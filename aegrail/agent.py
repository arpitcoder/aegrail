"""The Agent — where the contract is actually enforced.

An Agent wraps a neural model with a symbolic contract. The model
proposes; the contract disposes. Every tool call passes through
`Agent.call_tool`, which evaluates the contract before the tool function
ever runs.

    from aegrail import Agent, Budget, AuditSink

    agent = Agent(
        identity="cleanup-bot/v1",
        budget=Budget(usd=5.0, tokens=100_000, wall_seconds=300),
        audit=AuditSink.file("./audit.jsonl"),
    )

    @agent.tool(destructive=True)
    def delete_records(table: str, older_than_days: int) -> str:
        ...

    agent.call_tool("delete_records", table="users", older_than_days=90)
"""
from __future__ import annotations

import functools
import os
import time
import warnings
from typing import Any, Callable, Optional

from . import otel
from .argrules import ArgRuleSet
from .audit import AuditSink
from .budget import Budget
from .capability import CapabilityError, CapabilityGrant, ToolManifest
from .shadow import Mode, ObservationStore
from .exceptions import BudgetExceeded, HITLRejected, HITLTimeout, PolicyDenied
from .hitl import HITLClient
from .judge import JudgeClient
from .pending import PendingCall, PendingStore, Resumer
from .policy import PolicyClient
from .traces import TraceClient
from .verdict import Decision, Tier, Verdict, allow, deny, needs_human


class _ToolSpec:
    def __init__(
        self,
        fn: Callable,
        *,
        destructive: bool,
        description: str,
        inspect_args: bool = False,
    ):
        self.fn = fn
        self.destructive = destructive
        # Set when the tool's risk lives in its arguments rather than its
        # name — execute_sql, run_shell, http_post. These are the calls a
        # denylist cannot classify, so Tier 2 looks at the actual values.
        self.inspect_args = inspect_args
        self.description = description or (fn.__doc__ or "").strip().split("\n")[0]
        self.name = fn.__name__


class Agent:
    """A neural agent bound to a symbolic runtime contract."""

    def __init__(
        self,
        identity: str,
        budget: Optional[Budget] = None,
        audit: Optional[AuditSink] = None,
        *,
        policy: Optional[PolicyClient] = None,
        hitl: Optional[HITLClient] = None,
        poll_policy: bool = True,
        hitl_timeout_seconds: int = 300,
        redis_url: Optional[str] = None,
        resume_pending: bool = True,
        arg_rules: Optional[ArgRuleSet] = None,
        capability: Optional[CapabilityGrant] = None,
        strict_capability: bool = False,
        mode: Mode = Mode.ENFORCE,
    ):
        self.identity = identity
        self.budget = budget or Budget()
        self.audit = audit or AuditSink.file(
            os.environ.get("AEGRAIL_AUDIT_PATH", "./audit.jsonl")
        )
        self.hitl_timeout_seconds = hitl_timeout_seconds
        self._tools: dict[str, _ToolSpec] = {}

        self.hitl = hitl or HITLClient()

        if policy is not None:
            self.policy = policy
        elif poll_policy:
            self.policy = PolicyClient.start(
                agent_name=identity.split("/")[0],
                on_change=self._on_policy_change,
            )
        else:
            self.policy = PolicyClient(agent_name=identity.split("/")[0])

        # Tier 2. Runs in the control plane, so the agent process needs
        # no model credentials and the customer's tool arguments never
        # leave whichever model they configured.
        self.judge = JudgeClient(credentials=self.policy.credentials)

        # Layer 2. An explicit grant wins; otherwise take whatever the
        # control plane published for this agent. Resolved again in
        # attest() once tools are registered, because policy arrives
        # asynchronously and may land after construction.
        self._strict_capability = strict_capability
        self.capability = capability or CapabilityGrant.from_spec(
            {"allowed_tools": self.policy.snapshot.allowed_tools},
            strict=strict_capability,
            source=self.policy.snapshot.source,
        )
        self._attested = False

        # Shadow mode. Deliberately a constructor argument with no env-var
        # equivalent — a governance product that silently stops governing is
        # worse than no product, so switching it on must be a visible, explicit
        # act that shows up in review.
        self.mode = mode
        self.observations = ObservationStore(identity=identity) if not mode.enforces else None

        # Argument rules ship with the contract. Empty is a valid state:
        # an agent with no rules still has the structural layer.
        self.arg_rules = arg_rules or ArgRuleSet()

        # Every contract decision is also a trace. The runtime already
        # saw the call in order to rule on it, so the telemetry is a
        # byproduct of enforcement rather than a second integration.
        self.traces = TraceClient(credentials=self.policy.credentials).start()

        # Durable record of anything this agent is waiting on, so an
        # approval that lands after a restart is still carried out.
        self.pending = PendingStore(identity=self.identity, url=redis_url)
        self.resumer: Optional[Resumer] = None
        if resume_pending and self.pending.enabled:
            self.resumer = Resumer(
                store=self.pending,
                poll_review=self.hitl.poll,
                execute=self._execute_resumed,
                on_rejected=self._record_resumed_rejection,
                logger=lambda m: self.audit.record("resume.log", message=m),
            ).start()

        self.audit.record(
            "session.start",
            identity=self.identity,
            budget=self.budget.snapshot(),
            policy_source=self.policy.snapshot.source,
            pending_store=self.pending.describe(),
            capability_source=self.capability.source,
            capability_configured=self.capability.configured,
            mode=self.mode.value,
            shadow=not self.mode.enforces,
        )
        if not self.mode.enforces:
            # Loud on purpose. Anyone reading these logs must not believe the
            # agent is protected.
            warnings.warn(
                f"Agent '{identity}' is in SHADOW mode: contracts are evaluated "
                f"and recorded but NOT enforced. Every tool call will execute.",
                stacklevel=2,
            )

    # ── layer 2: capability attestation ─────────────────────────────
    def manifest(self) -> ToolManifest:
        """The tools this process actually registered."""
        return ToolManifest.from_specs(self._tools)

    def attest(self, *, raise_on_violation: Optional[bool] = None) -> ToolManifest:
        """Reconcile registered tools against the published grant.

        Call once after registering tools and before doing any work. Three
        things happen here, none of them on the hot path:

          1. The grant is re-read. Policy arrives asynchronously, so the one
             resolved in __init__ may predate the first successful fetch.
          2. Registered tools are reconciled against it. A tool the agent
             holds but the grant does not cover is a disagreement between
             deployed code and published policy — a deploy problem, and far
             better surfaced at startup than discovered one denied call at a
             time in production.
          3. The manifest hash is written to the audit chain, so the
             permitted set is provable alongside what was actually done.

        Violations raise by default when strict, and warn otherwise. Warning
        is not weakness: the calls are still denied by evaluate(). This only
        decides whether the process refuses to start.
        """
        # Re-resolve unless the caller pinned an explicit grant.
        published = self.policy.snapshot.allowed_tools
        if published and not self.capability.configured:
            self.capability = CapabilityGrant.from_spec(
                {"allowed_tools": published},
                strict=self._strict_capability,
                source=self.policy.snapshot.source,
            )

        man = self.manifest()
        violations = self.capability.reconcile(man)

        self.audit.record(
            "capability.attested",
            identity=self.identity,
            manifest_hash=man.digest(),
            tools=list(man.tools),
            destructive=sorted(man.destructive),
            granted=sorted(self.capability.allowed_tools),
            configured=self.capability.configured,
            violations=violations,
        )

        if violations:
            msg = (
                f"agent '{self.identity}' registered {len(violations)} tool(s) "
                f"outside its capability grant: {', '.join(violations)}. "
                f"These calls will be denied at runtime."
            )
            hard = self._strict_capability if raise_on_violation is None else raise_on_violation
            if hard:
                raise CapabilityError(msg)
            self.audit.record("capability.violation", identity=self.identity,
                              tools=violations, message=msg)

        if self._strict_capability and not self.capability.configured:
            raise CapabilityError(
                f"strict_capability is on but no grant was published for "
                f"'{self.identity}'. Refusing to run: an agent with no declared "
                f"capability cannot be held to one."
            )

        self._attested = True
        return man

    # ── durable resume ──────────────────────────────────────────────
    def _execute_resumed(self, call: PendingCall) -> None:
        """Run a tool whose approval arrived while nobody was waiting.

        The contract was already evaluated when the call was first made
        and a human has since approved it, so this does not re-run the
        checks — it carries out the decision that was reached.
        """
        spec = self._tools.get(call.tool)
        if spec is None:
            self.audit.record(
                "resume.abandoned",
                tool=call.tool,
                identity=self.identity,
                reason="tool is not registered in this process",
            )
            return

        result = spec.fn(*call.args.get("args", []), **call.args.get("kwargs", {}))
        self.audit.record(
            "tool.executed",
            tool=call.tool,
            identity=self.identity,
            destructive=spec.destructive,
            call=call.args,
            resumed=True,
            waited_seconds=round(call.waiting_seconds, 1),
            review_id=call.review_id,
        )
        return result

    def _record_resumed_rejection(self, call: PendingCall, status: str) -> None:
        self.audit.record(
            "hitl.denied",
            tool=call.tool,
            identity=self.identity,
            outcome=status,
            review_id=call.review_id,
            resumed=True,
        )

    # ── policy ──────────────────────────────────────────────────────
    def _on_policy_change(self, snap) -> None:
        """A fleet policy landed. Tighten local ceilings, log the change."""
        self.budget.tighten(
            usd=snap.budget_usd_per_day,
            tokens=snap.budget_tokens_per_day,
            wall_seconds=snap.budget_max_wall_seconds,
        )
        self.audit.record(
            "policy.applied",
            policy_hash=snap.policy_hash,
            source=snap.source,
            deny_tools=sorted(snap.deny_tools),
            require_hitl_for=sorted(snap.require_hitl_for),
            budget=self.budget.snapshot(),
        )

    # ── tool registration ───────────────────────────────────────────
    def tool(
        self,
        _fn: Optional[Callable] = None,
        *,
        destructive: bool = False,
        description: str = "",
        inspect_args: bool = False,
    ):
        """Register a callable as a contract-governed tool.

        `destructive=True` marks the tool as irreversible. Irreversible
        actions require human approval unless the contract explicitly
        says otherwise.

        `inspect_args=True` marks a tool whose risk is carried by its
        arguments rather than its name — execute_sql, run_shell,
        http_post. Those calls are additionally sent to the Tier 2 judge,
        which can escalate them but never approve past Tier 1.
        """

        def decorate(fn: Callable) -> Callable:
            spec = _ToolSpec(
                fn,
                destructive=destructive,
                description=description,
                inspect_args=inspect_args,
            )
            self._tools[spec.name] = spec

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return self.call_tool(spec.name, *args, **kwargs)

            wrapper.unguarded = fn  # escape hatch for tests
            return wrapper

        return decorate(_fn) if _fn is not None else decorate

    @property
    def tools(self) -> dict:
        return dict(self._tools)

    # ── enforcement ─────────────────────────────────────────────────
    def call_tool(self, name: str, /, *args: Any, **kwargs: Any) -> Any:
        # `name` is positional-only. Without the slash, a tool with its own
        # `name` argument — delete_bucket(name=...), create_user(name=...) —
        # collides with this parameter and raises TypeError, making a whole
        # class of perfectly ordinary tools uncallable by keyword.
        """Evaluate the contract, then run the tool if permitted."""
        spec = self._tools.get(name)
        if spec is None:
            raise KeyError(f"tool '{name}' is not registered with this agent")

        call_args = {"args": list(args), "kwargs": dict(kwargs)}

        with otel.contract_span(
            "tool_call",
            tool=name,
            identity=self.identity,
            attributes={
                "destructive": spec.destructive,
                "policy_hash": self.policy.snapshot.policy_hash,
                **{f"budget.{k}": v for k, v in self.budget.snapshot().items()},
            },
        ) as span:

            decision = self._evaluate(spec, name, call_args)
            span.set("verdict", str(decision.verdict))
            span.set("decided_by", str(decision.tier))

            # ── shadow mode ──────────────────────────────────────────
            # The single point where enforcement is skipped. Everything above
            # ran exactly as it would in production; what changes is only
            # whether the verdict is acted on.
            #
            # The event kind changes too: `tool.would_deny` is not
            # `tool.denied`. Nothing downstream — dashboard, auditor, alerting
            # — should ever have to infer from a flag whether a recorded
            # denial actually stopped anything.
            if not self.mode.enforces:
                self.observations.record(name, spec.destructive,
                                         call_args.get("kwargs", {}) or {}, decision)
                span.set("shadow", True)
                if decision.verdict is not Verdict.ALLOW:
                    self.audit.record(
                        f"tool.would_{decision.verdict.name.lower()}",
                        tool=name,
                        identity=self.identity,
                        call=call_args,
                        shadow=True,
                        **decision.to_audit(),
                    )
                decision = allow()

            if decision.verdict is Verdict.DENY:
                event = self.audit.record(
                    "tool.denied",
                    tool=name,
                    identity=self.identity,
                    call=call_args,
                    **decision.to_audit(),
                )
                span.set("audit_hash", event["hash"])
                self.traces.record(
                    operation="tool_call",
                    status="blocked",
                    tool_names=[name],
                    session_id=self.identity,
                    error_message=decision.why(),
                    payload={"audit_hash": event["hash"], **decision.to_audit()},
                )
                raise PolicyDenied(name, decision.why())

            if decision.verdict is Verdict.HITL:
                approval = self._require_human(name, spec, call_args, span, decision)
                span.set("hitl_review_id", approval)

            started = time.monotonic()
            result = spec.fn(*args, **kwargs)
            elapsed_ms = int((time.monotonic() - started) * 1000)

            event = self.audit.record(
                "tool.executed",
                tool=name,
                identity=self.identity,
                destructive=spec.destructive,
                call=call_args,
                budget=self.budget.snapshot(),
            )
            span.allow(audit_hash=event["hash"])
            self.traces.record(
                operation="tool_call",
                status="ok",
                duration_ms=elapsed_ms,
                tool_names=[name],
                session_id=self.identity,
                payload={
                    "verdict": "allowed",
                    # Which layer settled it. Only the deny path recorded this,
                    # so an allowed call gave no way to tell whether the
                    # contract had actually reasoned about it or simply had no
                    # rule that applied — the difference between governed and
                    # merely observed.
                    "decided_by": str(decision.tier),
                    "reasons": list(decision.reasons),
                    "destructive": spec.destructive,
                    "audit_hash": event["hash"],
                    "policy": self.policy.snapshot.source,
                },
            )
            return result

    def _evaluate(self, spec, name: str, call_args: dict) -> Decision:
        """Tier 1, then Tier 2 only if the call is still unclear.

        The two are combined with `max`, so Tier 2 can raise the verdict
        and never lower it. A judge that is wrong, slow, or manipulated
        makes this stricter — never permissive.
        """
        decision = allow()

        # Layer 2 first. If the tool is not this agent's to hold, nothing
        # further is worth evaluating: no argument is acceptable for a tool
        # it should never have registered, and the denial does not depend on
        # what was passed. Also the cheapest check here — a set lookup.
        decision = decision.combine(self.capability.evaluate(name))

        # Layer 3, tier 1. deny_tools remains as a fleet-wide override that
        # can revoke a tool without editing every agent's grant.
        if not self.policy.tool_allowed(name):
            decision = decision.combine(deny("denied by fleet policy"))

        try:
            self.budget.check()
        except BudgetExceeded as exc:
            decision = decision.combine(deny(str(exc)))

        if spec.destructive:
            decision = decision.combine(needs_human("tool is marked irreversible"))
        if self.policy.requires_hitl(name):
            decision = decision.combine(needs_human("fleet policy requires review"))

        # Tier 2a: declarative rules over the actual arguments. Runs on
        # every call, costs microseconds, and settles most of what would
        # otherwise reach a model.
        decision = decision.combine(
            self.arg_rules.evaluate(name, call_args.get("kwargs", {}) or {})
        )

        # Tier 2b runs only where the rules still cannot tell: a tool
        # whose risk is in its arguments, that nothing has already
        # stopped. Asking about a call that is already denied would cost
        # a model call to change nothing.
        if spec.inspect_args and decision.verdict is not Verdict.DENY:
            decision = decision.combine(
                self.judge.classify(
                    tool=name,
                    description=spec.description,
                    args=call_args.get("kwargs", {}) or {},
                )
            )

        return decision

    def _require_human(self, name, spec, call_args, span, decision=None) -> str:
        """Block on human approval. Timeout denies."""
        why = decision.why() if decision is not None else ""
        summary = (
            f"Agent `{self.identity}` wants to run `{name}`.\n\n"
            f"Tool: {spec.description or name}\n"
            f"Irreversible: {'yes' if spec.destructive else 'no'}\n"
            + (f"Why this needs review: {why}\n" if why else "")
            + f"Arguments: {call_args}\n\n"
            f"Approve only if this action is expected."
        )

        submitted = self.audit.record(
            "hitl.requested",
            tool=name,
            identity=self.identity,
            call=call_args,
        )
        span.set("audit_hash", submitted["hash"])

        if not self.hitl.enabled:
            # No MAVOPS_API_TOKEN — fail closed on irreversible actions
            # rather than silently allowing them.
            self.audit.record(
                "hitl.unavailable",
                tool=name,
                identity=self.identity,
                outcome="denied",
            )
            self.traces.record(
                operation="tool_call",
                status="blocked",
                tool_names=[name],
                session_id=self.identity,
                error_message="no approval path configured",
                payload={"verdict": "blocked", "reason": "hitl_unavailable"},
            )
            raise PolicyDenied(
                name,
                (f"{why} — " if why else "")
                + "requires human approval, and no approval path is "
                  "configured (failing closed)",
            )

        # Written before the wait begins, not after. A pod that dies
        # between submitting the review and recording it would otherwise
        # leave an approval nobody ever acts on.
        record: Optional[PendingCall] = None

        def _on_submitted(d):
            nonlocal record
            record = PendingCall(
                review_id=d.review_id,
                identity=self.identity,
                tool=name,
                args=call_args,
                approval_url=d.approval_url,
            )
            self.pending.put(record)
            print(
                f"\n  [aegrail] human approval required for '{name}'\n"
                f"  [aegrail] review: {d.approval_url}\n"
                f"  [aegrail] waiting up to {self.hitl_timeout_seconds}s…\n"
            )

        try:
            decision = self.hitl.request_approval(
                tool=name,
                title=f"Approve `{name}` for {self.identity}",
                content=summary,
                metadata={"identity": self.identity, "call": call_args},
                timeout_seconds=self.hitl_timeout_seconds,
                on_submitted=_on_submitted,
            )
        except (HITLRejected, HITLTimeout) as exc:
            # A timeout is not a decision. The record stays so the
            # resumer can still act if a human answers later; a
            # rejection is final and clears immediately.
            if record is not None and isinstance(exc, HITLRejected):
                self.pending.clear(record)
            self.audit.record(
                "hitl.denied",
                tool=name,
                identity=self.identity,
                outcome=type(exc).__name__,
                reason=getattr(exc, "reason", ""),
            )
            raise
        except Exception as exc:
            # The approval service was unreachable, rejected our
            # credential, or failed some other way. That is not a licence
            # to proceed: an irreversible action with no human reachable
            # is denied, and the reason is recorded.
            self.audit.record(
                "hitl.unreachable",
                tool=name,
                identity=self.identity,
                outcome="denied",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise PolicyDenied(
                name,
                (f"{why} — " if why else "")
                + f"requires human approval, and the review service is "
                  f"unreachable ({type(exc).__name__}) — failing closed",
            ) from exc

        if record is not None:
            self.pending.clear(record)
        self.audit.record(
            "hitl.approved",
            tool=name,
            identity=self.identity,
            review_id=decision.review_id,
            decided_via=decision.decided_via,
        )
        return decision.review_id

    # ── model spend ─────────────────────────────────────────────────
    def charge_model_call(
        self,
        *,
        model: str,
        usd: float = 0.0,
        tokens: int = 0,
        provider: str = "",
        duration_ms: int = 0,
    ) -> None:
        """Record an LLM call against the budget, enforcing ceilings."""
        with otel.contract_span(
            "model_call", tool=model, identity=self.identity
        ) as span:
            if not self.policy.model_allowed(model):
                self.audit.record(
                    "model.denied", model=model, identity=self.identity
                )
                self.traces.record(
                    operation="chat",
                    model=model,
                    provider=provider,
                    status="blocked",
                    session_id=self.identity,
                    error_message="model not in policy allowlist",
                )
                raise PolicyDenied(model, "model not in policy allowlist")
            self.budget.check(add_usd=usd, add_tokens=tokens)
            self.budget.charge(usd=usd, tokens=tokens)
            event = self.audit.record(
                "model.called",
                model=model,
                identity=self.identity,
                usd=usd,
                tokens=tokens,
                budget=self.budget.snapshot(),
            )
            span.allow(audit_hash=event["hash"], usd=usd, tokens=tokens)
            self.traces.record(
                operation="chat",
                model=model,
                provider=provider,
                status="ok",
                output_tokens=tokens,
                cost_usd=usd,
                duration_ms=duration_ms,
                session_id=self.identity,
                payload={"audit_hash": event["hash"]},
            )

    def begin_task(self, *, reset_spend: bool = False) -> None:
        """Start a new unit of work.

        Resets the wall-clock window so `wall_seconds` bounds this task
        rather than the lifetime of the process. Resident agents that
        serve many requests should call this per request.
        """
        self.budget.begin_task(reset_spend=reset_spend)

    # ── introspection ───────────────────────────────────────────────
    def contract_summary(self) -> dict:
        snap = self.policy.snapshot
        return {
            "identity": self.identity,
            "budget": self.budget.snapshot(),
            "deny_tools": sorted(snap.deny_tools),
            "require_hitl_for": sorted(snap.require_hitl_for),
            "destructive_tools": sorted(
                n for n, s in self._tools.items() if s.destructive
            ),
            "policy_hash": snap.policy_hash,
            "policy_source": snap.source,
            "audit_head": self.audit.head,
        }

    def close(self) -> None:
        if getattr(self, "traces", None):
            self.traces.stop()
        if getattr(self, "resumer", None):
            self.resumer.stop()
        self.audit.record(
            "session.end", identity=self.identity, budget=self.budget.snapshot()
        )
        if self.policy:
            self.policy.stop()

    def __repr__(self) -> str:
        return f"<Agent {self.identity} tools={sorted(self._tools)}>"
