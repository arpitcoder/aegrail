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
from .sequence import SequenceTracker
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
        begins: Optional[str] = None,
        completes: Optional[str] = None,
    ):
        self.fn = fn
        self.destructive = destructive
        # A critical section this call opens or closes. The pair is what
        # makes "safe state" a thing a program can check rather than a
        # thing a person asserts: an agent is at a safe point when no
        # section it opened is still open.
        self.begins = begins
        self.completes = completes
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
        trace_arguments: bool = False,
        agent_id: Optional[str] = None,
        sequence: Optional[SequenceTracker] = None,
        expect_manifest: Optional[Any] = None,
        max_policy_age: int = 300,
    ):
        self.identity = identity
        self.budget = budget or Budget()
        self.audit = audit or AuditSink.file(
            os.environ.get("AEGRAIL_AUDIT_PATH", "./audit.jsonl")
        )
        self.hitl_timeout_seconds = hitl_timeout_seconds
        # Off by default, and it stays that way. Tool arguments are where the
        # personal data lives — claim numbers, account ids, addresses — so
        # shipping them to a control plane is a decision a person makes
        # deliberately for a given agent, not a default they inherit.
        self.trace_arguments = trace_arguments
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

        # Rules over the trajectory rather than the call. Empty is the common
        # case and costs nothing: with no rules the tracker short-circuits
        # before it touches history.
        self.sequence = sequence or SequenceTracker()

        # The approved tool surface, if someone approved one. A hash pins it;
        # a full manifest pins it *and* can say which tool moved. Read from
        # the environment too, because the pin belongs to the deploy that was
        # reviewed rather than to the source file — the same reason you pin a
        # container digest in the manifest and not in the Dockerfile.
        self._expect_manifest = expect_manifest or os.environ.get(
            "AEGRAIL_EXPECT_MANIFEST"
        )

        # How long an agent may keep acting on a policy the control plane has
        # not confirmed. See _halt_check for why this does not simply deny
        # everything, and why it applies only to a managed agent.
        self.max_policy_age = max_policy_age
        self._halted = False
        self._halt_reason = ""
        self._halt_mode = "immediate"
        self._halt_at = 0.0
        # How long a controlled halt may wait for open work to finish. A
        # controlled stop that can be held open indefinitely is not a stop:
        # an agent that never closes its section would outlive the button.
        self._halt_escalate_after = 60.0
        # Critical sections this agent has opened and not yet closed.
        self._open_sections: list = []

        # Every contract decision is also a trace. The runtime already
        # saw the call in order to rule on it, so the telemetry is a
        # byproduct of enforcement rather than a second integration.
        self.traces = TraceClient(
            credentials=self.policy.credentials, agent_id=agent_id
        ).start()

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
            # Per-tool surface digests, so a later attestation can be diffed
            # against this one by anyone holding the chain — not only by a
            # process that happens to still have the approved manifest.
            surfaces=dict(man.surfaces),
            granted=sorted(self.capability.allowed_tools),
            configured=self.capability.configured,
            violations=violations,
        )

        self._check_manifest_pin(man)

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
            halted=snap.halted,
            deny_tools=sorted(snap.deny_tools),
            require_hitl_for=sorted(snap.require_hitl_for),
            budget=self.budget.snapshot(),
        )
        # Its own event, not a field on policy.applied. "The fleet was
        # stopped at 14:02" is the line someone goes looking for during an
        # incident, and it should not require reading every policy update
        # to find it.
        if snap.halted:
            self.audit.record("agent.halted", identity=self.identity,
                              source=snap.source, reason=snap.halt_reason)

    # ── tool registration ───────────────────────────────────────────
    def tool(
        self,
        _fn: Optional[Callable] = None,
        *,
        destructive: bool = False,
        description: str = "",
        inspect_args: bool = False,
        begins: Optional[str] = None,
        completes: Optional[str] = None,
    ):
        """Register a callable as a contract-governed tool.

        `destructive=True` marks the tool as irreversible. Irreversible
        actions require human approval unless the contract explicitly
        says otherwise.

        `inspect_args=True` marks a tool whose risk is carried by its
        arguments rather than its name — execute_sql, run_shell,
        http_post. Those calls are additionally sent to the Tier 2 judge,
        which can escalate them but never approve past Tier 1.

        `begins` / `completes` name a critical section — a span of work
        that must not be left half-done. `begins="payout"` opens it,
        `completes="payout"` closes it. A controlled halt waits for open
        sections to close before it takes full effect, and refuses to let
        a new one open in the meantime. Without these an agent has no
        boundaries, so a controlled halt and an immediate one are the
        same thing.
        """

        def decorate(fn: Callable) -> Callable:
            spec = _ToolSpec(
                fn,
                destructive=destructive,
                description=description,
                inspect_args=inspect_args,
                begins=begins,
                completes=completes,
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

    def _traced_args(self, call_args: dict) -> dict:
        """Arguments for the trace payload, or nothing.

        Enforcement always sees the real arguments — rules are evaluated
        against them regardless. This only governs what is *shipped* to the
        control plane, which is a separate decision with a separate blast
        radius: the audit chain stays local, a trace does not.
        """
        if not self.trace_arguments:
            return {}
        return {"arguments": call_args}

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
                    payload={
                        "audit_hash": event["hash"],
                        **decision.to_audit(),
                        **self._traced_args(call_args),
                    },
                )
                raise PolicyDenied(name, decision.why())

            if decision.verdict is Verdict.HITL:
                approval = self._require_human(name, spec, call_args, span, decision)
                span.set("hitl_review_id", approval)

            started = time.monotonic()
            result = spec.fn(*args, **kwargs)
            elapsed_ms = int((time.monotonic() - started) * 1000)

            # Only a call that actually ran arms a sequence rule. A blocked
            # read never handed the agent the data, so it must not restrict
            # what the rest of the session may do with it. This sits after
            # the call rather than before it for the same reason: a tool that
            # raised returned nothing to exfiltrate.
            self.sequence.record(name)
            self._enter_sections(spec, name)

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
                    **self._traced_args(call_args),
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

        # Before anything else. A halt needs no tool name, no arguments and
        # no reasoning to apply, and it must not be reachable only after a
        # cheaper check happens to pass.
        decision = decision.combine(self._halt_check(spec))

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

        # Tier 2c: what has already happened this session. Each call above
        # may be individually permissible — read a claim, post to a URL — and
        # the pair still be exfiltration. Deterministic, in-memory, and placed
        # before the judge so a sequence denial never pays for a model call.
        decision = decision.combine(self.sequence.evaluate(name))

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

    # ── the stop button ─────────────────────────────────────────────
    def halt(self, reason: str = "halted locally", *, mode: str = "immediate",
             escalate_after: float = 60.0) -> None:
        """Stop this agent acting. Process-local; the fleet switch is policy.

        Two modes, and the distinction is the one machinery safety has used
        for decades (ISO 13850 calls them stop categories):

        **immediate** — nothing else runs. Correct when you believe the agent
        is compromised, and correct by default: when in doubt you want it to
        stop, not to negotiate.

        **controlled** — open critical sections are allowed to close, and
        nothing new may open. The agent comes to rest at a boundary instead
        of mid-sequence. Use it when a half-finished sequence is worse than a
        few more seconds of running, which is most operational work and
        almost no security incidents.

        A controlled halt escalates to immediate after `escalate_after`
        seconds. Without that, an agent that simply never closes its section
        outlives the stop button, which is the one failure a stop button may
        not have.

        Note what this does *not* do: it stops tool calls that go through
        this agent. It does not revoke the process's credentials or close its
        connections. A kill switch that leaves the keys live is a well-known
        way to feel safe without being safe — see "Honest limits".
        """
        if mode not in ("immediate", "controlled"):
            raise ValueError("mode must be 'immediate' or 'controlled'")

        self._halted, self._halt_reason = True, reason
        self._halt_mode = mode
        self._halt_at = time.monotonic()
        self._halt_escalate_after = escalate_after
        self.audit.record(
            "agent.halted", identity=self.identity, source="local",
            reason=reason, mode=mode,
            open_sections=list(self._open_sections),
        )
        # A controlled halt with nothing open is already at rest. Say so,
        # rather than leaving an operator to infer it from silence.
        if mode == "controlled" and not self._open_sections:
            self._record_safe_point()

    def resume(self) -> None:
        self._halted, self._halt_reason = False, ""
        self._halt_mode, self._halt_at = "immediate", 0.0
        self.audit.record("agent.resumed", identity=self.identity,
                          source="local")

    @property
    def halted(self) -> bool:
        return self._halted or self.policy.snapshot.halted

    @property
    def at_safe_point(self) -> bool:
        """No critical section is open.

        This is what "a safe state" means here, and it is deliberately
        checkable rather than asserted. A safe state is not a prior state:
        it is one in which nothing further is half-done.
        """
        return not self._open_sections

    @property
    def open_sections(self) -> list:
        return list(self._open_sections)

    def _record_safe_point(self) -> None:
        self.audit.record("agent.safe_point", identity=self.identity,
                          reason=self._halt_reason)

    def _enter_sections(self, spec, name: str) -> None:
        """Track a critical section across a call that actually ran."""
        if spec.begins:
            self._open_sections.append(spec.begins)
        if spec.completes and spec.completes in self._open_sections:
            # Close the most recent matching section. Nesting the same name
            # is unusual but legal, and closing the oldest would leave the
            # inner one orphaned.
            for i in range(len(self._open_sections) - 1, -1, -1):
                if self._open_sections[i] == spec.completes:
                    del self._open_sections[i]
                    break
            if self._halted and self._halt_mode == "controlled" \
                    and not self._open_sections:
                self._record_safe_point()

    def _halt_check(self, spec) -> Decision:
        """The stop button, and what to do when nobody can press it.

        Two cases, and the second is the one that decides whether any of
        this is worth having.

        **Halted.** Everything denies. No exceptions for reads: if an
        operator stopped the fleet mid-incident, a read that feeds the next
        model call is not neutral.

        **Stale.** The agent has not heard from the control plane within
        `max_policy_age`. This is the case a kill switch exists for and the
        one most products get wrong. Fail open and the switch is defeated by
        anyone who can drop egress — including an agent that has been talked
        into killing its own network. Fail closed and a control-plane blip
        stops a customer's production fleet, which is a product they turn
        off within the week.

        So it splits on reversibility: destructive tools are revoked,
        everything else keeps working. An agent that cannot hear the kill
        switch assumes it was pressed, but only for the actions that would
        matter if it had been. The agent degrades instead of dying, and the
        half you would actually want stopped is stopped.

        Only for a *managed* agent. With no control plane configured there is
        nothing to be out of touch with, and degrading a developer's laptop
        agent would teach them to raise the timeout until it never fires.
        """
        if self._halted:
            return self._local_halt_check(spec)

        snap = self.policy.snapshot
        if snap.halted:
            return deny(
                "fleet policy has halted this agent"
                + (f": {snap.halt_reason}" if snap.halt_reason else "")
            )

        if not self.max_policy_age or not self.policy.managed:
            return allow()

        age = snap.age()
        if age <= self.max_policy_age:
            return allow()

        if not spec.destructive:
            return allow()

        pretty = "never" if age == float("inf") else f"{int(age)}s ago"
        return deny(
            f"policy last confirmed {pretty}, over the {self.max_policy_age}s "
            f"limit — irreversible tools are revoked until the control plane "
            f"is reachable again"
        )

    def _local_halt_check(self, spec) -> Decision:
        """A local halt, honouring its mode.

        An immediate halt denies everything. A controlled halt denies
        everything except the calls that close work already in flight —
        and only until the escalation deadline, after which it is an
        immediate halt like any other.
        """
        why = self._halt_reason
        if self._halt_mode != "controlled":
            return deny(f"agent is halted: {why}")

        if not self._open_sections:
            return deny(f"agent is halted at a safe point: {why}")

        waited = time.monotonic() - self._halt_at
        if waited >= self._halt_escalate_after:
            return deny(
                f"agent is halted: {why} — controlled stop gave up after "
                f"{int(waited)}s with {', '.join(self._open_sections)} still "
                f"open, escalated to an immediate stop"
            )

        if spec.completes and spec.completes in self._open_sections:
            return allow()

        return deny(
            f"agent is halted: {why} — controlled stop is waiting for "
            f"{', '.join(self._open_sections)} to close, and this call does "
            f"not close it"
        )

    def _check_manifest_pin(self, man: ToolManifest) -> None:
        """Refuse to run a toolset that is not the one that was approved.

        Attestation has always computed this hash and written it down. Writing
        it down is not change management — nobody reads an audit chain at
        startup. A pin is the part that makes the record load-bearing: if the
        surface moved since review, this process does not start.

        The failure is deliberately loud and deliberately annoying. A legitimate
        deploy updates the pin, which is the review step being asked for. A pin
        that warns and continues is decoration.
        """
        pin = self._expect_manifest
        if not pin:
            return

        approved = None
        if isinstance(pin, ToolManifest):
            approved, expected = pin, pin.digest()
        elif isinstance(pin, dict):
            approved = ToolManifest(
                tools=tuple(pin.get("tools", ())),
                destructive=frozenset(pin.get("destructive", ())),
                surfaces=tuple(sorted((pin.get("surfaces") or {}).items())),
            )
            expected = pin.get("manifest_hash") or approved.digest()
        else:
            expected = str(pin).split(":", 1)[-1].strip()

        actual = man.digest()
        if actual == expected:
            return

        # A hash alone cannot say what moved. A full approved manifest can,
        # and naming the tool is most of the value of catching this at all.
        detail = man.drift(approved) if approved is not None else None
        self.audit.record(
            "capability.drift",
            identity=self.identity,
            expected=expected,
            actual=actual,
            drift=detail,
        )

        where = ""
        if detail:
            parts = [f"{k}: {', '.join(v)}" for k, v in detail.items() if v]
            where = " — " + "; ".join(parts) if parts else ""

        msg = (
            f"agent '{self.identity}' has a tool surface that does not match "
            f"the approved manifest{where}. Expected {expected[:16]}…, got "
            f"{actual[:16]}…. A tool's name, description, signature or "
            f"destructive flag has changed since review. Re-review and update "
            f"the pin, or pin the manifest this build actually produces."
        )

        if not self.mode.enforces:
            # Shadow mode promises an onboarding agent will not be broken by
            # us. Breaking it at startup is still breaking it.
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            return
        raise CapabilityError(msg)

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
