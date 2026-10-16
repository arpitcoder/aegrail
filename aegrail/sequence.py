"""Rules over what an agent has already done.

Every tier so far judges one call in isolation, which leaves a whole
class of risk invisible. Reading a secret is allowed. Posting to a URL is
allowed. Reading a secret and *then* posting is exfiltration, and no
per-call check sees it, because neither call is wrong on its own.

The same shape covers most of what actually goes wrong with a read-heavy
agent holding broad credentials:

    read customer telemetry  → send outside the VPC
    list access keys         → write to a public bucket
    dump a database          → open an outbound connection

So the contract can also say: once this has happened, that becomes
restricted. State is per session and in memory — this is about a single
agent's trajectory, not a fleet-wide correlation, and keeping it local
means no round trip on the critical path.

Like every other tier, sequence rules may only escalate.
"""
from __future__ import annotations

import bisect
import fnmatch
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from .verdict import Decision, Tier, Verdict, allow


class SequenceRuleError(ValueError):
    pass


@dataclass
class _Occurrence:
    tool: str
    at: float


@dataclass
class SequenceRule:
    """"After `afterAny`, treat `then` as `verdict`."

    within_seconds bounds how long the trigger stays armed. Without it a
    read at the start of a long-running agent would restrict every send
    for the rest of the process, which is unusable in practice. With it,
    the rule says what people actually mean: these two things happening
    close together is the suspicious part.
    """

    after_any: list                 # tool names or globs that arm the rule
    then: str                       # tool name or glob that is restricted
    verdict: Verdict = Verdict.HITL
    reason: str = ""
    within_seconds: int = 0         # 0 = the whole session

    def __post_init__(self):
        # One compiled alternation instead of an fnmatch call per pattern
        # per history entry. A rule like then:"*" over a few hundred
        # entries was costing thousands of pattern matches on every call,
        # which put the deterministic tiers in the same latency class as
        # the thing they exist to avoid.
        self._after_re = re.compile(
            "|".join(fnmatch.translate(p) for p in (self.after_any or ["*"]))
        )
        self._then_re = re.compile(fnmatch.translate(self.then))
        # Tool names repeat constantly within a session, so the answer to
        # "does this name match" is worth remembering.
        self._match_cache: dict = {}

        if self.verdict is Verdict.ALLOW:
            raise SequenceRuleError(
                f"sequence rule for '{self.then}' has verdict=allow. Rules may "
                f"only escalate — an allow here would overrule the layers that "
                f"already ran."
            )
        if not self.after_any:
            raise SequenceRuleError("sequence rule needs at least one afterAny tool")
        if not self.reason:
            self.reason = f"{self.then} after {'/'.join(self.after_any)}"

    def _arms(self, tool: str) -> bool:
        hit = self._match_cache.get(tool)
        if hit is None:
            hit = self._after_re.match(tool) is not None
            self._match_cache[tool] = hit
        return hit

    def _armed_by(self, history: list, now: float) -> Optional[_Occurrence]:
        window = self.within_seconds
        for occ in reversed(history):
            if window and (now - occ.at) > window:
                # History is ordered, so once we are past the window
                # nothing older can arm this rule.
                break
            if self._arms(occ.tool):
                return occ
        return None

    def evaluate(self, tool: str, history: list, now: float) -> Optional[Decision]:
        if self._then_re.match(tool) is None:
            return None
        trigger = self._armed_by(history, now)
        if trigger is None:
            return None

        gap = int(now - trigger.at)
        return Decision(
            verdict=self.verdict,
            tier=Tier.DETERMINISTIC,
            reasons=[f"{self.reason} (saw '{trigger.tool}' {gap}s earlier)"],
        )

    @staticmethod
    def from_dict(raw: dict) -> "SequenceRule":
        d = dict(raw)
        verdict_raw = str(d.pop("verdict", "hitl")).lower()
        verdict = {
            "hitl": Verdict.HITL,
            "review": Verdict.HITL,
            "deny": Verdict.DENY,
            "block": Verdict.DENY,
            "allow": Verdict.ALLOW,  # accepted so __post_init__ rejects it clearly
        }.get(verdict_raw)
        if verdict is None:
            raise SequenceRuleError(f"unknown verdict {verdict_raw!r}")

        alias = {
            "afterAny": "after_any",
            "after": "after_any",
            "withinSeconds": "within_seconds",
        }
        kwargs = {alias.get(k, k): v for k, v in d.items()}
        if isinstance(kwargs.get("after_any"), str):
            kwargs["after_any"] = [kwargs["after_any"]]

        known = {f for f in SequenceRule.__dataclass_fields__}
        unknown = set(kwargs) - known
        if unknown:
            raise SequenceRuleError(f"unknown rule field(s): {sorted(unknown)}")
        return SequenceRule(verdict=verdict, **kwargs)


class SequenceTracker:
    """Per-session history, and the rules evaluated against it."""

    def __init__(self, rules: Optional[list] = None, max_history: int = 500):
        self.rules = list(rules or [])
        self._history: list = []
        # Parallel array of timestamps so the window boundary can be
        # located with bisect instead of a scan.
        self._times: list = []
        self._max = max_history
        self._lock = threading.Lock()

    @staticmethod
    def from_list(raw: Optional[list]) -> "SequenceTracker":
        return SequenceTracker([SequenceRule.from_dict(r) for r in (raw or [])])

    def evaluate(self, tool: str) -> Decision:
        """Judge this call against what came before. Does not record it."""
        if not self.rules:
            return allow()

        # Which rules even target this tool. Most calls match few or
        # none, and a rule that does not apply should cost nothing.
        applicable = None
        for rule in self.rules:
            if rule._then_re.match(tool):
                if applicable is None:
                    applicable = [rule]
                else:
                    applicable.append(rule)
        if applicable is None:
            return allow()

        now = time.time()
        widest = 0
        for r in applicable:
            if r.within_seconds == 0:
                widest = 0
                break
            widest = max(widest, r.within_seconds)

        with self._lock:
            if widest:
                # History is time-ordered, so the window boundary is a
                # binary search rather than a walk. A burst of calls
                # would otherwise make every subsequent call scan the
                # whole burst.
                idx = bisect.bisect_left(self._times, now - widest)
                history = self._history[idx:]
            else:
                history = list(self._history)

        decision = allow()
        for rule in applicable:
            fired = rule.evaluate(tool, history, now)
            if fired is not None:
                decision = decision.combine(fired)
        return decision

    def record(self, tool: str) -> None:
        """Note that a call actually happened.

        Only successful calls are recorded. A blocked read never gave the
        agent the data, so it must not arm a rule about what to do with
        that data — otherwise one denied attempt would restrict the rest
        of the session for no reason.
        """
        with self._lock:
            at = time.time()
            self._history.append(_Occurrence(tool=tool, at=at))
            self._times.append(at)
            if len(self._history) > self._max:
                drop = len(self._history) - self._max
                del self._history[:drop]
                del self._times[:drop]

    def history(self) -> list:
        with self._lock:
            return [(o.tool, o.at) for o in self._history]

    def clear(self) -> None:
        with self._lock:
            self._history.clear()
            self._times.clear()

    def __len__(self) -> int:
        return len(self.rules)
