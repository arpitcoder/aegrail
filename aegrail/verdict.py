"""What a contract evaluation concluded, and who concluded it.

Verdicts are ordered by severity, and combining two of them takes the
stricter one. That ordering is the entire safety argument for putting a
language model anywhere near this decision:

    final = max(deterministic_verdict, judge_verdict)

A judge can raise ALLOW to HITL. It cannot lower DENY to ALLOW, because
`max` will not let it. The constraint is enforced by the type rather than
by a rule someone has to remember, so a prompt-injected or simply wrong
judge can only ever make the system more cautious than the deterministic
layer already was.

That matters because the judge is a neural component, and the premise of
this product is that neural components cannot be trusted to gate their
own actions. A judge that could approve things would be exactly the hole
Aegrail exists to close.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


class Verdict(enum.IntEnum):
    """Ordered by severity. Higher always wins when two are combined."""

    ALLOW = 0
    HITL = 1  # a human must decide before this runs
    DENY = 2  # never runs, no human is asked

    def combine(self, other: "Verdict") -> "Verdict":
        return Verdict(max(int(self), int(other)))

    def __str__(self) -> str:
        return self.name.lower()


class Tier(enum.Enum):
    """Which layer reached the conclusion."""

    DETERMINISTIC = "deterministic"
    JUDGE = "judge"

    def __str__(self) -> str:
        return self.value


@dataclass
class Decision:
    """A verdict plus everything needed to justify it later.

    `reasons` accumulates one entry per contributing check, so the audit
    chain records not just what was decided but which clause decided it —
    the difference between evidence and a log line.
    """

    verdict: Verdict
    tier: Tier = Tier.DETERMINISTIC
    reasons: list = field(default_factory=list)
    judge_model: str = ""
    judge_latency_ms: int = 0
    escalated: bool = False

    def combine(self, other: "Decision") -> "Decision":
        """Merge another decision in. The stricter verdict survives.

        `escalated` records whether the incoming decision actually made
        things stricter, which is the only direction a judge is allowed
        to move the outcome.
        """
        merged = self.verdict.combine(other.verdict)
        return Decision(
            verdict=merged,
            # Attribute the outcome to whichever tier set the final
            # severity. If the judge merely agreed, the deterministic
            # layer is still what decided.
            tier=other.tier if other.verdict > self.verdict else self.tier,
            reasons=self.reasons + other.reasons,
            judge_model=other.judge_model or self.judge_model,
            judge_latency_ms=other.judge_latency_ms or self.judge_latency_ms,
            escalated=self.escalated or other.verdict > self.verdict,
        )

    def why(self) -> str:
        return "; ".join(self.reasons) if self.reasons else str(self.verdict)

    def to_audit(self) -> dict:
        out = {
            "verdict": str(self.verdict),
            "decided_by": str(self.tier),
            "reasons": list(self.reasons),
        }
        if self.judge_model:
            out["judge_model"] = self.judge_model
            out["judge_latency_ms"] = self.judge_latency_ms
            out["judge_escalated"] = self.escalated
        return out


def allow(reason: str = "") -> Decision:
    return Decision(Verdict.ALLOW, reasons=[reason] if reason else [])


def needs_human(reason: str, tier: Tier = Tier.DETERMINISTIC) -> Decision:
    return Decision(Verdict.HITL, tier=tier, reasons=[reason])


def deny(reason: str, tier: Tier = Tier.DETERMINISTIC) -> Decision:
    return Decision(Verdict.DENY, tier=tier, reasons=[reason])
