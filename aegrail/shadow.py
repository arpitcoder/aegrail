"""Shadow mode — watch an agent, then write its contract from evidence.

You cannot write argument rules for a domain you have never seen. The
observability and insurance packs were written from research: the tool names
in them are patterns because we were guessing at somebody else's naming. That
does not scale past a handful of customers, and for an agent nobody has
classified it does not work at all.

It also gets the first conversation wrong. Arriving with a policy that blocks
an unfamiliar agent is how governance gets switched off in week one. Arriving
with "let us watch for a week and show you what it does" is a different
conversation, and it ends with rules derived from evidence instead of
inference.

So: run every layer exactly as it would run in production, record what each
one *would* have decided, and execute the call regardless. At the end,
`propose_pack()` turns the observations into a starter contract.

Two things this module treats as non-negotiable.

**Shadow mode must be impossible to mistake for enforcement.** A governance
product that silently is not governing is worse than no product, because it
buys false confidence. So the mode is an explicit constructor argument with no
environment-variable shortcut, it is announced at construction, every audit
record carries `shadow: true`, and the event kind itself changes —
`tool.would_deny` is not `tool.denied`, and no dashboard or auditor will
confuse them.

**The proposal is a draft, not a policy.** It is derived from what an agent
did during a window in which nothing was blocked, so it encodes the observed
behaviour including anything already wrong with it. Every generated rule says
what it was derived from and how many calls it saw, so a reviewer can tell a
rule backed by 4,000 observations from one backed by two.
"""
from __future__ import annotations

import enum
import json
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional



class Mode(enum.Enum):
    """How the agent treats the verdicts it computes."""

    ENFORCE = "enforce"
    SHADOW = "shadow"

    @property
    def enforces(self) -> bool:
        return self is Mode.ENFORCE


# Cap what we retain per argument. An agent running for a week makes a lot of
# calls, and the point is to learn the shape of the values, not to accumulate
# a copy of the customer's data in memory.
MAX_SAMPLES_PER_ARG = 20
MAX_DISTINCT_TRACKED = 200


@dataclass
class ArgObservation:
    """What one argument of one tool actually looked like."""

    name: str
    seen: int = 0
    missing: int = 0
    types: Counter = field(default_factory=Counter)
    # Only for numerics — the range is what a threshold rule needs.
    numeric_min: Optional[float] = None
    numeric_max: Optional[float] = None
    # Bounded sample of distinct values. Useful for spotting an argument that
    # is really an enum, which becomes an `in` rule rather than a threshold.
    distinct: set = field(default_factory=set)
    truncated: bool = False

    def record(self, value: Any) -> None:
        if value is _ABSENT:
            self.missing += 1
            return
        self.seen += 1
        self.types[type(value).__name__] += 1
        if isinstance(value, bool):
            # bool is an int subclass; treating it as numeric produces
            # nonsense thresholds like "gt: 0.5".
            pass
        elif isinstance(value, (int, float)):
            self.numeric_min = value if self.numeric_min is None else min(self.numeric_min, value)
            self.numeric_max = value if self.numeric_max is None else max(self.numeric_max, value)
        if len(self.distinct) < MAX_SAMPLES_PER_ARG:
            try:
                self.distinct.add(value if isinstance(value, (str, int, float, bool)) else str(type(value).__name__))
            except TypeError:
                self.distinct.add("<unhashable>")
        else:
            self.truncated = True

    @property
    def always_present(self) -> bool:
        return self.missing == 0 and self.seen > 0

    @property
    def looks_enumerated(self) -> bool:
        """Few distinct values across many calls — a candidate `in` rule."""
        return (
            not self.truncated
            and 0 < len(self.distinct) <= 8
            and self.seen >= 10
            and all(isinstance(v, str) for v in self.distinct)
        )


@dataclass
class ToolObservation:
    name: str
    calls: int = 0
    destructive: bool = False
    args: dict = field(default_factory=dict)
    # Verdicts the layers would have produced, had anything been enforcing.
    would: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)

    def record(self, kwargs: dict, decision) -> None:
        self.calls += 1
        self.would[decision.verdict.name] += 1
        for r in (decision.reasons or []):
            self.reasons[r[:120]] += 1

        seen_now = set(kwargs)
        for k, v in kwargs.items():
            self.args.setdefault(k, ArgObservation(k)).record(v)
        # An argument absent from this call but seen before is evidence about
        # optionality, which is what decides whether a rule can require it.
        for k, obs in self.args.items():
            if k not in seen_now:
                obs.record(_ABSENT)


class _Absent:
    def __repr__(self):
        return "<absent>"


_ABSENT = _Absent()


class ObservationStore:
    """Everything shadow mode learned, and the pack it can propose."""

    def __init__(self, identity: str = ""):
        self.identity = identity
        self.tools: dict = {}
        self.sequences: Counter = Counter()
        self._last_tool: Optional[str] = None
        self._lock = threading.Lock()

    def record(self, tool: str, destructive: bool, kwargs: dict, decision) -> None:
        with self._lock:
            obs = self.tools.get(tool)
            if obs is None:
                if len(self.tools) >= MAX_DISTINCT_TRACKED:
                    return
                obs = self.tools[tool] = ToolObservation(tool, destructive=destructive)
            obs.destructive = obs.destructive or destructive
            obs.record(dict(kwargs), decision)

            if self._last_tool is not None:
                self.sequences[(self._last_tool, tool)] += 1
            self._last_tool = tool

    # ── reading back ────────────────────────────────────────────────
    @property
    def total_calls(self) -> int:
        return sum(t.calls for t in self.tools.values())

    def summary(self) -> str:
        """What a human reads at the end of an observation window."""
        if not self.tools:
            return "no tool calls observed"

        lines = [
            f"shadow observations for {self.identity or 'agent'}",
            f"  {self.total_calls} calls across {len(self.tools)} tools",
            "",
            f"  {'tool':28} {'calls':>6}  would have been",
        ]
        for name, t in sorted(self.tools.items(), key=lambda kv: -kv[1].calls):
            verdicts = ", ".join(f"{v.lower()}×{n}" for v, n in t.would.most_common())
            lines.append(f"  {name:28} {t.calls:>6}  {verdicts}")

        blocked = {n: t for n, t in self.tools.items() if t.would.get("DENY") or t.would.get("HITL")}
        if blocked:
            lines += ["", "  calls a policy would have stopped:"]
            for n, t in blocked.items():
                for reason, count in t.reasons.most_common(3):
                    lines.append(f"    {n}: {reason} (×{count})")
        return "\n".join(lines)

    def propose_pack(self, *, domain: str = "observed", min_calls: int = 5) -> str:
        """Generate a starter contract from what was actually seen.

        Deliberately conservative. It proposes the capability grant — which is
        purely factual, being the observed tool inventory — and thresholds only
        where there is enough data to mean anything. It does not invent
        argument rules for values it saw twice.

        Every rule carries the evidence it came from, so a reviewer can weigh
        one backed by thousands of calls against one backed by a handful.
        """
        if not self.tools:
            return "# no observations — nothing to propose\n"

        out = [
            f"# {domain} — DRAFT, generated from shadow observations",
            "#",
            f"# Derived from {self.total_calls} calls across {len(self.tools)} tools",
            f"# by agent '{self.identity}'.",
            "#",
            "# This is a starting point, not a policy. It was generated from a",
            "# window in which nothing was enforced, so it describes what the",
            "# agent DID — including anything already wrong with that. Review",
            "# every threshold before enabling it, and read the observed ranges",
            "# as evidence rather than as intent.",
            "#",
            "# Layers on top of infra-baseline.yaml. Do not repeat its rules here.",
            "",
            "# ── Layer 2: capability grant ─────────────────────────────────",
            "# Factual: the tools this agent actually used. Remove anything it",
            "# should not have, rather than adding — the point of an allowlist",
            "# is that someone decided, and observation cannot decide for you.",
            "allowedTools:",
        ]
        width = max(len(n) for n in self.tools)
        for name in sorted(self.tools):
            t = self.tools[name]
            note = ", destructive" if t.destructive else ""
            out.append(f"  - {name:<{width}}   # {t.calls} calls{note}")

        rules = []
        for name in sorted(self.tools):
            t = self.tools[name]
            if t.calls < min_calls:
                continue
            for arg_name, a in sorted(t.args.items()):
                # Numeric range → a threshold rule, set just above what was
                # observed, because the observed maximum is by definition
                # something the agent did in normal operation.
                if a.numeric_max is not None and a.seen >= min_calls and a.numeric_max > 0:
                    ceiling = _round_up(a.numeric_max)
                    rules.append(
                        f'  # observed {a.seen}x, range {a.numeric_min}–{a.numeric_max}\n'
                        f'  - tool: "{name}"\n'
                        f'    arg: "{arg_name}"\n'
                        f'    gt: {ceiling}\n'
                        f'    verdict: "hitl"\n'
                        f'    reason: "{arg_name} above the observed operating range '
                        f'(max seen {a.numeric_max} over {a.seen} calls)"'
                    )
                elif a.looks_enumerated:
                    vals = ", ".join(f'"{v}"' for v in sorted(a.distinct))
                    rules.append(
                        f'  # observed {a.seen}x, only these values\n'
                        f'  - tool: "{name}"\n'
                        f'    arg: "{arg_name}"\n'
                        f'    notIn: [{vals}]\n'
                        f'    onlyIfPresent: true\n'
                        f'    verdict: "hitl"\n'
                        f'    reason: "{arg_name} outside the observed set of values"'
                    )
                elif a.always_present and a.seen >= min_calls:
                    rules.append(
                        f'  # present on all {a.seen} observed calls\n'
                        f'  - tool: "{name}"\n'
                        f'    arg: "{arg_name}"\n'
                        f'    present: false\n'
                        f'    verdict: "hitl"\n'
                        f'    reason: "{arg_name} missing, though every observed call supplied it"'
                    )

        out.append("")
        if rules:
            out.append("argumentRules:")
            out.append("")
            out.append("\n\n".join(rules))
        else:
            out.append("# argumentRules: not enough data to propose any thresholds.")
            out.append("# Observe for longer, or lower min_calls.")

        # Sequence pairs are proposed as comments only. A frequently observed
        # pair is normal behaviour, so turning it into a rule would block the
        # agent's ordinary work — the useful sequence rules are about pairs
        # that did NOT occur, which observation cannot tell you.
        common = [p for p in self.sequences.most_common(8) if p[1] >= min_calls]
        if common:
            out += ["", "# Observed tool sequences, most frequent first. These are the agent's",
                    "# NORMAL trajectories — do not turn them into deny rules. They are here",
                    "# to help you spot which pairs would be abnormal, which is the question",
                    "# a sequence rule actually answers and observation cannot.",
                    "#"]
            for (a, b), n in common:
                out.append(f"#   {a} -> {b}   ({n}x)")
            out += ["#", "# sequenceRules: []"]

        return "\n".join(out) + "\n"

    def to_dict(self) -> dict:
        return {
            "identity": self.identity,
            "total_calls": self.total_calls,
            "tools": {
                n: {
                    "calls": t.calls,
                    "destructive": t.destructive,
                    "would": dict(t.would),
                    "args": {
                        a: {
                            "seen": o.seen,
                            "missing": o.missing,
                            "types": dict(o.types),
                            "min": o.numeric_min,
                            "max": o.numeric_max,
                        }
                        for a, o in t.args.items()
                    },
                }
                for n, t in self.tools.items()
            },
            "sequences": {f"{a}->{b}": n for (a, b), n in self.sequences.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)


def _round_up(v: float) -> float:
    """A threshold just above the observed maximum, at a readable number."""
    if v <= 0:
        return 1
    import math

    mag = 10 ** math.floor(math.log10(v))
    for mult in (1, 1.5, 2, 3, 5, 10):
        c = mag * mult
        if c > v:
            return int(c) if c == int(c) else c
    return int(mag * 10)
