"""Tier 2 — deterministic checks on the arguments themselves.

The layer most agent-governance products skip. They jump from "is this
tool allowed" straight to "let's ask a model", which spends a second and
a fraction of a cent to answer questions a regex already answers:

    execute_sql   query matches /\\b(DROP|TRUNCATE)\\b/   -> needs a human
    transfer_funds amount > 10_000                       -> needs a human
    write_file    path under /etc                        -> deny

These are declarative, evaluated in microseconds with no I/O, and — the
part that matters for an evidence pack — perfectly replayable. An auditor
re-runs them a year later and gets the identical answer, which is not
true of anything a model decided.

Rules may only escalate. A rule cannot mark something ALLOW, because
Tier 2 exists to catch things the structural layer let through, never to
overrule it. Attempting to write one is a configuration error and is
rejected when the rule is parsed rather than silently ignored later.
"""
from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .verdict import Decision, Tier, Verdict, allow

# Regexes come from a contract and are matched against agent-controlled
# strings, which is the classic setup for catastrophic backtracking. The
# cap bounds the damage: a pathological pattern is slow over 8 KB, not
# over a 10 MB payload someone pasted into a tool argument.
MAX_MATCH_CHARS = 8192

# ${VAR} or {{VAR}} left in a contract that was never templated.
_PLACEHOLDER = re.compile(r"\$\{[^}]+\}|\{\{[^}]+\}\}")


class RuleError(ValueError):
    """A rule that cannot be honoured as written."""


# How far into a nested argument to look, and how much of it to look at.
# Bounded because this runs on every call: a pathological payload must not be
# able to turn a microsecond check into a traversal of an arbitrary object
# graph.
_MAX_DEPTH = 6
_MAX_NODES = 256
# Only strings that plausibly hold JSON are parsed, and only short ones.
_MAX_JSON_PARSE = 4096


def _resolve(args: dict, path: str) -> Any:
    """Look up 'a.b.c' in nested arguments. Missing returns a sentinel."""
    cur: Any = args
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


def _leaves(args: Any) -> list:
    """Every scalar in the argument tree, bounded."""
    out: list = []
    budget = [_MAX_NODES]

    def walk(node: Any, depth: int) -> None:
        if depth > _MAX_DEPTH or budget[0] <= 0 or len(out) >= 64:
            return
        budget[0] -= 1
        if isinstance(node, dict):
            for v in node.values():
                walk(v, depth + 1)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v, depth + 1)
        elif isinstance(node, str) and node[:1] in ("{", "[") and len(node) <= _MAX_JSON_PARSE:
            try:
                walk(json.loads(node), depth + 1)
            except Exception:
                out.append(node)
        else:
            out.append(node)

    walk(args, 0)
    return out


def _resolve_all(args: dict, path: str) -> list:
    """Every value this rule could plausibly be about.

    An exact path match comes first and is usually the only one. Failing that,
    the argument tree is searched for the leaf key, because a great many tools
    wrap their real arguments:

        execute_sql(query="DROP TABLE users")                  caught
        execute_sql(request={"query": "DROP TABLE users"})     was not

    A rule written as `arg: query` plainly means that argument wherever the
    tool happens to put it. Matching only the top level meant a contract could
    look correct, read correctly in review, and enforce nothing against any
    tool with a params/body/request wrapper — which is most of them.

    Returning candidates rather than one value keeps absence honest: an empty
    result becomes _MISSING, so negative matchers still fire on a genuinely
    absent argument.
    """
    # arg: "*" means "any argument value, wherever it lives".
    #
    # For a tool whose danger is in a string rather than in a named parameter,
    # naming the parameter is a guess. execute_sql might take query=…,
    # statements=[…] or request={"sql": …}, and a rule pinned to one of those
    # silently misses the others. Only safe with a positive matcher — the
    # parse-time guard rejects a broad negative one, which would fire on every
    # call that simply lacked the argument.
    if path == "*":
        return _leaves(args) or [_MISSING]

    exact = _resolve(args, path)
    if exact is not _MISSING:
        return [exact]
    # A dotted path was explicit about where to look; do not go hunting.
    if "." in path:
        return [_MISSING]

    found: list = []
    budget = [_MAX_NODES]

    def walk(node: Any, depth: int) -> None:
        if depth > _MAX_DEPTH or budget[0] <= 0 or len(found) >= 8:
            return
        budget[0] -= 1
        if isinstance(node, dict):
            for k, v in node.items():
                if k == path:
                    found.append(v)
                walk(v, depth + 1)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v, depth + 1)
        elif isinstance(node, str) and node[:1] in ("{", "[") and len(node) <= _MAX_JSON_PARSE:
            # A JSON blob passed as a string is the same evasion one encoding
            # further out.
            try:
                walk(json.loads(node), depth + 1)
            except Exception:
                pass

    walk(args, 0)
    return found or [_MISSING]


class _Missing:
    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()


@dataclass
class ArgRule:
    """One declarative condition over a tool's arguments."""

    tool: str                       # exact name, or a glob like "db_*"
    arg: str = ""                   # argument name, dotted for nesting
    verdict: Verdict = Verdict.HITL
    reason: str = ""

    # Matchers. Exactly one is expected; more than one is an AND.
    matches: Optional[str] = None   # regex
    contains: Optional[str] = None
    glob: Optional[str] = None
    gt: Optional[float] = None
    gte: Optional[float] = None
    lt: Optional[float] = None
    lte: Optional[float] = None
    eq: Optional[Any] = None
    ne: Optional[Any] = None
    in_: Optional[list] = None
    not_in: Optional[list] = None
    # Fires when the value falls OUTSIDE [lo, hi].
    #
    # `gt: 1000` on a result limit reads as "cap the result set" and does not
    # do that, because every query API spells unbounded as a sentinel:
    # LIMIT -1, size 0, max_results 0. Those are smaller than the threshold, so
    # a gt rule waves through exactly the unbounded query it was written to
    # stop. `outside: [1, 1000]` says the thing the author meant.
    outside: Optional[list] = None
    present: Optional[bool] = None
    # Suppresses the absence-fires behaviour of ne / notIn, for arguments
    # that are genuinely optional rather than a safety marker.
    only_if_present: bool = False

    _compiled: Optional[Any] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self._reject_unexpanded_placeholders()
        if self.verdict is Verdict.ALLOW:
            raise RuleError(
                f"rule for '{self.tool}' has verdict=allow. Argument rules may "
                f"only escalate — an allow here would overrule the structural "
                f"layer, which is the opposite of what this tier is for."
            )
        if self.matches is not None:
            try:
                self._compiled = re.compile(self.matches)
            except re.error as exc:
                # Fail at parse time, not on the tool call that needed it.
                raise RuleError(
                    f"rule for '{self.tool}' has an invalid regex "
                    f"{self.matches!r}: {exc}"
                ) from exc
        self._guard_broad_negative_matcher()
        if not self.reason:
            self.reason = f"argument rule on {self.tool}.{self.arg or '*'}"

    def _reject_unexpanded_placeholders(self) -> None:
        """A ${...} that was never substituted is a silent policy hole.

        It does not error — it becomes a literal string that no real
        value ever equals. In a positive matcher the rule then never
        fires, and in a negative one it fires on everything. Both are
        worse than a rule that refuses to load.
        """
        for name in ("matches", "contains", "glob", "eq", "ne", "in_", "not_in"):
            value = getattr(self, name, None)
            candidates = value if isinstance(value, (list, tuple)) else [value]
            for c in candidates:
                if isinstance(c, str) and _PLACEHOLDER.search(c):
                    raise RuleError(
                        f"rule for '{self.tool}' contains an unexpanded "
                        f"placeholder {c!r}. Substitute it before loading the "
                        f"contract — left as-is it matches nothing, which "
                        f"silently disables the rule or denies every call."
                    )

    def _guard_broad_negative_matcher(self) -> None:
        """A negative matcher on every tool denies calls that never had
        the argument at all.

        Absence fires a negative matcher, by design — that is what makes
        "escalate anything not tagged ephemeral" catch untagged
        resources. Combined with tool "*" it also catches every call that
        was never about that argument, which denies the whole product.
        """
        broad = self.tool in ("*", "**")
        negative = self.ne is not None or self.not_in is not None
        # only_if_present suppresses the absence-fires behaviour, which is
        # exactly the remedy this error recommends — so a rule that already
        # sets it is safe. Without this clause the guard was unsatisfiable:
        # following its own advice still raised.
        if broad and negative and self.arg and not self.only_if_present:
            raise RuleError(
                f"rule on argument '{self.arg}' applies to every tool and uses "
                f"a negative matcher. Absence fires a negative matcher, so this "
                f"denies every call that never had '{self.arg}' at all. Scope "
                f"it to the tools that take that argument, or add "
                f"onlyIfPresent: true."
            )

    # ── matching ────────────────────────────────────────────────────
    def applies_to(self, tool: str) -> bool:
        return self.tool == tool or fnmatch.fnmatch(tool, self.tool)

    def _value_matches(self, value: Any) -> bool:
        if self.present is not None:
            return (value is not _MISSING) == self.present

        if value is _MISSING:
            # Absence means different things to different matchers.
            #
            # For a positive matcher — eq, contains, matches — an argument
            # that was not supplied simply is not what we are looking for,
            # so the rule does not fire.
            #
            # For a negative one — ne, notIn — absence IS the condition.
            # "escalate anything whose tags.ephemeral is not 'true'" has
            # to catch a resource carrying no tags at all, or the safest
            # rule anyone writes ("deny unless positively marked safe")
            # would pass exactly the unlabelled resources it exists to
            # stop.
            if self.only_if_present:
                return False
            return self.ne is not None or self.not_in is not None

        if self.eq is not None and value != self.eq:
            return False
        if self.ne is not None and value == self.ne:
            return False
        if self.in_ is not None and value not in self.in_:
            return False
        if self.not_in is not None and value in self.not_in:
            return False

        if self.outside is not None:
            try:
                num = float(value)
            except (TypeError, ValueError):
                # A range against a non-numeric argument cannot be satisfied.
                return False
            lo, hi = float(self.outside[0]), float(self.outside[1])
            if lo <= num <= hi:
                return False

        if any(b is not None for b in (self.gt, self.gte, self.lt, self.lte)):
            try:
                num = float(value)
            except (TypeError, ValueError):
                # A numeric bound against a non-numeric argument is a
                # rule that cannot be satisfied, not a violation.
                return False
            if self.gt is not None and not num > self.gt:
                return False
            if self.gte is not None and not num >= self.gte:
                return False
            if self.lt is not None and not num < self.lt:
                return False
            if self.lte is not None and not num <= self.lte:
                return False

        if self.contains is not None or self._compiled is not None or self.glob is not None:
            text = value if isinstance(value, str) else str(value)
            text = text[:MAX_MATCH_CHARS]
            if self.contains is not None and self.contains not in text:
                return False
            if self._compiled is not None and not self._compiled.search(text):
                return False
            if self.glob is not None and not fnmatch.fnmatch(text, self.glob):
                return False

        return True

    def evaluate(self, tool: str, args: dict) -> Optional[Decision]:
        """Return a Decision if this rule fires, else None."""
        if not self.applies_to(tool):
            return None
        if self.arg:
            candidates = _resolve_all(args, self.arg)
            value = next((c for c in candidates if self._value_matches(c)), None)
            if value is None:
                return None
        else:
            value = args
            if not self._value_matches(value):
                return None

        shown = repr(value)
        if len(shown) > 120:
            shown = shown[:117] + "…"
        return Decision(
            verdict=self.verdict,
            tier=Tier.DETERMINISTIC,
            reasons=[f"{self.reason} ({self.arg}={shown})" if self.arg else self.reason],
        )

    # ── construction ────────────────────────────────────────────────
    @staticmethod
    def from_dict(raw: dict) -> "ArgRule":
        """Build from the contract's wire form.

        Accepts the YAML/JSON spelling (`in`, `notIn`) as well as the
        Python one, so a contract reads naturally in either.
        """
        d = dict(raw)
        verdict_raw = str(d.pop("verdict", "hitl")).lower()
        verdict = {
            "hitl": Verdict.HITL,
            "review": Verdict.HITL,
            "deny": Verdict.DENY,
            "block": Verdict.DENY,
            "allow": Verdict.ALLOW,  # accepted here so __post_init__ can reject it clearly
        }.get(verdict_raw)
        if verdict is None:
            raise RuleError(f"unknown verdict {verdict_raw!r}")

        alias = {
            "in": "in_",
            "notIn": "not_in",
            "not_in": "not_in",
            "onlyIfPresent": "only_if_present",
        }
        kwargs = {alias.get(k, k): v for k, v in d.items()}

        known = {f for f in ArgRule.__dataclass_fields__ if not f.startswith("_")}
        unknown = set(kwargs) - known
        if unknown:
            # Silently ignoring a misspelled matcher would mean a rule
            # that looks present and enforces nothing.
            raise RuleError(f"unknown rule field(s): {sorted(unknown)}")

        return ArgRule(verdict=verdict, **kwargs)


class ArgRuleSet:
    """All argument rules in force, evaluated together."""

    def __init__(self, rules: Optional[list] = None):
        self.rules = list(rules or [])

    @staticmethod
    def from_list(raw: Optional[list]) -> "ArgRuleSet":
        return ArgRuleSet([ArgRule.from_dict(r) for r in (raw or [])])

    def evaluate(self, tool: str, args: dict) -> Decision:
        """Every matching rule contributes; the strictest wins.

        All rules are evaluated rather than stopping at the first match,
        so the audit record names every clause the call violated instead
        of only whichever happened to be listed first.
        """
        decision = allow()
        for rule in self.rules:
            fired = rule.evaluate(tool, args)
            if fired is not None:
                decision = decision.combine(fired)
        return decision

    def covers(self, tool: str) -> bool:
        return any(r.applies_to(tool) for r in self.rules)

    def __len__(self) -> int:
        return len(self.rules)
