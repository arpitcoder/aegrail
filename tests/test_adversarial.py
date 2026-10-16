"""Evasions a real agent stumbles into, and an attacker reaches for.

Every case here was found by probing the shipped contracts rather than by
reading the code, and several of them worked. They are the difference between
a contract that reads correctly in review and one that enforces anything.

Two classes dominate, and both make a rule look right while doing nothing:

  wrapped arguments   `execute_sql(query=…)` is caught and
                      `execute_sql(request={"query": …})` was not, though most
                      tools wrap their arguments in exactly that way.

  sentinel values     every query API spells unbounded as a small number —
                      LIMIT -1, size 0 — so `gt: 1000` waves through precisely
                      the unbounded query it was written to stop.
"""
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from aegrail.argrules import ArgRuleSet, RuleError
from aegrail.capability import CapabilityGrant
from aegrail.sequence import SequenceTracker
from aegrail.verdict import Verdict


def _blocks(rules, tool, kwargs) -> bool:
    return rules.evaluate(tool, kwargs).verdict is not Verdict.ALLOW


# ── wrapped and nested arguments ─────────────────────────────────────
SQL = ArgRuleSet.from_list([
    {"tool": "*sql*", "arg": "query", "matches": r"(?i)drop\s+table",
     "verdict": "deny", "reason": "schema-destructive SQL"},
])


@pytest.mark.parametrize("kwargs,shape", [
    ({"query": "DROP TABLE users"}, "flat"),
    ({"request": {"query": "DROP TABLE users"}}, "wrapped in request"),
    ({"body": {"payload": {"query": "DROP TABLE users"}}}, "two levels deep"),
    ({"params": '{"query": "DROP TABLE users"}'}, "JSON encoded as a string"),
    ({"items": [{"query": "DROP TABLE users"}]}, "inside a list"),
])
def test_nesting_does_not_hide_a_dangerous_value(kwargs, shape):
    """A rule naming an argument means it wherever the tool puts it."""
    assert _blocks(SQL, "execute_sql", kwargs), f"escaped when {shape}"


def test_nesting_does_not_manufacture_false_positives():
    for kwargs in ({"query": "SELECT 1"},
                   {"request": {"query": "SELECT count(*) FROM users"}},
                   {"note": "we should never drop table users"}):
        d = SQL.evaluate("execute_sql", kwargs)
        # The last one contains the words but not as a query argument; the
        # rule is scoped to `query`, so it must not fire on prose in a note.
        if "note" in kwargs:
            assert d.verdict is Verdict.ALLOW


def test_explicit_dotted_path_is_honoured():
    """`arg: a.b` was specific on purpose; do not go hunting elsewhere."""
    rules = ArgRuleSet.from_list([
        {"tool": "t", "arg": "outer.limit", "gt": 10, "verdict": "deny", "reason": "x"},
    ])
    assert _blocks(rules, "t", {"outer": {"limit": 50}})
    # A `limit` somewhere else is a different argument and must not fire.
    assert not _blocks(rules, "t", {"unrelated": {"limit": 50}})


# ── the sentinel problem ─────────────────────────────────────────────
def test_gt_alone_misses_the_unbounded_sentinels():
    """Documents the trap rather than asserting it is fine.

    Kept as a test so nobody 'fixes' gt into meaning something else: gt is
    correct, it just does not express "cap the result set".
    """
    rules = ArgRuleSet.from_list([
        {"tool": "q", "arg": "limit", "gt": 1000, "verdict": "deny", "reason": "x"},
    ])
    assert _blocks(rules, "q", {"limit": 5000})
    assert not _blocks(rules, "q", {"limit": -1})   # LIMIT -1 == unbounded
    assert not _blocks(rules, "q", {"limit": 0})    # size 0 == unbounded


@pytest.mark.parametrize("limit,blocked", [
    (5000, True), (-1, True), (0, True), (10 ** 9, True),
    (500, False), (1, False), (1000, False),
])
def test_outside_catches_both_ends(limit, blocked):
    """`outside` says what the author meant: within these bounds or refuse."""
    rules = ArgRuleSet.from_list([
        {"tool": "q", "arg": "limit", "outside": [1, 1000],
         "verdict": "deny", "reason": "result set outside safe bounds"},
    ])
    assert _blocks(rules, "q", {"limit": limit}) is blocked


def test_outside_ignores_non_numeric_values():
    rules = ArgRuleSet.from_list([
        {"tool": "q", "arg": "limit", "outside": [1, 1000], "verdict": "deny", "reason": "x"},
    ])
    assert not _blocks(rules, "q", {"limit": "all"})


# ── type coercion ────────────────────────────────────────────────────
@pytest.mark.parametrize("value", [5000, "5000", 5000.0, 5e9])
def test_numeric_thresholds_survive_type_changes(value):
    """A model emits JSON; a number may arrive quoted."""
    rules = ArgRuleSet.from_list([
        {"tool": "q", "arg": "limit", "gt": 1000, "verdict": "deny", "reason": "x"},
    ])
    assert _blocks(rules, "q", {"limit": value})


# ── matching any argument ────────────────────────────────────────────
def test_wildcard_arg_finds_a_pattern_anywhere():
    """For a tool whose danger is a string, naming the parameter is a guess."""
    rules = ArgRuleSet.from_list([
        {"tool": "*sql*", "arg": "*", "matches": r"(?i)drop\s+table",
         "verdict": "deny", "reason": "destructive SQL anywhere in the arguments"},
    ])
    for kwargs in ({"query": "DROP TABLE t"},
                   {"statements": ["SELECT 1", "DROP TABLE t"]},
                   {"body": '{"sql":"drop table t"}'},
                   {"a": {"b": {"c": "Drop Table t"}}}):
        assert _blocks(rules, "execute_sql", kwargs), kwargs
    assert not _blocks(rules, "execute_sql", {"query": "SELECT 1"})


def test_wildcard_arg_with_a_negative_matcher_is_rejected():
    """Absence fires a negative matcher, so this would deny every call."""
    with pytest.raises(RuleError):
        ArgRuleSet.from_list([
            {"tool": "*", "arg": "*", "notIn": ["ok"], "verdict": "deny", "reason": "x"},
        ])


# ── bounded work ─────────────────────────────────────────────────────
def test_deeply_nested_payload_cannot_stall_the_check():
    """A pathological argument must not turn microseconds into seconds."""
    import time
    deep: dict = {"leaf": "x"}
    for _ in range(200):
        deep = {"n": deep}
    rules = ArgRuleSet.from_list([
        {"tool": "t", "arg": "leaf", "eq": "x", "verdict": "deny", "reason": "x"},
    ])
    t0 = time.perf_counter()
    rules.evaluate("t", deep)
    assert time.perf_counter() - t0 < 0.05


def test_self_referential_arguments_do_not_recurse_forever():
    loop: dict = {"a": 1}
    loop["self"] = loop
    rules = ArgRuleSet.from_list([
        {"tool": "t", "arg": "missing", "eq": "x", "verdict": "deny", "reason": "x"},
    ])
    rules.evaluate("t", loop)   # must return rather than blow the stack


# ── tool identity ────────────────────────────────────────────────────
@pytest.mark.parametrize("name", [
    "READ_CLAIM", "Read_Claim", " read_claim", "read_claim ",
    "read_claim\n", "read_claim​", "read__claim",
])
def test_capability_grant_is_not_fooled_by_lookalike_names(name):
    """Only the exact tool is granted; a near-miss is a different tool."""
    g = CapabilityGrant.from_spec({"allowed_tools": ["read_claim"]})
    assert not g.permits(name)


# ── trajectory ───────────────────────────────────────────────────────
def test_sequence_rule_holds_under_concurrent_evaluation():
    seq = SequenceTracker.from_list([
        {"afterAny": ["read_*"], "then": "http_*", "verdict": "deny",
         "withinSeconds": 600, "reason": "exfil"},
    ])
    seq.record("read_claim")
    seen = []
    threads = [threading.Thread(target=lambda: seen.append(
        seq.evaluate("http_post").verdict)) for _ in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(seen) == {Verdict.DENY}, "a concurrent caller slipped through"


def test_a_blocked_read_does_not_arm_a_sequence_rule():
    """The agent never got the data, so nothing downstream is tainted."""
    seq = SequenceTracker.from_list([
        {"afterAny": ["read_*"], "then": "http_*", "verdict": "deny",
         "withinSeconds": 600, "reason": "exfil"},
    ])
    # record() is only called on success, so nothing is recorded here.
    assert seq.evaluate("http_post").verdict is Verdict.ALLOW


def test_retrying_a_denied_call_stays_denied():
    """Agents retry. Varying the wrapper must not wear the rule down."""
    for kwargs in (
        {"query": "DROP TABLE users"},
        {"query": "drop table users"},
        {"query": "DROP  TABLE   users"},
        {"request": {"query": "DROP TABLE users"}},
        {"query": "DROP\tTABLE users"},
    ):
        assert _blocks(SQL, "execute_sql", kwargs), kwargs
