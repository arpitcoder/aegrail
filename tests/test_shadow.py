"""Shadow mode — observe without enforcing, then propose a contract.

This is the one feature that deliberately turns enforcement off, so the tests
matter more here than anywhere else. Two properties are load-bearing:

  1. Shadow mode never blocks. If it can block, an onboarding customer's agent
     breaks during the window we promised was observation-only.

  2. Shadow mode is never mistaken for enforcement. A recorded denial that did
     not stop anything must not be readable as one that did — otherwise the
     audit chain lies, which is the whole product.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import yaml

from aegrail import Agent, AuditSink, Budget
from aegrail.argrules import ArgRuleSet
from aegrail.capability import CapabilityGrant
from aegrail.exceptions import PolicyDenied
from aegrail.shadow import Mode, ObservationStore
from aegrail.verdict import Verdict, deny, needs_human, allow


DENY_EVERYTHING = [
    {"tool": "*", "arg": "danger", "eq": "yes", "verdict": "deny",
     "reason": "a rule that would certainly deny in enforce mode"},
]


def _agent(tmp_path, mode=Mode.ENFORCE, **kw):
    return Agent(
        identity="shadow-test/v1",
        budget=Budget(usd=1e6, wall_seconds=1e6),
        audit=AuditSink.file(str(tmp_path / "audit.jsonl")),
        poll_policy=False,
        resume_pending=False,
        mode=mode,
        **kw,
    )


def _events(tmp_path):
    p = tmp_path / "audit.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ── property 1: shadow never blocks ──────────────────────────────────
def test_enforce_mode_blocks(tmp_path):
    """Control. Without this passing, the shadow test below proves nothing."""
    a = _agent(tmp_path, arg_rules=ArgRuleSet.from_list(DENY_EVERYTHING))

    def risky(danger: str = "no") -> str:
        return "ran"

    a.tool(description="risky")(risky)
    with pytest.raises(PolicyDenied):
        a.call_tool("risky", danger="yes")


def test_shadow_mode_executes_what_enforce_would_deny(tmp_path):
    a = _agent(tmp_path, mode=Mode.SHADOW,
               arg_rules=ArgRuleSet.from_list(DENY_EVERYTHING))

    def risky(danger: str = "no") -> str:
        return "ran"

    a.tool(description="risky")(risky)
    assert a.call_tool("risky", danger="yes") == "ran"


def test_shadow_mode_does_not_block_on_capability(tmp_path):
    """Layer 2 denies hardest of all, so it is the one most likely to leak."""
    a = _agent(tmp_path, mode=Mode.SHADOW,
               capability=CapabilityGrant.from_spec({"allowed_tools": ["permitted"]}))

    def forbidden() -> str:
        return "ran"

    a.tool(description="forbidden")(forbidden)
    assert a.call_tool("forbidden") == "ran"


def test_shadow_mode_never_asks_for_human_approval(tmp_path):
    """A HITL freeze during an observation window is still an outage.

    The agent would block for the timeout and then deny — exactly the
    disruption shadow mode exists to avoid.
    """
    a = _agent(tmp_path, mode=Mode.SHADOW, hitl_timeout_seconds=1)

    def irreversible() -> str:
        return "ran"

    a.tool(destructive=True, description="irreversible")(irreversible)
    assert a.call_tool("irreversible") == "ran"


# ── property 2: shadow is never mistaken for enforcement ─────────────
def test_would_deny_is_a_distinct_audit_event(tmp_path):
    """`tool.would_deny` must not be readable as `tool.denied`."""
    a = _agent(tmp_path, mode=Mode.SHADOW,
               arg_rules=ArgRuleSet.from_list(DENY_EVERYTHING))

    def risky(danger: str = "no") -> str:
        return "ran"

    a.tool(description="risky")(risky)
    a.call_tool("risky", danger="yes")

    kinds = [e["kind"] for e in _events(tmp_path)]
    assert "tool.would_deny" in kinds
    assert "tool.denied" not in kinds, "a shadow observation was logged as a real denial"
    # And the call did happen, so it must also be recorded as executed.
    assert "tool.executed" in kinds


def test_shadow_events_carry_the_flag(tmp_path):
    a = _agent(tmp_path, mode=Mode.SHADOW,
               arg_rules=ArgRuleSet.from_list(DENY_EVERYTHING))

    def risky(danger: str = "no") -> str:
        return "ran"

    a.tool(description="risky")(risky)
    a.call_tool("risky", danger="yes")

    would = [e for e in _events(tmp_path) if e["kind"] == "tool.would_deny"]
    assert would and would[0].get("shadow") is True


def test_session_start_declares_the_mode(tmp_path):
    """Someone reading the chain from the beginning must know immediately."""
    _agent(tmp_path, mode=Mode.SHADOW)
    start = [e for e in _events(tmp_path) if e["kind"] == "session.start"][0]
    assert start["mode"] == "shadow"
    assert start["shadow"] is True


def test_constructing_in_shadow_mode_warns(tmp_path):
    with pytest.warns(UserWarning, match="SHADOW mode"):
        _agent(tmp_path, mode=Mode.SHADOW)


def test_enforce_mode_is_the_default_and_is_silent(tmp_path):
    import warnings as w

    with w.catch_warnings():
        w.simplefilter("error")   # any warning becomes a failure
        a = _agent(tmp_path)
    assert a.mode is Mode.ENFORCE
    assert a.observations is None


def test_shadow_mode_cannot_be_switched_on_by_environment(tmp_path, monkeypatch):
    """No env-var shortcut, deliberately.

    A stray variable in a deployment manifest must not be able to silently
    disable enforcement across a fleet.
    """
    for var in ("AEGRAIL_SHADOW", "AEGRAIL_MODE", "MAVOPS_SHADOW", "SHADOW"):
        monkeypatch.setenv(var, "1")
    assert _agent(tmp_path).mode is Mode.ENFORCE


# ── the observations themselves ──────────────────────────────────────
def test_records_tools_arguments_and_would_be_verdicts(tmp_path):
    a = _agent(tmp_path, mode=Mode.SHADOW,
               arg_rules=ArgRuleSet.from_list(DENY_EVERYTHING))

    def query(limit: int = 10, danger: str = "no") -> str:
        return "ok"

    a.tool(description="query")(query)
    for i in (5, 50, 500):
        a.call_tool("query", limit=i)
    a.call_tool("query", limit=1, danger="yes")

    obs = a.observations
    assert obs.total_calls == 4
    t = obs.tools["query"]
    assert t.calls == 4
    assert t.would["ALLOW"] == 3 and t.would["DENY"] == 1
    assert t.args["limit"].numeric_min == 1
    assert t.args["limit"].numeric_max == 500


def test_absent_arguments_are_recorded_as_absent():
    """Optionality is what decides whether a rule can require an argument."""
    store = ObservationStore("x")
    store.record("t", False, {"a": 1, "b": 2}, allow())
    store.record("t", False, {"a": 1}, allow())
    assert store.tools["t"].args["a"].always_present
    assert not store.tools["t"].args["b"].always_present
    assert store.tools["t"].args["b"].missing == 1


def test_booleans_are_not_treated_as_numbers():
    """bool subclasses int; a threshold of 'gt: 0.5' on a flag is nonsense."""
    store = ObservationStore("x")
    for _ in range(5):
        store.record("t", False, {"flag": True}, allow())
    assert store.tools["t"].args["flag"].numeric_max is None


def test_sequences_are_recorded():
    store = ObservationStore("x")
    for _ in range(3):
        store.record("read", False, {}, allow())
        store.record("send", False, {}, allow())
    assert store.sequences[("read", "send")] == 3


def test_summary_names_what_would_have_been_stopped(tmp_path):
    a = _agent(tmp_path, mode=Mode.SHADOW,
               arg_rules=ArgRuleSet.from_list(DENY_EVERYTHING))

    def risky(danger: str = "no") -> str:
        return "ran"

    a.tool(description="risky")(risky)
    a.call_tool("risky", danger="yes")
    text = a.observations.summary()
    assert "risky" in text and "deny" in text.lower()


# ── the proposal ─────────────────────────────────────────────────────
def _observed_store():
    store = ObservationStore("claims-agent/v1")
    for i in range(30):
        store.record("read_claim", False, {"claim_id": f"C-{i}"}, allow())
        store.record("approve_payout", False,
                     {"amount": 100 + i * 10, "currency": "USD"}, allow())
    return store


def test_proposed_pack_is_valid_yaml_and_parses_as_a_contract():
    """The payoff. A proposal nobody can load is worthless."""
    text = _observed_store().propose_pack(domain="claims")
    spec = yaml.safe_load(text)
    # It must survive the same parse-time guards as a hand-written pack.
    ArgRuleSet.from_list(spec.get("argumentRules"))
    CapabilityGrant.from_spec({"allowed_tools": spec.get("allowedTools")})


def test_proposed_grant_is_exactly_what_was_observed():
    spec = yaml.safe_load(_observed_store().propose_pack())
    assert sorted(spec["allowedTools"]) == ["approve_payout", "read_claim"]


def test_proposed_threshold_sits_above_the_observed_range():
    """A threshold at or below what the agent already did blocks normal work."""
    spec = yaml.safe_load(_observed_store().propose_pack())
    payout = [r for r in spec["argumentRules"]
              if r["tool"] == "approve_payout" and r.get("arg") == "amount"]
    assert payout, "no threshold proposed for an argument with a clear numeric range"
    assert payout[0]["gt"] > 390, "threshold at or below the observed maximum"


def test_proposal_does_not_invent_rules_from_thin_evidence():
    store = ObservationStore("x")
    store.record("rare", False, {"n": 5}, allow())
    spec = yaml.safe_load(store.propose_pack(min_calls=5)) or {}
    assert not spec.get("argumentRules"), "proposed a rule from a single observation"


def test_proposal_does_not_turn_normal_sequences_into_deny_rules():
    """Observed pairs are the agent's normal behaviour.

    A sequence rule is about what should NOT happen, which observation cannot
    establish — so these may only appear as commentary.
    """
    store = ObservationStore("x")
    for _ in range(20):
        store.record("read", False, {}, allow())
        store.record("send", False, {}, allow())
    spec = yaml.safe_load(store.propose_pack()) or {}
    assert not spec.get("sequenceRules"), "generated a sequence rule from normal behaviour"
    assert "read -> send" in store.propose_pack()


def test_proposal_marks_itself_a_draft():
    text = _observed_store().propose_pack(domain="claims")
    assert "DRAFT" in text
    assert "not a policy" in text


def test_empty_observations_propose_nothing():
    assert "no observations" in ObservationStore("x").propose_pack()


def test_observations_serialise_for_the_control_plane():
    d = json.loads(_observed_store().to_json())
    assert d["total_calls"] == 60
    assert d["tools"]["approve_payout"]["args"]["amount"]["max"] == 390
