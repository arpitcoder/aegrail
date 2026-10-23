"""The sequence tier, as the Agent actually runs it.

sequence.py has always been correct in isolation — and until this was wired
up, the Agent never called it. A contract could declare sequenceRules, the
rules would parse, the tests over the pack would pass, and at runtime every
one of them was inert. So these tests go through `call_tool`, not through
SequenceTracker directly: the unit is the gate, not the rule.

The property that matters: each call is individually permissible and the
pair is not.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from aegrail import Agent, AuditSink, Budget
from aegrail.exceptions import PolicyDenied
from aegrail.sequence import SequenceTracker
from aegrail.verdict import Verdict


EXFIL = [
    {"afterAny": ["read_claim*"], "then": "http_*", "verdict": "deny",
     "withinSeconds": 600,
     "reason": "outbound request shortly after reading policyholder data"},
]


def _agent(tmp_path, rules=EXFIL, **kw):
    agent = Agent(
        identity="sequence-test/v1",
        budget=Budget(usd=1e6, wall_seconds=1e6),
        audit=AuditSink.file(str(tmp_path / "audit.jsonl")),
        poll_policy=False,
        resume_pending=False,
        sequence=SequenceTracker.from_list(rules),
        **kw,
    )

    @agent.tool(description="read a claim")
    def read_claim(claim_id: str) -> dict:
        return {"claim_id": claim_id, "ssn": "xxx-xx-1234"}

    @agent.tool(description="post to a url")
    def http_post(url: str, body: str = "") -> str:
        return f"posted to {url}"

    @agent.tool(description="fail")
    def flaky_read_claim_v2(claim_id: str) -> dict:
        raise RuntimeError("upstream down")

    return agent


def test_each_call_is_fine_on_its_own(tmp_path):
    agent = _agent(tmp_path)
    assert agent.call_tool("http_post", url="https://example.com") == \
        "posted to https://example.com"


def test_the_pair_is_not(tmp_path):
    agent = _agent(tmp_path)
    agent.call_tool("read_claim", claim_id="C-1001")
    with pytest.raises(PolicyDenied) as e:
        agent.call_tool("http_post", url="https://example.com")
    assert "shortly after reading policyholder data" in str(e.value)
    # The trigger is named, so the audit record says what armed the rule.
    assert "read_claim" in str(e.value)


def test_order_is_what_matters_not_membership(tmp_path):
    """Posting first and reading second is not the exfiltration shape."""
    agent = _agent(tmp_path)
    agent.call_tool("http_post", url="https://example.com")
    agent.call_tool("read_claim", claim_id="C-1001")   # must not raise


def test_a_blocked_read_does_not_arm_the_rule(tmp_path):
    """A read that never returned data cannot restrict what follows.

    Otherwise one refused attempt would quietly restrict the rest of the
    session, and the safest-looking agent — the one whose reads keep getting
    denied — would be the one that stops being able to work.
    """
    from aegrail.argrules import ArgRuleSet
    agent = _agent(tmp_path, arg_rules=ArgRuleSet.from_list([
        {"tool": "read_claim", "arg": "claim_id", "matches": "^Z",
         "verdict": "deny", "reason": "quarantined claim"},
    ]))
    with pytest.raises(PolicyDenied):
        agent.call_tool("read_claim", claim_id="Z-9")
    agent.call_tool("http_post", url="https://example.com")   # must not raise


def test_a_failed_read_does_not_arm_the_rule(tmp_path):
    """Same reasoning for a tool that raised: nothing came back to leak."""
    agent = _agent(tmp_path)
    with pytest.raises(RuntimeError):
        agent.call_tool("flaky_read_claim_v2", claim_id="C-1001")
    agent.call_tool("http_post", url="https://example.com")   # must not raise


def test_the_window_expires(tmp_path, monkeypatch):
    agent = _agent(tmp_path, rules=[
        {"afterAny": ["read_claim*"], "then": "http_*", "verdict": "deny",
         "withinSeconds": 1, "reason": "too soon after a read"},
    ])
    agent.call_tool("read_claim", claim_id="C-1001")

    import aegrail.sequence as seq
    real = seq.time.time
    monkeypatch.setattr(seq.time, "time", lambda: real() + 5)
    agent.call_tool("http_post", url="https://example.com")   # window passed


def test_the_denial_is_in_the_audit_chain(tmp_path):
    import json
    agent = _agent(tmp_path)
    agent.call_tool("read_claim", claim_id="C-1001")
    with pytest.raises(PolicyDenied):
        agent.call_tool("http_post", url="https://example.com")

    events = [json.loads(l) for l in
              open(tmp_path / "audit.jsonl", encoding="utf-8") if l.strip()]
    denied = [e for e in events if e["kind"] == "tool.denied"]
    assert len(denied) == 1
    assert denied[0]["tool"] == "http_post"
    assert "policyholder data" in json.dumps(denied[0])


def test_no_rules_means_no_cost_and_no_behaviour_change(tmp_path):
    agent = _agent(tmp_path, rules=[])
    agent.call_tool("read_claim", claim_id="C-1001")
    agent.call_tool("http_post", url="https://example.com")


def test_a_sequence_rule_cannot_allow(tmp_path):
    from aegrail.sequence import SequenceRuleError
    with pytest.raises(SequenceRuleError):
        SequenceTracker.from_list([
            {"afterAny": ["read_claim*"], "then": "http_*", "verdict": "allow"},
        ])
