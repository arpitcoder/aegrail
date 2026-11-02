"""The stop button, and what happens when nobody can press it.

A kill switch is only worth having if it cannot be defeated by the thing
going wrong. Two properties:

  1. Halted means halted — every tool, including the harmless ones.
  2. An agent that has lost the control plane does not keep its irreversible
     tools. Otherwise dropping egress disables the switch, and dropping
     egress is something a compromised agent can do to itself.

The second property has a cost, so it is bounded: reads keep working, and
the degrade only applies to an agent that is managed in the first place.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from aegrail import Agent, AuditSink, Budget
from aegrail.exceptions import PolicyDenied
from aegrail.policy import PolicyClient, PolicySnapshot


def _policy(*, halted=False, halt_reason="", age=0.0, managed=True):
    """A policy client pinned to a known snapshot and reachability."""
    c = PolicyClient(agent_name="kill-test")
    fetched = 0.0 if age == float("inf") else time.monotonic() - age
    c._snapshot = PolicySnapshot(halted=halted, halt_reason=halt_reason,
                                 fetched_at=fetched, source="test")
    type(c).managed = property(lambda self, _m=managed: _m)
    return c


def _agent(tmp_path, policy=None, **kw):
    agent = Agent(
        identity="kill-switch/v1",
        budget=Budget(usd=1e6, wall_seconds=1e6),
        audit=AuditSink.file(str(tmp_path / "audit.jsonl")),
        policy=policy or _policy(),
        resume_pending=False,
        **kw,
    )

    @agent.tool(description="read something")
    def read_doc(doc_id: str) -> str:
        return "contents"

    @agent.tool(destructive=True, description="delete something")
    def delete_doc(doc_id: str) -> str:
        return "deleted"

    return agent


def _kinds(tmp_path):
    path = tmp_path / "audit.jsonl"
    return [json.loads(l)["kind"] for l in open(path, encoding="utf-8") if l.strip()]


# ── the button ───────────────────────────────────────────────────────

def test_halt_stops_everything_including_reads(tmp_path):
    """No read-only exemption. A read feeds the next model call."""
    agent = _agent(tmp_path)
    assert agent.call_tool("read_doc", doc_id="d1") == "contents"

    agent.halt("incident 4471")
    for tool in ("read_doc", "delete_doc"):
        with pytest.raises(PolicyDenied) as e:
            agent.call_tool(tool, doc_id="d1")
        assert "halted" in str(e.value)
        assert "incident 4471" in str(e.value)


def test_resume_restores_service(tmp_path):
    agent = _agent(tmp_path)
    agent.halt()
    agent.resume()
    assert agent.call_tool("read_doc", doc_id="d1") == "contents"


def test_fleet_policy_can_halt(tmp_path):
    agent = _agent(tmp_path, policy=_policy(halted=True, halt_reason="prod freeze"))
    assert agent.halted is True
    with pytest.raises(PolicyDenied) as e:
        agent.call_tool("read_doc", doc_id="d1")
    assert "prod freeze" in str(e.value)


def test_halt_and_resume_are_audited(tmp_path):
    agent = _agent(tmp_path)
    agent.halt("drill")
    agent.resume()
    kinds = _kinds(tmp_path)
    assert "agent.halted" in kinds
    assert "agent.resumed" in kinds


def test_a_halted_call_is_recorded_as_denied(tmp_path):
    agent = _agent(tmp_path)
    agent.halt()
    with pytest.raises(PolicyDenied):
        agent.call_tool("delete_doc", doc_id="d1")
    assert "tool.denied" in _kinds(tmp_path)


# ── when nobody can press it ─────────────────────────────────────────

def test_a_stale_policy_revokes_irreversible_tools(tmp_path):
    agent = _agent(tmp_path, policy=_policy(age=3600), max_policy_age=300)
    with pytest.raises(PolicyDenied) as e:
        agent.call_tool("delete_doc", doc_id="d1")
    assert "irreversible tools are revoked" in str(e.value)


def test_a_stale_policy_keeps_reads_working(tmp_path):
    """Degrade, don't die. A product that halts on a blip gets turned off."""
    agent = _agent(tmp_path, policy=_policy(age=3600), max_policy_age=300)
    assert agent.call_tool("read_doc", doc_id="d1") == "contents"


def test_a_policy_that_never_arrived_is_stale(tmp_path):
    """Started up, never reached home. The dangerous case, not a fresh one."""
    agent = _agent(tmp_path, policy=_policy(age=float("inf")), max_policy_age=300)
    with pytest.raises(PolicyDenied) as e:
        agent.call_tool("delete_doc", doc_id="d1")
    assert "never" in str(e.value)


def _denial(agent, tool="delete_doc"):
    """Why a call was refused.

    A destructive tool with no approval path wired up always fails closed,
    which is correct and unrelated to staleness. So these tests assert on
    the *reason*: the staleness gate must not be what stopped the call.
    """
    with pytest.raises(PolicyDenied) as e:
        agent.call_tool(tool, doc_id="d1")
    return str(e.value)


def test_a_fresh_policy_changes_nothing(tmp_path):
    agent = _agent(tmp_path, policy=_policy(age=10), max_policy_age=300)
    assert "revoked" not in _denial(agent)


def test_an_unmanaged_agent_never_degrades(tmp_path):
    """No control plane configured means nothing to be out of touch with."""
    agent = _agent(tmp_path, policy=_policy(age=float("inf"), managed=False),
                   max_policy_age=300)
    assert "revoked" not in _denial(agent)


def test_the_degrade_can_be_switched_off(tmp_path):
    agent = _agent(tmp_path, policy=_policy(age=3600), max_policy_age=0)
    assert "revoked" not in _denial(agent)


def test_a_304_keeps_the_policy_fresh(tmp_path):
    """A 304 carries no body but proves the control plane is reachable.

    Without this an agent whose policy is simply stable degrades while
    perfectly healthy — and a control that fires when nothing is wrong is
    one people disable.
    """
    c = _policy(age=3600)
    assert c.snapshot.age() > 300
    c._touch()
    assert c.snapshot.age() < 1


def test_halt_outranks_a_tool_that_would_otherwise_pass(tmp_path):
    """The gate runs before capability, arguments and budget."""
    agent = _agent(tmp_path, policy=_policy(halted=True))
    with pytest.raises(PolicyDenied):
        agent.call_tool("read_doc", doc_id="d1")
