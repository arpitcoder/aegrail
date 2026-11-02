"""Stopping at a boundary, and the report that comes out of it.

Two things are under test. First, that a controlled stop actually comes to
rest rather than freezing mid-sequence — and that it cannot be held open
forever by an agent that simply never finishes. Second, that the incident
report reads the chain honestly: what ran, what cannot be taken back, and
what was refused.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from aegrail import Agent, AuditSink, Budget
from aegrail.exceptions import PolicyDenied
from aegrail.incident import build, render


class _ApprovingHuman:
    """A reviewer who always says yes.

    Needed because a destructive tool with no approval path fails closed,
    which is correct and would otherwise keep every irreversible call in
    these tests from ever running — leaving the report with nothing to
    report on.
    """

    enabled = True

    def request_approval(self, **kw):
        return type("D", (), {"approved": True, "review_id": "test",
                              "approval_url": "", "decided_via": "test"})()

    def poll(self, *a, **kw):
        return None


def _agent(tmp_path):
    agent = Agent(
        identity="deploy-bot/v1",
        budget=Budget(usd=1e6, wall_seconds=1e6),
        audit=AuditSink.file(str(tmp_path / "audit.jsonl")),
        poll_policy=False,
        resume_pending=False,
        hitl=_ApprovingHuman(),
    )

    @agent.tool(description="read status")
    def read_status(svc: str) -> str:
        return "ok"

    @agent.tool(begins="deploy", description="take a service out of rotation")
    def drain(svc: str) -> str:
        return f"drained {svc}"

    @agent.tool(completes="deploy", description="put it back in rotation")
    def undrain(svc: str) -> str:
        return f"undrained {svc}"

    @agent.tool(destructive=True, description="delete a snapshot")
    def delete_snapshot(snap: str) -> str:
        return f"deleted {snap}"

    return agent


def _events(tmp_path):
    return [json.loads(l) for l in
            open(tmp_path / "audit.jsonl", encoding="utf-8") if l.strip()]


# ── coming to rest ───────────────────────────────────────────────────

def test_a_controlled_stop_lets_open_work_finish(tmp_path):
    agent = _agent(tmp_path)
    agent.call_tool("drain", svc="api")
    assert agent.open_sections == ["deploy"]
    assert agent.at_safe_point is False

    agent.halt("rolling restart cancelled", mode="controlled")

    # The call that closes the section is the one exception.
    assert agent.call_tool("undrain", svc="api") == "undrained api"
    assert agent.at_safe_point is True


def test_a_controlled_stop_refuses_everything_else(tmp_path):
    agent = _agent(tmp_path)
    agent.call_tool("drain", svc="api")
    agent.halt("stop", mode="controlled")

    for tool, kw in [("read_status", {"svc": "api"}),
                     ("drain", {"svc": "web"}),
                     ("delete_snapshot", {"snap": "s1"})]:
        with pytest.raises(PolicyDenied) as e:
            agent.call_tool(tool, **kw)
        assert "waiting for deploy to close" in str(e.value)


def test_once_at_rest_a_controlled_stop_denies_everything(tmp_path):
    agent = _agent(tmp_path)
    agent.call_tool("drain", svc="api")
    agent.halt("stop", mode="controlled")
    agent.call_tool("undrain", svc="api")

    with pytest.raises(PolicyDenied) as e:
        agent.call_tool("read_status", svc="api")
    assert "safe point" in str(e.value)


def test_a_controlled_stop_with_nothing_open_is_immediate(tmp_path):
    agent = _agent(tmp_path)
    agent.halt("stop", mode="controlled")
    assert agent.at_safe_point is True
    with pytest.raises(PolicyDenied):
        agent.call_tool("read_status", svc="api")


def test_an_immediate_stop_does_not_wait(tmp_path):
    """The default, and the right one when you think it is compromised."""
    agent = _agent(tmp_path)
    agent.call_tool("drain", svc="api")
    agent.halt("compromised")
    with pytest.raises(PolicyDenied) as e:
        agent.call_tool("undrain", svc="api")
    assert "agent is halted" in str(e.value)


def test_a_controlled_stop_cannot_be_held_open_forever(tmp_path):
    """An agent that never closes its section must not outlive the button."""
    agent = _agent(tmp_path)
    agent.call_tool("drain", svc="api")
    agent.halt("stop", mode="controlled", escalate_after=0.0)
    with pytest.raises(PolicyDenied) as e:
        agent.call_tool("undrain", svc="api")
    assert "escalated to an immediate stop" in str(e.value)


def test_the_safe_point_is_recorded(tmp_path):
    agent = _agent(tmp_path)
    agent.call_tool("drain", svc="api")
    agent.halt("stop", mode="controlled")
    agent.call_tool("undrain", svc="api")
    kinds = [e["kind"] for e in _events(tmp_path)]
    assert "agent.safe_point" in kinds


def test_the_halt_records_what_was_open(tmp_path):
    agent = _agent(tmp_path)
    agent.call_tool("drain", svc="api")
    agent.halt("stop", mode="controlled")
    halt = [e for e in _events(tmp_path) if e["kind"] == "agent.halted"][-1]
    assert halt["open_sections"] == ["deploy"]
    assert halt["mode"] == "controlled"


def test_an_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        _agent(tmp_path).halt("stop", mode="gentle")


# ── the report ───────────────────────────────────────────────────────

def test_the_report_separates_what_cannot_be_taken_back(tmp_path):
    agent = _agent(tmp_path)
    agent.call_tool("read_status", svc="api")
    agent.call_tool("delete_snapshot", snap="snap-7")
    agent.halt("suspicious activity")

    report = build(_events(tmp_path))
    assert [e["tool"] for e in report["irreversible"]] == ["delete_snapshot"]
    assert "read_status" in [e["tool"] for e in report["reversible"]]


def test_the_report_names_the_arguments(tmp_path):
    agent = _agent(tmp_path)
    agent.call_tool("delete_snapshot", snap="snap-7")
    agent.halt("stop")
    assert "snap-7" in render(build(_events(tmp_path)))


def test_the_report_shows_refusals(tmp_path):
    agent = _agent(tmp_path)
    agent.halt("stop")
    with pytest.raises(PolicyDenied):
        agent.call_tool("read_status", svc="api")
    report = build(_events(tmp_path))
    assert len(report["denied"]) == 1


def test_the_report_says_when_nothing_irreversible_ran(tmp_path):
    agent = _agent(tmp_path)
    agent.call_tool("read_status", svc="api")
    agent.halt("stop")
    assert "no irreversible action ran" in render(build(_events(tmp_path)))


def test_the_report_flags_an_unreached_safe_point(tmp_path):
    agent = _agent(tmp_path)
    agent.call_tool("drain", svc="api")
    agent.halt("stop", mode="controlled")
    out = render(build(_events(tmp_path)))
    assert "NOT reached" in out
    assert "deploy" in out


def test_the_report_is_readable_with_no_halt(tmp_path):
    agent = _agent(tmp_path)
    agent.call_tool("read_status", svc="api")
    assert "No halt recorded" in render(build(_events(tmp_path)))
