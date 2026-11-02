"""Spans that a customer's existing dashboard already understands.

The point of following the GenAI semantic conventions rather than inventing
a shape is that nobody has to install anything of ours to see enforcement.
`execute_tool` and `gen_ai.tool.name` already mean something in Datadog,
Grafana and Honeycomb; `aegrail.contract.tool_call` meant something only to
us.

So these tests assert on the wire format, not on our own helpers. They are
skipped when the optional OTel extra is absent, which is also the state most
installs are in — the shim has to no-op cleanly there, and that is tested
too.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from aegrail import Agent, AuditSink, Budget
from aegrail.exceptions import PolicyDenied
from aegrail.capability import CapabilityGrant


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    import aegrail.otel as otel
    monkeypatch.setattr(otel, "_TRACER", provider.get_tracer("test"))
    monkeypatch.delenv("AEGRAIL_OTEL_DISABLED", raising=False)
    yield exporter


def _agent(tmp_path):
    agent = Agent(
        identity="claims-triage/v1",
        budget=Budget(usd=1e6, wall_seconds=1e6),
        audit=AuditSink.file(str(tmp_path / "audit.jsonl")),
        poll_policy=False,
        resume_pending=False,
        capability=CapabilityGrant.from_spec({"allowed_tools": ["read_claim"]}),
    )

    @agent.tool(description="read a claim")
    def read_claim(claim_id: str) -> dict:
        return {"id": claim_id}

    @agent.tool(description="delete a claim")
    def delete_claim(claim_id: str) -> str:
        return "gone"

    return agent


def _by_name(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


# ── tool spans ───────────────────────────────────────────────────────

def test_the_span_is_named_for_the_conventions(tmp_path, spans):
    _agent(tmp_path).call_tool("read_claim", claim_id="C-1")
    assert _by_name(spans, "execute_tool read_claim")


def test_the_required_attributes_are_present(tmp_path, spans):
    _agent(tmp_path).call_tool("read_claim", claim_id="C-1")
    attrs = _by_name(spans, "execute_tool read_claim")[0].attributes
    assert attrs["gen_ai.tool.name"] == "read_claim"
    assert attrs["gen_ai.operation.name"] == "execute_tool"


def test_the_span_kind_is_internal(tmp_path, spans):
    _agent(tmp_path).call_tool("read_claim", claim_id="C-1")
    span = _by_name(spans, "execute_tool read_claim")[0]
    assert span.kind is trace.SpanKind.INTERNAL


def test_our_own_attributes_survive(tmp_path, spans):
    """Compliance is not a reason to drop the only novel part."""
    _agent(tmp_path).call_tool("read_claim", claim_id="C-1")
    attrs = _by_name(spans, "execute_tool read_claim")[0].attributes
    assert attrs["aegrail.verdict"] == "allowed"
    assert "aegrail.decided_by" in attrs
    assert "aegrail.audit_hash" in attrs


def test_a_refusal_sets_error_type(tmp_path, spans):
    """Conditionally required by the conventions when a tool call fails."""
    agent = _agent(tmp_path)
    with pytest.raises(PolicyDenied):
        agent.call_tool("delete_claim", claim_id="C-1")
    attrs = _by_name(spans, "execute_tool delete_claim")[0].attributes
    assert attrs["error.type"] == "PolicyDenied"
    assert attrs["aegrail.verdict"] == "blocked"


# ── arguments stay opt-in ────────────────────────────────────────────

def test_arguments_are_not_recorded_by_default(tmp_path, spans):
    _agent(tmp_path).call_tool("read_claim", claim_id="C-1")
    attrs = _by_name(spans, "execute_tool read_claim")[0].attributes
    assert "gen_ai.tool.call.arguments" not in attrs


def test_arguments_appear_when_asked_for(tmp_path, spans):
    agent = Agent(identity="claims-triage/v1", budget=Budget(usd=1e6),
                  audit=AuditSink.file(str(tmp_path / "a.jsonl")),
                  poll_policy=False, resume_pending=False,
                  trace_arguments=True)

    @agent.tool(description="read a claim")
    def read_claim(claim_id: str) -> dict:
        return {}

    agent.call_tool("read_claim", claim_id="C-1")
    attrs = _by_name(spans, "execute_tool read_claim")[0].attributes
    assert "C-1" in attrs["gen_ai.tool.call.arguments"]


# ── the tree ─────────────────────────────────────────────────────────

def test_session_parents_the_tool_calls(tmp_path, spans):
    """The fix for a trace that was a flat list of siblings."""
    agent = _agent(tmp_path)
    with agent.session():
        agent.call_tool("read_claim", claim_id="C-1")
        agent.call_tool("read_claim", claim_id="C-2")

    parent = _by_name(spans, "invoke_agent claims-triage")[0]
    children = _by_name(spans, "execute_tool read_claim")
    assert len(children) == 2
    for child in children:
        assert child.parent.span_id == parent.context.span_id
        assert child.context.trace_id == parent.context.trace_id


def test_the_agent_span_carries_its_attributes(tmp_path, spans):
    agent = _agent(tmp_path)
    with agent.session(model="claude-opus-5"):
        agent.call_tool("read_claim", claim_id="C-1")
    attrs = _by_name(spans, "invoke_agent claims-triage")[0].attributes
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["gen_ai.agent.name"] == "claims-triage"
    assert attrs["gen_ai.request.model"] == "claude-opus-5"


def test_without_a_session_calls_are_still_valid_spans(tmp_path, spans):
    """session() is optional. Omitting it costs the tree, not the span."""
    _agent(tmp_path).call_tool("read_claim", claim_id="C-1")
    span = _by_name(spans, "execute_tool read_claim")[0]
    assert span.parent is None
    assert span.attributes["gen_ai.tool.name"] == "read_claim"
