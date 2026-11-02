"""Enforcement that shows up in the dashboard you already have.

Aegrail emits OpenTelemetry GenAI semantic conventions — `execute_tool`,
`gen_ai.tool.name`, `invoke_agent` — rather than a span shape of its own.
Your Datadog, Grafana or Honeycomb already knows what those mean, so the
contract layer appears in the views you have instead of a view we would
like you to install.

What the conventions do not cover — the verdict, which tier decided, the
audit digest — rides along under `aegrail.*`.

    pip install 'aegrail[otel]'
    python examples/otel_traces.py

Prints the spans to the terminal instead of shipping them anywhere.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
except ImportError:
    sys.exit("this example needs the otel extra:  pip install 'aegrail[otel]'")

import aegrail.otel as otel
from aegrail import Agent, Budget, CapabilityGrant
from aegrail.exceptions import PolicyDenied

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
otel._TRACER = provider.get_tracer("aegrail-example")

agent = Agent(
    identity="claims-triage/v1",
    budget=Budget(usd=10.0, wall_seconds=600),
    capability=CapabilityGrant.from_spec({
        "allowed_tools": ["read_claim", "write_note"],
    }),
)


@agent.tool(description="read a claim")
def read_claim(claim_id: str) -> dict:
    return {"id": claim_id, "assessed": 2_400.0}


@agent.tool(description="attach a note")
def write_note(claim_id: str, text: str) -> str:
    return "noted"


@agent.tool(destructive=True, description="delete a claim")
def delete_claim(claim_id: str) -> str:
    return "gone"


# One turn. Everything inside nests under a single invoke_agent span —
# without it the calls are siblings at the root, and the trace tells you
# what ran but never what the agent was doing at the time.
with agent.session(model="claude-opus-5"):
    agent.call_tool("read_claim", claim_id="C-1001")
    agent.call_tool("write_note", claim_id="C-1001", text="triaged")
    try:
        agent.call_tool("delete_claim", claim_id="C-1001")
    except PolicyDenied:
        pass

spans = exporter.get_finished_spans()
by_parent = {}
for s in spans:
    key = s.parent.span_id if s.parent else None
    by_parent.setdefault(key, []).append(s)

INTERESTING = ("gen_ai.operation.name", "gen_ai.tool.name", "gen_ai.agent.name",
               "gen_ai.request.model", "aegrail.verdict", "aegrail.decided_by",
               "error.type")


def show(span, depth=0):
    pad = "  " * depth
    print(f"\n{pad}  {span.name}   [{span.kind.name}]")
    for key in INTERESTING:
        if key in span.attributes:
            print(f"{pad}      {key:<26} {span.attributes[key]}")
    for child in by_parent.get(span.context.span_id, []):
        show(child, depth + 1)


print("\n  trace", format(spans[0].context.trace_id, "032x"))
for root in by_parent.get(None, []):
    show(root)

print("""
  Three tool calls under one agent turn, one of them refused — and every
  attribute above is a name the ecosystem already defines, except the
  aegrail.* ones, which are the part nobody else emits.

  The same identifiers go to the MavOps control plane, so clicking through
  from your own dashboard lands on the matching enforcement record.
""")
