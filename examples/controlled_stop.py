"""Stopping at a boundary, then reading back what happened.

An immediate stop is right when you think the agent is compromised. It is
wrong for most operational work, because an agent frozen between "took the
service out of rotation" and "put it back" has stopped *and* left a mess.

Machinery safety has had names for this for decades. ISO 13850 calls an
immediate stop Category 0 and a controlled one Category 1: keep power long
enough to come to rest, then remove it. The EU AI Act asks for the same
thing in Article 14 — a stop that brings the system "to a halt in a safe
state".

A safe state is not a prior state. Nothing here is undone. A safe state is
one where nothing is half-done, and `begins`/`completes` is what makes that
checkable rather than asserted.

    python examples/controlled_stop.py

No network, no API key, no services.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aegrail import Agent, Budget
from aegrail.exceptions import PolicyDenied
from aegrail.incident import build, render

agent = Agent(identity="deploy-bot/v1",
              budget=Budget(usd=10.0, wall_seconds=600))


@agent.tool(description="check a service's health")
def read_status(svc: str) -> str:
    return f"{svc}: healthy"


@agent.tool(begins="rotation", description="take a service out of rotation")
def drain(svc: str) -> str:
    return f"drained {svc}"


@agent.tool(completes="rotation", description="return a service to rotation")
def undrain(svc: str) -> str:
    return f"undrained {svc}"


def attempt(label, fn):
    try:
        print(f"  {label:<34} {fn()}")
    except PolicyDenied as e:
        print(f"  {label:<34} DENIED\n  {'':<34} {e}")


print("\n1. a rolling restart is underway\n")
attempt("read_status(api)", lambda: agent.call_tool("read_status", svc="api"))
attempt("drain(api)", lambda: agent.call_tool("drain", svc="api"))
print(f"\n  open sections: {agent.open_sections}")
print(f"  at a safe point: {agent.at_safe_point}")

print("\n2. someone stops it — controlled\n")
agent.halt("deploy window closed", mode="controlled")

attempt("read_status(api)", lambda: agent.call_tool("read_status", svc="api"))
attempt("drain(web)", lambda: agent.call_tool("drain", svc="web"))
attempt("undrain(api)", lambda: agent.call_tool("undrain", svc="api"))

print(f"\n  at a safe point: {agent.at_safe_point}")

print("\n3. at rest, nothing more runs\n")
attempt("read_status(api)", lambda: agent.call_tool("read_status", svc="api"))

print("""
  An immediate halt would have refused undrain(api) too, leaving `api`
  out of rotation with nobody driving. A controlled halt lets the one
  call that closes open work through, and nothing else — and it escalates
  to an immediate stop after `escalate_after` seconds, so an agent that
  never finishes cannot outlive the button.

4. and the report an operator actually reads""")

path = os.environ.get("AEGRAIL_AUDIT_PATH", "./audit.jsonl")
import json
with open(path, encoding="utf-8") as fh:
    events = [json.loads(line) for line in fh if line.strip()]
print(render(build(events)))
