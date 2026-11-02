"""Stopping an agent, and the case where nobody can reach it to stop it.

A stop button is easy. The hard half is what happens when the agent has lost
the control plane — because that is both the moment you most want the button,
and the moment it does not work. An agent that keeps its own last-known policy
forever can be freed by anything that drops egress, including an agent that
has been talked into dropping its own.

    python examples/kill_switch.py

No network, no API key, no services.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aegrail import Agent, Budget
from aegrail.exceptions import PolicyDenied
from aegrail.policy import PolicyClient, PolicySnapshot


def control_plane(*, halted=False, reason="", last_seen_seconds_ago=0.0):
    """Stand in for the fleet control plane at a known state."""
    c = PolicyClient(agent_name="kill-switch-demo")
    c._snapshot = PolicySnapshot(
        halted=halted, halt_reason=reason,
        fetched_at=time.monotonic() - last_seen_seconds_ago,
        source="demo",
    )
    type(c).managed = property(lambda self: True)
    return c


def build(policy):
    agent = Agent(
        identity="ops-bot/v1",
        budget=Budget(usd=10.0, wall_seconds=600),
        policy=policy,
        # How long this agent may keep acting on a policy nobody has
        # confirmed. Past it, irreversible tools are revoked.
        max_policy_age=300,
    )

    @agent.tool(description="read a runbook")
    def read_runbook(name: str) -> str:
        return f"runbook {name}"

    @agent.tool(destructive=True, description="restart a service")
    def restart_service(name: str) -> str:
        return f"restarted {name}"

    return agent


def attempt(label, fn):
    try:
        print(f"  {label:<40} {fn()}")
    except PolicyDenied as e:
        print(f"  {label:<40} DENIED\n  {'':<40} {e}")


print("\n1. normal operation — policy confirmed 10s ago\n")
agent = build(control_plane(last_seen_seconds_ago=10))
attempt("read_runbook", lambda: agent.call_tool("read_runbook", name="db"))

print("\n2. someone presses stop\n")
agent.halt("incident 4471 — agent misbehaving in prod")
attempt("read_runbook", lambda: agent.call_tool("read_runbook", name="db"))
attempt("restart_service", lambda: agent.call_tool("restart_service", name="db"))
print("""
  Reads stop too. There is no harmless-tool exemption: if an operator
  stopped the fleet mid-incident, a read that feeds the next model call
  is not neutral.""")

agent.resume()

print("\n3. the control plane goes away. nobody can press anything.\n")
stranded = build(control_plane(last_seen_seconds_ago=3600))
attempt("read_runbook", lambda: stranded.call_tool("read_runbook", name="db"))
attempt("restart_service", lambda: stranded.call_tool("restart_service", name="db"))

print("""
  The agent degrades rather than dying. Reads keep working, so a blip in
  the control plane does not stop a customer's production fleet — that is
  a product people switch off within a week. The irreversible half is
  revoked, so the kill switch cannot be defeated by cutting a cable.

  An agent that cannot hear the kill switch assumes it was pressed, but
  only for the actions that would have mattered if it had been.

  Every halt, resume and degraded denial is in the chain:
    python -m aegrail.verify ./audit.jsonl
""")
