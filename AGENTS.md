# AGENTS.md

Instructions for coding agents integrating this library. Everything below is
verified against the test suite in `tests/`.

## What this is

`aegrail` decides whether an AI agent is allowed to make a given tool call.
Deterministic rules decide; an optional model judge may only make a decision
*stricter*, never more permissive.

## Install

```bash
pip install aegrail
```

Zero runtime dependencies. Python 3.9+. Optional extras: `aegrail[yaml]` to load
contracts from YAML, `aegrail[otel]` to emit spans.

## Minimal correct integration

Call `agent.call_tool(name, **kwargs)` instead of calling the function directly.
That is the whole integration — it sits at the tool-call boundary, so it works
with LangChain, LlamaIndex, raw function calling, or a hand-written loop.

```python
from aegrail import Agent, ArgRuleSet, CapabilityGrant
from aegrail.exceptions import PolicyDenied

agent = Agent(
    identity="my-agent/v1",
    capability=CapabilityGrant.from_spec({"allowed_tools": ["read_row", "pay"]}),
    arg_rules=ArgRuleSet.from_list([
        {"tool": "pay", "arg": "amount", "gt": 10_000,
         "verdict": "deny", "reason": "above the unattended limit"},
    ]),
)

@agent.tool(description="read a row")
def read_row(row_id: str) -> dict:
    return db.get(row_id)

@agent.tool(destructive=True, description="send money")
def pay(row_id: str, amount: float) -> str:
    return payments.send(row_id, amount)

agent.attest()          # reconcile registered tools against the grant

try:
    agent.call_tool("pay", row_id="r1", amount=250_000)
except PolicyDenied as e:
    ...                 # refused; `e` carries the rule that fired
```

## Behaviour you will get wrong if you guess

- **A tool marked `destructive=True` requires an approval path.** With no HITL
  client configured, *every* call to it is denied — including small, otherwise
  legal ones. It fails closed. Either pass a `HITLClient`, or do not mark the
  tool destructive.
- **A tool that was never registered raises `KeyError`, not `PolicyDenied`.**
  `PolicyDenied` is for a tool that *is* registered but absent from the
  capability grant.
- **`Verdict` is an `IntEnum`:** `ALLOW=0 < HITL=1 < DENY=2`. The final verdict
  is `max(deterministic, judge)`. Do not invert this.
- **Use `outside: [lo, hi]`, not `gt:`, for bounds.** Query APIs spell
  "unbounded" as a sentinel (`LIMIT -1`, `size: 0`), which is *smaller* than any
  `gt` threshold and sails through it.
- **Rules match arguments wherever they live** — top level, nested inside a
  `params`/`body`/`request` object, or inside a JSON-encoded string. Naming an
  argument means it everywhere.
- **Negative matchers fire on absence.** A rule like `tags.ephemeral != "true"`
  must fire on a resource with no tags at all, or every untagged resource walks
  through the check written to catch it.

## Traces

Enforcement emits a trace per tool call to the control plane — verdict, which
tier decided it, duration, and the audit hash. Off if `MAVOPS_API_TOKEN` is
unset; disable explicitly with `AEGRAIL_TRACES_DISABLED=1`.

```python
agent = Agent(
    identity="my-agent/v1",
    trace_arguments=True,                     # default False — see below
    agent_id="3f2504e0-4f89-11d3-9a0c-0305e82c3301",
)
```

- **`trace_arguments` is off by default and should usually stay off.**
  Enforcement always evaluates the real arguments; this only controls whether
  they are *shipped*. Tool arguments are where personal data lives.
- **`agent_id` must be a UUID.** The control plane parses it as one and
  discards anything else, leaving traces unlinked. The SDK warns if you pass a
  non-UUID rather than letting it fail silently.
- **Traces flush on exit.** `start()` registers an `atexit` hook and `stop()`
  drains the queue with a bounded timeout, so a short-lived agent process does
  not lose its telemetry. You do not normally need to call `stop()`.
- **Self-hosted or cloud is one environment variable.** `MAVOPS_API_BASE`
  defaults to the hosted control plane; point it at your own cluster and
  nothing else changes.

## Onboarding an agent whose behaviour you don't know

```python
from aegrail import Mode
agent = Agent(identity="unknown/v1", mode=Mode.SHADOW)   # observes, blocks nothing
print(agent.observations.propose_pack(domain="lending"))  # draft contract
```

Shadow mode is a constructor argument on purpose, not an environment variable,
so enabling it is visible in code review. Audit events change kind too:
`tool.would_deny`, never `tool.denied`.

## Audit

Every decision appends to a SHA-256 chain, linked in-process before anything
leaves the agent.

```bash
python -m aegrail.verify ./audit.jsonl
```

Exits non-zero and names the sequence number if any record was altered.

## Contract packs

`contracts/` holds ready-made rules: `infra-baseline.yaml` (destructive actions
common to every domain) plus `domains/insurance.yaml` and
`domains/observability.yaml` layered on top. Load with `aegrail[yaml]`, or pass
the equivalent plain dicts.

## Honest limits

This is a guardrail, not a sandbox. It runs in-process and does not stop code
that deliberately bypasses it by calling the underlying function. It defends
against a manipulated *model*, not a malicious *developer*. In production, scope
the agent's own credentials and use network policy as the backstop.
