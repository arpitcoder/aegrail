# Aegrail

[![tests](https://github.com/arpitcoder/aegrail/actions/workflows/test.yml/badge.svg)](https://github.com/arpitcoder/aegrail/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/aegrail)](https://pypi.org/project/aegrail/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)

**Runtime contract enforcement for AI agents.** Deterministic rules decide what
an agent may do. A model is allowed in the loop only for calls a rule genuinely
cannot read — and it can only ever make a decision *stricter*, never permissive.

```bash
pip install aegrail
```

Zero dependencies. Standard library only, down to a hand-rolled Redis client.
This runs inside your production agent; you shouldn't have to audit our
dependency tree to install it.

---

## Why not just use a model to check the model?

Because you can prompt-inject the judge. If an attacker can talk your agent
into dropping a table, they can talk your safety classifier into approving it.

So Aegrail inverts it:

```python
final_verdict = max(deterministic_verdict, model_verdict)
```

That's enforced in the type system — `Verdict` is an `IntEnum` where
`ALLOW=0 < HITL=1 < DENY=2`. A compromised model can add caution. It can never
subtract it.

---

## Sixty seconds

```python
from aegrail import Agent, ArgRuleSet, Budget, CapabilityGrant

agent = Agent(
    identity="claims-triage/v1",
    budget=Budget(usd=10.0, wall_seconds=600),

    # Layer 2 — the tools this agent may ever hold. An allowlist: anything
    # absent is denied even if the process registered it.
    capability=CapabilityGrant.from_spec({
        "allowed_tools": ["read_claim", "assess_claim", "write_note",
                          "approve_payout"]
    }),

    # Layer 3 — what any individual call may do.
    arg_rules=ArgRuleSet.from_list([
        {"tool": "*_query*", "arg": "limit", "outside": [1, 1000],
         "verdict": "deny", "reason": "result set outside safe bounds"},
        {"tool": "approve_payout", "arg": "amount", "gt": 10_000,
         "verdict": "hitl", "reason": "above the unattended settlement limit"},
    ]),
)

@agent.tool(description="read a claim")
def read_claim(claim_id: str) -> dict:
    return db.fetch(claim_id)

@agent.tool(destructive=True, description="pay a claim")
def approve_payout(claim_id: str, amount: float) -> str:
    return payments.send(claim_id, amount)

# Reconcile what the process registered against what it was granted, and
# write the tool manifest hash into the audit chain.
agent.attest()

agent.call_tool("read_claim", claim_id="C-1001")      # runs
agent.call_tool("approve_payout", claim_id="C-1001", amount=250_000)
# PolicyDenied: above the unattended settlement limit (amount=250000)

# A tool this process registered but the grant omits is refused before any
# argument is even looked at:
@agent.tool(destructive=True, description="delete a claim")
def delete_claim(claim_id: str) -> str:
    return db.delete(claim_id)

agent.call_tool("delete_claim", claim_id="C-1001")
# PolicyDenied: 'delete_claim' is not in this agent's capability grant
```

(A name that was never registered at all raises `KeyError` instead — the
capability layer governs tools the process actually holds.)

Your agent framework calls `agent.call_tool(...)` instead of the function
directly. That's the whole integration — it works with LangChain, LlamaIndex,
raw OpenAI function calling, or your own loop, because it sits at the tool-call
boundary rather than inside any framework.

---

## The three layers

| | Runs | Cost | Decides |
|---|---|---|---|
| **1 · Prompt** | before each model call | ~700µs | what goes *to* the model |
| **2 · Capability** | once, at session start | 470ns | what the agent may *ever* do |
| **3 · Contract** | every tool call | ~19µs | what *this* call may do |

The ordering is the design. Layer 2 removes a tool from reach entirely, so
layer 3 only reasons about calls that were legitimate to attempt. Denying
`shell_exec` outright is cheaper and stronger than inspecting its arguments
forever.

Layer 3 runs in tiers, each only reached if the one before it couldn't settle
the call:

```
structural  tool name, destructive flag
argument    declarative rules over the actual arguments
sequence    trajectory — reading customer data then posting externally
judge       free-form arguments no rule can read (optional, rare)
human       freeze the agent, resume when someone answers
```

### Arguments are found wherever they live

```python
execute_sql(query="DROP TABLE users")                  # caught
execute_sql(request={"query": "DROP TABLE users"})     # also caught
execute_sql(statements=["DROP TABLE users"])           # also caught
execute_sql(params='{"query": "DROP TABLE users"}')    # also caught
```

A rule naming an argument means it wherever the tool puts it. Most tools wrap
their arguments in a `params`/`body`/`request` object, and a contract that only
checked the top level would read correctly in review and enforce nothing.

### `outside` exists because `gt` lies

```yaml
- tool: "*_query*"
  arg: limit
  outside: [1, 1000]      # catches 5000 AND -1 AND 0
```

Every query API spells unbounded as a sentinel — `LIMIT -1`, `size: 0`. A
`gt: 1000` rule waves through exactly the unbounded query it was written to
stop, because the sentinel is *smaller* than the threshold.

### Sequence rules see what per-call checks cannot

```yaml
- afterAny: ["read_claim*", "read_policyholder*"]
  then: "http_*"
  within_seconds: 600
  verdict: deny
  reason: outbound request shortly after reading policyholder data
```

Reading a record is fine. Posting to a URL is fine. Doing both within the same
minute is exfiltration, and no per-call rule can see it.

---

## Onboarding an agent nobody has classified

You can't write argument rules for a domain you've never seen, and arriving
with a policy that blocks an unfamiliar agent is how governance gets switched
off in week one. So start by watching:

```python
from aegrail import Mode

agent = Agent(identity="unknown/v1", mode=Mode.SHADOW)
# ... run normally for a week; nothing is blocked ...

print(agent.observations.propose_pack(domain="lending"))
```

That emits a draft contract built from the tool inventory, argument ranges and
trajectories the agent actually produced. Thresholds sit just above what was
observed. Low-cardinality arguments become `in` rules.

It deliberately will **not** generate sequence rules: an observed pair is by
definition the agent's normal behaviour, and a sequence rule is about what
should *not* happen — which observation cannot establish.

Shadow mode is built to be impossible to mistake for enforcement. No
environment-variable switch (it's a constructor argument, so turning it on
shows up in code review), a warning at construction, and the audit event kind
changes: `tool.would_deny` is not `tool.denied`.

---

## Audit

Every decision is appended to a SHA-256 hash chain, linked **in-process before
anything leaves the agent**. Tampering with a record breaks every link after it.

```bash
python -m aegrail.verify ./audit.jsonl
```

```
✓ CHAIN INTACT  chain intact across 11 events (head=9daea2a2b971…)

  7  tool.executed
  1  policy.applied
  1  session.start
  1  hitl.requested
  1  hitl.approved
```

`session.start` carries the tool manifest hash, so the chain proves both what
the agent did and what it was *permitted* to do — which is the question asked
after an incident, not before one.

---

## Failure semantics

| Situation | Result |
|---|---|
| Judge unreachable | escalate to a human — an unanswered question is not permission |
| Judge never configured | allow — otherwise an unconfigured install sends everything to a human and is unusable |
| Human doesn't answer in time | **deny** — silence is not consent |
| Tool absent from the allowlist | deny |
| Required argument absent | negative matchers fire — absence is not compliance |

That last row is the one people get wrong. A rule saying
`tags.ephemeral != "true"` has to fire on a resource with *no* tags, or every
untagged resource walks straight through the check designed to catch it.

---

## Honest limits

**This is a guardrail, not a sandbox.** Aegrail runs in-process. Code that
deliberately bypasses it — calling the underlying function directly — is not
stopped. It defends against a manipulated *model*, not a malicious *developer*.
In production, scope the agent's own credentials and use network policy as the
backstop.

**Deterministic prompt-injection detection is a losing arms race.** Layer 1
catches the unsophisticated majority for almost no cost. The honest claim isn't
"we detect prompt injection" — it's that a *successful* injection doesn't get
to do anything, which is layers 2 and 3.

---

## Control plane (optional)

Aegrail enforces locally and works entirely standalone. Point it at a MavOps
control plane and you additionally get fleet-wide policy distribution, the
approvals inbox, traces, and the audit chain in one place:

```bash
export MAVOPS_API_BASE=https://api.mavops.xyz   # or your own cluster
export MAVOPS_API_TOKEN=mvo_...
```

Same SDK either way. Self-hosted, tool arguments and audit records never leave
your network.

MIT licensed — read the code that audits your agents.
