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
  withinSeconds: 600
  verdict: deny
  reason: outbound request shortly after reading policyholder data
```

Reading a record is fine. Posting to a URL is fine. Doing both within the same
minute is exfiltration, and no per-call rule can see it.

`python examples/exfiltration.py`:

```
http_post(status.internal)                   posted 2 bytes to https://status.internal
read_claim(C-1001)                           {'policyholder': 'A. Nigam', 'ssn': 'xxx-xx-1234'}
http_post(status.internal)  [after the read] DENIED
    outbound request shortly after reading policyholder data
    (saw 'read_claim' 0s earlier)
```

The first and third calls are identical. Capability, arguments and budget all
say yes to both; the only thing that changed is what the agent did in between.
Deterministic and in-memory — no model is asked, and the denial replays the
same way a year later.

Two details that decide whether this is usable in production:

**Only calls that actually ran arm a rule.** A blocked or failed read handed
the agent nothing, so it must not restrict the rest of the session —
otherwise the agent whose reads keep getting denied is the one that stops
being able to work.

**Every window is bounded.** Without `withinSeconds`, one read at startup
restricts every outbound call for the life of the process. People respond to
unusable contracts by deleting them.

State is per-session and in-process: this sees one agent's trajectory, not a
fleet. Correlating across agents and restarts is a control-plane job.

---

## The tool registry is part of your supply chain

Approve an MCP server today. Tomorrow its tool *description* changes. Same
server, same tool name, same signature — but the description is not
documentation, it is text delivered straight into the model's context by
whoever controls that integration.

Your allowlist still matches. Your argument rules still pass. Nothing about
any individual call is wrong. The agent is simply no longer the agent that was
reviewed.

So the manifest hashes the whole execution surface — name, destructive flag,
description, call signature — per tool:

```python
approved = agent.manifest()
approved.digest()          # 37073c680178422c…
dict(approved.surfaces)    # {'read_ticket': '19ebd82d…', 'send_email': 'a3470979…'}
```

Pin it with the deploy that was reviewed, not in the source file — the same
reason you pin a container digest in the manifest and not the Dockerfile:

```bash
export AEGRAIL_EXPECT_MANIFEST=37073c680178422c...
```

`python examples/supply_chain.py`:

```
REFUSED TO START
  agent 'support-bot/v1' has a tool surface that does not match the approved
  manifest — changed: send_email
  Expected 37073c680178422c…, got acb4df5fcde71602…
```

Nothing was called. The refusal is at startup, before the model is handed a
description nobody reviewed — and the drift is written to the audit chain
before the process exits, so the evidence outlives the crash.

Pass the full approved manifest rather than just its hash and the error names
the tool that moved. A hash alone can only say that something did.

The failure is deliberately loud. A legitimate deploy updates the pin, which
is the review step being asked for. A pin that warns and continues is
decoration.

---

## The stop button

Fleet policy carries a halt. Set it and every call denies at the deterministic
tier — including reads, because an operator who stopped the fleet mid-incident
did not mean "keep feeding the model".

```python
agent.halt("incident 4471")     # process-local, reachable from a signal handler
agent.resume()
agent.halted                    # local halt OR fleet policy
```

The hard half is not the button. It is the moment the agent cannot hear it.

An SDK that keeps its last-known policy forever has a kill switch that anyone
who drops egress can defeat — including an agent that has been talked into
dropping its own. So the policy snapshot is timestamped, and an agent that has
not heard from the control plane within `max_policy_age` degrades:

```python
Agent(identity="ops-bot/v1", max_policy_age=300)   # seconds; 0 disables
```

```
read_runbook      runbook db
restart_service   DENIED
                  policy last confirmed 3600s ago, over the 300s limit —
                  irreversible tools are revoked until the control plane is
                  reachable again
```

**Reads keep working; irreversible tools are revoked.** Failing fully closed
would mean a control-plane blip stops a customer's production fleet, and that
is a safety control people disable within a week. Failing open means the
switch is defeated by a cut cable. Splitting on reversibility keeps the agent
useful while making the switch undefeatable for the actions that matter.

An agent that cannot hear the kill switch assumes it was pressed — but only
for the actions that would have mattered if it had been.

Two details that keep it from crying wolf: a `304 Not Modified` refreshes the
age, since it proves the control plane is reachable and the policy unchanged;
and the degrade applies only to an agent that is actually managed, because
offline development is not a degraded state.

### Stopping at a boundary, not mid-sequence

An immediate halt is right when you think the agent is compromised. It is
wrong for most operational work: an agent frozen between "took the service out
of rotation" and "put it back" has stopped *and* left a mess.

Tools can name the critical section they open and close:

```python
@agent.tool(begins="rotation")
def drain(svc): ...

@agent.tool(completes="rotation")
def undrain(svc): ...

agent.halt("deploy window closed", mode="controlled")
```

```
read_status(api)   DENIED   controlled stop is waiting for rotation to close,
                            and this call does not close it
drain(web)         DENIED   (same)
undrain(api)       undrained api
read_status(api)   DENIED   agent is halted at a safe point
```

Only the call that closes open work gets through. Then the agent is at rest
and nothing more runs. `agent.at_safe_point` is a property you can check, not
a claim — **a safe state is not a prior state**, it is one where nothing is
half-done.

A controlled stop escalates to an immediate one after `escalate_after`
seconds. An agent that simply never finishes must not outlive the button.

### Afterwards: a report, not a rollback

```
python -m aegrail.incident ./audit.jsonl
```

```
AGENT STOPPED   2026-11-02 10:18:57   (controlled)
reason          deploy window closed
open at stop    rotation
safe point      reached — no work was left half-done

Cannot be taken back (0)
  nothing — no irreversible action ran

Ran, but changed nothing irreversibly (3)
  10:18:57  read_status(svc='api')
  10:18:57  drain(svc='api')
  10:18:57  undrain(svc='api')
```

Built entirely from the hash chain, so it is as tamper-evident as the actions
were. It deliberately undoes nothing:

- Unwinding during an incident means running new writes with code in a process
  you have just decided not to trust.
- Compensation is not inverse. Restarting a service gives you a cold cache and
  a new PID, not the state you had.
- The post-incident state is the evidence. Reverting it automatically
  bulldozes what an investigator needs.

A human reads it and decides. Slower, and right.

Halts, resumes, safe points and degraded denials are all chain events — who
stopped it, when, what was open, and what was refused afterwards.

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

## Traces your dashboard already understands

Aegrail emits the OpenTelemetry **GenAI semantic conventions**, not a span
shape of its own. `execute_tool` and `gen_ai.tool.name` already mean something
in Datadog, Grafana and Honeycomb — so enforcement appears in the views you
have, with nothing of ours installed.

```bash
pip install 'aegrail[otel]'
python examples/otel_traces.py
```

```
invoke_agent claims-triage   [INTERNAL]
    gen_ai.operation.name      invoke_agent
    gen_ai.agent.name          claims-triage
    gen_ai.request.model       claude-opus-5

  execute_tool read_claim   [INTERNAL]
      gen_ai.tool.name           read_claim
      aegrail.verdict            allowed
      aegrail.decided_by         deterministic

  execute_tool delete_claim   [INTERNAL]
      gen_ai.tool.name           delete_claim
      aegrail.verdict            blocked
      aegrail.decided_by         deterministic
      error.type                 PolicyDenied
```

What the conventions do not cover — the verdict, the tier that decided it, the
audit digest — stays under `aegrail.*`. Dropping those to look compliant would
throw away the only part of this telemetry nobody else emits.

`agent.session()` opens the `invoke_agent` span that the tool calls nest under.
It is optional, and worth the one line: without a parent, calls are siblings at
the root and a trace tells you what ran but never what the agent was doing at
the time.

```python
with agent.session(model="claude-opus-5"):
    agent.call_tool("read_claim", claim_id="C-1001")
    agent.call_tool("write_note", claim_id="C-1001", text="triaged")
```

Call arguments are **opt-in**, matching both the conventions and
`trace_arguments=False`. One flag governs the span and the control plane
together.

Conventions move — they are still marked Development and have already changed
repositories. Every span carries `aegrail.conventions` so a trace says which
spelling it used.

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

## Examples

Both run in-process. No API key, no services, no install:

```bash
git clone https://github.com/arpitcoder/aegrail && cd aegrail
python examples/claims_triage.py    # capability, arguments, failing closed
python examples/exfiltration.py     # two permitted calls, forbidden order
python examples/supply_chain.py     # an approved tool, re-approved by nobody
python examples/kill_switch.py      # stopping an agent, and losing the ability to
python examples/controlled_stop.py  # coming to rest at a boundary, then the report
python examples/otel_traces.py      # the span tree, needs the otel extra
python -m aegrail.verify ./audit.jsonl
```

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
