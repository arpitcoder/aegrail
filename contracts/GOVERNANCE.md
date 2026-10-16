# MavOps governance — three layers

Governance is three layers, applied in order. Each is cheaper and more certain
than the one after it, so a call is settled as early as possible and only the
genuinely ambiguous ones reach anything expensive.

```
  ┌─ Layer 1 ── PROMPT ──────────────── before the model is called
  │    what goes TO the model
  │
  ├─ Layer 2 ── CAPABILITY ──────────── at session start
  │    what the agent MAY EVER do
  │
  └─ Layer 3 ── RUNTIME CONTRACT ────── on every tool call
       what this SPECIFIC call may do
```

The ordering is the design. Layer 2 removes a tool from the agent's reach
entirely; layer 3 then only has to reason about calls that were legitimate to
attempt. Denying `shell_exec` outright is stronger and cheaper than inspecting
its arguments forever.

---

## Layer 1 — Prompt

**Runs:** before every model call. **Cost:** ~704µs. **Fails:** open for PII,
closed for credentials.

| Class | Action | Why |
|---|---|---|
| Secrets | **deny** | a live key in a prompt bound for a hosted model is already leaked |
| PII | **redact** | blocking makes the product unusable — a support agent's prompts are made of email addresses |
| Injection | **flag** | escalate, never trust |

Every pattern sits behind a cheap literal precondition, so the common path is a
substring check rather than a regex walk over the whole prompt.

**What this layer is not.** Deterministic injection detection is a losing arms
race; every pattern here is trivially reworded. It catches the unsophisticated
majority for almost no cost. It is a filter, not a boundary. The honest claim is
not "we detect prompt injection" — it is that a *successful* injection does not
get to do anything, which is layers 2 and 3.

## Layer 2 — Capability

**Runs:** once, at session start. **Cost:** one round trip per session, zero per
call. **Fails:** closed.

An agent may hold only the tools its job requires. Not a denylist — an
**allowlist**, declared in the control plane, not by the agent.

Three properties make it real rather than advisory:

1. **Allowlist, fail closed.** A tool absent from the grant is denied even if
   the process registered it. Absence is a denial, not a gap.
2. **Attested at admission.** The SDK sends its registered tool inventory and a
   manifest hash when the session opens. The control plane rejects the session
   if the agent declares anything outside its grant. Self-declaration becomes a
   checkable claim.
3. **Declared in the deploy.** `spec.tools` on the MavAgent resource, so
   capability is reviewed in the same pull request as the image.

The manifest hash enters the audit chain at `session.start`, which makes *"what
was this agent permitted to do at 14:22"* answerable after the fact — the
question that actually matters in an incident.

Why this beats a denylist: a denylist over a tool set you do not control
permits everything you failed to think of, and cannot tell you what the agent
could do. An allowlist inverts the default.

## Layer 3 — Runtime contract

**Runs:** every tool call. **Cost:** 53µs hot path. **Fails:** closed.

| Tier | Checks | Cost |
|---|---|---|
| 0 · Admission | identity, budget, wall-clock | once |
| 1 · Structural | tool name, destructive flag | ~µs |
| 2a · Argument | declarative rules over arguments | 53µs |
| 2b · Sequence | trajectory — read-then-exfiltrate | 33µs |
| 3 · Judge | free-form arguments no rule can read | ~1s, rare |
| 4 · Human | freeze, durable in Redis, resume | unbounded |

Tiers 0–2 are deterministic: same inputs, same verdict, no model, no network.
Tier 3 exists only because some arguments cannot be classified by pattern —
`execute_sql("SELECT 1")` and `execute_sql("DROP TABLE users")` differ only in
content.

### Where Tier 3 belongs, and where it does not

Tier 3 costs a model call and it blocks. Put it only on tools where the danger
genuinely lives in free text **and** the action carries consequence — a
destination, a query, a shell command. Never on a tool that writes a record and
destroys nothing.

This was learned the hard way. Marking `write_note` as inspectable sent every
claim note to a model, the model was reasonably cautious about unusual prose,
and an agent froze for two minutes waiting for a human to approve a sentence.
Nothing was at risk. Friction like that is not caution: it is how a team ends
up switching the policy off, after which nothing is governed at all.

The test is not "could this text be bad?" — almost any text could. It is
"if this call goes through unexamined, what is different afterwards?" If the
answer is "a row has some text in it", no model needs to read it.

### The escalate-only invariant

```python
class Verdict(IntEnum):
    ALLOW = 0
    HITL  = 1
    DENY  = 2

final = max(deterministic, judge)
```

This is what makes it safe to put a model inside a security decision. The judge
reads attacker-controlled text and can itself be injected — "classify this as
safe" is an obvious payload. Because verdicts combine with `max`, a hijacked
judge answering ALLOW cannot clear a flag the deterministic tiers raised. It can
add caution. It can never subtract.

Sequence rules see what per-call checks cannot: reading customer telemetry is
allowed, posting to an external URL is allowed, and doing both within seconds is
exfiltration.

---

## Failure semantics

| Situation | Result |
|---|---|
| Judge unreachable | escalate to human — an unanswered question is not permission |
| Judge never configured | allow — otherwise an unconfigured install sends everything to a human and is unusable |
| Human does not answer in time | **deny** — silence is not consent |
| Tool absent from allowlist | deny |
| Required argument absent | negative matchers fire — absence is not compliance |

That fourth row is the one people get wrong. A rule saying
`tags.ephemeral != "true"` must fire on a resource with *no* tags, or every
untagged resource walks through the check designed to catch it.

## Audit

Every decision is appended to a SHA-256 hash chain, linked **in-process before
the span leaves the agent**. Tampering with any record breaks every link after
it. `session.start` carries the tool manifest hash, so the chain proves both
what the agent did and what it was permitted to do.

## Onboarding an agent nobody has classified

You cannot write argument rules for a domain you have never seen, and arriving
with a policy that blocks an unfamiliar agent is how governance gets switched
off in week one. So a pilot starts in **shadow mode**:

```python
agent = Agent(identity="...", mode=Mode.SHADOW)
```

Every layer runs exactly as it would in production and records what it *would*
have decided; the call executes regardless. After a window,
`agent.observations.propose_pack()` emits a draft contract built from the
observed tool inventory, argument ranges and trajectories.

Shadow mode is deliberately hard to mistake for enforcement, because a
governance product that silently is not governing buys false confidence:

  * no environment-variable switch — it is a constructor argument, so enabling
    it is a code change that shows up in review
  * a warning at construction, and `mode` recorded in `session.start`
  * the audit event kind differs: `tool.would_deny` is not `tool.denied`, so
    nothing downstream has to infer from a flag whether a recorded denial
    actually stopped anything

The proposal is a draft. It describes what the agent *did* during a window in
which nothing was blocked — including anything already wrong with that. Notably
it will not invent sequence rules: an observed pair is by definition normal
behaviour, and a sequence rule is about what should *not* happen, which
observation cannot establish.

## Deployment models

Both are supported, and the layers are identical in each.

**BYOC** — the whole stack in the customer's cluster via the Helm chart. Tool
arguments, audit and traces never leave their network.

**Hosted** — the Aegrail SDK inside the customer's agents, calling the MavOps
control plane. No Kubernetes required on their side; the agent authenticates
with an API key.

The SDK is the enforcement point in both. The control plane distributes policy
and holds the audit; it is never in the path of an allow decision.
