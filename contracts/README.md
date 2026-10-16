# Contract library

A contract is not one universal ruleset. What counts as dangerous
depends entirely on what the agent is for:

| Domain | Routine | Dangerous |
|---|---|---|
| Observability | reading telemetry across every source | that telemetry leaving the VPC |
| Healthcare | reading PHI — it is the job | exporting it, or sending it to a hosted model |
| SRE / platform | killing a pod in `scratch` | the same call in `kube-system` |
| Fintech | reading balances | a transfer above a ceiling, or to a new payee |
| Support | issuing a small refund | deleting an account, bulk refunds |
| Coding | opening a PR | force-push to `main`, editing CI, rotating a secret |

A rule that fits one of those is noise in another. `http_post` is
unremarkable for a support agent and is the exfiltration step for an
observability agent holding read access to a customer's estate.

So contracts ship as **domain packs**. A customer picks the pack that
matches their agent, gets defaults that are sensible on day one, and
tightens from there — rather than authoring a policy from an empty file
and discovering the gaps in production.

## Layout

```
contracts/
  infra-baseline.yaml        cloud and Kubernetes blast radius
  domains/
    observability.yaml       read-heavy agents with broad credentials
    insurance.yaml           claims, policies and policyholder data
    ...                      (one per domain as they are written)
```

Packs compose. An observability agent that also remediates loads both
the observability pack and the infra baseline; the strictest verdict
across all of them wins, because every tier only escalates.

## What a pack contains

```yaml
argumentRules:    # Tier 2a — deterministic, on the arguments of one call
sequenceRules:    # Tier 2b — deterministic, on what already happened
judge:            # Tier 3 — how to frame the question for this domain
```

The `judge` block matters more than it looks. "Is this destructive?"
means something different to a healthcare agent than to an SRE agent,
and a judge asked the generic question gives generically useless
answers. Each pack tells the judge what this domain's agents are *for*,
so it can recognise when a call sits outside that.

## Writing a pack

Three habits that make the difference:

**Invert the default.** `tags.ephemeral != "true"` beats `env == "prod"`.
The first catches the resource nobody remembered to label, which is the
one that gets deleted by accident. The second only catches correctly
labelled resources, which are the ones least likely to be a problem.

**Name what the pack does not cover.** Every pack here ends with a
section saying which risks it leaves to another tier or to nobody. A
policy that quietly implies completeness is worse than one that admits
its edges.

**Prefer `hitl` to `deny`.** A denied call the agent legitimately needed
turns into a stuck workflow and, eventually, someone disabling the
policy. An escalated call turns into a ten-second decision. Reserve
`deny` for things no human should be approving in a hurry.
