"""A claims agent that cannot overpay, cannot reach a tool it wasn't granted,
and leaves a record you can verify afterwards.

    python examples/claims_triage.py

No network, no API key, no services. Everything here runs in-process.
"""
from aegrail import Agent, ArgRuleSet, Budget, CapabilityGrant
from aegrail.exceptions import PolicyDenied

agent = Agent(
    identity="claims-triage/v1",
    budget=Budget(usd=10.0, wall_seconds=600),

    # Layer 2 — the tools this agent may ever hold. An allowlist: a tool the
    # process registered but the grant omits is refused before its arguments
    # are even read. Note `delete_claim` is registered below and missing here.
    capability=CapabilityGrant.from_spec({
        "allowed_tools": ["read_claim", "write_note", "approve_payout"],
    }),

    # Layer 3 — what any individual call is allowed to do.
    arg_rules=ArgRuleSet.from_list([
        {"tool": "approve_payout", "arg": "amount", "gt": 10_000,
         "verdict": "deny", "reason": "above the unattended settlement limit"},
        # `outside` rather than `gt`, because every query API spells "unbounded"
        # as a sentinel — LIMIT -1, size: 0 — and a `gt` rule waves through
        # exactly the unbounded query it was written to stop.
        {"tool": "*_query*", "arg": "limit", "outside": [1, 1000],
         "verdict": "deny", "reason": "result set outside safe bounds"},
    ]),
)

CLAIMS = {"C-1001": {"policyholder": "A. Nigam", "assessed": 2_400.0}}


@agent.tool(description="read a claim")
def read_claim(claim_id: str) -> dict:
    return CLAIMS.get(claim_id, {})


@agent.tool(description="attach a note to a claim")
def write_note(claim_id: str, text: str) -> str:
    return f"note added to {claim_id}"


@agent.tool(destructive=True, description="pay a claim")
def approve_payout(claim_id: str, amount: float) -> str:
    return f"paid {amount:.2f} on {claim_id}"


# Registered by this process, but absent from the grant above.
@agent.tool(destructive=True, description="delete a claim")
def delete_claim(claim_id: str) -> str:
    return f"deleted {claim_id}"


def attempt(label, fn):
    try:
        print(f"  {label:<44} {fn()}")
    except PolicyDenied as e:
        print(f"  {label:<44} DENIED\n  {'':<44} {e}")


agent.attest()   # reconcile registered tools against the grant, hash the manifest

print("\nclaims-triage/v1\n")

attempt("read_claim(C-1001)",
        lambda: agent.call_tool("read_claim", claim_id="C-1001"))

attempt("write_note(C-1001)",
        lambda: agent.call_tool("write_note", claim_id="C-1001", text="triaged"))

# Over the rule's threshold — refused on the arguments.
attempt("approve_payout(C-1001, 250000)",
        lambda: agent.call_tool("approve_payout", claim_id="C-1001", amount=250_000))

# Under the threshold, but the tool is marked destructive and this agent has no
# approval path wired up. It fails closed rather than paying. Give it a HITL
# client and this same call pauses for a human instead of being refused.
attempt("approve_payout(C-1001, 2400)",
        lambda: agent.call_tool("approve_payout", claim_id="C-1001", amount=2_400))

# Registered above, deliberately left out of the grant.
attempt("delete_claim(C-1001)",
        lambda: agent.call_tool("delete_claim", claim_id="C-1001"))

print("\nEvery decision above is appended to a SHA-256 chain. Verify it with:")
print("  python -m aegrail.verify ./audit.jsonl\n")
