"""Two permitted calls that must not happen in that order.

Every tier before this one judges a call on its own. Reading a claim is the
agent's job. Posting to a URL is the agent's job. Reading a claim and *then*
posting is exfiltration — and no per-call check can see it, because neither
call is wrong.

    python examples/exfiltration.py

No network, no API key, no services.
"""
import os
import sys

# So a fresh clone just runs: python examples/<name>.py
# Python puts the *script's* directory on sys.path, not the working directory,
# so without this the import fails for anyone who has not pip-installed first.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from aegrail import Agent, Budget, CapabilityGrant, SequenceTracker
from aegrail.exceptions import PolicyDenied

agent = Agent(
    identity="claims-triage/v1",
    budget=Budget(usd=10.0, wall_seconds=600),
    capability=CapabilityGrant.from_spec({
        "allowed_tools": ["read_claim", "search_claims", "http_post"],
    }),

    # Layer 4 — rules over the trajectory rather than the call.
    sequence=SequenceTracker.from_list([
        {"afterAny": ["read_claim*", "search_claims*"],
         "then": "http_*",
         "verdict": "deny",
         # Bounded, or a single read at startup would restrict every send for
         # the rest of the process. The two happening close together is the
         # suspicious part, not the two ever happening.
         "withinSeconds": 600,
         "reason": "outbound request shortly after reading policyholder data"},
    ]),
)

CLAIMS = {"C-1001": {"policyholder": "A. Nigam", "ssn": "xxx-xx-1234"}}


@agent.tool(description="read a claim")
def read_claim(claim_id: str) -> dict:
    return CLAIMS.get(claim_id, {})


@agent.tool(description="search claims")
def search_claims(q: str) -> list:
    return list(CLAIMS)


@agent.tool(description="post to a url")
def http_post(url: str, body: str = "") -> str:
    return f"posted {len(body)} bytes to {url}"


def attempt(label, fn):
    try:
        print(f"  {label:<44} {fn()}")
    except PolicyDenied as e:
        print(f"  {label:<44} DENIED\n  {'':<44} {e}")


agent.attest()

print("\nclaims-triage/v1 — each call below is individually permitted\n")

# On its own, posting is ordinary work. Nothing sensitive has been read.
attempt("http_post(status.internal)",
        lambda: agent.call_tool("http_post", url="https://status.internal",
                                body="ok"))

# So is reading a claim. This is the tool the agent exists to use.
attempt("read_claim(C-1001)",
        lambda: agent.call_tool("read_claim", claim_id="C-1001"))

# The same post as the first one. Same tool, same arguments, now refused —
# the only thing that changed is what the agent did before it.
attempt("http_post(status.internal)  [after the read]",
        lambda: agent.call_tool("http_post", url="https://status.internal",
                                body="ok"))

print("""
  The first and third calls are identical. Capability, arguments and budget
  all say yes to both. The difference is history, and it is deterministic:
  no model was asked, and the denial replays the same way a year from now.

  Verify the chain with:
    python -m aegrail.verify ./audit.jsonl
""")
