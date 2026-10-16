"""Aegrail — runtime contract enforcement for AI agents.

The neural model proposes; the symbolic contract disposes. Aegrail runs
inside the agent process and evaluates a contract at the tool-call
boundary — budget ceilings, denied tools, human-in-the-loop on
irreversible actions — before any tool function executes.

Every decision is written to a tamper-evident audit chain and emitted as
an OpenTelemetry span, so observability and compliance evidence fall out
of enforcement rather than being bolted on beside it.

    from aegrail import Agent, Budget, AuditSink

    agent = Agent(
        identity="cleanup-bot/v1",
        budget=Budget(usd=5.0, tokens=100_000, wall_seconds=300),
        audit=AuditSink.file("./audit.jsonl"),
    )

    @agent.tool(destructive=True)
    def delete_records(table: str, older_than_days: int) -> str:
        return db.execute(...)

MIT licensed. Read the code that audits your agents.
"""

from .agent import Agent
from .argrules import ArgRule, ArgRuleSet, RuleError
from .audit import AuditSink
from .budget import Budget
from .capability import CapabilityError, CapabilityGrant, ToolManifest
from .exceptions import (
    AegrailError,
    BudgetExceeded,
    HITLRejected,
    HITLTimeout,
    PolicyDenied,
)
from .hitl import HITLClient, ReviewDecision
from .inputguard import InputGuard
from .policy import PolicyClient, PolicySnapshot
from .sequence import SequenceRule, SequenceTracker
from .shadow import Mode, ObservationStore
from .verdict import Decision, Tier, Verdict

__version__ = "0.4.0"

__all__ = [
    # the agent itself
    "Agent",
    "Budget",
    "AuditSink",
    # layer 1 — what goes to the model
    "InputGuard",
    # layer 2 — what the agent may ever do
    "CapabilityGrant",
    "ToolManifest",
    "CapabilityError",
    # layer 3 — what this call may do
    "ArgRule",
    "ArgRuleSet",
    "RuleError",
    "SequenceRule",
    "SequenceTracker",
    "Decision",
    "Tier",
    "Verdict",
    # onboarding an agent nobody has classified yet
    "Mode",
    "ObservationStore",
    # control plane
    "PolicyClient",
    "PolicySnapshot",
    "HITLClient",
    "ReviewDecision",
    # errors
    "AegrailError",
    "BudgetExceeded",
    "HITLRejected",
    "HITLTimeout",
    "PolicyDenied",
    "__version__",
]
