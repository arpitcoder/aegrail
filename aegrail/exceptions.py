"""Exceptions raised when a contract clause blocks an action.

Every exception here means the agent tried to do something the contract
did not permit. They are not errors in the agent's logic — they are the
enforcement layer doing its job.
"""
from __future__ import annotations


class AegrailError(Exception):
    """Base for every Aegrail enforcement exception."""


class BudgetExceeded(AegrailError):
    """A budget ceiling (USD, tokens, or wall-clock) was hit."""

    def __init__(self, dimension: str, limit: float, attempted: float):
        self.dimension = dimension
        self.limit = limit
        self.attempted = attempted
        super().__init__(
            f"budget exceeded on {dimension}: limit={limit}, attempted={attempted}"
        )


class PolicyDenied(AegrailError):
    """A tool call was denied outright by fleet policy."""

    def __init__(self, tool: str, reason: str = ""):
        self.tool = tool
        self.reason = reason
        msg = f"policy denies tool '{tool}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class HITLRejected(AegrailError):
    """A human reviewed the action and rejected it."""

    def __init__(self, tool: str, reason: str = ""):
        self.tool = tool
        self.reason = reason
        msg = f"human rejected '{tool}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class HITLTimeout(AegrailError):
    """No human decided within the approval window. Timeout means deny."""

    def __init__(self, tool: str, waited_seconds: int):
        self.tool = tool
        self.waited_seconds = waited_seconds
        super().__init__(
            f"no human decision on '{tool}' after {waited_seconds}s — denying "
            f"(silence is not consent)"
        )
