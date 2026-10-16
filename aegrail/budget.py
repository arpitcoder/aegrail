"""Budget ceilings and the running tally against them."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .exceptions import BudgetExceeded


@dataclass
class Budget:
    """Declared ceilings for one agent session.

    A zero value means "no ceiling on this dimension".
    """

    usd: float = 0.0
    tokens: int = 0
    wall_seconds: int = 0

    def __post_init__(self):
        self._spent_usd = 0.0
        self._spent_tokens = 0
        self._started_at = time.monotonic()
        self._lock = threading.Lock()

    def begin_task(self, *, reset_spend: bool = False) -> None:
        """Start a fresh wall-clock window.

        `wall_seconds` bounds how long one piece of work may take. Measured
        from process start it would instead bound how long the process may
        live, so a resident agent would refuse every tool call once it had
        simply been up longer than the limit — regardless of what it was
        doing. Long-running services call this at the top of each task.

        Cost and token ceilings are cumulative across the session by
        default, since a per-task reset would let an agent spend the cap
        again on every request. Pass reset_spend=True only where each task
        genuinely has its own budget.
        """
        with self._lock:
            self._started_at = time.monotonic()
            if reset_spend:
                self._spent_usd = 0.0
                self._spent_tokens = 0

    # ── reads ────────────────────────────────────────────────────────
    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._spent_usd

    @property
    def spent_tokens(self) -> int:
        with self._lock:
            return self._spent_tokens

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._started_at

    def snapshot(self) -> dict:
        """Current spend, for audit events and OTel span attributes."""
        with self._lock:
            return {
                "usd_spent": round(self._spent_usd, 6),
                "usd_limit": self.usd,
                "tokens_spent": self._spent_tokens,
                "tokens_limit": self.tokens,
                "elapsed_seconds": round(self.elapsed_seconds, 2),
                "wall_limit": self.wall_seconds,
            }

    # ── enforcement ──────────────────────────────────────────────────
    def check(self, add_usd: float = 0.0, add_tokens: int = 0) -> None:
        """Raise BudgetExceeded if this spend would cross a ceiling.

        Checks before charging so the ceiling is never actually crossed.
        """
        if self.wall_seconds and self.elapsed_seconds > self.wall_seconds:
            raise BudgetExceeded(
                "wall_seconds", self.wall_seconds, round(self.elapsed_seconds, 2)
            )

        with self._lock:
            if self.usd and (self._spent_usd + add_usd) > self.usd:
                raise BudgetExceeded("usd", self.usd, round(self._spent_usd + add_usd, 6))
            if self.tokens and (self._spent_tokens + add_tokens) > self.tokens:
                raise BudgetExceeded(
                    "tokens", self.tokens, self._spent_tokens + add_tokens
                )

    def charge(self, usd: float = 0.0, tokens: int = 0) -> None:
        """Record spend after the call succeeded."""
        with self._lock:
            self._spent_usd += usd
            self._spent_tokens += tokens

    def tighten(self, usd: float = 0.0, tokens: int = 0, wall_seconds: int = 0) -> None:
        """Apply a fleet-pushed ceiling.

        Only ever lowers a ceiling — a fleet policy can restrict a session
        further but cannot widen what the agent declared locally.
        """
        if usd and (not self.usd or usd < self.usd):
            self.usd = usd
        if tokens and (not self.tokens or tokens < self.tokens):
            self.tokens = tokens
        if wall_seconds and (not self.wall_seconds or wall_seconds < self.wall_seconds):
            self.wall_seconds = wall_seconds
