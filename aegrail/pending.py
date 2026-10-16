"""Durable state for approvals an agent is waiting on.

Without this, a frozen agent is a blocked thread and nothing else. If the
pod restarts — a deploy, an OOM kill, a node drain — the pending action
disappears silently. A human approves in Slack twenty minutes later and
nothing happens, with no error anywhere to explain why.

So the pending call is written to Redis before the wait begins, and a
resumer thread looks for orphaned records on startup. The approval
outlives the process that asked for it.

What is durable is the *action*, not the caller. A restart cannot resume
the HTTP request that triggered the work — that connection is gone. It
can, and does, still carry out the tool call the human approved.

Redis is optional. Without it the agent still enforces and still blocks;
it just cannot survive a restart mid-approval, which is the pre-existing
behaviour.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from .redis_client import RedisClient, RedisError

KEY_PREFIX = "aegrail:pending"

# Records outlive the approval window so a decision made near the
# deadline still finds its record, but do not accumulate forever.
DEFAULT_TTL_SECONDS = 24 * 3600


@dataclass
class PendingCall:
    """A tool call held up waiting for a human."""

    review_id: str
    identity: str
    tool: str
    args: dict = field(default_factory=dict)
    kwargs: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    approval_url: str = ""

    def key(self) -> str:
        return f"{KEY_PREFIX}:{self.identity}:{self.review_id}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(raw: str) -> "PendingCall":
        return PendingCall(**json.loads(raw))

    @property
    def waiting_seconds(self) -> float:
        return time.time() - self.created_at


class PendingStore:
    """Redis-backed record of what an agent is waiting on.

    Degrades to a no-op when Redis is unreachable: an agent that cannot
    reach Redis should still enforce its contract. Losing durability is
    bad; refusing to run is worse.
    """

    def __init__(
        self,
        identity: str,
        url: Optional[str] = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        self.identity = identity
        self.ttl_seconds = ttl_seconds
        self.url = url or os.environ.get("REDIS_URL", "")
        self._redis: Optional[RedisClient] = None
        self.reason: str = "not configured"

        if not self.url:
            return
        try:
            client = RedisClient(self.url)
            if client.ping():
                self._redis = client
                self.reason = "connected"
            else:
                self.reason = "ping failed"
        except Exception as exc:
            self.reason = f"{type(exc).__name__}: {exc}"

    @property
    def enabled(self) -> bool:
        return self._redis is not None

    def describe(self) -> str:
        if self.enabled:
            return f"redis ({self.url})"
        return f"in-memory only — {self.reason}"

    # ── writes ──────────────────────────────────────────────────────
    def put(self, call: PendingCall) -> None:
        if not self._redis:
            return
        try:
            self._redis.set(call.key(), call.to_json(), ttl_seconds=self.ttl_seconds)
        except RedisError:
            pass  # durability is best-effort; enforcement is not

    def clear(self, call: PendingCall) -> None:
        if not self._redis:
            return
        try:
            self._redis.delete(call.key())
        except RedisError:
            pass

    # ── reads ───────────────────────────────────────────────────────
    def orphans(self) -> list:
        """Pending calls left behind by a previous process."""
        if not self._redis:
            return []
        out = []
        try:
            for key in self._redis.scan(f"{KEY_PREFIX}:{self.identity}:*"):
                raw = self._redis.get(key)
                if not raw:
                    continue
                try:
                    out.append(PendingCall.from_json(raw))
                except (TypeError, ValueError):
                    # A record we cannot parse is a record we cannot act
                    # on. Drop it rather than crash the resumer on every
                    # pass forever.
                    self._redis.delete(key)
        except RedisError:
            return []
        return out


class Resumer:
    """Carries out approvals that landed while nobody was listening.

    Runs once shortly after startup and then on an interval. On the first
    pass it finds work stranded by a previous process; later passes catch
    anything whose caller went away mid-wait.
    """

    def __init__(
        self,
        store: PendingStore,
        poll_review: Callable[[str], object],
        execute: Callable[[PendingCall], None],
        on_rejected: Optional[Callable[[PendingCall, str], None]] = None,
        interval_seconds: int = 30,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.store = store
        self.poll_review = poll_review
        self.execute = execute
        self.on_rejected = on_rejected
        self.interval = interval_seconds
        self.log = logger or (lambda _msg: None)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "Resumer":
        if not self.store.enabled or self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True, name="aegrail-resumer")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        # A short delay so the first sweep does not race the rest of
        # startup, then steady state.
        self._stop.wait(3)
        while not self._stop.is_set():
            try:
                self.sweep()
            except Exception as exc:
                self.log(f"resumer sweep failed: {type(exc).__name__}: {exc}")
            self._stop.wait(self.interval)

    def sweep(self) -> int:
        """One pass. Returns how many pending calls were resolved."""
        resolved = 0
        for call in self.store.orphans():
            try:
                decision = self.poll_review(call.review_id)
            except Exception:
                continue  # transient; leave the record for the next pass

            status = getattr(decision, "status", "pending")
            if status == "pending":
                continue

            if status == "approved":
                self.log(
                    f"resuming '{call.tool}' — approved after "
                    f"{int(call.waiting_seconds)}s, across a restart"
                )
                try:
                    self.execute(call)
                except Exception as exc:
                    self.log(f"resume of '{call.tool}' failed: {exc}")
            elif status in ("rejected", "expired"):
                self.log(f"'{call.tool}' was {status}; discarding pending call")
                if self.on_rejected:
                    try:
                        self.on_rejected(call, status)
                    except Exception:
                        pass

            self.store.clear(call)
            resolved += 1
        return resolved
