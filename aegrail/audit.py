"""Tamper-evident audit chain.

Every event carries the SHA-256 digest of the event before it. Rewriting
any past event breaks every digest that follows, so tampering is
detectable by replaying the chain.

The digest is computed HERE — inside the agent process, before the event
is written to disk or shipped to a collector. That is the whole point: by
the time an event reaches the network, the storage layer, or MavOps's own
servers, its position in the chain is already frozen. Nobody downstream
can rewrite history without the break showing up in `verify()`.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from typing import Any, Callable, Iterable

GENESIS = "0" * 64


def _canonical(obj: Any) -> str:
    """Stable JSON so the same event always hashes to the same digest."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _digest(prev_hash: str, payload: dict) -> str:
    return hashlib.sha256((prev_hash + _canonical(payload)).encode("utf-8")).hexdigest()


class AuditSink:
    """Append-only hash-linked event log.

    Construct with one of the classmethods rather than directly:

        AuditSink.file("/tmp/audit.jsonl")   # local JSONL
        AuditSink.memory()                    # tests / dry runs
        AuditSink.null()                      # disabled
    """

    def __init__(self, emit: Callable[[dict], None], *, name: str = "sink"):
        self._emit = emit
        self._name = name
        self._lock = threading.Lock()
        self._prev_hash = GENESIS
        self._seq = 0

    # ── constructors ────────────────────────────────────────────────
    @classmethod
    def file(cls, path: str) -> "AuditSink":
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        def emit(event: dict) -> None:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(_canonical(event) + "\n")

        sink = cls(emit, name=f"file:{path}")
        sink.path = path
        # Resume the chain if the file already has events, so restarts do
        # not silently fork the chain into two unlinked segments.
        sink._resume_from(path)
        return sink

    @classmethod
    def memory(cls) -> "AuditSink":
        events: list[dict] = []
        sink = cls(events.append, name="memory")
        sink.events = events
        return sink

    @classmethod
    def null(cls) -> "AuditSink":
        return cls(lambda _event: None, name="null")

    # ── chain state ─────────────────────────────────────────────────
    def _resume_from(self, path: str) -> None:
        if not os.path.exists(path):
            return
        last = None
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        last = json.loads(line)
                    except json.JSONDecodeError:
                        continue
        if last:
            self._prev_hash = last.get("hash", GENESIS)
            self._seq = int(last.get("seq", 0))

    @property
    def head(self) -> str:
        """Digest of the most recent event. This is the chain tip."""
        with self._lock:
            return self._prev_hash

    # ── write ───────────────────────────────────────────────────────
    def record(self, kind: str, **fields: Any) -> dict:
        """Append one event and return it (including its digest).

        The digest is computed before `self._emit` is called, so whatever
        the emit target is — disk, network, collector — it receives an
        already-sealed event.
        """
        with self._lock:
            self._seq += 1
            payload = {
                "seq": self._seq,
                "id": str(uuid.uuid4()),
                "ts": time.time(),
                "kind": kind,
                "prev": self._prev_hash,
                **fields,
            }
            payload["hash"] = _digest(self._prev_hash, payload)
            self._prev_hash = payload["hash"]

        self._emit(payload)
        return payload

    # ── verification ────────────────────────────────────────────────
    @staticmethod
    def verify(events: Iterable[dict]) -> tuple[bool, str]:
        """Replay a chain and confirm every link.

        Returns (ok, message). An auditor can run this without contacting
        MavOps at all — the proof is self-contained in the events.
        """
        prev = GENESIS
        count = 0
        for event in events:
            count += 1
            if event.get("prev") != prev:
                return False, (
                    f"chain break at seq={event.get('seq')}: "
                    f"prev={event.get('prev')!r} expected={prev!r}"
                )
            recorded = event.get("hash")
            body = {k: v for k, v in event.items() if k != "hash"}
            expected = _digest(prev, body)
            if recorded != expected:
                return False, (
                    f"digest mismatch at seq={event.get('seq')}: "
                    f"recorded={recorded!r} computed={expected!r}"
                )
            prev = recorded
        return True, f"chain intact across {count} events (head={prev[:12]}…)"

    @staticmethod
    def verify_file(path: str) -> tuple[bool, str]:
        def _read():
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

        return AuditSink.verify(_read())

    def __repr__(self) -> str:
        return f"<AuditSink {self._name} seq={self._seq} head={self._prev_hash[:12]}…>"
