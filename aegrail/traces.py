"""Ship contract evaluations to the control plane as traces.

Observability is not a second product bolted on beside enforcement — the
contract runtime already sees every model call and every tool call, so
the trace is a byproduct of the decision it was making anyway.

OTel spans (see otel.py) cover the case where a customer has their own
collector. This covers the case where they want the data in MavOps
itself. Both are emitted from the same evaluation; neither is required.

Posting is fire-and-forget on a background thread. A trace is telemetry:
losing one is acceptable, blocking a tool call to record one is not.
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import re
import threading
import urllib.error
import urllib.request
import warnings
from datetime import datetime, timezone
from typing import Optional

from .credentials import CredentialSource

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)


def _looks_like_uuid(value: str) -> bool:
    return bool(_UUID_RE.match(value))


class TraceClient:
    """Fire-and-forget writer for /v1/traces."""

    def __init__(
        self,
        api_base: Optional[str] = None,
        credentials: Optional[CredentialSource] = None,
        agent_id: Optional[str] = None,
        max_queue: int = 256,
    ):
        self.api_base = (
            api_base or os.environ.get("MAVOPS_API_BASE", "https://api.mavops.xyz")
        ).rstrip("/")
        self.credentials = credentials or CredentialSource()
        self.agent_id = agent_id or os.environ.get("MAVOPS_AGENT_ID", "")
        # The control plane parses this as a UUID and drops it silently when it
        # is not one — traces still arrive, just unlinked from any agent, which
        # looks like a server bug from the outside. Say so here instead.
        if self.agent_id and not _looks_like_uuid(self.agent_id):
            warnings.warn(
                f"MAVOPS_AGENT_ID={self.agent_id!r} is not a UUID; the control "
                "plane will discard it and these traces will not be linked to "
                "an agent.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.enabled = os.environ.get("AEGRAIL_TRACES_DISABLED", "").lower() not in (
            "1",
            "true",
            "yes",
        )

        # Bounded so a control plane outage cannot grow the queue without
        # limit inside a long-running agent.
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None

    def start(self) -> "TraceClient":
        if not self.enabled or self._worker is not None:
            return self
        self._worker = threading.Thread(
            target=self._drain, daemon=True, name="aegrail-traces"
        )
        self._worker.start()
        # The common shape of an agent in production is a process that does its
        # work and exits. The worker is a daemon, so without this every trace
        # still queued at that moment is discarded — and silently, which is the
        # worst way to lose telemetry: the dashboard just looks quiet.
        atexit.register(self.stop)
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Flush what is queued, then stop the worker.

        Blocks up to `timeout` seconds. Telemetry must not hang a process on
        the way out, so the join is bounded and anything still unsent past the
        deadline is abandoned — but the normal case now flushes instead of
        dropping. Idempotent: calling it twice, or after atexit has run, is a
        no-op.
        """
        worker, self._worker = self._worker, None
        if worker is None:
            return
        self._stop.set()
        worker.join(timeout=timeout)

    # ── emit ────────────────────────────────────────────────────────
    def record(
        self,
        *,
        operation: str,
        status: str = "ok",
        model: str = "",
        provider: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        duration_ms: int = 0,
        tool_names: Optional[list] = None,
        session_id: str = "",
        error_message: str = "",
        payload: Optional[dict] = None,
    ) -> None:
        """Queue one trace. Never raises, never blocks."""
        if not self.enabled:
            return

        body = {
            "provider": provider or "aegrail",
            "model": model or "n/a",
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "status": status,
            "tool_call_count": len(tool_names or []),
            "tool_names": tool_names or [],
            "session_id": session_id,
        }
        if self.agent_id:
            body["agent_id"] = self.agent_id
        if error_message:
            body["error_message"] = error_message
        if payload:
            body["payload"] = payload

        try:
            self._q.put_nowait(body)
        except queue.Full:
            # Dropping telemetry is the correct failure here. The audit
            # chain is the record that must not be lost; this is not it.
            pass

    # ── worker ──────────────────────────────────────────────────────
    def _drain(self) -> None:
        # Exit only when asked to stop AND the queue is empty, so a stop()
        # flushes the backlog rather than discarding it. The short poll keeps
        # shutdown responsive without busy-waiting.
        while True:
            try:
                body = self._q.get(timeout=0.25)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            try:
                self._post(body)
            except Exception:
                pass  # telemetry is best-effort by design

    def _post(self, body: dict) -> None:
        cred = self.credentials.get()
        if not cred:
            return
        req = urllib.request.Request(
            f"{self.api_base}/v1/traces",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {cred.token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10).read()
