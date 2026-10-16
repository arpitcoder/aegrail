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

import json
import os
import queue
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Optional

from .credentials import CredentialSource


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
        return self

    def stop(self) -> None:
        self._stop.set()

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
        while not self._stop.is_set():
            try:
                body = self._q.get(timeout=1.0)
            except queue.Empty:
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
