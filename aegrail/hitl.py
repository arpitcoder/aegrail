"""Human-in-the-loop approvals.

When a contract clause says an action needs a human, Aegrail submits a
review to MavOps, which fans it out to Slack with Approve / Reject
buttons, then blocks until a decision lands or the window expires.

Timeout means DENY. Silence is not consent.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from .credentials import CredentialSource
from .exceptions import HITLRejected, HITLTimeout


@dataclass
class ReviewDecision:
    status: str  # approved | rejected | expired | pending
    review_id: str
    approval_url: str = ""
    reason: str = ""
    edited_content: str = ""
    decided_via: str = ""


class HITLClient:
    """Submits reviews to MavOps and waits on the human decision."""

    def __init__(
        self,
        api_base: Optional[str] = None,
        token: Optional[str] = None,
        agent_id: Optional[str] = None,
    ):
        self.api_base = (
            api_base or os.environ.get("MAVOPS_API_BASE", "https://api.mavops.xyz")
        ).rstrip("/")
        # Resolved per request — projected SA tokens rotate.
        self.credentials = CredentialSource(api_token=token)
        self.agent_id = agent_id or os.environ.get("MAVOPS_AGENT_ID", "")

    @property
    def enabled(self) -> bool:
        return bool(self.credentials.get())

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.api_base}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.credentials.get().token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(
            f"{self.api_base}{path}",
            headers={
                "Authorization": f"Bearer {self.credentials.get().token}",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ── submit ──────────────────────────────────────────────────────
    def submit(
        self,
        *,
        title: str,
        content: str,
        kind: str = "generic",
        notify_channels: Optional[list] = None,
        metadata: Optional[dict] = None,
        expires_in_seconds: int = 3600,
    ) -> ReviewDecision:
        body = {
            "title": title,
            "kind": kind,
            "content": content,
            "notify_channels": notify_channels or ["slack"],
            "metadata": metadata or {},
            "expires_in_seconds": expires_in_seconds,
        }
        if self.agent_id:
            body["agent_id"] = self.agent_id

        res = self._post("/v1/reviews", body)
        return ReviewDecision(
            status=res.get("status", "pending"),
            review_id=res.get("id", ""),
            approval_url=res.get("approval_url", ""),
        )

    def poll(self, review_id: str) -> ReviewDecision:
        res = self._get(f"/v1/reviews/{review_id}")
        return ReviewDecision(
            status=res.get("status", "pending"),
            review_id=review_id,
            reason=res.get("rejection_reason", "") or "",
            edited_content=res.get("approved_content", "") or "",
            decided_via=res.get("decided_via", "") or "",
        )

    # ── submit + block ──────────────────────────────────────────────
    def request_approval(
        self,
        *,
        tool: str,
        title: str,
        content: str,
        metadata: Optional[dict] = None,
        timeout_seconds: int = 300,
        poll_interval: int = 3,
        on_submitted=None,
    ) -> ReviewDecision:
        """Submit and block until a human decides.

        Raises HITLRejected on reject, HITLTimeout when the window closes
        with no decision.
        """
        decision = self.submit(
            title=title,
            content=content,
            kind="generic",
            metadata={**(metadata or {}), "tool": tool},
            expires_in_seconds=timeout_seconds,
        )
        if on_submitted:
            on_submitted(decision)

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            time.sleep(poll_interval)
            try:
                current = self.poll(decision.review_id)
            except Exception:
                continue  # transient — keep waiting inside the window

            if current.status == "approved":
                return current
            if current.status == "rejected":
                raise HITLRejected(tool, current.reason)
            if current.status == "expired":
                raise HITLTimeout(tool, timeout_seconds)

        raise HITLTimeout(tool, timeout_seconds)
