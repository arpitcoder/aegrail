"""Fleet policy client.

Polls MavOps for this agent's effective contract every 60 seconds. The
server is the source of truth for fleet-wide clauses; Aegrail is what
actually enforces them at the tool-call boundary.

Without MAVOPS_API_TOKEN the client runs in permissive local mode so
`python main.py` works offline during development.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .credentials import CredentialSource
from typing import Optional


@dataclass
class PolicySnapshot:
    budget_usd_per_day: float = 0.0
    budget_tokens_per_day: int = 0
    budget_max_wall_seconds: int = 0
    require_hitl_for: set = field(default_factory=set)
    deny_tools: set = field(default_factory=set)
    # Layer 2. Empty means "no grant published", which permits everything —
    # see capability.py for why this direction. A denylist says what is
    # forbidden; this says what is permitted, and absence from it is a denial.
    allowed_tools: set = field(default_factory=set)
    allowed_models: set = field(default_factory=set)  # empty == all allowed
    policy_hash: str = ""
    source: str = "local-default"


class PolicyClient:
    """Thread-safe reader over a background-refreshed policy snapshot."""

    def __init__(
        self,
        agent_name: str,
        api_base: Optional[str] = None,
        token: Optional[str] = None,
        poll_interval: int = 60,
    ):
        self.agent_name = agent_name
        self.api_base = (
            api_base or os.environ.get("MAVOPS_API_BASE", "https://api.mavops.xyz")
        ).rstrip("/")
        # Credential is resolved per request, not captured here: a
        # projected ServiceAccount token rotates while the pod runs.
        self.credentials = CredentialSource(api_token=token)
        self.poll_interval = poll_interval
        self._lock = threading.Lock()
        self._snapshot = PolicySnapshot()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._on_change = None

    # ── lifecycle ───────────────────────────────────────────────────
    @classmethod
    def start(
        cls,
        agent_name: str,
        *,
        poll_interval: int = 60,
        on_change=None,
        block_seconds: float = 5.0,
    ) -> "PolicyClient":
        client = cls(agent_name, poll_interval=poll_interval)
        client._on_change = on_change
        first = threading.Event()

        def loop():
            while not client._stop.is_set():
                try:
                    client._fetch_once()
                except Exception:
                    pass  # keep last-known-good policy on transient failure
                finally:
                    first.set()
                client._stop.wait(client.poll_interval)

        client._thread = threading.Thread(target=loop, daemon=True, name="aegrail-policy")
        client._thread.start()
        # Block briefly so the first policy is in place before the first
        # request — otherwise the agent runs unbounded for a moment.
        first.wait(block_seconds)
        return client

    def stop(self) -> None:
        self._stop.set()

    def _fetch_once(self) -> None:
        cred = self.credentials.get()
        if not cred:
            return  # permissive local mode

        url = f"{self.api_base}/v1/policies/effective?agent={self.agent_name}"
        headers = {
            "Authorization": f"Bearer {cred.token}",
            "Accept": "application/json",
        }
        # The server supports ETag on policy_hash; a 304 means nothing
        # changed and we can keep the snapshot we already have.
        with self._lock:
            current_hash = self._snapshot.policy_hash
        if current_hash:
            headers["If-None-Match"] = f'"{current_hash}"'

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 304:
                    return
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 304:
                return
            raise

        # Server shape (service.EffectiveResult):
        #   { "agent": ..., "policy_hash": ..., "sources": [...],
        #     "spec": { "budget": {"usd_per_day": ..., "tokens_per_day": ...,
        #                          "max_wall_seconds": ...},
        #               "require_hitl_for": [...], "deny_tools": [...],
        #               "allowed_tools": [...], "allowed_models": [...] } }
        spec = data.get("spec") or {}
        budget = spec.get("budget") or {}
        sources = data.get("sources") or []

        snap = PolicySnapshot(
            budget_usd_per_day=float(budget.get("usd_per_day", 0) or 0),
            budget_tokens_per_day=int(budget.get("tokens_per_day", 0) or 0),
            budget_max_wall_seconds=int(budget.get("max_wall_seconds", 0) or 0),
            require_hitl_for=set(spec.get("require_hitl_for") or []),
            deny_tools=set(spec.get("deny_tools") or []),
            allowed_tools=set(spec.get("allowed_tools") or []),
            allowed_models=set(spec.get("allowed_models") or []),
            policy_hash=str(data.get("policy_hash", "")),
            source=(
                ", ".join(s.get("name", "?") for s in sources) if sources else "mavops"
            ),
        )

        with self._lock:
            changed = snap.policy_hash != self._snapshot.policy_hash
            self._snapshot = snap

        if changed and self._on_change:
            try:
                self._on_change(snap)
            except Exception:
                pass

    # ── reads ───────────────────────────────────────────────────────
    @property
    def snapshot(self) -> PolicySnapshot:
        with self._lock:
            return self._snapshot

    def requires_hitl(self, action: str) -> bool:
        snap = self.snapshot
        if action in snap.require_hitl_for:
            return True
        # Support prefix rules like "delete_*"
        for rule in snap.require_hitl_for:
            if rule.endswith("*") and action.startswith(rule[:-1]):
                return True
        return False

    def tool_allowed(self, tool: str) -> bool:
        snap = self.snapshot
        if tool in snap.deny_tools:
            return False
        for rule in snap.deny_tools:
            if rule.endswith("*") and tool.startswith(rule[:-1]):
                return False
        return True

    def model_allowed(self, model: str) -> bool:
        snap = self.snapshot
        return not snap.allowed_models or model in snap.allowed_models

    def __repr__(self) -> str:
        s = self.snapshot
        return (
            f"<PolicyClient {self.agent_name} cred={self.credentials.get().source} source={s.source} "
            f"hitl={sorted(s.require_hitl_for)} deny={sorted(s.deny_tools)}>"
        )
