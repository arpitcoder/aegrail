"""Tier 2 — asking a model whether a specific call is destructive.

Some tools cannot be classified by name. `execute_sql`, `run_shell`,
`http_post` and `write_file` are harmless or catastrophic depending
entirely on their arguments, and no denylist separates:

    execute_sql("SELECT count(*) FROM users")
    execute_sql("DROP TABLE users")

So a model looks at the actual invocation. Three properties make that
safe to do:

1. It runs only after the deterministic layer has already ruled, and its
   answer is combined with `max` — it can raise a verdict, never lower
   one. See verdict.py. A prompt-injected judge makes the system more
   cautious, never less.
2. It is unreachable-safe. No answer means escalate, never allow.
3. It runs in the control plane, not in the agent. The customer's tool
   arguments contain their SQL, their file paths, their payloads; those
   go to whichever model *they* configured, in *their* cluster, and the
   agent process needs no model credentials at all.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from .credentials import CredentialSource
from .verdict import Decision, Tier, Verdict


class JudgeClient:
    """Asks the control plane to classify one tool invocation."""

    def __init__(
        self,
        api_base: Optional[str] = None,
        credentials: Optional[CredentialSource] = None,
        timeout_seconds: float = 8.0,
        enabled: Optional[bool] = None,
    ):
        self.api_base = (
            api_base or os.environ.get("MAVOPS_API_BASE", "https://api.mavops.xyz")
        ).rstrip("/")
        self.credentials = credentials or CredentialSource()
        self.timeout = timeout_seconds
        if enabled is None:
            enabled = os.environ.get("AEGRAIL_JUDGE_DISABLED", "").lower() not in (
                "1",
                "true",
                "yes",
            )
        self.enabled = enabled

        # Verdicts are cached by (tool, arguments) so a loop calling the
        # same thing repeatedly does not pay for a model call each time.
        # Keyed on exact arguments: a different argument is a different
        # question, and that is the whole reason this tier exists.
        self._cache: dict = {}
        self._cache_max = 512

    def _cache_key(self, tool: str, args: dict) -> str:
        return tool + "|" + json.dumps(args, sort_keys=True, default=str)

    def classify(self, *, tool: str, description: str, args: dict) -> Decision:
        """Judge one invocation. Never raises; escalates on any failure."""
        if not self.enabled:
            return Decision(Verdict.ALLOW, tier=Tier.JUDGE)

        key = self._cache_key(tool, args)
        if key in self._cache:
            return self._cache[key]

        cred = self.credentials.get()
        if not cred:
            # Never configured, as opposed to configured and failing.
            # Escalating here would mean an agent with no judge set up at
            # all sends every inspectable call to a human, which is not
            # caution — it is an unusable product. A judge that HAS been
            # reached and then fails still escalates; see below.
            return Decision(Verdict.ALLOW, tier=Tier.JUDGE)

        body = json.dumps({
            "tool": tool,
            "description": description,
            "arguments": args,
        }).encode()

        started = time.monotonic()
        try:
            req = urllib.request.Request(
                f"{self.api_base}/v1/judge/classify",
                data=body,
                headers={
                    "Authorization": f"Bearer {cred.token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            # Unreachable, slow, or broken. An unanswered question about
            # a potentially destructive action is not permission.
            return self._escalate(
                tool, f"judge unavailable ({type(exc).__name__})", cached_key=None
            )

        latency_ms = int((time.monotonic() - started) * 1000)

        raw = str(data.get("verdict", "")).lower()
        verdict = {
            "allow": Verdict.ALLOW,
            "hitl": Verdict.HITL,
            "review": Verdict.HITL,
            "deny": Verdict.DENY,
            "block": Verdict.DENY,
        }.get(raw)

        if verdict is None:
            # A malformed answer is not an answer.
            return self._escalate(
                tool, f"judge returned an unusable verdict {raw!r}", cached_key=None
            )

        decision = Decision(
            verdict=verdict,
            tier=Tier.JUDGE,
            reasons=[str(data.get("reasoning", "")).strip()[:400] or f"judge: {raw}"],
            judge_model=str(data.get("model", "")),
            judge_latency_ms=latency_ms,
        )
        self._remember(key, decision)
        return decision

    def _escalate(self, tool: str, reason: str, cached_key) -> Decision:
        """Fail closed. Not cached — the next call should retry."""
        return Decision(
            verdict=Verdict.HITL,
            tier=Tier.JUDGE,
            reasons=[f"{reason}; escalating '{tool}' to a human"],
        )

    def _remember(self, key: str, decision: Decision) -> None:
        if len(self._cache) >= self._cache_max:
            # Crude eviction, but this is a per-process cache of a
            # bounded tool surface, not a general-purpose LRU.
            self._cache.clear()
        self._cache[key] = decision
