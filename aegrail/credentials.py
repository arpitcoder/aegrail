"""Where the agent's credential comes from.

Resolved explicit-first:

1. A projected ServiceAccount token someone deliberately mounted. This
   is the BYOC default — kubelet mints it, rotates it before expiry, and
   it dies with the pod. Nothing is ever stored or shared.
2. A static MavOps API key (``mvo_…``) in the environment. For agents
   outside Kubernetes: a Lambda, a VM, CI, a laptop.
3. The default SA token Kubernetes mounts in every pod unasked. Last
   resort only — it carries the `kubernetes` audience, so a control
   plane that checks its own audience will reject it.
4. Nothing, in which case the agent runs on its locally declared
   contract alone and fails closed on irreversible actions.

The projected-token case is why this module exists at all. Those files
are rewritten in place on rotation, so a credential read once at startup
stops working roughly an hour later — silently, because the failure
looks like an ordinary 401. Every read goes back to disk.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

# Where kubelet mounts a projected token by convention. Overridable so
# the same code works with a differently-named volume.
DEFAULT_SA_TOKEN_PATH = "/var/run/secrets/mavops.xyz/serviceaccount/token"

# Kubernetes' own default SA token, present in every pod. Used as a
# fallback when no dedicated audience-scoped volume is mounted.
K8S_DEFAULT_SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"


class Credential:
    """A bearer credential plus how it was obtained."""

    def __init__(self, token: str, source: str):
        self.token = token
        self.source = source

    def __bool__(self) -> bool:
        return bool(self.token)

    def __repr__(self) -> str:
        shown = f"{self.token[:8]}…" if self.token else "none"
        return f"<Credential {self.source} {shown}>"


class CredentialSource:
    """Resolves the current credential on every call.

    File-backed tokens are re-read when their mtime changes, so rotation
    is picked up without a restart. A short cache keeps the steady-state
    poll from stat-ing the file on every single request.
    """

    def __init__(
        self,
        token_path: Optional[str] = None,
        api_token: Optional[str] = None,
        cache_seconds: float = 5.0,
    ):
        self._explicit_path = token_path or os.environ.get("MAVOPS_SA_TOKEN_PATH")
        self._api_token = api_token or os.environ.get("MAVOPS_API_TOKEN", "")
        self._cache_seconds = cache_seconds

        self._lock = threading.Lock()
        self._cached: Optional[Credential] = None
        self._cached_at = 0.0
        self._cached_mtime: Optional[float] = None
        self._cached_path: Optional[str] = None

    # ── resolution ──────────────────────────────────────────────────
    def _candidate_paths(self, *, include_implicit: bool) -> list[str]:
        if self._explicit_path:
            return [self._explicit_path]
        paths = [DEFAULT_SA_TOKEN_PATH]
        if include_implicit:
            # Kubernetes mounts this in every pod whether or not anyone
            # asked for it, and it carries the default `kubernetes`
            # audience — so it is a last resort, never a preference.
            paths.append(K8S_DEFAULT_SA_TOKEN_PATH)
        return paths

    def _read_file_token(self, *, include_implicit: bool) -> Optional[Credential]:
        for path in self._candidate_paths(include_implicit=include_implicit):
            try:
                mtime = os.path.getmtime(path)
                with open(path, "r", encoding="utf-8") as fh:
                    token = fh.read().strip()
            except (OSError, IOError):
                continue
            if token:
                self._cached_mtime = mtime
                self._cached_path = path
                kind = (
                    "k8s-projected"
                    if path != K8S_DEFAULT_SA_TOKEN_PATH
                    else "k8s-default-sa"
                )
                return Credential(token, kind)
        return None

    def _file_changed(self) -> bool:
        if not self._cached_path:
            return True
        try:
            return os.path.getmtime(self._cached_path) != self._cached_mtime
        except OSError:
            return True

    def get(self) -> Credential:
        """Current credential. Cheap to call on every request."""
        now = time.monotonic()
        with self._lock:
            fresh = self._cached is not None and (now - self._cached_at) < self._cache_seconds
            if fresh and self._cached.source == "api-key":
                return self._cached
            if fresh and self._cached.source == "none":
                return self._cached
            if fresh and not self._file_changed():
                return self._cached

            # Precedence is explicit-over-implicit:
            #   1. a token file someone deliberately configured or mounted
            #   2. an API key set in the environment
            #   3. the default SA token Kubernetes mounts unasked
            #
            # (3) below (2) matters: that token carries the `kubernetes`
            # audience, so a control plane checking its own audience will
            # always reject it. Preferring it would mean a correctly-set
            # API key is silently ignored in favour of one that cannot
            # work, and the failure surfaces as an unexplained 401.
            cred = self._read_file_token(include_implicit=False)
            if cred is None and self._api_token:
                cred = Credential(self._api_token, "api-key")
            if cred is None:
                cred = self._read_file_token(include_implicit=True)
            if cred is None:
                cred = Credential("", "none")

            self._cached = cred
            self._cached_at = now
            return cred

    # ── convenience ─────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return bool(self.get())

    def auth_header(self) -> dict:
        cred = self.get()
        return {"Authorization": f"Bearer {cred.token}"} if cred else {}

    def describe(self) -> str:
        cred = self.get()
        if not cred:
            return "no credential — local contract only"
        if cred.source == "api-key":
            return "static API key (MAVOPS_API_TOKEN)"
        if cred.source == "k8s-projected":
            return f"Kubernetes projected token ({self._cached_path})"
        return f"Kubernetes default ServiceAccount token ({self._cached_path})"
