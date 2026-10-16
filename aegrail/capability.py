"""Layer 2 — what an agent may ever do.

Layers 1 and 3 both reason about a *call*: this prompt, this invocation. This
layer reasons about the agent itself, once, and it is the only one that can say
"that tool is not yours" without looking at arguments at all.

The distinction matters because a denylist over a tool set you do not control
is close to useless. Today an agent registers its own tools in-process:

    agent.tool(description="delete_bucket")(delete_bucket)

The control plane never sees that. It can publish `deny_tools`, but that only
blocks what someone thought to name, permits everything they did not, and can
never answer "what could this agent do?" — which is the question asked after an
incident, not before one.

So capability is inverted here:

  * An **allowlist**, not a denylist. A tool absent from the grant is denied,
    even though the process registered it. Absence is a denial, not a gap.

  * **Attested at admission.** The agent sends its inventory and a manifest
    hash when the session opens; the control plane refuses the session if the
    agent claims anything outside its grant. Self-declaration becomes a claim
    that can be checked, once per session rather than per call — so this adds
    nothing to the hot path.

  * **Recorded in the audit chain.** The manifest hash lands in `session.start`,
    making the permitted set provable after the fact alongside what was done.

An empty grant means "not configured" and permits everything. That is
deliberate and it is the one place this layer fails open: an allowlist that
defaults to empty-means-nothing would brick every existing agent the moment the
SDK is upgraded, and a security control that breaks the fleet gets switched off.
Configuration is what turns it on; `strict=True` refuses to run without it.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .verdict import Decision, Tier, Verdict, allow, deny


class CapabilityError(Exception):
    """Raised when an agent's tools cannot be reconciled with its grant."""


@dataclass(frozen=True)
class ToolManifest:
    """The tools a process actually registered, and a hash over them.

    The hash covers names and their destructive flags, sorted, so it is stable
    across process restarts and dict ordering but changes the moment the agent
    gains a tool or reclassifies one. That is exactly the event worth noticing:
    a deploy that quietly widened what an agent can reach.
    """

    tools: tuple = ()
    destructive: frozenset = frozenset()

    @staticmethod
    def from_specs(specs: dict) -> "ToolManifest":
        names = tuple(sorted(specs))
        destructive = frozenset(
            n for n, s in specs.items() if getattr(s, "destructive", False)
        )
        return ToolManifest(tools=names, destructive=destructive)

    def digest(self) -> str:
        payload = json.dumps(
            [[n, n in self.destructive] for n in self.tools],
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def as_dict(self) -> dict:
        return {
            "tools": list(self.tools),
            "destructive": sorted(self.destructive),
            "manifest_hash": self.digest(),
        }


@dataclass
class CapabilityGrant:
    """What the control plane says this agent is allowed to hold.

    Patterns are supported (`read_*`) because tool families are real, but they
    should be rare: the point of the layer is that someone wrote down the list.
    """

    allowed_tools: frozenset = frozenset()
    # Refuse to run at all when the control plane has no grant for this agent.
    # Off by default so upgrading the SDK cannot brick a fleet; on for anything
    # that matters.
    strict: bool = False
    source: str = "unset"

    # Compiled once. `permits` is called per registered tool at session start,
    # never on the hot path, but a fleet policy with many patterns over an
    # agent with many tools is still needless work to repeat.
    _exact: frozenset = field(default=frozenset(), repr=False)
    _patterns: tuple = field(default=(), repr=False)

    def __post_init__(self):
        exact, patterns = set(), []
        for entry in self.allowed_tools:
            if any(ch in entry for ch in "*?["):
                patterns.append(entry)
            else:
                exact.add(entry)
        self._exact = frozenset(exact)
        self._patterns = tuple(patterns)

    @property
    def configured(self) -> bool:
        return bool(self.allowed_tools)

    def permits(self, tool: str) -> bool:
        # An unconfigured grant permits everything. See the module docstring
        # for why this fails open rather than closed.
        if not self.configured:
            return not self.strict
        if tool in self._exact:
            return True
        return any(fnmatch.fnmatchcase(tool, p) for p in self._patterns)

    def evaluate(self, tool: str) -> Decision:
        """The per-call check. Cheap: a set lookup in the common case."""
        if self.permits(tool):
            return allow()
        if not self.configured and self.strict:
            return deny(
                f"no capability grant published for this agent, and strict mode "
                f"is on; refusing '{tool}'"
            )
        return Decision(
            verdict=Verdict.DENY,
            tier=Tier.DETERMINISTIC,
            reasons=[
                f"'{tool}' is not in this agent's capability grant "
                f"({len(self.allowed_tools)} tools allowed, source={self.source})"
            ],
        )

    def reconcile(self, manifest: ToolManifest) -> list:
        """Tools the process registered that the grant does not cover.

        Called once at session start. A non-empty result means the deployed
        code and the published policy disagree about what this agent is —
        which is a deploy problem, not a runtime one, and should be surfaced
        loudly rather than discovered one denied call at a time.
        """
        if not self.configured:
            return []
        return [t for t in manifest.tools if not self.permits(t)]

    @staticmethod
    def from_spec(spec: Optional[dict], *, strict: bool = False,
                  source: str = "mavops") -> "CapabilityGrant":
        spec = spec or {}
        raw: Iterable = spec.get("allowed_tools") or ()
        return CapabilityGrant(
            allowed_tools=frozenset(str(t) for t in raw),
            strict=strict,
            source=source if raw else "unset",
        )
