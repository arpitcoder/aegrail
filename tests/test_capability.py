"""Layer 2 — what an agent may ever do.

The property under test throughout: a tool absent from the grant is denied,
regardless of what the process registered, what arguments were passed, or what
any other layer decided.
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from aegrail.capability import CapabilityError, CapabilityGrant, ToolManifest
from aegrail.verdict import Verdict


class _Spec:
    def __init__(self, destructive=False):
        self.destructive = destructive


# ── the allowlist itself ─────────────────────────────────────────────
def test_absence_is_denial():
    """The whole point of an allowlist. A denylist permits what it forgot."""
    g = CapabilityGrant.from_spec({"allowed_tools": ["datadog_query"]})
    assert g.evaluate("datadog_query").verdict is Verdict.ALLOW
    assert g.evaluate("delete_bucket").verdict is Verdict.DENY
    # Nobody wrote 'shell_exec' anywhere. That is exactly why it is denied.
    assert g.evaluate("shell_exec").verdict is Verdict.DENY


def test_patterns_are_supported_but_anchored():
    g = CapabilityGrant.from_spec({"allowed_tools": ["read_*"]})
    assert g.permits("read_logs")
    assert not g.permits("unread_logs")      # must not match mid-string
    assert not g.permits("read")             # the underscore is required


def test_unconfigured_grant_permits_everything():
    """Fails open, deliberately.

    An allowlist that defaults to permitting nothing would deny every call in
    every existing deployment the moment the SDK is upgraded. A security
    control that breaks the fleet gets switched off, and then protects nothing.
    """
    g = CapabilityGrant.from_spec({})
    assert not g.configured
    assert g.evaluate("anything_at_all").verdict is Verdict.ALLOW


def test_strict_mode_refuses_an_unconfigured_grant():
    """The opposite posture, for anyone who wants it."""
    g = CapabilityGrant.from_spec({}, strict=True)
    assert g.evaluate("anything_at_all").verdict is Verdict.DENY


def test_denial_says_why():
    g = CapabilityGrant.from_spec({"allowed_tools": ["a", "b"]}, source="fleet")
    d = g.evaluate("c")
    assert "not in this agent's capability grant" in d.reasons[0]
    assert "2 tools allowed" in d.reasons[0]


# ── the manifest ─────────────────────────────────────────────────────
def test_manifest_hash_is_stable_across_ordering():
    """Restarts must not change the hash, or it is useless as an identity."""
    a = ToolManifest.from_specs({"b": _Spec(), "a": _Spec()})
    b = ToolManifest.from_specs({"a": _Spec(), "b": _Spec()})
    assert a.digest() == b.digest()


def test_manifest_hash_changes_when_a_tool_is_added():
    """The event worth noticing: a deploy that widened what an agent reaches."""
    before = ToolManifest.from_specs({"a": _Spec()})
    after = ToolManifest.from_specs({"a": _Spec(), "delete_everything": _Spec()})
    assert before.digest() != after.digest()


def test_manifest_hash_changes_when_a_tool_is_reclassified():
    """Same names, but one became irreversible. That is a different agent."""
    before = ToolManifest.from_specs({"a": _Spec(destructive=False)})
    after = ToolManifest.from_specs({"a": _Spec(destructive=True)})
    assert before.digest() != after.digest()


# ── reconciliation: deployed code vs published policy ────────────────
def test_reconcile_reports_tools_outside_the_grant():
    g = CapabilityGrant.from_spec({"allowed_tools": ["read_logs"]})
    man = ToolManifest.from_specs({"read_logs": _Spec(), "drop_table": _Spec()})
    assert g.reconcile(man) == ["drop_table"]


def test_reconcile_is_silent_when_unconfigured():
    g = CapabilityGrant.from_spec({})
    man = ToolManifest.from_specs({"anything": _Spec()})
    assert g.reconcile(man) == []


# ── integration with the agent ───────────────────────────────────────
def _agent(tmp_path, **kw):
    from aegrail import Agent, AuditSink, Budget
    return Agent(
        identity="cap-test/v1",
        budget=Budget(usd=10.0, wall_seconds=600),
        audit=AuditSink.file(str(tmp_path / "audit.jsonl")),
        poll_policy=False,
        resume_pending=False,
        **kw,
    )


def test_agent_denies_a_tool_outside_its_grant(tmp_path):
    from aegrail.exceptions import PolicyDenied

    a = _agent(tmp_path, capability=CapabilityGrant.from_spec(
        {"allowed_tools": ["allowed_tool"]}))

    def allowed_tool() -> str:
        return "ok"

    def forbidden_tool() -> str:
        return "should never run"

    a.tool(description="allowed_tool")(allowed_tool)
    a.tool(description="forbidden_tool")(forbidden_tool)

    assert a.call_tool("allowed_tool") == "ok"
    with pytest.raises(PolicyDenied):
        a.call_tool("forbidden_tool")


def test_registering_a_tool_does_not_grant_it(tmp_path):
    """The core weakness this layer closes.

    Tools are declared by the agent process. Before layer 2 that declaration
    was also the authorisation — registering a tool was all it took to be able
    to call it.
    """
    from aegrail.exceptions import PolicyDenied

    a = _agent(tmp_path, capability=CapabilityGrant.from_spec(
        {"allowed_tools": ["read_only"]}))

    def exfiltrate() -> str:
        return "sent"

    a.tool(description="exfiltrate")(exfiltrate)
    with pytest.raises(PolicyDenied):
        a.call_tool("exfiltrate")


def test_capability_denial_cannot_be_argued_away(tmp_path):
    """No argument makes a forbidden tool acceptable.

    Layer 2 runs before argument inspection precisely so that the denial does
    not depend on what was passed.
    """
    from aegrail.exceptions import PolicyDenied

    a = _agent(tmp_path, capability=CapabilityGrant.from_spec(
        {"allowed_tools": ["read_only"]}))

    def delete_bucket(name: str = "", dry_run: bool = True) -> str:
        return "deleted"

    a.tool(description="delete_bucket")(delete_bucket)
    for kwargs in ({}, {"dry_run": True}, {"name": "x", "dry_run": True}):
        with pytest.raises(PolicyDenied):
            a.call_tool("delete_bucket", **kwargs)


def test_attest_records_the_manifest_and_flags_violations(tmp_path):
    import json

    path = tmp_path / "audit.jsonl"
    a = _agent(tmp_path, capability=CapabilityGrant.from_spec(
        {"allowed_tools": ["good"]}))

    def good() -> str:
        return "ok"

    def bad() -> str:
        return "no"

    a.tool(description="good")(good)
    a.tool(description="bad")(bad)

    man = a.attest()
    events = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    attested = [e for e in events if e.get("kind") == "capability.attested"]
    assert attested, "attestation was not written to the audit chain"
    rec = attested[-1]
    assert rec["manifest_hash"] == man.digest()
    assert rec["violations"] == ["bad"]


def test_strict_attest_refuses_to_start_on_violation(tmp_path):
    a = _agent(tmp_path, strict_capability=True,
               capability=CapabilityGrant.from_spec(
                   {"allowed_tools": ["good"]}, strict=True))

    def bad() -> str:
        return "no"

    a.tool(description="bad")(bad)
    with pytest.raises(CapabilityError) as e:
        a.attest()
    assert "outside its capability grant" in str(e.value)


def test_strict_attest_refuses_when_no_grant_published(tmp_path):
    """An agent with no declared capability cannot be held to one."""
    a = _agent(tmp_path, strict_capability=True,
               capability=CapabilityGrant.from_spec({}, strict=True))

    with pytest.raises(CapabilityError) as e:
        a.attest()
    assert "no grant was published" in str(e.value)


def test_lenient_attest_warns_but_still_denies_at_runtime(tmp_path):
    """Warning is not weakness — the calls are denied either way."""
    from aegrail.exceptions import PolicyDenied

    a = _agent(tmp_path, capability=CapabilityGrant.from_spec(
        {"allowed_tools": ["good"]}))

    def bad() -> str:
        return "no"

    a.tool(description="bad")(bad)
    a.attest()  # does not raise
    with pytest.raises(PolicyDenied):
        a.call_tool("bad")
