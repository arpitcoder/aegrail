"""Every shipped contract pack must load and must not break ordinary work.

Packs are data, not code, so nothing else catches a mistake in them. Two real
bugs shipped in packs before this file existed:

  * an observability rule with tool "*" and a negative matcher, which denied
    every call in the product because absence fires a negative matcher;
  * an insurance rule written `present: true` to mean "require this argument",
    when it actually fires when the argument IS present — so it blocked every
    payout that had an amount, i.e. all of them.

Both parse cleanly. Both are catastrophic. The tests here are therefore less
about schema and more about behaviour: does the pack still permit the work the
agent exists to do?
"""
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import yaml

from aegrail.argrules import ArgRuleSet
from aegrail.capability import CapabilityGrant
from aegrail.sequence import SequenceTracker
from aegrail.verdict import Verdict

# Packs sit at the repo root. Honour an override so the same suite runs
# against a checkout laid out differently.
ROOT = os.environ.get(
    "AEGRAIL_CONTRACTS",
    os.path.join(os.path.dirname(__file__), "..", "contracts"),
)
PACKS = sorted(glob.glob(os.path.join(ROOT, "*.yaml")) +
               glob.glob(os.path.join(ROOT, "domains", "*.yaml")))


def _load(path):
    return yaml.safe_load(open(path))


def test_packs_are_discovered():
    """A typo in the glob would make every test below vacuously pass."""
    assert PACKS, f"no contract packs found under {ROOT}"
    names = {os.path.basename(p) for p in PACKS}
    assert "infra-baseline.yaml" in names
    assert "observability.yaml" in names
    assert "insurance.yaml" in names


@pytest.mark.parametrize("path", PACKS, ids=lambda p: os.path.basename(p))
def test_pack_parses(path):
    """Parse-time guards reject unsafe rules, so loading is a real check."""
    spec = _load(path)
    ArgRuleSet.from_list(spec.get("argumentRules"))
    SequenceTracker.from_list(spec.get("sequenceRules"))
    CapabilityGrant.from_spec({"allowed_tools": spec.get("allowedTools")})


@pytest.mark.parametrize("path", PACKS, ids=lambda p: os.path.basename(p))
def test_pack_does_not_deny_everything(path):
    """The failure that makes a pack useless rather than wrong.

    A rule broad enough to fire on an unrelated call turns the product off. If
    a plausible no-argument call to an innocuous tool is refused, the pack is
    almost certainly over-broad.
    """
    rules = ArgRuleSet.from_list(_load(path).get("argumentRules"))
    for tool in ("read_thing", "list_items", "get_status", "healthcheck"):
        d = rules.evaluate(tool, {})
        assert d.verdict is Verdict.ALLOW, (
            f"{os.path.basename(path)} refuses a benign '{tool}' with no "
            f"arguments: {d.reasons}"
        )


@pytest.mark.parametrize("path", PACKS, ids=lambda p: os.path.basename(p))
def test_granted_tools_are_not_denied_by_the_same_pack(path):
    """A grant that lists a tool the pack unconditionally denies is incoherent.

    Layer 2 says the agent may hold it; layer 3 then refuses every call. That
    combination is always a mistake in the pack, and it is invisible until an
    agent tries the tool in production.
    """
    spec = _load(path)
    granted = spec.get("allowedTools") or []
    if not granted:
        pytest.skip("pack publishes no capability grant")
    # Only rules with no `arg` are unconditional. A rule that denies
    # deny_claim when reason_code is missing is argument-conditional and
    # perfectly coherent — the tool is usable, just not carelessly.
    unconditional = ArgRuleSet.from_list([
        r for r in (spec.get("argumentRules") or [])
        if not r.get("arg") and str(r.get("verdict", "")).lower() in ("deny", "block")
    ])
    for tool in granted:
        d = unconditional.evaluate(tool, {})
        assert d.verdict is not Verdict.DENY, (
            f"{os.path.basename(path)} grants '{tool}' but denies it "
            f"unconditionally: {d.reasons}"
        )


@pytest.mark.parametrize("path", PACKS, ids=lambda p: os.path.basename(p))
def test_every_rule_explains_itself(path):
    """A verdict a human cannot act on is a support ticket.

    The reason reaches the approver in the HITL request and the auditor in the
    chain, so an empty or placeholder one defeats both.
    """
    for r in (_load(path).get("argumentRules") or []):
        reason = (r.get("reason") or "").strip()
        assert len(reason) > 15, f"{os.path.basename(path)}: weak reason {reason!r} on {r.get('tool')}"


@pytest.mark.parametrize("path", PACKS, ids=lambda p: os.path.basename(p))
def test_no_unexpanded_placeholders(path):
    """`${TENANT_IDS}` left in a shipped pack becomes a literal string match.

    It reads as configured and enforces nothing.
    """
    # Comments legitimately mention ${...} when documenting the guard against
    # it, so only rule lines count.
    offending = [
        l for l in open(path).read().splitlines()
        if "${" in l and not l.lstrip().startswith("#")
    ]
    assert not offending, f"{os.path.basename(path)} has unexpanded placeholders: {offending}"


# ── behavioural checks, per pack ─────────────────────────────────────
# Generic structure tests cannot know that a claims agent must be able to pay
# a claim. These encode the happy path each pack must not obstruct, and the
# specific harm it exists to stop.

def _pack(name):
    return _load(os.path.join(ROOT, "domains", name))


def test_insurance_allows_routine_claim_handling():
    rules = ArgRuleSet.from_list(_pack("insurance.yaml")["argumentRules"])
    routine = [
        ("read_claim", {"claim_id": "C-1"}),
        ("assess_claim", {"claim_id": "C-1", "damage_estimate": 4200}),
        ("approve_payout", {"amount": 4200, "currency": "USD",
                            "payee_matches_policyholder": "true"}),
        ("write_note", {"claim_id": "C-1", "text": "Settled per estimate."}),
        ("search_claims", {"policy_id": "P-1", "limit": 20}),
    ]
    for tool, kw in routine:
        d = rules.evaluate(tool, kw)
        assert d.verdict is Verdict.ALLOW, f"{tool} obstructed: {d.reasons}"


@pytest.mark.parametrize("kwargs,expected,label", [
    ({"currency": "USD"}, Verdict.HITL, "payout with no amount"),
    ({"amount": 75_000, "currency": "USD", "payee_matches_policyholder": "true"},
     Verdict.HITL, "above the unattended limit"),
    ({"amount": 900_000, "currency": "USD", "payee_matches_policyholder": "true"},
     Verdict.DENY, "beyond any automated authority"),
    ({"amount": 500, "currency": "USD", "payee_matches_policyholder": "false"},
     Verdict.HITL, "payee is not the policyholder"),
])
def test_insurance_payout_thresholds(kwargs, expected, label):
    rules = ArgRuleSet.from_list(_pack("insurance.yaml")["argumentRules"])
    assert rules.evaluate("approve_payout", kwargs).verdict is expected, label


def test_insurance_denials_need_a_reason_and_a_human():
    rules = ArgRuleSet.from_list(_pack("insurance.yaml")["argumentRules"])
    # A denial is an adverse regulated act: never fully automatic.
    assert rules.evaluate("deny_claim", {"claim_id": "C", "reason_code": "NC-14"}).verdict is Verdict.HITL
    # And one that cannot be explained must not happen at all.
    assert rules.evaluate("deny_claim", {"claim_id": "C"}).verdict is Verdict.DENY


@pytest.mark.parametrize("field", ["race", "religion", "genetic_data"])
def test_insurance_refuses_protected_attributes(field):
    """Unlawful in most markets, and silent: a model handed the field uses it."""
    rules = ArgRuleSet.from_list(_pack("insurance.yaml")["argumentRules"])
    d = rules.evaluate("assess_claim", {"claim_id": "C", field: "x"})
    assert d.verdict is Verdict.DENY, f"{field} reached an assessment decision"


def test_insurance_blocks_read_then_send():
    """Neither call is wrong alone, which is why no per-call rule sees it."""
    seq = SequenceTracker.from_list(_pack("insurance.yaml")["sequenceRules"])
    seq.record("read_policyholder")
    assert seq.evaluate("http_post").verdict is Verdict.DENY


def test_insurance_grant_excludes_policy_termination():
    """A claims agent must not be able to end someone's cover."""
    grant = CapabilityGrant.from_spec(
        {"allowed_tools": _pack("insurance.yaml")["allowedTools"]})
    for tool in ("cancel_policy", "rescind_policy", "update_premium",
                 "delete_claim", "export_bulk"):
        assert not grant.permits(tool), f"claims agent was granted {tool}"


def test_observability_allows_a_real_investigation():
    rules = ArgRuleSet.from_list(_pack("observability.yaml")["argumentRules"])
    for tool, kw in [
        ("datadog_query", {"query": "avg:latency", "lookback_days": 1, "start_time": "-1h"}),
        ("cloudwatch_query", {"log_group": "/prod/app", "lookback_days": 1, "start_time": "-1h"}),
        ("write_brief", {"title": "t", "content": "c"}),
    ]:
        assert rules.evaluate(tool, kw).verdict is Verdict.ALLOW, f"{tool}: {kw}"


def test_observability_stops_cost_blowout_and_exfiltration():
    spec = _pack("observability.yaml")
    rules = ArgRuleSet.from_list(spec["argumentRules"])
    assert rules.evaluate("datadog_query", {
        "query": "x", "lookback_days": 90, "start_time": "-90d", "limit": 5_000_000
    }).verdict is not Verdict.ALLOW

    seq = SequenceTracker.from_list(spec["sequenceRules"])
    seq.record("datadog_query")
    assert seq.evaluate("http_post").verdict is Verdict.DENY


def test_baseline_applies_to_every_domain():
    """infra-baseline is the layer no pack repeats, so it must stand alone."""
    rules = ArgRuleSet.from_list(_load(os.path.join(ROOT, "infra-baseline.yaml"))["argumentRules"])
    assert rules.evaluate("run_shell", {"command": "rm -rf /"}).verdict is Verdict.DENY
    assert rules.evaluate("k8s_delete_namespace",
                          {"namespace": "kube-system"}).verdict is Verdict.DENY
