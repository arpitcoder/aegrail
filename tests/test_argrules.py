"""Tier 2 must catch the cases a denylist cannot, and only escalate."""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from aegrail.argrules import ArgRule, ArgRuleSet, RuleError
from aegrail.verdict import Verdict


def rules(*raw):
    return ArgRuleSet.from_list(list(raw))


# ── the case that motivates the whole tier ──────────────────────────
def test_same_tool_different_arguments():
    rs = rules({
        "tool": "execute_sql",
        "arg": "query",
        "matches": r"(?i)\b(DROP|TRUNCATE|DELETE\s+FROM)\b",
        "verdict": "hitl",
        "reason": "destructive SQL",
    })
    assert rs.evaluate("execute_sql", {"query": "SELECT count(*) FROM users"}).verdict is Verdict.ALLOW
    assert rs.evaluate("execute_sql", {"query": "DROP TABLE users"}).verdict is Verdict.HITL
    assert rs.evaluate("execute_sql", {"query": "delete from orders"}).verdict is Verdict.HITL


# ── cloud and infrastructure blast radius ───────────────────────────
def test_protected_namespace_is_denied():
    rs = rules({
        "tool": "kubectl_delete",
        "arg": "namespace",
        "in": ["kube-system", "mavops", "prod"],
        "verdict": "deny",
        "reason": "protected namespace",
    })
    assert rs.evaluate("kubectl_delete", {"namespace": "kube-system"}).verdict is Verdict.DENY
    assert rs.evaluate("kubectl_delete", {"namespace": "scratch"}).verdict is Verdict.ALLOW


def test_expensive_instance_needs_a_human():
    rs = rules({
        "tool": "create_instance",
        "arg": "instance_type",
        "matches": r"^(p4d|p5|x2|u-)",
        "verdict": "hitl",
        "reason": "high-cost instance class",
    })
    assert rs.evaluate("create_instance", {"instance_type": "p4d.24xlarge"}).verdict is Verdict.HITL
    assert rs.evaluate("create_instance", {"instance_type": "t4g.medium"}).verdict is Verdict.ALLOW


def test_deleting_anything_not_marked_ephemeral():
    """The useful inversion: safe only if positively marked safe."""
    rs = rules({
        "tool": "delete_*",
        "arg": "tags.ephemeral",
        "ne": "true",
        "verdict": "hitl",
        "reason": "resource is not tagged ephemeral",
    })
    assert rs.evaluate("delete_bucket", {"tags": {"ephemeral": "true"}}).verdict is Verdict.ALLOW
    assert rs.evaluate("delete_bucket", {"tags": {"env": "prod"}}).verdict is Verdict.HITL


def test_scaling_production_to_zero():
    rs = rules({
        "tool": "scale_deployment",
        "arg": "replicas",
        "lt": 1,
        "verdict": "hitl",
        "reason": "scaling to zero takes the service down",
    })
    assert rs.evaluate("scale_deployment", {"replicas": 0}).verdict is Verdict.HITL
    assert rs.evaluate("scale_deployment", {"replicas": 3}).verdict is Verdict.ALLOW


def test_writes_outside_the_workspace():
    rs = rules({
        "tool": "write_file", "arg": "path", "glob": "/etc/*",
        "verdict": "deny", "reason": "system path",
    })
    assert rs.evaluate("write_file", {"path": "/etc/passwd"}).verdict is Verdict.DENY
    assert rs.evaluate("write_file", {"path": "/workspace/out.txt"}).verdict is Verdict.ALLOW


def test_tool_glob_matches_a_family():
    rs = rules({"tool": "aws_*_delete", "verdict": "hitl", "reason": "any AWS delete"})
    assert rs.evaluate("aws_s3_delete", {}).verdict is Verdict.HITL
    assert rs.evaluate("aws_s3_list", {}).verdict is Verdict.ALLOW


# ── the escalate-only invariant ─────────────────────────────────────
def test_a_rule_cannot_grant_allow():
    with pytest.raises(RuleError, match="may only escalate"):
        ArgRule.from_dict({"tool": "anything", "verdict": "allow"})


def test_strictest_matching_rule_wins_and_all_reasons_survive():
    rs = rules(
        {"tool": "execute_sql", "arg": "query", "contains": "DROP",
         "verdict": "hitl", "reason": "destructive statement"},
        {"tool": "execute_sql", "arg": "database", "eq": "prod",
         "verdict": "deny", "reason": "production database"},
    )
    d = rs.evaluate("execute_sql", {"query": "DROP TABLE x", "database": "prod"})
    assert d.verdict is Verdict.DENY
    assert len(d.reasons) == 2


# ── things that must not fire ───────────────────────────────────────
def test_absent_argument_does_not_fire():
    rs = rules({"tool": "t", "arg": "missing", "contains": "x", "verdict": "deny"})
    assert rs.evaluate("t", {"other": "x"}).verdict is Verdict.ALLOW


def test_numeric_bound_against_text_does_not_fire():
    rs = rules({"tool": "t", "arg": "n", "gt": 10, "verdict": "deny"})
    assert rs.evaluate("t", {"n": "not-a-number"}).verdict is Verdict.ALLOW


# ── configuration errors surface at parse time ──────────────────────
def test_bad_regex_is_rejected_when_the_rule_is_written():
    with pytest.raises(RuleError, match="invalid regex"):
        ArgRule.from_dict({"tool": "t", "arg": "q", "matches": "([unclosed"})


def test_misspelled_matcher_is_rejected_rather_than_ignored():
    """A typo must not become a rule that silently enforces nothing."""
    with pytest.raises(RuleError, match="unknown rule field"):
        ArgRule.from_dict({"tool": "t", "arg": "q", "contians": "DROP"})


def test_oversized_argument_is_truncated_not_scanned_whole():
    rs = rules({"tool": "t", "arg": "blob", "contains": "needle", "verdict": "deny"})
    # The needle sits past the cap, so it is not found — bounded work,
    # deliberately, rather than scanning an unbounded payload.
    assert rs.evaluate("t", {"blob": "x" * 20000 + "needle"}).verdict is Verdict.ALLOW


def test_absent_argument_fires_a_negative_matcher():
    """The fail-safe inversion: an untagged resource is not a safe one.

    'escalate anything whose tags.ephemeral is not true' has to catch a
    resource with no tags at all, or the safest rule anyone writes passes
    exactly the unlabelled resources it exists to stop.
    """
    rs = rules({
        "tool": "delete_*", "arg": "tags.ephemeral", "ne": "true",
        "verdict": "hitl", "reason": "not tagged ephemeral",
    })
    assert rs.evaluate("delete_bucket", {}).verdict is Verdict.HITL
    assert rs.evaluate("delete_bucket", {"tags": {}}).verdict is Verdict.HITL
    assert rs.evaluate("delete_bucket", {"tags": {"ephemeral": "true"}}).verdict is Verdict.ALLOW


def test_absent_argument_still_ignores_positive_matchers():
    rs = rules({"tool": "t", "arg": "q", "eq": "x", "verdict": "deny"})
    assert rs.evaluate("t", {}).verdict is Verdict.ALLOW


def test_unexpanded_placeholder_is_rejected():
    """A ${...} left in a contract matches nothing, silently."""
    with pytest.raises(RuleError, match="unexpanded placeholder"):
        ArgRule.from_dict({"tool": "t", "arg": "id", "in": ["${TENANT_IDS}"],
                           "verdict": "deny"})


def test_broad_negative_matcher_is_rejected():
    """tool:* with notIn denies every call that never had the argument."""
    with pytest.raises(RuleError, match="applies to every tool"):
        ArgRule.from_dict({"tool": "*", "arg": "account_id",
                           "notIn": ["123"], "verdict": "deny"})


def test_only_if_present_opts_out_of_absence_firing():
    rs = rules({"tool": "*_query", "arg": "account_id", "onlyIfPresent": True,
                "notIn": ["123"], "verdict": "deny"})
    assert rs.evaluate("db_query", {}).verdict is Verdict.ALLOW
    assert rs.evaluate("db_query", {"account_id": "999"}).verdict is Verdict.DENY
    assert rs.evaluate("db_query", {"account_id": "123"}).verdict is Verdict.ALLOW
