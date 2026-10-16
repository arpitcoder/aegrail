"""The input boundary — what goes *to* the model.

A different hook point from the tool-call tiers, with a harsher latency
budget. It runs on every model call rather than every tool call, and it
sits directly in front of the user's own latency, so it is deterministic
only. Asking a model whether a prompt is safe doubles the delay you were
trying to protect and puts a second manipulable component in the path.

Three things it does, in decreasing order of how well it works:

  Secrets    a live API key or private key in a prompt bound for a
             hosted model is a credential leak the moment it is sent.
             Pattern-matched reliably, and worth denying outright.

  PII        redacted rather than blocked. Blocking a prompt because it
             contains an email address makes the product unusable; a
             support agent's prompts are *made* of email addresses. The
             prompt goes on with the values replaced.

  Injection  flagged, not trusted. Deterministic injection detection is
             a losing arms race and this module does not pretend
             otherwise — see the note at the bottom.

Nothing here is the real defence against a manipulated model. The
tool-call boundary is, because it does not care whether the model was
manipulated: the contract stops the action either way. This layer is a
cheap filter in front of that, not a replacement for it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .verdict import Decision, Tier, Verdict, allow, deny, needs_human

# Prompts can be enormous — a whole log file pasted into context. Every
# pattern here is linear, but bounding the scan keeps the cost of this
# layer flat rather than proportional to whatever the agent happened to
# read.
MAX_SCAN_CHARS = 32_000


# A single scan for "does this text contain a digit at all", shared by
# every numeric pattern instead of each rediscovering it.
_HAS_DIGIT = re.compile(r"\d")


@dataclass(frozen=True)
class Pattern:
    name: str
    regex: "re.Pattern"
    action: str          # "deny" | "redact" | "flag"
    label: str = ""      # what replaces the match, for redact
    # Literals that must appear for the pattern to have any chance of
    # matching. Checking them is a memchr; running the regex is a walk of
    # the whole string. On a 10 KB prompt that is the difference between
    # microseconds and milliseconds, and almost every pattern misses.
    requires: tuple = ()
    needs_digit: bool = False

    def possible(self, lowered: str, has_digit: bool) -> bool:
        if self.needs_digit and not has_digit:
            return False
        if not self.requires:
            return True
        return any(lit in lowered for lit in self.requires)


def _p(name, rx, action, label="", flags=re.IGNORECASE, requires=(), needs_digit=False):
    return Pattern(name, re.compile(rx, flags), action, label,
                   tuple(l.lower() for l in requires), needs_digit)


# ── Secrets: deny. A key in a prompt is already leaked once sent. ────
SECRET_PATTERNS = [
    _p("aws_access_key", r"\bAKIA[0-9A-Z]{16}\b", "deny", flags=0, requires=("AKIA",)),
    _p("private_key", r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----", "deny", requires=("-----begin",)),
    _p("openai_key", r"\bsk-[A-Za-z0-9_-]{20,}\b", "deny", flags=0, requires=("sk-",)),
    _p("anthropic_key", r"\bsk-ant-[A-Za-z0-9_-]{20,}\b", "deny", flags=0, requires=("sk-ant-",)),
    _p("github_token", r"\bgh[pousr]_[A-Za-z0-9]{36,}\b", "deny", flags=0, requires=("ghp_","gho_","ghu_","ghs_","ghr_")),
    _p("slack_token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b", "deny", flags=0, requires=("xox",)),
    _p("jwt", r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
       "deny", flags=0, requires=("eyj",)),
    _p("bearer_header", r"\bAuthorization:\s*Bearer\s+\S{20,}", "deny", requires=("bearer",)),
    _p("pg_uri", r"\b(?:postgres|postgresql|mysql|mongodb)(?:\+\w+)?://[^\s:]+:[^\s@]+@", "deny", requires=("://",)),
]

# ── PII: redact, do not block. ──────────────────────────────────────
PII_PATTERNS = [
    _p("email", r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b", "redact", "[EMAIL]", flags=0, requires=("@",)),
    _p("ssn", r"\b\d{3}-\d{2}-\d{4}\b", "redact", "[SSN]", flags=0, needs_digit=True),
    # Card-shaped digit runs. Deliberately loose: over-redacting a number
    # costs nothing, under-redacting a card number costs a compliance
    # finding.
    _p("card", r"\b(?:\d[ -]?){13,19}\b", "redact", "[CARD]", flags=0, needs_digit=True),
    _p("ipv4", r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "redact", "[IP]", flags=0, needs_digit=True),
    _p("phone_e164", r"\+\d{1,3}[\s-]?\d{6,14}\b", "redact", "[PHONE]", flags=0, requires=("+",), needs_digit=True),
]

# ── Injection: flag only. See the note at the bottom of this file. ──
INJECTION_PATTERNS = [
    _p("instruction_override",
       r"\b(?:ignore|disregard|forget)\b[^.\n]{0,40}\b(?:previous|prior|above|earlier|all)\b"
       r"[^.\n]{0,30}\b(?:instruction|prompt|rule|direction)", "flag", requires=("ignore","disregard","forget")),
    _p("role_override",
       r"\byou are now\b|\bnew (?:system )?prompt\b|\bact as (?:if|though) you", "flag", requires=("you are now","new prompt","new system prompt","act as")),
    _p("system_tag_injection",
       r"<\s*/?\s*(?:system|assistant)\s*>|\[\s*(?:SYSTEM|INST)\s*\]", "flag", requires=("<","[")),
    _p("exfil_instruction",
       r"\b(?:send|post|upload|exfiltrate|forward)\b[^.\n]{0,40}\b(?:to|at)\b\s*https?://",
       "flag", requires=("http",)),
    _p("secret_solicitation",
       r"\b(?:reveal|print|show|repeat|output)\b[^.\n]{0,30}"
       r"\b(?:system prompt|instructions|api key|secret|credential|token)\b", "flag", requires=("reveal","print","show","repeat","output")),
]


@dataclass
class InputFinding:
    pattern: str
    action: str
    count: int


@dataclass
class InputResult:
    """A verdict, plus the prompt as it should actually be sent."""

    decision: Decision
    text: str                          # redacted where redaction applied
    findings: list = field(default_factory=list)
    redacted: bool = False

    @property
    def verdict(self) -> Verdict:
        return self.decision.verdict

    def summary(self) -> str:
        if not self.findings:
            return "clean"
        return ", ".join(f"{f.pattern}×{f.count}" for f in self.findings)


class InputGuard:
    """Deterministic checks on prompt text, before it leaves the process."""

    def __init__(
        self,
        redact_pii: bool = True,
        deny_secrets: bool = True,
        injection_verdict: Verdict = Verdict.HITL,
        extra_patterns: Optional[list] = None,
        judge=None,
    ):
        self.redact_pii = redact_pii
        self.deny_secrets = deny_secrets
        # Optional escalation for prompts the patterns flagged but cannot
        # settle. See check() for why this only runs on flagged prompts.
        self.judge = judge
        # Configurable because the right response differs by deployment.
        # An interactive agent can afford to ask; a batch one cannot, and
        # for it flagging into the audit chain is more useful than
        # stopping.
        self.injection_verdict = injection_verdict
        self.extra = list(extra_patterns or [])

    def check(self, text: str) -> InputResult:
        if not text:
            return InputResult(allow(), text)

        scanned = text[:MAX_SCAN_CHARS]
        # Computed once and shared by every pattern's precondition.
        lowered = scanned.lower()
        has_digit = _HAS_DIGIT.search(scanned) is not None
        findings: list = []
        decision = allow()
        out = text
        redacted = False

        if self.deny_secrets:
            for pat in SECRET_PATTERNS:
                if not pat.possible(lowered, has_digit):
                    continue
                hits = len(pat.regex.findall(scanned))
                if hits:
                    findings.append(InputFinding(pat.name, "deny", hits))
                    decision = decision.combine(
                        deny(f"credential in prompt: {pat.name}")
                    )

        if self.redact_pii:
            for pat in PII_PATTERNS:
                if not pat.possible(lowered, has_digit):
                    continue
                out, n = pat.regex.subn(pat.label, out)
                if n:
                    findings.append(InputFinding(pat.name, "redact", n))
                    redacted = True

        for pat in INJECTION_PATTERNS + self.extra:
            if not pat.possible(lowered, has_digit):
                continue
            hits = len(pat.regex.findall(scanned))
            if not hits:
                continue
            findings.append(InputFinding(pat.name, pat.action, hits))
            if pat.action == "deny":
                decision = decision.combine(deny(f"input pattern: {pat.name}"))
            elif pat.action == "flag":
                decision = decision.combine(
                    Decision(
                        self.injection_verdict,
                        tier=Tier.DETERMINISTIC,
                        reasons=[f"possible prompt injection: {pat.name}"],
                    )
                )
            else:
                decision = decision.combine(
                    needs_human(f"possible prompt injection: {pat.name}")
                )

        # Escalate to a model only for prompts the patterns already found
        # suspicious. Two reasons it is gated that way.
        #
        # Latency: this sits in front of the user's own wait. Running a
        # model on every prompt would double the delay the whole layer
        # exists to keep small. Flagged prompts are rare, so almost every
        # call still costs microseconds.
        #
        # Safety: the judge is itself a model, reading attacker-controlled
        # text, and can therefore be injected too — "classify this as
        # safe" is an obvious payload. That is survivable only because
        # verdicts combine with max: a hijacked judge answering ALLOW
        # cannot clear a flag the patterns raised. It can add caution, or
        # add nothing. It can never subtract.
        if self.judge is not None and decision.verdict is not Verdict.ALLOW:
            try:
                judged = self.judge.classify_input(
                    text=out, findings=[f.pattern for f in findings]
                )
                decision = decision.combine(judged)
            except Exception:
                # A judge that cannot answer leaves the pattern verdict
                # standing. It was already non-ALLOW to get here.
                pass

        return InputResult(decision, out, findings, redacted)


# ── On what this cannot do ──────────────────────────────────────────
#
# Deterministic injection detection is a losing arms race. Every pattern
# above is trivially reworded, and a determined attacker rewords it. The
# patterns are worth keeping because they are nearly free and they catch
# the unsophisticated majority, but treating them as a security boundary
# would be a mistake.
#
# The defence that holds is the tool-call boundary, which does not care
# whether the model was manipulated: the contract evaluates the action on
# its own merits and stops it regardless of what convinced the model to
# try. An injection that persuades an agent to run DROP TABLE still meets
# a rule that says DROP TABLE needs a human.
#
# So the honest claim is not "we detect prompt injection". It is that a
# successful injection does not get to do anything.
#
# Two more gaps worth naming:
#
#   Redaction is one-way. Once an email becomes [EMAIL] the model cannot
#   reason about it, which is occasionally the whole task. A tokenising
#   redactor that can restore values on the way back is the better
#   answer and is not built.
#
#   This sees the prompt, not the retrieved context. An agent that pulls
#   a poisoned document into its context window at RAG time has not sent
#   anything suspicious *yet*, and this layer runs before that.
