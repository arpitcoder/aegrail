"""Standalone audit-chain verifier.

    python -m aegrail.verify ./audit.jsonl

Runs offline. Contacts nothing. An auditor can confirm the chain is
intact without an account, an API key, or trusting MavOps — the proof is
self-contained in the events themselves.

Exit code 0 when the chain verifies, 1 when it does not.
"""
from __future__ import annotations

import json
import sys

from .audit import AuditSink

GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m aegrail.verify <audit.jsonl>", file=sys.stderr)
        return 2

    path = argv[0]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            events = [json.loads(line) for line in fh if line.strip()]
    except FileNotFoundError:
        print(f"{RED}no such file: {path}{RESET}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"{RED}malformed JSONL at line {exc.lineno}: {exc.msg}{RESET}", file=sys.stderr)
        return 1

    ok, msg = AuditSink.verify(events)

    print(f"\n{BOLD}aegrail audit verification{RESET}")
    print(f"{DIM}  file    {path}{RESET}")
    print(f"{DIM}  events  {len(events)}{RESET}")
    if events:
        print(f"{DIM}  window  {events[0].get('kind')} → {events[-1].get('kind')}{RESET}")

    if ok:
        print(f"\n  {GREEN}{BOLD}✓ CHAIN INTACT{RESET}  {msg}\n")
        _summarize(events)
        return 0

    print(f"\n  {RED}{BOLD}✗ TAMPERING DETECTED{RESET}  {msg}\n")
    return 1


def _summarize(events: list[dict]) -> None:
    counts: dict[str, int] = {}
    for e in events:
        counts[e.get("kind", "?")] = counts.get(e.get("kind", "?"), 0) + 1
    print(f"{BOLD}  event breakdown{RESET}")
    for kind, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>4}  {kind}")
    print()


if __name__ == "__main__":
    raise SystemExit(main())
