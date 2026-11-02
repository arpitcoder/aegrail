"""What happened, read back out of the chain.

When an agent is stopped, the first question is never "what is its state" —
it is "what did it already do, and which of it cannot be taken back". That
answer is already in the audit log; it is just not in a shape anyone wants
to read at two in the morning.

So this is a report, not an action. It deliberately does not undo anything:

  * Unwinding during an incident means running new writes with code in a
    process you have just decided you do not trust.
  * Compensation is not inverse. Restarting a service gives you a cold
    cache and a new PID, not the state you had.
  * The post-incident state is the evidence. Reverting it automatically
    bulldozes the thing an investigator needs.

A human reads this and decides. That is slower, and it is right.

    python -m aegrail.incident ./audit.jsonl
"""
from __future__ import annotations

import json
import sys
import time
from typing import Iterable, Optional

# Events that change something in the world, in the order they matter.
_EXECUTED = "tool.executed"
_DENIED = "tool.denied"
_HALTED = "agent.halted"
_RESUMED = "agent.resumed"
_SAFE = "agent.safe_point"
_DRIFT = "capability.drift"


def _ts(value: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(value))


def _args(event: dict) -> str:
    """Arguments as written, or a note that they were not recorded.

    Tool arguments are off by default in traces, and an incident report
    that silently shows nothing would read as "it did nothing" rather than
    "we did not keep this".
    """
    call = event.get("call") or {}
    kwargs = call.get("kwargs") or {}
    positional = call.get("args") or []
    if not kwargs and not positional:
        return ""
    parts = [repr(a) for a in positional]
    parts += [f"{k}={v!r}" for k, v in kwargs.items()]
    return ", ".join(parts)


def build(events: Iterable[dict], *, since: Optional[float] = None) -> dict:
    """Summarise a chain. `since` defaults to the last halt."""
    events = [e for e in events if isinstance(e, dict)]

    halts = [e for e in events if e.get("kind") == _HALTED]
    last_halt = halts[-1] if halts else None
    if since is None and last_halt is not None:
        # Everything this agent did in the session that ended in a stop.
        # Anchoring at the halt itself would report an empty incident,
        # since by then it had stopped doing things.
        since = 0.0

    window = [e for e in events if since is None or e.get("ts", 0) >= since]
    executed = [e for e in window if e.get("kind") == _EXECUTED]

    return {
        "halt": last_halt,
        "resumed": any(e.get("kind") == _RESUMED for e in window),
        "safe_point": next(
            (e for e in window if e.get("kind") == _SAFE), None),
        "drift": [e for e in window if e.get("kind") == _DRIFT],
        "irreversible": [e for e in executed if e.get("destructive")],
        "reversible": [e for e in executed if not e.get("destructive")],
        "denied": [e for e in window if e.get("kind") == _DENIED],
        "events": len(window),
    }


def render(report: dict) -> str:
    out = []
    w = out.append
    halt = report.get("halt")

    w("")
    if halt:
        mode = halt.get("mode", "immediate")
        w(f"  AGENT STOPPED   {_ts(halt.get('ts', 0))}   ({mode})")
        w(f"  reason          {halt.get('reason', '—')}")
        still_open = halt.get("open_sections") or []
        if still_open:
            w(f"  open at stop    {', '.join(still_open)}")
        if report.get("safe_point"):
            w("  safe point      reached — no work was left half-done")
        elif still_open:
            w("  safe point      NOT reached — the sections above were open")
    else:
        w("  No halt recorded in this chain.")

    irreversible = report["irreversible"]
    w("")
    w(f"  Cannot be taken back ({len(irreversible)})")
    if not irreversible:
        w("    nothing — no irreversible action ran")
    for e in irreversible:
        w(f"    {_ts(e.get('ts', 0))}  {e.get('tool')}({_args(e)})")

    reversible = report["reversible"]
    w("")
    w(f"  Ran, but changed nothing irreversibly ({len(reversible)})")
    for e in reversible[-10:]:
        w(f"    {_ts(e.get('ts', 0))}  {e.get('tool')}({_args(e)})")
    if len(reversible) > 10:
        w(f"    … and {len(reversible) - 10} earlier")

    denied = report["denied"]
    if denied:
        w("")
        w(f"  Refused ({len(denied)})")
        for e in denied[-10:]:
            reasons = e.get("reasons") or []
            why = reasons[0] if reasons else e.get("reason", "")
            w(f"    {_ts(e.get('ts', 0))}  {e.get('tool')}  {why}")

    if report["drift"]:
        w("")
        w("  Tool surface changed since review — see capability.drift")

    w("")
    w("  This report changes nothing. Decide what to undo, and do it")
    w("  deliberately — an automatic revert during an incident is new")
    w("  writes from a process you have already stopped trusting.")
    w("")
    return "\n".join(out)


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    path = argv[0] if argv else "./audit.jsonl"
    try:
        with open(path, encoding="utf-8") as fh:
            events = [json.loads(line) for line in fh if line.strip()]
    except FileNotFoundError:
        print(f"no audit log at {path}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"{path} is not a valid audit chain: {exc}", file=sys.stderr)
        return 2

    print(render(build(events)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
