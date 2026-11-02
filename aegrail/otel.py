"""OpenTelemetry emission for contract evaluations.

Every contract decision becomes a span, and the spans follow the GenAI
semantic conventions rather than a shape of our own. That distinction is
the whole point: a customer's existing Datadog or Grafana already knows
what `execute_tool` and `gen_ai.tool.name` mean, so enforcement shows up
in the dashboards they have instead of the dashboard we would like them
to install.

What the conventions do not cover — the verdict, which tier decided, the
audit digest — stays under `aegrail.*`. Extra attributes are allowed, and
dropping ours to look compliant would throw away the only part of this
telemetry nobody else emits.

Conventions as of the `open-telemetry/semantic-conventions-genai` repo:
operation `execute_tool`, span name `execute_tool {gen_ai.tool.name}`,
span kind INTERNAL, `error.type` on failure, arguments opt-in. They are
still marked Development, so names may move; `GENAI_CONVENTIONS` pins the
version this was written against.

OTel is an optional dependency. Without it installed, every function here
degrades to a no-op so the agent still runs.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Optional

# The semantic-conventions-genai revision these attribute names follow.
# Recorded on the span so a trace tells you which spelling it used — the
# conventions are in Development and have already moved repositories once.
GENAI_CONVENTIONS = "semconv-genai/development"

_TRACER = None
_AVAILABLE = False

try:  # pragma: no cover - depends on optional extra
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import SpanKind, Status, StatusCode

    _AVAILABLE = True
except Exception:  # pragma: no cover
    _otel_trace = None
    Status = StatusCode = SpanKind = None


def _tracer():
    global _TRACER
    if not _AVAILABLE:
        return None
    if _TRACER is None:
        _TRACER = _otel_trace.get_tracer("aegrail", "0.1.0")
    return _TRACER


def enabled() -> bool:
    """True when OTel is installed and not explicitly disabled."""
    if os.environ.get("AEGRAIL_OTEL_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    return _AVAILABLE


@contextmanager
def agent_span(identity: str, *, model: str = ""):
    """The agent's own turn, as the parent of everything it does.

    Without this the tool spans have no common parent and a trace is a flat
    list of calls — technically correct and useless for answering "what was
    the agent doing when it tried that". `invoke_agent` is the conventions'
    answer, and nesting is free once the span exists.
    """
    tracer = _tracer()
    if tracer is None or not enabled():
        yield _NullSpan()
        return

    name = identity.split("/")[0] if identity else ""
    span_name = f"invoke_agent {name}" if name else "invoke_agent"
    kind = SpanKind.INTERNAL if SpanKind else None
    with tracer.start_as_current_span(span_name, kind=kind) as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        if name:
            span.set_attribute("gen_ai.agent.name", name)
        if model:
            span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("aegrail.identity", identity)
        yield _Span(span)


@contextmanager
def contract_span(
    name: str,
    *,
    tool: str = "",
    identity: str = "",
    attributes: Optional[dict] = None,
):
    """Wrap one contract evaluation in a span.

    Named `execute_tool {tool}` per the GenAI conventions, so the span lands
    in an existing dashboard as a tool call rather than as a vendor event
    nobody has a view for. The verdict and the audit digest ride along under
    `aegrail.*`, which is what makes the row worth looking at.
    """
    tracer = _tracer()
    if tracer is None or not enabled():
        yield _NullSpan()
        return

    span_name = f"execute_tool {tool}" if tool else f"aegrail.contract.{name}"
    kind = SpanKind.INTERNAL if SpanKind else None
    with tracer.start_as_current_span(span_name, kind=kind) as span:
        if tool:
            # Required by the conventions. Set first and unconditionally:
            # a tool span without it is not a tool span.
            span.set_attribute("gen_ai.tool.name", tool)
            span.set_attribute("gen_ai.operation.name", "execute_tool")
        span.set_attribute("aegrail.conventions", GENAI_CONVENTIONS)
        span.set_attribute("aegrail.tool", tool)
        span.set_attribute("aegrail.identity", identity)
        for key, value in (attributes or {}).items():
            _set(span, f"aegrail.{key}", value)
        wrapper = _Span(span)
        try:
            yield wrapper
        except Exception as exc:
            span.set_attribute("aegrail.verdict", "blocked")
            span.set_attribute("aegrail.blocked_by", type(exc).__name__)
            # Conditionally required by the conventions when a tool call
            # fails. A refusal is a failure of the call, not of the agent —
            # the verdict attribute is what tells them apart.
            span.set_attribute("error.type", type(exc).__qualname__)
            if Status and StatusCode:
                span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def _set(span, key: str, value: Any) -> None:
    if isinstance(value, (str, bool, int, float)):
        span.set_attribute(key, value)
    elif value is None:
        return
    else:
        span.set_attribute(key, str(value))


class _Span:
    def __init__(self, span):
        self._span = span

    def set(self, key: str, value: Any) -> None:
        _set(self._span, f"aegrail.{key}", value)

    def allow(self, **attrs: Any) -> None:
        self._span.set_attribute("aegrail.verdict", "allowed")
        for k, v in attrs.items():
            self.set(k, v)

    def set_raw(self, key: str, value: Any) -> None:
        """Set a conventions attribute, unprefixed."""
        _set(self._span, key, value)

    def context(self) -> tuple:
        """(trace_id, span_id) as 32- and 16-char hex, or ("", "").

        Lets the control-plane trace carry the same identifiers as the OTel
        span, so a customer who clicks through from their own dashboard
        lands on the matching enforcement record instead of guessing.
        """
        try:
            ctx = self._span.get_span_context()
            if not ctx or not ctx.trace_id:
                return "", ""
            return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")
        except Exception:
            return "", ""


class _NullSpan:
    def set(self, key: str, value: Any) -> None:  # noqa: D401
        return

    def set_raw(self, key: str, value: Any) -> None:
        return

    def allow(self, **attrs: Any) -> None:
        return

    def context(self) -> tuple:
        return "", ""
