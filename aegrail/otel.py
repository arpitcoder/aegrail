"""OpenTelemetry emission for contract evaluations.

Every contract decision becomes a span. That makes the enforcement layer
readable by anything that already speaks OTel — a customer's existing
collector, Grafana, Datadog, an OpenSearch-backed control plane — with no
MavOps-specific integration.

OTel is an optional dependency. Without it installed, every function here
degrades to a no-op so the agent still runs.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Optional

_TRACER = None
_AVAILABLE = False

try:  # pragma: no cover - depends on optional extra
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import Status, StatusCode

    _AVAILABLE = True
except Exception:  # pragma: no cover
    _otel_trace = None
    Status = StatusCode = None


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
def contract_span(
    name: str,
    *,
    tool: str = "",
    identity: str = "",
    attributes: Optional[dict] = None,
):
    """Wrap one contract evaluation in a span.

    Span name is `aegrail.contract.<name>`. Attributes carry the verdict,
    the budget state, and the audit chain digest so a trace row and an
    audit row can be correlated after the fact.
    """
    tracer = _tracer()
    if tracer is None or not enabled():
        yield _NullSpan()
        return

    with tracer.start_as_current_span(f"aegrail.contract.{name}") as span:
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


class _NullSpan:
    def set(self, key: str, value: Any) -> None:  # noqa: D401
        return

    def allow(self, **attrs: Any) -> None:
        return
