"""Traces are best-effort, but 'best-effort' must not mean 'silently lost'.

The failure these cover is the one that actually happened: a short-lived agent
did its work, the process ended, and every queued trace was discarded without a
word. From the dashboard that is indistinguishable from an idle agent.
"""
import threading
import time
import warnings

import pytest

from aegrail.traces import TraceClient, _looks_like_uuid


class _Cred:
    token = "t"


class _Creds:
    def get(self):
        return _Cred()


def _client(sink, **kw):
    c = TraceClient(credentials=_Creds(), **kw)
    c._post = lambda body: sink.append(body)   # don't touch the network
    return c


def test_stop_flushes_the_backlog():
    sink = []
    c = _client(sink).start()
    for i in range(25):
        c.record(operation="tool_call", tool_names=[f"t{i}"])
    c.stop(timeout=5.0)
    assert len(sink) == 25, f"stop() dropped {25 - len(sink)} queued traces"


def test_stop_is_idempotent():
    sink = []
    c = _client(sink).start()
    c.record(operation="tool_call", tool_names=["a"])
    c.stop()
    c.stop()          # atexit will call it again after an explicit stop
    assert len(sink) == 1


def test_stop_is_bounded_when_posting_hangs():
    # A wedged control plane must not hold the process open indefinitely.
    sink = []
    c = _client(sink).start()
    c._post = lambda body: time.sleep(30)
    c.record(operation="tool_call", tool_names=["slow"])
    began = time.monotonic()
    c.stop(timeout=1.0)
    assert time.monotonic() - began < 5.0, "stop() did not honour its timeout"


def test_disabled_client_records_nothing(monkeypatch):
    monkeypatch.setenv("AEGRAIL_TRACES_DISABLED", "1")
    sink = []
    c = _client(sink).start()
    c.record(operation="tool_call", tool_names=["a"])
    c.stop()
    assert sink == []


def test_queue_is_bounded_and_drops_rather_than_blocking():
    # Never block a tool call to record telemetry about it.
    c = TraceClient(credentials=_Creds(), max_queue=4)
    for i in range(50):
        c.record(operation="tool_call", tool_names=[f"t{i}"])   # no worker running
    assert c._q.qsize() <= 4


def test_warns_when_agent_id_is_not_a_uuid(monkeypatch):
    monkeypatch.setenv("MAVOPS_AGENT_ID", "agent-abc-123")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TraceClient(credentials=_Creds())
    assert any("not a UUID" in str(w.message) for w in caught)


def test_no_warning_for_a_real_uuid(monkeypatch):
    monkeypatch.setenv("MAVOPS_AGENT_ID", "3f2504e0-4f89-11d3-9a0c-0305e82c3301")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TraceClient(credentials=_Creds())
    assert not caught


@pytest.mark.parametrize("value,ok", [
    ("3f2504e0-4f89-11d3-9a0c-0305e82c3301", True),
    ("3F2504E0-4F89-11D3-9A0C-0305E82C3301", True),
    ("agent-abc-123", False),
    ("", False),
])
def test_uuid_shape(value, ok):
    assert _looks_like_uuid(value) is ok
