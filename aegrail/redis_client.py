"""A very small Redis client.

Aegrail has no runtime dependencies, and that is a feature: it installs
into someone else's agent process, so every package it drags in is a
version conflict waiting to happen in a codebase we do not control.
Durable pending-approval state needs six commands, so it speaks RESP
directly rather than pulling in redis-py.

Not a general Redis client. It does what pending.py needs and nothing
else — no pipelining, no pub/sub, no cluster support.
"""
from __future__ import annotations

import socket
import threading
from typing import Optional
from urllib.parse import urlparse

CRLF = b"\r\n"


class RedisError(Exception):
    pass


class RedisClient:
    """Synchronous RESP client with a lock around a single connection.

    One connection guarded by a mutex, rather than a pool: the call rate
    here is a handful of operations per approval, and a pool would be
    more moving parts than the problem deserves.
    """

    def __init__(self, url: str, timeout: float = 5.0):
        parsed = urlparse(url if "://" in url else f"redis://{url}")
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 6379
        self.password = parsed.password
        self.db = int((parsed.path or "/0").lstrip("/") or 0)
        self.timeout = timeout

        self._lock = threading.Lock()
        self._sock: Optional[socket.socket] = None
        self._buf = b""

    # ── connection ──────────────────────────────────────────────────
    def _connect(self) -> None:
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._sock.settimeout(self.timeout)
        self._buf = b""
        if self.password:
            self._command_locked("AUTH", self.password)
        if self.db:
            self._command_locked("SELECT", str(self.db))

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

    # ── protocol ────────────────────────────────────────────────────
    def _encode(self, *args: str) -> bytes:
        out = [b"*%d" % len(args), CRLF]
        for a in args:
            b = a.encode("utf-8") if isinstance(a, str) else a
            out += [b"$%d" % len(b), CRLF, b, CRLF]
        return b"".join(out)

    def _read_line(self) -> bytes:
        while CRLF not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise RedisError("connection closed by server")
            self._buf += chunk
        line, self._buf = self._buf.split(CRLF, 1)
        return line

    def _read_exact(self, n: int) -> bytes:
        while len(self._buf) < n + 2:  # payload plus trailing CRLF
            chunk = self._sock.recv(4096)
            if not chunk:
                raise RedisError("connection closed by server")
            self._buf += chunk
        data, self._buf = self._buf[:n], self._buf[n + 2 :]
        return data

    def _read_reply(self):
        line = self._read_line()
        kind, rest = line[:1], line[1:]

        if kind == b"+":
            return rest.decode()
        if kind == b"-":
            raise RedisError(rest.decode())
        if kind == b":":
            return int(rest)
        if kind == b"$":
            length = int(rest)
            if length == -1:
                return None
            return self._read_exact(length).decode()
        if kind == b"*":
            count = int(rest)
            if count == -1:
                return None
            return [self._read_reply() for _ in range(count)]
        raise RedisError(f"unexpected reply type: {line!r}")

    def _command_locked(self, *args: str):
        self._sock.sendall(self._encode(*args))
        return self._read_reply()

    def command(self, *args: str):
        """Run one command, reconnecting once if the socket went away.

        Long-lived agents sit idle between approvals, easily longer than
        a Redis idle timeout, so the first command after a quiet period
        routinely lands on a dead socket. One transparent retry turns
        that from an error into a reconnect.
        """
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._sock is None:
                        self._connect()
                    return self._command_locked(*args)
                except (OSError, RedisError) as exc:
                    if self._sock is not None:
                        try:
                            self._sock.close()
                        except OSError:
                            pass
                        self._sock = None
                    if attempt == 2:
                        raise RedisError(str(exc)) from exc
            return None

    # ── the six operations pending.py needs ─────────────────────────
    def ping(self) -> bool:
        try:
            return self.command("PING") == "PONG"
        except RedisError:
            return False

    def set(self, key: str, value: str, ttl_seconds: Optional[int] = None) -> None:
        if ttl_seconds:
            self.command("SET", key, value, "EX", str(int(ttl_seconds)))
        else:
            self.command("SET", key, value)

    def get(self, key: str) -> Optional[str]:
        return self.command("GET", key)

    def delete(self, key: str) -> int:
        return int(self.command("DEL", key) or 0)

    def scan(self, match: str, count: int = 100) -> list:
        """Cursor-based key scan.

        SCAN rather than KEYS: KEYS blocks the whole server while it walks
        the keyspace, which is a poor thing for an SDK to do to a database
        it shares with the control plane.
        """
        cursor, keys = "0", []
        while True:
            cursor, batch = self.command("SCAN", cursor, "MATCH", match, "COUNT", str(count))
            keys.extend(batch or [])
            if cursor == "0":
                break
        return keys
