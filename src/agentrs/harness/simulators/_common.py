"""Shared pieces of the local provider simulators.

Every simulator is a FastAPI app with in-memory state. They are written from our knowledge of each
provider's public API documentation and have NOT been checked against the live services: they exist
so connectors can be exercised end to end without real credentials. Every response carries
``X-Agentrs-Simulator: 1`` (and ``X-Agentrs-Simulator-Provider``) so a simulated response can never
be mistaken for a real one in logs or recordings.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone, UTC
from typing import Any, Callable, Optional, Union

import httpx

SIM_HEADER = "X-Agentrs-Simulator"

# A webhook target is either a URL (delivered with a real HTTP POST) or a callable
# ``(headers: dict[str, str], body: bytes) -> int | (int, bytes) | None`` for in-process tests.
WebhookTarget = Union[str, Callable[[dict, bytes], Any], None]


class SimClock:
    """Simulator clock: real time plus an offset (``advance``), or frozen at ``start``."""

    def __init__(self, start: Optional[float] = None, frozen: bool = False):
        self._base = time.time() if start is None else float(start)
        self._t0 = time.monotonic()
        self._offset = 0.0
        self.frozen = frozen

    def now(self) -> float:
        t = self._base + self._offset
        if not self.frozen:
            t += time.monotonic() - self._t0
        return t

    def advance(self, seconds: float) -> None:
        self._offset += float(seconds)


@dataclass
class RateState:
    allowed: bool
    limit: int
    remaining: int
    reset: float  # epoch seconds when the window resets
    used: int
    window: float


class RateLimiter:
    """Fixed-window counters keyed by (bucket, principal)."""

    def __init__(self, clock: SimClock):
        self.clock = clock
        self._w: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int, window: float, cost: int = 1) -> RateState:
        now = self.clock.now()
        with self._lock:
            start, used = self._w.get(key, [now, 0])
            if now >= start + window:
                start, used = now, 0
            allowed = used + cost <= limit
            if allowed:
                used += cost
            self._w[key] = [start, used]
        return RateState(allowed, int(limit), max(0, int(limit - used)), start + window, int(used), float(window))

    def peek(self, key: str, limit: int, window: float) -> RateState:
        now = self.clock.now()
        start, used = self._w.get(key, [now, 0])
        if now >= start + window:
            start, used = now, 0
        return RateState(used < limit, int(limit), max(0, int(limit - used)), start + window, int(used), float(window))


class SimulatorMarker:
    """Pure ASGI middleware adding the simulator marker headers to every HTTP response."""

    def __init__(self, app, provider: str):
        self.app = app
        self.provider = provider.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def _send(msg):
            if msg["type"] == "http.response.start":
                msg = dict(msg)
                msg["headers"] = list(msg.get("headers") or []) + [
                    (b"x-agentrs-simulator", b"1"), (b"x-agentrs-simulator-provider", self.provider)]
            await send(msg)

        await self.app(scope, receive, _send)


def iso_z(t: float, ms: bool = True) -> str:
    d = datetime.fromtimestamp(t, tz=UTC)
    if ms:
        return d.strftime("%Y-%m-%dT%H:%M:%S.") + f"{d.microsecond // 1000:03d}Z"
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def deliver_webhook(target: WebhookTarget, headers: dict[str, str], body: bytes, log: list) -> dict[str, Any]:
    """Deliver one webhook and record it in ``log``. Returns {status, body}."""
    rec: dict[str, Any] = {"headers": dict(headers), "body": body, "status": None, "response": b""}
    if target is None:
        rec["status"] = "no_target"
    elif callable(target):
        out = target(dict(headers), body)
        if isinstance(out, tuple):
            rec["status"], rec["response"] = out[0], (out[1] or b"")
        else:
            rec["status"] = 200 if out is None else int(out)
    else:
        with httpx.Client(trust_env=False, timeout=5.0) as c:  # never through an ambient proxy
            r = c.post(str(target), content=body, headers=headers)
            rec["status"], rec["response"] = r.status_code, r.content
    log.append(rec)
    return rec


def jdump(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=False).encode()


async def read_params(request) -> dict[str, Any]:
    """Merge query string, form-encoded and JSON bodies (providers accept several encodings)."""
    out: dict[str, Any] = dict(request.query_params)
    ctype = request.headers.get("content-type", "")
    body = await request.body()
    if not body:
        return out
    if "application/json" in ctype:
        try:
            data = json.loads(body)
        except ValueError:
            return out
        if isinstance(data, dict):
            out.update(data)
    elif "application/x-www-form-urlencoded" in ctype or "multipart/form-data" in ctype:
        form = await request.form()
        for k in form:
            vals = form.getlist(k)
            out[k] = vals if (k.endswith("[]") or len(vals) > 1) else vals[0]
    return out
