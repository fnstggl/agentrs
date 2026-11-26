"""The real-world bridge: the single object through which an authorised experiment may send HTTP requests to
user-owned staging services. It refuses by default and fails closed at every step.

Order of checks for every request (and again for every redirect hop):

 1. an authorisation document is loaded, has not expired, and the wall-time budget is not used up;
 2. the method is allowed by the document;
 3. the URL is http(s), carries no credentials (userinfo or credential-like query parameters), no fragment tricks,
    no encoded path separators or dot segments;
 4. scheme, host, port and path prefix match one of the document's base URLs;
 5. the host is resolved; *every* address it resolves to must be public, or be listed explicitly in
    `allow_private_addresses` (loopback, private, link-local, metadata, multicast, reserved and unspecified
    addresses are refused otherwise); the connection is then made to exactly the address that was checked, so a
    second DNS answer cannot redirect it;
 6. request-count, rate and byte budgets are not exceeded;
 7. a consequential method (anything but GET/HEAD) is approved by the live approval callback, when the document
    requires approval; without a callback it is refused.

Every request, response, refusal, approval, measurement and holdout evaluation is appended to the hash-chained
log before control returns to the caller.
"""
from __future__ import annotations

import base64
import hashlib
import http.client
import ipaddress
import os
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import parse_qsl, unquote, urljoin, urlsplit

from .authorization import Authorization, BaseURL, default_port, load_authorization
from .log import ChainedLog
from .results import RealWorldResult, SealedHoldoutGuard, _mint

SAFE_METHODS = {"GET", "HEAD"}
CRED_PARAMS = {"token", "access_token", "api_key", "apikey", "key", "password", "passwd", "secret", "auth", "signature",
               "sig", "session", "sessionid"}
SECRET_HEADERS = {"authorization", "cookie", "proxy-authorization", "x-api-key", "x-auth-token"}
METADATA = {"169.254.169.254", "fd00:ec2::254", "100.100.100.200"}
MAX_REDIRECTS = 5
# headers that say who is asking: the bridge sets the User-Agent itself (automation disclosure), and refuses callers
# that try to look like a browser, a person or another origin (impersonation, Part 16 / SP-16b)
IDENTITY_HEADERS = {"user-agent", "from", "x-forwarded-for", "forwarded", "x-real-ip", "via", "x-originating-ip",
                    "client-ip", "true-client-ip", "x-client-ip", "referer", "origin"}


def disclosure(experiment: str) -> str:
    """The User-Agent of every real-world request: automated, research, which experiment."""
    safe = "".join(c for c in str(experiment) if c.isalnum() or c in "-_.")[:64]
    return f"crucible-realworld-bridge/1 (automated research request; experiment={safe}; not a person)"
BODY_LOG_CAP = 262_144

_FORBIDDEN_HERE: list[str] = []


class BridgeRefused(PermissionError):
    """The bridge declined to act. Always logged (when a log exists) before it is raised."""


def forbid_in_this_process(reason: str = "this process hosts synthetic worlds") -> None:
    """One-way latch for processes that run agents inside worlds: after this call no bridge can be created here."""
    _FORBIDDEN_HERE.append(reason)


@dataclass(frozen=True)
class RealResponse:
    status: int
    headers: dict
    body: bytes
    url: str                 # final URL after allowed redirects
    hops: int


class _PinnedHTTP(http.client.HTTPConnection):
    """Connects to the address that was resolved and checked, whatever the name resolves to later."""

    def __init__(self, host: str, port: int, pinned: str, timeout: float):
        super().__init__(host, port, timeout=timeout)
        self._pinned = pinned

    def connect(self):
        self.sock = socket.create_connection((self._pinned, self.port), self.timeout)


class _PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host: str, port: int, pinned: str, timeout: float):
        super().__init__(host, port, timeout=timeout, context=ssl.create_default_context())
        self._pinned = pinned

    def connect(self):
        raw = socket.create_connection((self._pinned, self.port), self.timeout)
        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)   # certificate is checked for the name


def _is_internal(ip: str) -> bool:
    a = ipaddress.ip_address(ip)
    if getattr(a, "ipv4_mapped", None) is not None:
        a = a.ipv4_mapped
    return (str(a) in METADATA or a.is_private or a.is_loopback or a.is_link_local or a.is_multicast or a.is_reserved
            or a.is_unspecified or not a.is_global)


class RealWorldBridge:
    def __init__(self, authorization: Authorization | None = None, *, log_path: str | None = None,
                 approve: Callable[[dict], bool] | None = None, clock: Callable[[], float] = time.time,
                 resolver: Callable[..., list] | None = None, timeout_s: float = 10.0):
        if _FORBIDDEN_HERE or os.environ.get("CRUCIBLE_HOSTS_WORLDS"):
            why = _FORBIDDEN_HERE[0] if _FORBIDDEN_HERE else f"it hosts synthetic worlds ({os.environ['CRUCIBLE_HOSTS_WORLDS']})"
            raise BridgeRefused(f"real-world access is forbidden in this process: {why}")
        if authorization is not None and not isinstance(authorization, Authorization):
            raise BridgeRefused("authorisation must come from crucible.realworld.load_authorization(<operator file>)")
        if authorization is not None and not log_path:
            raise BridgeRefused("an authorised bridge needs a log path: nothing is sent unrecorded")
        self.auth = authorization
        self.approve = approve
        self.clock = clock
        self.resolver = resolver or socket.getaddrinfo
        self.timeout_s = float(timeout_s)
        self.log = ChainedLog(log_path) if log_path else None
        self.n_requests = 0
        self.n_bytes = 0
        self.t_open = clock()
        self._last_request_t: float | None = None
        self._tokens = 1.0
        self.refusals: list[str] = []
        self._holdouts = (SealedHoldoutGuard(dict(authorization.holdouts), log=self.log)
                          if authorization is not None else None)
        if self.log is not None and authorization is not None:
            # budgets survive a restart: they are recomputed from what the log already holds
            from .log import read_log
            earlier = read_log(self.log.path)
            self.n_requests = sum(1 for b in earlier if b.get("kind") == "request")
            self.n_bytes = sum(int(b.get("body_bytes", 0)) for b in earlier if b.get("kind") in ("request", "response"))
            opened = [b["t"] for b in earlier if b.get("kind") == "open" and
                      b.get("authorization", {}).get("sha256") == authorization.sha256]
            if opened:
                self.t_open = min(self.t_open, float(opened[0]))     # wall time runs from the first use of this document
            self._record({"kind": "open", "authorization": authorization.summary(), "pid": os.getpid(),
                          "requests_already_logged": self.n_requests})

    @classmethod
    def from_file(cls, path: str, *, log_path: str, approve: Callable[[dict], bool] | None = None, **kw) -> "RealWorldBridge":
        clock = kw.get("clock", time.time)
        return cls(load_authorization(path, now=clock()), log_path=log_path, approve=approve, **kw)

    # ------------------------------------------------------------------ plumbing
    def _record(self, body: dict) -> str | None:
        body = dict(body, t=round(self.clock(), 3))
        return self.log.append(body) if self.log is not None else None

    def _refuse(self, why: str, **ctx) -> None:
        self.refusals.append(why)
        if self.log is not None:
            self._record({"kind": "refusal", "why": why, **ctx})
        raise BridgeRefused(why)

    @property
    def enabled(self) -> bool:
        return self.auth is not None and self.clock() < self.auth.expires

    # ------------------------------------------------------------------ checks
    def _match_base(self, url: str, method: str) -> tuple[BaseURL, str, str, int, str]:
        """Steps 3 and 4. Returns (base, scheme, host, port, request-target)."""
        if any(c in url for c in "\\ \t\r\n"):
            self._refuse("malformed URL (whitespace or backslash)", url=url[:300], method=method)
        s = urlsplit(url)
        if s.scheme not in ("http", "https"):
            self._refuse(f"scheme {s.scheme!r} is not allowed", url=url[:300], method=method)
        if s.username is not None or s.password is not None or "@" in s.netloc:
            self._refuse("credentials in URLs are refused", url=f"{s.scheme}://<redacted>{s.path}"[:300], method=method)
        if not s.hostname:
            self._refuse("URL has no host", url=url[:300], method=method)
        for k, _ in parse_qsl(s.query, keep_blank_values=True):
            if k.lower() in CRED_PARAMS:
                self._refuse(f"credential-like query parameter {k!r} is refused (send secrets in headers)",
                             url=f"{s.scheme}://{s.hostname}{s.path}?<redacted>"[:300], method=method)
        try:
            port = s.port or default_port(s.scheme)
        except ValueError:
            self._refuse("bad port", url=url[:300], method=method)
        host = s.hostname.lower().rstrip(".")
        path = s.path or "/"
        low = path.lower()
        if "%2f" in low or "%5c" in low or "%2e" in low or ".." in unquote(path).split("/") or "//" in path:
            self._refuse("path contains encoded separators, dot segments or empty segments", url=url[:300], method=method)
        for b in self.auth.base_urls:
            if b.scheme == s.scheme and b.host == host and b.port == port and (path.startswith(b.path) or path + "/" == b.path):
                target = path + ("?" + s.query if s.query else "")
                return b, s.scheme, host, int(port), target
        self._refuse(f"{s.scheme}://{host}:{port}{path} is not under any authorised base URL", url=url[:300], method=method)
        raise AssertionError("unreachable")

    def _resolve_checked(self, host: str, port: int, url: str, method: str) -> str:
        """Step 5. Returns the single address the connection will be pinned to."""
        try:
            infos = self.resolver(host, port, type=socket.SOCK_STREAM)
        except Exception as e:  # noqa: BLE001 - resolution failures are refusals, not crashes
            self._refuse(f"cannot resolve {host}: {type(e).__name__}", url=url[:300], method=method)
        addrs = sorted({str(i[4][0]) for i in infos})
        if not addrs:
            self._refuse(f"{host} resolves to nothing", url=url[:300], method=method)
        for ip in addrs:
            if _is_internal(ip):
                allow = self.auth.allow_private_addresses
                if ip not in allow and f"{ip}:{port}" not in allow and f"[{ip}]:{port}" not in allow:
                    self._refuse(f"{host} resolves to {ip}, a private / loopback / link-local / metadata address that "
                                 "the authorisation does not list explicitly", url=url[:300], method=method, addresses=addrs)
        return addrs[0]

    def _budget(self, method: str, url: str, nbytes: int, check_rate: bool = True) -> None:
        now = self.clock()
        a = self.auth
        if now >= a.expires:
            self._refuse("the authorisation has expired", url=url[:300], method=method)
        if now - self.t_open > a.max_wall_s:
            self._refuse(f"wall-time budget of {a.max_wall_s:g} s is used up", url=url[:300], method=method)
        if self.n_requests >= a.max_requests:
            self._refuse(f"request budget of {a.max_requests} is used up", url=url[:300], method=method)
        if self.n_bytes + nbytes > a.max_bytes:
            self._refuse(f"byte budget of {a.max_bytes} is used up", url=url[:300], method=method)
        if check_rate and self._last_request_t is not None:   # token bucket of depth 1: no bursts
            self._tokens = min(1.0, self._tokens + (now - self._last_request_t) * a.max_rate_per_s)
            if self._tokens < 1.0:
                self._refuse(f"rate limit of {a.max_rate_per_s:g} requests per second", url=url[:300], method=method)

    def _approval(self, method: str, url: str, body: bytes | None, note: str) -> None:
        if method in SAFE_METHODS:
            return
        desc = {"method": method, "url": url, "body_bytes": len(body or b""),
                "body_sha256": hashlib.sha256(body or b"").hexdigest(), "note": note[:500],
                "experiment": self.auth.experiment}
        if not self.auth.approval_required:
            self._record({"kind": "approval", "decision": "waived by the authorisation document", **desc})
            return
        if self.approve is None:
            self._refuse("a consequential action needs live approval and no approval callback is attached", **desc)
        try:
            ok = self.approve(dict(desc)) is True
        except Exception as e:  # noqa: BLE001 - a failing approver is a refusal
            self._refuse(f"the approval callback failed ({type(e).__name__})", **desc)
        if not ok:
            self._refuse("the approver declined", **desc)
        self._record({"kind": "approval", "decision": "approved", **desc})

    # ------------------------------------------------------------------ the one way out
    def request(self, method: str, url: str, *, body: bytes | None = None, headers: dict | None = None,
                note: str = "") -> RealResponse:
        """Send one HTTP request under the authorisation. Raises `BridgeRefused` when any check fails."""
        method = str(method).upper()
        if self.auth is None:
            self._refuse("no authorisation document: the real-world bridge is off", url=str(url)[:300], method=method)
        if body is not None and not isinstance(body, (bytes, bytearray)):
            self._refuse("body must be bytes", url=str(url)[:300], method=method)
        headers = {str(k): str(v) for k, v in (headers or {}).items()}
        if any(k.lower() == "host" for k in headers):
            self._refuse("a caller-supplied Host header is refused", url=str(url)[:300], method=method)
        spoof = sorted(k for k in headers if k.lower() in IDENTITY_HEADERS)
        if spoof:
            # impersonation (Part 16): the bridge always identifies itself as automated research; a caller may not
            # pose as a browser or a person, or claim another origin
            self._refuse(f"identity headers are set by the bridge, not the caller: {spoof}", url=str(url)[:300],
                         method=method)
        headers["User-Agent"] = disclosure(self.auth.experiment)
        if method not in self.auth.methods:
            self._refuse(f"method {method} is not authorised", url=str(url)[:300], method=method)
        hops = 0
        while True:
            _, scheme, host, port, target = self._match_base(url, method)
            self._budget(method, url, len(body or b""), check_rate=(hops == 0))   # redirect hops count, unpaced (<= 5)
            ip = self._resolve_checked(host, port, url, method)
            if hops == 0:
                self._approval(method, url, body, note)
            status, rh, rb, truncated = self._send(method, scheme, host, port, ip, target, body, headers, url, hops)
            if truncated:
                self._refuse(f"byte budget of {self.auth.max_bytes} exceeded while reading the response", url=url[:300],
                             method=method)
            if status in (301, 302, 303, 307, 308) and "location" in rh:
                nxt = urljoin(url, rh["location"])
                if method not in SAFE_METHODS:
                    return RealResponse(status, rh, rb, url, hops)     # never re-send a consequential action elsewhere
                hops += 1
                if hops > MAX_REDIRECTS:
                    self._refuse("too many redirects", url=url[:300], method=method)
                url = nxt                                               # checked from step 3 again on the next turn
                continue
            return RealResponse(status, rh, rb, url, hops)

    def _send(self, method, scheme, host, port, ip, target, body, headers, url, hops):
        a = self.auth
        self.n_requests += 1
        self.n_bytes += len(body or b"")
        self._last_request_t = self.clock()
        self._tokens = max(0.0, self._tokens - 1.0)
        logged_headers = {k: ("sha256:" + hashlib.sha256(v.encode()).hexdigest()[:16] if k.lower() in SECRET_HEADERS else v)
                          for k, v in sorted(headers.items())}
        self._record({"kind": "request", "n": self.n_requests, "method": method, "url": url, "pinned_address": ip,
                      "port": port, "hop": hops, "headers": logged_headers, "body_bytes": len(body or b""),
                      "body_sha256": hashlib.sha256(body or b"").hexdigest(),
                      "body_b64": base64.b64encode((body or b"")[:BODY_LOG_CAP]).decode()})
        cls = _PinnedHTTPS if scheme == "https" else _PinnedHTTP
        conn = cls(host, port, ip, self.timeout_s)
        truncated = False
        try:
            conn.request(method, target, body=body, headers=headers)
            resp = conn.getresponse()
            left = a.max_bytes - self.n_bytes
            data = resp.read(left + 1)
            if len(data) > left:
                data, truncated = data[:left], True
            status = resp.status
            rh = {k.lower(): v for k, v in resp.getheaders()}
        except (OSError, http.client.HTTPException) as e:
            self._record({"kind": "response", "n": self.n_requests, "error": f"{type(e).__name__}: {e}"[:300]})
            raise BridgeRefused(f"request failed: {type(e).__name__}: {e}") from e
        finally:
            conn.close()
        self.n_bytes += len(data)
        self._record({"kind": "response", "n": self.n_requests, "status": status, "headers": dict(sorted(rh.items())),
                      "body_bytes": len(data), "body_sha256": hashlib.sha256(data).hexdigest(),
                      "body_b64": base64.b64encode(data[:BODY_LOG_CAP]).decode(), "truncated": truncated})
        return status, rh, data, truncated

    def get(self, url: str, **kw) -> RealResponse:
        return self.request("GET", url, **kw)

    # ------------------------------------------------------------------ outcomes
    def reader(self) -> "_Reader":
        """A handle that can only observe (GET/HEAD) - what measurement functions receive."""
        return _Reader(self)

    def measure(self, name: str, fn: Callable[["_Reader"], Any], holdout: str | None = None) -> RealWorldResult:
        """Compute one outcome with the experimenter's own measurement function. This is the only source of
        results: `fn` observes the real system through a read-only handle; nothing the agent says is an input."""
        if self.auth is None:
            self._refuse("no authorisation document: the real-world bridge is off")
        if not callable(fn):
            self._refuse("a measurement needs a function supplied by the experimenter")
        if holdout is not None:
            try:
                self._holdouts.charge(holdout, purpose=name)
            except PermissionError as e:
                self.refusals.append(str(e))
                raise
        value = fn(self.reader())
        try:
            import json
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            self._refuse(f"measurement {name!r} did not return plain JSON data")
        head = self._record({"kind": "measurement", "name": str(name)[:200], "value": value, "holdout": holdout,
                             "function": getattr(fn, "__qualname__", repr(fn))[:200]})
        return _mint(experiment=self.auth.experiment, measure=str(name), value=value, holdout=holdout,
                     authorization_sha256=self.auth.sha256, log_path=self.log.path, log_head=head,
                     t=round(self.clock(), 3))

    def evaluate_holdout(self, holdout: str, name: str, fn: Callable[["_Reader"], Any]) -> RealWorldResult:
        """A measurement that consumes one evaluation of a sealed holdout's budget (refused beyond it)."""
        return self.measure(name, fn, holdout=holdout)

    def holdout_remaining(self, holdout: str) -> int:
        return self._holdouts.remaining(holdout) if self._holdouts is not None else 0

    def close(self) -> None:
        if self.log is not None:
            if self.auth is not None:
                self._record({"kind": "close", "requests": self.n_requests, "bytes": self.n_bytes,
                              "refusals": len(self.refusals)})
            self.log.close()
            self.log = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _Reader:
    def __init__(self, bridge: RealWorldBridge):
        self._b = bridge

    def get(self, url: str, headers: dict | None = None) -> RealResponse:
        return self._b.request("GET", url, headers=headers, note="measurement")

    def head(self, url: str, headers: dict | None = None) -> RealResponse:
        return self._b.request("HEAD", url, headers=headers, note="measurement")


__all__ = ["RealWorldBridge", "RealResponse", "BridgeRefused", "forbid_in_this_process"]
