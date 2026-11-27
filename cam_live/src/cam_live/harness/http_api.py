"""cam_live.harness.http_api — the ``api`` execution modality: authorized HTTP requests to documented APIs.

Owned by workstream ES.

* Every request (every redirect hop and every retry too) passes ``ExecContext.check`` with its method and URL.
* Credentials are referenced by *name*: ``arguments["auth"]["secret"]`` names a secret, the value is fetched from
  ``ExecContext.secrets`` at the moment of use, is never written into a ``PrimitiveOp`` or an ``Observation`` and
  is scrubbed from the response digest. Literal credential headers in the arguments are rejected.
* Responses become ``Observation``s: a bounded untrusted text digest plus exact ``Quantity`` objects for the status,
  the measured latency, the rate-limit headers and the numeric fields of a JSON body.
* Rate-limit headers are honoured (``Retry-After``, ``X-RateLimit-*``, ``RateLimit-*``): an exhausted window is
  waited for when the wait is within bounds and is otherwise reported as ``RESOURCE_LIMIT``; it is never bypassed.

Also exported for the other ES executors: :func:`schema_errors` (JSON-Schema subset), :func:`resolve_secret`,
:func:`scrub`, :func:`structure_hash`, :class:`RateLimiter`, and :class:`ApiModality`, which presents several
``api``-modality executors (generic HTTP, publishing, Discord) as one ``ModalityExecutor``.
"""
from __future__ import annotations

import base64
import contextlib
import email.utils
import hashlib
import inspect
import json
import re
import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx

from ..contracts import (ActionResult, ActionSpec, Cost, ExecContext, FailureClass, Level, Modality, Observation, OutcomeKind,
                         PrimitiveOp, Provenance, Quantity, ResultStatus, ToolDescriptor, Trust, new_id)
from .failures import (FailureEvidence, RetryPolicy, authorization_result, classify, failed_result,
                       run_with_retries, wait)

# --------------------------------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------------------------------
_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list,
          "null": type(None)}


def schema_errors(schema: Mapping[str, Any], value: Any, path: str = "arguments") -> List[str]:
    """Validate ``value`` against the JSON-Schema subset used by tool descriptors and skills
    (type, enum, const, required, properties, additionalProperties, items, minimum, maximum, minLength, maxLength,
    pattern, minItems, maxItems). Returns human-readable errors; ``[]`` means valid."""
    errs: List[str] = []
    if not schema:
        return errs
    t = schema.get("type")
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        ok = False
        for name in ts:
            py = _TYPES.get(name)
            if py is None:
                continue
            if isinstance(value, bool) and name in ("integer", "number"):
                continue
            if name == "number" and isinstance(value, float) and value != value:
                continue
            if isinstance(value, py):
                ok = True
        if not ok:
            return [f"{path}: expected {'|'.join(ts)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} is not one of {list(schema['enum'])}")
    if "const" in schema and value != schema["const"]:
        errs.append(f"{path}: must equal {schema['const']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errs.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errs.append(f"{path}: does not match {schema['pattern']!r}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if isinstance(schema.get("items"), Mapping):
            for i, v in enumerate(value):
                errs.extend(schema_errors(schema["items"], v, f"{path}[{i}]"))
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}: missing required '{k}'")
        extra = schema.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                errs.extend(schema_errors(props[k], v, f"{path}.{k}"))
            elif extra is False:
                errs.append(f"{path}: unexpected '{k}'")
            elif isinstance(extra, Mapping):
                errs.extend(schema_errors(extra, v, f"{path}.{k}"))
    return errs


def granted_action(ctx: Any, action: ActionSpec) -> ActionSpec:
    """The top-level action whose grant covers what is being executed (a skill sets ``ctx.granted_action`` while
    it runs its steps; otherwise it is the action itself)."""
    return getattr(ctx, "granted_action", None) or action


@contextlib.contextmanager
def acting_under(ctx: Any, action: ActionSpec):
    """While an executor runs sub-steps on behalf of ``action`` (a skill run, a publication through another
    adapter), the executors of those steps ask the secret broker in the name of this granted action."""
    had = getattr(ctx, "granted_action", None)
    if had is None:
        ctx.granted_action = action
    try:
        yield ctx
    finally:
        if had is None:
            ctx.granted_action = None


def grant_bound(secrets: Any) -> bool:
    """True for a broker handle whose ``get`` needs the granted action, its decision and a destination."""
    get = None if isinstance(secrets, Mapping) else (getattr(secrets, "get", None) or (secrets if callable(secrets) else None))
    if get is None:
        return False
    try:
        required = [p for p in inspect.signature(get).parameters.values()
                    if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    except (TypeError, ValueError):
        return False
    return len(required) >= 4


def resolve_secret(secrets: Any, name: str, *, action: Optional[ActionSpec] = None, decision: Any = None,
                   destination: str = "") -> Optional[str]:
    """Fetch a secret value by name from the executor-side broker handle at the moment of use.

    Two handle shapes are supported: a grant-bound broker ``get(name, action, decision, destination)`` (the
    authorization service releases the secret only for the granted action that is executing and only for a
    destination the secret is bound to), and a plain ``get(name)`` / mapping / callable. Returns ``None`` when
    there is no broker, no such secret, or the broker refuses. The value must only be used to build the request."""
    if secrets is None or not name:
        return None
    try:
        if isinstance(secrets, Mapping):
            v = secrets.get(name)
        else:
            get = getattr(secrets, "get", None) or (secrets if callable(secrets) else None)
            if get is None:
                return None
            v = get(name, action, decision, destination) if grant_bound(secrets) else get(name)
    except (KeyError, LookupError, PermissionError):
        return None
    return str(v) if v else None


def scrub(value: Any, secret_values: Iterable[str]) -> Any:
    """Replace every occurrence of a known secret value inside strings of a JSON-like structure."""
    vals = [v for v in secret_values if v and len(v) >= 4]
    if not vals:
        return value
    if isinstance(value, str):
        for v in vals:
            value = value.replace(v, "[secret]")
            b = base64.b64encode(v.encode()).decode()
            if len(b) >= 8:
                value = value.replace(b, "[secret]")
        return value
    if isinstance(value, dict):
        return {scrub(k, vals): scrub(v, vals) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v, vals) for v in value]
    return value


def structure_hash(data: Any, depth: int = 6) -> str:
    """Hash of the *shape* of a JSON-like value (keys and value types, not values). Two responses or pages with the
    same fields have the same hash; a renamed or removed field changes it."""
    def shape(v: Any, d: int) -> Any:
        if d <= 0:
            return "…"
        if isinstance(v, dict):
            return {str(k): shape(x, d - 1) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
        if isinstance(v, list):
            kinds = sorted({json.dumps(shape(x, d - 1), sort_keys=True) for x in v[:32]})
            return ["list", kinds]
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, (int, float)):
            return "num"
        if v is None:
            return "null"
        return "str"
    return hashlib.sha256(json.dumps(shape(data, depth), sort_keys=True).encode()).hexdigest()[:16]


def host_hash(url_or_host: str) -> str:
    host = urlsplit(url_or_host).netloc if "://" in url_or_host else url_or_host
    return hashlib.sha256(host.lower().encode()).hexdigest()[:12]


# --------------------------------------------------------------------------------------------------------------
# Rate limits
# --------------------------------------------------------------------------------------------------------------
@dataclass
class RateLimitInfo:
    limit: Optional[float] = None
    remaining: Optional[float] = None
    reset_after_s: Optional[float] = None
    retry_after_s: Optional[float] = None
    bucket: str = ""
    is_global: bool = False
    scope: str = ""

    def quantities(self, prefix: str) -> List[Quantity]:
        out = []
        for name, v, unit in (("limit", self.limit, ""), ("remaining", self.remaining, ""),
                              ("reset_after_s", self.reset_after_s, "s"), ("retry_after_s", self.retry_after_s, "s")):
            if v is not None:
                out.append(Quantity(f"{prefix}.ratelimit.{name}", float(v), unit=unit))
        return out


def _num(v: Any) -> Optional[float]:
    try:
        return float(str(v).strip().split(",")[0].split(";")[0])
    except (TypeError, ValueError):
        return None


def parse_rate_limit(headers: Mapping[str, str], now: float, body: Any = None) -> RateLimitInfo:
    """Read the rate-limit state a server reported: ``Retry-After`` (seconds or HTTP date), ``X-RateLimit-*`` and
    the IETF ``RateLimit-*`` headers, and a JSON body's ``retry_after`` / ``global`` fields (Discord's 429 body)."""
    h = {str(k).lower(): str(v) for k, v in headers.items()}
    info = RateLimitInfo()
    for key in ("x-ratelimit-limit", "ratelimit-limit"):
        if key in h and info.limit is None:
            info.limit = _num(h[key])
    for key in ("x-ratelimit-remaining", "ratelimit-remaining"):
        if key in h and info.remaining is None:
            info.remaining = _num(h[key])
    if "x-ratelimit-reset-after" in h:
        info.reset_after_s = _num(h["x-ratelimit-reset-after"])
    elif "ratelimit-reset" in h:
        info.reset_after_s = _num(h["ratelimit-reset"])
    elif "x-ratelimit-reset" in h:
        v = _num(h["x-ratelimit-reset"])
        if v is not None:
            info.reset_after_s = max(0.0, v - now) if v > 1e9 else v     # epoch seconds vs. delta seconds
    if "retry-after" in h:
        v = _num(h["retry-after"])
        if v is None:
            try:
                v = max(0.0, email.utils.parsedate_to_datetime(h["retry-after"]).timestamp() - now)
            except (TypeError, ValueError):
                v = None
        info.retry_after_s = v
    info.bucket = h.get("x-ratelimit-bucket", "")
    info.scope = h.get("x-ratelimit-scope", "")
    info.is_global = h.get("x-ratelimit-global", "").lower() == "true" or info.scope == "global"
    if isinstance(body, dict):
        v = _num(body.get("retry_after"))
        if v is not None:
            info.retry_after_s = v if info.retry_after_s is None else max(v, info.retry_after_s)
        if body.get("global") is True:
            info.is_global = True
    return info


class RateLimiter:
    """Remembers, per key (a host or a rate-limit bucket), until when the other side asked us not to send."""

    def __init__(self) -> None:
        self._until: Dict[str, float] = {}
        self._route_bucket: Dict[str, str] = {}

    def wait_needed(self, key: str, now: float) -> float:
        keys = ["*", key]
        if key in self._route_bucket:
            keys.append(self._route_bucket[key])
        return max(0.0, max(self._until.get(k, 0.0) for k in keys) - now)

    def update(self, key: str, info: RateLimitInfo, status: int, now: float) -> None:
        if info.bucket:
            self._route_bucket[key] = info.bucket
        target = info.bucket or key
        if status == 429:
            delay = info.retry_after_s if info.retry_after_s is not None else (info.reset_after_s or 1.0)
            self._until["*" if info.is_global else target] = now + delay
        elif info.remaining is not None and info.remaining <= 0 and info.reset_after_s is not None:
            self._until[target] = now + info.reset_after_s
        elif status == 503 and info.retry_after_s is not None:
            self._until[target] = now + info.retry_after_s


# --------------------------------------------------------------------------------------------------------------
# Response digests
# --------------------------------------------------------------------------------------------------------------
def path_slug(url: str, keep: int = 3) -> str:
    segs = [s for s in urlsplit(url).path.split("/") if s]
    out = []
    for s in segs[-keep:]:
        out.append("N" if re.fullmatch(r"\d+|[0-9a-fA-F-]{16,}", s) else re.sub(r"[^A-Za-z0-9_]", "_", s)[:24])
    return ".".join(out) or "root"


def json_quantities(obj: Any, prefix: str, limit: int = 48, max_depth: int = 5) -> List[Quantity]:
    """Exact numeric leaves (and list lengths) of a JSON value as quantities named by their key path."""
    out: List[Quantity] = []

    def walk(v: Any, name: str, depth: int) -> None:
        if len(out) >= limit or depth > max_depth:
            return
        if isinstance(v, bool):
            out.append(Quantity(name, 1.0 if v else 0.0, kind=OutcomeKind.BINARY))
        elif isinstance(v, int):
            out.append(Quantity(name, float(v), kind=OutcomeKind.COUNT if v >= 0 else OutcomeKind.REAL))
        elif isinstance(v, float):
            if v == v and abs(v) != float("inf"):
                out.append(Quantity(name, v))
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                walk(v[k], f"{name}.{re.sub(r'[^A-Za-z0-9_]', '_', str(k))[:32]}", depth + 1)
        elif isinstance(v, list):
            out.append(Quantity(f"{name}.count", float(len(v)), kind=OutcomeKind.COUNT))
            for i, x in enumerate(v[:4]):
                if isinstance(x, (dict, list)):
                    walk(x, f"{name}.{i}", depth + 1)

    walk(obj, prefix, 0)
    return out[:limit]


_CREDENTIAL_HEADER = re.compile(r"authorization|cookie|token|secret|api[-_]?key|password|credential", re.I)
_SAFE_RESPONSE_HEADERS = ("content-type", "etag", "last-modified", "location", "content-length")
_METHODS = ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE")
_REDIRECTS = (301, 302, 303, 307, 308)

_AUTH_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["secret"], "properties": {
    "secret": {"type": "string", "minLength": 1, "description": "name of the secret; never its value"},
    "scheme": {"type": "string", "enum": ["bearer", "bot", "basic", "header", "query"]},
    "name": {"type": "string", "description": "header or query-parameter name for scheme header / query"}}}
_REQUEST_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["url"], "properties": {
    "method": {"type": "string", "enum": list(_METHODS)},
    "url": {"type": "string", "minLength": 8, "maxLength": 4096},
    "query": {"type": "object"},
    "headers": {"type": "object", "additionalProperties": {"type": "string"}},
    "json": {},
    "body": {"type": "string"},
    "auth": _AUTH_SCHEMA}}
_GET_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["url"], "properties": {
    "url": _REQUEST_SCHEMA["properties"]["url"], "query": {"type": "object"},
    "headers": _REQUEST_SCHEMA["properties"]["headers"], "auth": _AUTH_SCHEMA}}


@dataclass
class _Prepared:
    method: str
    url: str
    headers: Dict[str, str]
    body: bytes
    auth: Dict[str, str] = field(default_factory=dict)


class HttpApiExecutor:
    """``ModalityExecutor`` for the ``api`` modality (tools ``http.get`` and ``http.request``)."""

    modality = Modality.API

    def __init__(self, *, allowed_schemes: Sequence[str] = ("https", "http"), timeout_s: float = 30.0,
                 max_response_bytes: int = 1_000_000, max_text: int = 4000, max_quantities: int = 48,
                 max_redirects: int = 5, max_rate_wait_s: float = 30.0, retry: Optional[RetryPolicy] = None,
                 transport: Optional[httpx.BaseTransport] = None, sleep: Optional[Callable[[float], None]] = None,
                 user_agent: str = "cam-live/1.0 (api executor)"):
        self.allowed_schemes = tuple(allowed_schemes)
        self.timeout_s, self.max_response_bytes, self.max_text = timeout_s, max_response_bytes, max_text
        self.max_quantities, self.max_redirects, self.max_rate_wait_s = max_quantities, max_redirects, max_rate_wait_s
        self.retry = retry or RetryPolicy()
        self.sleep, self.user_agent = sleep, user_agent
        self.limiter = RateLimiter()
        self._client = httpx.Client(transport=transport, follow_redirects=False, trust_env=False)

    # ---- ModalityExecutor ----------------------------------------------------------------------------------
    def tools(self) -> List[ToolDescriptor]:
        return [
            ToolDescriptor(name="http.get", modality=Modality.API, level=Level.PRIMITIVE, auth_scope="http.read",
                           side_effects="external_read", arg_schema=_GET_SCHEMA, est_cost=Cost(seconds=1.0, primitive_ops=1),
                           description="Send one HTTP GET request to an API endpoint. Returns the status, a bounded "
                                       "digest of the response body, its numeric JSON fields as exact quantities "
                                       "and any rate-limit state the server reported."),
            ToolDescriptor(name="http.request", modality=Modality.API, level=Level.PRIMITIVE, auth_scope="http.write",
                           side_effects="external_write", arg_schema=_REQUEST_SCHEMA,
                           est_cost=Cost(seconds=1.0, primitive_ops=1),
                           description="Send one HTTP request (GET, HEAD, POST, PUT, PATCH or DELETE) with an optional "
                                       "JSON or text body to an API endpoint. A credential is referenced by the name "
                                       "of a stored secret. Returns the status, a bounded digest of the response "
                                       "body, its numeric JSON fields and any rate-limit state the server reported."),
        ]

    def validate(self, action: ActionSpec) -> List[str]:
        if action.operation not in ("http.get", "http.request"):
            return [f"unknown operation {action.operation!r} for the api executor"]
        a = action.arguments
        errs = schema_errors(_GET_SCHEMA if action.operation == "http.get" else _REQUEST_SCHEMA, a)
        if errs:
            return errs
        u = urlsplit(a["url"])
        if u.scheme not in self.allowed_schemes:
            errs.append(f"arguments.url: scheme {u.scheme!r} is not one of {list(self.allowed_schemes)}")
        if not u.hostname:
            errs.append("arguments.url: no host")
        if u.username or u.password:
            errs.append("arguments.url: credentials must not be embedded in the URL; reference a secret by name")
        for k in (a.get("headers") or {}):
            if _CREDENTIAL_HEADER.search(k):
                errs.append(f"arguments.headers.{k}: credential headers are injected from a named secret ('auth')")
        if "json" in a and "body" in a:
            errs.append("arguments: give either 'json' or 'body', not both")
        auth = a.get("auth") or {}
        if auth.get("scheme") in ("header", "query") and not auth.get("name"):
            errs.append("arguments.auth.name: required for scheme header / query")
        return errs

    def close(self) -> None:
        self._client.close()

    def execute(self, action: ActionSpec, ctx: ExecContext) -> ActionResult:
        clock, started = ctx.clock, ctx.clock.now()
        errs = self.validate(action)
        if errs:
            return failed_result(action, classify(FailureEvidence(arg_errors=errs, modality=Modality.API)), clock, started)
        req = self._prepare(action)
        return run_with_retries(lambda n: self._attempt(action, ctx, req, started), self.retry, clock,
                                sleep=self.sleep, deadline=ctx.deadline, cancel=ctx.cancel)

    # ---- internals -----------------------------------------------------------------------------------------
    def _prepare(self, action: ActionSpec) -> _Prepared:
        a = action.arguments
        method = "GET" if action.operation == "http.get" else str(a.get("method", "GET")).upper()
        u = urlsplit(a["url"])
        query = parse_qsl(u.query, keep_blank_values=True) + [(str(k), str(v)) for k, v in (a.get("query") or {}).items()]
        url = urlunsplit((u.scheme, u.netloc, u.path or "/", urlencode(query), ""))
        headers = {str(k): str(v) for k, v in (a.get("headers") or {}).items()}
        headers.setdefault("User-Agent", self.user_agent)
        headers.setdefault("Accept", "application/json, text/plain;q=0.8, */*;q=0.5")
        body = b""
        if "json" in a:
            body = json.dumps(a["json"], sort_keys=True).encode()
            headers.setdefault("Content-Type", "application/json")
        elif "body" in a:
            body = str(a["body"]).encode()
            headers.setdefault("Content-Type", "text/plain; charset=utf-8")
        return _Prepared(method=method, url=url, headers=headers, body=body, auth=dict(a.get("auth") or {}))

    @staticmethod
    def _inject(req_headers: Dict[str, str], url: str, auth: Dict[str, str], value: str) -> str:
        scheme = auth.get("scheme", "bearer")
        if scheme == "bearer":
            req_headers["Authorization"] = f"Bearer {value}"
        elif scheme == "bot":
            req_headers["Authorization"] = f"Bot {value}"
        elif scheme == "basic":
            req_headers["Authorization"] = "Basic " + base64.b64encode(value.encode()).decode()
        elif scheme == "header":
            req_headers[auth["name"]] = value
        elif scheme == "query":
            u = urlsplit(url)
            q = parse_qsl(u.query, keep_blank_values=True) + [(auth["name"], value)]
            url = urlunsplit((u.scheme, u.netloc, u.path, urlencode(q), ""))
        return url

    def _attempt(self, action: ActionSpec, ctx: ExecContext, req: _Prepared, started: float) -> ActionResult:
        clock = ctx.clock
        ops: List[PrimitiveOp] = []
        method, url, body, send_auth = req.method, req.url, req.body, bool(req.auth)
        origin0 = urlsplit(url)[:2]
        t_wall = _time.perf_counter()
        for hop in range(self.max_redirects + 1):
            host = urlsplit(url).netloc
            op = PrimitiveOp(op="http.request", modality=Modality.API, started=clock.now(),
                             arguments={"method": method, "url": url, "host": host, "hop": hop,
                                        "body_sha256": hashlib.sha256(body).hexdigest() if body else "",
                                        "body_bytes": len(body),
                                        "auth_secret": req.auth.get("secret", "") if send_auth else ""})
            ops.append(op)
            refused = authorization_result(action, ctx.check(op, action), clock, started, ops=ops)
            if refused is not None:
                return refused
            if ctx.cancel is not None and ctx.cancel.is_set():
                op.status, op.ended = ResultStatus.CANCELLED, clock.now()
                return ActionResult(action_id=action.action_id, status=ResultStatus.CANCELLED, started=started,
                                    ended=clock.now(), ops=ops)
            pause = self.limiter.wait_needed(host, clock.now())
            if pause > 0:
                if pause > self.max_rate_wait_s or (ctx.deadline and clock.now() + pause > ctx.deadline):
                    op.status, op.failure_class, op.ended = ResultStatus.FAILED, FailureClass.RESOURCE_LIMIT, clock.now()
                    op.note = "rate-limit window not yet reset; request not sent"
                    rep = classify(FailureEvidence(limit_exceeded=True, retry_after_s=pause, modality=Modality.API,
                                                   detail=f"rate limit of {host}: {pause:.1f}s until the window resets"))
                    return failed_result(action, rep, clock, started, ops=ops)
                wait(clock, pause, self.sleep)
            headers, send_url, secret_values = dict(req.headers), url, []
            if send_auth:
                value = resolve_secret(ctx.secrets, req.auth["secret"], action=granted_action(ctx, action),
                                       decision=getattr(ctx, "authorization", None), destination=url)
                if value is None:
                    op.status, op.failure_class, op.ended = ResultStatus.FAILED, FailureClass.MISSING_PERMISSION, clock.now()
                    rep = classify(FailureEvidence(exception=PermissionError(
                        f"secret {req.auth['secret']!r} is not available to this executor"), modality=Modality.API))
                    return failed_result(action, rep, clock, started, ops=ops)
                secret_values.append(value)
                send_url = self._inject(headers, url, req.auth, value)
            timeout = min(self.timeout_s, action.timeout_s or self.timeout_s)
            try:
                with self._client.stream(method, send_url, headers=headers, content=body or None,
                                         timeout=timeout) as resp:
                    chunks, size, truncated = [], 0, False
                    for chunk in resp.iter_bytes():
                        chunks.append(chunk)
                        size += len(chunk)
                        if size > self.max_response_bytes:
                            truncated = True
                            break
                    raw = b"".join(chunks)[: self.max_response_bytes]
                    status, rheaders = resp.status_code, dict(resp.headers)
            except httpx.HTTPError as e:
                op.ended = clock.now()
                rep = classify(FailureEvidence(exception=e, timed_out=isinstance(e, httpx.TimeoutException),
                                               modality=Modality.API,
                                               detail=scrub(f"{type(e).__name__}: {e}", secret_values)))
                op.status = ResultStatus.TIMEOUT if isinstance(e, httpx.TimeoutException) else ResultStatus.FAILED
                op.failure_class = rep.failure_class
                return failed_result(action, rep, clock, started, ops=ops)
            op.ended = clock.now()
            parsed = self._parse(raw, rheaders)
            info = parse_rate_limit(rheaders, clock.now(), parsed)
            self.limiter.update(host, info, status, clock.now())
            if status in _REDIRECTS and rheaders.get("location"):
                nxt = urljoin(url, rheaders["location"])
                op.note = f"redirect {status}"
                if urlsplit(nxt)[:2] != origin0:
                    send_auth = False                      # a credential never follows a redirect to another origin
                if status == 303 or (status in (301, 302) and method == "POST"):
                    method, body = "GET", b""
                url = nxt
                continue
            elapsed = _time.perf_counter() - t_wall
            obs = self._digest(action, method, url, status, rheaders, raw, parsed, truncated, info, elapsed,
                               secret_values, clock, hop)
            op.obs_after = obs.obs_id
            if status >= 400:
                rep = classify(FailureEvidence(http_status=status, retry_after_s=info.retry_after_s,
                                               limit_exceeded=status == 429, modality=Modality.API,
                                               detail=f"HTTP {status} from {method} {urlsplit(url).path}: {obs.text[:200]}"))
                op.status, op.failure_class = ResultStatus.FAILED, rep.failure_class
                return failed_result(action, rep, clock, started, ops=ops, observations=[obs])
            from .skills import evaluate_condition        # local import: skills imports this module
            verified = evaluate_condition(action.verify, [obs], action.arguments)
            res = ActionResult(action_id=action.action_id, status=ResultStatus.OK, started=started, ended=clock.now(),
                               observations=[obs], ops=ops, verified=verified,
                               cost=Cost(seconds=elapsed, primitive_ops=len(ops)))
            if verified is False:
                rep = classify(FailureEvidence(verified=False, executed_ok=True, modality=Modality.API,
                                               detail=f"verification condition not met: {action.verify.description}"))
                fail = failed_result(action, rep, clock, started, ops=ops, observations=[obs])
                fail.verified, fail.cost = False, res.cost
                return fail
            return res
        rep = classify(FailureEvidence(target_missing=True, modality=Modality.API,
                                       detail=f"more than {self.max_redirects} redirects"))
        return failed_result(action, rep, clock, started, ops=ops)

    @staticmethod
    def _parse(raw: bytes, headers: Mapping[str, str]) -> Any:
        ctype = str(headers.get("content-type", "")).lower()
        if "json" in ctype or raw[:1] in (b"{", b"["):
            try:
                return json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                return None
        return None

    def _digest(self, action: ActionSpec, method: str, url: str, status: int, headers: Mapping[str, str], raw: bytes,
                parsed: Any, truncated: bool, info: RateLimitInfo, elapsed: float, secret_values: List[str],
                clock: Any, hops: int) -> Observation:
        slug = path_slug(url)
        if parsed is not None:
            parsed = scrub(parsed, secret_values)
            text = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
        else:
            text = scrub(raw.decode("utf-8", errors="replace"), secret_values)
        q = [Quantity("http.status", float(status)), Quantity("http.elapsed_s", float(elapsed), unit="s"),
             Quantity("http.response_bytes", float(len(raw)), unit="bytes", kind=OutcomeKind.COUNT)]
        q += info.quantities("http")
        if parsed is not None:
            q += json_quantities(parsed, f"http.{slug}", limit=self.max_quantities)
        data: Dict[str, Any] = {
            "status": status, "method": method, "url": scrub(url, secret_values), "redirects": hops,
            "headers": {k: v for k, v in ((k.lower(), v) for k, v in headers.items()) if k in _SAFE_RESPONSE_HEADERS},
            "truncated": truncated or len(text) > self.max_text,
            "structure_hash": structure_hash(parsed) if parsed is not None else "",
        }
        if parsed is not None and len(text) <= 16 * self.max_text:
            data["json"] = parsed
        return Observation(
            obs_id=new_id("obs"), ts=clock.now(), kind="tool_result", text=text[: self.max_text], quantities=q,
            provenance=Provenance(source="http_api", uri=data["url"], retrieved_at=clock.now(), trust=Trust.UNTRUSTED,
                                  content_sha256=hashlib.sha256(raw).hexdigest()),
            parent_action_id=action.action_id, data=data)


class ApiModality:
    """Several ``api``-modality executors (generic HTTP, publishing, Discord, …) behind one ``ModalityExecutor``.
    An operation is routed to the member that lists it in ``tools()``."""

    modality = Modality.API

    def __init__(self, members: Sequence[Any]):
        self.members = list(members)

    def _owner(self, operation: str) -> Optional[Any]:
        for m in self.members:
            if any(t.name == operation for t in m.tools()):
                return m
        return None

    def tools(self) -> List[ToolDescriptor]:
        return [t for m in self.members for t in m.tools()]

    def validate(self, action: ActionSpec) -> List[str]:
        m = self._owner(action.operation)
        return m.validate(action) if m is not None else [f"unknown api operation {action.operation!r}"]

    def execute(self, action: ActionSpec, ctx: ExecContext) -> ActionResult:
        m = self._owner(action.operation)
        if m is None:
            rep = classify(FailureEvidence(arg_errors=[f"unknown api operation {action.operation!r}"],
                                           modality=Modality.API))
            return failed_result(action, rep, ctx.clock, ctx.clock.now())
        return m.execute(action, ctx)

    def close(self) -> None:
        for m in self.members:
            m.close()
