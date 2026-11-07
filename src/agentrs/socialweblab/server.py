"""HTTP front end of SocialWebLab: JSON API, server-rendered HTML pages and a privileged
control channel, all on top of the engine's service layer (``engine.SocialWebLab``).

Everything is implemented by one socket-free function, ``LabApp.dispatch(method, path, query,
headers, body) -> (status, headers, body)``; ``SocialWebLabServer`` only adapts it to a
``ThreadingHTTPServer`` and ``InProcessTransport`` calls it directly, so authentication, rate
limiting and error handling are identical in-process and over HTTP.

Authentication
    * user sessions: cookie ``swl_session`` (set by ``POST /api/login`` or the ``/login`` form) or
      ``Authorization: Bearer <session token>``;
    * observers: ``X-Api-Key: <key>`` (issued through the control channel) - read-only access to
      the JSON API; observer reads never count as item views.
    Every ``/api/*`` route except ``POST /api/login`` needs one of the two (401 otherwise); every
    HTML page except ``/login`` redirects anonymous visitors to ``/login?next=...``.

Rate limiting
    Token buckets (capacity = ``requests_per_minute``, refill ``requests_per_minute / 60`` per
    second) keyed by API key or session token, measured on the LAB clock (``lab.now()``), so under
    the virtual clock buckets only refill when the lab's time advances. Counted requests: every
    request authenticated by an API key, and every ``/api/*`` request or form ``POST`` made with a
    session (server-rendered HTML ``GET`` pages are not counted). Exhausted buckets get ``429`` with
    ``Retry-After`` (lab seconds) and ``{"error", "retry_after"}``.

JSON API (errors are ``{"error": msg}`` with the ``LabError`` status)::

    GET  /api/events?since=&limit=        {"events": [...], "now"}
    GET  /api/items/<id>                  lab.public_item (a view is counted for sessions only)
    GET  /api/users/<handle>              lab.public_user
    GET  /api/feed?space=&sort=&limit=    {"items": [...]}
    GET  /api/search?q=&space=            {"items": [...]}
    GET  /api/spaces                      {"spaces": [...]}
    POST /api/login {handle, password}    {"token"} + Set-Cookie
    POST /api/logout                      {"ok": true}
    POST /api/items {space, title, body, client_ref?}           {"id"}
    POST /api/items/<id>/responses {body, kind}                 {"id"}  (kind: comment | answer)
    POST /api/items/<id>/react {kind}                           {"ok", "new"}
    POST /api/items/<id>/edit {body, title?}                    {"version"}
    POST /api/items/<id>/delete                                 {"ok": true}
    POST /api/items/<id>/accept {response_id}                   {"ok": true}
    GET  /api/notifications               {"notifications": [...], "unread"}
    POST /api/notifications/read          {"marked"}

    JSON ``POST`` requests authenticated by cookie must carry ``Content-Type: application/json``
    (blocks cross-site form posts), and a present ``Origin`` header must match ``Host``.

HTML pages: ``/login``, ``/`` (hot feed + spaces), ``/s/<space>?sort=hot|new``, ``/item/<id>``,
``/new``, ``/search?q=``, ``/notifications``; plus the form endpoints ``POST /login``,
``/logout``, ``/new`` (no-JS fallback), ``/item/<id>/{reply,react,edit,delete,accept}``,
``/notifications/read`` (Post/Redirect/Get; errors re-render the page with a ``role=alert``).

Control channel: ``POST /__control/<token>/<op>`` with a JSON body; ``token`` is a random secret
(``secrets.token_urlsafe``) that is never rendered into a page; a wrong token gets a plain 404.
Ops: ``now``, ``advance {seconds}``, ``create_account {handle, password, spaces?, followers?,
kind?}``, ``login {handle, password}``, ``issue_api_key {source_id, requests_per_minute}``,
``revoke_api_key {api_key}``, ``verify_help {item_id, within_s}``, ``item_truth {item_id}``,
``snapshot {path}``, ``restore {path}``, ``stats``.

API keys and rate-limit buckets live in the ``LabApp`` (process memory), not in the lab database.
"""

from __future__ import annotations

import hmac
import html
import json
import math
import secrets
import sqlite3
import threading
import urllib.error
import urllib.parse
import urllib.request
import weakref
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit

from agentrs.socialweblab.engine import LabError, SocialWebLab

SESSION_COOKIE = "swl_session"
CONTROL_PREFIX = "/__control/"
MAX_BODY = 1 << 20
USER_RESPONSE_KINDS = ("comment", "answer")
BANNER = "SocialWebLab — simulated environment"

Response = tuple[int, dict, bytes]


# ====================================================================== rate limiting
@dataclass
class TokenBucket:
    """Token bucket on an external clock (the lab's)."""

    requests_per_minute: float
    tokens: float = -1.0
    last: float | None = None

    def __post_init__(self) -> None:
        if self.requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        if self.tokens < 0:
            self.tokens = float(self.requests_per_minute)

    @property
    def capacity(self) -> float:
        return float(self.requests_per_minute)

    @property
    def rate(self) -> float:  # tokens per second
        return self.requests_per_minute / 60.0

    def take(self, now: float) -> tuple[bool, float]:
        """Consume one token. Returns (allowed, retry_after_seconds)."""
        if self.last is not None and now > self.last:
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now  # (a clock that moved backwards, e.g. after restore(), just re-anchors)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True, 0.0
        return False, (1.0 - self.tokens) / self.rate


@dataclass
class ApiKey:
    key: str
    source_id: str
    bucket: TokenBucket
    revoked: bool = False


@dataclass
class _Auth:
    uid: int | None = None
    session: str | None = None
    via_cookie: bool = False
    api_key: ApiKey | None = None
    error: str | None = None  # an invalid credential was presented

    @property
    def observer(self) -> bool:
        return self.api_key is not None


# ====================================================================== the application
class LabApp:
    """Routes, auth, rate limits and rendering for one lab. Thread-safe."""

    def __init__(self, lab: SocialWebLab, control_token: str | None = None,
                 session_requests_per_minute: float = 120, publish_delay_ms: int = 300,
                 search_delay_ms: int = 300, poll_ms: int = 2000):
        self.lab = lab
        self.control_token = control_token or secrets.token_urlsafe(24)
        self.session_rpm = session_requests_per_minute
        self.publish_delay_ms = publish_delay_ms
        self.search_delay_ms = search_delay_ms
        self.poll_ms = poll_ms
        self._lock = threading.Lock()
        self._keys: dict[str, ApiKey] = {}
        self._session_buckets: dict[str, TokenBucket] = {}

    # ------------------------------------------------------------------ api keys
    def issue_api_key(self, source_id: str, requests_per_minute: float = 60) -> str:
        key = "swlk_" + secrets.token_urlsafe(24)
        with self._lock:
            self._keys[key] = ApiKey(key, str(source_id), TokenBucket(float(requests_per_minute)))
        return key

    def revoke_api_key(self, key: str) -> bool:
        with self._lock:
            k = self._keys.get(key)
            if k is None or k.revoked:
                return False
            k.revoked = True
            return True

    # ------------------------------------------------------------------ entry point
    def dispatch(self, method: str, path: str, query: dict | None = None, headers: dict | None = None,
                 body: bytes | None = None) -> Response:
        method = method.upper()
        query = {str(k): (v[-1] if isinstance(v, (list, tuple)) else v) for k, v in (query or {}).items()}
        query = {k: "" if v is None else str(v) for k, v in query.items()}
        headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        if body is not None and len(body) > MAX_BODY:
            return _json(413, {"error": "request body too large"})
        auth = _Auth()
        try:
            if path.startswith(CONTROL_PREFIX):
                return self._control_route(method, path, body)
            auth = self._authenticate(headers)
            if path.startswith("/api/"):
                return self._api(method, path, query, headers, body, auth)
            return self._pages(method, path, query, headers, body, auth)
        except LabError as e:
            if path.startswith("/api/"):
                return _json(e.status, {"error": str(e)})
            return _html(e.status, self._safe_error_page(str(e), auth.uid))
        except Exception as e:  # noqa: BLE001 - never crash the server on a request
            if path.startswith("/api/"):
                return _json(500, {"error": f"internal error: {type(e).__name__}"})
            return _html(500, self._safe_error_page("internal error", auth.uid))

    def _safe_error_page(self, msg: str, uid: int | None) -> str:
        try:
            return self._error_page(msg, uid)
        except Exception:  # noqa: BLE001 - e.g. the lab database is gone; plain page
            return f"<!doctype html><title>Error</title><p role=alert>{_e(msg)}</p>"

    # ------------------------------------------------------------------ auth + limits
    def _authenticate(self, headers: dict) -> _Auth:
        a = _Auth()
        key = headers.get("x-api-key")
        if key:
            with self._lock:
                k = self._keys.get(key)
            if k is None or k.revoked:
                a.error = "invalid API key"
            else:
                a.api_key = k
            return a
        tok, cookie = None, False
        authz = headers.get("authorization", "")
        if authz.lower().startswith("bearer "):
            tok = authz[7:].strip()
        elif "cookie" in headers:
            c = SimpleCookie()
            try:
                c.load(headers["cookie"])
            except CookieError:
                c = SimpleCookie()
            if SESSION_COOKIE in c:
                tok, cookie = c[SESSION_COOKIE].value, True
        if tok:
            uid = self.lab.user_for_token(tok)
            if uid is None:
                a.error = "invalid or expired session"
            else:
                a.uid, a.session, a.via_cookie = uid, tok, cookie
        return a

    def _limit(self, auth: _Auth) -> Response | None:
        """Charge one request to the caller's bucket; a 429 response if exhausted."""
        now = self.lab.now()
        with self._lock:
            if auth.api_key is not None:
                bucket = auth.api_key.bucket
            elif auth.session is not None:
                bucket = self._session_buckets.get(auth.session)
                if bucket is None:
                    bucket = self._session_buckets[auth.session] = TokenBucket(float(self.session_rpm))
            else:
                return None
            ok, retry = bucket.take(now)
        if ok:
            return None
        ra = max(1, math.ceil(retry))
        return 429, {"Content-Type": "application/json", "Retry-After": str(ra)}, \
            json.dumps({"error": "rate limit exceeded", "retry_after": ra}).encode()

    @staticmethod
    def _origin_ok(headers: dict) -> bool:
        origin = headers.get("origin")
        if not origin:
            return True  # non-browser clients (and same-origin GET-initiated navigations) send none
        if origin == "null":
            return False
        return urlsplit(origin).netloc == headers.get("host", "")

    # ------------------------------------------------------------------ JSON API
    def _api(self, method: str, path: str, q: dict, headers: dict, body: bytes | None, auth: _Auth) -> Response:
        lab = self.lab
        parts = [p for p in path.split("/") if p][1:]  # drop "api"
        if method == "POST" and parts == ["login"]:
            data = _json_body(body)
            tok = lab.login(str(data.get("handle", "")), str(data.get("password", "")))
            return 200, {"Content-Type": "application/json", "Set-Cookie": _session_cookie(tok)}, \
                json.dumps({"token": tok}).encode()
        if auth.error:
            return _json(401, {"error": auth.error})
        if auth.uid is None and not auth.observer:
            return _json(401, {"error": "login required"})
        limited = self._limit(auth)
        if limited:
            return limited
        uid = auth.uid
        if method == "GET":
            if parts == ["events"]:
                since = _int(q.get("since"), 0)
                limit = max(1, min(1000, _int(q.get("limit"), 200)))
                return _json(200, {"events": lab.public_events(since, limit), "now": lab.now()})
            if len(parts) == 2 and parts[0] == "items":
                return _json(200, lab.public_item(_id(parts[1]), viewer=uid))
            if len(parts) == 2 and parts[0] == "users":
                return _json(200, lab.public_user(parts[1]))
            if parts == ["feed"]:
                sort = q.get("sort") or "hot"
                if sort not in ("hot", "new"):
                    raise LabError("sort must be 'hot' or 'new'")
                limit = max(1, min(100, _int(q.get("limit"), 20)))
                return _json(200, {"items": lab.feed(uid, q.get("space") or None, limit=limit, sort=sort)})
            if parts == ["search"]:
                return _json(200, {"items": lab.search(q.get("q", ""), q.get("space") or None)})
            if parts == ["spaces"]:
                return _json(200, {"spaces": lab.spaces()})
            if parts == ["notifications"]:
                if uid is None:
                    return _json(403, {"error": "notifications need a user session"})
                ns = lab.notifications(uid)
                return _json(200, {"notifications": ns, "unread": sum(1 for n in ns if not n["read"])})
            return _json(404, {"error": "no such route"})
        if method != "POST":
            return _json(405, {"error": "method not allowed"})
        if auth.observer:
            return _json(403, {"error": "observer API keys are read-only"})
        if auth.via_cookie and (not headers.get("content-type", "").startswith("application/json")
                                or not self._origin_ok(headers)):
            return _json(403, {"error": "cross-site request refused"})
        assert uid is not None
        data = _json_body(body)
        if parts == ["logout"]:
            self._logout(auth.session)
            return 200, {"Content-Type": "application/json", "Set-Cookie": _session_cookie("", clear=True)}, \
                b'{"ok": true}'
        if parts == ["items"]:
            ref = data.get("client_ref")
            iid = lab.create_item(uid, str(data.get("space", "")), str(data.get("title", "")),
                                  str(data.get("body", "")), client_ref=str(ref)[:64] if ref else None)
            return _json(200, {"id": iid})
        if parts == ["notifications", "read"]:
            return _json(200, {"marked": lab.mark_read(uid)})
        if len(parts) == 3 and parts[0] == "items":
            iid, op = _id(parts[1]), parts[2]
            if op == "responses":
                kind = str(data.get("kind", "comment"))
                if kind not in USER_RESPONSE_KINDS:
                    raise LabError("kind must be 'comment' or 'answer'")
                return _json(200, {"id": lab.respond(uid, iid, str(data.get("body", "")), kind=kind)})
            if op == "react":
                new = lab.react(uid, iid, str(data.get("kind", "")))
                return _json(200, {"ok": True, "new": new})
            if op == "edit":
                title = data.get("title")
                v = lab.edit_item(uid, iid, str(data.get("body", "")), title=str(title) if title else None)
                return _json(200, {"version": v})
            if op == "delete":
                lab.delete_item(uid, iid)
                return _json(200, {"ok": True})
            if op == "accept":
                lab.accept(uid, iid, _id(data.get("response_id")))
                return _json(200, {"ok": True})
        return _json(404, {"error": "no such route"})

    def _logout(self, token: str | None) -> None:
        if not token:
            return
        logout = getattr(self.lab, "logout", None)
        if callable(logout):
            logout(token)
        else:  # the engine has no logout service yet: drop the session row
            self.lab._x("DELETE FROM sessions WHERE token=?", (token,))
        with self._lock:
            self._session_buckets.pop(token, None)

    # ------------------------------------------------------------------ control channel
    def _control_route(self, method: str, path: str, body: bytes | None) -> Response:
        token, _, op = path[len(CONTROL_PREFIX):].partition("/")
        if method != "POST" or not hmac.compare_digest(token.encode(), self.control_token.encode()):
            return _not_found()  # do not reveal that the endpoint exists
        try:
            payload = _json_body(body)
            return _json(200, self.control(op, payload))
        except LabError as e:
            return _json(e.status, {"error": str(e)})
        except sqlite3.IntegrityError as e:
            return _json(409, {"error": f"conflict: {e}"})
        except (KeyError, ValueError, TypeError) as e:
            return _json(400, {"error": f"bad control request: {type(e).__name__}: {e}"})
        except Exception as e:  # noqa: BLE001 - control errors are reported, never crash the server
            return _json(500, {"error": f"{type(e).__name__}: {e}"})

    def control(self, op: str, p: dict) -> dict:
        """Privileged operations (the token has been checked)."""
        lab = self.lab
        if op == "now":
            return {"now": lab.now()}
        if op == "advance":
            n = lab.advance(float(p["seconds"]))
            return {"jobs_run": n, "now": lab.now()}
        if op == "create_account":
            uid = lab.create_account(str(p["handle"]), str(p["password"]), kind=str(p.get("kind", "agent")),
                                     spaces=list(p.get("spaces") or []), followers=int(p.get("followers", 0)))
            return {"uid": uid}
        if op == "login":
            return {"token": lab.login(str(p["handle"]), str(p["password"]))}
        if op == "issue_api_key":
            return {"api_key": self.issue_api_key(str(p["source_id"]), float(p.get("requests_per_minute", 60)))}
        if op == "revoke_api_key":
            return {"revoked": self.revoke_api_key(str(p["api_key"]))}
        if op == "verify_help":
            return lab.helpful_answer_within(int(p["item_id"]), float(p["within_s"]))
        if op == "item_truth":
            return lab.item_truth(int(p["item_id"]))
        if op == "snapshot":
            return {"path": lab.snapshot(str(p["path"]))}
        if op == "restore":
            lab.restore(str(p["path"]))
            return {"ok": True, "now": lab.now()}
        if op == "stats":
            return lab.stats()
        raise LabError(f"unknown control op {op!r}", 404)

    # ------------------------------------------------------------------ HTML pages
    def _pages(self, method: str, path: str, q: dict, headers: dict, body: bytes | None, auth: _Auth) -> Response:
        if path == "/login":
            if method == "POST":
                return self._login_post(_form_body(body), headers)
            return _html(200, self._login_page(_safe_next(q.get("next", "/")), None))
        if auth.uid is None:
            if method == "GET":
                return _redirect("/login?next=" + quote(path + ("?" + urllib.parse.urlencode(q) if q else ""), safe=""))
            return _html(401, self._login_page("/", "Please log in first."))
        uid = auth.uid
        if method == "POST":
            if not self._origin_ok(headers):
                return _html(403, self._error_page("cross-site request refused", uid))
            limited = self._limit(auth)
            if limited:
                st, hd, _ = limited
                return st, {**hd, "Content-Type": "text/html; charset=utf-8"}, \
                    self._error_page("Too many requests. Please slow down.", uid).encode()
            return self._form_post(path, _form_body(body), auth)
        if method not in ("GET", "HEAD"):
            return _html(405, self._error_page("method not allowed", uid))
        if path == "/":
            return _html(200, self._home_page(uid))
        if path.startswith("/s/"):
            sort = q.get("sort") or "hot"
            return _html(200, self._space_page(uid, unquote(path[3:]), "new" if sort == "new" else "hot"))
        if path.startswith("/item/"):
            return _html(200, self._item_page(uid, _id(path[6:]), None))
        if path == "/new":
            return _html(200, self._new_page(uid, q.get("space", ""), None))
        if path == "/search":
            return _html(200, self._search_page(uid, q.get("q", "")))
        if path == "/notifications":
            return _html(200, self._notifications_page(uid))
        return _html(404, self._error_page("Page not found.", uid))

    def _login_post(self, form: dict, headers: dict) -> Response:
        nxt = _safe_next(form.get("next", "/"))
        if not self._origin_ok(headers):
            return _html(403, self._login_page(nxt, "cross-site request refused"))
        try:
            tok = self.lab.login(form.get("handle", ""), form.get("password", ""))
        except LabError as e:
            return _html(e.status, self._login_page(nxt, "Invalid handle or password."))
        return 303, {"Location": nxt, "Set-Cookie": _session_cookie(tok), "Content-Type": "text/plain"}, b""

    def _form_post(self, path: str, f: dict, auth: _Auth) -> Response:
        lab, uid = self.lab, auth.uid
        assert uid is not None
        if path == "/logout":
            self._logout(auth.session)
            return 303, {"Location": "/login", "Set-Cookie": _session_cookie("", clear=True),
                         "Content-Type": "text/plain"}, b""
        if path == "/notifications/read":
            lab.mark_read(uid)
            return _redirect("/notifications")
        if path == "/new":
            try:
                iid = lab.create_item(uid, f.get("space", ""), f.get("title", ""), f.get("body", ""),
                                      client_ref=f.get("client_ref") or None)
            except LabError as e:
                return _html(e.status, self._new_page(uid, f.get("space", ""), str(e), f))
            return _redirect(f"/item/{iid}")
        if path.startswith("/item/"):
            sid, _, op = path[6:].partition("/")
            iid = _id(sid)
            try:
                if op == "reply":
                    kind = f.get("kind", "answer")
                    if kind not in USER_RESPONSE_KINDS:
                        raise LabError("reply type must be Answer or Comment")
                    rid = lab.respond(uid, iid, f.get("body", ""), kind=kind)
                    return _redirect(f"/item/{iid}#response-{rid}")
                if op == "react":
                    lab.react(uid, iid, f.get("kind", ""))
                    return _redirect(f"/item/{iid}")
                if op == "edit":
                    lab.edit_item(uid, iid, f.get("body", ""), title=f.get("title") or None)
                    return _redirect(f"/item/{iid}")
                if op == "delete":
                    lab.delete_item(uid, iid)
                    return _redirect("/")
                if op == "accept":
                    lab.accept(uid, iid, _id(f.get("response_id")))
                    return _redirect(f"/item/{iid}")
            except LabError as e:
                if e.status == 404 and "item" in str(e):
                    raise
                return _html(e.status, self._item_page(uid, iid, str(e)))
        return _html(404, self._error_page("Page not found.", uid))

    # ------------------------------------------------------------------ rendering helpers
    def _layout(self, title: str, main: str, uid: int | None, script: str = "") -> str:
        lab = self.lab
        nav = ""
        if uid is not None:
            unread = sum(1 for n in lab.notifications(uid, unread_only=True))
            handle = _e(lab.handle(uid))
            nav = (f'<nav aria-label="Main"><a href="/" id="nav-home">Home</a> '
                   f'<a href="/new" id="nav-new">New post</a> <a href="/search" id="nav-search">Search</a> '
                   f'<a href="/notifications" id="nav-notifications">Notifications '
                   f'<span id="unread-badge">({unread})</span></a>'
                   f'<span class="who">Signed in as {handle}</span>'
                   f'<form method="post" action="/logout" class="inline"><button type="submit" id="logout">'
                   f'Log out</button></form></nav>')
            script = _POLL_JS.replace("__POLL_MS__", str(int(self.poll_ms))) + script
        clock = _fmt_time(lab.now())
        return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width, initial-scale=1">'
                f'<title>{_e(title)} · SocialWebLab</title><style>{_CSS}</style></head><body>'
                f'<header><div role="note" id="sim-banner">{BANNER}: every other account is a simulated user '
                f'and all activity stays inside this lab.</div>{nav}'
                f'<p class="clock">Lab time: <time id="lab-time">{clock}</time></p></header>'
                f'<main id="main">{main}</main>'
                + (f"<script>{script}</script>" if script else "") + "</body></html>")

    def _error_page(self, msg: str, uid: int | None) -> str:
        return self._layout("Error", f'<h1>Something went wrong</h1><div role="alert" id="page-error">{_e(msg)}</div>'
                            f'<p><a href="/">Back to the home page</a></p>', uid)

    def _login_page(self, nxt: str, error: str | None) -> str:
        err = f'<div role="alert" id="login-error">{_e(error)}</div>' if error else ""
        main = (f'<h1>Log in to SocialWebLab</h1>{err}'
                f'<form method="post" action="/login" id="login-form">'
                f'<input type="hidden" name="next" value="{_e(nxt)}">'
                f'<div class="field"><label for="login-handle">Handle</label>'
                f'<input id="login-handle" name="handle" autocomplete="username"></div>'
                f'<div class="field"><label for="login-password">Password</label>'
                f'<input id="login-password" name="password" type="password" autocomplete="current-password"></div>'
                f'<button type="submit" id="login-submit">Log in</button></form>')
        return self._layout("Log in", main, None)

    def _feed_list(self, items: list[dict], list_id: str) -> str:
        if not items:
            return f'<p id="{list_id}-empty">No posts yet.</p>'
        rows = "".join(
            f'<li><a href="/item/{it["id"]}">{_e(it["title"])}</a> '
            f'<span class="meta">in {_e(it["space"])} · by {_e(it["author"])} · {it["responses"]} responses · '
            f'{it.get("reactions", 0)} upvotes · {_fmt_time(it["created_at"])}</span></li>' for it in items)
        return f'<ol id="{list_id}" class="feed">{rows}</ol>'

    def _home_page(self, uid: int) -> str:
        lab = self.lab
        spaces = "".join(f'<li><a href="/s/{quote(s["name"])}">{_e(s["name"])}</a> '
                         f'<span class="meta">({_e(s["app"])})</span></li>' for s in lab.spaces())
        main = (f'<h1>Your feed</h1>{self._feed_list(lab.feed(uid, limit=15, sort="hot"), "feed")}'
                f'<section aria-labelledby="spaces-h"><h2 id="spaces-h">Spaces</h2>'
                f'<ul id="spaces" class="spaces">{spaces}</ul></section>')
        return self._layout("Home", main, uid)

    def _space_page(self, uid: int, space: str, sort: str) -> str:
        lab = self.lab
        if space not in {s["name"] for s in lab.spaces()}:
            raise LabError("no such space", 404)
        sp = quote(space)
        links = ('<nav aria-label="Sort" class="sort">Sort: '
                 + ('<strong aria-current="page">Hot</strong>' if sort == "hot" else f'<a href="/s/{sp}?sort=hot">Hot</a>')
                 + " · " + ('<strong aria-current="page">New</strong>' if sort == "new"
                            else f'<a href="/s/{sp}?sort=new">New</a>') + "</nav>")
        main = (f'<h1>{_e(space)}</h1>{links}<p><a href="/new?space={sp}" id="new-in-space">New post in '
                f'{_e(space)}</a></p>{self._feed_list(lab.feed(uid, space, limit=20, sort=sort), "space-feed")}')
        return self._layout(space, main, uid)

    def _item_page(self, uid: int, iid: int, error: str | None) -> str:
        lab = self.lab
        it = lab.public_item(iid, viewer=uid)  # a logged-in page view counts as a view
        is_author = lab.handle(uid) == it["author"]
        open_ = it["state"] == "visible"
        c = it["counts"]
        err = f'<div role="alert" id="page-error">{_e(error)}</div>' if error else ""
        edited = f' · edited {_fmt_time(it["edited_at"])}' if it["edited_at"] else ""
        state = "" if open_ else f' · <strong id="item-state">{_e(it["state"])}</strong>'
        react = ""
        if open_ and not is_author:
            mine = _viewer_reaction(lab, uid, iid)
            if mine:
                react = f'<p id="your-reaction">You {"upvoted" if mine == "up" else "downvoted"} this post.</p>'
            else:
                react = (f'<form method="post" action="/item/{iid}/react" class="inline" id="react-form">'
                         f'<button type="submit" name="kind" value="up" id="upvote">Upvote</button> '
                         f'<button type="submit" name="kind" value="down" id="downvote">Downvote</button></form>')
        manage = ""
        if is_author and it["state"] not in ("removed", "deleted"):
            manage = (f'<section aria-labelledby="manage-h" id="manage"><h2 id="manage-h">Manage your post</h2>'
                      f'<form method="post" action="/item/{iid}/edit" id="edit-form">'
                      f'<div class="field"><label for="edit-title">Edit title</label>'
                      f'<input id="edit-title" name="title" value="{_e(it["title"])}"></div>'
                      f'<div class="field"><label for="edit-body">Edit body</label>'
                      f'<textarea id="edit-body" name="body" rows="3">{_e(it["body"])}</textarea></div>'
                      f'<button type="submit" id="edit-submit">Save changes</button></form>'
                      f'<form method="post" action="/item/{iid}/delete" class="inline">'
                      f'<button type="submit" id="delete-item">Delete post</button></form></section>')
        rows = []
        for r in it["responses"]:
            acc = ' · <strong class="accepted">Accepted answer</strong>' if r["accepted"] else ""
            accept = ""
            if is_author and r["kind"] == "answer" and it["status"] == "open" and open_:
                accept = (f'<form method="post" action="/item/{iid}/accept" class="inline">'
                          f'<input type="hidden" name="response_id" value="{r["id"]}">'
                          f'<button type="submit" id="accept-{r["id"]}">Accept this answer</button></form>')
            rows.append(f'<li id="response-{r["id"]}" class="response"><p class="meta">{_e(r["kind"])} by '
                        f'{_e(r["author"])} · <time datetime="{_iso(r["created_at"])}">{_fmt_time(r["created_at"])}'
                        f'</time>{acc}</p><p class="rbody">{_e(r["body"])}</p>{accept}</li>')
        responses = (f'<section aria-labelledby="responses-h"><h2 id="responses-h">Responses ({c["responses"]})</h2>'
                     + (f'<ol id="responses">{"".join(rows)}</ol>' if rows else '<p id="no-responses">No responses yet.</p>')
                     + "</section>")
        reply = ""
        if open_:
            reply = (f'<section aria-labelledby="reply-h"><h2 id="reply-h">Reply</h2>'
                     f'<form method="post" action="/item/{iid}/reply" id="reply-form">'
                     f'<div class="field"><label for="reply-kind">Reply type</label>'
                     f'<select id="reply-kind" name="kind"><option value="answer">Answer</option>'
                     f'<option value="comment">Comment</option></select></div>'
                     f'<div class="field"><label for="reply-body">Your reply</label>'
                     f'<textarea id="reply-body" name="body" rows="3"></textarea></div>'
                     f'<button type="submit" id="reply-submit">Post reply</button></form></section>')
        main = (f'{err}<nav aria-label="Breadcrumb"><a href="/s/{quote(it["space"])}" id="crumb-space">'
                f'{_e(it["space"])}</a></nav><article id="item" data-item-id="{iid}">'
                f'<h1 id="item-title">{_e(it["title"])}</h1>'
                f'<p class="meta" id="item-meta">Posted by {_e(it["author"])} ({it["author_followers"]} followers) in '
                f'{_e(it["space"])} · <time datetime="{_iso(it["created_at"])}">{_fmt_time(it["created_at"])}</time>'
                f'{edited} · status {_e(it["status"])}{state}</p>'
                f'<div class="body" id="item-body">{_e(it["body"])}</div>'
                f'<p id="item-counts">{c["reactions"]} upvotes · {c["downs"]} downvotes · {c["views"]} views · '
                f'{c["answers"]} answers · {c["responses"]} responses</p>{react}</article>'
                f'{manage}{responses}{reply}')
        return self._layout(it["title"], main, uid)

    def _new_page(self, uid: int, space: str, error: str | None, form: dict | None = None) -> str:
        form = form or {}
        opts = "".join(f'<option value="{_e(s["name"])}"{" selected" if s["name"] == space else ""}>{_e(s["name"])}'
                       f'</option>' for s in self.lab.spaces())
        alert = (f'<div role="alert" id="new-error">{_e(error)}</div>' if error
                 else '<div role="alert" id="new-error" hidden></div>')
        main = (f'<h1>New post</h1>{alert}<form method="post" action="/new" id="new-form">'
                f'<input type="hidden" name="client_ref" id="new-client-ref" value="">'
                f'<div class="field"><label for="new-space">Space</label><select id="new-space" name="space">'
                f'{opts}</select></div>'
                f'<div class="field"><label for="new-title">Title</label>'
                f'<input id="new-title" name="title" maxlength="200" value="{_e(form.get("title", ""))}"></div>'
                f'<div class="field"><label for="new-body">Body</label>'
                f'<textarea id="new-body" name="body" rows="5">{_e(form.get("body", ""))}</textarea></div>'
                f'<button type="submit" id="publish">Publish</button>'
                f'<p id="publish-status" hidden></p></form>')
        js = _NEW_JS.replace("__DELAY__", str(int(self.publish_delay_ms)))
        return self._layout("New post", main, uid, js)

    def _search_page(self, uid: int, q: str) -> str:
        res = ""
        if q.strip():
            res = ('<section id="results" aria-busy="true" aria-live="polite" aria-labelledby="results-h">'
                   '<h2 id="results-h">Results</h2><p role="status" id="search-status">Loading results…</p></section>')
        main = (f'<h1>Search</h1><form method="get" action="/search" role="search" id="search-form">'
                f'<div class="field"><label for="search-q">Search posts</label>'
                f'<input type="search" id="search-q" name="q" value="{_e(q)}"></div>'
                f'<button type="submit" id="search-submit">Search</button></form>{res}')
        js = _SEARCH_JS.replace("__DELAY__", str(int(self.search_delay_ms))) if q.strip() else ""
        return self._layout("Search", main, uid, js)

    def _notifications_page(self, uid: int) -> str:
        lab = self.lab
        rows = []
        for n in lab.notifications(uid):
            try:
                title = lab.public_item(n["item_id"])["title"]
            except LabError:
                title = "(deleted)"
            what = {"response": "New response on", "followed_posted": "Someone you follow posted"}.get(n["kind"], n["kind"])
            rows.append(f'<li class="{"unread" if not n["read"] else "read"}">{_e(what)} '
                        f'<a href="/item/{n["item_id"]}">{_e(title)}</a> <span class="meta">'
                        f'{_fmt_time(n["created_at"])}{"" if n["read"] else " · unread"}</span></li>')
        body = f'<ol id="notifications">{"".join(rows)}</ol>' if rows else '<p id="no-notifications">No notifications.</p>'
        main = (f'<h1>Notifications</h1><form method="post" action="/notifications/read" class="inline">'
                f'<button type="submit" id="mark-read">Mark all as read</button></form>{body}')
        return self._layout("Notifications", main, uid)


# ====================================================================== module helpers
_APPS: weakref.WeakKeyDictionary[SocialWebLab, LabApp] = weakref.WeakKeyDictionary()


def app_for(lab: SocialWebLab) -> LabApp:
    """The default ``LabApp`` of a lab (created on first use)."""
    if lab not in _APPS:
        _APPS[lab] = LabApp(lab)
    return _APPS[lab]


def dispatch(lab: SocialWebLab | LabApp | SocialWebLabServer, method: str, path: str,
             query: dict | None = None, headers: dict | None = None, body: bytes | None = None) -> Response:
    """Socket-free request handling. ``lab`` may be a lab (its default ``LabApp`` is used), a
    ``LabApp`` or a ``SocialWebLabServer``."""
    return _resolve_app(lab).dispatch(method, path, query, headers, body)


def _resolve_app(x) -> LabApp:
    if isinstance(x, LabApp):
        return x
    if isinstance(x, SocialWebLabServer):
        return x.app
    if isinstance(x, SocialWebLab):
        return app_for(x)
    raise TypeError(f"cannot dispatch to {type(x).__name__}")


def _e(s: Any) -> str:
    return html.escape("" if s is None else str(s), quote=True)


def _iso(t: float | None) -> str:
    return datetime.fromtimestamp(t or 0, tz=UTC).isoformat(timespec="seconds") if t else ""


def _fmt_time(t: float | None) -> str:
    return datetime.fromtimestamp(t, tz=UTC).strftime("%Y-%m-%d %H:%M UTC") if t else ""


def _json(status: int, obj: Any) -> Response:
    return status, {"Content-Type": "application/json"}, json.dumps(obj).encode()


def _html(status: int, text: str) -> Response:
    return status, {"Content-Type": "text/html; charset=utf-8"}, text.encode()


def _redirect(location: str) -> Response:
    return 303, {"Location": location, "Content-Type": "text/plain"}, b""


def _not_found() -> Response:
    return 404, {"Content-Type": "text/html; charset=utf-8"}, b"<!doctype html><title>Not found</title><p>Not found</p>"


def _session_cookie(tok: str, clear: bool = False) -> str:
    if clear:
        return f"{SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0"
    return f"{SESSION_COOKIE}={tok}; Path=/; HttpOnly; SameSite=Lax"


def _json_body(body: bytes | None) -> dict:
    if not body:
        return {}
    try:
        obj = json.loads(body)
    except ValueError as e:
        raise LabError("request body must be JSON") from e
    if not isinstance(obj, dict):
        raise LabError("request body must be a JSON object")
    return obj


def _form_body(body: bytes | None) -> dict:
    if not body:
        return {}
    try:
        return {k: v[-1] for k, v in parse_qs(body.decode("utf-8"), keep_blank_values=True).items()}
    except UnicodeDecodeError as e:
        raise LabError("bad form encoding") from e


def _int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _id(v: Any) -> int:
    try:
        i = int(str(v))
    except (TypeError, ValueError) as e:
        raise LabError("not found", 404) from e
    if i <= 0:
        raise LabError("not found", 404)
    return i


def _safe_next(nxt: str) -> str:
    """Only same-site absolute paths (no open redirects, never the control channel)."""
    if not nxt.startswith("/") or nxt.startswith(("//", CONTROL_PREFIX)) or "\\" in nxt:
        return "/"
    return nxt


def _viewer_reaction(lab: SocialWebLab, uid: int, iid: int) -> str | None:
    r = lab._q("SELECT kind FROM reactions WHERE item_id=? AND user_id=?", (iid, uid))
    return r[0]["kind"] if r else None


# ====================================================================== page assets
_CSS = """
:root { color-scheme: light; }
body { font: 15px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0; background: #f5f6f8; color: #1d1f23; }
header { background: #fff; border-bottom: 1px solid #d5d9de; padding: 6px 16px; }
#sim-banner { background: #fff4cc; border: 1px solid #e0c060; padding: 3px 8px; font-size: 13px; border-radius: 4px; }
nav[aria-label=Main] { margin: 6px 0 0; display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }
.who { color: #555; font-size: 13px; }
.clock { margin: 4px 0 0; font-size: 12px; color: #666; }
main { max-width: 760px; margin: 0 auto; padding: 8px 16px 24px; }
h1 { font-size: 22px; margin: 8px 0; } h2 { font-size: 17px; margin: 12px 0 4px; }
.meta { color: #666; font-size: 13px; }
.feed li, .spaces li { margin: 3px 0; }
.body { white-space: pre-wrap; background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 8px; }
.field { margin: 6px 0; } .field label { display: block; font-size: 13px; color: #444; }
.field input, .field select, .field textarea { font: inherit; padding: 4px 6px; min-width: 320px; }
form.inline { display: inline; }
button { font: inherit; padding: 3px 12px; border: 1px solid #9aa3ad; border-radius: 6px; background: #fff; cursor: pointer; }
button:disabled { opacity: .55; }
[role=alert] { color: #a3160e; background: #fde8e6; padding: 6px 10px; border-radius: 6px; }
[role=status] { color: #555; font-style: italic; }
.response { background: #fff; border: 1px solid #e3e3e3; border-radius: 6px; padding: 4px 8px; margin: 4px 0; }
.rbody { margin: 2px 0; white-space: pre-wrap; }
"""

# Unread badge: polls /api/notifications; backs off on errors / 429 (the virtual clock may not move).
_POLL_JS = r"""
(() => {
  const badge = document.getElementById('unread-badge');
  if (!badge) return;
  let delay = __POLL_MS__;
  const tick = async () => {
    try {
      const r = await fetch('/api/notifications', {headers: {'Accept': 'application/json'}});
      if (r.ok) { const d = await r.json(); badge.textContent = '(' + d.unread + ')'; delay = __POLL_MS__; }
      else { delay = Math.min(delay * 2, 60000); }
    } catch (e) { delay = Math.min(delay * 2, 60000); }
    setTimeout(tick, delay);
  };
  setTimeout(tick, delay);
})();
"""

_NEW_JS = r"""
(() => {
  const form = document.getElementById('new-form');
  const btn = document.getElementById('publish');
  const status = document.getElementById('publish-status');
  const alertBox = document.getElementById('new-error');
  const ref = document.getElementById('new-client-ref');
  ref.value = (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Math.random()).slice(2);
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    if (btn.disabled) return;
    btn.disabled = true;
    alertBox.hidden = true; alertBox.textContent = '';
    status.hidden = false; status.setAttribute('role', 'status'); status.textContent = 'Publishing…';
    form.setAttribute('aria-busy', 'true');
    const payload = {space: form.space.value, title: form.title.value, body: form.body.value, client_ref: ref.value};
    let res, data;
    try {
      res = await fetch('/api/items', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                       body: JSON.stringify(payload)});
      data = await res.json();
    } catch (e) { res = null; data = {error: 'network error, please retry'}; }
    await sleep(__DELAY__);
    if (res && res.ok && data.id) { window.location.assign('/item/' + data.id); return; }
    status.hidden = true; status.removeAttribute('role'); status.textContent = '';
    form.removeAttribute('aria-busy');
    alertBox.textContent = 'Could not publish: ' + ((data && data.error) || 'unknown error');
    alertBox.hidden = false;
    btn.disabled = false;
  });
})();
"""

_SEARCH_JS = r"""
(() => {
  const box = document.getElementById('results');
  const status = document.getElementById('search-status');
  const q = new URLSearchParams(location.search).get('q') || '';
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  (async () => {
    await sleep(__DELAY__);
    let items = null, err = null;
    try {
      const r = await fetch('/api/search?q=' + encodeURIComponent(q));
      const d = await r.json();
      if (r.ok) items = d.items; else err = d.error || ('HTTP ' + r.status);
    } catch (e) { err = 'network error'; }
    status.remove();
    if (err !== null) {
      const a = document.createElement('div'); a.setAttribute('role', 'alert'); a.id = 'search-error';
      a.textContent = 'Search failed: ' + err; box.appendChild(a);
    } else {
      const p = document.createElement('p'); p.id = 'search-summary';
      p.textContent = items.length + (items.length === 1 ? ' result' : ' results') + ' for "' + q + '"';
      box.appendChild(p);
      if (items.length) {
        const ol = document.createElement('ol'); ol.id = 'result-list';
        for (const it of items) {
          const li = document.createElement('li');
          const a = document.createElement('a'); a.href = '/item/' + it.id; a.textContent = it.title;
          const m = document.createElement('span'); m.className = 'meta';
          m.textContent = ' in ' + it.space + ' · by ' + it.author + ' · ' + it.responses + ' responses';
          li.appendChild(a); li.appendChild(m); ol.appendChild(li);
        }
        box.appendChild(ol);
      }
    }
    box.setAttribute('aria-busy', 'false');
  })();
})();
"""


# ====================================================================== HTTP server
class _Handler(BaseHTTPRequestHandler):
    app: LabApp  # bound in SocialWebLabServer.start
    server_version = "SocialWebLab/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep test output quiet
        pass

    def _handle(self, method: str) -> None:
        parts = urlsplit(self.path)
        query = {k: v[-1] for k, v in parse_qs(parts.query, keep_blank_values=True).items()}
        headers = {k.lower(): v for k, v in self.headers.items()}
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            status, hdrs, out = _json(413, {"error": "request body too large"})
            self.close_connection = True
        else:
            body = self.rfile.read(n) if n else None
            status, hdrs, out = self.app.dispatch(method, unquote(parts.path), query, headers, body)
        self.send_response(status)
        for k, v in hdrs.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(out)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if method != "HEAD":
            self.wfile.write(out)

    def do_GET(self):
        self._handle("GET")

    def do_HEAD(self):
        self._handle("HEAD")

    def do_POST(self):
        self._handle("POST")


class SocialWebLabServer:
    """The lab's web application on a real socket (``ThreadingHTTPServer``)."""

    def __init__(self, lab: SocialWebLab, host: str = "127.0.0.1", port: int = 0, **app_kw):
        self.lab = lab
        self.host = host
        self.port = port
        self.app = LabApp(lab, **app_kw)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def control_token(self) -> str:
        return self.app.control_token

    @property
    def running(self) -> bool:
        return self._httpd is not None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> SocialWebLabServer:
        if self._httpd is not None:
            return self
        handler = type("BoundHandler", (_Handler,), {"app": self.app})
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="socialweblab-http", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def dispatch(self, method: str, path: str, query: dict | None = None, headers: dict | None = None,
                 body: bytes | None = None) -> Response:
        return self.app.dispatch(method, path, query, headers, body)

    def control(self, op: str, **payload) -> dict:
        """Call a control op in-process (the same code the control route runs)."""
        return self.app.control(op, payload)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


# ====================================================================== clients
class _TransportBase:
    def __init__(self, api_key: str | None = None, session_token: str | None = None):
        self.api_key = api_key
        self.session_token = session_token

    def _headers(self, json_body: bool = False) -> dict:
        h = {"Accept": "application/json"}
        if json_body:
            h["Content-Type"] = "application/json"
        if self.api_key:
            h["X-Api-Key"] = self.api_key
        if self.session_token:
            h["Authorization"] = f"Bearer {self.session_token}"
        return h

    @staticmethod
    def _decode(raw: bytes) -> Any:
        try:
            return json.loads(raw) if raw else {}
        except ValueError:
            return {"error": "non-JSON response", "body": raw[:200].decode("utf-8", "replace")}


class InProcessTransport(_TransportBase):
    """Calls ``dispatch`` directly (no sockets): same routes, auth and rate limits as HTTP."""

    def __init__(self, target: SocialWebLabServer | LabApp | SocialWebLab, api_key: str | None = None,
                 session_token: str | None = None):
        super().__init__(api_key, session_token)
        self.app = _resolve_app(target)

    def get(self, path: str, **params) -> tuple[int, Any]:
        q = {k: str(v) for k, v in params.items() if v is not None}
        status, _, body = self.app.dispatch("GET", path, q, self._headers(), None)
        return status, self._decode(body)

    def post(self, path: str, json_body: dict | None = None) -> tuple[int, Any]:
        status, _, body = self.app.dispatch("POST", path, {}, self._headers(True), json.dumps(json_body or {}).encode())
        return status, self._decode(body)


class HttpTransport(_TransportBase):
    """Real HTTP client (stdlib; never goes through HTTP(S)_PROXY)."""

    def __init__(self, base_url: str, api_key: str | None = None, session_token: str | None = None,
                 timeout: float = 10.0):
        super().__init__(api_key, session_token)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _call(self, req: urllib.request.Request) -> tuple[int, Any]:
        try:
            with self._opener.open(req, timeout=self.timeout) as r:
                return r.status, self._decode(r.read())
        except urllib.error.HTTPError as e:
            return e.code, self._decode(e.read())

    def get(self, path: str, **params) -> tuple[int, Any]:
        q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = self.base_url + path + (("?" + q) if q else "")
        return self._call(urllib.request.Request(url, headers=self._headers(), method="GET"))

    def post(self, path: str, json_body: dict | None = None) -> tuple[int, Any]:
        return self._call(urllib.request.Request(self.base_url + path, data=json.dumps(json_body or {}).encode(),
                                                 headers=self._headers(True), method="POST"))


class LabHTTPError(RuntimeError):
    def __init__(self, status: int, payload: Any):
        msg = payload.get("error") if isinstance(payload, dict) else str(payload)
        super().__init__(f"HTTP {status}: {msg}")
        self.status = status
        self.payload = payload
        self.retry_after = payload.get("retry_after") if isinstance(payload, dict) else None


@dataclass
class LabClient:
    """Convenience JSON-API client over either transport. Non-2xx responses raise ``LabHTTPError``."""

    transport: Any
    last_status: int = field(default=0, init=False)

    def _ok(self, res: tuple[int, Any]) -> Any:
        status, data = res
        self.last_status = status
        if not 200 <= status < 300:
            raise LabHTTPError(status, data)
        return data

    def events(self, since: int = 0, limit: int = 200) -> list[dict]:
        return self._ok(self.transport.get("/api/events", since=since, limit=limit))["events"]

    def item(self, item_id: int) -> dict:
        return self._ok(self.transport.get(f"/api/items/{int(item_id)}"))

    def user(self, handle: str) -> dict:
        return self._ok(self.transport.get(f"/api/users/{quote(handle, safe='')}"))

    def feed(self, space: str | None = None, sort: str = "hot", limit: int = 20) -> list[dict]:
        return self._ok(self.transport.get("/api/feed", space=space, sort=sort, limit=limit))["items"]

    def search(self, q: str, space: str | None = None) -> list[dict]:
        return self._ok(self.transport.get("/api/search", q=q, space=space))["items"]

    def spaces(self) -> list[dict]:
        return self._ok(self.transport.get("/api/spaces"))["spaces"]

    def login(self, handle: str, password: str) -> str:
        tok = self._ok(self.transport.post("/api/login", {"handle": handle, "password": password}))["token"]
        self.transport.session_token = tok
        return tok

    def logout(self) -> None:
        self._ok(self.transport.post("/api/logout", {}))
        self.transport.session_token = None

    def create_item(self, space: str, title: str, body: str, client_ref: str | None = None) -> int:
        payload = {"space": space, "title": title, "body": body}
        if client_ref:
            payload["client_ref"] = client_ref
        return self._ok(self.transport.post("/api/items", payload))["id"]

    def respond(self, item: int, body: str, kind: str = "comment") -> int:
        return self._ok(self.transport.post(f"/api/items/{int(item)}/responses", {"body": body, "kind": kind}))["id"]

    def react(self, item: int, kind: str) -> bool:
        return self._ok(self.transport.post(f"/api/items/{int(item)}/react", {"kind": kind}))["new"]

    def edit(self, item: int, body: str, title: str | None = None) -> int:
        payload = {"body": body} | ({"title": title} if title else {})
        return self._ok(self.transport.post(f"/api/items/{int(item)}/edit", payload))["version"]

    def delete(self, item: int) -> None:
        self._ok(self.transport.post(f"/api/items/{int(item)}/delete", {}))

    def accept(self, item: int, response_id: int) -> None:
        self._ok(self.transport.post(f"/api/items/{int(item)}/accept", {"response_id": int(response_id)}))

    def notifications(self) -> dict:
        return self._ok(self.transport.get("/api/notifications"))

    def mark_read(self) -> int:
        return self._ok(self.transport.post("/api/notifications/read", {}))["marked"]


__all__ = [
    "SESSION_COOKIE",
    "HttpTransport",
    "InProcessTransport",
    "LabApp",
    "LabClient",
    "LabHTTPError",
    "SocialWebLabServer",
    "TokenBucket",
    "app_for",
    "dispatch",
]
