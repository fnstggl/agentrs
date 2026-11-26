"""Test support for workstream ES: a small real HTTP application, and a scope-based policy engine.

The application is a genuine HTTP server on a loopback port; the executors under test talk to it over real sockets.
The policy engine is a minimal ``contracts.PolicyEngine`` (authorization is another workstream's deliverable; this
one exists only so that the ES executors can be exercised against allow / deny / needs-approval verdicts).
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlsplit

from cam_live.contracts import (ActionSpec, AuthorizationDecision, AuthVerdict, ExecContext, Modality, PrimitiveOp,
                                WallClock, new_id)


# --------------------------------------------------------------------------------------------------------------
# A small content application with two interface generations
# --------------------------------------------------------------------------------------------------------------
class NotesApp:
    """Items are created as drafts and then published. ``generation`` selects the interface:

    * 1: POST /api/items {title, body} -> {id, title, status}; POST /api/items/<id>/publish; GET /api/items
    * 2: POST /api/v2/entries {name, text} -> {entry_id, ...}; POST /api/v2/entries/<id>/state {state};
         GET /api/v2/entries. The generation-1 paths answer 404.

    ``publish_is_noop`` makes generation 1's publish endpoint answer 200 without publishing (a silent change).
    """

    TOKEN = "s3cr3t-app-token-0123456789"

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.items: List[Dict[str, Any]] = []
        self.generation = 1
        self.publish_is_noop = False
        self.rename_id_field = False
        self.flaky_left = 0
        self.limited_left = 0
        self.retry_after = "0.05"
        self.requests: List[Dict[str, Any]] = []
        self.latency_s = 0.0
        self.other_origin = ""
        self._next = 1

    # ---- state ---------------------------------------------------------------------------------------------
    def _listing(self) -> Dict[str, Any]:
        if self.generation == 1:
            return {"items": [dict(i) for i in self.items], "total": len(self.items),
                    "published": sum(1 for i in self.items if i["status"] == "published")}
        return {"entries": [{"entry_id": i["id"], "name": i["title"], "text": i["body"], "state": i["status"]}
                            for i in self.items],
                "count": len(self.items), "live": sum(1 for i in self.items if i["status"] == "published")}

    def handle(self, method: str, url: str, headers: Dict[str, str], body: bytes):
        u = urlsplit(url)
        path, query = u.path.rstrip("/") or "/", parse_qs(u.query)
        try:
            payload = json.loads(body.decode() or "null") if body else None
        except ValueError:
            return 400, {}, {"error": "malformed json"}
        with self.lock:
            self.requests.append({"method": method, "path": path, "headers": dict(headers), "body": payload})
            if self.latency_s:
                time.sleep(self.latency_s)
            # ---- generic behaviours for the executor tests -------------------------------------------------
            if path == "/api/private":
                if headers.get("authorization") != f"Bearer {self.TOKEN}":
                    return 401, {}, {"error": "unauthorized"}
                return 200, {}, {"secret_area": True, "echo": headers.get("authorization"), "visits": 12}
            if path == "/api/keyed":
                if headers.get("x-service-key") != self.TOKEN and query.get("key", [""])[0] != self.TOKEN:
                    return 403, {}, {"error": "forbidden"}
                return 200, {}, {"ok": True}
            if path == "/api/flaky":
                if self.flaky_left > 0:
                    self.flaky_left -= 1
                    return 503, {}, {"error": "temporarily unavailable"}
                return 200, {}, {"ok": True, "value": 7}
            if path == "/api/limited":
                if self.limited_left > 0:
                    self.limited_left -= 1
                    return 429, {"Retry-After": self.retry_after, "X-RateLimit-Limit": "5",
                                 "X-RateLimit-Remaining": "0", "X-RateLimit-Reset-After": self.retry_after}, \
                        {"error": "rate limited"}
                return 200, {"X-RateLimit-Limit": "5", "X-RateLimit-Remaining": "4",
                             "X-RateLimit-Reset-After": "1"}, {"ok": True}
            if path == "/api/exhausting":
                return 200, {"X-RateLimit-Limit": "1", "X-RateLimit-Remaining": "0",
                             "X-RateLimit-Reset-After": self.retry_after}, {"ok": True}
            if path == "/api/slow":
                time.sleep(0.3)
                return 200, {}, {"ok": True}
            if path == "/api/redirect":
                return 302, {"Location": "/api/items"}, {}
            if path == "/api/redirect_out":
                return 307, {"Location": self.other_origin + "/api/echo"}, {}
            if path == "/api/echo":
                return 200, {}, {"headers": {k: v for k, v in headers.items()}, "method": method}
            if path == "/api/forbidden":
                return 403, {}, {"error": "forbidden"}
            if path == "/api/legal":
                return 451, {}, {"error": "unavailable for legal reasons"}
            if path == "/api/text":
                return 200, {"Content-Type": "text/plain"}, "plain words, 3 of them"
            # ---- the content application -------------------------------------------------------------------
            if self.generation == 1:
                return self._gen1(method, path, payload)
            return self._gen2(method, path, payload)

    def _gen1(self, method, path, payload):
        parts = path.strip("/").split("/")
        if parts[:2] != ["api", "items"]:
            return 404, {}, {"error": "not found"}
        if len(parts) == 2 and method == "GET":
            return 200, {}, self._listing()
        if len(parts) == 2 and method == "POST":
            if not isinstance(payload, dict) or not isinstance(payload.get("title"), str) or not payload["title"]:
                return 422, {}, {"error": "title is required"}
            item = {"id": self._next, "title": payload["title"], "body": payload.get("body", ""), "status": "draft"}
            self._next += 1
            self.items.append(item)
            out = dict(item)
            if self.rename_id_field:
                out["item_id"] = out.pop("id")
            return 201, {}, out
        item = next((i for i in self.items if str(i["id"]) == parts[2]), None) if len(parts) >= 3 else None
        if item is None:
            return 404, {}, {"error": "no such item"}
        if len(parts) == 3 and method == "GET":
            return 200, {}, dict(item)
        if len(parts) == 4 and parts[3] == "publish" and method == "POST":
            if not self.publish_is_noop:
                item["status"] = "published"
            return 200, {}, {"id": item["id"], "status": "published"}
        return 405, {}, {"error": "method not allowed"}

    def _gen2(self, method, path, payload):
        parts = path.strip("/").split("/")
        if parts[:3] != ["api", "v2", "entries"]:
            return 404, {}, {"error": "not found"}
        if len(parts) == 3 and method == "GET":
            return 200, {}, self._listing()
        if len(parts) == 3 and method == "POST":
            if not isinstance(payload, dict) or not isinstance(payload.get("name"), str) or not payload["name"]:
                return 422, {}, {"error": "name is required"}
            item = {"id": self._next, "title": payload["name"], "body": payload.get("text", ""), "status": "draft"}
            self._next += 1
            self.items.append(item)
            return 201, {}, {"entry_id": item["id"], "name": item["title"], "state": "draft"}
        item = next((i for i in self.items if str(i["id"]) == parts[3]), None) if len(parts) >= 4 else None
        if item is None:
            return 404, {}, {"error": "no such entry"}
        if len(parts) == 5 and parts[4] == "state" and method == "POST":
            if not isinstance(payload, dict) or payload.get("state") not in ("draft", "published"):
                return 422, {}, {"error": "state must be draft or published"}
            item["status"] = payload["state"]
            return 200, {}, {"entry_id": item["id"], "state": item["status"]}
        return 405, {}, {"error": "method not allowed"}


class AppServer:
    """Serves a ``handle(method, url, headers, body) -> (status, headers, json-or-text)`` object over HTTP."""

    def __init__(self, app: Any):
        self.app = app
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            disable_nagle_algorithm = True       # headers and body are separate writes; no 40 ms delayed-ACK stall

            def _serve(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                status, headers, payload = outer.app.handle(self.command, self.path,
                                                            {k.lower(): v for k, v in self.headers.items()}, body)
                raw = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
                self.send_response(status)
                hs = {"Content-Type": "application/json"}
                hs.update(headers)
                for k, v in hs.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                try:
                    if self.command != "HEAD":
                        self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    return                                  # the client gave up (timeout tests)

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = _serve

            def log_message(self, *a):
                return

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


# --------------------------------------------------------------------------------------------------------------
# Policy engine for tests
# --------------------------------------------------------------------------------------------------------------
class ScopedPolicy:
    """A small ``PolicyEngine``: a primitive operation is allowed when its action's scope is granted and no deny
    rule matches; ``approval_rule`` sends matching operations to an approval queue keyed by the operation's
    content, and a resolved approval answers later checks of the same operation."""

    def __init__(self, scopes=("http.read", "http.write"), deny: Optional[Callable[[PrimitiveOp, ActionSpec], str]] = None,
                 approval_rule: Optional[Callable[[PrimitiveOp, ActionSpec], bool]] = None):
        self.scopes = set(scopes)
        self.deny, self.approval_rule = deny, approval_rule
        self.checks: List[PrimitiveOp] = []
        self.queue: Dict[str, Dict[str, Any]] = {}
        self._stopped = False

    @staticmethod
    def op_key(op: PrimitiveOp) -> str:
        args = {k: v for k, v in op.arguments.items() if k not in ("hop",)}
        return hashlib.sha256(json.dumps([op.op, args], sort_keys=True, default=str).encode()).hexdigest()[:24]

    def check_primitive(self, op: PrimitiveOp, action: ActionSpec) -> AuthorizationDecision:
        self.checks.append(op)
        if self._stopped:
            return AuthorizationDecision(AuthVerdict.DENY, reasons=["emergency stop"])
        if action.auth_scope and not set(action.auth_scope.split()) <= self.scopes:
            return AuthorizationDecision(AuthVerdict.DENY, reasons=[f"scope {action.auth_scope!r} is not granted"],
                                         policy_ids=["scope"])
        why = self.deny(op, action) if self.deny else ""
        if why:
            return AuthorizationDecision(AuthVerdict.DENY, reasons=[why],
                                         policy_ids=["platform.rule" if why.startswith("platform") else "destination"])
        if self.approval_rule and self.approval_rule(op, action):
            key = self.op_key(op)
            entry = self.queue.setdefault(key, {"approval_id": new_id("apr"), "state": "pending", "op": op.to_dict(),
                                                "action_id": action.action_id})
            if entry["state"] == "approved":
                return AuthorizationDecision(AuthVerdict.ALLOW, approval_id=entry["approval_id"], grant="approved")
            if entry["state"] == "rejected":
                return AuthorizationDecision(AuthVerdict.DENY, reasons=["rejected by the operator"],
                                             approval_id=entry["approval_id"], policy_ids=["approval"])
            return AuthorizationDecision(AuthVerdict.NEEDS_APPROVAL, reasons=["operator approval required"],
                                         approval_id=entry["approval_id"])
        return AuthorizationDecision(AuthVerdict.ALLOW)

    # the rest of the PolicyEngine protocol
    def authorize(self, action, task, context=None):
        return AuthorizationDecision(AuthVerdict.ALLOW, action.content_hash(), grant="test")

    def check_grant(self, action, decision):
        return decision.verdict == AuthVerdict.ALLOW

    def record(self, action, result):
        return None

    def approvals(self):
        return [dict(v) for v in self.queue.values() if v["state"] == "pending"]

    def resolve_approval(self, approval_id, approve, by, note=""):
        for v in self.queue.values():
            if v["approval_id"] == approval_id:
                v["state"], v["by"], v["note"] = ("approved" if approve else "rejected"), by, note
                return AuthorizationDecision(AuthVerdict.ALLOW if approve else AuthVerdict.DENY, approval_id=approval_id)
        return AuthorizationDecision(AuthVerdict.DENY, reasons=["unknown approval"])

    def emergency_stop(self, reason, by):
        self._stopped = True

    def stopped(self):
        return self._stopped

    def revoke(self, scope, by):
        self.scopes.discard(scope)

    def limits_state(self):
        return {}


def make_ctx(policy: Any, secrets: Any = None, clock: Any = None, run_dir: str = "") -> ExecContext:
    return ExecContext(clock=clock or WallClock(), check=policy.check_primitive, obs_sink=lambda o: None,
                       run_dir=run_dir, secrets=secrets)


def act(operation: str, arguments: Dict[str, Any], scope: str = "", modality: str = Modality.API, **kw) -> ActionSpec:
    if not scope:
        scope = "http.read" if operation == "http.get" else "http.write" if operation.startswith("http.") else ""
    return ActionSpec(action_id=new_id("act"), operation=operation, modality=modality, arguments=arguments,
                      auth_scope=scope, **kw)


# --------------------------------------------------------------------------------------------------------------
# Workflows on the notes application, executed through the api executor (they are what skills are induced from)
# --------------------------------------------------------------------------------------------------------------
def listing_probe(url: str, generation: int = 1) -> ActionSpec:
    return act("http.get", {"url": url + ("/api/items" if generation == 1 else "/api/v2/entries")})


def create_and_publish(http, ctx, url: str, title: str, body: str, generation: int = 1,
                       counter: Optional[List[int]] = None, observe_before: bool = True):
    """The primitive-only way: create an item, publish it, look at the listing. Returns (trace pairs, before
    observations, after observations). ``counter`` (a one-element list) counts the actions issued."""
    from cam_live.contracts import ResultStatus

    def run(a):
        if counter is not None:
            counter[0] += 1
        r = http.execute(a, ctx)
        assert r.status == ResultStatus.OK, (r.failure_class, r.failure_detail)
        return r

    # the state before (used to derive conditions when a skill is induced); not part of the task itself
    before_obs = http.execute(listing_probe(url, generation), ctx).observations if observe_before else []
    if generation == 1:
        a1 = act("http.request", {"method": "POST", "url": url + "/api/items", "json": {"title": title, "body": body}})
        r1 = run(a1)
        item = r1.observations[0].data["json"]["id"]
        a2 = act("http.request", {"method": "POST", "url": f"{url}/api/items/{item}/publish"})
    else:
        a1 = act("http.request", {"method": "POST", "url": url + "/api/v2/entries", "json": {"name": title, "text": body}})
        r1 = run(a1)
        item = r1.observations[0].data["json"]["entry_id"]
        a2 = act("http.request", {"method": "POST", "url": f"{url}/api/v2/entries/{item}/state",
                                  "json": {"state": "published"}})
    r2 = run(a2)
    after = run(listing_probe(url, generation))
    return [(a1, r1), (a2, r2)], before_obs, after.observations
