"""Helpers for the E-fix regression tests (computer environment). Nothing here stands in for the code under test:
``CountingServer`` is a real HTTP server on 127.0.0.1 that counts every request it receives (also WebSocket
handshakes), so a test can show whether a request really left the browser or the executor."""
import base64
import collections
import hashlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from cam_live.contracts import (ActionResult, AuthorizationDecision, AuthVerdict, Cost, Level, Modality,  # noqa: E402
                                Observation, PrimitiveOp, Provenance, Quantity, ResultStatus, ToolDescriptor, Trust,
                                new_id)

from eb_support import GrantPolicy  # noqa: E402


class CountingServer:
    """``routes[path] = fn(method, path, query, headers, body) -> (status, headers, body)``; ``page(path, html)``."""

    def __init__(self):
        self.routes, self.hits, self.log, self.ws = {}, collections.Counter(), [], []
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _go(self):
                u = urlsplit(self.path)
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                outer.hits[(self.command, u.path)] += 1
                outer.log.append((self.command, self.path, dict(self.headers), body))
                if self.headers.get("Upgrade", "").lower() == "websocket":
                    key = self.headers.get("Sec-WebSocket-Key", "")
                    acc = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest())
                    self.send_response(101)
                    self.send_header("Upgrade", "websocket")
                    self.send_header("Connection", "Upgrade")
                    self.send_header("Sec-WebSocket-Accept", acc.decode())
                    self.end_headers()
                    outer.ws.append(u.path)
                    self.close_connection = True
                    return
                fn = outer.routes.get(u.path)
                status, headers, out = (404, {}, b"not found") if fn is None else \
                    fn(self.command, u.path, dict(parse_qsl(u.query)), dict(self.headers), body)
                out = out.encode() if isinstance(out, str) else out
                self.send_response(status)
                headers = dict(headers)
                headers.setdefault("Content-Type", "text/html; charset=utf-8")
                headers["Content-Length"] = str(len(out))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(out)

            do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = do_PATCH = _go

            def log_message(self, *a):
                return

            def handle(self):
                try:
                    super().handle()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.host = f"127.0.0.1:{self.port}"
        self.url = f"http://{self.host}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def page(self, path, html, status=200, headers=None):
        self.routes[path] = lambda *a: (status, dict(headers or {}), html)

    def count(self, path, method=""):
        return sum(n for (m, p), n in self.hits.items() if p == path and (not method or m == method))

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def html(title, body, head=""):
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>{head}</head><body>{body}</body></html>"


class SlowFirstGrantCheck(GrantPolicy):
    """GrantPolicy whose FIRST check_grant call takes ``delay`` seconds (a slow AuthRPC round trip) and, with
    ``first_ok=False``, refuses (a forged grant)."""

    def __init__(self, delay=0.8, first_ok=True, **kw):
        super().__init__(**kw)
        self.delay, self.first_ok, self.n_checks = delay, first_ok, 0

    def check_grant(self, action, decision):
        with self._lock:
            self.n_checks += 1
            first = self.n_checks == 1
        if first:
            time.sleep(self.delay)
            if not self.first_ok:
                return False
        return super().check_grant(action, decision)


class FourArgBroker:
    """Shaped like the authorization service's ``ExecutorSecrets``: ``get(name, action, decision, destination)``
    releases a value only for a name it knows (without the ``secret:`` prefix, like the vault); ``scrub`` knows the
    exact value only."""

    def __init__(self, values):
        self.values, self.asked = dict(values), []

    def get(self, name, action, decision, destination):
        self.asked.append((name, destination))
        if name not in self.values:
            raise PermissionError(f"no secret named {name!r}")
        return self.values[name]

    def scrub(self, text):
        for v in self.values.values():
            text = text.replace(v, "[secret]")
        return text


class ProbeExecutor:
    """A real ``ModalityExecutor`` (modality ``code``) whose results are configured by the test: it returns one
    observation built by ``make(action, ctx)``; ``block`` makes it wait on an event; every call is counted."""
    modality = Modality.CODE

    def __init__(self, make=None, side="external_write"):
        self.make, self.side, self.calls = make, side, []
        self.release = threading.Event()
        self.release.set()

    def tools(self):
        schema = {"type": "object", "properties": {"n": {"type": "integer"}, "secret": {"type": "string"}},
                  "additionalProperties": False}
        return [ToolDescriptor("probe.run", Modality.CODE, "test executor", schema, Level.PRIMITIVE, "probe",
                               self.side, Cost(seconds=0.01))]

    def validate(self, action):
        return []

    def execute(self, action, ctx):
        now = ctx.clock.now()
        self.calls.append(action.action_id)
        op = PrimitiveOp("probe.run", self.modality, dict(action.arguments), started=now, ended=now)
        dec = ctx.check(op, action)
        res = ActionResult(action.action_id, ResultStatus.OK, started=now, ended=now, ops=[op])
        if dec.verdict != AuthVerdict.ALLOW:
            res.status, res.failure_detail = ResultStatus.DENIED, "; ".join(dec.reasons)
            return res
        self.release.wait(30)
        if self.make is not None:
            res.observations.append(self.make(action, ctx))
        return res

    def close(self):
        return None


def obs(text="", data=None, quantities=None):
    now = time.time()
    return Observation(obs_id=new_id("obs"), ts=now, kind="tool_result", text=text, quantities=list(quantities or []),
                       provenance=Provenance(source="probe", retrieved_at=now, trust=Trust.UNTRUSTED),
                       data=dict(data or {}))


def allow(policy, action):
    d = policy.authorize(action, None, {})
    assert d.verdict == AuthVerdict.ALLOW, d
    return d


def strict_json(o):
    return json.loads(json.dumps(o.to_dict(), allow_nan=False, default=str))
