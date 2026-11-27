"""Shared helpers for the G-env verification probes (workstream G, lens "env").

Nothing in here stands in for the code under test: the ComputerEnvironment, BrowserExecutor (real Chromium),
Perception, InternalExecutor, ExecRPC service, PolicyService (workstream EA), HttpApiExecutor, PublishingExecutor,
SkillExecutor and ModalitySelector are the real ones. ``HostileServer`` is a real local HTTP server (127.0.0.1)
that serves the adversarial pages the probes need; every request it receives is counted, so a probe can show
whether a request really left the browser.

Run every probe from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/env/<probe>.py
Scratch run directories: $G_ENV_SCRATCH (default: a fresh temporary directory).
"""
import base64
import collections
import hashlib
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qsl, urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
CAM = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))            # .../cam_live
for p in (os.path.join(CAM, "src"), os.path.join(CAM, "tests", "harness", "env"),
          os.path.join(CAM, "tests", "harness", "skills")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cam_live import contracts as C  # noqa: E402

RESULTS = []


def scratch(name: str) -> str:
    base = os.environ.get("G_ENV_SCRATCH") or tempfile.mkdtemp(prefix="g_env_")
    d = os.path.join(base, f"{name}_{os.getpid()}_{int(time.time() * 1000) % 100000}")
    os.makedirs(d, exist_ok=True)
    return d


def report(pid: str, verdict: str, text: str, **facts):
    """verdict: HOLDS (property confirmed) | DEFECT (attack succeeded / claim refuted) | NOTE."""
    RESULTS.append({"id": pid, "verdict": verdict, "text": text, "facts": facts})
    extra = ("  " + json.dumps(facts, sort_keys=True, default=str)[:900]) if facts else ""
    print(f"[{verdict:6s}] {pid}: {text}{extra}", flush=True)


def summary(name: str) -> int:
    n = collections.Counter(r["verdict"] for r in RESULTS)
    print(f"\n== {name}: {dict(n)} ==")
    out = os.environ.get("G_ENV_JSON")
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, name + ".json"), "w") as f:
            json.dump(RESULTS, f, indent=1, default=str)
    return 0


def action(op: str, modality: str, level: str = "primitive", aid: str = "", timeout: float = 60.0, verify=None,
           steps=None, **args) -> C.ActionSpec:
    return C.ActionSpec(action_id=aid or C.new_id("gact"), operation=op, modality=modality, level=level,
                        arguments=args, timeout_s=timeout, verify=verify or C.VerificationCondition(),
                        steps=list(steps or []))


def modality_of(op: str) -> str:
    if op.startswith("browser."):
        return C.Modality.STRUCTURED_BROWSER
    if op.startswith(("http.", "publish.", "discord.")):
        return C.Modality.API
    if op.startswith("skill."):
        return C.Modality.SKILL
    return C.Modality.INTERNAL


class Drv:
    """authorize -> execute, exactly what the runtime loop does."""

    def __init__(self, env, policy, task):
        self.env, self.policy, self.task = env, policy, task
        self.last_decision = None

    def run(self, op: str, aid: str = "", level: str = "primitive", timeout: float = 60.0, verify=None, **args):
        a = action(op, modality_of(op), level=level, aid=aid, timeout=timeout, verify=verify, **args)
        return self.submit(a)

    def submit(self, a):
        self.last_decision = dec = self.policy.authorize(a, self.task, {})
        res = self.env.execute(a, dec)
        res._action = a          # type: ignore[attr-defined]
        return res


def brief(res) -> str:
    return f"{res.status}/{res.failure_class} {res.failure_detail[:160]!r}"


# ----------------------------------------------------------------------------------------------------------------
# A real local HTTP server for adversarial content
# ----------------------------------------------------------------------------------------------------------------
class HostileServer:
    """``routes[path] = fn(method, path, query, headers, body) -> (status, headers, body)``; counts every hit.
    Also accepts WebSocket upgrades on any path (counted in ``ws``) and echoes nothing back."""

    def __init__(self, name: str = "hostile"):
        self.name, self.routes = name, {}
        self.hits = collections.Counter()
        self.log = []                      # (method, path, headers)
        self.ws = []                       # paths of accepted websocket handshakes
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
                    try:
                        self.connection.settimeout(2.0)
                        data = self.connection.recv(4096)       # first frame the page sends (masked)
                        if data:
                            outer.ws.append(f"frame:{len(data)}bytes")
                    except OSError:
                        pass
                    self.close_connection = True
                    return
                fn = outer.routes.get(u.path)
                if fn is None:
                    for pre, f in outer.routes.items():
                        if pre.endswith("*") and u.path.startswith(pre[:-1]):
                            fn = f
                            break
                if fn is None:
                    status, headers, out = 404, {}, b"not found"
                else:
                    status, headers, out = fn(self.command, u.path, dict(parse_qsl(u.query)), dict(self.headers), body)
                if isinstance(out, str):
                    out = out.encode()
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

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.host = f"127.0.0.1:{self.port}"
        self.url = f"http://{self.host}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def page(self, path: str, html: str, status: int = 200, headers=None):
        self.routes[path] = lambda *a: (status, dict(headers or {}), html)

    def count(self, path: str, method: str = "") -> int:
        return sum(n for (m, p), n in self.hits.items() if p == path and (not method or m == method))

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def html(title: str, body: str, head: str = "") -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>{head}</head><body>{body}</body></html>"


def staging_task(objective_id: str, **resources) -> C.TaskSpec:
    return C.TaskSpec(objective_id=objective_id, objective="G-env verification probe (no strategy)", mode="staging",
                      resources=dict(resources),
                      limits=C.Limits(duration_s=3600, money=1, max_decisions=5000, max_external_actions=5000))


def real_policy(run_dir: str, task: C.TaskSpec):
    """The real authorization service of workstream EA, in process, with the default policy."""
    from cam_live.harness.auth import PolicyService
    svc = PolicyService(run_dir, fsync=False)
    svc.register_task(task, "operator")
    return svc


def trust_labels(res) -> set:
    return {o.provenance.trust for o in res.observations}
