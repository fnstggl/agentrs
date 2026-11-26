"""cam_live.harness.staging.server — tiny threaded HTTP infrastructure for the local staging sites (workstream EB).

Standard library only. Each site is a :class:`Site` served on its own localhost port (its own origin), so tests can
exercise per-destination policy. Nothing here is a model of any real platform; the content is abstract and neutral.
"""
from __future__ import annotations

import collections
import email
import email.policy
import html as _html
import json
import threading
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

esc = _html.escape


@dataclass
class Request:
    method: str
    path: str
    query: Dict[str, str]
    form: Dict[str, str] = field(default_factory=dict)
    files: Dict[str, Tuple[str, bytes]] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    cookies: Dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode() or "null")
        except ValueError:
            return None


@dataclass
class Response:
    status: int = 200
    body: bytes = b""
    ctype: str = "text/html; charset=utf-8"
    headers: List[Tuple[str, str]] = field(default_factory=list)


_STYLE = ("body{font-family:sans-serif;margin:24px;max-width:900px}table{border-collapse:collapse}"
          "td,th{border:1px solid #999;padding:4px 10px;text-align:left}nav a{margin-right:14px}"
          "label{display:block;margin:10px 0 2px}.err{color:#a00}.ok{color:#060}"
          "button,input[type=submit]{margin-top:12px;padding:6px 14px}")


def page(title: str, body: str, *, status: int = 200, headers: Optional[List[Tuple[str, str]]] = None,
         head: str = "") -> Response:
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{esc(title)}</title>"
           f"<style>{_STYLE}</style>{head}</head><body>{body}</body></html>")
    return Response(status, doc.encode(), headers=list(headers or []))


def as_json(obj: Any, status: int = 200) -> Response:
    return Response(status, json.dumps(obj, sort_keys=True).encode(), "application/json")


def redirect(location: str, headers: Optional[List[Tuple[str, str]]] = None) -> Response:
    return Response(303, b"", headers=[("Location", location)] + list(headers or []))


def not_found() -> Response:
    return page("Not found", "<h1>Not found</h1><p>No such page.</p>", status=404)


class Site:
    """Base class: route table of ``(method, path) -> handler`` plus prefix routes; counts every request."""
    name = "site"
    description = ""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.hits: collections.Counter = collections.Counter()
        self.routes: Dict[Tuple[str, str], Callable[[Request], Response]] = {}
        self.prefix_routes: List[Tuple[str, str, Callable[[Request], Response]]] = []
        self.base_url = ""

    def route(self, method: str, path: str, fn: Callable[[Request], Response]) -> None:
        self.routes[(method, path)] = fn

    def prefix(self, method: str, path_prefix: str, fn: Callable[[Request], Response]) -> None:
        self.prefix_routes.append((method, path_prefix, fn))

    def dispatch(self, req: Request) -> Response:
        with self.lock:
            self.hits[f"{req.method} {req.path}"] += 1
            fn = self.routes.get((req.method, req.path))
            if fn is None:
                for m, pre, f in self.prefix_routes:
                    if m == req.method and req.path.startswith(pre):
                        fn = f
                        break
            if fn is None:
                return not_found()
            return fn(req)

    def hit_count(self, path: str, method: Optional[str] = None) -> int:
        with self.lock:
            return sum(n for k, n in self.hits.items()
                       if k.split(" ", 1)[1] == path and (method is None or k.startswith(method + " ")))

    def state(self) -> Dict[str, Any]:
        return {}


def _parse_multipart(ctype: str, body: bytes) -> Tuple[Dict[str, str], Dict[str, Tuple[str, bytes]]]:
    msg = email.message_from_bytes(b"Content-Type: " + ctype.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + body,
                                   policy=email.policy.HTTP)
    form: Dict[str, str] = {}
    files: Dict[str, Tuple[str, bytes]] = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        fname = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if fname is not None:
            files[str(name)] = (fname, payload)
        else:
            form[str(name)] = payload.decode("utf-8", "replace")
    return form, files


def _make_handler(site: Site):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "CamLiveStaging/1.0"

        def log_message(self, fmt, *args):  # quiet: request counts live in Site.hits
            return

        def _run(self, method: str) -> None:
            u = urllib.parse.urlsplit(self.path)
            query = {k: v[-1] for k, v in urllib.parse.parse_qs(u.query, keep_blank_values=True).items()}
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else b""
            ctype = self.headers.get("Content-Type", "")
            form: Dict[str, str] = {}
            files: Dict[str, Tuple[str, bytes]] = {}
            if ctype.startswith("application/x-www-form-urlencoded"):
                form = {k: v[-1] for k, v in urllib.parse.parse_qs(body.decode(), keep_blank_values=True).items()}
            elif ctype.startswith("multipart/form-data"):
                form, files = _parse_multipart(ctype, body)
            cookies: Dict[str, str] = {}
            for piece in (self.headers.get("Cookie") or "").split(";"):
                if "=" in piece:
                    k, v = piece.strip().split("=", 1)
                    cookies[k] = v
            req = Request(method, u.path, query, form, files, {k.lower(): v for k, v in self.headers.items()},
                          cookies, body)
            try:
                resp = site.dispatch(req)
            except Exception as e:  # noqa: BLE001 — a staging bug must surface as a 500, not hang the browser
                resp = page("Server error", f"<h1>Server error</h1><pre>{esc(repr(e))}</pre>", status=500)
            self.send_response(resp.status)
            self.send_header("Content-Type", resp.ctype)
            self.send_header("Content-Length", str(len(resp.body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in resp.headers:
                self.send_header(k, v)
            self.end_headers()
            if method != "HEAD":
                self.wfile.write(resp.body)

        def do_GET(self):
            self._run("GET")

        def do_POST(self):
            self._run("POST")

        def do_HEAD(self):
            self._run("GET")

    return Handler


class SiteServer:
    def __init__(self, site: Site, host: str = "127.0.0.1", port: int = 0):
        self.site = site
        self.httpd = ThreadingHTTPServer((host, port), _make_handler(site))
        self.httpd.daemon_threads = True
        self.host, self.port = host, self.httpd.server_address[1]
        self.url = f"http://{host}:{self.port}"
        site.base_url = self.url
        self._thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05},
                                        daemon=True, name=f"staging-{site.name}")

    def start(self) -> "SiteServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self._thread.join(timeout=5)
