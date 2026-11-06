"""Local HTTP server that hosts MiniWeb apps as real HTML pages.

The same 14 MiniWeb task families (``agentrs.environments.miniweb.families``)
run unchanged: each browser session owns one ``App`` instance whose
server-side state, handlers and verifier are exactly the pure-Python ones. Only
the presentation layer changes: widgets are rendered to semantic HTML and a
small inline JS client forwards user interactions back to the server.

URL layout (``<sid>`` is a random per-episode session id)::

    GET  /s/<sid>/<app path>      full HTML page for the app's current state
    POST /s/<sid>/__event         {"type": "click"|"input", "key", "value"?}
                                  -> {"v", "url", "title", "html", "loading"}
    GET  /s/<sid>/__render?v=N    current render if its version is newer than N
    POST /__control/<token>/<op>  privileged control channel (see below)

Interaction semantics mirror ``MiniWebEnv._apply``: clicking a link calls
``app.goto(href)``, clicking anything else calls ``on_click``, typing/clearing
and selecting call ``on_input``. After an interaction the client either
morphs the DOM in place (keyed by ``data-key``, so existing nodes - and any
agent ids attached to them - survive, like an SPA re-render) or, when the app
URL changed, performs a real browser navigation, so a new document (and new
element ids) is created, exactly when MiniWeb regenerates ids.

Control channel
---------------
``/__control/<token>/...`` is protected by a random secret token generated in
the server process and handed only to the environment (never rendered into
any page). It creates sessions (receiving the full ``TaskSpec`` including its
private part), advances the app clock (one tick per agent action, two for
``wait``), records ``finish`` answers and returns ``app.verify()``. A wrong
token gets a plain 404. The browser additionally aborts every request to
``/__control/`` at the network layer, and agent navigation is restricted to the
app's allowed paths, so the agent has no route to the verifier.
"""

from __future__ import annotations

import hmac
import html
import json
import random
import secrets
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlsplit

from agentrs.environments.base import TaskSpec
from agentrs.environments.miniweb.core import CLICKABLE_ROLES, EDITABLE_ROLES, App, Page, W

SESSION_PREFIX = "/s/"
CONTROL_PREFIX = "/__control/"
MAX_BODY = 1 << 20  # 1 MiB request bodies are plenty


def path_allowed(app: App, url: str) -> bool:
    """MiniWeb navigation allowlist semantics (exact path or ``prefix*``)."""
    return any(url == p or (p.endswith("*") and url.startswith(p[:-1])) for p in app.allowed_paths)


@dataclass
class Session:
    sid: str
    spec: TaskSpec
    app: App
    lock: threading.RLock = field(default_factory=threading.RLock)
    version: int = 1  # bumped on every server-side state change
    page_url: str = ""  # app URL of the page most recently rendered
    created: float = field(default_factory=time.time)

    @property
    def prefix(self) -> str:
        return f"{SESSION_PREFIX}{self.sid}"


# ---------------------------------------------------------------------- rendering
_CSS = """
:root { color-scheme: light; }
body { font: 15px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0; background: #f6f7f9; color: #1d1f23; }
main { max-width: 640px; margin: 0 auto; padding: 12px 16px 24px; }
.w { display: block; margin: 6px 0; }
h2.w { font-size: 20px; margin: 10px 0 6px; }
p.w { margin: 4px 0; }
button.w, .menuitem { font: inherit; padding: 4px 12px; border: 1px solid #9aa3ad; border-radius: 6px;
  background: #fff; cursor: pointer; text-align: left; }
button.w:disabled { opacity: .55; cursor: default; }
.switch[aria-checked="true"] { background: #2f6fde; color: #fff; border-color: #2f6fde; }
.menuitem { margin-left: 18px; max-width: 280px; }
.field label { display: block; font-size: 13px; color: #4a5059; }
.field input, .field select { font: inherit; padding: 4px 6px; min-width: 240px; }
.check label { margin-left: 6px; }
a.w { color: #1a56c4; }
a.w[aria-disabled="true"] { color: #8a8f96; text-decoration: none; cursor: default; }
.alert { color: #a3160e; background: #fde8e6; padding: 6px 10px; border-radius: 6px; }
.status { color: #555; font-style: italic; }
.dialog { border: 2px solid #333; background: #fff; padding: 10px; border-radius: 8px; }
/* Long pages: fixed 60px row pitch, so a 720px-high viewport shows 12 rows and a
   half-viewport scroll moves 6 rows - the same window as MiniWebEnv (viewport=12). */
main[data-scrollable="1"] { padding-top: 12px; }
main[data-scrollable="1"] .w { height: 50px; box-sizing: border-box; margin: 10px 0; }
"""

# Inline client. It never contains the control token. It forwards clicks and
# value changes on ``[data-key]`` widgets, applies the returned render (keyed DOM
# morph, or a real navigation when the app URL changed) and polls while the app
# shows a loading indicator. ``data-busy`` on <html> is set from the moment an
# interaction starts until its result is applied, so automation can wait for it.
_CLIENT_JS = r"""
(() => {
  const PREFIX = __PREFIX__;
  let appUrl = __URL__;
  const root = document.documentElement;
  let v = Number(root.dataset.v || 0);
  let inflight = 0;
  const setBusy = (on) => {
    if (on) { inflight++; root.setAttribute('data-busy', '1'); }
    else if (--inflight <= 0) { inflight = 0; root.removeAttribute('data-busy'); }
  };
  const CONTROL = 'a, button, input[type=checkbox], input[type=radio], [role=menuitem], [role=switch], [role=radio], [role=tab], [role=link], [role=button], [role=checkbox]';

  function patch(o, n) {
    for (const a of Array.from(o.attributes)) {
      if (!a.name.startsWith('data-agentrs') && !n.hasAttribute(a.name)) o.removeAttribute(a.name);
    }
    for (const a of Array.from(n.attributes)) {
      if (o.getAttribute(a.name) !== a.value) o.setAttribute(a.name, a.value);
    }
    const oc = Array.from(o.childNodes), nc = Array.from(n.childNodes);
    const same = oc.length === nc.length && oc.every((c, i) => c.nodeName === nc[i].nodeName);
    if (!same) {
      o.replaceChildren(...nc);
    } else {
      oc.forEach((c, i) => {
        if (c.nodeType === Node.TEXT_NODE) { if (c.nodeValue !== nc[i].nodeValue) c.nodeValue = nc[i].nodeValue; }
        else if (c.nodeType === Node.ELEMENT_NODE) patch(c, nc[i]);
      });
    }
    if (o instanceof HTMLInputElement) {
      if (o.type === 'checkbox' || o.type === 'radio') o.checked = n.hasAttribute('checked');
      else if (o !== document.activeElement) { const nv = n.getAttribute('value') || ''; if (o.value !== nv) o.value = nv; }
    } else if (o instanceof HTMLSelectElement) {
      const sel = n.querySelector('option[selected]');
      o.value = sel ? sel.getAttribute('value') : '';
    }
  }

  // Keyed reconciliation. Nodes whose widget disappears are kept (per document)
  // and reused if a widget with the same key comes back, so a widget keeps its
  // DOM node - and any attributes tools attached to it - for the whole page load.
  const detached = new Map();
  function morph(container, markup) {
    const tpl = document.createElement('template');
    tpl.innerHTML = markup;
    const old = new Map();
    for (const c of Array.from(container.children)) if (c.dataset.key !== undefined) old.set(c.dataset.key, c);
    const keep = new Set();
    let prev = null;
    for (const n of Array.from(tpl.content.children)) {
      const key = n.dataset.key;
      let node = old.get(key) || detached.get(key);
      if (node && node.tagName === n.tagName && !keep.has(node)) { detached.delete(key); patch(node, n); }
      else node = n;
      keep.add(node);
      const want = prev ? prev.nextSibling : container.firstChild;
      if (want !== node) container.insertBefore(node, want);
      prev = node;
    }
    for (const c of Array.from(container.children)) {
      if (!keep.has(c)) { if (c.dataset.key !== undefined) detached.set(c.dataset.key, c); c.remove(); }
    }
  }

  // Apply a server render. Returns true when a real navigation was started.
  function apply(st) {
    if (!st || typeof st.v !== 'number') return false;
    if (typeof st.url === 'string' && st.url !== appUrl) {
      location.assign(PREFIX + st.url);
      return true;
    }
    if (typeof st.html === 'string' && st.v > v) {
      morph(document.getElementById('app'), st.html);
      v = st.v;
      root.dataset.v = String(v);
      document.title = st.title;
      setLoading(!!st.loading);
    }
    return false;
  }

  async function send(ev) {
    setBusy(true);
    let navigating = false;
    try {
      const r = await fetch(PREFIX + '/__event', {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(ev),
      });
      const st = await r.json();
      if (st && st.error) console.warn('agentrs: server rejected event:', st.error);
      navigating = apply(st);
    } catch (e) {
      console.error('agentrs: event failed', e);
    } finally {
      if (!navigating) setBusy(false);
    }
  }

  let polling = false;
  function setLoading(on) {
    root.dataset.loading = on ? '1' : '0';
    if (on && !polling) poll();
  }
  async function poll() {
    polling = true;
    try {
      while (root.dataset.loading === '1') {
        await new Promise((res) => setTimeout(res, 100));
        try {
          const r = await fetch(PREFIX + '/__render?v=' + v);
          if (apply(await r.json())) break;
        } catch (e) { /* transient; retry */ }
      }
    } finally { polling = false; }
  }

  document.addEventListener('click', (e) => {
    const w = e.target.closest('[data-key]');
    const ctl = e.target.closest(CONTROL);
    if (!w || !ctl || !w.contains(ctl)) return;
    if (ctl.tagName === 'A') e.preventDefault();
    if (ctl.disabled || ctl.getAttribute('aria-disabled') === 'true') return;
    send({type: 'click', key: w.dataset.key});
  });
  document.addEventListener('input', (e) => {
    const t = e.target;
    if (!(t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement)) return;
    if (t.type === 'checkbox' || t.type === 'radio') return;
    const w = t.closest('[data-key]');
    if (w) send({type: 'input', key: w.dataset.key, value: t.value});
  });
  document.addEventListener('change', (e) => {
    const t = e.target;
    if (!(t instanceof HTMLSelectElement)) return;
    const w = t.closest('[data-key]');
    if (w) send({type: 'input', key: w.dataset.key, value: t.value});
  });
  if (root.dataset.loading === '1') poll();
})();
"""


def _esc(s: Any) -> str:
    return html.escape("" if s is None else str(s), quote=True)


def _field_id(key: str) -> str:
    return "f-" + "".join(c if c.isalnum() or c in "-_" else "_" for c in key)


def widget_html(w: W, prefix: str) -> str:
    """Render one MiniWeb widget as semantic HTML carrying ``data-key``."""
    k, name, role = _esc(w.key), _esc(w.name), w.role
    dis = " disabled" if w.disabled else ""
    aria_dis = ' aria-disabled="true"' if w.disabled else ""
    if role == "text":
        return f'<p class="w" data-key="{k}">{name}</p>'
    if role == "heading":
        return f'<h2 class="w" data-key="{k}">{name}</h2>'
    if role in ("alert", "status"):
        return f'<div class="w {role}" role="{role}" data-key="{k}">{name}</div>'
    if role == "dialog":
        return f'<div class="w dialog" role="dialog" data-key="{k}">{name}</div>'
    if role == "button":
        return f'<button type="button" class="w" data-key="{k}"{dis}>{name}</button>'
    if role == "link":
        if w.disabled or w.href is None:
            return f'<a class="w" role="link" data-key="{k}"{aria_dis}>{name}</a>'
        return f'<a class="w" href="{_esc(prefix + w.href)}" data-key="{k}">{name}</a>'
    if role in ("textbox", "searchbox"):
        fid = _field_id(w.key)
        typ = "search" if role == "searchbox" else "text"
        return (f'<div class="w field" data-key="{k}"><label for="{fid}">{name}</label>'
                f'<input type="{typ}" id="{fid}" value="{_esc(w.value or "")}" autocomplete="off"{dis}></div>')
    if role == "combobox":
        fid = _field_id(w.key)
        opts = ['<option value="" hidden></option>']
        for o in w.options or []:
            sel = " selected" if o == (w.value or "") else ""
            opts.append(f'<option value="{_esc(o)}"{sel}>{_esc(o)}</option>')
        return (f'<div class="w field" data-key="{k}"><label for="{fid}">{name}</label>'
                f'<select id="{fid}"{dis}>{"".join(opts)}</select></div>')
    if role == "checkbox":
        fid = _field_id(w.key)
        chk = " checked" if w.checked else ""
        return (f'<div class="w check" data-key="{k}"><input type="checkbox" id="{fid}"{chk}{dis}>'
                f'<label for="{fid}">{name}</label></div>')
    if role == "switch":
        state = "true" if w.checked else "false"
        return f'<button type="button" class="w switch" role="switch" aria-checked="{state}" data-key="{k}"{dis}>{name}</button>'
    if role == "menubutton":
        ex = "true" if w.expanded else "false"
        return (f'<button type="button" class="w" aria-haspopup="menu" aria-expanded="{ex}" '
                f'data-key="{k}"{dis}>{name}</button>')
    if role == "menuitem":
        return f'<div class="w menuitem" role="menuitem" tabindex="0" data-key="{k}"{aria_dis}>{name}</div>'
    if role in CLICKABLE_ROLES:  # radio, tab, ...
        extra = ""
        if w.checked is not None:
            extra = f' aria-checked="{"true" if w.checked else "false"}"'
        return f'<div class="w" role="{_esc(role)}" tabindex="0" data-key="{k}"{extra}{aria_dis}>{name}</div>'
    return f'<div class="w" role="{_esc(role)}" data-key="{k}">{name}</div>'


def widgets_html(page: Page, app: App, prefix: str) -> str:
    parts = [widget_html(w, prefix) for w in page.widgets]
    if app.loading:  # same convention as MiniWebEnv.observe: trailing status element
        parts.append('<div class="w status" role="status" data-key="__loading__">Loading...</div>')
    return "\n".join(parts)


def page_html(sess: Session, page: Page) -> str:
    client = (_CLIENT_JS.replace("__PREFIX__", json.dumps(sess.prefix))
              .replace("__URL__", json.dumps(page.url)))
    client = client.replace("</", "<\\/")  # never terminate the script element early
    return (
        "<!doctype html>\n"
        f'<html lang="en" data-v="{sess.version}" data-loading="{"1" if sess.app.loading else "0"}">\n'
        '<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(page.title)}</title><style>{_CSS}</style></head>\n"
        f'<body><main id="app" data-scrollable="{"1" if page.scrollable else "0"}">\n'
        f"{widgets_html(page, sess.app, sess.prefix)}\n</main>\n"
        f"<script>{client}</script></body></html>\n"
    )


# ---------------------------------------------------------------------- server
class LocalWebServer:
    """Threaded HTTP server hosting one MiniWeb ``App`` per session."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, max_sessions: int = 256):
        self.host = host
        self.token = secrets.token_urlsafe(24)  # control-channel secret; never rendered
        self.max_sessions = max_sessions
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._port = port

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "LocalWebServer":
        if self._httpd is not None:
            return self
        handler = type("_BoundHandler", (_Handler,), {"app_server": self})
        self._httpd = ThreadingHTTPServer((self.host, self._port), handler)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="agentrs-localweb",
                                        kwargs={"poll_interval": 0.2}, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        with self._lock:
            self._sessions.clear()

    @property
    def running(self) -> bool:
        return self._httpd is not None

    @property
    def port(self) -> int:
        assert self._httpd is not None, "server not started"
        return self._httpd.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ------------------------------------------------------------------ sessions
    def create_session(self, spec: TaskSpec) -> Session:
        from agentrs.environments.miniweb.families import FAMILIES

        if spec.family not in FAMILIES:
            raise ValueError(f"unknown MiniWeb family {spec.family!r}")
        app = FAMILIES[spec.family](spec, random.Random(spec.seed))  # same seeding as MiniWebEnv
        sess = Session(sid=secrets.token_hex(6), spec=spec, app=app)
        sess.page_url = app.render().url
        with self._lock:
            if len(self._sessions) >= self.max_sessions:  # evict the oldest
                oldest = min(self._sessions.values(), key=lambda s: s.created)
                self._sessions.pop(oldest.sid, None)
            self._sessions[sess.sid] = sess
        return sess

    def get_session(self, sid: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(sid)

    def close_session(self, sid: str) -> None:
        with self._lock:
            self._sessions.pop(sid, None)

    # ------------------------------------------------------------------ app semantics
    @staticmethod
    def render_state(sess: Session) -> dict:
        page = sess.app.render()
        sess.page_url = page.url
        return {"v": sess.version, "url": page.url, "title": page.title,
                "html": widgets_html(page, sess.app, sess.prefix), "loading": sess.app.loading}

    @staticmethod
    def apply_event(sess: Session, ev: dict) -> Optional[str]:
        """Apply a client interaction with MiniWebEnv semantics. Returns an error or None."""
        app = sess.app
        typ, key = ev.get("type"), ev.get("key")
        if not isinstance(key, str):
            return "missing widget key"
        page = app.render()
        w = next((x for x in page.widgets if x.key == key), None)
        if w is None:
            return f"unknown widget {key!r} (stale page?)"
        if w.disabled:
            return f"widget {key!r} is disabled"
        if typ == "click":
            if w.role == "link" and w.href is not None:
                app.goto(w.href)
            elif w.on_click is not None:
                w.on_click()
            # clicking fields / static text: valid, no effect
        elif typ == "input":
            value = ev.get("value")
            if not isinstance(value, str) or len(value) > 10_000:
                return "invalid value"
            if w.role in EDITABLE_ROLES and w.on_input is not None:
                w.on_input(value)
            elif w.role == "combobox" and w.on_input is not None:
                if value == "":
                    w.on_input("")
                else:
                    match = [o for o in (w.options or []) if o.strip().lower() == value.strip().lower()]
                    if not match:
                        return f"option {value!r} not found"
                    w.on_input(match[0])
            else:
                return f"widget {key!r} is not editable"
        else:
            return f"unknown event type {typ!r}"
        sess.version += 1
        return None

    @staticmethod
    def tick(sess: Session, n: int) -> bool:
        """Advance the app clock ``n`` ticks. Returns True if any scheduled callback ran."""
        before = list(sess.app._pending)
        sess.app.advance(n)
        ran = any(p not in sess.app._pending for p in before)
        if ran:
            sess.version += 1
        return ran

    @staticmethod
    def state(sess: Session) -> dict:
        app = sess.app
        page = app.render()
        return {"sid": sess.sid, "family": app.family, "url": page.url, "title": page.title,
                "terminal": bool(app.terminal), "loading": bool(app.loading),
                "finished": bool(app.server.get("finished", False)), "version": sess.version,
                "tick": app.tick, "allowed_paths": list(app.allowed_paths)}

    def control(self, op: str, payload: dict) -> tuple[int, dict]:
        """Privileged operations (token already checked)."""
        if op == "ping":
            return 200, {"ok": True}
        if op == "session":
            spec = TaskSpec.model_validate(payload["task"])
            sess = self.create_session(spec)
            with sess.lock:
                return 200, {"sid": sess.sid, "prefix": sess.prefix, **self.state(sess)}
        sess = self.get_session(str(payload.get("sid", "")))
        if sess is None:
            return 404, {"error": "unknown session"}
        with sess.lock:
            if op == "state":
                return 200, self.state(sess)
            if op == "tick":
                n = int(payload.get("n", 1))
                if not 0 <= n <= 100:
                    return 400, {"error": "bad tick count"}
                changed = self.tick(sess, n)
                return 200, {"changed": changed, **self.state(sess)}
            if op == "finish":
                answer = payload.get("answer")
                if answer is not None and not isinstance(answer, str):
                    return 400, {"error": "answer must be a string"}
                sess.app.server["answer"] = answer
                sess.app.server["finished"] = True
                sess.version += 1
                return 200, self.state(sess)
            if op == "verify":
                ver = sess.app.verify()
                ver.verifier_id = f"localweb/{sess.app.family}/v1"
                return 200, {"verification": json.loads(ver.model_dump_json()), **self.state(sess)}
            if op == "close":
                self.close_session(sess.sid)
                return 200, {"ok": True}
        return 404, {"error": f"unknown op {op!r}"}


class _Handler(BaseHTTPRequestHandler):
    app_server: LocalWebServer  # bound in LocalWebServer.start
    server_version = "agentrs-localweb/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep test output quiet
        pass

    # ------------------------------------------------------------------ helpers
    def _send(self, status: int, body: bytes, ctype: str, extra: Optional[dict] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, obj: dict) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json")

    def _html(self, status: int, text: str) -> None:
        self._send(status, text.encode(), "text/html; charset=utf-8")

    def _not_found(self) -> None:
        self._html(404, "<!doctype html><title>Not found</title><p>Not found</p>")

    def _body(self) -> Optional[dict]:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            return None
        raw = self.rfile.read(n) if n else b""
        try:
            obj = json.loads(raw or b"{}")
        except ValueError:
            return None
        return obj if isinstance(obj, dict) else None

    def _session_route(self, path: str) -> tuple[Optional[Session], str]:
        rest = path[len(SESSION_PREFIX):]
        sid, _, app_path = rest.partition("/")
        return self.app_server.get_session(sid), "/" + app_path

    # ------------------------------------------------------------------ verbs
    def do_GET(self):  # noqa: N802 - stdlib naming
        parts = urlsplit(self.path)
        path = unquote(parts.path)
        if not path.startswith(SESSION_PREFIX):
            return self._not_found()
        sess, app_path = self._session_route(path)
        if sess is None:
            return self._not_found()
        if app_path == "/__render":
            since = parse_qs(parts.query).get("v", ["0"])[0]
            with sess.lock:
                st = self.app_server.render_state(sess)
            if str(since).isdigit() and int(since) >= st["v"]:
                st = {"v": st["v"]}  # nothing new
            return self._json(200, st)
        with sess.lock:
            app = sess.app
            # A GET for a different allowed path is a navigation (address bar, ``navigate``).
            # Disallowed paths do not change state; we redirect back to the current page.
            if app_path not in (sess.page_url, app.url) and path_allowed(app, app_path):
                app.goto(app_path)
                sess.version += 1
            page = app.render()
            sess.page_url = page.url
            if page.url != app_path:
                return self._send(303, b"", "text/plain", {"Location": sess.prefix + page.url})
            body = page_html(sess, page)
        return self._html(200, body)

    do_HEAD = do_GET

    def do_POST(self):  # noqa: N802
        path = unquote(urlsplit(self.path).path)
        body = self._body()
        if body is None:
            return self._json(400, {"error": "bad request body"})
        if path.startswith(CONTROL_PREFIX):
            token, _, op = path[len(CONTROL_PREFIX):].partition("/")
            if not hmac.compare_digest(token.encode(), self.app_server.token.encode()):
                return self._not_found()  # do not reveal that the endpoint exists
            try:
                status, out = self.app_server.control(op, body)
            except Exception as e:  # control errors are reported, never crash the server
                status, out = 500, {"error": f"{type(e).__name__}: {e}"}
            return self._json(status, out)
        if path.startswith(SESSION_PREFIX):
            sess, app_path = self._session_route(path)
            if sess is None or app_path != "/__event":
                return self._not_found()
            with sess.lock:
                err = self.app_server.apply_event(sess, body)
                st = self.app_server.render_state(sess)
            st["ok"] = err is None
            if err:
                st["error"] = err
            return self._json(200, st)
        return self._not_found()


__all__ = ["LocalWebServer", "Session", "widget_html", "page_html", "path_allowed"]
