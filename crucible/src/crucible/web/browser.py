"""A controlled headless browser for Crucible's browser mode (Playwright + Chromium).

`BrowserSession(app_url)` launches Chromium so that every ``*.crucible`` host name resolves to the local web app
(``--host-resolver-rules=MAP *.crucible 127.0.0.1:<port>``; every other name fails to resolve) and installs a
route handler that aborts any request whose host is neither ``*.crucible`` nor the app server itself. The
network is therefore closed: pages, forms and redirects stay inside the synthetic internet.

Agents drive it with primitive methods (`goto`, `click`, `fill`, `select`, `check`, `press`, `submit`, `back`)
or with one JSON-able call, `act({"action": ..., ...})`, which returns the next observation:

    {"ok": bool, "error": str | None, "url", "title", "text" (visible text, truncated),
     "messages": [{"role": "alert" | "status", "text"}],
     "elements": [{"id", "role", "name", "tag", "value"?, "options"?, "href"?, "form"?, "required"?}]}

Element ids are the page's own ``id`` attributes where present (all form controls of the Crucible pages have
stable ids such as ``bank-transfer-amount``) and ``e<n>`` (document order) otherwise; any of them can be used as
the target of an action. `serve_in_thread(app)` runs a FastAPI app with uvicorn on a free local port.
"""
from __future__ import annotations

import socket
import threading
import time
from typing import Any
from urllib.parse import urljoin, urlsplit

__all__ = ["BrowserSession", "BrowserError", "serve_in_thread", "playwright_available"]

_OBSERVE_JS = r"""
(maxText) => {
  const SEL = 'a[href], button, input, select, textarea, summary, [role=button], [role=link], [role=checkbox], ' +
              '[role=tab], [role=menuitem], [contenteditable=""], [contenteditable=true]';
  const implicit = (el) => {
    const tag = el.tagName.toLowerCase(), type = (el.getAttribute('type') || '').toLowerCase();
    if (tag === 'a') return 'link';
    if (tag === 'button' || tag === 'summary') return 'button';
    if (tag === 'select') return el.multiple ? 'listbox' : 'combobox';
    if (tag === 'textarea') return 'textbox';
    if (tag === 'input') {
      if (['submit', 'button', 'reset', 'image'].includes(type)) return 'button';
      if (type === 'checkbox') return 'checkbox';
      if (type === 'radio') return 'radio';
      if (type === 'number') return 'spinbutton';
      if (type === 'search') return 'searchbox';
      return 'textbox';
    }
    return tag;
  };
  const clean = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const nameOf = (el) => {
    const al = el.getAttribute('aria-label');
    if (al) return clean(al);
    const lb = el.getAttribute('aria-labelledby');
    if (lb) {
      const t = lb.split(/\s+/).map(id => document.getElementById(id)).filter(Boolean).map(e => e.innerText).join(' ');
      if (clean(t)) return clean(t);
    }
    if (el.labels && el.labels.length) {
      const t = Array.from(el.labels).map(l => l.innerText).join(' ');
      if (clean(t)) return clean(t.replace(/\s\*$/, ''));
    }
    const tag = el.tagName.toLowerCase();
    if (tag === 'input' && ['submit', 'button', 'reset'].includes((el.type || '').toLowerCase())) return clean(el.value);
    if (['a', 'button', 'summary'].includes(tag) || el.getAttribute('role')) {
      const t = clean(el.innerText);
      if (t) return t;
    }
    return clean(el.getAttribute('placeholder') || el.getAttribute('title') || el.getAttribute('name') || '');
  };
  const used = new Set();
  let n = 0;
  const out = [];
  for (const el of document.querySelectorAll(SEL)) {
    if (el.type === 'hidden') continue;
    const st = window.getComputedStyle(el);
    if (st.display === 'none' || st.visibility === 'hidden') continue;
    let cid = el.id && !used.has(el.id) ? el.id : null;
    if (!cid) { do { n += 1; cid = 'e' + n; } while (used.has(cid) || document.getElementById(cid)); }
    used.add(cid);
    el.setAttribute('data-cid', cid);
    const tag = el.tagName.toLowerCase();
    const item = {id: cid, role: el.getAttribute('role') || implicit(el), name: nameOf(el), tag: tag};
    if (tag === 'input' || tag === 'textarea' || tag === 'select') {
      if (el.type === 'checkbox' || el.type === 'radio') item.value = el.checked;
      else if (el.type === 'password') item.value = el.value ? '********' : '';
      else item.value = el.value;
    }
    if (tag === 'select') item.options = Array.from(el.options).slice(0, 60).map(o => ({value: o.value, label: clean(o.text)}));
    if (tag === 'a') item.href = el.getAttribute('href');
    if (el.required || el.getAttribute('aria-required') === 'true') item.required = true;
    if (el.disabled) item.disabled = true;
    const f = el.form || el.closest('form');
    if (f && f.id) item.form = f.id;
    out.push(item);
  }
  const messages = Array.from(document.querySelectorAll('[role=alert], [role=status]')).map(
    e => ({role: e.getAttribute('role'), text: clean(e.innerText).slice(0, 600)}));
  const text = document.body ? document.body.innerText : '';
  return {title: document.title, text: text.length > maxText ? text.slice(0, maxText) + ' ...[truncated]' : text,
          messages: messages, elements: out};
}
"""

_NAVIGATES_JS = r"""
(el) => {
  const a = el.closest('a[href]');
  if (a) { const h = a.getAttribute('href') || ''; return !h.startsWith('#') && !h.startsWith('javascript:'); }
  const tag = el.tagName.toLowerCase(), type = (el.getAttribute('type') || '').toLowerCase();
  if (tag === 'button') return !!el.form && (type === '' || type === 'submit');
  if (tag === 'input') return !!el.form && (type === 'submit' || type === 'image');
  return false;
}
"""


class BrowserError(RuntimeError):
    pass


def playwright_available() -> bool:
    """True if Playwright and its Chromium build can be launched here."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            import os
            return os.path.exists(p.chromium.executable_path)
    except Exception:  # noqa: BLE001
        return False


def _css_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


class BrowserSession:
    """One headless Chromium (one context, one page) confined to ``*.crucible`` and the app server.

    `app_url` is the web app's base URL (``http://127.0.0.1:<port>``); `start` an optional first URL (e.g.
    ``"http://id.crucible/"``). Always `close()` it (or use it as a context manager)."""

    def __init__(self, app_url: str, *, start: str | None = None, headless: bool = True, timeout_ms: int = 15_000,
                 max_text: int = 4000, extra_args: tuple[str, ...] = ()):
        from playwright.sync_api import sync_playwright
        u = urlsplit(app_url)
        if not u.hostname or not u.port:
            raise ValueError(f"app_url needs a host and port, got {app_url!r}")
        self.app_url = app_url.rstrip("/")
        self.server = (u.hostname.lower(), u.port)
        self.max_text = max_text
        self.timeout_ms = timeout_ms
        self.blocked: list[str] = []
        self._pw = sync_playwright().start()
        try:
            rules = f"MAP *.crucible {u.hostname}:{u.port}, MAP * ~NOTFOUND, EXCLUDE {u.hostname}"
            self._browser = self._pw.chromium.launch(headless=headless, args=[
                f"--host-resolver-rules={rules}", "--no-proxy-server", "--disable-background-networking",
                "--disable-component-update", "--disable-sync", "--disable-default-apps", "--no-first-run",
                "--disable-extensions", "--renderer-process-limit=2", "--disable-gpu", *extra_args])
            self._ctx = self._browser.new_context(java_script_enabled=True, service_workers="block",
                                                  accept_downloads=False)
            self._ctx.set_default_timeout(timeout_ms)
            self._ctx.route("**/*", self._route)
            self.page = self._ctx.new_page()
        except Exception:
            self._pw.stop()
            raise
        self._closed = False
        if start:
            self.goto(start)

    # ------------------------------------------------------------------ network control
    def _allowed(self, url: str) -> bool:
        u = urlsplit(url)
        if u.scheme in ("data", "blob", "about"):
            return True
        if u.scheme not in ("http", "https"):
            return False
        host = (u.hostname or "").lower()
        if host.endswith(".crucible"):
            return True
        return (host, u.port) == self.server

    def _route(self, route, request) -> None:
        if self._allowed(request.url):
            route.continue_()
        else:
            self.blocked.append(request.url)
            route.abort("blockedbyclient")

    # ------------------------------------------------------------------ element lookup
    def _locate(self, target: Any):
        p = self.page
        if hasattr(target, "click") and hasattr(target, "evaluate"):   # already a Playwright Locator
            return target
        if isinstance(target, dict):
            target = target.get("id") or target.get("selector") or target.get("text") or target.get("name")
        if target is None:
            raise BrowserError("no target element given")
        t = str(target)
        cands = [f"[data-cid={_css_str(t)}]", f"[id={_css_str(t)}]"]
        for sel in cands:
            loc = p.locator(sel)
            if loc.count():
                return loc.first
        if t[:1] in "#.[" or "=" in t or t.startswith(("css=", "xpath=", "text=")):
            try:
                loc = p.locator(t)
                if loc.count():
                    return loc.first
            except Exception:  # noqa: BLE001 - not a valid selector: try it as text
                pass
        for loc in (p.get_by_role("button", name=t, exact=True), p.get_by_role("link", name=t, exact=True),
                    p.get_by_label(t, exact=True), p.get_by_text(t, exact=True),
                    p.get_by_role("button", name=t), p.get_by_role("link", name=t)):
            if loc.count():
                return loc.first
        raise BrowserError(f"no element matches {t!r}")

    def _settle(self) -> None:
        try:
            self.page.wait_for_load_state("load", timeout=self.timeout_ms)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ actions
    def _url(self, url: str) -> str:
        url = url.strip()
        if url.startswith("/"):
            return urljoin(self.page.url if self.page.url.startswith("http") else self.app_url + "/", url)
        if "://" not in url:
            if "." not in url.split("/", 1)[0]:          # "bank" or "bank/invoices"
                head, _, rest = url.partition("/")
                url = f"{head}.crucible/{rest}"
            url = "http://" + url
        return url

    def goto(self, url: str) -> None:
        try:
            self.page.goto(self._url(url), wait_until="load")
        except Exception as e:  # noqa: BLE001
            raise BrowserError(_short(e)) from None

    def click(self, target: Any) -> None:
        loc = self._locate(target)
        try:
            nav = loc.evaluate(_NAVIGATES_JS)
            if nav:
                with self.page.expect_navigation(wait_until="load", timeout=self.timeout_ms):
                    loc.click()
            else:
                loc.click()
                self._settle()
        except Exception as e:  # noqa: BLE001
            raise BrowserError(_short(e)) from None

    def fill(self, target: Any, text: Any) -> None:
        loc = self._locate(target)
        try:
            typ = (loc.get_attribute("type") or "").lower()
            if typ in ("checkbox", "radio"):
                loc.set_checked(str(text).lower() in ("1", "true", "yes", "on"))
            elif loc.evaluate("el => el.tagName.toLowerCase()") == "select":
                self._select(loc, str(text))
            else:
                loc.fill("" if text is None else str(text))
        except Exception as e:  # noqa: BLE001
            raise BrowserError(_short(e)) from None

    def _select(self, loc, value: str) -> None:
        opts = loc.evaluate("el => Array.from(el.options).map(o => [o.value, o.text.trim()])")
        if any(v == value for v, _ in opts):
            loc.select_option(value=value)
        elif any(lab == value for _, lab in opts):
            loc.select_option(label=value)
        else:
            m = [v for v, lab in opts if value and (value in lab or value in v)]
            if not m:
                raise BrowserError(f"no option {value!r} (options: {[v for v, _ in opts][:20]})")
            loc.select_option(value=m[0])

    def select(self, target: Any, value: str) -> None:
        loc = self._locate(target)
        try:
            self._select(loc, str(value))
        except BrowserError:
            raise
        except Exception as e:  # noqa: BLE001
            raise BrowserError(_short(e)) from None

    def check(self, target: Any, on: bool = True) -> None:
        self.fill(target, "true" if on else "false")

    def press(self, target: Any, key: str = "Enter") -> None:
        loc = self._locate(target) if target is not None else self.page.locator(":focus").first
        try:
            submits = key == "Enter" and loc.evaluate(
                "el => !!el.form && el.tagName.toLowerCase() === 'input'")
            if submits:
                with self.page.expect_navigation(wait_until="load", timeout=self.timeout_ms):
                    loc.press(key)
            else:
                loc.press(key)
                self._settle()
        except Exception as e:  # noqa: BLE001
            raise BrowserError(_short(e)) from None

    def submit(self, target: Any = None) -> None:
        """Submit the form identified by `target` (a form id, or any element inside the form) by clicking its
        submit button; without a target, the form of the focused element."""
        loc = self._locate(target) if target is not None else self.page.locator(":focus").first
        try:
            fid = loc.evaluate("el => { const f = el.tagName.toLowerCase() === 'form' ? el : (el.form || "
                               "el.closest('form')); if (!f) return null; if (!f.id) f.id = 'form-' + "
                               "Math.random().toString(36).slice(2); return f.id; }")
        except Exception as e:  # noqa: BLE001
            raise BrowserError(_short(e)) from None
        if not fid:
            raise BrowserError("target is not in a form")
        btn = self.page.locator(f"form[id={_css_str(fid)}] button[type=submit], "
                                f"form[id={_css_str(fid)}] input[type=submit]").first
        if not btn.count():
            raise BrowserError(f"form {fid} has no submit button")
        self.click(btn)

    def back(self) -> None:
        try:
            self.page.go_back(wait_until="load")
        except Exception as e:  # noqa: BLE001
            raise BrowserError(_short(e)) from None

    # ------------------------------------------------------------------ observation
    def observe(self) -> dict:
        d = None
        for attempt in range(3):
            try:
                d = self.page.evaluate(_OBSERVE_JS, self.max_text)
                break
            except Exception as e:  # noqa: BLE001 - mid-navigation, or an error page without a body
                d = {"title": "", "text": "", "messages": [], "elements": [], "observe_error": _short(e)}
                self._settle()
        return {"url": self.page.url, **d}

    def act(self, action: dict) -> dict:
        """Run one action and return ``{"ok", "error", **observe()}``. Actions:
        ``{"action": "goto", "url"}``, ``{"action": "click", "target"}``, ``{"action": "fill", "target", "value"}``,
        ``{"action": "select", "target", "value"}``, ``{"action": "check", "target", "value": bool}``,
        ``{"action": "press", "target"?, "key"}``, ``{"action": "submit", "target"?}``, ``{"action": "back"}``,
        ``{"action": "observe"}``. `target` may also be given as ``id``, ``selector`` or ``text``."""
        if not isinstance(action, dict):
            return {"ok": False, "error": "action must be an object", **self.observe()}
        kind = str(action.get("action") or action.get("type") or "").lower()
        target = action.get("target", action.get("id", action.get("selector", action.get("text"))))
        value = action.get("value", action.get("text_value", action.get("input")))
        err = None
        try:
            if kind == "goto":
                self.goto(str(action.get("url") or target or ""))
            elif kind == "click":
                self.click(target)
            elif kind in ("fill", "type"):
                self.fill(target, "" if value is None else value)
            elif kind == "select":
                self.select(target, "" if value is None else str(value))
            elif kind == "check":
                self.check(target, bool(value) if not isinstance(value, str) else value.lower() in ("1", "true", "on"))
            elif kind == "press":
                self.press(target, str(action.get("key") or value or "Enter"))
            elif kind == "submit":
                self.submit(target)
            elif kind == "back":
                self.back()
            elif kind in ("observe", "noop", ""):
                pass
            else:
                err = f"unknown action {kind!r}"
        except BrowserError as e:
            err = str(e)
        obs = self.observe()
        return {"ok": err is None, "error": err, **obs}

    # ------------------------------------------------------------------ lifetime
    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        for fn in (lambda: self._ctx.close(), lambda: self._browser.close(), lambda: self._pw.stop()):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "BrowserSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):  # best effort: never leak a Chromium
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def _short(e: Exception) -> str:
    s = str(e).strip().splitlines()
    return (s[0] if s else type(e).__name__)[:300]


def serve_in_thread(app: Any, host: str = "127.0.0.1", port: int = 0, log_level: str = "warning"):
    """Serve an ASGI app with uvicorn in a daemon thread on `host` (a free port by default).
    Returns ``(base_url, stop_fn)``."""
    import uvicorn
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(64)
    server = uvicorn.Server(uvicorn.Config(app, log_level=log_level, lifespan="off", access_log=False))
    t = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, name="crucible-web", daemon=True)
    t.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if not t.is_alive() or time.monotonic() > deadline:
            server.should_exit = True
            sock.close()
            raise RuntimeError("web server failed to start")
        time.sleep(0.02)
    url = f"http://{host}:{sock.getsockname()[1]}"

    def stop() -> None:
        server.should_exit = True
        t.join(timeout=10)
        try:
            sock.close()
        except OSError:
            pass
    return url, stop
