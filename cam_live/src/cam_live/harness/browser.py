"""cam_live.harness.browser — the ``structured_browser`` executor on Playwright Chromium (spec 12A.2). Workstream EB.

* **Sessions.** ``default`` / ``identity:<name>`` are persistent contexts (one profile directory per authorized
  identity; cookies survive a restart). ``research:<name>`` are isolated incognito contexts that only issue
  read-type requests. ``cdp:<name>`` is an externally supplied Chromium attached with ``connect_over_cdp``.
* **Observations** (via :mod:`cam_live.harness.perception`): URL, title, bounded visible-text digest, accessibility
  tree, DOM outline, interactive elements with roles and per-observation refs, forms, links, tables, tabs and a
  diff against the previous observation of the tab. Screenshots only on demand.
* **Primitives**: navigate, open/close/select tab, inspect, read, click, fill, select_option, press, scroll, upload,
  download, wait_for, screenshot. Elements are addressed by ``ref`` + ``obs_id`` or by semantic locator
  (role/name) with recorded fallbacks. A ref whose element vanished or changed since its observation fails with
  ``FailureClass.STALE_OBSERVATION``; a semantic locator that matches nothing fails with ``PAGE_CHANGED``.
* **Authorization.** Every primitive calls ``ExecContext.check`` before it runs. Network requests are checked in
  two layers before they leave the browser:

  1. the *route layer* (Playwright ``context.route`` / ``route_web_socket``) sees every request a page starts —
     navigations (typed, clicked, scripted, popups, new tabs) and downloads per URL, writes (any method other than
     GET / HEAD / OPTIONS, e.g. a script's same-origin POST) per URL, reads per origin and action, WebSockets per URL;
  2. the *network layer*: Chromium is launched with a private, password-protected forward proxy in this process
     (:class:`_NetGuard`) as its only way out (also for loopback). It sees what the route layer cannot — every
     redirect hop (each hop is checked before it is sent, so a redirect to a forbidden destination never reaches
     it), requests of workers, WebSocket tunnels and anything else — and forwards a request only when the route
     layer has just authorized exactly that request or the policy allows it now. HTTPS and WebSocket tunnels are
     judged per host and port (the proxy cannot see inside TLS); plain HTTP per URL. A host name is resolved once
     and the connection goes only to those addresses; a non-global answer is put to the policy as ``peer_ip``.

  Service workers are blocked, QUIC is off and WebRTC may not use unproxied UDP; ``WebTransport`` is removed from
  pages (none of them could be checked). A denial is final: the result is ``DENIED`` with a hard failure class.
  Externally supplied browsers (``cdp:``) cannot be given the proxy: for them only the route layer applies and a
  redirect hop is judged after it happened (the tab is reset); see docs/HARNESS_ADAPTER_GUIDE.md.
* ``screen()`` exposes screenshot + raw mouse/keyboard for the visual executor (workstream EC).

Playwright's API is driven from one private asyncio thread, so the executor may be called from any thread; actions
on one session are serialised, different sessions run concurrently.

Provenance: the role / accessible-name rules in ``_PAGE_JS`` are adapted from
``agentrs/src/agentrs/environments/web/dom_extract.js`` (commit 3968583); the rest was written for CAM Live.
"""
from __future__ import annotations

import asyncio
import base64
import collections
import concurrent.futures
import glob
import hashlib
import hmac
import ipaddress
import json
import os
import pwd
import re
import secrets as _secrets
import shutil
import socket
import stat
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

from ..contracts import (ActionResult, ActionSpec, AuthorizationDecision, AuthVerdict, Clock, Cost, ExecContext,
                         FailureClass, Level, Modality, Observation, PrimitiveOp, ResultStatus, ToolDescriptor,
                         VerificationCondition, WallClock)
from . import SERVICE_USERS
from .failures import refusal_class
from .perception import Perception, digest, sha256_of

REF_ATTR = "data-camlive-ref"
_REF_RE = re.compile(r"^e\d{1,9}$")
_ALLOW_SCHEMES = ("http", "https")

_PAGE_JS = r"""
(o) => {
  o = Object.assign({mode:'extract', next:1, maxElements:150, maxText:20000, maxRows:40, maxTables:12, maxCols:30,
                     maxCells:600, maxUrl:1024, maxTitle:300, attr:'data-camlive-ref', ref:null, el:null}, o || {});
  const clipUrl = (u) => { u = String(u == null ? '' : u); return u.length > o.maxUrl ? u.slice(0, o.maxUrl) + '…' : u; };
  const A = o.attr; let next = o.next;
  if (!window.__camlive_doc) window.__camlive_doc = Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
  const norm = (s) => String(s == null ? '' : s).replace(/\s+/g, ' ').trim();
  const clip = (s, n) => { s = norm(s); return s.length > n ? s.slice(0, n - 1) + '…' : s; };
  const IROLES = ['button','link','textbox','searchbox','checkbox','radio','combobox','listbox','menuitem',
    'menuitemcheckbox','menuitemradio','switch','tab','slider','spinbutton','option','treeitem'];
  const SEL = ['a[href]','button','input:not([type=hidden])','select','textarea','summary','[contenteditable=""]',
    '[contenteditable=true]','[tabindex]:not([tabindex="-1"])','[onclick]'].concat(IROLES.map(r => '[role=' + r + ']')).join(',');
  function implicitRole(el) {
    const tag = el.tagName.toLowerCase();
    switch (tag) {
      case 'a': case 'area': return el.hasAttribute('href') ? 'link' : null;
      case 'button': case 'summary': return 'button';
      case 'input': {
        const t = (el.getAttribute('type') || 'text').toLowerCase();
        if (t === 'hidden') return null;
        if (['button','submit','reset','image','file','color'].includes(t)) return 'button';
        if (t === 'checkbox') return 'checkbox';
        if (t === 'radio') return 'radio';
        if (t === 'range') return 'slider';
        if (t === 'number') return 'spinbutton';
        if (t === 'search') return 'searchbox';
        return 'textbox';
      }
      case 'textarea': return 'textbox';
      case 'select': return (el.multiple || el.size > 1) ? 'listbox' : 'combobox';
      case 'h1': case 'h2': case 'h3': case 'h4': case 'h5': case 'h6': return 'heading';
      case 'img': return norm(el.getAttribute('alt')) ? 'img' : null;
      case 'dialog': return 'dialog';
      case 'output': return 'status';
    }
    const ce = el.getAttribute('contenteditable');
    if (ce !== null && ce !== 'false' && el.isContentEditable) return 'textbox';
    return null;
  }
  function roleOf(el) {
    const ex = (el.getAttribute('role') || '').trim().split(/\s+/)[0].toLowerCase();
    if (ex && !['none','presentation','generic'].includes(ex)) return ex;
    return implicitRole(el);
  }
  function visible(el) {
    if (typeof el.checkVisibility === 'function') {
      if (!el.checkVisibility({checkOpacity: true, checkVisibilityCSS: true})) return false;
    } else { const cs = getComputedStyle(el); if (cs.display === 'none' || cs.visibility !== 'visible') return false; }
    const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0;
  }
  function inView(el) {
    const r = el.getBoundingClientRect();
    return r.bottom > 0 && r.right > 0 && r.top < window.innerHeight && r.left < window.innerWidth;
  }
  function textOfIds(ids) {
    return ids.split(/\s+/).map((id) => { const e = id && document.getElementById(id); return e ? (e.innerText || e.textContent || '') : ''; }).join(' ');
  }
  function labelText(el) {
    if (!el.labels || !el.labels.length) return '';
    return Array.from(el.labels).map((l) => { const c = l.cloneNode(true);
      c.querySelectorAll('input,select,textarea,button').forEach((x) => x.remove()); return c.textContent || ''; }).join(' ');
  }
  function accName(el) {
    const lb = el.getAttribute('aria-labelledby');
    if (lb) { const t = norm(textOfIds(lb)); if (t) return clip(t, 160); }
    const al = norm(el.getAttribute('aria-label')); if (al) return clip(al, 160);
    const tag = el.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
      const type = (el.type || '').toLowerCase();
      if (['button','submit','reset'].includes(type)) return clip(el.value || (type === 'submit' ? 'Submit' : type === 'reset' ? 'Reset' : ''), 160);
      const lt = norm(labelText(el)); if (lt) return clip(lt, 160);
      const ph = norm(el.getAttribute('placeholder')); if (ph) return clip(ph, 160);
      const ti = norm(el.getAttribute('title')); if (ti) return clip(ti, 160);
      return clip(el.getAttribute('name') || el.id || '', 160);
    }
    if (tag === 'IMG') return clip(el.getAttribute('alt') || '', 160);
    const t = norm(el.innerText || el.textContent); if (t) return clip(t, 160);
    return clip(el.getAttribute('title') || el.getAttribute('value') || '', 160);
  }
  function refOf(el, assign) { let r = el.getAttribute(A); if (!r && assign) { r = 'e' + (next++); el.setAttribute(A, r); } return r; }
  function describe(el) {
    const tag = el.tagName.toLowerCase(), type = (el.getAttribute('type') || '').toLowerCase();
    const d = {role: roleOf(el) || 'clickable', name: accName(el), tag};
    if (type) d.type = type;
    if (tag === 'input' || tag === 'textarea') {
      if (type === 'checkbox' || type === 'radio') d.checked = !!el.checked;
      else if (type === 'password') d.value = el.value ? '•'.repeat(Math.min(8, el.value.length)) : '';
      else if (type === 'file') d.value = Array.from(el.files || []).map((f) => f.name).join(', ');
      else d.value = clip(el.value, 120);
      if (el.placeholder) d.placeholder = clip(el.placeholder, 80);
      if (el.required) d.required = true;
    } else if (tag === 'select') {
      d.value = Array.from(el.selectedOptions).map((x) => norm(x.label || x.textContent)).join(', ');
      d.options = Array.from(el.options).slice(0, 30).map((x) => clip(x.label || x.textContent, 60));
    } else if (el.isContentEditable) d.value = clip(el.innerText, 120);
    const ac = el.getAttribute('aria-checked'); if (ac !== null && d.checked === undefined) d.checked = ac === 'true';
    for (const s of ['expanded','pressed','selected']) { const v = el.getAttribute('aria-' + s); if (v !== null) d[s] = v === 'true'; }
    if (el.disabled || el.getAttribute('aria-disabled') === 'true') d.disabled = true;
    if (tag === 'a' && el.href) { d.href = el.href.length > o.maxUrl ? el.href.slice(0, o.maxUrl) : el.href;
      if (el.href.length > o.maxUrl) d.href_truncated = el.href.length; }
    const r = el.getBoundingClientRect(); d.box = [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)];
    d.in_viewport = inView(el);
    return d;
  }
  if (o.mode === 'status') {
    const el = o.el || (o.ref ? document.querySelector('[' + A + '="' + o.ref + '"]') : null);
    if (!el || !el.isConnected) return {found: false, next, doc: window.__camlive_doc};
    const d = describe(el); d.found = true; d.visible = visible(el); d.ref = refOf(el, true); d.next = next;
    d.doc = window.__camlive_doc; d.editable = el.isContentEditable || ((el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') && !el.readOnly);
    return d;
  }
  const all = Array.from(document.querySelectorAll(SEL)).filter(visible);
  const elements = [];
  for (const el of all) {
    if (elements.length >= o.maxElements) break;
    const d = describe(el); d.ref = refOf(el, true);
    if (el.form) d.form = Array.prototype.indexOf.call(document.forms, el.form);
    elements.push(d);
  }
  const forms = Array.from(document.forms).slice(0, 20).map((f, i) => ({index: i, id: f.id || '',
    name: clip(norm(f.getAttribute('aria-label')) || f.getAttribute('name') || f.id || '', 120), action: clipUrl(f.action || ''),
    method: (f.method || 'get').toLowerCase(),
    fields: Array.from(f.elements).map((e) => e.getAttribute && e.getAttribute(A)).filter(Boolean)}));
  const links = elements.filter((e) => e.role === 'link' && e.href).slice(0, 80).map((e) => ({ref: e.ref, text: e.name, href: e.href}));
  const tables = Array.from(document.querySelectorAll('table')).filter(visible).slice(0, o.maxTables).map((t) => {
    const cells = (tr) => Array.from(tr.children).filter((c) => c.tagName === 'TH' || c.tagName === 'TD').slice(0, o.maxCols).map((c) => clip(c.innerText || c.textContent, 80));
    let headers = t.tHead && t.tHead.rows.length ? cells(t.tHead.rows[t.tHead.rows.length - 1]) : [];
    let trs = Array.from(t.rows).filter((tr) => !(t.tHead && t.tHead.contains(tr)));
    if (!headers.length && trs.length && Array.from(trs[0].children).every((c) => c.tagName === 'TH') && trs[0].children.length > 1) { headers = cells(trs[0]); trs = trs.slice(1); }
    const width = Math.max(1, Math.min(o.maxCols, headers.length || (trs[0] ? trs[0].children.length : 1)));
    const nr = Math.max(1, Math.min(o.maxRows, Math.floor(o.maxCells / width)));
    return {caption: t.caption ? clip(t.caption.innerText, 80) : clip(t.getAttribute('aria-label'), 80), headers,
            rows: trs.slice(0, nr).map(cells), n_rows: trs.length,
            n_cols: Math.max(headers.length, ...trs.slice(0, 5).map((tr) => tr.children.length))};
  });
  const kv = [];
  Array.from(document.querySelectorAll('dl')).filter(visible).forEach((dl) => { let label = null;
    for (const c of dl.children) { if (c.tagName === 'DT') label = clip(c.innerText, 80); else if (c.tagName === 'DD' && label && kv.length < 60) kv.push({label, value: clip(c.innerText, 80)}); } });
  const outline = [];
  const OSEL = 'h1,h2,h3,h4,h5,h6,[role=heading],nav,main,header,footer,aside,form,dialog,section[aria-label],[role=dialog],[role=alertdialog],[role=alert],[role=status],[role=navigation],[role=main],[role=search],[role=tablist],output,table,canvas,iframe';
  for (const el of document.querySelectorAll(OSEL)) {
    if (outline.length >= 80) break;
    if (!visible(el)) continue;
    const tag = el.tagName.toLowerCase(), role = roleOf(el) || tag;
    const e = {kind: role};
    if (/^h[1-6]$/.test(tag)) { e.kind = 'heading'; e.level = Number(tag[1]); e.text = clip(el.innerText, 100); }
    else if (['alert','status','dialog','alertdialog'].includes(role) || tag === 'output') { e.name = norm(el.getAttribute('aria-label')); e.text = clip(el.innerText, 160); }
    else if (tag === 'table') { e.name = el.caption ? clip(el.caption.innerText, 80) : ''; e.rows = el.rows.length; }
    else if (tag === 'canvas' || tag === 'iframe') { const r = el.getBoundingClientRect(); e.box = [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)]; if (tag === 'iframe') e.src = clipUrl(el.src || ''); }
    else { e.name = norm(el.getAttribute('aria-label')) || el.id || ''; }
    outline.push(e);
  }
  const raw = document.body ? (document.body.innerText || '') : '';
  const text = raw.split('\n').map(norm).filter(Boolean).join('\n');
  const ae = document.activeElement;
  return {url: clipUrl(location.href), title: clip(document.title, o.maxTitle), doc: window.__camlive_doc, next,
          text: text.length > o.maxText ? text.slice(0, o.maxText) : text, text_len: text.length,
          elements, elements_total: all.length, forms, links, tables, kv, outline,
          canvases: document.querySelectorAll('canvas').length,
          focused: ae && ae.getAttribute ? (ae.getAttribute(A) || '') : '',
          scroll: {x: Math.round(window.scrollX), y: Math.round(window.scrollY),
                   max_y: Math.max(0, Math.round(document.documentElement.scrollHeight - window.innerHeight))},
          viewport: {w: window.innerWidth, h: window.innerHeight}};
}
"""

_MUT_JS = ("if(!window.__camlive_mo){window.__camlive_last=performance.now();window.__camlive_mo=new MutationObserver("
           "()=>{window.__camlive_last=performance.now();});window.__camlive_mo.observe(document,{subtree:true,"
           "childList:true,attributes:true,characterData:true});}")
_SETTLE_JS = ("(o)=>new Promise((res)=>{" + _MUT_JS + "const t0=performance.now();const tick=()=>{const n=performance"
              ".now();if(n-window.__camlive_last>=o.quiet||n-t0>=o.max)res(n-t0);else setTimeout(tick,25);};tick();})")
_NET_TRANSIENT = ("ERR_NAME_NOT_RESOLVED", "ERR_CONNECTION", "ERR_INTERNET_DISCONNECTED", "ERR_TIMED_OUT",
                  "ERR_NETWORK", "ERR_ADDRESS_UNREACHABLE", "ERR_EMPTY_RESPONSE", "ERR_SSL", "ERR_PROXY",
                  "ERR_TUNNEL", "ERR_SOCKET")


# ---------------------------------------------------------------------------------------------------------------
# Which Chromium binary: the service uids cannot reach the operator's own Playwright download (under /root), so a
# shared copy is installed once by the operator and found here.
# ---------------------------------------------------------------------------------------------------------------
#: where the operator's shared Chromium lives (python -m cam_live.harness.browser --install-shared)
SHARED_BROWSERS_DIR = os.environ.get("CAM_LIVE_BROWSERS") or os.path.join(sys.prefix, "share", "cam_live",
                                                                            "ms-playwright")
_CHROME_GLOBS = ("chromium-*/chrome-linux*/chrome", "chromium-*/chrome-linux/chrome")


#: Chromium features switched off in addition to Playwright's own list: background requests a page did not cause
EXTRA_DISABLED_FEATURES = ("AutofillServerCommunication",)
_PW_FEATURES: Optional[List[str]] = None


def playwright_disabled_features() -> List[str]:
    """The ``--disable-features`` list Playwright itself passes to Chromium (read from the installed driver), or []
    when it cannot be read exactly. Chromium honours only the last ``--disable-features`` switch, so an extra
    feature can only be switched off by repeating Playwright's list."""
    global _PW_FEATURES
    if _PW_FEATURES is not None:
        return _PW_FEATURES
    _PW_FEATURES = []
    try:
        import playwright
        base = os.path.join(os.path.dirname(playwright.__file__), "driver", "package", "lib")
        for rel in ("coreBundle.js", os.path.join("server", "chromium", "chromiumSwitches.js")):
            path = os.path.join(base, rel)
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                src = f.read()
            m = re.search(r"disabledFeatures = \[(.*?)\]\.filter\(Boolean\)", src, re.S)
            if not m:
                continue
            body = re.sub(r"//[^\n]*", "", m.group(1))
            items = re.findall(r'"([A-Za-z0-9_]+)"', body)
            rest = re.sub(r'"[A-Za-z0-9_]+"', "", body).replace(",", "").strip()
            if items and not rest:                       # only plain names: the list is known exactly
                _PW_FEATURES = items
            break
    except Exception:  # noqa: BLE001 — without the exact list nothing extra is switched off
        _PW_FEATURES = []
    return _PW_FEATURES


def _revision_key(path: str) -> int:
    m = re.search(r"chromium-(\d+)", path)
    return int(m.group(1)) if m else 0


def _in_browsers_dir(d: str) -> List[str]:
    found: List[str] = []
    for g in _CHROME_GLOBS:
        found += glob.glob(os.path.join(d, g))
    return sorted(set(found), key=_revision_key, reverse=True)


def executable_by(path: str, uid: Optional[int] = None, gid: Optional[int] = None) -> bool:
    """Whether ``uid`` (default: this process) can execute ``path``: every directory on the way searchable and the
    file executable, judged from the permission bits (root's own access rights are not taken as the user's)."""
    if not path or not os.path.isfile(path):
        return False
    if uid is None:
        return os.access(path, os.X_OK)
    if uid == 0:
        return True
    groups = {gid} if gid is not None else set()
    try:
        name = pwd.getpwuid(uid).pw_name
        groups |= {g for g in os.getgrouplist(name, gid if gid is not None else pwd.getpwuid(uid).pw_gid)}
    except KeyError:
        pass

    def allowed(st: os.stat_result, bit: int) -> bool:
        if st.st_uid == uid:
            return bool(st.st_mode & (bit << 6))
        if st.st_gid in groups:
            return bool(st.st_mode & (bit << 3))
        return bool(st.st_mode & bit)

    real = os.path.realpath(path)
    cur = os.path.dirname(real)
    while True:
        if not allowed(os.stat(cur), 0o1):
            return False
        if cur == "/":
            break
        cur = os.path.dirname(cur)
    return allowed(os.stat(real), 0o1)


def searchable_by(path: str, uid: int, gid: Optional[int] = None) -> str:
    """The first directory on the way to ``path`` (inclusive) that ``uid`` cannot search, or ""."""
    groups = {gid} if gid is not None else set()
    try:
        pw = pwd.getpwuid(uid)
        groups |= set(os.getgrouplist(pw.pw_name, pw.pw_gid))
    except KeyError:
        pass
    cur = os.path.realpath(path)
    chain = []
    while True:
        chain.append(cur)
        if cur == "/":
            break
        cur = os.path.dirname(cur)
    for d in reversed(chain):
        st = os.stat(d)
        bit = 0o100 if st.st_uid == uid else 0o010 if st.st_gid in groups else 0o001
        if uid != 0 and not st.st_mode & bit:
            return d
    return ""


def chromium_candidates(bundled: str = "") -> List[str]:
    """Chromium binaries in order of preference: ``$CAM_LIVE_CHROMIUM``, ``$PLAYWRIGHT_BROWSERS_PATH``, the shared
    install (:data:`SHARED_BROWSERS_DIR`), then Playwright's own download for this ``HOME``."""
    out: List[str] = []
    if os.environ.get("CAM_LIVE_CHROMIUM"):
        out.append(os.environ["CAM_LIVE_CHROMIUM"])
    pbp = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    if pbp and pbp != "0":
        out += _in_browsers_dir(pbp)
    out += _in_browsers_dir(SHARED_BROWSERS_DIR)
    if bundled:
        out.append(bundled)
    return list(dict.fromkeys(out))


def find_chromium(uid: Optional[int] = None, gid: Optional[int] = None, bundled: str = "") -> str:
    """The first candidate Chromium that ``uid`` can execute, or "" when there is none."""
    for c in chromium_candidates(bundled):
        if executable_by(c, uid, gid):
            return c
    return ""


def install_shared_chromium(dest: str = "", source: str = "") -> str:
    """Operator step (run once, as the user whose Playwright download exists): copy that Chromium revision to
    ``dest`` (default :data:`SHARED_BROWSERS_DIR`) readable and executable by every service uid. Returns the path
    of the binary in the shared copy. Nothing is downloaded."""
    dest = os.path.abspath(dest or SHARED_BROWSERS_DIR)
    exe = source or BrowserExecutor._bundled_chromium()
    if not exe or not os.path.isfile(exe):
        raise FileNotFoundError(f"no Playwright Chromium to copy (looked at {exe!r}); run 'playwright install chromium'")
    rev_dir = os.path.dirname(os.path.dirname(exe))                       # .../chromium-<rev>
    target = os.path.join(dest, os.path.basename(rev_dir))
    final = os.path.join(target, os.path.relpath(exe, rev_dir))
    if not executable_by(final, 65534, 65534):                            # absent or not shared yet
        os.makedirs(dest, mode=0o755, exist_ok=True)
        tmp = target + f".tmp.{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(rev_dir, tmp, symlinks=True)
        os.chmod(tmp, 0o755)
        for root, dirs, files in os.walk(tmp):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o755)
            for f in files:
                fp = os.path.join(root, f)
                if os.path.islink(fp):
                    continue
                mode = os.stat(fp).st_mode
                os.chmod(fp, (mode | 0o444 | (0o111 if mode & 0o100 else 0)) & ~0o022)
        shutil.rmtree(target, ignore_errors=True)
        os.replace(tmp, target)
    if not executable_by(final, 65534, 65534):
        raise PermissionError(f"{final} was copied but other users cannot reach it: a directory above {dest} is not "
                              f"searchable by others; choose a shared location (--dest)")
    return final


@dataclass
class BrowserConfig:
    headless: bool = True
    viewport: Tuple[int, int] = (1280, 800)
    identities: Tuple[str, ...] = ("default",)      # authorized identities = persistent profiles that may exist
    executable_path: str = ""                       # "" = Playwright's bundled Chromium
    extra_args: Tuple[str, ...] = ()
    run_as_user: str = "auto"                       # "auto" | "" (own uid) | a unix user name
    nav_timeout_s: float = 30.0
    op_timeout_s: float = 10.0
    settle_quiet_ms: int = 120
    settle_max_ms: int = 1500
    max_elements: int = 150
    max_page_text: int = 20000
    max_a11y_chars: int = 6000
    max_download_bytes: int = 64 * 1024 * 1024
    research_read_only: bool = True
    cdp_endpoints: Dict[str, str] = field(default_factory=dict)   # session name (without "cdp:") -> endpoint URL
    origin_cache_s: float = 30.0


class _Fail(Exception):
    def __init__(self, status: str, failure_class: str, detail: str):
        super().__init__(detail)
        self.status, self.failure_class, self.detail = status, failure_class, detail


def _norm_url(url: str) -> str:
    u = urlsplit(url)
    return urlunsplit((u.scheme, u.netloc, u.path or "/", u.query, ""))


def _origin(url: str) -> str:
    u = urlsplit(url)
    return f"{u.scheme}://{u.netloc}"


def denial_class(dec: Optional[AuthorizationDecision]) -> Tuple[str, str, str]:
    """(status, failure class, detail) for a non-ALLOW decision (see :func:`refusal_class`)."""
    if dec is not None and dec.verdict == AuthVerdict.NEEDS_APPROVAL:
        return ResultStatus.NEEDS_APPROVAL, FailureClass.NONE, "; ".join(dec.reasons) or "needs approval"
    fc = refusal_class(getattr(dec, "reasons", []) or [], getattr(dec, "policy_ids", []) or [])
    return ResultStatus.DENIED, fc, "; ".join(getattr(dec, "reasons", []) or []) or "denied by the authorization layer"


SECRET_REF = "secret:"


def secret_name(ref: Any) -> str:
    """The vault name in a secret reference. Arguments name a stored secret as ``secret:<name>`` (the form the
    authorization service accepts in a credential field); the vault knows it as ``<name>``."""
    ref = str(ref or "").strip()
    return ref[len(SECRET_REF):] if ref.startswith(SECRET_REF) else ref


def broker_get(secrets: Any, name: str, op: PrimitiveOp, action: ActionSpec, decision: Any = None,
               destination: str = "") -> str:
    """Fetch a secret at the moment of use from the broker handle in ``ExecContext.secrets``. Supports the
    authorization workstream's ``get(name, action, decision, destination)`` and simpler ``get(name)`` handles."""
    import inspect
    name = secret_name(name)
    if secrets is None:
        raise _Fail(ResultStatus.DENIED, FailureClass.MISSING_PERMISSION, "no secret broker is attached")
    fn = next((getattr(secrets, m) for m in ("get", "get_secret", "resolve") if callable(getattr(secrets, m, None))), None)
    if fn is None:
        raise _Fail(ResultStatus.DENIED, FailureClass.MISSING_PERMISSION, "secret broker has no usable accessor")
    try:
        n = len([p for p in inspect.signature(fn).parameters.values()
                 if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])
    except (TypeError, ValueError):
        n = 1
    args = {4: (name, action, decision, destination), 3: (name, op, action), 2: (name, action)}.get(n, (name,))
    try:
        v = fn(*args)
    except Exception as e:  # noqa: BLE001 — the broker refused; never echo its message into an observation
        raise _Fail(ResultStatus.DENIED, FailureClass.MISSING_PERMISSION,
                    f"the secret broker refused {name!r}: {type(e).__name__}")
    if isinstance(v, (bytes, bytearray)):
        v = bytes(v).decode()
    if not isinstance(v, str) or not v:
        raise _Fail(ResultStatus.DENIED, FailureClass.MISSING_PERMISSION, f"secret {name!r} is unavailable")
    return v


class _Loop:
    """One asyncio loop on a private thread; every Playwright call of the executor runs there."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True, name="camlive-browser")
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def call(self, coro: Awaitable[Any], timeout: Optional[float] = None) -> Any:
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return fut.result(timeout)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            raise TimeoutError("browser operation timed out")

    def stop(self) -> None:
        if self.thread.is_alive():
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=10)


class _Tab:
    def __init__(self, tab_id: str, page: Any):
        self.id, self.page = tab_id, page
        self.prev_state: Optional[Dict[str, Any]] = None
        self.prev_obs_id = ""
        self.sigs: "collections.OrderedDict[str, Dict[str, Tuple[str, str]]]" = collections.OrderedDict()
        self.dialogs: List[str] = []


class _Session:
    def __init__(self, name: str, kind: str):
        self.name, self.kind = name, kind
        self.context: Any = None
        self.browser: Any = None
        self.tabs: Dict[str, _Tab] = {}
        self.active = ""
        self.lock = threading.RLock()
        self.ctx: Optional[ExecContext] = None        # context of the action in progress (or the last one)
        self.action: Optional[ActionSpec] = None
        self.approved: Dict[str, int] = {}            # explicitly checked URLs awaiting their request
        self.checked: set = set()                     # URLs approved during the current action
        self.origins_ok: Dict[str, bool] = {}         # origins navigated to during the current action
        self.origin_cache: Dict[str, Tuple[bool, float]] = {}
        self.blocked: List[Dict[str, Any]] = []
        # handed from the route layer to the network layer: requests that were just authorized (method, url) ->
        # [count, expiry], and host:port tunnels -> expiry
        self.net_tokens: Dict[Tuple[str, str], List[float]] = {}
        self.tunnel_ok: Dict[str, float] = {}
        self.guard: Optional["_NetGuard"] = None
        self.read_only = False
        self.n_tabs = 0
        self.expect_download = False
        self.profile_dir = ""
        self.restore: List[Dict[str, Any]] = []


_HOP_BY_HOP = ("connection", "keep-alive", "proxy-connection", "proxy-authorization", "proxy-authenticate", "te",
               "trailer")


def _non_global(ip: str) -> bool:
    try:
        return not ipaddress.ip_address(ip).is_global
    except ValueError:
        return True
_NO_BODY_STATUS = (204, 304)


class _NetGuard:
    """The browser's only way out: a forward HTTP proxy on 127.0.0.1 that checks every request against the policy
    before forwarding it (one per browser session; password-protected so no other local process can use it).

    Plain HTTP requests are seen whole (method and URL); HTTPS and WebSocket traffic arrives as ``CONNECT
    host:port`` tunnels and is judged per host and port. Each client connection carries exactly one request (the
    response is sent with ``Connection: close``), so nothing can be smuggled behind a checked request."""

    MAX_HEAD = 1 << 20

    def __init__(self, executor: "BrowserExecutor", sess: "_Session"):
        self.x, self.sess = executor, sess
        self.user, self.password = "cam", _secrets.token_urlsafe(24)
        self._auth = "Basic " + base64.b64encode(f"{self.user}:{self.password}".encode()).decode()
        self.server: Any = None
        self.port = 0
        self.stats = {"requests": 0, "tunnels": 0, "forwarded": 0, "blocked": 0, "unauthenticated": 0}

    async def start(self) -> "_NetGuard":
        self.server = await asyncio.start_server(self._client, "127.0.0.1", 0, limit=self.MAX_HEAD)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    def playwright_proxy(self) -> Dict[str, str]:
        # "<-loopback>" removes Chromium's implicit bypass of localhost: loopback destinations are checked as well
        return {"server": f"http://127.0.0.1:{self.port}", "bypass": "<-loopback>", "username": self.user,
                "password": self.password}

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
            try:
                await asyncio.wait_for(self.server.wait_closed(), 2)
            except Exception:  # noqa: BLE001 — open tunnels are torn down with the browser
                pass

    @staticmethod
    async def _reply(w: Any, status: str, extra: str = "", body: bytes = b"") -> None:
        try:
            w.write(f"HTTP/1.1 {status}\r\n{extra}Content-Type: text/plain\r\nContent-Length: {len(body)}\r\n"
                    f"Connection: close\r\n\r\n".encode("latin-1") + body)
            await w.drain()
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    async def _pipe(r: Any, w: Any, idle_s: float = 300.0) -> None:
        try:
            while True:
                data = await asyncio.wait_for(r.read(65536), idle_s)
                if not data:
                    break
                w.write(data)
                await w.drain()
        except Exception:  # noqa: BLE001 — either side went away
            pass
        finally:
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass

    async def _client(self, reader: Any, writer: Any) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 60)
        except Exception:  # noqa: BLE001 — oversized, incomplete or idle: nothing is forwarded
            return await self._reply(writer, "400 Bad Request")
        lines = head.decode("latin-1").split("\r\n")
        parts = lines[0].split(" ")
        if len(parts) != 3:
            return await self._reply(writer, "400 Bad Request")
        method, target, version = parts
        headers: List[Tuple[str, str]] = []
        for ln in lines[1:]:
            if not ln:
                continue
            if ":" not in ln:
                return await self._reply(writer, "400 Bad Request")
            k, v = ln.split(":", 1)
            headers.append((k.strip(), v.strip()))
        hmap = {k.lower(): v for k, v in headers}
        if not hmac.compare_digest(hmap.get("proxy-authorization", ""), self._auth):
            self.stats["unauthenticated"] += 1
            return await self._reply(writer, "407 Proxy Authentication Required",
                                     'Proxy-Authenticate: Basic realm="cam-live"\r\n')
        self.stats["requests"] += 1
        if method == "CONNECT":
            return await self._tunnel(target, reader, writer)
        return await self._forward(method, target, version, headers, hmap, reader, writer)

    @staticmethod
    async def _resolve(host: str, port: int) -> Tuple[List[str], bool]:
        """(addresses to connect to, IPv4 first; whether ``host`` was a name). The name is resolved once, here; the
        connection goes to exactly these addresses, so what was checked is what is reached."""
        h = host.strip("[]")
        try:
            return [str(ipaddress.ip_address(h))], False
        except ValueError:
            pass
        infos = await asyncio.get_running_loop().getaddrinfo(h, port, type=socket.SOCK_STREAM)
        addrs = list(dict.fromkeys(i[4][0] for i in sorted(infos, key=lambda i: i[0] != socket.AF_INET)))
        return addrs, True

    @staticmethod
    async def _connect(addrs: List[str], port: int) -> Tuple[Any, Any]:
        last: Optional[BaseException] = None
        for ip in addrs:
            try:
                return await asyncio.wait_for(asyncio.open_connection(ip, port), 15)
            except Exception as e:  # noqa: BLE001 — the next address of the same, checked answer
                last = e
        raise OSError(f"no address of the destination accepted the connection: {last!r}")

    async def _tunnel(self, target: str, reader: Any, writer: Any) -> None:
        m = re.match(r"^(\[[0-9A-Fa-f:.]+\]|[^:\[\]/\s]+):(\d{1,5})$", target)
        if not m:
            return await self._reply(writer, "400 Bad Request")
        host, port = m.group(1), int(m.group(2))
        try:
            addrs, named = await asyncio.wait_for(self._resolve(host, port), 15)
        except Exception:  # noqa: BLE001
            return await self._reply(writer, "502 Bad Gateway")
        ok, why = await self.x._net_decide(self.sess, "CONNECT", f"https://{target}/", tunnel=target,
                                           peer_ips=addrs if named else [])
        if not ok:
            self.stats["blocked"] += 1
            return await self._reply(writer, "403 Forbidden", body=f"blocked by the authorization layer: {why}".encode()[:300])
        try:
            ur, uw = await self._connect(addrs, port)
        except Exception:  # noqa: BLE001
            return await self._reply(writer, "502 Bad Gateway")
        self.stats["tunnels"] += 1
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
        await asyncio.gather(self._pipe(reader, uw), self._pipe(ur, writer))

    async def _forward(self, method: str, target: str, version: str, headers: List[Tuple[str, str]],
                       hmap: Dict[str, str], reader: Any, writer: Any) -> None:
        u = urlsplit(target)
        if u.scheme != "http" or not u.hostname:
            return await self._reply(writer, "400 Bad Request")
        try:
            port = u.port or 80
        except ValueError:
            return await self._reply(writer, "400 Bad Request")
        dest = hmap.get("sec-fetch-dest", "")
        try:
            addrs, named = await asyncio.wait_for(self._resolve(u.hostname, port), 15)
        except Exception:  # noqa: BLE001 — a name that does not resolve: like a failed connection
            addrs, named = [], True
        ok, why = await self.x._net_decide(self.sess, method, target, document=dest in ("document", "iframe", "frame"),
                                           main=dest == "document", resource=dest, peer_ips=addrs if named else [])
        if not ok:
            self.stats["blocked"] += 1
            return await self._reply(writer, "403 Forbidden", body=f"blocked by the authorization layer: {why}".encode()[:300])
        try:
            ur, uw = await self._connect(addrs, port)
        except Exception:  # noqa: BLE001 — the destination is down: the browser sees what it would see without
            try:                                # a proxy (an empty response, a transient network failure)
                writer.close()
            except Exception:  # noqa: BLE001
                pass
            return
        upgrade = "upgrade" in hmap
        keep = [(k, v) for k, v in headers if k.lower() not in _HOP_BY_HOP]
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        conn = "Connection: Upgrade\r\n" if upgrade else "Connection: close\r\n"
        uw.write((f"{method} {path} {version}\r\n" + "".join(f"{k}: {v}\r\n" for k, v in keep) + conn + "\r\n")
                 .encode("latin-1"))
        try:
            if upgrade:
                await uw.drain()
                self.stats["forwarded"] += 1
                await asyncio.gather(self._pipe(reader, uw), self._pipe(ur, writer))
                return
            await self._copy_body(reader, uw, hmap)
            await uw.drain()
            self.stats["forwarded"] += 1
            rhead = await asyncio.wait_for(ur.readuntil(b"\r\n\r\n"), 120)
            rlines = rhead.decode("latin-1").split("\r\n")
            status = int(rlines[0].split(" ")[1]) if len(rlines[0].split(" ")) > 1 else 0
            rkeep = [ln for ln in rlines[1:] if ln and ln.split(":", 1)[0].strip().lower() not in _HOP_BY_HOP]
            writer.write((rlines[0] + "\r\n" + "".join(f"{ln}\r\n" for ln in rkeep) + "Connection: close\r\n\r\n")
                         .encode("latin-1"))
            await writer.drain()
            length = next((ln.split(":", 1)[1].strip() for ln in rkeep if ln.lower().startswith("content-length:")), None)
            if method == "HEAD" or status in _NO_BODY_STATUS or 100 <= status < 200:
                return
            if length is not None and length.isdigit():
                left = int(length)
                while left > 0:
                    chunk = await asyncio.wait_for(ur.read(min(65536, left)), 300)
                    if not chunk:
                        break
                    writer.write(chunk)
                    await writer.drain()
                    left -= len(chunk)
            else:
                await self._pipe(ur, writer)
        except Exception:  # noqa: BLE001 — the browser sees a broken response, nothing else
            pass
        finally:
            for w in (uw, writer):
                try:
                    w.close()
                except Exception:  # noqa: BLE001
                    pass

    @staticmethod
    async def _copy_body(reader: Any, uw: Any, hmap: Dict[str, str]) -> None:
        if "chunked" in hmap.get("transfer-encoding", "").lower():
            while True:
                line = await asyncio.wait_for(reader.readuntil(b"\r\n"), 120)
                uw.write(line)
                size = int(line.split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while True:                     # trailers up to the empty line
                        t = await asyncio.wait_for(reader.readuntil(b"\r\n"), 120)
                        uw.write(t)
                        if t == b"\r\n":
                            return
                uw.write(await asyncio.wait_for(reader.readexactly(size + 2), 120))
        n = hmap.get("content-length", "0").strip()
        left = int(n) if n.isdigit() else 0
        while left > 0:
            chunk = await asyncio.wait_for(reader.read(min(65536, left)), 120)
            if not chunk:
                return
            uw.write(chunk)
            left -= len(chunk)


_GUARD_JS = ("(() => { for (const k of ['WebTransport']) { try { Object.defineProperty(window, k, {value: undefined, "
             "configurable: false, writable: false}); } catch (e) {} } })();")
_NO_SW_JS = ("(() => { try { if (navigator.serviceWorker) Object.defineProperty(navigator.serviceWorker, 'register', "
             "{value: () => Promise.reject(new DOMException('service workers are disabled', 'SecurityError'))}); } "
             "catch (e) {} })();")


def _tool(name: str, desc: str, props: Dict[str, Any], required: Tuple[str, ...] = (), side: str = "none",
          scope: str = "browser.read", seconds: float = 0.5) -> ToolDescriptor:
    common = {"session": {"type": "string", "description": "browser session name"},
              "tab": {"type": "string", "description": "tab id; default is the active tab"}}
    return ToolDescriptor(name=name, modality=Modality.STRUCTURED_BROWSER, description=desc,
                          arg_schema={"type": "object", "properties": {**props, **common},
                                      "required": list(required), "additionalProperties": False},
                          level=Level.PRIMITIVE, auth_scope=scope, side_effects=side,
                          est_cost=Cost(seconds=seconds, primitive_ops=1))


_TARGET = {"ref": {"type": "string", "description": "element ref from an observation, e.g. e12"},
           "obs_id": {"type": "string", "description": "id of the observation the ref was read from"},
           "role": {"type": "string", "description": "accessible role for a semantic locator"},
           "name": {"type": "string", "description": "accessible name / label / visible text"},
           "selector": {"type": "string", "description": "CSS selector (last-resort locator)"},
           "nth": {"type": "integer", "minimum": 0, "description": "index among several matches"}}

TOOLS: List[ToolDescriptor] = [
    _tool("browser.navigate", "Load a URL in a tab and return the resulting page observation.",
          {"url": {"type": "string"}, "wait_until": {"type": "string", "enum": ["domcontentloaded", "load", "networkidle"]}},
          ("url",), "external_read", seconds=2.0),
    _tool("browser.open_tab", "Open a new tab, optionally loading a URL in it.", {"url": {"type": "string"}},
          (), "external_read", seconds=2.0),
    _tool("browser.close_tab", "Close a tab.", {}, ()),
    _tool("browser.select_tab", "Make a tab the active tab and return its page observation.", {}, ("tab",)),
    _tool("browser.inspect", "Return the full structured observation of the page: text digest, accessibility tree, "
          "interactive elements with refs, forms, links, tables, tabs. Optionally also a screenshot.",
          {"screenshot": {"type": "boolean"}, "full": {"type": "boolean"}}, ()),
    _tool("browser.read", "Return the visible text of the page or of one element, with paging by character offset.",
          {**_TARGET, "offset": {"type": "integer", "minimum": 0}, "max_chars": {"type": "integer", "minimum": 1,
                                                                                "maximum": 20000}}, ()),
    _tool("browser.click", "Click an element identified by ref or by semantic locator.",
          {**_TARGET, "double": {"type": "boolean"}, "button": {"type": "string", "enum": ["left", "right", "middle"]}},
          (), "external_write", "browser.write"),
    _tool("browser.fill", "Replace the content of a form field with text, or type a stored secret into a password "
          "field by reference ('secret:<name>'; the value is fetched at the moment of use and never returned).",
          {**_TARGET, "text": {"type": "string"},
           "secret": {"type": "string", "description": "reference of a stored secret: 'secret:<name>'"}},
          (), "local", "browser.write"),
    _tool("browser.select_option", "Choose an option of a select element by label or value.",
          {**_TARGET, "label": {"type": "string"}, "value": {"type": "string"}}, (), "local", "browser.write"),
    _tool("browser.press", "Press a key or key combination, on an element or on the page.",
          {**_TARGET, "key": {"type": "string"}}, ("key",), "external_write", "browser.write"),
    _tool("browser.scroll", "Scroll the page by pixels, to its top or bottom, or scroll an element into view.",
          {**_TARGET, "dx": {"type": "number"}, "dy": {"type": "number"}, "to": {"type": "string",
                                                                              "enum": ["top", "bottom"]}}, ()),
    _tool("browser.upload", "Attach a file from the workspace to a file input.",
          {**_TARGET, "path": {"type": "string", "description": "path inside the workspace"}}, ("path",),
          "external_write", "browser.upload"),
    _tool("browser.download", "Download a resource by URL, or by clicking an element, into the workspace.",
          {**_TARGET, "url": {"type": "string"}, "filename": {"type": "string"}}, (), "external_read",
          "browser.download", seconds=2.0),
    _tool("browser.wait_for", "Wait until text appears or disappears, an element reaches a state, the URL matches, "
          "or the page changes.",
          {"text": {"type": "string"}, "text_gone": {"type": "string"}, "url": {"type": "string"},
           "role": {"type": "string"}, "name": {"type": "string"},
           "state": {"type": "string", "enum": ["visible", "hidden", "attached", "detached"]},
           "change": {"type": "boolean"}, "timeout_s": {"type": "number", "minimum": 0.05, "maximum": 120}}, ()),
    _tool("browser.screenshot", "Capture the viewport (or the full page) as a PNG stored in the run directory.",
          {"full_page": {"type": "boolean"}}, ()),
    _tool("browser.close_session", "Close a research or externally attached session.", {}, ("session",)),
]
_TOOLS = {t.name: t for t in TOOLS}
_NEEDS_TARGET = {"browser.click", "browser.fill", "browser.select_option", "browser.upload"}


class BrowserScreen:
    """Raw screen access to one tab for the visual executor: PNG screenshots, mouse and keyboard.

    Coordinates are CSS pixels of the viewport (device scale factor 1, so they equal screenshot pixels). The
    methods do not ask the authorization layer about the gesture itself — the visual executor must call
    ``ExecContext.check`` for each primitive — but any navigation or request a gesture triggers is still checked
    per URL with the ``ctx`` bound here.
    """

    def __init__(self, executor: "BrowserExecutor", session: str, tab: Optional[str], ctx: Optional[ExecContext],
                 action: Optional[ActionSpec]):
        self._x, self._session, self._tab, self._ctx, self._action = executor, session, tab, ctx, action

    def _run(self, fn: Callable[[Any], Awaitable[Any]]) -> Any:
        x = self._x
        sess = x._session_sync(self._session, self._ctx, self._action)
        with sess.lock:
            if self._ctx is not None:
                sess.ctx, sess.action = self._ctx, self._action or sess.action
            return x._loop.call(x._with_page(sess, self._tab, fn), timeout=x.cfg.nav_timeout_s + 10)

    def screenshot(self) -> bytes:
        return self._run(lambda p: p.screenshot(type="png", full_page=False))

    def size(self) -> Tuple[int, int]:
        v = self._run(lambda p: p.evaluate("() => [window.innerWidth, window.innerHeight]"))
        return int(v[0]), int(v[1])

    def url(self) -> str:
        async def f(p):
            return p.url
        return self._run(f)

    def move(self, x: float, y: float) -> None:
        self._run(lambda p: p.mouse.move(x, y))

    def click(self, x: float, y: float, button: str = "left") -> None:
        self._run(lambda p: p.mouse.click(x, y, button=button))

    def double_click(self, x: float, y: float) -> None:
        self._run(lambda p: p.mouse.dblclick(x, y))

    def drag(self, x0: float, y0: float, x1: float, y1: float, steps: int = 12) -> None:
        async def f(p):
            await p.mouse.move(x0, y0)
            await p.mouse.down()
            await p.mouse.move(x1, y1, steps=max(1, steps))
            await p.mouse.up()
        self._run(f)

    def scroll(self, dx: float, dy: float, x: Optional[float] = None, y: Optional[float] = None) -> None:
        async def f(p):
            if x is not None and y is not None:
                await p.mouse.move(x, y)
            await p.mouse.wheel(dx, dy)
        self._run(f)

    def type(self, text: str, delay_ms: float = 0) -> None:
        self._run(lambda p: p.keyboard.type(text, delay=delay_ms))

    def key(self, combo: str) -> None:
        self._run(lambda p: p.keyboard.press(combo))

    def element_at(self, x: float, y: float) -> Dict[str, Any]:
        """DOM facts under a point (tag, role, name, href) so a proposed coordinate can be validated."""
        async def f(p):
            h = await p.evaluate_handle("([x, y]) => document.elementFromPoint(x, y)", [x, y])
            el = h.as_element()
            return {"found": False} if el is None else await self._x._status(p, handle=el)
        return self._run(f)

    def settle(self) -> None:
        self._run(lambda p: self._x._settle(p))


class BrowserExecutor:
    """``contracts.ModalityExecutor`` for ``Modality.STRUCTURED_BROWSER``."""
    modality = Modality.STRUCTURED_BROWSER

    def __init__(self, run_dir: str, *, clock: Optional[Clock] = None, config: Optional[BrowserConfig] = None,
                 perception: Optional[Perception] = None):
        self.run_dir = os.path.abspath(run_dir)
        self.clock = clock or WallClock()
        self.cfg = config or BrowserConfig()
        self.perception = perception or Perception(self.clock)
        self.workspace = os.path.join(self.run_dir, "workspace")
        self.root = os.path.join(self.run_dir, "browser")
        for d in (self.workspace, os.path.join(self.workspace, "downloads"), self.root,
                  os.path.join(self.root, "profiles"), os.path.join(self.run_dir, "artifacts", "screenshots")):
            os.makedirs(d, exist_ok=True)
        self._loop = _Loop()
        self._pw: Any = None
        self._research_browser: Any = None
        self._sessions: Dict[str, _Session] = {}
        self._guard = threading.RLock()
        self._next_ref = 1
        self._restorable: Dict[str, Any] = {}
        self._closed = False
        self._version = ""
        self._launch = self._plan_launch()

    # ---- launch plan (uid separation) -----------------------------------------------------------------------
    def _plan_launch(self) -> Dict[str, Any]:
        plan = self._plan_uid()
        if not plan["user"] and not plan["exe"]:
            # Chromium runs under this process's own uid: Playwright's own download is used if this uid can run
            # it, otherwise a shared copy (a service uid cannot reach the operator's download)
            bundled = self._bundled_chromium()
            if not executable_by(bundled):
                plan["exe"] = find_chromium(None, None, bundled)
                if not plan["exe"]:
                    plan["reason"] = (plan["reason"] + "; " if plan["reason"] else "") + (
                        f"no Chromium this uid can execute (Playwright's download {bundled or '?'} is not reachable): "
                        f"install a shared copy with 'python -m cam_live.harness.browser --install-shared'")
        return plan

    def _plan_uid(self) -> Dict[str, Any]:
        cfg = self.cfg
        plan: Dict[str, Any] = {"user": "", "uid": os.getuid(), "wrapper": "", "reason": "", "exe": cfg.executable_path}
        want = cfg.run_as_user
        if want == "auto":
            want = SERVICE_USERS["browser"]
            auto = True
        else:
            auto = False
        if not want:
            plan["reason"] = "uid separation disabled by configuration"
            return plan
        try:
            pw = pwd.getpwnam(want)
        except KeyError:
            plan["reason"] = f"user {want!r} does not exist"
            if not auto:
                raise RuntimeError(plan["reason"])
            return plan
        if os.geteuid() != 0:
            plan["reason"] = "not root: cannot switch uid"
            return plan
        if pw.pw_uid == os.getuid():
            plan["reason"] = "already running as the browser user"
            return plan
        setpriv = shutil.which("setpriv")
        if not setpriv:
            plan["reason"] = "setpriv not found"
            return plan
        bundled = self._bundled_chromium()
        exe = cfg.executable_path or find_chromium(pw.pw_uid, pw.pw_gid, bundled) or bundled
        ok = subprocess.run([setpriv, f"--reuid={pw.pw_uid}", f"--regid={pw.pw_gid}", "--clear-groups", "test", "-x",
                             exe], capture_output=True).returncode == 0
        if not ok:
            plan["reason"] = (f"Chromium at {exe} is not executable by {want}; install a shared copy once with "
                              f"'python -m cam_live.harness.browser --install-shared' (or set CAM_LIVE_CHROMIUM / "
                              f"BrowserConfig.executable_path)")
            plan["exe"] = cfg.executable_path
            return plan
        for d in (self.run_dir, self.root):           # the browser uid must be able to traverse to its profile
            st = os.stat(d)
            os.chmod(d, st.st_mode | 0o011)
        blocked = searchable_by(self.root, pw.pw_uid, pw.pw_gid)
        if blocked:
            plan["reason"] = (f"the run directory is not reachable by {want}: {blocked} is not searchable for it; "
                              f"Chromium runs under this process's uid")
            return plan
        wrapper = os.path.join(self.root, f"chromium-as-{want}.sh")
        with open(wrapper, "w") as f:
            f.write("#!/bin/sh\n# generated by cam_live.harness.browser: run Chromium under the browser uid\n"
                    f"export HOME='{self.root}/home'\n"
                    f"exec '{setpriv}' --reuid={pw.pw_uid} --regid={pw.pw_gid} --init-groups --inh-caps=-all -- "
                    f"'{exe}' \"$@\"\n")
        os.chmod(wrapper, 0o755)
        for d in (os.path.join(self.root, "home"), os.path.join(self.root, "profiles"),
                  os.path.join(self.root, "dl")):
            os.makedirs(d, exist_ok=True)
            os.chown(d, pw.pw_uid, pw.pw_gid)
            os.chmod(d, 0o700)
        plan.update(user=want, uid=pw.pw_uid, gid=pw.pw_gid, wrapper=wrapper, exe=exe, reason="")
        return plan

    _bundled: Optional[str] = None

    @staticmethod
    def _bundled_chromium() -> str:
        """Playwright's own Chromium for this process's ``HOME`` (it may not exist)."""
        if BrowserExecutor._bundled is not None:
            return BrowserExecutor._bundled
        from playwright.sync_api import sync_playwright
        box: Dict[str, str] = {}

        def probe() -> None:      # Playwright's sync API refuses to start inside a running asyncio loop
            with sync_playwright() as p:
                box["exe"] = p.chromium.executable_path

        t = threading.Thread(target=probe)
        t.start()
        t.join()
        BrowserExecutor._bundled = box.get("exe", "")
        return BrowserExecutor._bundled

    def _launch_args(self) -> List[str]:
        # QUIC (UDP) and unproxied WebRTC UDP would leave the browser without passing the network guard
        args = ["--no-sandbox", "--disable-dev-shm-usage", "--no-first-run", "--no-default-browser-check",
                "--disable-background-networking", "--disable-component-update", "--password-store=basic",
                "--disable-quic", "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
                "--webrtc-ip-handling-policy=disable_non_proxied_udp"]
        pw = playwright_disabled_features()
        if pw:
            args.append("--disable-features=" + ",".join(list(dict.fromkeys(pw + list(EXTRA_DISABLED_FEATURES)))))
        return args + list(self.cfg.extra_args)

    def _ignored_default_args(self) -> List[str]:
        pw = playwright_disabled_features()
        return ["--disable-features=" + ",".join(pw)] if pw else []

    # ---- ModalityExecutor -----------------------------------------------------------------------------------
    def tools(self) -> List[ToolDescriptor]:
        return list(TOOLS)

    def validate(self, action: ActionSpec) -> List[str]:
        t = _TOOLS.get(action.operation)
        if t is None:
            return [f"unknown operation {action.operation!r}"]
        from .env import validate_args
        a = action.arguments
        errs = validate_args(t.arg_schema, a)
        op = action.operation
        if op in ("browser.navigate", "browser.open_tab", "browser.download"):
            url = a.get("url") or (action.target if op == "browser.navigate" else "")
            if op == "browser.navigate" and not url:
                errs.append("url is required")
            if url and urlsplit(str(url)).scheme not in _ALLOW_SCHEMES and url != "about:blank":
                errs.append("only http(s) URLs can be opened")
        if op in _NEEDS_TARGET or (op == "browser.download" and not a.get("url")):
            if not (a.get("ref") or a.get("role") or a.get("name") or a.get("selector")):
                errs.append("an element is required: ref + obs_id, or role/name, or selector")
        ref = str(a.get("ref") or "")
        if ref:
            r, _, o = ref.partition("@")
            if not _REF_RE.match(r):
                errs.append(f"malformed element ref {ref!r}")
            elif not (o or a.get("obs_id")):
                errs.append("obs_id is required with ref (the observation the ref was read from)")
        if op == "browser.fill" and ("text" in a) == ("secret" in a):
            errs.append("exactly one of text or secret is required")
        if op == "browser.select_option" and not ("label" in a or "value" in a):
            errs.append("label or value is required")
        if op == "browser.wait_for" and not any(k in a for k in ("text", "text_gone", "url", "name", "role", "change")):
            errs.append("a condition is required: text, text_gone, url, role/name or change")
        sess = str(a.get("session") or "default")
        if self._session_kind(sess) is None:
            errs.append(f"unknown or unauthorized session {sess!r}")
        return errs

    def execute(self, action: ActionSpec, ctx: ExecContext) -> ActionResult:
        now = self.clock.now()
        res = ActionResult(action_id=action.action_id, status=ResultStatus.OK, started=now, ended=now)
        errs = self.validate(action)
        if errs:
            res.status, res.failure_class, res.failure_detail = (ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS,
                                                                 "; ".join(errs))
            return res
        name = str(action.arguments.get("session") or "default")
        try:
            sess = self._session_sync(name, ctx, action)
        except _Fail as f:
            res.status, res.failure_class, res.failure_detail = f.status, f.failure_class, f.detail
            return res
        budget = max(1.0, float(action.timeout_s or 60.0))
        if ctx.deadline and isinstance(self.clock, WallClock):
            budget = max(0.5, min(budget, ctx.deadline - self.clock.now()))
        with sess.lock:
            sess.ctx, sess.action = ctx, action
            # what was authorized for an earlier action is not authorized for this one
            sess.checked, sess.blocked = set(), []
            sess.origins_ok, sess.origin_cache, sess.net_tokens, sess.tunnel_ok = {}, {}, {}, {}
            try:
                self._loop.call(self._run_action(sess, action, ctx, res, budget), timeout=budget + 15)
            except TimeoutError:
                res.status, res.failure_class, res.failure_detail = (ResultStatus.TIMEOUT, FailureClass.UNKNOWN,
                                                                     f"no result within {budget:.0f} s")
            if sess.kind == "identity" and sess.context is not None:
                try:
                    self._loop.call(self._save_storage(sess), timeout=10)
                except Exception:  # noqa: BLE001 — losing one cookie snapshot must not fail the action
                    pass
        res.ended = self.clock.now()
        res.cost = Cost(seconds=max(0.0, res.ended - res.started), primitive_ops=len(res.ops))
        return res

    def close(self) -> None:
        with self._guard:
            if self._closed:
                return
            self._closed = True
        try:
            self._loop.call(self._close_all(), timeout=30)
        except Exception:  # noqa: BLE001 — shutting down: nothing useful can be done with a late browser error
            pass
        self._loop.stop()

    # ---- hooks used by ComputerEnvironment ------------------------------------------------------------------
    def capabilities(self) -> Dict[str, Any]:
        import importlib.metadata as md
        with self._guard:
            sessions = {n: {"kind": s.kind, "tabs": len(s.tabs), "active": s.active, "read_only": s.read_only}
                        for n, s in self._sessions.items()}
        return {"engine": "chromium", "version": self._version, "build": "full Chromium (new headless mode)"
                if self.cfg.headless else "full Chromium", "playwright": md.version("playwright"),
                "headless": self.cfg.headless,
                "viewport": list(self.cfg.viewport), "identities": list(self.cfg.identities),
                "session_kinds": ["identity (persistent profile)", "research (isolated, read-only requests)",
                                  "cdp (externally supplied Chromium)"],
                "cdp_sessions": sorted(self.cfg.cdp_endpoints), "sessions": sessions, "screenshots": "on demand",
                "uid_separation": {"enabled": bool(self._launch["user"]), "user": self._launch["user"],
                                   "reason": self._launch["reason"]},
                "chromium_executable": self._launch.get("exe") or "playwright default",
                "network_guard": {"proxy": "every request leaves through a policy-checking proxy (per session)",
                                  "sessions": {n: dict(s.guard.stats) for n, s in list(self._sessions.items())
                                               if s.guard is not None},
                                  "service_workers": "blocked", "quic": "disabled",
                                  "webrtc_udp": "proxied only", "webtransport": "removed"},
                "restorable_tabs": {n: [t.get("url") for t in v.get("tabs", [])]
                                    for n, v in self._restorable.get("sessions", {}).items()}}

    def observe(self) -> List[Observation]:
        """Current state of the active tab of every open session (no navigation, no screenshot)."""
        out: List[Observation] = []
        with self._guard:
            sessions = [s for s in self._sessions.values() if s.tabs]
        for sess in sessions[:4]:
            with sess.lock:
                try:
                    out.append(self._loop.call(self._observe(sess, self._tab(sess, None), mode="auto"), timeout=20))
                except Exception:  # noqa: BLE001 — an unreadable tab is simply not part of this observation
                    continue
        return out

    def snapshot_state(self) -> Dict[str, Any]:
        with self._guard:
            sessions = {n: {"kind": s.kind, "active": s.active,
                            "tabs": [{"tab": t.id, "url": t.page.url} for t in list(s.tabs.values())
                                     if urlsplit(t.page.url).scheme in _ALLOW_SCHEMES]}
                        for n, s in self._sessions.items() if s.kind == "identity"}
            for n, v in self._restorable.get("sessions", {}).items():
                sessions.setdefault(n, v)
            return {"sessions": sessions, "next_ref": self._next_ref}

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Remember the tabs of a previous process. They are reopened (each URL checked by policy) by the first
        action that uses the session; cookies come back with the persistent profile."""
        self._restorable = dict(state or {})
        self._next_ref = max(self._next_ref, int(self._restorable.get("next_ref", 1)))

    def verify(self, cond: VerificationCondition, action: ActionSpec, ctx: ExecContext) -> Optional[bool]:
        """Evaluate a verification condition against the live page (never against what the action assumed)."""
        if cond.kind not in ("text_present", "url_matches", "element_state"):
            return None
        name = str(action.arguments.get("session") or "default")
        with self._guard:
            sess = self._sessions.get(name)
        if sess is None or not sess.tabs:
            return None
        with sess.lock:
            return self._loop.call(self._verify(sess, cond, action), timeout=self.cfg.op_timeout_s + 10)

    def screen(self, session: str = "default", tab: Optional[str] = None, ctx: Optional[ExecContext] = None,
               action: Optional[ActionSpec] = None) -> BrowserScreen:
        return BrowserScreen(self, session, tab, ctx, action)

    def attach_cdp(self, name: str, endpoint: str) -> str:
        """Operator API: register an externally supplied Chromium (CDP endpoint) as session ``cdp:<name>``."""
        self.cfg.cdp_endpoints[name] = endpoint
        return f"cdp:{name}"

    def chromium_pids(self) -> List[int]:
        """Process ids of browser processes launched by this executor (diagnostics / uid checks)."""
        out = subprocess.run(["pgrep", "-f", self.root], capture_output=True, text=True).stdout.split()
        return [int(p) for p in out if p.isdigit()]

    # ---- sessions ---------------------------------------------------------------------------------------------
    def _session_kind(self, name: str) -> Optional[str]:
        if name == "default" and "default" in self.cfg.identities:
            return "identity"
        if name.startswith("identity:") and name[9:] in self.cfg.identities and re.match(r"^[\w.-]{1,40}$", name[9:]):
            return "identity"
        if name.startswith("research:") and re.match(r"^[\w.-]{1,40}$", name[9:]):
            return "research"
        if name.startswith("cdp:") and name[4:] in self.cfg.cdp_endpoints:
            return "cdp"
        return None

    def _session_sync(self, name: str, ctx: Optional[ExecContext], action: Optional[ActionSpec]) -> _Session:
        with self._guard:
            if self._closed:
                raise _Fail(ResultStatus.FAILED, FailureClass.UNKNOWN, "browser executor is closed")
            sess = self._sessions.get(name)
            if sess is not None:
                return sess
            kind = self._session_kind(name)
            if kind is None:
                raise _Fail(ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS, f"unknown session {name!r}")
            sess = _Session(name, kind)
            sess.ctx, sess.action = ctx, action
            try:
                self._loop.call(self._open_session(sess), timeout=90)
            except _Fail:
                raise
            except Exception as e:  # noqa: BLE001
                raise _Fail(ResultStatus.FAILED, FailureClass.UNKNOWN, f"could not start browser session: {e}")
            self._sessions[name] = sess
            return sess

    async def _ensure_pw(self) -> Any:
        if self._pw is None:
            from playwright.async_api import async_playwright
            self._pw = await async_playwright().start()
        return self._pw

    async def _open_session(self, sess: _Session) -> None:
        pw, cfg = await self._ensure_pw(), self.cfg
        exe = self._launch["wrapper"] or self._launch.get("exe") or cfg.executable_path or None
        # without an explicit binary, ask for the full Chromium build (Playwright's default headless launch would
        # otherwise run the reduced "headless shell")
        channel = None if exe else "chromium"
        vp = {"width": cfg.viewport[0], "height": cfg.viewport[1]}
        dl = os.path.join(self.root, "dl") if self._launch["user"] else None
        if sess.kind in ("identity", "research"):
            sess.guard = await _NetGuard(self, sess).start()
        if sess.kind == "identity":
            ident = "default" if sess.name == "default" else sess.name[9:]
            sess.profile_dir = os.path.join(self.root, "profiles", ident)
            os.makedirs(sess.profile_dir, exist_ok=True)
            if self._launch["user"] and os.stat(sess.profile_dir).st_uid != self._launch["uid"]:
                subprocess.run(["chown", "-R", f"{self._launch['uid']}:{self._launch['gid']}", sess.profile_dir],
                               check=True)
            os.chmod(sess.profile_dir, 0o700)
            sess.context = await pw.chromium.launch_persistent_context(
                sess.profile_dir, headless=cfg.headless, args=self._launch_args(), viewport=vp, executable_path=exe,
                channel=channel, accept_downloads=True, device_scale_factor=1, downloads_path=dl,
                proxy=sess.guard.playwright_proxy(), service_workers="block",
                ignore_default_args=self._ignored_default_args())
            if sess.context.browser is not None:
                self._version = sess.context.browser.version
            await self._load_storage(sess)
        elif sess.kind == "research":
            if self._research_browser is None:
                # every research context gets its own guard; the browser-wide setting only ensures that nothing
                # can leave without one
                self._research_browser = await pw.chromium.launch(
                    headless=cfg.headless, args=self._launch_args(), executable_path=exe, channel=channel,
                    downloads_path=dl, proxy={"server": "http://127.0.0.1:9", "bypass": "<-loopback>"},
                    ignore_default_args=self._ignored_default_args())
                self._version = self._research_browser.version
            sess.context = await self._research_browser.new_context(viewport=vp, accept_downloads=True,
                                                                    device_scale_factor=1, service_workers="block",
                                                                    proxy=sess.guard.playwright_proxy())
            sess.read_only = cfg.research_read_only
        else:
            try:
                sess.browser = await pw.chromium.connect_over_cdp(cfg.cdp_endpoints[sess.name[4:]])
            except Exception as e:  # noqa: BLE001
                raise _Fail(ResultStatus.FAILED, FailureClass.TRANSIENT_NETWORK, f"CDP attach failed: {e}")
            sess.context = sess.browser.contexts[0] if sess.browser.contexts else await sess.browser.new_context()
        await sess.context.add_init_script(_NO_SW_JS)        # registration is refused, visibly, in every session
        await sess.context.add_init_script(_MUT_JS)
        await sess.context.add_init_script(_GUARD_JS)
        await sess.context.route("**/*", lambda route: self._on_route(sess, route))
        await sess.context.route_web_socket(re.compile(r".*"), lambda ws: self._on_ws(sess, ws))
        sess.context.on("page", lambda page: self._adopt(sess, page))
        for page in list(sess.context.pages):
            self._adopt(sess, page)
        sess.restore = list(self._restorable.get("sessions", {}).pop(sess.name, {}).get("tabs", []))

    def _adopt(self, sess: _Session, page: Any) -> _Tab:
        for t in sess.tabs.values():
            if t.page is page:
                return t
        sess.n_tabs += 1
        tab = _Tab(f"t{sess.n_tabs}", page)
        sess.tabs[tab.id] = tab
        if not sess.active:
            sess.active = tab.id

        def on_close(_p: Any = None) -> None:
            sess.tabs.pop(tab.id, None)
            if sess.active == tab.id:
                sess.active = next(reversed(sess.tabs), "") if sess.tabs else ""

        async def on_dialog(d: Any) -> None:
            tab.dialogs.append(f"{d.type}: {d.message}"[:300])
            try:
                await d.dismiss()
            except Exception:  # noqa: BLE001
                pass

        async def on_download(d: Any) -> None:
            if not sess.expect_download:
                sess.blocked.append({"url": d.url, "why": "unrequested download cancelled", "verdict": "cancelled",
                                     "main": False})
                try:
                    await d.cancel()
                except Exception:  # noqa: BLE001
                    pass

        page.on("close", on_close)
        page.on("dialog", lambda d: asyncio.ensure_future(on_dialog(d)))
        page.on("download", lambda d: asyncio.ensure_future(on_download(d)))
        return tab

    def _tab(self, sess: _Session, tab_id: Optional[str]) -> _Tab:
        tid = tab_id or sess.active
        if tid not in sess.tabs:
            raise _Fail(ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS if tab_id else FailureClass.STALE_OBSERVATION,
                        f"no such tab {tid!r}" if tid else "the session has no open tab")
        return sess.tabs[tid]

    async def _with_page(self, sess: _Session, tab_id: Optional[str], fn: Callable[[Any], Awaitable[Any]]) -> Any:
        if not sess.tabs:
            self._adopt(sess, await sess.context.new_page())
        return await fn(self._tab(sess, tab_id).page)

    def _storage_path(self, sess: _Session) -> str:
        return sess.profile_dir.rstrip("/") + ".storage.json"

    async def _save_storage(self, sess: _Session) -> None:
        cookies = [c for c in await sess.context.cookies() if c.get("expires", -1) in (-1, 0)]
        path, tmp = self._storage_path(sess), self._storage_path(sess) + ".tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"session_cookies": cookies}, f)
        os.replace(tmp, path)

    async def _load_storage(self, sess: _Session) -> None:
        try:
            with open(self._storage_path(sess)) as f:
                cookies = json.load(f).get("session_cookies", [])
        except (OSError, ValueError):
            return
        if cookies:
            for c in cookies:
                c.pop("expires", None)
            try:
                await sess.context.add_cookies(cookies)
            except Exception:  # noqa: BLE001 — a cookie Chromium rejects is dropped, the rest of the profile stands
                pass

    async def _close_session(self, sess: _Session) -> None:
        if sess.guard is not None:
            await sess.guard.close()
        try:
            if sess.kind == "identity" and sess.context is not None:
                await self._save_storage(sess)
            if sess.kind == "cdp":
                await sess.context.unroute("**/*")
                if sess.browser is not None:
                    await sess.browser.close()        # disconnects; the external Chromium keeps running
            elif sess.context is not None:
                await sess.context.close()
        except Exception:  # noqa: BLE001
            pass
        sess.tabs.clear()

    async def _close_all(self) -> None:
        for sess in list(self._sessions.values()):
            await self._close_session(sess)
        self._sessions.clear()
        if self._research_browser is not None:
            try:
                await self._research_browser.close()
            except Exception:  # noqa: BLE001
                pass
        if self._pw is not None:
            await self._pw.stop()
            self._pw = None

    # ---- authorization ----------------------------------------------------------------------------------------
    async def _check(self, ctx: Optional[ExecContext], op: PrimitiveOp, action: Optional[ActionSpec]) -> AuthorizationDecision:
        if ctx is None or ctx.check is None or action is None:
            return AuthorizationDecision(AuthVerdict.DENY, reasons=["no authorization context for this operation"])
        try:
            dec = await asyncio.get_running_loop().run_in_executor(None, ctx.check, op, action)
        except Exception as e:  # noqa: BLE001 — fail closed
            return AuthorizationDecision(AuthVerdict.DENY, reasons=[f"authorization check failed: {type(e).__name__}"])
        if not isinstance(dec, AuthorizationDecision):
            return AuthorizationDecision(AuthVerdict.DENY, reasons=["authorization check returned no decision"])
        return dec

    # The route layer (every request a page starts) and the network layer (_NetGuard: every request that leaves the
    # browser, including redirect hops) share these helpers. Both run on the executor's event loop.
    _TOKEN_S = 30.0
    _READ_METHODS = ("GET", "HEAD", "OPTIONS")

    def _grant_token(self, sess: _Session, method: str, url: str) -> None:
        """The route layer authorized exactly this request: the network layer forwards it without asking again."""
        key, now = (method.upper(), _norm_url(url)), self.clock.now()
        tok = sess.net_tokens.setdefault(key, [0, 0.0])
        tok[0], tok[1] = tok[0] + 1, now + self._TOKEN_S
        u = urlsplit(url)
        if u.scheme in ("https", "wss", "ws"):        # Chromium tunnels these through the proxy with CONNECT
            try:
                port = u.port or (80 if u.scheme == "ws" else 443)
            except ValueError:
                return
            host = u.hostname or ""
            sess.tunnel_ok[f"[{host}]:{port}" if ":" in host else f"{host}:{port}"] = now + self._TOKEN_S

    def _take_token(self, sess: _Session, method: str, url: str) -> bool:
        key, now = (method.upper(), _norm_url(url)), self.clock.now()
        tok = sess.net_tokens.get(key)
        if not tok or tok[0] <= 0 or tok[1] < now:
            return False
        tok[0] -= 1
        return True

    def _blocked(self, sess: _Session, url: str, why: str, verdict: str, main: bool, via: str) -> None:
        sess.blocked.append({"url": url, "why": why, "verdict": verdict, "main": main, "via": via})

    async def _read_allowed(self, sess: _Session, origin: str, resource: str, method: str) -> AuthorizationDecision:
        """Reads (GET / HEAD / OPTIONS of a subresource) are judged per origin, once per action."""
        now = self.clock.now()
        if sess.origins_ok.get(origin):
            return AuthorizationDecision(AuthVerdict.ALLOW)
        cached = sess.origin_cache.get(origin)
        if cached is None or cached[1] < now:
            op = PrimitiveOp(op="browser.request", modality=self.modality, started=now, ended=now,
                             arguments={"url": origin + "/", "resource_type": resource, "method": method,
                                        "session": sess.name})
            dec = await self._check(sess.ctx, op, sess.action)
            sess.origin_cache[origin] = (dec.verdict == AuthVerdict.ALLOW, now + self.cfg.origin_cache_s)
            return dec
        return AuthorizationDecision(AuthVerdict.ALLOW if cached[0] else AuthVerdict.DENY,
                                     reasons=[] if cached[0] else ["request to this origin denied earlier in this action"])

    async def _net_decide(self, sess: _Session, method: str, url: str, *, tunnel: str = "", document: bool = False,
                          main: bool = False, resource: str = "", peer_ips: Sequence[str] = ()) -> Tuple[bool, str]:
        """Network layer: may this request leave the browser now? (Called by the session's :class:`_NetGuard`.)

        ``peer_ips`` are the addresses a host *name* resolved to (the connection goes to them only). Each that is not
        a global address (loopback, private, link-local, ...) is put to the policy as ``peer_ip``, whatever the route
        layer decided for the name: a public name whose DNS answer points at this machine or its network (DNS
        rebinding) is judged by the address actually reached."""
        now, method = self.clock.now(), method.upper()
        for peer_ip in [p for p in peer_ips if _non_global(p)]:
            op = PrimitiveOp(op="browser.request", modality=self.modality, started=now, ended=now,
                             arguments={"url": url, "method": method, "peer_ip": peer_ip, "via": "network",
                                        "resource_type": "tunnel" if tunnel else (resource or "other"),
                                        "session": sess.name})
            dec = await self._check(sess.ctx, op, sess.action)
            if dec.verdict != AuthVerdict.ALLOW:
                why = "; ".join(dec.reasons) or f"{url} resolves to {peer_ip}"
                self._blocked(sess, url, why, dec.verdict, main, "network")
                return False, why
        if tunnel:
            if sess.tunnel_ok.get(tunnel, 0.0) >= now:
                return True, ""
            op = PrimitiveOp(op="browser.request", modality=self.modality, started=now, ended=now,
                             arguments={"url": url, "method": "CONNECT", "resource_type": "tunnel", "via": "network",
                                        "session": sess.name})
            dec = await self._check(sess.ctx, op, sess.action)
            if dec.verdict == AuthVerdict.ALLOW:
                sess.tunnel_ok[tunnel] = now + self._TOKEN_S
                return True, ""
            why = "; ".join(dec.reasons) or "tunnel denied"
            self._blocked(sess, url, why, dec.verdict, False, "network")
            return False, why
        if self._take_token(sess, method, url):
            return True, ""
        if sess.read_only and method not in self._READ_METHODS:
            why = "research sessions only issue read requests"
            self._blocked(sess, url, why, AuthVerdict.DENY, main, "network")
            return False, why
        if document:              # a navigation the route layer did not see: a redirect hop of a page or frame
            op = PrimitiveOp(op="browser.navigate", modality=self.modality, started=now, ended=now,
                             arguments={"url": url, "via": "redirect", "method": method, "session": sess.name})
            dec = await self._check(sess.ctx, op, sess.action)
            if dec.verdict == AuthVerdict.ALLOW:
                sess.checked.add(_norm_url(url))
                sess.origins_ok[_origin(url)] = True
        elif method in self._READ_METHODS:
            dec = await self._read_allowed(sess, _origin(url), resource or "other", method)
        else:
            op = PrimitiveOp(op="browser.request", modality=self.modality, started=now, ended=now,
                             arguments={"url": url, "method": method, "resource_type": resource or "other",
                                        "via": "network", "session": sess.name})
            dec = await self._check(sess.ctx, op, sess.action)
        if dec.verdict == AuthVerdict.ALLOW:
            return True, ""
        why = "; ".join(dec.reasons) or "request denied"
        self._blocked(sess, url, why, dec.verdict, main, "network")
        return False, why

    async def _on_route(self, sess: _Session, route: Any) -> None:
        """Route layer: every request a page of this session starts, before it is sent."""
        req = route.request
        url, now = req.url, self.clock.now()
        scheme = urlsplit(url).scheme

        async def block(why: str, verdict: str = AuthVerdict.DENY, main: bool = False) -> None:
            self._blocked(sess, url, why, verdict, main, "route")
            try:
                if main:        # "204 No Content" cancels a top-level navigation and leaves the current page in place
                    await route.fulfill(status=204, body="")
                else:
                    await route.abort("blockedbyclient")
            except Exception:  # noqa: BLE001
                pass

        async def go() -> None:
            self._grant_token(sess, req.method, url)
            await route.continue_()

        try:
            if scheme in ("data", "blob", "about"):
                return await route.continue_()
            is_nav = req.is_navigation_request()
            try:
                main = is_nav and req.frame.parent_frame is None
            except Exception:  # noqa: BLE001 — no frame (service worker): not a top-level navigation
                main = False
            if scheme not in _ALLOW_SCHEMES:
                return await block(f"scheme {scheme!r} is not allowed", main=main)
            method = req.method.upper()
            if sess.read_only and method not in self._READ_METHODS:
                return await block("research sessions only issue read requests", main=main)
            key, origin = _norm_url(url), _origin(url)
            if is_nav:
                if sess.approved.get(key, 0) > 0:
                    sess.approved[key] -= 1
                else:
                    op = PrimitiveOp(op="browser.navigate", modality=self.modality, started=now, ended=now,
                                     arguments={"url": url, "via": "page", "method": method, "session": sess.name})
                    dec = await self._check(sess.ctx, op, sess.action)
                    if dec.verdict != AuthVerdict.ALLOW:
                        return await block("; ".join(dec.reasons) or "navigation denied", dec.verdict, main)
                sess.checked.add(key)
                sess.origins_ok[origin] = True
                return await go()
            if method in self._READ_METHODS:
                dec = await self._read_allowed(sess, origin, req.resource_type, method)
            else:
                # a write (a script's POST, a form submission into a frame, a beacon): reported and judged per URL,
                # also on an origin the page itself came from
                op = PrimitiveOp(op="browser.request", modality=self.modality, started=now, ended=now,
                                 arguments={"url": url, "method": method, "resource_type": req.resource_type,
                                            "via": "page", "session": sess.name})
                dec = await self._check(sess.ctx, op, sess.action)
            if dec.verdict == AuthVerdict.ALLOW:
                return await go()
            return await block("; ".join(dec.reasons) or "request denied", dec.verdict)
        except Exception:  # noqa: BLE001 — the page went away mid-request; nothing left to continue or abort
            return

    async def _on_ws(self, sess: _Session, ws: Any) -> None:
        """Route layer for WebSockets: judged per URL before the connection is opened (the tunnel it opens is
        then checked again by the network layer)."""
        url, now = ws.url, self.clock.now()
        try:
            op = PrimitiveOp(op="browser.request", modality=self.modality, started=now, ended=now,
                             arguments={"url": url, "method": "GET", "resource_type": "websocket", "via": "page",
                                        "session": sess.name})
            dec = AuthorizationDecision(AuthVerdict.DENY, reasons=["research sessions do not open WebSockets"]) \
                if sess.read_only else await self._check(sess.ctx, op, sess.action)
            if dec.verdict == AuthVerdict.ALLOW:
                self._grant_token(sess, "GET", url)
                ws.connect_to_server()
                return
            self._blocked(sess, url, "; ".join(dec.reasons) or "websocket denied", dec.verdict, False, "route")
            await ws.close(code=1008, reason="blocked by the authorization layer")
        except Exception:  # noqa: BLE001 — the page went away
            return

    # ---- observation ------------------------------------------------------------------------------------------
    async def _settle(self, page: Any, max_ms: Optional[int] = None) -> None:
        for _ in range(3):
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=self.cfg.nav_timeout_s * 1000)
                await page.evaluate(_SETTLE_JS, {"quiet": self.cfg.settle_quiet_ms,
                                                 "max": max_ms or self.cfg.settle_max_ms})
                return
            except Exception as e:  # noqa: BLE001 — a navigation destroyed the context: wait for the new document
                if "closed" in str(e).lower():
                    return
                await asyncio.sleep(0.05)

    async def _a11y(self, page: Any) -> str:
        try:
            return digest(await page.locator("body").aria_snapshot(timeout=5000), self.cfg.max_a11y_chars)
        except Exception as e:  # noqa: BLE001
            return f"(accessibility tree unavailable: {type(e).__name__})"

    async def _state(self, sess: _Session, tab: _Tab) -> Dict[str, Any]:
        page, st = tab.page, None
        for _ in range(3):
            try:
                st = await page.evaluate(_PAGE_JS, {"mode": "extract", "next": self._next_ref,
                                                    "maxElements": self.cfg.max_elements,
                                                    "maxText": self.cfg.max_page_text, "attr": REF_ATTR})
                break
            except Exception as e:  # noqa: BLE001
                if "closed" in str(e).lower():
                    raise _Fail(ResultStatus.FAILED, FailureClass.STALE_OBSERVATION, "the tab was closed")
                await self._settle(page)
        if st is None:
            raise _Fail(ResultStatus.FAILED, FailureClass.UNKNOWN, "the page could not be read")
        self._next_ref = max(self._next_ref, int(st.pop("next", self._next_ref)))
        st.update(tab=tab.id, session=sess.name, session_kind=sess.kind, surface=f"browser:{sess.name}:{tab.id}",
                  generation=st["doc"])
        tabs = []
        for t in list(sess.tabs.values()):        # a popup may be adopted while this coroutine awaits
            try:
                title = await t.page.title()
            except Exception:  # noqa: BLE001
                title = ""
            url = t.page.url
            tabs.append({"tab": t.id, "url": url[:1024] + ("…" if len(url) > 1024 else ""), "title": title[:120],
                         "active": t.id == sess.active})
        st["tabs"] = tabs
        if tab.dialogs:
            st["dialogs"], tab.dialogs = list(tab.dialogs), []
        return st

    async def _observe(self, sess: _Session, tab: _Tab, *, mode: str = "auto", parent: str = "",
                       a11y: Optional[bool] = None) -> Observation:
        prev = tab.prev_state
        st = await self._state(sess, tab)
        if mode == "auto":
            mode = "diff" if prev is not None and prev.get("doc") == st.get("doc") else "full"
        if a11y or (a11y is None and mode == "full"):
            st["a11y"] = await self._a11y(tab.page)
        obs = self.perception.page(st, prev, mode=mode, parent_action_id=parent, base_obs_id=tab.prev_obs_id)
        if st.get("dialogs"):
            obs.data["dialogs"] = st["dialogs"]
        tab.prev_state, tab.prev_obs_id = st, obs.obs_id
        tab.sigs[obs.obs_id] = {e["ref"]: (e.get("role", ""), e.get("name", "")) for e in st.get("elements", [])}
        if prev is not None and prev.get("doc") != st.get("doc"):
            for k in list(tab.sigs)[:-1]:           # a new document: every earlier observation of this tab is outdated
                tab.sigs.pop(k)
        while len(tab.sigs) > 8:
            tab.sigs.popitem(last=False)
        return obs

    # ---- element resolution -------------------------------------------------------------------------------------
    async def _status(self, page: Any, *, ref: str = "", handle: Any = None) -> Dict[str, Any]:
        st = await page.evaluate(_PAGE_JS, {"mode": "status", "ref": ref or None, "el": handle, "attr": REF_ATTR,
                                            "next": self._next_ref})
        self._next_ref = max(self._next_ref, int(st.get("next", self._next_ref)))
        return st

    async def _resolve(self, tab: _Tab, a: Dict[str, Any]) -> Tuple[Any, Dict[str, Any], str]:
        """Locator for the element an action names. Returns (locator, element facts, how it was found)."""
        page = tab.page
        raw = str(a.get("ref") or "")
        if raw:
            ref, _, oid = raw.partition("@")
            oid = oid or str(a.get("obs_id") or "")
            if oid not in tab.sigs:
                raise _Fail(ResultStatus.FAILED, FailureClass.STALE_OBSERVATION,
                            f"observation {oid} is outdated for tab {tab.id}: inspect the page again")
            st = await self._status(page, ref=ref)
            if not st.get("found"):
                raise _Fail(ResultStatus.FAILED, FailureClass.STALE_OBSERVATION,
                            f"element {ref} of observation {oid} is no longer in the page")
            rec = tab.sigs[oid].get(ref)
            if rec is None:
                raise _Fail(ResultStatus.FAILED, FailureClass.STALE_OBSERVATION,
                            f"observation {oid} has no element {ref}")
            if rec != (st.get("role", ""), st.get("name", "")):
                raise _Fail(ResultStatus.FAILED, FailureClass.STALE_OBSERVATION,
                            f"element {ref} changed since observation {oid}: was {rec[0]} \"{rec[1]}\", "
                            f"now {st.get('role')} \"{st.get('name')}\"")
            return page.locator(f'[{REF_ATTR}="{ref}"]'), st, "locator=ref"
        role, name, sel = a.get("role"), a.get("name"), a.get("selector")
        chain: List[Tuple[str, Any]] = []
        if role and name is not None:
            chain += [("role+name", page.get_by_role(role, name=name, exact=True)),
                      ("role+name~", page.get_by_role(role, name=name))]
        elif role:
            chain.append(("role", page.get_by_role(role)))
        if name is not None:
            chain += [("label", page.get_by_label(name, exact=True)), ("placeholder", page.get_by_placeholder(name)),
                      ("text", page.get_by_text(name, exact=True))]
        if sel:
            chain.append(("css", page.locator(sel)))
        tried: List[str] = []
        for how, loc in chain:
            try:
                vis = loc.filter(visible=True)
                n = await vis.count()
            except Exception:  # noqa: BLE001 — an invalid selector or role is just a failed strategy
                n, vis = 0, loc
            if n == 0:
                tried.append(how)
                continue
            if n > 1:
                if a.get("nth") is None:
                    raise _Fail(ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS,
                                f"{n} elements match {how} ({role or ''} \"{name or sel}\"): give nth or use a ref")
                if int(a["nth"]) >= n:
                    raise _Fail(ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS, f"nth={a['nth']} but only {n} match")
                vis = vis.nth(int(a["nth"]))
            handle = await vis.element_handle(timeout=2000)
            st = await self._status(page, handle=handle)
            note = f"locator={how}" + (f" (fallback after {', '.join(tried)})" if tried else "")
            return page.locator(f'[{REF_ATTR}="{st["ref"]}"]'), st, note
        raise _Fail(ResultStatus.FAILED, FailureClass.PAGE_CHANGED,
                    f"no visible element for role={role!r} name={name!r} selector={sel!r} (tried {', '.join(tried)})")

    @staticmethod
    def _pw_fail(e: Exception, what: str) -> _Fail:
        msg = str(e).split("\n")[0][:300]
        low = str(e)
        if "ERR_BLOCKED_BY_CLIENT" in low:
            return _Fail(ResultStatus.DENIED, FailureClass.MISSING_PERMISSION, f"{what}: blocked by the authorization layer")
        if "ERR_UNSAFE_PORT" in low or "ERR_INVALID_URL" in low or "invalid URL" in low:
            return _Fail(ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS, f"{what}: {msg}")
        if "ERR_CERT" in low:
            return _Fail(ResultStatus.FAILED, FailureClass.WRONG_SITE_ASSUMPTION, f"{what}: {msg}")
        if any(k in low for k in _NET_TRANSIENT):
            return _Fail(ResultStatus.FAILED, FailureClass.TRANSIENT_NETWORK, f"{what}: {msg}")
        if type(e).__name__ == "TimeoutError":
            if "goto" in low or "navigat" in low:
                return _Fail(ResultStatus.TIMEOUT, FailureClass.TRANSIENT_NETWORK, f"{what}: {msg}")
            return _Fail(ResultStatus.TIMEOUT, FailureClass.WRONG_SITE_ASSUMPTION, f"{what}: {msg}")
        if "not a <select>" in low or "not an <input>" in low or "cannot be filled" in low.lower() or \
                "did not find some options" in low:
            return _Fail(ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS, f"{what}: {msg}")
        return _Fail(ResultStatus.FAILED, FailureClass.UNKNOWN, f"{what}: {msg}")

    # ---- the action -------------------------------------------------------------------------------------------
    async def _run_action(self, sess: _Session, action: ActionSpec, ctx: ExecContext, res: ActionResult,
                          budget: float) -> None:
        a, opname, now = dict(action.arguments), action.operation, self.clock.now()
        public = {k: v for k, v in a.items() if k != "text" or len(str(v)) <= 2000}
        op = PrimitiveOp(op=opname, modality=self.modality, arguments=public, started=now, ended=now)
        res.ops.append(op)
        try:
            if ctx.cancel is not None and ctx.cancel.is_set():
                raise _Fail(ResultStatus.CANCELLED, FailureClass.NONE, "cancelled before start")
            if sess.restore:
                await self._restore_tabs(sess, ctx, action)
            if not sess.tabs and opname not in ("browser.open_tab", "browser.close_session"):
                self._adopt(sess, await sess.context.new_page())
            tab = None if opname in ("browser.open_tab", "browser.close_session") else self._tab(sess, a.get("tab"))
            if tab is not None:
                op.obs_before = tab.prev_obs_id
            handler = getattr(self, "_op_" + opname.split(".", 1)[1])
            obs = await asyncio.wait_for(handler(sess, tab, action, ctx, op, res), timeout=budget)
            if obs is not None:
                res.observations.append(obs)
                op.obs_after = obs.obs_id
            nav = [b for b in sess.blocked if b.get("main")]
            if nav and opname not in ("browser.inspect", "browser.read", "browser.wait_for", "browser.screenshot"):
                raise _Fail(*denial_class(AuthorizationDecision(nav[0]["verdict"], reasons=[
                    f"navigation to {nav[0]['url']} denied: {nav[0]['why']}"])))
        except _Fail as f:
            res.status, res.failure_class, res.failure_detail = f.status, f.failure_class, f.detail
            op.status, op.failure_class = f.status, f.failure_class
        except asyncio.TimeoutError:
            res.status, res.failure_class, res.failure_detail = (ResultStatus.TIMEOUT, FailureClass.UNKNOWN,
                                                                 f"{opname} exceeded {budget:.0f} s")
            op.status = ResultStatus.TIMEOUT
            await self._stop_loading(sess)
        except Exception as e:  # noqa: BLE001 — classify whatever Playwright raised
            f = self._pw_fail(e, opname)
            res.status, res.failure_class, res.failure_detail = f.status, f.failure_class, f.detail
            op.status, op.failure_class = f.status, f.failure_class
        op.ended = self.clock.now()
        for b in sess.blocked:
            res.ops.append(PrimitiveOp(op="browser.request_blocked", modality=self.modality, started=op.started,
                                       ended=op.ended, arguments={"url": b["url"]}, status=ResultStatus.DENIED,
                                       failure_class=FailureClass.MISSING_PERMISSION, note=b["why"][:200]))
        if res.status != ResultStatus.OK and not res.observations:
            res.observations.append(self.perception.error(f"{opname} {res.status}: {res.failure_detail}",
                                                          failure_class=res.failure_class,
                                                          parent_action_id=action.action_id, source=self.modality))

    async def _stop_loading(self, sess: _Session) -> None:
        """Abandon whatever a timed-out action left loading, so the next action starts from a quiet tab."""
        for t in list(sess.tabs.values()):
            try:
                cdp = await sess.context.new_cdp_session(t.page)
                await asyncio.wait_for(cdp.send("Page.stopLoading"), timeout=2)
                await cdp.detach()
            except Exception:  # noqa: BLE001
                continue

    async def _gate(self, ctx: ExecContext, op: PrimitiveOp, action: ActionSpec, extra: Optional[Dict[str, Any]] = None) -> None:
        if extra:
            op.arguments = {**op.arguments, **extra}
        dec = await self._check(ctx, op, action)
        if dec.verdict != AuthVerdict.ALLOW:
            raise _Fail(*denial_class(dec))

    async def _goto(self, sess: _Session, tab: _Tab, url: str, wait_until: str, timeout_s: float) -> Tuple[int, str]:
        key = _norm_url(url)
        sess.approved[key] = sess.approved.get(key, 0) + 1
        sess.checked.add(key)
        n_blocked = len(sess.blocked)
        try:
            for attempt in range(3):
                try:
                    resp = await tab.page.goto(url, wait_until=wait_until, timeout=timeout_s * 1000)
                    break
                except Exception as e:  # noqa: BLE001 — an earlier, abandoned load of this tab got in the way
                    if "interrupted by another navigation" not in str(e) or attempt == 2:
                        raise
                    sess.approved[key] = sess.approved.get(key, 0) + 1
                    await asyncio.sleep(0.15)
        except Exception as e:  # noqa: BLE001
            new = sess.blocked[n_blocked:]
            if new:
                if new[0]["verdict"] == "cancelled":
                    raise _Fail(ResultStatus.FAILED, FailureClass.WRONG_SITE_ASSUMPTION,
                                f"{url} is a download, not a page: use browser.download")
                raise _Fail(*denial_class(AuthorizationDecision(new[0]["verdict"], reasons=[
                    f"navigation to {new[0]['url']} denied: {new[0]['why']}"])))
            if "Download is starting" in str(e):
                raise _Fail(ResultStatus.FAILED, FailureClass.WRONG_SITE_ASSUMPTION,
                            f"{url} is a download, not a page: use browser.download")
            raise self._pw_fail(e, f"navigate {url}")
        finally:
            if sess.approved.get(key, 0) > 0:
                sess.approved[key] -= 1
        refused = [x for x in sess.blocked[n_blocked:] if x.get("main")]
        if refused:                     # the network layer refused the document (a redirect hop, its address)
            raise _Fail(*denial_class(AuthorizationDecision(refused[0]["verdict"], reasons=[
                f"navigation to {refused[0]['url']} denied: {refused[0]['why']}"])))
        return (resp.status if resp is not None else 0), tab.page.url

    async def _post_nav_check(self, sess: _Session, tab: _Tab, ctx: ExecContext, action: ActionSpec, before: str) -> str:
        """After anything that may have navigated: the URL the tab ended on must itself be authorized (redirect
        hops are not seen by the request hook)."""
        final = tab.page.url
        if urlsplit(final).scheme not in _ALLOW_SCHEMES or _norm_url(final) in sess.checked or \
                _norm_url(final) == _norm_url(before):
            return ""
        now = self.clock.now()
        op = PrimitiveOp(op="browser.navigate", modality=self.modality, started=now, ended=now,
                         arguments={"url": final, "via": "redirect", "session": sess.name})
        dec = await self._check(ctx, op, action)
        if dec.verdict != AuthVerdict.ALLOW:
            try:
                await tab.page.goto("about:blank")
            except Exception:  # noqa: BLE001
                pass
            st, fc, detail = denial_class(dec)
            raise _Fail(st, fc, f"redirect to {final} denied: {detail}")
        sess.checked.add(_norm_url(final))
        return final

    async def _restore_tabs(self, sess: _Session, ctx: ExecContext, action: ActionSpec) -> None:
        todo, sess.restore = sess.restore, []
        for i, t in enumerate(todo):
            url = t.get("url", "")
            if urlsplit(url).scheme not in _ALLOW_SCHEMES:
                continue
            now = self.clock.now()
            op = PrimitiveOp(op="browser.navigate", modality=self.modality, started=now, ended=now,
                             arguments={"url": url, "via": "session_restore", "session": sess.name})
            if (await self._check(ctx, op, action)).verdict != AuthVerdict.ALLOW:
                continue
            blank = [x for x in sess.tabs.values() if x.page.url in ("about:blank", "")]
            tab = blank[0] if blank else self._adopt(sess, await sess.context.new_page())
            try:
                await self._goto(sess, tab, url, "domcontentloaded", self.cfg.nav_timeout_s)
            except _Fail:
                continue

    # ---- primitives ---------------------------------------------------------------------------------------------
    async def _op_navigate(self, sess, tab, action, ctx, op, res) -> Observation:
        a = action.arguments
        url = str(a.get("url") or action.target)
        await self._gate(ctx, op, action, {"url": url, "session": sess.name})
        status, final = await self._goto(sess, tab, url, a.get("wait_until", "domcontentloaded"),
                                         min(self.cfg.nav_timeout_s, float(action.timeout_s or 30)))
        await self._post_nav_check(sess, tab, ctx, action, "about:blank")
        await self._settle(tab.page)
        obs = await self._observe(sess, tab, mode="auto", parent=action.action_id)
        op.note = f"http {status}; final url {final}"
        obs.data["http_status"] = status
        if status in (401, 403, 407, 429, 451):
            res.observations.append(obs)
            op.obs_after = obs.obs_id
            raise _Fail(ResultStatus.FAILED, FailureClass.PLATFORM_RESTRICTION, f"the site answered HTTP {status}")
        if status in (404, 410):
            res.observations.append(obs)
            op.obs_after = obs.obs_id
            raise _Fail(ResultStatus.FAILED, FailureClass.WRONG_SITE_ASSUMPTION, f"the site answered HTTP {status}")
        if status >= 500:
            res.observations.append(obs)
            op.obs_after = obs.obs_id
            raise _Fail(ResultStatus.FAILED, FailureClass.TRANSIENT_NETWORK, f"the site answered HTTP {status}")
        return obs

    async def _op_open_tab(self, sess, tab, action, ctx, op, res) -> Observation:
        url = action.arguments.get("url")
        await self._gate(ctx, op, action, {"session": sess.name})
        new = self._adopt(sess, await sess.context.new_page())
        sess.active = new.id
        op.note = f"tab {new.id}"
        res.artifacts["tab"] = new.id
        if url:
            status, final = await self._goto(sess, new, str(url), "domcontentloaded", self.cfg.nav_timeout_s)
            await self._post_nav_check(sess, new, ctx, action, "about:blank")
            await self._settle(new.page)
            op.note += f"; http {status}; final url {final}"
        return await self._observe(sess, new, mode="full", parent=action.action_id)

    async def _op_close_tab(self, sess, tab, action, ctx, op, res) -> Optional[Observation]:
        await self._gate(ctx, op, action, {"tab": tab.id, "session": sess.name})
        await tab.page.close()
        sess.tabs.pop(tab.id, None)
        if sess.active == tab.id:
            sess.active = next(reversed(sess.tabs), "") if sess.tabs else ""
        op.note = f"closed {tab.id}; active tab is now {sess.active or 'none'}"
        return await self._observe(sess, sess.tabs[sess.active], mode="auto", parent=action.action_id) \
            if sess.active else None

    async def _op_select_tab(self, sess, tab, action, ctx, op, res) -> Observation:
        await self._gate(ctx, op, action, {"session": sess.name})
        sess.active = tab.id
        await tab.page.bring_to_front()
        return await self._observe(sess, tab, mode="auto", parent=action.action_id)

    async def _op_inspect(self, sess, tab, action, ctx, op, res) -> Observation:
        a = action.arguments
        await self._gate(ctx, op, action, {"page_url": tab.page.url, "session": sess.name})
        obs = await self._observe(sess, tab, mode="full" if a.get("full", True) else "diff", parent=action.action_id,
                                  a11y=True)
        if a.get("screenshot"):
            res.observations += [obs, await self._shot(sess, tab, action, res, False)]
            op.obs_after = obs.obs_id
            return None
        return obs

    async def _shot(self, sess, tab, action, res, full_page: bool) -> Observation:
        png = await tab.page.screenshot(type="png", full_page=full_page)
        path = os.path.join(self.run_dir, "artifacts", "screenshots", f"{action.action_id}-{len(res.artifacts)}.png")
        with open(path, "wb") as f:
            f.write(png)
        res.artifacts["screenshot"] = path
        obs = self.perception.image(png, uri=tab.page.url, ref=path, parent_action_id=action.action_id,
                                    source=self.modality)
        obs.data.update(surface=f"browser:{sess.name}:{tab.id}", generation=(tab.prev_state or {}).get("doc", ""),
                        tab=tab.id, session=sess.name)
        return obs

    async def _op_screenshot(self, sess, tab, action, ctx, op, res) -> Observation:
        await self._gate(ctx, op, action, {"page_url": tab.page.url, "session": sess.name})
        return await self._shot(sess, tab, action, res, bool(action.arguments.get("full_page")))

    async def _op_read(self, sess, tab, action, ctx, op, res) -> Observation:
        a = action.arguments
        await self._gate(ctx, op, action, {"page_url": tab.page.url, "session": sess.name})
        if a.get("ref") or a.get("role") or a.get("name") or a.get("selector"):
            loc, st, op.note = await self._resolve(tab, a)
            text = await loc.first.inner_text(timeout=self.cfg.op_timeout_s * 1000)
        else:
            text = await tab.page.evaluate("() => document.body ? document.body.innerText : ''")
        text = "\n".join(x for x in (re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in text.split("\n")) if x)
        off, n = int(a.get("offset", 0)), int(a.get("max_chars", self.perception.max_text - 200))
        part = text[off:off + n]
        obs = self.perception.document(part, uri=tab.page.url, media_type="text/plain",
                                       parent_action_id=action.action_id, source=self.modality)
        obs.data.update(offset=off, total_chars=len(text), more=off + n < len(text), tab=tab.id, session=sess.name)
        return obs

    async def _acted(self, sess, tab, action, ctx, before: str) -> Observation:
        """Common tail of state-changing primitives: let the page settle, authorize where it ended up, observe."""
        await self._settle(tab.page)
        target = tab
        if sess.active in sess.tabs and tab.id not in sess.tabs:
            target = sess.tabs[sess.active]
        await self._post_nav_check(sess, target, ctx, action, before)
        return await self._observe(sess, target, mode="auto", parent=action.action_id)

    async def _op_click(self, sess, tab, action, ctx, op, res) -> Observation:
        a = action.arguments
        loc, st, op.note = await self._resolve(tab, a)
        await self._gate(ctx, op, action, {"element": {k: st.get(k) for k in ("ref", "role", "name", "tag", "href", "type")
                                                       if st.get(k)}, "page_url": tab.page.url, "session": sess.name})
        if st.get("disabled"):
            raise _Fail(ResultStatus.FAILED, FailureClass.WRONG_SITE_ASSUMPTION, f"{st['role']} \"{st['name']}\" is disabled")
        before, n_tabs = tab.page.url, len(sess.tabs)
        try:
            await loc.first.click(timeout=self.cfg.op_timeout_s * 1000, button=a.get("button", "left"),
                                  click_count=2 if a.get("double") else 1)
        except Exception as e:  # noqa: BLE001
            raise self._pw_fail(e, f"click {st.get('role')} \"{st.get('name')}\"")
        if len(sess.tabs) > n_tabs:
            op.note += f"; opened tab {list(sess.tabs)[-1]}"
        return await self._acted(sess, tab, action, ctx, before)

    async def _op_fill(self, sess, tab, action, ctx, op, res) -> Observation:
        a = action.arguments
        loc, st, op.note = await self._resolve(tab, a)
        facts = {"element": {k: st.get(k) for k in ("ref", "role", "name", "tag", "type") if st.get(k)},
                 "page_url": tab.page.url, "session": sess.name}
        await self._gate(ctx, op, action, facts)
        if "secret" in a:
            if st.get("type") != "password":
                raise _Fail(ResultStatus.DENIED, FailureClass.MISSING_PERMISSION,
                            "a stored secret may only be typed into a password field")
            text = broker_get(ctx.secrets, str(a["secret"]), op, action, getattr(ctx, "authorization", None),
                              tab.page.url)
        else:
            text = str(a["text"])
        if st.get("disabled"):
            raise _Fail(ResultStatus.FAILED, FailureClass.WRONG_SITE_ASSUMPTION, f"{st['role']} \"{st['name']}\" is disabled")
        if not st.get("editable", True):
            raise _Fail(ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS, f"{st['role']} \"{st['name']}\" is not a text field")
        try:
            await loc.first.fill(text, timeout=self.cfg.op_timeout_s * 1000)
        except Exception as e:  # noqa: BLE001
            f = self._pw_fail(e, f"fill {st.get('role')} \"{st.get('name')}\"")
            if f.failure_class == FailureClass.UNKNOWN:
                f = _Fail(ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS, f.detail)
            raise f
        return await self._acted(sess, tab, action, ctx, tab.page.url)

    async def _op_select_option(self, sess, tab, action, ctx, op, res) -> Observation:
        a = action.arguments
        loc, st, op.note = await self._resolve(tab, a)
        await self._gate(ctx, op, action, {"element": {k: st.get(k) for k in ("ref", "role", "name", "tag") if st.get(k)},
                                           "page_url": tab.page.url, "session": sess.name})
        if st.get("tag") != "select":
            raise _Fail(ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS,
                        f"{st.get('role')} \"{st.get('name')}\" is not a select element")
        want = {"label": a["label"]} if "label" in a else {"value": a["value"]}
        if "label" in a and st.get("options") is not None and a["label"] not in st["options"]:
            raise _Fail(ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS,
                        f"no option labelled {a['label']!r}; options: {st['options']}")
        try:
            await loc.first.select_option(timeout=self.cfg.op_timeout_s * 1000, **want)
        except Exception as e:  # noqa: BLE001
            raise _Fail(ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS, str(e).split("\n")[0][:300])
        return await self._acted(sess, tab, action, ctx, tab.page.url)

    async def _op_press(self, sess, tab, action, ctx, op, res) -> Observation:
        a = action.arguments
        loc = None
        facts: Dict[str, Any] = {"page_url": tab.page.url, "session": sess.name}
        if a.get("ref") or a.get("role") or a.get("name") or a.get("selector"):
            loc, st, op.note = await self._resolve(tab, a)
            facts["element"] = {k: st.get(k) for k in ("ref", "role", "name", "tag", "type") if st.get(k)}
        await self._gate(ctx, op, action, facts)
        before = tab.page.url
        try:
            if loc is not None:
                await loc.first.press(str(a["key"]), timeout=self.cfg.op_timeout_s * 1000)
            else:
                await tab.page.keyboard.press(str(a["key"]))
        except Exception as e:  # noqa: BLE001
            f = self._pw_fail(e, f"press {a['key']}")
            raise _Fail(f.status, FailureClass.INVALID_ARGUMENTS if "Unknown key" in str(e) else f.failure_class, f.detail)
        return await self._acted(sess, tab, action, ctx, before)

    async def _op_scroll(self, sess, tab, action, ctx, op, res) -> Observation:
        a = action.arguments
        await self._gate(ctx, op, action, {"page_url": tab.page.url, "session": sess.name})
        if a.get("ref") or a.get("role") or a.get("name") or a.get("selector"):
            loc, st, op.note = await self._resolve(tab, a)
            await loc.first.scroll_into_view_if_needed(timeout=self.cfg.op_timeout_s * 1000)
        elif a.get("to"):
            await tab.page.evaluate("(to) => window.scrollTo(0, to === 'top' ? 0 : document.documentElement.scrollHeight)",
                                    a["to"])
        else:
            await tab.page.evaluate("([dx, dy]) => window.scrollBy({left: dx, top: dy, behavior: 'instant'})",
                                    [float(a.get("dx", 0)), float(a.get("dy", 0))])
        await self._settle(tab.page, 600)
        return await self._observe(sess, tab, mode="auto", parent=action.action_id)

    def _workspace_file(self, path: str) -> str:
        p = path if os.path.isabs(path) else os.path.join(self.workspace, path)
        real = os.path.realpath(p)
        if os.path.commonpath([real, os.path.realpath(self.workspace)]) != os.path.realpath(self.workspace):
            raise _Fail(ResultStatus.DENIED, FailureClass.MISSING_PERMISSION,
                        "only files inside the workspace may be uploaded")
        if not os.path.isfile(real):
            raise _Fail(ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS, f"no such workspace file: {path}")
        return real

    async def _op_upload(self, sess, tab, action, ctx, op, res) -> Observation:
        a = action.arguments
        real = self._workspace_file(str(a["path"]))
        loc, st, op.note = await self._resolve(tab, a)
        with open(real, "rb") as f:
            blob = f.read()
        await self._gate(ctx, op, action, {"path": real, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
                                           "element": {k: st.get(k) for k in ("ref", "role", "name", "type") if st.get(k)},
                                           "page_url": tab.page.url, "session": sess.name})
        if st.get("type") != "file":
            raise _Fail(ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS,
                        f"{st.get('role')} \"{st.get('name')}\" is not a file input")
        await loc.first.set_input_files({"name": os.path.basename(real), "mimeType": "application/octet-stream",
                                         "buffer": blob}, timeout=self.cfg.op_timeout_s * 1000)
        return await self._acted(sess, tab, action, ctx, tab.page.url)

    def _download_path(self, suggested: str) -> str:
        name = re.sub(r"[^\w.\-]+", "_", os.path.basename(suggested or "download"))[:120] or "download"
        base, ext = os.path.splitext(name)
        path, n = os.path.join(self.workspace, "downloads", name), 1
        while os.path.exists(path):
            path, n = os.path.join(self.workspace, "downloads", f"{base}-{n}{ext}"), n + 1
        return path

    async def _op_download(self, sess, tab, action, ctx, op, res) -> Observation:
        a = action.arguments
        if a.get("url"):
            url = str(a["url"])
            await self._gate(ctx, op, action, {"url": url, "session": sess.name})
            try:
                resp = await sess.context.request.get(url, timeout=self.cfg.nav_timeout_s * 1000, max_redirects=0)
            except Exception as e:  # noqa: BLE001
                raise self._pw_fail(e, f"download {url}")
            if resp.status in (301, 302, 303, 307, 308):
                raise _Fail(ResultStatus.FAILED, FailureClass.WRONG_SITE_ASSUMPTION,
                            f"download URL redirects to {resp.headers.get('location', '?')}: request that URL explicitly")
            if resp.status >= 400:
                fc = (FailureClass.PLATFORM_RESTRICTION if resp.status in (401, 403, 407, 429, 451) else
                      FailureClass.TRANSIENT_NETWORK if resp.status >= 500 else FailureClass.WRONG_SITE_ASSUMPTION)
                raise _Fail(ResultStatus.FAILED, fc, f"the site answered HTTP {resp.status}")
            body = await resp.body()
            if len(body) > self.cfg.max_download_bytes:
                raise _Fail(ResultStatus.FAILED, FailureClass.RESOURCE_LIMIT, f"download of {len(body)} bytes exceeds the limit")
            cd = resp.headers.get("content-disposition", "")
            m = re.search(r'filename="?([^";]+)"?', cd)
            path = self._download_path(a.get("filename") or (m.group(1) if m else os.path.basename(urlsplit(url).path)))
            with open(path, "wb") as f:
                f.write(body)
            mt = resp.headers.get("content-type", "")
        else:
            loc, st, op.note = await self._resolve(tab, a)
            href = st.get("href") or ""
            await self._gate(ctx, op, action, {"url": href or tab.page.url, "page_url": tab.page.url, "element": {
                k: st.get(k) for k in ("ref", "role", "name", "href") if st.get(k)}, "session": sess.name})
            sess.expect_download = True
            try:
                async with tab.page.expect_download(timeout=self.cfg.nav_timeout_s * 1000) as info:
                    await loc.first.click(timeout=self.cfg.op_timeout_s * 1000)
                dl = await info.value
                url = dl.url
                if _norm_url(url) != _norm_url(href):
                    now = self.clock.now()
                    dec = await self._check(ctx, PrimitiveOp(op="browser.download", modality=self.modality, started=now,
                                                             ended=now, arguments={"url": url, "session": sess.name}), action)
                    if dec.verdict != AuthVerdict.ALLOW:
                        await dl.cancel()
                        raise _Fail(*denial_class(dec))
                path = self._download_path(a.get("filename") or dl.suggested_filename)
                await dl.save_as(path)
                mt = ""
            except _Fail:
                raise
            except Exception as e:  # noqa: BLE001
                raise self._pw_fail(e, "download by click")
            finally:
                sess.expect_download = False
        size = os.path.getsize(path)
        if size > self.cfg.max_download_bytes:
            os.unlink(path)
            raise _Fail(ResultStatus.FAILED, FailureClass.RESOURCE_LIMIT, f"download of {size} bytes exceeds the limit")
        res.artifacts["download"] = path
        op.note = (op.note + "; " if op.note else "") + f"saved {size} bytes to {path}"
        obs = self.perception.file(path, media_type=mt, parent_action_id=action.action_id, source=self.modality)
        obs.provenance.uri = url
        obs.refs["file"] = path
        obs.data.update(path=path, bytes=size)
        return obs

    async def _op_wait_for(self, sess, tab, action, ctx, op, res) -> Observation:
        a, page = action.arguments, tab.page
        await self._gate(ctx, op, action, {"page_url": page.url, "session": sess.name})
        t_ms = float(a.get("timeout_s", 10.0)) * 1000
        try:
            if "text" in a:
                await page.wait_for_function("(t) => document.body && document.body.innerText.includes(t)",
                                             arg=a["text"], timeout=t_ms)
            if "text_gone" in a:
                await page.wait_for_function("(t) => !(document.body && document.body.innerText.includes(t))",
                                             arg=a["text_gone"], timeout=t_ms)
            if "url" in a:
                pat = re.compile(str(a["url"]))
                await page.wait_for_url(lambda u: bool(pat.search(u)), timeout=t_ms)
            if "role" in a or "name" in a:
                loc = (page.get_by_role(a["role"], name=a.get("name"), exact=True) if a.get("role")
                       else page.get_by_text(a["name"], exact=True))
                await loc.first.wait_for(state=a.get("state", "visible"), timeout=t_ms)
            if a.get("change"):
                h0 = (tab.prev_state or {}).get("text", None)
                await page.wait_for_function("(t) => t === null || (document.body ? document.body.innerText.split('\\n')"
                                             ".map(s => s.replace(/\\s+/g, ' ').trim()).filter(Boolean).join('\\n') : '') !== t",
                                             arg=h0, timeout=t_ms)
        except re.error as e:
            raise _Fail(ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS, f"bad url pattern: {e}")
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "TimeoutError":
                res.observations.append(await self._observe(sess, tab, mode="auto", parent=action.action_id))
                raise _Fail(ResultStatus.TIMEOUT, FailureClass.WRONG_SITE_ASSUMPTION,
                            f"the awaited state was not reached within {t_ms / 1000:.1f} s")
            raise self._pw_fail(e, "wait_for")
        await self._settle(page, 400)
        return await self._observe(sess, tab, mode="auto", parent=action.action_id)

    async def _op_close_session(self, sess, tab, action, ctx, op, res) -> None:
        if sess.kind == "identity":
            raise _Fail(ResultStatus.FAILED, FailureClass.INVALID_ARGUMENTS,
                        "identity sessions are closed by the operator, not by an action")
        await self._gate(ctx, op, action, {"session": sess.name})
        await self._close_session(sess)
        with self._guard:
            self._sessions.pop(sess.name, None)
        return None

    # ---- verification -----------------------------------------------------------------------------------------
    async def _verify(self, sess: _Session, cond: VerificationCondition, action: ActionSpec) -> Optional[bool]:
        tab = self._tab(sess, action.arguments.get("tab") if action.arguments.get("tab") in sess.tabs else None)
        page, spec = tab.page, cond.spec or {}
        await self._settle(page, 600)
        if cond.kind == "text_present":
            text = await page.evaluate("() => document.body ? document.body.innerText : ''")
            want = str(spec.get("text", ""))
            found = (want in text) if spec.get("case_sensitive", True) else (want.lower() in text.lower())
            return found != bool(spec.get("absent", False))
        if cond.kind == "url_matches":
            try:
                return bool(re.search(str(spec.get("pattern", "")), page.url))
            except re.error:
                return None
        role, name, state = spec.get("role"), spec.get("name"), spec.get("state", "visible")
        loc = (page.get_by_role(role, name=name, exact=True) if role else
               page.locator(spec["selector"]) if spec.get("selector") else page.get_by_text(str(name), exact=True))
        n = await loc.count()
        if state in ("hidden", "absent", "detached"):
            return n == 0 or not await loc.first.is_visible()
        if n == 0:
            return False
        el = loc.first
        if state == "visible":
            return await el.is_visible()
        if state in ("checked", "unchecked"):
            return (await el.is_checked()) == (state == "checked")
        if state in ("enabled", "disabled"):
            return (await el.is_enabled()) == (state == "enabled")
        if state == "value":
            return (await el.input_value()) == str(spec.get("value", ""))
        return None


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m cam_live.harness.browser",
                                 description="Chromium for the browser executor: shared install and lookup")
    ap.add_argument("--install-shared", action="store_true",
                    help="copy this user's Playwright Chromium to the shared location (operator, once)")
    ap.add_argument("--dest", default="", help=f"shared location (default {SHARED_BROWSERS_DIR})")
    ap.add_argument("--find", default="", metavar="USER", help="print the Chromium that USER can execute")
    a = ap.parse_args(argv)
    if a.install_shared:
        print(install_shared_chromium(a.dest))
        return 0
    if a.find:
        p = pwd.getpwnam(a.find)
        exe = find_chromium(p.pw_uid, p.pw_gid, BrowserExecutor._bundled_chromium())
        print(exe or "none")
        return 0 if exe else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
