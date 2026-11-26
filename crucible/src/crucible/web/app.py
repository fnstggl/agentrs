"""Browser mode: real HTML pages for every Crucible service, served from one FastAPI app.

Routing is by host name, like the synthetic internet it renders: ``bank.crucible`` is the bank, ``id.crucible`` the
identity portal and home page, ``<host>.crucible`` an agent-deployed service. Clients that cannot set the Host
header (or resolve ``*.crucible``) use the path prefix ``/_h/<host>/...`` on the server's own address instead.

Two rules make browser mode equivalent to the tool API:

* **Page renders never change the world.** Every GET renders through `World.query` (a read-only endpoint
  evaluated in a transaction that is always rolled back): nothing is recorded, nothing is charged.
* **Every form submission is one recorded command.** A POST to ``/_act`` converts the form fields to the tool's
  arguments with the tool's own JSON schema and calls `World.execute(actor, {"tool", "args"})` - the very call an
  API agent makes - as the logged-in actor. Errors come back as the ActionResult's error code and message.

So a scenario played through forms and the same tool calls made headlessly produce the same event log and the
same `World.digest()` (tested in ``tests/test_web.py``). Login (`id.crucible/login`) checks the password set by
``admin.join`` (or an API token issued by a `WorldServer`); sessions are server-side, carried by a host-only
cookie and propagated to other hosts by a one-time single-sign-on ticket. ``admin.*`` tools are never offered or
executed. All content is HTML-escaped by Jinja's autoescaping and pages carry a script-free CSP.

Run: ``python -m crucible.web.app --store sqlite:///worlds.db --world <id> --port 8080`` (see ``--help``).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import threading
import time as _time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import parse_qsl, quote, urlencode, urlsplit

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..api.render import compact_json, fmt_time
from ..core.errors import CrucibleError
from ..core.kernel import World

__all__ = ["make_web_app", "WebBackend", "form_to_args", "SERVICES", "main"]

HERE = os.path.dirname(os.path.abspath(__file__))
COOKIE = "crucible_sid"
HOST_RE = re.compile(r"[a-z][a-z0-9-]{0,30}")
MENTION_RE = re.compile(r"\b([a-z][a-z0-9-]{1,30})\.crucible\b")
HIDDEN_HOSTS = ("admin",)

# (host, menu label, description) - the well-known part of the internet, in menu order
SERVICES = [
    ("id", "Home", "identity, organisations, invitations, grants"),
    ("bank", "Bank", "accounts, transfers, statements, invoices, direct debits"),
    ("mail", "Mail", "private messages"),
    ("forum", "Forum", "public boards, posts, replies, votes"),
    ("search", "Search", "search the public web of this world"),
    ("docs", "Docs", "documents, versions, sharing"),
    ("code", "Code", "repositories, commits, sandboxed runs and tests"),
    ("issues", "Issues", "issue trackers"),
    ("cal", "Calendar", "events and invitations"),
    ("support", "Support", "support tickets with service operators"),
    ("ads", "Ads", "search-ad campaigns"),
    ("market", "Market", "listings, orders, ratings"),
    ("cloud", "Cloud", "compute providers, instances, deployments"),
    ("biz", "Registry", "company registry"),
    ("contracts", "Contracts", "contracts and claims"),
    ("jobs", "Jobs", "job openings, applications, bounties"),
    ("research", "Research", "market surveys"),
]
SERVICE_HOSTS = {h for h, _, _ in SERVICES}

MONEY_KEYS = {"balance", "amount", "price_cents", "reward_cents", "salary_cents_month", "bid_cents",
              "daily_budget_cents", "spent_today", "total_spent", "overdraft", "charged_cents",
              "price_core_hour_cents", "max_per_period", "salary", "price", "fee", "revenue_cents"}
TIME_KEYS = {"t", "since", "created", "updated", "start", "due", "deadline", "opened", "delivered",
             "first_response", "resolved", "expires", "time"}
TEXTAREA_NAMES = {"body", "text", "description", "note", "content", "bio", "reason", "comment"}


# ====================================================================== backend
class WebBackend:
    """Where the world lives: a `WorldServer` (+ world id), a `World`, or a zero-argument callable returning one.

    All world access goes through `lock` (the WorldServer's own lock when there is one), so the FastAPI
    threadpool is safe; `after_write` flushes recorded events to the store."""

    def __init__(self, provider: Any, world_id: str | None = None):
        self.server = None
        self.world_id = world_id
        if isinstance(provider, World):
            self._get = lambda: provider
            self.lock = threading.RLock()
        elif hasattr(provider, "execute_action") and hasattr(provider, "world"):   # a WorldServer
            if world_id is None:
                raise ValueError("make_web_app(server, world_id): a WorldServer needs the world id")
            self.server = provider
            self._get = lambda: provider.world(world_id)
            self.lock = getattr(provider, "_lock", None) or threading.RLock()
        elif callable(provider):
            self._get = provider
            self.lock = threading.RLock()
        else:
            raise TypeError("provider must be a WorldServer, a World or a callable returning a World")

    def world(self) -> World:
        return self._get()

    def after_write(self, w: World) -> None:
        if self.server is not None and hasattr(self.server, "_flush"):
            self.server._flush(w)
        elif w.store is not None:
            w.flush()

    def token_principal(self, token: str) -> str | None:
        if self.server is None or not token.startswith("cru_"):
            return None
        try:
            return self.server.authenticate(self.world_id, token).principal
        except Exception:  # noqa: BLE001 - any auth failure is just "no"
            return None


# ====================================================================== sessions
@dataclass
class Session:
    sid: str
    principal: str
    csrf: str
    seen_seq: int = 0
    results: "OrderedDict[str, dict]" = field(default_factory=OrderedDict)
    mail: "OrderedDict[str, dict]" = field(default_factory=OrderedDict)
    n: int = 0

    def remember(self, rec: dict) -> str:
        self.n += 1
        rid = str(self.n)
        self.results[rid] = rec
        while len(self.results) > 40:
            self.results.popitem(last=False)
        return rid


class Sessions:
    def __init__(self):
        self._lock = threading.Lock()
        self.by_id: dict[str, Session] = {}
        self.tickets: dict[str, tuple[str, float]] = {}

    def create(self, principal: str, seen_seq: int) -> Session:
        s = Session(secrets.token_urlsafe(24), principal, secrets.token_urlsafe(18), seen_seq)
        with self._lock:
            self.by_id[s.sid] = s
        return s

    def get(self, sid: str | None) -> Session | None:
        return self.by_id.get(sid) if sid else None

    def drop(self, sid: str) -> None:
        with self._lock:
            self.by_id.pop(sid, None)
            for t in [t for t, (s, _) in self.tickets.items() if s == sid]:
                self.tickets.pop(t, None)

    def ticket(self, sid: str) -> str:
        t = secrets.token_urlsafe(24)
        now = _time.monotonic()
        with self._lock:
            for k in [k for k, (_, exp) in self.tickets.items() if exp < now]:
                self.tickets.pop(k, None)
            self.tickets[t] = (sid, now + 60)
        return t

    def redeem(self, ticket: str) -> Session | None:
        with self._lock:
            rec = self.tickets.pop(ticket, None)
        if rec is None or rec[1] < _time.monotonic():
            return None
        return self.get(rec[0])


# ====================================================================== cells (rendered by macros.html)
class Link:
    kind = "link"

    def __init__(self, text: Any, href: str, id: str | None = None, label: str | None = None, desc: str = ""):
        self.text, self.href, self.id, self.label, self.desc = str(text), href, id, label, desc


class Rich:
    """Text in which `<host>.crucible` mentions become links (everything else stays escaped text)."""
    kind = "rich"

    def __init__(self, parts: list):
        self.parts = parts


class Pre:
    kind = "pre"

    def __init__(self, text: str, id: str | None = None):
        self.text, self.id = text, id


class Many:
    kind = "many"

    def __init__(self, items: list, sep: str = ", "):
        self.items, self.sep = items, sep


def money(c: Any) -> str:
    if not isinstance(c, (int, float)) or isinstance(c, bool):
        return str(c)
    sign = "-" if c < 0 else ""
    c = abs(c)
    if isinstance(c, float) and c != int(c):
        return f"{sign}${c / 100:,.4f}"
    c = int(c)
    return f"{sign}${c // 100:,}.{c % 100:02d}"


# ====================================================================== forms
@dataclass
class Field:
    name: str
    label: str
    kind: str                     # text | number | textarea | json | select | checkbox | hidden | pathcontent
    id: str = ""
    required: bool = False
    value: Any = ""
    help: str = ""
    options: list = field(default_factory=list)
    attrs: list = field(default_factory=list)
    rows: int = 4
    input_type: str = "text"


@dataclass
class Form:
    id: str
    action: str
    tool: str = ""
    method: str = "post"
    title: str = ""
    doc: str = ""
    submit: str = "Submit"
    submit_aria: str = ""
    label: str = ""
    fields: list = field(default_factory=list)
    hidden: list = field(default_factory=list)
    buttons: list = field(default_factory=list)      # [(name, value, text, aria-label)]
    inline: bool = False
    kind: str = "form"


def _stype(s: dict) -> str | None:
    t = s.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    return t


def _scalar_items(s: dict) -> bool:
    it = s.get("items") or {}
    return _stype(it) in ("string", "integer", "number", "boolean") or "enum" in it


class FormError(ValueError):
    pass


def _coerce(name: str, s: dict, raw: str) -> Any:
    if "enum" in s:
        for e in s["enum"]:
            if str(e) == raw or (isinstance(e, bool) and raw.lower() == str(e).lower()):
                return e
        raise FormError(f"{name}: {raw!r} is not one of {', '.join(str(e) for e in s['enum'])}")
    t = _stype(s)
    if t == "string":
        return raw
    if t == "integer":
        try:
            return int(raw.strip().replace("_", ""))
        except ValueError:
            raise FormError(f"{name}: {raw!r} is not a whole number") from None
    if t == "number":
        try:
            return float(raw.strip())
        except ValueError:
            raise FormError(f"{name}: {raw!r} is not a number") from None
    if t == "boolean":
        v = raw.strip().lower()
        if v in ("true", "on", "1", "yes"):
            return True
        if v in ("false", "off", "0", "no"):
            return False
        raise FormError(f"{name}: {raw!r} is not true/false")
    if t == "array":
        if _scalar_items(s):
            items = [x.strip() for x in re.split(r"[,\n]", raw) if x.strip()]
            return [_coerce(name, s.get("items") or {}, x) for x in items]
        return _json(name, raw, list)
    if t == "object":
        return _json(name, raw, dict)
    try:                                                  # untyped: JSON if it parses, else the text itself
        return json.loads(raw)
    except ValueError:
        return raw


def _json(name: str, raw: str, typ: type) -> Any:
    try:
        v = json.loads(raw)
    except ValueError as e:
        raise FormError(f"{name}: invalid JSON ({e})") from None
    if not isinstance(v, typ):
        raise FormError(f"{name}: expected a JSON {'object' if typ is dict else 'array'}")
    return v


def form_to_args(params: dict, data: dict[str, list[str]]) -> dict:
    """Convert submitted form fields to tool arguments using the tool's JSON schema.

    Empty optional fields are omitted (the kernel fills schema defaults, exactly as for an API call that leaves
    them out); integers, numbers, booleans, enums, comma-separated lists and JSON objects are parsed. A field
    ``<name>.path`` + ``<name>.content`` builds a one-file object ``{path: content}`` (the commit editor); a
    ``_args`` field holds a JSON object of extra arguments (for schemas without declared properties).
    Raises `FormError` for input that cannot be converted (nothing is sent to the world then)."""
    props = params.get("properties") or {}
    req = set(params.get("required") or [])
    args: dict[str, Any] = {}
    extra = (data.get("_args") or [""])[-1].strip()
    if extra:
        args.update(_json("arguments", extra, dict))
    for name, s in props.items():
        if f"{name}.path" in data:
            path = (data.get(f"{name}.path") or [""])[-1].strip()
            if not path:
                if name in req:
                    raise FormError(f"{name}: a file path is required")
                continue
            args[name] = {path: (data.get(f"{name}.content") or [""])[-1]}
            continue
        vals = data.get(name)
        if not vals:
            continue
        raw = vals[-1]
        if raw == "" or (raw.strip() == "" and _stype(s) != "string"):
            if name in req and _stype(s) == "string" and "enum" not in s:
                args[name] = raw
            continue
        args[name] = _coerce(name, s, raw)
    return args


# ====================================================================== request context
class PageError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


class Ctx:
    """One request: the world, the actor, the host, URL helpers and the page being built."""

    def __init__(self, web: "WebApp", w: World, req: Request, host: str, path: str, mode: str,
                 session: Session | None, query: dict[str, str]):
        self.web, self.w, self.req, self.host, self.path, self.mode = web, w, req, host, path, mode
        self.session = session
        self.p = session.principal if session else None
        self.qp = query
        self.title = host
        self.sections: list[dict] = []
        self.errors: list[str] = []
        self._ids: dict[str, int] = {}
        self._cache: dict[str, Any] = {}
        self.result_rec = session.results.get(query.get("r", "")) if session else None

    # ------------------------------------------------------------------ urls
    def base(self, host: str) -> str:
        if self.mode == "path":
            return f"/_h/{host}"
        hp = self.req.headers.get("host", "")
        port = hp.rsplit(":", 1)[1] if ":" in hp else ""
        return f"{self.req.url.scheme}://{host}.crucible{':' + port if port else ''}"

    def url(self, host: str | None = None, path: str = "/", /, **params) -> str:
        host = host or self.host
        params = {k: v for k, v in params.items() if v is not None and v != ""}
        qs = ("?" + urlencode(params)) if params else ""
        if host == self.host:
            prefix = f"/_h/{host}" if self.mode == "path" else ""
            return f"{prefix}{path}{qs}"
        return f"{self.base(host)}{path}{qs}"

    def here(self) -> str:
        """The current page (without the result marker) as a local URL."""
        q = {k: v for k, v in self.qp.items() if k != "r"}
        return self.url(self.host, self.path, **q)

    # ------------------------------------------------------------------ queries (never recorded)
    def q(self, tool: str, **args) -> Any:
        ep = self.w.endpoints.get(tool)
        if ep is not None:                       # page sizes: clamp to what the endpoint accepts
            props = (ep.params or {}).get("properties") or {}
            for k in ("limit", "replies", "comments"):
                if isinstance(args.get(k), int) and k in props:
                    lo, hi = props[k].get("minimum"), props[k].get("maximum")
                    args[k] = min(args[k], hi) if hi is not None else args[k]
                    args[k] = max(args[k], lo) if lo is not None else args[k]
        try:
            return self.w.query(self.p, tool, args)
        except CrucibleError as e:
            raise PageError(f"{e.code}: {e.message}", 404 if e.code == "not_found" else 400) from None

    def tq(self, tool: str, **args) -> Any:
        """Query, showing a failure as a page error instead of aborting the page."""
        try:
            return self.q(tool, **args)
        except PageError as e:
            self.errors.append(f"{tool}: {e.message}")
            return None

    def accounts(self) -> list[dict]:
        if "accounts" not in self._cache:
            try:
                acc = self.w.query(self.p, "bank.accounts", {}) if "bank.accounts" in self.w.endpoints else []
            except CrucibleError:
                acc = []
            acc = sorted(acc, key=lambda a: (a.get("owner") != self.p, a.get("account", "")))
            self._cache["accounts"] = acc
        return self._cache["accounts"]

    def orgs(self) -> list[dict]:
        if "orgs" not in self._cache:
            try:
                me = self.w.query(self.p, "id.me", {}) if "id.me" in self.w.endpoints else {}
            except CrucibleError:
                me = {}
            self._cache["orgs"] = list(me.get("orgs") or [])
        return self._cache["orgs"]

    def account_options(self) -> list[tuple[str, str]]:
        return [(a["account"], f"{a['account']} - {a.get('name', '')} ({a.get('owner')}, "
                               f"{a.get('balance_fmt') or money(a.get('balance', 0))})") for a in self.accounts()]

    def org_options(self, blank: str | None = None) -> list[tuple[str, str]]:
        opts = [("", blank)] if blank is not None else []
        return opts + [(o["org"], f"{o['org']} ({o['role']})") for o in self.orgs()]

    # ------------------------------------------------------------------ cells
    def who(self, pid: Any) -> Any:
        if not isinstance(pid, str):
            return pid
        if pid.startswith(("p:", "o:")):
            return Link(pid, self.url("id", "/profile", p=pid))
        if pid.startswith("svc:"):
            return Link(pid, self.url(pid[4:], "/"))
        return pid

    def acct(self, a: Any) -> Any:
        return Link(a, self.url("bank", "/account", id=a)) if isinstance(a, str) and a.startswith("acct_") else a

    def rich(self, text: Any) -> Rich:
        text = "" if text is None else str(text)
        parts: list = []
        pos = 0
        for mt in MENTION_RE.finditer(text):
            if mt.start() > pos:
                parts.append(text[pos:mt.start()])
            parts.append(Link(mt.group(0), self.url(mt.group(1), "/")))
            pos = mt.end()
        if pos < len(text):
            parts.append(text[pos:])
        return Rich(parts)

    def auto(self, key: str, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool):
            return "yes" if v else "no"
        if key in MONEY_KEYS and isinstance(v, (int, float)):
            return money(v)
        if key in TIME_KEYS and isinstance(v, int):
            return fmt_time(v)
        if key in ("owner", "author", "seller", "buyer", "payer", "creator", "organiser", "advertiser", "poster",
                   "customer", "principal", "holder", "applicant", "from", "proposer", "assignee", "org",
                   "provider", "grantee", "person", "payee_owner", "mailbox", "by", "employer", "employee"):
            return self.who(v)
        if key in ("account", "payee_acct", "from_account", "to_account"):
            return self.acct(v)
        if key in ("hosts",) and isinstance(v, list):
            return Many([Link(f"{h}.crucible", self.url(h, "/")) for h in v]) if v else None
        if key in ("to", "parties", "signed", "orgs") and isinstance(v, list):
            return Many([self.who(x) for x in v])
        if isinstance(v, list) and all(isinstance(x, (str, int, float)) for x in v):
            return ", ".join(str(x) for x in v)
        if isinstance(v, (dict, list)):
            return compact_json(v, 300)
        return v

    # ------------------------------------------------------------------ page building
    def uid(self, base: str) -> str:
        base = re.sub(r"[^A-Za-z0-9_-]+", "-", base).strip("-") or "x"
        n = self._ids.get(base, 0) + 1
        self._ids[base] = n
        return base if n == 1 else f"{base}-{n}"

    def add(self, kind: str, title: str = "", id: str | None = None, **kw) -> dict:
        s = {"kind": kind, "title": title, "id": self.uid(id or ("s-" + (title or kind).lower())), **kw}
        self.sections.append(s)
        return s

    def table(self, title: str, rows: list[dict] | None, cols: list, *, empty: str = "Nothing here yet.",
              note: str = "", id: str | None = None) -> dict:
        """`cols`: (key, label) or (key, label, fn(row) -> cell)."""
        out = []
        for r in rows or []:
            line = []
            for c in cols:
                key, fn = c[0], (c[2] if len(c) > 2 else None)
                line.append(fn(r) if fn else self.auto(key, r.get(key) if isinstance(r, dict) else None))
            out.append(line)
        return self.add("table", title, id=id, cols=[c[1] for c in cols], rows=out, empty=empty, note=note)

    def kv(self, title: str, d: dict, keys: list | None = None, labels: dict | None = None,
           id: str | None = None, extra: list | None = None) -> dict:
        labels = labels or {}
        keys = keys if keys is not None else list(d.keys())
        items = [(labels.get(k, k.replace("_", " ")), self.auto(k, d.get(k))) for k in keys if k in d]
        items += extra or []
        return self.add("kv", title, id=id, pairs=items)

    def note(self, text: Any, id: str | None = None) -> dict:
        return self.add("note", "", id=id or "note", text=text)

    def tabs(self, title: str, links: list[tuple[str, str, bool]]) -> dict:
        return self.add("tabs", title, id="tabs", links=[{"text": t, "href": h, "current": cur,
                                                          "id": self.uid("tab-" + t.lower())}
                                                         for t, h, cur in links])

    def links(self, title: str, links: list[Link], empty: str = "None.") -> dict:
        return self.add("links", title, links=links, empty=empty)

    def forms(self, title: str, forms: list[Form | None], note: str = "", id: str | None = None) -> dict | None:
        forms = [f for f in forms if f is not None]
        if not forms:
            return None
        return self.add("forms", title, id=id, forms=forms, note=note)

    def form(self, tool: str, *, title: str | None = None, fields: list[str] | None = None,
             hide: dict | None = None, values: dict | None = None, widgets: dict | None = None,
             labels: dict | None = None, helps: dict | None = None, submit: str | None = None,
             submit_aria: str = "", next: str | None = None, key: str | None = None, inline: bool = False,
             buttons: list | None = None, doc: bool = True, omit: list[str] | None = None) -> Form | None:
        """A form that executes `tool` (POST /_act) with fields generated from the tool's JSON schema.

        `fields` (order/subset), `hide` {name: value} (hidden inputs), `values` (prefill), `widgets`
        {name: kind or {"kind", "options", ...}}, `next` (local path on success; `{key}` is replaced by the
        result's `key`), `key` (suffix for unique ids), `buttons` [(name, value, text, aria)] (several submit
        buttons that each set one argument)."""
        try:
            ep = self.w.resolve(tool, self.p)
        except CrucibleError:
            return None
        if ep.host in HIDDEN_HOSTS:
            return None
        params = ep.params or {}
        props = params.get("properties") or {}
        req = set(params.get("required") or [])
        hide = dict(hide or {})
        values = dict(values or {})
        widgets = dict(widgets or {})
        labels = labels or {}
        helps = helps or {}
        fid = self.uid(tool.replace(".", "-") + (f"-{key}" if key else ""))
        prev = self.result_rec if (self.result_rec and self.result_rec.get("form_id") == fid
                                   and not self.result_rec.get("ok")) else None
        names = fields if fields is not None else list(props.keys())
        names = [n for n in names if n in props and n not in (omit or [])]
        button_names = {b[0] for b in buttons or []}
        flds = []
        for n in list(hide) + [n for n in names if n not in hide and n not in button_names]:
            s = props.get(n, {})
            if n in hide:
                v = hide[n]
                flds.append(Field(n, n, "hidden", value=_fmt_value(v)))
                continue
            f = self._field(fid, n, s, n in req, widgets.get(n), labels.get(n), helps.get(n))
            if n in values:
                f.value = values[n] if f.kind == "pathcontent" else _fmt_value(values[n])
            if prev is not None:
                raw = prev.get("form") or {}
                if f.kind == "pathcontent":
                    f.value = ((raw.get(f"{n}.path") or [""])[-1], (raw.get(f"{n}.content") or [""])[-1])
                elif f.kind == "checkbox":
                    f.value = bool(raw.get(n))
                elif n in raw:
                    f.value = raw[n][-1]
            flds.append(f)
        if params.get("additionalProperties") is True:
            flds.append(Field("_args", "Arguments (JSON object)", "json", id=f"{fid}-args", rows=5,
                              value=(prev.get("form", {}).get("_args", [""])[-1] if prev else "{}"),
                              help="This endpoint does not declare its parameters; see its description."))
        hidden = [("_tool", tool), ("_csrf", self.session.csrf if self.session else ""), ("_form", fid),
                  ("_back", self.here())]
        if next:
            hidden.append(("_next", next))
        return Form(id=fid, action=self.url(self.host, "/_act"), tool=tool,
                    title=(title if title is not None else ("" if inline else tool)),
                    doc=(ep.doc.strip() if doc and not inline else ""), submit=submit or _verb(tool),
                    submit_aria=submit_aria, fields=flds, hidden=hidden, buttons=buttons or [], inline=inline)

    def _field(self, fid: str, n: str, s: dict, required: bool, widget: Any, label: str | None,
               help_: str | None) -> Field:
        t = _stype(s)
        desc = (s.get("description") or "").strip()
        bits = [desc] if desc else []
        if n in MONEY_KEYS or "cents" in n or desc == "cents":
            bits = [x for x in bits if x != "cents"] + ["in cents (100 = $1.00)"]
        if "default" in s:
            bits.append(f"default {json.dumps(s['default'])}")
        lab = label or n.replace("_", " ").capitalize()
        f = Field(n, lab, "text", id=f"{fid}-{n}", required=required)
        w = widget if isinstance(widget, dict) else ({"kind": widget} if widget else {})
        kind = w.get("kind")
        if kind is None:
            if "enum" in s:
                kind = "select"
            elif t == "boolean":
                kind = "select" if required else "checkbox"
            elif t in ("integer", "number"):
                kind = "number"
            elif t == "array":
                kind = "text" if _scalar_items(s) else "json"
            elif t == "object":
                kind = "json"
            elif t is None:
                kind = "json"
            elif n in TEXTAREA_NAMES or (s.get("maxLength", 0) or 0) > 1000:
                kind = "textarea"
            elif n.endswith("account") and n != "to" and self.accounts():
                kind = "accounts"
            elif n in ("org", "as_org", "from_org") and self.orgs():
                kind = "orgs"
            else:
                kind = "text"
        if kind == "select" and not w.get("options"):
            if "enum" in s:
                opts = [(str(e), str(e)) for e in s["enum"]]
            else:
                opts = [("true", "yes"), ("false", "no")]
            if not required:
                opts = [("", f"(default{': ' + str(s['default']) if 'default' in s else ''})")] + opts
            f.options = opts
        elif kind == "accounts":
            kind = "select"
            f.options = ([] if required else [("", "(none)")]) + self.account_options()
        elif kind == "orgs":
            kind = "select"
            f.options = [("", "(none - yourself)" if not required else "(choose an organisation)")] \
                + self.org_options()
        if w.get("options"):
            f.options = list(w["options"])
        if kind == "number":
            f.input_type = "number"
            f.attrs = [(k2, s[k1]) for k1, k2 in (("minimum", "min"), ("maximum", "max")) if k1 in s]
            f.attrs.append(("step", "any" if t == "number" else "1"))
        if kind == "text" and t == "array":
            bits.append("comma-separated")
        if kind == "json":
            bits.append("JSON")
            f.rows = 6
        if kind == "textarea":
            f.rows = w.get("rows", 8 if n in ("body", "content") else 4)
        if kind == "pathcontent":
            f.value = ("", "")
        f.kind = kind
        f.help = help_ if help_ is not None else "; ".join(bits)
        return f

    def get_form(self, id: str, action: str, fields: list[Field], submit: str, label: str) -> Form:
        """A GET form (page filter): changes what is rendered, never the world."""
        fid = self.uid(id)
        for f in fields:
            f.id = f"{fid}-{f.name}"
        return Form(id=fid, action=action, method="get", submit=submit, fields=fields, label=label)


def _fmt_value(v: Any) -> Any:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v, indent=1)
    return "" if v is None else str(v)


_VERBS = {"transfer": "Send money", "open_account": "Open account", "post": "Publish", "reply": "Reply",
          "send": "Send", "create": "Create", "create_org": "Create organisation", "register": "Register company",
          "query": "Search", "commit": "Commit", "run": "Run", "test": "Run tests", "buy": "Buy", "list": "List",
          "rent": "Rent", "deploy": "Deploy", "propose": "Propose", "sign": "Sign", "apply": "Apply",
          "survey": "Commission survey", "edit": "Save", "share": "Share", "comment": "Comment",
          "invoice": "Send invoice", "pay_invoice": "Pay", "vote": "Vote", "delete": "Delete", "read": "Open"}


def _verb(tool: str) -> str:
    ep = tool.split(".", 1)[-1]
    return _VERBS.get(ep, ep.replace("_", " ").capitalize())


# ====================================================================== the app
Route = tuple[str, "re.Pattern[str]", Callable[[Ctx], Any]]


class WebApp:
    def __init__(self, backend: WebBackend, *, secure_cookies: bool = False, notifications: int = 8):
        self.b = backend
        self.sessions = Sessions()
        self.secure = secure_cookies
        self.n_notifications = notifications
        self.env = Environment(loader=FileSystemLoader(os.path.join(HERE, "templates")),
                               autoescape=select_autoescape(["html"]), trim_blocks=False, lstrip_blocks=False)
        self.routes: dict[str, list[Route]] = {}
        _register_pages(self)

    def route(self, host: str, pattern: str, method: str = "GET"):
        def deco(fn):
            self.routes.setdefault(host, []).append((method, re.compile(pattern + r"\Z"), fn))
            return fn
        return deco

    # ------------------------------------------------------------------ dispatch
    def handle(self, req: Request, host: str, path: str, mode: str, form: dict[str, list[str]] | None) -> Response:
        query = dict(parse_qsl(req.url.query, keep_blank_values=True))
        session = self.sessions.get(req.cookies.get(COOKIE))
        with self.b.lock:
            w = self.b.world()
            if session is not None:
                p = w.principal(session.principal)
                if p is None or p.get("status") != "active":
                    self.sessions.drop(session.sid)
                    session = None
            c = Ctx(self, w, req, host, path, mode, session, query)
            if path == "/_sso":
                return self._sso(c)
            if path == "/_logout" and req.method == "POST":
                return self._logout(c, form or {})
            if host == "id" and path == "/login":
                return self._login(c, form) if req.method == "POST" else self._login_page(c)
            if session is None:
                return self._to_login(c)
            net = w.modules.get("net")
            if net is not None and host not in self.routes and host not in HIDDEN_HOSTS:
                target = net.resolve_host(w.state, host)
                if target is not None and net.web_dep(w.state, target)[1] is not None:
                    return self._webapp(c, host, path, form)
            if not self._host_exists(w, host):
                return self.render(c, status=404, title=f"{host}.crucible not found",
                                   errors=[f"There is no service at {host}.crucible (DNS lookup failed)."])
            if path == "/_act" and req.method == "POST":
                return self._act(c, form or {})
            for method, pat, fn in self.routes.get(host if host in self.routes else "*", []):
                mt = pat.match(path)
                if mt and method == req.method:
                    try:
                        out = fn(c, **mt.groupdict())
                    except PageError as e:
                        c.errors.append(e.message)
                        return self.render(c, status=e.status)
                    if isinstance(out, Response):
                        return out
                    return self.render(c)
            return self.render(c, status=404, title="Page not found",
                               errors=[f"No page {path} on {host}.crucible."])

    def _webapp(self, c: Ctx, host: str, path: str, form: dict[str, list[str]] | None) -> Response:
        """A web application on the synthetic internet: every request is the visitor's `net.fetch` command, so the
        browser and the API see the same application with the same recorded effects."""
        w = c.w
        g = lambda k: ((form or {}).get(k) or [""])[-1]  # noqa: E731
        if path == "/_login":
            back = c.qp.get("return") or g("return") or "/"
            if not back.startswith("/") or back.startswith("//"):
                back = "/"
            if c.req.method == "POST":
                if g("_csrf") != c.session.csrf:
                    return self.render(c, status=403, title="Forbidden", errors=["Bad CSRF token."])
                w.execute(c.p, {"tool": "net.login", "args": {"site": host}})
                self.b.after_write(w)
                return _redirect(c.url(host, back))
            c.title = f"Log in to {host}.crucible"
            c.add("form", id="applogin", form=Form(
                id="applogin", action=c.url(host, "/_login"), title=f"{host}.crucible wants to know who you are",
                doc="The site learns your principal id and nothing else.", submit=f"Log in as {c.p}",
                hidden=[("return", back), ("_csrf", c.session.csrf)]))
            return self.render(c)
        url = f"http://{host}.crucible{path}" + (f"?{c.req.url.query}" if c.req.url.query else "")
        args: dict[str, Any] = {"url": url, "method": c.req.method if c.req.method in ("GET", "POST") else "GET"}
        if args["method"] == "POST":
            args["form"] = {k: (v[-1] if v else "") for k, v in (form or {}).items() if not k.startswith("_")}
        r = w.execute(c.p, {"tool": "net.fetch", "args": args})
        self.b.after_write(w)
        if not r.ok:
            return self.render(c, status=400, title=f"{host}.crucible", errors=[r.error.get("message", "error")])
        res = r.result
        if res.get("login_required"):
            return _redirect(c.url(host, "/_login", **{"return": path}))
        if res.get("location"):
            loc = str(res["location"])
            if loc.startswith("/") and not loc.startswith("//"):
                loc = c.url(host, loc)
            elif not (urlsplit(loc).hostname or "").endswith(".crucible"):
                loc = c.url(host, "/")
            return _redirect(loc)
        headers = {"Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                                              "connect-src 'self'; object-src 'none'; base-uri 'none'; "
                                              "frame-ancestors 'none'; img-src 'self' data:; "
                                              "style-src 'self' 'unsafe-inline'",
                   "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"}
        return Response(content=res["body"], status_code=int(res["status"]), media_type=res["content_type"],
                        headers=headers)

    def _host_exists(self, w: World, host: str) -> bool:
        if host in HIDDEN_HOSTS:
            return False
        if host == "id" or any(e.host == host for e in w.endpoints.values()):
            return True
        fn = w.hooks.get("host_endpoints")
        return bool(fn and fn(w, host, SYSTEM_VIEW))

    # ------------------------------------------------------------------ rendering
    def render(self, c: Ctx, template: str = "page.html", status: int = 200, title: str | None = None,
               errors: list[str] | None = None, **extra) -> HTMLResponse:
        w = c.w
        if errors:
            c.errors.extend(errors)
        ctx = {"title": title or c.title, "host": c.host, "sections": c.sections, "errors": c.errors,
               "world_id": self.b.world_id or w.world_id, "clock": fmt_time(w.now), "seq": w.seq,
               "static_url": "/_static", "home_url": c.url("id", "/"), "principal": c.p,
               "me_url": c.url("id", "/profile", p=c.p) if c.p else "",
               "logout_url": c.url(c.host, "/_logout"), "csrf": c.session.csrf if c.session else "",
               "notifications_url": c.url("id", "/notifications"), "result": None,
               "nav": [{"host": h, "label": lab, "href": c.url(h, "/")} for h, lab, _ in SERVICES
                       if h == "id" or any(e.host == h for e in w.endpoints.values())]}
        if c.session is not None:
            ctx["notifications"], ctx["n_new"] = self._notifications(c)
            ctx["result"] = self._result_view(c)
        ctx.update(extra)
        html = self.env.get_template(template).render(**ctx)
        return HTMLResponse(html, status_code=status, headers=_SEC_HEADERS)

    def _notifications(self, c: Ctx) -> tuple[list[dict], int]:
        w, s = c.w, c.session
        evs, _ = w.visible_events(c.p, max(0, w.seq - 5000), limit=5000)
        mine = [e for e in evs if isinstance(e.vis, tuple) and e.type not in ("cmd", "cmd.rejected", "ctr", "idem")]
        mine = mine[-self.n_notifications:][::-1]
        out = [{"time": fmt_time(e.t), "type": e.type, "actor": e.actor, "text": compact_json(e.payload, 180),
                "new": e.seq > s.seen_seq, "seq": e.seq} for e in mine]
        n_new = sum(1 for x in out if x["new"])
        s.seen_seq = max(s.seen_seq, w.seq)
        return out, n_new

    def _result_view(self, c: Ctx) -> dict | None:
        rec = c.result_rec
        if rec is None:
            return None
        if rec.get("ok"):
            res = rec.get("result")
            items = []
            if isinstance(res, dict):
                items = [(k, c.auto(k, v)) for k, v in list(res.items())[:30]]
            elif res is not None:
                items = [("result", compact_json(res, 600))]
            return {"ok": True, "tool": rec["tool"], "pairs": items, "summary": ""}
        err = rec.get("error") or {}
        return {"ok": False, "tool": rec["tool"], "code": err.get("code", "error"), "message": err.get("message", ""),
                "sent": rec.get("sent", True)}

    # ------------------------------------------------------------------ auth
    def _cookie(self, resp: Response, sid: str) -> Response:
        resp.set_cookie(COOKIE, sid, httponly=True, samesite="lax", secure=self.secure, path="/")
        return resp

    def _to_login(self, c: Ctx) -> Response:
        if c.mode == "path":
            nxt = c.req.url.path + (("?" + c.req.url.query) if c.req.url.query else "")
        else:
            nxt = str(c.req.url)
        if c.req.method != "GET":
            nxt = c.url(c.host, "/") if c.mode == "path" else c.base(c.host) + "/"
        return _redirect(c.url("id", "/login", next=nxt))

    def _safe_next(self, c: Ctx, nxt: str | None) -> str:
        if not nxt:
            return c.url("id", "/")
        if nxt.startswith("/") and not nxt.startswith("//") and "\\" not in nxt:
            return nxt
        u = urlsplit(nxt)
        if u.scheme in ("http", "https") and (u.hostname or "").endswith(".crucible"):
            return nxt
        return c.url("id", "/")

    def _continue(self, c: Ctx, session: Session, nxt: str) -> Response:
        nxt = self._safe_next(c, nxt)
        u = urlsplit(nxt)
        if u.scheme and u.hostname and u.hostname != "id.crucible":
            host = u.hostname[: -len(".crucible")]
            local = (u.path or "/") + (("?" + u.query) if u.query else "")
            t = self.sessions.ticket(session.sid)
            return _redirect(f"{u.scheme}://{u.netloc}/_sso?" + urlencode({"ticket": t, "next": local}))
        return _redirect(nxt)

    def _login_page(self, c: Ctx, error: str = "", handle: str = "", status: int = 200) -> Response:
        nxt = c.qp.get("next", "")
        if c.session is not None and not error:
            return self._continue(c, c.session, nxt)
        c.title = "Sign in"
        return self.render(c, "login.html", status=status, action=c.url("id", "/login"), next=nxt,
                           login_error=error, handle=handle)

    def _login(self, c: Ctx, form: dict[str, list[str]]) -> Response:
        handle = (form.get("handle") or [""])[-1].strip()
        if handle.startswith("p:"):
            handle = handle[2:]
        pw = (form.get("password") or [""])[-1]
        c.qp["next"] = (form.get("next") or [""])[-1]
        pid = f"p:{handle}"
        w = c.w
        p = w.principal(pid)
        ok = False
        if p is not None and p.get("kind") == "person" and p.get("status") == "active" and pw:
            ident = w.modules.get("identity")
            if ident is not None and ident.check_password(w, pid, pw):
                ok = True
            elif self.b.token_principal(pw) == pid:
                ok = True
        if not ok:
            c.session = None
            c.p = None
            return self._login_page(c, "Unknown handle or wrong password.", handle, status=401)
        s = self.sessions.create(pid, max(0, w.seq - 200))
        c.session, c.p = s, pid
        return self._cookie(self._continue(c, s, c.qp["next"]), s.sid)

    def _sso(self, c: Ctx) -> Response:
        s = self.sessions.redeem(c.qp.get("ticket", ""))
        nxt = c.qp.get("next", "/")
        if not nxt.startswith("/") or nxt.startswith("//"):
            nxt = "/"
        if s is None:
            return self._to_login(c)
        return self._cookie(_redirect(nxt), s.sid)

    def _logout(self, c: Ctx, form: dict[str, list[str]]) -> Response:
        if c.session is not None and (form.get("_csrf") or [""])[-1] == c.session.csrf:
            self.sessions.drop(c.session.sid)
        resp = _redirect(c.url("id", "/login"))
        resp.delete_cookie(COOKIE, path="/")
        return resp

    # ------------------------------------------------------------------ actions (recorded commands)
    def _act(self, c: Ctx, form: dict[str, list[str]]) -> Response:
        s = c.session
        g = lambda k: (form.get(k) or [""])[-1]  # noqa: E731
        back = g("_back")
        if not back.startswith("/") or back.startswith("//"):
            back = c.url(c.host, "/")
        if g("_csrf") != s.csrf:
            return self.render(c, status=403, title="Forbidden",
                               errors=["This form has expired (bad CSRF token). Reload the page and try again."])
        tool = g("_tool")
        thost = tool.split(".", 1)[0]
        rec: dict[str, Any] = {"tool": tool, "form_id": g("_form"), "form": form, "sent": False}
        if thost in HIDDEN_HOSTS or (thost != c.host and thost != "world"):
            rec.update(ok=False, error={"code": "forbidden",
                                        "message": f"{tool} cannot be submitted from {c.host}.crucible"})
            return _redirect(_with_r(back, s.remember(rec)))
        try:
            ep = c.w.resolve(tool, c.p)
            params = ep.params or {}
        except CrucibleError:
            params = {"properties": {}, "additionalProperties": True}
        try:
            args = form_to_args(params, {k: v for k, v in form.items() if not k.startswith("_") or k == "_args"})
        except FormError as e:
            rec.update(ok=False, error={"code": "invalid_input", "message": str(e)})
            return _redirect(_with_r(back, s.remember(rec)))
        res = c.w.execute(c.p, {"tool": tool, "args": args})
        self.b.after_write(c.w)
        d = res.to_dict()
        rec.update(ok=d["ok"], result=d.get("result"), error=d.get("error"), args=args, sent=True)
        if d["ok"] and tool == "mail.read" and isinstance(d.get("result"), dict):
            s.mail[str(d["result"].get("msg"))] = d["result"]
            while len(s.mail) > 50:
                s.mail.popitem(last=False)
        rid = s.remember(rec)
        target = back
        if d["ok"] and g("_next"):
            nxt = _fill(g("_next"), d.get("result"))
            if nxt and nxt.startswith("/") and not nxt.startswith("//"):
                target = c.url(c.host, nxt) if c.mode == "path" and not nxt.startswith("/_h/") else nxt
        return _redirect(_with_r(target, rid))


SYSTEM_VIEW = "sys"
_SEC_HEADERS = {"Content-Security-Policy": "default-src 'self'; script-src 'none'; object-src 'none'; "
                                          "base-uri 'none'; frame-ancestors 'none'; img-src 'self' data:; "
                                          "style-src 'self'",
                "X-Content-Type-Options": "nosniff", "Referrer-Policy": "same-origin", "Cache-Control": "no-store"}


def _redirect(url: str) -> RedirectResponse:
    return RedirectResponse(url, status_code=303, headers={"Cache-Control": "no-store"})


def _with_r(url: str, rid: str) -> str:
    base, _, frag = url.partition("#")
    path, _, qs = base.partition("?")
    q = [(k, v) for k, v in parse_qsl(qs, keep_blank_values=True) if k != "r"] + [("r", rid)]
    return f"{path}?{urlencode(q)}" + (f"#{frag}" if frag else "")


def _fill(template: str, result: Any) -> str | None:
    if "{" not in template:
        return template
    if not isinstance(result, dict):
        return None
    missing = []

    def rep(m):
        v = result.get(m.group(1))
        if v is None or isinstance(v, (dict, list)):
            missing.append(m.group(1))
            return ""
        return quote(str(v), safe="")
    out = re.sub(r"\{(\w+)\}", rep, template)
    return None if missing else out


# ====================================================================== pages
def _register_pages(app: WebApp) -> None:  # noqa: C901 - one place that lists every page
    r = app.route

    # ------------------------------------------------------------------ id.crucible: home portal
    @r("id", "/")
    def home(c: Ctx):
        w = c.w
        c.title = "Home"
        c.note(f"Signed in as {c.p}. World time {fmt_time(w.now)}.", id="welcome")
        c.links("Services", [Link(f"{h}.crucible", c.url(h, "/"), id=f"svc-{h}", desc=d) for h, _, d in SERVICES
                             if h == "id" or any(e.host == h for e in w.endpoints.values())])
        known = [rec["value"] for kid in w.state.find_ids("knowledge", "principal", c.p)
                 for rec in [w.state.tables["knowledge"][kid]] if rec["kind"] == "host"
                 and rec["value"] not in SERVICE_HOSTS]
        c.links("Services you know", [Link(f"{h}.crucible", c.url(h, "/")) for h in sorted(set(known))],
                empty="None yet: search results, posts, documents and mail that mention <host>.crucible "
                      "introduce services.")
        c.add("form", id="go", form=c.get_form("go", c.url("id", "/go"),
                                               [Field("host", "Service host name", "text",
                                                      help="e.g. shop for shop.crucible")],
                                               "Open service", "Open a service by name"))
        c.table("Your accounts", c.accounts(), [("account", "Account"), ("name", "Name"), ("owner", "Owner"),
                                                ("balance_fmt", "Balance"), ("status", "Status")],
                empty="No bank accounts.")
        c.table("Your organisations", c.orgs(), [("org", "Organisation"), ("role", "Role"), ("title", "Title")],
                empty="You are not a member of any organisation.")
        inv = c.tq("id.invites") or []
        c.table("Invitations", inv, [("id", "Invite", lambda r_: r_.get("id") or r_.get("invite")),
                                     ("org", "Organisation"), ("role", "Role"),
                                     ("respond", "Respond", lambda r_: c.form(
                                         "id.respond_invite", hide={"invite": r_.get("id") or r_.get("invite")},
                                         key=r_.get("id") or r_.get("invite"), inline=True,
                                         buttons=[("accept", "true", "Accept", f"Accept invite to {r_.get('org')}"),
                                                  ("accept", "false", "Decline",
                                                   f"Decline invite to {r_.get('org')}")]))],
                empty="No pending invitations.")
        c.forms("Organisations", [c.form("id.create_org", title="Create an organisation",
                                         next="/profile?p={org}")])
        c.forms("Profile", [c.form("id.update_profile", title="Update your profile")])
        c.links("More", [Link("Directory", c.url("id", "/directory"), id="link-directory"),
                         Link("Grants (delegated capabilities)", c.url("id", "/grants"), id="link-grants"),
                         Link("All notifications", c.url("id", "/notifications"), id="link-notifications"),
                         Link("All identity endpoints", c.url("id", "/_api"), id="link-api")])

    @r("id", "/go")
    def go(c: Ctx):
        h = c.qp.get("host", "").strip().lower()
        if h.endswith(".crucible"):
            h = h[:-9]
        if not HOST_RE.fullmatch(h or ""):
            raise PageError(f"{h!r} is not a valid host name")
        return _redirect(c.url(h, "/"))

    @r("id", "/profile")
    def profile(c: Ctx):
        pid = c.qp.get("p") or c.p
        prof = c.q("id.profile", principal=pid)
        c.title = f"Profile of {prof.get('id', pid)}"
        c.kv("Profile", prof, ["id", "handle", "name", "kind", "status", "since", "bio", "reputation", "orgs"])
        if prof.get("id", "").startswith("o:"):
            org = prof["id"]
            c.table("Members", prof.get("members"), [("person", "Person"), ("role", "Role"), ("title", "Title")])
            my_role = next((o["role"] for o in c.orgs() if o["org"] == org), None)
            c.links("Organisation services", [
                Link("Company registry entry", c.url("biz", "/company", org=org)),
                Link("Bank accounts", c.url("bank", "/")),
                Link("Code repositories", c.url("code", "/", owner=org)),
                Link("Issue projects", c.url("issues", "/", owner=org)),
                Link("Support queue", c.url("support", "/", owner=org))])
            if my_role in ("owner", "admin"):
                c.forms("Administer", [
                    c.form("id.invite", title="Invite a person", hide={"org": org}),
                    c.form("id.set_role", title="Change a member's role", hide={"org": org}),
                    c.form("id.remove", title="Remove a member", hide={"org": org}),
                    c.form("id.grant", title="Delegate a capability over this organisation", hide={"owner": org}),
                    c.form("id.update_profile", title="Organisation bio", hide={"org": org})])
            if my_role:
                c.forms("Membership", [c.form("id.leave", title="Leave this organisation", hide={"org": org},
                                              submit="Leave organisation")])
        else:
            c.links("Activity", [Link("Send a message", c.url("mail", "/compose", to=prof.get("id"))),
                                 Link("Repositories", c.url("code", "/", owner=prof.get("id")))])

    @r("id", "/directory")
    def directory(c: Ctx):
        kind = c.qp.get("kind", "orgs")
        if kind not in ("orgs", "persons"):
            kind = "orgs"
        c.title = "Directory"
        c.tabs("Directory kind", [("Organisations", c.url("id", "/directory", kind="orgs"), kind == "orgs"),
                                  ("People", c.url("id", "/directory", kind="persons"), kind == "persons")])
        rows = c.q("id.directory", kind=kind, limit=200)
        c.table("Organisations" if kind == "orgs" else "People", rows,
                [("id", "Id", lambda r_: c.who(r_.get("id"))), ("name", "Name"), ("kind", "Kind")])

    @r("id", "/grants")
    def grants(c: Ctx):
        c.title = "Grants"
        me = c.q("id.me")
        c.table("Grants you hold", me.get("grants_held"),
                [("id", "Grant"), ("owner", "Owner"), ("cap", "Capability"), ("scope", "Scope"),
                 ("limit", "Limit (cents)"), ("used", "Used (cents)"), ("expires", "Expires")])
        issued = me.get("grants_issued")
        if issued is not None:
            c.table("Grants you issued", issued,
                    [("id", "Grant"), ("grantee", "Grantee"), ("cap", "Capability"), ("scope", "Scope"),
                     ("revoke", "Revoke", lambda r_: c.form("id.revoke", hide={"grant": r_.get("id")},
                                                           key=r_.get("id"), inline=True, submit="Revoke",
                                                           submit_aria=f"Revoke grant {r_.get('id')}"))])
        c.forms("Delegate", [c.form("id.grant", title="Grant a capability"),
                             c.form("id.revoke", title="Revoke a grant you issued")])

    @r("id", "/notifications")
    def notifications(c: Ctx):
        w = c.w
        c.title = "Notifications"
        show_all = c.qp.get("all") == "1"
        c.tabs("Which events", [("Addressed to you", c.url("id", "/notifications"), not show_all),
                                ("Everything you can see", c.url("id", "/notifications", all="1"), show_all)])
        evs, dropped = w.visible_events(c.p, max(0, w.seq - 20000), limit=200 if show_all else 20000)
        if not show_all:
            evs = [e for e in evs if isinstance(e.vis, tuple) and e.type not in ("ctr", "idem")][-200:]
        rows = [{"seq": e.seq, "t": e.t, "type": e.type, "actor": e.actor, "payload": e.payload} for e in evs[::-1]]
        c.table("Events (newest first)", rows, [("seq", "#"), ("t", "Time"), ("type", "Type"),
                                                ("actor", "By"), ("payload", "Details",
                                                                  lambda r_: compact_json(r_["payload"], 400))])

    # ------------------------------------------------------------------ bank.crucible
    @r("bank", "/")
    def bank_home(c: Ctx):
        c.title = "Bank"
        c.tabs("Bank sections", [("Accounts", c.url("bank", "/"), True), ("Invoices", c.url("bank", "/invoices"), False),
                                 ("Direct debits", c.url("bank", "/mandates"), False)])
        c.table("Accounts", c.accounts(), [("account", "Account"), ("name", "Name"), ("owner", "Owner"),
                                           ("balance_fmt", "Balance"), ("overdraft", "Overdraft"),
                                           ("status", "Status")], empty="You have no accounts.")
        c.forms("Move money", [c.form("bank.transfer", title="Transfer",
                                      labels={"to": "To (account id, principal or handle)",
                                              "amount": "Amount (cents)", "from_account": "From account"})])
        c.forms("New account", [c.form("bank.open_account", title="Open an account",
                                       widgets={"owner": {"kind": "select",
                                                          "options": c.org_options("Yourself")}},
                                       labels={"owner": "Owner"})])

    @r("bank", "/account")
    def bank_account(c: Ctx):
        acct = c.qp.get("id", "")
        bal = c.q("bank.balance", account=acct)
        c.title = f"Account {acct}"
        c.kv("Balance", bal, ["account", "balance_fmt", "status"], labels={"balance_fmt": "balance"})
        try:
            limit = max(1, min(int(c.qp.get("limit", "50")), 500))
        except ValueError:
            limit = 50
        st = c.q("bank.statement", account=acct, limit=limit)
        c.table("Statement (newest first)", st, [("txn", "Transaction"), ("t", "Time"), ("amount", "Amount"),
                                                 ("kind", "Kind"), ("memo", "Memo"),
                                                 ("counterparties", "Counterparties",
                                                  lambda r_: Many([c.acct(x) for x in r_.get("counterparties", [])])),
                                                 ("ref", "Ref")], empty="No transactions.")
        c.forms("Pay from this account", [c.form("bank.transfer", title="Transfer", values={"from_account": acct},
                                                 labels={"to": "To (account id, principal or handle)",
                                                         "amount": "Amount (cents)"})])

    @r("bank", "/invoices")
    def bank_invoices(c: Ctx):
        c.title = "Invoices"
        c.tabs("Bank sections", [("Accounts", c.url("bank", "/"), False), ("Invoices", c.url("bank", "/invoices"), True),
                                 ("Direct debits", c.url("bank", "/mandates"), False)])
        payer = c.tq("bank.invoices", role="payer") or []
        c.table("Invoices to pay", payer,
                [("id", "Invoice"), ("payee_owner", "From"), ("amount", "Amount"), ("memo", "Memo"), ("due", "Due"),
                 ("status", "Status"),
                 ("pay", "Pay", lambda r_: c.form("bank.pay_invoice", hide={"invoice": r_["id"]}, key=r_["id"],
                                                  inline=True, fields=["from_account"], submit="Pay",
                                                  submit_aria=f"Pay invoice {r_['id']}")
                  if r_.get("status") == "open" else None)], empty="No invoices addressed to you.")
        payee = c.tq("bank.invoices", role="payee") or []
        c.table("Invoices you sent", payee, [("id", "Invoice"), ("payer", "Payer"), ("payee_acct", "To account"),
                                             ("amount", "Amount"), ("memo", "Memo"), ("due", "Due"),
                                             ("status", "Status")], empty="You have not sent invoices.")
        c.forms("Request money", [c.form("bank.invoice", title="Send an invoice",
                                         labels={"amount": "Amount (cents)", "to_account": "Pay into account"})])

    @r("bank", "/mandates")
    def bank_mandates(c: Ctx):
        c.title = "Direct debits"
        c.tabs("Bank sections", [("Accounts", c.url("bank", "/"), False), ("Invoices", c.url("bank", "/invoices"), False),
                                 ("Direct debits", c.url("bank", "/mandates"), True)])
        c.forms("Mandates", [c.form("bank.authorize_debit", title="Authorise a direct debit"),
                             c.form("bank.cancel_mandate", title="Cancel a mandate")])

    # ------------------------------------------------------------------ mail.crucible
    @r("mail", "/")
    def mail_home(c: Ctx):
        folder = c.qp.get("folder", "inbox")
        if folder not in ("inbox", "sent"):
            folder = "inbox"
        mailbox = c.qp.get("mailbox") or None
        c.title = "Sent mail" if folder == "sent" else "Inbox"
        c.tabs("Folders", [("Inbox", c.url("mail", "/"), folder == "inbox"),
                           ("Sent", c.url("mail", "/", folder="sent"), folder == "sent"),
                           ("Compose", c.url("mail", "/compose"), False)])
        args = {"folder": folder, "limit": 100}
        if mailbox:
            args["mailbox"] = mailbox
        if c.qp.get("unread") == "1":
            args["unread_only"] = True
        msgs = c.q("mail.inbox", **args)

        def open_btn(m):
            hide = {"msg": m["msg"]}
            if m.get("mailbox") and m["mailbox"] != c.p:
                hide["mailbox"] = m["mailbox"]
            return c.form("mail.read", hide=hide, key=m["msg"], inline=True, submit="Open",
                          submit_aria=f"Open message {m['msg']}: {m.get('subject', '')}",
                          next="/msg?id={msg}" + (f"&mailbox={quote(m['mailbox'])}" if "mailbox" in hide else ""))
        c.table("Messages", msgs, [("msg", "Message"), ("from", "From"), ("to", "To"), ("subject", "Subject"),
                                   ("t", "Time"), ("read", "Read"), ("open", "Open", open_btn)],
                empty="No messages.")
        c.add("form", id="mail-search", form=c.get_form(
            "mail-search", c.url("mail", "/search"), [Field("q", "Search your mail", "text", required=True)],
            "Search mail", "Search your mail"))
        org_boxes = [o["org"] for o in c.orgs() if o["role"] in ("owner", "admin")]
        if org_boxes:
            c.links("Organisation mailboxes", [Link(o, c.url("mail", "/", mailbox=o)) for o in org_boxes])

    @r("mail", "/search")
    def mail_search(c: Ctx):
        q = c.qp.get("q", "")
        c.title = f"Mail search: {q}"
        rows = c.q("mail.search", q=q, limit=50) if q else []
        c.table("Matching messages", rows, [("msg", "Message", lambda r_: Link(r_["msg"], c.url(
            "mail", "/msg", id=r_["msg"], mailbox=r_.get("mailbox") if r_.get("mailbox") != c.p else None))),
            ("from", "From"), ("subject", "Subject"), ("t", "Time"), ("snippet", "Snippet")], empty="No matches.")

    @r("mail", "/msg")
    def mail_msg(c: Ctx):
        mid = c.qp.get("id", "")
        mailbox = c.qp.get("mailbox") or None
        m = c.session.mail.get(mid)
        c.title = f"Message {mid}"
        if m is None:
            c.note("Opening a message marks it read (a recorded mail.read command).")
            c.forms("Open", [c.form("mail.read", hide={"msg": mid, **({"mailbox": mailbox} if mailbox else {})},
                                    submit="Open message", next=c.path + "?" + urlencode(
                                        {"id": mid, **({"mailbox": mailbox} if mailbox else {})}))])
            return
        c.title = m.get("subject") or c.title
        c.add("article", m.get("subject") or "(no subject)", id="message",
              meta=[Many(["From ", c.who(m.get("from"))], ""), Many(["To "] + [c.who(x) for x in m.get("to", [])], " "),
                    fmt_time(m.get("t", 0))], body=c.rich(m.get("body")), forms=[], children=[])
        reply_to = m.get("from")
        subj = m.get("subject") or ""
        c.forms("Reply", [c.form("mail.send", title="Reply", values={
            "to": reply_to, "subject": subj if subj.lower().startswith("re:") else f"Re: {subj}"},
            hide={"in_reply_to": mid}, next="/?folder=sent")])
        c.forms("Delete", [c.form("mail.delete", hide={"msg": mid, **({"mailbox": mailbox} if mailbox else {})},
                                  fields=["folder"], submit="Delete message", next="/")])

    @r("mail", "/compose")
    def mail_compose(c: Ctx):
        c.title = "Compose"
        c.tabs("Folders", [("Inbox", c.url("mail", "/"), False), ("Sent", c.url("mail", "/", folder="sent"), False),
                           ("Compose", c.url("mail", "/compose"), True)])
        c.forms("New message", [c.form("mail.send", title="New message",
                                       values={k: c.qp[k] for k in ("to", "subject") if c.qp.get(k)},
                                       labels={"to": "To", "from_org": "Send as organisation"},
                                       next="/?folder=sent")])

    # ------------------------------------------------------------------ forum.crucible
    def board_options(c: Ctx) -> list:
        return [(b["board"], f"{b['title']} ({b['board']})") for b in (c.tq("forum.boards") or [])]

    def post_cols(c: Ctx) -> list:
        return [("title", "Title", lambda r_: Link(r_.get("title") or r_["post"], c.url("forum", "/post", id=r_["post"]))),
                ("board", "Board", lambda r_: Link(r_.get("board"), c.url("forum", "/b", board=r_.get("board")))),
                ("author", "Author"), ("score", "Score"), ("replies", "Replies"), ("t", "Posted"),
                ("snippet", "Snippet", lambda r_: c.rich(r_.get("snippet")))]

    @r("forum", "/")
    def forum_home(c: Ctx):
        c.title = "Forum"
        boards = c.q("forum.boards")
        c.table("Boards", boards, [("board", "Board", lambda r_: Link(r_.get("title") or r_["board"], c.url(
            "forum", "/b", board=r_["board"]), id=f"board-{r_['board']}")), ("description", "Description"),
            ("posts", "Posts")])
        c.table("Your feed", c.tq("forum.feed", limit=20) or [], post_cols(c), empty="Your feed is empty.")
        c.forms("Write", [c.form("forum.post", title="New post", widgets={"board": {"kind": "select",
                                                                                     "options": board_options(c)}},
                                 next="/post?id={post}")])

    @r("forum", "/b")
    def forum_board(c: Ctx):
        board = c.qp.get("board", "")
        sort = c.qp.get("sort", "new")
        if sort not in ("new", "top", "active"):
            sort = "new"
        try:
            offset = max(0, int(c.qp.get("offset", "0")))
        except ValueError:
            offset = 0
        posts = c.q("forum.list", board=board, sort=sort, limit=25, offset=offset)
        c.title = f"Board: {board}"
        c.tabs("Sort order", [(s.capitalize(), c.url("forum", "/b", board=board, sort=s), s == sort)
                              for s in ("new", "top", "active")])
        c.table("Posts", posts, post_cols(c), empty="No posts on this board yet.")
        more = []
        if offset:
            more.append(Link("Previous page", c.url("forum", "/b", board=board, sort=sort, offset=max(0, offset - 25))))
        if len(posts) == 25:
            more.append(Link("Next page", c.url("forum", "/b", board=board, sort=sort, offset=offset + 25)))
        if more:
            c.links("Pages", more)
        c.forms("Write", [c.form("forum.post", title=f"New post in {board}", hide={"board": board},
                                 next="/post?id={post}")])
        c.forms("Follow", [c.form("forum.follow", hide={"target": f"board:{board}"}, inline=True,
                                  submit="Follow board", key="board"),
                           c.form("forum.unfollow", hide={"target": f"board:{board}"}, inline=True,
                                  submit="Unfollow board", key="board")])

    @r("forum", "/post")
    def forum_post(c: Ctx):
        pid = c.qp.get("id", "")
        p = c.q("forum.read", post=pid, replies=100)
        c.title = p.get("title") or pid
        votes = c.form("forum.vote", hide={"post": pid}, inline=True,
                       buttons=[("value", "1", "Upvote", "Upvote this post"),
                                ("value", "-1", "Downvote", "Downvote this post")])
        children = [{"id": f"reply-{x.get('reply')}", "label": f"Reply by {x.get('author')}",
                     "meta": [c.who(x.get("author")), fmt_time(x.get("t", 0))], "body": c.rich(x.get("body"))}
                    for x in p.get("replies", [])]
        c.add("article", p.get("title") or pid, id="post",
              meta=[Many(["by ", c.who(p.get("author"))], ""),
                    Many(["in ", Link(p.get("board"), c.url("forum", "/b", board=p.get("board")))], ""),
                    fmt_time(p.get("t", 0)), f"score {p.get('score', 0)}", f"your vote {p.get('your_vote', 0)}",
                    f"{p.get('n_replies', len(children))} replies"]
              + ([Many(["tags: ", ", ".join(p.get("tags") or [])], "")] if p.get("tags") else []),
              body=c.rich(p.get("body")), forms=[votes] if votes else [], children=children)
        c.forms("Reply", [c.form("forum.reply", title="Reply", hide={"post": pid})])
        extra = [c.form("forum.follow", hide={"target": p.get("author")}, inline=True,
                        submit=f"Follow {p.get('author')}", key="author")]
        if p.get("author") == c.p or p.get("author") in {o["org"] for o in c.orgs()}:
            extra.append(c.form("forum.delete", hide={"post": pid}, inline=True, submit="Delete post", next="/"))
        c.forms("More", extra)

    # ------------------------------------------------------------------ search.crucible
    def result_link(c: Ctx, r_: dict) -> Any:
        kind, ref, title = r_.get("kind"), r_.get("ref"), r_.get("title") or r_.get("id")
        if r_.get("sponsored"):
            hosts = r_.get("hosts") or []
            return Link(title, c.url(hosts[0], "/")) if hosts else title
        target = {"post": ("forum", "/post", {"id": ref}), "doc": ("docs", "/d", {"id": ref}),
                  "listing": ("market", "/listing", {"id": ref}), "company": ("biz", "/company", {"org": ref}),
                  "job": ("jobs", "/posting", {"id": ref}), "repo": ("code", "/repo", {"id": ref})}.get(kind)
        if kind == "service" and (r_.get("hosts") or ref):
            h = (r_.get("hosts") or [ref])[0]
            return Link(title, c.url(h, "/"))
        if target and ref:
            return Link(title, c.url(target[0], target[1], **target[2]))
        return Link(title, c.url("search", "/doc", id=r_.get("id")))

    @r("search", "/")
    def search_home(c: Ctx):
        c.title = "Search"
        rec = c.result_rec
        last_q = (rec.get("args") or {}).get("q", "") if rec and rec.get("tool") == "search.query" else ""
        c.forms("Search the web", [c.form("search.query", title="", fields=["q", "kind"], values={"q": last_q},
                                          labels={"q": "Search terms", "kind": "Only this kind (optional)"},
                                          widgets={"kind": {"kind": "select", "options": [("", "(any kind)")] + [
                                              (k, k) for k in ("post", "doc", "listing", "company", "job",
                                                               "service", "repo")]}},
                                          submit="Search", doc=False)],
                note="Searching is a recorded action: sponsored results are billed to advertisers and you "
                     "learn the services that results mention.", id="search-form")
        if rec and rec.get("tool") == "search.query" and rec.get("ok"):
            res = rec.get("result") or {}
            c.table(f"Results for “{last_q}” ({res.get('matched', 0)} matched)", res.get("results"),
                    [("title", "Result", lambda r_: result_link(c, r_)), ("kind", "Kind"),
                     ("sponsored", "Sponsored"), ("snippet", "Snippet", lambda r_: c.rich(r_.get("snippet"))),
                     ("owner", "Owner"), ("hosts", "Services"), ("t", "Time"),
                     ("id", "Indexed text", lambda r_: Link("indexed text", c.url("search", "/doc", id=r_["id"]),
                                                            label=f"Indexed text of {r_['id']}")
                      if not r_.get("sponsored") else None)],
                    empty="No results.", id="results")

    @r("search", "/doc")
    def search_doc(c: Ctx):
        d = c.q("search.get", id=c.qp.get("id", ""))
        c.title = d.get("title") or d.get("id")
        c.kv("Indexed document", d, ["id", "kind", "owner", "ref", "hosts", "t"])
        c.add("text", "Text", id="text", body=c.rich(d.get("text")))

    # ------------------------------------------------------------------ docs.crucible
    @r("docs", "/")
    def docs_home(c: Ctx):
        scope = c.qp.get("scope", "mine")
        if scope not in ("mine", "org", "shared", "public"):
            scope = "mine"
        c.title = "Documents"
        c.tabs("Document scope", [(s.capitalize(), c.url("docs", "/", scope=s), s == scope)
                                  for s in ("mine", "org", "shared", "public")])
        args = {"scope": scope, "limit": 100}
        if scope == "org" and c.qp.get("org"):
            args["org"] = c.qp["org"]
        rows = c.tq("docs.list", **args) or []
        c.table("Documents", rows, [("doc", "Document", lambda r_: Link(r_.get("title") or r_["doc"], c.url(
            "docs", "/d", id=r_["doc"]))), ("doc", "Id"), ("owner", "Owner"), ("visibility", "Visibility"),
            ("version", "Version"), ("updated", "Updated")], empty="No documents.")
        c.forms("Write", [c.form("docs.create", title="New document", next="/d?id={doc}")])

    @r("docs", "/d")
    def docs_view(c: Ctx):
        did = c.qp.get("id", "")
        args = {"doc": did}
        if c.qp.get("version", "").isdigit():
            args["version"] = int(c.qp["version"])
        d = c.q("docs.get", **args)
        c.title = d.get("title") or did
        c.kv("Document", d, ["doc", "owner", "visibility", "version", "latest_version", "access", "created",
                             "updated", "hosts"])
        c.add("text", "Content", id="content", body=c.rich(d.get("body")))
        hist = c.tq("docs.history", doc=did, limit=50) or []
        c.table("History", hist, [("version", "Version", lambda r_: Link(f"v{r_['version']}", c.url(
            "docs", "/d", id=did, version=r_["version"]))), ("author", "Author"), ("t", "Time"), ("title", "Title")])
        if d.get("access") in ("write", "admin"):
            c.forms("Edit", [c.form("docs.edit", title="Edit document", hide={"doc": did,
                                                                              "base_version": d.get("latest_version")},
                                    values={"title": d.get("title"), "body": d.get("body")}, next="/d?id={doc}")])
        if d.get("access") == "admin":
            c.table("Shared with", d.get("shares"), [("principal", "Principal"), ("perm", "Permission")],
                    empty="Not shared with anyone.")
            c.forms("Sharing", [c.form("docs.share", title="Share", hide={"doc": did}),
                                c.form("docs.set_visibility", title="Visibility", hide={"doc": did},
                                       values={"visibility": d.get("visibility")})])

    # ------------------------------------------------------------------ code.crucible
    @r("code", "/")
    def code_home(c: Ctx):
        owner = c.qp.get("owner") or None
        c.title = f"Repositories of {owner}" if owner else "Your repositories"
        repos = c.q("code.list", **({"owner": owner} if owner else {}))
        c.table("Repositories", repos, [("repo", "Repository", lambda r_: Link(r_["repo"], c.url(
            "code", "/repo", id=r_["repo"]))), ("visibility", "Visibility"), ("license", "Licence"),
            ("description", "Description"), ("head", "Head")], empty="No repositories.")
        c.add("form", id="code-owner", form=c.get_form("code-owner", c.url("code", "/"), [
            Field("owner", "Owner (p:<handle> or o:<handle>)", "text", value=owner or "")], "Show", "Browse an owner"))
        c.forms("New", [c.form("code.create_repo", title="Create a repository", next="/repo?id={repo}"),
                        c.form("code.fork", title="Fork a repository", next="/repo?id={repo}")])

    @r("code", "/repo")
    def code_repo(c: Ctx):
        rid = c.qp.get("id", "")
        ref = c.qp.get("ref") or None
        ls = c.q("code.ls", repo=rid, **({"ref": ref} if ref else {}))
        c.title = f"Repository {rid}"
        c.kv("Repository", ls, ["repo", "owner", "visibility", "license", "description", "head"])
        c.table("Files", ls.get("files"), [("path", "Path", lambda r_: Link(r_["path"], c.url(
            "code", "/file", repo=rid, path=r_["path"], ref=ref))), ("bytes", "Bytes")], empty="No files yet.")
        log = c.tq("code.log", repo=rid, limit=30) or []
        c.table("Commits", log, [("commit", "Commit", lambda r_: Link(r_["commit"], c.url(
            "code", "/repo", id=rid, ref=r_["commit"]))), ("author", "Author"), ("message", "Message"),
            ("t", "Time"), ("files", "Files")], empty="No commits.")
        c.forms("Change", [c.form("code.commit", title="Commit a file", hide={"repo": rid},
                                  widgets={"files": "pathcontent"}, labels={"files": "File"},
                                  helps={"files": "Writes one file; to delete a file, commit it via the API."},
                                  next="/repo?id=" + quote(rid, safe=""))])
        c.forms("Execute", [c.form("code.run", title="Run a function", hide={"repo": rid},
                                   labels={"target": "Target (file.py:function)", "input": "Input (JSON)"}),
                            c.form("code.test", title="Run the tests", hide={"repo": rid})])
        c.forms("Settings", [c.form("code.set_visibility", title="Visibility and licence", hide={"repo": rid},
                                    values={"visibility": ls.get("visibility")})])
        c.links("Related", [Link("Deploy this repository", c.url("cloud", "/", repo=rid)),
                            Link("Issue projects", c.url("issues", "/", owner=ls.get("owner")))])

    @r("code", "/file")
    def code_file(c: Ctx):
        rid, path = c.qp.get("repo", ""), c.qp.get("path", "")
        ref = c.qp.get("ref") or None
        f = c.q("code.read", repo=rid, path=path, **({"ref": ref} if ref else {}))
        c.title = f"{rid}: {path}"
        c.links("Repository", [Link(rid, c.url("code", "/repo", id=rid))])
        c.add("text", path, id="file-content", body=Pre(f.get("content", ""), id="file-text"))
        c.forms("Edit", [c.form("code.commit", title="Edit and commit", hide={"repo": rid},
                                widgets={"files": "pathcontent"}, labels={"files": "File"},
                                values={"files": (path, f.get("content", ""))},
                                next="/file?" + urlencode({"repo": rid, "path": path}))])

    # ------------------------------------------------------------------ issues.crucible
    @r("issues", "/")
    def issues_home(c: Ctx):
        owner = c.qp.get("owner") or None
        c.title = "Issue trackers"
        projs = c.q("issues.projects", **({"owner": owner} if owner else {}))
        c.table("Projects" + (f" of {owner}" if owner else ""), projs, [
            ("project", "Project", lambda r_: Link(r_.get("name") or r_["project"], c.url(
                "issues", "/project", id=r_["project"]))), ("project", "Id"), ("owner", "Owner"), ("repo", "Repo"),
            ("public", "Public"), ("open_issues", "Open issues")], empty="No projects.")
        c.table("Assigned to you", c.tq("issues.assigned", limit=50) or [], [
            ("issue", "Issue", lambda r_: Link(r_.get("title") or r_["issue"], c.url("issues", "/issue",
                                                                                     id=r_["issue"]))),
            ("project", "Project"), ("status", "Status")], empty="Nothing assigned to you.")
        c.forms("New", [c.form("issues.create_project", title="Create a project", next="/project?id={project}")])

    @r("issues", "/project")
    def issues_project(c: Ctx):
        pid = c.qp.get("id", "")
        status = c.qp.get("status", "open")
        if status not in ("open", "in_progress", "closed", "any"):
            status = "open"
        rows = c.q("issues.list", project=pid, status=status, limit=100)
        c.title = f"Project {pid}"
        c.tabs("Status", [(s.replace("_", " "), c.url("issues", "/project", id=pid, status=s), s == status)
                          for s in ("open", "in_progress", "closed", "any")])
        c.table("Issues", rows, [("issue", "Issue", lambda r_: Link(r_.get("title") or r_["issue"], c.url(
            "issues", "/issue", id=r_["issue"]))), ("status", "Status"), ("labels", "Labels"),
            ("assignee", "Assignee"), ("creator", "Creator"), ("comments", "Comments"), ("updated", "Updated")],
            empty="No issues.")
        c.forms("New", [c.form("issues.create", title="File an issue", hide={"project": pid},
                               next="/issue?id={issue}")])

    @r("issues", "/issue")
    def issues_issue(c: Ctx):
        iid = c.qp.get("id", "")
        it = c.q("issues.get", issue=iid, comments=200)
        c.title = it.get("title") or iid
        c.kv("Issue", it, ["issue", "project", "status", "labels", "assignee", "creator", "t", "updated"],
             extra=[("project page", Link(it.get("project"), c.url("issues", "/project", id=it.get("project"))))])
        children = [{"id": f"comment-{x.get('comment')}", "label": f"Comment by {x.get('author')}",
                     "meta": [c.who(x.get("author")), fmt_time(x.get("t", 0))], "body": c.rich(x.get("body"))}
                    for x in it.get("comments", [])]
        c.add("article", "Description", id="issue-body", meta=[], body=c.rich(it.get("body")), forms=[],
              children=children)
        c.forms("Discuss", [c.form("issues.comment", title="Comment", hide={"issue": iid})])
        c.forms("Triage", [c.form("issues.update", title="Update", hide={"issue": iid},
                                  values={"status": it.get("status")}),
                           c.form("issues.assign", title="Assign", hide={"issue": iid}),
                           c.form("issues.close", title="Close", hide={"issue": iid})])

    # ------------------------------------------------------------------ cal.crucible
    @r("cal", "/")
    def cal_home(c: Ctx):
        c.title = "Calendar"
        args = {}
        for k in ("from_t", "to_t"):
            if c.qp.get(k, "").isdigit():
                args[k] = int(c.qp[k])
        rows = c.q("cal.agenda", **args)
        c.table("Agenda", rows, [("event", "Event", lambda r_: Link(r_.get("title") or r_["event"], c.url(
            "cal", "/event", id=r_["event"]))), ("start", "Start"), ("duration", "Duration (s)"),
            ("organiser", "Organiser"), ("response", "Your response"), ("conflicts", "Conflicts")],
            empty="No events in this window.")
        c.note(f"Times are simulated seconds; now is {c.w.now} ({fmt_time(c.w.now)}).", id="now")
        c.forms("Schedule", [c.form("cal.create", title="New event", values={"start": c.w.now + 3600,
                                                                             "duration": 1800},
                                    next="/event?id={event}")])

    @r("cal", "/event")
    def cal_event(c: Ctx):
        eid = c.qp.get("id", "")
        e = c.q("cal.get", event=eid)
        c.title = e.get("title") or eid
        c.kv("Event", e, ["event", "title", "start", "duration", "organiser", "creator", "status"],
             extra=[("description", c.rich(e.get("description")))])
        c.table("Responses", [{"principal": k, "response": v} for k, v in (e.get("responses") or {}).items()],
                [("principal", "Attendee"), ("response", "Response")])
        c.forms("Respond", [c.form("cal.respond", hide={"event": eid}, title="Respond",
                                   buttons=[("accept", "true", "Accept", "Accept invitation"),
                                            ("accept", "false", "Decline", "Decline invitation")]),
                            c.form("cal.cancel", hide={"event": eid}, title="Cancel event")])

    # ------------------------------------------------------------------ support.crucible
    @r("support", "/")
    def support_home(c: Ctx):
        owner = c.qp.get("owner") or None
        status = c.qp.get("status", "any")
        if status not in ("open", "answered", "resolved", "unresolved", "any"):
            status = "any"
        c.title = f"Support queue of {owner}" if owner else "Your support tickets"
        rows = c.q("support.list", status=status, limit=100, **({"owner": owner} if owner else {}))
        c.table("Tickets", rows, [("ticket", "Ticket", lambda r_: Link(r_.get("subject") or r_["ticket"], c.url(
            "support", "/ticket", id=r_["ticket"]))), ("host", "Service"), ("customer", "Customer"),
            ("severity", "Severity"), ("status", "Status"), ("opened", "Opened"), ("messages", "Messages")],
            empty="No tickets.")
        c.add("form", id="support-queue", form=c.get_form("support-queue", c.url("support", "/"), [
            Field("owner", "Operator queue (o:<handle>, needs support.reply)", "text", value=owner or "")],
            "Show queue", "Show an operator's queue"))
        c.add("form", id="support-stats", form=c.get_form("support-stats", c.url("support", "/stats"), [
            Field("owner", "Operator (o:<handle>)", "text", required=True)], "Show service levels",
            "Public service levels of an operator"))
        c.forms("New ticket", [c.form("support.open", title="Open a ticket", values={"host": c.qp.get("host", "")},
                                      next="/ticket?id={ticket}")])

    @r("support", "/ticket")
    def support_ticket(c: Ctx):
        tid = c.qp.get("id", "")
        t = c.q("support.get", ticket=tid)
        c.title = t.get("subject") or tid
        c.kv("Ticket", t, ["ticket", "host", "owner", "customer", "severity", "status", "opened", "first_response",
                           "resolved", "your_role"])
        children = [{"id": c.uid("msg"), "label": f"Message by {x.get('author') or x.get('by')}",
                     "meta": [c.who(x.get("author") or x.get("by")), fmt_time(x.get("t", 0))],
                     "body": c.rich(x.get("body"))} for x in t.get("messages", [])]
        c.add("article", "Conversation", id="conversation", meta=[], body=c.rich(t.get("body")), forms=[],
              children=children)
        c.forms("Answer", [c.form("support.reply", title="Reply", hide={"ticket": tid}),
                           c.form("support.resolve", title="Mark resolved", hide={"ticket": tid})])

    @r("support", "/stats")
    def support_stats(c: Ctx):
        owner = c.qp.get("owner", "")
        st = c.q("support.stats", owner=owner)
        c.title = f"Service levels of {owner}"
        if isinstance(st, dict):
            c.kv("Live queue", {k: v for k, v in st.items() if not isinstance(v, list)})
            for k, v in st.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    c.table(k.replace("_", " ").capitalize(), v, [(x, x) for x in v[0].keys()])
        else:
            c.add("json", "Statistics", text=json.dumps(st, indent=1))

    # ------------------------------------------------------------------ ads.crucible
    @r("ads", "/")
    def ads_home(c: Ctx):
        c.title = "Ad campaigns"
        rows = c.q("ads.stats")

        def actions(r_):
            cid = r_["campaign"]
            if r_.get("status") == "active":
                return c.form("ads.pause", hide={"campaign": cid}, key=cid, inline=True, submit="Pause",
                              submit_aria=f"Pause campaign {cid}")
            return c.form("ads.resume", hide={"campaign": cid}, key=cid, inline=True, submit="Resume",
                          submit_aria=f"Resume campaign {cid}")
        c.table("Your campaigns", rows, [("campaign", "Campaign"), ("status", "Status"), ("keywords", "Keywords"),
                                         ("bid_cents", "Bid"), ("daily_budget_cents", "Daily budget"),
                                         ("impressions", "Impressions"), ("spent_today", "Spent today"),
                                         ("total_spent", "Total spent"), ("account", "Account"),
                                         ("act", "Action", actions)], empty="No campaigns.")
        c.forms("Advertise", [c.form("ads.create_campaign", title="New campaign",
                                     labels={"bid_cents": "Bid per impression (cents)",
                                             "daily_budget_cents": "Daily budget (cents)"})])

    # ------------------------------------------------------------------ market.crucible
    def market_tabs(c: Ctx, cur: str) -> None:
        c.tabs("Market sections", [("Browse", c.url("market", "/"), cur == "browse"),
                                   ("Orders", c.url("market", "/orders"), cur == "orders"),
                                   ("Inventory", c.url("market", "/inventory"), cur == "inventory")])

    @r("market", "/")
    def market_home(c: Ctx):
        c.title = "Market"
        market_tabs(c, "browse")
        args = {"limit": 100}
        for k, allowed in (("kind", ("api", "code", "goods", "service")), ("sort", ("new", "price", "rating"))):
            if c.qp.get(k) in allowed:
                args[k] = c.qp[k]
        if c.qp.get("need"):
            args["need"] = c.qp["need"]
        rows = c.q("market.browse", **args)
        c.add("form", id="market-filter", form=c.get_form("market-filter", c.url("market", "/"), [
            Field("kind", "Kind", "select", value=c.qp.get("kind", ""),
                  options=[("", "(any)"), ("api", "api"), ("code", "code"), ("goods", "goods"), ("service", "service")]),
            Field("need", "Need format", "text", value=c.qp.get("need", "")),
            Field("sort", "Sort", "select", value=c.qp.get("sort", ""),
                  options=[("", "(default)"), ("new", "new"), ("price", "price"), ("rating", "rating")])],
            "Filter", "Filter listings"))
        c.table("Listings", rows, [("listing", "Listing", lambda r_: Link(r_.get("title") or r_["listing"], c.url(
            "market", "/listing", id=r_["listing"]))), ("kind", "Kind"), ("seller", "Seller"),
            ("price_cents", "Price"), ("need", "Need"), ("host", "Service", lambda r_: Link(
                f"{r_['host']}.crucible", c.url(r_["host"], "/")) if r_.get("host") else None),
            ("rating", "Rating"), ("sales", "Sales")], empty="No listings.")
        c.forms("Sell", [c.form("market.list", title="New listing", labels={"price_cents": "Price (cents)"},
                                widgets={"seller": {"kind": "select", "options": [("", "(yourself)")] + [
                                    (o["org"], o["org"]) for o in c.orgs()]}},
                                next="/listing?id={listing}")])

    @r("market", "/listing")
    def market_listing(c: Ctx):
        lid = c.qp.get("id", "")
        L = c.q("market.view", listing=lid)
        c.title = L.get("title") or lid
        market_tabs(c, "")
        c.kv("Listing", L, ["listing", "kind", "seller", "price_cents", "status", "need", "host", "endpoint",
                            "rating", "ratings", "sales"], extra=[("description", c.rich(L.get("description")))])
        c.table("Recent reviews", L.get("recent_reviews"), [(k, k) for k in ("by", "stars", "text", "t")],
                empty="No reviews yet.")
        c.forms("Buy", [c.form("market.buy", title="Buy", hide={"listing": lid}, next="/orders")])
        c.forms("Review", [c.form("market.rate", title="Rate this listing", hide={"listing": lid})])
        sellers = {c.p} | {o["org"] for o in c.orgs()}
        if L.get("seller") in sellers:
            c.forms("Manage", [c.form("market.update", title="Update listing", hide={"listing": lid},
                                      values={"price_cents": L.get("price_cents")})])

    @r("market", "/orders")
    def market_orders(c: Ctx):
        role = c.qp.get("role", "buyer")
        if role not in ("buyer", "seller"):
            role = "buyer"
        c.title = "Orders"
        market_tabs(c, "orders")
        c.tabs("Role", [("As buyer", c.url("market", "/orders", role="buyer"), role == "buyer"),
                        ("As seller", c.url("market", "/orders", role="seller"), role == "seller")])
        rows = c.q("market.orders", role=role)

        def actions(o):
            oid, st = o["id"], o.get("status")
            out = []
            if role == "buyer" and st in ("paid", "delivered"):
                out.append(c.form("market.confirm", hide={"order": oid}, key=oid, inline=True, submit="Confirm",
                                  submit_aria=f"Confirm delivery of order {oid}"))
                out.append(c.form("market.dispute", hide={"order": oid}, key=oid, inline=True, submit="Dispute",
                                  submit_aria=f"Dispute order {oid}"))
            if role == "seller" and st == "paid":
                out.append(c.form("market.deliver", hide={"order": oid}, key=oid, inline=True, submit="Deliver",
                                  submit_aria=f"Mark order {oid} delivered"))
            if role == "seller" and st in ("paid", "delivered", "disputed"):
                out.append(c.form("market.refund", hide={"order": oid}, key=oid, inline=True, submit="Refund",
                                  submit_aria=f"Refund order {oid}"))
            return Many([x for x in out if x], " ") if out else None
        c.table("Orders", rows, [("id", "Order"), ("listing", "Listing", lambda o: Link(o["listing"], c.url(
            "market", "/listing", id=o["listing"]))), ("buyer", "Buyer"), ("seller", "Seller"), ("qty", "Qty"),
            ("amount", "Amount"), ("status", "Status"), ("updated", "Updated"), ("act", "Actions", actions)],
            empty="No orders.")

    @r("market", "/inventory")
    def market_inventory(c: Ctx):
        c.title = "Inventory"
        market_tabs(c, "inventory")
        rows = c.q("market.inventory")
        cols = [(k, k) for k in (rows[0].keys() if rows and isinstance(rows[0], dict) else ["sku", "qty"])]
        c.table("Goods you hold", rows, cols, empty="No goods.")

    # ------------------------------------------------------------------ cloud.crucible
    @r("cloud", "/")
    def cloud_home(c: Ctx):
        c.title = "Cloud"
        provs = c.q("cloud.providers")
        c.table("Compute providers", provs, [("provider", "Provider"), ("name", "Name"),
                                             ("price_core_hour_cents", "Price per core-hour"),
                                             ("cores_free", "Free cores"), ("cores_total", "Total cores"),
                                             ("outages_30d", "Outages (30d)"), ("in_outage", "In outage")])
        insts = c.tq("cloud.instances") or []
        c.table("Your instances", insts, [("instance", "Instance"), ("owner", "Owner"), ("provider", "Provider"),
                                          ("cores", "Cores"), ("status", "Status"),
                                          ("price_core_hour_cents", "Price per core-hour"),
                                          ("used_cu_this_hour", "CU used this hour"),
                                          ("release", "Release", lambda r_: c.form(
                                              "cloud.release", hide={"instance": r_["instance"]}, key=r_["instance"],
                                              inline=True, submit="Release",
                                              submit_aria=f"Release instance {r_['instance']}")
                                           if r_.get("status") == "running" else None)], empty="No instances.")
        deps = c.tq("cloud.deployments") or []
        c.table("Your deployments", deps, [
            ("host", "Service", lambda r_: Link(f"{r_.get('host')}.crucible", c.url(r_.get("host"), "/"))
             if r_.get("host") else None),
            ("status", "Status"), ("repo", "Repo"), ("instance", "Instance"),
            ("stats", "Analytics", lambda r_: Link("status", c.url("cloud", "/status", host=r_.get("host")),
                                                   label=f"Status of {r_.get('host')}")),
            ("undeploy", "Undeploy", lambda r_: c.form("cloud.undeploy", hide={"host": r_.get("host")},
                                                       key=r_.get("host"), inline=True, submit="Undeploy",
                                                       submit_aria=f"Undeploy {r_.get('host')}")
             if r_.get("status") == "live" else None)], empty="No deployments.")
        inst_opts = [(i["instance"], f"{i['instance']} ({i['cores']} cores, {i['status']})") for i in insts]
        c.forms("Compute", [
            c.form("cloud.rent", title="Rent an instance",
                   widgets={"provider": {"kind": "select", "options": [(p["provider"], f"{p['name']} ({p['provider']})")
                                                                       for p in provs]}}),
            c.form("cloud.deploy", title="Deploy a repository as <host>.crucible",
                   values={"repo": c.qp.get("repo", "")},
                   widgets={"instance": {"kind": "select", "options": inst_opts}} if inst_opts else None)])
        c.links("Operate a cloud", [Link("Provider console", c.url("cloud", "/provider"))])

    @r("cloud", "/provider")
    def cloud_provider(c: Ctx):
        c.title = "Provider console"
        c.forms("Provider", [c.form("cloud.register_provider", title="Become a compute provider"),
                             c.form("cloud.set_price", title="Set price"),
                             c.form("cloud.buy_hardware", title="Buy hardware")])

    @r("cloud", "/status")
    def cloud_status(c: Ctx):
        host = c.qp.get("host", "")
        args = {"host": host}
        if c.qp.get("days", "").isdigit():
            args["days"] = int(c.qp["days"])
        st = c.q("cloud.status", **args)
        c.title = f"Status of {host}.crucible"
        if isinstance(st, dict):
            c.kv("Service", {k: v for k, v in st.items() if not isinstance(v, list)})
            for k, v in st.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    c.table(k.replace("_", " ").capitalize(), v, [(x, x) for x in v[0].keys()])
        else:
            c.add("json", "Status", text=json.dumps(st, indent=1))

    # ------------------------------------------------------------------ biz.crucible
    @r("biz", "/")
    def biz_home(c: Ctx):
        c.title = "Company registry"
        status = c.qp.get("status")
        args = {"limit": 200}
        if status in ("registered", "bankrupt", "dissolved"):
            args["status"] = status
        rows = c.q("biz.registry", **args)
        c.tabs("Status", [("All", c.url("biz", "/"), status not in ("registered", "bankrupt", "dissolved"))] + [
            (s.capitalize(), c.url("biz", "/", status=s), s == status) for s in ("registered", "bankrupt", "dissolved")])
        c.table("Companies", rows, [("org", "Company", lambda r_: Link(r_.get("name") or r_["org"], c.url(
            "biz", "/company", org=r_["org"]))), ("org", "Id"), ("sector", "Sector"), ("since", "Since"),
            ("status", "Status")], empty="No companies.")
        c.forms("Register", [c.form("biz.register", title="Register an organisation as a company",
                                    labels={"account": "Pay the fee from"}, next="/company?org={company}")])
        c.forms("Wind up", [c.form("biz.dissolve", title="Dissolve a company")])

    @r("biz", "/company")
    def biz_company(c: Ctx):
        org = c.qp.get("org", "")
        co = c.q("biz.company", org=org)
        c.title = f"Company {org}"
        c.kv("Registry entry", co, [k for k in co.keys() if k != "shareholders"])
        c.table("Shareholders", co.get("shareholders"), [("holder", "Holder"), ("shares", "Shares")])
        c.links("Related", [Link("Organisation profile", c.url("id", "/profile", p=org))])

    # ------------------------------------------------------------------ contracts.crucible
    @r("contracts", "/")
    def contracts_home(c: Ctx):
        c.title = "Contracts"
        status = c.qp.get("status")
        statuses = ("proposed", "active", "terminating", "completed", "breached", "terminated", "rejected")
        rows = c.q("contracts.list", **({"status": status} if status in statuses else {}))
        c.tabs("Status", [("All", c.url("contracts", "/"), status not in statuses)] +
               [(s.capitalize(), c.url("contracts", "/", status=s), s == status) for s in statuses])
        c.table("Your contracts", rows, [("contract", "Contract", lambda r_: Link(r_.get("title") or r_["contract"],
                                                                                 c.url("contracts", "/contract",
                                                                                       id=r_["contract"]))),
                                         ("contract", "Id"), ("status", "Status"), ("parties", "Parties"),
                                         ("signed", "Signed")], empty="No contracts.")
        claims = c.tq("contracts.claims") or []
        c.table("Claims", claims, [(k, k) for k in (claims[0].keys() if claims else ["id"])]
                + [("settle", "Settle", lambda r_: c.form("contracts.settle_claim",
                                                          hide={"claim": r_.get("id") or r_.get("claim")},
                                                          key=r_.get("id") or r_.get("claim"), inline=True,
                                                          fields=["account"], submit="Settle"))],
                empty="No claims.")
        example = json.dumps([{"type": "payment", "from": c.p, "to": "p:<handle>", "amount": 1000}], indent=1)
        c.forms("Propose", [c.form("contracts.propose", title="Propose a contract",
                                   labels={"parties": "Parties (comma-separated principals)",
                                           "clauses": "Clauses (JSON array)"},
                                   helps={"clauses": "JSON array of clause objects. Types: payment, membership, "
                                                     "grant, shares, deliverable, notice. Example: " + example},
                                   values={"parties": c.p}, next="/contract?id={contract}")])

    @r("contracts", "/contract")
    def contracts_view(c: Ctx):
        cid = c.qp.get("id", "")
        k = c.q("contracts.view", contract=cid)
        c.title = k.get("title") or cid
        c.kv("Contract", k, ["id", "title", "status", "proposer", "parties", "signed", "expires", "t", "employer",
                             "employee", "accounts", "paid", "delivered"],
             extra=[("text", c.rich(k.get("text")))])
        c.add("json", "Clauses", id="clauses", text=json.dumps(k.get("clauses"), indent=1))
        c.forms("Act", [c.form("contracts.sign", title="Sign", hide={"contract": cid},
                               labels={"accounts": "Payment accounts (JSON: clause index -> account)"}),
                        c.form("contracts.reject", title="Reject", hide={"contract": cid}),
                        c.form("contracts.terminate", title="Terminate", hide={"contract": cid})])

    # ------------------------------------------------------------------ jobs.crucible
    @r("jobs", "/")
    def jobs_home(c: Ctx):
        c.title = "Jobs"
        role = c.qp.get("role")
        roles = ("developer", "support", "ops", "sales", "analyst", "general")
        rows = c.q("jobs.openings", limit=100, **({"role": role} if role in roles else {}))
        c.tabs("Role", [("All", c.url("jobs", "/"), role not in roles)] +
               [(x.capitalize(), c.url("jobs", "/", role=x), x == role) for x in roles])
        c.table("Openings", rows, [("posting", "Opening", lambda r_: Link(r_.get("title") or r_["posting"], c.url(
            "jobs", "/posting", id=r_["posting"]))), ("org", "Organisation"), ("role", "Role"),
            ("salary_cents_month", "Salary / month"), ("description", "Description")], empty="No openings.")
        bounties = c.tq("jobs.bounties") or []
        c.table("Bounties", bounties, [("bounty", "Bounty", lambda r_: Link(r_.get("title") or r_["bounty"], c.url(
            "jobs", "/bounty", id=r_["bounty"]))), ("need", "Need"), ("reward_cents", "Reward"),
            ("min_quality", "Min quality"), ("deadline", "Deadline"), ("poster", "Poster"), ("attempts", "Attempts")],
            empty="No bounties.")
        c.forms("Hire", [c.form("jobs.post", title="Post a job opening", next="/posting?id={posting}"),
                         c.form("jobs.post_bounty", title="Post a bounty")])

    @r("jobs", "/posting")
    def jobs_posting(c: Ctx):
        pid = c.qp.get("id", "")
        posting = next((x for x in (c.tq("jobs.openings", limit=200) or []) if x.get("posting") == pid), None)
        c.title = (posting or {}).get("title") or f"Opening {pid}"
        if posting:
            c.kv("Opening", posting, ["posting", "org", "title", "role", "salary_cents_month", "description"])
        else:
            c.note("This opening is not (or no longer) listed.")
        c.forms("Apply", [c.form("jobs.apply", title="Apply", hide={"posting": pid})])
        try:
            apps = c.w.query(c.p, "jobs.applications", {"posting": pid})
        except CrucibleError:
            apps = None
        if apps is not None:
            c.table("Applications", apps, [("application", "Application"), ("applicant", "Applicant"),
                                           ("status", "Status"), ("note", "Note"),
                                           ("public_profile", "Profile")], empty="No applications yet.")
            c.forms("Offer", [c.form("jobs.offer", title="Offer employment",
                                     widgets={"application": {"kind": "select", "options": [
                                         (a["application"], f"{a['applicant']} ({a['status']})") for a in apps]}}
                                     if apps else None),
                              c.form("jobs.close", title="Close this opening", hide={"posting": pid})])

    @r("jobs", "/bounty")
    def jobs_bounty(c: Ctx):
        bid = c.qp.get("id", "")
        b = next((x for x in (c.tq("jobs.bounties") or []) if x.get("bounty") == bid), None)
        c.title = (b or {}).get("title") or f"Bounty {bid}"
        if b:
            c.kv("Bounty", b)
        c.forms("Submit", [c.form("jobs.submit", title="Submit code", hide={"bounty": bid})])

    # ------------------------------------------------------------------ research.crucible
    @r("research", "/")
    def research_home(c: Ctx):
        c.title = "Market research"
        c.forms("Survey", [c.form("research.survey", title="Commission a market survey",
                                  labels={"account": "Pay from"})])

    # ------------------------------------------------------------------ every host: endpoint console
    @r("*", "/")
    def deployed_home(c: Ctx):
        api_page(c, deployed=True)

    for h in sorted(SERVICE_HOSTS) + ["*"]:
        app.route(h, "/_api")(lambda c: api_page(c, deployed=False))

    def api_page(c: Ctx, deployed: bool) -> None:
        host = c.host
        d = c.q("world.describe", host=host)
        eps = d.get("endpoints") or []
        c.title = f"{host}.crucible" if deployed else f"All endpoints of {host}.crucible"
        if deployed:
            owner_fn = c.w.hooks.get("host_owner")
            owner = owner_fn(c.w.state, host) if owner_fn else None
            c.kv("Service", {"host": f"{host}.crucible", "operator": owner, "endpoints": len(eps)},
                 extra=[("support", Link("Open a support ticket", c.url("support", "/", host=host))),
                        ("service levels", Link("Support statistics", c.url("support", "/stats", owner=owner)))
                        if owner else ("service levels", None)])
            known = c.w.knows(c.p, "host", host)
            if not known:
                c.forms("Bookmark", [c.form("world.describe", hide={"host": host}, inline=True,
                                            submit="Add to my tools",
                                            submit_aria=f"Add {host}.crucible to my tool list")],
                        note="Adding the service records a world.describe call: its endpoints then appear in your "
                             "API tool list too.")
        c.note("Each form below calls one endpoint as you (a recorded command); the result is shown at the top.",
               id="api-note")
        for e in eps:
            tool = e["name"]
            ann = e.get("annotations") or {}
            f = c.form(tool, title=tool, submit=f"Call {tool.split('.', 1)[1]}")
            if f is not None:
                c.add("form", id="ep-" + tool.replace(".", "-"), form=f)
            elif ann.get("host") not in HIDDEN_HOSTS:
                c.note(f"{tool} is unavailable.")


def _path_mode(path: str) -> tuple[str, str] | None:
    m = re.match(r"/_h/([^/]+)(/.*)?\Z", path)
    if not m:
        return None
    return m.group(1).lower(), m.group(2) or "/"


def make_web_app(server_or_world_provider: Any, world_id: str | None = None, *, secure_cookies: bool = False,
                 notifications: int = 8) -> FastAPI:
    """The browser-mode FastAPI app for one world.

    `server_or_world_provider`: a `WorldServer` (then `world_id` names the world), a kernel `World`, or a
    zero-argument callable returning the `World`. Pages are routed by Host header (``<host>.crucible``) or by the
    path prefix ``/_h/<host>/``. ``app.state.web`` is the `WebApp` (sessions, routes)."""
    backend = WebBackend(server_or_world_provider, world_id)
    web = WebApp(backend, secure_cookies=secure_cookies, notifications=notifications)
    app = FastAPI(title="Crucible browser mode", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.web = web
    app.mount("/_static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")

    async def dispatch(request: Request, full: str = "") -> Response:
        path = request.url.path
        pm = _path_mode(path)
        if pm is not None:
            host, sub = pm
            mode = "path"
        else:
            hn = (request.headers.get("host") or "").rsplit(":", 1)[0].lower().rstrip(".")
            if hn.endswith(".crucible"):
                host, sub, mode = hn[: -len(".crucible")], path, "host"
            else:
                if path in ("/", ""):
                    return _redirect("/_h/id/")
                return HTMLResponse("<!doctype html><title>Not found</title><h1>Not found</h1>"
                                    "<p>Use http://&lt;host&gt;.crucible/ or /_h/&lt;host&gt;/.</p>", status_code=404)
        if not HOST_RE.fullmatch(host):
            return HTMLResponse("<!doctype html><title>Bad host</title><h1>Bad host name</h1>", status_code=404)
        form = None
        if request.method == "POST":
            body = (await request.body()).decode("utf-8", "replace")
            ctype = request.headers.get("content-type", "")
            if "multipart/form-data" in ctype:
                fd = await request.form()
                form = {}
                for k, v in fd.multi_items():
                    form.setdefault(k, []).append(v if isinstance(v, str) else "")
            else:
                form = {}
                for k, v in parse_qsl(body, keep_blank_values=True):
                    form.setdefault(k, []).append(v)
        return await run_in_threadpool(web.handle, request, host, sub, mode, form)

    app.add_api_route("/", dispatch, methods=["GET", "POST"], include_in_schema=False)
    app.add_api_route("/{full:path}", dispatch, methods=["GET", "POST"], include_in_schema=False)
    return app


# ====================================================================== CLI
def main(argv: list[str] | None = None) -> int:
    from ..hosting import engage
    engage("web.app")                          # Part 16: no real-world bridge in a process that hosts worlds
    ap = argparse.ArgumentParser(prog="python -m crucible.web.app",
                                 description="Serve a Crucible world as browsable *.crucible web sites.")
    ap.add_argument("--store", default="memory://", help="memory://, sqlite:///worlds.db or postgresql://...")
    ap.add_argument("--world", help="world id or name in the store (default: create one with --create)")
    ap.add_argument("--create", metavar="CONFIG", help="create a world from a generator config (e.g. tiny)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--join", action="append", default=[], metavar="HANDLE:PASSWORD[:ENDOWMENT_CENTS]",
                    help="admit a human/agent actor with a password (repeatable)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(argv)
    import crucible
    import uvicorn

    from ..api.local import WorldServer
    crucible.load_all_modules()
    srv = WorldServer(a.store)
    wid = a.world
    if wid is None:
        if not a.create:
            ap.error("give --world or --create")
        wid = srv.create_world(a.create, seed=a.seed)
    for j in a.join:
        parts = j.split(":")
        if len(parts) < 2:
            ap.error(f"--join {j!r}: expected HANDLE:PASSWORD[:ENDOWMENT]")
        srv.join_world(wid, parts[0], kind="human", password=parts[1],
                       endowment=int(parts[2]) if len(parts) > 2 else 0)
    app = make_web_app(srv, wid)
    print(f"world {wid}: browse http://{a.host}:{a.port}/_h/id/ (or map *.crucible to {a.host}:{a.port}, e.g. "
          f"chromium --host-resolver-rules='MAP *.crucible {a.host}:{a.port}' http://id.crucible/)", flush=True)
    try:
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    finally:
        srv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
