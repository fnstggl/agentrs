"""The synthetic internet fabric (modules `net` - host `net` - and `appdb` - host `db`).

Every web application in Crucible - built-in pages aside - is ordinary agent code deployed through `cloud`: a
deployment whose manifest has a ``web`` section serves HTTP routes::

    "web": {"routes": {"GET /": "index", "GET /item/{id}": "item", "POST /items": "create"},
            "static": {"/style.css": "static/style.css"}, "rate_per_min": 120, "login": "optional"}

A route handler is called as ``handler(request, caller)`` inside the sandbox with::

    request = {"method", "path", "params", "query", "form", "json", "headers", "session", "user", "now", "host"}

and returns ``{"status"?, "html" | "json" | "text" | "redirect", "session"?, "headers"?}``. Applications keep
persistent data in their own namespace of the application database (`db.*`, callable only by the application's
service principal), may send mail, bill users through bank invoices and call any tool they hold grants for.

Name resolution: ``<host>.crucible`` names every deployment; organisations may also register names
(`net.register`) and point them at a deployment they operate (`net.point`). Requests are one recorded command
(`net.fetch`), whether they come from an API agent or a real Chromium browser through the web front end, so both
interfaces have identical semantics. Sessions are server-side per (site, visitor); "Login with Crucible ID"
(`net.login`) tells a site who the visitor is - only after the visitor consents.

Realistic failure: rate limits (429), overload (503), outages (503), handler errors (500), timeouts (the sandbox's),
unknown names (DNS failure), missing pages (404).

Discovery: a crawler visits sites daily from seeds (registered names, deployments, links found on pages, names
mentioned in public posts), follows links (bounded), and indexes pages that are public (HTTP 200 for an anonymous
visitor, no ``<meta name="robots" content="noindex">``). The link graph feeds a ranking signal. The index is
therefore incomplete and lags behind the web, like a real one.
"""
from __future__ import annotations

import html as _html
import json
import math
import re
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound, RateLimited
from ..core.kernel import SYSTEM, Ctx, Module, applier, endpoint, process, register_module
from .identity import ident

DAY = 86400
NAME = r"[a-z][a-z0-9-]{2,30}"
CRAWLER = "p:webcrawler"
THIN_PAGE_CHARS = 300      # visible text below this is ranked down proportionally (doorway pages)
MAX_BODY = 200_000
TAG_RE = re.compile(r"<[^>]+>")
HREF_RE = re.compile(r"""href\s*=\s*["']([^"'#]+)""", re.I)
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.I | re.S)
NOINDEX_RE = re.compile(r"""<meta[^>]+name=["']robots["'][^>]+content=["'][^"']*noindex""", re.I)
MENTION_RE = re.compile(r"\b([a-z][a-z0-9-]{2,30})\.crucible\b")


def page_text(body: str) -> str:
    body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", body, flags=re.S | re.I)
    return re.sub(r"\s+", " ", _html.unescape(TAG_RE.sub(" ", body))).strip()


def match_route(routes: dict, method: str, path: str):
    """Return (handler, params) for the first matching route; `{name}` matches one path segment."""
    for key in sorted(routes, key=lambda k: (k.count("{"), -len(k))):
        m_, pat = key.split(" ", 1)
        if m_ != method:
            continue
        rx = "^" + re.sub(r"\\\{([A-Za-z_][A-Za-z0-9_]*)\\\}", r"(?P<\1>[^/]+)", re.escape(pat)) + "$"
        mt = re.match(rx, path)
        if mt:
            return routes[key], mt.groupdict()
    return None, {}


@register_module
class Net(Module):
    name = "net"
    depends = ("identity", "bank", "cloud")

    def install(self, world):
        st = world.state
        st.declare_table("domains", index=("owner", "host"))
        st.declare_table("web_sessions", index=("host", "visitor"))
        st.declare_table("web_pages", index=("host",))
        st.declare_table("web_links", index=("dst", "src_host"))
        st.declare_table("web_rate")
        st.declare_table("web_days", index=("host",))
        world.hooks["net"] = self
        prev = world.hooks.get("rank_signal")

        def rank(w, doc):
            base = prev(w, doc) if prev else 1.0
            if doc.get("kind") != "page" or not doc.get("hosts"):
                return base
            return base * self.page_authority(w, doc)
        world.hooks["rank_signal"] = rank

    def page_authority(self, world, doc) -> float:
        """Ranking signal of an indexed page. Links count as a vote only when they come from somebody else: one
        vote per independent owner (`goals.metrics.is_related`), however many sites that owner links from - a link
        farm on the page owner's own hosts adds nothing. A page with almost no text (a doorway page that only
        repeats the words people search for) is ranked below pages that say something: its score is scaled by
        the share of THIN_PAGE_CHARS of visible text it has."""
        from ..goals.metrics import is_related
        st = world.state
        host, owner = doc["hosts"][0], doc.get("owner")
        voters = set()
        for src in sorted({lk["src_host"] for lk in st.find("web_links", "dst", host) if lk["src_host"] != host}):
            h = st.get("hosts", src)
            if h is not None and owner and not is_related(world, owner, h["owner"]):
                voters.add(h["owner"])
        page = st.get("web_pages", doc.get("ref") or "")
        chars = page["chars"] if page is not None else len(doc.get("text") or "")
        return (1.0 + 0.35 * math.log1p(len(voters))) * min(1.0, max(chars, 1) / THIN_PAGE_CHARS)

    def genesis(self, ctx):
        idm = ctx.world.modules["identity"]
        if not ctx.state.has("principals", CRAWLER):
            idm.create_person(ctx, "webcrawler", "Search crawler", "npc", {"role": "crawler"})
        ctx.schedule(ctx.now + DAY - 7200, "net.crawl", {}, prio=85, period=DAY, key="net.crawl")

    def p(self, key, default):
        return self.config.get(key, self.world.spec.params.get(key, default))

    # ------------------------------------------------------------------ names
    def resolve_host(self, st, name: str) -> str | None:
        """Name -> deployment host (registered names first, then deployment hosts)."""
        name = name.lower()
        if name.endswith(".crucible"):
            name = name[: -len(".crucible")]
        d = st.get("domains", name)
        if d is not None and d["status"] == "active" and d.get("host"):
            return d["host"]
        if st.get("hosts", name) is not None:
            return name
        return None

    def web_dep(self, st, host: str):
        cloud = self.world.modules["cloud"]
        dep = cloud.live_deployment(st, host)
        if dep is None or "web" not in dep["manifest"]:
            return None, None
        return st.get("hosts", host)["deployment"], dep

    @endpoint("register", S.obj({"name": S.s_str(pattern=NAME, maxLength=31), "org": S.s_str(maxLength=40),
                                 "account": S.s_str(maxLength=40)}, ["name"]), cost_s=60)
    def ep_register(self, ctx, name, org=None, account=None):
        """Register a name `<name>.crucible` for you or an organisation (fee `domain_fee_cents`, default 2,000,
        paid to the registry). Names of built-in services and existing deployments are taken."""
        st = ctx.state
        owner = ident(ctx.world).resolve_handle(st, org) if org else ctx.actor
        ident(ctx.world).authorize(ctx, owner, "org.admin")
        if st.has("domains", name) or st.get("hosts", name) is not None or \
                any(e.host == name for e in ctx.world.endpoints.values()):
            raise Conflict(f"{name}.crucible is taken")
        bank = ctx.world.modules["bank"]
        src = account or bank.primary_account(st, owner)
        if src is None:
            raise InvalidAction("no account to pay the registration fee")
        bank.move(ctx, src, "acct_fees", int(self.p("domain_fee_cents", 2000)), "fee", f"registration of {name}.crucible")
        ctx.emit("net.domain", {"name": name, "owner": owner, "host": None}, vis="public")
        return {"domain": f"{name}.crucible", "owner": owner}

    @endpoint("point", S.obj({"name": S.s_str(maxLength=40), "host": S.s_str(maxLength=31)}, ["name", "host"]))
    def ep_point(self, ctx, name, host):
        """Point a name you own at a deployment you operate."""
        st = ctx.state
        name = name[: -len(".crucible")] if name.endswith(".crucible") else name
        d = st.get("domains", name)
        if d is None:
            raise NotFound("no such registered name")
        ident(ctx.world).authorize(ctx, d["owner"], "org.admin")
        h = st.get("hosts", host)
        if h is None or h["owner"] != d["owner"]:
            raise Forbidden("point names only at deployments of the same owner")
        ctx.emit("net.domain", {"name": name, "owner": d["owner"], "host": host}, vis="public")
        return {"domain": f"{name}.crucible", "host": host}

    @endpoint("whois", S.obj({"name": S.s_str(maxLength=40)}, ["name"]), read_only=True)
    def ep_whois(self, ctx, name):
        """Public registration record of a name."""
        st = ctx.state
        n = name[: -len(".crucible")] if name.endswith(".crucible") else name
        d = st.get("domains", n)
        if d is not None:
            return {"name": f"{n}.crucible", "owner": d["owner"], "host": d.get("host"), "registered": d["t"]}
        h = st.get("hosts", n)
        if h is not None:
            return {"name": f"{n}.crucible", "owner": h["owner"], "host": n, "registered": None}
        raise NotFound(f"{n}.crucible does not resolve")

    # ------------------------------------------------------------------ HTTP
    @endpoint("fetch", S.obj({"url": S.s_str("http://<name>.crucible/path?query", maxLength=500),
                              "method": S.s_enum(["GET", "POST"]),
                              "form": {"type": "object", "additionalProperties": {"type": ["string", "number",
                                                                                           "boolean", "null"]},
                                       "description": "form fields (POST)"},
                              "json": S.s_any_obj("JSON body (POST)")}, ["url"]), cost_s=3)
    def ep_fetch(self, ctx, url, method="GET", form=None, json=None):
        """Request a page or API route of a web application on the synthetic internet, as you. Returns
        {status, content_type, body, location?}. The same call is made when you browse with Chromium."""
        return self.fetch(ctx, url, method, form or {}, json, ctx.actor)

    def fetch(self, ctx: Ctx, url: str, method: str, form: dict, body_json, visitor: str) -> dict:
        st = ctx.state
        u = urlsplit(url if "://" in url else "http://" + url)
        name = (u.hostname or "").lower()
        if not name.endswith(".crucible"):
            return {"status": 502, "content_type": "text/plain", "body": "only *.crucible hosts exist on this network"}
        host = self.resolve_host(st, name)
        if host is None:
            return {"status": 404, "content_type": "text/plain", "body": f"{name}: DNS lookup failed"}
        dep_id, dep = self.web_dep(st, host)
        if dep is None:
            return {"status": 404, "content_type": "text/plain", "body": f"{name} serves no web pages"}
        web = dep["manifest"]["web"]
        path = u.path or "/"
        cloud = ctx.world.modules["cloud"]
        why = cloud.availability(st, dep)
        if why:
            self._hit(ctx, host, "unavailable")
            return {"status": 503, "content_type": "text/plain", "body": f"{name} is unavailable ({why})"}
        if path in web["static"] and method == "GET":
            files = ctx.world.modules["code"].files(st, dep["repo"], dep["commit"])
            fn = web["static"][path]
            ctype = {"css": "text/css", "js": "application/javascript", "svg": "image/svg+xml",
                     "txt": "text/plain", "json": "application/json"}.get(fn.rsplit(".", 1)[-1], "text/plain")
            self._hit(ctx, host, "static")
            return {"status": 200, "content_type": ctype, "body": files[fn][:MAX_BODY]}
        minute = ctx.now // 60
        rk = f"{host}|{visitor}|{minute}"
        used = (st.get("web_rate", rk) or {}).get("n", 0)
        if used >= web["rate_per_min"]:
            self._hit(ctx, host, "rate_limited")
            return {"status": 429, "content_type": "text/plain", "body": "too many requests; slow down"}
        ctx.emit("net.rate", {"id": rk, "minute": minute}, vis="internal")
        handler, params = match_route(web["routes"], method, path)
        if handler is None:
            self._hit(ctx, host, "not_found")
            return {"status": 404, "content_type": "text/plain", "body": f"no page {path} on {name}"}
        skey = f"{host}|{visitor}"
        sess = st.get("web_sessions", skey)
        user = visitor if sess is not None and sess.get("login") else None
        if web.get("login") == "required" and user is None and visitor != CRAWLER:
            self._hit(ctx, host, "login_required")
            return {"status": 401, "content_type": "text/plain", "body": f"log in to {name} with your Crucible ID",
                    "login_required": True}
        req = {"method": method, "path": path, "params": params,
               "query": dict(parse_qsl(u.query, keep_blank_values=True)), "form": dict(form), "json": body_json,
               "session": dict((sess or {}).get("data") or {}), "user": user, "now": ctx.now, "host": host,
               "visitor": "anonymous" if user is None else user}
        res = cloud.execute_handler(ctx, dep_id, handler, req, user or "anonymous", float(cloud.p("cu_per_ms", 50.0)))
        if res["status"] != 200:
            self._hit(ctx, host, "error" if res["status"] == 500 else "overload")
            return {"status": res["status"], "content_type": "text/plain", "body": str(res.get("error", ""))[:2000]}
        out = res["body"] if isinstance(res["body"], dict) else {"text": str(res["body"])}
        status = int(out.get("status", 200)) if isinstance(out.get("status", 200), int) else 200
        if "session" in out and isinstance(out["session"], dict):
            data = out["session"]
            if len(json_dumps(data)) > 20_000:
                data = {}
            ctx.emit("net.session", {"id": skey, "host": host, "visitor": visitor, "data": data,
                                     "login": bool((sess or {}).get("login"))}, vis="internal")
        self._hit(ctx, host, "ok")
        if visitor != CRAWLER:
            ctx.world.learn(ctx, visitor, "host", host)
        if out.get("redirect"):
            loc = str(out["redirect"])
            return {"status": 303, "content_type": "text/plain", "body": "", "location": loc}
        if "html" in out:
            return {"status": status, "content_type": "text/html", "body": str(out["html"])[:MAX_BODY]}
        if "json" in out:
            return {"status": status, "content_type": "application/json", "body": json_dumps(out["json"])[:MAX_BODY]}
        return {"status": status, "content_type": "text/plain", "body": str(out.get("text", ""))[:MAX_BODY]}

    def _hit(self, ctx, host, outcome):
        ctx.emit("net.hit", {"host": host, "day": ctx.now // DAY, "outcome": outcome}, vis="internal")

    @endpoint("login", S.obj({"site": S.s_str("name of the site", maxLength=40)}, ["site"]), cost_s=5)
    def ep_login(self, ctx, site):
        """Log in to a site with your Crucible ID: the site learns your principal id (and nothing else)."""
        st = ctx.state
        host = self.resolve_host(st, site)
        if host is None:
            raise NotFound(f"{site} does not resolve")
        skey = f"{host}|{ctx.actor}"
        sess = st.get("web_sessions", skey)
        ctx.emit("net.session", {"id": skey, "host": host, "visitor": ctx.actor,
                                 "data": dict((sess or {}).get("data") or {}), "login": True}, vis=(ctx.actor,))
        return {"site": f"{host}.crucible", "logged_in_as": ctx.actor}

    @endpoint("logout", S.obj({"site": S.s_str(maxLength=40)}, ["site"]))
    def ep_logout(self, ctx, site):
        """Log out of a site (its session for you is cleared)."""
        host = self.resolve_host(ctx.state, site)
        if host is None:
            raise NotFound(f"{site} does not resolve")
        ctx.emit("net.session", {"id": f"{host}|{ctx.actor}", "host": host, "visitor": ctx.actor, "data": {},
                                 "login": False}, vis=(ctx.actor,))
        return {"ok": True}

    @endpoint("sites", S.obj({"limit": S.s_int(minimum=1, maximum=200)}), read_only=True)
    def ep_sites(self, ctx, limit=50):
        """Sites you have heard of: those mentioned to you (mail, posts, links you followed) and their titles.
        There is no complete directory; search (`search.query` with kind "page") finds indexed pages."""
        st = ctx.state
        out = []
        for kid in st.find_ids("knowledge", "principal", ctx.actor):
            k = st.tables["knowledge"][kid]
            if k["kind"] == "host":
                _, dep = self.web_dep(st, k["value"])
                if dep is not None:
                    out.append({"site": f"{k['value']}.crucible", "title": dep["manifest"]["web"].get("title", "")})
        return out[:limit]

    # ------------------------------------------------------------------ crawling and indexing
    @process("crawl")
    def pr_crawl(self, ctx):
        st = ctx.state
        if self.p("crawl", True) is False:
            return
        seeds = set()
        for h, rec in st.scan("hosts"):
            if self.web_dep(st, h)[1] is not None:
                seeds.add(h)
        max_pages = int(self.p("crawl_pages_per_site", 6))
        for host in sorted(seeds):
            frontier = ["/"]
            seen: set[str] = set()
            while frontier and len(seen) < max_pages:
                path = frontier.pop(0)
                if path in seen:
                    continue
                seen.add(path)
                try:
                    r = ctx.run_as(CRAWLER, lambda sub: self.fetch(sub, f"http://{host}.crucible{path}", "GET", {},
                                                                   None, CRAWLER))
                except CrucibleError:
                    continue
                url = f"{host}.crucible{path}"
                if r["status"] != 200 or r["content_type"] != "text/html":
                    if st.has("web_pages", url):
                        ctx.emit("net.page_gone", {"url": url}, vis="internal")
                        from .search import unindex
                        unindex(ctx, f"page:{url}")
                    continue
                body = r["body"]
                title = (TITLE_RE.search(body).group(1).strip() if TITLE_RE.search(body) else url)[:200]
                text = page_text(body)[:4000]
                links = []
                for href in HREF_RE.findall(body)[:60]:
                    lu = urlsplit(href)
                    if lu.hostname and lu.hostname.endswith(".crucible"):
                        dst = self.resolve_host(st, lu.hostname)
                        if dst:
                            links.append(dst)
                            if dst == host and lu.path and lu.path not in seen:
                                frontier.append(lu.path)
                    elif not lu.scheme and href.startswith("/") and href not in seen:
                        frontier.append(href.split("?")[0])
                noindex = bool(NOINDEX_RE.search(body))
                ctx.emit("net.page", {"url": url, "host": host, "title": title, "links": sorted(set(links)),
                                      "noindex": noindex, "chars": len(text)}, vis="internal")
                from .search import index, unindex
                if noindex:
                    unindex(ctx, f"page:{url}")
                else:
                    index(ctx, f"page:{url}", "page", title, text, st.tables["hosts"][host]["owner"], ref=url,
                          hosts=[host] + sorted(set(links) - {host})[:6])

    # ------------------------------------------------------------------ appliers
    @applier("net.domain")
    def ap_domain(self, ev):
        p = ev.payload
        st = self.world.state
        old = st.get("domains", p["name"])
        st.put("domains", p["name"], {"owner": p["owner"], "host": p["host"], "status": "active",
                                      "t": old["t"] if old else ev.t})

    @applier("net.rate")
    def ap_rate(self, ev):
        st = self.world.state
        r = st.tables["web_rate"].get(ev.payload["id"])
        st.put("web_rate", ev.payload["id"], {"n": (r["n"] if r else 0) + 1, "minute": ev.payload["minute"]})
        if len(st.tables["web_rate"]) > 20_000:           # bounded: keep only the current minute
            for k in [k for k, v in st.tables["web_rate"].items() if v["minute"] < ev.payload["minute"]]:
                st.delete("web_rate", k)

    @applier("net.session")
    def ap_session(self, ev):
        p = ev.payload
        self.world.state.put("web_sessions", p["id"], {"host": p["host"], "visitor": p["visitor"], "data": p["data"],
                                                       "login": p["login"], "t": ev.t})

    @applier("net.hit")
    def ap_hit(self, ev):
        p = ev.payload
        st = self.world.state
        k = f"{p['host']}|{p['day']}"
        r = dict(st.tables["web_days"].get(k) or {"host": p["host"], "day": p["day"]})
        r[p["outcome"]] = r.get(p["outcome"], 0) + 1
        st.put("web_days", k, r)

    @applier("net.page")
    def ap_page(self, ev):
        p = ev.payload
        st = self.world.state
        st.put("web_pages", p["url"], {"host": p["host"], "title": p["title"], "links": p["links"],
                                       "noindex": p["noindex"], "chars": p["chars"], "t": ev.t})
        for k in [k for k in st.find_ids("web_links", "src_host", p["host"])
                  if st.tables["web_links"][k]["src"] == p["url"]]:
            st.delete("web_links", k)
        for dst in p["links"]:
            st.put("web_links", f"{p['url']}|{dst}", {"src": p["url"], "src_host": p["host"], "dst": dst})

    @applier("net.page_gone")
    def ap_page_gone(self, ev):
        st = self.world.state
        if st.has("web_pages", ev.payload["url"]):
            st.delete("web_pages", ev.payload["url"])


def json_dumps(x) -> str:
    return json.dumps(x, sort_keys=True, default=str, separators=(",", ":"))


@register_module
class AppDB(Module):
    """Per-application document storage (host `db`). Only a service principal can call it, and only for its own
    namespace (its host); quotas bound size. State is world state: it is snapshotted, forked and replayed."""
    name = "appdb"
    host = "db"
    depends = ("identity",)
    MAX_DOCS = 20_000
    MAX_DOC = 20_000

    def install(self, world):
        world.state.declare_table("app_docs", index=("ns", "nc"))

    def _ns(self, ctx, as_site=None) -> str:
        st = ctx.state
        p = st.get("principals", ctx.actor)
        if p is not None and p["kind"] == "service":
            return p["info"]["host"]
        if as_site:      # an operator inspecting its own application's data
            h = st.get("hosts", as_site)
            if h is not None and ident(ctx.world).allowed(ctx, ctx.actor, h["owner"], "org.admin"):
                return as_site
        raise Forbidden("the application database is for deployed applications (call it from your service code)")

    def _key(self, ns, coll, did):
        return f"{ns}|{coll}|{did}"

    @endpoint("put", S.obj({"collection": S.s_str(pattern=r"[a-z][a-z0-9_]{0,30}", maxLength=31),
                            "id": S.s_str(pattern=r"[A-Za-z0-9_.:-]{1,64}", maxLength=64),
                            "doc": S.s_any_obj()}, ["collection", "doc"]))
    def ep_put(self, ctx, collection, doc, id=None):
        """Insert or replace a document in your application's collection; returns its id."""
        st = ctx.state
        ns = self._ns(ctx)
        if len(json_dumps(doc)) > self.MAX_DOC:
            raise InvalidAction("document too large")
        if id is None:
            id = f"{collection[:3]}{st.next_counter('appdb:' + ns):06d}"
        k = self._key(ns, collection, id)
        if not st.has("app_docs", k) and len(st.find_ids("app_docs", "ns", ns)) >= self.MAX_DOCS:
            raise InvalidAction("database quota exceeded")
        ctx.emit("db.put", {"id": k, "ns": ns, "coll": collection, "doc_id": id, "doc": doc}, vis="internal")
        return {"id": id}

    @endpoint("get", S.obj({"collection": S.s_str(maxLength=31), "id": S.s_str(maxLength=64),
                            "site": S.s_str(maxLength=31)}, ["collection", "id"]), read_only=True)
    def ep_get(self, ctx, collection, id, site=None):
        """Read one document (None if absent)."""
        r = ctx.state.get("app_docs", self._key(self._ns(ctx, site), collection, id))
        return None if r is None else {"id": id, **r["doc"]}

    @endpoint("query", S.obj({"collection": S.s_str(maxLength=31), "where": S.s_any_obj("field -> required value"),
                              "order_by": S.s_str(maxLength=40), "desc": S.s_bool(), "limit": S.s_int(minimum=1, maximum=500),
                              "site": S.s_str(maxLength=31)}, ["collection"]), read_only=True)
    def ep_query(self, ctx, collection, where=None, order_by=None, desc=False, limit=100, site=None):
        """Documents of a collection matching all `where` fields, optionally ordered."""
        st = ctx.state
        ns = self._ns(ctx, site)
        rows = [{"id": r["doc_id"], **r["doc"]} for r in st.find("app_docs", "nc", f"{ns}|{collection}")
                if all(r["doc"].get(k) == v for k, v in (where or {}).items())]
        if order_by:
            rows.sort(key=lambda d: (d.get(order_by) is None, d.get(order_by) if isinstance(d.get(order_by), (int, float, str)) else str(d.get(order_by))), reverse=bool(desc))
        else:
            rows.sort(key=lambda d: d["id"])
        return rows[:limit]

    @endpoint("delete", S.obj({"collection": S.s_str(maxLength=31), "id": S.s_str(maxLength=64)}, ["collection", "id"]))
    def ep_delete(self, ctx, collection, id):
        """Delete a document."""
        k = self._key(self._ns(ctx), collection, id)
        if ctx.state.has("app_docs", k):
            ctx.emit("db.delete", {"id": k}, vis="internal")
        return {"ok": True}

    @applier("db.put")
    def ap_put(self, ev):
        p = ev.payload
        self.world.state.put("app_docs", p["id"], {"ns": p["ns"], "nc": f"{p['ns']}|{p['coll']}", "doc_id": p["doc_id"],
                                                   "doc": p["doc"], "t": ev.t})

    @applier("db.delete")
    def ap_delete(self, ev):
        self.world.state.delete("app_docs", ev.payload["id"])
