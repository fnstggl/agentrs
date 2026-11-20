"""Read-only researcher dashboard (FastAPI): HTML pages plus JSON endpoints.

    app = make_app(store_url="sqlite:///crucible-data/crucible.db", worlds=["w1"],
                   experiments=["results.json"], trajectories=["traj.jsonl"])
    uvicorn.run(app, host="127.0.0.1", port=8050)          # or: crucible dashboard w1

Pages per world (`/w/<id>/...`): overview (time, metrics, money by sector, customers per host), actors,
organisations, economy (balance distribution, transactions with filters), communication (counts by type and
visibility; private payloads are never shown unless the app is created with ``operator=True``), services and
infrastructure, the event stream with filters, agent trajectories (loaded files or Runner trajectories passed in),
fork lineage and snapshots, evaluation (invariants, metrics, emergence), trace / explain and causal chains, and
`/w/<id>/metrics` in Prometheus text format. `/experiments` shows experiment, evaluation and league JSON files
(tables and learning / reward curves). JSON mirrors live under `/api/...`.

The dashboard never mutates a world: it only serves GET routes, loads worlds itself with `World.load` (no writer
lease) behind a store wrapper that refuses writes, reads state directly, and follows new events appended by a
writer process (incremental catch-up with hash-chain verification).
"""
from __future__ import annotations

import html
import json
import math
import os
import threading
import time
from collections import Counter, defaultdict
from typing import Any

from fastapi import Query, Request   # module level: route annotations are resolved from module globals
from markupsafe import Markup

from ..analytics import DAY, NOISE_TYPES, accounts_of, has_table, hosts_of, money, world_events

COMM_HOSTS = {"mail", "docs", "cal", "support", "issues", "forum"}
COMM_PREFIXES = ("mail.", "docs.", "cal.", "support.msg", "issues.", "forum.", "contracts.note")
PALETTE = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)"]
TEMPLATES = os.path.join(os.path.dirname(__file__), "templates")


# ====================================================================== read-only store
class ReadOnlyStore:
    """Proxy that exposes a store's read API and refuses every write."""
    _READ = {"world_info", "list_worlds", "read_events", "last_seq", "get_snapshot", "list_snapshots",
             "latest_snapshot", "lineage", "get_blob", "kind"}

    def __init__(self, store):
        self._s = store

    def __getattr__(self, name):
        if name in self._READ:
            return getattr(self._s, name)
        raise PermissionError(f"the dashboard is read-only (store.{name} refused)")

    def close(self):
        pass


# ====================================================================== privacy
def is_comm(ev_type: str, payload: Any) -> bool:
    if ev_type.startswith(COMM_PREFIXES):
        return True
    if ev_type in ("cmd", "cmd.rejected") and isinstance(payload, dict):
        return str(payload.get("tool", "")).split(".", 1)[0] in COMM_HOSTS
    return False


def is_private(vis) -> bool:
    return isinstance(vis, (tuple, list))


def redact(ev_type: str, payload: Any, vis, operator: bool) -> tuple[Any, bool]:
    """(payload to show, redacted?) - private communication payloads (and recorded program RPCs, which may carry
    messages) are replaced by their shape unless the operator flag is set."""
    if operator:
        return payload, False
    if ev_type == "ext" or (is_comm(ev_type, payload) and is_private(vis)):
        keys = sorted(payload) if isinstance(payload, dict) else []
        shape = {"redacted": True, "keys": keys}
        if isinstance(payload, dict):
            if isinstance(payload.get("to"), list):
                shape["recipients"] = len(payload["to"])
            if "tool" in payload:
                shape["tool"] = payload["tool"]
        return shape, True
    return payload, False


# ====================================================================== charts (server-side SVG)
def _e(s) -> str:
    return html.escape(str(s), quote=True)


def svg_bars(items: list[tuple[str, float]], width: int = 720, bar_h: int = 18, fmt=None) -> Markup:
    items = [(k, float(v)) for k, v in items if v is not None]
    if not items:
        return Markup('<p class="muted">no data</p>')
    fmt = fmt or (lambda v: f"{v:,.0f}")
    lab_w = min(260, 8 + 7 * max(len(str(k)) for k, _ in items))
    vmax = max(abs(v) for _, v in items) or 1.0
    h = bar_h * len(items) + 8
    plot = width - lab_w - 90
    parts = [f'<svg viewBox="0 0 {width} {h}" width="{width}" height="{h}" role="img">']
    for i, (k, v) in enumerate(items):
        y = 4 + i * bar_h
        w = max(1.0, plot * abs(v) / vmax)
        color = PALETTE[0] if v >= 0 else "var(--bad)"
        parts.append(f'<text x="{lab_w - 6}" y="{y + bar_h - 6}" text-anchor="end">{_e(k)}</text>')
        parts.append(f'<rect x="{lab_w}" y="{y + 2}" width="{w:.1f}" height="{bar_h - 5}" rx="2" fill="{color}">'
                     f'<title>{_e(k)}: {_e(fmt(v))}</title></rect>')
        parts.append(f'<text x="{lab_w + w + 4:.1f}" y="{y + bar_h - 6}">{_e(fmt(v))}</text>')
    parts.append("</svg>")
    return Markup("".join(parts))


def svg_lines(series: dict[str, list], width: int = 720, height: int = 240, xlabel: str = "day",
              ylabel: str = "", max_series: int = 6) -> Markup:
    """Line chart; series = {name: [[x, y], ...]}. Up to `max_series` series (largest final values first)."""
    ser = {k: [(float(x), float(y)) for x, y in v if y is not None] for k, v in series.items() if v}
    ser = {k: v for k, v in ser.items() if v}
    if not ser:
        return Markup('<p class="muted">no data</p>')
    ser = dict(sorted(ser.items(), key=lambda kv: -abs(kv[1][-1][1]))[:max_series])
    xs = [x for v in ser.values() for x, _ in v]
    ys = [y for v in ser.values() for _, y in v]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(0.0, min(ys)), max(ys)
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y1 = y0 + 1
    L, R, T, B = 56, 150, 24, 28
    pw, ph = width - L - R, height - T - B

    def px(x):
        return L + pw * (x - x0) / (x1 - x0)

    def py(y):
        return T + ph * (1 - (y - y0) / (y1 - y0))
    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">',
             f'<line x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}" stroke="var(--line)"/>',
             f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T + ph}" stroke="var(--line)"/>']
    for f in (0.0, 0.5, 1.0):
        yv = y0 + f * (y1 - y0)
        parts.append(f'<text x="{L - 4}" y="{py(yv) + 4:.1f}" text-anchor="end">{_e(_short_num(yv))}</text>')
        xv = x0 + f * (x1 - x0)
        parts.append(f'<text x="{px(xv):.1f}" y="{T + ph + 16}" text-anchor="middle">{_e(_short_num(xv))}</text>')
    parts.append(f'<text x="{L + pw / 2:.1f}" y="{height - 2}" text-anchor="middle">{_e(xlabel)}</text>')
    if ylabel:
        parts.append(f'<text x="4" y="12">{_e(ylabel)}</text>')
    for i, (k, v) in enumerate(ser.items()):
        c = PALETTE[i % len(PALETTE)]
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in v)
        parts.append(f'<polyline fill="none" stroke="{c}" stroke-width="2" points="{pts}"><title>{_e(k)}</title>'
                     f'</polyline>')
        parts.append(f'<rect x="{L + pw + 10}" y="{T + 4 + 16 * i}" width="10" height="10" fill="{c}"/>')
        parts.append(f'<text x="{L + pw + 24}" y="{T + 13 + 16 * i}">{_e(str(k)[:22])}</text>')
    parts.append("</svg>")
    return Markup("".join(parts))


def _short_num(v: float) -> str:
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.1f}M"
    if a >= 1e4:
        return f"{v / 1e3:.0f}k"
    if a >= 100 or v == int(v):
        return f"{v:.0f}"
    return f"{v:.2f}"


# ====================================================================== world cache
class _Worlds:
    def __init__(self, store, live: dict, allowed: set | None, refresh_s: float):
        self.store = ReadOnlyStore(store) if store is not None else None
        self.live = dict(live)
        self.allowed = allowed
        self.refresh_s = refresh_s
        self.loaded: dict[str, Any] = {}
        self.checked: dict[str, float] = {}
        self.lock = threading.RLock()

    def ids(self) -> list[str]:
        out = list(self.live)
        if self.store is not None:
            out += [w["id"] for w in self.store.list_worlds() if w["id"] not in self.live]
        if self.allowed is not None:
            out = [w for w in out if w in self.allowed]
        return out

    def get(self, wid: str):
        from fastapi import HTTPException
        if self.allowed is not None and wid not in self.allowed:
            raise HTTPException(404, f"world {wid} is not exposed by this dashboard")
        if wid in self.live:
            return self.live[wid]
        if self.store is None:
            raise HTTPException(404, f"no world {wid}")
        with self.lock:
            w = self.loaded.get(wid)
            if w is None:
                if self.store.world_info(wid) is None:
                    raise HTTPException(404, f"no world {wid}")
                import crucible
                crucible.load_all_modules()
                from ..core.kernel import World
                w = World.load(self.store, wid)
                w.store = self.store
                self.loaded[wid] = w
                self.checked[wid] = time.monotonic()
            elif time.monotonic() - self.checked.get(wid, 0) >= self.refresh_s:
                self.checked[wid] = time.monotonic()
                self._catch_up(wid, w)
            return self.loaded[wid]

    def _catch_up(self, wid, w) -> None:
        """Apply events a writer appended since we loaded (hash chain verified); reload on any problem."""
        from ..core.canonical import canonical_bytes, chain_hash
        from ..core.kernel import Event, World
        last = self.store.last_seq(wid)
        if last <= w.seq:
            return
        try:
            h = w.last_hash
            for d in self.store.read_events(wid, w.seq, last):
                ev = Event.from_dict(d)
                if ev.seq != w.seq + 1:
                    raise ValueError("gap")
                h = chain_hash(h, canonical_bytes(ev.body()))
                if h != ev.hash:
                    raise ValueError("hash chain broken")
                w.state.begin()
                w._apply(ev)
                w.state.commit()
                w.seq, w.last_hash = ev.seq, ev.hash
                w.state.meta["seq"], w.state.meta["hash"] = w.seq, w.last_hash
            w.log_base = w.seq
            w._rebuild_heap()
            for mod in w.modules.values():
                mod.on_load(w)
        except Exception:  # noqa: BLE001 - fall back to a full reload
            fresh = World.load(self.store, wid)
            fresh.store = self.store
            self.loaded[wid] = fresh
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass

    def close(self):
        for w in self.loaded.values():
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass


# ====================================================================== data sources
def _load_json(path: str) -> dict | None:
    try:
        with open(path) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {"data": d}
    except (OSError, ValueError):
        return None


def _load_traj(src) -> list[dict]:
    from ..training.trajectories import load_npz, read_jsonl, step_records
    if isinstance(src, str):
        try:
            return load_npz(src) if src.endswith(".npz") else read_jsonl(src)
        except (OSError, ValueError):
            return []
    return step_records(src)


# ====================================================================== the app
def make_app(store=None, *, store_url: str | None = None, worlds=None, world=None, experiments=(),
             trajectories=(), operator: bool = False, refresh_s: float = 2.0, root: str = ""):
    """Build the dashboard. `store` (or `store_url`) gives persisted worlds; `worlds` restricts which ids are
    exposed; `world` passes live `World` objects (one, a list, or {id: World}) e.g. from a notebook or test;
    `experiments` are experiment / evaluation / league JSON paths (or dicts); `trajectories` are JSONL / NPZ
    paths, Runner trajectory dicts, or step-record lists. `operator=True` shows private communication payloads."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
    from fastapi.templating import Jinja2Templates

    if store is None and store_url:
        from ..storage.postgres import open_store
        store = open_store(store_url)
    live = {}
    if world is not None:
        ws = world if isinstance(world, dict) else {w.world_id: w for w in (world if isinstance(world, (list, tuple))
                                                                             else [world])}
        live.update(ws)
    W = _Worlds(store, live, set(worlds) if worlds else None, refresh_s)
    exps: list[tuple[str, dict]] = []
    for i, e in enumerate(experiments or ()):
        d = e if isinstance(e, dict) else _load_json(e)
        if d is not None:
            exps.append((os.path.basename(e) if isinstance(e, str) else f"experiment-{i}", d))
    traj_records: list[dict] = []
    for src in trajectories or ():
        traj_records += _load_traj(src)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app):
        yield
        W.close()

    app = FastAPI(title="Crucible dashboard", docs_url=None, redoc_url=None, lifespan=lifespan)
    tpl = Jinja2Templates(directory=TEMPLATES)
    app.state.worlds = W

    def page(request: Request, title: str, sections: list, wid: str | None = None, subtitle: str = ""):
        return tpl.TemplateResponse(request, "page.html", {"title": title, "subtitle": subtitle, "sections": sections,
                                                           "wid": wid, "root": root, "operator": operator})

    def link(text, href):
        return {"text": text, "href": f"{root}{href}"}

    def ent(wid, e):
        return link(e, f"/w/{wid}/trace/{e}") if e else None

    @app.middleware("http")
    async def only_get(request: Request, call_next):
        if request.method not in ("GET", "HEAD"):
            return JSONResponse({"error": "read-only dashboard"}, status_code=405)
        return await call_next(request)

    # ------------------------------------------------------------------ data builders (shared HTML / JSON)
    def overview(w) -> dict:
        from ..analytics.metrics_export import world_metrics
        m = world_metrics(w)
        info = W.store.world_info(w.world_id) if W.store is not None else None
        return {"metrics": m, "parent": (info or {}).get("parent"), "status": (info or {}).get("status", "live")}

    def actors(w) -> list[dict]:
        st = w.state
        out = []
        if not has_table(w, "persons"):
            return out
        for pid, p in st.scan("persons"):
            pr = st.tables["principals"].get(pid, {})
            orgs = sorted(m["org"] for m in st.find("members", "person", pid)) if has_table(w, "members") else []
            out.append({"id": pid, "kind": p["kind"], "status": pr.get("status"), "actions": pr.get("actions", 0),
                        "time_used_s": pr.get("time_used", 0), "cash": _cash(w, pid), "orgs": orgs,
                        "role": (p.get("info") or {}).get("role"), "created": p.get("created")})
        return out

    def orgs(w) -> list[dict]:
        from ..goals.metrics import m_independent_revenue
        st = w.state
        out = []
        if not has_table(w, "orgs"):
            return out
        for oid, o in st.scan("orgs"):
            comp = st.get("companies", oid) if has_table(w, "companies") else None
            out.append({"id": oid, "name": o["name"], "status": o["status"], "kind": o["kind"],
                        "members": len(st.find("members", "org", oid)) if has_table(w, "members") else 0,
                        "cash": _cash(w, oid), "revenue_30d": int(m_independent_revenue(w, oid, 30 * DAY))
                        if has_table(w, "acct_txns") else 0, "hosts": hosts_of(w, oid),
                        "registered": bool(comp), "distress": bool(comp and comp.get("distress_since") is not None),
                        "sector": (comp or {}).get("sector")})
        return out

    def economy(w, kind=None, owner=None, limit=100) -> dict:
        st = w.state
        bal = [x["balance"] for _, x in st.scan("accounts") if x["owner"] != "sys"] if has_table(w, "accounts") else []
        bins = defaultdict(int)
        for b in bal:
            if b <= 0:
                bins["<= $0"] += 1
            else:
                e = int(math.floor(math.log10(max(b, 1))))
                bins[f"${10 ** e / 100:,.2f}-${10 ** (e + 1) / 100:,.2f}"] += 1
        order = sorted(bins, key=lambda k: (k != "<= $0", float(k.split("-")[0].strip("$<= ").replace(",", "") or 0)))
        txs = []
        if has_table(w, "txns"):
            accts = set(accounts_of(w, owner)) if owner else None
            for tid in reversed(st.ids("txns")):
                tx = st.tables["txns"][tid]
                if kind and tx["kind"] != kind:
                    continue
                if accts is not None and not any(a in accts for a, _ in tx["postings"]):
                    continue
                txs.append({"id": tid, "t": tx["t"], "kind": tx["kind"], "amount": sum(d for _, d in tx["postings"] if d > 0),
                            "postings": tx["postings"], "memo": tx["memo"], "by": tx["by"]})
                if len(txs) >= limit:
                    break
        gini = _gini([max(0, b) for b in bal])
        return {"balance_histogram": [(k, bins[k]) for k in order], "n_accounts": len(bal), "gini": gini,
                "transactions": txs}

    def comms(w, since=0) -> dict:
        counts: dict[tuple, int] = Counter()
        per_day: dict[int, Counter] = defaultdict(Counter)
        public = []
        for ev in world_events(w, since):
            if not is_comm(ev.type, ev.payload) or ev.type in ("cmd", "cmd.rejected"):
                continue
            v = "public" if ev.vis == "public" else ("internal" if ev.vis == "internal" else "private")
            counts[(ev.type, v)] += 1
            per_day[ev.t // DAY][v] += 1
            if ev.vis == "public" and ev.type == "forum.post" and len(public) < 200:
                public.append({"seq": ev.seq, "t": ev.t, "author": ev.payload.get("author"),
                               "board": ev.payload.get("board"), "title": ev.payload.get("title")})
            elif operator and v == "private" and len(public) < 200:
                public.append({"seq": ev.seq, "t": ev.t, "author": ev.actor, "board": ev.type,
                               "title": json.dumps(ev.payload, default=str)[:200]})
        return {"counts": [{"type": t, "visibility": v, "n": n} for (t, v), n in sorted(counts.items())],
                "per_day": {d: dict(c) for d, c in sorted(per_day.items())}, "posts": public[-50:],
                "private_payloads_shown": operator}

    def services(w) -> dict:
        st = w.state
        cons = w.modules.get("consumers")
        shares = cons.market_shares(w) if cons is not None and "c_prov" in st.arrays else {}
        out: dict = {"providers": [], "instances": [], "deployments": [], "days": {}}
        if has_table(w, "providers"):
            out["providers"] = [{"id": k, "name": p["name"], "cores": p["cores"], "price": p["price"],
                                 "reliability": p["reliability"], "status": p["status"],
                                 "outages": len(p["outages"]), "in_outage": p["outage_start"] <= w.now < p["outage_until"]}
                                for k, p in st.scan("providers")]
        if has_table(w, "instances"):
            out["instances"] = [{"id": k, "owner": i["owner"], "provider": i["provider"], "cores": i["cores"],
                                 "status": i["status"], "price": i["price"]} for k, i in st.scan("instances")]
        if has_table(w, "deployments"):
            out["deployments"] = [{"id": k, "host": d["host"], "owner": d["owner"], "status": d["status"],
                                   "repo": d["repo"], "trusted": d["trusted"], "requests": d["requests"],
                                   "customers": shares.get(d["host"], 0), "errors": list(d.get("errors") or [])[-3:]}
                                  for k, d in st.scan("deployments")]
        if has_table(w, "svc_days"):
            d0 = w.now // DAY - 6
            for k, r in st.scan("svc_days"):
                h, d = k.rsplit("|", 1)
                if int(d) >= d0:
                    out["days"].setdefault(h, []).append({"day": int(d), **{f: r.get(f, 0) for f in
                                                          ("requests", "ok", "errors", "unavailable", "overload",
                                                           "revenue_cents")}})
        return out

    def events(w, type_=None, actor=None, entity=None, since=None, limit=100, noise=False) -> list[dict]:
        from ..analytics.trace import _mentions, _pattern, describe_event, entity_aliases
        lo = int(since) if since not in (None, "") else max(0, w.seq - 20_000)
        aliases = entity_aliases(w, entity) if entity else None
        pat = _pattern(aliases) if aliases else None
        rows = []
        for ev in world_events(w, lo):
            if type_ and not ev.type.startswith(type_):
                continue
            if not type_ and not noise and ev.type in NOISE_TYPES:
                continue
            if actor and ev.actor != actor:
                continue
            if aliases and not _mentions(ev, pat, aliases):
                continue
            payload, red = redact(ev.type, ev.payload, ev.vis, operator)
            rows.append({"seq": ev.seq, "t": ev.t, "type": ev.type, "actor": ev.actor, "cmd": ev.cmd,
                         "vis": "public" if ev.vis == "public" else ("internal" if ev.vis == "internal"
                                                                     else f"private({len(ev.vis)})"),
                         "text": "[private communication]" if red else describe_event(ev), "payload": payload,
                         "redacted": red})
        return rows[-int(limit):]

    def trajs_for(wid) -> list[dict]:
        return [r for r in traj_records if r.get("world_id") in (wid, None)]

    def trajectories_summary(wid) -> dict:
        by: dict[str, list] = defaultdict(list)
        for r in trajs_for(wid):
            by[r["actor"]].append(r)
        out = {}
        for actor, rs in sorted(by.items()):
            rs.sort(key=lambda r: (r["t"], r["step"]))
            cum, curve = 0.0, []
            for r in rs:
                cum += float(r.get("reward") or 0.0)
                curve.append([round(r["t"] / DAY, 4), round(cum, 6)])
            tools = Counter(r.get("tool") or "(wait)" for r in rs)
            out[actor] = {"policy": rs[-1].get("policy"), "steps": len(rs),
                          "actions": sum(1 for r in rs if r.get("action")),
                          "ok": sum(1 for r in rs if r.get("ok")), "rejected": sum(1 for r in rs if r.get("ok") is False),
                          "reward": round(cum, 6), "tools": dict(tools.most_common(15)), "curve": curve,
                          "last": [{"step": r["step"], "t": r["t"], "tool": r.get("tool"), "ok": r.get("ok"),
                                    "reward": r.get("reward")} for r in rs[-15:]]}
        return out

    def lineage(w) -> dict:
        if W.store is None:
            return {"lineage": [], "children": [], "snapshots": []}
        lin = W.store.lineage(w.world_id)
        children = [x for x in W.store.list_worlds() if (x.get("parent") or {}).get("world") == w.world_id]
        snaps = W.store.list_snapshots(w.world_id)
        return {"lineage": [{"id": x["id"], "status": x["status"], "parent": x.get("parent")} for x in lin],
                "children": [{"id": x["id"], "parent": x["parent"], "status": x["status"]} for x in children],
                "snapshots": [{k: v for k, v in s.items() if k != "blob"} for s in snaps]}

    def evaluation(w, emergence: bool = False) -> dict:
        from ..analytics.metrics_export import world_metrics
        out = {"invariant_violations": w.check_invariants(), "digest": w.digest(), "seq": w.seq,
               "metrics": world_metrics(w)}
        if emergence:
            from ..evaluation.emergence import emergence_report
            out["emergence"] = emergence_report(w)
        return out

    def trace_data(w, entity, since=0) -> dict:
        from ..analytics.trace import explain, timeline
        ex = explain(w, entity, since=since)
        tl = timeline(w, entity, since=since, limit=200, payload=True)
        for r in tl:
            p, red = redact(r["type"], r.pop("payload"), tuple(r["vis"]) if isinstance(r["vis"], list) else r["vis"],
                            operator)
            if red:
                r["text"] = "[private communication]"
        ex["timeline"] = tl
        return ex

    # ------------------------------------------------------------------ HTML
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        rows = []
        for wid in W.ids():
            info = W.store.world_info(wid) if W.store is not None and wid not in W.live else None
            par = (info or {}).get("parent")
            last = W.store.last_seq(wid) if info else W.live[wid].seq
            fam = info["spec"]["family"] if info else W.live[wid].spec.family
            seed = info["spec"]["seed"] if info else W.live[wid].spec.seed
            rows.append([link(wid, f"/w/{wid}"), fam, seed, (info or {}).get("status", "live"), last,
                         f"{par['world']}@{par['seq']}" if par else None])
        secs = [{"kind": "table", "title": "Worlds", "columns": ["world", "family", "seed", "status", "events",
                                                                 "parent"], "rows": rows}]
        if exps:
            secs.append({"kind": "links", "title": "Experiments",
                         "items": [(n, f"{root}/experiments/{i}") for i, (n, _) in enumerate(exps)]})
        return page(request, "Worlds", secs)

    @app.get("/w/{wid}", response_class=HTMLResponse)
    def world_page(request: Request, wid: str):
        w = W.get(wid)
        o = overview(w)
        m = o["metrics"]
        cards = [("simulated time", f"day {m['day']:.2f}"), ("events (seq)", m["seq"]),
                 ("principals", sum(m.get("principals", {}).values())), ("organisations", sum(m.get("orgs", {}).values())),
                 ("money in circulation", money(m.get("money_in_circulation"))), ("txns / day", m.get("txns_per_day")),
                 ("services live", m.get("services_live")), ("requests ok (24h)", m.get("requests_ok")),
                 ("requests failed (24h)", m.get("requests_failed")), ("open tickets", m.get("tickets_open")),
                 ("consumers unserved", m.get("consumers_unserved")), ("status", o["status"])]
        secs = [{"kind": "cards", "items": cards},
                {"kind": "svg", "title": "Money by sector", "svg": svg_bars(list(m.get("money_by_sector", {}).items()),
                                                                            fmt=lambda v: money(v))},
                {"kind": "svg", "title": "Customers per host", "svg": svg_bars(
                    sorted(m.get("customers_by_host", {}).items(), key=lambda kv: -kv[1]))},
                {"kind": "table", "title": "Kernel", "columns": ["stat", "value"],
                 "rows": [[k, v] for k, v in sorted(m.get("kernel", {}).items())]},
                {"kind": "table", "title": "Contracts / claims / listings / tickets", "columns": ["table", "status", "n"],
                 "rows": [[t, k, v] for t in ("contracts", "claims", "listings", "tickets")
                          for k, v in sorted(m.get(t, {}).items())]}]
        sub = f"{m['family']} seed {m['seed']} · spec {m['spec_hash']}" + \
              (f" · forked from {o['parent']['world']}@{o['parent']['seq']}" if o["parent"] else "")
        return page(request, f"World {wid}", secs, wid, sub)

    @app.get("/w/{wid}/actors", response_class=HTMLResponse)
    def actors_page(request: Request, wid: str, kind: str = ""):
        w = W.get(wid)
        rows = [[ent(wid, a["id"]), a["kind"], a["role"], a["status"], a["actions"], money(a["cash"]),
                 ", ".join(a["orgs"])] for a in actors(w) if not kind or a["kind"] == kind]
        kinds = Counter(p["kind"] for _, p in w.state.scan("principals"))
        return page(request, "Actors", [{"kind": "cards", "items": sorted(kinds.items())},
                                        {"kind": "form", "action": f"{root}/w/{wid}/actors",
                                         "fields": [{"name": "kind", "label": "kind", "value": kind,
                                                     "options": ["", "agent", "npc", "human"]}]},
                                        {"kind": "table", "columns": ["principal", "kind", "role", "status", "actions",
                                                                      "cash", "organisations"], "rows": rows}], wid)

    @app.get("/w/{wid}/orgs", response_class=HTMLResponse)
    def orgs_page(request: Request, wid: str):
        w = W.get(wid)
        rows = [[ent(wid, o["id"]), o["name"], {"text": o["status"], "cls": "sev3" if o["status"] != "active" else ""},
                 "distress" if o["distress"] else "", o["members"], money(o["cash"]), money(o["revenue_30d"]),
                 ", ".join(o["hosts"]), o["sector"]] for o in orgs(w)]
        return page(request, "Organisations", [{"kind": "table", "columns": [
            "org", "name", "status", "", "members", "cash", "independent revenue 30d", "hosts", "sector"],
            "rows": rows}], wid)

    @app.get("/w/{wid}/economy", response_class=HTMLResponse)
    def economy_page(request: Request, wid: str, kind: str = "", owner: str = "", limit: int = Query(100, le=1000)):
        w = W.get(wid)
        e = economy(w, kind or None, owner or None, limit)
        from ..analytics.metrics_export import world_metrics
        m = world_metrics(w)
        kinds = [""] + sorted({t["kind"] for _, t in w.state.scan("txns")}) if has_table(w, "txns") else [""]
        secs = [{"kind": "cards", "items": [("accounts (non-system)", e["n_accounts"]), ("Gini of balances", e["gini"]),
                                            ("txns / day", m.get("txns_per_day")),
                                            ("volume / day", money(m.get("txn_volume_per_day")))]},
                {"kind": "svg", "title": "Balance distribution (accounts per balance band)",
                 "svg": svg_bars(e["balance_histogram"])},
                {"kind": "svg", "title": "Transactions in the last 24h by kind",
                 "svg": svg_bars([(k, v["n"]) for k, v in m.get("txns_by_kind", {}).items()])},
                {"kind": "form", "action": f"{root}/w/{wid}/economy", "title": "Transactions",
                 "fields": [{"name": "kind", "label": "kind", "value": kind, "options": kinds},
                            {"name": "owner", "label": "owner (principal)", "value": owner},
                            {"name": "limit", "label": "limit", "value": limit, "size": 5}]},
                {"kind": "table", "columns": ["txn", "time", "kind", "amount", "postings", "memo", "by"],
                 "rows": [[t["id"], _fmt_t(t["t"]), t["kind"], money(t["amount"]),
                           "; ".join(f"{a} {money(d)}" for a, d in t["postings"]), t["memo"], t["by"]]
                          for t in e["transactions"]]}]
        return page(request, "Economy", secs, wid)

    @app.get("/w/{wid}/comms", response_class=HTMLResponse)
    def comms_page(request: Request, wid: str):
        w = W.get(wid)
        c = comms(w)
        note = ("Operator view: private payloads are shown." if operator else
                "Private communication is shown as counts only (start the dashboard with --operator to inspect "
                "payloads).")
        secs = [{"kind": "table", "title": "Communication events by type and visibility", "note": note,
                 "columns": ["type", "visibility", "events"], "rows": [[x["type"], x["visibility"], x["n"]]
                                                                      for x in c["counts"]]},
                {"kind": "svg", "title": "Communication per day",
                 "svg": svg_lines({v: [[d, c["per_day"][d].get(v, 0)] for d in c["per_day"]]
                                   for v in ("public", "private", "internal")}, ylabel="events")},
                {"kind": "table", "title": "Public posts" + (" and private messages" if operator else ""),
                 "columns": ["seq", "time", "author", "board/type", "title"],
                 "rows": [[p["seq"], _fmt_t(p["t"]), p["author"], p["board"], p["title"]] for p in c["posts"]]}]
        return page(request, "Communication", secs, wid)

    @app.get("/w/{wid}/services", response_class=HTMLResponse)
    def services_page(request: Request, wid: str):
        w = W.get(wid)
        s = services(w)
        secs = [{"kind": "table", "title": "Deployments", "columns": ["host", "owner", "status", "repo", "trusted",
                                                                       "requests", "customers", "recent errors"],
                 "rows": [[ent(wid, d["host"]), ent(wid, d["owner"]), d["status"], d["repo"], d["trusted"],
                           d["requests"], d["customers"], "; ".join(d["errors"])[:120]] for d in s["deployments"]]},
                {"kind": "svg", "title": "Requests ok per day (last 7 days)",
                 "svg": svg_lines({h: [[r["day"], r["ok"]] for r in sorted(rs, key=lambda r: r["day"])]
                                   for h, rs in s["days"].items()}, ylabel="ok requests")},
                {"kind": "table", "title": "Providers", "columns": ["provider", "name", "cores", "price c/core-h",
                                                                     "reliability", "status", "outages", "in outage"],
                 "rows": [[p["id"], p["name"], p["cores"], p["price"], p["reliability"], p["status"], p["outages"],
                           p["in_outage"]] for p in s["providers"]]},
                {"kind": "table", "title": "Instances", "columns": ["instance", "owner", "provider", "cores", "status",
                                                                     "price"],
                 "rows": [[i["id"], ent(wid, i["owner"]), i["provider"], i["cores"], i["status"], i["price"]]
                          for i in s["instances"]]}]
        return page(request, "Services & infrastructure", secs, wid)

    @app.get("/w/{wid}/events", response_class=HTMLResponse)
    def events_page(request: Request, wid: str, type: str = "", actor: str = "", entity: str = "", since: str = "",
                    limit: int = Query(100, le=2000), noise: int = 0):
        w = W.get(wid)
        rows = events(w, type or None, actor or None, entity or None, since or None, limit, bool(noise))
        secs = [{"kind": "form", "action": f"{root}/w/{wid}/events",
                 "fields": [{"name": "type", "label": "type prefix", "value": type},
                            {"name": "actor", "label": "actor", "value": actor},
                            {"name": "entity", "label": "mentions entity", "value": entity},
                            {"name": "since", "label": "since seq", "value": since, "size": 8},
                            {"name": "limit", "label": "limit", "value": limit, "size": 5},
                            {"name": "noise", "label": "bookkeeping", "value": str(noise), "options": ["0", "1"]}]},
                {"kind": "table", "columns": ["seq", "time", "type", "actor", "visibility", "event"],
                 "rows": [[link(r["seq"], f"/w/{wid}/chain/{r['seq']}"), _fmt_t(r["t"]), r["type"], r["actor"],
                           r["vis"], {"redacted": r["text"]} if r["redacted"] else r["text"]] for r in rows]}]
        return page(request, "Event stream", secs, wid, f"{len(rows)} events (newest last)")

    @app.get("/w/{wid}/trajectories", response_class=HTMLResponse)
    def trajectories_page(request: Request, wid: str):
        W.get(wid)
        t = trajectories_summary(wid)
        secs = [{"kind": "table", "title": "Agents", "columns": ["actor", "policy", "steps", "actions", "ok", "rejected",
                                                                  "return"],
                 "rows": [[ent(wid, a), v["policy"], v["steps"], v["actions"], v["ok"], v["rejected"], v["reward"]]
                          for a, v in t.items()]},
                {"kind": "svg", "title": "Cumulative reward", "svg": svg_lines({a: v["curve"] for a, v in t.items()},
                                                                               ylabel="return")}]
        for a, v in t.items():
            secs.append({"kind": "svg", "title": f"{a}: tools used", "svg": svg_bars(list(v["tools"].items()))})
            secs.append({"kind": "table", "columns": ["step", "time", "tool", "ok", "reward"],
                         "rows": [[x["step"], _fmt_t(x["t"]), x["tool"], x["ok"], x["reward"]] for x in v["last"]]})
        if not t:
            secs.append({"kind": "pre", "text": "No trajectories loaded for this world (start the dashboard with "
                                                "--trajectories FILE.jsonl, or pass Runner trajectories to make_app)."})
        return page(request, "Agent trajectories", secs, wid)

    @app.get("/w/{wid}/lineage", response_class=HTMLResponse)
    def lineage_page(request: Request, wid: str):
        w = W.get(wid)
        lin = lineage(w)
        secs = [{"kind": "table", "title": "Ancestry (this world first)", "columns": ["world", "status", "forked at"],
                 "rows": [[link(x["id"], f"/w/{x['id']}"), x["status"],
                           f"{x['parent']['world']}@{x['parent']['seq']}" if x["parent"] else "genesis"]
                          for x in lin["lineage"]]},
                {"kind": "table", "title": "Forks of this world", "columns": ["world", "at seq", "status"],
                 "rows": [[link(x["id"], f"/w/{x['id']}"), x["parent"]["seq"], x["status"]] for x in lin["children"]]},
                {"kind": "table", "title": "Snapshots", "columns": ["snapshot", "seq", "bytes", "hash"],
                 "rows": [[s.get("id"), s.get("seq"), s.get("bytes"), str(s.get("hash"))[:16]] for s in lin["snapshots"]]}]
        return page(request, "Lineage", secs, wid)

    @app.get("/w/{wid}/evaluation", response_class=HTMLResponse)
    def evaluation_page(request: Request, wid: str, emergence: int = 0):
        w = W.get(wid)
        ev = evaluation(w, bool(emergence))
        secs = [{"kind": "cards", "items": [("invariants", "ok" if not ev["invariant_violations"] else
                                             {"text": f"{len(ev['invariant_violations'])} violated", "cls": "sev3"}),
                                            ("state digest", ev["digest"][:16]), ("seq", ev["seq"])]}]
        if ev["invariant_violations"]:
            secs.append({"kind": "pre", "text": "\n".join(ev["invariant_violations"])})
        if "emergence" in ev:
            e = ev["emergence"]
            sd = e["strategy_diversity"]
            secs += [{"kind": "cards", "title": "Emergence", "items": [
                ("agent hosts reused", f"{e['tool_reuse']['n_reused']}/{e['tool_reuse']['n_agent_hosts']}"),
                ("strategy clusters", sd["n_clusters"]), ("mean pairwise JS", sd["mean_pairwise_js"]),
                ("dependency edges", e["dependencies"]["n_edges"]),
                ("longest service chain", e["dependencies"]["max_service_chain"]),
                ("specialists", e["specialization"]["summary"].get("specialists"))]},
                {"kind": "svg", "title": "Market concentration (HHI) per need",
                 "svg": svg_lines({n: [[x["t"] / DAY, x["hhi"]] for x in s] for n, s in
                                   e["concentration"]["series"].items()}, ylabel="HHI")},
                {"kind": "svg", "title": "Diffusion: customers per host",
                 "svg": svg_lines({h: [[t / DAY, v] for t, v in c] for h, c in e["diffusion"]["curves"].items()},
                                  ylabel="customers")}]
        else:
            secs.append({"kind": "links", "items": [("compute emergence metrics (log scan)",
                                                     f"{root}/w/{wid}/evaluation?emergence=1")]})
        secs.append({"kind": "pre", "title": "Metrics snapshot (JSON)",
                     "text": json.dumps(ev["metrics"], indent=1, sort_keys=True, default=str)})
        return page(request, "Evaluation", secs, wid)

    @app.get("/w/{wid}/trace", response_class=HTMLResponse)
    def trace_form(request: Request, wid: str, entity: str = ""):
        w = W.get(wid)
        if entity:
            return trace_page(request, wid, entity)
        cand = [o["id"] for o in orgs(w)] + [d["host"] for d in services(w)["deployments"]]
        return page(request, "Trace / explain", [
            {"kind": "form", "action": f"{root}/w/{wid}/trace", "fields": [{"name": "entity", "label": "entity "
                                                                            "(org, host, account, person)", "value": "",
                                                                            "size": 30}]},
            {"kind": "links", "items": [(c, f"{root}/w/{wid}/trace/{c}") for c in cand]}], wid)

    @app.get("/w/{wid}/trace/{entity:path}", response_class=HTMLResponse)
    def trace_page(request: Request, wid: str, entity: str, since: int = 0):
        w = W.get(wid)
        ex = trace_data(w, entity, since)
        secs = [{"kind": "pre", "title": "Explanation", "text": ex["text"]},
                {"kind": "table", "title": "Findings", "columns": ["day", "severity", "kind", "finding"],
                 "rows": [[f["day"], {"text": "!" * f["severity"], "cls": f"sev{f['severity']}"}, f["kind"], f["text"]]
                          for f in ex["findings"]]}]
        series = ex.get("series", {})
        if series.get("customers_by_day"):
            cb = series["customers_by_day"]
            if cb and isinstance(next(iter(cb.values())), dict):   # org: {host: {day: n}}
                lines = {h: [[d, v] for d, v in sorted(s.items())] for h, s in cb.items()}
            else:
                lines = {ex["entity"]: [[d, v] for d, v in sorted(cb.items())]}
                for h, s in (series.get("rivals_customers_by_day") or {}).items():
                    lines[h] = [[d, v] for d, v in sorted(s.items())]
            secs.append({"kind": "svg", "title": "Customers per day", "svg": svg_lines(lines, ylabel="customers")})
        if series.get("revenue_by_day"):
            secs.append({"kind": "svg", "title": "Revenue and costs per day", "svg": svg_lines(
                {"revenue": [[d, v] for d, v in sorted(series["revenue_by_day"].items())],
                 "costs": [[d, v] for d, v in sorted(series.get("costs_by_day", {}).items())]}, ylabel="cents")})
        secs.append({"kind": "table", "title": "Timeline", "columns": ["seq", "time", "type", "actor", "event"],
                     "rows": [[link(r["seq"], f"/w/{wid}/chain/{r['seq']}"), _fmt_t(r["t"]), r["type"], r["actor"],
                               r["text"]] for r in ex["timeline"][-100:]]})
        return page(request, f"Why: {ex['entity']}", secs, wid, ex["headline"])

    @app.get("/w/{wid}/chain/{seq}", response_class=HTMLResponse)
    def chain_page(request: Request, wid: str, seq: int):
        from ..analytics.trace import causal_chain
        w = W.get(wid)
        c = causal_chain(w, seq)
        ev = c.get("event") or {}
        _, red = redact(ev.get("type", ""), ev.get("payload"), tuple(ev["vis"]) if isinstance(ev.get("vis"), list)
                        else ev.get("vis"), operator)
        if red:
            return page(request, f"Causal chain of #{seq}", [{"kind": "pre", "text": "[private communication]"}], wid)
        secs = [{"kind": "pre", "text": c["text"]},
                {"kind": "table", "columns": ["seq", "time", "relation", "event"],
                 "rows": [[link(l["seq"], f"/w/{wid}/chain/{l['seq']}"), _fmt_t(l["t"]), l["relation"], l["text"]]
                          for l in c["links"]]}]
        return page(request, f"Causal chain of #{seq}", secs, wid)

    @app.get("/w/{wid}/metrics", response_class=PlainTextResponse)
    def prom(wid: str):
        from ..analytics.metrics_export import prometheus_text
        return PlainTextResponse(prometheus_text(W.get(wid)), media_type="text/plain; version=0.0.4")

    @app.get("/experiments", response_class=HTMLResponse)
    def experiments_page(request: Request):
        rows = []
        for i, (n, d) in enumerate(exps):
            rows.append([link(n, f"/experiments/{i}"), d.get("format", "?"),
                         ", ".join(d.get("policies", {})) if isinstance(d.get("policies"), dict) else
                         ", ".join(e["id"] for e in d.get("entries", [])) or d.get("policy")])
        return page(request, "Experiments", [{"kind": "table", "columns": ["file", "format", "policies"], "rows": rows}])

    @app.get("/experiments/{i}", response_class=HTMLResponse)
    def experiment_page(request: Request, i: int):
        if not 0 <= i < len(exps):
            raise HTTPException(404, "no such experiment")
        from ..cli import report_markdown
        n, d = exps[i]
        secs: list = [{"kind": "pre", "title": "Report", "text": report_markdown(d)}]
        fmt = d.get("format", "")
        if fmt.startswith("crucible-experiment"):
            lines = {}
            for pol, runs in d.get("curves", {}).items():
                for r in runs:
                    lines[f"{pol[:14]} s{r['seed']}"] = r["points"]
            secs.append({"kind": "svg", "title": "Learning curves: cumulative verified reward within the lifetime",
                         "svg": svg_lines(lines, ylabel="return", max_series=8)})
        if fmt.startswith("crucible-league"):
            secs.append({"kind": "svg", "title": "Elo", "svg": svg_bars([(r["id"][:40], r["elo"]) for r in d["ratings"]])})
            hist: dict[str, list] = defaultdict(list)
            for k, rec in enumerate(d.get("records", [])):
                for pid, sc in rec["scores"].items():
                    hist[pid[:30]].append([k, sc / 100.0])
            secs.append({"kind": "svg", "title": "Score (independent revenue, $) per match",
                         "svg": svg_lines(hist, xlabel="match", ylabel="$")})
        return page(request, f"Experiment {n}", secs)

    # ------------------------------------------------------------------ JSON
    @app.get("/api/worlds")
    def api_worlds():
        return {"worlds": W.ids()}

    @app.get("/api/w/{wid}/overview")
    def api_overview(wid: str):
        return _json(overview(W.get(wid)))

    @app.get("/api/w/{wid}/metrics")
    def api_metrics(wid: str):
        from ..analytics.metrics_export import world_metrics
        return _json(world_metrics(W.get(wid)))

    @app.get("/api/w/{wid}/actors")
    def api_actors(wid: str):
        return _json(actors(W.get(wid)))

    @app.get("/api/w/{wid}/orgs")
    def api_orgs(wid: str):
        return _json(orgs(W.get(wid)))

    @app.get("/api/w/{wid}/economy")
    def api_economy(wid: str, kind: str = "", owner: str = "", limit: int = Query(100, le=5000)):
        return _json(economy(W.get(wid), kind or None, owner or None, limit))

    @app.get("/api/w/{wid}/comms")
    def api_comms(wid: str):
        return _json(comms(W.get(wid)))

    @app.get("/api/w/{wid}/services")
    def api_services(wid: str):
        return _json(services(W.get(wid)))

    @app.get("/api/w/{wid}/events")
    def api_events(wid: str, type: str = "", actor: str = "", entity: str = "", since: str = "",
                   limit: int = Query(100, le=10000), noise: int = 0):
        return _json(events(W.get(wid), type or None, actor or None, entity or None, since or None, limit, bool(noise)))

    @app.get("/api/w/{wid}/trajectories")
    def api_trajectories(wid: str):
        W.get(wid)
        return _json(trajectories_summary(wid))

    @app.get("/api/w/{wid}/lineage")
    def api_lineage(wid: str):
        return _json(lineage(W.get(wid)))

    @app.get("/api/w/{wid}/evaluation")
    def api_evaluation(wid: str, emergence: int = 0):
        return _json(evaluation(W.get(wid), bool(emergence)))

    @app.get("/api/w/{wid}/explain/{entity:path}")
    def api_explain(wid: str, entity: str, since: int = 0):
        return _json(trace_data(W.get(wid), entity, since))

    @app.get("/api/w/{wid}/chain/{seq}")
    def api_chain(wid: str, seq: int):
        from ..analytics.trace import causal_chain
        c = causal_chain(W.get(wid), seq)
        ev = c.get("event") or {}
        p, red = redact(ev.get("type", ""), ev.get("payload"), tuple(ev["vis"]) if isinstance(ev.get("vis"), list)
                        else ev.get("vis"), operator)
        if red:
            return {"seq": seq, "redacted": True}
        return _json(c)

    @app.get("/api/experiments")
    def api_experiments():
        return [{"index": i, "name": n, "format": d.get("format")} for i, (n, d) in enumerate(exps)]

    @app.get("/api/experiments/{i}")
    def api_experiment(i: int):
        if not 0 <= i < len(exps):
            raise HTTPException(404, "no such experiment")
        return _json(exps[i][1])

    return app


# ====================================================================== helpers
def _cash(w, owner: str) -> int:
    st = w.state
    if not has_table(w, "accounts"):
        return 0
    return int(sum(st.tables["accounts"][a]["balance"] for a in accounts_of(w, owner)))


def _gini(x) -> float:
    x = sorted(x)
    n = len(x)
    s = sum(x)
    if n == 0 or s == 0:
        return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(x))
    return round((2 * cum) / (n * s) - (n + 1) / n, 4)


def _fmt_t(t: int) -> str:
    t = int(t or 0)
    return f"d{t // DAY} {(t % DAY) // 3600:02d}:{(t % 3600) // 60:02d}"


def _json(x):
    from fastapi.responses import JSONResponse
    return JSONResponse(json.loads(json.dumps(x, default=_default)))


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if hasattr(o, "items"):
        return dict(o)
    return str(o)
