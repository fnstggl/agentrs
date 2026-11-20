"""Counterfactual experiments by forking: "what would have happened without X?".

`fork_compare(world, intervention, horizon_s, agents=None)` forks the world twice at its current state (two
isolated in-memory copies of one snapshot; the source world is not touched), applies the intervention to one
fork through *recorded system calls* (so the intervened fork's log says exactly what was changed), advances both
forks by the same schedule - optionally with the same deterministic scripted agents rebuilt from the same
factories and seeds in each fork - and returns outcome deltas (treatment minus baseline):

* customers per host and market shares / HHI per need,
* independent revenue and sales revenue per organisation over the horizon,
* requests served (ok / errors / unavailable / overload) per host,
* requests served by the services that *depended on* the intervened host (callers found in the log),
* household spending and unserved consumers.

Exactness. Both forks start from the same bytes and every random draw is seeded from the world seed and event
sequence numbers, so without an intervention the two futures are identical - *if* all recorded outcomes are
reproducible. Sandboxed programs report CPU time, which feeds capacity and overload; with the default cost model
that is measured wall/CPU time and differs run to run. Set ``params.cost_model = "deterministic"`` (executed-line
counting) when creating the world for exact counterfactuals; `fork_compare` reports ``exact: False`` otherwise and
can check determinism with a second baseline fork (``check_determinism=True``).

Interventions (`Intervention`): `undeploy_host`, `close_listings`, `suspend_principal` (e.g. to disable a
communication channel or an actor), `set_params` (changes the fork's WorldSpec.params; recorded as an internal
`note` event), `system_call` (any operator tool), `combine`, or a custom function.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.kernel import SYSTEM, World

DAY = 86400


# ====================================================================== interventions
@dataclass
class Intervention:
    description: str
    calls: list[tuple[str, dict, str]] = field(default_factory=list)     # (tool, args, as_principal)
    params: dict | None = None
    fn: Callable[[World], Any] | None = None
    resolve: Callable[[World], list[tuple[str, dict, str]]] | None = None
    target_hosts: tuple = ()

    def apply(self, world: World) -> list[dict]:
        """Apply to `world` through recorded system calls. Returns what was done (with results)."""
        out = []
        calls = list(self.calls) + (self.resolve(world) if self.resolve else [])
        for tool, args, as_p in calls:
            res = world.system_call(tool, args, as_principal=as_p)
            out.append({"tool": tool, "args": args, "as": as_p, "result": res})
        if self.params:
            world.spec.params.update(self.params)
            world._run_system("intervention", lambda ctx: ctx.emit(
                "note", {"what": "intervention", "kind": "params", "params": dict(self.params)}, vis="internal"))
            out.append({"params": dict(self.params)})
        if self.fn is not None:
            out.append({"custom": self.description, "result": self.fn(world)})
        return out


def system_call(tool: str, args: dict, as_principal: str = SYSTEM, description: str = "") -> Intervention:
    return Intervention(description or f"{tool}({args})", calls=[(tool, dict(args), as_principal)])


def undeploy_host(host: str) -> Intervention:
    """Stop a deployed service (its tools become unavailable; consumers lose it as a provider)."""
    return Intervention(f"undeploy {host}", calls=[("cloud.undeploy", {"host": host}, SYSTEM)], target_hosts=(host,))


def close_listings(host: str | None = None, listing: str | None = None) -> Intervention:
    """Close a marketplace listing (or every active listing of a host): the offering disappears from the market
    and from search, while the service itself keeps running for callers who already know it."""
    def resolve(w):
        st = w.state
        ids = [listing] if listing else [l for l in st.find_ids("listings", "host", host)
                                         if st.tables["listings"][l]["status"] == "active"]
        return [("market.update", {"listing": l, "status": "closed"}, SYSTEM) for l in ids]
    return Intervention(f"close listings of {host or listing}", resolve=resolve,
                        target_hosts=(host,) if host else ())


def suspend_principal(principal: str) -> Intervention:
    """Suspend a principal (an actor, or a service principal): it can no longer act or send messages."""
    return Intervention(f"suspend {principal}", calls=[("admin.set_status", {"principal": principal,
                                                                             "status": "suspended"}, SYSTEM)])


def set_params(description: str = "", **params) -> Intervention:
    return Intervention(description or f"set params {params}", params=params)


def custom(fn: Callable[[World], Any], description: str) -> Intervention:
    return Intervention(description, fn=fn)


def combine(*ivs: Intervention) -> Intervention:
    def resolve(w):
        out = []
        for iv in ivs:
            out += list(iv.calls) + (iv.resolve(w) if iv.resolve else [])
        return out
    params = {}
    for iv in ivs:
        params.update(iv.params or {})
    fns = [iv.fn for iv in ivs if iv.fn]
    return Intervention(" + ".join(iv.description for iv in ivs), resolve=resolve, params=params or None,
                        fn=(lambda w: [f(w) for f in fns]) if fns else None,
                        target_hosts=tuple(h for iv in ivs for h in iv.target_hosts))


# ====================================================================== measurement
def measure(world: World, t0: int) -> dict:
    """Outcomes accumulated since `t0` plus the current market state."""
    from ..goals.metrics import m_independent_revenue
    st = world.state
    out: dict[str, Any] = {"time": world.now}
    cons = world.modules.get("consumers")
    if cons is not None and "c_prov" in st.arrays:
        out["customers"] = dict(sorted(cons.market_shares(world).items()))
        out["unserved"] = int((st.arrays["c_prov"] < 0).sum())
        needs = sorted({o["need"] for _, o in st.scan("cons_offers")})
        shares, hhi = {}, {}
        for n in needs:
            ms = cons.market_shares(world, n)
            tot = sum(ms.values())
            shares[n] = {h: round(v / tot, 4) for h, v in sorted(ms.items())} if tot else {}
            hhi[n] = round(sum((v / tot) ** 2 for v in ms.values()), 4) if tot else None
        out["shares"] = shares
        out["hhi"] = hhi
    window = max(1, world.now - t0)
    if "orgs" in st.tables and "accounts" in st.tables:
        rev, sales = {}, {}
        for org, o in st.scan("orgs"):
            rev[org] = int(m_independent_revenue(world, org, window))
            s = 0
            for a in st.find_ids("accounts", "owner", org):
                for lid in st.find_ids("acct_txns", "acct", a):
                    e = st.tables["acct_txns"][lid]
                    if e["t"] > t0 and e["amount"] > 0 and st.tables["txns"][e["txn"]]["kind"] == "sale":
                        s += e["amount"]
            sales[org] = s
        out["revenue"] = rev
        out["sales"] = sales
        out["org_status"] = {org: o["status"] for org, o in st.scan("orgs")}
    if "svc_days" in st.tables:
        d0 = t0 // DAY
        req: dict[str, dict[str, int]] = {}
        for k, r in st.scan("svc_days"):
            h, d = k.rsplit("|", 1)
            if int(d) < d0:
                continue
            x = req.setdefault(h, {"ok": 0, "errors": 0, "unavailable": 0, "overload": 0})
            for f in x:
                x[f] += int(r.get(f, 0))
        out["requests"] = req
    if "accounts" in st.tables and st.has("accounts", "acct_households"):
        spent = 0
        for lid in st.find_ids("acct_txns", "acct", "acct_households"):
            e = st.tables["acct_txns"][lid]
            if e["t"] > t0 and e["amount"] < 0:
                spent += -e["amount"]
        out["household_spending"] = spent
    out["digest"] = world.digest()
    return out


def _delta(a: Any, b: Any) -> Any:
    """b - a over nested dicts of numbers (missing values count as 0)."""
    if isinstance(a, dict) or isinstance(b, dict):
        a = a if isinstance(a, dict) else {}
        b = b if isinstance(b, dict) else {}
        out = {}
        for k in sorted(set(a) | set(b)):
            d = _delta(a.get(k), b.get(k))
            if d not in (None, 0, {}):
                out[k] = d
        return out
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
        return None if a == b else {"base": a, "treatment": b}
    if a is None and b is None:
        return None
    da = 0 if a is None else a
    db = 0 if b is None else b
    d = db - da
    return round(d, 6) if isinstance(d, float) else d


# ====================================================================== forks
def _advance(w: World, horizon_s: int, agents, dt: int, seed: int) -> None:
    end = w.now + int(horizon_s)
    pairs = None
    if agents is not None:
        pairs = agents(w) if callable(agents) else [(a, f()) for a, f in agents]
    if pairs:
        from ..agents.base import Runner
        run = Runner(w, pairs, dt=dt, seed=seed, record=False)
        while w.now + dt <= end:
            run.round()
    while w.now < end:
        w.advance_time(min(end, w.now + dt))


def _dependents(world: World, hosts) -> dict[str, list[str]]:
    """Services and principals that called each host before the fork (from the event log)."""
    from .emergence import call_edges
    out: dict[str, list[str]] = {}
    if not hosts:
        return out
    edges = call_edges(world)
    for h in hosts:
        out[h] = sorted({c for (c, dst), _ in edges.items() if dst == h})
    return out


def fork_compare(world: World, intervention: Intervention, horizon_s: int, agents=None, *, dt: int = 3600,
                 seed: int = 0, check_determinism: bool = False, keep_forks: bool = False) -> dict:
    """Run baseline and intervened forks of `world` for `horizon_s` simulated seconds; see module docstring.

    `agents`: None, a list of (actor, zero-argument policy factory), or a callable(world) -> [(actor, Policy)];
    every fork gets fresh policies from the same factories and the same runner seed."""
    t_wall = time.perf_counter()
    blob = world.snapshot_blob()
    t0, seq0 = world.now, world.seq
    exact = world.spec.params.get("cost_model") == "deterministic"
    base = World.from_blob(blob, world_id=f"{world.world_id}~cf-base")
    treat = World.from_blob(blob, world_id=f"{world.world_id}~cf-treat")
    forks = [base, treat]
    try:
        before = measure(base, t0)
        applied = intervention.apply(treat)
        deps = _dependents(world, intervention.target_hosts)
        for w in forks:
            _advance(w, horizon_s, agents, dt, seed)
        mb, mt = measure(base, t0), measure(treat, t0)
        det = None
        if check_determinism:
            again = World.from_blob(blob, world_id=f"{world.world_id}~cf-base2")
            forks.append(again)
            _advance(again, horizon_s, agents, dt, seed)
            det = again.digest() == base.digest()
        delta = _delta({k: v for k, v in mb.items() if k not in ("digest", "time")},
                       {k: v for k, v in mt.items() if k not in ("digest", "time")})
        dep_out = {}
        for h, callers in deps.items():
            rows = {}
            for c in callers:
                ch = c.split(":", 1)[1] if c.startswith("svc:") else None
                if ch:
                    rb = mb.get("requests", {}).get(ch, {})
                    rt = mt.get("requests", {}).get(ch, {})
                    rows[c] = {"base": rb, "treatment": rt, "delta_ok": rt.get("ok", 0) - rb.get("ok", 0)}
                else:
                    rows[c] = {"kind": "direct caller"}
            dep_out[h] = rows
        res = {"intervention": intervention.description, "applied": applied, "horizon_s": int(horizon_s),
               "fork_seq": seq0, "fork_time": t0, "exact": exact, "determinism_check": det,
               "before": before, "baseline": mb, "treatment": mt, "delta": delta, "dependents": dep_out,
               "wall_s": round(time.perf_counter() - t_wall, 3)}
        res["text"] = summarize(res)
        if keep_forks:
            res["forks"] = (base, treat)
        return res
    finally:
        if not keep_forks:
            for w in forks:
                w.close()


def summarize(res: dict) -> str:
    from ..analytics import money
    lines = [f"Counterfactual: {res['intervention']} at t={res['fork_time']} (day {res['fork_time'] / DAY:.2f}), "
             f"horizon {res['horizon_s'] / DAY:.2f} days, exact={res['exact']}"]
    d = res["delta"]
    rb, rt = res["baseline"].get("revenue", {}), res["treatment"].get("revenue", {})
    for org, v in sorted(d.get("revenue", {}).items(), key=lambda kv: kv[1]):
        lines.append(f"  revenue {org}: {money(rb.get(org, 0))} -> {money(rt.get(org, 0))} ({'+' if v > 0 else ''}"
                     f"{money(v)})")
    cb, ct = res["baseline"].get("customers", {}), res["treatment"].get("customers", {})
    for h, v in sorted(d.get("customers", {}).items(), key=lambda kv: kv[1]):
        lines.append(f"  customers {h}: {cb.get(h, 0)} -> {ct.get(h, 0)} ({'+' if v > 0 else ''}{v})")
    if "unserved" in d:
        lines.append(f"  unserved consumers: {'+' if d['unserved'] > 0 else ''}{d['unserved']}")
    if "household_spending" in d:
        lines.append(f"  household spending: {money(d['household_spending'])}")
    for h, rows in res.get("dependents", {}).items():
        for c, r in rows.items():
            if "delta_ok" in r:
                lines.append(f"  dependent {c} of {h}: requests ok {r['base'].get('ok', 0)} -> "
                             f"{r['treatment'].get('ok', 0)}")
    return "\n".join(lines)
