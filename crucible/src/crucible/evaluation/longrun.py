"""Long-run harness with honest liveness measures (evaluator side).

`run_long(family, seed, days, ...)` advances a world for many simulated days (with no focal agent, or with given
agents acting once per simulated hour) and records, per day and per week:

    households          balance of the household pool (demand is capped by household income once it is empty)
    per sector (need)   potential demand (request rate of the customers who currently have the need), demand
                        arriving at sellers, demand served, verified quality, revenue, sellers, unserved customers
    economy             active firms, company registrations (entries), exits and bankruptcies, price changes, new
                        listings, capability adoptions (where that substrate exists), founders' entries and exits
    health              process failures by process name, rejected commands, invariant violations
                        (`evaluation.invariants.InvariantMonitor`, attached for the whole run), event counts
    cost                wall-clock seconds, resident memory, bytes of event log written

Liveness is judged relative to recent levels, per sector. On day d a sector is **collapsed** when the demand it
served that day is below `collapse_fraction` (default 0.5) of its mean over the trailing 28 days (at least 7 days
of history) while customers who still have the need exist. The day is recorded, the reference level is frozen,
and the harness reports the proximate cause it finds in the log around that day: a failing process, the exit of a
seller, a suspension or termination for unpaid compute, a provider outage, a stopped service, or a recorded shock.
The sector has **recovered** on the first later day on which it serves at least `recover_fraction` (default 0.8)
of the frozen reference. "Demand served > 0" plays no role. Independently of every demand number, a scheduled
process that failed `proc_failure_limit` times or more is flagged; and because a relative rule cannot see a
sector that never worked, a sector that served less than half of the demand arriving at its sellers over the
last four weeks is flagged as underserved.

Shocks: `shock={"day": d, "kind": "bankrupt_largest" | "bankrupt_sector" | "provider_outage", ...}` injects one disturbance through
recorded operator calls (module `disturb`, tools under the operator-only host `admin`), using mechanisms that
already exist: the company registry's bankruptcy procedure, and the cloud's outage state. The report says which
sectors were hit, how far they fell, and whether and when they recovered.

At the end the event log is replayed and the digests compared. Output: a JSON summary and a per-day CSV.
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import time
from collections import Counter

import numpy as np

from ..core import schema as S
from ..core.errors import InvalidAction, NotFound
from ..core.kernel import Module, World, endpoint, register_module
from .invariants import InvariantMonitor

DAY = 86400
WEEK = 7 * DAY


# ====================================================================== operator tools for recorded disturbances
@register_module
class Disturb(Module):
    """Operator-only tools (host `admin`: refused for, and hidden from, every in-world actor) that trigger
    mechanisms the world already has. They are recorded commands, so a disturbed run replays and regenerates."""
    name = "disturb"
    host = "admin"
    depends = ("identity", "bank")

    @endpoint("shock_bankrupt", S.obj({"org": S.s_str(maxLength=40)}, ["org"]))
    def ep_bankrupt(self, ctx, org):
        """Operator only: put a registered company through the registry's bankruptcy procedure now."""
        biz = ctx.world.modules.get("business")
        if biz is None or not biz.is_registered(ctx.state, org):
            raise NotFound(f"{org} is not a registered company")
        biz._wind_up(ctx, org, "bankrupt")
        return {"org": org, "status": "bankrupt"}

    @endpoint("shock_outage", S.obj({"provider": S.s_str(maxLength=40), "hours": S.s_int(minimum=1, maximum=720)},
                                    ["provider", "hours"]))
    def ep_outage(self, ctx, provider, hours):
        """Operator only: a compute provider goes down for `hours` (the cloud's own outage state)."""
        p = ctx.state.get("providers", provider) if "providers" in ctx.state.tables else None
        if p is None or p["status"] != "active":
            raise InvalidAction(f"{provider} is not an active provider")
        ctx.emit("cloud.outage", {"id": provider, "start": ctx.now, "until": ctx.now + int(hours) * 3600}, vis="public")
        return {"provider": provider, "until": ctx.now + int(hours) * 3600}


# ====================================================================== measurement
def _rss_mb() -> float:
    try:
        with open("/proc/self/statm") as fh:
            return round(int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6, 1)
    except (OSError, ValueError, IndexError):
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


class _Recorder:
    """Streams committed events into per-day tallies and an incident list (for cause attribution)."""

    def __init__(self, world):
        self.world = world
        self.day = self._blank()
        self.incidents: list[dict] = []
        self.bytes_total = 0
        world.listeners.append(self.on_event)

    def _blank(self) -> dict:
        return {"events": 0, "types": Counter(), "sector": {}, "proc_failed": Counter(), "rejected": 0,
                "entries": 0, "exits": 0, "bankruptcies": 0, "price_changes": 0, "new_listings": 0, "adoptions": 0,
                "founder_entries": 0, "founder_exits": 0, "founder_failures": 0, "log_bytes": 0, "commands": 0}

    def close(self) -> None:
        try:
            self.world.listeners.remove(self.on_event)
        except ValueError:
            pass

    def _incident(self, ev, kind, **kw) -> None:
        self.incidents.append({"t": ev.t, "day": ev.t // DAY, "seq": ev.seq, "kind": kind, **kw})

    def on_event(self, ev) -> None:
        d, t, p = self.day, ev.type, ev.payload
        d["events"] += 1
        d["types"][t] += 1
        n = len(json.dumps(ev.to_dict(), default=str, separators=(",", ":")))
        d["log_bytes"] += n
        self.bytes_total += n
        st = self.world.state
        if t == "consumers.hour":
            # the hourly demand step has just been applied: read what it wrote to the summary table (requests that
            # arrived at each offering, how many were served, their verified quality), not the event's payload
            slot_need = {o["slot"]: str(o["need"]) for o in st.tables["cons_offers"].values()}
            now = self.world.now
            for sm in st.tables["cons_summary"].values():
                need = slot_need.get(sm["slot"])
                if need is None or sm.get("t") != now:
                    continue
                s = d["sector"].setdefault(need, {"requests": 0, "served": 0, "q_sum": 0.0})
                s["requests"] += int(sm["requests"])
                s["served"] += int(sm["ok"])
                s["q_sum"] += float(sm["q"] or 0.0) * int(sm["ok"])
        elif t == "proc.failed":
            d["proc_failed"][p.get("proc", "?")] += 1
            self._incident(ev, "process_failure", proc=p.get("proc"), error=str(p.get("error", ""))[:160])
        elif t == "cmd.rejected":
            d["rejected"] += 1
        elif t == "cmd":
            d["commands"] += 1
            if p.get("system") and str(p.get("tool", "")).startswith("admin.shock_"):
                self._incident(ev, "shock", tool=p["tool"], args=p.get("args"))
        elif t == "biz.registered":
            d["entries"] += 1
        elif t == "biz.status":
            d["exits"] += 1
            d["bankruptcies"] += p["status"] == "bankrupt"
            self._incident(ev, "exit", org=p["org"], status=p["status"], cash=p.get("cash"), owed=p.get("owed"))
        elif t == "market.listing_update":
            d["price_changes"] += "price" in p["changes"]
        elif t == "market.listing":
            d["new_listings"] += p.get("kind") == "api"
        elif t == "caps.install_status":
            d["adoptions"] += p.get("status") == "active" and p.get("reason") == "adopted"
        elif t == "cloud.instance_status":
            if p["status"] in ("suspended", "terminated"):
                inst = st.tables["instances"].get(p["id"]) or {}
                self._incident(ev, "suspension" if p["status"] == "suspended" else "termination", instance=p["id"],
                               org=inst.get("owner"), provider=inst.get("provider"), reason=str(p.get("reason", ""))[:80])
        elif t == "cloud.dep_status":
            if p["status"] in ("stopped", "failed"):
                dep = st.tables["deployments"].get(p["id"]) or {}
                self._incident(ev, "service_stopped", host=dep.get("host"), org=dep.get("owner"),
                               reason=str(p.get("reason", ""))[:80])
        elif t == "cloud.outage":
            self._incident(ev, "outage", provider=p["id"], start=p["start"], until=p["until"])
        elif t == "entry.entered":
            d["founder_entries"] += 1
        elif t == "entry.exited":
            d["founder_exits"] += 1
            d["founder_failures"] += p.get("outcome") == "failed"

    def take(self) -> dict:
        d, self.day = self.day, self._blank()
        return d


def _sector_state(world) -> dict:
    """Per need: customers who have it now, how many of them are unserved, their potential request rate, and the
    sellers able to serve (active listing on a live deployment). Evaluator-side read of the population arrays."""
    st = world.state
    out: dict[str, dict] = {}
    names = [str(n) for n in (st.meta_get("cons_needs") or [])]
    if not names or "c_need" not in st.arrays:
        return out
    need, prov, rate = st.arrays["c_need"], st.arrays["c_prov"], st.arrays["c_rate"]
    active = st.arrays["c_active"] if "c_active" in st.arrays else None
    cloud = world.modules.get("cloud")
    sellers: dict[str, set] = {n: set() for n in names}
    for lid in st.find_ids("listings", "status", "active"):
        l = st.tables["listings"][lid]
        if l["kind"] == "api" and l["need"] is not None and str(l["need"]) in sellers \
                and cloud is not None and cloud.live_deployment(st, l["host"] or "") is not None:
            sellers[str(l["need"])].add(l["seller"])
    for i, n in enumerate(names):
        m = need == i
        if active is not None:
            m = m & active
        out[n] = {"customers": int(m.sum()), "unserved": int((m & (prov < 0)).sum()),
                  "potential": round(float(rate[m].sum()), 1), "sellers": len(sellers[n]),
                  "seller_orgs": sorted(sellers[n])}
    return out


def _sector_revenue(world) -> dict[str, int]:
    """Revenue to date by need, from the cloud's own per-service records (the harness takes daily differences, so
    a day's revenue covers exactly the hours whose requests it counts)."""
    st = world.state
    out: dict[str, int] = {}
    if "svc_days" not in st.tables or "listings" not in st.tables:
        return out
    host_need = {l["host"]: str(l["need"]) for _, l in st.scan("listings")
                 if l["kind"] == "api" and l["host"] and l["need"] is not None}
    for _, r in st.scan("svc_days"):
        need = host_need.get(r["host"])
        if need is not None and r["revenue_cents"]:
            out[need] = out.get(need, 0) + int(r["revenue_cents"])
    return out


def _org_sectors(world) -> dict[str, set]:
    st = world.state
    out: dict[str, set] = {}
    if "listings" in st.tables:
        for _, l in st.scan("listings"):
            if l["kind"] == "api" and l["need"] is not None:
                out.setdefault(l["seller"], set()).add(str(l["need"]))
    return out


# ====================================================================== liveness
def _proximate_cause(sector, day, incidents, org_sectors, providers_of) -> dict:
    """The most specific recorded incident that can explain a fall of this sector's output on `day`."""
    lo = day - 3
    cands = []
    for inc in incidents:
        if not (lo <= inc["day"] <= day):
            continue
        k = inc["kind"]
        if k == "process_failure":
            if inc["day"] == day and inc.get("proc") == "consumers.hour":
                cands.append((0, inc))
            elif inc["day"] == day:
                cands.append((6, inc))
        elif k == "shock":
            cands.append((1, inc))
        elif k == "exit" and sector in org_sectors.get(inc.get("org"), ()):
            cands.append((2, inc))
        elif k in ("suspension", "termination") and sector in org_sectors.get(inc.get("org"), ()):
            cands.append((3 if "payment" in inc.get("reason", "") or "unpaid" in inc.get("reason", "") else 5, inc))
        elif k == "outage" and inc.get("provider") in providers_of.get(sector, ()) \
                and inc["start"] // DAY <= day <= max(inc["start"], inc["until"] - 1) // DAY:
            cands.append((4, inc))
        elif k == "service_stopped" and sector in org_sectors.get(inc.get("org"), ()):
            cands.append((5, inc))
    if not cands:
        return {"kind": "unknown", "evidence": []}
    cands.sort(key=lambda c: (c[0], -c[1]["seq"]))
    top = cands[0][1]
    label = {"process_failure": "process failure", "shock": "recorded shock", "exit": "exit of a seller",
             "suspension": "suspension for unpaid bills", "termination": "instance terminated",
             "outage": "provider outage", "service_stopped": "service stopped"}[top["kind"]]
    if top["kind"] in ("suspension", "termination") and cands[0][0] == 5:
        label = "instance " + ("suspended" if top["kind"] == "suspension" else "terminated")
    return {"kind": top["kind"], "label": label, "incident": top,
            "evidence": [c[1] for c in cands[:6]]}


class _Liveness:
    def __init__(self, collapse_fraction=0.5, recover_fraction=0.8, window=28, min_history=7):
        self.cf, self.rf, self.window, self.min_history = collapse_fraction, recover_fraction, window, min_history
        self.served: dict[str, list] = {}
        self.open: dict[str, dict] = {}
        self.episodes: list[dict] = []

    def day(self, day, sectors, sstate, incidents, org_sectors, providers_of) -> list[dict]:
        new = []
        for need in sorted(set(sectors) | set(sstate)):
            served = sectors.get(need, {}).get("served", 0)
            hist = self.served.setdefault(need, [])
            ep = self.open.get(need)
            if ep is not None:
                ep["min_served"] = min(ep["min_served"], served)
                if served >= self.rf * ep["reference"]:
                    ep.update(recovered_day=day, days_to_recover=day - ep["day"])
                    del self.open[need]
            elif len(hist) >= self.min_history:
                ref = float(np.mean(hist[-self.window:]))
                customers = sstate.get(need, {}).get("customers", 0)
                if ref > 0 and served < self.cf * ref and customers > 0:
                    ep = {"sector": need, "day": day, "served": served, "reference": round(ref, 1),
                          "ratio": round(served / ref, 3), "customers_with_need": customers,
                          "min_served": served, "recovered_day": None, "days_to_recover": None,
                          "cause": _proximate_cause(need, day, incidents, org_sectors, providers_of)}
                    self.open[need] = ep
                    self.episodes.append(ep)
                    new.append(ep)
            if need not in self.open:           # the trailing mean is of normal days: collapsed days do not drag it down
                hist.append(served)
        return new


# ====================================================================== shocks
def _largest_seller(world, sector=None) -> tuple[str, str] | None:
    """(org, sector) of the seller with the most customers (in `sector`, or overall)."""
    st = world.state
    if "cons_offers" not in st.tables or "c_prov" not in st.arrays:
        return None
    counts = np.bincount(st.arrays["c_prov"][st.arrays["c_prov"] >= 0], minlength=st.count("cons_offers") + 1)
    best = None
    for lid, o in st.scan("cons_offers"):
        l = st.tables["listings"].get(lid)
        if l is None or l["status"] != "active" or (sector and str(o["need"]) != sector):
            continue
        key = (int(counts[o["slot"]]), lid)
        if best is None or key > best[0]:
            best = (key, l["seller"], str(o["need"]))
    return (best[1], best[2]) if best else None


def inject_shock(world, shock: dict) -> dict:
    """Inject one disturbance through recorded operator calls. Returns what was done."""
    kind = shock.get("kind", "bankrupt_largest")
    st = world.state
    if kind == "bankrupt_largest":
        pick = _largest_seller(world, shock.get("sector"))
        if pick is None:
            return {"kind": kind, "done": False, "why": "no seller with customers"}
        org, sector = pick
        if "admin.shock_bankrupt" in world.endpoints:
            world.system_call("admin.shock_bankrupt", {"org": org})
            how = "admin.shock_bankrupt"
        else:       # a world built without the operator tools: the owner's own wind-up, as a recorded system call
            owner = next(m["person"] for m in st.find("members", "org", org) if m["role"] == "owner")
            world.system_call("biz.dissolve", {"org": org}, as_principal=owner)
            how = "biz.dissolve as the owner"
        return {"kind": kind, "done": True, "org": org, "sectors": [sector], "how": how, "t": world.now}
    if kind == "bankrupt_sector":       # every seller of one sector (default: the sector of the largest seller)
        pick = _largest_seller(world, shock.get("sector"))
        if pick is None or "admin.shock_bankrupt" not in world.endpoints:
            return {"kind": kind, "done": False, "why": "no seller with customers, or no operator tool (module disturb)"}
        sector = pick[1]
        orgs = sorted({l["seller"] for _, l in st.scan("listings") if l["kind"] == "api" and l["status"] == "active"
                       and l["need"] is not None and str(l["need"]) == sector})
        for org in orgs:
            world.system_call("admin.shock_bankrupt", {"org": org})
        return {"kind": kind, "done": True, "orgs": orgs, "org": pick[0], "sectors": [sector],
                "how": "admin.shock_bankrupt", "t": world.now}
    if kind == "provider_outage":
        prov = shock.get("provider")
        if prov is None:        # the provider hosting the most running cores
            used: dict[str, int] = {}
            for _, i in st.scan("instances"):
                if i["status"] == "running":
                    used[i["provider"]] = used.get(i["provider"], 0) + i["cores"]
            if not used:
                return {"kind": kind, "done": False, "why": "no running instance"}
            prov = sorted(used, key=lambda k: (-used[k], k))[0]
        hours = int(shock.get("hours", 48))
        if "admin.shock_outage" not in world.endpoints:
            return {"kind": kind, "done": False, "why": "world has no operator outage tool (module disturb)"}
        world.system_call("admin.shock_outage", {"provider": prov, "hours": hours})
        cloud = world.modules["cloud"]
        hit = set()
        for _, l in st.scan("listings"):
            dep = cloud.live_deployment(st, l["host"] or "") if l["kind"] == "api" and l["status"] == "active" else None
            if dep and st.tables["instances"][dep["instance"]]["provider"] == prov and l["need"] is not None:
                hit.add(str(l["need"]))
        return {"kind": kind, "done": True, "provider": prov, "hours": hours, "sectors": sorted(hit),
                "how": "admin.shock_outage", "t": world.now}
    raise ValueError(f"unknown shock kind {kind!r}")


# ====================================================================== the harness
CSV_BASE = ["day", "events", "commands", "rejected", "proc_failures", "violations_new", "active_firms", "entries",
            "exits", "bankruptcies", "price_changes", "new_listings", "adoptions", "founder_entries",
            "founder_exits", "founder_failures", "wall_s", "rss_mb", "log_bytes", "served_total", "requests_total",
            "revenue_total", "households_cents", "collapsed"]
CSV_SECTOR = ["potential", "requests", "served", "quality", "revenue", "sellers", "customers", "unserved"]


def run_long(family: str, seed: int, days: int, overrides: dict | None = None, out_dir: str | None = None,
             check_every_days: float = 1, *, extra_modules=(), agents=None, shock: dict | None = None,
             collapse_fraction: float = 0.5, recover_fraction: float = 0.8, proc_failure_limit: int = 3,
             label: str | None = None, snapshot_every_days: int | None = None, resume: bool = False,
             replay_check: bool = True, world: World | None = None, base_blob: bytes | None = None,
             before_day=None, progress=None, close: bool = True, min_history_days: int = 7) -> dict:
    """Advance a world for `days` simulated days and report liveness. See the module docstring.

    agents       optional [(handle, policy, endowment_cents)]: admitted with `admin.join`, each acts once per hour
    shock        optional {"day": d, "kind": "bankrupt_largest"|"bankrupt_sector"|"provider_outage",
                 "sector"|"provider"|"hours"}
    world        use this world instead of generating one (e.g. a prepared fork); it is not closed if close=False.
                 For a fork, pass the snapshot it was made from as `base_blob` so the replay check has a base.
    before_day   optional callable (world, day) run before each simulated day (for tests that break things)
    resume       continue from `<out_dir>/<label>.snap` written by an earlier call with snapshot_every_days
    """
    from ..agents.base import Runner
    from ..worldgen import generate
    t_start = time.time()
    label = label or f"{family}_s{seed}"
    extra = list(extra_modules)
    if shock is not None and "disturb" not in extra and world is None:
        extra.append("disturb")
    snap_path = os.path.join(out_dir, f"{label}.snap") if out_dir else None
    state_path = os.path.join(out_dir, f"{label}.state.pkl") if out_dir else None
    saved = None
    if resume and snap_path and os.path.exists(snap_path) and os.path.exists(state_path):
        with open(snap_path, "rb") as fh:
            base_blob = fh.read()
        with open(state_path, "rb") as fh:
            saved = pickle.load(fh)
        world = World.from_blob(base_blob, world_id=saved["world_id"])
    elif world is None:
        spec = generate({"family": family, "overrides": dict(overrides or {}), "extra_modules": extra}, seed)
        world = World.create(spec, world_id=f"long-{label}")
    w = world
    actors = []
    if saved is None:
        for handle, policy, endowment in (agents or []):
            w.system_call("admin.join", {"handle": handle, "endowment": int(endowment)})
            actors.append((f"p:{handle}", policy))
    else:
        actors = [(f"p:{h}", pol) for h, pol, _ in (agents or [])]
    mon = InvariantMonitor(w, proc_failure_limit=proc_failure_limit).attach(every_s=int(check_every_days * DAY))
    rec = _Recorder(w)
    live = _Liveness(collapse_fraction, recover_fraction, min_history=int(min_history_days))
    rows: list[dict] = []
    shock_report = None
    day0 = w.now // DAY
    if saved is not None:
        rows, live.served, live.open, live.episodes = saved["rows"], saved["served"], saved["open"], saved["episodes"]
        rec.incidents, rec.bytes_total, shock_report = saved["incidents"], saved["bytes_total"], saved["shock"]
        mon.proc_failures, mon.violations = saved["proc_failures"], saved["violations"]
        mon._seen = {(v.kind, v.where) for v in mon.violations}
    runner = Runner(w, actors, dt=3600, record=False, seed=seed) if actors else None
    rev_before = _sector_revenue(w)
    n_viol = len(mon.violations)
    try:
        for day in range(day0, int(days)):
            t_day = time.time()
            if before_day is not None:
                before_day(w, day)
            if shock is not None and shock_report is None and day == int(shock["day"]):
                shock_report = inject_shock(w, shock)
                shock_report["day"] = day
            if runner is not None:
                runner.run(24)
            else:
                w.advance_time((day + 1) * DAY)
            d = rec.take()
            sstate = _sector_state(w)
            org_sectors = _org_sectors(w)
            providers_of: dict[str, set] = {}
            if "instances" in w.state.tables:
                for _, i in w.state.scan("instances"):
                    for s in org_sectors.get(i["owner"], ()):
                        providers_of.setdefault(s, set()).add(i["provider"])
            sectors = {}
            rev_now = _sector_revenue(w)
            revenue = {k: v - rev_before.get(k, 0) for k, v in rev_now.items()}
            rev_before = rev_now
            for need in sorted(set(d["sector"]) | set(sstate)):
                s = d["sector"].get(need, {"requests": 0, "served": 0, "q_sum": 0.0})
                ss = sstate.get(need, {})
                sectors[need] = {"potential": ss.get("potential", 0.0), "requests": s["requests"], "served": s["served"],
                                 "quality": round(s["q_sum"] / s["served"], 4) if s["served"] else None,
                                 "revenue": revenue.get(need, 0), "failed": s["requests"] - s["served"],
                                 "sellers": ss.get("sellers", 0),
                                 "customers": ss.get("customers", 0), "unserved": ss.get("unserved", 0)}
            new_eps = live.day(day, sectors, sstate, rec.incidents, org_sectors, providers_of)
            firms = len(w.state.find_ids("companies", "status", "registered")) if "companies" in w.state.tables else 0
            row = {"day": day, "events": d["events"], "commands": d["commands"], "rejected": d["rejected"],
                   "proc_failed": dict(d["proc_failed"]), "proc_failures": sum(d["proc_failed"].values()),
                   "violations_new": len(mon.violations) - n_viol, "active_firms": firms,
                   **{k: int(d[k]) for k in ("entries", "exits", "bankruptcies", "price_changes", "new_listings",
                                             "adoptions", "founder_entries", "founder_exits", "founder_failures")},
                   "wall_s": round(time.time() - t_day, 3), "rss_mb": _rss_mb(), "log_bytes": d["log_bytes"],
                   "served_total": sum(s["served"] for s in sectors.values()),
                   "requests_total": sum(s["requests"] for s in sectors.values()),
                   "revenue_total": sum(s["revenue"] for s in sectors.values()),
                   "households_cents": (w.state.tables["accounts"]["acct_households"]["balance"]
                                        if "accounts" in w.state.tables and "acct_households" in w.state.tables["accounts"]
                                        else None),
                   "collapsed": sorted(live.open), "new_collapses": [e["sector"] for e in new_eps],
                   "sectors": sectors, "top_types": d["types"].most_common(5)}
            n_viol = len(mon.violations)
            rows.append(row)
            if progress is not None:
                progress(row)
            if snapshot_every_days and snap_path and (day + 1) % int(snapshot_every_days) == 0 and day + 1 < days:
                w.flush()
                blob = w.snapshot_blob()
                with open(snap_path + ".tmp", "wb") as fh:
                    fh.write(blob)
                with open(state_path + ".tmp", "wb") as fh:
                    pickle.dump({"world_id": w.world_id, "rows": rows, "served": live.served, "open": live.open,
                                 "episodes": live.episodes, "incidents": rec.incidents,
                                 "bytes_total": rec.bytes_total, "shock": shock_report,
                                 "proc_failures": mon.proc_failures, "violations": mon.violations}, fh)
                os.replace(snap_path + ".tmp", snap_path)
                os.replace(state_path + ".tmp", state_path)
        final_violations = mon.check()
        summary = _summarise(w, family, seed, days, label, rows, live, mon, rec, shock, shock_report,
                             final_violations, collapse_fraction, recover_fraction, proc_failure_limit, extra,
                             overrides, check_every_days)
        if replay_check:
            t0 = time.time()
            evs = [e.to_dict() for e in w.log]
            if base_blob is None and (not evs or evs[0]["seq"] != 1):
                summary["replay"] = {"identical": None, "skipped": "the log does not start at genesis and no base "
                                                                   "snapshot was given"}
            else:
                rep = World.replay(w.spec, evs, base_blob=base_blob)
                summary["replay"] = {"identical": rep.digest() == w.digest(), "digest": w.digest(), "events": len(evs),
                                     "from": "snapshot" if base_blob is not None else "genesis",
                                     "wall_s": round(time.time() - t0, 1)}
                rep.close()
        else:
            summary["replay"] = None
        summary["wall_s"] = round(time.time() - t_start, 1)
        summary["healthy"] = bool(not summary["liveness"]["unrecovered"] and not summary["flags"]["repeated_process_failures"]
                                  and not summary["flags"]["underserved_sectors"]
                                  and summary["invariants"]["violations"] == 0
                                  and (summary["replay"] is None or summary["replay"]["identical"] is not False))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"{label}.json"), "w") as fh:
                json.dump(summary, fh, indent=1, default=str)
            write_csv(os.path.join(out_dir, f"{label}_days.csv"), rows)
            summary["files"] = {"json": os.path.join(out_dir, f"{label}.json"),
                                "csv": os.path.join(out_dir, f"{label}_days.csv")}
        return summary
    finally:
        rec.close()
        mon.detach()
        if close:
            w.close()


def write_csv(path: str, rows: list[dict]) -> None:
    sectors = sorted({s for r in rows for s in r["sectors"]})
    cols = CSV_BASE + [f"{s}:{k}" for s in sectors for k in CSV_SECTOR]
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for r in rows:
            line = [";".join(r[c]) if c == "collapsed" else ("" if r[c] is None else r[c]) for c in CSV_BASE]
            for s in sectors:
                x = r["sectors"].get(s, {})
                line += [x.get(k, "") if x.get(k) is not None else "" for k in CSV_SECTOR]
            wr.writerow(line)


def _weeks(rows: list[dict]) -> list[dict]:
    out = []
    for i in range(0, len(rows), 7):
        chunk = rows[i:i + 7]
        if len(chunk) < 7:
            break
        sectors: dict[str, dict] = {}
        for r in chunk:
            for s, x in r["sectors"].items():
                a = sectors.setdefault(s, {"requests": 0, "served": 0, "revenue": 0, "q_sum": 0.0, "potential": 0.0})
                a["requests"] += x["requests"]
                a["served"] += x["served"]
                a["revenue"] += x["revenue"]
                a["potential"] += x["potential"]
                a["q_sum"] += (x["quality"] or 0.0) * x["served"]
        for s, a in sectors.items():
            a["quality"] = round(a.pop("q_sum") / a["served"], 4) if a["served"] else None
            a["potential"] = round(a["potential"], 1)
            a["sellers_end"] = chunk[-1]["sectors"].get(s, {}).get("sellers", 0)
            a["unserved_end"] = chunk[-1]["sectors"].get(s, {}).get("unserved", 0)
        pf: Counter = Counter()
        for r in chunk:
            pf.update(r["proc_failed"])
        out.append({"week": len(out) + 1, "days": [chunk[0]["day"], chunk[-1]["day"]],
                    "served": sum(r["served_total"] for r in chunk), "requests": sum(r["requests_total"] for r in chunk),
                    "revenue": sum(r["revenue_total"] for r in chunk), "sectors": sectors,
                    "active_firms_end": chunk[-1]["active_firms"],
                    **{k: sum(r[k] for r in chunk) for k in ("entries", "exits", "bankruptcies", "price_changes",
                                                             "new_listings", "adoptions", "founder_entries",
                                                             "founder_exits", "founder_failures", "rejected",
                                                             "events", "violations_new", "log_bytes")},
                    "process_failures": dict(pf), "wall_s": round(sum(r["wall_s"] for r in chunk), 1),
                    "rss_mb_end": chunk[-1]["rss_mb"],
                    "collapsed_days": sum(1 for r in chunk if r["collapsed"]),
                    "structural_change": bool(sum(r[k] for r in chunk for k in ("entries", "exits", "new_listings",
                                                                                 "adoptions", "price_changes")))})
    return out


def underserved_sectors(rows: list[dict], fraction: float = 0.5, window: int = 28, min_days: int = 14) -> list[dict]:
    """Sectors that have been failing most of their customers for weeks: over the last `window` days less than
    `fraction` of the demand that arrived at sellers was served. This is a level, not a fall, so the relative
    collapse rule cannot see it (a sector that never worked has nothing to fall from)."""
    out = []
    if len(rows) < min_days:
        return out
    for s in sorted({k for r in rows for k in r["sectors"]}):
        last = [r["sectors"].get(s, {}) for r in rows[-window:]]
        req, ok = sum(int(x.get("requests") or 0) for x in last), sum(int(x.get("served") or 0) for x in last)
        if req > 0 and ok < fraction * req:
            all_req = sum(int(r["sectors"].get(s, {}).get("requests") or 0) for r in rows)
            all_ok = sum(int(r["sectors"].get(s, {}).get("served") or 0) for r in rows)
            out.append({"sector": s, "served_share_last_window": round(ok / req, 4), "window_days": len(last),
                        "requests_per_day": round(req / len(last), 1), "served_per_day": round(ok / len(last), 1),
                        "served_share_whole_run": round(all_ok / all_req, 4) if all_req else None})
    return out


def _summarise(w, family, seed, days, label, rows, live, mon, rec, shock, shock_report, final_violations,
               collapse_fraction, recover_fraction, proc_failure_limit, extra, overrides, check_every_days) -> dict:
    weeks = _weeks(rows)
    pf: Counter = Counter()
    for r in rows:
        pf.update(r["proc_failed"])
    repeated = [{"proc": p, "failures": int(n), "first_day": mon.proc_failures.get(p, {}).get("first_t", 0) // DAY,
                 "last_error": mon.proc_failures.get(p, {}).get("last_error", "")}
                for p, n in sorted(pf.items()) if n >= proc_failure_limit]
    served_w = [x["served"] for x in weeks]
    mid = float(np.median(served_w[1:8])) if len(served_w) >= 3 else None
    late = [x for x in served_w[9:]]
    # the household sector pays for consumer demand out of a pool fed by a fixed daily income: once the pool is
    # empty, demand is capped by income whatever sellers do (an understandable, non-industry cause of lower output).
    # A day's row is taken just after the next day's income arrived, so the budget binds once that balance no
    # longer covers a day of spending at the day's level.
    bound = next((r["day"] for r in rows if r["households_cents"] is not None and r["revenue_total"] > 0
                  and r["households_cents"] < r["revenue_total"]), None)
    out = {
        "family": family, "seed": seed, "days": int(days), "label": label, "world": w.world_id,
        "spec_hash": w.spec.hash(), "modules": [m["name"] for m in w.spec.modules], "extra_modules": list(extra),
        "overrides": overrides or {}, "check_every_days": check_every_days,
        "events": w.seq, "commands": w.stats["commands"], "rejected_commands": sum(r["rejected"] for r in rows),
        "weeks": weeks,
        "output": {"served_by_week": served_w, "median_weeks_2_8": mid,
                   "last_week_over_median_weeks_2_8": round(served_w[-1] / mid, 3) if mid else None,
                   "mean_after_week_9_over_median_weeks_2_8": round(float(np.mean(late)) / mid, 3) if mid and late else None,
                   "weeks_with_structural_change_after_9": sum(1 for x in weeks[9:] if x["structural_change"]),
                   "weeks_after_9": len(weeks[9:])},
        "household_budget": {"bound_from_day": bound,
                             "rule": "first day after which the household pool (including the next day's income) "
                                     "holds less than that day's consumer spending",
                             "balance_end": rows[-1]["households_cents"] if rows else None},
        "liveness": {"rule": f"collapsed when a day's served demand < {collapse_fraction} x trailing 28-day mean "
                             f"(>= 7 days of history) while customers with the need exist; recovered at "
                             f">= {recover_fraction} x that mean",
                     "collapses": live.episodes, "unrecovered": sorted(live.open),
                     "sectors": {s: {"served_first_week": sum(h[:7]), "served_last_week": sum(r["sectors"].get(s, {}).get("served", 0) for r in rows[-7:]),
                                     "sellers_end": rows[-1]["sectors"].get(s, {}).get("sellers", 0) if rows else 0,
                                     "unserved_end": rows[-1]["sectors"].get(s, {}).get("unserved", 0) if rows else 0}
                                 for s, h in sorted(live.served.items())}},
        "flags": {"repeated_process_failures": repeated, "process_failures": dict(pf),
                  "collapsed_sectors": sorted({e["sector"] for e in live.episodes}),
                  "underserved_sectors": underserved_sectors(rows)},
        "invariants": {**mon.summary(), "open_at_end": [v.to_dict() for v in final_violations][:50],
                       "first": [v.to_dict() for v in mon.violations][:50]},
        "economy": {"entries": sum(r["entries"] for r in rows), "exits": sum(r["exits"] for r in rows),
                    "bankruptcies": sum(r["bankruptcies"] for r in rows),
                    "price_changes": sum(r["price_changes"] for r in rows),
                    "new_listings": sum(r["new_listings"] for r in rows),
                    "adoptions": sum(r["adoptions"] for r in rows),
                    "active_firms_start": rows[0]["active_firms"] if rows else 0,
                    "active_firms_end": rows[-1]["active_firms"] if rows else 0},
        "cost": {"wall_s_days": round(sum(r["wall_s"] for r in rows), 1),
                 "s_per_sim_day": round(sum(r["wall_s"] for r in rows) / max(1, len(rows)), 3),
                 "rss_mb_end": rows[-1]["rss_mb"] if rows else None, "rss_mb_max": max((r["rss_mb"] for r in rows), default=None),
                 "log_bytes": rec.bytes_total, "log_bytes_per_day": round(rec.bytes_total / max(1, len(rows))),
                 "invariant_checks": mon.checks, "invariant_wall_s": round(mon.wall_s, 2)},
        "incidents": {k: n for k, n in sorted(Counter(i["kind"] for i in rec.incidents).items())},
    }
    if "founders" in w.state.tables:
        from ..actors.entry import entrant_orgs, entry_stats
        es = entry_stats(w)
        ents = entrant_orgs(w)
        now = _sector_state(w)
        es["sectors_served_by_entrants_at_end"] = sorted(s for s, x in now.items() if set(x["seller_orgs"]) & ents)
        es["sectors_served_only_by_entrants_at_end"] = sorted(
            s for s, x in now.items() if x["seller_orgs"] and set(x["seller_orgs"]) <= ents)
        out["entry"] = es
    if shock is not None:
        out["shock"] = _shock_outcome(shock, shock_report, rows, live, w, recover_fraction)
    return out


def _shock_outcome(shock, report, rows, live, w, recover_fraction) -> dict:
    if not report or not report.get("done"):
        return {"requested": shock, "report": report, "sectors": {}}
    day = report["day"]
    out = {"requested": shock, "report": report, "sectors": {}}
    by_day = {r["day"]: r for r in rows}
    for s in report.get("sectors", []):
        before = [by_day[d]["sectors"].get(s, {}).get("served", 0) for d in range(max(0, day - 7), day) if d in by_day]
        ref = float(np.mean(before)) if before else 0.0
        after = [(d, by_day[d]["sectors"].get(s, {}).get("served", 0)) for d in sorted(by_day) if d >= day]
        low = min(after, key=lambda x: x[1]) if after else (day, 0)
        rec_day = None
        fell = False
        for d, v in after:
            if v < recover_fraction * ref:
                fell = True
            elif fell:
                rec_day = d
                break
        eps = [e for e in live.episodes if e["sector"] == s and e["day"] >= day]
        out["sectors"][s] = {"served_per_day_before": round(ref, 1), "lowest_day": low[0], "lowest_served": low[1],
                             "fell_below_recover_level": fell, "recovered_day": rec_day,
                             "days_to_recover": None if rec_day is None else rec_day - day,
                             "flagged_collapsed_on": eps[0]["day"] if eps else None,
                             "cause_reported": eps[0]["cause"].get("label") if eps else None,
                             "served_last_day": after[-1][1] if after else None,
                             "sellers_end": rows[-1]["sectors"].get(s, {}).get("sellers", 0)}
    return out


def liveness_table(summary: dict) -> str:
    """A plain-text weekly table (served per sector, sellers, entries/exits, failures) for reports."""
    sectors = sorted({s for wk in summary["weeks"] for s in wk["sectors"]})
    head = ["wk"] + [f"{s} served/sellers" for s in sectors] + ["firms", "in", "out", "price", "procfail", "viol", "collapsed-days"]
    lines = [" | ".join(head)]
    for wk in summary["weeks"]:
        cells = [str(wk["week"])]
        for s in sectors:
            x = wk["sectors"].get(s, {})
            cells.append(f"{x.get('served', 0)}/{x.get('sellers_end', 0)}")
        cells += [str(wk["active_firms_end"]), str(wk["entries"]), str(wk["exits"]), str(wk["price_changes"]),
                  str(sum(wk["process_failures"].values())), str(wk["violations_new"]), str(wk["collapsed_days"])]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


__all__ = ["run_long", "inject_shock", "liveness_table", "underserved_sectors", "write_csv", "Disturb"]
