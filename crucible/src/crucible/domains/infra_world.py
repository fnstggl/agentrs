"""Infrastructure operations world (kernel module `fleet`, host `fleet`) for the Tier 2 domain
`engineering_infrastructure` (MASTER_SPEC 9.7).

An operator organisation runs compute sites (data centres) that serve the workloads of independent tenant
organisations. Ground truth is executed physics, no judge:

* **compute and capacity**: every site has leased machines (cores, idle and peak power). A machine is off, booting
  (it draws power for `boot_h` hours before it serves), on, failed, under repair, on order or retired. Interactive
  requests and batch core-hours share the cores that are on; what does not fit is dropped (interactive) or waits
  (batch);
* **workloads**: tenants' interactive services (requests per region and hour from a hidden demand process with a
  daily cycle per region, growth and noise; CPU milliseconds per request that change, unannounced, when a tenant
  releases new software) and batch jobs (core-hours, maximum parallelism, deadline);
* **networking**: regions and sites are nodes of a network whose links have capacity and latency; request traffic
  is routed on the lowest-latency path and the 4.0 engineering-lab flow physics (`worlds.engineering_lab.evaluate`)
  gives link queueing delay and drops. Fibre cuts (where configured) remove links for their repair time;
* **latency and reliability**: a request is good when it is delivered and its network delay plus its M/M/1 sojourn
  time at the site, ``1 - exp(-(slo - d_net)(1 - rho)/s)``, meets the service's SLO;
* **energy**: facility power = IT power (idle + utilisation x dynamic + booting machines) x PUE; PUE is constant or
  rises with outside temperature and utilisation (`thermal`); energy is paid per kWh at a flat, time-of-use or
  spot price (`spot`: an autonomous utility prices from grid load including the operator's own draw, settled after
  the hour);
* **failures**: machines fail with a hazard that grows with accumulated wear (hours on at load, power cycles) and
  with inlet temperature; technicians repair a limited number per day; a site whose facility power exceeds its
  supply rating trips its breaker (`breakers`) and goes dark for `trip_h` hours, otherwise it is throttled;
* **delayed effects**: boot lag, procurement lead time, repair time, wear, deployment at the next maintenance
  window, dashboards published late, tenants reacting to a week of experienced reliability;
* **controllers and software**: production is run by a controller program from a repository
  (`control(obs) -> plan`: machines on per site, placement of each service's regional traffic over sites, batch
  job placement). The incumbent `baseline.py` is trusted and runs in-process with the parameters in the
  repository's `controller.json`; any other program runs in the sandbox, its output recorded for regeneration.
  Tenants deploy new releases of their services, which changes the work they cause.

Autonomous participants: tenants (each decides from its own experienced reliability how much of its traffic it
keeps on this operator, and when to release software), the utility (spot pricing) and the repair crew.
Hidden (internal events only): demand parameters, true CPU cost per request, wear, weather, background grid load
and fibre-cut hazards. The operator sees its own fleet, its own traffic of past hours, noisy CPU telemetry and a
delayed, probe-sampled reliability dashboard.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np

from ..core import schema as S
from ..core.canonical import digest, to_jsonable
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound
from ..core.kernel import Ctx, Module, applier, endpoint, process, register_module
from ..services.identity import ident
from ..worlds.engineering_lab import Network, baseline_route
from ..worlds.engineering_lab import evaluate as net_evaluate

HOUR = 3600
DAY = 86400
OFF, ON, BOOT, FAILED, REPAIR, ORDERED, RETIRED = 0, 1, 2, 3, 4, 5, 6
STATUS = ("off", "on", "booting", "failed", "repairing", "ordered", "retired")
LEASED = (OFF, ON, BOOT, FAILED, REPAIR)
TARGET_RE = r"[A-Za-z_][A-Za-z0-9_./-]{0,119}(:[A-Za-z_][A-Za-z0-9_]{0,63})?"
HARNESS_FILE = "_crucible_fleet_harness.py"
HARNESS_TARGET = "_crucible_fleet_harness:run"
DROP_PENALTY_MS = 250.0

DEFAULTS: dict[str, Any] = {
    "n_sites": 1, "n_regions": 3, "topology": "star",      # star | mesh | ring
    "n_services": 3, "n_tenants": 3,
    "batch": False, "jobs_per_day": [2, 5],
    "tariff": "tou",                                       # flat | tou | spot
    "follow_sun": False,                                   # regional demand peaks at different hours
    "thermal": False, "breakers": False, "wear": True, "link_cuts": False,
    "boot_h": 2, "trip_h": 6, "lead_days": 5,
    "machine": {"cores": 32, "idle_kw": 0.22, "peak_kw": 0.52},
    "fleet_factor": [2.3, 2.8],                            # installed machines / machines needed at the peak
    "power_cap": None,                                     # [lo, hi] x the incumbent's facility power at peak
    "lease_cents_day": 700, "repair_parts_cents": 12_000, "repair_tech_h": 6, "repair_h": 8,
    "technicians": 2, "tech_wage_cents_day": 20_000,
    "price_per_m_cents": [40, 90], "batch_value_cents_per_core_h": 3,
    "sla_target": 0.995, "sla_credit": 0.1,
    "cut_rate_per_day": 0.03,                              # per link (hidden)
    "heat_waves": 0.0,                                     # daily probability that a heat wave starts (hidden)
    "wear_h0": 6e-5, "wear_k": 3.0, "wear_on": 0.0005, "wear_cycle": 0.012,
    "maintenance_hour": 3, "dashboard_delay_h": 2,
    "experiment_fee_cents": 2_000, "experiment_hour_cents": 40, "max_experiment_hours": 72,
    "controller_timeout_s": 5.0,
    "operator": {"handle": "opsco", "name": "OpsCo", "lead": "opsco-lead", "cash": 30_000_000},
    "seat": "agent",
    "spot": {"kappa": 2.5, "bg_share": 0.85},
}

# Parameters of the incumbent controller (its repository's controller.json); the defaults are what production runs
# at genesis: every installed machine on, traffic to the nearest site, batch as soon as it arrives.
BASELINE_PARAMS = {"power": "always_on", "headroom": 1.6, "min_on": 1, "forecast": "day_ago_max",
                   "placement": "nearest", "batch": "asap", "spread": 0.0}

BASELINE_SOURCE = '''"""Incumbent fleet controller (trusted). Parameters come from obs["params"] (controller.json):

power      "always_on": every available machine on | "follow_demand": enough machines for the forecast peak of the
           next boot window times `headroom` (never fewer than `min_on`)
forecast   "last_hour" | "day_ago_max": max of the same hours yesterday and the last hour, per service and region
placement  "nearest": all of a region's traffic to its lowest-latency site | "spread": a share `spread` of it
           split over the other sites in proportion to their cores
batch      "asap": run every job now at the site with most spare cores | "cheapest": run a job only in the hours
           with the lowest price that still meet its deadline (with a 50% safety margin)
"""


def _lat(topology, cut):
    adj = {}
    for l in topology["links"]:
        if l["id"] in cut:
            continue
        adj.setdefault(l["u"], []).append((l["v"], float(l["latency"])))
        adj.setdefault(l["v"], []).append((l["u"], float(l["latency"])))
    return adj


def _dist(adj, src):
    import heapq
    d = {src: 0.0}
    h = [(0.0, src)]
    while h:
        x, u = heapq.heappop(h)
        if x > d.get(u, 1e18):
            continue
        for v, w in sorted(adj.get(u, [])):
            if x + w < d.get(v, 1e18):
                d[v] = x + w
                heapq.heappush(h, (x + w, v))
    return d


def _num(x, lo, hi, dflt):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return dflt
    if x != x:
        return dflt
    return min(hi, max(lo, x))


def control(obs):
    p = obs.get("params") or {}
    power = p.get("power", "always_on")
    headroom = _num(p.get("headroom", 1.6), 1.0, 5.0, 1.6)
    min_on = int(_num(p.get("min_on", 1), 0, 10000, 1))
    spread = _num(p.get("spread", 0.0), 0.0, 1.0, 0.0)
    sites = obs["sites"]
    ids = [s["id"] for s in sites]
    adj = _lat(obs["topology"], set(obs.get("cut_links") or []))
    live = [s for s in sites if not s.get("tripped_until")]
    placement = {}
    for svc in obs["services"]:
        placement[svc["id"]] = {}
        for r in obs["regions"]:
            d = _dist(adj, r)
            cand = sorted((d[s["id"]], s["id"]) for s in live if s["id"] in d)
            if not cand:
                continue
            near = cand[0][1]
            if p.get("placement") == "spread" and len(cand) > 1 and spread > 0:
                others = [c[1] for c in cand[1:]]
                cores = {s["id"]: max(1, s["machines"]["on"] + s["machines"]["booting"] + s["machines"]["off"])
                         for s in sites}
                tot = sum(cores[o] for o in others)
                fr = {near: 1.0 - spread}
                for o in others:
                    fr[o] = spread * cores[o] / tot
                placement[svc["id"]][r] = fr
            else:
                placement[svc["id"]][r] = {near: 1.0}
    on = {}
    for s in sites:
        m = s["machines"]
        avail = m["on"] + m["booting"] + m["off"]
        if power != "follow_demand":
            on[s["id"]] = avail
            continue
        need = 0.0
        for svc in obs["services"]:
            for r in obs["regions"]:
                fr = placement[svc["id"]].get(r, {}).get(s["id"], 0.0)
                if fr <= 0:
                    continue
                last = float((svc.get("rps_last") or {}).get(r, 0.0))
                if p.get("forecast", "day_ago_max") == "last_hour":
                    f = last
                else:
                    f = max([last] + [float(x) for x in (svc.get("rps_next_day_ago") or {}).get(r, [])])
                need += f * fr * float(svc["cpu_ms"]) / 1000.0
        cores = need * headroom
        n = int(-(-cores // max(1, s["cores_per_machine"])))
        on[s["id"]] = max(min_on, min(avail, n))
    batch = {}
    jobs = sorted(obs.get("batch") or [], key=lambda j: j["deadline_hour"])
    for j in jobs:
        if p.get("batch") == "cheapest":
            hours_needed = j["core_hours_left"] / max(1.0, j["cores_max"])
            left = j["deadline_hour"] - obs["hour"]
            if left > 1.5 * hours_needed + 1:
                price = [s["price_cents_kwh"] for s in live if s.get("price_cents_kwh") is not None]
                today = [s.get("tariff_today") for s in live if s.get("tariff_today")]
                if today and price:
                    sched = sorted(today[0])
                    k = max(1, int(-(-1.5 * hours_needed // 1)))
                    cut_price = sched[min(len(sched) - 1, k)]
                    if min(price) > cut_price:
                        continue
        best = max(live, key=lambda s: (s["machines"]["on"] + s["machines"]["booting"], s["id"]), default=None)
        if best is not None:
            batch[j["job"]] = best["id"]
    return {"on": on, "placement": placement, "batch": batch}
'''

HARNESS_SOURCE = '''"""Installed by Crucible next to your files: calls your controller once per observation."""
import importlib
import json
import os
import random
import traceback


def run(job, caller=None):
    mod_name, _, fn_name = job["target"].partition(":")
    if mod_name.endswith(".py"):
        mod_name = mod_name[:-3]
    mod_name = mod_name.replace("/", ".")
    try:
        fn = getattr(importlib.import_module(mod_name), fn_name or "control")
    except BaseException as e:  # noqa: BLE001
        return {"load_error": (type(e).__name__ + ": " + str(e).replace(os.getcwd() + os.sep, ""))[:300], "outputs": []}
    out = []
    for obs in job["observations"]:
        random.seed(obs.get("hour", 0))
        try:
            plan = fn(obs)
            json.dumps(plan)
            out.append({"ok": True, "out": plan})
        except BaseException as e:  # noqa: BLE001
            tb = traceback.extract_tb(e.__traceback__)
            where = " at %s:%d" % (os.path.basename(tb[-1].filename), tb[-1].lineno) if tb else ""
            out.append({"ok": False, "error": (type(e).__name__ + where)[:300]})
    return {"outputs": out}
'''

README = """# Fleet controllers

Production runs `control(obs) -> plan` once per hour (see `baseline.py` for the observation fields):

    plan = {"on": {site: machines_on},
            "placement": {service: {region: {site: fraction}}},
            "batch": {job: site}}

* `on`: machines to keep on (or booting) at each site; machines switched on serve only after `boot_h` hours and
  every power cycle wears a machine;
* `placement`: how each region's requests for each service are split over sites (fractions are normalised; a
  missing entry sends the region to its nearest live site);
* `batch`: which jobs run this hour and where (a job not listed waits).

`baseline.py` is trusted and runs in-process with `controller.json` as `obs["params"]`; any other program runs in
the sandbox. Test candidates with `fleet.experiment`, deploy with `fleet.propose_change` (applied at the next
maintenance window).
"""

_COMPILED: dict[str, Any] = {}


def baseline_control(obs: dict) -> dict:
    fn = _COMPILED.get("baseline")
    if fn is None:
        ns: dict = {"__name__": "fleet_baseline"}
        exec(compile(BASELINE_SOURCE, "<fleet-baseline>", "exec"), ns)  # kernel-authored reference code
        fn = _COMPILED["baseline"] = ns["control"]
    return fn(obs)


# ============================================================================ physics (pure functions)
def sojourn_ok(slo_ms: float, net_ms: float, svc_ms: float, rho: float) -> float:
    """P(network delay + M/M/1 sojourn <= SLO): the sojourn time is exponential with mean svc_ms / (1 - rho)."""
    budget = slo_ms - net_ms
    if budget <= 0:
        return 0.0
    rho = min(max(rho, 0.0), 0.99)
    return 1.0 - math.exp(-budget * (1.0 - rho) / max(1e-6, svc_ms))


def pue(site: dict, temp_c: float, util: float, thermal: bool) -> float:
    if not thermal:
        return float(site["pue0"])
    return float(site["pue0"]) + 0.014 * max(0.0, temp_c - 18.0) + 0.06 * util


def inlet_temp(temp_c: float, util: float, thermal: bool) -> float:
    return 20.0 + (0.28 * max(0.0, temp_c - 18.0) if thermal else 0.0) + 8.0 * util


def topology_without(topology: dict, cut: list[str]) -> dict:
    cut = set(cut or [])
    return {"nodes": list(topology["nodes"]), "links": [dict(l) for l in topology["links"] if l["id"] not in cut]}


def normalise_plan(conf: dict, plan: Any, counts: dict, live: list[str], net: Network) -> tuple[dict, list[str]]:
    """Validate a controller plan against the fleet. Returns (plan, errors); invalid parts fall back: machines on
    unchanged, a region's traffic to its nearest live site, unlisted batch jobs wait."""
    errs: list[str] = []
    plan = plan if isinstance(plan, dict) else {}
    if not isinstance(plan, dict) or not plan:
        errs.append("plan must be a dict with on/placement/batch")
    sites = [s["id"] for s in conf["sites"]]
    on = {}
    raw_on = plan.get("on") if isinstance(plan.get("on"), dict) else {}
    for s in sites:
        c = counts[s]
        avail = c["on"] + c["booting"] + c["off"]
        v = raw_on.get(s, c["on"] + c["booting"])
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v != v:
            errs.append(f"on[{s}] is not a number")
            v = c["on"] + c["booting"]
        on[s] = int(min(avail, max(0, int(v))))
    nearest = {}
    for r in conf["regions"]:
        best = None
        for s in live:
            d = _path_latency(net, r, s)
            if d is not None and (best is None or (d, s) < best):
                best = (d, s)
        nearest[r] = best[1] if best else None
    raw_pl = plan.get("placement") if isinstance(plan.get("placement"), dict) else {}
    placement = {}
    for svc in conf["services"]:
        sid = svc["id"]
        placement[sid] = {}
        rp = raw_pl.get(sid) if isinstance(raw_pl.get(sid), dict) else {}
        for r in conf["regions"]:
            fr = rp.get(r)
            out = {}
            if isinstance(fr, dict):
                for s, x in fr.items():
                    if s in sites and isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and x > 0:
                        out[s] = float(x)
            tot = sum(out.values())
            if tot <= 0:
                if fr is not None:
                    errs.append(f"placement[{sid}][{r}] invalid")
                out = {nearest[r]: 1.0} if nearest[r] else {}
                tot = 1.0
            placement[sid][r] = {s: round(x / tot, 6) for s, x in sorted(out.items())}
    raw_b = plan.get("batch") if isinstance(plan.get("batch"), dict) else {}
    batch = {str(j): s for j, s in raw_b.items() if isinstance(s, str) and s in sites}
    return {"on": on, "placement": placement, "batch": batch}, errs[:5]


def _path_latency(net: Network, src: str, dst: str) -> float | None:
    if src == dst:
        return 0.0
    r = baseline_route(net.topology, [{"id": "x", "src": src, "dst": dst, "volume": 0.0}])["routes"].get("x")
    if not r:
        return None
    path = r[0][0]
    return sum(float(net.links[net.adj[(a, b)]]["latency"]) for a, b in zip(path, path[1:]))


def hour_physics(conf: dict, counts: dict, plan: dict, exo: dict, jobs: list[dict], net: Network) -> dict:
    """One hour of the fleet. `counts[site]` = {on, booting} after the plan's power changes (booting machines draw
    idle power and do not serve), `exo` = {demand: {svc: {region: rps}}, cpu_ms: {svc}, temp: {site},
    price: {site} (None for spot: settled afterwards), tripped: [sites]}. Returns exact metrics."""
    mach = conf["machine"]
    thermal = bool(conf["thermal"])
    sites = {s["id"]: s for s in conf["sites"]}
    tripped = set(exo.get("tripped") or [])
    # network volumes
    vol: dict[tuple[str, str], float] = {}
    reqs: dict[tuple[str, str, str], float] = {}
    for svc in conf["services"]:
        sid = svc["id"]
        for r, rps in exo["demand"][sid].items():
            for s, fr in plan["placement"][sid].get(r, {}).items():
                x = float(rps) * fr
                if x <= 0:
                    continue
                reqs[(sid, r, s)] = reqs.get((sid, r, s), 0.0) + x
                vol[(r, s)] = vol.get((r, s), 0.0) + x * float(svc["kb"]) * 8e-6     # Gbit/s
    demands = [{"id": f"{r}>{s}", "src": r, "dst": s, "volume": v} for (r, s), v in sorted(vol.items()) if r != s]
    netres = net_evaluate(net, demands, baseline_route(net.topology, demands), 1.0, DROP_PENALTY_MS) if demands else None
    nd: dict[str, tuple[float, float]] = {}
    if netres:
        for did, (v, cost, dropped) in netres["per_demand"].items():
            dv = v - dropped
            delay = (cost - dropped * DROP_PENALTY_MS) / dv if dv > 1e-12 else DROP_PENALTY_MS
            nd[did] = (dropped / v if v > 0 else 0.0, delay)
    # interactive load per site (core-hours of what arrives)
    arrive: dict[tuple[str, str, str], float] = {}
    load = {s: 0.0 for s in sites}
    for (sid, r, s), x in reqs.items():
        drop, _ = nd.get(f"{r}>{s}", (0.0, 0.0))
        a = x * (1.0 - drop)
        arrive[(sid, r, s)] = a
        load[s] += a * 3600.0 * float(exo["cpu_ms"][sid]) / 3.6e6
    out_sites = {}
    served_frac = {}
    batch_done: dict[str, float] = {}
    for s, site in sites.items():
        c = counts[s]
        n_on = 0 if s in tripped else int(c["on"])
        n_boot = 0 if s in tripped else int(c["booting"])
        cap = float(n_on * mach["cores"])
        throttle = 1.0
        for _ in range(2):            # second pass only when a power cap throttles the site
            capt = cap * throttle
            f = min(1.0, capt / load[s]) if load[s] > 0 else 1.0
            inter = min(load[s], capt)
            free = max(0.0, 0.95 * capt - inter)
            bcores = 0.0
            done: dict[str, float] = {}
            for j in sorted((j for j in jobs if plan["batch"].get(j["job"]) == s), key=lambda j: (j["deadline_hour"], j["job"])):
                g = min(float(j["cores_max"]), float(j["core_hours_left"]), free - bcores)
                if g <= 0:
                    continue
                done[j["job"]] = g
                bcores += g
            util = (inter + bcores) / capt if capt > 0 else 0.0
            it_kw = (n_on + n_boot) * mach["idle_kw"] + util * n_on * (mach["peak_kw"] - mach["idle_kw"]) * throttle
            p = pue(site, exo["temp"][s], util, thermal)
            kw = it_kw * p
            capkw = site.get("power_cap_kw")
            if capkw and kw > capkw and not conf["breakers"] and throttle == 1.0 and kw > 0:
                throttle = max(0.05, capkw / kw)
                continue
            break
        trip = bool(capkw and kw > capkw * 1.0001 and conf["breakers"])
        if trip:                          # the breaker opens half-way through the hour
            f *= 0.5
            done = {k: v * 0.5 for k, v in done.items()}
        rho = min(0.99, (inter / capt if capt > 0 else 0.99) + 0.3 * (bcores / capt if capt > 0 else 0.0))
        served_frac[s] = (f, rho)
        batch_done.update(done)
        out_sites[s] = {"on": n_on, "booting": n_boot, "cores": cap, "load_core_h": round(load[s], 4),
                        "util": round(util, 5), "throttle": round(throttle, 4), "it_kw": round(it_kw, 4),
                        "pue": round(p, 4), "kw": round(kw, 4), "kwh": round(kw, 4), "trip": trip,
                        "t_in": round(inlet_temp(exo["temp"][s], util, thermal), 3),
                        "batch_core_h": round(sum(done.values()), 4), "served_frac": round(f, 5)}
    out_svc = {}
    for svc in conf["services"]:
        sid = svc["id"]
        offered = sum(float(v) for v in exo["demand"][sid].values()) * 3600.0
        good = delivered = 0.0
        for (sv, r, s), x in reqs.items():
            if sv != sid:
                continue
            drop, delay = nd.get(f"{r}>{s}", (0.0, 0.0))
            f, rho = served_frac[s]
            d = x * 3600.0 * (1.0 - drop) * f
            delivered += d
            good += d * sojourn_ok(float(svc["slo_ms"]), delay, float(svc["service_ms"]), rho)
        unplaced = offered - sum(x * 3600.0 for (sv, _, _), x in reqs.items() if sv == sid)
        out_svc[sid] = {"offered": round(offered, 2), "delivered": round(delivered, 2), "good": round(good, 2),
                        "unplaced": round(max(0.0, unplaced), 2)}
    net_out = {"max_util": round(netres["max_util"], 5) if netres else 0.0,
               "dropped_fraction": round(netres["dropped_fraction"], 6) if netres else 0.0,
               "volume_gbps": round(netres["volume"], 5) if netres else 0.0}
    return {"sites": out_sites, "services": out_svc, "batch": {k: round(v, 4) for k, v in sorted(batch_done.items())},
            "network": net_out}


# ============================================================================ generators (seeded, at genesis)
def _topology(rng, cfg: dict, regions: list[str], sites: list[dict]) -> dict:
    nodes = list(regions) + [s["id"] for s in sites]
    links = []

    def add(u, v, cap, lat):
        a, b = sorted([u, v])
        lid = f"{a}-{b}"
        if any(l["id"] == lid for l in links):
            return
        links.append({"id": lid, "u": a, "v": b, "capacity": round(float(cap), 3), "latency": round(float(lat), 2)})
    kind = cfg["topology"]
    if kind == "star":
        nodes.append("core")
        for r in regions:
            add(r, "core", rng.uniform(8, 14), rng.uniform(6, 22))
        for s in sites:
            add(s["id"], "core", rng.uniform(25, 40), rng.uniform(1, 3))
    elif kind == "mesh":
        for i, s in enumerate(sites):
            for t in sites[i + 1:]:
                add(s["id"], t["id"], rng.uniform(8, 14), rng.uniform(18, 40))
        for i, r in enumerate(regions):
            home = sites[i % len(sites)]["id"]
            add(r, home, rng.uniform(6, 10), rng.uniform(2, 6))
            if len(sites) > 1:
                other = sites[(i + 1) % len(sites)]["id"]
                add(r, other, rng.uniform(1.5, 3.0), rng.uniform(25, 45))
    else:   # ring
        ring = []
        for i in range(max(len(regions), len(sites))):
            if i < len(regions):
                ring.append(regions[i])
            if i < len(sites):
                ring.append(sites[i]["id"])
        for a, b in zip(ring, ring[1:] + ring[:1]):
            add(a, b, rng.uniform(6, 10), rng.uniform(5, 15))
        if len(ring) > 3:
            add(ring[0], ring[len(ring) // 2], rng.uniform(2, 4), rng.uniform(20, 35))
    return {"nodes": nodes, "links": links}


def make_world(rng, cfg: dict) -> dict:
    """Public configuration and hidden regime of one world, drawn from the module config."""
    R = int(cfg["n_regions"])
    S_ = int(cfg["n_sites"])
    regions = [f"r{i}" for i in range(R)]
    sites = []
    for i in range(S_):
        sites.append({"id": f"s{i}", "region": regions[i % R], "pue0": round(float(rng.uniform(1.18, 1.38)), 3),
                      "base_price": round(float(rng.uniform(9.0, 16.0)), 3),
                      "temp_mean": round(float(rng.uniform(12.0, 24.0)), 2)})
    topo = _topology(rng, cfg, regions, sites)
    tenants = [f"tenant{k + 1}" for k in range(int(cfg["n_tenants"]))]
    services, hidden_svc = [], {}
    for k in range(int(cfg["n_services"])):
        sid = f"svc{k}"
        services.append({"id": sid, "tenant": tenants[k % len(tenants)],
                         "slo_ms": int(rng.integers(70, 140)), "service_ms": round(float(rng.uniform(2.0, 6.0)), 3),
                         "kb": round(float(rng.uniform(20, 120)), 2),
                         "price_per_m_cents": int(rng.integers(*cfg["price_per_m_cents"]))})
        base = {r: round(float(rng.uniform(150, 900)), 2) for r in regions}
        phase = {r: (int(rng.integers(0, 24)) if cfg["follow_sun"] else 14 + int(rng.integers(-2, 3))) for r in regions}
        hidden_svc[sid] = {"base": base, "peak_hour": phase, "amp": round(float(rng.uniform(0.45, 0.75)), 3),
                           "growth": round(float(rng.uniform(-0.002, 0.012)), 5),
                           "sigma": round(float(rng.uniform(0.05, 0.12)), 4),
                           "cpu_ms": round(float(rng.uniform(6.0, 22.0)), 4),
                           "release_every": int(rng.integers(4, 10)), "release_sigma": 0.18}
    return {"regions": regions, "sites": sites, "topology": topo, "services": services, "tenants": tenants,
            "hidden": {"services": hidden_svc,
                       "bg": {"amp": round(float(rng.uniform(0.2, 0.35)), 3), "sigma": 0.05},
                       "weather": {s["id"]: {"amp": round(float(rng.uniform(4, 9)), 3)} for s in sites}}}


def demand_rps(hid: dict, region: str, hour: int, mult: float, rng=None) -> float:
    """Hidden demand of one service in one region during `hour` (rps)."""
    day = hour / 24.0
    hod = hour % 24
    ph = hid["peak_hour"][region]
    cyc = 1.0 + hid["amp"] * math.cos(2 * math.pi * (hod - ph) / 24.0)
    g = math.exp(hid["growth"] * day)
    x = hid["base"][region] * cyc * g * mult
    if rng is not None:
        x *= math.exp(float(rng.normal(0.0, hid["sigma"])))
    return max(0.0, x)


def tariff(site: dict, kind: str, hod: int) -> float:
    if kind == "tou":
        f = 0.65 if hod < 7 else (1.7 if 17 <= hod <= 20 else 1.0)
        return site["base_price"] * f
    return site["base_price"]


# ============================================================================ the module
@register_module
class Fleet(Module):
    name = "fleet"
    host = "fleet"
    depends = ("identity", "bank", "code")

    def __init__(self, config=None):
        super().__init__(config)
        cfg = json.loads(json.dumps(DEFAULTS))
        for k, v in self.config.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k] = dict(cfg[k], **v)
            else:
                cfg[k] = v
        self.cfg = cfg

    def install(self, world):
        st = world.state
        st.declare_table("fleet")                       # conf, regime, deployed, pending, totals, tripped, cuts
        st.declare_table("fleet_hours")                 # exact production hour records (internal)
        st.declare_table("fleet_days")                  # daily accounting (internal)
        st.declare_table("fleet_jobs", index=("status",))
        st.declare_table("fleet_changes", index=("by",))
        st.declare_table("fleet_releases", index=("service",))
        st.declare_table("fleet_orders", index=("status",))
        st.declare_table("fleet_experiments", index=("by",))
        prev = world.hooks.get("on_join")

        def on_join(ctx, pid, kind):
            if prev is not None:
                prev(ctx, pid, kind)
            self._seat(ctx, pid)
        world.hooks["on_join"] = on_join

    def on_load(self, world):
        world.runtime.pop("fleet_net", None)

    # ------------------------------------------------------------------ helpers
    def conf(self, st) -> dict:
        c = st.get("fleet", "conf")
        if c is None:
            raise NotFound("no fleet in this world")
        return c

    def regime(self, st) -> dict:
        return st.get("fleet", "regime")["params"]

    def net(self, world, cut: tuple = ()) -> Network:
        cache = world.runtime.setdefault("fleet_net", {})
        key = tuple(sorted(cut))
        if key not in cache:
            topo = to_jsonable(self.conf(world.state)["topology"])
            cache[key] = Network(topology_without(topo, list(key)))
        return cache[key]

    def operates(self, ctx, actor=None) -> bool:
        conf = self.conf(ctx.state)
        return ident(ctx.world).allowed(ctx, actor or ctx.actor, conf["operator"], "fleet.operate")

    def _require_operator(self, ctx) -> dict:
        conf = self.conf(ctx.state)
        ident(ctx.world).authorize(ctx, conf["operator"], "fleet.operate")
        return conf

    def _seat(self, ctx: Ctx, pid: str) -> None:
        seat = self.cfg.get("seat")
        if not seat or pid != f"p:{seat}" or not ctx.state.has("fleet", "conf"):
            return
        conf = self.conf(ctx.state)
        ident(ctx.world).add_member(ctx, conf["operator"], pid, "admin", "infrastructure engineer")

    def counts(self, st) -> dict:
        conf = self.conf(st)
        site = st.arrays["fl_site"]
        status = st.arrays["fl_status"]
        out = {}
        for i, s in enumerate(conf["sites"]):
            m = site == i
            out[s["id"]] = {STATUS[k]: int(np.sum(m & (status == k))) for k in range(len(STATUS))}
        return out

    def tripped(self, st, h: int) -> list[str]:
        t = (st.get("fleet", "tripped") or {}).get("until", {})
        return sorted(s for s, u in t.items() if u > h)

    def cut_links(self, st, h: int) -> list[str]:
        c = (st.get("fleet", "cuts") or {}).get("until", {})
        return sorted(l for l, u in c.items() if u > h)

    def _next_maintenance(self, st) -> int | None:
        j = st.get("sched", "fleet_maint")
        return j["at"] if j else None

    def demand_mult(self, st, sid: str) -> float:
        """Tenant's share of its traffic kept here (its own decision)."""
        return float((st.get("fleet", "shares") or {}).get("share", {}).get(sid, 1.0))

    # ------------------------------------------------------------------ genesis
    def genesis(self, ctx):
        cfg = self.cfg
        w = ctx.world
        idm, bank, code = ident(w), w.modules["bank"], w.modules["code"]
        rng = w.rng_for("fleet-genesis")
        g = make_world(rng, cfg)
        op = cfg["operator"]
        lead = idm.create_person(ctx, op["lead"], f"{op['name']} operations lead", "npc", {"role": "operator"})
        bank.open_account(ctx, lead, "checking")
        org = idm.create_org(ctx, op["handle"], op["name"], lead, "company", {"fleet": True})
        acct = bank.open_account(ctx, org, "operating")
        bank.move(ctx, "acct_genesis", acct, int(op["cash"]), "endowment", f"initial capital of {org}")
        util_lead = idm.create_person(ctx, "gridpower-ceo", "GridPower CEO", "npc", {"role": "utility"})
        utility = idm.create_org(ctx, "gridpower", "GridPower", util_lead, "company", {"utility": True})
        uacct = bank.open_account(ctx, utility, "operating")
        vend_lead = idm.create_person(ctx, "rackline-ceo", "RackLine CEO", "npc", {"role": "vendor"})
        vendor = idm.create_org(ctx, "rackline", "RackLine leasing", vend_lead, "company", {"vendor": True})
        vacct = bank.open_account(ctx, vendor, "operating")
        tenants = {}
        for tname in g["tenants"]:
            tl = idm.create_person(ctx, f"{tname}-cto", f"{tname} CTO", "npc", {"role": "tenant"})
            bank.open_account(ctx, tl, "checking")
            to = idm.create_org(ctx, tname, tname.title(), tl, "company", {"tenant": True})
            ta = bank.open_account(ctx, to, "operating")
            bank.move(ctx, "acct_genesis", ta, 5_000_000, "endowment", f"initial capital of {to}")
            tenants[tname] = {"org": to, "lead": tl, "account": ta,
                              "tolerance": round(float(rng.uniform(0.97, 0.995)), 4),
                              "patience": round(float(rng.uniform(0.15, 0.45)), 3),
                              "value_per_m_cents": int(rng.integers(200, 400))}
        for svc in g["services"]:
            svc["tenant_org"] = tenants[svc["tenant"]]["org"]
        # fleet sizing from the hidden peak (what the incumbent provisioned for)
        mach = cfg["machine"]
        hid = g["hidden"]["services"]
        need = {s["id"]: 0.0 for s in g["sites"]}
        net = Network(g["topology"])
        for svc in g["services"]:
            for r in g["regions"]:
                best = min(((_path_latency(net, r, s["id"]) or 1e9, s["id"]) for s in g["sites"]))[1]
                peak = hid[svc["id"]]["base"][r] * (1 + hid[svc["id"]]["amp"])
                need[best] += peak * hid[svc["id"]]["cpu_ms"] / 1000.0
        jobs_core_h = 0.0
        if cfg["batch"]:
            jobs_core_h = float(np.mean(cfg["jobs_per_day"])) * 900.0 / 24.0 / max(1, len(g["sites"]))
        m_site = []
        for i, s in enumerate(g["sites"]):
            n_need = (need[s["id"]] + jobs_core_h) / (0.65 * mach["cores"])
            n = max(4, int(math.ceil(n_need * float(rng.uniform(*cfg["fleet_factor"])))))
            s["installed"] = n
            if cfg.get("power_cap"):
                kw = n * (mach["idle_kw"] + 0.45 * (mach["peak_kw"] - mach["idle_kw"])) * \
                    pue(s, s["temp_mean"] + 2.0, 0.45, cfg["thermal"])
                s["power_cap_kw"] = round(kw * float(rng.uniform(*cfg["power_cap"])), 3)
            else:
                s["power_cap_kw"] = None
            m_site += [i] * n
        wear0 = [round(float(x), 4) for x in rng.uniform(0.02, 0.25, len(m_site))]
        public_sites = [{k: s[k] for k in ("id", "region", "pue0", "power_cap_kw", "installed")}
                        | ({"base_price": s["base_price"]} if cfg["tariff"] != "spot" else {}) for s in g["sites"]]
        rid = code.create_repo(ctx, org, "controllers", "private", f"Fleet controllers of {op['name']}", "closed")
        commit = code.commit_files(ctx, rid, {"baseline.py": BASELINE_SOURCE, "README.md": README,
                                              "controller.json": json.dumps(BASELINE_PARAMS, indent=1)},
                                   "incumbent fleet controller", lead, trusted=True)
        ctx.emit("fleet.conf", {
            "operator": org, "lead": lead, "account": acct, "utility": utility, "utility_account": uacct,
            "vendor": vendor, "vendor_account": vacct, "repo": rid, "baseline_ref": commit,
            "regions": g["regions"], "sites": public_sites, "topology": g["topology"],
            "services": [{k: s[k] for k in ("id", "tenant", "tenant_org", "slo_ms", "service_ms", "kb",
                                            "price_per_m_cents")} for s in g["services"]],
            "machine": dict(mach), "thermal": bool(cfg["thermal"]), "breakers": bool(cfg["breakers"]),
            "tariff": cfg["tariff"], "batch": bool(cfg["batch"]), "boot_h": int(cfg["boot_h"]),
            "trip_h": int(cfg["trip_h"]), "lead_days": int(cfg["lead_days"]),
            "lease_cents_day": int(cfg["lease_cents_day"]), "repair_parts_cents": int(cfg["repair_parts_cents"]),
            "tech_wage_cents_day": int(cfg["tech_wage_cents_day"]), "technicians": int(cfg["technicians"]),
            "maintenance_hour": int(cfg["maintenance_hour"]) % 24, "dashboard_delay_h": int(cfg["dashboard_delay_h"]),
            "sla_target": float(cfg["sla_target"]), "experiment_fee_cents": int(cfg["experiment_fee_cents"]),
            "experiment_hour_cents": int(cfg["experiment_hour_cents"]),
            "max_experiment_hours": int(cfg["max_experiment_hours"])}, vis="public")
        hidden = dict(g["hidden"])
        hidden["sites"] = {s["id"]: {"base_price": s["base_price"], "temp_mean": s["temp_mean"],
                                     "pue0": s["pue0"]} for s in g["sites"]}
        hidden["tenants"] = tenants
        hidden["grid_cap_kw"] = round(sum(
            s["installed"] * mach["peak_kw"] * s["pue0"] for s in g["sites"]) / max(1e-6, 1 - cfg["spot"]["bg_share"]), 3)
        ctx.emit("fleet.regime", {"day": 0, "cause": "genesis", "params": hidden}, vis="internal")
        ctx.emit("fleet.machines_init", {"site": m_site, "wear": wear0}, vis="internal")
        ctx.emit("fleet.deployed", {"id": "dep_genesis", "repo": rid, "ref": commit, "target": "baseline.py:control",
                                    "by": org, "principal": lead, "change": None}, vis="public")
        docs = w.modules.get("docs")
        if docs is not None:
            docs.create_doc(ctx, org, f"{op['name']} fleet operations", README, "public")
        mh = int(cfg["maintenance_hour"]) % 24
        ctx.schedule(HOUR, "fleet.hour", {}, prio=40, period=HOUR, key="fleet_hour")
        ctx.schedule(DAY + 60, "fleet.daily", {}, prio=45, period=DAY, key="fleet_daily")
        ctx.schedule(mh * HOUR + 30 if mh else DAY + 30, "fleet.maintenance", {}, prio=60, period=DAY, key="fleet_maint")
        if cfg["batch"]:
            ctx.schedule(30, "fleet.jobs", {}, prio=44, period=DAY, key="fleet_jobs")

    # ------------------------------------------------------------------ exogenous inputs of an hour (hidden)
    def exogenous(self, st, h: int) -> dict:
        conf = self.conf(st)
        reg = self.regime(st)
        rng = self.world.rng_for("fleet-demand", h)
        demand, cpu = {}, {}
        for svc in conf["services"]:
            hid = reg["services"][svc["id"]]
            mult = self.demand_mult(st, svc["id"])
            demand[svc["id"]] = {r: round(demand_rps(hid, r, h, mult, rng), 3) for r in conf["regions"]}
            cpu[svc["id"]] = float(hid["cpu_ms"])
        wr = self.world.rng_for("fleet-weather", h)
        heat = (st.get("fleet", "heat") or {}).get("until", {})
        temp = {}
        for s in conf["sites"]:
            sh = reg["sites"][s["id"]]
            amp = reg["weather"][s["id"]]["amp"]
            t = sh["temp_mean"] + amp * math.cos(2 * math.pi * ((h % 24) - 15) / 24.0) + float(wr.normal(0, 1.2))
            if heat.get(s["id"], -1) > h:
                t += 11.0
            temp[s["id"]] = round(t, 3)
        price = {}
        for s in conf["sites"]:
            sh = reg["sites"][s["id"]]
            price[s["id"]] = None if conf["tariff"] == "spot" else round(tariff(sh, conf["tariff"], h % 24), 4)
        return {"demand": demand, "cpu_ms": cpu, "temp": temp, "price": price, "tripped": self.tripped(st, h),
                "cut": self.cut_links(st, h)}

    def spot_price(self, st, h: int, our_kw: float) -> float:
        """The utility's settlement price for an hour: background grid load (hidden daily cycle and noise) plus the
        operator's own draw, priced convexly in the grid's utilisation."""
        reg = self.regime(st)
        bg = reg["bg"]
        cap = float(reg["grid_cap_kw"])
        share = float(self.cfg["spot"]["bg_share"])
        rng = self.world.rng_for("fleet-grid", h)
        load_bg = cap * share * 0.75 * (1 + bg["amp"] * math.cos(2 * math.pi * ((h % 24) - 18) / 24.0)) * \
            math.exp(float(rng.normal(0, bg["sigma"])))
        u = (load_bg + our_kw) / cap
        base = float(np.mean([s["base_price"] for s in reg["sites"].values()]))
        return round(base * (0.6 + float(self.cfg["spot"]["kappa"]) * u * u), 4)

    # ------------------------------------------------------------------ the controller's observation
    def observation(self, st, h: int, counts: dict, exo: dict, params: dict | None) -> dict:
        conf = self.conf(st)
        rows = {}
        for k in (1, 2, 24, 23, 22):
            r = st.get("fleet_hours", f"h{h - k:06d}")
            if r is not None:
                rows[k] = r
        last = rows.get(1)
        sites = []
        for s in conf["sites"]:
            c = counts[s["id"]]
            ls = last["exact"]["sites"][s["id"]] if last else None
            today = None if conf["tariff"] == "spot" else [round(tariff(self.regime(st)["sites"][s["id"]],
                                                                         conf["tariff"], k), 4) for k in range(24)]
            sites.append({"id": s["id"], "region": s["region"], "cores_per_machine": conf["machine"]["cores"],
                          "machines": {k: c[k] for k in ("on", "booting", "off", "failed", "repairing")},
                          "power_cap_kw": s["power_cap_kw"], "tripped_until": (st.get("fleet", "tripped") or {}).get(
                              "until", {}).get(s["id"]) if s["id"] in exo["tripped"] else None,
                          "price_cents_kwh": exo["price"][s["id"]] if exo["price"][s["id"]] is not None else
                          (last["price"].get(s["id"]) if last else None),
                          "tariff_today": today,
                          "temp_c": last["temp"].get(s["id"]) if last else None,
                          "last": {k: ls[k] for k in ("util", "kw", "pue")} if ls else None})
        services = []
        for svc in conf["services"]:
            sid = svc["id"]
            m = last["measured"]["cpu_ms"].get(sid) if last else None
            services.append({"id": sid, "slo_ms": svc["slo_ms"], "service_ms": svc["service_ms"], "kb": svc["kb"],
                             "cpu_ms": m if m else 10.0,
                             "rps_last": dict(last["rps"][sid]) if last else {},
                             "rps_next_day_ago": {r: [rows[k]["rps"][sid][r] for k in (24, 23, 22) if k in rows]
                                                  for r in conf["regions"]}})
        jobs = []
        for jid in st.find_ids("fleet_jobs", "status", "open"):
            j = st.tables["fleet_jobs"][jid]
            jobs.append({"job": jid, "core_hours_left": j["left"], "cores_max": j["cores_max"],
                         "deadline_hour": j["deadline_hour"]})
        return {"hour": h, "hour_of_day": h % 24, "day": h // 24, "boot_h": conf["boot_h"],
                "params": params or {}, "sites": sites, "services": services, "regions": list(conf["regions"]),
                "topology": to_jsonable(conf["topology"]), "cut_links": exo["cut"], "batch": jobs}

    def _params(self, st, repo: str, ref: str) -> dict | None:
        code = self.world.modules["code"]
        try:
            files = code.files(st, repo, ref)
        except CrucibleError:
            return None
        raw = files.get("controller.json")
        if raw is None:
            return {}
        try:
            p = json.loads(raw)
        except ValueError:
            return {}
        return p if isinstance(p, dict) else {}

    def _trusted(self, st, dep: dict) -> bool:
        """The incumbent program, unchanged (its parameters may differ): it runs in-process."""
        code = self.world.modules["code"]
        mod = dep["target"].partition(":")[0]
        fn = dep["target"].partition(":")[2] or "control"
        path = mod if mod.endswith(".py") else mod.replace(".", "/") + ".py"
        try:
            files = code.files(st, dep["repo"], dep["ref"])
        except CrucibleError:
            return False
        return files.get(path) == BASELINE_SOURCE and fn == "control"

    def run_controller(self, ctx: Ctx, dep: dict, observations: list[dict], principal: str, recorded: str) -> list:
        """Plans for a list of observations (one sandbox job). Returns [{"ok", "out"|"error"}]."""
        st = ctx.state
        if self._trusted(st, dep):
            params = self._params(st, dep["repo"], dep["ref"]) or {}
            outs = []
            for o in observations:
                o = dict(o, params=params)
                try:
                    outs.append({"ok": True, "out": to_jsonable(baseline_control(o))})
                except Exception as e:  # noqa: BLE001 - bad parameters in controller.json
                    outs.append({"ok": False, "error": type(e).__name__})
            return outs
        code = ctx.world.modules["code"]
        if not code.can_read(ctx, dep["repo"], principal):
            raise Forbidden(f"{principal} cannot read {dep['repo']}")
        files = dict(code.files_with_deps(ctx, dep["repo"], dep["ref"], principal))
        files[HARNESS_FILE] = HARNESS_SOURCE
        pool = code.sandbox_pool(ctx.world)
        key = f"{ctx.world.world_id}|fleet|{dep['repo']}|{dep['ref']}"
        job = {"kind": "call", "target": HARNESS_TARGET, "caller": principal,
               "args": {"target": dep["target"], "observations": observations}}
        timeout = min(60.0, float(self.cfg["controller_timeout_s"]) * max(1, len(observations)) + 2.0)

        def go():
            w = pool.get(key, lambda: files)
            r = w.run(job, rpc=None, timeout=timeout)
            res = r.get("result") if isinstance(r.get("result"), dict) else {}
            return {"ok": bool(r.get("ok")), "result": res,
                    "cpu_ms": float(r.get("work_lines") or 0) * 0.001 if recorded == "cmd" else 0.0}
        res = ctx.external("fleet.controller", go)
        result = res.get("result") or {}
        raw = result.get("outputs") if isinstance(result.get("outputs"), list) else []
        if result.get("load_error"):
            return [{"ok": False, "error": str(result["load_error"])[:300]} for _ in observations]
        return [raw[i] if i < len(raw) and isinstance(raw[i], dict) else {"ok": False, "error": "no output"}
                for i in range(len(observations))]

    # ------------------------------------------------------------------ power transitions
    def transitions(self, st, plan_on: dict, h: int) -> tuple[list[int], list[int]]:
        """Machines to switch on (lowest wear first) and off (booting first, then highest wear)."""
        conf = self.conf(st)
        site = st.arrays["fl_site"]
        status = st.arrays["fl_status"]
        wear = st.arrays["fl_wear"]
        tripped = set(self.tripped(st, h))
        up, down = [], []
        for i, s in enumerate(conf["sites"]):
            m = site == i
            on = np.flatnonzero(m & (status == ON))
            boot = np.flatnonzero(m & (status == BOOT))
            off = np.flatnonzero(m & (status == OFF))
            if s["id"] in tripped:
                down += on.tolist() + boot.tolist()
                continue
            target = int(plan_on.get(s["id"], len(on) + len(boot)))
            cur = len(on) + len(boot)
            if target > cur:
                order = off[np.lexsort((off, wear[off]))]
                up += order[: target - cur].tolist()
            elif target < cur:
                k = cur - target
                b = boot[::-1][:k].tolist()
                k -= len(b)
                o = on[np.lexsort((on, -wear[on]))][:k].tolist() if k > 0 else []
                down += b + o
        return sorted(up), sorted(down)

    # ------------------------------------------------------------------ the production hour
    @process("hour")
    def pr_hour(self, ctx):
        st = ctx.state
        h = ctx.now // HOUR - 1
        if h < 0 or not st.has("fleet", "conf"):
            return
        conf = self.conf(st)
        cfg = self.cfg
        status = st.arrays["fl_status"]
        ready = st.arrays["fl_ready"]
        became_on = np.flatnonzero((status == BOOT) & (ready <= h)).tolist()
        repaired = np.flatnonzero((status == REPAIR) & (ready <= h)).tolist()
        if became_on or repaired:
            ctx.emit("fleet.ready", {"on": became_on, "off": repaired, "hour": h}, vis="internal")
        exo = self.exogenous(st, h)
        dep = st.get("fleet", "deployed")
        counts0 = self.counts(st)
        obs = self.observation(st, h, counts0, exo, None)
        try:
            out = self.run_controller(ctx, dep, [obs], dep["principal"], "proc")[0]
        except CrucibleError as e:
            out = {"ok": False, "error": e.message[:200]}
        live = [s["id"] for s in conf["sites"] if s["id"] not in exo["tripped"]]
        net = self.net(ctx.world, tuple(exo["cut"]))
        plan, perr = normalise_plan(conf, out.get("out") if out.get("ok") else None, counts0, live, net)
        if not out.get("ok"):
            perr = [f"controller failed: {out.get('error', '')}"[:200]] + perr
        up, down = self.transitions(st, plan["on"], h)
        counts = {}
        site_arr = st.arrays["fl_site"]
        new_status = status.copy()
        new_status[up] = BOOT
        new_status[down] = OFF
        for i, s in enumerate(conf["sites"]):
            m = site_arr == i
            counts[s["id"]] = {"on": int(np.sum(m & (new_status == ON))), "booting": int(np.sum(m & (new_status == BOOT)))}
        jobs = [dict(st.tables["fleet_jobs"][j], job=j) for j in st.find_ids("fleet_jobs", "status", "open")]
        res = hour_physics(conf, counts, plan, exo, jobs, net)
        # spot settlement (the utility's price depends on the operator's own draw)
        price = dict(exo["price"])
        if conf["tariff"] == "spot":
            tot = sum(v["kw"] for v in res["sites"].values())
            p = self.spot_price(st, h, tot)
            price = {s: p for s in price}
        # failures: hazard from wear and inlet temperature (common random numbers keyed by the hour)
        failed: list[int] = []
        if cfg["wear"]:
            fr = self.world.rng_for("fleet-fail", h).random(len(status))
            wear = st.arrays["fl_wear"]
            for i, s in enumerate(conf["sites"]):
                rs = res["sites"][s["id"]]
                if rs["on"] == 0:
                    continue
                idx = np.flatnonzero((site_arr == i) & (new_status == ON))
                hz = cfg["wear_h0"] * np.exp(cfg["wear_k"] * wear[idx]) * \
                    math.exp(0.12 * max(0.0, rs["t_in"] - 27.0)) * (0.6 + rs["util"])
                failed += idx[fr[idx] < 1.0 - np.exp(-hz)].tolist()
        trips = sorted(s for s, v in res["sites"].items() if v["trip"])
        energy = {}
        bank = ctx.world.modules["bank"]
        for s, v in res["sites"].items():
            cents = int(math.ceil(v["kwh"] * price[s])) if v["kwh"] > 0 else 0
            energy[s] = cents
            if cents > 0:
                avail = bank.available(st, conf["account"])
                pay = min(cents, max(0, avail))
                if pay > 0:
                    bank.move(ctx, conf["account"], conf["utility_account"], pay, "energy",
                              f"{v['kwh']:.1f} kWh at {s}, hour {h}")
                energy[s] = pay
        # measured telemetry (what the operator sees): exact own traffic, CPU per request with sensor noise
        mr = self.world.rng_for("fleet-telemetry", h)
        cpu_meas = {sid: round(float(exo["cpu_ms"][sid] * math.exp(float(mr.normal(0, 0.03)))), 4)
                    for sid in exo["cpu_ms"]}
        good = sum(v["good"] for v in res["services"].values())
        offered = sum(v["offered"] for v in res["services"].values())
        probe = mr.random(len(res["services"]))
        att_meas = round(min(1.0, (good / offered if offered else 1.0) * (1 + 0.004 * float(mr.normal()))), 5)
        del probe
        ctx.emit("fleet.hour", {
            "id": f"h{h:06d}", "hour": h, "exact": res, "plan": plan, "plan_errors": perr,
            "up": up, "down": down, "failed": sorted(set(failed)), "trips": trips,
            "price": price, "temp": exo["temp"], "energy_cents": energy, "cut": exo["cut"],
            "rps": {sid: {r: v for r, v in d.items()} for sid, d in exo["demand"].items()},
            "measured": {"cpu_ms": cpu_meas, "attainment": att_meas},
            "deployment": dep["id"], "controller_ok": bool(out.get("ok"))}, vis="internal")
        for s in trips:
            ctx.emit("fleet.trip", {"site": s, "hour": h, "until": h + 1 + int(conf["trip_h"])}, vis="public")
        old = st.get("fleet_hours", f"h{h - conf['dashboard_delay_h']:06d}")
        if old is not None:
            ctx.emit("fleet.dashboard", {"hour": old["hour"]}, vis=(conf["operator"],))
        # repairs at 08:00: the crew takes failed machines first-come first-served
        if h % 24 == 8:
            self._repairs(ctx, h)
        if self.cfg["link_cuts"]:
            self._cuts(ctx, h)

    def _repairs(self, ctx, h):
        st = ctx.state
        conf = self.conf(st)
        techs = int((st.get("fleet", "staff") or {}).get("technicians", conf["technicians"]))
        cap = int(techs * 8 // max(1, int(self.cfg["repair_tech_h"])))
        status = st.arrays["fl_status"]
        failed_at = st.arrays["fl_ready"]
        idx = np.flatnonzero(status == FAILED)
        idx = idx[np.lexsort((idx, failed_at[idx]))][:cap]
        if not len(idx):
            return
        bank = ctx.world.modules["bank"]
        cost = int(len(idx) * conf["repair_parts_cents"])
        pay = min(cost, max(0, bank.available(st, conf["account"])))
        if pay < cost:
            idx = idx[: pay // max(1, conf["repair_parts_cents"])]
            pay = int(len(idx) * conf["repair_parts_cents"])
        if pay > 0:
            bank.move(ctx, conf["account"], conf["vendor_account"], pay, "maintenance", f"parts for {len(idx)} repairs")
        if len(idx):
            ctx.emit("fleet.repair", {"machines": idx.tolist(), "ready": h + int(self.cfg["repair_h"])}, vis="internal")

    def _cuts(self, ctx, h):
        st = ctx.state
        conf = self.conf(st)
        rng = self.world.rng_for("fleet-cuts", h)
        cur = (st.get("fleet", "cuts") or {}).get("until", {})
        p = 1.0 - math.exp(-float(self.cfg["cut_rate_per_day"]) / 24.0)
        new = {}
        for l in conf["topology"]["links"]:
            u = float(rng.random())
            dur = int(round(math.exp(float(rng.normal(math.log(14.0), 0.5)))))
            if u < p and cur.get(l["id"], -1) <= h:
                new[l["id"]] = h + 1 + max(2, dur)
        if new:
            ctx.emit("fleet.cut", {"links": new, "hour": h}, vis="public")

    @process("maintenance")
    def pr_maintenance(self, ctx):
        st = ctx.state
        pend = st.get("fleet", "pending")
        if pend is None:
            return
        ctx.emit("fleet.deployed", {"id": st.new_id("dep"), "repo": pend["repo"], "ref": pend["ref"],
                                    "target": pend["target"], "by": pend["by"], "principal": pend["by"],
                                    "change": pend["change"]}, vis="public")

    @process("jobs")
    def pr_jobs(self, ctx):
        """Tenants submit the day's batch jobs (sizes and deadlines are their own)."""
        st = ctx.state
        conf = self.conf(st)
        d = ctx.now // DAY
        rng = self.world.rng_for("fleet-jobs", d)
        lo, hi = self.cfg["jobs_per_day"]
        n = int(rng.integers(lo, hi + 1))
        h0 = ctx.now // HOUR
        for k in range(n):
            t = conf["services"][int(rng.integers(0, len(conf["services"])))]["tenant_org"]
            core_h = float(round(rng.uniform(300, 1500), 1))
            jid = f"job{d:04d}-{k}"
            ctx.emit("fleet.job", {"id": jid, "tenant": t, "core_h": core_h, "cores_max": int(rng.integers(64, 200)),
                                   "deadline_hour": int(h0 + rng.integers(20, 40)), "submitted": h0},
                     vis=(t, conf["operator"]))

    @process("daily")
    def pr_daily(self, ctx):
        st = ctx.state
        conf = self.conf(st)
        cfg = self.cfg
        d = ctx.now // DAY - 1
        bank = ctx.world.modules["bank"]
        status = st.arrays["fl_status"]
        leased = int(np.sum(np.isin(status, LEASED)))
        lease = leased * conf["lease_cents_day"]
        techs = int((st.get("fleet", "staff") or {}).get("technicians", conf["technicians"]))
        wages = techs * conf["tech_wage_cents_day"]
        paid = {}
        for kind, amt, dst, memo in (("hardware", lease, conf["vendor_account"], f"lease of {leased} machines"),
                                     ("wage", wages, "acct_external", f"{techs} technicians")):
            pay = min(int(amt), max(0, bank.available(st, conf["account"])))
            if pay > 0:
                bank.move(ctx, conf["account"], dst, pay, kind, f"day {d}: {memo}")
            paid[kind] = pay
        # tenants: billing, SLA credits, their own income, their decisions
        hours = [st.get("fleet_hours", f"h{h:06d}") for h in range(d * 24, d * 24 + 24)]
        hours = [r for r in hours if r is not None]
        reg = self.regime(st)
        credits = 0
        bills = {}
        for tname, t in sorted(reg["tenants"].items()):
            svcs = [s for s in conf["services"] if s["tenant_org"] == t["org"]]
            good = sum(r["exact"]["services"][s["id"]]["good"] for r in hours for s in svcs)
            off = sum(r["exact"]["services"][s["id"]]["offered"] for r in hours for s in svcs)
            bill = int(sum(r["exact"]["services"][s["id"]]["good"] * s["price_per_m_cents"] / 1e6
                           for r in hours for s in svcs))
            done_jobs = [st.tables["fleet_jobs"][j] for j in st.find_ids("fleet_jobs", "status", "done")
                         if st.tables["fleet_jobs"][j]["tenant"] == t["org"] and
                         st.tables["fleet_jobs"][j].get("done_day") == d]
            bill += int(sum(j["core_h"] for j in done_jobs) * cfg["batch_value_cents_per_core_h"])
            income = int(good * t["value_per_m_cents"] / 1e6)
            if income > 0:
                bank.move(ctx, "acct_external", t["account"], income, "income", f"{tname} users, day {d}")
            att = good / off if off > 0 else 1.0
            credit = int(bill * cfg["sla_credit"]) if att < conf["sla_target"] else 0
            charge = bill - credit
            if charge > 0:
                charge = min(charge, max(0, bank.available(st, t["account"])))
                if charge > 0:
                    bank.move(ctx, t["account"], conf["account"], charge, "contract", f"hosting for {tname}, day {d}")
            credits += credit
            bills[t["org"]] = {"bill": bill, "credit": credit, "charged": charge, "good": round(good, 1),
                               "offered": round(off, 1)}
        ctx.emit("fleet.day", {"day": d, "leased": leased, "paid": paid, "bills": bills, "credits": credits},
                 vis="internal")
        for t, b in bills.items():
            ctx.emit("fleet.statement", {"day": d, "tenant": t, **b}, vis=(t, conf["operator"]))
        self._tenant_decisions(ctx, d)
        self._deliveries(ctx)
        if cfg["heat_waves"]:
            self._heat(ctx, d + 1)

    def _heat(self, ctx, day):
        st = ctx.state
        conf = self.conf(st)
        rng = self.world.rng_for("fleet-heat", day)
        cur = (st.get("fleet", "heat") or {}).get("until", {})
        new = {}
        for s in conf["sites"]:
            if float(rng.random()) < float(self.cfg["heat_waves"]) and cur.get(s["id"], -1) <= day * 24:
                new[s["id"]] = day * 24 + 24 * int(rng.integers(2, 5))
        if new:
            ctx.emit("fleet.heat", {"until": new, "day": day}, vis="internal")

    def _tenant_decisions(self, ctx, d):
        """Each tenant reviews its own last seven days (good / offered requests it was billed for) and decides how
        much of its traffic to keep here; it also releases new software on its own schedule."""
        st = ctx.state
        conf = self.conf(st)
        reg = self.regime(st)
        rng = self.world.rng_for("fleet-tenants", d)
        for tname, t in sorted(reg["tenants"].items()):
            recs = [st.get("fleet_days", f"d{k:04d}") for k in range(max(0, d - 6), d + 1)]
            recs = [r for r in recs if r is not None]
            good = sum(r["bills"].get(t["org"], {}).get("good", 0.0) for r in recs)
            off = sum(r["bills"].get(t["org"], {}).get("offered", 0.0) for r in recs)
            if off <= 0 or d < 6:
                continue
            att = good / off
            for svc in [s for s in conf["services"] if s["tenant_org"] == t["org"]]:
                share = self.demand_mult(st, svc["id"])
                if att < t["tolerance"]:
                    new = max(0.2, share * (1.0 - t["patience"] * min(1.0, (t["tolerance"] - att) * 20)))
                else:
                    new = min(1.0, share + 0.02)
                if abs(new - share) > 1e-6:
                    try:
                        ctx.call("fleet.tenant_share", {"service": svc["id"], "share": round(new, 4)},
                                 as_principal=t["lead"])
                    except CrucibleError:
                        pass
        for svc in conf["services"]:
            hid = reg["services"][svc["id"]]
            rels = st.find_ids("fleet_releases", "service", svc["id"])
            last_day = st.tables["fleet_releases"][rels[-1]]["day"] if rels else 0
            if d - last_day >= hid["release_every"] and float(rng.random()) < 0.5:
                lead = reg["tenants"][svc["tenant"]]["lead"]
                try:
                    ctx.call("fleet.release", {"service": svc["id"], "notes": f"release {len(rels) + 1}"},
                             as_principal=lead)
                except CrucibleError:
                    pass

    def _deliveries(self, ctx):
        st = ctx.state
        for oid in st.find_ids("fleet_orders", "status", "ordered"):
            o = st.tables["fleet_orders"][oid]
            if o["due"] <= ctx.now:
                ctx.emit("fleet.delivered", {"order": oid, "machines": list(o["machines"])},
                         vis=(self.conf(st)["operator"],))

    # ------------------------------------------------------------------ endpoints: public
    @endpoint("topology", S.obj(), read_only=True, cost_s=5)
    def ep_topology(self, ctx):
        """The network (public): regions, sites and links with capacity (Gbit/s) and latency (ms); links cut now."""
        conf = self.conf(ctx.state)
        return {"topology": to_jsonable(conf["topology"]), "regions": list(conf["regions"]),
                "sites": [s["id"] for s in conf["sites"]], "cut_links": self.cut_links(ctx.state, ctx.now // HOUR)}

    @endpoint("services", S.obj(), read_only=True)
    def ep_services(self, ctx):
        """Tenant services hosted here (public): owner, latency SLO, response size, price, release history."""
        st = ctx.state
        conf = self.conf(st)
        out = []
        for s in conf["services"]:
            rels = [st.tables["fleet_releases"][r] for r in st.find_ids("fleet_releases", "service", s["id"])]
            out.append({**to_jsonable(s), "releases": [{"version": r["version"], "day": r["day"], "notes": r["notes"]}
                                                        for r in rels]})
        return out

    @endpoint("status", S.obj(), read_only=True)
    def ep_status(self, ctx):
        """Operator console: fleet by site and status, deployed controller, pending change, maintenance window,
        tariff, technicians, orders, open batch jobs, trips and cut links (operators see everything here)."""
        st = ctx.state
        conf = self.conf(st)
        mine = self.operates(ctx)
        dep = st.get("fleet", "deployed")
        pend = st.get("fleet", "pending")
        h = ctx.now // HOUR
        out = {"operator": conf["operator"], "you_operate": mine, "tariff": conf["tariff"],
               "deployed": {k: dep[k] for k in ("repo", "ref", "target", "since")},
               "next_maintenance": self._next_maintenance(st), "maintenance_hour": conf["maintenance_hour"],
               "tripped": self.tripped(st, h), "cut_links": self.cut_links(st, h),
               "sites": [{k: s[k] for k in ("id", "region", "power_cap_kw", "pue0")} for s in conf["sites"]]}
        if mine:
            out["pending"] = None if pend is None else {k: pend[k] for k in ("change", "repo", "ref", "target")}
            out["machines"] = self.counts(st)
            out["technicians"] = int((st.get("fleet", "staff") or {}).get("technicians", conf["technicians"]))
            out["orders"] = [dict(st.tables["fleet_orders"][o], order=o)
                             for o in st.find_ids("fleet_orders", "status", "ordered")]
            out["open_jobs"] = len(st.find_ids("fleet_jobs", "status", "open"))
            out["prices"] = {"lease_cents_day": conf["lease_cents_day"], "repair_parts_cents": conf["repair_parts_cents"],
                             "tech_wage_cents_day": conf["tech_wage_cents_day"], "lead_days": conf["lead_days"],
                             "experiment_fee_cents": conf["experiment_fee_cents"],
                             "experiment_hour_cents": conf["experiment_hour_cents"]}
            bank = ctx.world.modules["bank"]
            out["cash_cents"] = bank.balance(st, conf["account"])
        return to_jsonable(out)

    # ------------------------------------------------------------------ endpoints: operator
    @endpoint("measure", S.obj({"window_h": S.s_int("hours to look back", minimum=1, maximum=24 * 30)}),
              read_only=True, cost_s=5)
    def ep_measure(self, ctx, window_h=24):
        """Operations dashboard (operators): per hour and site, machines on, utilisation, facility power, PUE,
        energy price and cost; per service, requests offered and the attainment measured by probes (noisy),
        CPU ms per request (noisy telemetry); published `dashboard_delay_h` hours late."""
        st = ctx.state
        conf = self._require_operator(ctx)
        hours = st.ids("fleet_hours")
        if not hours:
            return {"rows": [], "delay_h": conf["dashboard_delay_h"]}
        latest = st.tables["fleet_hours"][hours[-1]]["hour"] - conf["dashboard_delay_h"]
        lo = ctx.now // HOUR - window_h
        rows = []
        for hid in hours[-(window_h + conf["dashboard_delay_h"] + 1):]:
            r = st.tables["fleet_hours"][hid]
            if not (lo <= r["hour"] <= latest):
                continue
            rows.append({"hour": r["hour"], "hour_of_day": r["hour"] % 24,
                         "sites": {s: {k: v[k] for k in ("on", "booting", "util", "kw", "pue", "throttle", "trip")}
                                   | {"price": r["price"][s], "energy_cents": r["energy_cents"][s]}
                                   for s, v in r["exact"]["sites"].items()},
                         "requests": {sid: round(v["offered"]) for sid, v in r["exact"]["services"].items()},
                         "cpu_ms": dict(r["measured"]["cpu_ms"]), "attainment": r["measured"]["attainment"],
                         "failed_machines": len(r["failed"]), "plan_errors": list(r["plan_errors"])})
        return to_jsonable({"rows": rows, "delay_h": conf["dashboard_delay_h"]})

    @endpoint("machines", S.obj({"site": S.s_str(maxLength=10)}), read_only=True)
    def ep_machines(self, ctx, site=None):
        """Machines of a site (operators): status, health score (a noisy view of wear), power cycles, hours on."""
        st = ctx.state
        conf = self._require_operator(ctx)
        ids = [s["id"] for s in conf["sites"]]
        if site is not None and site not in ids:
            raise NotFound("no site " + site)
        rng = self.world.rng_for("fleet-health", ctx.now // HOUR)
        out = []
        sa, stt, wear = st.arrays["fl_site"], st.arrays["fl_status"], st.arrays["fl_wear"]
        cyc, hon = st.arrays["fl_cycles"], st.arrays["fl_hours"]
        noise = rng.normal(0, 0.05, len(sa))
        for i in range(len(sa)):
            if site is not None and ids[int(sa[i])] != site:
                continue
            out.append({"machine": i, "site": ids[int(sa[i])], "status": STATUS[int(stt[i])],
                        "health": round(float(max(0.0, 1.0 - wear[i] + noise[i])), 3), "cycles": int(cyc[i]),
                        "hours_on": int(hon[i])})
        return out

    @endpoint("experiment", S.obj({
        "repo": S.s_str("repository holding the controller", maxLength=81),
        "target": S.s_str("module:function, e.g. controller.py:control", maxLength=184),
        "ref": S.s_str("commit (default: head)", maxLength=40),
        "hours": S.s_int("simulated hours", minimum=1, maximum=72)}, ["repo", "target"]), cost_s=300)
    def ep_experiment(self, ctx, repo, target, ref=None, hours=24):
        """Offline test (operators, paid from the operator account): run a candidate controller and the deployed
        one side by side for `hours` simulated hours from now, on demand freshly sampled from the current traffic
        process and the current fleet (no failures, trips or tenant reactions). Returns energy, machine-hours,
        attainment and batch progress of both."""
        st = ctx.state
        conf = self._require_operator(ctx)
        if not re.fullmatch(TARGET_RE, target):
            raise InvalidAction("target must look like module.py:function")
        hours = int(min(hours, conf["max_experiment_hours"]))
        code = ctx.world.modules["code"]
        if not code.can_read(ctx, repo):
            raise Forbidden(f"no read access to {repo}")
        cid = ref or code.repo(st, repo)["head"]
        if cid is None:
            raise InvalidAction(f"{repo} is empty")
        mod = target.partition(":")[0]
        if (mod if mod.endswith(".py") else mod.replace(".", "/") + ".py") not in code.tree(st, repo, cid):
            raise NotFound(f"{mod} not in {repo}@{cid}")
        bank = ctx.world.modules["bank"]
        fee = int(conf["experiment_fee_cents"] + conf["experiment_hour_cents"] * hours)
        bank.move(ctx, conf["account"], "acct_fees", fee, "compute", f"fleet experiment {repo}:{target}")
        rng = ctx.rng("fleet-experiment")
        seed = int(rng.integers(0, 2**31 - 1))
        cand = {"repo": repo, "ref": cid, "target": target}
        dep = st.get("fleet", "deployed")
        res = {}
        for name, d in (("candidate", cand), ("deployed", dep)):
            res[name] = self.simulate(ctx, d, hours, seed)
        xid = st.new_id("flx")
        ctx.emit("fleet.experiment", {"id": xid, "by": ctx.actor, "repo": repo, "ref": cid, "target": target,
                                      "hours": hours, "results": res, "fee": fee}, vis=(ctx.actor, conf["operator"]))
        return {"experiment": xid, "hours": hours, "fee_cents": fee, **res}

    def simulate(self, ctx, dep: dict, hours: int, seed: int) -> dict:
        """Run a controller on sampled hours from the current state (counts only; boot lag kept)."""
        st = ctx.state
        conf = self.conf(st)
        reg = self.regime(st)
        h0 = ctx.now // HOUR
        cnt = self.counts(st)
        counts = {s: {"on": c["on"], "booting": c["booting"], "off": c["off"], "failed": c["failed"],
                      "repairing": c["repairing"]} for s, c in cnt.items()}
        boot_q: list[tuple[int, str]] = []
        rng = np.random.default_rng(seed)
        jobs = [dict(st.tables["fleet_jobs"][j], job=j) for j in st.find_ids("fleet_jobs", "status", "open")]
        tot = {"energy_cents": 0.0, "kwh": 0.0, "machine_hours": 0.0, "good": 0.0, "offered": 0.0,
               "batch_core_h": 0.0, "plan_errors": 0, "controller_errors": 0}
        hist: list[dict] = []
        for k in range(hours):
            h = h0 + k
            demand = {}
            for svc in conf["services"]:
                hid = reg["services"][svc["id"]]
                mult = self.demand_mult(st, svc["id"])
                demand[svc["id"]] = {r: demand_rps(hid, r, h, mult, rng) for r in conf["regions"]}
            for (t_ready, s) in [x for x in boot_q if x[0] <= h]:
                counts[s]["booting"] -= 1
                counts[s]["on"] += 1
                boot_q.remove((t_ready, s))
            exo = {"demand": demand, "cpu_ms": {s["id"]: reg["services"][s["id"]]["cpu_ms"] for s in conf["services"]},
                   "temp": {s["id"]: reg["sites"][s["id"]]["temp_mean"] for s in conf["sites"]},
                   "price": {s["id"]: (None if conf["tariff"] == "spot" else
                                       tariff(reg["sites"][s["id"]], conf["tariff"], h % 24)) for s in conf["sites"]},
                   "tripped": [], "cut": []}
            obs = self._sim_obs(st, h, counts, exo, hist, jobs)
            out = self.run_controller(ctx, dep, [obs], ctx.actor, "cmd")[0]
            if not out.get("ok"):
                tot["controller_errors"] += 1
            net = self.net(ctx.world)
            plan, perr = normalise_plan(conf, out.get("out") if out.get("ok") else None, counts,
                                        [s["id"] for s in conf["sites"]], net)
            tot["plan_errors"] += len(perr)
            for s, c in counts.items():
                target = plan["on"][s]
                cur = c["on"] + c["booting"]
                if target > cur:
                    n = min(target - cur, c["off"])
                    c["off"] -= n
                    c["booting"] += n
                    boot_q += [(h + conf["boot_h"], s)] * n
                elif target < cur:
                    n = cur - target
                    b = min(n, c["booting"])
                    c["booting"] -= b
                    for _ in range(b):
                        rm = max(x for x in boot_q if x[1] == s)
                        boot_q.remove(rm)
                    c["on"] -= n - b
                    c["off"] += n
            res = hour_physics(conf, counts, plan, exo, jobs, net)
            if conf["tariff"] == "spot":
                p = self.spot_price(st, h, sum(v["kw"] for v in res["sites"].values()))
                exo["price"] = {s: p for s in exo["price"]}
            for s, v in res["sites"].items():
                tot["energy_cents"] += v["kwh"] * exo["price"][s]
                tot["kwh"] += v["kwh"]
                tot["machine_hours"] += v["on"] + v["booting"]
            for v in res["services"].values():
                tot["good"] += v["good"]
                tot["offered"] += v["offered"]
            for j in jobs:
                g = res["batch"].get(j["job"], 0.0)
                j["left"] = max(0.0, j["left"] - g)
                j["core_hours_left"] = j["left"]
                tot["batch_core_h"] += g
            jobs = [j for j in jobs if j["left"] > 1e-9]
            hist.append({"hour": h, "rps": demand})
        out = {k: round(v, 3) if isinstance(v, float) else v for k, v in tot.items()}
        out["attainment"] = round(tot["good"] / tot["offered"], 6) if tot["offered"] else 1.0
        return out

    def _sim_obs(self, st, h, counts, exo, hist, jobs) -> dict:
        conf = self.conf(st)
        sites = []
        for s in conf["sites"]:
            today = None if conf["tariff"] == "spot" else [round(tariff(self.regime(st)["sites"][s["id"]],
                                                                         conf["tariff"], k), 4) for k in range(24)]
            sites.append({"id": s["id"], "region": s["region"], "cores_per_machine": conf["machine"]["cores"],
                          "machines": dict(counts[s["id"]]), "power_cap_kw": s["power_cap_kw"],
                          "tripped_until": None, "price_cents_kwh": exo["price"][s["id"]], "tariff_today": today,
                          "temp_c": None, "last": None})
        last = hist[-1] if hist else None
        services = []
        for svc in conf["services"]:
            sid = svc["id"]
            prev = [x for x in hist if h - x["hour"] in (24, 23, 22)]
            services.append({"id": sid, "slo_ms": svc["slo_ms"], "service_ms": svc["service_ms"], "kb": svc["kb"],
                             "cpu_ms": exo["cpu_ms"][sid], "rps_last": dict(last["rps"][sid]) if last else {},
                             "rps_next_day_ago": {r: [x["rps"][sid][r] for x in prev] for r in conf["regions"]}})
        return to_jsonable({"hour": h, "hour_of_day": h % 24, "day": h // 24, "boot_h": conf["boot_h"],
                            "params": {}, "sites": sites, "services": services, "regions": list(conf["regions"]),
                            "topology": conf["topology"], "cut_links": [],
                            "batch": [{"job": j["job"], "core_hours_left": j["left"], "cores_max": j["cores_max"],
                                       "deadline_hour": j["deadline_hour"]} for j in jobs]})

    @endpoint("propose_change", S.obj({
        "repo": S.s_str(maxLength=81), "target": S.s_str("module:function", maxLength=184),
        "ref": S.s_str("commit to deploy (default: current head, pinned now)", maxLength=40),
        "note": S.s_str(maxLength=500)}, ["repo", "target"]), cost_s=60)
    def ep_propose_change(self, ctx, repo, target, ref=None, note=""):
        """Schedule deployment of a controller to production at the next maintenance window (replaces any pending
        change). Operators only."""
        st = ctx.state
        conf = self._require_operator(ctx)
        if not re.fullmatch(TARGET_RE, target):
            raise InvalidAction("target must look like module.py:function")
        code = ctx.world.modules["code"]
        if not code.can_read(ctx, repo):
            raise Forbidden(f"no read access to {repo}")
        cid = ref or code.repo(st, repo)["head"]
        if cid is None:
            raise InvalidAction(f"{repo} is empty")
        mod = target.partition(":")[0]
        if (mod if mod.endswith(".py") else mod.replace(".", "/") + ".py") not in code.tree(st, repo, cid):
            raise NotFound(f"{mod} not in {repo}@{cid}")
        chg = st.new_id("flchg")
        at = self._next_maintenance(st)
        ctx.emit("fleet.change", {"id": chg, "repo": repo, "ref": cid, "target": target, "by": ctx.actor,
                                  "note": note[:500], "deploy_at": at}, vis=(ctx.actor, conf["operator"]))
        return {"change": chg, "ref": cid, "deploy_at": at}

    @endpoint("order", S.obj({"site": S.s_str(maxLength=10), "n": S.s_int(minimum=1, maximum=200)},
                             ["site", "n"]), cost_s=120)
    def ep_order(self, ctx, site, n):
        """Lease more machines for a site (operators): delivered after the lead time; the lease is paid daily
        from delivery."""
        st = ctx.state
        conf = self._require_operator(ctx)
        ids = [s["id"] for s in conf["sites"]]
        if site not in ids:
            raise NotFound("no site " + site)
        if len(st.arrays["fl_site"]) + n > 5000:
            raise Conflict("fleet size limit")
        first = len(st.arrays["fl_site"])
        oid = st.new_id("flord")
        due = ctx.now + conf["lead_days"] * DAY
        ctx.emit("fleet.ordered", {"id": oid, "site": ids.index(site), "n": int(n), "first": first, "due": due},
                 vis=(conf["operator"],))
        return {"order": oid, "due": due, "machines": list(range(first, first + n))}

    @endpoint("retire", S.obj({"site": S.s_str(maxLength=10), "n": S.s_int(minimum=1, maximum=5000)},
                              ["site", "n"]), cost_s=60)
    def ep_retire(self, ctx, site, n):
        """Return leased machines that are off (operators): the lease stops; the machines are gone for good."""
        st = ctx.state
        conf = self._require_operator(ctx)
        ids = [s["id"] for s in conf["sites"]]
        if site not in ids:
            raise NotFound("no site " + site)
        i = ids.index(site)
        sa, stt, wear = st.arrays["fl_site"], st.arrays["fl_status"], st.arrays["fl_wear"]
        cand = np.flatnonzero((sa == i) & np.isin(stt, (OFF, FAILED)))
        cand = cand[np.lexsort((cand, -wear[cand]))][: int(n)]
        if not len(cand):
            raise Conflict("no machine at this site is off or failed")
        ctx.emit("fleet.retired", {"machines": cand.tolist()}, vis=(conf["operator"],))
        return {"retired": len(cand)}

    @endpoint("staff", S.obj({"technicians": S.s_int(minimum=0, maximum=50)}, ["technicians"]), cost_s=60)
    def ep_staff(self, ctx, technicians):
        """Set the number of repair technicians (operators): each costs a daily wage and repairs
        8 / repair hours machines a day."""
        conf = self._require_operator(ctx)
        ctx.emit("fleet.staff", {"technicians": int(technicians)}, vis=(conf["operator"],))
        return {"technicians": int(technicians)}

    # ------------------------------------------------------------------ endpoints: tenants
    @endpoint("my_services", S.obj(), read_only=True)
    def ep_my_services(self, ctx):
        """A tenant's own services: daily statements (good and offered requests, bill, SLA credit) and the share of
        its traffic it keeps here."""
        st = ctx.state
        conf = self.conf(st)
        mine = ctx.world.principals_of(ctx.actor)
        orgs = [s["tenant_org"] for s in conf["services"] if s["tenant_org"] in mine or
                ident(ctx.world).allowed(ctx, ctx.actor, s["tenant_org"], "fleet.tenant")]
        out = []
        for o in sorted(set(orgs)):
            days = []
            for did in st.ids("fleet_days")[-14:]:
                b = st.tables["fleet_days"][did]["bills"].get(o)
                if b:
                    days.append({"day": st.tables["fleet_days"][did]["day"], **to_jsonable(b)})
            out.append({"tenant": o, "services": {s["id"]: self.demand_mult(st, s["id"]) for s in conf["services"]
                                                  if s["tenant_org"] == o}, "days": days})
        return out

    @endpoint("tenant_share", S.obj({"service": S.s_str(maxLength=20),
                                     "share": S.s_num("share of the service's traffic sent here", minimum=0.0,
                                                      maximum=1.0)}, ["service", "share"]), cost_s=60)
    def ep_tenant_share(self, ctx, service, share):
        """A tenant moves traffic of its own service to (or back from) other providers."""
        st = ctx.state
        conf = self.conf(st)
        svc = next((s for s in conf["services"] if s["id"] == service), None)
        if svc is None:
            raise NotFound("no service " + service)
        ident(ctx.world).authorize(ctx, svc["tenant_org"], "org.admin")
        ctx.emit("fleet.share", {"service": service, "share": float(share)}, vis=(svc["tenant_org"], conf["operator"]))
        return {"service": service, "share": float(share)}

    @endpoint("release", S.obj({"service": S.s_str(maxLength=20), "notes": S.s_str(maxLength=500)}, ["service"]),
              cost_s=120)
    def ep_release(self, ctx, service, notes=""):
        """A tenant deploys a new release of its service (its resource use per request may change)."""
        st = ctx.state
        conf = self.conf(st)
        svc = next((s for s in conf["services"] if s["id"] == service), None)
        if svc is None:
            raise NotFound("no service " + service)
        ident(ctx.world).authorize(ctx, svc["tenant_org"], "org.admin")
        rels = st.find_ids("fleet_releases", "service", service)
        version = len(rels) + 2
        hid = self.regime(st)["services"][service]
        rng = self.world.rng_for("fleet-release", service, version)
        factor = float(math.exp(rng.normal(-0.02, hid["release_sigma"])))
        rid = f"{service}-v{version}"
        ctx.emit("fleet.release", {"id": rid, "service": service, "version": version, "day": ctx.now // DAY,
                                   "notes": notes[:500]}, vis="public")
        ctx.emit("fleet.release_effect", {"service": service, "factor": round(factor, 5)}, vis="internal")
        return {"release": rid, "version": version}

    # ------------------------------------------------------------------ appliers
    @applier("fleet.conf")
    def ap_conf(self, ev):
        self.world.state.put("fleet", "conf", to_jsonable(ev.payload))
        self.world.runtime.pop("fleet_net", None)

    @applier("fleet.regime")
    def ap_regime(self, ev):
        self.world.state.put("fleet", "regime", {"params": to_jsonable(ev.payload["params"]), "day": ev.payload["day"]})

    @applier("fleet.machines_init")
    def ap_machines_init(self, ev):
        st = self.world.state
        n = len(ev.payload["site"])
        st.set_array("fl_site", np.array(ev.payload["site"], np.int16))
        st.set_array("fl_status", np.zeros(n, np.int8))
        st.set_array("fl_wear", np.array(ev.payload["wear"], np.float64))
        st.set_array("fl_ready", np.zeros(n, np.int64))
        st.set_array("fl_cycles", np.zeros(n, np.int32))
        st.set_array("fl_hours", np.zeros(n, np.int32))
        # at genesis every installed machine is on (the incumbent's operating state)
        st.array_mut("fl_status")[:] = ON

    @applier("fleet.deployed")
    def ap_deployed(self, ev):
        p = ev.payload
        st = self.world.state
        st.put("fleet", "deployed", dict(to_jsonable(p), since=ev.t))
        if p.get("change"):
            st.delete("fleet", "pending")
            st.patch("fleet_changes", p["change"], status="deployed", deployed_at=ev.t)

    @applier("fleet.change")
    def ap_change(self, ev):
        p = ev.payload
        st = self.world.state
        old = st.get("fleet", "pending")
        if old is not None:
            st.patch("fleet_changes", old["change"], status="superseded")
        st.insert("fleet_changes", p["id"], dict(to_jsonable(p), t=ev.t, status="pending"))
        st.put("fleet", "pending", {"change": p["id"], "repo": p["repo"], "ref": p["ref"], "target": p["target"],
                                    "by": p["by"]})

    @applier("fleet.ready")
    def ap_ready(self, ev):
        st = self.world.state
        stt = st.array_mut("fl_status")
        stt[np.array(ev.payload["on"], np.int64)] = ON
        stt[np.array(ev.payload["off"], np.int64)] = OFF

    @applier("fleet.hour")
    def ap_hour(self, ev):
        p = ev.payload
        st = self.world.state
        conf = st.get("fleet", "conf")
        stt = st.array_mut("fl_status")
        ready = st.array_mut("fl_ready")
        cyc = st.array_mut("fl_cycles")
        wear = st.array_mut("fl_wear")
        hon = st.array_mut("fl_hours")
        up = np.array(p["up"], np.int64)
        down = np.array(p["down"], np.int64)
        stt[up] = BOOT
        ready[up] = p["hour"] + int(conf["boot_h"])
        stt[down] = OFF
        cyc[up] += 1
        site = st.arrays["fl_site"]
        w_on = float(self.cfg["wear_on"])
        for i, s in enumerate(conf["sites"]):
            rs = p["exact"]["sites"][s["id"]]
            m = (site == i) & (stt == ON)
            wear[m] += w_on * (0.5 + rs["util"])
            hon[m] += 1
        wear[up] += float(self.cfg["wear_cycle"])
        f = np.array(p["failed"], np.int64)
        stt[f] = FAILED
        ready[f] = p["hour"]
        if p["trips"]:
            ids = [s["id"] for s in conf["sites"]]
            for s in p["trips"]:
                m = (site == ids.index(s)) & np.isin(stt, (ON, BOOT))
                stt[m] = OFF
        st.insert("fleet_hours", p["id"], dict(to_jsonable(p), t=ev.t, published=False))
        for jid, g in p["exact"]["batch"].items():
            j = st.get("fleet_jobs", jid)
            if j is None:
                continue
            left = max(0.0, j["left"] - g)
            st.patch("fleet_jobs", jid, left=round(left, 4), **({"status": "done", "done_day": p["hour"] // 24}
                                                               if left <= 1e-6 else {}))
        for jid in st.find_ids("fleet_jobs", "status", "open"):
            j = st.tables["fleet_jobs"][jid]
            if j["deadline_hour"] <= p["hour"]:
                st.patch("fleet_jobs", jid, status="missed")
        tot = st.get("fleet", "totals") or {"energy": 0, "hours": 0}
        st.put("fleet", "totals", {"energy": tot["energy"] + sum(p["energy_cents"].values()), "hours": tot["hours"] + 1})

    @applier("fleet.trip")
    def ap_trip(self, ev):
        st = self.world.state
        t = dict((st.get("fleet", "tripped") or {}).get("until", {}))
        t[ev.payload["site"]] = ev.payload["until"]
        st.put("fleet", "tripped", {"until": t})

    @applier("fleet.cut")
    def ap_cut(self, ev):
        st = self.world.state
        c = dict((st.get("fleet", "cuts") or {}).get("until", {}))
        c.update(ev.payload["links"])
        st.put("fleet", "cuts", {"until": c})
        self.world.runtime.pop("fleet_net", None)

    @applier("fleet.heat")
    def ap_heat(self, ev):
        st = self.world.state
        c = dict((st.get("fleet", "heat") or {}).get("until", {}))
        c.update(ev.payload["until"])
        st.put("fleet", "heat", {"until": c})

    @applier("fleet.dashboard")
    def ap_dashboard(self, ev):
        self.world.state.patch("fleet_hours", f"h{ev.payload['hour']:06d}", published=True)

    @applier("fleet.repair")
    def ap_repair(self, ev):
        st = self.world.state
        idx = np.array(ev.payload["machines"], np.int64)
        st.array_mut("fl_status")[idx] = REPAIR
        st.array_mut("fl_ready")[idx] = ev.payload["ready"]
        w = st.array_mut("fl_wear")
        w[idx] = w[idx] * 0.3

    @applier("fleet.job")
    def ap_job(self, ev):
        p = ev.payload
        self.world.state.insert("fleet_jobs", p["id"], {"tenant": p["tenant"], "core_h": p["core_h"],
                                                        "left": p["core_h"], "cores_max": p["cores_max"],
                                                        "deadline_hour": p["deadline_hour"],
                                                        "submitted": p["submitted"], "status": "open"})

    @applier("fleet.day")
    def ap_day(self, ev):
        self.world.state.insert("fleet_days", f"d{ev.payload['day']:04d}", dict(to_jsonable(ev.payload), t=ev.t))

    @applier("fleet.statement")
    def ap_statement(self, ev):
        pass

    @applier("fleet.share")
    def ap_share(self, ev):
        st = self.world.state
        s = dict((st.get("fleet", "shares") or {}).get("share", {}))
        s[ev.payload["service"]] = float(ev.payload["share"])
        st.put("fleet", "shares", {"share": s})

    @applier("fleet.release")
    def ap_release(self, ev):
        p = ev.payload
        self.world.state.insert("fleet_releases", p["id"], {"service": p["service"], "version": p["version"],
                                                            "day": p["day"], "notes": p["notes"], "t": ev.t})

    @applier("fleet.release_effect")
    def ap_release_effect(self, ev):
        st = self.world.state
        reg = to_jsonable(st.get("fleet", "regime"))
        svc = reg["params"]["services"][ev.payload["service"]]
        svc["cpu_ms"] = round(float(svc["cpu_ms"]) * float(ev.payload["factor"]), 4)
        st.put("fleet", "regime", reg)

    @applier("fleet.ordered")
    def ap_ordered(self, ev):
        p = ev.payload
        st = self.world.state
        n = int(p["n"])
        st.insert("fleet_orders", p["id"], {"site": p["site"], "n": n, "due": p["due"],
                                            "machines": list(range(p["first"], p["first"] + n)), "status": "ordered"})
        for name, dtype, fill in (("fl_site", np.int16, p["site"]), ("fl_status", np.int8, ORDERED),
                                  ("fl_wear", np.float64, 0.0), ("fl_ready", np.int64, 0),
                                  ("fl_cycles", np.int32, 0), ("fl_hours", np.int32, 0)):
            a = st.arrays[name]
            st.set_array(name, np.concatenate([a, np.full(n, fill, dtype)]).astype(dtype))

    @applier("fleet.delivered")
    def ap_delivered(self, ev):
        st = self.world.state
        st.patch("fleet_orders", ev.payload["order"], status="delivered")
        st.array_mut("fl_status")[np.array(ev.payload["machines"], np.int64)] = OFF

    @applier("fleet.retired")
    def ap_retired(self, ev):
        self.world.state.array_mut("fl_status")[np.array(ev.payload["machines"], np.int64)] = RETIRED

    @applier("fleet.staff")
    def ap_staff(self, ev):
        self.world.state.put("fleet", "staff", {"technicians": int(ev.payload["technicians"])})

    @applier("fleet.experiment")
    def ap_experiment(self, ev):
        p = ev.payload
        self.world.state.insert("fleet_experiments", p["id"], dict(to_jsonable(p), t=ev.t))

    # ------------------------------------------------------------------ invariants & observation
    def check_invariants(self, world):
        st = world.state
        if not st.has("fleet", "conf"):
            return []
        errs = []
        conf = st.get("fleet", "conf")
        n = len(st.arrays["fl_site"])
        for a in ("fl_status", "fl_wear", "fl_ready", "fl_cycles", "fl_hours"):
            if len(st.arrays[a]) != n:
                errs.append(f"{a} length {len(st.arrays[a])} != {n}")
        if n and (st.arrays["fl_status"].min() < 0 or st.arrays["fl_status"].max() >= len(STATUS)):
            errs.append("bad machine status")
        if n and st.arrays["fl_site"].max() >= len(conf["sites"]):
            errs.append("machine at an unknown site")
        hours = [st.tables["fleet_hours"][h]["hour"] for h in st.ids("fleet_hours")]
        if hours and hours != list(range(hours[0], hours[0] + len(hours))):
            errs.append("production hours not contiguous")
        if hours and hours[-1] != world.now // HOUR - 1:
            errs.append(f"production behind the clock: last hour {hours[-1]}, now {world.now // HOUR}")
        for hid in st.ids("fleet_hours")[-48:]:
            r = st.tables["fleet_hours"][hid]
            for sid, v in r["exact"]["services"].items():
                if v["good"] > v["delivered"] + 1e-6 or v["delivered"] > v["offered"] + 1e-6:
                    errs.append(f"{hid}: {sid} good <= delivered <= offered violated")
            for sid, v in r["exact"]["sites"].items():
                if v["util"] > 1.0 + 1e-6 or v["kw"] < 0:
                    errs.append(f"{hid}: {sid} utilisation or power out of range")
        for jid, j in st.scan("fleet_jobs"):
            if j["left"] < -1e-9 or j["left"] > j["core_h"] + 1e-9:
                errs.append(f"job {jid} progress out of range")
        return errs

    def observe(self, world, actor):
        st = world.state
        if not st.has("fleet", "conf"):
            return None
        conf = st.get("fleet", "conf")
        if conf["operator"] not in world.principals_of(actor):
            return None
        dep = st.get("fleet", "deployed")
        out = {"deployed": f"{dep['repo']}@{dep['ref']}:{dep['target']}", "machines": self.counts(st)}
        hours = st.ids("fleet_hours")
        if hours:
            h = st.tables["fleet_hours"][hours[-1]]["hour"] - conf["dashboard_delay_h"]
            r = st.get("fleet_hours", f"h{h:06d}")
            if r is not None:
                out["dashboard"] = {"hour": h, "attainment": r["measured"]["attainment"],
                                    "kw": round(sum(v["kw"] for v in r["exact"]["sites"].values()), 2),
                                    "energy_cents": sum(r["energy_cents"].values())}
        return out


# ============================================================================ evaluator-side accounting
COST_KINDS = ("energy", "hardware", "maintenance", "wage", "compute")


def operator_costs(world, t0: int, t1: int) -> dict:
    """Exact infrastructure cost of the operator between t0 and t1 from the ledger, by kind, plus SLA credits
    (revenue given up for missed service levels)."""
    st = world.state
    conf = st.get("fleet", "conf")
    acct = conf["account"]
    by = {k: 0 for k in COST_KINDS}
    for lid in st.indexes[("acct_txns", "acct")].get(acct, ()):
        e = st.tables["acct_txns"][lid]
        if not (t0 < e["t"] <= t1):
            continue
        tx = st.tables["txns"][e["txn"]]
        if tx["kind"] in by:
            by[tx["kind"]] += -sum(d for a, d in tx["postings"] if a == acct)
    d0, d1 = t0 // DAY, t1 // DAY
    credits = sum(r["credits"] for _, r in st.scan("fleet_days") if d0 <= r["day"] < d1)
    by["sla_credits"] = int(credits)
    by["total"] = int(sum(by.values()))
    return by


def reliability(world, h0: int, h1: int) -> dict:
    st = world.state
    good = offered = 0.0
    trips = failures = 0
    jobs_done = jobs_missed = 0
    for h in range(h0, h1):
        r = st.get("fleet_hours", f"h{h:06d}")
        if r is None:
            continue
        for v in r["exact"]["services"].values():
            good += v["good"]
            offered += v["offered"]
        trips += len(r["trips"])
        failures += len(r["failed"])
    for _, j in st.scan("fleet_jobs"):
        dl = j["deadline_hour"]
        if h0 <= dl < h1 or (j["status"] == "done" and h0 // 24 <= j.get("done_day", -1) < (h1 + 23) // 24):
            if j["status"] == "done":
                jobs_done += 1
            elif j["status"] == "missed":
                jobs_missed += 1
    return {"good": good, "offered": offered, "attainment": good / offered if offered else 1.0,
            "trips": trips, "failures": failures, "jobs_done": jobs_done, "jobs_missed": jobs_missed,
            "batch_on_time": jobs_done / (jobs_done + jobs_missed) if jobs_done + jobs_missed else 1.0}


__all__ = ["Fleet", "hour_physics", "normalise_plan", "sojourn_ok", "pue", "make_world", "demand_rps",
           "operator_costs", "reliability", "BASELINE_SOURCE", "BASELINE_PARAMS", "baseline_control", "STATUS"]
