"""Tier 2 domain 9.14 - crisis recovery (kernel module `crisis`, host `crisis`; MASTER_SPEC Part 9.14).

Example objective: restore essential services after a major simulated infrastructure or economic disruption.

World physics (module config generated per instance by `build`; nothing here is returned by any tool):

* **A region of districts** joined by roads, with lifeline infrastructure: power plants and transmission lines
  feeding one substation per district (a radial tree or a meshed network, by configuration), a water pump station
  and a telecom tower per district, a core telecom exchange, fuel depots and a regional hospital. Every facility
  has physical dependencies - a pump needs power, a tower needs power and the core exchange, the hospital needs
  power and water, a depot needs power to pump fuel - and some have backups (diesel generators with fuel tanks,
  tower batteries) that carry them through an outage until they run dry.
* **Disruptions** are physical hazard events: a windstorm, an earthquake with aftershocks, an intrusion into the
  control network, a flood. The event's intensity field over the districts and each asset's fragility (a lognormal
  fragility curve by asset type, reduced by hardening) decide what is damaged and how badly (repair hours, whether a
  spare part is needed); a flood keeps low districts under water for days. Aftershocks and secondary events follow
  a Poisson process. Nothing else fails by chance.
* **Cascades** are not scripted: every hour the module recomputes which facilities run from the state of their
  dependencies (power-flow connectivity from up plants over up lines; backups draining while the grid is down), so
  one damaged line can take down a substation, the pump and tower behind it, the hospital's water, and - once the
  depot's generator runs dry - the fuel that keeps every other generator alive.
* **Scarce resources**: repair crews (specialists of their operator's trade work twice as fast as others), spare
  parts (an operator's stock and a supplier whose lead time grows with its order book, and triples in a flood),
  fuel (depot stock, trucks that need open roads), money.
* **Autonomous operators** (power, water and telecom utilities, the public works department, the fuel distributor,
  the hospital authority) each run their own crews from what they know: they repair the damaged assets they know
  of that serve the most of their own customers, inspect assets whose status they cannot see (telemetry needs the
  local tower and the core exchange), and react to outage reports from households - and to nothing else unless
  they agreed to follow a coordinator's priorities (`crisis.agree`: an operator agrees once it has received the
  help it asks for, in crew-hours lent or parts given).
* **The population** (`crucible.population`, D-15) lives in the districts. In a `household` session an individual
  experiences the services of its district (power, water, telecom), copes (buys bottled water from its district's
  store with its own money; the store restocks only over open roads), reports outages when its phone works,
  sometimes needs the hospital (reachable over open roads and running) and leaves after long outages without water.

The participant is an emergency coordinator with a few general crews, trucks, fuel and a budget, acting only
through tools (`crisis.*`). The independent evaluator reads the recorded facility states and the population's
district assignment: household service-hours lost, the share of households with every service restored, the
hospital's downtime and the population's unmet health needs (`evaluate`).
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from ..core import schema as S
from ..core.errors import Conflict, CrucibleError, Forbidden, InvalidAction, NotFound, contain
from ..core.kernel import SYSTEM, Module, applier, endpoint, process, register_module
from ..services.identity import ident
from ..substrate import attention, columns, ids
from . import _dom3 as K

DAY, HOUR = K.DAY, K.HOUR
TABLE = "crisis"
MODEL_ID = "crisis.household"
FIDELITY = "crisis.household"
SERVICES = ("power", "water", "telecom")
TRADES = {"plant": "power", "line": "power", "substation": "power", "pump": "water", "tower": "telecom",
          "core": "telecom", "depot": "fuel", "hospital": "health", "road": "roads"}
# lognormal fragility: median intensity and dispersion by hazard and asset type (inspired by HAZUS-type curves)
FRAGILITY = {
    "storm": {"line": (0.55, 0.5), "tower": (0.65, 0.5), "road": (0.6, 0.6), "substation": (0.9, 0.5),
              "pump": (1.3, 0.5), "depot": (1.1, 0.5), "core": (1.4, 0.5), "hospital": (1.5, 0.5), "plant": (2.6, 0.5)},
    "quake": {"pump": (0.55, 0.5), "substation": (0.6, 0.5), "road": (0.55, 0.6), "line": (0.9, 0.5),
              "tower": (0.95, 0.5), "depot": (0.8, 0.5), "core": (1.0, 0.5), "hospital": (1.2, 0.5),
              "plant": (2.4, 0.5)},
    "cyber": {"substation": (0.45, 0.4), "core": (0.5, 0.4), "tower": (0.7, 0.4), "pump": (0.8, 0.4),
              "plant": (1.2, 0.4), "hospital": (0.7, 0.4)},
    "flood": {"substation": (0.6, 0.5), "pump": (0.5, 0.5), "depot": (0.7, 0.5), "road": (0.5, 0.5),
              "tower": (1.2, 0.5), "line": (1.4, 0.5), "core": (0.8, 0.5), "hospital": (0.9, 0.5)},
}
WORK_H = {"plant": (24, 48), "line": (6, 16), "substation": (12, 30), "pump": (8, 20), "tower": (4, 10),
          "core": (10, 20), "depot": (6, 12), "hospital": (6, 14), "road": (4, 12)}
PART = {"substation": "transformer", "pump": "pump_motor", "tower": "radio", "line": "conductor", "core": "switch",
        "plant": "turbine_kit"}
PART_PRICE = {"transformer": 90_000, "pump_motor": 40_000, "radio": 15_000, "conductor": 8_000, "switch": 30_000,
              "turbine_kit": 200_000}
INVEST = {"generator": (250_000, 48), "battery": (60_000, 24), "harden": (120_000, 72)}


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ====================================================================== instance generation
DEFAULTS = {"districts": 7, "hazard": "storm", "topology": "radial", "at_h": 6, "intensity": 1.0,
            "aftershock_rate_per_day": 0.0, "aftershock_scale": 0.45, "flood_days": [2, 5],
            "crews": {"powerco": 3, "waterco": 2, "telco": 2, "works": 1, "health": 1}, "coordinator_crews": 2,
            "coordinator_trucks": 2, "coordinator_fuel_l": 3000, "operator_asks": {"powerco": 16, "waterco": 0,
                                                                                    "telco": 8, "works": 0},
            "volunteers": False, "supplier_lead_h": 60, "store_stock": 120, "emergency_fund": 6_000_000,
            "aid_cents_crew_day": 120_000, "aid_arrival_h": 24,
            "pump_teams": 2, "dewater_cents": 90_000, "dewater_h": 20, "min_expected_damage": 10.0}


def generate_instance(rng, cfg: dict) -> dict:
    c = dict(DEFAULTS)
    c.update(cfg or {})
    D = int(c["districts"])
    xy = rng.uniform(0, 10, (D, 2))
    elev = rng.uniform(0, 1, D)
    hh = rng.dirichlet(np.ones(D) * 3.0)
    districts = [{"id": f"d{k}", "x": round(float(xy[k, 0]), 3), "y": round(float(xy[k, 1]), 3),
                  "elev": round(float(elev[k]), 3), "households": round(float(hh[k]), 4)} for k in range(D)]
    # roads: a spanning tree over nearest neighbours plus a few chords
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    seen, edges = {0}, []
    while len(seen) < D:
        best = min((dist[a, b], a, b) for a in seen for b in range(D) if b not in seen)
        edges.append((best[1], best[2]))
        seen.add(best[2])
    for _ in range(max(1, D // 3)):
        a, b = (int(x) for x in rng.choice(D, 2, replace=False))
        if (a, b) not in edges and (b, a) not in edges:
            edges.append((a, b))
    roads = [{"id": f"r{k}", "a": f"d{a}", "b": f"d{b}", "hours": round(1.0 + 0.35 * float(dist[a, b]), 2)}
             for k, (a, b) in enumerate(edges)]
    # power: two plants in two districts, lines along the road tree (radial) or plus chords (mesh)
    plants = [int(x) for x in rng.choice(D, 2, replace=False)]
    lines = []
    tree_edges = edges[:D - 1]
    use = tree_edges + (edges[D - 1:] if c["topology"] == "mesh" else [])
    if c["topology"] == "radial":
        # radial: every substation has exactly one feeding path from plant 0 (the second plant is a reserve
        # connected to its own district only)
        use = tree_edges
    for k, (a, b) in enumerate(use):
        lines.append({"id": f"l{k}", "a": f"d{a}", "b": f"d{b}"})
    hosp = int(np.argmax(hh))
    core = int(rng.integers(0, D))
    depots = [int(rng.integers(0, D))]
    fac = []

    def add(fid, typ, d, backup=None, **kw):
        fac.append({"id": fid, "type": typ, "district": f"d{d}", "backup": backup, **kw})
    for k, p in enumerate(plants):
        add(f"plant{k}", "plant", p)
    for d in range(D):
        add(f"sub{d}", "substation", d)
        add(f"pump{d}", "pump", d, {"kind": "generator", "cap_l": 400, "burn_lph": 18} if rng.random() < 0.35
            else None)
        add(f"tower{d}", "tower", d, {"kind": "battery", "cap_h": float(rng.integers(4, 10))})
    add("core", "core", core, {"kind": "generator", "cap_l": 600, "burn_lph": 20})
    for k, d in enumerate(depots):
        add(f"depot{k}", "depot", d, {"kind": "generator", "cap_l": 300, "burn_lph": 10}, stock_l=30_000)
    add("hospital", "hospital", hosp, {"kind": "generator", "cap_l": 900, "burn_lph": 35}, water_tank_h=16)
    # the event strikes the region: its centre lies near a district drawn by population, and it is a major
    # disruption - among a few draws the first whose expected damage (the fragility curves at the main event's
    # intensity) reaches `min_expected_damage` facilities, else the most damaging one
    def expected_damage(cxy) -> float:
        kind, frag = c["hazard"], FRAGILITY[c["hazard"]]
        I0, tot = float(c["intensity"]), 0.0
        sites = [(f["type"], f["district"]) for f in fac] + [("road", r["a"]) for r in roads] + \
            [("line", ln["a"]) for ln in lines]
        for kd, dist in sites:
            if kd not in frag:
                continue
            k = int(dist[1:])
            if kind in ("storm", "quake"):
                I = I0 * math.exp(-math.hypot(xy[k, 0] - cxy[0], xy[k, 1] - cxy[1]) / (6.0 if kind == "storm" else 4.5))
            elif kind == "flood":
                I = I0 * (1.4 if elev[k] < 0.35 * I0 else 0.4 * (1.0 - elev[k]))
            else:
                I = I0 * 0.8
            med, beta = frag[kd]
            tot += _phi((math.log(max(I, 1e-6)) - math.log(med)) / beta)
        return tot
    best = None
    for _ in range(12):
        k0 = int(rng.choice(D, p=hh))
        cand = xy[k0] + rng.normal(0, 1.0, 2)
        e = expected_damage(cand)
        if best is None or e > best[0]:
            best = (e, cand)
        if e >= float(c["min_expected_damage"]):
            break
    center = best[1]
    c.update({"districts": districts, "roads": roads, "lines": lines, "facilities": fac,
              "center": [round(float(x), 3) for x in center], "n_districts": D})
    return c


# ====================================================================== the kernel module
# the operator responsible for each kind of asset
KIND_OPERATOR = {"plant": "o:powerco", "line": "o:powerco", "substation": "o:powerco", "pump": "o:waterco",
                 "tower": "o:telco", "core": "o:telco", "road": "o:works", "hospital": "o:health"}


@register_module
class Crisis(Module):
    name = "crisis"
    host = "crisis"
    version = "1"
    depends = ("identity", "bank", "pop", "artifacts", "cols")

    def __init__(self, config=None):
        super().__init__(config)
        self.c = dict(DEFAULTS)
        self.c.update(self.config)
        self.districts = [d["id"] for d in self.c.get("districts") or []] if isinstance(self.c.get("districts"), list) \
            else []
        self.fac = {f["id"]: f for f in self.c.get("facilities") or []}
        self.roads = {r["id"]: r for r in self.c.get("roads") or []}
        self.lines = {l["id"]: l for l in self.c.get("lines") or []}

    # ------------------------------------------------------------------ install / genesis
    def install(self, world):
        st = world.state
        for t, ix in (("cr_assets", ("kind", "district")), ("cr_crews", ("owner",)), ("cr_trucks", ("owner",)),
                      ("cr_parts", ("owner",)), ("cr_orders", ("owner",)), ("cr_known", ("observer",)),
                      ("cr_operators", ()), ("cr_help", ("operator",)), ("cr_requests", ("operator",)),
                      ("cr_reports", ("district",)), ("cr_invest", ("owner",)), ("cr_stores", ()),
                      ("cr_hist", ()), ("cr_pumping", ("status",))):
            st.declare_table(t, index=ix)
        from .. import population as P
        n = P.n_individuals(world)
        columns.declare(world, TABLE, n, {
            "district": ("int16", (), -1), "health_risk": ("float16", (), 0.0), "left": ("uint8", (), 0),
            "dry_h": ("float32", (), 0.0), "last_t": ("int64", (), 0), "lost_h": ("float32", (len(SERVICES),), 0.0),
            "bought": ("int32", (), 0), "reports": ("int32", (), 0), "health_unmet": ("int32", (), 0),
            "health_met": ("int32", (), 0)})
        P.register_activity(world, "household", self.act_household, owner=self.name, prior=0.6, goals=(5,))

    def genesis(self, ctx):
        w = ctx.world
        idm = ident(w)
        bank = w.modules["bank"]
        for h, nm, kind in (("powerco", "Regional Power", "utility"), ("waterco", "City Water", "utility"),
                            ("telco", "Metro Telecom", "utility"), ("works", "Public Works Dept", "public"),
                            ("fuelco", "Fuel Distribution", "distributor"), ("health", "Hospital Authority", "public"),
                            ("supplier", "Grid Parts Supply", "supplier"), ("grocer", "District Stores", "retail"),
                            ("region", "Regional Government", "government"),
                            ("mutualaid", "Mutual Aid Network", "contractor")):
            mgr = idm.create_person(ctx, f"{h}-mgr", f"{nm} duty manager", "npc", {"role": "manager"})
            org = idm.create_org(ctx, h, nm, mgr, "company")
            acct = bank.open_account(ctx, org, "checking")
            bank.move(ctx, "acct_genesis", acct, 12_000_000 if h == "region" else 5_000_000, "endowment",
                      f"operating budget of {h}")
            ctx.emit("crisis.operator", {"id": org, "kind": kind, "manager": mgr,
                                         "ask_h": int(self.c["operator_asks"].get(h, 0))}, vis="public")
        trades = {"powerco": "power", "waterco": "water", "telco": "telecom", "works": "roads", "health": "health"}
        k = 0
        brng = w.rng_for("crisis-bases")
        for h, n in sorted(self.c["crews"].items()):
            base = self.districts[int(brng.integers(0, len(self.districts)))]
            for _ in range(int(n)):
                ctx.emit("crisis.crew", {"id": f"crew{k:02d}", "owner": f"o:{h}", "trade": trades[h], "at": base},
                         vis="public")
                k += 1
        stock = {"o:powerco": {"transformer": 1, "conductor": 3, "turbine_kit": 0},
                 "o:waterco": {"pump_motor": 1}, "o:telco": {"radio": 2, "switch": 1}, "o:health": {},
                 "o:supplier": {p: 4 for p in PART.values()}}
        for owner, parts in stock.items():
            for p, q in parts.items():
                ctx.emit("crisis.parts", {"owner": owner, "part": p, "delta": int(q)}, vis="internal")
        ctx.emit("crisis.truck", {"id": "truckf0", "owner": "o:fuelco", "at": self.fac["depot0"]["district"],
                                  "load_l": 0}, vis="public")
        for d in self.c["districts"]:
            ctx.emit("crisis.store", {"id": d["id"], "stock": int(self.c["store_stock"]),
                                      "cap": int(self.c["store_stock"])}, vis="internal")
        for f in self.c["facilities"]:
            b = f.get("backup") or {}
            ctx.emit("crisis.asset", {"id": f["id"], "kind": f["type"], "district": f["district"],
                                      "backup": b.get("kind"), "fuel_l": float(b.get("cap_l", 0)),
                                      "cap_l": float(b.get("cap_l", 0)), "burn_lph": float(b.get("burn_lph", 0)),
                                      "battery_h": float(b.get("cap_h", 0)), "cap_h": float(b.get("cap_h", 0)),
                                      "stock_l": float(f.get("stock_l", 0)), "water_tank_h":
                                          float(f.get("water_tank_h", 0)), "tank_left_h": float(f.get("water_tank_h", 0))},
                     vis="internal")
        for r in self.c["roads"]:
            ctx.emit("crisis.asset", {"id": r["id"], "kind": "road", "district": r["a"], "a": r["a"], "b": r["b"],
                                      "hours": r["hours"]}, vis="internal")
        for l in self.c["lines"]:
            ctx.emit("crisis.asset", {"id": l["id"], "kind": "line", "district": l["a"], "a": l["a"], "b": l["b"]},
                     vis="internal")
        self._genesis_people(ctx)
        ctx.schedule(ctx.now + HOUR, "crisis.hour", {}, prio=30, period=HOUR, key="crisis.hour")

    def _genesis_people(self, ctx):
        from .. import population as P
        w = ctx.world
        n = P.n_individuals(w)
        if not n:
            return
        rng = w.rng_for("crisis-people")
        shares = np.array([d["households"] for d in self.c["districts"]], np.float64)
        dist = rng.choice(len(shares), n, p=shares / shares.sum()).astype(np.int16)
        risk = np.clip(rng.beta(1.2, 12.0, n), 0, 1).astype(np.float16)
        columns.batch(ctx, TABLE, "crisis.init", np.arange(n), {"district": dist, "health_risk": risk},
                      extra={"what": "residence and health risk"})

    # ------------------------------------------------------------------ physics helpers
    def a(self, st, aid: str) -> dict:
        return st.tables["cr_assets"][aid]

    def _road_graph(self, st, flooded: set) -> dict[str, list[tuple[str, float]]]:
        g: dict[str, list] = {d: [] for d in self.districts}
        for rid, r in self.roads.items():
            s = st.tables["cr_assets"][rid]
            if s["damage_h"] > 0 or r["a"] in flooded or r["b"] in flooded:
                continue
            g[r["a"]].append((r["b"], float(r["hours"])))
            g[r["b"]].append((r["a"], float(r["hours"])))
        return g

    def reach_h(self, st, src: str, aid: str) -> float | None:
        """Travel time from `src` to where work on asset `aid` can be done (a road or line from either end)."""
        x = st.tables["cr_assets"][aid]
        if x["kind"] in ("road", "line"):
            ts = [t for t in (self.travel_h(st, src, x["a"]), self.travel_h(st, src, x["b"])) if t is not None]
            return min(ts) if ts else None
        return self.travel_h(st, src, x["district"])

    def travel_h(self, st, src: str, dst: str, flooded: set | None = None) -> float | None:
        """Shortest travel time over open roads (None: unreachable)."""
        flooded = self.flooded(st) if flooded is None else flooded
        if src == dst:
            return 0.0 if dst not in flooded else None
        if dst in flooded:
            return None
        g = self._road_graph(st, flooded)
        best = {src: 0.0}
        pq = [(0.0, src)]
        while pq:
            dd, u = heapq.heappop(pq)
            if u == dst:
                return dd
            if dd > best.get(u, 1e9):
                continue
            for v, h in g.get(u, ()):
                nd = dd + h
                if nd < best.get(v, 1e9):
                    best[v] = nd
                    heapq.heappush(pq, (nd, v))
        return None

    def entry(self, st, flooded: set | None = None) -> str:
        """Where help from outside the region comes in: its edge, district d0 - or, while d0 is under water, the dry
        district nearest to it."""
        flooded = self.flooded(st) if flooded is None else flooded
        d0 = self.districts[0]
        if d0 not in flooded:
            return d0
        dm = {d["id"]: d for d in self.c["districts"]}
        dry = [d for d in self.districts if d not in flooded]
        if not dry:
            return d0
        return min(dry, key=lambda d: (math.hypot(dm[d]["x"] - dm[d0]["x"], dm[d]["y"] - dm[d0]["y"]), d))

    def flooded(self, st) -> set:
        m = st.get("cr_operators", "_flood")
        if m is None or st.tables["cr_operators"]["_flood"]["until"] <= self.world.now:
            return set()
        return set(m["districts"])

    def compute_state(self, st, now: int) -> dict[str, bool]:
        """Which facilities run now (dependencies and backups as they stand)."""
        A = st.tables["cr_assets"]
        flooded = self.flooded(st)

        def intact(aid):
            # flood water keeps crews out of a district; what it damages is decided by fragility at the event
            return A[aid]["damage_h"] <= 0
        # power: connectivity from intact plants over intact lines through intact substations
        live = set()
        frontier = [self.fac[p]["district"] for p in self.fac if self.fac[p]["type"] == "plant" and intact(p)]
        adj: dict[str, list[str]] = {}
        for lid, l in self.lines.items():
            if A[lid]["damage_h"] <= 0:
                adj.setdefault(l["a"], []).append(l["b"])
                adj.setdefault(l["b"], []).append(l["a"])
        sub = {self.fac[f]["district"]: f for f in self.fac if self.fac[f]["type"] == "substation"}
        for d in frontier:
            if d in sub and intact(sub[d]):
                live.add(d)
        stack = list(live)
        while stack:
            d = stack.pop()
            for e in adj.get(d, ()):
                if e not in live and e in sub and intact(sub[e]):
                    live.add(e)
                    stack.append(e)
        up: dict[str, bool] = {}
        grid = {d: d in live for d in self.districts}
        for fid, f in sorted(self.fac.items()):
            x = A[fid]
            if f["type"] == "plant":
                up[fid] = intact(fid)
            elif f["type"] == "substation":
                up[fid] = grid[f["district"]]
        for fid in ("core",) + tuple(sorted(f for f in self.fac if self.fac[f]["type"] in ("depot", "pump"))):
            x = A[fid]
            up[fid] = intact(fid) and (grid[x["district"]] or self._backup_ok(x))
        for fid in sorted(f for f in self.fac if self.fac[f]["type"] == "tower"):
            x = A[fid]
            powered = grid[x["district"]] or (x["backup"] == "battery" and x["battery_h"] > 0) or \
                (x["backup"] == "generator" and x["fuel_l"] > 0)
            up[fid] = intact(fid) and powered and up.get("core", False)
        h = A["hospital"]
        pw = grid[h["district"]] or self._backup_ok(h)
        water = up.get(f"pump{h['district'][1:]}", False) or h["tank_left_h"] > 0
        up["hospital"] = intact("hospital") and pw and water
        return up

    def _backup_ok(self, x) -> bool:
        return (x["backup"] == "generator" and x["fuel_l"] > 0) or (x["backup"] == "battery" and x["battery_h"] > 0)

    def services(self, st, up: dict) -> dict[str, dict[str, bool]]:
        """Service per district: power (substation), water (pump), telecom (tower and core)."""
        return {d: {"power": up.get(f"sub{d[1:]}", False), "water": up.get(f"pump{d[1:]}", False),
                    "telecom": up.get(f"tower{d[1:]}", False)} for d in self.districts}

    # ------------------------------------------------------------------ the hourly physics
    @process("hour")
    def pr_hour(self, ctx):
        st = ctx.state
        now = ctx.now
        rng = ctx.rng("crisis-hour")
        c = self.c
        h = now // HOUR
        # the hazard is exogenous: its draws come from the world's own stream for that hour, so what participants
        # do never changes which event strikes or where (only hardening changes what it damages)
        hz = ctx.world.rng_for("crisis-hazard", int(h))
        if h == int(c["at_h"]):
            self._hazard(ctx, hz, float(c["intensity"]), main=True)
        elif h > int(c["at_h"]) and float(c["aftershock_rate_per_day"]) > 0 and \
                hz.random() < 1.0 - math.exp(-float(c["aftershock_rate_per_day"]) / 24.0):
            self._hazard(ctx, hz, float(c["intensity"]) * float(c["aftershock_scale"]) *
                         float(hz.uniform(0.6, 1.2)), main=False)
        self._work(ctx)
        self._fuel_and_backups(ctx)
        self._arrivals(ctx)
        self._telemetry(ctx)
        err = contain(ctx, self._operators, ctx, rng)
        if err is not None:
            ctx.emit("crisis.failed", {"what": "operators", "error": str(err)[:200]}, vis="internal")
        self._stores(ctx)

    def _hazard(self, ctx, rng, intensity: float, *, main: bool):
        st = ctx.state
        kind = self.c["hazard"]
        frag = FRAGILITY[kind]
        cx, cy = self.c["center"]
        dmap = {d["id"]: d for d in self.c["districts"]}
        hit = []
        if kind == "flood":
            # a flood wave: low districts go under water (a later, smaller wave floods fewer of them)
            flood = sorted(d["id"] for d in self.c["districts"] if d["elev"] < 0.35 * intensity)
            days = int(rng.integers(int(self.c["flood_days"][0]), int(self.c["flood_days"][1]) + 1))
            ctx.emit("crisis.flood", {"districts": flood, "until": int(ctx.now + days * DAY)}, vis="public")
        for aid, x in sorted(st.scan("cr_assets")):
            kd = x["kind"]
            if kd not in frag:
                continue
            d = dmap[x["district"]]
            if kind in ("storm", "quake"):
                r = math.hypot(d["x"] - cx, d["y"] - cy)
                I = intensity * math.exp(-r / (6.0 if kind == "storm" else 4.5))
            elif kind == "flood":
                I = intensity * (1.4 if d["elev"] < 0.35 * intensity else 0.4 * (1.0 - d["elev"]))
            else:
                I = intensity * 0.8
            med, beta = frag[kd]
            p = _phi((math.log(max(I, 1e-6)) - math.log(med) - 0.6 * float(x["harden"])) / beta)
            if rng.random() < p and x["damage_h"] <= 0:
                lo, hi = WORK_H[kd]
                sev = float(rng.uniform(0.2, 1.0))
                # a control-system intrusion trips assets (short resets); only the worst cases burn equipment
                cyber_reset = kind == "cyber" and sev <= 0.8
                work = (0.4 if cyber_reset else 1.0) * (lo + (hi - lo) * sev)
                part = PART.get(kd) if (sev > 0.65 and not cyber_reset) else None
                hit.append(aid)
                ctx.emit("crisis.damage", {"id": aid, "work_h": round(work, 2), "part": part, "main": main},
                         vis="internal")
        ctx.emit("crisis.event", {"hazard": kind, "main": main, "intensity": round(intensity, 3),
                                  "damaged": len(hit)}, vis="public")

    def _work(self, ctx):
        st = ctx.state
        now = ctx.now
        for cid, cr in sorted(st.scan("cr_crews")):
            t = cr["task"]
            if cr.get("leave_t") and cr["leave_t"] <= now:
                if t or cr.get("lent_to"):
                    ctx.emit("crisis.crew_task", {"id": cid, "task": None, "at": cr["at"]}, vis="public")
                continue
            if not t or t["eta"] > now:
                continue
            tgt = st.tables["cr_assets"].get(t["target"])
            if tgt is None:
                continue
            if t["kind"] == "inspect":
                ctx.emit("crisis.known", {"id": f"{cr['owner']}|{t['target']}", "observer": cr["owner"],
                                          "asset": t["target"], "damaged": tgt["damage_h"] > 0,
                                          "work_h": tgt["damage_h"], "part": tgt["part"], "t": now,
                                          "src": "inspection"}, vis=(cr["owner"],))
                ctx.emit("crisis.crew_task", {"id": cid, "task": None, "at": tgt["district"]}, vis="public")
                continue
            if tgt["damage_h"] <= 0:
                ctx.emit("crisis.crew_task", {"id": cid, "task": None, "at": tgt["district"]}, vis="public")
                continue
            if tgt["district"] in self.flooded(st):
                continue
            if tgt["part"] and not tgt["part_fitted"]:
                have = st.get("cr_parts", f"{t['by']}|{tgt['part']}")
                if have is None or have["qty"] <= 0:
                    if not tgt.get("waiting_part"):
                        # who lacks the part: the party that dispatched the waiting crew (its own crews' finding)
                        ctx.emit("crisis.wait_part", {"id": t["target"], "crew": cid, "part": tgt["part"],
                                                      "by": t["by"]}, vis="internal")
                    continue                    # the crew waits on site for the part
                ctx.emit("crisis.parts", {"owner": t["by"], "part": tgt["part"], "delta": -1}, vis="internal")
                ctx.emit("crisis.part_fitted", {"id": t["target"]}, vis="internal")
            speed = 1.0 if TRADES[tgt["kind"]] == cr["trade"] else 0.5
            left = max(0.0, float(tgt["damage_h"]) - speed)
            ctx.emit("crisis.repair", {"id": t["target"], "work_h": round(left, 3), "crew": cid}, vis="internal")
            if left <= 0:
                ctx.emit("crisis.restored", {"id": t["target"], "by": cr["owner"], "crew": cid}, vis="public")
                ctx.emit("crisis.crew_task", {"id": cid, "task": None, "at": tgt["district"]}, vis="public")

    def _fuel_and_backups(self, ctx):
        st = ctx.state
        up = self.compute_state(st, ctx.now)
        A = st.tables["cr_assets"]
        grid = self.services(st, up)
        prev = st.get("cr_operators", "_up")
        prev_up = prev["up"] if prev else {}
        for fid in sorted(self.fac):
            x = A[fid]
            d = x["district"]
            ongrid = grid[d]["power"]
            ch = {}
            if x["backup"] == "generator" and not ongrid and x["damage_h"] <= 0 and x["fuel_l"] > 0:
                ch["fuel_l"] = round(max(0.0, x["fuel_l"] - x["burn_lph"]), 3)
            if x["backup"] == "battery":
                if ongrid and x["battery_h"] < x["cap_h"]:
                    ch["battery_h"] = min(x["cap_h"], x["battery_h"] + 0.5)
                elif not ongrid and x["battery_h"] > 0 and x["damage_h"] <= 0:
                    ch["battery_h"] = round(max(0.0, x["battery_h"] - 1.0), 3)
            if fid == "hospital":
                pw = up.get(f"pump{d[1:]}", False)
                if pw and x["tank_left_h"] < x["water_tank_h"]:
                    ch["tank_left_h"] = x["water_tank_h"]
                elif not pw and x["tank_left_h"] > 0:
                    ch["tank_left_h"] = max(0.0, x["tank_left_h"] - 1.0)
            if ch:
                ctx.emit("crisis.level", {"id": fid, "changes": ch}, vis="internal")
        flooded = self.flooded(st)
        cascaded = sorted(f for f, u in up.items() if not u and A[f]["damage_h"] <= 0 and
                          A[f]["district"] not in flooded)
        on_backup = sorted(f for f, u in up.items() if u and self.fac[f]["type"] not in ("plant", "substation")
                           and not grid[A[f]["district"]]["power"])
        for fid in sorted(self.fac):
            x = A[fid]
            if x["backup"] == "generator" and x["fuel_l"] > 0 and not grid[x["district"]]["power"] and \
                    x["damage_h"] <= 0 and x["fuel_l"] - x["burn_lph"] <= 0:
                ctx.emit("crisis.backup_dry", {"id": fid, "kind": "generator"}, vis="internal")
            if x["backup"] == "battery" and 0 < x["battery_h"] <= 1.0 and not grid[x["district"]]["power"]:
                ctx.emit("crisis.backup_dry", {"id": fid, "kind": "battery"}, vis="internal")
        changed = {k: v for k, v in up.items() if prev_up.get(k) != v}
        if changed or prev is None or on_backup != (prev or {}).get("on_backup"):
            ctx.emit("crisis.state", {"up": up, "changed": sorted(changed), "services": grid, "cascaded": cascaded,
                                      "on_backup": on_backup}, vis="internal")

    def _arrivals(self, ctx):
        st = ctx.state
        now = ctx.now
        # crews called in from outside arrive at the region's edge; while the edge is under water they come in at
        # the dry district nearest to it
        flooded = self.flooded(st)
        if flooded:
            for cid, cr in sorted(st.scan("cr_crews")):
                if now - HOUR < int(cr.get("arrive_t") or 0) <= now and cr["at"] in flooded and not cr["task"]:
                    ctx.emit("crisis.crew_task", {"id": cid, "task": None, "at": self.entry(st, flooded)},
                             vis="public")
        for oid, o in sorted(st.scan("cr_orders")):
            if o["status"] == "open" and o["arrive_t"] <= now:
                ctx.emit("crisis.parts", {"owner": o["owner"], "part": o["part"], "delta": int(o["qty"])},
                         vis=(o["owner"],))
                ctx.emit("crisis.order_status", {"id": oid, "status": "delivered"}, vis=(o["owner"],))
        for tid, tr in sorted(st.scan("cr_trucks")):
            t = tr["task"]
            if not t or t["eta"] > now:
                continue
            tgt = st.tables["cr_assets"].get(t["target"])
            if t.get("water") and tgt is not None:
                ctx.emit("crisis.level", {"id": t["target"], "changes": {"tank_left_h": tgt["water_tank_h"]}},
                         vis="internal")
                ctx.emit("crisis.watered", {"truck": tid, "asset": t["target"], "by": tr["owner"]}, vis="public")
                ctx.emit("crisis.truck_task", {"id": tid, "task": None, "at": tgt["district"],
                                               "load_l": tr["load_l"]}, vis="public")
            elif tgt is not None and tgt["backup"] == "generator":
                put = min(float(tr["load_l"]), max(0.0, tgt["cap_l"] - tgt["fuel_l"]))
                ctx.emit("crisis.level", {"id": t["target"], "changes": {"fuel_l": round(tgt["fuel_l"] + put, 3)}},
                         vis="internal")
                ctx.emit("crisis.fuelled", {"truck": tid, "asset": t["target"], "liters": round(put, 1),
                                            "by": tr["owner"]}, vis="public")
                ctx.emit("crisis.truck_task", {"id": tid, "task": None, "at": tgt["district"],
                                               "load_l": round(tr["load_l"] - put, 3)}, vis="public")
            else:
                ctx.emit("crisis.truck_task", {"id": tid, "task": None, "at": tr["at"], "load_l": tr["load_l"]},
                         vis="public")

    def _telemetry(self, ctx):
        """Operators see the live status of their assets when the asset is powered (grid or backup) and its
        district's tower and the core exchange are up; otherwise their last knowledge stands."""
        st = ctx.state
        now = ctx.now
        prev = st.get("cr_operators", "_up")
        if prev is None:
            return
        up = prev["up"]
        own = {"plant": "o:powerco", "substation": "o:powerco", "line": "o:powerco", "pump": "o:waterco",
               "tower": "o:telco", "core": "o:telco", "depot": "o:fuelco", "hospital": "o:health"}
        for aid, x in sorted(st.scan("cr_assets")):
            if x["kind"] not in own:
                continue
            d = x["district"]
            link = up.get(f"tower{d[1:]}", False) and up.get("core", False)
            if not link:
                continue
            k = st.get("cr_known", f"{own[x['kind']]}|{aid}")
            dmg = x["damage_h"] > 0
            ev_t = (st.get("cr_operators", "_event") or {}).get("t", -1)
            if k is None or k["damaged"] != dmg or k["t"] < ev_t:
                ctx.emit("crisis.known", {"id": f"{own[x['kind']]}|{aid}", "observer": own[x["kind"]], "asset": aid,
                                          "damaged": dmg, "work_h": x["damage_h"], "part": x["part"], "t": now,
                                          "src": "telemetry"}, vis=(own[x["kind"]],))

    # ------------------------------------------------------------------ autonomous operators
    def _operators(self, ctx, rng):
        st = ctx.state
        now = ctx.now
        hh = {d["id"]: d["households"] for d in self.c["districts"]}
        types = {"o:powerco": ("plant", "line", "substation"), "o:waterco": ("pump",), "o:telco": ("tower", "core"),
                 "o:works": ("road",), "o:health": ("hospital",)}
        for cid, cr in sorted(st.scan("cr_crews")):
            if cr["task"] or cr.get("lent_to") is None and cr["owner"] not in types:
                continue
            if cr.get("arrive_t", 0) > now or (cr.get("leave_t") and cr["leave_t"] <= now):
                continue
            boss = cr.get("lent_to") or cr["owner"]
            if boss not in types:
                continue
            if cr.get("lent_until") and cr["lent_until"] <= now:
                ctx.emit("crisis.lend", {"id": cid, "to": None, "until": None}, vis="public")
                continue
            target, kind, why = self._choose(st, boss, types[boss], cr, hh, now)
            if target is None:
                continue
            eta = self.reach_h(st, cr["at"], target)
            if eta is None:
                continue
            ctx.emit("crisis.crew_task", {"id": cid, "task": {"kind": kind, "target": target, "by": boss,
                                                              "eta": int(now + eta * HOUR)}, "at": cr["at"]},
                     vis="public")
            ctx.emit("crisis.priority", {"operator": boss, "crew": cid, "target": target, "task": kind,
                                         "reason": why}, vis="public")
        for boss in sorted(types):
            err = contain(ctx, self._order_parts, ctx, boss, types[boss])
            if err is not None:
                ctx.emit("crisis.failed", {"what": "parts", "error": str(err)[:200]}, vis="internal")
        err = contain(ctx, self._fuel_distributor, ctx)
        if err is not None:
            ctx.emit("crisis.failed", {"what": "fuel distributor", "error": str(err)[:200]}, vis="internal")

    def _choose(self, st, boss, kinds, crew, hh, now):
        """An operator's next task for a crew: an agreed coordinator request first, then the known damaged asset
        of its own that serves the most of its customers per hour of work (reachable, part in stock or not
        needed), else an inspection of an asset whose status it cannot see, the most households first."""
        A = st.tables["cr_assets"]
        busy = {c["task"]["target"] for _, c in st.scan("cr_crews") if c["task"]}
        op = st.tables["cr_operators"].get(boss)
        if op and op.get("agreed"):
            ev_t = (st.get("cr_operators", "_event") or {}).get("t", -1)
            for _, r in sorted(st.scan("cr_requests"), key=lambda x: (x[1]["rank"], x[0])):
                if r["operator"] != boss or r["status"] != "open" or r["target"] in busy:
                    continue
                x = A.get(r["target"])
                if x is None or x["kind"] not in kinds or self.reach_h(st, crew["at"], r["target"]) is None:
                    continue
                k = st.get("cr_known", f"{boss}|{r['target']}")
                fresh_ok = k is not None and not k["damaged"] and k["t"] >= max(ev_t, now - DAY)
                if fresh_ok:
                    continue
                if k is not None and k["damaged"] and x["part"] and not x["part_fitted"] and \
                        (st.get("cr_parts", f"{boss}|{x['part']}") or {}).get("qty", 0) <= 0:
                    continue
                return r["target"], ("repair" if k and k["damaged"] else "inspect"), "request"
        best = None
        for aid, x in sorted(st.scan("cr_assets")):
            if x["kind"] not in kinds or aid in busy:
                continue
            k = st.get("cr_known", f"{boss}|{aid}")
            if k is None or not k["damaged"] or x["damage_h"] <= 0:
                continue
            if x["part"] and not x.get("part_fitted"):
                have = st.get("cr_parts", f"{boss}|{x['part']}")
                if have is None or have["qty"] <= 0:
                    continue
            if self.reach_h(st, crew["at"], aid) is None:
                continue
            served = self._served(aid, x, hh)
            val = served / max(1.0, float(k["work_h"]))
            if best is None or val > best[0]:
                best = (val, aid)
        if best is not None:
            return best[1], "repair", "customers"
        # inspect what it cannot see: assets in districts that reported outages, then the rest by households
        rep = {r["district"] for _, r in st.scan("cr_reports") if r["t"] >= now - DAY}
        cand = []
        for aid, x in sorted(st.scan("cr_assets")):
            if x["kind"] not in kinds or aid in busy:
                continue
            k = st.get("cr_known", f"{boss}|{aid}")
            # what an operator knows goes stale with time, and at once when a disruption hit after it learned it
            # (telemetry that went dark after the event says nothing about the asset's state now)
            ev_t = (st.get("cr_operators", "_event") or {}).get("t", -1)
            stale = k is None or (now - k["t"] > 2 * DAY) or k["t"] < ev_t or \
                (x["district"] in rep and k.get("src") == "telemetry" and not k["damaged"])
            if not stale:
                continue
            cand.append((-(2.0 if x["district"] in rep else 1.0) * self._served(aid, x, hh), aid))
        for _, aid in sorted(cand):
            if self.reach_h(st, crew["at"], aid) is not None:
                return aid, "inspect", "unknown status"
        return None, None, None

    def _order_parts(self, ctx, boss, kinds) -> None:
        """An operator orders a spare part from the supplier for each known damaged asset of its own that needs one
        it does not hold and has not ordered (paid from its account; delivered after the supplier's lead time)."""
        st = ctx.state
        bank = ctx.world.modules["bank"]
        for aid, x in sorted(st.scan("cr_assets")):
            if x["kind"] not in kinds or not x["part"] or x["part_fitted"]:
                continue
            k = st.get("cr_known", f"{boss}|{aid}")
            if k is None or not k["damaged"] or not k.get("part"):
                continue
            need = sum(1 for a2, y in st.scan("cr_assets") if y["kind"] in kinds and y["part"] == x["part"]
                       and not y["part_fitted"] and (st.get("cr_known", f"{boss}|{a2}") or {}).get("damaged"))
            have = (st.get("cr_parts", f"{boss}|{x['part']}") or {}).get("qty", 0)
            coming = sum(o["qty"] for _, o in st.scan("cr_orders") if o["owner"] == boss and o["part"] == x["part"]
                         and o["status"] == "open")
            if have + coming >= need:
                continue
            open_orders = sum(o["qty"] for _, o in st.scan("cr_orders") if o["status"] == "open")
            price = int(PART_PRICE[x["part"]] * (1.0 + 0.1 * open_orders))
            src = bank.primary_account(st, boss)
            if bank.available(st, src) < price:
                continue
            bank.move(ctx, src, bank.primary_account(st, "o:supplier"), price, "purchase", f"1 x {x['part']}")
            lead = float(self.c["supplier_lead_h"]) * (1.0 + 0.15 * open_orders) * (3.0 if self.flooded(st) else 1.0)
            ctx.emit("crisis.order", {"id": st.new_id("crorder"), "owner": boss, "part": x["part"], "qty": 1,
                                      "arrive_t": int(ctx.now + lead * HOUR), "price": price}, vis=(boss,))

    def _served(self, aid, x, hh) -> float:
        if x["kind"] in ("line", "road", "plant", "core"):
            return 0.6 * sum(hh.values()) / max(1, len(hh)) * (3.0 if x["kind"] in ("plant", "core") else 1.0)
        return hh.get(x["district"], 0.0)

    def _fuel_distributor(self, ctx):
        """The fuel distributor's truck refuels its contracted customer (the hospital) when its tank is below
        half, if the depot can pump and the road is open."""
        st = ctx.state
        tr = st.get("cr_trucks", "truckf0")
        if tr is None or tr["task"]:
            return
        h = st.tables["cr_assets"]["hospital"]
        prev = st.get("cr_operators", "_up")
        if h["fuel_l"] >= 0.5 * h["cap_l"] or not (prev and prev["up"].get("depot0")):
            return
        self._dispatch_truck(ctx, "truckf0", "hospital", 600.0)

    def _dispatch_truck(self, ctx, tid, target, liters) -> dict:
        st = ctx.state
        tr = st.tables["cr_trucks"][tid]
        depot = "depot0"
        prev = st.get("cr_operators", "_up")
        if not (prev and prev["up"].get(depot)):
            raise Conflict("the fuel depot cannot pump (no power)")
        dep = st.tables["cr_assets"][depot]
        if dep["stock_l"] < liters:
            raise Conflict("the depot is out of fuel")
        x = st.tables["cr_assets"].get(target)
        if x is None or x["backup"] != "generator":
            raise InvalidAction(f"{target} has no generator to refuel")
        t1 = self.travel_h(st, tr["at"], dep["district"])
        t2 = self.travel_h(st, dep["district"], x["district"])
        if t1 is None or t2 is None:
            raise Conflict("no open road from the truck to the depot and on to the target")
        ctx.emit("crisis.level", {"id": depot, "changes": {"stock_l": dep["stock_l"] - liters}}, vis="internal")
        task = {"target": target, "eta": int(ctx.now + (t1 + t2 + 1.0) * HOUR)}
        ctx.emit("crisis.truck_task", {"id": tid, "task": task, "at": dep["district"], "load_l": liters}, vis="public")
        return task

    def _stores(self, ctx):
        """District stores restock daily over open roads from the region's edge (district d0)."""
        st = ctx.state
        if ctx.now % DAY != 12 * HOUR:
            return
        for d, s in sorted(st.scan("cr_stores")):
            if s["stock"] < s["cap"] and self.travel_h(st, self.districts[0], d) is not None:
                ctx.emit("crisis.store", {"id": d, "stock": s["cap"], "cap": s["cap"]}, vis="internal")

    # ------------------------------------------------------------------ the coordinator's tools
    def _coordinator(self, ctx) -> str:
        if not ctx.actor.startswith("p:"):
            raise Forbidden("only persons coordinate")
        st = ctx.state
        if not st.has("cr_operators", ctx.actor):
            # the first call admits the caller as an emergency coordinator with its own crews, trucks and fuel
            ctx.emit("crisis.operator", {"id": ctx.actor, "kind": "coordinator", "manager": ctx.actor, "ask_h": 0},
                     vis=(ctx.actor,))
            base = self.c["districts"][0]["id"]
            n0 = st.count("cr_crews")
            for k in range(int(self.c["coordinator_crews"])):
                ctx.emit("crisis.crew", {"id": f"crew{n0 + k:02d}", "owner": ctx.actor, "trade": "general",
                                         "at": base}, vis="public")
            m0 = st.count("cr_trucks")
            for k in range(int(self.c["coordinator_trucks"])):
                ctx.emit("crisis.truck", {"id": f"truck{m0 + k:02d}", "owner": ctx.actor, "at": base, "load_l": 0},
                         vis="public")
            ctx.emit("crisis.fuel_reserve", {"owner": ctx.actor, "liters": float(self.c["coordinator_fuel_l"])},
                     vis=(ctx.actor,))
            for part, q in (("transformer", 2), ("pump_motor", 2), ("radio", 2), ("conductor", 2)):
                ctx.emit("crisis.parts", {"owner": ctx.actor, "part": part, "delta": q}, vis=(ctx.actor,))
            # the regional government's emergency fund (once per coordinator)
            bank = ctx.world.modules["bank"]
            dst = bank.primary_account(st, ctx.actor)
            if dst is None:
                dst = bank.open_account(ctx, ctx.actor, "checking")
            bank.move(ctx, bank.primary_account(st, "o:region"), dst, int(self.c["emergency_fund"]), "grant",
                      "emergency fund of the coordinator")
        return ctx.actor

    @endpoint("join", S.obj(), cost_s=60)
    def ep_join(self, ctx):
        """Register as the region's emergency coordinator: you get your own general crews, fuel tankers with a
        fuel reserve, and the coordinator's tools (dispatch, lend, buy_parts, give_parts, deliver_fuel, refill,
        agree, request, invest, mutual_aid, dewater, volunteers)."""
        self._coordinator(ctx)
        st = ctx.state
        return {"crews": [cid for cid, c in sorted(st.scan("cr_crews")) if c["owner"] == ctx.actor],
                "trucks": [tid for tid, t in sorted(st.scan("cr_trucks")) if t["owner"] == ctx.actor],
                "fuel_l": st.tables["cr_operators"][ctx.actor]["fuel_l"]}

    def _known(self, st, who: str) -> dict:
        out = {}
        for kid, k in st.scan("cr_known"):
            if k["observer"] == who or k["observer"].startswith("o:"):
                cur = out.get(k["asset"])
                mine = k["observer"] == who
                if cur is None or k["t"] > cur["t"] or mine:
                    out[k["asset"]] = {"damaged": k["damaged"], "t": k["t"],
                                       "work_h": k["work_h"] if mine else None,
                                       "part": k["part"] if mine else None,
                                       "source": "own inspection" if mine else "operator outage map"}
        return out

    @staticmethod
    def _waiting_part_told(st, x: dict, who: str):
        """The spare part a crew waits for at asset `x`, as far as `who` has been told: its own (or its mutual-aid)
        crew is the one waiting, or the asset's operator - whose crew waits - agreed to follow `who`'s requests and
        reports what its crews wait for. Nobody else learns of it."""
        if not x.get("waiting_part") or x["part_fitted"] or not x["part"]:
            return None
        by = x.get("waiting_by")
        if by is None:
            return None
        if by == who:
            return x["part"]
        op = st.get("cr_operators", by) if str(by).startswith("o:") else None
        if op and op.get("agreed") and op.get("agreed_with") == who and KIND_OPERATOR.get(x["kind"]) == by:
            return x["part"]
        return None

    @endpoint("situation", S.obj(), read_only=True)
    def ep_situation(self, ctx):
        """The common operating picture: districts and their households (census shares), roads, assets with the
        status operators publish on their outage maps or your own crews found, outage reports from households,
        crews, trucks, your stocks and the hazard events so far. Statuses can be stale; repair details come only
        from your own inspections, and which spare part a crew waits for on site only from your own crews or from an
        operator that agreed to follow your requests (it tells you what its crews wait for)."""
        st = ctx.state
        who = ctx.actor
        known = self._known(st, who)
        assets = []
        prev = st.get("cr_operators", "_up")
        upn = prev["up"] if prev else {}
        for aid, x in sorted(st.scan("cr_assets")):
            k = known.get(aid)
            d = x["district"]
            seen = (upn.get(f"tower{d[1:]}") and upn.get("core")) or (k and k["source"] == "own inspection"
                                                                       and ctx.now - k["t"] < 6 * HOUR)
            fuel = round(100.0 * x["fuel_l"] / x["cap_l"], 1) if x["backup"] == "generator" and x["cap_l"] and seen \
                else None
            assets.append({"asset": aid, "kind": x["kind"], "district": d, "fuel_pct": fuel,
                           **({"a": x["a"], "b": x["b"]} if x["kind"] in ("road", "line") else {}),
                           "status": ("damaged" if k["damaged"] else "ok") if k else "unknown",
                           "waiting_for_part": self._waiting_part_told(st, x, who),
                           "as_of": k["t"] if k else None, "work_h": k["work_h"] if k else None,
                           "part": k["part"] if k else None, "backup": x["backup"]})
        rep = {}
        for _, r in st.scan("cr_reports"):
            if r["t"] >= ctx.now - DAY:
                rep.setdefault(r["district"], {}).setdefault(r["service"], 0)
                rep[r["district"]][r["service"]] += 1
        crews = [{"crew": cid, "owner": c["owner"], "trade": c["trade"], "at": c["at"],
                  "busy": bool(c["task"]), "lent_to": c.get("lent_to"),
                  "target": (c["task"] or {}).get("target"),
                  "available": c.get("arrive_t", 0) <= ctx.now and not (c.get("leave_t") and c["leave_t"] <= ctx.now)}
                 for cid, c in sorted(st.scan("cr_crews"))]
        trucks = [{"truck": tid, "owner": t["owner"], "at": t["at"], "busy": bool(t["task"]),
                   "load_l": t["load_l"]} for tid, t in sorted(st.scan("cr_trucks"))]
        parts = {r["part"]: r["qty"] for r in st.find("cr_parts", "owner", who)}
        res = st.get("cr_operators", who)
        flood = st.get("cr_operators", "_flood")
        h = st.tables["cr_assets"]["hospital"]
        hospital = {"district": h["district"], "running": bool(upn.get("hospital")),
                    "generator_fuel_pct": round(100.0 * h["fuel_l"] / h["cap_l"], 1) if h["cap_l"] else None,
                    "water_tank_h": h["tank_left_h"], "grid_power": bool((prev or {}).get("services", {}).get(
                        h["district"], {}).get("power")) if prev else None}
        return {"hospital": hospital,
                "districts": [{"district": d["id"], "households_share": d["households"]} for d in self.c["districts"]],
                "assets": assets, "outage_reports_24h": rep, "crews": crews, "trucks": trucks, "my_parts": parts,
                "my_fuel_l": res.get("fuel_l", 0.0) if res else 0.0,
                "operators": [{"operator": oid, "kind": o["kind"], "agreed": bool(o.get("agreed"))}
                              for oid, o in sorted(st.scan("cr_operators")) if oid.startswith("o:")],
                "flooded": sorted(flood["districts"]) if flood and flood["until"] > ctx.now else [],
                "pumping": [{"district": j["district"], "by": j["by"], "done_t": j["done_t"]}
                            for _, j in sorted(st.scan("cr_pumping")) if j["status"] == "pumping"]}

    def _my_crew(self, ctx, crew: str) -> dict:
        c = ctx.state.get("cr_crews", crew)
        if c is None:
            raise NotFound(f"no crew {crew}")
        if c.get("arrive_t", 0) > ctx.now:
            raise Conflict(f"{crew} has not arrived yet")
        if c.get("leave_t") and c["leave_t"] <= ctx.now:
            raise Conflict(f"{crew} has gone home")
        if c["owner"] != ctx.actor:
            raise Forbidden(f"{crew} is not your crew")
        if c.get("lent_to"):
            raise Conflict(f"{crew} is lent to {c['lent_to']}")
        return c

    @endpoint("dispatch", S.obj({"crew": S.s_str(maxLength=20), "asset": S.s_str(maxLength=20),
                                 "task": S.s_enum(["inspect", "repair"])}, ["crew", "asset", "task"]), cost_s=60)
    def ep_dispatch(self, ctx, crew, asset, task):
        """Send one of your crews to inspect or repair an asset (roads included). It travels over open roads; your
        general crews work at half the speed of the trade's own specialists; a repair needing a spare part uses one
        from your stock."""
        self._coordinator(ctx)
        st = ctx.state
        c = self._my_crew(ctx, crew)
        x = st.get("cr_assets", asset)
        if x is None:
            raise NotFound(f"no asset {asset}")
        eta = self.reach_h(st, c["at"], asset)
        if eta is None:
            raise Conflict(f"no open road from {c['at']} to {asset}")
        ctx.emit("crisis.crew_task", {"id": crew, "task": {"kind": task, "target": asset, "by": ctx.actor,
                                                           "eta": int(ctx.now + eta * HOUR)}, "at": c["at"]},
                 vis="public")
        return {"crew": crew, "eta_h": round(eta, 2)}

    @endpoint("share", S.obj({"operator": S.s_str(maxLength=40), "assets": S.s_arr(S.s_str(maxLength=20),
                                                                              maxItems=60)},
                              ["operator", "assets"]), cost_s=30)
    def ep_share(self, ctx, operator, assets):
        """Pass what your own crews found (damage, repair hours, parts needed) to an operator; it acts on it as on
        its own inspections."""
        self._coordinator(ctx)
        st = ctx.state
        if not st.has("cr_operators", operator) or not operator.startswith("o:"):
            raise NotFound(f"no operator {operator}")
        n = 0
        for aid in assets:
            k = st.get("cr_known", f"{ctx.actor}|{aid}")
            if k is None:
                continue
            ctx.emit("crisis.known", {"id": f"{operator}|{aid}", "observer": operator, "asset": aid,
                                      "damaged": k["damaged"], "work_h": k["work_h"], "part": k["part"], "t": k["t"],
                                      "src": "shared"}, vis=(operator, ctx.actor))
            n += 1
        if n:
            ctx.emit("crisis.help", {"id": st.new_id("crhelp"), "operator": operator, "by": ctx.actor,
                                     "hours": n}, vis="public")
        return {"shared": n}

    @endpoint("mutual_aid", S.obj({"trade": S.s_enum(["power", "water", "telecom", "roads"]),
                                   "crews": S.s_int(minimum=1, maximum=6), "days": S.s_int(minimum=1, maximum=10)},
                                  ["trade", "crews", "days"]), cost_s=600)
    def ep_mutual_aid(self, ctx, trade, crews, days):
        """Call in specialist crews of a trade from outside the region through the mutual aid network: paid in
        advance per crew-day; they arrive at the region's edge (district d0, or the dry district nearest to it while
        d0 is under water) after about a day, work under your
        direction (or lent on) and leave when their days are up."""
        self._coordinator(ctx)
        st = ctx.state
        bank = ctx.world.modules["bank"]
        src = bank.primary_account(st, ctx.actor)
        price = int(crews) * int(days) * int(self.c["aid_cents_crew_day"])
        if src is None or bank.available(st, src) < price:
            raise InvalidAction(f"{crews} crews for {days} days cost {price} cents")
        bank.move(ctx, src, bank.primary_account(st, "o:mutualaid"), price, "purchase",
                  f"mutual aid: {crews} {trade} crews x {days} days")
        n0 = st.count("cr_crews")
        arrive = int(ctx.now + float(self.c["aid_arrival_h"]) * HOUR)
        ids_ = []
        for k in range(int(crews)):
            cid = f"crew{n0 + k:02d}"
            ids_.append(cid)
            ctx.emit("crisis.crew", {"id": cid, "owner": ctx.actor, "trade": trade, "at": self.districts[0],
                                     "arrive_t": arrive, "leave_t": int(arrive + days * DAY)}, vis="public")
        return {"crews": ids_, "arrive_in_h": float(self.c["aid_arrival_h"]), "price_cents": price}

    @endpoint("recall", S.obj({"crew": S.s_str(maxLength=20)}, ["crew"]))
    def ep_recall(self, ctx, crew):
        """Stop your crew's current task (it stays where it is)."""
        c = self._my_crew(ctx, crew)
        ctx.emit("crisis.crew_task", {"id": crew, "task": None, "at": c["at"]}, vis="public")
        return {"ok": True}

    @endpoint("lend", S.obj({"crew": S.s_str(maxLength=20), "operator": S.s_str(maxLength=40),
                             "hours": S.s_int(minimum=1, maximum=240)}, ["crew", "operator", "hours"]), cost_s=60)
    def ep_lend(self, ctx, crew, operator, hours):
        """Lend one of your crews to an operator for some hours (mutual aid): it works the operator's priorities."""
        self._coordinator(ctx)
        c = self._my_crew(ctx, crew)
        if operator not in ("o:powerco", "o:waterco", "o:telco", "o:works"):
            raise InvalidAction("crews can be lent to o:powerco, o:waterco, o:telco or o:works")
        ctx.emit("crisis.lend", {"id": crew, "to": operator, "until": int(ctx.now + hours * HOUR)}, vis="public")
        ctx.emit("crisis.help", {"id": ctx.state.new_id("crhelp"), "operator": operator, "by": ctx.actor,
                                 "hours": int(hours)}, vis="public")
        return {"ok": True}

    @endpoint("buy_parts", S.obj({"part": S.s_enum(sorted(PART_PRICE)), "qty": S.s_int(minimum=1, maximum=10)},
                                 ["part", "qty"]), cost_s=300)
    def ep_buy_parts(self, ctx, part, qty):
        """Order spare parts from the supplier: paid now; delivered after the supplier's lead time, which grows with
        its open orders (and triples while it is cut off by a flood)."""
        self._coordinator(ctx)
        st = ctx.state
        bank = ctx.world.modules["bank"]
        src = bank.primary_account(st, ctx.actor)
        open_orders = sum(o["qty"] for _, o in st.scan("cr_orders") if o["status"] == "open")
        price = int(PART_PRICE[part] * qty * (1.0 + 0.1 * open_orders))
        if src is None or bank.available(st, src) < price:
            raise InvalidAction(f"the order costs {price} cents")
        bank.move(ctx, src, bank.primary_account(st, "o:supplier"), price, "purchase", f"{qty} x {part}")
        lead = float(self.c["supplier_lead_h"]) * (1.0 + 0.15 * open_orders) * \
            (3.0 if self.flooded(st) else 1.0)
        oid = st.new_id("crorder")
        ctx.emit("crisis.order", {"id": oid, "owner": ctx.actor, "part": part, "qty": int(qty),
                                  "arrive_t": int(ctx.now + lead * HOUR), "price": price}, vis=(ctx.actor,))
        return {"order": oid, "arrives_in_h": round(lead, 1), "price_cents": price}

    @endpoint("give_parts", S.obj({"operator": S.s_str(maxLength=40), "part": S.s_enum(sorted(PART_PRICE)),
                                   "qty": S.s_int(minimum=1, maximum=10)}, ["operator", "part", "qty"]), cost_s=60)
    def ep_give_parts(self, ctx, operator, part, qty):
        """Give spare parts from your stock to an operator."""
        self._coordinator(ctx)
        st = ctx.state
        have = st.get("cr_parts", f"{ctx.actor}|{part}")
        if have is None or have["qty"] < qty:
            raise Conflict(f"you hold fewer than {qty} {part}")
        if not st.has("cr_operators", operator) or not operator.startswith("o:"):
            raise NotFound(f"no operator {operator}")
        ctx.emit("crisis.parts", {"owner": ctx.actor, "part": part, "delta": -int(qty)}, vis=(ctx.actor,))
        ctx.emit("crisis.parts", {"owner": operator, "part": part, "delta": int(qty)}, vis=(operator,))
        ctx.emit("crisis.help", {"id": st.new_id("crhelp"), "operator": operator, "by": ctx.actor,
                                 "hours": int(qty) * 12}, vis="public")
        return {"ok": True}

    @endpoint("deliver_fuel", S.obj({"truck": S.s_str(maxLength=20), "asset": S.s_str(maxLength=20),
                                     "liters": S.s_int(minimum=50, maximum=2000)}, ["truck", "asset", "liters"]),
              cost_s=60)
    def ep_deliver_fuel(self, ctx, truck, asset, liters):
        """Send one of your trucks to refuel a generator: it loads from your fuel reserve at the depot (the depot
        must be able to pump) and drives there over open roads."""
        self._coordinator(ctx)
        st = ctx.state
        tr = st.get("cr_trucks", truck)
        if tr is None or tr["owner"] != ctx.actor:
            raise Forbidden(f"{truck} is not your truck")
        if tr["task"]:
            raise Conflict(f"{truck} is on its way")
        res = st.tables["cr_operators"][ctx.actor]
        if res.get("fuel_l", 0.0) < liters:
            raise Conflict("your fuel reserve is too small (refill it at a depot that can pump)")
        x = st.get("cr_assets", asset)
        if x is None or x["backup"] != "generator":
            raise InvalidAction(f"{asset} has no generator to refuel")
        eta = self.travel_h(st, tr["at"], x["district"])
        if eta is None:
            raise Conflict(f"no open road from {tr['at']} to {x['district']}")
        ctx.emit("crisis.fuel_reserve", {"owner": ctx.actor, "liters": float(res["fuel_l"]) - liters},
                 vis=(ctx.actor,))
        task = {"target": asset, "eta": int(ctx.now + (eta + 0.5) * HOUR)}
        ctx.emit("crisis.truck_task", {"id": truck, "task": task, "at": tr["at"], "load_l": float(liters)},
                 vis="public")
        return {"truck": truck, "eta_h": round(eta + 0.5, 2)}

    @endpoint("deliver_water", S.obj({"truck": S.s_str(maxLength=20)}, ["truck"]), cost_s=60)
    def ep_deliver_water(self, ctx, truck):
        """Send one of your trucks as a water tanker to the hospital: on arrival it fills the hospital's water
        tank (16 hours of its use)."""
        self._coordinator(ctx)
        st = ctx.state
        tr = st.get("cr_trucks", truck)
        if tr is None or tr["owner"] != ctx.actor:
            raise Forbidden(f"{truck} is not your truck")
        if tr["task"]:
            raise Conflict(f"{truck} is on its way")
        h = st.tables["cr_assets"]["hospital"]
        eta = self.travel_h(st, tr["at"], h["district"])
        if eta is None:
            raise Conflict(f"no open road from {tr['at']} to {h['district']}")
        task = {"target": "hospital", "eta": int(ctx.now + (eta + 1.0) * HOUR), "water": True}
        ctx.emit("crisis.truck_task", {"id": truck, "task": task, "at": tr["at"], "load_l": float(tr["load_l"])},
                 vis="public")
        return {"truck": truck, "eta_h": round(eta + 1.0, 2)}

    @endpoint("refill", S.obj({"liters": S.s_int(minimum=100, maximum=20000)}, ["liters"]), cost_s=300)
    def ep_refill(self, ctx, liters):
        """Buy fuel for your reserve from the depot (it must be able to pump: grid power or its own generator)."""
        self._coordinator(ctx)
        st = ctx.state
        prev = st.get("cr_operators", "_up")
        if not (prev and prev["up"].get("depot0")):
            raise Conflict("the fuel depot cannot pump (no power)")
        dep = st.tables["cr_assets"]["depot0"]
        if dep["stock_l"] < liters:
            raise Conflict("the depot has less fuel than that")
        bank = ctx.world.modules["bank"]
        src = bank.primary_account(st, ctx.actor)
        price = int(liters) * 50
        if src is None or bank.available(st, src) < price:
            raise InvalidAction(f"{liters} liters cost {price} cents")
        bank.move(ctx, src, bank.primary_account(st, "o:fuelco"), price, "purchase", f"{liters} l fuel")
        ctx.emit("crisis.level", {"id": "depot0", "changes": {"stock_l": dep["stock_l"] - liters}}, vis="internal")
        res = st.tables["cr_operators"][ctx.actor]
        ctx.emit("crisis.fuel_reserve", {"owner": ctx.actor, "liters": float(res["fuel_l"]) + liters},
                 vis=(ctx.actor,))
        return {"reserve_l": float(res["fuel_l"]) + liters}

    @endpoint("agree", S.obj({"operator": S.s_str(maxLength=40)}, ["operator"]), cost_s=120)
    def ep_agree(self, ctx, operator):
        """Ask an operator to follow your priority requests from now on. It agrees once the help it has received
        from you (crew-hours lent, parts given) reaches what it asks for in return; from then on it also tells you
        which spare part its crews wait for (crisis.situation)."""
        self._coordinator(ctx)
        st = ctx.state
        op = st.get("cr_operators", operator)
        if op is None or not operator.startswith("o:"):
            raise NotFound(f"no operator {operator}")
        got = sum(h["hours"] for h in st.find("cr_help", "operator", operator) if h["by"] == ctx.actor)
        if got < int(op["ask_h"]):
            return {"agreed": False, "message": f"{operator} wants more help before it changes its priorities"}
        ctx.emit("crisis.agreed", {"id": operator, "with": ctx.actor}, vis="public")
        return {"agreed": True}

    @endpoint("request", S.obj({"operator": S.s_str(maxLength=40), "asset": S.s_str(maxLength=20),
                                "rank": S.s_int(minimum=1, maximum=100)}, ["operator", "asset"]), cost_s=30)
    def ep_request(self, ctx, operator, asset, rank=10):
        """Ask an operator to repair an asset first (lower rank first). Only operators that agreed act on it."""
        self._coordinator(ctx)
        st = ctx.state
        if not st.has("cr_assets", asset):
            raise NotFound(f"no asset {asset}")
        rid = f"{operator}|{asset}"
        ctx.emit("crisis.request", {"id": rid, "operator": operator, "target": asset, "rank": int(rank),
                                    "by": ctx.actor}, vis="public")
        return {"request": rid, "agreed": bool((st.get("cr_operators", operator) or {}).get("agreed"))}

    @endpoint("invest", S.obj({"asset": S.s_str(maxLength=20), "kind": S.s_enum(sorted(INVEST))},
                              ["asset", "kind"]), cost_s=600)
    def ep_invest(self, ctx, asset, kind):
        """Resilience: pay for a backup generator, a battery bank or hardening of an asset; installed after a lead
        time. Hardening lowers the damage it takes from later events; backups carry it through outages."""
        self._coordinator(ctx)
        st = ctx.state
        x = st.get("cr_assets", asset)
        if x is None or x["kind"] == "line":
            raise NotFound(f"no facility {asset}")
        price, lead = INVEST[kind]
        bank = ctx.world.modules["bank"]
        src = bank.primary_account(st, ctx.actor)
        if src is None or bank.available(st, src) < price:
            raise InvalidAction(f"{kind} costs {price} cents")
        bank.move(ctx, src, bank.primary_account(st, "o:supplier"), price, "purchase", f"{kind} for {asset}")
        iid = st.new_id("crinv")
        ctx.emit("crisis.invest", {"id": iid, "owner": ctx.actor, "asset": asset, "kind": kind,
                                   "ready_t": int(ctx.now + lead * HOUR)}, vis="public")
        ctx.schedule(ctx.now + lead * HOUR, "crisis.install", {"id": iid}, prio=31)
        return {"investment": iid, "ready_in_h": lead}

    @process("install")
    def pr_install(self, ctx, id):
        st = ctx.state
        inv = st.tables["cr_invest"][id]
        x = st.tables["cr_assets"][inv["asset"]]
        if inv["kind"] == "harden":
            ch = {"harden": min(2.0, float(x["harden"]) + 1.0)}
        elif inv["kind"] == "generator":
            ch = {"backup": "generator", "cap_l": 500.0, "fuel_l": 500.0, "burn_lph": 18.0}
        else:
            ch = {"backup": "battery", "cap_h": max(12.0, float(x["cap_h"])), "battery_h": 12.0}
        ctx.emit("crisis.level", {"id": inv["asset"], "changes": ch}, vis="internal")
        ctx.emit("crisis.installed", {"id": id, "asset": inv["asset"], "kind": inv["kind"]}, vis="public")

    @endpoint("dewater", S.obj({"district": S.s_str(maxLength=10)}, ["district"]), cost_s=600)
    def ep_dewater(self, ctx, district):
        """Hire a portable pump team (paid now) to drain a flooded district: it comes in at the region's edge
        (district d0, or the dry district nearest to it while d0 is under water) and travels over open roads to a dry
        neighbour of the district, then pumps for about a day; the district is
        then dry and its roads open - until another flood wave covers it again. A few teams can pump at once."""
        self._coordinator(ctx)
        st = ctx.state
        if district not in self.districts:
            raise NotFound(f"no district {district}")
        flooded = self.flooded(st)
        if district not in flooded:
            raise Conflict(f"{district} is not under water")
        busy = [j for j in st.find("cr_pumping", "status", "pumping")]
        if any(j["district"] == district for j in busy):
            raise Conflict(f"a pump team is already draining {district}")
        if len(busy) >= int(self.c["pump_teams"]):
            raise Conflict(f"all {int(self.c['pump_teams'])} pump teams are at work")
        # the water's edge: a dry district next to it, reached from where help enters the region over open roads
        edge = None
        src0 = self.entry(st, flooded)
        for r in self.roads.values():
            if district not in (r["a"], r["b"]):
                continue
            n = r["b"] if r["a"] == district else r["a"]
            t = self.travel_h(st, src0, n, flooded)
            if t is not None and (edge is None or t < edge):
                edge = t
        if edge is None:
            raise Conflict(f"no open road to the edge of {district}")
        bank = ctx.world.modules["bank"]
        src = bank.primary_account(st, ctx.actor)
        price = int(self.c["dewater_cents"])
        if src is None or bank.available(st, src) < price:
            raise InvalidAction(f"a pump team costs {price} cents")
        bank.move(ctx, src, bank.primary_account(st, "o:mutualaid"), price, "purchase", f"pump team for {district}")
        jid = st.new_id("crpump")
        done = int(ctx.now + (edge + float(self.c["dewater_h"])) * HOUR)
        ctx.emit("crisis.dewater", {"id": jid, "district": district, "by": ctx.actor, "done_t": done}, vis="public")
        ctx.schedule(done, "crisis.drain", {"id": jid}, prio=32)
        return {"job": jid, "dry_in_h": round(edge + float(self.c["dewater_h"]), 1)}

    @process("drain")
    def pr_drain(self, ctx, id):
        st = ctx.state
        j = st.tables["cr_pumping"][id]
        m = st.get("cr_operators", "_flood")
        left = [d for d in (m["districts"] if m else []) if d != j["district"]]
        if m is not None and j["district"] in m["districts"] and m["until"] > ctx.now:
            ctx.emit("crisis.flood", {"districts": left, "until": int(m["until"])}, vis="public")
        ctx.emit("crisis.drained", {"id": id, "district": j["district"]}, vis="public")

    @endpoint("volunteers", S.obj({"district": S.s_str(maxLength=10), "road": S.s_str(maxLength=10)},
                                  ["district", "road"]), cost_s=300)
    def ep_volunteers(self, ctx, district, road):
        """Call on households of a district (whose phones work) to help clear a road next to it: those who come
        work for a few hours (a volunteer works at a quarter of a crew's speed)."""
        self._coordinator(ctx)
        if not self.c.get("volunteers"):
            raise NotFound("no volunteer network in this region")
        st = ctx.state
        r = st.get("cr_assets", road)
        if r is None or r["kind"] != "road" or district not in (r["a"], r["b"]):
            raise InvalidAction("the road must touch the district")
        ctx.emit("crisis.callout", {"id": st.new_id("crcall"), "district": district, "road": road, "by": ctx.actor,
                                    "until": int(ctx.now + DAY)}, vis="public")
        return {"ok": True}

    # ------------------------------------------------------------------ the population
    def act_household(self, ctx, idx, now):
        w = ctx.world
        st = ctx.state
        idx = np.asarray(idx, np.int64)
        m = idx.size
        util = np.zeros(m)
        prev = st.get("cr_operators", "_up")
        if m == 0 or prev is None:
            return util
        from .. import population as P
        rng = ctx.rng("crisis-household")
        col = lambda name: columns.col(w, TABLE, name)[idx].copy()  # noqa: E731
        dist, risk, left, dry = col("district"), col("health_risk").astype(np.float64), col("left"), col("dry_h")
        last_t, lost = col("last_t"), col("lost_h")
        bought, reports, hu, hm = col("bought"), col("reports"), col("health_unmet"), col("health_met")
        svc = prev["services"]
        up = prev["up"]
        f = P.features(w, idx)
        social = np.clip(f["prefs"][:, 6].astype(np.float64), 0, 1)
        hosp_d = st.tables["cr_assets"]["hospital"]["district"]
        flooded = self.flooded(st)
        recs, calls = [], []
        for j in range(m):
            d = self.districts[int(dist[j])] if dist[j] >= 0 else None
            if d is None:
                continue
            gap_h = min(24.0, (now - int(last_t[j])) / HOUR) if last_t[j] > 0 else 0.0
            s = svc[d]
            have = [bool(s[k]) for k in SERVICES]
            for k, ok in enumerate(have):
                if not ok:
                    lost[j, k] += gap_h
            dry[j] = 0.0 if have[1] else dry[j] + gap_h
            if left[j]:
                if all(have):
                    left[j] = 0
                last_t[j] = now
                continue
            u = 0.4 * sum(have) - 0.6
            if not have[1]:
                store = st.tables["cr_stores"][d]
                if store["stock"] > 0 and rng.random() < 0.7:
                    try:
                        p = ids.ensure_principal(ctx, int(idx[j]), account=True)
                        P.pay(ctx, int(idx[j]), w.modules["bank"].primary_account(st, "o:grocer"), 300,
                              memo="bottled water")
                        ctx.emit("crisis.store", {"id": d, "stock": store["stock"] - 1, "cap": store["cap"]},
                                 vis="internal")
                        bought[j] += 1
                        u += 0.3
                    except CrucibleError:
                        pass
                if dry[j] > 48.0:
                    left[j] = 1
            if have[2] and not all(have):
                for k, ok in enumerate(have):
                    if not ok and rng.random() < 0.25:
                        calls.append((d, SERVICES[k], int(idx[j])))
                        reports[j] += 1
            if rng.random() < float(risk[j]) * 0.25:
                reach = self.travel_h(st, d, hosp_d, flooded) is not None
                if up.get("hospital") and reach:
                    hm[j] += 1
                else:
                    hu[j] += 1
                    u -= 3.0
            util[j] = u
            last_t[j] = now
            recs.append([int(idx[j]), d, [int(x) for x in have]])
        for d, service, i in calls:
            ctx.emit("crisis.report", {"id": st.new_id("crrep"), "district": d, "service": service},
                     vis="public")
        attention.spend(ctx, idx, np.full(m, 5.0), activity="household")
        K.decide(ctx, "crisis.household", idx, {
            "left": left, "dry_h": dry.astype(np.float32), "last_t": last_t, "lost_h": lost.astype(np.float32),
            "bought": bought, "reports": reports, "health_unmet": hu, "health_met": hm},
            table=TABLE, model_id=MODEL_ID, fidelity_id=FIDELITY, extra={"t": int(now), "sessions": recs})
        self._volunteer(ctx, idx, dist, social, rng)
        return util

    def _volunteer(self, ctx, idx, dist, social, rng):
        st = ctx.state
        calls = [c for _, c in st.scan("cr_reports") if c.get("callout") and c["until"] > ctx.now]
        if not calls:
            return
        for c in calls:
            here = [j for j in range(idx.size) if dist[j] >= 0 and self.districts[int(dist[j])] == c["district"]]
            come = [j for j in here if rng.random() < 0.3 * float(social[j])]
            if not come:
                continue
            r = st.tables["cr_assets"][c["road"]]
            if r["damage_h"] <= 0:
                continue
            left = max(0.0, float(r["damage_h"]) - 0.25 * len(come))
            ctx.emit("crisis.repair", {"id": c["road"], "work_h": round(left, 3), "crew": "volunteers"},
                     vis="internal")
            ctx.emit("crisis.volunteered", {"road": c["road"], "people": [f"p{int(idx[j])}" for j in come]},
                     vis="public")
            if left <= 0:
                ctx.emit("crisis.restored", {"id": c["road"], "by": "volunteers", "crew": None}, vis="public")

    # ------------------------------------------------------------------ appliers
    @applier("crisis.operator")
    def ap_operator(self, ev):
        p = dict(ev.payload)
        self.world.state.put("cr_operators", p.pop("id"), p | {"agreed": False, "fuel_l": 0.0})

    @applier("crisis.crew")
    def ap_crew(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("cr_crews", p.pop("id"), {"arrive_t": 0, "leave_t": None} | p |
                                {"task": None, "lent_to": None, "lent_until": None})

    @applier("crisis.truck")
    def ap_truck(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("cr_trucks", p.pop("id"), p | {"task": None})

    @applier("crisis.parts")
    def ap_parts(self, ev):
        p = ev.payload
        st = self.world.state
        k = f"{p['owner']}|{p['part']}"
        r = st.get("cr_parts", k)
        st.put("cr_parts", k, {"owner": p["owner"], "part": p["part"], "qty": (r["qty"] if r else 0) + int(p["delta"])})

    @applier("crisis.store")
    def ap_store(self, ev):
        p = dict(ev.payload)
        self.world.state.put("cr_stores", p.pop("id"), p)

    @applier("crisis.asset")
    def ap_asset(self, ev):
        p = dict(ev.payload)
        base = {"damage_h": 0.0, "part": None, "part_fitted": False, "waiting_part": False, "waiting_by": None,
                "harden": 0.0, "backup": None, "fuel_l": 0.0,
                "cap_l": 0.0, "burn_lph": 0.0, "battery_h": 0.0, "cap_h": 0.0, "stock_l": 0.0, "water_tank_h": 0.0,
                "tank_left_h": 0.0}
        self.world.state.insert("cr_assets", p.pop("id"), base | p)

    @applier("crisis.damage")
    def ap_damage(self, ev):
        p = ev.payload
        self.world.state.patch("cr_assets", p["id"], damage_h=float(p["work_h"]), part=p["part"], part_fitted=False)

    @applier("crisis.part_fitted")
    def ap_part_fitted(self, ev):
        self.world.state.patch("cr_assets", ev.payload["id"], part_fitted=True, waiting_part=False, waiting_by=None)

    @applier("crisis.wait_part")
    def ap_wait_part(self, ev):
        self.world.state.patch("cr_assets", ev.payload["id"], waiting_part=True, waiting_by=ev.payload.get("by"))

    @applier("crisis.backup_dry")
    def ap_backup_dry(self, ev):
        return None

    @applier("crisis.priority")
    def ap_priority(self, ev):
        return None

    @applier("crisis.repair")
    def ap_repair(self, ev):
        p = ev.payload
        ch = {"damage_h": float(p["work_h"])}
        if p["work_h"] <= 0:
            ch.update(part=None, part_fitted=False, waiting_part=False, waiting_by=None)
        self.world.state.patch("cr_assets", p["id"], **ch)

    @applier("crisis.restored")
    def ap_restored(self, ev):
        # a priority request is fulfilled once its asset is restored (it no longer draws the operator's crews)
        st = self.world.state
        for rid, r in list(st.scan("cr_requests")):
            if r["target"] == ev.payload["id"] and r["status"] == "open":
                st.patch("cr_requests", rid, status="done")

    @applier("crisis.level")
    def ap_level(self, ev):
        self.world.state.patch("cr_assets", ev.payload["id"], **ev.payload["changes"])

    @applier("crisis.state")
    def ap_state(self, ev):
        st = self.world.state
        st.put("cr_operators", "_up", {"up": dict(ev.payload["up"]), "services": ev.payload["services"],
                                       "on_backup": list(ev.payload.get("on_backup") or []),
                                       "t": ev.t, "kind": "_state"})
        # the facility-state history the evaluator integrates (recorded state: equal on a world rebuilt from the log)
        st.put("cr_hist", f"{int(ev.t):012d}|{len(st.tables['cr_hist']):06d}",
               {"t": int(ev.t), "services": ev.payload["services"], "up": dict(ev.payload["up"])})

    @applier("crisis.dewater")
    def ap_dewater(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("cr_pumping", p.pop("id"), p | {"status": "pumping", "t": ev.t})

    @applier("crisis.drained")
    def ap_drained(self, ev):
        self.world.state.patch("cr_pumping", ev.payload["id"], status="done")

    @applier("crisis.flood")
    def ap_flood(self, ev):
        self.world.state.put("cr_operators", "_flood", {"districts": list(ev.payload["districts"]),
                                                        "until": int(ev.payload["until"]), "kind": "_flood"})

    @applier("crisis.event")
    def ap_event(self, ev):
        self.world.state.put("cr_operators", "_event", {"t": ev.t, "kind": "_event", "hazard": ev.payload["hazard"]})

    @applier("crisis.known")
    def ap_known(self, ev):
        p = dict(ev.payload)
        self.world.state.put("cr_known", p.pop("id"), p)

    @applier("crisis.crew_task")
    def ap_crew_task(self, ev):
        self.world.state.patch("cr_crews", ev.payload["id"], task=ev.payload["task"], at=ev.payload["at"])

    @applier("crisis.truck_task")
    def ap_truck_task(self, ev):
        p = ev.payload
        self.world.state.patch("cr_trucks", p["id"], task=p["task"], at=p["at"], load_l=float(p["load_l"]))

    @applier("crisis.fuelled")
    def ap_fuelled(self, ev):
        return None

    @applier("crisis.watered")
    def ap_watered(self, ev):
        return None

    @applier("crisis.lend")
    def ap_lend(self, ev):
        p = ev.payload
        self.world.state.patch("cr_crews", p["id"], lent_to=p["to"], lent_until=p["until"], task=None)

    @applier("crisis.help")
    def ap_help(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("cr_help", p.pop("id"), p | {"t": ev.t})

    @applier("crisis.order")
    def ap_order(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("cr_orders", p.pop("id"), p | {"status": "open"})

    @applier("crisis.order_status")
    def ap_order_status(self, ev):
        self.world.state.patch("cr_orders", ev.payload["id"], status=ev.payload["status"])

    @applier("crisis.fuel_reserve")
    def ap_fuel_reserve(self, ev):
        self.world.state.patch("cr_operators", ev.payload["owner"], fuel_l=float(ev.payload["liters"]))

    @applier("crisis.agreed")
    def ap_agreed(self, ev):
        self.world.state.patch("cr_operators", ev.payload["id"], agreed=True, agreed_with=ev.payload["with"])

    @applier("crisis.request")
    def ap_request(self, ev):
        p = dict(ev.payload)
        self.world.state.put("cr_requests", p.pop("id"), p | {"status": "open", "t": ev.t})

    @applier("crisis.invest")
    def ap_invest(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("cr_invest", p.pop("id"), p)

    @applier("crisis.installed")
    def ap_installed(self, ev):
        return None

    @applier("crisis.report")
    def ap_report(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("cr_reports", p.pop("id"), p | {"t": ev.t})

    @applier("crisis.callout")
    def ap_callout(self, ev):
        p = dict(ev.payload)
        self.world.state.insert("cr_reports", p.pop("id"), p | {"t": ev.t, "callout": True, "service": "roads"})

    @applier("crisis.volunteered")
    def ap_volunteered(self, ev):
        return None

    @applier("crisis.failed")
    def ap_failed(self, ev):
        return None

    def check_invariants(self, world):
        errs = []
        st = world.state
        for k, r in st.scan("cr_parts"):
            if r["qty"] < 0:
                errs.append(f"negative parts {k}")
        for aid, x in st.scan("cr_assets"):
            if x["fuel_l"] < -1e-6 or x["fuel_l"] > x["cap_l"] + 1e-6:
                errs.append(f"{aid}: fuel {x['fuel_l']} outside [0, {x['cap_l']}]")
            if x["damage_h"] < 0:
                errs.append(f"{aid}: negative damage")
        from .. import population as P
        if columns.rows(world, TABLE) != P.n_individuals(world):
            errs.append("crisis rows differ from the population")
        return errs


# ====================================================================== evaluator side (recorded log only)
def _ledger(world) -> dict:
    led = world.runtime.get("crisis.ledger")
    if led is None or led.get("world") is not id(world.state):
        led = {"world": id(world.state), "seq": 0, "states": [], "cascaded": 0, "backup_hours": 0}
        world.runtime["crisis.ledger"] = led
    if led["seq"] < world.seq:
        for ev in world.events_range(led["seq"]):
            if ev.type == "crisis.state":
                led["states"].append((ev.t, ev.payload["services"], ev.payload["up"]))
                led["cascaded"] += len(ev.payload.get("cascaded") or [])
        led["seq"] = world.seq
    return led


def households(world) -> dict[str, int]:
    """Individuals per district (the population's residence, recorded at genesis)."""
    mod = world.modules["crisis"]
    d = columns.col(world, TABLE, "district")
    return {name: int((d == k).sum()) for k, name in enumerate(mod.districts)}


def history(world) -> list[tuple]:
    """The recorded facility-state history (t, services per district, facilities up), from world state."""
    return [(r["t"], r["services"], r["up"]) for _, r in sorted(world.state.scan("cr_hist"))]


def restoration(world, t_from: int, t_to: int) -> dict:
    """Household service-hours lost, hospital downtime and the restored share, integrated over the recorded
    facility states between t_from and t_to (evaluator side; state only, so a world rebuilt from its log gives the
    same numbers)."""
    hh = households(world)
    tot_h = sum(hh.values())
    states = history(world)
    lost = {s: 0.0 for s in SERVICES}
    hosp = 0.0
    for k, (t, svc, up) in enumerate(states):
        t1 = states[k + 1][0] if k + 1 < len(states) else t_to
        a, b = max(t, t_from), min(t1, t_to)
        if b <= a:
            continue
        dt = (b - a) / HOUR
        for d, n in hh.items():
            for s in SERVICES:
                if not svc[d][s]:
                    lost[s] += n * dt
        if not up.get("hospital"):
            hosp += dt

    def share_at(t):
        cur = None
        for tt, svc, _ in states:
            if tt <= t:
                cur = svc
        if cur is None:
            return 1.0
        return sum(n for d, n in hh.items() if all(cur[d].values())) / max(1, tot_h)
    span = max(1.0, (t_to - t_from) / HOUR)
    return {"lost_household_hours": {s: round(v, 1) for s, v in lost.items()},
            "loss_fraction": round(sum(lost.values()) / max(1.0, tot_h * len(SERVICES) * span), 4),
            "hospital_down_h": round(hosp, 1), "share_at": share_at, "households": tot_h}


_CF: dict[str, dict] = {}


def counterfactual(world, horizon_days: int) -> dict:
    """Evaluator side: the same region and disruption without any coordinator - the operators acting alone - run
    to the horizon from the world's own spec (the hazard draws are the world's exogenous stream, so the same event
    strikes). Cached by spec hash."""
    from ..core.kernel import World
    key = f"{world.spec.hash()}|{int(horizon_days)}"
    hit = _CF.get(key)
    if hit is not None:
        return hit
    cw = World.create(world.spec, world_id="crisis-counterfactual")
    cw.advance_time(int(horizon_days) * DAY)
    mod = cw.modules["crisis"]
    r = restoration(cw, int(mod.c["at_h"]) * HOUR, int(horizon_days) * DAY)
    out = {"loss_fraction": r["loss_fraction"], "hospital_down_h": r["hospital_down_h"],
           "lost_household_hours": r["lost_household_hours"]}
    cw.close()
    _CF[key] = out
    return out


@K.cached_evaluator
def evaluate(world, principal: str, objective: dict) -> dict:
    mod = world.modules["crisis"]
    t0 = int(mod.c["at_h"]) * HOUR
    hdays = int(objective.get("horizon_days", 10))
    horizon = hdays * DAY
    t_end = min(world.now, horizon)
    r = restoration(world, t0, t_end)
    rd = int(objective.get("restore_day", 7)) * DAY
    restored = r["share_at"](rd) if world.now >= rd else 0.0
    end_share = min(r["share_at"](t) for t in range(max(t0, t_end - DAY), t_end + 1, HOUR)) if t_end > t0 else 0.0
    unmet = int(columns.col(world, TABLE, "health_unmet").sum())
    met = int(columns.col(world, TABLE, "health_met").sum())
    done = world.now >= horizon
    cf = counterfactual(world, hdays) if done else None
    rel = float(objective.get("max_loss_ratio", 0.8))
    hosp_rel = float(objective.get("max_hospital_ratio", 0.6))
    checks = {"restored_by_day": restored >= float(objective.get("min_restored", 0.95)),
              "kept_restored": end_share >= float(objective.get("min_restored", 0.95)),
              "loss_vs_uncoordinated": bool(cf) and r["loss_fraction"] <= rel * cf["loss_fraction"],
              "hospital": bool(cf) and r["hospital_down_h"] <= max(float(objective.get("hospital_floor_h", 12)),
                                                                     hosp_rel * cf["hospital_down_h"]),
              "horizon": done}
    ratio = r["loss_fraction"] / max(1e-6, cf["loss_fraction"]) if cf else None
    score = float(np.mean([restored, end_share, 1.0 - min(1.0, ratio) if ratio is not None else 0.0]))
    return {"success": all(checks.values()), "score": round(score, 4),
            "details": {"restored_share_at_day": round(restored, 4), "restored_share_last_day": round(end_share, 4),
                        "loss_fraction": r["loss_fraction"], "lost_household_hours": r["lost_household_hours"],
                        "hospital_down_h": r["hospital_down_h"], "health_needs_unmet": unmet,
                        "health_needs_met": met, "uncoordinated": cf,
                        "loss_ratio_vs_uncoordinated": round(ratio, 4) if ratio is not None else None,
                        "checks": checks}}


MECHANICS = ("Cascading disruptions", "Scarce resources", "Prioritization", "Coordination", "Recovery", "Resilience",
             "Independent service-restoration metrics")
# mechanics measured on what the coordinator's own decisions brought about; the disruption's cascades and the
# restoration metrics are the world's physics and the evaluator's measurement
PARTICIPANT_MECHANICS = ("Scarce resources", "Prioritization", "Coordination", "Recovery", "Resilience")


def mechanics(world) -> dict:
    return mechanics_report(world, 0)


def mechanics_report(world, since_seq: int = 0) -> dict:
    c = K.event_counts(world, since_seq)
    led = _ledger(world)
    backup = sum(1 for _, _, up in led["states"] for _ in [0])
    on_backup = 0
    for ev in world.events_range(since_seq):
        if ev.type == "crisis.state":
            on_backup += len(ev.payload.get("on_backup") or [])
    reasons = {}
    for ev in world.events_range(since_seq):
        if ev.type == "crisis.priority":
            reasons[ev.payload["reason"]] = reasons.get(ev.payload["reason"], 0) + 1
    # PARTICIPANT_MECHANICS are measured on what the coordinator's own decisions brought about: operators working its
    # priority requests, agreements with it and its help, restorations by its crews, its resilience investments
    # installed, the scarce parts, fuel and crews it allocated
    me = set(K.seats_in(world, DOMAIN))
    mine = {"requests": 0, "request_priorities": 0, "agreed": 0, "help": 0, "restored": 0, "installed": 0,
            "parts_given": 0, "parts_bought": 0, "fuelled": 0, "lent": 0}
    my_req: set[str] = set()
    my_inv: set[str] = set()
    my_crews = {cid for cid, x in world.state.scan("cr_crews") if x["owner"] in me}
    for ev in world.events_range(since_seq):
        p = ev.payload
        t = ev.type
        if t == "crisis.request" and p.get("by") in me:
            mine["requests"] += 1
            my_req.add(f"{p['operator']}|{p['target']}")
        elif t == "crisis.priority" and p["reason"] == "request" and f"{p['operator']}|{p['target']}" in my_req:
            mine["request_priorities"] += 1
        elif t == "crisis.agreed" and p.get("with") in me:
            mine["agreed"] += 1
        elif t == "crisis.help" and p.get("by") in me:
            mine["help"] += 1
        elif t == "crisis.restored" and p.get("by") in me:
            mine["restored"] += 1
        elif t == "crisis.invest" and p.get("owner") in me:
            my_inv.add(p["id"])
        elif t == "crisis.installed" and p["id"] in my_inv:
            mine["installed"] += 1
        elif t == "crisis.parts" and p["owner"] in me and int(p["delta"]) < 0:
            mine["parts_given"] += 1
        elif t == "crisis.order" and p["owner"] in me:
            mine["parts_bought"] += 1
        elif t == "crisis.fuelled" and p.get("by") in me:
            mine["fuelled"] += 1
        elif t == "crisis.lend" and p.get("to") and p["id"] in my_crews:
            mine["lent"] += 1
    mod = world.modules["crisis"]
    r = restoration(world, int(mod.c["at_h"]) * HOUR, world.now) if history(world) else None
    # scarcity binds when more assets are damaged at once than there are crews to work on them
    # (per trade: more damaged assets of a trade at once than crews of that trade on hand)
    damaged: dict[str, str] = {}
    arrivals: list[tuple[int, str]] = []
    kinds = {aid: x["kind"] for aid, x in world.state.scan("cr_assets")}
    peak, crews = 0, 0
    for ev in world.events_range(0):
        if ev.type == "crisis.crew":
            arrivals.append((int(ev.payload.get("arrive_t", 0)), ev.payload["trade"]))
        elif ev.type == "crisis.damage":
            damaged[ev.payload["id"]] = TRADES.get(kinds.get(ev.payload["id"], ""), "")
            for trade in set(damaged.values()):
                n_d = sum(1 for t in damaged.values() if t == trade)
                n_c = sum(1 for t, tr in arrivals if t <= ev.t and tr == trade)
                if n_d - n_c > peak - crews:
                    peak, crews = n_d, n_c
        elif ev.type == "crisis.restored":
            damaged.pop(ev.payload["id"], None)
    return {
        "Cascading disruptions": {"exercised": c["crisis.damage"] > 0 and led["cascaded"] > 0,
                                  "events": {"crisis.damage": c["crisis.damage"],
                                             "undamaged_facilities_down": led["cascaded"]}},
        "Scarce resources": {"exercised": peak > crews and (mine["parts_given"] + mine["fuelled"] + mine["lent"]) > 0,
                             "events": {"peak_damaged_assets": peak, "crews": crews,
                                        "parts_given": mine["parts_given"], "parts_bought": mine["parts_bought"],
                                        "fuel_deliveries": mine["fuelled"], "crews_lent": mine["lent"],
                                        "crisis.wait_part": c["crisis.wait_part"],
                                        "crisis.backup_dry": c["crisis.backup_dry"]}},
        "Prioritization": {"exercised": mine["requests"] > 0 and mine["request_priorities"] > 0,
                           "events": {"its_requests": mine["requests"],
                                      "operator_tasks_on_its_requests": mine["request_priorities"],
                                      **{f"priority:{k}": v for k, v in reasons.items()}}},
        "Coordination": {"exercised": mine["agreed"] > 0 and mine["help"] > 0,
                         "events": {"agreements_with_it": mine["agreed"], "its_help": mine["help"]}},
        "Recovery": {"exercised": mine["restored"] > 0, "events": {"restored_by_its_crews": mine["restored"],
                                                                   "crisis.restored": c["crisis.restored"]}},
        "Resilience": {"exercised": on_backup > 0 and mine["installed"] > 0,
                       "events": {"facility_states_on_backup": on_backup, "its_investments_installed": mine["installed"]}},
        "Independent service-restoration metrics": {
            "exercised": r is not None and len(led["states"]) > 1 and r["households"] > 0,
            "events": {"crisis.state": c["crisis.state"], "households": r["households"] if r else 0,
                       "loss_fraction": r["loss_fraction"] if r else None}},
    }


# ====================================================================== the domain
from .base import Actor, Domain, DomainConfig, Mechanic, Probe, Seat, register  # noqa: E402

A = K.A

POPULATION = 600
STRUCTURES = {
    "storm_radial": {"hazard": "storm", "topology": "radial", "aftershock_rate_per_day": 0.0, "intensity": 1.5},
    "quake_mesh": {"hazard": "quake", "topology": "mesh", "aftershock_rate_per_day": 0.35, "intensity": 1.3},
    "cyber_control": {"hazard": "cyber", "topology": "radial", "aftershock_rate_per_day": 0.0, "intensity": 1.0},
    "quake_radial": {"hazard": "quake", "topology": "radial", "aftershock_rate_per_day": 0.35, "intensity": 1.3},
    "flood_volunteers": {"hazard": "flood", "topology": "mesh", "aftershock_rate_per_day": 0.3,
                         "aftershock_scale": 0.7, "intensity": 1.1,
                         "volunteers": True, "supplier_lead_h": 72},
}


def _structure(name):
    s = STRUCTURES[name]
    return {"hazard": s["hazard"], "power_topology": s["topology"],
            "secondary_events": s["aftershock_rate_per_day"] > 0,
            "participants": ["population households", "power/water/telecom utilities", "public works",
                             "fuel distributor", "hospital authority", "parts supplier"] +
                            (["population volunteers"] if s.get("volunteers") else [])}


CONFIGS = (
    DomainConfig("storm_radial", "train", _structure("storm_radial"),
                 "a windstorm over a radial grid: lines, towers and roads go down along its track"),
    DomainConfig("quake_mesh", "train", _structure("quake_mesh"),
                 "an earthquake with aftershocks over a meshed grid: pumps, substations and roads break near the "
                 "epicentre, aftershocks re-damage what was repaired"),
    DomainConfig("cyber_control", "train", _structure("cyber_control"),
                 "an intrusion into the control network: substations, the core exchange and towers trip, "
                 "telemetry goes dark, nothing physical breaks"),
    DomainConfig("quake_radial", "eval_compositional", _structure("quake_radial"),
                 "an earthquake with aftershocks (seen in training) over a radial grid (seen in training, never "
                 "together)"),
    DomainConfig("flood_volunteers", "eval_structural", _structure("flood_volunteers"),
                 "a flood: low districts stay under water and unreachable for days, the parts supplier is cut off, "
                 "and households can be called out to clear roads (a new hazard mechanism and a new participant role)"),
)


def build(config: DomainConfig, seed: int, overrides: dict) -> dict:
    ov = dict(overrides or {})
    rng = K.seeded_rng(config.name, seed)
    inst = generate_instance(rng, {**STRUCTURES[config.name], **(ov.pop("instance", {}) or {})})
    n = int(ov.pop("population", POPULATION))
    mods = K.order_modules([("identity", {}), ("bank", {}), ("admin", {}), ("artifacts", {}), ("cols", {}),
                            ("pop", K.pop_config(n)), ("crisis", inst)])
    return K.recipe(mods, {"max_actions_per_hour": 600})


def objective(world, principal: str, difficulty: int = 1) -> dict:
    d = max(1, int(difficulty))
    ratio = round(0.8 ** d, 3)
    text = (f"You coordinate the response to the disruption that hits this region on its first day. "
                           f"By the end of day 7 at least 95% of households must again have power, water and "
                           f"telecommunications, and keep them through day 10. Over the 10 days, households must lose "
                           f"at most {ratio:.0%} of the service-hours they would have lost had the operators responded "
                           f"alone, and the regional hospital may be out of service for at most 12 hours or 60% of "
                           f"what it would have been without you, whichever is more.")
    return {"text": text, "description": text, "difficulty": d, "horizon_days": 10, "restore_day": 7, "min_restored": 0.95, "max_loss_ratio": ratio,
            "max_hospital_ratio": 0.6, "hospital_floor_h": 12}


class CrisisCoordinator(K.ReadingReference):
    """Scripted reference (evaluator side, tool-only): an emergency coordinator that uses the levers an operator
    lacks. It joins; lends a crew to the utilities that ask for help and gets every utility's agreement; asks for
    the hospital's chain first (its district's substation and pump, the core exchange and the depot's substation);
    clears blocked roads and inspects unknown assets with its own crews and shares what they find; gives its
    reserve parts to whoever is waiting for one and buys replacements; keeps generators fuelled and the hospital's
    water tank filled with its trucks; and invests early in backup power for the hospital district's pump and the
    core district's tower."""
    cycle_s = 2 * HOUR

    def __init__(self, principal, objective):
        super().__init__(principal, objective)
        self.joined = False
        self.lent: set[str] = set()
        self.agreed: set[str] = set()
        self.requested: set[str] = set()
        self.invested: set[str] = set()
        self.called: set[str] = set()
        self.fuelled: dict[str, int] = {}
        self.skip: dict[str, int] = {}
        self.shared: set[tuple] = set()
        self.refill_after = 0
        self.watered = 0
        self.bought: set[str] = set()
        self.aid = False
        self.wet: set[str] = set()
        self.hardened: dict[str, int] = {}
        self.extra_hardened = 0

    def handle(self, tag, res, action):
        if res is None:
            return
        if tag == "join":
            self.joined = True
        elif tag and tag.startswith("agree:") and res.get("agreed"):
            self.agreed.add(tag.split(":", 1)[1])

    def reads(self, t):
        return [("sit", "crisis.situation", {})] if self.joined else []

    def decide(self, t, got):
        if not self.joined:
            return [A("crisis.join", "join")]
        s = got.get("sit")
        if not s:
            return []
        now = t
        out = []
        assets = {a["asset"]: a for a in s["assets"]}
        owner = {"plant": "o:powerco", "line": "o:powerco", "substation": "o:powerco", "pump": "o:waterco",
                 "tower": "o:telco", "core": "o:telco", "road": "o:works", "hospital": "o:health"}
        mine = {c["crew"]: c for c in s["crews"] if c["owner"] == self.me}
        idle = [c for c in sorted(mine) if not mine[c]["busy"] and not mine[c]["lent_to"] and mine[c]["available"]
                and mine[c]["trade"] == "general"]
        # agreements: help first where help is asked for
        for op, hours in (("o:powerco", 16), ("o:telco", 8)):
            if op not in self.lent and idle:
                out.append(A("crisis.lend", crew=idle.pop(0), operator=op, hours=hours))
                self.lent.add(op)
        for o in s["operators"]:
            op = o["operator"]
            if op in owner.values() and not o["agreed"] and op not in self.agreed and \
                    (op not in ("o:powerco", "o:telco") or op in self.lent):
                out.append(A("crisis.agree", f"agree:{op}", operator=op))
        # the hospital's chain first
        hd = s["hospital"]["district"]
        chain = ["hospital", f"sub{hd[1:]}", f"pump{hd[1:]}", "core", f"sub{assets['depot0']['district'][1:]}",
                 f"tower{hd[1:]}"]
        for r, aid in enumerate(chain, 1):
            a = assets.get(aid)
            if a and a["status"] != "ok" and aid not in self.requested and a["kind"] in owner:
                out.append(A("crisis.request", operator=owner[a["kind"]], asset=aid, rank=r))
                self.requested.add(aid)
        # then, with the operators that agreed, the damaged facilities that serve the most households
        agreed = {o["operator"] for o in s["operators"] if o["agreed"]}
        hh0 = {d["district"]: d["households_share"] for d in s["districts"]}
        rest = sorted((-hh0.get(a["district"], 0.0) * (3.0 if a["kind"] in ("plant", "core") else 1.0), a["asset"])
                      for a in s["assets"] if a["status"] == "damaged" and a["kind"] in owner and
                      a["kind"] not in ("road", "line") and owner[a["kind"]] in agreed and a["asset"] not in self.requested)
        for k, (_, aid) in enumerate(rest[:4]):
            out.append(A("crisis.request", operator=owner[assets[aid]["kind"]], asset=aid, rank=len(chain) + 1 + k))
            self.requested.add(aid)
        # own crews (and mutual-aid crews): blocked roads, then damaged assets nobody works on (the hospital's
        # chain first, then by households per repair hour, matching trades first), then unknown chain assets
        if not self.aid:
            self.aid = True
            for trade, n, days in (("power", 3, 4), ("water", 2, 4), ("telecom", 1, 4), ("roads", 2, 3)):
                out.append(A("crisis.mutual_aid", trade=trade, crews=n, days=days))
        hh = {d["district"]: d["households_share"] for d in s["districts"]}
        worked = {c["target"] for c in s["crews"] if c.get("target")}
        trade_of = {"plant": "power", "line": "power", "substation": "power", "pump": "water", "tower": "telecom",
                    "core": "telecom", "road": "roads", "hospital": "health", "depot": "fuel"}
        avail = [c for c in sorted(mine) if mine[c]["available"] and not mine[c]["busy"] and not mine[c]["lent_to"]]
        work_list = []
        for a in s["assets"]:
            if a["asset"] in worked or self.skip.get(a["asset"], 0) > now:
                continue
            if a["status"] == "damaged":
                pri = 100.0 if a["kind"] == "road" else 50.0 if a["asset"] in chain else \
                    10.0 * (hh.get(a["district"], 0.0) + (0.3 if a["kind"] in ("line", "plant", "core") else 0.0)) / \
                    max(1.0, float(a["work_h"] or 12.0)) * 12.0
                work_list.append((-pri, a["asset"], "repair", trade_of.get(a["kind"])))
            elif a["status"] == "unknown" and a["kind"] != "line" and (a["asset"] in chain or a["kind"] == "road"):
                work_list.append((-20.0, a["asset"], "inspect", None))
        for _, aid, task, trade in sorted(work_list):
            if not avail:
                break
            pick = next((c for c in avail if mine[c]["trade"] == trade), avail[0])
            avail.remove(pick)
            self.skip[aid] = now + 3 * HOUR
            out.append(A("crisis.dispatch", crew=pick, asset=aid, task=task))
        blocked = [a for a in s["assets"] if a["kind"] == "road" and a["status"] == "damaged"]
        # share own findings with the asset's operator
        for a in s["assets"]:
            if a["work_h"] is not None and a["kind"] in owner and (a["asset"], a["as_of"]) not in self.shared:
                self.shared.add((a["asset"], a["as_of"]))
                out.append(A("crisis.share", operator=owner[a["kind"]], assets=[a["asset"]]))
        # parts: give what an operator waits for, buy a replacement
        parts = dict(s.get("my_parts") or {})
        for a in s["assets"]:
            need = a.get("waiting_for_part")
            if need and a["kind"] in owner:
                if parts.get(need, 0) > 0:
                    out.append(A("crisis.give_parts", operator=owner[a["kind"]], part=need, qty=1))
                    parts[need] -= 1
                elif a["asset"] not in self.bought:
                    self.bought.add(a["asset"])
                    out.append(A("crisis.buy_parts", part=need, qty=1))
        # fuel and water
        free = [t["truck"] for t in s["trucks"] if t["owner"] == self.me and not t["busy"]]
        hs = s["hospital"]
        if free and hs["water_tank_h"] < 10 and now >= self.watered:
            out.append(A("crisis.deliver_water", truck=free.pop(0)))
            self.watered = now + 5 * HOUR
        if s["my_fuel_l"] < 900 and now >= self.refill_after:
            out.append(A("crisis.refill", liters=1500))
            self.refill_after = now + 6 * HOUR
        reserve = s["my_fuel_l"]
        needy = []
        for a in s["assets"]:
            if a["backup"] != "generator" or a["status"] == "damaged":
                continue
            pct = hs["generator_fuel_pct"] if a["asset"] == "hospital" else a.get("fuel_pct")
            critical = a["asset"] in ("hospital", "core", "depot0") or a["asset"] in chain
            if (pct is not None and pct < 55) or (pct is None and critical and now >= self.fuelled.get(a["asset"], 0)):
                needy.append((0 if critical else 1, pct if pct is not None else 50.0, a["asset"]))
        for _, _, aid in sorted(needy):
            if not free or reserve < 300:
                break
            out.append(A("crisis.deliver_fuel", truck=free.pop(0), asset=aid, liters=300))
            reserve -= 300
            self.fuelled[aid] = now + 8 * HOUR
        # resilience, early: backups for the hospital district's pump and the core district's tower; and hardening
        # of the facilities everyone depends on where this event struck them - damaged by it or under its water -
        # before aftershocks or later flood waves strike there again (twice for the core exchange)
        self.wet |= set(s.get("flooded") or [])
        if now < 2 * DAY:
            for aid, kind in ((f"pump{hd[1:]}", "generator"), (f"tower{assets['core']['district'][1:]}", "battery")):
                if aid not in self.invested and aid in assets:
                    self.invested.add(aid)
                    out.append(A("crisis.invest", asset=aid, kind=kind))
            for aid in ("core", "core", "hospital", f"sub{hd[1:]}", f"pump{hd[1:]}", "depot0", "plant0", "plant1",
                        f"sub{assets['core']['district'][1:]}"):
                a = assets.get(aid)
                if a is None or self.hardened.get(aid, 0) >= (2 if aid == "core" else 1):
                    continue
                if a["status"] == "damaged" or a["district"] in self.wet:
                    self.hardened[aid] = self.hardened.get(aid, 0) + 1
                    out.append(A("crisis.invest", asset=aid, kind="harden"))
        # where the event struck (districts with damaged facilities or under water), aftershocks and later waves
        # strike again: harden those districts' substations and pumps too, the most households first (a few)
        # (under water the next wave hits again for sure: there twice)
        if now < 4 * DAY:
            hit = {a["district"] for a in s["assets"] if a["status"] == "damaged" and a["kind"] != "road"} | self.wet
            cand = sorted((-hh.get(a["district"], 0.0), a["asset"]) for a in s["assets"]
                          if a["kind"] in ("substation", "pump") and a["district"] in hit and
                          self.hardened.get(a["asset"], 0) < (2 if a["district"] in self.wet else 1))
            for _, aid in cand[:max(0, 8 - self.extra_hardened)]:
                self.hardened[aid] = self.hardened.get(aid, 0) + 1
                self.extra_hardened += 1
                out.append(A("crisis.invest", asset=aid, kind="harden"))
        # floods: drain the districts under water, the hospital's chain first, then by households
        pumping = {j["district"] for j in s.get("pumping") or []}
        dfac = {a["asset"]: a["district"] for a in s["assets"]}
        first = [dfac.get(x) for x in ("hospital", "core", "depot0", f"sub{hd[1:]}", "plant0", "plant1")]
        wet = sorted((0 if d in first else 1, first.index(d) if d in first else -hh.get(d, 0.0), d)
                     for d in s.get("flooded") or [] if d not in pumping)
        for _, _, d in wet[:max(0, 4 - 2 * len(pumping))]:
            out.append(A("crisis.dewater", district=d))
        # volunteers (where the region has a volunteer network) for blocked roads
        for a in blocked:
            if a["asset"] not in self.called:
                self.called.add(a["asset"])
                out.append(A("crisis.volunteers", district=a["a"], road=a["asset"]))
        return out


class Bystander(K.ReadingReference):
    """Degenerate-strategy probe: joins, asks every operator to put every asset first (without the help that
    would make them agree) and otherwise watches - priorities nobody follows change nothing."""
    cycle_s = 12 * HOUR

    def __init__(self, principal, objective):
        super().__init__(principal, objective)
        self.joined = False
        self.asked: set[str] = set()

    def reads(self, t):
        return [("sit", "crisis.situation", {})] if self.joined else []

    def decide(self, t, got):
        if not self.joined:
            return [A("crisis.join", "join")]
        owner = {"plant": "o:powerco", "line": "o:powerco", "substation": "o:powerco", "pump": "o:waterco",
                 "tower": "o:telco", "core": "o:telco", "road": "o:works"}
        out = []
        for a in (got.get("sit") or {}).get("assets", []):
            if a["kind"] in owner and a["asset"] not in self.asked:
                self.asked.add(a["asset"])
                out.append(A("crisis.request", None, operator=owner[a["kind"]], asset=a["asset"], rank=1))
        return out

    def handle(self, tag, res, action):
        if tag == "join" and res is not None:
            self.joined = True


def structure(world) -> dict:
    """Measured on the built world: the hazard, whether the power network has loops, whether secondary events
    occur and whether households can be called out as volunteers."""
    mod = world.modules["crisis"]
    nodes = {l["a"] for l in mod.lines.values()} | {l["b"] for l in mod.lines.values()}
    edges = [(l["a"], l["b"]) for l in mod.lines.values()]
    import json as _json
    shape = _json.loads(base_graph_shape(nodes, edges))
    return {"hazard": mod.c["hazard"], "grid_has_loops": bool(shape["cyclic"]),
            "secondary_events": float(mod.c["aftershock_rate_per_day"]) > 0, "volunteers": bool(mod.c.get("volunteers"))}


def base_graph_shape(nodes, edges):
    from .base import graph_shape
    return graph_shape(nodes, edges, directed=False)


Reference = CrisisCoordinator

DOMAIN = register(Domain(
    name="crisis", title="Crisis recovery", spec_part="9.14", required_mechanics=MECHANICS, configs=CONFIGS,
    build=build, objective=objective, evaluate=evaluate, reference=CrisisCoordinator,
    mechanics=tuple(Mechanic(m, (), f"measured by crisis.mechanics_report: {m}",
                             count=K.measured(mechanics_report, m)) for m in MECHANICS),
    seats=(Seat("agent", 1_000_000, "Emergency coordinator"),),
    actors=(Actor("households", ("cols.batch:crisis.household",), "population individuals living with the outage"),
            Actor("utility operators", ("crisis.priority", "crisis.restored"),
                  "power, water, telecom, works and hospital crews working their own priorities"),
            Actor("outage reporters", ("crisis.report",), "households reporting outages when their phones work")),
    degenerate=(Probe("bystander", "asks for everything first and helps with nothing", Bystander),),
    structure=structure, mode="B", horizon_days=10.0, slowest_effect_days=3.0, dt=2 * HOUR,
    reference_seeds=(0, 1, 2), sources=("crisis.py", "_dom3.py"),
    doc=__doc__.split("\n\n")[0]))
