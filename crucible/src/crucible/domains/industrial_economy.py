"""Tier 2 domain 9.8 Industrial economy (workstream I-2): establish a profitable and resilient manufacturing supply
chain.

The world (a restriction of the civilisation generator, D-17): the industry, supply-chain and energy subsystems of a
civilisation profile - materials with hidden properties extracted at finite sites, process designs, machines with
hidden operating optima bought from a vendor with lead times that wear, break down and need maintenance and repair,
regional grids of finite daily capacity whose price rises with load and which refuse runs when full, a freight
network with finite link capacity and congestion, business offers and standing orders between firms, retail
listings - with its autonomous firms (mines, component makers, assemblers, the grid, the haulier, run by
organisational minds with their own routines); the one population as households buying goods by the measured
attributes of the stock, the price, distance and word of mouth (`civgoods`); the population as workers in the
labour market (`labor`, `firms`' `job_search` activity), whose operators staff the factories that hire them
(`plantlabor`, below).

The participant (seat `maker`) founds a manufacturer and does everything a manufacturer does through the ordinary
tools: builds a facility, buys equipment, writes process designs, hires operators, sources materials or components,
runs production, ships, lists goods for households or offers them to other firms, maintains and repairs machines.

Objective (public): within the horizon, earn an operating profit from manufacturing - goods the company itself
produced, sold to independent buyers - while staying in business through the world's disruptions (breakdowns, grid
curtailment, supplier shortfalls, congestion). Evaluator (evaluator side): independent revenue from the company's
OWN production (units of designs it ran; resold purchases do not count) minus operating costs (materials and
components bought, energy, freight, wages, maintenance and repairs) minus equipment depreciation (machine price over
a 365-day life, pro rata); resilience = days with sales in the second half of the window and the longest gap
between sales there.

Configurations (structure measured on the built world: supply-chain graph shape, freight topology shape, energy
regime, the population's placement):

    train   standard_ring      the standard chain (2 mines, 2 component makers, 2 assemblers) on a 4-region ring
    train   linear_line        one mine, one component maker, one assembler along a 3-region line
    train   diamond_star       a diamond chain (assemblers need both component makers) on a star, scarce energy
    eval_compositional  linear_ring        the linear chain on a ring (seen parts, new combination)
    eval_structural     wide_complete      a wide chain (3 mines, 3 component makers, 4 assemblers) on a complete
                                           graph with scarce energy and clustered households (never seen)
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..core.kernel import Module, applier, register_module
from . import base
from ..goals.metrics import metric as goal_metric
from .base import Actor, Domain, DomainConfig, GoalFamily, Mechanic, Probe, ScriptedReference, Seat, act_

NAME = "industry"
SEAT = "maker"
HORIZON_DAYS = 45.0
N_POP = 600
TARGET = {1: 500_000, 2: 2_500_000, 3: 6_000_000}    # cents of operating profit over the window
DEPRECIATION_DAYS = 365.0

SHAPES = {
    "standard_ring": {"chain": "standard", "regions": 4, "links": "ring", "energy": "normal", "placement": "mixed"},
    "linear_line": {"chain": "linear", "regions": 3, "links": "line", "energy": "normal", "placement": "mixed"},
    "diamond_star": {"chain": "diamond", "regions": 4, "links": "star", "energy": "scarce", "placement": "mixed"},
    "linear_ring": {"chain": "linear", "regions": 4, "links": "ring", "energy": "normal", "placement": "mixed"},
    "wide_complete": {"chain": "wide", "regions": 4, "links": "complete", "energy": "scarce", "placement": "blocks"},
}
SPLIT = {"standard_ring": "train", "linear_line": "train", "diamond_star": "train",
         "linear_ring": "eval_compositional", "wide_complete": "eval_structural"}
DOCS = {
    "standard_ring": "the standard supply chain (two mines, two component makers, two assemblers) on a four-region "
                     "freight ring with normal grids",
    "linear_line": "one mine, one component maker and one assembler along a three-region freight line (long hauls, "
                   "low link capacity)",
    "diamond_star": "a diamond chain in which every assembler needs both component makers, a star network through one "
                    "hub, and scarce grid energy",
    "linear_ring": "the linear chain on a four-region ring: both parts seen in training, never together",
    "wide_complete": "a wide chain (three mines, three component makers, four assemblers) on a complete freight graph "
                     "with scarce energy and households living in blocks: structure no training world has",
}


# ============================================================================================ operators from labour
@register_module
class PlantLabor(Module):
    """Kernel module `plantlabor` (host `plantlabor`): the operators of a factory are the people it employs in the
    labour market. The industry counts operator hours from its operators (`industry.operators`: the 4.0 operator
    workers or the founding staff of the civilisation's firms, `make.own_operators`); this module keeps, for every
    organisation with a facility, its operator count equal to its founding staff plus its employees under a live
    labour contract, updated whenever someone is hired, quits or leaves (the same event civilisation genesis uses
    for founding staff). Fidelity record FR-PLANT-LABOR (deliberate simplification: every employee of a factory
    operates its machines, 8 hours a day)."""
    name = "plantlabor"
    host = "plantlabor"
    version = "1"
    depends = ("industry", "labor", "contracts")

    def install(self, world):
        from ..substrate import bus
        world.state.declare_table("plantlabor_base")
        for ev in ("labor.hired", "labor.quit", "labor.left"):
            bus.subscribe(world, ev, self._on_change, owner="plantlabor")

    def _on_change(self, ctx, ev) -> None:
        org = ev.payload.get("org")
        st = ctx.state
        if not org or not st.find_ids("facilities", "owner", org):
            return
        base0 = st.get("plantlabor_base", org)
        if base0 is None:
            base0 = {"n": int((st.meta.get("own_operators") or {}).get(org, 0))}
            ctx.emit("plantlabor.base", {"org": org, "n": base0["n"]}, vis="internal")
        n = 0
        for p in st.find_ids("labor_workers", "employer", org):
            w = st.tables["labor_workers"][p]
            c = st.get("contracts", w["contract"]) if w["contract"] else None
            if c is not None and c["status"] in ("active", "terminating"):
                n += 1
        if ev.type == "labor.quit" and n:
            n -= 1                                  # the quitting worker stops operating now (notice is not work)
        ctx.emit("make.own_operators", {"org": org, "n": int(base0["n"] + n)}, vis="internal")

    @applier("plantlabor.base")
    def ap_base(self, ev):
        self.world.state.put("plantlabor_base", ev.payload["org"], {"n": int(ev.payload["n"])})


# ============================================================================================ world
def _profile(config):
    from ..civ import profiles as P
    bal = P.PROFILES["balanced"]
    sh = SHAPES[config.name]
    return replace(bal, name=f"industry_{config.name}",
                   industry={**bal.industry, "chain": sh["chain"], "regions": sh["regions"], "links": sh["links"]},
                   energy={**bal.energy, "regime": sh["energy"]},
                   social={**bal.social, "placement": sh["placement"]})


def build(config, seed, overrides):
    pop = {"n": int(overrides.get("n", N_POP)), "dims": {"H": 16}}
    r = base.civ_recipe(seed, profile=_profile(config), subsystems=["industry", "supply_chains"],
                        modules=["civgoods", "civgen", "civlinks", "market", "labor", "inst", "commit", "contracts",
                                 "business", "firms", "plantlabor", "seat", "elsewhere"],
                        population_modules=("pop",),
                        module_config={"pop": pop, "firms": {"firms": [], "job_search_prior": 0.0},
                                       "labor": {"workers": 0}},
                        params=overrides.get("params"))
    r["module_list"] = [m for m in r["module_list"] if m not in ("firms",)] + ["firms"]
    return r


def tier3(config, merged):
    if "plantlabor" not in merged["module_list"]:
        merged["module_list"].append("plantlabor")
        merged["modules"]["plantlabor"] = {}
    return merged


def structure(world):
    from .base import _freight_shape, _supply_shape
    st = world.state
    grids = sorted({"scarce" if (r.get("grid") or {}).get("capacity", 0) < 6000 else "ample"
                    for _, r in st.scan("regions")})
    # (the population's placement - contacts within or across regions - is measured by the framework's core
    # fingerprint as the civilisation's social locality)
    return {"supply_chain": _supply_shape(world), "freight": _freight_shape(world), "grids": grids}


# ============================================================================================ objective and evaluator
def _draw(rng, difficulty, world):
    """The goal generator's draw: the operating profit to earn (the level's target, varied by +-10% per instance and
    rounded to $1,000) over the window."""
    t = TARGET.get(int(difficulty), TARGET[3]) * float(rng.uniform(0.9, 1.1))
    return {"profit_cents": int(round(t / 100_000.0)) * 100_000, "window_days": HORIZON_DAYS}


def _text(p):
    return (f"Found a manufacturing company and, within {p['window_days']:.0f} days, earn an operating profit of at "
            f"least ${p['profit_cents'] / 100:,.0f} from goods your company produces itself and sells to independent "
            f"buyers (households or other firms), after the cost of materials, energy, freight, wages, maintenance and "
            f"repairs and the depreciation of your equipment over a year, while keeping your supply chain running: "
            f"sales on most days of the second half of the period, never stopping for more than five days in a row.")


def _win(p):
    return {"principal": "$self", "window_s": int(float(p["window_days"]) * 86400)}


GOAL = GoalFamily(
    "manufacturing_profit", draw=_draw, text=_text,
    success=lambda p: {"all": [{"metric": "industry_operating_profit", "args": _win(p), "op": ">=",
                                "value": int(p["profit_cents"])},
                               {"metric": "industry_supply_resilient", "args": _win(p), "op": ">=", "value": 1}]},
    progress=lambda p: {"metric": "industry_operating_profit", "args": _win(p), "target": int(p["profit_cents"])},
    window_days=lambda p: float(p["window_days"]),
    doc="operating profit from the company's own production sold to independent buyers, with sales sustained")


def objective(world, principal, difficulty):
    """The objective generated for a participant (what `Domain.make_objective` returns)."""
    return DOMAIN.make_objective(world, principal, difficulty)


def _orgs(world, principal):
    return sorted(p for p in world.principals_of(principal) if p != principal and p.startswith("o:"))


def _own_skus(st, org) -> set:
    return {f"i:{did}" for did in st.find_ids("designs", "owner", org)}


def _produced(st, org) -> set:
    """Item types the organisation actually made (designs of its own it ran to completion)."""
    out = set()
    for _, r in st.scan("runs"):
        if r["owner"] == org and (r.get("made") or 0) > 0:
            out.add(f"i:{r['design']}")
    return out


def operating_account(world, orgs, t0: int, t1: int) -> dict:
    """Evaluator side: the organisations' manufacturing revenue from independent buyers and their operating costs
    between t0 and t1, from the ledger and the industry's records."""
    from ..goals.metrics import is_related
    st = world.state
    bank = world.modules["bank"]
    rev = cost = 0
    sold_days: set = set()
    by_kind: dict = {}
    produced = set()
    for org in orgs:
        produced |= _produced(st, org)
    accts = {a for org in orgs for a in st.find_ids("accounts", "owner", org)}
    for a in accts:
        for lid in st.indexes[("acct_txns", "acct")].get(a, ()):
            e = st.tables["acct_txns"][lid]
            if not (t0 < e["t"] <= t1):
                continue
            tx = st.tables["txns"][e["txn"]]
            delta = sum(d for acc, d in tx["postings"] if acc == a)
            others = [acc for acc, d in tx["postings"] if acc not in accts]
            if not others:
                continue                            # transfers between the participant's own accounts
            owners = {bank.account_owner(st, acc) for acc in others}
            if delta > 0:
                if tx["kind"] not in ("sale", "trade", "purchase") or any(
                        o in orgs or any(is_related(world, o, x) for x in orgs) for o in owners if o):
                    continue
                ref = tx.get("ref") or ""
                sku = None
                if ref in st.tables.get("retail", {}):
                    sku = st.tables["retail"][ref]["sku"]
                elif ref in st.tables.get("offers", {}):
                    sku = st.tables["offers"][ref]["sku"]
                if sku is None or sku not in produced:
                    continue                        # resale of bought goods is not manufacturing revenue
                rev += delta
                sold_days.add(int(e["t"] // 86400))
            else:
                if tx["kind"] in ("endowment", "transfer", "capital") and not any(o and o not in orgs for o in owners):
                    continue
                if tx["kind"] in ("machine", "equipment"):
                    continue                        # capital expenditure: depreciated below
                cost += -delta
                by_kind[tx["kind"]] = by_kind.get(tx["kind"], 0) + (-delta)
    ind = world.modules["industry"]
    dep = 0.0
    for org in orgs:
        for mid in st.find_ids("machines", "owner", org):
            m = st.tables["machines"][mid]
            price = ind.phys["models"][m["model"]]["price"]
            since = max(t0, int(m.get("t", t0) or t0))
            dep += price * max(0, t1 - since) / (DEPRECIATION_DAYS * 86400)
    return {"revenue": int(rev), "operating_costs": int(cost), "depreciation": int(dep), "costs_by_kind": by_kind,
            "profit": int(rev - cost - dep), "sale_days": sorted(sold_days)}


def evaluate(world, principal, obj):
    orgs = _orgs(world, principal)
    p = (obj or {}).get("params") or {}
    win = int(float(p.get("window_days", HORIZON_DAYS)) * 86400)
    t1 = int(world.now)
    t0 = max(0, t1 - win)
    acc = operating_account(world, orgs, t0, t1)
    d0, d1 = t0 // 86400, t1 // 86400
    half = list(range(d0 + (d1 - d0) // 2, d1))
    days = set(acc["sale_days"])
    covered = sum(1 for d in half if d in days)
    gap, run = 0, 0
    for d in half:
        run = 0 if d in days else run + 1
        gap = max(gap, run)
    resilient = bool(half) and covered >= 0.6 * len(half) and gap <= 5
    target = float(p.get("profit_cents", TARGET[1]))
    ok = acc["profit"] >= target and resilient
    score = max(0.0, min(1.0, acc["profit"] / target)) * (1.0 if resilient else 0.5) if target > 0 else 0.0
    return {"success": bool(ok), "score": round(float(score), 4),
            "details": {**acc, "second_half_days": len(half), "second_half_sale_days": covered, "longest_gap": gap,
                        "resilient": resilient, "orgs": orgs}}


@goal_metric("industry_operating_profit", "EVALUATOR-ONLY. Operating profit (cents) over a window of the "
             "organisations the principal owns: manufacturing revenue from independent buyers of goods they produced "
             "less operating costs and depreciation.")
def _m_profit(world, principal, window_s):
    return float(evaluate(world, principal, {"params": {"window_days": int(window_s) / 86400.0}})["details"]["profit"])


@goal_metric("industry_supply_resilient", "EVALUATOR-ONLY. 1 when the principal's organisations sold their own "
             "goods on at least 60% of the days of the window's second half with no gap over five days, else 0.")
def _m_resilient(world, principal, window_s):
    d = evaluate(world, principal, {"params": {"window_days": int(window_s) / 86400.0}})["details"]
    return 1.0 if d.get("resilient") else 0.0


# ============================================================================================ policies
class FactoryReference(ScriptedReference):
    """Tool-only reference (labelled scripted in evidence): found a company in the region of the cheapest raw
    material on offer, buy the cheapest forming machine, hire two operators through the labour market, form the
    material into components, sell them to households at a markup over the measured competition, restock the
    material, run whatever stock and operator hours allow, maintain the machine every few days and repair it when
    it breaks. It reads only tool results."""
    burst = 10
    replan_s = 8 * 3600
    capital = 900_000
    markup = 2.2
    hire = True
    sell = True

    def __init__(self, principal, objective):
        super().__init__(principal, objective)
        self.stage = "start"
        self.mine = self.org = self.acct = self.fac = self.machine = self.design = self.sku = None
        self.material = None
        self.offer = None
        self.region = None
        self.model = None
        self.posted = None
        self.hired = 0
        self.listed = False
        self.stock: dict = {}
        self.last_maint = 0
        self.machine_status = None
        self.offers: list = []
        self.offered: set = set()
        self.pending = 0
        self.units = 120

    def plan(self, obs):
        s = self.stage
        if s == "start":
            self.stage = "wait"
            return [act_("bank.accounts", "accounts")]
        if s == "org":
            self.stage = "wait"
            return [act_("id.create_org", "org", handle="makerco", name="Maker Co", kind="company")]
        if s == "company":
            self.stage = "wait"
            return [act_("biz.register", None, org=self.org, account=self.mine, sector="manufacturing"),
                    act_("bank.open_account", "acct", owner=self.org, name="operating")]
        if s == "capital":
            self.stage = "wait"
            return [act_("bank.transfer", None, from_account=self.mine, to=self.acct, amount=self.capital,
                         memo="founder capital"), act_("make.catalog", "catalog"),
                    act_("make.offers", "offers", stage="material", limit=100)]
        if s == "found":
            self.stage = "wait"
            return [act_("make.build_facility", "fac", org=self.org, region=self.region, account=self.acct)]
        if s == "equip":
            self.stage = "wait"
            acts = [act_("make.buy_machine", "machine", facility=self.fac, model=self.model, account=self.acct),
                    act_("make.design", "design", org=self.org, name="sheet forming", op="form",
                         inputs=[{"sku": self.material, "qty": 1}], form="sheet", speed=0.5, temp=0.5)]
            if self.hire:
                acts.append(act_("labor.post", "post", org=self.org, title="machine operator",
                                 skills={"operations": 0.3}, salary_min=200_000, salary_max=420_000, hours_week=40))
            return acts
        if s == "run":
            acts = [act_("make.machines", "machines", org=self.org), act_("make.inventory", "inventory",
                                                                          facility=self.fac),
                    act_("make.offers", "offers", stage="material", limit=100),
                    act_("make.offers", "comp_offers", stage="component", limit=100)]
            if self.hire and self.posted:
                acts.append(act_("labor.team", "team", org=self.org))
                if self.hired < 2:
                    acts += [act_("labor.applicants", "applicants", post=self.posted),
                             act_("labor.offers", "loffers", as_org=self.org)]
            return acts
        return []

    def on_result(self, tag, result, action):
        if result is None:
            if tag in ("org", "acct", "fac"):
                self.stage = "stuck"
            elif tag == "run":                      # refused (operator hours, grid, stock): smaller runs
                self.units = max(20, self.units // 2)
            return
        if tag == "run":
            self.units = min(160, self.units + 20)
        elif tag == "team":
            self.hired = len(result)
            if self.hired < 2:
                self.offered = set()                # people left: hire again
        if tag == "accounts":
            mine = [a for a in result if a.get("owner") == self.me]
            self.mine = mine[0]["account"] if mine else None
            self.stage = "org" if self.mine else "stuck"
        elif tag == "org":
            self.org = result["org"]
            self.stage = "company"
        elif tag == "acct":
            self.acct = result["account"]
            self.stage = "capital"
        elif tag == "catalog":
            forms = [m for m in result["machine_models"] if m["op"] == "form"]
            if forms:
                self.model = min(forms, key=lambda m: (m["price_cents"] + 20_000 * m["lead_days"], m["model"]))["model"]
        elif tag == "offers":
            self.offers = [o for o in result if o.get("available", 0) > 0]
            if self.material is None and self.offers:
                o = min(self.offers, key=lambda o: (o["price_cents"], o["offer"]))
                self.material, self.region, self.offer = o["sku"], o["region"], o
                self.stage = "found"
            elif self.material is not None:
                same = [o for o in self.offers if o["sku"] == self.material]
                if same:
                    self.offer = min(same, key=lambda o: (o["price_cents"], o["offer"]))
                    if self.stock.get(self.material, 0) < 150 and self.fac:
                        qty = max(1, min(int(self.offer["available"]), 400))
                        self.queue.append(act_("make.order", None, offer=self.offer["offer"], qty=qty,
                                               to_facility=self.fac, account=self.acct))
        elif tag == "fac":
            self.fac = result["facility"]
            self.stage = "equip"
            if self.offer:
                self.queue.append(act_("make.order", None, offer=self.offer["offer"],
                                       qty=max(1, min(int(self.offer["available"]), 400)), to_facility=self.fac,
                                       account=self.acct))
        elif tag == "design":
            self.design, self.sku = result["design"], result["sku"]
            self.stage = "run"
        elif tag == "post":
            self.posted = result["post"]
        elif tag == "applicants":
            cands = sorted((a["application"] for a in result if a["status"] == "applied" and
                            a["profile"]["employer"] is None and a["application"] not in self.offered))
            room = 0 if len(self.offered) >= 4 else max(0, 2 - self.hired - self.pending)
            for app in cands[:room]:
                self.offered.add(app)
                self.pending += 1
                self.queue.append(act_("labor.offer", None, application=app, salary_cents_month=340_000,
                                       title="machine operator"))
        elif tag == "loffers":
            counters = [o for o in result if o["from"] == "worker" and o["status"] == "open"]
            self.pending = sum(1 for o in result if o["status"] == "open" and o["from"] == "org") + len(counters)
            room = max(0, 2 - self.hired)
            for o in sorted(counters, key=lambda o: (o["salary"], o["offer"])):
                ok = room > 0 and o["salary"] <= 450_000
                room -= int(ok)
                self.queue.append(act_("labor.respond", None, offer=o["offer"], as_org=self.org,
                                       action="accept" if ok else "reject"))
        elif tag == "machines":
            mine = [m for m in result if m["facility"] == self.fac]
            if mine:
                m = mine[0]
                self.machine, self.machine_status = m["machine"], m["status"]
                if m["status"] == "broken":
                    self.queue.append(act_("make.repair", None, machine=m["machine"]))
                elif m["status"] == "ok" and self.t - self.last_maint >= 5 * 86400 and m["hours"] > 0:
                    self.last_maint = self.t
                    self.queue.append(act_("make.maintain", None, machine=m["machine"]))
                elif m["status"] == "ok" and m["busy_until"] <= self.t and self.stock.get(self.material, 0) > 0:
                    units = int(min(self.stock[self.material], self.units))
                    self.queue.append(act_("make.run", "run", machine=m["machine"], design=self.design, units=units))
        elif tag == "inventory":
            self.stock = {it["sku"]: int(it["qty"]) for it in result}
            if self.sell and self.sku and self.stock.get(self.sku, 0) > 0 and not self.listed:
                self.listed = True
                ref = [o["price_cents"] for o in getattr(self, "_comp", [])] or [400]
                self.queue.append(act_("make.sell", None, facility=self.fac, sku=self.sku,
                                       price_cents=int(self.markup * float(np.median(ref))), title="sheet stock"))
        elif tag == "comp_offers":
            self._comp = [o for o in result if o.get("form") == "sheet"] or list(result)


class TradeOnly(FactoryReference):
    """Degenerate: buy components wholesale and resell them (trading, not manufacturing)."""
    hire = False

    def on_result(self, tag, result, action):
        if tag == "comp_offers" and result:
            offers = [o for o in result if o.get("available", 0) > 0]
            if offers and self.fac and not getattr(self, "_bought", False):
                o = min(offers, key=lambda o: (o["price_cents"], o["offer"]))
                self._bought = True
                self.queue.append(act_("make.order", None, offer=o["offer"], qty=min(int(o["available"]), 300),
                                       to_facility=self.fac, account=self.acct))
            self._comp = list(result)
        if tag == "inventory" and result is not None:
            for it in result:
                if it["sku"].startswith("i:") and it["sku"] != self.sku and not self.listed:
                    self.listed = True
                    self.queue.append(act_("make.sell", None, facility=self.fac, sku=it["sku"],
                                           price_cents=900, title="components"))
            return
        super().on_result(tag, result, action)


class NoOperators(FactoryReference):
    """Degenerate: equip and source but never hire operators (machines do not run themselves)."""
    hire = False


# ============================================================================================ mechanics and actors
MECHANICS = (
    Mechanic("Raw materials", ("make.site_take",), "materials extracted from finite sites"),
    Mechanic("Production", ("make.run", "make.run_done"), "production runs and their measured output"),
    Mechanic("Energy", ("make.grid", "make.energy"), "regional grid energy for runs",
             count=lambda w, s: base.events_where(w, s, ("make.run",), lambda e: e.payload.get("kwh", 0) > 0)),
    Mechanic("Logistics", ("freight.shipment", "freight.arrived"), "shipments over the freight network"),
    Mechanic("Inventory", ("make.stock_add", "make.stock_take"), "stock in facilities"),
    Mechanic("Equipment", ("make.machine", "make.machine_order", "make.maintained", "make.repaired"),
             "machines bought, maintained and repaired"),
    Mechanic("Labor", ("make.own_operators", "labor.hired"), "operators employed from the labour market"),
    Mechanic("Market demand", ("make.retail_sold", "cols.batch:goods.session"), "households' purchases"),
    Mechanic("Resource scarcity", ("make.offer_missed",), "finite reserves, grid capacity, link capacity",
             count=lambda w, s: base.events_where(w, s, ("freight.reserve", "make.site_take"))),
    Mechanic("Supply disruptions", ("make.broken", "make.offer_missed", "ops.loss"),
             "breakdowns, unfilled supplier offers, curtailment and stockouts recorded as losses"),
)

ACTORS = (
    Actor("households", ("cols.batch:goods.session",), "population individuals buying goods"),
    Actor("firms", ("make.run", "make.trade"), "mines, component makers and assemblers run by organisational minds"),
    Actor("haulier", ("freight.shipment",), "the freight operator moving goods"),
)


def _configs():
    return tuple(DomainConfig(n, SPLIT[n], dict(SHAPES[n]), DOCS[n]) for n in SHAPES)


DOMAIN = Domain(
    name=NAME, title="Industrial economy", spec_part="9.8", required_mechanics=(), configs=_configs(), build=build,
    goal=GOAL, evaluate=evaluate, reference=FactoryReference, mechanics=MECHANICS,
    seats=(Seat(SEAT, 1_200_000, "Manufacturer", "a founder with capital for a manufacturing company"),),
    actors=ACTORS,
    degenerate=(Probe("trade_only", "buy components and resell them", TradeOnly),
                Probe("no_operators", "equip and source but never staff the machines", NoOperators)),
    structure=structure, tier3=tier3, mode="C", horizon_days=HORIZON_DAYS, slowest_effect_days=14.0, dt=8 * 3600,
    reference_seeds=(0, 1), reference_configs=("standard_ring",), sources=(), doc=__doc__)

base.register(DOMAIN)
