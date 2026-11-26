"""The worked example of docs/v5/components/I.md: a small civilisation-backed Tier 2 domain, shipped as a TEST FIXTURE
only (`fixture=True`: never counted by Gate N, never listed by execution mode B).

Household retail of components: the seat founds a trading company, rents a facility in a region, buys components
wholesale from the supply chain's component makers (`make.offers` / `make.order`, shipped over the freight network)
and lists them for households (`make.sell`). Households are the one population (`civgoods`' `goods` activity): an
individual hears of listings by browsing, search and word of mouth, and buys by the measured attributes of the stock,
the price, the distance and its own outside option - so a listing sells only if it is competitive with the
assemblers' products and other offers. Nothing here is scripted demand.

Configurations differ in the BUILT world's structure (supply-chain graph, freight topology), measured by
`crucible.domains.base.fingerprint`:

    train   ring_standard   standard chain (2 mines, 2 forges, 2 assemblers), 4-region ring
    train   line_linear     linear chain (1 mine, 1 forge, 1 assembler), 3-region line
    train   star_diamond    diamond chain, 4-region star
    eval_compositional  ring_linear     the linear chain on a ring (both seen in training, never together)
    eval_structural     complete_wide   a wide chain on a complete graph (shapes no training world has)

Objective: earn at least `target` cents of revenue from independent households within the horizon. Evaluator
(evaluator side): `goals.metrics` independent revenue of the seat's organisations (payments from unrelated payers
only, so transfers between the seat and its own company count for nothing) over the horizon.
"""
from __future__ import annotations

import copy
from dataclasses import replace

from crucible.domains import base
from crucible.domains.base import Actor, Domain, DomainConfig, Mechanic, Probe, ScriptedReference, Seat, act_

NAME = "fixture_retail"
HORIZON_DAYS = 21.0
TARGET = {1: 400_000, 2: 1_000_000, 3: 2_500_000}          # cents of independent revenue in the horizon

SHAPES = {
    "ring_standard": {"chain": "standard", "regions": 4, "links": "ring"},
    "line_linear": {"chain": "linear", "regions": 3, "links": "line"},
    "star_diamond": {"chain": "diamond", "regions": 4, "links": "star"},
    "ring_linear": {"chain": "linear", "regions": 4, "links": "ring"},
    "complete_wide": {"chain": "wide", "regions": 4, "links": "complete"},
}


def _profile(config):
    from crucible.civ import profiles as P
    bal = P.PROFILES["balanced"]
    return replace(bal, name=f"fixture_{config.name}", industry={**bal.industry, **SHAPES[config.name]})


def build(config, seed, overrides):
    return base.civ_recipe(seed, profile=_profile(config), subsystems=["industry", "supply_chains"],
                           modules=["civgoods", "civgen", "civlinks", "market"], population_modules=("pop",),
                           params=overrides.get("params"))


def objective(world, principal, difficulty):
    t = TARGET.get(int(difficulty), TARGET[3])
    return {"text": f"Earn at least ${t / 100:,.0f} of revenue from independent households within "
                    f"{HORIZON_DAYS:.0f} days by selling them goods.", "target_cents": t,
            "window_days": HORIZON_DAYS, "difficulty": int(difficulty)}


def evaluate(world, principal, obj):
    from crucible.goals.metrics import evaluate_metric
    win = int(float(obj.get("window_days", HORIZON_DAYS)) * 86400)
    rev = {o: float(evaluate_metric(world, "independent_revenue", {"principal": o, "window_s": win}))
           for o in base.seat_orgs(world, principal)}
    total = sum(rev.values())
    t = float(obj["target_cents"])
    return {"success": total >= t, "score": min(1.0, total / t) if t else 0.0,
            "details": {"independent_revenue_cents": total, "orgs": rev}}


class RetailReference(ScriptedReference):
    """Tool-only reference: found a company, rent a facility next to a component maker, buy its components and
    sell them to households at a markup; restock when the shelf runs low. It reads only tool results."""
    burst = 6

    def __init__(self, principal, objective):
        super().__init__(principal, objective)
        self.stage = "start"
        self.mine = None
        self.org = self.acct = self.fac = None
        self.offer = None
        self.listed = set()

    def plan(self, obs):
        if self.stage == "start":
            self.stage = "accounts"
            return [act_("bank.accounts", "accounts")]
        if self.stage == "org":
            self.stage = "wait"
            return [act_("id.create_org", "org", handle="tradeco", name="Tradeco", kind="company")]
        if self.stage == "company":
            self.stage = "wait"
            return [act_("biz.register", None, org=self.org, account=self.mine),
                    act_("bank.open_account", "acct", owner=self.org, name="operating")]
        if self.stage == "capital":
            self.stage = "offers"
            return [act_("bank.transfer", None, from_account=self.mine, to=self.acct, amount=450_000,
                         memo="founder capital"),
                    act_("make.offers", "offers", stage="component", limit=50)]
        if self.stage == "facility":
            self.stage = "wait"
            return [act_("make.build_facility", "fac", org=self.org, region=self.offer["region"],
                         account=self.acct)]
        if self.stage == "stock":
            self.stage = "trading"
            qty = max(1, min(int(self.offer["available"]), 120_000 // max(1, int(self.offer["price_cents"]))))
            return [act_("make.order", "ordered", offer=self.offer["offer"], qty=qty, to_facility=self.fac,
                         account=self.acct)]
        if self.stage == "trading":
            return [act_("make.inventory", "inventory", facility=self.fac),
                    act_("make.offers", "offers", stage="component", limit=50)]
        return []

    def on_result(self, tag, result, action):
        if result is None:
            if tag in ("org", "acct", "fac"):
                self.stage = "stuck"          # a refused founding step: the failure is in `failures`
            return
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
        elif tag == "offers":
            offers = [o for o in result if o.get("available", 0) > 0]
            if self.offer is None and offers:
                self.offer = min(offers, key=lambda o: (o["price_cents"], o["offer"]))
                self.stage = "facility"
            elif self.offer is not None:
                for o in offers:
                    if o["offer"] == self.offer["offer"]:
                        self.offer = o
        elif tag == "fac":
            self.fac = result["facility"]
            self.stage = "stock"
        elif tag == "inventory":
            items = result if isinstance(result, list) else result.get("items", [])
            for it in items:
                sku, qty = it.get("sku"), int(it.get("qty", 0))
                if qty > 0 and sku not in self.listed:
                    self.listed.add(sku)
                    self.queue.append(act_("make.sell", None, facility=self.fac, sku=sku,
                                           price_cents=int(self.offer["price_cents"] * 2.5), title="Components"))
            low = sum(int(it.get("qty", 0)) for it in items) < 20
            if low and self.offer and int(self.offer.get("available", 0)) > 0:
                qty = max(1, min(int(self.offer["available"]), 60_000 // max(1, int(self.offer["price_cents"]))))
                self.queue.append(act_("make.order", "ordered", offer=self.offer["offer"], qty=qty,
                                       to_facility=self.fac, account=self.acct))


class SelfFunding(ScriptedReference):
    """Degenerate: a company that the seat pays itself (related-party transfers are not independent revenue)."""
    def __init__(self, principal, objective):
        super().__init__(principal, objective)
        self.mine = self.org = self.acct = None
        self.step = 0

    def plan(self, obs):
        if self.mine is None:
            return [act_("bank.accounts", "accounts")]
        if self.org is None:
            return [act_("id.create_org", "org", handle="selfco", name="Selfco", kind="company")]
        if self.acct is None:
            return [act_("biz.register", None, org=self.org, account=self.mine),
                    act_("bank.open_account", "acct", owner=self.org, name="operating")]
        self.step += 1
        return [act_("bank.transfer", None, from_account=self.mine, to=self.acct, amount=40_000,
                     memo=f"sale {self.step}")] if self.step <= 10 else []

    def on_result(self, tag, result, action):
        if result is None:
            return
        if tag == "accounts":
            mine = [a for a in result if a.get("owner") == self.me]
            self.mine = mine[0]["account"] if mine else None
        elif tag == "org":
            self.org = result["org"]
        elif tag == "acct":
            self.acct = result["account"]


class ListWithoutStock(RetailReference):
    """Degenerate: everything the reference does except buying stock - a listing with an empty shelf."""
    def plan(self, obs):
        if self.stage == "stock":
            self.stage = "listed"
            return [act_("make.sell", None, facility=self.fac, sku=self.offer["sku"],
                         price_cents=int(self.offer["price_cents"] * 2.5), title="Components")]
        if self.stage == "listed":
            return []
        return super().plan(obs)


def structure(world):
    """The domain's own structure: which kinds of firms exist (by role) - presence only."""
    st = world.state
    roles = set()
    for _, s in (st.scan("minds") if "minds" in st.tables else []):
        roles.add(str(s.get("role") or s.get("kind") or "mind"))
    return {"firm_roles": sorted(roles)}


def _configs():
    return tuple(DomainConfig(n, s, dict(SHAPES[n]), d) for n, s, d in (
        ("ring_standard", "train", "standard chain on a four-region freight ring"),
        ("line_linear", "train", "one mine, one forge, one assembler along a three-region line"),
        ("star_diamond", "train", "diamond chain: every assembler needs both component makers; star freight"),
        ("ring_linear", "eval_compositional", "the linear chain on the ring: familiar parts, new combination"),
        ("complete_wide", "eval_structural", "a wide chain (3 mines, 3 forges, 4 assemblers), complete freight graph"),
    ))


MECHANICS = (
    Mechanic("Wholesale supply", ("make.trade",), "components bought from the supply chain's offers"),
    Mechanic("Freight", ("freight.shipment", "freight.arrived"), "shipments over the freight network"),
    Mechanic("Retail listing", ("make.retail", "make.retail_update"), "listings households can find"),
    Mechanic("Household demand", ("make.retail_sold",), "purchases by population individuals"),
    Mechanic("Production", ("make.run",), "the firms' own production runs"),
)

DOMAIN = Domain(
    name=NAME, title="Household retail (framework fixture)", spec_part="fixture",
    required_mechanics=tuple(m.name for m in MECHANICS), configs=_configs(), build=build, objective=objective,
    evaluate=evaluate, reference=RetailReference, mechanics=MECHANICS,
    seats=(Seat("trader", 600_000, "Trader"),),
    actors=(Actor("households", ("cols.batch:goods.session",), "population individuals shopping for goods"),
            Actor("firms", ("make.run", "make.trade"), "mines, forges and assemblers run by organisational minds")),
    degenerate=(Probe("self_funding", "the seat pays its own company", SelfFunding),
                Probe("list_without_stock", "a listing with nothing on the shelf", ListWithoutStock)),
    structure=structure, horizon_days=HORIZON_DAYS, slowest_effect_days=2.0, dt=6 * 3600,
    reference_seeds=(0, 1), fixture=True,
    doc="Worked example of the domain-authoring guide (docs/v5/components/I.md); test fixture only.")


def register():
    """Register the fixture (idempotent); returns the Domain."""
    return base.register(DOMAIN)


def planted(**changes):
    """A copy of the fixture with planted defects (positive controls of the gate checks)."""
    d = copy.copy(DOMAIN)
    for k, v in changes.items():
        setattr(d, k, v)
    return d
