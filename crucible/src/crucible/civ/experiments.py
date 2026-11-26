"""The counterfactual experiments of Gate O / SP-10.3 (crucible.civ.counterfactual runs them).

Effect types (Part 10.2 categories), each an intervention at a fork against placebo arms, measured downstream in
another subsystem than the one intervened on (the measured quantity is never the coupling's own bookkeeping):

    1 science_technology      remove `science_process`        -> true process quality of producers' settings
    2 technology_cost         revert adopted process settings -> energy and labour per good unit produced
    3 usage_infrastructure    take the busiest web services down -> energy drawn by datacenters from the grid
    4 infrastructure_software remove `grid_curtailment` (scarce grid) -> revenue of web businesses, failed requests
    5 social_adoption         remove `goods_wom`              -> goods bought by the population
    7 coordination            remove `org_coordination`       -> units produced by multi-operator firms
    8 supply_disruption       black out the component makers' grid region -> units on retail shelves, prices,
                                                                 goods bought

Chains (Part 10.3), mined from provenance in the reference arm and confirmed by removing their root:

    agent_shop     a scripted agent writes and deploys a web shop and announces it on the forum
                   arms: no_agent (it does nothing) -> agent-artifact chain; no_post (it deploys, never posts) ->
                   social chain (forum post -> discovery -> purchases -> compute -> energy)
    agent_science  a scripted researcher measures a process phenomenon and publishes a fitted model
                   arm: no_paper (it measures, never publishes) -> technical chain (paper -> replication -> R&D
                   trial -> process settings -> production yield -> goods)

Every measure is evaluator-side (it may use hidden physics, e.g. the true quality of an operating point).
"""
from __future__ import annotations

import numpy as np

from . import couplings as C
from .counterfactual import Experiment

DAY = 86400


# --------------------------------------------------------------------------------------------- window helpers
def _window(w, ctx):
    return w.events_range(ctx["seq0"])


def _late(w, t0, t1, frac=0.5):
    return t0 + int((t1 - t0) * (1 - frac))


def m_goods_bought(w, t0, t1, ctx):
    return float(sum(1 for e in _window(w, ctx) if e.type == "make.retail_sold"))


def m_goods_adopters(w, t0, t1, ctx):
    return float(len({e.payload["buyer"] for e in _window(w, ctx) if e.type == "make.retail_sold"}))


def m_shelf_units(w, t0, t1, ctx):
    """Units on retail shelves, averaged over the daily closes of the second half of the window (and now)."""
    ind = w.modules["industry"]
    st = w.state
    return float(sum(ind.qty(st, r["fac"], r["sku"]) for _, r in st.scan("retail") if r["status"] == "active"))


def m_retail_price(w, t0, t1, ctx):
    ps = [r["price"] for _, r in w.state.scan("retail") if r["status"] == "active"]
    return float(np.mean(ps)) if ps else 0.0


def m_stockouts(w, t0, t1, ctx):
    return float(sum(1 for e in _window(w, ctx) if e.type == "ops.loss" and e.payload.get("kind") == "stockout"))


def m_dc_kwh(w, t0, t1, ctx):
    provs = set(w.state.ids("civ_dc"))
    return float(sum(e.payload["kwh"] for e in _window(w, ctx) if e.type == "make.energy"
                     and e.payload.get("consumer") in provs))


def m_dc_region_grid(w, t0, t1, ctx):
    """Energy drawn in the datacenter regions (production runs and datacenters together) over the window."""
    regs = {r["region"] for _, r in w.state.scan("civ_dc")}
    tot = 0.0
    for e in _window(w, ctx):
        if e.type == "make.energy" and e.payload["region"] in regs:
            tot += e.payload["kwh"]
        elif e.type == "make.run":
            m = w.state.get("machines", e.payload["machine"])
            f = w.state.get("facilities", m["facility"]) if m else None
            if f and f["region"] in regs:
                tot += e.payload["kwh"]
    return float(tot)


def _site_orgs(w) -> set[str]:
    st = w.state
    out = set()
    for h, rec in st.scan("hosts"):
        if rec.get("owner"):
            out.add(rec["owner"])
    return out


def m_web_revenue(w, t0, t1, ctx):
    """Money the population paid web businesses (invoice payments to site operators) in the window."""
    st = w.state
    bank = w.modules["bank"]
    owners = _site_orgs(w)
    tot = 0
    for e in _window(w, ctx):
        if e.type != "bank.txn" or e.payload.get("kind") != "invoice":
            continue
        for a, d in e.payload["postings"]:
            if d > 0 and bank.account_owner(st, a) in owners:
                tot += d
    return float(tot)


def m_failed_requests(w, t0, t1, ctx):
    return float(sum(1 for e in _window(w, ctx) if e.type == "cloud.usage"
                     and e.payload.get("outcome") in ("unavailable", "overload")))


def _bound_models(w) -> set[str]:
    return {b["model"] for b in (w.modules["civlinks"].config.get("bindings") or [])} if "civlinks" in w.modules \
        else set()


def m_true_quality(w, t0, t1, ctx):
    """Mean true quality (hidden physics) of the current operating points of designs run on bound machine models."""
    st = w.state
    ind = w.modules["industry"]
    models = _bound_models(w)
    qs = []
    for _, mid in sorted((m["facility"], k) for k, m in st.scan("machines")):
        m = st.tables["machines"][mid]
        if m["model"] not in models:
            continue
        for _, r in st.scan("runs"):
            if r["machine"] == mid:
                d = st.tables["designs"][r["design"]]
                qs.append(ind.quality(ind.phys["models"][m["model"]], d["speed"], d["temp"]))
                break
    return float(np.mean(qs)) if qs else 0.0


def _runs_done(w, t0, t1, ctx, models=None):
    st = w.state
    out = []
    late = _late(w, t0, t1)
    for e in _window(w, ctx):
        if e.type != "make.run_done" or e.t < late:
            continue
        r = st.tables["runs"][e.payload["id"]]
        m = st.tables["machines"][r["machine"]]
        if models is None or m["model"] in models:
            out.append((r, e.payload))
    return out


def m_bound_yield(w, t0, t1, ctx):
    rs = _runs_done(w, t0, t1, ctx, _bound_models(w))
    u = sum(r["units"] for r, _ in rs)
    return float(sum(p["made"] for _, p in rs) / u) if u else 0.0


def m_unit_cost(w, t0, t1, ctx):
    """Energy (at the grid base price) and operator hours (at the routine's 1300 c/h) per good unit, over runs of
    bound-model machines that finished in the second half of the window."""
    st = w.state
    rs = _runs_done(w, t0, t1, ctx, _bound_models(w))
    made = sum(p["made"] for _, p in rs)
    if not made:
        return 0.0
    cost = 0.0
    for r, _ in rs:
        f = st.tables["facilities"][st.tables["machines"][r["machine"]]["facility"]]
        g = st.tables["regions"][f["region"]].get("grid") or {"base_price": 10.0}
        cost += r["kwh"] * g["base_price"] + r["labor_h"] * 1300.0
    return float(cost / made)


def m_multi_operator_output(w, t0, t1, ctx):
    st = w.state
    big = {o for o in {f["owner"] for _, f in st.scan("facilities")} if C.headcount(w, st, o) >= 2}
    return float(sum(p["made"] for r, p in _runs_done(w, t0, t1, ctx) if r["owner"] in big))


def m_units_made(w, t0, t1, ctx):
    return float(sum(p["made"] for _, p in _runs_done(w, t0, t1, ctx)))


# --------------------------------------------------------------------------------------------- interventions
def remove(cid):
    def fn(w):
        C.remove(w, cid)
    fn.__name__ = f"remove_{cid}"
    return fn


def revert_adopted(w):
    """Undo every process setting a producer adopted from a paper and kept after its own trial (as the owning
    firm, back to the setting it had before), and stop further R&D adoption."""
    st = w.state
    for did, rec in sorted(st.scan("civ_rnd")):
        if not rec.get("kept"):
            continue
        d = st.get("designs", did)
        owner = next((x["person"] for x in st.find("members", "org", d["owner"]) if x.get("role") == "owner"), None)
        if owner:
            w.system_call("make.tune", {"design": did, "speed": float(rec["from"][0]), "temp": float(rec["from"][1])},
                          as_principal=owner)
    C.remove(w, "science_process")


def undeploy_busiest(k: int = 5):
    def fn(w):
        st = w.state
        cu: dict[str, float] = {}
        for key, h in st.scan("svc_hours"):
            cu[h["host"]] = cu.get(h["host"], 0.0) + float(h["cu"])
        for host, _ in sorted(cu.items(), key=lambda x: (-x[1], x[0]))[:k]:
            rec = st.get("hosts", host)
            if not rec:
                continue
            owner = rec["owner"]
            persons = [x["person"] for x in st.find("members", "org", owner) if x.get("role") == "owner"] \
                if owner.startswith("o:") else [owner]
            if persons:
                w.system_call("cloud.undeploy", {"host": host}, as_principal=persons[0])
    fn.__name__ = f"undeploy_busiest_{k}"
    return fn


def blackout_components(w):
    """Cut the grids of every region hosting a component maker to 1 kWh/day for the rest of the window."""
    st = w.state
    regs = set()
    for mid, m in st.scan("minds"):
        if m.get("role") == "components":
            for f in st.find("facilities", "owner", m["org"]):
                regs.add(f["region"])
    for r in sorted(regs):
        w.system_call("admin.civ_grid", {"region": r, "capacity_kwh_day": 1})


# --------------------------------------------------------------------------------------------- agent experiments
def _agent_setup(kind, **kw):
    def setup(w, ctx):
        from . import reference
        ag = reference.WebShopAgent(seed=ctx["seed"], **kw) if kind == "shop" else reference.ResearcherAgent(**kw)
        ag.admit(w)
        if ctx["arm"] == "no_post" and kind == "shop":
            ag.post = False
        if ctx["arm"] == "no_paper" and kind == "science":
            ag.publish = False
        ctx["agent"] = ag
        ctx["active"] = ctx["arm"] != "no_agent"
        _agent_providers(w, ctx)
    return setup


def _prelude_science(w, ctx):
    from . import reference
    ctx["agent"] = reference.ResearcherAgent()
    ctx["agent"].admit(w)
    ctx["active"] = True


def _prelude_during(w, arm, ctx):
    ctx["agent"].act(w)


def _agent_during(w, arm, ctx):
    if ctx.get("active"):
        ctx["agent"].act(w)


def _noop(w):
    """The intervention of an agent arm is in its setup (the agent is told what not to do)."""


def m_agent_visitors(w, t0, t1, ctx):
    """Population individuals who visited the agent's site (the site exists in every arm but `no_agent`)."""
    from ..substrate import columns
    st = w.state
    host = getattr(ctx.get("agent"), "handle", None)
    rec = st.get("pop_res", host) if host and "pop_res" in st.tables else None
    if rec is None:
        return 0.0
    return float(columns.col(w, "res", "visits")[int(rec["row"])])


def m_agent_revenue(w, t0, t1, ctx):
    st = w.state
    bank = w.modules["bank"]
    p = getattr(ctx.get("agent"), "p", None)
    tot = 0
    for e in _window(w, ctx):
        if e.type == "bank.txn" and e.payload.get("kind") == "invoice":
            for a, d in e.payload["postings"]:
                if d > 0 and bank.account_owner(st, a) == p:
                    tot += d
    return float(tot)


def _agent_providers(w, ctx) -> set[str]:
    """The datacenter the agent rents from: the cheapest provider when the experiment forked (the same scope in
    every arm, also in those where the agent rents nothing)."""
    if "providers_at_fork" not in ctx:
        ps = [(r["price"], k) for k, r in w.state.scan("providers") if r["status"] == "active"]
        ctx["providers_at_fork"] = [min(ps)[1]] if ps else []
    return set(ctx["providers_at_fork"])


def m_agent_provider_kwh(w, t0, t1, ctx):
    """Energy drawn from the grid by the datacenter of the provider the agent rents from."""
    provs = _agent_providers(w, ctx)
    return float(sum(e.payload["kwh"] for e in _window(w, ctx) if e.type == "make.energy"
                     and e.payload.get("consumer") in provs))


def m_agent_provider_cu(w, t0, t1, ctx):
    """Compute used on that provider by every deployment it hosts."""
    st = w.state
    provs = _agent_providers(w, ctx)
    tot = 0.0
    for key, h in st.scan("svc_hours"):
        if h["hour"] * 3600 < t0:
            continue
        rec = st.get("hosts", h["host"])
        dep = st.get("deployments", rec["deployment"]) if rec and rec.get("deployment") else None
        inst = st.get("instances", dep["instance"]) if dep else None
        if inst and inst["provider"] in provs:
            tot += float(h["cu"])
    return tot


def m_adopted_from_papers(w, t0, t1, ctx):
    return float(sum(1 for e in _window(w, ctx) if e.type == "links.rnd"))


# --------------------------------------------------------------------------------------------- the battery
EXPERIMENTS: dict[str, Experiment] = {e.name: e for e in [
    Experiment("science_technology", "replicated public science about a machine model changes producers' process "
               "settings (removing the science->industry relation leaves them where their own trials put them). The "
               "discovery is a scripted reference researcher's paper (labelled), present in every arm; the autonomous "
               "rival labs' polynomial papers are not reproduced by independent replication (relative error "
               "0.5-0.9), so they are never trusted",
               "research_intensive", 1.0, 14.0, {"remove_science_process": remove("science_process")},
               {"true_quality": m_true_quality, "bound_yield": m_bound_yield, "adopted": m_adopted_from_papers},
               "true_quality", category=1, setup=_agent_setup("science"), during=_agent_during),
    Experiment("technology_cost", "an adopted process setting changes industrial production cost per good unit: "
               "forked after producers adopted (and kept) the settings of a scripted researcher's replicated paper",
               "research_intensive", 14.0, 8.0, {"revert_adopted": revert_adopted},
               {"unit_cost": m_unit_cost, "bound_yield": m_bound_yield, "true_quality": m_true_quality},
               "unit_cost", category=2, prelude=_prelude_science, prelude_during=_prelude_during),
    Experiment("usage_infrastructure", "service usage by the population draws datacenter energy from the grid",
               "balanced", 2.0, 4.0, {"undeploy_busiest_5": undeploy_busiest(5)},
               {"dc_kwh": m_dc_kwh, "region_grid_kwh": m_dc_region_grid, "web_revenue": m_web_revenue},
               "dc_kwh", category=3),
    Experiment("infrastructure_software", "grid capacity limits compute, which limits web businesses' sales",
               "infrastructure_constrained", 2.0, 5.0, {"remove_grid_curtailment": remove("grid_curtailment")},
               {"web_revenue": m_web_revenue, "failed_requests": m_failed_requests, "dc_kwh": m_dc_kwh},
               "failed_requests", category=4),
    Experiment("social_adoption", "word of mouth between contacts changes what the population buys",
               "balanced", 2.0, 6.0, {"remove_goods_wom": remove("goods_wom")},
               {"goods_bought": m_goods_bought, "adopters": m_goods_adopters, "stockouts": m_stockouts},
               "goods_bought", category=5),
    Experiment("coordination", "organisation size costs coordination: effective labour and output fall",
               "cooperative_institutions", 2.0, 6.0, {"remove_org_coordination": remove("org_coordination")},
               {"multi_operator_output": m_multi_operator_output, "units_made": m_units_made},
               "multi_operator_output", category=7),
    Experiment("supply_disruption", "a supply disruption upstream changes what households find on shelves and pay",
               "balanced", 3.0, 6.0, {"blackout_components": blackout_components},
               {"shelf_units": m_shelf_units, "retail_price": m_retail_price, "goods_bought": m_goods_bought,
                "stockouts": m_stockouts}, "goods_bought", category=8),
    Experiment("agent_shop", "a scripted agent's web shop: discovery, purchases, compute and energy",
               "balanced", 1.0, 6.0, {"no_agent": _noop, "no_post": _noop},
               {"agent_visitors": m_agent_visitors, "agent_revenue": m_agent_revenue,
                "provider_cu": m_agent_provider_cu, "provider_kwh": m_agent_provider_kwh,
                "web_revenue": m_web_revenue},
               "provider_kwh", setup=_agent_setup("shop"), during=_agent_during, chain=True),
    Experiment("agent_science", "a scripted researcher's paper: replication, R&D trials, process settings, output",
               "research_intensive", 1.0, 14.0, {"no_paper": _noop},
               {"adopted": m_adopted_from_papers, "true_quality": m_true_quality, "bound_yield": m_bound_yield,
                "goods_bought": m_goods_bought},
               "true_quality", setup=_agent_setup("science"), during=_agent_during, chain=True),
]}

EFFECT_EXPERIMENTS = [n for n, e in EXPERIMENTS.items() if e.category is not None]
CHAIN_EXPERIMENTS = [n for n, e in EXPERIMENTS.items() if e.chain]
