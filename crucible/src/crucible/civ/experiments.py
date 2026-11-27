"""The counterfactual experiments of Gate O / SP-10.3 (crucible.civ.counterfactual runs them).

Effect types (Part 10.2 categories), each an intervention at a fork against placebo arms, measured downstream in
another subsystem than the one intervened on (the measured quantity is never the coupling's own bookkeeping):

    1 science_technology      remove `science_process`        -> true process quality of producers' settings
    2 technology_cost         revert adopted process settings -> energy and labour per good unit produced
    3 usage_infrastructure    take the busiest web services down -> energy drawn by datacenters from the grid
    4 infrastructure_software cut the datacenter regions' grids (scarce profile) -> failed requests, revenue of web
                              businesses (second arm: remove `grid_curtailment`)
    5 social_adoption         remove `goods_wom`              -> goods bought by the population
    7 coordination            remove `org_coordination`       -> units produced by multi-operator firms
    8 supply_disruption       black out the component makers' grid region -> units on retail shelves, prices,
                                                                 goods bought

Chains (Part 10.3), mined from provenance in the reference arm and confirmed by removing their root:

    agent_shop     a scripted agent writes and deploys a web shop and announces it on the forum every day (8 days)
                   arms: no_agent (it does nothing) -> agent-artifact chain; no_post (it deploys, never posts) ->
                   social chain (forum post -> discovery -> purchases -> compute -> energy)
                   stage measures: visitors (population), payments by population individuals and all invoice
                   payments (finance), compute its deployment served to others (computing infrastructure: low
                   reshuffling noise, unlike the provider's whole load) and the provider's whole compute and energy
    agent_science  a scripted researcher measures a process phenomenon and publishes a fitted model
                   arm: no_paper (it measures, never publishes) -> technical chain (paper -> replication -> R&D
                   trial -> process settings -> production yield and energy per good unit -> goods)

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
    """Requests to deployed services that failed for want of infrastructure in the window: API and service calls the
    cloud refused (`cloud.usage` unavailable/overload, `cloud.usage_bulk` unavailable/overload) and web page requests
    the network answered 503 because the site's provider was down (`net.hit` unavailable). At a54f53d this measure
    counted the API path only and read 0 in every arm and seed while the grid cut refused datacenter draws (dc_kwh
    -458, CI [-706, -210]): population web visits, most of the requests, go through `net.fetch`, which records a
    refusal as `net.hit`, not `cloud.usage` - the instrument could not see the failures it was meant to count."""
    n = 0
    for e in _window(w, ctx):
        if e.type == "cloud.usage":
            n += e.payload.get("outcome") in ("unavailable", "overload")
        elif e.type == "cloud.usage_bulk":
            n += int(e.payload.get("unavailable") or 0) + int(e.payload.get("overload") or 0)
        elif e.type == "net.hit":
            n += e.payload.get("outcome") == "unavailable"
    return float(n)


def m_curtailed_hours(w, t0, t1, ctx):
    """Diagnostic only (the coupling's own record, never a confirming measure): datacenter-hours the grid curtailed."""
    return float(sum(1 for e in _window(w, ctx) if e.type == "links.curtailed"))


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
                qs.append(ind.quality(ind.phys["models"][m["model"]], d["speed"], d["temp"]))  # not-a-content-score: industrial physics: part quality of a production run's process settings
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


def m_bound_kwh_per_unit(w, t0, t1, ctx):
    """Grid energy drawn per good unit by runs of bound-model machines that finished in the second half of the
    window (the process settings set the energy of a run: industry.energy_per_unit)."""
    rs = _runs_done(w, t0, t1, ctx, _bound_models(w))
    made = sum(p["made"] for _, p in rs)
    return float(sum(r["kwh"] for r, _ in rs) / made) if made else 0.0


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
    """Undo every process setting a producer adopted from a paper - kept after its own trial, or still in trial at
    the fork (as the owning firm, back to the setting it had before) - and stop further R&D adoption. A setting in
    trial must be undone too: with the relation removed its trial never concludes, so it would otherwise stay
    (at e77fef3 the arm's true quality equalled the factual arm's in every seed: the adoptions were still in
    trial at the fork). Settings whose trial failed were already undone by the producer."""
    st = w.state
    for did, rec in sorted(st.scan("civ_rnd")):
        if rec.get("kept") is False:
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


def cut_datacenter_grids(share: float):
    def fn(w):
        st = w.state
        for r in sorted({rec["region"] for _, rec in st.scan("civ_dc")}):
            g = st.tables["regions"][r]["grid"]
            w.system_call("admin.civ_grid", {"region": r, "capacity_kwh_day": int(g["capacity"] * share)})
    fn.__name__ = f"cut_datacenter_grids_{share}"
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


def m_agent_compute(w, t0, t1, ctx):
    """Compute units the agent's deployment served in the window to requests from outside its operator's side
    (metered per request by the sandbox and billed by the provider: `cloud.usage` / `cloud.usage_bulk` of its
    deployments; a request the cloud marks as the operator's own, `cloud.usage_rel`, is load but not demand and is
    left out). The site's own visitors, the search crawler and other services' calls all count; in `no_agent` there
    is no deployment (0)."""
    host = getattr(ctx.get("agent"), "handle", None)
    st = w.state
    deps = {k for k, d in st.scan("deployments") if d.get("host") == host} if host and "deployments" in st.tables \
        else set()
    if not deps:
        return 0.0
    tot = 0.0
    pending: dict[str, float] = {}           # dep -> cu of its last request, until we know whose it was
    for e in _window(w, ctx):
        if e.type == "cloud.usage" and e.payload.get("dep") in deps:
            tot += pending.pop(e.payload["dep"], 0.0)
            pending[e.payload["dep"]] = float(e.payload.get("cu") or 0.0)
        elif e.type == "cloud.usage_rel" and e.payload.get("dep") in deps:
            pending.pop(e.payload["dep"], None)
        elif e.type == "cloud.usage_bulk" and e.payload.get("dep") in deps:
            tot += float(e.payload.get("cu") or 0.0)
    return float(tot + sum(pending.values()))


def _pop_payments_to(w, ctx, who) -> list[tuple[str, int]]:
    """(paying population principal, cents) of every ledger posting in the window that moved a population
    individual's money to `who` (independently verified adoption: the ledger, not the seller, says it happened)."""
    from ..substrate import ids
    st, bank = w.state, w.modules["bank"]
    out = []
    for e in _window(w, ctx):
        if e.type != "bank.txn":
            continue
        payers = [bank.account_owner(st, a) for a, d in e.payload["postings"] if d < 0]
        payers = [x for x in payers if ids.index(x) is not None]
        if not payers:
            continue
        for a, d in e.payload["postings"]:
            if d > 0 and bank.account_owner(st, a) == who:
                out.append((payers[0], int(d)))
    return out


def m_agent_pop_revenue(w, t0, t1, ctx):
    """economic: money population individuals paid the agent in the window."""
    return float(sum(c for _, c in _pop_payments_to(w, ctx, getattr(ctx.get("agent"), "p", None))))


def m_agent_pop_buyers(w, t0, t1, ctx):
    """adoption: distinct population individuals who paid the agent in the window."""
    return float(len({x for x, _ in _pop_payments_to(w, ctx, getattr(ctx.get("agent"), "p", None))}))


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


# --------------------------------------------------------------------------------------------- invention (SP-2.1b)
INVENTOR = {"handle": "inventor", "host": "inventor-svc", "price": 3}


def _invention_setup(w, ctx):
    """A scripted reference inventor (labelled `scripted`, crucible.civ.reference.InventorAgent) is admitted in every
    arm; in the factual arm it writes, deploys and declares a service for one of the civilisation's published work
    formats at the best known solver tier (better than every incumbent), announces it on the forum and answers the
    public requests for that kind of service on the help board once a day. In `no_invention` it does nothing. Everything after that is the world's: individuals of the population
    search, try the declared capability on their own work, adopt it by their own accepted outcomes, pay per call."""
    from ..economy.needs import NEEDS
    from ..innovation.worlds import FORMAT_SCHEMAS
    from .reference import InventorAgent
    fams = (w.modules["work"].config.get("families") or []) if "work" in w.modules else []
    # the first of the world's published work formats whose interface an operator can declare (a public schema)
    cands = sorted(f["need"] for f in fams if f.get("need") in FORMAT_SCHEMAS)
    if not cands:
        raise ValueError(f"no work format of this civilisation has a public interface schema: {fams}")
    need = cands[0]
    tier = len(NEEDS[need].tiers) - 1
    board = w.modules["work"].cfg("ask_board") if "work" in w.modules else "help"
    ag = InventorAgent(INVENTOR["handle"], need, INVENTOR["host"], tier=tier, price=INVENTOR["price"], at=w.now,
                       cores=1, announce=True, board=board)
    ag.ensure(w)
    provs = w.query(ag.actor, "cloud.providers")
    ctx["provider"] = sorted(provs, key=lambda p: (-p["cores_free"], p["provider"]))[0]["provider"] if provs else None
    ctx["agent"], ctx["need"] = ag, need
    ctx["active"] = ctx["arm"] != "no_invention"


def _inv_decl(w, ctx):
    return getattr(ctx.get("agent"), "declaration", None)


def _inv_rows(w, ctx):
    d = _inv_decl(w, ctx)
    if d is None or "work_exp" not in w.state.tables:
        return []
    return [r for _, r in w.state.scan("work_exp") if r["prov"] == d]


def m_inv_revenue(w, t0, t1, ctx):
    """economic: money population individuals paid the inventor in the window (its admission endowment and any
    money from non-population accounts excluded)."""
    from ..substrate import ids
    st, bank = w.state, w.modules["bank"]
    who = getattr(ctx.get("agent"), "actor", None)
    tot = 0
    for e in _window(w, ctx):
        if e.type != "bank.txn":
            continue
        if not any(d < 0 and ids.index(bank.account_owner(st, a)) is not None for a, d in e.payload["postings"]):
            continue
        for a, d in e.payload["postings"]:
            if d > 0 and bank.account_owner(st, a) == who:
                tot += d
    return float(tot)


def m_incumbent_revenue(w, t0, t1, ctx):
    """economic: what the incumbent SaaS firms (merchants of the population's subscriptions) were paid by the
    population in the window."""
    st, bank = w.state, w.modules["bank"]
    inc = {r["merchant"] for _, r in st.scan("sub_subs")} if "sub_subs" in st.tables else set()
    from ..substrate import ids
    tot = 0
    for e in _window(w, ctx):
        if e.type != "bank.txn":
            continue
        pay = [a for a, d in e.payload["postings"] if d < 0 and ids.index(bank.account_owner(st, a)) is not None]
        if not pay:
            continue
        for a, d in e.payload["postings"]:
            if d > 0 and bank.account_owner(st, a) in inc:
                tot += d
    return float(tot)


def m_inv_adopters(w, t0, t1, ctx):
    """behaviour: individuals who rely on the invention for a step of their own work at the end of the window."""
    return float(len({r["i"] for r in _inv_rows(w, ctx) if r["active"]}))


def m_inv_uses(w, t0, t1, ctx):
    """behaviour: steps of the population's work delegated to the invention."""
    return float(sum(int(r["uses"]) for r in _inv_rows(w, ctx)))


def m_inv_compute(w, t0, t1, ctx):
    """resource demand: compute units used by the inventor's deployed service."""
    tot = 0.0
    for key, h in w.state.scan("svc_hours"):
        if h["host"] == INVENTOR["host"] and h["hour"] * 3600 >= t0:
            tot += float(h["cu"])
    return tot


def m_inv_provider_kwh(w, t0, t1, ctx):
    """resource demand: energy drawn from the grid by the datacenter of the provider the inventor rents from (the
    provider it would pick at the fork, in every arm)."""
    prov = ctx.get("provider")
    return float(sum(e.payload["kwh"] for e in _window(w, ctx) if e.type == "make.energy"
                     and e.payload.get("consumer") == prov))


def _needs(w):
    from ..innovation import relations
    try:
        return relations.label(w)
    except Exception:  # noqa: BLE001 - a world whose relation graph cannot be read has no derived needs
        return []


def m_inv_needs(w, t0, t1, ctx):
    """future technological opportunities: needs the evaluator-side relation graph (crucible.innovation.relations:
    bottlenecks, scarcity of capacity, unreliable producers) derives at the end of the window that involve the
    invention's capability or service - demand for capacity, substitutes or complements that did not exist before."""
    d = _inv_decl(w, ctx)
    keys = {f"cap:{ctx.get('need')}", f"svc:{INVENTOR['host']}", INVENTOR["host"]} | ({d} if d else set())
    import json as _j
    return float(sum(1 for n in _needs(w) if any(k in _j.dumps(n, default=str) for k in keys)))


def m_needs_total(w, t0, t1, ctx):
    return float(len(_needs(w)))


INVENTION_KINDS = {"economic_outcomes": ("inventor_revenue", "incumbent_revenue"),
                   "participant_behaviour": ("adopters", "delegated_uses"),
                   "resource_demand": ("service_compute", "provider_kwh"),
                   "future_opportunities": ("invention_needs", "needs_total")}


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
    Experiment("infrastructure_software", "grid capacity limits compute, which limits web businesses' sales: the "
               "grids of the datacenter regions are cut to 10% at the fork (the binding condition of the SP-10.2 "
               "battery; at 40% no grid refused a datacenter in any arm or seed at e77fef3: 0 outages, 0 failed "
               "requests) and, as a second arm, the curtailment relation is removed instead",
               "infrastructure_constrained", 2.0, 5.0, {"grid_cut_datacenters": cut_datacenter_grids(0.1),
                                                        "remove_grid_curtailment": remove("grid_curtailment")},
               {"web_revenue": m_web_revenue, "failed_requests": m_failed_requests, "dc_kwh": m_dc_kwh,
                "curtailed_hours": m_curtailed_hours},
               "failed_requests", category=4),
    Experiment("social_discovery", "social discovery produces independently verified product adoption: a "
               "labelled scripted agent's web shop is announced (or not) in a forum post every day; the population "
               "finds it through the forum, search and contacts and buys with its own money (the ledger verifies "
               "every purchase); the counterfactual arm builds the same shop and never posts. Eight days and a "
               "daily post: at e77fef3 the shop's three posts over six days brought 3-14 visitors and 0-2 buyers per "
               "seed (a seed without a buyer cannot show the effect under the every-seed sign rule)",
               "balanced", 1.0, 8.0, {"no_post": _noop},
               {"population_buyers": m_agent_pop_buyers, "population_revenue": m_agent_pop_revenue,
                "agent_visitors": m_agent_visitors, "agent_compute": m_agent_compute},
               "population_buyers", category=5, setup=_agent_setup("shop", posts=8), during=_agent_during),
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
    Experiment("agent_shop", "a scripted agent's web shop: discovery, purchases, compute and energy. Exposure as in "
               "social_discovery (a forum post every day over eight days): at a54f53d, with three posts over six "
               "days, the population stage (visitors -8.3, CI [-12.6, -4.0]) stayed below twice the reshuffling "
               "noise (5.9) in both arms while social_discovery's daily posts gave -18.7 against a noise of 4.5; "
               "the measure-to-subsystem mapping (crucible.civ.gates.CHAIN_ARMS) and the confirmation rule are "
               "unchanged",
               "balanced", 1.0, 8.0, {"no_agent": _noop, "no_post": _noop},
               {"agent_visitors": m_agent_visitors, "agent_revenue": m_agent_revenue,
                "population_revenue": m_agent_pop_revenue, "agent_compute": m_agent_compute,
                "provider_cu": m_agent_provider_cu, "provider_kwh": m_agent_provider_kwh,
                "web_revenue": m_web_revenue},
               "provider_kwh", setup=_agent_setup("shop", posts=8), during=_agent_during, chain=True),
    Experiment("invention", "a technological invention (a scripted reference inventor's service for a published "
               "work format, better than every incumbent's solver) changes economic outcomes, participant behaviour, "
               "resource demand and future technological opportunities (SP-2.1b); counterfactual: the inventor is "
               "admitted but never builds it",
               "balanced", 2.0, 8.0, {"no_invention": _noop},
               {"inventor_revenue": m_inv_revenue, "incumbent_revenue": m_incumbent_revenue,
                "adopters": m_inv_adopters, "delegated_uses": m_inv_uses, "service_compute": m_inv_compute,
                "provider_kwh": m_inv_provider_kwh, "invention_needs": m_inv_needs, "needs_total": m_needs_total},
               "adopters", setup=_invention_setup, during=_agent_during, chain=True),
    Experiment("agent_science", "a scripted researcher's paper: replication, R&D trials, process settings, output",
               "research_intensive", 1.0, 14.0, {"no_paper": _noop},
               {"adopted": m_adopted_from_papers, "true_quality": m_true_quality, "bound_yield": m_bound_yield,
                "kwh_per_unit": m_bound_kwh_per_unit, "goods_bought": m_goods_bought},
               "true_quality", setup=_agent_setup("science"), during=_agent_during, chain=True),
]}

EFFECT_EXPERIMENTS = [n for n, e in EXPERIMENTS.items() if e.category is not None]
INVENTION_EXPERIMENTS = ["invention"]
CHAIN_EXPERIMENTS = [n for n, e in EXPERIMENTS.items() if e.chain]
