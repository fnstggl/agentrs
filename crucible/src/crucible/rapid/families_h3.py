"""Workstream H3 families: hidden causal systems (8.1), uncertain discovery (8.5), network systems (8.7) and changing
environments (8.10).

The causal-system and changing-environment families are the generator's `compose()` recipes - a topology, the
primitives on the lever paths, lags and observation channels - plus named structural operations (`recipes.Composed`:
a previous operator's regime for historical records, a regime switch spliced into a path, a drifting set point, a
capacity input). The seed draws names and coefficients; unit costs are priced from the mechanism itself so that the
economic optimum sits at a hidden interior setting (`recipes.price_levers_at`), never at a corner or the middle of
a range. The discovery and network families run on the search and graph-dynamical engine kinds with structured
actions (queries, staged commitments, graph edits, allocations, discrete choices).

Evaluation families (`split="val"`, partition v5-eval-1) carry `novelty`: `compositional` (familiar primitives in a
combination no training family has) or `structural` (a structure - topology class, observation channel, scenario -
absent from every training family). `EVAL_NOVELTY` lists them; `tests/test_v5_rapid_h3.py` measures both claims
from fingerprints.
"""
from __future__ import annotations

from .families import FAMILIES, family
from .recipes import Composed, price_key_and_costly, price_levers_at, settled_port, settled_revenue, u

C = "crucible.rapid.claims_h3:"
R = "crucible.rapid.refs_h3:"

EVAL_NOVELTY: dict[str, str] = {}


def _novel(name: str, kind: str) -> None:
    EVAL_NOVELTY[name] = kind
    FAMILIES[name].extra["novelty"] = kind


def _base(d: dict, **recipe) -> dict:
    r = {"n_levers": 2, "temporal": "instant", "channels": ["noisy"], "noise": float(d.get("observation_noise", 0.3)),
         "n_distractors": int(d.get("n_variables", 0)), "horizon": int(d.get("horizon", 60)), "price": 20.0}
    r.update(recipe)
    return r


SHOP = "Maximise your net income over the horizon."


def _pg(name: str, **kw):
    """A `primitive_graph` family; its oracle searches lever schedules on the realised noise (`refs_h3`)."""
    kw.setdefault("oracle", R + "lever_schedule_oracle")
    return family(name, engine="primitive_graph", **kw)


# ====================================================================== 8.1 hidden causal systems
@_pg("cx_competing_causes", category="8.1", item="8.1.4", split="train",
        doc="Two spending levers moved together in the previous operator's records, both following a hidden demand "
            "cycle that also drives sales: either lever 'explains' the history equally well. Only one of them causes "
            "sales; the other costs money and does nothing.",
        distinct_because="compose chain whose second lever is causally inert, plus one hidden common driver of both "
                         "levers' historical regimes and of the outcome: two collinear candidate causes that passive "
                         "records cannot separate and intervention can",
        axes={"horizon": 60, "observation_noise": 0.3, "n_variables": 0, "experiment_cost": 1.0,
              "resources": 60.0}, objective="maximize_net",
        claims=(C + "competing_causes",), identify=C + "which_lever_causes", lever_view=True)
def cx_competing_causes(rng, d):
    c = Composed(rng, _base(d, topology="chain", motifs=["saturation", "linear_nonlinear"],
                            affordances=["levers", "experiment"]))
    a, b = c.levers
    m0, m1 = c.motifs
    c.drop_input(m0, b)
    U = c.prim("hidden_confounding", {"mean": 0.0, "sd": u(rng, 0.4, 0.5), "ar": 0.9}, role="confounder")
    c.natural(a, [(U, 0, 2.5)], bias=5.0, sd=0.4)
    c.natural(b, [(U, 0, 2.5)], bias=5.0, sd=0.4)
    c.add_input(m1, "x", U, 0, u(rng, 2.0, 3.0))
    exp = {"vars": [c.name_of(a), c.name_of(b)], "measure": [c.gauge_of(c.sold)], "cost": float(d["experiment_cost"]),
           "budget": float(d["resources"]), "max_ticks": 3, "max_replicates": 4}
    cfg = c.finish(warmup=40, brief="A shop with two spending levers. Gauges: units sold and an engagement count. "
                                    "You have the previous operator's records, and a pilot store for trials.",
                   objective={"type": "maximize_net", "text": SHOP}, experiment=exp,
                   claim={"lever": a, "inert": b, "outcome": c.sold, "confounder": U})
    costs = price_levers_at(cfg, {a: u(rng, 6.0, 8.0), b: 0.0})
    cfg["nodes"][int(b[1:])]["lever"]["cost"] = costs[a] * u(rng, 0.7, 1.0)
    return cfg


@_pg("cx_reactive_history", category="8.1", item="8.1.1", split="train",
        doc="The previous operator raised spending whenever sales had just fallen, and sales swing with a hidden "
            "demand shock: in the records more spending goes with lower sales, although spending raises sales "
            "a few ticks later.",
        distinct_because="reverse causation in the historical regime (the lever's passive mechanism reads the lagged "
                         "outcome: a cycle through the natural regime) with lagged effects through a hub; the passive "
                         "association is negative and the interventional effect positive",
        axes={"horizon": 60, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(C + "reactive_reversal",), identify=C + "lever_raises_sales_any", lever_view=True)
def cx_reactive_history(rng, d):
    c = Composed(rng, _base(d, topology="hub", motifs=["saturation"], temporal="lagged"))
    a, b = c.levers
    U = c.prim("hidden_confounding", {"mean": 0.0, "sd": u(rng, 0.55, 0.75), "ar": 0.85}, role="shock")
    k = u(rng, 3.0, 4.0)
    c.add_input(c.sold, "x", U, 0, k)
    cfg = c.finish(warmup=40, brief="A shop with two spending levers. Gauges: units sold and orders received. You "
                                    "have the previous operator's records.",
                   objective={"type": "maximize_net", "text": SHOP},
                   claim={"lever": a, "outcome": c.sold, "shock": U})
    bd = u(rng, 6.0, 8.0)
    cfg["nodes"][int(b[1:])]["lever"]["default"] = bd
    cfg["nodes"][int(c.sold[1:])]["params"]["half"] = u(rng, 20.0, 35.0)
    s_bar = settled_revenue(cfg, {a: 2.5, b: bd}) / cfg["money"]["price"]
    kap = u(rng, 0.8, 1.1)
    nd = cfg["nodes"][int(a[1:])]
    nd["natural"] = {"ports": {"x": [[c.sold, 1, -kap]]}, "bias": {"x": 2.5 + kap * s_bar}, "sd": 0.3}
    price_key_and_costly(cfg, a, u(rng, 7.0, 9.0), b, u(rng, 1.5, 2.0))
    return cfg


@_pg("cx_overdose", category="8.1", item="8.1.3", split="train",
        doc="A treatment lever has a dose-response that rises and then falls: past a hidden dose more of it lowers "
            "output, and the plant was left running at an overdose. A second lever helps with diminishing returns.",
        distinct_because="compose diamond (a mediated lagged path and a direct path) into a non-monotonic node whose "
                         "peak lies at a low hidden dose, plus a separate saturating lever path: the interventional "
                         "response to the first lever changes sign along its range",
        axes={"horizon": 60, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(C + "dose_peak",), identify=C + "dose_is_non_monotonic", lever_view=True)
def cx_overdose(rng, d):
    c = Composed(rng, _base(d, topology="diamond", motifs=["linear_nonlinear", "non_monotonic"], temporal="lagged"))
    a, b = c.levers
    med, nm = c.motifs
    c.drop_input(nm, b)
    sb = c.prim("saturation", {"cap": u(rng, 15.0, 25.0), "half": u(rng, 3.0, 6.0)}, x=[(b, 0, 1.0)])
    c.add_input(c.sold, "x", sb, 0, 1.0)
    c.set_lever(a, default=u(rng, 6.5, 8.5))
    cfg = c.finish(brief="A treatment plant with two dosing levers. Gauges: output and a process reading.",
                   objective={"type": "maximize_net", "text": SHOP},
                   claim={"lever": a, "second": b, "outcome": c.sold, "response": nm})
    ta = u(rng, 1.8, 3.0)
    # the response peaks at dose ta: the non-monotonic node's settled input at that dose is its peak
    cfg["nodes"][int(nm[1:])]["params"]["peak"] = max(settled_port(cfg, nm, {a: ta, b: 0.0}), 0.5)
    cfg["nodes"][int(c.sold[1:])]["params"]["half"] = u(rng, 25.0, 40.0)
    costs = price_levers_at(cfg, {a: ta, b: u(rng, 5.5, 8.0)})
    cfg["nodes"][int(a[1:])]["lever"]["cost"] = max(costs[a], 0.0) + 0.05 * costs[b]
    cfg["claim"]["peak_dose"] = ta
    return cfg


@_pg("cx_balancing_loop", category="8.1", item="8.1.5", split="train",
        doc="Output feeds back on the system through an unobserved intermediate that pushes against change: a step in "
            "the lever overshoots at once and settles about a third lower, so short tests overstate its lasting "
            "value.",
        distinct_because="compose feedback topology with a balancing loop (outcome -> hidden feedback node with "
                         "negative weight, one-tick lag); the opposite of a reinforcing loop: long-run response "
                         "below the short-run one",
        axes={"horizon": 60, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(C + "balancing_loop",), identify=C + "lever_raises_sales", lever_view=True)
def cx_balancing_loop(rng, d):
    c = Composed(rng, _base(d, topology="feedback", motifs=["saturation"]))
    a, b = c.levers
    fb = next(n for n in c.motifs if c.prim_of(n) == "feedback")
    c.set_params(fb, g=u(rng, 0.05, 0.15))
    sat = c.node(c.core)["params"]
    # loop gain below one at every operating point: the weight is a share of the inverse of the steepest slope
    w = u(rng, 0.6, 0.8) * sat["half"] / sat["cap"]
    c.node(fb)["ports"]["x"] = [[a, 0, 1.0], [c.core, 1, -w]]
    c.set_lever(a, default=u(rng, 0.5, 1.5))
    c.set_lever(b, default=u(rng, 6.0, 8.0))
    c.set_params(c.sold, half=u(rng, 20.0, 35.0))
    cfg = c.finish(brief="A process with two control levers. Gauges: output and throughput.",
                   objective={"type": "maximize_net", "text": SHOP},
                   claim={"lever": a, "second": b, "outcome": c.sold, "core": c.core, "hidden": fb})
    price_key_and_costly(cfg, a, u(rng, 7.0, 9.0), b, u(rng, 1.5, 2.0))
    return cfg


@_pg("cx_latched_pathway", category="8.1", item="8.1.6", split="train",
        doc="A second lever pays only while a hidden latch is on. Pushing the first lever above a hidden level turns "
            "the latch on; it stays on until the lever drops below a lower level, so the same setting of the first "
            "lever opens or closes the pathway depending on the past.",
        distinct_because="compose collider of a hysteresis latch and a saturating path (multiplicative): the active "
                         "causal structure depends on a path-dependent hidden state, not on the current setting "
                         "(unlike a memoryless threshold gate)",
        axes={"horizon": 60, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(C + "latch_memory",), identify=C + "latch_threshold", lever_view=True)
def cx_latched_pathway(rng, d):
    c = Composed(rng, _base(d, topology="collider", motifs=["hysteresis", "saturation"], temporal="lagged"))
    a, b = c.levers
    hy, sb, prod = c.motifs
    c.set_lever(a, default=u(rng, 0.5, 1.5))
    cfg = c.finish(brief="A unit with two levers. Gauges: output and a combined process reading.",
                   objective={"type": "maximize_net", "text": SHOP},
                   claim={"lever": b, "latch_lever": a, "latch": hy, "outcome": c.sold})
    p = cfg["nodes"][int(hy[1:])]["params"]
    tb = u(rng, 5.5, 7.5)
    gain = (settled_revenue(cfg, {a: 10.0, b: tb}) - settled_revenue(cfg, {a: 0.0, b: tb}))
    costs = price_levers_at(cfg, {a: 10.0, b: tb})
    cfg["nodes"][int(b[1:])]["lever"]["cost"] = costs[b]
    cfg["nodes"][int(a[1:])]["lever"]["cost"] = u(rng, 0.6, 0.75) * gain / p["hi"]
    return cfg


# ====================================================================== 8.10 changing environments
def _switch_tick(rng, H: int) -> float:
    return float(int(rng.integers(int(0.35 * H), int(0.65 * H) + 1)))


AFTER = 100_000        # a tick after every timed switch (pricing the regime that comes later)


@_pg("ch_sign_reversal", category="8.10", item="8.10.1", split="train",
        doc="The key lever raises sales until a hidden tick; from then on the same lever lowers them, permanently. "
            "What used to work has to be abandoned.",
        distinct_because="compose chain with a timed regime switch spliced into the lever path (gain from positive "
                         "to negative at a hidden tick) over a base level: an exogenous, permanent sign reversal of a "
                         "causal effect (not a Markov regime process, not triggered by the participant)",
        axes={"horizon": 60, "observation_noise": 0.3, "n_variables": 0, "experiment_cost": 2.0, "resources": 40.0},
        objective="maximize_net", claims=(C + "sign_reversal",), identify=C + "reversal_time", lever_view=True)
def ch_sign_reversal(rng, d):
    H = int(d["horizon"])
    c = Composed(rng, _base(d, topology="chain", motifs=["saturation", "regime_change"]))
    a, b = c.levers
    m0, rg = c.motifs
    c.drop_input(m0, b)
    c.set_params(rg, at=float(int(rng.integers(int(0.5 * H), int(0.7 * H) + 1))), after=-u(rng, 0.25, 0.45))
    sb = c.prim("saturation", {"cap": u(rng, 8.0, 14.0), "half": u(rng, 2.0, 4.0)}, x=[(b, 0, 1.0)])
    c.add_input(c.sold, "x", sb, 0, 1.0)
    c.set_bias(c.sold, "x", u(rng, 14.0, 22.0))
    c.set_params(c.sold, half=u(rng, 30.0, 45.0))
    c.process_noise(u(rng, 0.8, 1.2))
    c.set_lever(a, default=u(rng, 0.5, 1.5))
    c.set_lever(b, default=u(rng, 2.5, 4.0))
    # a costly market survey reads the lever's current contribution (it changes sign with the regime)
    c.gauge(rg, [("measurement_noise", {"sd": float(d["observation_noise"]) * 2})], on_demand=True,
            query_cost=float(d["experiment_cost"]))
    cfg = c.finish(brief="A shop with two levers. Gauges: units sold and a demand indicator; a market survey can "
                         "be commissioned (query), paid from a survey budget.",
                   objective={"type": "maximize_net", "text": SHOP},
                   experiment={"budget": float(d["resources"]), "cost": 0.0, "vars": [], "measure": [],
                               "max_ticks": 1, "max_replicates": 1},
                   claim={"lever": a, "second": b, "outcome": c.sold, "regime": rg})
    price_key_and_costly(cfg, a, u(rng, 6.5, 9.0), b, u(rng, 1.2, 1.5))
    return cfg


@_pg("ch_constraint_tightens", category="8.10", item="8.10.2", split="train",
        doc="Output needs materials and machine hours together. At a hidden tick the machines lose much of their "
            "capacity per hour: from then on materials beyond what the machines can process are wasted, and the "
            "best mix changes.",
        distinct_because="compose collider of two saturating inputs joined by complementarity, with a timed regime "
                         "switch on one input's productivity: a resource constraint that tightens mid-episode and "
                         "changes the optimal mix of complements",
        axes={"horizon": 60, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(C + "constraint_tightens",), identify=C + "constraint_time", lever_view=True)
def ch_constraint_tightens(rng, d):
    H = int(d["horizon"])
    c = Composed(rng, _base(d, topology="collider", motifs=["saturation", "saturation", "complementarity"]))
    a, b = c.levers
    pa, pb, comp = c.motifs
    c.set_params(comp, scale=u(rng, 2.0, 3.0), rho=-3.0)
    rg = c.splice(pb, "regime_change", {"before": 1.0, "after": u(rng, 0.3, 0.45), "at": _switch_tick(rng, H),
                                        "trig": 1e18}, trigger=[])
    c.set_params(c.sold, half=u(rng, 30.0, 45.0))
    c.process_noise(u(rng, 0.8, 1.2))
    c.set_lever(a, default=u(rng, 1.0, 2.0))
    c.set_lever(b, default=u(rng, 1.0, 2.0))
    cfg = c.finish(brief="A workshop. Levers: materials bought and machine hours booked. Gauges: output and work "
                         "in progress.",
                   objective={"type": "maximize_net", "text": SHOP},
                   claim={"lever": a, "constrained": b, "outcome": c.sold, "regime": rg})
    price_levers_at(cfg, {a: u(rng, 7.0, 9.0), b: u(rng, 2.0, 3.5)})
    return cfg


@_pg("ch_drifting_ideal", category="8.10", item="8.10.3", split="train",
        doc="Customers like a design feature most at a particular level, and that level drifts gradually during the "
            "episode (up in some worlds, down in others). A setting that was right early is wrong late.",
        distinct_because="compose chain of a non-monotonic response whose input is shifted by a hidden slowly "
                         "accumulating stock (decay with constant inflow): a gradually drifting optimum, not an "
                         "abrupt switch",
        axes={"horizon": 60, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(C + "drifting_optimum",), identify=C + "drift_direction", lever_view=True)
def ch_drifting_ideal(rng, d):
    H = int(d["horizon"])
    c = Composed(rng, _base(d, topology="chain", motifs=["non_monotonic"]))
    a, b = c.levers
    nm = c.motifs[0]
    c.drop_input(nm, b)
    up = bool(rng.random() < 0.5)
    a0, a1 = (u(rng, 1.5, 3.0), u(rng, 7.5, 9.0)) if up else (u(rng, 7.5, 9.0), u(rng, 1.5, 3.0))
    peak = u(rng, 2.5, 4.0)
    rate = u(rng, 0.03, 0.05)
    reach = 1.0 - (1.0 - rate) ** H                        # share of the stock's asymptote reached by the end
    # the response peaks where its input equals `peak`; input = a + peak - a0 -/+ D(t), D grows to |a1 - a0|
    k = abs(a1 - a0) * rate / reach
    D = c.prim("decay", {"rate": rate, "init": 0.0}, role="taste", bias={"x": k}, x=[])
    c.set_input(nm, "x", [(a, 0, 1.0), (D, 0, -1.0 if up else 1.0)])
    c.set_bias(nm, "x", peak - a0)
    c.set_params(nm, peak=peak, height=u(rng, 18.0, 26.0))
    sb = c.prim("saturation", {"cap": u(rng, 6.0, 10.0), "half": u(rng, 2.0, 4.0)}, x=[(b, 0, 1.0)])
    c.add_input(c.sold, "x", sb, 0, 1.0)
    c.set_params(c.sold, half=u(rng, 30.0, 45.0))
    c.process_noise(u(rng, 0.8, 1.2))
    c.set_lever(a, default=a0)
    c.set_lever(b, default=u(rng, 5.0, 7.0))
    cfg = c.finish(brief="A product with a design feature you set and a promotion lever. Gauges: units sold and "
                         "a store-traffic count.",
                   objective={"type": "maximize_net", "text": SHOP},
                   claim={"lever": a, "second": b, "outcome": c.sold, "taste": D, "direction": "up" if up else "down",
                          "start_best": a0, "end_best": a1})
    cb = price_levers_at(cfg, {b: u(rng, 2.0, 4.0), a: a0})[b]
    cfg["nodes"][int(a[1:])]["lever"]["cost"] = 0.04 * cb
    return cfg


@_pg("ch_demand_shift", category="8.10", item="8.10.4", split="train",
        doc="Sales are capped by capacity and by demand, and demand responds slowly to how well it was served. At a "
            "hidden tick demand shifts abruptly - up in some worlds, down in others - and capacity has to follow.",
        distinct_because="compose collider (capacity and demand complements) with endogenous demand fed back from "
                         "sales and an abrupt timed demand multiplier: a one-off level shift on top of slow endogenous "
                         "adjustment",
        axes={"horizon": 60, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(C + "demand_shift",), identify=C + "shift_direction", lever_view=True)
def ch_demand_shift(rng, d):
    H = int(d["horizon"])
    c = Composed(rng, _base(d, topology="collider", motifs=["saturation", "saturation", "complementarity"]))
    a, b = c.levers
    cap_node, promo, comp = c.motifs
    c.set_params(comp, scale=1.0, rho=-6.0, w=0.5)
    c.set_params(cap_node, cap=u(rng, 55.0, 75.0), half=u(rng, 6.0, 10.0))
    base = u(rng, 14.0, 20.0)
    E = c.prim("endogenous_demand", {"base": base, "elasticity": u(rng, 0.01, 0.03), "rate": 0.2, "s0": base,
                                     "init": base}, role="demand", served=[(comp, 1, 1.0)])
    tot = c.splice(promo, "linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0})
    c.add_input(tot, "x", E, 0, 1.0)
    up = bool(rng.random() < 0.5)
    rg = c.splice(tot, "regime_change", {"before": 1.0, "after": u(rng, 1.6, 2.0) if up else u(rng, 0.4, 0.55),
                                         "at": _switch_tick(rng, H), "trig": 1e18}, trigger=[])
    c.set_params(c.sold, half=u(rng, 30.0, 45.0))
    c.process_noise(u(rng, 0.8, 1.2), into=tot)
    c.set_lever(a, default=u(rng, 4.0, 6.0))
    c.set_lever(b, default=u(rng, 5.0, 7.0))
    cfg = c.finish(brief="A service. Levers: capacity and promotion. Gauges: customers served and a load index.",
                   objective={"type": "maximize_net", "text": SHOP},
                   claim={"lever": a, "second": b, "outcome": c.sold, "regime": rg,
                          "direction": "up" if up else "down"})
    ta = u(rng, 2.5, 4.0)
    price_key_and_costly(cfg, a, ta, b, u(rng, 1.4, 1.9))
    cfg["claim"]["capacity_before"] = ta
    return cfg


@_pg("ch_tech_transition", category="8.10", item="8.10.5", split="train",
        doc="An established technology pays at once but becomes obsolete at a hidden tick. A new technology earns "
            "little at first; its yield per unit grows with experience of using it (learning by doing).",
        distinct_because="compose collider rewired into learning by doing (the new lever's yield is multiplied by a "
                         "decaying stock of its own past use) next to a saturating old path with a timed obsolescence "
                         "switch: a transition whose payoff needs investment before the switch",
        axes={"horizon": 60, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(C + "tech_transition",), identify=C + "obsolescence_time", lever_view=True)
def ch_tech_transition(rng, d):
    H = int(d["horizon"])
    c = Composed(rng, _base(d, topology="collider", motifs=["saturation", "decay", "multiplicative"]))
    a, b = c.levers
    old, exp, prod = c.motifs
    rate = u(rng, 0.08, 0.12)
    c.set_params(exp, rate=rate)
    c.set_input(exp, "x", [(b, 1, 1.0)])
    c.set_input(prod, "a", [(b, 0, 1.0)])
    # at full use (8 per tick, experience 8/rate) the new process yields about as much as the old one at its best
    c.set_params(prod, k=u(rng, 0.25, 0.35) * rate)
    rg = c.splice(old, "regime_change", {"before": 1.0, "after": u(rng, 0.1, 0.25), "at": _switch_tick(rng, H),
                                         "trig": 1e18}, trigger=[])
    c.set_input(c.sold, "x", [(rg, 0, 1.0), (prod, 0, 1.0)])
    c.set_params(c.sold, half=u(rng, 30.0, 45.0))
    c.process_noise(u(rng, 0.8, 1.2))
    c.set_lever(a, default=u(rng, 5.0, 7.0))
    c.set_lever(b, default=0.0)
    cfg = c.finish(brief="A producer with an established process and a new one. Levers: use of each. Gauges: "
                         "output and a process reading.",
                   objective={"type": "maximize_net", "text": SHOP},
                   claim={"lever": a, "new": b, "outcome": c.sold, "regime": rg, "experience": exp})
    price_levers_at(cfg, {a: u(rng, 6.0, 8.0)})
    # the new process, once learned, earns two to three times its running cost per unit
    per_unit = settled_revenue(cfg, {a: 0.0, b: 8.0}, t0=AFTER) / 8.0
    cfg["nodes"][int(b[1:])]["lever"]["cost"] = max(per_unit, 0.0) * u(rng, 0.35, 0.5)
    return cfg


@_pg("ch_rule_change", category="8.10", item="8.10.6", split="train",
        doc="A platform ranks offers by one criterion until a hidden tick and by another from then on: the lever that "
            "earned visibility stops paying and the one that did not starts to.",
        distinct_because="compose chain split into two saturating paths with opposite timed regime switches at the "
                         "same hidden tick: which action the environment rewards changes (an institutional rule "
                         "change), not the sign of one effect",
        axes={"horizon": 60, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(C + "rule_change",), identify=C + "rule_change_time", lever_view=True)
def ch_rule_change(rng, d):
    H = int(d["horizon"])
    c = Composed(rng, _base(d, topology="chain", motifs=["saturation"], temporal="lagged",
                            channels=["noisy", "delayed"]))
    a, b = c.levers
    pa = c.motifs[0]
    c.drop_input(pa, b)
    at = _switch_tick(rng, H)
    lo = u(rng, 0.08, 0.15)
    ra = c.splice(pa, "regime_change", {"before": 1.0, "after": lo, "at": at, "trig": 1e18}, trigger=[])
    lag_a = int(c.node(pa)["ports"]["x"][0][1])
    pb = c.prim("saturation", dict(c.node(pa)["params"]), x=[(b, lag_a, 1.0)])
    rb = c.prim("regime_change", {"before": lo, "after": 1.0, "at": at, "trig": 1e18}, x=[(pb, 0, 1.0)], trigger=[])
    c.add_input(c.sold, "x", rb, 0, 1.0)
    c.set_params(c.sold, half=u(rng, 30.0, 45.0))
    c.process_noise(u(rng, 0.8, 1.2))
    c.set_lever(a, default=u(rng, 5.0, 7.0))
    c.set_lever(b, default=u(rng, 0.0, 1.0))
    cfg = c.finish(brief="A seller on a platform. Levers: discount depth and listing quality spend. Gauges: units "
                         "sold and listing views.",
                   objective={"type": "maximize_net", "text": SHOP},
                   claim={"lever": a, "second": b, "outcome": c.sold, "regime_a": ra, "regime_b": rb})
    ta, tb = u(rng, 6.5, 8.5), u(rng, 6.5, 8.5)
    ca = price_levers_at(cfg, {a: ta, b: 0.0})[a]
    cb = (settled_revenue(cfg, {a: 0.0, b: tb + 0.25}, t0=AFTER) -
          settled_revenue(cfg, {a: 0.0, b: tb - 0.25}, t0=AFTER)) / 0.5
    cfg["nodes"][int(a[1:])]["lever"]["cost"] = ca
    cfg["nodes"][int(b[1:])]["lever"]["cost"] = max(cb, 0.0)
    return cfg
