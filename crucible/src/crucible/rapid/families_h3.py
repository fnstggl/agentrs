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


# ====================================================================== 8.1 hidden causal systems
@family("cx_competing_causes", engine="primitive_graph", category="8.1", item="8.1.4", split="train",
        doc="Two spending levers moved together in the previous operator's records, both following a hidden demand "
            "cycle that also drives sales: either lever 'explains' the history equally well. Only one of them causes "
            "sales; the other costs money and does nothing.",
        distinct_because="compose chain whose second lever is causally inert, plus one hidden common driver of both "
                         "levers' historical regimes and of the outcome: two collinear candidate causes that passive "
                         "records cannot separate and intervention can",
        axes={"horizon": 60, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(C + "competing_causes",), identify=C + "which_lever_causes", lever_view=True)
def cx_competing_causes(rng, d):
    c = Composed(rng, _base(d, topology="chain", motifs=["saturation", "linear_nonlinear"]))
    a, b = c.levers
    m0, m1 = c.motifs
    c.drop_input(m0, b)
    U = c.prim("hidden_confounding", {"mean": 0.0, "sd": u(rng, 0.4, 0.5), "ar": 0.9}, role="confounder")
    c.natural(a, [(U, 0, 2.5)], bias=5.0, sd=0.4)
    c.natural(b, [(U, 0, 2.5)], bias=5.0, sd=0.4)
    c.add_input(m1, "x", U, 0, u(rng, 2.0, 3.0))
    cfg = c.finish(warmup=40, brief="A shop with two spending levers. Gauges: units sold and an engagement count. "
                                    "You have the previous operator's records.",
                   objective={"type": "maximize_net", "text": SHOP},
                   claim={"lever": a, "inert": b, "outcome": c.sold, "confounder": U})
    costs = price_levers_at(cfg, {a: u(rng, 6.0, 8.0), b: 0.0})
    cfg["nodes"][int(b[1:])]["lever"]["cost"] = costs[a] * u(rng, 0.7, 1.0)
    return cfg


@family("cx_reactive_history", engine="primitive_graph", category="8.1", item="8.1.1", split="train",
        doc="The previous operator raised spending whenever sales had just fallen, and sales swing with a hidden "
            "demand shock: in the records more spending goes with lower sales, although spending raises sales "
            "a few ticks later.",
        distinct_because="reverse causation in the historical regime (the lever's passive mechanism reads the lagged "
                         "outcome: a cycle through the natural regime) with lagged effects through a hub; the passive "
                         "association is negative and the interventional effect positive",
        axes={"horizon": 60, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(C + "reactive_reversal",), identify=C + "lever_raises_sales", lever_view=True)
def cx_reactive_history(rng, d):
    c = Composed(rng, _base(d, topology="hub", motifs=["saturation"], temporal="lagged"))
    a, b = c.levers
    U = c.prim("hidden_confounding", {"mean": 0.0, "sd": u(rng, 0.7, 0.9), "ar": 0.85}, role="shock")
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


@family("cx_overdose", engine="primitive_graph", category="8.1", item="8.1.3", split="train",
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


@family("cx_balancing_loop", engine="primitive_graph", category="8.1", item="8.1.5", split="train",
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


@family("cx_latched_pathway", engine="primitive_graph", category="8.1", item="8.1.6", split="train",
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
