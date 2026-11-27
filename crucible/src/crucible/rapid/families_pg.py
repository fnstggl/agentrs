"""Families on the `primitive_graph` engine: hidden causal systems, experimental science, delayed consequences,
engineering control, a strategic game and the look-alike markets of Part 7.2. Each builder composes registered
primitives with `GraphBuilder`; the seed draws names and coefficients, the difficulty axes change structure size,
noise, cost and horizon."""
from __future__ import annotations

from .families import family
from .generator import GraphBuilder, lookalike_market

C = "crucible.rapid.claims:"
I = "crucible.rapid.identify:"


def u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def _distractors(gb: GraphBuilder, rng, n: int, noise: float) -> None:
    """n hidden processes that drive nothing but are shown on gauges (causally inert variables)."""
    for _ in range(int(n)):
        d = gb.prim("hidden_confounding", {"mean": u(rng, 20, 60), "sd": u(rng, 1, 3), "ar": 0.8}, role="distractor")
        gb.gauge(d, [("measurement_noise", {"sd": noise * 5})])


# ====================================================================== 8.1 hidden causal systems
@family("hc_confounded", engine="primitive_graph", category="8.1", item="8.1.1", split="train",
        doc="A hidden demand cycle drives both the previous operator's spending lever and sales: the records show a "
            "strong positive spending-sales correlation, but spending itself lowers sales; a second lever helps "
            "with sharply diminishing returns. Predict sales at settings partly outside what can be set, as "
            "interventions: the records give the wrong sign, and a straight line through tests gives the wrong "
            "shape.",
        distinct_because="hidden confounder with two children (natural-regime lever and outcome) plus historical "
                         "records: the passive association and the interventional effect have opposite signs; "
                         "scored by interventional predictions at held-out conditions",
        axes={"horizon": 60, "observation_noise": 0.3, "n_variables": 0}, objective="predict_heldout",
        claims=(C + "confounded_reversal",), identify=I + "lever_sign", lever_view=True,
        reference="crucible.rapid.refs_adapt:fir_shape_predict", oracle="crucible.rapid.oracles:science_oracle",
        version="2")
def hc_confounded(rng, d):
    gb = GraphBuilder(rng)
    U = gb.prim("hidden_confounding", {"mean": 0.0, "sd": u(rng, 0.5, 0.6), "ar": 0.85}, role="confounder")
    X = gb.lever(0.0, 10.0, 5.0, cost=u(rng, 40, 70), natural={"x": [(U, 0, 2.0)], "bias": 5.0, "sd": 0.3})
    Q = gb.lever(0.0, 10.0, 2.0, cost=u(rng, 60, 100))
    base = gb.prim("saturation", {"cap": u(rng, 40, 55), "half": u(rng, 1.0, 2.0)}, x=[(Q, 0, 1.0)])
    sales = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, role="outcome",
                    bias={"x": u(rng, 35, 45)}, x=[(U, 0, u(rng, 3.5, 4.5)), (X, 0, -u(rng, 0.8, 1.2)),
                                                   (base, 0, 1.0)])
    proxy = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 50.0}, x=[(U, 0, 3.0)])
    noise = float(d["observation_noise"])
    gy = gb.gauge(sales, [("measurement_noise", {"sd": noise * 3})])
    gb.gauge(proxy, [("measurement_noise", {"sd": 4.0})])
    _distractors(gb, rng, d["n_variables"], noise)
    held = [{gb.name_of(X): p, gb.name_of(Q): q} for p, q in
            ((13.0, 0.5), (15.0, 13.0), (0.5, 16.0), (12.0, 12.0), (16.0, 3.0), (2.0, 14.0), (14.0, 18.0))]
    text = ("Predict mean units sold if the two spending levers were held at these values (partly outside what you "
            f"can set): {held}. File predictions with the journal: message to 'journal' with content "
            "{\"predictions\": [{\"set\": <condition>, \"value\": <number>}, ...]}. Only the last filing counts.")
    return gb.build(horizon=int(d["horizon"]), money={"revenue_node": sales, "price": 100.0, "cap": 200.0},
                    affordances=["levers", "message"], warmup=40, endowment=500_000,
                    objective={"type": "predict_heldout", "heldout": held, "gauge": gy, "text": text},
                    journal=True,
                    brief="You take over a shop. Two spending levers; gauges: units sold and a footfall count. "
                          "You have the previous operator's records.",
                    claim={"lever": X, "outcome": sales, "confounder": U})


@family("hc_conditional", engine="primitive_graph", category="8.1", item="8.1.2", split="train",
        doc="One lever works only through a pathway that is open when a second lever is above a hidden threshold; "
            "below it the first lever does nothing, although an upstream engagement count still moves with it. A "
            "slow hidden demand cycle moves sales as well. Predict sales at combinations of the two settings "
            "outside what can be set, as interventions: the answer depends on which side of the hidden threshold "
            "the second setting lies and on how the first lever's effect saturates.",
        distinct_because="a threshold gates a saturating pathway multiplicatively (conditional causal path) behind "
                         "an exogenous hidden cycle, with an upstream reading that moves whether or not the path is "
                         "open; the objective is the interventional prediction at held-out conditions (scored on "
                         "unobserved conditions); no confounded history",
        axes={"horizon": 80, "observation_noise": 0.3, "n_variables": 0}, objective="predict_heldout",
        claims=(C + "conditional_pathway",), identify="crucible.rapid.claims_w3b:identify_gate_threshold",
        reference="crucible.rapid.refs_adapt:interleaved_law_predict", version="2")
def hc_conditional(rng, d):
    gb = GraphBuilder(rng)
    A = gb.lever(0.0, 10.0, 1.0, cost=0.0)
    Bv = gb.lever(0.0, 10.0, 2.0, cost=0.0)
    theta = u(rng, 3.0, 8.0)
    gate = gb.prim("threshold", {"theta": theta, "lo": 0.0, "hi": 1.0}, x=[(Bv, 0, 1.0)])
    eff = gb.prim("saturation", {"cap": u(rng, 40, 70), "half": u(rng, 2.0, 5.0)}, x=[(A, 0, 1.0)])
    path = gb.prim("multiplicative", {"k": 1.0}, a=[(eff, 0, 1.0)], b=[(gate, 0, 1.0)])
    season = gb.prim("hidden_confounding", {"mean": 0.0, "sd": 0.5, "ar": 0.8}, role="exogenous")
    sales = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, role="outcome",
                    bias={"x": u(rng, 25, 35)}, x=[(path, 0, 1.0), (Bv, 0, -u(rng, 0.3, 1.0)), (season, 0, 2.0)])
    noise = float(d["observation_noise"])
    gy = gb.gauge(sales, [("measurement_noise", {"sd": noise * 3})])
    gb.gauge(eff, [("measurement_noise", {"sd": noise * 3})], lag=1)
    _distractors(gb, rng, d["n_variables"], noise)
    held = [{gb.name_of(A): p, gb.name_of(Bv): q} for p, q in
            ((12.0, 2.0), (12.0, 10.5), (14.0, 5.0), (14.0, 12.0), (16.0, 8.5), (11.0, 1.0), (13.0, 9.5))]
    text = ("Predict mean units sold if the two levers were set to these values (partly outside what you can set): "
            f"{held}. File predictions with the journal: message to 'journal' with content {{\"predictions\": "
            "[{\"set\": <condition>, \"value\": <number>}, ...]}. Only the last filing counts.")
    return gb.build(horizon=int(d["horizon"]), affordances=["levers", "message"],
                    objective={"type": "predict_heldout", "heldout": held, "gauge": gy, "text": text},
                    journal=True,
                    brief="A workshop with two levers. Gauges: units sold and an engagement count (reported a tick "
                          "late).",
                    claim={"lever": A, "gate_lever": Bv, "theta": theta, "outcome": sales})


@family("hc_feedback_hidden", engine="primitive_graph", category="8.1", item="8.1.5", split="val",
        doc="Sales build a hidden reputation stock that raises later sales: the long-run effect of a lever is two "
            "to three times its immediate effect, through an intermediate variable nobody observes. Predict where "
            "weekly sales settle at marketing levels partly outside what can be set; short tests understate the "
            "answer.",
        distinct_because="a positive feedback loop through an unobserved decaying stock with lag (cycle in the "
                         "causal graph); the observed short-run response understates the long-run one; scored by "
                         "predictions of the settled response at held-out conditions",
        axes={"horizon": 80, "observation_noise": 0.3}, objective="predict_heldout",
        claims=(C + "feedback_amplifies",), identify="crucible.rapid.claims_w3b:identify_feedback_multiplier",
        reference="crucible.rapid.refs_adapt:arx_settled_predict", version="2")
def hc_feedback_hidden(rng, d):
    gb = GraphBuilder(rng)
    M = gb.lever(0.0, 10.0, 0.5, cost=0.0)
    sat = gb.prim("saturation", {"cap": u(rng, 25, 35), "half": u(rng, 3.5, 4.5)}, x=[(M, 0, 1.0)])
    rep = gb.prim("decay", {"rate": 0.1, "init": 0.0}, role="reputation", x=[])
    sales = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, role="outcome",
                    bias={"x": u(rng, 15, 25)}, x=[(sat, 0, 1.0), (rep, 1, u(rng, 5.8, 6.4))])
    gb.nodes[int(rep[1:])]["ports"]["x"] = [[sales, 2, 0.01]]
    gy = gb.gauge(sales, [("measurement_noise", {"sd": float(d["observation_noise"]) * 3})])
    gb.gauge(sat, [("measurement_noise", {"sd": float(d["observation_noise"]) * 3})])
    held = [{gb.name_of(M): p} for p in (0.0, 1.5, 4.0, 11.0, 13.0, 16.0)]
    text = ("Predict where weekly units sold would settle if marketing were held at these values (partly outside "
            f"what you can set): {held}. File predictions with the journal: message to 'journal' with content "
            "{\"predictions\": [{\"set\": <condition>, \"value\": <number>}, ...]}. Only the last filing counts.")
    cfg = gb.build(horizon=int(d["horizon"]), money={"revenue_node": sales, "price": 100.0, "cap": 400.0},
                   affordances=["levers", "message"], endowment=500_000,
                   objective={"type": "predict_heldout", "heldout": held, "gauge": gy, "text": text,
                              "settle_ticks": 300},
                   journal=True, brief="A shop with a marketing lever. Gauges: units sold and leads.",
                   claim={"lever": M, "outcome": sales, "hidden": rep})
    return cfg


# ====================================================================== Part 7.2 look-alike markets
def _market(rule):
    def build(rng, d):
        cfg = lookalike_market(rng, rule, d)
        return cfg
    return build


family("mk_elastic", engine="primitive_graph", category="8.1", item="8.1.4", split="train",
       doc="Look-alike market, rule 'elastic': lowering the price raises purchases (logistic demand in price).",
       distinct_because="shares its public interface with mk_signal, mk_gate and mk_drift; here conversion is a "
                        "decreasing logistic function of price",
       axes={"horizon": 60, "observation_noise": 0.15}, objective="predict_heldout",
       claims=(C + "market_rule",), identify=I + "market_rule",
       reference="crucible.rapid.refs_adapt:market_predict", oracle="crucible.rapid.oracles:science_oracle", version="2", lever_view=True)(_market("elastic"))
family("mk_signal", engine="primitive_graph", category="8.1", item="8.1.3", split="train",
       doc="Look-alike market, rule 'signal': buyers read price as quality, so lowering a low price lowers "
           "purchases (non-monotonic response).",
       distinct_because="same interface as the other markets; conversion is non-monotonic in price (rises, "
                        "peaks, falls)",
       axes={"horizon": 60, "observation_noise": 0.15}, objective="predict_heldout",
       claims=(C + "market_rule",), identify=I + "market_rule",
       reference="crucible.rapid.refs_adapt:market_predict", oracle="crucible.rapid.oracles:science_oracle", version="2", lever_view=True)(_market("signal"))
family("mk_gate", engine="primitive_graph", category="8.1", item="8.1.6", split="val",
       doc="Look-alike market, rule 'gate': price has almost no effect until a compatibility certification (a "
           "discrete decision) is bought; then demand is elastic.",
       distinct_because="same interface; the active causal path from price depends on a discrete state "
                        "(complementarity of certification and price response)",
       axes={"horizon": 60, "observation_noise": 0.15}, objective="predict_heldout",
       claims=(C + "market_rule",), identify=I + "market_rule",
       reference="crucible.rapid.refs_adapt:market_predict", oracle="crucible.rapid.oracles:science_oracle", version="2")(_market("gate"))
family("mk_drift", engine="primitive_graph", category="8.10", item="8.10.1", split="train",
       doc="Look-alike market, rule 'drift': a rival's share grows with the shop's price history; once it crosses "
           "a hidden level, demand becomes several times more price-sensitive, permanently.",
       distinct_because="same interface; a regime change of the price effect triggered by an accumulating "
                        "competitor stock (endogenous, path-dependent change of the causal effect)",
       axes={"horizon": 60, "observation_noise": 0.15}, objective="predict_heldout",
       claims=(C + "market_rule",), identify=I + "market_rule",
       reference="crucible.rapid.refs_adapt:market_predict", oracle="crucible.rapid.oracles:science_oracle", version="2", lever_view=True)(_market("drift"))


# ====================================================================== 8.2 experimental science
def _science(rng, d, law: str):
    gb = GraphBuilder(rng)
    x1 = gb.lever(0.0, 6.0, 3.0, cost=0.0)
    x2 = gb.lever(0.0, 6.0, 3.0, cost=0.0)
    noise = float(d["observation_noise"])
    if law == "saturating_product":
        s1 = gb.prim("saturation", {"cap": u(rng, 8, 14), "half": u(rng, 1.5, 4.0)}, x=[(x1, 0, 1.0)])
        l2 = gb.prim("linear_nonlinear", {"form": "linear", "a": u(rng, 0.2, 0.5), "b": u(rng, 0.5, 1.0)},
                     x=[(x2, 0, 1.0)])
        y = gb.prim("multiplicative", {"k": 1.0}, role="outcome", a=[(s1, 0, 1.0)], b=[(l2, 0, 1.0)])
        instruments = None
    else:                                                     # threshold law, measured with instruments
        th = gb.prim("threshold", {"theta": u(rng, 2.0, 4.0), "lo": 0.0, "hi": u(rng, 3.0, 6.0)}, x=[(x2, 0, 1.0)])
        lin = gb.prim("linear_nonlinear", {"form": "quadratic", "a": u(rng, 0.5, 1.2), "b": u(rng, 1.0, 3.0),
                                           "c": u(rng, 0.05, 0.15)}, x=[(x1, 0, 1.0)])
        y = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, role="outcome",
                    x=[(th, 0, 1.0), (lin, 0, 1.0)])
        instruments = {"field": {"sd_mult": 1.0, "cost_mult": 1.0}, "lab": {"sd_mult": 0.15, "cost_mult": 8.0}}
    gy = gb.gauge(y, [("measurement_noise", {"sd": noise})])
    heldout = [{gb.name_of(x1): a, gb.name_of(x2): b} for a, b in
               ((7.5, 1.0), (7.5, 4.0), (9.0, 2.0), (9.0, 5.0), (8.0, 0.5), (8.5, 3.0))]
    exp = {"vars": [gb.name_of(x1), gb.name_of(x2)], "measure": [gy], "cost": float(d["experiment_cost"]),
           "budget": float(d["resources"]), "max_ticks": 3, "max_replicates": 8,
           "box": {gb.name_of(x1): [0.0, 6.0], gb.name_of(x2): [0.0, 6.0]}}
    if instruments:
        exp["instruments"] = instruments
    text = ("Find the law that relates the two settings to the reading. Predict the noise-free reading at these "
            f"conditions, which lie outside what you can set: {heldout}. File predictions with the journal: "
            "message to 'journal' with content {\"predictions\": [{\"set\": <condition>, \"value\": <number>}, ...]}. "
            "Only the last filing counts.")
    return gb.build(horizon=int(d["horizon"]), affordances=["levers", "experiment", "message"], experiment=exp,
                    objective={"type": "predict_heldout", "heldout": heldout, "gauge": gy, "text": text},
                    journal=True, brief="A laboratory apparatus with two settings and one sensor.",
                    claim={"levers": [x1, x2], "outcome": y, "law": law})


@family("sci_functional_law", engine="primitive_graph", category="8.2", item="8.2.1", split="train",
        doc="Discover an unknown functional law (a saturating response times a linear one) by experiment and predict "
            "it at conditions outside the range that can be set; scored on those unobserved conditions.",
        distinct_because="objective is prediction at held-out conditions (scientific law discovery) through designed "
                         "experiments and journal filings; no money",
        axes={"horizon": 40, "observation_noise": 0.3, "experiment_cost": 1.0, "resources": 200.0},
        objective="predict_heldout", claims=(C + "law_extrapolates",), identify=I + "law_shape")
def sci_functional_law(rng, d):
    return _science(rng, d, "saturating_product")


@family("sci_instruments", engine="primitive_graph", category="8.2", item="8.2.6", split="val",
        doc="Very noisy measurements, a limited budget and two instruments that differ in precision and cost: "
            "replicate or buy precision to identify a threshold law well enough to predict it out of range.",
        distinct_because="instrument selection under a budget with heavy measurement noise; a threshold plus "
                         "quadratic law instead of a saturating product",
        axes={"horizon": 40, "observation_noise": 2.0, "experiment_cost": 1.0, "resources": 120.0},
        objective="predict_heldout", claims=(C + "instrument_precision",), identify=I + "law_shape")
def sci_instruments(rng, d):
    return _science(rng, d, "threshold_quadratic")


# ====================================================================== 8.6 delayed consequences
@family("dl_delayed_returns", engine="primitive_graph", category="8.6", item="8.6.1", split="train",
        doc="Investment builds capacity only after a hidden construction delay; capacity depreciates. Paying now "
            "earns nothing for many ticks.",
        distinct_because="a pipeline delay into a depreciating stock feeding a saturating output (credit assignment "
                         "across the construction lag)",
        axes={"horizon": 60, "observation_noise": 0.3, "resources": 0.0}, objective="maximize_net",
        claims=(C + "delayed_return",), identify=I + "lever_sign", lever_view=True,
        reference="crucible.rapid.claims_v2:invest_learn_lag_then_stop")
def dl_delayed_returns(rng, d):
    gb = GraphBuilder(rng)
    inv = gb.lever(0.0, 10.0, 0.0, cost=u(rng, 550, 750))
    lag = int(rng.integers(6, 11))
    pipe = gb.prim("delay", {"lag": lag, "init": 0.0}, x=[(inv, 0, 1.0)])
    cap = gb.prim("stock", {"init": float(d["resources"]), "cap": 1e6}, role="capacity", inflow=[(pipe, 0, 1.0)],
                  outflow=[])
    gb.nodes[int(cap[1:])]["ports"]["outflow"] = [[cap, 1, 0.08]]
    out = gb.prim("saturation", {"cap": u(rng, 70, 90), "half": u(rng, 30, 50)}, role="output",
                  x=[(cap, 0, 1.0)])
    sold = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 10.0}, role="outcome", x=[(out, 0, 1.0)])
    gb.gauge(sold, [("measurement_noise", {"sd": float(d["observation_noise"]) * 3})])
    return gb.build(horizon=int(d["horizon"]), money={"revenue_node": sold, "price": 100.0, "cap": 200.0},
                    affordances=["levers"], endowment=500_000,
                    objective={"type": "maximize_net", "text": "Maximise your net income over the horizon."},
                    brief="A plant with an investment lever. Gauge: units produced.",
                    claim={"lever": inv, "outcome": sold, "lag": lag})


@family("dl_delayed_adoption", engine="primitive_graph", category="8.6", item="8.6.3", split="train",
        doc="Marketing builds awareness only after a hidden delay, and awareness fades. Nobody buys until awareness "
            "passes a hidden level; then trial grows with diminishing returns, word of mouth from earlier adopters "
            "amplifies it, and the market empties. Predict how many have adopted forty ticks after a launch held at "
            "constant spends, some above what can be set: spending pays only after the delay and the threshold, "
            "and much of its effect comes later through adopters.",
        distinct_because="delayed decaying awareness stock gating (threshold) a saturating trial rate that word of "
                         "mouth from the installed base amplifies (network effect) and the remaining market limits, "
                         "accumulating into a stock of adopters; scored by the adopters after a held-out constant "
                         "launch, so credit runs across the delay and the reinforcing loop",
        axes={"horizon": 60, "observation_noise": 0.3}, objective="predict_heldout",
        claims=("crucible.rapid.claims_w3b:claim_adoption_delayed_and_amplified",),
        identify="crucible.rapid.claims_w3b:identify_awareness_delay", lever_view=True,
        reference="crucible.rapid.refs_adapt:adoption_predict", oracle="crucible.rapid.oracles:science_oracle")
def dl_delayed_adoption(rng, d):
    gb = GraphBuilder(rng)
    spend = gb.lever(0.0, 10.0, 0.0, cost=0.0)
    lag = int(rng.integers(2, 5))
    M = u(rng, 400.0, 800.0)
    aware = gb.prim("decay", {"rate": u(rng, 0.1, 0.2), "init": 0.0}, role="awareness",
                    x=[(spend, lag, u(rng, 1.5, 3.0))])
    gate = gb.prim("threshold", {"theta": u(rng, 8.0, 20.0), "lo": 0.0, "hi": 1.0}, x=[(aware, 0, 1.0)])
    trial = gb.prim("saturation", {"cap": u(rng, 6.0, 12.0), "half": u(rng, 15.0, 40.0)}, x=[(aware, 0, 1.0)])
    gated = gb.prim("multiplicative", {"k": 1.0}, a=[(trial, 0, 1.0)], b=[(gate, 0, 1.0)])
    adopters = gb.prim("stock", {"init": 0.0, "cap": M}, role="outcome", inflow=[], outflow=[])
    wom = gb.prim("network_effect", {"strength": u(rng, 3.0, 6.0) / M}, x=[(gated, 0, 1.0)], base=[(adopters, 1, 1.0)])
    rem = gb.prim("linear_nonlinear", {"form": "linear", "a": -1.0 / M, "b": 1.0}, x=[(adopters, 1, 1.0)])
    flow = gb.prim("multiplicative", {"k": 1.0}, a=[(wom, 0, 1.0)], b=[(rem, 0, 1.0)])
    gb.nodes[int(adopters[1:])]["ports"]["inflow"] = [[flow, 0, 1.0]]
    noise = float(d["observation_noise"])
    gy = gb.gauge(adopters, [("measurement_noise", {"sd": noise * 4})])
    gb.gauge(aware, [("measurement_noise", {"sd": noise * 3})])
    held = [{gb.name_of(spend): x} for x in (1.0, 3.0, 5.0, 8.0, 11.0, 14.0)]
    text = ("Predict how many customers would have adopted forty ticks after a launch if marketing spend were held "
            f"at these levels from the start (partly above what you can set): {held}. File predictions with the "
            "journal: message to 'journal' with content {\"predictions\": [{\"set\": <condition>, \"value\": "
            "<number>}, ...]}. Only the last filing counts.")
    return gb.build(horizon=int(d["horizon"]), money={"revenue_node": flow, "price": 50.0, "cap": 100.0},
                    affordances=["levers", "message"], endowment=200_000,
                    objective={"type": "predict_heldout", "heldout": held, "gauge": gy, "text": text,
                               "settle_ticks": 40},
                    journal=True,
                    brief="A new product. Lever: marketing spend per tick. Gauges: customers who have adopted and an "
                          "awareness survey.",
                    claim={"lever": spend, "outcome": adopters, "awareness": aware, "lag": lag, "market": M})


@family("net_congestion_routes", engine="primitive_graph", category="8.7", item="8.7.1", split="train",
        doc="Traffic is sent over two routes; each has its own link of hidden capacity and both share a bottleneck "
            "link of hidden capacity, and every link's delay rises steeply as it nears its capacity (BPR "
            "congestion). Predict the delay on the first route at flows above what can be sent: a straight line "
            "through light-traffic tests misses the steep part, and the second route's traffic slows the first "
            "through the shared link.",
        distinct_because="two routes of two links each, one link shared: BPR congestion delays (convex in load over "
                         "capacity) summed along a route, so route delays interact through the shared bottleneck; "
                         "scored by the delay at held-out flows above the settable range",
        axes={"horizon": 50, "observation_noise": 0.3}, objective="predict_heldout",
        claims=("crucible.rapid.claims_w3b:claim_shared_congestion",),
        identify="crucible.rapid.claims_w3b:identify_tighter_route", lever_view=True,
        reference="crucible.rapid.refs_adapt:congestion_predict", oracle="crucible.rapid.oracles:science_oracle")
def net_congestion_routes(rng, d):
    gb = GraphBuilder(rng)
    fa = gb.lever(0.0, 10.0, 2.0, cost=0.0)
    fb = gb.lever(0.0, 10.0, 2.0, cost=0.0)
    beta = float(rng.choice([3.0, 4.0]))
    own_a = gb.prim("congestion", {"t0": u(rng, 4.0, 8.0), "alpha": u(rng, 0.5, 1.0), "beta": beta},
                    bias={"cap": u(rng, 7.0, 11.0)}, load=[(fa, 0, 1.0)], cap=[])
    own_b = gb.prim("congestion", {"t0": u(rng, 4.0, 8.0), "alpha": u(rng, 0.5, 1.0), "beta": beta},
                    bias={"cap": u(rng, 7.0, 11.0)}, load=[(fb, 0, 1.0)], cap=[])
    shared = gb.prim("congestion", {"t0": u(rng, 2.0, 5.0), "alpha": u(rng, 0.5, 1.0), "beta": beta},
                     bias={"cap": u(rng, 11.0, 16.0)}, load=[(fa, 0, 1.0), (fb, 0, 1.0)], cap=[])
    delay_a = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, role="outcome",
                      x=[(own_a, 0, 1.0), (shared, 0, 1.0)])
    delay_b = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, x=[(own_b, 0, 1.0), (shared, 0, 1.0)])
    noise = float(d["observation_noise"])
    ga = gb.gauge(delay_a, [("measurement_noise", {"sd": noise * 2})])
    gb.gauge(delay_b, [("measurement_noise", {"sd": noise * 2})])
    held = [{gb.name_of(fa): x, gb.name_of(fb): y} for x, y in
            ((12.0, 2.0), (11.0, 8.0), (14.0, 5.0), (6.0, 13.0), (13.0, 12.0), (2.0, 15.0), (16.0, 1.0))]
    text = ("Predict the delay on the first route if these flows were sent over the two routes (partly above what "
            f"you can send): {held}. File predictions with the journal: message to 'journal' with content "
            "{\"predictions\": [{\"set\": <condition>, \"value\": <number>}, ...]}. Only the last filing counts.")
    return gb.build(horizon=int(d["horizon"]), money={"revenue_node": delay_a, "price": 1.0, "cap": 1.0},
                    affordances=["levers", "message"], endowment=100_000,
                    objective={"type": "predict_heldout", "heldout": held, "gauge": ga, "text": text},
                    journal=True,
                    brief="Two routes through a network. Levers: the flow sent on each route. Gauges: the delay on "
                          "each route.",
                    claim={"first": fa, "second": fb, "outcome": delay_a, "own_a": own_a, "own_b": own_b,
                           "shared": shared})


@family("plan_spoilage", engine="primitive_graph", category="8.4", item="8.4.2", split="train",
        doc="Units made each tick join a stock that spoils at a hidden rate; customers buy more when shelves are "
            "fuller, with sharply diminishing returns, so beyond some point extra production only feeds the bin. "
            "Predict where weekly sales settle at production rates above what can be set: a straight line through "
            "small tests keeps rising, while spoilage takes almost all of the extra units.",
        distinct_because="a perishing stock (inflow production, outflow sales plus a hidden spoilage fraction of the "
                         "stock) whose sales saturate in the stock on hand: the settled split between sales and "
                         "waste moves with the production rate; scored by settled sales at held-out rates above "
                         "the settable range",
        axes={"horizon": 60, "observation_noise": 0.3}, objective="predict_heldout",
        claims=("crucible.rapid.claims_w3b:claim_spoilage",),
        identify="crucible.rapid.claims_w3b:identify_spoilage_rate", lever_view=True,
        reference="crucible.rapid.refs_adapt:spoilage_predict", oracle="crucible.rapid.oracles:science_oracle")
def plan_spoilage(rng, d):
    gb = GraphBuilder(rng)
    make = gb.lever(0.0, 10.0, 0.0, cost=0.0)
    spoil = u(rng, 0.15, 0.35)
    stock = gb.prim("stock", {"init": 0.0, "cap": 1e6}, role="stock", inflow=[(make, 0, 1.0)], outflow=[])
    cap = u(rng, 6.0, 10.0)
    # sales never exceed what is on the shelf: cap / half <= 0.56 and spoilage <= 0.35, so sales + waste stay below
    # the stock of the previous tick (no phantom units)
    sales = gb.prim("saturation", {"cap": cap, "half": cap * u(rng, 1.8, 3.0)}, role="outcome",
                    x=[(stock, 1, 1.0)])
    waste = gb.prim("linear_nonlinear", {"form": "linear", "a": spoil, "b": 0.0}, x=[(stock, 1, 1.0)])
    gb.nodes[int(stock[1:])]["ports"]["outflow"] = [[sales, 0, 1.0], [waste, 0, 1.0]]
    noise = float(d["observation_noise"])
    gs = gb.gauge(sales, [("measurement_noise", {"sd": noise})])
    gb.gauge(waste, [("measurement_noise", {"sd": noise})])
    held = [{gb.name_of(make): x} for x in (2.0, 5.0, 8.0, 11.0, 14.0, 18.0)]
    text = ("Predict where sales per tick would settle if production were held at these rates (partly above what "
            f"you can set): {held}. File predictions with the journal: message to 'journal' with content "
            "{\"predictions\": [{\"set\": <condition>, \"value\": <number>}, ...]}. Only the last filing counts.")
    return gb.build(horizon=int(d["horizon"]), money={"revenue_node": sales, "price": 10.0, "cap": 20.0},
                    affordances=["levers", "message"], endowment=100_000,
                    objective={"type": "predict_heldout", "heldout": held, "gauge": gs, "text": text,
                               "settle_ticks": 200},
                    journal=True,
                    brief="A perishable good. Lever: units made per tick. Gauges: units sold and units thrown away.",
                    claim={"lever": make, "outcome": sales, "stock": stock, "spoil": spoil, "waste": waste})


@family("plan_serial_stages", engine="primitive_graph", category="8.4", item="8.4.3", split="train",
        doc="Raw material released into a three-stage chain: each stage works off its queue with diminishing "
            "returns up to a hidden capacity, and goods take a hidden transit time between stages. Predict what the "
            "chain ships per tick forty ticks after a constant release, at rates partly above what can be set: the "
            "slowest stage caps the chain, the queue in front of it grows, and short tests mislead because goods "
            "are still in transit.",
        distinct_because="a serial chain of three queue stocks, each drained by a saturating processing rate of "
                         "hidden capacity, joined by hidden transit lags; a bottleneck stage caps throughput and "
                         "accumulates work in process; scored by shipments after a held-out constant release",
        axes={"horizon": 60, "observation_noise": 0.3}, objective="predict_heldout",
        claims=("crucible.rapid.claims_w3b:claim_serial_bottleneck",),
        identify="crucible.rapid.claims_w3b:identify_bottleneck_stage", lever_view=True,
        reference="crucible.rapid.refs_adapt:serial_chain_predict", oracle="crucible.rapid.oracles:science_oracle")
def plan_serial_stages(rng, d):
    gb = GraphBuilder(rng)
    rel = gb.lever(0.0, 10.0, 0.0, cost=0.0)
    caps = [u(rng, 8.0, 14.0) for _ in range(3)]
    caps[int(rng.integers(3))] = u(rng, 4.5, 7.0)              # one stage is the bottleneck
    lags = [int(rng.integers(1, 4)), int(rng.integers(1, 4))]
    noise = float(d["observation_noise"])
    stocks, procs = [], []
    prev, prev_lag = rel, 0
    for k in range(3):
        st = gb.prim("stock", {"init": 0.0, "cap": 1e6}, role=f"queue{k}", inflow=[(prev, prev_lag, 1.0)], outflow=[])
        # a stage never processes more than its queue held (cap / half <= 1): no phantom goods downstream
        pr = gb.prim("saturation", {"cap": caps[k], "half": caps[k] * u(rng, 1.0, 2.0)},
                     role="outcome" if k == 2 else f"processed{k}", x=[(st, 1, 1.0)])
        gb.nodes[int(st[1:])]["ports"]["outflow"] = [[pr, 0, 1.0]]
        stocks.append(st)
        procs.append(pr)
        prev, prev_lag = pr, (lags[k] if k < 2 else 0)
    gship = gb.gauge(procs[2], [("measurement_noise", {"sd": noise})])
    for k in range(3):
        gb.gauge(stocks[k], [("measurement_noise", {"sd": noise * 2})])
    for k in range(2):
        gb.gauge(procs[k], [("measurement_noise", {"sd": noise})])
    held = [{gb.name_of(rel): x} for x in (2.0, 4.0, 6.0, 9.0, 12.0, 15.0)]
    text = ("Predict what the chain would ship per tick forty ticks after a constant release of raw material at "
            f"these rates (partly above what you can set): {held}. File predictions with the journal: message to "
            "'journal' with content {\"predictions\": [{\"set\": <condition>, \"value\": <number>}, ...]}. Only the "
            "last filing counts.")
    return gb.build(horizon=int(d["horizon"]), money={"revenue_node": procs[2], "price": 10.0, "cap": 20.0},
                    affordances=["levers", "message"], endowment=100_000,
                    objective={"type": "predict_heldout", "heldout": held, "gauge": gship, "text": text,
                               "settle_ticks": 40},
                    journal=True,
                    brief="A three-stage chain. Lever: raw material released per tick. Gauges: goods shipped by the last "
                          "stage, the queue at each stage, and the goods processed by the first two stages.",
                    claim={"lever": rel, "outcome": procs[2], "stocks": stocks, "procs": procs, "caps": caps,
                           "lags": lags})


@family("net_interdependent", engine="primitive_graph", category="8.7", item="8.7.4", split="val",
        doc="Two interdependent networks (a power grid and the communications network that controls it): in each, "
            "components on a ring carry a base load plus the demand put on the network; an overloaded component "
            "fails and sheds its load onto its neighbours (a cascade). Failures in one network add stress to the "
            "other a tick later. Predict the first network's service after fifteen ticks from an intact start at "
            "demand pairs partly above what can be set: a demand that one network survives alone can bring both "
            "down through the other, and a collapse can sustain itself.",
        distinct_because="two cascading-failure rings coupled by lagged cross-stress (each network's failed share "
                         "stresses the other): thresholds, load redistribution and mutual reinforcement give "
                         "bistable collapse; scored by the service after an intact start at held-out demand pairs",
        axes={"horizon": 60, "observation_noise": 0.3}, objective="predict_heldout",
        claims=("crucible.rapid.claims_w3b:claim_interdependent_cascade",),
        identify="crucible.rapid.claims_w3b:identify_first_threshold", lever_view=True,
        reference="crucible.rapid.refs_adapt:interdependent_predict", oracle="crucible.rapid.oracles:science_oracle")
def net_interdependent(rng, d):
    gb = GraphBuilder(rng)
    la = gb.lever(0.0, 10.0, 0.0, cost=0.0)
    lb = gb.lever(0.0, 10.0, 0.0, cost=0.0)
    m = 20
    nodes = []
    for _ in range(2):
        base, spread = 5.0, u(rng, 0.5, 0.9)
        thr = u(rng, 3.5, 8.0)                          # demand at which the most loaded component fails
        nodes.append({"load": base, "spread": spread, "cap": base * (1.0 + spread) + thr, "thr": thr})
    kab, kba = u(rng, 2.0, 7.0), u(rng, 2.0, 7.0)       # extra stress from a fully failed other network
    fa = gb.prim("cascading_failure", {"m": m, "load": nodes[0]["load"], "spread": nodes[0]["spread"],
                                       "cap": nodes[0]["cap"]}, role="failed_a", stress=[(la, 0, 1.0)])
    fb = gb.prim("cascading_failure", {"m": m, "load": nodes[1]["load"], "spread": nodes[1]["spread"],
                                       "cap": nodes[1]["cap"]}, role="failed_b", stress=[(lb, 0, 1.0), (fa, 1, kba)])
    gb.nodes[int(fa[1:])]["ports"]["stress"] = [[la, 0, 1.0], [fb, 1, kab]]
    sa = gb.prim("linear_nonlinear", {"form": "linear", "a": -1.0, "b": 1.0}, role="outcome", x=[(fa, 0, 1.0)])
    sb = gb.prim("linear_nonlinear", {"form": "linear", "a": -1.0, "b": 1.0}, x=[(fb, 0, 1.0)])
    noise = float(d["observation_noise"])
    ga = gb.gauge(sa, [("measurement_noise", {"sd": noise * 0.05})])
    gb.gauge(sb, [("measurement_noise", {"sd": noise * 0.05})])
    held = [{gb.name_of(la): x, gb.name_of(lb): y} for x, y in
            ((2.0, 2.0), (5.0, 9.0), (9.0, 1.0), (11.0, 3.0), (3.0, 12.0), (6.0, 6.0), (12.0, 12.0), (1.0, 7.0))]
    text = ("Predict the share of the first network's components in service fifteen ticks after an intact start if "
            f"these demands were put on the two networks (partly above what you can set): {held}. File predictions "
            "with the journal: message to 'journal' with content {\"predictions\": [{\"set\": <condition>, "
            "\"value\": <number>}, ...]}. Only the last filing counts.")
    return gb.build(horizon=int(d["horizon"]), money={"revenue_node": sa, "price": 100.0, "cap": 1.0},
                    affordances=["levers", "message"], endowment=100_000,
                    objective={"type": "predict_heldout", "heldout": held, "gauge": ga, "text": text,
                               "settle_ticks": 15},
                    journal=True,
                    brief="Two interdependent networks. Levers: the demand put on each. Gauges: the share of each "
                          "network's components in service.",
                    claim={"first": la, "second": lb, "outcome": sa, "failed": [fa, fb], "thr": [n["thr"] for n in nodes],
                           "coupling": [kab, kba]})


@family("dl_short_term_damage", engine="primitive_graph", category="8.6", item="8.6.6", split="train",
        doc="Overdriving raises output now but accumulates wear (shown by a noisy vibration reading) that drains "
            "only below a hidden rate; past a hidden level the machine latches into a permanently degraded state. "
            "Predict the output a new machine gives after forty ticks at constant drives, some above what can be "
            "set: the short-run gain of a high drive turns into a long-run loss.",
        distinct_because="a stock of wear (inflow the drive, a hidden drain) drives a hysteresis latch that never "
                         "resets (irreversible long-term deterioration from a short-term gain); scored by the "
                         "long-run output at held-out constant drives, which needs the drain, the latch level and "
                         "the degraded yield, each learned from one episode in which the latch can trip only once",
        axes={"horizon": 60, "observation_noise": 0.3}, objective="predict_heldout",
        claims=(C + "damage_persists",), identify="crucible.rapid.claims_w3b:identify_latch_level", lever_view=True,
        reference="crucible.rapid.refs_adapt:wear_latch_predict", oracle="crucible.rapid.oracles:science_oracle",
        version="2")
def dl_short_term_damage(rng, d):
    gb = GraphBuilder(rng)
    boost = gb.lever(0.0, 10.0, 2.0, cost=0.0)
    drain = u(rng, 2.5, 3.5)
    wear = gb.prim("stock", {"init": 0.0, "cap": 1e6}, role="wear", inflow=[(boost, 0, 1.0)], outflow=[],
                   bias={"outflow": drain})
    hi = u(rng, 30, 45)
    broken = gb.prim("hysteresis", {"lo": -1.0, "hi": hi, "off": 1.0, "on": u(rng, 0.3, 0.45), "init": 0.0},
                     role="health", x=[(wear, 0, 1.0)])
    raw = gb.prim("linear_nonlinear", {"form": "linear", "a": u(rng, 2.5, 3.5), "b": u(rng, 18, 24)},
                  x=[(boost, 0, 1.0)])
    out = gb.prim("multiplicative", {"k": 1.0}, role="outcome", a=[(raw, 0, 1.0)], b=[(broken, 0, 1.0)])
    gy = gb.gauge(out, [("measurement_noise", {"sd": float(d["observation_noise"]) * 3})])
    gb.gauge(wear, [("measurement_noise", {"sd": 1.0})])
    gb.gauge(broken, [("measurement_noise", {"sd": 0.02})], on_demand=True, query_cost=0.0)
    held = [{gb.name_of(boost): x} for x in (1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0)]
    text = ("Predict the output a new machine would give after forty ticks if it were driven at these constant "
            f"levels (partly above what you can set): {held}. File predictions with the journal: message to "
            "'journal' with content {\"predictions\": [{\"set\": <condition>, \"value\": <number>}, ...]}. Only "
            "the last filing counts.")
    return gb.build(horizon=int(d["horizon"]), money={"revenue_node": out, "price": 100.0, "cap": 200.0},
                    affordances=["levers", "message"], endowment=500_000,
                    objective={"type": "predict_heldout", "heldout": held, "gauge": gy, "text": text,
                               "settle_ticks": 40},
                    journal=True,
                    brief="A machine with a drive lever. Gauges: units produced and a vibration reading.",
                    claim={"lever": boost, "outcome": out, "threshold": hi, "drain": drain, "wear": wear})


# ====================================================================== 8.3 complex engineering (control)
@family("eng_delayed_controller", engine="primitive_graph", category="8.3", item="8.3.3", split="train",
        doc="A heated process whose heater acts only after a hidden delay, through a first-order (decay) plant, "
            "while an unobserved disturbance drifts. Predict where the process temperature settles at heater "
            "powers above what can be set: short tests mislead (the response has not arrived yet, or has not "
            "settled), and the drifting disturbance confounds readings taken at different times.",
        distinct_because="identification of a delayed first-order (decay) plant with an unobserved drifting "
                         "disturbance and a band-pass payoff (two thresholds) on its output; scored by the settled "
                         "response at held-out heater powers above the settable range (the closed-loop control "
                         "of the same item is pe_deadtime_control)",
        axes={"horizon": 80, "observation_noise": 0.5}, objective="predict_heldout",
        claims=(C + "controller_delay",), identify="crucible.rapid.claims_w3b:identify_heater_delay", lever_view=True,
        reference="crucible.rapid.refs_adapt:delayed_plant_predict", oracle="crucible.rapid.oracles:science_oracle",
        version="2")
def eng_delayed_controller(rng, d):
    gb = GraphBuilder(rng)
    heat = gb.lever(0.0, 10.0, 3.0, cost=0.0)
    dist = gb.prim("hidden_confounding", {"mean": 0.0, "sd": u(rng, 0.2, 0.3), "ar": 0.95}, role="disturbance")
    lag = int(rng.integers(2, 5))
    T = gb.prim("decay", {"rate": u(rng, 0.12, 0.3), "init": 10.0}, role="temperature", bias={"x": u(rng, 1.0, 3.0)},
                x=[(heat, lag, u(rng, 0.5, 1.0)), (dist, 0, 1.0)])
    sp = u(rng, 16, 25) if rng.random() < 0.5 else u(rng, 35, 44)
    band = u(rng, 1.5, 2.5)
    lo = gb.prim("threshold", {"theta": sp - band, "lo": 0.0, "hi": 1.0}, x=[(T, 0, 1.0)])
    hi = gb.prim("threshold", {"theta": sp + band, "lo": 1.0, "hi": 0.0}, x=[(T, 0, 1.0)])
    inband = gb.prim("multiplicative", {"k": 1.0}, role="outcome", a=[(lo, 0, 1.0)], b=[(hi, 0, 1.0)])
    gt = gb.gauge(T, [("measurement_noise", {"sd": float(d["observation_noise"])})])
    gb.gauge(inband, [])
    held = [{gb.name_of(heat): x} for x in (10.5, 11.5, 12.5, 14.0, 16.0)]
    text = ("Predict where the process temperature would settle if the heater were held at these powers (above "
            f"what you can set): {held}. File predictions with the journal: message to 'journal' with content "
            "{\"predictions\": [{\"set\": <condition>, \"value\": <number>}, ...]}. Only the last filing counts.")
    return gb.build(horizon=int(d["horizon"]), money={"revenue_node": inband, "price": 3000.0, "cap": 1.0},
                    affordances=["levers", "program", "message"], endowment=200_000,
                    program_max_ticks=int(d["horizon"]),
                    objective={"type": "predict_heldout", "heldout": held, "gauge": gt, "text": text,
                               "settle_ticks": 200},
                    journal=True,
                    brief="A heated process. Lever: heater power. Gauges: process temperature and an "
                          "in-specification signal.",
                    claim={"lever": heat, "outcome": T, "lag": lag, "setpoint": sp})


# ====================================================================== 8.8 strategic games
@family("game_adaptive_rival", engine="primitive_graph", category="8.8", item="8.8.1", split="train",
        doc="Repeated price competition against a rival that learns the shop's typical price and undercuts it; how "
            "fast it learns is the opponent's capability. Predict the long-run sales of holding a price, once the "
            "rival has adapted to it: the short-run response to a price change overstates it.",
        distinct_because="logit strategic interaction with an adaptive (learning) opponent whose action is a state "
                         "driven by the participant's past actions",
        axes={"horizon": 60, "observation_noise": 0.15, "opponent_capability": 0.2}, objective="predict_heldout",
        participants=("adaptive_opponent:undercut",), claims=(C + "rival_adapts",), identify=I + "lever_sign",
        lever_view=True, reference="crucible.rapid.refs_adapt:rival_model_predict",
        oracle="crucible.rapid.oracles:science_oracle", version="2")
def game_adaptive_rival(rng, d):
    gb = GraphBuilder(rng)
    p = gb.lever(1.0, 10.0, 4.0, cost=0.0)
    lr = float(d["opponent_capability"])
    opp = gb.prim("adaptive_opponent", {"lr": lr, "lm": lr, "margin": u(rng, 0.5, 1.0), "lo": 1.0, "hi": 10.0,
                                        "init": u(rng, 4.0, 6.0), "mode": "undercut"}, role="opponent",
                  x=[(p, 1, 1.0)])
    units = gb.prim("strategic_interaction", {"market": 100.0, "beta": u(rng, 0.6, 0.9), "outside": 0.3},
                    role="units", own=[(p, 0, 1.0)], rival=[(opp, 0, 1.0)])
    rev = gb.prim("multiplicative", {"k": 10.0}, role="outcome", a=[(units, 0, 1.0)], b=[(p, 0, 1.0)])
    gu = gb.gauge(units, [("measurement_noise", {"sd": float(d["observation_noise"]) * 10})])
    gb.gauge(opp, [("measurement_noise", {"sd": 0.05})])
    held = [{gb.name_of(p): v} for v in (1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 11.0, 12.5)]
    text = ("Predict how many units you would sell per period once the rival has adapted, if you held your price "
            f"at each of these values from the start (partly outside what you can set): {held}. File predictions "
            "with the journal: message to 'journal' with content {\"predictions\": [{\"set\": <condition>, "
            "\"value\": <number>}, ...]}. Only the last filing counts.")
    return gb.build(horizon=int(d["horizon"]), money={"revenue_node": rev, "price": 1.0, "cap": 20_000.0},
                    affordances=["levers", "message"], endowment=200_000,
                    objective={"type": "predict_heldout", "heldout": held, "gauge": gu, "text": text,
                               "settle_ticks": 80},
                    journal=True,
                    brief="You set your price each tick against one rival. Gauges: your units sold and the "
                          "rival's posted price.",
                    claim={"lever": p, "opponent": opp, "outcome": rev})
