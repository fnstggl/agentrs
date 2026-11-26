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
            "strong positive spending-sales correlation, but spending itself lowers sales slightly and costs money.",
        distinct_because="hidden confounder with two children (natural-regime lever and outcome) plus historical "
                         "records: the passive association and the interventional effect have opposite signs",
        axes={"horizon": 60, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(C + "confounded_reversal",), identify=I + "lever_sign", lever_view=True)
def hc_confounded(rng, d):
    gb = GraphBuilder(rng)
    U = gb.prim("hidden_confounding", {"mean": 0.0, "sd": u(rng, 0.4, 0.5), "ar": 0.9}, role="confounder")
    X = gb.lever(0.0, 10.0, 5.0, cost=u(rng, 40, 70), natural={"x": [(U, 0, 2.0)], "bias": 5.0, "sd": 0.3})
    Q = gb.lever(0.0, 10.0, 2.0, cost=u(rng, 60, 100))
    base = gb.prim("saturation", {"cap": u(rng, 25, 35), "half": u(rng, 2, 4)}, x=[(Q, 0, 1.0)])
    sales = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, role="outcome",
                    bias={"x": u(rng, 35, 45)}, x=[(U, 0, u(rng, 3.5, 4.5)), (X, 0, -u(rng, 0.3, 0.8)),
                                                   (base, 0, 1.0)])
    proxy = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 50.0}, x=[(U, 0, 3.0)])
    noise = float(d["observation_noise"])
    gb.gauge(sales, [("measurement_noise", {"sd": noise * 3})])
    gb.gauge(proxy, [("measurement_noise", {"sd": 4.0})])
    _distractors(gb, rng, d["n_variables"], noise)
    return gb.build(horizon=int(d["horizon"]), money={"revenue_node": sales, "price": 100.0, "cap": 200.0},
                    affordances=["levers"], warmup=40, endowment=500_000,
                    objective={"type": "maximize_net", "text": "Maximise your net income over the horizon."},
                    brief="You take over a shop. Two spending levers; gauges: units sold and a footfall count. "
                          "You have the previous operator's records.",
                    claim={"lever": X, "outcome": sales, "confounder": U})


@family("hc_conditional", engine="primitive_graph", category="8.1", item="8.1.2", split="train",
        doc="One lever works only through a pathway that is open when a second, costly lever is above a hidden "
            "threshold; below it the first lever does nothing.",
        distinct_because="a threshold gates a saturating pathway multiplicatively (conditional causal path); no "
                         "confounder, no history",
        axes={"horizon": 60, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(C + "conditional_pathway",), identify=I + "lever_sign", lever_view=True)
def hc_conditional(rng, d):
    gb = GraphBuilder(rng)
    A = gb.lever(0.0, 10.0, 1.0, cost=u(rng, 250, 350))
    Bv = gb.lever(0.0, 10.0, 2.0, cost=u(rng, 60, 90))
    theta = u(rng, 6.5, 8.5)
    gate = gb.prim("threshold", {"theta": theta, "lo": 0.0, "hi": 1.0}, x=[(Bv, 0, 1.0)])
    eff = gb.prim("saturation", {"cap": u(rng, 50, 70), "half": u(rng, 1.5, 2.5)}, x=[(A, 0, 1.0)])
    path = gb.prim("multiplicative", {"k": 1.0}, a=[(eff, 0, 1.0)], b=[(gate, 0, 1.0)])
    season = gb.prim("hidden_confounding", {"mean": 0.0, "sd": 0.5, "ar": 0.8}, role="exogenous")
    sales = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, role="outcome",
                    bias={"x": u(rng, 25, 35)}, x=[(path, 0, 1.0), (Bv, 0, -0.5), (season, 0, 2.0)])
    noise = float(d["observation_noise"])
    gb.gauge(sales, [("measurement_noise", {"sd": noise * 3})])
    gb.gauge(eff, [("measurement_noise", {"sd": noise * 3})], lag=1)
    _distractors(gb, rng, d["n_variables"], noise)
    return gb.build(horizon=int(d["horizon"]), money={"revenue_node": sales, "price": 100.0, "cap": 200.0},
                    affordances=["levers"], endowment=500_000,
                    objective={"type": "maximize_net", "text": "Maximise your net income over the horizon."},
                    brief="A workshop with two spending levers. Gauges: units sold and an engagement count "
                          "(reported a tick late).",
                    claim={"lever": A, "gate_lever": Bv, "theta": theta, "outcome": sales})


@family("hc_feedback_hidden", engine="primitive_graph", category="8.1", item="8.1.5", split="val",
        doc="Sales build a hidden reputation stock that raises later sales: the long-run effect of a lever is about "
            "twice its immediate effect, through an intermediate variable nobody observes.",
        distinct_because="a positive feedback loop through an unobserved decaying stock with lag (cycle in the "
                         "causal graph); the observed short-run response understates the long-run one",
        axes={"horizon": 80, "observation_noise": 0.3}, objective="maximize_net",
        claims=(C + "feedback_amplifies",), identify=I + "lever_sign", lever_view=True)
def hc_feedback_hidden(rng, d):
    gb = GraphBuilder(rng)
    M = gb.lever(0.0, 10.0, 0.5, cost=u(rng, 450, 550))
    sat = gb.prim("saturation", {"cap": u(rng, 25, 35), "half": u(rng, 3.5, 4.5)}, x=[(M, 0, 1.0)])
    rep = gb.prim("decay", {"rate": 0.1, "init": 0.0}, role="reputation", x=[])
    sales = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, role="outcome",
                    bias={"x": u(rng, 15, 25)}, x=[(sat, 0, 1.0), (rep, 1, u(rng, 4.5, 5.5))])
    gb.nodes[int(rep[1:])]["ports"]["x"] = [[sales, 2, 0.01]]
    gb.gauge(sales, [("measurement_noise", {"sd": float(d["observation_noise"]) * 3})])
    gb.gauge(sat, [("measurement_noise", {"sd": float(d["observation_noise"]) * 3})])
    cfg = gb.build(horizon=int(d["horizon"]), money={"revenue_node": sales, "price": 100.0, "cap": 400.0},
                   affordances=["levers"], endowment=500_000,
                   objective={"type": "maximize_net", "text": "Maximise your net income over the horizon."},
                   brief="A shop with a marketing lever. Gauges: units sold and leads.",
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
       axes={"horizon": 60, "observation_noise": 0.15}, objective="maximize_net",
       claims=(C + "market_rule",), identify=I + "market_rule", lever_view=True)(_market("elastic"))
family("mk_signal", engine="primitive_graph", category="8.1", item="8.1.3", split="train",
       doc="Look-alike market, rule 'signal': buyers read price as quality, so lowering a low price lowers "
           "purchases (non-monotonic response).",
       distinct_because="same interface as the other markets; conversion is non-monotonic in price (rises, "
                        "peaks, falls)",
       axes={"horizon": 60, "observation_noise": 0.15}, objective="maximize_net",
       claims=(C + "market_rule",), identify=I + "market_rule", lever_view=True)(_market("signal"))
family("mk_gate", engine="primitive_graph", category="8.1", item="8.1.6", split="val",
       doc="Look-alike market, rule 'gate': price has almost no effect until a compatibility certification (a "
           "discrete decision) is bought; then demand is elastic.",
       distinct_because="same interface; the active causal path from price depends on a discrete state "
                        "(complementarity of certification and price response)",
       axes={"horizon": 60, "observation_noise": 0.15}, objective="maximize_net",
       claims=(C + "market_rule",), identify=I + "market_rule")(_market("gate"))
family("mk_drift", engine="primitive_graph", category="8.10", item="8.10.1", split="train",
       doc="Look-alike market, rule 'drift': a rival's share grows with the shop's price history; once it crosses "
           "a hidden level, demand becomes several times more price-sensitive, permanently.",
       distinct_because="same interface; a regime change of the price effect triggered by an accumulating "
                        "competitor stock (endogenous, path-dependent change of the causal effect)",
       axes={"horizon": 60, "observation_noise": 0.15}, objective="maximize_net",
       claims=(C + "market_rule",), identify=I + "market_rule", lever_view=True)(_market("drift"))


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


@family("dl_short_term_damage", engine="primitive_graph", category="8.6", item="8.6.6", split="train",
        doc="Overdriving raises output now but accumulates hidden wear; past a hidden level the machine latches "
            "into a permanently degraded state.",
        distinct_because="a stock of wear drives a hysteresis latch that never resets (irreversible long-term "
                         "deterioration from a short-term gain)",
        axes={"horizon": 60, "observation_noise": 0.3}, objective="maximize_net",
        claims=(C + "damage_persists",), identify=I + "lever_sign", lever_view=True)
def dl_short_term_damage(rng, d):
    gb = GraphBuilder(rng)
    boost = gb.lever(0.0, 10.0, 2.0, cost=u(rng, 5, 15))
    wear = gb.prim("stock", {"init": 0.0, "cap": 1e6}, role="wear", inflow=[(boost, 0, 1.0)], outflow=[],
                   bias={"outflow": 3.0})
    hi = u(rng, 30, 45)
    broken = gb.prim("hysteresis", {"lo": -1.0, "hi": hi, "off": 1.0, "on": u(rng, 0.3, 0.45), "init": 0.0},
                     role="health", x=[(wear, 0, 1.0)])
    raw = gb.prim("linear_nonlinear", {"form": "linear", "a": u(rng, 2.5, 3.5), "b": u(rng, 18, 24)},
                  x=[(boost, 0, 1.0)])
    out = gb.prim("multiplicative", {"k": 1.0}, role="outcome", a=[(raw, 0, 1.0)], b=[(broken, 0, 1.0)])
    gb.gauge(out, [("measurement_noise", {"sd": float(d["observation_noise"]) * 3})])
    gb.gauge(broken, [("measurement_noise", {"sd": 0.02})], on_demand=True, query_cost=0.0)
    return gb.build(horizon=int(d["horizon"]), money={"revenue_node": out, "price": 100.0, "cap": 200.0},
                    affordances=["levers"], endowment=500_000,
                    objective={"type": "maximize_net", "text": "Maximise your net income over the horizon."},
                    brief="A machine with a drive lever. Gauge: units produced.",
                    claim={"lever": boost, "outcome": out, "threshold": hi})


# ====================================================================== 8.3 complex engineering (control)
@family("eng_delayed_controller", engine="primitive_graph", category="8.3", item="8.3.3", split="train",
        doc="Keep a process variable inside a hidden band while an unobserved disturbance drifts; the heater acts "
            "only after a delay. A controller program can be deployed to act every tick.",
        distinct_because="closed-loop control through a delayed first-order (decay) plant with band-pass payoff "
                         "(two thresholds); offers sandboxed controller programs",
        axes={"horizon": 80, "observation_noise": 0.5}, objective="maximize_net",
        claims=(C + "controller_delay",), identify=I + "lever_sign", lever_view=True)
def eng_delayed_controller(rng, d):
    gb = GraphBuilder(rng)
    heat = gb.lever(0.0, 10.0, 3.0, cost=u(rng, 15, 30))
    dist = gb.prim("hidden_confounding", {"mean": 0.0, "sd": u(rng, 0.2, 0.3), "ar": 0.95}, role="disturbance")
    lag = int(rng.integers(2, 5))
    T = gb.prim("decay", {"rate": 0.2, "init": 10.0}, role="temperature", bias={"x": 2.0},
                x=[(heat, lag, 0.8), (dist, 0, 1.0)])
    sp = u(rng, 16, 25) if rng.random() < 0.5 else u(rng, 35, 44)
    band = u(rng, 1.5, 2.5)
    lo = gb.prim("threshold", {"theta": sp - band, "lo": 0.0, "hi": 1.0}, x=[(T, 0, 1.0)])
    hi = gb.prim("threshold", {"theta": sp + band, "lo": 1.0, "hi": 0.0}, x=[(T, 0, 1.0)])
    inband = gb.prim("multiplicative", {"k": 1.0}, role="outcome", a=[(lo, 0, 1.0)], b=[(hi, 0, 1.0)])
    gb.gauge(T, [("measurement_noise", {"sd": float(d["observation_noise"])})])
    gb.gauge(inband, [])
    return gb.build(horizon=int(d["horizon"]), money={"revenue_node": inband, "price": 3000.0, "cap": 1.0},
                    affordances=["levers", "program"], endowment=200_000, program_max_ticks=int(d["horizon"]),
                    objective={"type": "maximize_net", "text": "You are paid every tick the process is in "
                                                               "specification; heating costs money."},
                    brief="A heated process. Lever: heater power. Gauges: process temperature and an "
                          "in-specification signal.",
                    claim={"lever": heat, "outcome": T, "lag": lag, "setpoint": sp})


# ====================================================================== 8.8 strategic games
@family("game_adaptive_rival", engine="primitive_graph", category="8.8", item="8.8.1", split="train",
        doc="Repeated price competition against a rival that learns the shop's typical price and undercuts it; how "
            "fast it learns is the opponent's capability.",
        distinct_because="logit strategic interaction with an adaptive (learning) opponent whose action is a state "
                         "driven by the participant's past actions",
        axes={"horizon": 60, "observation_noise": 0.15, "opponent_capability": 0.2}, objective="maximize_net",
        participants=("adaptive_opponent:undercut",), claims=(C + "rival_adapts",), identify=I + "lever_sign",
        lever_view=True)
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
    gb.gauge(units, [("measurement_noise", {"sd": float(d["observation_noise"]) * 10})])
    gb.gauge(opp, [("measurement_noise", {"sd": 0.05})])
    return gb.build(horizon=int(d["horizon"]), money={"revenue_node": rev, "price": 1.0, "cap": 20_000.0},
                    affordances=["levers"], endowment=200_000,
                    objective={"type": "maximize_net", "text": "Maximise your net income over the horizon."},
                    brief="You set your price each tick against one rival. Gauges: your units sold and the "
                          "rival's posted price.",
                    claim={"lever": p, "opponent": opp, "outcome": rev})
