"""Families built for the redesigned admission (workstream H, recovery wave): worlds in which a simple policy -
the best constant setting in hindsight, a tuned threshold rule, explore-then-commit - cannot come close to a
learner that adapts, because what is best keeps changing for reasons the readings do not show directly, or
because the value lies in information that has to be acquired by experiment.

Each builder composes registered primitives with `GraphBuilder`; seeds draw names and coefficients; difficulty axes
change noise, horizon, cost and size, never the structure. Each family claims its mechanism with controlled
interventions on copies of its worlds (`claims_v2`) and answers its experimental question from the public
interface (`claims_v2.identify_*`).
"""
from __future__ import annotations

from .families import family
from .generator import GraphBuilder

CV = "crucible.rapid.claims_v2:"
RV = "crucible.rapid.refs_v2:"
R = "crucible.rapid.refs_h3:"


def u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def _distractors(gb: GraphBuilder, rng, n: int, noise: float) -> None:
    for _ in range(int(n)):
        d = gb.prim("hidden_confounding", {"mean": u(rng, 20, 60), "sd": u(rng, 1, 3), "ar": 0.8}, role="distractor")
        gb.gauge(d, [("measurement_noise", {"sd": noise * 5})])


SHOP = "Maximise your net income over the horizon."


# ====================================================================== 8.10.3 wandering preferences
# evaluation split (D-9): its mechanism motif set equals the 4.0 validation family scm_invu's (a moving peak is
# still the inverted-U mechanism scm_invu holds out), so it may not be trained on
@family("ch_wandering_taste", engine="primitive_graph", category="8.10", item="8.10.3", split="val",
        doc="Buyers like a design best when it matches their taste, and their taste wanders as a persistent hidden "
            "random walk across the whole design range. Sales show how well the current design fits, never in which "
            "direction taste has moved, so the design has to be re-probed and moved continually.",
        distinct_because="a hidden persistent AR(1) source shifts the peak of a non-monotonic response "
                         "(x - taste) feeding a saturating outcome: the optimum moves continually in a direction "
                         "no reading reveals; no timed switch, no stock, no history",
        axes={"horizon": 120, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(CV + "taste_moves_the_optimum",), identify=CV + "identify_taste_drift", oracle=R + "lever_schedule_oracle",
        reference=CV + "track_moving_peak", lever_view=True)
def ch_wandering_taste(rng, d):
    gb = GraphBuilder(rng)
    X = gb.lever(0.0, 10.0, 0.0, cost=u(rng, 2.0, 5.0))
    P = gb.lever(0.0, 10.0, 0.0, cost=u(rng, 30.0, 50.0))
    w = u(rng, 0.9, 1.2)
    T = gb.prim("hidden_confounding", {"mean": u(rng, 4.5, 5.5), "sd": u(rng, 0.5, 0.6), "ar": 0.97},
                role="taste")
    fit = gb.prim("non_monotonic", {"peak": w, "height": u(rng, 55.0, 70.0)}, role="fit",
                  bias={"x": w}, x=[(X, 0, 1.0), (T, 0, -1.0)])
    promo = gb.prim("saturation", {"cap": u(rng, 6.0, 10.0), "half": u(rng, 2.0, 4.0)}, x=[(P, 0, 1.0)])
    sales = gb.prim("saturation", {"cap": 120.0, "half": u(rng, 60.0, 90.0)}, role="outcome",
                    x=[(fit, 0, 1.0), (promo, 0, 1.0)])
    noise = float(d["observation_noise"])
    gb.gauge(sales, [("measurement_noise", {"sd": noise * 3})])
    _distractors(gb, rng, d["n_variables"], noise)
    return gb.build(horizon=int(d["horizon"]), money={"revenue_node": sales, "price": 100.0, "cap": 150.0},
                    affordances=["levers"], endowment=500_000, objective={"type": "maximize_net", "text": SHOP},
                    brief="A product with a design setting and a promotion lever. Gauge: units sold.",
                    claim={"lever": X, "second": P, "outcome": sales, "taste": T, "fit": fit, "width": w})


# ====================================================================== 8.10.1 hidden sign reversal (level-neutral)
@family("ch_silent_reversal", engine="primitive_graph", category="8.10", item="8.10.1", split="train",
        doc="A lever raises sales until a hidden tick and lowers them afterwards; the switch is level-neutral (at the "
            "middle setting sales are the same before and after), so only an intervention - moving the lever and "
            "watching the response - shows that the old strategy has become harmful.",
        distinct_because="a timed regime change that flips the sign of a linear lever effect around a pivot "
                         "(complementary bias keeps the level at the pivot unchanged), plus a cost-only second lever: "
                         "a sign change invisible in levels, unlike scm_regime's strength switch",
        axes={"horizon": 100, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(CV + "reversal_is_level_neutral",), identify=CV + "identify_reversal", oracle=R + "lever_schedule_oracle",
        reference=CV + "sign_tracker", lever_view=True)
def ch_silent_reversal(rng, d):
    gb = GraphBuilder(rng)
    H = int(d["horizon"])
    X = gb.lever(0.0, 10.0, 5.0, cost=u(rng, 5.0, 15.0))
    Q = gb.lever(0.0, 10.0, 0.0, cost=u(rng, 40.0, 70.0))
    pivot = 5.0
    slope = u(rng, 3.0, 4.5)
    at = float(int(rng.integers(int(0.3 * H), int(0.7 * H))))
    # effect = slope * (x - pivot), times +1 before the switch and -1 after: level-neutral at the pivot
    dev = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": -pivot}, x=[(X, 0, 1.0)])
    sw = gb.prim("regime_change", {"before": slope, "after": -slope, "at": at, "trig": 1e18}, role="regime",
                 x=[(dev, 0, 1.0)], trigger=[])
    side = gb.prim("saturation", {"cap": u(rng, 4.0, 8.0), "half": u(rng, 2.0, 4.0)}, x=[(Q, 0, 1.0)])
    season = gb.prim("hidden_confounding", {"mean": 0.0, "sd": u(rng, 0.8, 1.2), "ar": 0.7}, role="exogenous")
    sales = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, role="outcome",
                    bias={"x": u(rng, 35.0, 45.0)}, x=[(sw, 0, 1.0), (side, 0, 1.0), (season, 0, 2.0)])
    noise = float(d["observation_noise"])
    gb.gauge(sales, [("measurement_noise", {"sd": noise * 3})])
    _distractors(gb, rng, d["n_variables"], noise)
    return gb.build(horizon=H, money={"revenue_node": sales, "price": 100.0, "cap": 200.0}, affordances=["levers"],
                    endowment=500_000, objective={"type": "maximize_net", "text": SHOP},
                    brief="A shop with two spending levers. Gauge: units sold.",
                    claim={"lever": X, "second": Q, "outcome": sales, "regime": sw, "at": at, "pivot": pivot,
                           "slope": slope})


# ====================================================================== 8.10.6 platform rule change (discrete)
@family("ch_format_switch", engine="primitive_graph", category="8.10", item="8.10.6", split="train",
        doc="A marketplace ranks listings by format. Until a hidden tick its rules favour one listing format, "
            "afterwards the other; the change is announced nowhere and the average sales of the two formats are "
            "unchanged. A seller must notice from its own results that the old format has started to lose and "
            "switch, while also pricing well.",
        distinct_because="a discrete decision whose effect sign flips at a hidden tick (timed regime change on a "
                         "decision's value) under heavy demand noise: the changing element is a categorical choice, "
                         "not a lever level, and single readings cannot tell a rule change from noise",
        axes={"horizon": 100, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(CV + "option_effect_flips",), identify=CV + "identify_option_flip", oracle=CV + "oracle_option_switch",
        reference=CV + "probe_and_switch")
def ch_format_switch(rng, d):
    gb = GraphBuilder(rng)
    H = int(d["horizon"])
    D = gb.discrete([1.0, -1.0], cost=[0.0, 0.0], default=0)
    slope = u(rng, 6.0, 9.0)
    at = float(int(rng.integers(int(0.3 * H), int(0.7 * H))))
    sw = gb.prim("regime_change", {"before": slope, "after": -slope, "at": at, "trig": 1e18}, role="regime",
                 x=[(D, 0, 1.0)], trigger=[])
    season = gb.prim("hidden_confounding", {"mean": 0.0, "sd": u(rng, 4.0, 5.0), "ar": 0.3}, role="exogenous")
    units = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, role="outcome",
                    bias={"x": u(rng, 30.0, 40.0)}, x=[(sw, 0, 1.0), (season, 0, 1.0)])
    noise = float(d["observation_noise"])
    gb.gauge(units, [("measurement_noise", {"sd": noise * 3})])
    _distractors(gb, rng, d["n_variables"], noise)
    return gb.build(horizon=H, money={"revenue_node": units, "price": 100.0, "cap": 200.0},
                    affordances=["discrete"], endowment=500_000,
                    objective={"type": "maximize_net", "text": SHOP},
                    brief="A marketplace seller choosing a listing format. Gauge: units sold.",
                    claim={"decision": D, "outcome": units, "regime": sw, "at": at})


# ====================================================================== 8.2.2 competing hypotheses
def _predict_text(heldout) -> str:
    return ("Predict the noise-free reading at these conditions, which lie outside what you can set: "
            f"{heldout}. File predictions with the journal: message to 'journal' with content "
            "{\"predictions\": [{\"set\": <condition>, \"value\": <number>}, ...]}. Only the last filing counts.")


def _hypotheses(rng, d, pair: str, split_note: str):
    """A law that is one of two public candidate hypotheses (which one, the seed decides); both fit the settable
    range similarly but predict very differently outside it."""
    gb = GraphBuilder(rng)
    x1 = gb.lever(0.0, 6.0, 0.0, cost=0.0)
    x2 = gb.lever(0.0, 6.0, 0.0, cost=0.0)
    noise = float(d["observation_noise"])
    h = u(rng, 1.5, 3.0)
    y6 = u(rng, 10.0, 14.0)
    first = bool(rng.random() < 0.5)
    beta = u(rng, 0.6, 1.2)
    if pair == "saturating_vs_power":
        if first:
            f = gb.prim("saturation", {"cap": y6 * (6.0 + h) / 6.0, "half": h}, x=[(x1, 0, 1.0)])
        else:
            f = gb.prim("linear_nonlinear", {"form": "power", "a": y6 / 6.0 ** 0.5, "b": 0.0, "k": 0.5},
                        x=[(x1, 0, 1.0)])
        g = gb.prim("linear_nonlinear", {"form": "linear", "a": beta, "b": 0.0}, x=[(x2, 0, 1.0)])
        y = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, role="outcome",
                    x=[(f, 0, 1.0), (g, 0, 1.0)])
        names = ("saturating", "power")
        held_pts = ((9.0, 1.0), (12.0, 3.0), (15.0, 2.0), (10.0, 5.0), (14.0, 0.5), (11.0, 4.0))
        brief = ("A laboratory apparatus with two settings and one sensor. Two theories compete: the reading "
                 "saturates in the first setting (diminishing returns towards a ceiling), or it keeps growing as a "
                 "power of it. The second setting adds to the reading.")
    else:                                                     # additive_vs_interactive
        if first:
            f = gb.prim("saturation", {"cap": y6 * (6.0 + h) / 6.0, "half": h}, x=[(x1, 0, 1.0)])
            hg = u(rng, 2.0, 4.0)
            g = gb.prim("saturation", {"cap": beta * 6.0 * (6.0 + hg) / 6.0 * 2.0, "half": hg}, x=[(x2, 0, 1.0)])
            y = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, role="outcome",
                        x=[(f, 0, 1.0), (g, 0, 1.0)])
        else:
            f = gb.prim("saturation", {"cap": y6 * (6.0 + h) / 6.0, "half": h}, x=[(x1, 0, 1.0)])
            g = gb.prim("linear_nonlinear", {"form": "linear", "a": beta / 3.0, "b": 0.4}, x=[(x2, 0, 1.0)])
            y = gb.prim("multiplicative", {"k": 1.0}, role="outcome", a=[(f, 0, 1.0)], b=[(g, 0, 1.0)])
        names = ("additive", "interactive")
        held_pts = ((3.0, 14.0), (5.0, 20.0), (1.0, 16.0), (6.0, 24.0), (2.0, 18.0), (4.0, 22.0))
        brief = ("A laboratory apparatus with two settings and one sensor. Two theories compete: the settings act "
                 "separately and their effects add up, or the second setting scales the effect of the first.")
    gy = gb.gauge(y, [("measurement_noise", {"sd": noise})])
    heldout = [{gb.name_of(x1): a, gb.name_of(x2): b} for a, b in held_pts]
    return gb.build(horizon=int(d["horizon"]), affordances=["levers", "message"],
                    objective={"type": "predict_heldout", "heldout": heldout, "gauge": gy,
                               "text": _predict_text(heldout)},
                    journal=True, brief=brief,
                    claim={"levers": [x1, x2], "outcome": y, "law": names[0] if first else names[1],
                           "hypotheses": list(names), "pair": pair})


@family("sci_competing_hypotheses", engine="primitive_graph", category="8.2", item="8.2.2", split="train",
        doc="Two public candidate laws - saturating or power growth in the first setting - fit the settable range "
            "about equally well but predict very differently outside it; the world obeys one of them. Design "
            "discriminating experiments, decide, and predict held-out conditions.",
        distinct_because="the law is one of two public hypotheses (saturation or power) with an additive second "
                         "factor; prediction at held-out conditions where only the true hypothesis extrapolates",
        axes={"horizon": 40, "observation_noise": 0.3}, objective="predict_heldout",
        claims=(CV + "hypotheses_diverge_out_of_range",), identify=CV + "identify_hypothesis",
        reference=CV + "test_hypotheses_and_predict")
def sci_competing_hypotheses(rng, d):
    return _hypotheses(rng, d, "saturating_vs_power", "train")


@family("sci_interaction_hypotheses", engine="primitive_graph", category="8.2", item="8.2.2", split="val",
        doc="Two public candidate laws - the settings' effects add up, or the second scales the first - agree "
            "roughly inside the settable range and diverge outside it; the world obeys one of them.",
        distinct_because="competing hypotheses about the combination of two factors (additive collider vs "
                         "multiplicative interaction), not about the shape of one response",
        axes={"horizon": 40, "observation_noise": 0.3}, objective="predict_heldout",
        claims=(CV + "hypotheses_diverge_out_of_range",), identify=CV + "identify_hypothesis",
        reference=CV + "test_hypotheses_and_predict")
def sci_interaction_hypotheses(rng, d):
    return _hypotheses(rng, d, "additive_vs_interactive", "val")


# ====================================================================== 8.2.3 hidden conservation
@family("sci_hidden_conservation", engine="primitive_graph", category="8.2", item="8.2.3", split="train",
        doc="Material circulates through three tanks: two are gauged, the third is hidden and returns material at "
            "a hidden rate. The total is conserved but unknown. Set the two transfer rates, watch the gauged "
            "levels settle, infer the hidden total and return rate, and predict where a gauged level would settle "
            "at transfer rates outside the settable range.",
        distinct_because="a closed cycle of three stocks with rate-controlled flows (multiplicative lagged flows, "
                         "conserved total, one stock unobserved); the law to find is a conservation relation, not "
                         "a response curve",
        axes={"horizon": 80, "observation_noise": 0.3}, objective="predict_heldout",
        claims=(CV + "total_is_conserved",), identify=CV + "identify_total", reference=CV + "infer_conservation")
def sci_hidden_conservation(rng, d):
    gb = GraphBuilder(rng)
    r1 = gb.lever(0.05, 0.5, 0.2, cost=0.0)
    r2 = gb.lever(0.05, 0.5, 0.2, cost=0.0)
    M = u(rng, 60.0, 140.0)
    rho = u(rng, 0.08, 0.3)
    a0 = u(rng, 0.3, 0.5)
    A = gb.prim("stock", {"init": a0 * M, "cap": 1e9}, role="tank_a")
    B = gb.prim("stock", {"init": (1 - a0) * M * 0.5, "cap": 1e9}, role="tank_b")
    Cn = gb.prim("stock", {"init": (1 - a0) * M * 0.5, "cap": 1e9}, role="tank_c")
    fab = gb.prim("multiplicative", {"k": 1.0}, a=[(r1, 0, 1.0)], b=[(A, 1, 1.0)])
    fbc = gb.prim("multiplicative", {"k": 1.0}, a=[(r2, 0, 1.0)], b=[(B, 1, 1.0)])
    fca = gb.prim("linear_nonlinear", {"form": "linear", "a": rho, "b": 0.0}, x=[(Cn, 1, 1.0)])
    for nid, inflow, outflow in ((A, fca, fab), (B, fab, fbc), (Cn, fbc, fca)):
        gb.nodes[int(nid[1:])]["ports"] = {"inflow": [[inflow, 0, 1.0]], "outflow": [[outflow, 0, 1.0]]}
    noise = float(d["observation_noise"])
    ga = gb.gauge(A, [("measurement_noise", {"sd": noise})])
    gbb = gb.gauge(B, [("measurement_noise", {"sd": noise})])
    held = [{gb.name_of(r1): a, gb.name_of(r2): b} for a, b in
            ((0.8, 0.1), (0.02, 0.4), (0.7, 0.7), (1.0, 0.3), (0.03, 0.03), (0.9, 0.05))]
    text = (f"Predict where the reading {gbb} would settle if the two transfer rates were held at these values "
            f"(outside what you can set): {held}. " + _predict_text(held).split(". ", 1)[1])
    return gb.build(horizon=int(d["horizon"]), affordances=["levers", "message"],
                    objective={"type": "predict_heldout", "heldout": held, "gauge": gbb, "text": text,
                               "settle_ticks": 400},
                    journal=True,
                    brief=(f"Three tanks in a loop: material flows from the first tank (gauge {ga}) to the second "
                           f"(gauge {gbb}) at the first rate, from the second to a third, ungauged tank at the "
                           "second rate, and back from the third at a fixed rate nobody has measured."),
                    claim={"levers": [r1, r2], "tanks": [A, B, Cn], "total": M, "rho": rho, "gauges": [ga, gbb]})


# ====================================================================== 8.2.5 / 8.1 selection-biased measurement
@family("sci_selection_bias", engine="primitive_graph", category="8.2", item="8.2.5", split="val",
        doc="An outcome responds linearly to two settings, but it is measured through a survey in which only units "
            "above a hidden cut report; the reading is the mean of the reporters, biased upward and flattened "
            "against changes in the settings. Predict the true mean outcome at settings outside the range.",
        distinct_because="a selection-effect observation channel (truncation at a hidden cut) on a linear law: "
                         "the measured response is a biased, flattened image of the true one; the task is to "
                         "invert the channel, not to find a curved law",
        axes={"horizon": 40, "observation_noise": 0.3}, objective="predict_heldout",
        claims=(CV + "selection_flattens",), identify=CV + "identify_selection_slope",
        reference=CV + "correct_selection_and_predict")
def sci_selection_bias(rng, d):
    gb = GraphBuilder(rng)
    x1 = gb.lever(0.0, 6.0, 0.0, cost=0.0)
    x2 = gb.lever(0.0, 6.0, 0.0, cost=0.0)
    b1, b2 = u(rng, 0.8, 1.6), u(rng, -0.8, 0.8)
    a = u(rng, -2.0, 2.0)
    y = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": a}, role="outcome",
                x=[(x1, 0, b1), (x2, 0, b2)])
    sd = u(rng, 2.0, 3.0)
    cut = u(rng, 1.0, 3.0)
    gy = gb.gauge(y, [("selection_effect", {"n": 64, "sd": sd, "c": cut}),
                      ("measurement_noise", {"sd": float(d["observation_noise"]) * 0.3})])
    held = [{gb.name_of(x1): p, gb.name_of(x2): q} for p, q in
            ((9.0, 1.0), (-3.0, 2.0), (12.0, 4.0), (-5.0, 5.0), (8.0, -2.0), (10.0, 8.0))]
    text = ("Predict the true mean outcome of all units (not only those who report) at these settings, outside "
            f"what you can set: {held}. " + _predict_text(held).split(". ", 1)[1])
    return gb.build(horizon=int(d["horizon"]), affordances=["levers", "message"],
                    objective={"type": "predict_heldout", "heldout": held, "gauge": gy, "text": text},
                    journal=True,
                    brief=("A survey of 64 units under two settings. Only units whose outcome clears an "
                           f"undisclosed bar report; the reading {gy} is the mean of those who report. Units' "
                           f"outcomes scatter around the population mean with a spread of about {sd:.1f}."),
                    claim={"levers": [x1, x2], "outcome": y, "cut": cut, "sd": sd, "coef": [a, b1, b2]})


# ====================================================================== 8.2.4 delayed transformation
@family("sci_slow_response", engine="primitive_graph", category="8.2", item="8.2.4", split="val",
        doc="A transformation responds to two settings only after a hidden dead time and then settles slowly "
            "(first-order lag with a hidden time constant): readings taken soon after a change are far from where "
            "the system settles. Predict the settled reading at settings outside the range.",
        distinct_because="a pipeline delay into a decaying (first-order lag) stock in front of a saturating law: "
                         "the measured response is a transient; the task is to separate dynamics from the steady "
                         "law, unlike instantaneous laws",
        axes={"horizon": 120, "observation_noise": 0.3}, objective="predict_heldout",
        claims=(CV + "response_is_delayed_and_lagged",), identify=CV + "identify_dead_time",
        reference=RV + "discover_settled")
def sci_slow_response(rng, d):
    gb = GraphBuilder(rng)
    x1 = gb.lever(0.0, 6.0, 0.0, cost=0.0)
    x2 = gb.lever(0.0, 6.0, 0.0, cost=0.0)
    h = u(rng, 1.5, 3.0)
    law1 = gb.prim("saturation", {"cap": u(rng, 10.0, 16.0), "half": h}, x=[(x1, 0, 1.0)])
    law = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, x=[(law1, 0, 1.0), (x2, 0, u(rng, 0.5, 1.2))])
    lag = int(rng.integers(2, 5))
    dl = gb.prim("delay", {"lag": lag, "init": 0.0}, x=[(law, 0, 1.0)])
    rate = u(rng, 0.12, 0.25)
    st = gb.prim("decay", {"rate": rate, "init": 0.0}, role="outcome", x=[(dl, 0, rate)])
    gy = gb.gauge(st, [("measurement_noise", {"sd": float(d["observation_noise"])})])
    held = [{gb.name_of(x1): p, gb.name_of(x2): q} for p, q in
            ((7.0, 1.0), (8.0, 3.0), (9.0, 6.0), (10.0, 2.0), (7.5, 5.0), (8.5, 0.0))]
    text = ("Predict where the reading would settle if these settings (outside what you can set) were held: "
            f"{held}. " + _predict_text(held).split(". ", 1)[1])
    return gb.build(horizon=int(d["horizon"]), affordances=["levers", "message"],
                    objective={"type": "predict_heldout", "heldout": held, "gauge": gy, "text": text,
                               "settle_ticks": 300},
                    journal=True, brief="A process with two settings and one sensor; it does not answer at once.",
                    claim={"levers": [x1, x2], "outcome": st, "lag": lag, "rate": rate, "half": h})


# ====================================================================== 8.1.1 confounded history, causal prediction
@family("hc_causal_prediction", engine="primitive_graph", category="8.1", item="8.1.1", split="val",
        doc="The previous operator's records show the spending lever and sales rising and falling together, "
            "because a hidden demand cycle drove both; intervening shows that spending's own effect saturates and "
            "is far smaller. Predict sales at spending levels outside the settable range, as an intervention, not "
            "as the records suggest.",
        distinct_because="hidden confounder with a natural-regime lever and a saturating interventional effect, "
                         "with the objective of predicting the effect of unseen interventions (correlation versus "
                         "causation scored on held-out conditions)",
        axes={"horizon": 80, "observation_noise": 0.3}, objective="predict_heldout",
        claims=(CV + "records_overstate_the_effect",), identify=CV + "identify_effect_size",
        reference=RV + "discover_averaged")
def hc_causal_prediction(rng, d):
    gb = GraphBuilder(rng)
    U = gb.prim("hidden_confounding", {"mean": 0.0, "sd": u(rng, 0.18, 0.22), "ar": 0.8}, role="confounder")
    X = gb.lever(0.0, 6.0, 3.0, cost=0.0, natural={"x": [(U, 0, 3.0)], "bias": 3.0, "sd": 0.2})
    Z = gb.lever(0.0, 6.0, 0.0, cost=0.0)
    eff = gb.prim("saturation", {"cap": u(rng, 8.0, 12.0), "half": u(rng, 1.5, 3.0)}, x=[(X, 0, 1.0)])
    sales = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": u(rng, 20.0, 30.0)}, role="outcome",
                    x=[(U, 0, u(rng, 4.0, 6.0)), (eff, 0, 1.0), (Z, 0, u(rng, 1.0, 2.0))])
    gy = gb.gauge(sales, [("measurement_noise", {"sd": float(d["observation_noise"]) * 3})])
    held = [{gb.name_of(X): p, gb.name_of(Z): q} for p, q in
            ((7.0, 1.0), (8.0, 3.0), (9.0, 6.0), (10.0, 2.0), (7.5, 5.0), (8.5, 0.0))]
    text = ("Predict mean units sold if spending and the second lever were set to these values (outside what you "
            f"can set): {held}. " + _predict_text(held).split(". ", 1)[1])
    return gb.build(horizon=int(d["horizon"]), affordances=["levers", "message"], warmup=40,
                    objective={"type": "predict_heldout", "heldout": held, "gauge": gy, "text": text},
                    journal=True,
                    brief="A shop: a spending lever and a second lever; gauge: units sold. You have the previous "
                          "operator's records.",
                    claim={"levers": [X, Z], "outcome": sales, "confounder": U, "effect": eff})


# ====================================================================== 8.6.6 wear-triggered reversal (endogenous)
@family("dl_wear_reversal", engine="primitive_graph", category="8.6", item="8.6.6", split="train",
        doc="Driving a machine hard raises output, but every tick at a high setting adds to hidden wear; once the "
            "wear passes a hidden level the high setting starts to damage output instead, for good. When that "
            "happens depends on how hard the machine was driven - the participant's own past - so no clock and no "
            "reading announces it; the middle setting's output does not change.",
        distinct_because="a regime change triggered endogenously by an accumulating stock of the participant's own "
                         "past use (path-dependent, permanent), flipping the sign of a lever effect around a pivot; "
                         "unlike timed switches the switch time moves with the policy",
        axes={"horizon": 100, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(CV + "wear_triggers_reversal",), identify=CV + "identify_wear_reversal",
        oracle=R + "lever_schedule_oracle", reference=CV + "sign_tracker", lever_view=True)
def dl_wear_reversal(rng, d):
    gb = GraphBuilder(rng)
    H = int(d["horizon"])
    X = gb.lever(0.0, 10.0, 5.0, cost=u(rng, 5.0, 15.0))
    Q = gb.lever(0.0, 10.0, 0.0, cost=u(rng, 40.0, 70.0))
    pivot = 5.0
    slope = u(rng, 3.0, 4.5)
    over = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": -pivot}, x=[(X, 0, 1.0)])
    use = gb.prim("saturation", {"cap": 1.0, "half": 1e-6}, x=[(over, 0, 1.0)])          # 1 while above the pivot
    wear = gb.prim("stock", {"init": 0.0, "cap": 1e9}, role="wear", inflow=[(use, 0, 1.0)], outflow=[])
    level = float(int(rng.integers(int(0.25 * H), int(0.55 * H))))
    sw = gb.prim("regime_change", {"before": slope, "after": -slope, "at": -1.0, "trig": level}, role="regime",
                 x=[(over, 0, 1.0)], trigger=[(wear, 1, 1.0)])
    side = gb.prim("saturation", {"cap": u(rng, 4.0, 8.0), "half": u(rng, 2.0, 4.0)}, x=[(Q, 0, 1.0)])
    season = gb.prim("hidden_confounding", {"mean": 0.0, "sd": u(rng, 0.8, 1.2), "ar": 0.7}, role="exogenous")
    sales = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, role="outcome",
                    bias={"x": u(rng, 35.0, 45.0)}, x=[(sw, 0, 1.0), (side, 0, 1.0), (season, 0, 2.0)])
    noise = float(d["observation_noise"])
    gb.gauge(sales, [("measurement_noise", {"sd": noise * 3})])
    _distractors(gb, rng, d["n_variables"], noise)
    return gb.build(horizon=H, money={"revenue_node": sales, "price": 100.0, "cap": 200.0}, affordances=["levers"],
                    endowment=500_000, objective={"type": "maximize_net", "text": SHOP},
                    brief="A machine with a drive setting and a second, costly lever. Gauge: output sold.",
                    claim={"lever": X, "second": Q, "outcome": sales, "regime": sw, "wear": wear, "level": level,
                           "pivot": pivot})


# ====================================================================== 8.10.5 technological transition (interaction flip)
@family("ch_complement_to_substitute", engine="primitive_graph", category="8.10", item="8.10.5", split="val",
        doc="Two inputs were complements: output needs both, so raising the second input pays only next to the "
            "first. At a hidden tick a new technology makes the second input a substitute for the first - with the "
            "first already high, the second now crowds out the new process and lowers output. The best use of the "
            "second input reverses although no reading announces the transition.",
        distinct_because="a timed regime switch between a complementarity (CES) path and a substitution path for "
                         "the same two inputs: what changes is how two inputs combine (an interaction), not a "
                         "single lever's sign",
        axes={"horizon": 100, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(CV + "interaction_flips",), identify=CV + "identify_interaction_flip",
        oracle=R + "lever_schedule_oracle", reference=CV + "second_input_tracker", lever_view=True)
def ch_complement_to_substitute(rng, d):
    gb = GraphBuilder(rng)
    H = int(d["horizon"])
    A = gb.lever(0.0, 10.0, 10.0, cost=u(rng, 5.0, 10.0))
    Bv = gb.lever(0.0, 10.0, 5.0, cost=u(rng, 60.0, 90.0))
    at = float(int(rng.integers(int(0.3 * H), int(0.7 * H))))
    comp = gb.prim("complementarity", {"scale": u(rng, 5.0, 6.0), "rho": -2.0, "w": 0.5}, a=[(A, 0, 1.0)],
                   b=[(Bv, 0, 1.0)])
    # after the transition the old second input interferes with the new process (negative substitution weight):
    # it now crowds out output instead of completing it
    subs = gb.prim("substitution", {"cap": u(rng, 55.0, 65.0), "k": u(rng, 4.0, 6.0), "w": -u(rng, 0.5, 0.7)},
                   a=[(A, 0, 1.0)], b=[(Bv, 0, 1.0)])
    old = gb.prim("regime_change", {"before": 1.0, "after": 0.0, "at": at, "trig": 1e18}, role="old_regime",
                  x=[(comp, 0, 1.0)], trigger=[])
    new = gb.prim("regime_change", {"before": 0.0, "after": 1.0, "at": at, "trig": 1e18}, role="new_regime",
                  x=[(subs, 0, 1.0)], trigger=[])
    season = gb.prim("hidden_confounding", {"mean": 0.0, "sd": u(rng, 0.8, 1.2), "ar": 0.7}, role="exogenous")
    sales = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, role="outcome",
                    bias={"x": u(rng, 10.0, 15.0)}, x=[(old, 0, 1.0), (new, 0, 1.0), (season, 0, 2.0)])
    noise = float(d["observation_noise"])
    gb.gauge(sales, [("measurement_noise", {"sd": noise * 3})])
    _distractors(gb, rng, d["n_variables"], noise)
    return gb.build(horizon=H, money={"revenue_node": sales, "price": 100.0, "cap": 200.0}, affordances=["levers"],
                    endowment=500_000, objective={"type": "maximize_net", "text": SHOP},
                    brief="A workshop with two inputs to buy each tick. Gauge: output sold.",
                    claim={"lever": Bv, "first": A, "outcome": sales, "at": at})


# ====================================================================== 8.10.1 delayed sign reversal
@family("ch_lagged_reversal", engine="primitive_graph", category="8.10", item="8.10.1", split="val",
        doc="A lever's effect on sales arrives only after a hidden delay of a few ticks, and at a hidden tick its "
            "sign reverses (level-neutral at the middle setting). A participant has to probe long enough for the "
            "delayed response to show before it can tell whether the old strategy has turned harmful.",
        distinct_because="a pipeline delay in front of a timed sign reversal of a linear effect: credit assignment "
                         "across a lag combined with belief revision (unlike the immediate reversal of "
                         "ch_silent_reversal)",
        axes={"horizon": 120, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(CV + "lagged_reversal",), identify=CV + "identify_lagged_reversal", oracle=R + "lever_schedule_oracle",
        reference=CV + "lagged_sign_tracker", lever_view=True)
def ch_lagged_reversal(rng, d):
    gb = GraphBuilder(rng)
    H = int(d["horizon"])
    X = gb.lever(0.0, 10.0, 5.0, cost=u(rng, 5.0, 15.0))
    Q = gb.lever(0.0, 10.0, 0.0, cost=u(rng, 40.0, 70.0))
    pivot = 5.0
    slope = u(rng, 3.0, 4.5)
    lag = int(rng.integers(2, 4))
    at = float(int(rng.integers(int(0.3 * H), int(0.7 * H))))
    dev = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": -pivot}, x=[(X, 0, 1.0)])
    dl = gb.prim("delay", {"lag": lag, "init": 0.0}, x=[(dev, 0, 1.0)])
    sw = gb.prim("regime_change", {"before": slope, "after": -slope, "at": at, "trig": 1e18}, role="regime",
                 x=[(dl, 0, 1.0)], trigger=[])
    side = gb.prim("saturation", {"cap": u(rng, 4.0, 8.0), "half": u(rng, 2.0, 4.0)}, x=[(Q, 0, 1.0)])
    season = gb.prim("hidden_confounding", {"mean": 0.0, "sd": u(rng, 0.8, 1.2), "ar": 0.7}, role="exogenous")
    sales = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, role="outcome",
                    bias={"x": u(rng, 35.0, 45.0)}, x=[(sw, 0, 1.0), (side, 0, 1.0), (season, 0, 2.0)])
    noise = float(d["observation_noise"])
    gb.gauge(sales, [("measurement_noise", {"sd": noise * 3})])
    _distractors(gb, rng, d["n_variables"], noise)
    return gb.build(horizon=H, money={"revenue_node": sales, "price": 100.0, "cap": 200.0}, affordances=["levers"],
                    endowment=500_000, objective={"type": "maximize_net", "text": SHOP},
                    brief="A shop with two spending levers; spending takes a few ticks to show. Gauge: units sold.",
                    claim={"lever": X, "second": Q, "outcome": sales, "regime": sw, "at": at + lag, "pivot": pivot,
                           "lag": lag, "slope": slope})


# ====================================================================== 8.1.3 dose response beyond the range (prediction)
@family("sci_dose_response", engine="primitive_graph", category="8.1", item="8.1.3", split="train",
        doc="A dose rises in benefit and then turns harmful past a hidden peak that lies near the top of the "
            "settable range, so the decline beyond it is barely visible inside the range; a second factor adds. "
            "Predict the response at doses beyond the range (an intervention nobody has run).",
        distinct_because="a non-monotonic (rise-and-fall) dose response with an additive linear factor, judged by "
                         "prediction beyond the peak: extrapolating the rising part, as a monotone fit does, gets "
                         "the sign of the effect of more dose wrong",
        axes={"horizon": 40, "observation_noise": 0.3}, objective="predict_heldout",
        claims=(CV + "dose_turns_harmful",), identify=CV + "identify_peak", reference=RV + "discover_and_predict")
def sci_dose_response(rng, d):
    gb = GraphBuilder(rng)
    x1 = gb.lever(0.0, 6.0, 0.0, cost=0.0)
    x2 = gb.lever(0.0, 6.0, 0.0, cost=0.0)
    peak = u(rng, 3.5, 5.0)
    f = gb.prim("non_monotonic", {"peak": peak, "height": u(rng, 10.0, 16.0)}, x=[(x1, 0, 1.0)])
    y = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": u(rng, 2.0, 5.0)}, role="outcome",
                x=[(f, 0, 1.0), (x2, 0, u(rng, 0.4, 1.0))])
    gy = gb.gauge(y, [("measurement_noise", {"sd": float(d["observation_noise"])})])
    held = [{gb.name_of(x1): p, gb.name_of(x2): q} for p, q in
            ((8.0, 1.0), (10.0, 3.0), (12.0, 5.0), (9.0, 0.0), (11.0, 2.0), (14.0, 4.0))]
    return gb.build(horizon=int(d["horizon"]), affordances=["levers", "message"],
                    objective={"type": "predict_heldout", "heldout": held, "gauge": gy, "text": _predict_text(held)},
                    journal=True, brief="A treatment trial: a dose and a second factor; one response sensor.",
                    claim={"levers": [x1, x2], "outcome": y, "peak": peak})
