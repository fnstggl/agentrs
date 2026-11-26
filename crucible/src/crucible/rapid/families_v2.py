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
R = "crucible.rapid.refs_h3:"


def u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def _distractors(gb: GraphBuilder, rng, n: int, noise: float) -> None:
    for _ in range(int(n)):
        d = gb.prim("hidden_confounding", {"mean": u(rng, 20, 60), "sd": u(rng, 1, 3), "ar": 0.8}, role="distractor")
        gb.gauge(d, [("measurement_noise", {"sd": noise * 5})])


SHOP = "Maximise your net income over the horizon."


# ====================================================================== 8.10.3 wandering preferences
@family("ch_wandering_taste", engine="primitive_graph", category="8.10", item="8.10.3", split="train",
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
                         "decision's value), next to a priced lever with an interior optimum; the changing element "
                         "is a categorical choice, not a lever level",
        axes={"horizon": 100, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(CV + "option_effect_flips",), identify=CV + "identify_option_flip", oracle=R + "lever_schedule_oracle",
        reference=CV + "probe_and_switch")
def ch_format_switch(rng, d):
    gb = GraphBuilder(rng)
    H = int(d["horizon"])
    D = gb.discrete([1.0, -1.0], cost=[0.0, 0.0], default=0)
    Pr = gb.lever(0.0, 10.0, 5.0, cost=0.0)
    slope = u(rng, 12.0, 16.0)
    at = float(int(rng.integers(int(0.3 * H), int(0.7 * H))))
    sw = gb.prim("regime_change", {"before": slope, "after": -slope, "at": at, "trig": 1e18}, role="regime",
                 x=[(D, 0, 1.0)], trigger=[])
    # price: demand falls with price (logistic), revenue per unit rises with it - an interior optimum
    conv = gb.prim("linear_nonlinear", {"form": "logistic", "a": u(rng, 50.0, 60.0), "b": 0.0,
                                        "c": u(rng, 0.6, 0.9), "m": u(rng, 4.5, 6.0)}, x=[(Pr, 0, -1.0)],
                   bias={"x": 10.0})
    season = gb.prim("hidden_confounding", {"mean": 0.0, "sd": u(rng, 0.8, 1.2), "ar": 0.7}, role="exogenous")
    units = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, role="outcome",
                    bias={"x": u(rng, 15.0, 20.0)}, x=[(sw, 0, 1.0), (conv, 0, 1.0), (season, 0, 2.0)])
    rev = gb.prim("multiplicative", {"k": 1.0}, role="revenue", a=[(units, 0, 1.0)], b=[(Pr, 0, 1.0)],
                  bias={"b": 1.0})
    noise = float(d["observation_noise"])
    gb.gauge(units, [("measurement_noise", {"sd": noise * 3})])
    _distractors(gb, rng, d["n_variables"], noise)
    return gb.build(horizon=H, money={"revenue_node": rev, "price": 10.0, "cap": 2000.0},
                    affordances=["levers", "discrete"], endowment=500_000,
                    objective={"type": "maximize_net", "text": SHOP},
                    brief="A marketplace seller: a listing format to choose and a price to set. Gauge: units sold.",
                    claim={"decision": D, "lever": Pr, "outcome": units, "regime": sw, "at": at})


# ====================================================================== 8.1.6 state-dependent causal structure
@family("hc_hidden_state_sign", engine="primitive_graph", category="8.1", item="8.1.6", split="train",
        doc="Whether a lever helps or hurts depends on a hidden, slowly wandering market state: above a hidden "
            "set point the same push lowers sales, below it it raises them. A footfall gauge tracks the state but "
            "is also raised by the lever itself (a confounded proxy), so neither levels nor the passive "
            "correlation tell which way to push; only repeated small interventions do.",
        distinct_because="a hidden persistent AR(1) state enters a state-dependent action effect "
                         "(gain * (x - pivot) * (setpoint - state)) and, with the lever, a proxy gauge: the active "
                         "causal sign follows a hidden state; no timed switch, no stock",
        axes={"horizon": 160, "observation_noise": 0.3, "n_variables": 0}, objective="maximize_net",
        claims=(CV + "sign_follows_hidden_state",), identify=CV + "identify_state_sign",
        oracle=R + "lever_schedule_oracle", reference=CV + "sign_tracker", lever_view=True)
def hc_hidden_state_sign(rng, d):
    gb = GraphBuilder(rng)
    X = gb.lever(0.0, 10.0, 5.0, cost=u(rng, 5.0, 10.0))
    Q = gb.lever(0.0, 10.0, 0.0, cost=u(rng, 40.0, 70.0))
    sp = u(rng, -0.3, 0.3)
    S = gb.prim("hidden_confounding", {"mean": sp, "sd": u(rng, 0.3, 0.36), "ar": 0.94}, role="state")
    dev = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": -5.0}, x=[(X, 0, 1.0)])
    eff = gb.prim("state_dependent_effect", {"gain": u(rng, 4.5, 6.0), "setpoint": sp}, role="effect",
                  a=[(dev, 0, 1.0)], s=[(S, 0, 1.0)])
    side = gb.prim("saturation", {"cap": u(rng, 4.0, 7.0), "half": u(rng, 2.0, 4.0)}, x=[(Q, 0, 1.0)])
    sales = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, role="outcome",
                    bias={"x": u(rng, 35.0, 45.0)}, x=[(eff, 0, 1.0), (side, 0, 1.0), (S, 0, -6.0)])
    proxy = gb.prim("linear_nonlinear", {"form": "linear", "a": 1.0, "b": 0.0}, role="proxy",
                    bias={"x": 30.0}, x=[(S, 0, 8.0), (X, 0, 0.8)])
    noise = float(d["observation_noise"])
    gb.gauge(sales, [("measurement_noise", {"sd": noise * 3})])
    gb.gauge(proxy, [("measurement_noise", {"sd": noise * 6})])
    _distractors(gb, rng, d["n_variables"], noise)
    return gb.build(horizon=int(d["horizon"]), money={"revenue_node": sales, "price": 100.0, "cap": 200.0},
                    affordances=["levers"], endowment=500_000, objective={"type": "maximize_net", "text": SHOP},
                    brief="A shop with two spending levers. Gauges: units sold and a footfall count.",
                    claim={"lever": X, "second": Q, "outcome": sales, "state": S, "setpoint": sp, "effect": eff})


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
