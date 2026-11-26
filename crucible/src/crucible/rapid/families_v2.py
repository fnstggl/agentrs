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
