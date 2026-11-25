"""Characteristic-behaviour experiments, one per primitive. Each runs the real implementation and measures the
behaviour that makes the primitive what it is (hysteresis depends on the path, confounding makes a correlation that
an intervention removes, a delay shifts an impulse by exactly its lag, ...). `CHECKS[name]()` returns
`{"ok": bool, ...measured}`; the gate test for Part 7.1 and the unit tests call them."""
from __future__ import annotations

import numpy as np

from .primitives import cascade, get, ration, ring, run


def _grid(lo, hi, n=41):
    return np.linspace(lo, hi, n)


def linear_nonlinear() -> dict:
    x = _grid(-2, 2)
    lin, _ = run("linear_nonlinear", {"form": "linear", "a": 1.5, "b": 0.3}, {"x": x})
    quad, _ = run("linear_nonlinear", {"form": "quadratic", "a": 1.0, "b": 0.0, "c": 0.5}, {"x": x})
    logi, _ = run("linear_nonlinear", {"form": "logistic", "a": 2.0, "b": 0.0, "c": 3.0, "m": 0.0}, {"x": x})
    d2_lin = float(np.abs(np.diff(lin, 2)).max())
    d2_quad = np.diff(quad, 2)
    return {"ok": d2_lin < 1e-9 and float(d2_quad.min()) > 0 and np.allclose(d2_quad, d2_quad[0])
            and bool(np.all(np.diff(logi) > 0)) and 0 < logi.min() and logi.max() < 2.0,
            "linear_max_second_difference": d2_lin, "quadratic_second_difference": float(d2_quad.mean()),
            "logistic_range": [float(logi.min()), float(logi.max())]}


def threshold() -> dict:
    x = _grid(0, 2)
    y, _ = run("threshold", {"theta": 1.0, "lo": 0.0, "hi": 5.0}, {"x": x})
    below, above = y[x < 1.0], y[x >= 1.0]
    return {"ok": bool(np.all(below == 0.0) and np.all(above == 5.0)), "jump": float(above.min() - below.max())}


def saturation() -> dict:
    x = _grid(0, 20, 81)
    y, _ = run("saturation", {"cap": 10.0, "half": 2.0}, {"x": x})
    inc = np.diff(y)
    half, _ = run("saturation", {"cap": 10.0, "half": 2.0}, {"x": [2.0]})
    return {"ok": bool(np.all(inc > 0) and np.all(np.diff(inc) < 0) and y.max() < 10.0 and abs(half[0] - 5) < 1e-12),
            "max": float(y.max()), "first_increment": float(inc[0]), "last_increment": float(inc[-1])}


def non_monotonic() -> dict:
    x = _grid(0, 6, 121)
    y, _ = run("non_monotonic", {"peak": 2.0, "height": 3.0}, {"x": x})
    i = int(np.argmax(y))
    return {"ok": abs(x[i] - 2.0) < 0.06 and abs(y[i] - 3.0) < 1e-6 and bool(np.all(np.diff(y[i:]) < 0))
            and bool(np.all(np.diff(y[:i + 1]) > 0)), "argmax": float(x[i]), "max": float(y[i])}


def complementarity() -> dict:
    p = {"scale": 1.0, "rho": -2.0, "w": 0.5}
    f = lambda a, b: float(run("complementarity", p, {"a": [a], "b": [b]})[0][0])  # noqa: E731
    alone = f(10.0, 0.0)
    cross = f(2.0, 2.0) - f(1.0, 2.0) - f(2.0, 1.0) + f(1.0, 1.0)
    return {"ok": alone < 1e-3 and cross > 0 and f(2, 2) > 1.9, "one_input_alone": alone, "cross_partial": cross}


def substitution() -> dict:
    p = {"cap": 10.0, "k": 4.0, "w": 2.0}
    f = lambda a, b: float(run("substitution", p, {"a": [a], "b": [b]})[0][0])  # noqa: E731
    gain_low_b, gain_high_b = f(1, 0) - f(0, 0), f(1, 3) - f(0, 3)
    return {"ok": gain_high_b < gain_low_b and abs(f(2, 0) - f(0, 1)) < 1e-12,
            "gain_of_a_without_b": gain_low_b, "gain_of_a_with_b": gain_high_b}


def multiplicative() -> dict:
    p = {"k": 0.7}
    f = lambda a, b: float(run("multiplicative", p, {"a": [a], "b": [b]})[0][0])  # noqa: E731
    return {"ok": abs(f(4, 3) - 2 * f(2, 3)) < 1e-12 and f(5, 0) == 0.0 and abs(
        (f(2, 2) - f(1, 2) - f(2, 1) + f(1, 1)) - 0.7) < 1e-12, "doubling_ratio": f(4, 3) / f(2, 3)}


def hidden_confounding() -> dict:
    """U (latent) drives X under the passive regime and drives Y; X has no effect on Y. Passive data show a strong
    X-Y correlation; when X is set at random (an intervention), the slope of Y on X vanishes."""
    T, B = 400, 1
    pu = {"mean": 0.0, "sd": 1.0, "ar": 0.8}
    rng = np.random.default_rng(3)
    u = get("hidden_confounding")
    s = u.init(pu, B)
    us = []
    for t in range(T):
        v, s = u.apply(pu, {}, s, rng.standard_normal((B, 1)), np.zeros((B, 0)), t)
        us.append(v[0])
    us = np.array(us)
    x_pass = us + 0.3 * rng.standard_normal(T)
    y_of = lambda x: run("linear_nonlinear", {"form": "linear", "a": 2.0, "b": 0.0}, {"x": us})[0] \
        + 0.0 * x + 0.3 * rng.standard_normal(T)  # noqa: E731
    y_pass = y_of(x_pass)
    x_do = rng.uniform(-2, 2, T)
    y_do = y_of(x_do)
    corr = float(np.corrcoef(x_pass, y_pass)[0, 1])
    slope_pass = float(np.polyfit(x_pass, y_pass, 1)[0])
    slope_do = float(np.polyfit(x_do, y_do, 1)[0])
    return {"ok": corr > 0.8 and slope_pass > 1.0 and abs(slope_do) < 0.25, "passive_correlation": corr,
            "passive_slope": slope_pass, "interventional_slope": slope_do}


def delay() -> dict:
    x = np.zeros((20, 1))
    x[3, 0] = 1.0
    y, _ = run("delay", {"lag": 5, "init": 0.0}, {"x": x}, ticks=20)
    return {"ok": int(np.argmax(y[:, 0])) == 8 and float(y.sum()) == 1.0, "impulse_at": 3,
            "response_at": int(np.argmax(y[:, 0]))}


def stock() -> dict:
    T = 30
    rng = np.random.default_rng(1)
    inf = rng.uniform(0, 3, (T, 1))
    out = rng.uniform(0, 2, (T, 1))
    pr = get("stock")
    p = {"init": 1.0, "cap": 25.0}
    s = pr.init(p, 1)
    bal = []
    for t in range(T):
        lvl0 = s["level"].copy()
        lvl, s = pr.apply(p, {"inflow": inf[t], "outflow": out[t]}, s, np.zeros((1, 0)), np.zeros((1, 0)), t)
        bal.append(float(lvl[0] - lvl0[0] - (inf[t, 0] - s["taken"][0] - s["spill"][0])))
    acc, _ = run("stock", {"init": 0.0, "cap": 1e9}, {"inflow": np.ones((10, 1)), "outflow": np.zeros((10, 1))},
                 ticks=10)
    return {"ok": max(abs(b) for b in bal) < 1e-12 and float(acc[-1, 0]) == 10.0,
            "max_balance_error": max(abs(b) for b in bal), "accumulated": float(acc[-1, 0])}


def decay() -> dict:
    x = np.zeros((30, 1))
    x[0, 0] = 1.0
    y, _ = run("decay", {"rate": 0.2, "init": 0.0}, {"x": x}, ticks=30)
    ratio = y[1:, 0] / y[:-1, 0]
    hl = float(np.log(2) / -np.log(0.8))
    return {"ok": bool(np.allclose(ratio, 0.8)), "per_tick_ratio": float(ratio.mean()), "half_life_ticks": hl}


def hysteresis() -> dict:
    p = {"lo": 1.0, "hi": 3.0, "off": 0.0, "on": 1.0, "init": 0.0}
    up = np.array([0.0, 1.0, 2.0, 3.5, 2.0]).reshape(-1, 1)      # passes 2.0 on the way up, then again after 3.5
    y, _ = run("hysteresis", p, {"x": up}, ticks=len(up))
    first, second = float(y[2, 0]), float(y[4, 0])
    return {"ok": first == 0.0 and second == 1.0, "output_at_2_rising": first, "output_at_2_after_peak": second}


def feedback() -> dict:
    y, _ = run("feedback", {"g": 0.6}, {"x": np.ones((80, 1))}, ticks=80)
    yn, _ = run("feedback", {"g": -0.5}, {"x": np.ones((80, 1))}, ticks=80)
    return {"ok": abs(y[-1, 0] - 2.5) < 1e-9 and abs(yn[-1, 0] - 1 / 1.5) < 1e-9,
            "steady_state_g06": float(y[-1, 0]), "steady_state_gm05": float(yn[-1, 0])}


def path_dependence() -> dict:
    p = {"eta": 0.3, "k": 8.0, "init": 0.5}
    a = np.array([1.0] * 4 + [-1.0] * 4 + [0.0] * 40).reshape(-1, 1)
    b = np.array([-1.0] * 4 + [1.0] * 4 + [0.0] * 40).reshape(-1, 1)
    ya, _ = run("path_dependence", p, {"x": a}, ticks=len(a))
    yb, _ = run("path_dependence", p, {"x": b}, ticks=len(b))
    fa, fb = float(ya[-1, 0]), float(yb[-1, 0])
    return {"ok": fa > 0.8 and fb < 0.2 and float(a.sum()) == float(b.sum()), "final_share_push_A_first": fa,
            "final_share_push_B_first": fb, "same_total_push": float(a.sum())}


def network_effect() -> dict:
    base = np.linspace(0.0, 1.0, 11)
    v, _ = run("network_effect", {"strength": 3.0}, {"x": np.ones(11), "base": base})
    total = base * v
    return {"ok": bool(np.all(np.diff(v) > 0) and np.all(np.diff(total, 2) > 0)), "value_at_empty": float(v[0]),
            "value_at_full": float(v[-1])}


def congestion() -> dict:
    p = {"t0": 1.0, "alpha": 0.15, "beta": 4.0}
    lat = lambda r: float(run("congestion", p, {"load": [r], "cap": [1.0]})[0][0])  # noqa: E731
    low, high = lat(0.3) - lat(0.1), lat(1.1) - lat(0.9)
    return {"ok": high > 20 * low and abs(lat(1.0) - 1.15) < 1e-12, "rise_low_load": low, "rise_near_capacity": high}


def resource_competition() -> dict:
    claims = np.array([[2.0, 3.0, 5.0], [1.0, 1.0, 1.0]])
    supply = np.array([6.0, 10.0])
    al = ration(claims, supply)
    own = [float(run("resource_competition", {}, {"own": [4.0], "rival": [r], "supply": [6.0]})[0][0])
           for r in (0.0, 4.0, 8.0)]
    return {"ok": float(al[0].sum()) <= 6.0 + 1e-12 and np.allclose(al[1], 1.0) and own[0] == 4.0
            and own[0] > own[1] > own[2], "allocations_short": al[0].tolist(), "own_allocation_vs_rival": own}


def strategic_interaction() -> dict:
    p = {"market": 100.0, "beta": 1.0, "outside": 0.5}
    prices = np.linspace(0.1, 6.0, 300)

    def best(rival):
        d, _ = run("strategic_interaction", p, {"own": prices, "rival": np.full(300, rival)})
        return float(prices[int(np.argmax(prices * d))])
    b1, b2 = best(1.0), best(4.0)
    return {"ok": b2 > b1 + 0.05, "best_response_to_1": b1, "best_response_to_4": b2}


def partial_observability() -> dict:
    p = {"mode": "sum", "weights": [1.0, 1.0, 0.0]}
    s1 = np.array([[2.0, 1.0, 5.0]])
    s2 = np.array([[0.5, 2.5, -3.0]])
    o1, _ = run("partial_observability", p, {"xs": s1})
    o2, _ = run("partial_observability", p, {"xs": s2})
    return {"ok": float(o1[0]) == float(o2[0]) and not np.array_equal(s1, s2), "observation": float(o1[0]),
            "states_differ": True}


def state_dependent_effect() -> dict:
    p = {"gain": 1.0, "setpoint": 5.0}
    lo, _ = run("state_dependent_effect", p, {"a": [1.0], "s": [3.0]})
    hi, _ = run("state_dependent_effect", p, {"a": [1.0], "s": [7.0]})
    return {"ok": lo[0] > 0 > hi[0], "effect_below_setpoint": float(lo[0]), "effect_above_setpoint": float(hi[0])}


def regime_change() -> dict:
    p = {"before": 1.0, "after": -1.0, "at": 10.0, "trig": 1e18}
    y, _ = run("regime_change", p, {"x": np.ones((20, 1)), "trigger": np.zeros((20, 1))}, ticks=20)
    pt = {"before": 2.0, "after": 0.0, "at": -1.0, "trig": 3.0}
    trig = np.array([0, 1, 4, 0, 0, 0], dtype=float).reshape(-1, 1)
    yt, _ = run("regime_change", pt, {"x": np.ones((6, 1)), "trigger": trig}, ticks=6)
    return {"ok": bool(np.all(y[:10, 0] == 1.0) and np.all(y[10:, 0] == -1.0) and yt[1, 0] == 2.0
                       and np.all(yt[2:, 0] == 0.0)), "sign_before": float(y[0, 0]), "sign_after": float(y[-1, 0]),
            "trigger_switch_permanent": bool(np.all(yt[2:, 0] == 0.0))}


def adaptive_opponent() -> dict:
    p = {"lr": 0.3, "lm": 0.3, "margin": 0.5, "lo": 0.0, "hi": 10.0, "init": 5.0, "mode": "undercut"}
    xs = np.array([8.0] * 30 + [3.0] * 30).reshape(-1, 1)
    o, _ = run("adaptive_opponent", p, {"x": xs}, ticks=60)
    slow, _ = run("adaptive_opponent", dict(p, lr=0.03, lm=0.03), {"x": xs}, ticks=60)
    return {"ok": abs(o[29, 0] - 7.5) < 0.1 and abs(o[59, 0] - 2.5) < 0.1 and abs(slow[29, 0] - 7.5) > 0.5,
            "after_high": float(o[29, 0]), "after_low": float(o[59, 0]), "slow_learner_after_high": float(slow[29, 0])}


def measurement_noise() -> dict:
    n = 4000
    y, _ = run("measurement_noise", {"sd": 0.5}, {"x": np.full(n, 3.0)}, seed=5)
    se = 0.5 / np.sqrt(n)
    return {"ok": abs(float(y.mean()) - 3.0) < 4 * se and abs(float(y.std()) - 0.5) < 0.03,
            "mean": float(y.mean()), "sd": float(y.std())}


def missing_observation() -> dict:
    n = 5000
    y, _ = run("missing_observation", {"p": 0.3, "k": 0.0, "m": 0.0}, {"x": np.ones(n)}, seed=2)
    rate = float(np.isnan(y).mean())
    xs = np.random.default_rng(4).normal(0, 1, n)
    ym, _ = run("missing_observation", {"p": 0.0, "k": 3.0, "m": 0.0}, {"x": xs}, seed=6)
    kept = float(np.nanmean(ym))
    return {"ok": abs(rate - 0.3) < 4 * np.sqrt(0.3 * 0.7 / n) and kept < float(xs.mean()) - 0.3,
            "missing_rate": rate, "mnar_mean_of_kept": kept, "true_mean": float(xs.mean())}


def selection_effect() -> dict:
    p = {"n": 64, "sd": 1.0, "c": 0.5}
    y, _ = run("selection_effect", p, {"x": np.zeros(400)}, seed=1)
    y2, _ = run("selection_effect", p, {"x": np.full(400, 0.5)}, seed=1)
    yu, _ = run("selection_effect", dict(p, c=-1e9), {"x": np.zeros(400)}, seed=1)
    bias = float(np.nanmean(y))
    return {"ok": bias > 0.9 and abs(float(np.nanmean(yu))) < 0.05 and float(np.nanmean(y2) - bias) < 0.5,
            "observed_mean_true_0": bias, "unselected_mean": float(np.nanmean(yu)),
            "observed_change_for_true_change_0.5": float(np.nanmean(y2) - bias)}


def cascading_failure() -> dict:
    m = 20
    load = np.full((2, m), 0.8)
    load[1, 0] = 1.05                          # one component just over capacity
    cap = np.ones((2, m))
    f = cascade(load, cap, ring(m))
    calm = np.full((1, m), 0.4)
    calm[0, 0] = 1.05
    g = cascade(calm, np.ones((1, m)), ring(m))
    node_low, _ = run("cascading_failure", {"m": 20, "load": 0.5, "spread": 0.1, "cap": 1.0}, {"stress": [0.0]})
    node_hi, _ = run("cascading_failure", {"m": 20, "load": 0.5, "spread": 0.1, "cap": 1.0}, {"stress": [0.47]})
    return {"ok": float(f[0].sum()) == 0.0 and float(f[1].sum()) >= 10 and float(g[0].sum()) == 1.0
            and float(node_low[0]) == 0.0 and float(node_hi[0]) > 0.5,
            "cascade_size_near_capacity": float(f[1].sum()), "cascade_size_with_slack": float(g[0].sum()),
            "failed_fraction_low_stress": float(node_low[0]), "failed_fraction_high_stress": float(node_hi[0])}


def endogenous_demand() -> dict:
    p = {"base": 100.0, "elasticity": 1.0, "rate": 0.5, "s0": 0.5, "init": 100.0}
    served_a = np.array([0.5] * 5 + [1.0] + [0.5] * 6).reshape(-1, 1)
    served_b = np.full((12, 1), 0.5)
    da, _ = run("endogenous_demand", p, {"served": served_a}, ticks=12)
    db, _ = run("endogenous_demand", p, {"served": served_b}, ticks=12)
    return {"ok": float(da[5, 0]) == float(db[5, 0]) and float(da[6, 0]) > float(db[6, 0]) + 10,
            "same_tick_effect": float(da[5, 0] - db[5, 0]), "next_tick_effect": float(da[6, 0] - db[6, 0])}


def learning_behaviour() -> dict:
    p = {"lr": 0.2, "prior": 1.0}
    x = np.ones((30, 1))
    o = np.full((30, 1), 0.2)
    r, _ = run("learning_behaviour", p, {"x": x, "o": o}, ticks=30)
    r0, _ = run("learning_behaviour", dict(p, lr=0.0), {"x": x, "o": o}, ticks=30)
    return {"ok": float(r[0, 0]) == 1.0 and float(r[-1, 0]) < 0.25 and float(r0[-1, 0]) == 1.0,
            "response_before": float(r[0, 0]), "response_after_experience": float(r[-1, 0]),
            "response_without_learning": float(r0[-1, 0])}


CHECKS = {f.__name__: f for f in (
    linear_nonlinear, threshold, saturation, non_monotonic, complementarity, substitution, multiplicative,
    hidden_confounding, delay, stock, decay, hysteresis, feedback, path_dependence, network_effect, congestion,
    resource_competition, strategic_interaction, partial_observability, state_dependent_effect, regime_change,
    adaptive_opponent, measurement_noise, missing_observation, selection_effect, cascading_failure, endogenous_demand,
    learning_behaviour)}
