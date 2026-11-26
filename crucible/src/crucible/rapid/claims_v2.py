"""Mechanism claims (controlled interventions on copies of worlds) and public identification procedures for the
`families_v2` families (evaluator side; nothing in a world imports this module).

Claims return {"ok", ...what was measured}; identification procedures play through a `PublicView` only and return
{"ok", "question", "estimate", "truth"}.
"""
from __future__ import annotations

import numpy as np

from . import dominance as D
from .identify import PublicView
from .oracles import fork


def _lever(env, nid: str) -> str:
    eng = env._engine
    return eng.config["nodes"][eng.idx[nid]]["lever"]["name"]


def _node_series(env, act, ticks: int, nid: str) -> np.ndarray:
    """Values of node `nid` over `ticks` ticks of a fork, action act(k) at tick k (evaluator side)."""
    e = fork(env)
    i = e._engine.idx[nid]
    out = []
    for k in range(ticks):
        e.step(act(k))
        out.append(float(e._state["h"][0, -1, i]))
    return np.asarray(out)


def _gauge_series(env, act, ticks: int) -> np.ndarray:
    e = fork(env)
    out = []
    for k in range(ticks):
        r = e.step(act(k))
        out.append(list(r["observation"]["numbers"].values())[0])
    return np.asarray(out)


def _hidden_series(env, nid: str, ticks: int) -> np.ndarray:
    """The trajectory of a hidden source node while waiting (sources do not depend on actions)."""
    return _node_series(env, lambda k: None, ticks, nid)


# ====================================================================== ch_wandering_taste
def taste_moves_the_optimum(env) -> dict:
    """(1) The hidden taste wanders: the best design (the peak of the fit, at x = taste) moves by more than three
    units over the episode. (2) Controlled intervention on the same noise: holding the design that was best in the
    first fifth sells less in the last fifth than holding the design that is best in the last fifth."""
    c = env._config["claim"]
    H = env.horizon
    T = _hidden_series(env, c["taste"], H)
    k = max(4, H // 5)
    early, late = float(np.clip(T[:k].mean(), 0, 10)), float(np.clip(T[-k:].mean(), 0, 10))
    x = _lever(env, c["lever"])
    s_early = _gauge_series(env, lambda t: {"kind": "levers", "values": {x: early}}, H)[-k:].mean()
    s_late = _gauge_series(env, lambda t: {"kind": "levers", "values": {x: late}}, H)[-k:].mean()
    return {"ok": bool(T.max() - T.min() > 3.0 and abs(late - early) > 0.5 and s_late > s_early),
            "taste_range": float(T.max() - T.min()), "best_early": early, "best_late": late,
            "late_sales_holding_early_best": float(s_early), "late_sales_holding_late_best": float(s_late)}


def identify_taste_drift(env) -> dict:
    """Question: did the best design move up or down between the first and the last fifth of the episode? Public
    procedure: sweep the design in the first and the last fifth, compare where sales peak."""
    view = PublicView(env)
    c = env._config["claim"]
    x = _lever(env, c["lever"])
    H = view.horizon
    grid = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

    def sweep():
        best, bv = None, -np.inf
        for g in grid:
            vals = []
            for _ in range(3):
                if view.done:
                    break
                n = view.step({"kind": "levers", "values": {x: g}})["observation"]["numbers"]
                vals.append(list(n.values())[0])
            if len(vals) >= 2 and np.mean(vals[1:]) > bv:
                best, bv = g, float(np.mean(vals[1:]))
        return best
    b0 = sweep()
    while view.t < H - 3 * len(grid):
        view.step({"kind": "levers", "values": {x: b0}})
    b1 = sweep()
    T = _hidden_series(fork_start(env), c["taste"], H)
    k = max(4, H // 5)
    truth = bool(T[-k:].mean() > T[:k].mean())
    est = bool(b1 > b0) if b1 != b0 else None
    return {"ok": est == truth if est is not None else abs(T[-k:].mean() - T[:k].mean()) < 1.0,
            "question": "did the best design move up", "estimate": est, "truth": truth}


def fork_start(env):
    from .env import Env
    return Env(env.family, env.seed, env._config, env.difficulty, world_seed=env.world_seed, env_id=env.env_id)


# ====================================================================== ch_silent_reversal
def reversal_is_level_neutral(env) -> dict:
    """Controlled interventions on the same noise: (1) before the hidden switch, holding the lever high sells more
    than holding it low; after it, less (the effect's sign flips); (2) at the pivot setting, sales before and
    after the switch are the same (the switch does not show in levels)."""
    c = env._config["claim"]
    x = _lever(env, c["lever"])
    H = env.horizon
    at = int(c["at"])
    hi = _gauge_series(env, lambda t: {"kind": "levers", "values": {x: 10.0}}, H)
    lo = _gauge_series(env, lambda t: {"kind": "levers", "values": {x: 0.0}}, H)
    mid = _gauge_series(env, lambda t: {"kind": "levers", "values": {x: c["pivot"]}}, H)
    pre, post = slice(2, at), slice(at + 1, H)
    d_pre, d_post = float((hi - lo)[pre].mean()), float((hi - lo)[post].mean())
    lvl = float(mid[post].mean() - mid[pre].mean())
    sd = float(np.std(mid[pre])) + 1e-9
    return {"ok": bool(d_pre > 3 * sd and d_post < -3 * sd and abs(lvl) < 1.5 * sd), "effect_before": d_pre,
            "effect_after": d_post, "pivot_level_change": lvl, "noise_sd": sd, "switch_tick": at}


def identify_reversal(env) -> dict:
    """Question: at which tick does the lever's effect change sign? Public procedure: alternate the lever between
    its ends two ticks at a time and find the first window where high sells less than low."""
    view = PublicView(env)
    c = env._config["claim"]
    x = _lever(env, c["lever"])
    est = None
    diffs = []
    while not view.done:
        vals = {}
        for g in (10.0, 0.0):
            v = []
            for _ in range(2):
                if view.done:
                    break
                n = view.step({"kind": "levers", "values": {x: g}})["observation"]["numbers"]
                v.append(list(n.values())[0])
            vals[g] = v[-1] if v else None
        if vals.get(10.0) is None or vals.get(0.0) is None:
            break
        diffs.append((view.t, vals[10.0] - vals[0.0]))
        if est is None and len(diffs) >= 2 and diffs[-1][1] < 0 and diffs[-2][1] < 0:
            est = diffs[-2][0] - 2
    truth = int(c["at"])
    return {"ok": est is not None and abs(est - truth) <= 10, "question": "when does the effect reverse",
            "estimate": est, "truth": truth}


def track_moving_peak(view, env) -> None:
    """Scripted public reference (perturb and observe, as solar trackers do): first find a good promotion level
    and a starting design by a coarse sweep; then every two ticks move the design one step in the current
    direction and reverse the direction whenever sales fell after the move. Public information only: the lever
    names (the brief lists the design setting first), the readings and the ranges in the schema."""
    sp = view.action_space()
    names = sorted(sp.kinds["levers"]["properties"]["values"]["properties"])
    c = env._config["claim"]
    x, p = _lever(env, c["lever"]), _lever(env, c["second"])
    del names

    def sales(r):
        return list(r["observation"]["numbers"].values())[0]
    best, bv = 5.0, -np.inf
    for g in (1.0, 3.0, 5.0, 7.0, 9.0):
        if view.done:
            return
        view.step({"kind": "levers", "values": {x: g, p: 3.0}})
        v = sales(view.step({"kind": "levers", "values": {x: g, p: 3.0}}))
        if v > bv:
            best, bv = g, v
    xv, direction, step, last = best, 1.0, 0.6, None
    while not view.done:
        view.step({"kind": "levers", "values": {x: xv, p: 3.0}})
        if view.done:
            return
        v = sales(view.step({"kind": "levers", "values": {x: xv, p: 3.0}}))
        if last is not None and v < last:
            direction = -direction
        last = v
        xv = float(np.clip(xv + direction * step, 0.0, 10.0))


# ====================================================================== ch_format_switch
def _decision(env, nid: str) -> tuple[str, list[str]]:
    eng = env._engine
    o = eng.config["nodes"][eng.idx[nid]]["options"]
    return o["name"], list(o["choices"])


def option_effect_flips(env) -> dict:
    """Controlled interventions on the same noise: holding the first format sells more than holding the second
    before the hidden switch and less after it, and the two formats' average is the same before and after (the
    rule change does not show in the market's overall level)."""
    c = env._config["claim"]
    name, opts = _decision(env, c["decision"])
    H = env.horizon
    at = int(c["at"])
    a = _gauge_series(env, lambda t: {"kind": "discrete", "decision": name, "choice": opts[0]}, H)
    b = _gauge_series(env, lambda t: {"kind": "discrete", "decision": name, "choice": opts[1]}, H)
    pre, post = slice(2, at), slice(at + 1, H)
    d_pre, d_post = float((a - b)[pre].mean()), float((a - b)[post].mean())
    avg = (a + b) / 2
    lvl = float(avg[post].mean() - avg[pre].mean())
    sd = float(np.std(avg[pre])) + 1e-9
    return {"ok": bool(d_pre > 3 * sd and d_post < -3 * sd and abs(lvl) < 1.5 * sd), "first_minus_second_before":
            d_pre, "first_minus_second_after": d_post, "average_level_change": lvl, "noise_sd": sd, "switch_tick": at}


def identify_option_flip(env) -> dict:
    """Question: at which tick does the better format change? Public procedure: alternate the two formats three
    ticks at a time and find the first two consecutive rounds in which the first format sells less."""
    view = PublicView(env)
    c = env._config["claim"]
    name, opts = _decision(env, c["decision"])
    est, diffs = None, []
    while not view.done:
        vals = {}
        for o in opts:
            v = []
            for _ in range(3):
                if view.done:
                    break
                n = view.step({"kind": "discrete", "decision": name, "choice": o})["observation"]["numbers"]
                v.append(list(n.values())[0])
            vals[o] = float(np.mean(v[1:])) if len(v) > 1 else None
        if vals.get(opts[0]) is None or vals.get(opts[1]) is None:
            break
        diffs.append((view.t, vals[opts[0]] - vals[opts[1]]))
        if est is None and len(diffs) >= 2 and diffs[-1][1] < 0 and diffs[-2][1] < 0:
            est = diffs[-2][0] - 4
    truth = int(c["at"])
    return {"ok": est is not None and abs(est - truth) <= 10, "question": "when does the better format change",
            "estimate": est, "truth": truth}


def probe_and_switch(view, env) -> None:
    """Scripted public reference for an option whose value can change: keep the option that has been selling
    better; every eight ticks try the other one for three ticks, and switch after two such probes in a row in
    which the other sold more on average than the current one did over its last four ticks (averaging, because
    single readings are dominated by demand noise)."""
    sp = view.action_space()
    (dname, opts), = sp.limits["decisions"].items()

    def units(r):
        return list(r["observation"]["numbers"].values())[0]
    cur, other = opts[0], opts[1]
    wins, k, recent = 0, 0, []
    while not view.done:
        k += 1
        if k % 8 == 0:
            seen = []
            for _ in range(3):
                if view.done:
                    return
                seen.append(units(view.step({"kind": "discrete", "decision": dname, "choice": other})))
            base = float(np.mean(recent[-4:])) if recent else -np.inf
            wins = wins + 1 if np.mean(seen[1:]) > base else 0
            if wins >= 2:
                cur, other, wins, recent = other, cur, 0, list(seen[1:])
        else:
            recent.append(units(view.step({"kind": "discrete", "decision": dname, "choice": cur})))


# ====================================================================== hc_hidden_state_sign
def sign_follows_hidden_state(env) -> dict:
    """Controlled interventions on the same noise: the effect of holding the lever high rather than low has the
    sign of (set point - hidden state) tick by tick (it changes sign as the state wanders), and it changes sign
    at least twice per episode; the footfall gauge correlates with the state."""
    c = env._config["claim"]
    x = _lever(env, c["lever"])
    H = env.horizon
    S = _hidden_series(env, c["state"], H)
    hi = _gauge_series(env, lambda t: {"kind": "levers", "values": {x: 10.0}}, H)
    lo = _gauge_series(env, lambda t: {"kind": "levers", "values": {x: 0.0}}, H)
    d = hi - lo
    want = np.sign(c["setpoint"] - S)
    big = np.abs(c["setpoint"] - S) > 0.1
    agree = float(np.mean(np.sign(d[big]) == want[big])) if big.any() else 0.0
    flips = int(np.sum(np.diff(want[big]) != 0)) if big.sum() > 1 else 0
    return {"ok": bool(agree > 0.9 and flips >= 2), "sign_agreement": agree, "sign_changes": flips}


def identify_state_sign(env) -> dict:
    """Question: is the lever's effect positive in the last fifth of the episode? Public procedure: hold the lever
    at the middle; in the last fifth alternate high and low two ticks at a time and compare sales."""
    view = PublicView(env)
    c = env._config["claim"]
    x = _lever(env, c["lever"])
    H = view.horizon
    while view.t < H - max(8, H // 5):
        view.step({"kind": "levers", "values": {x: 5.0}})
    diffs = []
    while not view.done:
        vals = {}
        for g in (10.0, 0.0):
            v = None
            for _ in range(2):
                if view.done:
                    break
                v = list(view.step({"kind": "levers", "values": {x: g}})["observation"]["numbers"].values())[0]
            vals[g] = v
        if vals.get(10.0) is None or vals.get(0.0) is None:
            break
        diffs.append(vals[10.0] - vals[0.0])
    S = _hidden_series(fork_start(env), c["state"], H)
    k = max(8, H // 5)
    truth = bool(np.mean(c["setpoint"] - S[-k:]) > 0)
    est = bool(np.mean(diffs) > 0) if diffs else None
    return {"ok": est == truth, "question": "is the effect positive at the end", "estimate": est, "truth": truth}


def sign_tracker(view, env) -> None:
    """Scripted public reference for a lever whose effect can change sign: in every block of ten ticks, hold the
    lever high for two ticks and low for two (the second tick of each counts, so a one-tick lag does not
    mislead), then hold the better end for the rest of the block. The cost-only second lever is held at a
    moderate level found by a short sweep at the start."""
    c = env._config["claim"]
    x, q = _lever(env, c["lever"]), _lever(env, c["second"])

    def sales(r):
        return list(r["observation"]["numbers"].values())[0]

    def net(r):
        res = r["observation"]["resources"]
        return res.get("revenue_cents_last_tick", 0.0) - res.get("cost_cents_last_tick", 0.0)
    best_q, bv = 0.0, -np.inf
    for g in (0.0, 2.0, 4.0, 6.0):
        if view.done:
            return
        view.step({"kind": "levers", "values": {q: g, x: 5.0}})
        v = net(view.step({"kind": "levers", "values": {q: g, x: 5.0}}))
        if v > bv:
            best_q, bv = g, v
    while not view.done:
        seen = {}
        for g in (10.0, 0.0):
            v = None
            for _ in range(2):
                if view.done:
                    return
                v = sales(view.step({"kind": "levers", "values": {x: g, q: best_q}}))
            seen[g] = v
        end = 10.0 if seen[10.0] >= seen[0.0] else 0.0
        for _ in range(6):
            if view.done:
                return
            view.step({"kind": "levers", "values": {x: end, q: best_q}})


# ====================================================================== competing hypotheses
def _law_values(env, conds) -> np.ndarray:
    from .engines.primitive_graph import true_response
    return np.asarray(true_response(env._engine, conds, env._config["objective"]["gauge"]))


def _fit_hyp(name: str, X: np.ndarray, y: np.ndarray):
    """Least-squares fit of a named public hypothesis to (x1, x2) -> y; returns a predictor and the RMS residual."""
    a, b = X[:, 0], X[:, 1]
    best = None
    for h in np.linspace(0.3, 8.0, 40):
        if name == "saturating":
            F = np.c_[a / (a + h), b, np.ones(len(a))]
        elif name == "power":
            k = 0.2 + h / 10.0
            F = np.c_[np.abs(a) ** k, b, np.ones(len(a))]
        elif name == "additive":
            F = np.c_[a / (a + h), b / (b + 3.0), np.ones(len(a))]
        else:                                         # interactive
            F = np.c_[a / (a + h) * b, a / (a + h), np.ones(len(a))]
        coef, *_ = np.linalg.lstsq(F, y, rcond=None)
        res = float(np.sqrt(np.mean((F @ coef - y) ** 2)))
        if best is None or res < best[0]:
            best = (res, h, coef)
    res, h, coef = best

    def pred(Xq):
        a, b = Xq[:, 0], Xq[:, 1]
        if name == "saturating":
            F = np.c_[a / (a + h), b, np.ones(len(a))]
        elif name == "additive":
            F = np.c_[a / (a + h), b / (b + 3.0), np.ones(len(a))]
        elif name == "power":
            F = np.c_[np.abs(a) ** (0.2 + h / 10.0), b, np.ones(len(a))]
        else:
            F = np.c_[a / (a + h) * b, a / (a + h), np.ones(len(a))]
        return F @ coef
    return pred, res


def hypotheses_diverge_out_of_range(env) -> dict:
    """Evaluator side, noise free: the rival hypothesis, fitted to the true law over the settable range, stays
    close to it there (the two are hard to tell apart without designed experiments) but misses the held-out
    conditions by far more."""
    c = env._config["claim"]
    x1, x2 = [_lever(env, n) for n in c["levers"]]
    grid = [(float(a), float(b)) for a in np.linspace(0, 6, 7) for b in np.linspace(0, 6, 7)]
    X = np.asarray(grid)
    y = _law_values(env, [{x1: a, x2: b} for a, b in grid])
    rival = [h for h in c["hypotheses"] if h != c["law"]][0]
    pred, res = _fit_hyp(rival, X, y)
    held = env._config["objective"]["heldout"]
    Xh = np.asarray([[cnd[x1], cnd[x2]] for cnd in held])
    yh = _law_values(env, held)
    miss = float(np.sqrt(np.mean((pred(Xh) - yh) ** 2)))
    span = float(np.ptp(y)) + 1e-9
    return {"ok": bool(res < 0.12 * span and miss > 2.5 * res and miss > 0.1 * float(np.ptp(yh) + span)),
            "rival": rival, "rival_rms_in_range": res, "rival_rms_heldout": miss, "range_span": span}


def _grid_measure(view, x1, x2, reps=2, n=5):
    X, y = [], []
    for a in np.linspace(0, 6, n):
        for b in np.linspace(0, 6, n):
            vals = []
            for _ in range(reps):
                if view.done:
                    return np.asarray(X), np.asarray(y)
                r = view.step({"kind": "levers", "values": {x1: float(a), x2: float(b)}})
                vals.append(list(r["observation"]["numbers"].values())[0])
            X.append([a, b])
            y.append(float(np.mean(vals)))
            if view.t >= view.horizon - 2:
                return np.asarray(X), np.asarray(y)
    return np.asarray(X), np.asarray(y)


def test_hypotheses_and_predict(view, env) -> None:
    """Scripted public reference: measure a grid of settings (the setting acts at once), fit both public
    hypotheses, keep the one with the smaller residual and file its predictions for the held-out conditions."""
    c = env._config["claim"]
    x1, x2 = [_lever(env, n) for n in c["levers"]]
    X, y = _grid_measure(view, x1, x2, reps=1, n=6)
    held = env._config["objective"]["heldout"]
    fits = {h: _fit_hyp(h, X, y) for h in c["hypotheses"]}
    h = min(fits, key=lambda k: fits[k][1])
    Xh = np.asarray([[cnd[x1], cnd[x2]] for cnd in held])
    pv = fits[h][0](Xh)
    if not view.done:
        view.step({"kind": "message", "to": "journal",
                   "content": {"predictions": [{"set": cnd, "value": float(v)} for cnd, v in zip(held, pv)]}})
    while not view.done:
        view.step(None)


def identify_hypothesis(env) -> dict:
    view = PublicView(env)
    c = env._config["claim"]
    x1, x2 = [_lever(env, n) for n in c["levers"]]
    X, y = _grid_measure(view, x1, x2, reps=1, n=6)
    fits = {h: _fit_hyp(h, X, y)[1] for h in c["hypotheses"]}
    est = min(fits, key=fits.get)
    return {"ok": est == c["law"], "question": "which hypothesis holds", "estimate": est, "truth": c["law"]}


# ====================================================================== hidden conservation
def total_is_conserved(env) -> dict:
    """Under random rate settings the three tanks' total stays exactly constant every tick (conservation), while
    the gauged two tanks' sum does not (the third tank is hidden)."""
    from .policies import random_policy
    c = env._config["claim"]
    e = fork(env)
    eng = e._engine
    idx = [eng.idx[t] for t in c["tanks"]]
    rng = np.random.default_rng(3)
    tot, vis = [], []
    for _ in range(40):
        e.step(random_policy(e.observe(), e.action_space(), rng))
        h = e._state["h"][0, -1]
        tot.append(float(sum(h[i] for i in idx)))
        vis.append(float(h[idx[0]] + h[idx[1]]))
    return {"ok": bool(np.ptp(tot) < 1e-6 * max(1.0, abs(tot[0])) and abs(tot[0] - c["total"]) < 1e-6 * c["total"]
                       and np.ptp(vis) > 0.05 * c["total"]),
            "total_range": float(np.ptp(tot)), "gauged_sum_range": float(np.ptp(vis)), "total": tot[0]}


def _settle(view, r1n, r2n, a, b, gauges, ticks):
    """Hold rates (a, b) for `ticks` ticks; mean gauged levels over the last five."""
    ra, rb = [], []
    for k in range(ticks):
        if view.done:
            break
        n = view.step({"kind": "levers", "values": {r1n: a, r2n: b}})["observation"]["numbers"]
        if k >= ticks - 5:
            ra.append(n.get(gauges[0], np.nan))
            rb.append(n.get(gauges[1], np.nan))
    return float(np.nanmean(ra)) if ra else None, float(np.nanmean(rb)) if rb else None


def _solve_conservation(obs):
    """From settled levels (r1, r2, A, B) at several rate settings: r1 A = rho C and C = M - A - B, so
    r1 A = rho (M - A - B) - linear in (rho M, rho) -> least squares."""
    Arow, yv = [], []
    for r1, r2, a, b in obs:
        Arow.append([1.0, -(a + b)])                  # rho*M - rho*(A+B) = r1*A
        yv.append(r1 * a)
    sol, *_ = np.linalg.lstsq(np.asarray(Arow), np.asarray(yv), rcond=None)
    rhoM, rho = float(sol[0]), float(sol[1])
    rho = max(rho, 1e-6)
    return rhoM / rho, rho


def infer_conservation(view, env) -> None:
    """Scripted public reference: hold three different rate settings long enough to settle, solve the
    conservation relation for the hidden total and return rate, and predict the settled second gauge at the
    held-out rates (A* = M / (1 + r1/r2 + r1/rho), B* = r1 A* / r2)."""
    c = env._config["claim"]
    r1n, r2n = [_lever(env, n) for n in c["levers"]]
    g = c["gauges"]
    span = max(8, (view.horizon - 2) // 3)
    obs = []
    for a, b in ((0.1, 0.4), (0.4, 0.1), (0.3, 0.3)):
        A, B = _settle(view, r1n, r2n, a, b, g, span)
        if A is not None:
            obs.append((a, b, A, B))
    held = env._config["objective"]["heldout"]
    if len(obs) >= 2 and not view.done:
        M, rho = _solve_conservation(obs)
        preds = []
        for cnd in held:
            a, b = cnd[r1n], cnd[r2n]
            Ast = M / (1.0 + a / b + a / rho)
            preds.append({"set": cnd, "value": float(a * Ast / b)})
        view.step({"kind": "message", "to": "journal", "content": {"predictions": preds}})
    while not view.done:
        view.step(None)


def identify_total(env) -> dict:
    view = PublicView(env)
    c = env._config["claim"]
    r1n, r2n = [_lever(env, n) for n in c["levers"]]
    obs = []
    span = max(8, (view.horizon - 2) // 3)
    for a, b in ((0.1, 0.4), (0.4, 0.1), (0.3, 0.3)):
        A, B = _settle(view, r1n, r2n, a, b, c["gauges"], span)
        if A is not None:
            obs.append((a, b, A, B))
    M, _ = _solve_conservation(obs) if len(obs) >= 2 else (None, None)
    return {"ok": M is not None and abs(M - c["total"]) < 0.1 * c["total"], "question": "what is the hidden total",
            "estimate": M, "truth": c["total"]}


# ====================================================================== selection bias
def _truncated_mean(mu, sd, cut):
    from math import erf, exp, pi, sqrt
    a = (cut - mu) / sd
    pdf = exp(-0.5 * a * a) / sqrt(2 * pi)
    sf = 0.5 * (1 - erf(a / sqrt(2)))
    return mu + sd * pdf / max(sf, 1e-12)


def selection_flattens(env) -> dict:
    """On the same noise, moving the first setting from low to high moves the reported mean by clearly less than
    it moves the true mean (truncation flattens the response), and at a low setting the reported mean sits above
    the true mean (selection bias)."""
    c = env._config["claim"]
    x1, x2 = [_lever(env, n) for n in c["levers"]]
    out = {}
    for lab, v in (("low", 0.0), ("high", 6.0)):
        act = {"kind": "levers", "values": {x1: v, x2: 3.0}}
        rd = _gauge_series(env, lambda t, act=act: act, 6)[1:].mean()
        tr = _node_series(env, lambda t, act=act: act, 6, c["outcome"])[1:].mean()
        out[lab] = (float(rd), float(tr))
    d_read = out["high"][0] - out["low"][0]
    d_true = out["high"][1] - out["low"][1]
    return {"ok": bool(d_true > 0 and d_read < 0.85 * d_true and out["low"][0] > out["low"][1] + 0.3),
            "reported_change": d_read, "true_change": d_true, "low_reported": out["low"][0],
            "low_true": out["low"][1]}


def _fit_selection(X, y, sd):
    """Fit mu = a + b1 x1 + b2 x2 and the cut to reported means under truncation (public spread sd), by a grid
    over the cut and least squares refinement of the linear part."""
    best = None
    for cut in np.linspace(-6.0, 10.0, 81):
        mu = np.asarray(y, dtype=float).copy()
        for _ in range(30):                      # fixed point: mu such that truncated_mean(mu) = y
            tm = np.asarray([_truncated_mean(m, sd, cut) for m in mu])
            mu = mu + (np.asarray(y) - tm)
        F = np.c_[np.ones(len(X)), X]
        coef, *_ = np.linalg.lstsq(F, mu, rcond=None)
        pred = np.asarray([_truncated_mean(m, sd, cut) for m in F @ coef])
        err = float(np.mean((pred - y) ** 2))
        if best is None or err < best[0]:
            best = (err, cut, coef)
    return best[1], best[2]


def _sd_from_brief(view) -> float:
    import re
    txt = " ".join(view.spec["brief"].values())
    m = re.search(r"spread of about ([0-9]+(?:\.[0-9]+)?)", txt)
    return float(m.group(1)) if m else 2.5


def correct_selection_and_predict(view, env) -> None:
    """Scripted public reference: measure the reported mean over a grid of settings, fit a linear law behind a
    truncation at an unknown cut (the spread is stated in the brief), predict the true mean at held-out settings."""
    c = env._config["claim"]
    x1, x2 = [_lever(env, n) for n in c["levers"]]
    X, y = _grid_measure(view, x1, x2, reps=1, n=5)
    held = env._config["objective"]["heldout"]
    if len(y) >= 6 and not view.done:
        cut, coef = _fit_selection(X, y, _sd_from_brief(view))
        preds = [{"set": cnd, "value": float(coef[0] + coef[1] * cnd[x1] + coef[2] * cnd[x2])} for cnd in held]
        view.step({"kind": "message", "to": "journal", "content": {"predictions": preds}})
    while not view.done:
        view.step(None)


def identify_selection_slope(env) -> dict:
    view = PublicView(env)
    c = env._config["claim"]
    x1, x2 = [_lever(env, n) for n in c["levers"]]
    X, y = _grid_measure(view, x1, x2, reps=1, n=5)
    _, coef = _fit_selection(X, y, _sd_from_brief(view))
    tr = c["coef"][1]
    return {"ok": abs(coef[1] - tr) < 0.25 * abs(tr), "question": "true slope in the first setting",
            "estimate": float(coef[1]), "truth": tr}


# ====================================================================== slow response
def response_is_delayed_and_lagged(env) -> dict:
    """Controlled step on the same noise: after raising the first setting nothing changes for exactly the dead
    time, then the outcome approaches its new level geometrically (first-order lag): after the dead time plus
    three time constants it has covered most but not all of the way."""
    c = env._config["claim"]
    x1, x2 = [_lever(env, n) for n in c["levers"]]
    n = 120
    a = _node_series(env, lambda t: {"kind": "levers", "values": {x1: 6.0 if t >= 5 else 0.0, x2: 0.0}}, n,
                     c["outcome"])
    b = _node_series(env, lambda t: {"kind": "levers", "values": {x1: 0.0, x2: 0.0}}, n, c["outcome"])
    d = a - b
    first = int(np.argmax(np.abs(d) > 1e-9)) if (np.abs(d) > 1e-9).any() else None
    final = float(d[-1])
    k = 5 + c["lag"] + int(round(3 / c["rate"]))
    frac = float(d[min(k, n - 1)] / final) if final else 0.0
    return {"ok": bool(first == 5 + c["lag"] and 0.75 < frac < 0.995), "first_change_at": first,
            "expected_first_change": 5 + c["lag"], "share_after_three_time_constants": frac}


def settle_and_predict(view, env) -> None:
    """Scripted public reference: hold each of a few settings until the reading stops moving, fit a saturating law
    in the first setting plus a linear one in the second to the settled levels, predict the held-out ones."""
    c = env._config["claim"]
    x1, x2 = [_lever(env, n) for n in c["levers"]]
    pts = [(0.0, 0.0), (3.0, 0.0), (6.0, 0.0), (6.0, 6.0)]
    hold = max(6, (view.horizon - 2) // len(pts))
    X, y = [], []
    for a, b in pts:
        vals = []
        for k in range(hold):
            if view.done:
                break
            n = view.step({"kind": "levers", "values": {x1: a, x2: b}})["observation"]["numbers"]
            vals.append(list(n.values())[0])
        if len(vals) >= 4:
            # extrapolate the geometric approach: fit v_t = L - (L - v0) r^t on the last values
            v = np.asarray(vals[-4:])
            dv = np.diff(v)
            r = float(np.clip(dv[1:].sum() / dv[:-1].sum(), 0.0, 0.95)) if abs(dv[:-1].sum()) > 1e-9 else 0.0
            L = float(v[-1] + dv[-1] * r / (1 - r)) if r > 0 else float(v[-1])
            X.append([a, b])
            y.append(L)
    held = env._config["objective"]["heldout"]
    if len(y) >= 4 and not view.done:
        pred, _ = _fit_hyp("saturating", np.asarray(X), np.asarray(y))
        Xh = np.asarray([[cnd[x1], cnd[x2]] for cnd in held])
        view.step({"kind": "message", "to": "journal",
                   "content": {"predictions": [{"set": cnd, "value": float(v)} for cnd, v in zip(held, pred(Xh))]}})
    while not view.done:
        view.step(None)


def identify_dead_time(env) -> dict:
    view = PublicView(env)
    c = env._config["claim"]
    x1, x2 = [_lever(env, n) for n in c["levers"]]
    base = []
    for _ in range(4):
        base.append(list(view.step({"kind": "levers", "values": {x1: 0.0, x2: 0.0}})["observation"]["numbers"]
                         .values())[0])
    sd = float(np.std(base)) + 1e-6
    t0 = view.t
    est = None
    for k in range(15):
        if view.done:
            break
        v = list(view.step({"kind": "levers", "values": {x1: 6.0, x2: 0.0}})["observation"]["numbers"].values())[0]
        if est is None and v - np.mean(base) > 4 * sd:
            est = view.t - 1 - t0
    return {"ok": est is not None and abs(est - c["lag"]) <= 1, "question": "dead time", "estimate": est,
            "truth": c["lag"]}


# ====================================================================== causal prediction under confounding
def records_overstate_the_effect(env) -> dict:
    """The previous operator's records (passive regime) show a slope of sales on spending several times larger
    than the interventional effect measured on copies of the world with spending set (same noise)."""
    c = env._config["claim"]
    eng = env._engine
    x, z = [_lever(env, n) for n in c["levers"]]
    gname = env._config["objective"]["gauge"]
    hx, hy = eng.history_of(env._state, x), eng.history_of(env._state, gname)
    passive = float(np.polyfit(hx, hy, 1)[0])
    lo = _gauge_series(env, lambda t: {"kind": "levers", "values": {x: 1.0, z: 0.0}}, 30)
    hi = _gauge_series(env, lambda t: {"kind": "levers", "values": {x: 5.0, z: 0.0}}, 30)
    causal = float((hi - lo).mean() / 4.0)
    return {"ok": bool(passive > 2.0 * max(causal, 1e-9) and causal > 0), "passive_slope": passive,
            "interventional_slope": causal}


def intervene_and_predict(view, env) -> None:
    """Scripted public reference: ignore the records; set spending across its range (several ticks each, the
    hidden demand cycle averages out), fit a saturating effect plus a linear second lever, predict the held-out
    interventions."""
    c = env._config["claim"]
    x, z = [_lever(env, n) for n in c["levers"]]
    pts = [(0.0, 0.0), (1.0, 0.0), (2.5, 0.0), (6.0, 0.0), (6.0, 6.0)]
    hold = max(3, (view.horizon - 2) // len(pts))
    X, y = [], []
    for a, b in pts:
        vals = []
        for _ in range(hold):
            if view.done:
                break
            vals.append(list(view.step({"kind": "levers", "values": {x: a, z: b}})["observation"]["numbers"]
                             .values())[0])
        if vals:
            X.append([a, b])
            y.append(float(np.mean(vals)))
    held = env._config["objective"]["heldout"]
    if len(y) >= 4 and not view.done:
        pred, _ = _fit_hyp("saturating", np.asarray(X), np.asarray(y))
        Xh = np.asarray([[cnd[x], cnd[z]] for cnd in held])
        view.step({"kind": "message", "to": "journal",
                   "content": {"predictions": [{"set": cnd, "value": float(v)} for cnd, v in zip(held, pred(Xh))]}})
    while not view.done:
        view.step(None)


def identify_effect_size(env) -> dict:
    """Question: is spending's interventional effect over its range smaller than the records suggest?"""
    view = PublicView(env)
    c = env._config["claim"]
    x, z = [_lever(env, n) for n in c["levers"]]
    vals = {}
    for a in (1.0, 5.0):
        vs = []
        for _ in range(8):
            if view.done:
                break
            vs.append(list(view.step({"kind": "levers", "values": {x: a, z: 0.0}})["observation"]["numbers"]
                           .values())[0])
        vals[a] = float(np.mean(vs)) if vs else 0.0
    causal = (vals[5.0] - vals[1.0]) / 4.0
    eng = env._engine
    hx, hy = eng.history_of(env._state, x), eng.history_of(env._state, env._config["objective"]["gauge"])
    passive = float(np.polyfit(hx, hy, 1)[0])
    truth = records_overstate_the_effect(fork_start(env))["ok"]
    return {"ok": bool((passive > 2 * causal) == truth), "question": "do the records overstate the effect",
            "estimate": bool(passive > 2 * causal), "truth": truth}


# ====================================================================== dl_delayed_returns (reference)
def invest_learn_lag_then_stop(view, env) -> None:
    """Scripted public reference for delayed investment returns: invest once and watch for the output to answer
    (that is the construction delay L); invest at a steady level while the investment can still pay back, and
    stop investing L + a margin ticks before the end (later investment never comes on line in time)."""
    sp = view.action_space()
    (name, p), = sp.kinds["levers"]["properties"]["values"]["properties"].items()
    H = view.horizon

    def out(r):
        return list(r["observation"]["numbers"].values())[0]
    base = out(view.step({"kind": "levers", "values": {name: float(p["maximum"])}}))
    lag, seen = None, [base]
    q = 0.5 * float(p["maximum"])
    while not view.done and lag is None and view.t < H // 2:
        v = out(view.step({"kind": "levers", "values": {name: q}}))
        if v - seen[0] > 2.0:
            lag = view.t - 1
        seen.append(v)
    lag = lag or 8
    # capacity built after `stop` cannot repay its cost: it comes on line L ticks later and has only the last
    # quarter of the episode (roughly a depreciating unit's payback time) left to earn
    stop = H - lag - max(3, H // 4)
    while not view.done:
        view.step({"kind": "levers", "values": {name: q if view.t < stop else 0.0}})


# ====================================================================== net_repair (reference)
def learn_which_links_matter(view, env) -> None:
    """Scripted public reference for network repair: learn from its own observations which links matter and which
    are fragile. A link's importance is the drop in served demand seen in the tick it broke (or the rise in the
    tick after its repair); its fragility is how often it broke. Each tick it repairs the broken link with the
    largest learned importance when that is worth a repair (links never seen are repaired once, to learn), and
    once a link that matters has broken three times it builds a parallel link between its ends (a bypass that
    does not inherit its fragility). Public information only: the link entities, the served-demand gauge (the
    first reading), the stated costs in the action schemas."""
    imp: dict[str, float] = {}
    fails: dict[str, int] = {}
    built: set = set()
    obs = view.observe()
    prev_up, prev_served, pending = None, None, None
    while not view.done:
        links = {x["id"]: x for x in obs["entities"] if x.get("kind") == "link" and not x["id"].startswith("new")}
        served = list(obs["numbers"].values())[0] if obs["numbers"] else 0.0
        up = {k: bool(v["up"]) for k, v in links.items()}
        if prev_up is not None:
            newly_down = [k for k in up if prev_up.get(k) and not up[k]]
            for k in newly_down:
                fails[k] = fails.get(k, 0) + 1
                drop = (prev_served or 0.0) - served
                imp[k] = max(imp.get(k, 0.0), drop / max(1, len(newly_down)))
            if pending is not None and up.get(pending):
                imp[pending] = max(imp.get(pending, 0.0), served - (prev_served or 0.0))
        broken = [k for k, v in up.items() if not v]
        act = None
        cands = sorted(broken, key=lambda k: -imp.get(k, 1e9))
        if cands:
            k = cands[0]
            if k not in imp or imp[k] > 0.5:
                act = {"kind": "discrete", "decision": "repair", "choice": k}
                pending = k
        bypass = [k for k in links if fails.get(k, 0) >= 3 and imp.get(k, 0.0) > 1.0 and k not in built]
        if bypass:
            k = bypass[0]
            a, b = links[k]["ends"]
            edit = {"kind": "graph_edit", "op": "add_edge", "src": a, "dst": b, "weight": 1.0}
            built.add(k)
            act = {"kind": "sequence", "steps": [edit] + ([act] if act else [])}
        prev_up, prev_served = up, served
        r = view.step(act)
        if not r["result"]["ok"] and act is not None and act["kind"] == "sequence":
            r = view.step(None) if view.done is False else r
        obs = r["observation"]


def oracle_option_switch(env) -> dict:
    """Evaluator-side upper reference: knowing the hidden switch tick, hold the first option before it and the
    second after it."""
    from .oracles import best_of
    c = env._config["claim"]
    name, opts = _decision(env, c["decision"])
    at = int(c["at"])
    pols = {f"switch_at_{at + d}": (lambda e, d=d: {"kind": "discrete", "decision": name,
                                                     "choice": opts[0] if e.t < at + d else opts[1]})
            for d in (0, 1)}
    return best_of(env, pols)


# ====================================================================== dl_wear_reversal
def wear_triggers_reversal(env) -> dict:
    """Controlled interventions on the same noise: (1) driving hard from the start flips the sign of the drive's
    effect once the accumulated wear reaches the hidden level (high beats low before, loses after); (2) the flip
    comes later when the machine was first run gently for a while (the switch time follows the participant's own
    past, not a clock)."""
    c = env._config["claim"]
    x = _lever(env, c["lever"])
    H = env.horizon
    L = int(c["level"])

    def first_flip(warm: int) -> int | None:
        for t0 in range(warm + 1, H - 2):
            hi = _gauge_series(env, lambda t: {"kind": "levers", "values": {x: 10.0 if t >= warm else 5.0}}, t0 + 2)
            lo = _gauge_series(env, lambda t: {"kind": "levers", "values": {x: 10.0 if warm <= t < t0 else
                                                                            (0.0 if t >= t0 else 5.0)}}, t0 + 2)
            if hi[-1] < lo[-1]:
                return t0
        return None
    f0, f1 = first_flip(0), first_flip(10)
    return {"ok": bool(f0 is not None and f1 is not None and abs(f0 - (L + 1)) <= 2 and f1 >= f0 + 8),
            "flip_driving_hard_from_start": f0, "flip_after_ten_gentle_ticks": f1, "wear_level": L}


def identify_wear_reversal(env) -> dict:
    """Question: after how many ticks of hard driving does the drive start to hurt? Public procedure: drive hard,
    every four ticks compare one tick low against the hard setting, and report when low first wins twice."""
    view = PublicView(env)
    c = env._config["claim"]
    x = _lever(env, c["lever"])
    hard, est, wins = 0, None, 0
    last_hi = None
    while not view.done and est is None:
        for _ in range(3):
            if view.done:
                break
            last_hi = list(view.step({"kind": "levers", "values": {x: 10.0}})["observation"]["numbers"].values())[0]
            hard += 1
        if view.done:
            break
        lo = list(view.step({"kind": "levers", "values": {x: 0.0}})["observation"]["numbers"].values())[0]
        wins = wins + 1 if lo > last_hi else 0
        if wins >= 2:
            est = hard - 3
    return {"ok": est is not None and abs(est - int(c["level"])) <= 6, "question": "hard ticks before damage",
            "estimate": est, "truth": int(c["level"])}


def probe_the_other_end(view, env) -> None:
    """Scripted public reference for a lever whose effect sign follows a hidden regime: hold the end of the
    lever's range that earned more; every seventh tick try the other end for one tick and move there if it
    earned more net income than the tick before (net income is observed exactly)."""
    sp = view.action_space()
    (name, p), = sp.kinds["levers"]["properties"]["values"]["properties"].items()
    ends = (float(p["minimum"]), float(p["maximum"]))
    cur = 1
    last = None
    while not view.done:
        if view.t % 7 == 6 and last is not None:
            r = view.step({"kind": "levers", "values": {name: ends[1 - cur]}})
            if D._net(r["observation"]) > last:
                cur = 1 - cur
            last = D._net(r["observation"])
            continue
        r = view.step({"kind": "levers", "values": {name: ends[cur]}})
        last = D._net(r["observation"])
