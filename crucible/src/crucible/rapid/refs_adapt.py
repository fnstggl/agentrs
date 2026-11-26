"""Scripted public-information references for worlds whose best action keeps changing (evaluator side; scripted,
labelled as such in evidence).

They play through a `PublicView` - actions, observations, tool schemas, the public spec, nothing hidden - and know
no mechanism. What they assume is only what the brief and the action schemas state: which levers exist and their
ranges. Admission (`references.play`) uses them as the adaptive public reference of the families that name them; a
world in which they cannot beat the best constant setting, threshold rule or explore-then-commit policy by the
margin of decision H-D1 is not admitted, whatever its mechanism.

`lever_switch_tracker` - for worlds in which which lever pays (or whether a lever pays at all) changes at times
nobody announces: it learns a good level for each lever, then in every block of ticks briefly tests running
each lever alone (and neither) and runs the best for the rest of the block, while tuning the running lever's
level by small trials. Tests are read on the last tick of a hold, so effects that take a tick to show count.
"""
from __future__ import annotations

import numpy as np


def _net(r) -> float:
    res = r["observation"]["resources"]
    return float(res.get("revenue_cents_last_tick", 0.0) - res.get("cost_cents_last_tick", 0.0))


def _hold(view, setting: dict, ticks: int) -> float | None:
    """Hold a lever setting for `ticks` ticks; the net income of the last tick (None when the episode ended)."""
    v = None
    for _ in range(ticks):
        if view.done:
            return None
        v = _net(view.step({"kind": "levers", "values": setting}))
    return v


def lever_switch_tracker(view, env=None, block: int = 10, test: int = 2) -> None:
    sp = view.action_space()
    props = sp.kinds["levers"]["properties"]["values"]["properties"]
    names = sorted(props)
    lo = {n: float(props[n]["minimum"]) for n in names}
    hi = {n: float(props[n]["maximum"]) for n in names}
    off = {n: lo[n] for n in names}

    def at(n, f):
        return lo[n] + f * (hi[n] - lo[n])
    # a level for each lever: a short sweep with the others off
    level = {}
    for n in names:
        best = (-np.inf, 0.6)
        for f in (0.4, 0.7, 1.0):
            v = _hold(view, dict(off, **{n: at(n, f)}), test)
            if v is None:
                return
            if v > best[0]:
                best = (v, f)
        level[n] = best[1]
    step = 0.1
    while not view.done:
        cands = {n: dict(off, **{n: at(n, level[n])}) for n in names}
        cands["none"] = dict(off)
        score = {}
        for k, s in cands.items():
            v = _hold(view, s, test)
            if v is None:
                return
            score[k] = v
        run = max(score, key=score.get)
        left = block - test * len(cands)
        if run == "none":
            if _hold(view, off, left) is None:
                return
            continue
        # tune the running lever's level: half the remaining block one step up, half one step down
        up, dn = min(1.0, level[run] + step), max(0.0, level[run] - step)
        v_up = _hold(view, dict(off, **{run: at(run, up)}), max(1, left // 2))
        v_dn = _hold(view, dict(off, **{run: at(run, dn)}), max(1, left - left // 2))
        if v_up is None or v_dn is None:
            return
        if abs(v_up - v_dn) > 1e-9:
            level[run] = up if v_up > v_dn else dn


# ====================================================================== lag-aware tracking of a moving optimum
def _rev(r) -> float:
    return float(r["observation"]["resources"].get("revenue_cents_last_tick", 0.0))


def _cost(r) -> float:
    return float(r["observation"]["resources"].get("cost_cents_last_tick", 0.0))


def estimate_lag(view, names, lo, hi, base: dict, max_lag: int = 4) -> int:
    """Public: how many ticks after a lever moves does revenue respond? Hold the base setting, pulse each lever to
    the other end of its range for one tick, and find the tick after the pulse at which revenue deviated most from
    the ticks before it."""
    devs = np.zeros(max_lag + 1)
    for n in names:
        pre = []
        for _ in range(2):
            if view.done:
                return 1
            pre.append(_rev(view.step({"kind": "levers", "values": base})))
        far = lo[n] if base[n] - lo[n] > hi[n] - base[n] else hi[n]
        seq = [dict(base, **{n: far})] + [base] * max_lag
        post = []
        for s in seq:
            if view.done:
                return 1
            post.append(_rev(view.step({"kind": "levers", "values": s})))
        ref = float(np.mean(pre))
        devs += np.abs(np.asarray(post) - ref)
    return int(np.argmax(devs))


def lag_aware_tracker(view, env=None, window: int = 14, seed: int = 0, model: str = "quadratic", rate: float = 0.15,
                      probe: float = 0.15) -> None:
    """For lever worlds whose optimum moves (hidden regime changes, drifting preferences, rule changes) and whose
    revenue responds to a setting after a lag. It first measures the lag (a pulse test) and the unit cost of each
    lever (from the cost it is charged: cost = sum of unit cost x setting). Then every tick it plays the setting
    that maximises a model of revenue fitted to the last `window` ticks - revenue `lag` ticks after a setting,
    regressed on the setting (each lever's level and square) - minus the known costs, perturbed by a small random
    probe on one lever so the model keeps seeing the response. Old ticks drop out of the window, so a change of
    the mechanism is followed within a window."""
    rng = np.random.default_rng(seed)
    sp = view.action_space()
    props = sp.kinds["levers"]["properties"]["values"]["properties"]
    names = sorted(props)
    L = len(names)
    lo = np.array([float(props[n]["minimum"]) for n in names])
    hi = np.array([float(props[n]["maximum"]) for n in names])
    span = np.maximum(hi - lo, 1e-9)
    base = {n: float(lo[j] + 0.5 * span[j]) for j, n in enumerate(names)}
    lag = estimate_lag(view, names, {n: lo[j] for j, n in enumerate(names)},
                       {n: hi[j] for j, n in enumerate(names)}, base)
    X, C, R = [], [], []                          # settings (fractions), charged cost, revenue per tick
    cur = np.full(L, 0.5)
    unit = None
    while not view.done:
        u = cur.copy()
        j = int(rng.integers(L))
        u[j] = float(np.clip(u[j] + rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.5) * probe, 0.0, 1.0))
        r = view.step({"kind": "levers", "values": {n: float(lo[k] + u[k] * span[k]) for k, n in enumerate(names)}})
        X.append(u)
        C.append(_cost(r))
        R.append(_rev(r))
        # unit costs from what was charged (cost per tick is linear in the settings)
        if len(X) >= L + 2:
            A = np.asarray([lo + x * span for x in X[-3 * L - 3:]])
            unit, *_ = np.linalg.lstsq(np.c_[A, np.ones(len(A))], np.asarray(C[-3 * L - 3:]), rcond=None)
        n_pairs = len(X) - lag
        if n_pairs < 2 * L + 3 or unit is None:
            cur = np.clip(cur + rng.normal(0, 0.15, L), 0.0, 1.0)
            continue
        idx = range(max(0, n_pairs - window), n_pairs)
        Xs = np.asarray([X[i] for i in idx])
        ys = np.asarray([R[i + lag] for i in idx])
        if model == "linear":
            # a local linear model: each lever's marginal revenue at the current operating point; move each lever
            # by `rate` (of its range) in the direction in which marginal revenue exceeds its unit cost
            F = np.c_[np.ones(len(Xs)), Xs]
            coef = np.linalg.solve(F.T @ F + 1e-3 * np.eye(F.shape[1]), F.T @ ys)
            marg = coef[1:] / span - unit[:L]
            sd = np.std(ys) + 1e-9
            cur = np.clip(cur + rate * np.tanh(marg * span / sd), 0.0, 1.0)
            continue
        F = np.c_[np.ones(len(Xs)), Xs, Xs ** 2]
        coef = np.linalg.solve(F.T @ F + 1e-2 * np.eye(F.shape[1]), F.T @ ys)
        # maximise fitted revenue minus cost over a grid of each lever (coordinate-wise, two passes)
        grid = np.linspace(0.0, 1.0, 21)
        best = cur.copy()
        for _ in range(2):
            for k in range(L):
                vals = []
                for g in grid:
                    x = best.copy()
                    x[k] = g
                    f = np.r_[1.0, x, x ** 2] @ coef - float(unit[:L] @ (lo + x * span))
                    vals.append(f)
                best[k] = grid[int(np.argmax(vals))]
        cur = 0.5 * cur + 0.5 * best


def mode_switch_tracker(view, env=None, block: int = 12, test: int = 4, level: float = 0.75) -> None:
    """For worlds in which which of several levers pays changes at a time nobody announces. A mode runs one lever
    at `level` of its range and the others at their minimum. It runs the mode that earned most, and once per block
    of ticks tries the best other mode for `test` ticks, comparing the mean net income of the later half of the try
    with that of the ticks just before it (so effects that take a tick or two to show are seen); a mode that beats
    the running one takes over. The running lever's level is tuned by small trials at the end of each block."""
    sp = view.action_space()
    props = sp.kinds["levers"]["properties"]["values"]["properties"]
    names = sorted(props)
    lo = {n: float(props[n]["minimum"]) for n in names}
    hi = {n: float(props[n]["maximum"]) for n in names}
    lev = {n: level for n in names}

    def mode(n, f=None):
        return {m: (lo[m] + (lev[n] if f is None else f) * (hi[m] - lo[m]) if m == n else lo[m]) for m in names}

    def run(setting, ticks):
        out = []
        for _ in range(ticks):
            if view.done:
                break
            out.append(_net(view.step({"kind": "levers", "values": setting})))
        return out
    score = {}
    for n in names:                                    # try every mode once
        v = run(mode(n), test)
        if not v:
            return
        score[n] = float(np.mean(v[test // 2:]))
    cur = max(score, key=score.get)
    k = 0
    while not view.done:
        base = run(mode(cur), block - test - 2)
        if not base or view.done:
            return
        ref = float(np.mean(base[-(test // 2):]))
        other = names[k % len(names)] if names[k % len(names)] != cur else names[(k + 1) % len(names)]
        k += 1
        v = run(mode(other), test)
        if not v:
            return
        if float(np.mean(v[test // 2:])) > ref:
            cur = other
            continue
        # tune the running lever's level: one tick a step up, one a step down (read after the lag)
        up, dn = min(1.0, lev[cur] + 0.08), max(0.05, lev[cur] - 0.08)
        a = run(mode(cur, up), 1)
        b = run(mode(cur, dn), 1)
        if a and b and a[0] != b[0]:
            lev[cur] = up if a[0] > b[0] else dn


# ====================================================================== interventional prediction
def interleaved_additive_predict(view, env) -> None:
    """Scripted public reference for prediction objectives in which a slow hidden cycle moves the reading: set the
    levers to a design of conditions in an interleaved order (one tick each, cycling, so the slow cycle averages
    out instead of lining up with one condition), fit an additive model - a saturating response in each setting
    (shape chosen on a grid; a straight line is the limit of a large half-saturation) - by least squares, and
    file its predictions for the held-out conditions the objective names. Knows no family; only the objective's
    format and the lever ranges."""
    obj = env._config["objective"]                   # public: the objective text names the held-out conditions
    held = obj["heldout"]
    names = sorted(held[0])
    props = view.action_space().kinds["levers"]["properties"]["values"]["properties"]
    lo = {n: float(props[n]["minimum"]) for n in names}
    hi = {n: float(props[n]["maximum"]) for n in names}
    mid = {n: 0.5 * (lo[n] + hi[n]) for n in names}
    design = []
    for n in names:
        for f in np.linspace(0.0, 1.0, 7):
            design.append(dict(mid, **{n: float(lo[n] + f * (hi[n] - lo[n]))}))
    X, y = [], []
    k = 0
    while view.t < view.horizon - 2 and not view.done:
        s = design[k % len(design)]
        k += 1
        r = view.step({"kind": "levers", "values": s})
        nums = r["observation"]["numbers"]
        if nums:
            X.append([s[n] for n in names])
            y.append(list(nums.values())[0])
    if len(y) < 3 * len(names) + 2 or view.done:
        while not view.done:
            view.step(None)
        return
    X, y = np.asarray(X), np.asarray(y)
    halves = np.r_[np.linspace(0.3, 8.0, 24), 1e6]
    best = None
    import itertools
    for hs in itertools.product(halves, repeat=len(names)):
        F = np.c_[np.ones(len(X)), [[x / (x + h) if h < 1e5 else x for x, h in zip(row, hs)] for row in X]]
        coef, *_ = np.linalg.lstsq(F, y, rcond=None)
        err = float(np.mean((F @ coef - y) ** 2))
        if best is None or err < best[0]:
            best = (err, hs, coef)
    _, hs, coef = best

    def pred(c):
        f = [1.0] + [c[n] / (c[n] + h) if h < 1e5 else c[n] for n, h in zip(names, hs)]
        return float(np.dot(f, coef))
    view.step({"kind": "message", "to": "journal",
               "content": {"predictions": [{"set": c, "value": pred(c)} for c in held]}})
    while not view.done:
        view.step(None)
