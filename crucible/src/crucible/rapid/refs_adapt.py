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


def _fit_additive_sat(X: np.ndarray, y: np.ndarray, names: list[str]):
    import itertools
    halves = np.r_[np.linspace(0.3, 8.0, 24), 1e6]
    best = None
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
    return pred


def settled_law_predict(view, env) -> None:
    """Scripted public reference for prediction objectives on a slow process (a dead time, then a first-order
    approach to where the reading settles): a step test from rest identifies the dead time (first tick the reading
    moves) and the approach rate (least squares on the step response); the design is then run with each condition
    held a few ticks, and the settled value behind every tick is recovered by inverting the identified dynamics
    (y' = y + r (s - y), s = where it settles); an additive saturating law is fitted to those values and its
    predictions for the held-out conditions are filed."""
    obj = env._config["objective"]
    held = obj["heldout"]
    names = sorted(held[0])
    props = view.action_space().kinds["levers"]["properties"]["values"]["properties"]
    lo = {n: float(props[n]["minimum"]) for n in names}
    hi = {n: float(props[n]["maximum"]) for n in names}

    def read(r):
        nums = r["observation"]["numbers"]
        return list(nums.values())[0] if nums else None
    rest = {n: lo[n] for n in names}
    top = {n: hi[n] for n in names}
    ys, xs = [], []
    for _ in range(3):
        ys.append(read(view.step({"kind": "levers", "values": rest})))
        xs.append(rest)
    step = []
    for _ in range(24):
        if view.done:
            return
        v = read(view.step({"kind": "levers", "values": top}))
        step.append(v)
        ys.append(v)
        xs.append(top)
    y0 = float(np.mean([v for v in ys[:3] if v is not None]))
    st = np.asarray([v if v is not None else np.nan for v in step])
    noise = float(np.nanstd(np.diff(st[-6:]))) + 1e-6
    dead = next((k for k, v in enumerate(st) if abs(v - y0) > 3 * noise + 0.3), 1)
    best = None
    for r in np.linspace(0.04, 0.6, 57):
        for dd in range(max(0, dead - 1), dead + 2):
            # with the input switched at the first step tick, after dd ticks of dead time the reading approaches s
            pred_shape = np.array([0.0 if k < dd else 1.0 - (1.0 - r) ** (k - dd + 1) for k in range(len(st))])
            ok = ~np.isnan(st)
            A = np.c_[np.ones(ok.sum()), pred_shape[ok]]
            coef, *_ = np.linalg.lstsq(A, st[ok], rcond=None)
            err = float(np.mean((A @ coef - st[ok]) ** 2))
            if best is None or err < best[0]:
                best = (err, r, dd)
    _, rate, dead = best
    design = []
    for n in names:
        for f in np.linspace(0.0, 1.0, 5):
            mid = {m: 0.5 * (lo[m] + hi[m]) for m in names}
            design.append(dict(mid, **{n: float(lo[n] + f * (hi[n] - lo[n]))}))
    k = 0
    hold = 4
    while view.t < view.horizon - 2 and not view.done:
        s = design[(k // hold) % len(design)]
        k += 1
        ys.append(read(view.step({"kind": "levers", "values": s})))
        xs.append(s)
    # invert the dynamics: the settled value targeted at tick t is s_t = y_t + (y_{t+1} - y_t) / r, driven by the
    # setting `dead` ticks earlier
    X, S = [], []
    for t in range(dead + 1, len(ys) - 1):
        if ys[t] is None or ys[t + 1] is None:
            continue
        src = xs[t - dead] if t - dead >= 0 else xs[0]
        X.append([src[n] for n in names])
        S.append(ys[t] + (ys[t + 1] - ys[t]) / rate)
    if len(S) < 10 or view.done:
        while not view.done:
            view.step(None)
        return
    X, S = np.asarray(X), np.asarray(S)
    # robustness: average the recovered values per setting before fitting (noise is amplified by 1/r)
    keys = {}
    for x, s_ in zip(map(tuple, X), S):
        keys.setdefault(x, []).append(s_)
    Xm = np.asarray(list(keys))
    Sm = np.asarray([float(np.median(v)) for v in keys.values()])
    pred = _fit_additive_sat(Xm, Sm, names)
    view.step({"kind": "message", "to": "journal",
               "content": {"predictions": [{"set": c, "value": pred(c)} for c in held]}})
    while not view.done:
        view.step(None)


# a small library of two-input laws, each with up to two shape parameters (half-saturation h, threshold q)
def _law_basis(form: str, X: np.ndarray, h: float, q: float) -> np.ndarray:
    a, b = X[:, 0], X[:, 1]
    one = np.ones(len(X))
    sat = lambda v: np.maximum(v, 0.0) / (np.maximum(v, 0.0) + h)      # noqa: E731
    step = lambda v: (v >= q).astype(float)                           # noqa: E731
    return {"linear": lambda: np.c_[one, a, b],
            "sat_a+lin_b": lambda: np.c_[one, sat(a), b],
            "lin_a+sat_b": lambda: np.c_[one, a, sat(b)],
            "sat_a*step_b": lambda: np.c_[one, sat(a) * step(b), b],
            "sat_b*step_a": lambda: np.c_[one, sat(b) * step(a), a],
            "lin_a*step_b": lambda: np.c_[one, a * step(b), b],
            "lin_b*step_a": lambda: np.c_[one, b * step(a), a],
            "sat_a+step_b": lambda: np.c_[one, sat(a), step(b), b],
            "sat_a*lin_b": lambda: np.c_[one, sat(a) * b, sat(a), b]}[form]()


LAW_FORMS = ("linear", "sat_a+lin_b", "lin_a+sat_b", "sat_a*step_b", "sat_b*step_a", "lin_a*step_b", "lin_b*step_a",
             "sat_a+step_b", "sat_a*lin_b")


def fit_law_cv(X: np.ndarray, y: np.ndarray, lo: float, hi: float):
    """Choose a law from `LAW_FORMS` (shape parameters on grids) by five-fold cross-validated error, refit on all
    data. Returns predict(Xq)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    folds = np.arange(len(y)) % 5
    H = np.linspace(0.3, 8.0, 16)
    Q = np.linspace(lo + 0.25, hi - 0.25, 20)
    best = None
    for form in LAW_FORMS:
        hs = H if "sat" in form else [1.0]
        qs = Q if "step" in form else [0.0]
        for h in hs:
            for q in qs:
                F = _law_basis(form, X, h, q)
                cv = 0.0
                for k in range(5):
                    tr, te = folds != k, folds == k
                    coef, *_ = np.linalg.lstsq(F[tr], y[tr], rcond=None)
                    cv += float(np.sum((F[te] @ coef - y[te]) ** 2))
                if best is None or cv < best[0]:
                    best = (cv, form, h, q)
    _, form, h, q = best
    coef, *_ = np.linalg.lstsq(_law_basis(form, X, h, q), y, rcond=None)
    return lambda Xq: _law_basis(form, np.asarray(Xq, dtype=float), h, q) @ coef


def interleaved_law_predict(view, env) -> None:
    """Scripted public reference for prediction objectives with two settings whose law may be gated (one setting
    works only beyond a threshold of the other): run a grid of conditions in an interleaved order (one tick each,
    cycling through the grid, so slow hidden cycles average out), choose a law from a small library by
    cross-validation and file its predictions for the held-out conditions the objective names."""
    obj = env._config["objective"]                   # public: the objective text names the held-out conditions
    held = obj["heldout"]
    names = sorted(held[0])
    props = view.action_space().kinds["levers"]["properties"]["values"]["properties"]
    lo = max(float(props[n]["minimum"]) for n in names)
    hi = min(float(props[n]["maximum"]) for n in names)
    grid = [(a, b) for a in np.linspace(lo, hi, 6) for b in np.linspace(lo, hi, 9)]
    order = np.random.default_rng(0).permutation(len(grid))
    X, y = [], []
    k = 0
    while view.t < view.horizon - 2 and not view.done:
        a, b = grid[int(order[k % len(grid)])]
        k += 1
        r = view.step({"kind": "levers", "values": {names[0]: float(a), names[1]: float(b)}})
        nums = r["observation"]["numbers"]
        if nums:
            X.append([a, b])
            y.append(list(nums.values())[0])
    if len(y) >= 12 and not view.done:
        pred = fit_law_cv(np.asarray(X), np.asarray(y), lo, hi)
        Xh = np.asarray([[c[names[0]], c[names[1]]] for c in held])
        view.step({"kind": "message", "to": "journal",
                   "content": {"predictions": [{"set": c, "value": float(v)} for c, v in zip(held, pred(Xh))]}})
    while not view.done:
        view.step(None)


def arx_settled_predict(view, env, order: int = 2, lags: int = 2, hold: tuple = (2, 6), seed: int = 0) -> None:
    """Scripted public reference for predicting where a process with feedback settles: hold random settings for a
    few ticks each (a persistently exciting design over the settable range), fit a dynamic model of the reading -
    its own last `order` values plus each setting's current and last `lags` values through a saturating basis
    (shape by cross-validation) - and file the model's steady state at the held-out conditions:
    y* = (c + sum_b b phi(x)) / (1 - sum_a a)."""
    rng = np.random.default_rng(seed)
    obj = env._config["objective"]
    held = obj["heldout"]
    names = sorted(held[0])
    props = view.action_space().kinds["levers"]["properties"]["values"]["properties"]
    lo = np.array([float(props[n]["minimum"]) for n in names])
    hi = np.array([float(props[n]["maximum"]) for n in names])
    X, y = [], []
    while view.t < view.horizon - 2 and not view.done:
        s = lo + rng.random(len(names)) * (hi - lo)
        for _ in range(int(rng.integers(hold[0], hold[1] + 1))):
            if view.t >= view.horizon - 2 or view.done:
                break
            r = view.step({"kind": "levers", "values": {n: float(v) for n, v in zip(names, s)}})
            nums = r["observation"]["numbers"]
            X.append(s.copy())
            y.append(list(nums.values())[0] if nums else np.nan)
    X, y = np.asarray(X), np.asarray(y, dtype=float)
    p = max(order, lags)
    if len(y) < 20 + p or view.done:
        while not view.done:
            view.step(None)
        return

    def basis(x, h):
        return np.c_[x, x / (x + h)] if h < 1e5 else x[:, None] if x.ndim == 1 else x

    best = None
    for h in (0.5, 1.0, 2.0, 4.0, 8.0, 1e6):
        rows, tgt = [], []
        for t in range(p, len(y)):
            if not np.isfinite(y[t - order:t + 1]).all():
                continue
            feats = [1.0] + [y[t - k] for k in range(1, order + 1)]
            for l in range(lags + 1):
                for j in range(len(names)):
                    xv = X[t - l, j]
                    feats += [xv, xv / (xv + h)] if h < 1e5 else [xv]
            rows.append(feats)
            tgt.append(y[t])
        F, T = np.asarray(rows), np.asarray(tgt)
        if len(T) < F.shape[1] + 5:
            continue
        folds = np.arange(len(T)) % 5
        cv = 0.0
        for k in range(5):
            tr, te = folds != k, folds == k
            coef = np.linalg.lstsq(F[tr], T[tr], rcond=None)[0]
            cv += float(np.sum((F[te] @ coef - T[te]) ** 2))
        if best is None or cv < best[0]:
            best = (cv, h, np.linalg.lstsq(F, T, rcond=None)[0])
    if best is None:
        while not view.done:
            view.step(None)
        return
    _, h, coef = best
    a_sum = float(np.sum(coef[1:1 + order]))

    def steady(cond):
        tot = coef[0]
        k = 1 + order
        for _ in range(lags + 1):
            for n in names:
                xv = float(cond[n])
                if h < 1e5:
                    tot += coef[k] * xv + coef[k + 1] * xv / (xv + h) if xv + h != 0 else 0.0
                    k += 2
                else:
                    tot += coef[k] * xv
                    k += 1
        return float(tot / max(1e-3, 1.0 - a_sum))
    view.step({"kind": "message", "to": "journal",
               "content": {"predictions": [{"set": c, "value": steady(c)} for c in held]}})
    while not view.done:
        view.step(None)


def latch_open_and_hold(view, env=None, hold: int = 5) -> None:
    """Scripted public reference for a pathway behind a hidden latch (hysteresis). Push both levers to the top
    (whichever lever is the latch, this opens it), then lower each lever in turn to mid-range for a few ticks: the
    one whose lowering leaves revenue where it was is the latch (a latch stays on above a lower level), the other
    is the gated lever. Then walk the latch lever down in steps while revenue stays up and hold the lowest setting
    that kept the pathway open (re-opening it if a step closed it), and tune the gated lever by small trials on
    net income. Every hold lasts several ticks and is read on its last ticks, so lagged effects count."""
    sp = view.action_space()
    props = sp.kinds["levers"]["properties"]["values"]["properties"]
    names = sorted(props)
    lo = {n: float(props[n]["minimum"]) for n in names}
    hi = {n: float(props[n]["maximum"]) for n in names}

    def run(setting, ticks=hold):
        out = []
        for _ in range(ticks):
            if view.done:
                break
            out.append(view.step({"kind": "levers", "values": setting}))
        return out

    def rev(rs, k=2):
        v = [float(r["observation"]["resources"].get("revenue_cents_last_tick", 0.0)) for r in rs[-k:]]
        return float(np.mean(v)) if v else 0.0

    def net(rs, k=2):
        v = [_net(r) for r in rs[-k:]]
        return float(np.mean(v)) if v else 0.0
    top = dict(hi)
    base = rev(run(top))
    n1, n2 = names[0], names[1]
    half = {n: 0.5 * (lo[n] + hi[n]) for n in names}
    r1 = rev(run(dict(top, **{n1: half[n1]}), hold - 1))          # the first lever to mid-range
    r2 = rev(run({n1: half[n1], n2: half[n2]}, hold - 1))          # then the second as well
    if view.done:
        return
    # lowering the latch lever leaves revenue where it was; lowering the gated lever lowers it
    A = n1 if base - r1 < r1 - r2 else n2
    B = [m for m in names if m != A][0]
    xb = lo[B] + 0.7 * (hi[B] - lo[B])
    keep = 0.5 * (lo[A] + hi[A])
    step = 0.1 * (hi[A] - lo[A])
    open_rev = base
    x = keep
    while x - step >= lo[A] - 1e-9 and not view.done:
        rs = run({A: x - step, B: xb})
        if not rs:
            return
        if rev(rs) < 0.7 * open_rev:                     # it closed: open again and hold the last good level
            run({A: hi[A], B: xb}, 4)
            break
        x -= step
        keep = x
    db = 0.08 * (hi[B] - lo[B])
    while not view.done:                                 # tune the gated lever on net income
        up = run({A: keep, B: min(hi[B], xb + db)}, 3)
        dn = run({A: keep, B: max(lo[B], xb - db)}, 3)
        if not up or not dn:
            return
        xb = min(hi[B], xb + db) if net(up, 1) > net(dn, 1) else max(lo[B], xb - db)
        run({A: keep, B: xb}, 4)


def drop_switch_tracker(view, env=None, test: int = 5, level: float = 0.7, drop: float = 0.6, settle: int = 4,
                        tune_every: int = 0) -> None:
    """For worlds where which lever pays changes at a time nobody announces and the change shows as a fall of
    income under the current setting. Modes run one lever at `level` of its range (the others at their minimum).
    Each mode is tried for `test` ticks at the start (read on its last two ticks: effects can take a few ticks to
    show) and the best is run; once the running mode has settled (`settle` ticks after any change), a fast and a
    slow running average of observed net income are kept, and when the fast one falls below `drop` x the slow one
    for two ticks in a row every mode is tried again and the best taken over. The running lever's level is tuned
    by small trials every twelve ticks."""
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

    def choose(first: str | None = None):
        score = {}
        order = ([first] if first else []) + [n for n in names if n != first]
        for n in order:
            v = run(mode(n), test)
            if not v:
                return None
            score[n] = float(np.mean(v[-2:]))
        return max(score, key=score.get)
    cur = choose()
    if cur is None:
        return
    since, fast, slow, low, k = 0, None, None, 0, 0
    while not view.done:
        v = run(mode(cur), 1)
        if not v:
            return
        k += 1
        since += 1
        if since <= settle:                       # just changed: wait until the change has worked through
            if slow is None and since == settle:
                fast = slow = v[0]
            continue
        if slow is None:
            fast = slow = v[0]
        fast = 0.5 * fast + 0.5 * v[0]
        slow = 0.9 * slow + 0.1 * v[0]
        low = low + 1 if fast < slow - (1.0 - drop) * abs(slow) else 0
        if low >= 2:
            # income fell: try the other modes in turn and take the first that earns clearly more than the
            # running mode does now (if none does, keep the best of them)
            now = fast
            best = (now, cur)
            for n in [m for m in names if m != cur]:
                v2 = run(mode(n), test)
                if not v2:
                    return
                sc = float(np.mean(v2[-2:]))
                if sc > best[0]:
                    best = (sc, n)
                if sc > now + 0.5 * abs(slow - now):
                    break
            cur, since, low, fast, slow = best[1], 0, 0, None, None
            continue
        if tune_every and k % tune_every == 0:
            # tune the level; the trial ticks count as observations of the running mode too (they are close to it)
            up, dn = min(1.0, lev[cur] + 0.08), max(0.05, lev[cur] - 0.08)
            a = run(mode(cur, up), 3)
            b = run(mode(cur, dn), 3)
            for x in a + b:
                fast = 0.5 * fast + 0.5 * x
                slow = 0.9 * slow + 0.1 * x
            if len(a) == 3 and len(b) == 3 and a[-1] != b[-1]:
                lev[cur] = up if a[-1] > b[-1] else dn


# ====================================================================== additive shape laws with input lags
def _shape_cols(shape: tuple, x: np.ndarray) -> np.ndarray:
    """Basis columns of one setting for a candidate shape: ("lin",), ("sat", h), ("quad",), ("step", q),
    ("sat_lin", h) - a saturating term plus a linear one."""
    kind = shape[0]
    if kind == "lin":
        return x[:, None]
    if kind == "quad":
        return np.c_[x, x * x]
    if kind == "sat":
        xp = np.maximum(x, 0.0)
        return (xp / (xp + shape[1]))[:, None]
    if kind == "sat_lin":
        xp = np.maximum(x, 0.0)
        return np.c_[xp / (xp + shape[1]), x]
    if kind == "step":
        return (x >= shape[1]).astype(float)[:, None]
    raise ValueError(shape)


def _shape_options(lo: float, hi: float) -> list[tuple]:
    span = hi - lo
    out = [("lin",), ("quad",)]
    out += [("sat", float(h)) for h in np.geomspace(0.05 * span, 1.2 * span, 10)]
    out += [("sat_lin", float(h)) for h in np.geomspace(0.05 * span, 1.2 * span, 6)]
    out += [("step", float(q)) for q in np.linspace(lo + 0.1 * span, hi - 0.1 * span, 9)]
    return out


def _design(Xl: list[np.ndarray], shapes: list[tuple]) -> np.ndarray:
    """Xl[l] holds the settings l ticks before each reading (n x J); one block of shape columns per lag."""
    cols = [np.ones((len(Xl[0]), 1))]
    for Xk in Xl:
        for j, sh in enumerate(shapes):
            cols.append(_shape_cols(sh, Xk[:, j]))
    return np.hstack(cols)


def _cv(F: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    folds = np.arange(len(y)) % k
    err = 0.0
    for i in range(k):
        tr, te = folds != i, folds == i
        coef, *_ = np.linalg.lstsq(F[tr], y[tr], rcond=None)
        err += float(np.sum((F[te] @ coef - y[te]) ** 2))
    return err


def fit_additive_lagged(X: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray, lags: int,
                        Z: np.ndarray | None = None, gain: float = 0.03):
    """An additive law y_t = c + sum_l sum_j b_lj phi_j(x_j,t-l), l = 0..L, with one shape phi_j per setting and the
    number of lags L <= `lags` chosen by five-fold cross-validation (coordinate search over `_shape_options`, two
    passes, for every L on the same readings; a more complex shape or a longer lag must lower the cross-validated
    error by `gain`). `Z` (optional) holds covariates that do not respond to the settings (readings of exogenous
    cycles); they enter linearly and are held at their means for predictions. Returns (predict_settled,
    (shapes, L)): predict_settled(Xq) is the law's response with each condition held (all lags at the condition)."""
    n, J = X.shape
    M = int(lags)
    yy = y[M:]
    ok = np.isfinite(yy)
    if Z is not None and Z.size:
        Zs = Z[M:]
        ok &= np.isfinite(Zs).all(axis=1)
        Zs = Zs[ok]
        zbar = Zs.mean(axis=0)
    else:
        Zs, zbar = None, None
    yy = yy[ok]

    def des(Xl, shapes, Zm=None):
        F = _design(Xl, shapes)
        if Zs is not None:
            F = np.c_[F, Zs if Zm is None else Zm]
        return F
    best_all = None
    for L in range(M + 1):
        Xl = [X[M - l:n - l][ok] for l in range(L + 1)]
        shapes = [("lin",)] * J
        best = _cv(des(Xl, shapes), yy)
        for _ in range(2):
            for j in range(J):
                for sh in _shape_options(float(lo[j]), float(hi[j])):
                    cand = list(shapes)
                    cand[j] = sh
                    e = _cv(des(Xl, cand), yy)
                    if e < best * (1.0 - gain):
                        best, shapes = e, cand
        if best_all is None or best < best_all[0] * (1.0 - gain):
            best_all = (best, L, shapes, Xl)
    _, L, shapes, Xl = best_all
    coef, *_ = np.linalg.lstsq(des(Xl, shapes), yy, rcond=None)

    def predict(Xq):
        Xq = np.asarray(Xq, dtype=float)
        Zm = None if Zs is None else np.repeat(zbar[None, :], len(Xq), axis=0)
        return des([Xq] * (L + 1), shapes, Zm) @ coef
    return predict, (shapes, L)


def _exogenous(X: np.ndarray, R: np.ndarray, alpha: float = 0.01) -> list[int]:
    """Columns of R (other readings) that do not respond to the randomised settings X: the F test of a regression
    of each reading on the settings (current and one lag) is not significant at `alpha`."""
    from math import exp, lgamma, log
    out = []
    n = len(X)
    F = np.c_[np.ones(n - 1), X[1:], X[:-1]]
    p = F.shape[1] - 1
    for k in range(R.shape[1]):
        r = R[1:, k]
        if not np.isfinite(r).all() or np.std(r) < 1e-9:
            continue
        coef, *_ = np.linalg.lstsq(F, r, rcond=None)
        rss1 = float(np.sum((r - F @ coef) ** 2))
        rss0 = float(np.sum((r - r.mean()) ** 2))
        d2 = n - 1 - p - 1
        if d2 <= 0 or rss1 <= 0:
            continue
        f = ((rss0 - rss1) / p) / (rss1 / d2)
        # upper tail of F(p, d2) by the regularised incomplete beta (continued fraction)
        x = d2 / (d2 + p * f)
        a, b = d2 / 2.0, p / 2.0

        def betacf(a, b, x, it=200):
            qab, qap, qam = a + b, a + 1.0, a - 1.0
            c, d = 1.0, 1.0 - qab * x / qap
            d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
            h = d
            for m in range(1, it):
                m2 = 2 * m
                aa = m * (b - m) * x / ((qam + m2) * (a + m2))
                d = 1.0 + aa * d
                d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
                c = 1.0 + aa / c if abs(c) > 1e-30 else 1e-30
                h *= d * c
                aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
                d = 1.0 + aa * d
                d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
                c = 1.0 + aa / c if abs(c) > 1e-30 else 1e-30
                de = d * c
                h *= de
                if abs(de - 1.0) < 3e-12:
                    break
            return h
        if x <= 0.0:
            pval = 0.0
        elif x >= 1.0:
            pval = 1.0
        else:
            front = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x))
            pval = front * betacf(a, b, x) / a if x < (a + 1.0) / (a + b + 2.0) else \
                1.0 - exp(lgamma(a + b) - lgamma(a) - lgamma(b) + b * log(1.0 - x) + a * log(x)) * \
                betacf(b, a, 1.0 - x) / b
        if pval > alpha:
            out.append(k)
    return out


def fir_shape_predict(view, env, lags: int = 1, hold: tuple = (1, 1), seed: int = 0, covariates: bool = True,
                      gain: float = 0.06) -> None:
    """Scripted public reference for prediction objectives on processes without feedback whose settings act after a
    few ticks: hold uniformly random settings for a few ticks each (a design that randomises the settings, so
    hidden demand cycles and past regimes cannot confound it), fit an additive law of the reading on each
    setting's current and `lags` previous values with one shape per setting (linear, quadratic, saturating,
    saturating plus linear, or a step; chosen by cross-validation), and file the law's response with each held-out
    condition held. Other readings that do not respond to the settings (an F test under the randomised design:
    readings of exogenous cycles) enter as covariates, which removes cycle noise without biasing the effects. The
    reading is the gauge the objective names (public: the objective text says what to predict)."""
    rng = np.random.default_rng(seed)
    obj = env._config["objective"]
    held = obj["heldout"]
    gauge = obj.get("gauge")
    names = sorted(held[0])
    props = view.action_space().kinds["levers"]["properties"]["values"]["properties"]
    lo = np.array([float(props[n]["minimum"]) for n in names])
    hi = np.array([float(props[n]["maximum"]) for n in names])
    X, y, R = [], [], []
    others = None
    while view.t < view.horizon - 2 and not view.done:
        s = lo + rng.random(len(names)) * (hi - lo)
        for _ in range(int(rng.integers(hold[0], hold[1] + 1))):
            if view.t >= view.horizon - 2 or view.done:
                break
            r = view.step({"kind": "levers", "values": {n: float(v) for n, v in zip(names, s)}})
            nums = r["observation"]["numbers"]
            key = gauge if gauge in nums else (next(iter(nums)) if nums else None)
            if others is None:
                others = sorted(k for k in nums if k != key)
            X.append(s.copy())
            y.append(nums.get(key, np.nan) if key else np.nan)
            R.append([float(nums.get(k, np.nan)) for k in others])
    X, y = np.asarray(X), np.asarray(y, dtype=float)
    if len(y) >= 20 + lags and not view.done:
        Z = None
        if covariates and others:
            Rm = np.asarray(R, dtype=float)
            keep = _exogenous(X, Rm)
            Z = Rm[:, keep] if keep else None
        pred, _ = fit_additive_lagged(X, y, lo, hi, lags, Z=Z, gain=gain)
        Xh = np.asarray([[float(c[n]) for n in names] for c in held])
        view.step({"kind": "message", "to": "journal",
                   "content": {"predictions": [{"set": c, "value": float(v)} for c, v in zip(held, pred(Xh))]}})
    while not view.done:
        view.step(None)


def fir_shape_predict_lagged(view, env) -> None:
    """`fir_shape_predict` for worlds whose settings act through lagged paths: settings held one or two ticks, up
    to three lags of each setting."""
    fir_shape_predict(view, env, lags=3, hold=(1, 2), gain=0.04)


# ====================================================================== seeding communities as they come online
def seed_as_communities_open(view, env=None, step: int = 2, wait: int = 1, cap: int = 8, reserve: int = 2,
                             grown: int = 3) -> None:
    """Scripted public reference for seeding a product in communities that come online at different times
    (complex contagion needs a critical mass, unknown per community): a community that shows its members gets
    `step` seeds; every `wait` periods, a seeded community whose adopters have not grown past its seeds plus
    `grown` gets `step` more, up to `cap` seeds per community, while `reserve` seeds are kept for each community
    not online yet. Public: community entities (members once online, adopters) and the seeds left."""
    sp = view.action_space()
    names = list(sp.kinds["allocate"]["properties"]["amounts"]["properties"])
    seeded = {n: 0 for n in names}
    when: dict = {}
    r = view.step(None)
    while not view.done:
        obs = r["observation"]
        left = int(obs["resources"].get("seeds_left", 0))
        ents = {x["id"]: x for x in obs["entities"] if x.get("kind") == "community"}
        online = [n for n in names if ents.get(n, {}).get("members", 0) > 0]
        unseen = sum(1 for n in names if n not in online)
        amt = {}
        for n in online:
            spare = left - reserve * unseen
            if seeded[n] == 0 and left >= step:
                k = step
            elif 0 < seeded[n] < cap and view.t - when[n] >= wait and ents[n]["adopters"] < seeded[n] + grown \
                    and spare >= step:
                k = step
            else:
                continue
            amt[n] = float(k)
            left -= k
        for n, k in amt.items():
            seeded[n] += int(k)
            when[n] = view.t
        r = view.step({"kind": "allocate", "amounts": amt} if amt else None)


# ====================================================================== prediction under a changed regime
def _change_point(X: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray, margin: int = 10) -> int:
    """The split that best separates two regimes of the reading's law: for every split with at least `margin`
    readings on each side, a quadratic law in each setting (plus their product) is fitted on each side; the split
    with the smallest total squared error, if it beats one law for all by a clear margin (F test at 1%), else 0."""
    n = len(y)
    if n < 2 * margin + 5:
        return 0
    Z = (X - lo) / np.maximum(hi - lo, 1e-9)
    F = np.c_[np.ones(n), Z, Z * Z, Z[:, :1] * Z[:, 1:2]] if X.shape[1] >= 2 else np.c_[np.ones(n), Z, Z * Z]
    p = F.shape[1]

    def sse(rows):
        coef, *_ = np.linalg.lstsq(F[rows], y[rows], rcond=None)
        return float(np.sum((F[rows] @ coef - y[rows]) ** 2))
    allr = np.arange(n)
    s0 = sse(allr)
    best, bt = np.inf, 0
    for tau in range(margin, n - margin):
        s = sse(allr[:tau]) + sse(allr[tau:])
        if s < best:
            best, bt = s, tau
    f = ((s0 - best) / p) / max(best / max(n - 2 * p, 1), 1e-12)
    return bt if f > 3.5 else 0


def fir_shape_predict_recent(view, env, lags: int = 1, gain: float = 0.06, seed: int = 0) -> None:
    """`fir_shape_predict` for a world whose law may have changed during the episode, asked about the current
    regime: settings are randomised every tick; at the end the most likely change point of the law is located
    (`_change_point`) and the additive law is fitted to the readings after it only."""
    rng = np.random.default_rng(seed)
    obj = env._config["objective"]
    held = obj["heldout"]
    gauge = obj.get("gauge")
    names = sorted(held[0])
    props = view.action_space().kinds["levers"]["properties"]["values"]["properties"]
    lo = np.array([float(props[n]["minimum"]) for n in names])
    hi = np.array([float(props[n]["maximum"]) for n in names])
    X, y = [], []
    while view.t < view.horizon - 2 and not view.done:
        s = lo + rng.random(len(names)) * (hi - lo)
        r = view.step({"kind": "levers", "values": {n: float(v) for n, v in zip(names, s)}})
        nums = r["observation"]["numbers"]
        key = gauge if gauge in nums else (next(iter(nums)) if nums else None)
        X.append(s.copy())
        y.append(nums.get(key, np.nan) if key else np.nan)
    X, y = np.asarray(X), np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    if len(y) >= 20 and not view.done:
        tau = _change_point(X, y, lo, hi)
        Xr, yr = X[tau:], y[tau:]
        pred, _ = fit_additive_lagged(Xr, yr, lo, hi, lags, gain=gain)
        Xh = np.asarray([[float(c[n]) for n in names] for c in held])
        view.step({"kind": "message", "to": "journal",
                   "content": {"predictions": [{"set": c, "value": float(v)} for c, v in zip(held, pred(Xh))]}})
    while not view.done:
        view.step(None)


def fir_shape_predict_recent_lagged(view, env) -> None:
    """`fir_shape_predict_recent` with up to three lags of each setting (worlds with short feedback or delays)."""
    fir_shape_predict_recent(view, env, lags=3, gain=0.04)


# ====================================================================== look-alike markets as prediction tasks
def _logit_market(X: np.ndarray, y: np.ndarray, pi: int, folds: np.ndarray):
    """units = M * q / (q + h) * (b + (1 - b) / (1 + exp(c (p - mid)))): a logit demand curve in price p times a
    saturating response to promotion q (shapes on grids, M by least squares). Returns (predict, cv error)."""
    p = X[:, pi]
    q = np.maximum(X[:, 1 - pi], 0.0)
    best = None
    for h in (0.5, 1.5, 3.0, 6.0, 12.0):
        sq = (q + 1.0) / (q + 1.0 + h)
        for mid in np.linspace(1.0, 10.0, 19):
            for cc in (0.3, 0.6, 1.0, 1.6, 2.5):
                for b in (0.0, 0.02, 0.05):
                    g = sq * (b + (1.0 - b) / (1.0 + np.exp(cc * (p - mid))))
                    M = float(np.sum(g * y) / max(np.sum(g * g), 1e-12))
                    err = float(np.sum((M * g - y) ** 2))
                    if best is None or err < best[0]:
                        best = (err, h, mid, cc, b)
    _, h, mid, cc, b = best

    def design(Xq):
        pq, qq = Xq[:, pi], np.maximum(Xq[:, 1 - pi], 0.0)
        return (qq + 1.0) / (qq + 1.0 + h) * (b + (1.0 - b) / (1.0 + np.exp(cc * (pq - mid))))
    cv = 0.0
    for k in range(5):
        tr, te = folds != k, folds == k
        g = design(X[tr])
        M = float(np.sum(g * y[tr]) / max(np.sum(g * g), 1e-12))
        cv += float(np.sum((M * design(X[te]) - y[te]) ** 2))
    g = design(X)
    M = float(np.sum(g * y) / max(np.sum(g * g), 1e-12))
    return (lambda Xq: M * design(np.asarray(Xq, dtype=float))), cv


def market_predict(view, env, seed: int = 0, gain: float = 0.04) -> None:
    """Scripted public reference for predicting a market's sales at held-out prices, promotion levels and
    certification states: price, promotion and the certification decision are randomised every tick; for each
    certification state, an additive law of log units (one shape per lever by cross-validation, the rival's price
    index as a covariate if it does not respond to the levers) is fitted, so the price response and the promotion
    response multiply; predictions are filed for the held-out conditions."""
    rng = np.random.default_rng(seed)
    obj = env._config["objective"]
    held = obj["heldout"]
    gauge = obj.get("gauge")
    sp = view.action_space()
    props = sp.kinds["levers"]["properties"]["values"]["properties"]
    lv = sorted(props)
    (dname, opts), = sp.limits["decisions"].items()
    lo = np.array([float(props[n]["minimum"]) for n in lv])
    hi = np.array([float(props[n]["maximum"]) for n in lv])
    X, y, C, R = [], [], [], []
    others = None
    while view.t < view.horizon - 2 and not view.done:
        s = lo + rng.random(len(lv)) * (hi - lo)
        ch = int(rng.integers(len(opts)))
        r = view.step({"kind": "sequence", "steps": [
            {"kind": "discrete", "decision": dname, "choice": opts[ch]},
            {"kind": "levers", "values": {n: float(v) for n, v in zip(lv, s)}}]})
        nums = r["observation"]["numbers"]
        if others is None:
            others = sorted(k for k in nums if k != gauge)
        X.append(s.copy())
        y.append(float(nums.get(gauge, np.nan)))
        C.append(ch)
        R.append([float(nums.get(k, np.nan)) for k in others])
    X, y, C, R = np.asarray(X), np.asarray(y), np.asarray(C), np.asarray(R, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() >= 20 and not view.done:
        keep = _exogenous(X[ok], R[ok]) if R.size else []
        pi = lv.index(next(n for n in lv if float(props[n]["minimum"]) == 1.0)) if any(
            float(props[n]["minimum"]) == 1.0 for n in lv) else 0          # the price lever's range starts at 1
        preds = {}
        for ch in range(len(opts)):
            m = ok & (C == ch)
            if m.sum() < 10:
                continue
            Z = R[m][:, keep] if keep else None
            ly = np.log(np.maximum(y[m], 0.05))
            f_add, _ = fit_additive_lagged(X[m], ly, lo, hi, 0, Z=Z, gain=gain)
            # two model classes compete by five-fold cross-validated error on units: the additive law of log units,
            # and a logit demand curve in price times a saturating response to promotion
            Xm, ym = X[m], y[m]
            folds = np.arange(len(ym)) % 5
            cv_add = 0.0
            for k in range(5):
                tr, te = folds != k, folds == k
                fk, _ = fit_additive_lagged(Xm[tr], np.log(np.maximum(ym[tr], 0.05)), lo, hi, 0, gain=gain)
                cv_add += float(np.sum((np.exp(fk(Xm[te])) - ym[te]) ** 2))
            f_log, cv_log = _logit_market(Xm, ym, pi, folds)
            preds[ch] = (lambda Xq, f=f_add: np.exp(f(Xq))) if cv_add <= cv_log else f_log
        out = []
        for c in held:
            ch = opts.index(c[dname])
            f = preds.get(ch) or next(iter(preds.values()))
            out.append({"set": c, "value": float(f(np.asarray([[float(c[n]) for n in lv]]))[0])})
        view.step({"kind": "message", "to": "journal", "content": {"predictions": out}})
    while not view.done:
        view.step(None)


# ====================================================================== adaptive rival: the long-run outcome of a price
def rival_model_predict(view, env, seed: int = 0, hold: tuple = (1, 4)) -> None:
    """Scripted public reference (D-64: it models the opponent's policy class): prices are held at random levels for
    a few ticks each; the rival's posted price is modelled as following an average of your past prices that it
    remembers at an unknown rate (chosen by fit from a grid): o' = a + b o + c m, m' = m + r (x - m); your units are
    modelled as a logit choice between your price, the rival's and not buying (market size, price sensitivity and
    the outside option fitted on grids and by least squares). Holding a price p, the rival settles at
    o* = (a + c p) / (1 - b), and the filed prediction is the units law at (p, o*)."""
    rng = np.random.default_rng(seed)
    obj = env._config["objective"]
    held = obj["heldout"]
    gauge = obj.get("gauge")
    name = next(iter(held[0]))
    props = view.action_space().kinds["levers"]["properties"]["values"]["properties"]
    lo, hi = float(props[name]["minimum"]), float(props[name]["maximum"])
    P, U, O = [], [], []
    other = None
    while view.t < view.horizon - 2 and not view.done:
        x = float(lo + rng.random() * (hi - lo))
        for _ in range(int(rng.integers(hold[0], hold[1] + 1))):
            if view.t >= view.horizon - 2 or view.done:
                break
            r = view.step({"kind": "levers", "values": {name: x}})
            nums = r["observation"]["numbers"]
            if other is None:
                other = next(k for k in nums if k != gauge)
            P.append(x)
            U.append(float(nums.get(gauge, np.nan)))
            O.append(float(nums.get(other, np.nan)))
    P, U, O = np.asarray(P), np.asarray(U), np.asarray(O)
    if len(P) >= 20 and not view.done:
        best = None
        for rate in np.linspace(0.03, 0.6, 20):
            m = np.zeros(len(P))
            mm = O[0]
            for t in range(len(P)):
                m[t] = mm
                mm = mm + rate * (P[t] - mm)          # the rival remembers prices up to the last one
            Y = O[1:]
            Xr = np.c_[np.ones(len(Y)), O[:-1], m[1:]]
            coef, *_ = np.linalg.lstsq(Xr, Y, rcond=None)
            err = float(np.sum((Xr @ coef - Y) ** 2))
            if best is None or err < best[0]:
                best = (err, coef)
        a, b, c = best[1]
        ok = np.isfinite(U) & np.isfinite(O)
        fit = None
        for beta in np.linspace(0.2, 1.6, 15):
            for K in np.geomspace(0.01, 5.0, 14):
                share = np.exp(-beta * P[ok]) / (np.exp(-beta * P[ok]) + np.exp(-beta * O[ok]) + K)
                Mk = float(np.sum(share * U[ok]) / max(np.sum(share * share), 1e-12))
                err = float(np.sum((Mk * share - U[ok]) ** 2))
                if fit is None or err < fit[0]:
                    fit = (err, beta, K, Mk)
        _, beta, K, Mk = fit
        out = []
        for cond in held:
            p_ = float(cond[name])
            o_ = (a + c * p_) / max(1e-6, 1.0 - b)
            o_ = float(np.clip(o_, min(O.min(), lo), max(O.max(), hi)))
            v = Mk * np.exp(-beta * p_) / (np.exp(-beta * p_) + np.exp(-beta * o_) + K)
            out.append({"set": cond, "value": float(v)})
        view.step({"kind": "message", "to": "journal", "content": {"predictions": out}})
    while not view.done:
        view.step(None)


# ====================================================================== drifting preferences: a bump that moves
def _bump(x: np.ndarray, c, w: float) -> np.ndarray:
    """A liked band: zero below c - w, rising to 1 at c, falling slowly beyond (r e^(1-r), r = (x - c + w) / w)."""
    r = np.maximum(x - c + w, 0.0) / w
    return r * np.exp(1.0 - r)


def drifting_bump_predict(view, env, seed: int = 0) -> None:
    """Scripted public reference for predicting sales under the taste at the end of an episode in which the liked
    level of a design setting drifts smoothly: settings are randomised every tick; sales are modelled as
    A + B bump(design - c(t), w) + C sat(promotion), the liked level moving as c(t) = c0 + D (1 - q^t) (a drift that
    slows down); c0, D, q, the band width w and the promotion's half-saturation come from grids, A, B, C by least
    squares; the law with c at the end of the episode is filed. Public: the objective names the design setting
    first, the promotion second."""
    rng = np.random.default_rng(seed)
    obj = env._config["objective"]
    held = obj["heldout"]
    gauge = obj.get("gauge")
    xn, pn = list(held[0])[:2]
    props = view.action_space().kinds["levers"]["properties"]["values"]["properties"]
    lo = {n: float(props[n]["minimum"]) for n in (xn, pn)}
    hi = {n: float(props[n]["maximum"]) for n in (xn, pn)}
    X, Q, y, T = [], [], [], []
    while view.t < view.horizon - 2 and not view.done:
        x = float(lo[xn] + rng.random() * (hi[xn] - lo[xn]))
        q = float(lo[pn] + rng.random() * (hi[pn] - lo[pn]))
        r = view.step({"kind": "levers", "values": {xn: x, pn: q}})
        nums = r["observation"]["numbers"]
        X.append(x)
        Q.append(q)
        y.append(float(nums.get(gauge, np.nan)))
        T.append(float(view.t))
    X, Q, y, T = map(np.asarray, (X, Q, y, T))
    ok = np.isfinite(y)
    X, Q, y, T = X[ok], Q[ok], y[ok], T[ok]
    if len(y) >= 20 and not view.done:
        H = float(view.horizon)
        best = None
        sat_cols = {h: Q / (Q + h) for h in (1.0, 2.5, 5.0, 10.0)}
        for qd in (0.93, 0.96, 0.98, 0.995):
            g = 1.0 - qd ** T
            for c0 in np.arange(lo[xn], hi[xn] + 0.01, 0.5):
                for Dl in np.arange(-9.0, 9.01, 0.5):
                    c = c0 + Dl * g
                    for w in (1.5, 2.5, 3.5, 5.0):
                        bmp = _bump(X, c, w)
                        for h, sc in sat_cols.items():
                            F_ = np.c_[np.ones(len(y)), bmp, sc]
                            coef, *_ = np.linalg.lstsq(F_, y, rcond=None)
                            err = float(np.sum((F_ @ coef - y) ** 2))
                            if best is None or err < best[0]:
                                best = (err, qd, c0, Dl, w, h, coef)
        _, qd, c0, Dl, w, h, coef = best
        c_end = c0 + Dl * (1.0 - qd ** H)
        out = []
        for cond in held:
            x, q = float(cond[xn]), float(cond[pn])
            v = coef[0] + coef[1] * float(_bump(np.array([x]), c_end, w)[0]) + coef[2] * max(q, 0.0) / (max(q, 0.0) + h)
            out.append({"set": cond, "value": float(v)})
        view.step({"kind": "message", "to": "journal", "content": {"predictions": out}})
    while not view.done:
        view.step(None)


# ====================================================================== technology transition: a learning curve
def learning_transition_predict(view, env, seed: int = 0, settle: int = 60) -> None:
    """Scripted public reference for predicting output under the conditions at the end of an episode in which an
    old process may lose its value at some period and a new process's yield grows with experience of using it
    (learning by doing): settings are randomised (held one to three ticks); output is modelled as y = z, or as a
    saturating y = cap z / (z + K), with z = c0 + c1 [before tau] sat(a) + c2 [after tau] sat(a) + c3 b E(t) and E an
    experience stock of past use of the new process that decays at a rate; tau, the rate, the saturations and the
    link come from grids, the c's by least squares.
    Holding (a, b) from the start, experience reaches b (1 - (1 - rate)^settle) / rate, and the law after tau is
    filed. Public: the objective names the established process's setting first, the new one second."""
    rng = np.random.default_rng(seed)
    obj = env._config["objective"]
    held = obj["heldout"]
    gauge = obj.get("gauge")
    an, bn = list(held[0])[:2]
    props = view.action_space().kinds["levers"]["properties"]["values"]["properties"]
    lo = {n: float(props[n]["minimum"]) for n in (an, bn)}
    hi = {n: float(props[n]["maximum"]) for n in (an, bn)}
    A, B, y = [], [], []
    while view.t < view.horizon - 2 and not view.done:
        a = float(lo[an] + rng.random() * (hi[an] - lo[an]))
        b = float(lo[bn] + rng.random() * (hi[bn] - lo[bn]))
        for _ in range(int(rng.integers(1, 4))):
            if view.t >= view.horizon - 2 or view.done:
                break
            r = view.step({"kind": "levers", "values": {an: a, bn: b}})
            A.append(a)
            B.append(b)
            y.append(float(r["observation"]["numbers"].get(gauge, np.nan)))
    A, B, y = np.asarray(A), np.asarray(B), np.asarray(y)
    n = len(y)
    ok = np.isfinite(y)
    if n >= 24 and not view.done:
        best = None
        ymax = float(np.nanmax(y))
        links = [(None, None)] + [(ymax * m, k) for m in (1.2, 1.6, 2.2, 3.0) for k in (10.0, 25.0, 45.0, 80.0)]
        for rate in (0.05, 0.08, 0.1, 0.12, 0.16, 0.2):
            E = np.zeros(n)
            e_ = 0.0
            for t in range(n):
                E[t] = e_
                e_ = (1.0 - rate) * e_ + B[t]
            for h in (1.0, 2.5, 5.0, 10.0):
                sa = A / (A + h)
                for tau in range(8, n - 8, 2):
                    pre = (np.arange(n) < tau).astype(float)
                    F_ = np.c_[np.ones(n), pre * sa, (1 - pre) * sa, B * E][ok]
                    for cap, K in links:
                        # output may saturate in the processes' combined contribution z: y = cap z / (z + K)
                        if cap is None:
                            tz = y[ok]
                        else:
                            yy = np.minimum(y[ok], 0.98 * cap)
                            tz = K * yy / (cap - yy)
                        coef, *_ = np.linalg.lstsq(F_, tz, rcond=None)
                        zh = F_ @ coef
                        yh = zh if cap is None else cap * np.maximum(zh, 0.0) / (np.maximum(zh, 0.0) + K)
                        err = float(np.sum((yh - y[ok]) ** 2))
                        if best is None or err < best[0]:
                            best = (err, rate, h, tau, coef, cap, K)
        _, rate, h, tau, coef, cap, K = best
        out = []
        for c in held:
            a, b = float(c[an]), float(c[bn])
            Es = b * (1.0 - (1.0 - rate) ** settle) / rate
            z = coef[0] + coef[2] * a / (a + h) + coef[3] * b * Es
            v = z if cap is None else cap * max(z, 0.0) / (max(z, 0.0) + K)
            out.append({"set": c, "value": float(v)})
        view.step({"kind": "message", "to": "journal", "content": {"predictions": out}})
    while not view.done:
        view.step(None)


# ====================================================================== a dose with a peak, acting after a delay
def held_dose_predict(view, env, hold: int = 7, use: int = 2, seed: int = 0) -> None:
    """Scripted public reference for predicting the settled output of a treatment whose dose-response rises to a
    peak and falls, acting after a delay, next to a second, saturating lever: random conditions are held for
    `hold` ticks each and only the last `use` readings of every hold are kept (the delayed effect has arrived);
    output is modelled as y = z or y = cap z / (z + K) with z = c0 + c1 bump(dose; peak, width) + c2 sat(second; h)
    (bump: r e^(1 - r), r = dose / peak, scaled by a width), shapes and link from grids, the c's by least squares.
    Public: the objective names the treatment first, the second lever second."""
    rng = np.random.default_rng(seed)
    obj = env._config["objective"]
    held = obj["heldout"]
    gauge = obj.get("gauge")
    an, bn = list(held[0])[:2]
    props = view.action_space().kinds["levers"]["properties"]["values"]["properties"]
    lo = {n: float(props[n]["minimum"]) for n in (an, bn)}
    hi = {n: float(props[n]["maximum"]) for n in (an, bn)}
    A, B, y = [], [], []
    while view.t < view.horizon - 2 and not view.done:
        a = float(lo[an] + rng.random() * (hi[an] - lo[an]))
        b = float(lo[bn] + rng.random() * (hi[bn] - lo[bn]))
        vals = []
        for _ in range(hold):
            if view.t >= view.horizon - 2 or view.done:
                break
            r = view.step({"kind": "levers", "values": {an: a, bn: b}})
            vals.append(float(r["observation"]["numbers"].get(gauge, np.nan)))
        if len(vals) == hold:
            for v in vals[-use:]:
                A.append(a)
                B.append(b)
                y.append(v)
    A, B, y = np.asarray(A), np.asarray(B), np.asarray(y)
    ok = np.isfinite(y)
    A, B, y = A[ok], B[ok], y[ok]
    if len(y) >= 12 and not view.done:
        ymax = float(np.max(y))
        links = [(None, None)] + [(ymax * m, k) for m in (1.2, 1.6, 2.5) for k in (10.0, 25.0, 50.0)]
        best = None
        for pk in np.linspace(0.5, 9.5, 19):
            bump = (A / pk) * np.exp(1.0 - A / pk)
            for h in (1.0, 3.0, 6.0, 12.0):
                F_ = np.c_[np.ones(len(y)), bump, B / (B + h)]
                for cap, K in links:
                    tz = y if cap is None else K * np.minimum(y, 0.98 * cap) / (cap - np.minimum(y, 0.98 * cap))
                    coef, *_ = np.linalg.lstsq(F_, tz, rcond=None)
                    zh = F_ @ coef
                    yh = zh if cap is None else cap * np.maximum(zh, 0.0) / (np.maximum(zh, 0.0) + K)
                    err = float(np.sum((yh - y) ** 2))
                    if best is None or err < best[0]:
                        best = (err, pk, h, coef, cap, K)
        _, pk, h, coef, cap, K = best
        out = []
        for c in held:
            a, b = float(c[an]), float(c[bn])
            z = coef[0] + coef[1] * (a / pk) * np.exp(1.0 - a / pk) + coef[2] * b / (b + h)
            v = z if cap is None else cap * max(z, 0.0) / (max(z, 0.0) + K)
            out.append({"set": c, "value": float(v)})
        view.step({"kind": "message", "to": "journal", "content": {"predictions": out}})
    while not view.done:
        view.step(None)


def held_hub_predict(view, env, hold: int = 6, use: int = 2, seed: int = 0) -> None:
    """Scripted public reference for predicting the settled reading of two settings that act together through one
    saturating channel after a delay: random conditions are held for `hold` ticks and only the last `use` readings
    of every hold are kept; the reading is modelled as y = z or y = cap z / (z + K), z = c0 + c1 sat(w1 x1 + w2 x2; h),
    with the direction (w1, w2), h and the link from grids and the c's by least squares."""
    rng = np.random.default_rng(seed)
    obj = env._config["objective"]
    held = obj["heldout"]
    gauge = obj.get("gauge")
    names = sorted(held[0])
    props = view.action_space().kinds["levers"]["properties"]["values"]["properties"]
    lo = np.array([float(props[n]["minimum"]) for n in names])
    hi = np.array([float(props[n]["maximum"]) for n in names])
    X, y = [], []
    while view.t < view.horizon - 2 and not view.done:
        s = lo + rng.random(len(names)) * (hi - lo)
        vals = []
        for _ in range(hold):
            if view.t >= view.horizon - 2 or view.done:
                break
            r = view.step({"kind": "levers", "values": {n: float(v) for n, v in zip(names, s)}})
            vals.append(float(r["observation"]["numbers"].get(gauge, np.nan)))
        if len(vals) == hold:
            for v in vals[-use:]:
                X.append(s.copy())
                y.append(v)
    X, y = np.asarray(X), np.asarray(y)
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    if len(y) >= 12 and not view.done and X.shape[1] == 2:
        ymax = float(np.max(y))
        links = [(None, None)] + [(ymax * m, k) for m in (1.2, 1.6, 2.5) for k in (10.0, 25.0, 50.0)]
        best = None
        for th in np.linspace(0.05, np.pi / 2 - 0.05, 12):
            idx = X @ np.array([np.cos(th), np.sin(th)])
            for h in np.geomspace(0.5, 40.0, 10):
                F_ = np.c_[np.ones(len(y)), idx / (idx + h)]
                for cap, K in links:
                    tz = y if cap is None else K * np.minimum(y, 0.98 * cap) / (cap - np.minimum(y, 0.98 * cap))
                    coef, *_ = np.linalg.lstsq(F_, tz, rcond=None)
                    zh = F_ @ coef
                    yh = zh if cap is None else cap * np.maximum(zh, 0.0) / (np.maximum(zh, 0.0) + K)
                    err = float(np.sum((yh - y) ** 2))
                    if best is None or err < best[0]:
                        best = (err, th, h, coef, cap, K)
        _, th, h, coef, cap, K = best
        w = np.array([np.cos(th), np.sin(th)])
        out = []
        for c in held:
            idx = float(np.array([float(c[nm]) for nm in names]) @ w)
            z = coef[0] + coef[1] * idx / (idx + h)
            v = z if cap is None else cap * max(z, 0.0) / (max(z, 0.0) + K)
            out.append({"set": c, "value": float(v)})
        view.step({"kind": "message", "to": "journal", "content": {"predictions": out}})
    while not view.done:
        view.step(None)
