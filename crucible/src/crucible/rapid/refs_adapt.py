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
