"""Evaluator side of the `experiment_lab` families (workstream H1): oracles, scripted public-information
references, identification procedures and intervention tests. Nothing in a world imports this module
(`admission.static_isolation` checks it); its functions read hidden structure and run only on copies of worlds.

The prediction oracle files the noise-free law at the held-out conditions; references design experiments through
a `PublicView`, fit the candidate laws a scientist would consider (the brief names the apparatus, not the law) by
least squares, and file predictions from the fitted law.
"""
from __future__ import annotations

import numpy as np

from .engines.experiment_lab import hill, kinetics_solve
from .identify import PublicView
from .oracles import best_of, fork, inside


# ====================================================================== helpers
def file_predictions(view, conds: list[dict], values) -> dict:
    preds = [{"set": cond, "value": float(v)} for cond, v in zip(conds, values)]
    return view.step({"kind": "message", "to": "journal", "content": {"predictions": preds}})


def finish(view) -> None:
    while not view.done:
        view.step(None)


def oracle_predict(env) -> dict:
    eng, st, c = inside(env)
    vals = eng.sc.heldout_truth()
    preds = [{"set": cond, "value": float(v)} for cond, v in zip(c["heldout"], vals)]

    def fn(e):
        return {"kind": "message", "to": "journal", "content": {"predictions": preds}} if e.t == 0 else None
    return best_of(env, {"knows_law": fn})


def run_designs(view, designs: list[dict]) -> list[tuple[dict, dict]]:
    """Run experiments in sequences of up to 16 per tick; returns (design, measured) pairs that succeeded."""
    out = []
    for i in range(0, len(designs), 16):
        if view.done:
            break
        chunk = designs[i:i + 16]
        r = view.step({"kind": "sequence", "steps": [{"kind": "experiment", "design": d} for d in chunk]})
        res = r["result"]
        got = res.get("results") if res["ok"] else res.get("done_steps")
        if isinstance(got, dict):
            got = [got]
        for d, g in zip(chunk, got or []):
            if isinstance(g, dict) and "measured" in g:
                out.append((d, g["measured"]))
    return out


def lm(resid, p0, iters: int = 60):
    """Levenberg-Marquardt on a residual function with a forward-difference Jacobian."""
    p = np.asarray(p0, dtype=float)
    r = resid(p)
    f = float(r @ r)
    lam = 1e-2
    for _ in range(iters):
        J = np.empty((r.size, p.size))
        for j in range(p.size):
            dp = np.zeros_like(p)
            dp[j] = 1e-5 * max(1.0, abs(p[j]))
            J[:, j] = (resid(p + dp) - r) / dp[j]
        A, g = J.T @ J, J.T @ r
        improved = False
        while lam < 1e8:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            pn = p + step
            rn = resid(pn)
            fn = float(rn @ rn)
            if np.isfinite(fn) and fn < f:
                rel = (f - fn) / max(f, 1e-300)
                p, r, f, lam, improved = pn, rn, fn, max(lam / 3, 1e-9), True
                break
            lam *= 4
        if not improved or rel < 1e-10:
            break
    return p, f


# ====================================================================== kinetics
def kin_model(scheme: str, p, T, A0, times):
    k30 = [np.exp(p[0]), np.exp(p[1])]
    q = [np.exp(p[2]), np.exp(p[3])]
    seed = float(np.exp(p[4])) if scheme == "autocatalytic" else 0.0
    return kinetics_solve(scheme, k30, q, seed, T, A0, times)


def kin_fit(data: list, scheme: str) -> tuple[np.ndarray, float]:
    T = np.array([x[0] for x in data])
    A0 = np.array([x[1] for x in data])
    tt = np.array([x[2] for x in data])
    y = np.array([x[3] for x in data])

    def resid(p):
        return kin_model(scheme, p, T, A0, tt)[:, 1] - y
    starts = ([np.log([0.25, 0.1, 2.0, 2.0])] if scheme == "sequential" else
              [np.log([0.8, 0.1, 2.0, 2.0, 0.04]), np.log([0.5, 0.05, 1.8, 1.8, 0.02])])
    if scheme == "sequential":
        starts.append(np.log([0.1, 0.25, 2.0, 2.0]))
    best = None
    for p0 in starts:
        p, f = lm(resid, p0)
        if best is None or f < best[1]:
            best = (p, f)
    return best


def kin_collect(view, c: dict, reps: int = 2) -> list:
    nT, nA = c["vars"][0]["name"], c["vars"][1]["name"]
    sp = c["species"]
    designs = [{"set": {nT: T, nA: a0}, "measure": [sp[1]], "instrument": "probe", "replicates": reps, "time": tt}
               for T in (20.0, 35.0, 50.0) for a0 in (0.5, 1.25, 2.0) for tt in (1, 2, 3, 5, 8)]
    data = []
    for d, m in run_designs(view, designs):
        for v in m[sp[1]]["values"]:
            data.append((d["set"][nT], d["set"][nA], d["time"], v))
    return data


def kin_select(data: list) -> tuple[str, np.ndarray]:
    fits = {s: kin_fit(data, s) for s in ("sequential", "autocatalytic")}
    n = len(data)
    bic = {s: n * np.log(f / n + 1e-300) + (4 if s == "sequential" else 5) * np.log(n) for s, (p, f) in fits.items()}
    s = min(bic, key=bic.get)
    return s, fits[s][0]


def reference_kinetics(view, env) -> None:
    c = env._config
    nT, nA = c["vars"][0]["name"], c["vars"][1]["name"]
    data = kin_collect(view, c)
    if data:
        s, p = kin_select(data)
        conds = c["heldout"]
        y = kin_model(s, p, [x[nT] for x in conds], [x[nA] for x in conds], [x["time"] for x in conds])[:, 1]
        file_predictions(view, conds, y)
    finish(view)


def identify_kinetics(env) -> dict:
    c = env._config
    data = kin_collect(PublicView(env), c, reps=1)
    s, _ = kin_select(data)
    truth = c["hidden"]["scheme"]
    return {"question": "the reaction scheme (sequential first-order or autocatalytic)", "estimate": s,
            "truth": truth, "ok": s == truth}


oracle_kinetics = oracle_predict


def scheme_curvature(env) -> dict:
    """At the start of a batch the second species accelerates (autocatalysis) or decelerates (sequential
    first-order conversion), whatever the coefficients: the sign of its initial second difference names the
    scheme."""
    eng, st, c = inside(env)
    nT, nA = c["vars"][0]["name"], c["vars"][1]["name"]
    out = {}
    for T in (20.0, 35.0, 50.0):
        b = [float(v) for v in eng.sc.conc({nT: T, nA: 1.0}, [0.05, 0.1, 0.15])[:, 1]]
        b0 = c["hidden"]["seed_b"]
        out[T] = (b[1] - b[0]) - (b[0] - b0)
    acc = all(v > 0 for v in out.values())
    dec = all(v < 0 for v in out.values())
    want = c["hidden"]["scheme"]
    return {"ok": bool((acc and want == "autocatalytic") or (dec and want == "sequential")),
            "initial_second_difference": out, "scheme": want}


def mass_balance(env) -> dict:
    """A conservation relationship: the chromatograph's three readings (noise switched off on a copy) sum to the
    initial load (plus the trace) at every sampling time and temperature."""
    e = fork(env)
    c = e._config
    for v in c["instruments"].values():
        v["sd"] = 0.0
    e._state["budget"] = 1e9
    nT, nA = c["vars"][0]["name"], c["vars"][1]["name"]
    worst = 0.0
    for T in (20.0, 50.0):
        for a0 in (0.5, 2.0):
            for tt in (1, 4, 12):
                r = e.step({"kind": "experiment", "design": {"set": {nT: T, nA: a0}, "measure": list(c["species"]),
                                                             "instrument": "chromatograph", "time": tt}})
                m = r["result"]["results"]["measured"]
                tot = sum(m[s]["mean"] for s in c["species"])
                worst = max(worst, abs(tot - a0 - c["hidden"]["seed_b"]))
    return {"ok": bool(worst < 1e-4), "max_imbalance": worst}


def rates_rise_with_temperature(env) -> dict:
    """The second species peaks earlier at higher temperature (both rates rise)."""
    eng, st, c = inside(env)
    nT, nA = c["vars"][0]["name"], c["vars"][1]["name"]
    grid = np.arange(0.5, 40.0, 0.5)
    peaks = {}
    for T in (20.0, 35.0, 50.0):
        b = eng.sc.conc({nT: T, nA: 1.0}, grid)[:, 1]
        peaks[T] = float(grid[int(np.argmax(b))])
    return {"ok": bool(peaks[20.0] > peaks[35.0] > peaks[50.0]), "peak_time": peaks}


# ====================================================================== drift_blanks
def db_collect(view, c: dict, doses=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0), reps: int = 3,
               reserve_ticks: int = 1) -> dict:
    """Every tick: a blank and one dose in the same tick, `reps` replicates each; the median difference is that
    tick's estimate of the dose's effect."""
    nx = c["vars"][0]["name"]
    q = c["quantity"]
    est: dict = {}
    k = 0
    while not view.done and view.t < view.horizon - reserve_ticks:
        x = doses[k % len(doses)]
        k += 1
        mk = lambda v: {"kind": "experiment", "design": {"set": {nx: v}, "measure": [q], "instrument": "meter",
                                                        "replicates": reps}}
        r = view.step({"kind": "sequence", "steps": [mk(0.0), mk(x)]})
        if not r["result"]["ok"]:
            break
        blank, dose = r["result"]["results"]
        est.setdefault(x, []).append(float(np.median(dose["measured"][q]["values"]) -
                                         np.median(blank["measured"][q]["values"])))
    return est


def hill_fit(xs, ys) -> tuple[float, float, float]:
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    best = None
    for n in np.arange(0.6, 3.01, 0.05):
        for ec in np.exp(np.linspace(np.log(0.5), np.log(12.0), 80)):
            g = hill(xs, 1.0, ec, n)
            den = float(g @ g)
            if den <= 0:
                continue
            em = float(g @ ys) / den
            sse = float(((em * g - ys) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, em, ec, n)
    return best[1], best[2], best[3]


def reference_drift_blanks(view, env) -> None:
    c = env._config
    nx = c["vars"][0]["name"]
    est = db_collect(view, c)
    if est:
        xs = [x for x, v in est.items() for _ in v]
        ys = [y for v in est.values() for y in v]
        em, ec, n = hill_fit(xs, ys)
        conds = c["heldout"]
        file_predictions(view, conds, hill([x[nx] for x in conds], em, ec, n))
    finish(view)


def identify_drift_blanks(env) -> dict:
    c = env._config
    est = db_collect(PublicView(env), c, doses=(4.0,), reps=3, reserve_ticks=20)
    e4 = float(np.mean(est.get(4.0, [0.0])))
    h = c["hidden"]
    truth = float(hill(4.0, h["emax"], h["ec50"], h["n"]))
    return {"question": "the drift-free effect at the highest settable dose", "estimate": e4, "truth": truth,
            "ok": bool(abs(e4 - truth) <= 0.15)}


oracle_drift_blanks = oracle_predict


def _db_fork(env, outliers: bool):
    e = fork(env)
    e._state["budget"] = 1e9
    if not outliers:
        e._config["hidden"]["out_p"] = 0.0
    return e


def baseline_drifts(env) -> dict:
    """Two blanks measured in the same tick agree far better than two blanks ten ticks apart (outliers switched
    off on the copy): the baseline moves between ticks."""
    e = _db_fork(env, outliers=False)
    c = e._config
    nx, q = c["vars"][0]["name"], c["quantity"]
    blk = {"kind": "experiment", "design": {"set": {nx: 0.0}, "measure": [q], "instrument": "meter", "replicates": 6}}
    halves, means = [], []
    while not e.done:
        r = e.step(blk)
        v = r["result"]["results"]["measured"][q]["values"]
        halves.append(float(np.mean(v[:3]) - np.mean(v[3:])))
        means.append(float(np.mean(v)))
    same = float(np.sqrt(np.mean(np.square(halves))))
    apart = float(np.sqrt(np.mean([(means[j] - means[i]) ** 2 for i in range(len(means))
                                   for j in range(i + 10, len(means))])))
    return {"ok": bool(apart > 1.5 * same), "same_tick_rms": same, "ten_or_more_ticks_apart_rms": apart}


def heavy_tails(env) -> dict:
    """Replicate readings of one preparation have excess kurtosis (gross outliers), unlike Gaussian noise."""
    e = _db_fork(env, outliers=True)
    c = e._config
    nx, q = c["vars"][0]["name"], c["quantity"]
    vals = []
    while not e.done:
        r = e.step({"kind": "experiment", "design": {"set": {nx: 2.0}, "measure": [q], "instrument": "meter",
                                                     "replicates": 6}})
        v = np.asarray(r["result"]["results"]["measured"][q]["values"])
        vals += list(v - v.mean())
    v = np.asarray(vals)
    kurt = float(((v - v.mean()) ** 4).mean() / (v.var() ** 2))
    return {"ok": bool(kurt > 4.0), "kurtosis": kurt}


def blanks_remove_drift(env) -> dict:
    """Estimating the effect at dose 3 from a blank measured in the same tick is accurate; using a blank measured
    at the start (stale) is off by the drift."""
    e = _db_fork(env, outliers=False)
    c = e._config
    nx, q = c["vars"][0]["name"], c["quantity"]
    h = c["hidden"]
    truth = float(hill(3.0, h["emax"], h["ec50"], h["n"]))
    mk = lambda v: {"kind": "experiment", "design": {"set": {nx: v}, "measure": [q], "instrument": "meter",
                                                    "replicates": 6}}
    r0 = e.step(mk(0.0))
    stale = float(np.mean(r0["result"]["results"]["measured"][q]["values"]))
    err_same, err_stale = [], []
    while not e.done:
        r = e.step({"kind": "sequence", "steps": [mk(0.0), mk(3.0)]})
        b, d = r["result"]["results"]
        bv, dv = np.mean(b["measured"][q]["values"]), np.mean(d["measured"][q]["values"])
        err_same.append(dv - bv - truth)
        err_stale.append(dv - stale - truth)
    rs, rt = float(np.sqrt(np.mean(np.square(err_same)))), float(np.sqrt(np.mean(np.square(err_stale))))
    return {"ok": bool(rt > 1.5 * rs), "rms_error_same_tick_blank": rs, "rms_error_stale_blank": rt}
