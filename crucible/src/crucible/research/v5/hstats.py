"""Statistics over worlds AND learner seeds (MASTER_SPEC Part 15.7; U.8, SP-15.7).

A comparison of two arms is a crossed design: every arm is run with several independent learner seeds on the same
(paired) worlds. Its data are two matrices X[seed, world]. The 4.0 instrument averaged seeds away and bootstrapped
worlds only; with real seed-to-seed variation that gave a 94% false-positive rate for identical learners (audit
p5). Here:

* `crossed_diff` - the difference of arm means with a standard error that carries both sources of variation:
  between-seed variance of each arm's seed means, plus the variance of the paired world differences (each term an
  unbiased moment estimate that also absorbs the residual, so the sum is slightly conservative), and Student-t
  limits with Welch-Satterthwaite degrees of freedom, which keeps the interval honest with three to five seeds;
* `crossed_bootstrap` - the pigeonhole (rows x columns) bootstrap of the same statistic, resampling learner seeds
  per arm and worlds jointly (paired), for any statistic of the two matrices;
* `null_calibration` - the realised false-positive rate of a method on null data (identical arms): simulated from a
  crossed random-effects model, or by random splits of a pool of real runs of one learner;
* `verdict` - "significant" needs at least `MIN_SEEDS` seeds per arm and an interval excluding zero; otherwise the
  result is reported as underpowered (never as a difference).
"""
from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np

MIN_SEEDS = 3
LEVEL = 0.95


def _t_quantile(p: float, df: float) -> float:
    """Student-t quantile (Cornish-Fisher expansion around the normal; accurate to ~1e-3 for df >= 2)."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(p)
    if not math.isfinite(df) or df > 1e6:
        return z
    df = max(df, 1.0)
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    g4 = (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3 + g4 / df ** 4


def _t_sf(t: float, df: float) -> float:
    """Upper tail of Student-t by numerical inversion of `_t_quantile` (bisection; adequate for p-values)."""
    t = abs(float(t))
    lo, hi = 1e-12, 0.5
    for _ in range(80):
        mid = (lo + hi) / 2
        if _t_quantile(1 - mid, df) > t:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def crossed_diff(XA: np.ndarray, XB: np.ndarray, *, paired_seeds: bool = False, level: float = LEVEL) -> dict:
    """mean(XB) - mean(XA) for matrices [seed, world] on the same worlds (columns), with a crossed-design interval.
    `paired_seeds`: row i of both arms used the same learner seed (shared seed effect)."""
    XA, XB = np.asarray(XA, dtype=float), np.asarray(XB, dtype=float)
    if XA.shape[1] != XB.shape[1]:
        raise ValueError("arms must be run on the same worlds")
    nA, nB, nW = XA.shape[0], XB.shape[0], XA.shape[1]
    est = float(XB.mean() - XA.mean())
    Dw = XB.mean(axis=0) - XA.mean(axis=0)
    terms = []
    if paired_seeds:
        if nA != nB:
            raise ValueError("paired seeds need the same number of seeds in both arms")
        ds = XB.mean(axis=1) - XA.mean(axis=1)
        if nA > 1:
            terms.append((float(np.var(ds, ddof=1)) / nA, nA - 1))
    else:
        for X, n in ((XA, nA), (XB, nB)):
            if n > 1:
                terms.append((float(np.var(X.mean(axis=1), ddof=1)) / n, n - 1))
    if nW > 1:
        terms.append((float(np.var(Dw, ddof=1)) / nW, nW - 1))
    var = sum(v for v, _ in terms)
    se = math.sqrt(var) if var > 0 else 0.0
    df = (var ** 2 / sum(v ** 2 / d for v, d in terms if d > 0)) if var > 0 else float("inf")
    q = _t_quantile(1 - (1 - level) / 2, df)
    p = 2 * _t_sf(est / se, df) if se > 0 else (0.0 if est != 0 else 1.0)
    return {"estimate": est, "se": se, "df": float(df), "lo": est - q * se, "hi": est + q * se, "p": float(min(1, p)),
            "level": level, "seeds": [nA, nB], "worlds": nW, "unit": "crossed (learner seed x world)",
            "components": {"seed": sum(v for v, _ in terms[:-1]) if nW > 1 else var,
                           "world": terms[-1][0] if nW > 1 else 0.0}}


def crossed_bootstrap(XA: np.ndarray, XB: np.ndarray, stat: Callable[[np.ndarray, np.ndarray], float] | None = None,
                      *, n_boot: int = 2000, seed: int = 0, level: float = LEVEL, paired_seeds: bool = False) -> dict:
    """Pigeonhole bootstrap: resample learner seeds (rows, per arm unless paired) and worlds (columns, jointly);
    returns the percentile interval of `stat` (default: difference of means) with a small-sample correction of
    the spread (the factor sqrt(n/(n-1)) for the smaller of the two dimensions)."""
    XA, XB = np.asarray(XA, dtype=float), np.asarray(XB, dtype=float)
    stat = stat or (lambda a, b: float(b.mean() - a.mean()))
    rng = np.random.default_rng(seed)
    nA, nB, nW = XA.shape[0], XB.shape[0], XA.shape[1]
    est = stat(XA, XB)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        w = rng.integers(nW, size=nW)
        ra = rng.integers(nA, size=nA)
        rb = ra if paired_seeds else rng.integers(nB, size=nB)
        vals[i] = stat(XA[np.ix_(ra, w)], XB[np.ix_(rb, w)])
    n_min = min(nA, nB, nW)
    infl = math.sqrt(n_min / (n_min - 1)) if n_min > 1 else float("inf")
    c = np.median(vals)
    a = (1 - level) / 2
    lo, hi = np.quantile(vals, [a, 1 - a])
    return {"estimate": float(est), "lo": float(c + (lo - c) * infl), "hi": float(c + (hi - c) * infl),
            "boot_se": float(np.std(vals, ddof=1) * infl), "n_boot": n_boot,
            "unit": "pigeonhole bootstrap (learner seeds x worlds)"}


def verdict(res: Mapping, min_effect: float = 0.0) -> dict:
    """'significant' only with >= MIN_SEEDS seeds per arm, an interval excluding zero and |estimate| >= min_effect."""
    seeds = min(res.get("seeds") or [MIN_SEEDS])
    if seeds < MIN_SEEDS:
        return {"call": "underpowered", "why": f"{seeds} learner seeds < {MIN_SEEDS}"}
    excl = res["lo"] > 0 or res["hi"] < 0
    big = abs(res["estimate"]) >= min_effect
    if excl and big:
        return {"call": "significant", "direction": "B>A" if res["estimate"] > 0 else "B<A"}
    return {"call": "no_difference_detected", "why": "interval includes 0" if not excl else
            f"|effect| {abs(res['estimate']):.3g} < prespecified {min_effect}"}


def naive_task_bootstrap(XA: np.ndarray, XB: np.ndarray, n_boot: int = 1000, seed: int = 0,
                         level: float = LEVEL) -> dict:
    """The 4.0 analysis (seeds averaged away, paired bootstrap over worlds only) - kept to show its miscalibration."""
    rng = np.random.default_rng(seed)
    d = np.asarray(XB).mean(axis=0) - np.asarray(XA).mean(axis=0)
    vals = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(n_boot)])
    a = (1 - level) / 2
    lo, hi = np.quantile(vals, [a, 1 - a])
    return {"estimate": float(d.mean()), "lo": float(lo), "hi": float(hi), "seeds": [np.shape(XA)[0]] * 2}


def simulate_null(rng: np.random.Generator, n_seeds: int, n_worlds: int, sd_seed: float = 0.15,
                  sd_world: float = 0.3, sd_inter: float = 0.05, sd_resid: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Two identical arms from a crossed random-effects model: shared world effects, independent learner-seed
    effects per arm, arm-by-world interaction and residual noise."""
    beta = rng.normal(0, sd_world, n_worlds)

    def arm():
        alpha = rng.normal(0, sd_seed, n_seeds)
        gamma = rng.normal(0, sd_inter, n_worlds)
        return alpha[:, None] + beta[None, :] + gamma[None, :] + rng.normal(0, sd_resid, (n_seeds, n_worlds))
    return arm(), arm()


def null_calibration(method: str = "crossed", trials: int = 400, n_seeds: int = 3, n_worlds: int = 12,
                     seed: int = 0, pool: np.ndarray | None = None, level: float = LEVEL, **model) -> dict:
    """False-positive rate of `method` ("crossed", "bootstrap" or "naive") on identical arms. With `pool`
    (a [seed, world] matrix of real runs of ONE learner), each trial splits 2 * n_seeds random rows into two arms
    and takes n_worlds random columns; otherwise data come from `simulate_null(**model)`."""
    rng = np.random.default_rng(seed)
    fp = 0
    for i in range(trials):
        if pool is not None:
            pool = np.asarray(pool, dtype=float)
            rows = rng.permutation(pool.shape[0])[: 2 * n_seeds]
            cols = rng.permutation(pool.shape[1])[:n_worlds]
            XA, XB = pool[np.ix_(rows[:n_seeds], cols)], pool[np.ix_(rows[n_seeds:], cols)]
        else:
            XA, XB = simulate_null(rng, n_seeds, n_worlds, **model)
        if method == "crossed":
            r = crossed_diff(XA, XB, level=level)
        elif method == "bootstrap":
            r = crossed_bootstrap(XA, XB, n_boot=300, seed=i, level=level)
            r["seeds"] = [n_seeds, n_seeds]
        elif method == "naive":
            r = naive_task_bootstrap(XA, XB, n_boot=300, seed=i, level=level)
        else:
            raise ValueError(method)
        fp += int(r["lo"] > 0 or r["hi"] < 0)
    return {"method": method, "trials": trials, "false_positive_rate": fp / trials, "nominal": 1 - level,
            "n_seeds": n_seeds, "n_worlds": n_worlds, "source": "real-run pool" if pool is not None else "simulated"}


def matrix(records: Sequence[Mapping], key: str = "score") -> tuple[np.ndarray, list, list]:
    """[seed, world] matrix from records {learner_seed, entry, seed (world), key}; worlds sorted."""
    seeds = sorted({r["learner_seed"] for r in records})
    worlds = sorted({(r["entry"], r["seed"]) for r in records})
    X = np.full((len(seeds), len(worlds)), np.nan)
    for r in records:
        X[seeds.index(r["learner_seed"]), worlds.index((r["entry"], r["seed"]))] = float(r[key])
    if np.isnan(X).any():
        raise ValueError("incomplete crossed design (every seed must be scored on every world)")
    return X, seeds, worlds
