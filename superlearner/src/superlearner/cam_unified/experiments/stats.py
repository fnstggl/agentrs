"""Interval estimates over training seeds and evaluation worlds (experiment register rule; spec Part 19).

Values are a matrix ``M[s, w]``: training seed s (rows; a learner without training seeds has one row) by evaluation
world w (columns; the same worlds for every seed). NaN cells (a world a learner could not run) are skipped.

Default method "t" (repaired 2026-11-26; audit 'h' statistics finding): a studentised random-effects interval for
the population mean over seeds and worlds. Under M[s, w] = mu + a_s + b_w + e_sw (independent seed, world and cell
effects), Var(grand mean) = sigma_a^2 / S + sigma_b^2 / W + sigma_e^2 / (S W). It is estimated by
s^2(row means) / S + s^2(column means) / W, which is unbiased for the first two terms and counts the cell term
twice (conservative by sigma_e^2 / (S W)), and the interval is mean +- t_{q, nu} se. The Satterthwaite degrees of
freedom nu_S = se^4 / ((s_r^2/S)^2 / (S-1) + (s_c^2/W)^2 / (W-1)) are liberal when the dominant term has few degrees
of freedom (its weight is itself estimated: coverage 0.918 at S=3, W=16, seed sd = world sd = 1), so the default uses
nu = max(min(S-1, W-1), 0.6 nu_S) (``DOF_SHRINK``), calibrated by ``coverage_simulation``: 0.945 at S=3/W=16,
0.948 at S=3/W=48, 0.965 at S=5, 0.931 at S=2/W=8 (2000 replicates each; the test suite re-checks >= 0.93 at S=3).
"t_conservative" uses min(S-1, W-1) (always >= nominal, wider) and "t_satterthwaite" the plain estimate. With 3
seeds the interval is wide (t_{0.975, 2} = 4.30): that is the honest price of 3 seeds. The percentile bootstrap of
the earlier version covered the mean only 0.77-0.79 of the time at S=3 (nominal 0.95); it remains available as
method="bootstrap" and is never the default. One training seed: the seed component is not estimable, the interval
covers world sampling only, and ``seed_variance_estimable`` is False.

Paired comparisons (``paired``): learners evaluated on the same worlds. With the same number of seeds the per-cell
difference D = A - B follows the same two-way model (paired by seed index; valid whether the seeds are matched or
independent) and gets the interval above. With different seed counts, or ``independent_seeds=True``, the seeds of A
and of B are independent samples: Var = s^2_A(rows) / S_A + s^2_B(rows) / S_B + s^2(d_w) / W with d_w the per-world
difference of seed means, and the same degrees-of-freedom rule over the three terms (the earlier version subtracted
B's seed mean and dropped its variance: coverage 0.49 at 3 vs 1 seeds; now 0.948 with the single-seed learner's seed
variance pooled from the other learner, ``single_seed="pooled"``, recorded in ``assumptions``).

``p_gt0`` is descriptive: the fraction of bootstrap draws (seeds of each learner resampled independently, worlds
jointly) with a positive difference. Decisions use the interval.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

METHODS = ("t", "t_conservative", "t_satterthwaite", "bootstrap")
SINGLE_SEED = ("pooled", "zero")


# ============================================================================ Student t (no scipy in this venv)
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularised incomplete beta function (modified Lentz)."""
    tiny, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1.0 - x) / b


def t_cdf(x: float, df: float) -> float:
    """P(T <= x) for Student's t with (real) df > 0."""
    if not math.isfinite(df) or df > 1e7:          # the normal limit (the beta form loses precision there)
        return 0.5 * math.erfc(-x / math.sqrt(2.0))
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + x * x))
    return 1.0 - tail if x > 0 else tail


def t_ppf(q: float, df: float) -> float:
    """Quantile of Student's t (bisection on t_cdf; |error| < 1e-10)."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    if q == 0.5:
        return 0.0
    if q < 0.5:
        return -t_ppf(1.0 - q, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < q:
        hi *= 2.0
        if hi > 1e12:
            return float("inf")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


# ============================================================================ components
def _mat(M) -> np.ndarray:
    return np.atleast_2d(np.asarray(M, dtype=np.float64))


def _row_col(A: np.ndarray) -> tuple:
    """(row means over present cells, column means over present cells), rows / columns without data dropped."""
    ok = np.isfinite(A)
    rs, cs = ok.sum(1), ok.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rows = np.where(rs > 0, np.where(ok, A, 0.0).sum(1) / np.maximum(rs, 1), np.nan)
        cols = np.where(cs > 0, np.where(ok, A, 0.0).sum(0) / np.maximum(cs, 1), np.nan)
    return rows[np.isfinite(rows)], cols[np.isfinite(cols)]


def _var(x: np.ndarray) -> float:
    return float(np.var(x, ddof=1)) if len(x) > 1 else 0.0


DOF_SHRINK = 0.6       # calibrated by coverage_simulation (see the module docstring and test_x_stats_coverage.py)


def _interval(mean: float, terms: list, level: float, method: str = "t") -> dict:
    """terms: [(variance contribution, degrees of freedom)] combined into se and degrees of freedom:
    "t_satterthwaite" nu_S (plain Welch-Satterthwaite), "t_conservative" min_i nu_i (exact lower bound of the
    effective dof of a positive combination of scaled chi-squares), "t" (default) max(min_i nu_i, DOF_SHRINK nu_S):
    the plain estimate is liberal when the dominant term has few dof because its weight is itself estimated."""
    terms = [(v, d) for v, d in terms if d > 0]
    var = sum(v for v, _ in terms)
    if not terms:
        return {"mean": mean, "lo": None, "hi": None, "se": None, "dof": None}
    if var <= 0.0:
        return {"mean": mean, "lo": mean, "hi": mean, "se": 0.0, "dof": None}
    den = sum(v * v / d for v, d in terms if v > 0)
    sat = var * var / den if den > 0 else float("inf")
    lo_d = min(d for _, d in terms)
    dof = {"t_satterthwaite": sat, "t_conservative": lo_d, "t": max(lo_d, DOF_SHRINK * sat)}[method]
    se = math.sqrt(var)
    q = t_ppf(0.5 + level / 2.0, dof)
    return {"mean": mean, "lo": mean - q * se, "hi": mean + q * se, "se": se, "dof": dof}


def _draws(A: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap means of A [S, W] (vectorised): seeds resampled, then worlds resampled, NaN cells skipped."""
    rng = np.random.default_rng(seed)
    S, W = A.shape
    si = rng.integers(0, S, (n_boot, S))
    wi = rng.integers(0, W, (n_boot, W))
    sub = A[si[:, :, None], wi[:, None, :]]                        # [n_boot, S, W]
    ok = np.isfinite(sub)
    cnt = ok.sum((1, 2))
    tot = np.where(ok, sub, 0.0).sum((1, 2))
    out = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return out[np.isfinite(out)]


def _draws_indep(A: np.ndarray, B: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap of mean(A) - mean(B): seeds of A and of B resampled independently, worlds jointly."""
    rng = np.random.default_rng(seed)
    W = A.shape[1]
    wi = rng.integers(0, W, (n_boot, W))
    out = []
    for X in (A, B):
        si = rng.integers(0, X.shape[0], (n_boot, X.shape[0]))
        sub = X[si[:, :, None], wi[:, None, :]]
        ok = np.isfinite(sub)
        out.append(np.where(ok.sum((1, 2)) > 0, np.where(ok, sub, 0.0).sum((1, 2)) / np.maximum(ok.sum((1, 2)), 1),
                            np.nan))
    d = out[0] - out[1]
    return d[np.isfinite(d)]


# ============================================================================ public API
def two_level(M, n_boot: int = 4000, level: float = 0.95, seed: int = 0, method: str = "t") -> dict:
    """Mean of M [S, W] with a `level` interval for the population mean over seeds and worlds (module docstring)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    A = _mat(M)
    S, W = A.shape
    ok = np.isfinite(A)
    base = {"n_seeds": S, "n_worlds": W, "n_cells": int(ok.sum()), "method": method, "level": level}
    if not ok.any():
        return {"mean": None, "lo": None, "hi": None, "se": None, "dof": None, **base,
                "seed_variance_estimable": False}
    point = float(np.nanmean(A))
    if method == "bootstrap":
        boots = _draws(A, n_boot, seed)
        a = (1 - level) / 2
        return {"mean": point, "lo": float(np.quantile(boots, a)), "hi": float(np.quantile(boots, 1 - a)),
                "se": float(np.std(boots)), "dof": None, **base, "seed_variance_estimable": S > 1}
    rows, cols = _row_col(A)
    terms = []
    if len(rows) > 1:
        terms.append((_var(rows) / len(rows), len(rows) - 1))
    if len(cols) > 1:
        terms.append((_var(cols) / len(cols), len(cols) - 1))
    out = _interval(point, terms, level, method)
    return {**out, **base, "seed_variance_estimable": len(rows) > 1}


def paired(A, B, n_boot: int = 4000, level: float = 0.95, seed: int = 0, method: str = "t",
           independent_seeds: Optional[bool] = None, single_seed: str = "pooled") -> dict:
    """Mean of A - B on the same worlds with a `level` interval. A, B: [S_a, W], [S_b, W]. Paired per cell by seed
    index when S_a == S_b (unless `independent_seeds=True`); otherwise the independent-seeds interval (docstring).

    `single_seed`: a learner with ONE training seed has no estimable seed variance. "pooled" (default) assumes it
    equals the other learner's (recorded in "assumptions"); "zero" declares the single-row learner deterministic in
    its training seed (a scripted / null / oracle reference, or a learner without training), so it contributes no
    seed term."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    A, B = _mat(A), _mat(B)
    if A.shape[1] != B.shape[1]:
        raise ValueError("paired comparison needs the same evaluation worlds")
    indep = bool(independent_seeds) or A.shape[0] != B.shape[0]
    if not indep:
        D = A - B
        out = two_level(D, n_boot, level, seed, method)
        out["p_gt0"] = float(np.mean(_draws(D, n_boot, seed + 1) > 0)) if out["mean"] is not None else None
        out["pairing"] = "per cell (seed index)"
        return out
    ra, _ = _row_col(A)
    rb, _ = _row_col(B)
    with np.errstate(invalid="ignore"):
        dw = np.nanmean(A, 0) - np.nanmean(B, 0)
    dw = dw[np.isfinite(dw)]
    base = {"n_seeds": [A.shape[0], B.shape[0]], "n_worlds": A.shape[1],
            "n_cells": int(np.isfinite(A).sum() + np.isfinite(B).sum()), "method": method, "level": level,
            "pairing": "per world (independent seeds)", "seed_variance_estimable": len(ra) > 1 and len(rb) > 1}
    if not len(dw):
        return {"mean": None, "lo": None, "hi": None, "se": None, "dof": None, "p_gt0": None, **base}
    mean = float(np.nanmean(A) - np.nanmean(B))
    boots = _draws_indep(A, B, n_boot, seed + 1)
    p = float(np.mean(boots > 0)) if len(boots) else None
    if method == "bootstrap":
        a = (1 - level) / 2
        return {"mean": mean, "lo": float(np.quantile(boots, a)), "hi": float(np.quantile(boots, 1 - a)),
                "se": float(np.std(boots)), "dof": None, "p_gt0": p, **base}
    if single_seed not in SINGLE_SEED:
        raise ValueError(f"single_seed must be one of {SINGLE_SEED}")
    terms, assumptions = [], []
    for name, r, other in (("A", ra, rb), ("B", rb, ra)):
        if len(r) > 1:
            terms.append((_var(r) / len(r), len(r) - 1))
        elif single_seed == "pooled" and len(other) > 1:
            terms.append((_var(other) / max(len(r), 1), len(other) - 1))
            assumptions.append(f"{name}: one seed; seed variance pooled from the other learner")
        else:
            assumptions.append(f"{name}: one seed; seed variance taken as zero")
    if len(dw) > 1:
        terms.append((_var(dw) / len(dw), len(dw) - 1))
    return {**_interval(mean, terms, level, method), "p_gt0": p, **base, "assumptions": assumptions}


def verdict(ci: dict, direction: str = ">", margin: float = 0.0) -> str:
    """Pre-registered decision rule on an interval: DEMONSTRATED when the whole interval lies beyond `margin` in
    `direction` (">" larger is better, "<" smaller is better), FAILED when the whole interval lies on the other side
    of `margin` (at or below it for ">"), INCONCLUSIVE when it straddles `margin` or no interval exists."""
    lo, hi = ci.get("lo"), ci.get("hi")
    if lo is None or hi is None:
        return "INCONCLUSIVE"
    if direction == ">":
        return "DEMONSTRATED" if lo > margin else ("FAILED" if hi <= margin else "INCONCLUSIVE")
    if direction == "<":
        return "DEMONSTRATED" if hi < margin else ("FAILED" if lo >= margin else "INCONCLUSIVE")
    raise ValueError("direction must be '>' or '<'")


def curve(Ms: dict, n_boot: int = 2000, seed: int = 0, method: str = "t") -> dict:
    """{x: M[s, w]} -> {x: two_level(M)} (e.g. a metric at every support size k)."""
    return {str(x): two_level(M, n_boot, seed=seed, method=method) for x, M in Ms.items()}


def area(Ms: dict, xs: Optional[Sequence] = None) -> np.ndarray:
    """Learning-curve area per (seed, world): the mean of the metric over the grid points `xs` (default: all keys of
    `Ms`, e.g. the doubling grid k = 0, 1, 2, ..., 64, so every doubling of experience weighs the same)."""
    keys = list(xs) if xs is not None else list(Ms)
    return np.nanmean(np.stack([np.asarray(Ms[k], dtype=np.float64) for k in keys], 0), 0)


def coverage_simulation(S: int = 3, W: int = 16, seed_sd: float = 1.0, world_sd: float = 1.0, noise_sd: float = 0.3,
                        reps: int = 2000, level: float = 0.95, method: str = "t", rng_seed: int = 0,
                        S_b: Optional[int] = None, n_boot: int = 400) -> dict:
    """Coverage of the population mean (0) by two_level (or, with `S_b`, of the true difference 0 by paired between
    two learners with independent seeds, S vs S_b, sharing the worlds' main effects) under the random-effects model
    of the module docstring. Returns {"coverage", "mean_width", "reps", ...}."""
    rng = np.random.default_rng(rng_seed)
    hit, width = 0, []
    for r in range(reps):
        b = rng.normal(0, world_sd, (1, W))
        A = rng.normal(0, seed_sd, (S, 1)) + b + rng.normal(0, noise_sd, (S, W))
        if S_b is None:
            ci = two_level(A, n_boot=n_boot, level=level, seed=r, method=method)
        else:
            Bm = rng.normal(0, seed_sd, (S_b, 1)) + b + rng.normal(0, noise_sd, (S_b, W))
            ci = paired(A, Bm, n_boot=n_boot, level=level, seed=r, method=method)
        if ci["lo"] is None:
            continue
        hit += ci["lo"] <= 0.0 <= ci["hi"]
        width.append(ci["hi"] - ci["lo"])
    n = len(width)
    return {"coverage": hit / max(n, 1), "mean_width": float(np.mean(width)) if width else None, "reps": n,
            "S": S, "S_b": S_b, "W": W, "seed_sd": seed_sd, "world_sd": world_sd, "noise_sd": noise_sd,
            "level": level, "method": method}
