"""Statistics for comparing learners. Deterministic (seeded resampling), dependency-free beyond numpy.

Conventions used throughout the research platform:

* A *run* is one (learner seed, world) pair; uncertainty is always over runs, never over time steps.
* Comparisons between learners are *paired* on identical worlds whenever the experiment allows it.
* Intervals are percentile bootstrap intervals; with fewer than 5 runs they are reported but flagged
  (`"low_n": True`), because a bootstrap over 3 points is not a confidence statement worth relying on.
* Nothing here pools different quantities into one score.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

MIN_N = 5


def _arr(x: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(x) if not isinstance(x, np.ndarray) else x, dtype=float).ravel()
    return a[np.isfinite(a)]


def mean_ci(x: Iterable[float], level: float = 0.95, n_boot: int = 4000, seed: int = 0) -> dict:
    """Mean with a percentile bootstrap interval."""
    a = _arr(x)
    if len(a) == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None, "low_n": True}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    m = a[idx].mean(axis=1)
    q = (1 - level) / 2
    return {"n": int(len(a)), "mean": float(a.mean()), "lo": float(np.quantile(m, q)),
            "hi": float(np.quantile(m, 1 - q)), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "low_n": len(a) < MIN_N}


def proportion_ci(successes: int, n: int, level: float = 0.95) -> dict:
    """Wilson score interval for a success probability (well behaved at 0 and 1 and at small n)."""
    if n <= 0:
        return {"n": 0, "p": None, "lo": None, "hi": None, "low_n": True}
    z = _z(level)
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = 0.0 if successes <= 0 else float(max(0.0, centre - half))     # the bounds are exact at the edges;
    hi = 1.0 if successes >= n else float(min(1.0, centre + half))     # do not let rounding move them
    return {"n": int(n), "p": float(p), "lo": lo, "hi": hi, "low_n": n < MIN_N}


def _z(level: float) -> float:
    # inverse normal CDF at (1 + level) / 2 by bisection (no scipy dependency)
    from math import erf, sqrt
    target = (1 + level) / 2
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def iqm(x: Iterable[float]) -> float | None:
    """Interquartile mean: the mean of the middle half. Less sensitive to a few extreme runs than the mean."""
    a = np.sort(_arr(x))
    if len(a) == 0:
        return None
    if len(a) < 4:
        return float(a.mean())
    k = len(a) // 4
    return float(a[k:len(a) - k].mean())


def paired_diff(a: Sequence[float], b: Sequence[float], level: float = 0.95, n_boot: int = 4000,
                n_perm: int = 10000, seed: int = 0) -> dict:
    """Paired comparison of `a` against `b` (same worlds, same order): mean of a - b with a bootstrap interval,
    a two-sided sign-flip permutation p-value, and the fraction of pairs where a is better."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same length")
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    if len(d) == 0:
        return {"n": 0, "mean_diff": None, "lo": None, "hi": None, "p_value": None, "frac_better": None,
                "low_n": True}
    ci = mean_ci(d, level, n_boot, seed)
    rng = np.random.default_rng(seed + 1)
    obs = abs(d.mean())
    if len(d) <= 16:                       # exact enumeration of all sign patterns
        signs = np.array([[1 if (k >> i) & 1 else -1 for i in range(len(d))] for k in range(2 ** len(d))])
        p = float((np.abs((signs * d).mean(axis=1)) >= obs - 1e-15).mean())
    else:
        signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
        p = float(((np.abs((signs * d).mean(axis=1)) >= obs - 1e-15).sum() + 1) / (n_perm + 1))
    return {"n": int(len(d)), "mean_diff": float(d.mean()), "lo": ci["lo"], "hi": ci["hi"], "p_value": p,
            "frac_better": float((d > 0).mean()), "frac_worse": float((d < 0).mean()), "low_n": len(d) < MIN_N}


def unpaired_diff(a: Iterable[float], b: Iterable[float], level: float = 0.95, n_boot: int = 4000,
                  seed: int = 0) -> dict:
    """Difference of means of two independent samples with a bootstrap interval (use only when pairing is
    impossible, e.g. different learner seeds on different worlds)."""
    a, b = _arr(a), _arr(b)
    if len(a) == 0 or len(b) == 0:
        return {"n_a": int(len(a)), "n_b": int(len(b)), "mean_diff": None, "lo": None, "hi": None, "low_n": True}
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(axis=1)
    q = (1 - level) / 2
    return {"n_a": int(len(a)), "n_b": int(len(b)), "mean_diff": float(a.mean() - b.mean()),
            "lo": float(np.quantile(da - db, q)), "hi": float(np.quantile(da - db, 1 - q)),
            "low_n": min(len(a), len(b)) < MIN_N}


def prob_improvement(a: Iterable[float], b: Iterable[float]) -> float | None:
    """P(a run of A beats a run of B), ties counted half (the Mann-Whitney statistic as a probability)."""
    a, b = _arr(a), _arr(b)
    if len(a) == 0 or len(b) == 0:
        return None
    gt = (a[:, None] > b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(a) * len(b)))


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values for a family of comparisons."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return [float(v) for v in adj]


# ---------------------------------------------------------------------------- learning curves
def auc(x: Sequence[float], y: Sequence[float], x_max: float | None = None, normalise: bool = True) -> float | None:
    """Area under a learning curve y(x) by the trapezoid rule, up to `x_max` (linear interpolation at the cut;
    the last value is held if the curve ends early). Normalised by the x-range, it is the mean height of the
    curve: comparable across experiments with the same budget."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) == 0:
        return None
    o = np.argsort(x)
    x, y = x[o], y[o]
    if x_max is None:
        x_max = float(x[-1])
    if x[-1] < x_max:
        x = np.append(x, x_max)
        y = np.append(y, y[-1])
    elif x[-1] > x_max:
        y_cut = float(np.interp(x_max, x, y))
        keep = x < x_max
        x = np.append(x[keep], x_max)
        y = np.append(y[keep], y_cut)
    if len(x) == 1 or x[-1] == x[0]:
        return float(y[0])
    area = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))
    return area / float(x[-1] - x[0]) if normalise else area


def steps_to_threshold(x: Sequence[float], y: Sequence[float], threshold: float, sustain: int = 1) -> float | None:
    """First x at which y reaches `threshold` and stays there for `sustain` consecutive points (None if never).
    Linear interpolation between the last point below and the first point of the sustained stretch."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    for i in range(len(y) - sustain + 1):
        if np.all(y[i:i + sustain] >= threshold):
            if i == 0 or y[i - 1] >= threshold or y[i] == y[i - 1]:
                return float(x[i])
            f = (threshold - y[i - 1]) / (y[i] - y[i - 1])
            return float(x[i - 1] + f * (x[i] - x[i - 1]))
    return None


def at_budget(x: Sequence[float], y: Sequence[float], budget: float) -> float | None:
    """Performance after a fixed experience budget (linear interpolation; None before the first point)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) == 0 or budget < x.min():
        return None
    o = np.argsort(x)
    return float(np.interp(budget, x[o], y[o]))


def curve_band(curves: Sequence[tuple[Sequence[float], Sequence[float]]], grid: Sequence[float], level: float = 0.95,
               seed: int = 0) -> dict:
    """Mean learning curve over runs on a common grid, with a bootstrap band over runs."""
    grid = np.asarray(grid, float)
    rows = []
    for x, y in curves:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if len(x) == 0:
            continue
        o = np.argsort(x)
        v = np.interp(grid, x[o], y[o], left=np.nan)
        v[grid > x[o][-1]] = np.nan          # do not extrapolate a run past its last measurement
        rows.append(v)
    if not rows:
        return {"x": grid.tolist(), "mean": [], "lo": [], "hi": [], "n": []}
    m = np.vstack(rows)
    mean, lo, hi, ns = [], [], [], []
    for j in range(m.shape[1]):
        c = mean_ci(m[:, j], level, 1000, seed + j)
        mean.append(c["mean"])
        lo.append(c["lo"])
        hi.append(c["hi"])
        ns.append(c["n"])
    return {"x": grid.tolist(), "mean": mean, "lo": lo, "hi": hi, "n": ns}
