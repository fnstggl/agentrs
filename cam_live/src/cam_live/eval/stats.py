"""cam_live.eval.stats — the statistics every result of ``cam_live.eval`` is reported with (spec 16.9).

All intervals are percentile bootstrap intervals over the *independent unit* of the experiment (a seed, i.e. one
agent lifetime sequence, or one world), never over correlated decisions of one lifetime. Tests are exact or
Monte-Carlo permutation tests on the same units. Everything is deterministic given ``seed``.

    mean_ci(xs)                       mean with a bootstrap interval
    diff_ci(xs, ys, paired=...)       difference of means, interval and a permutation p-value
    wilson(k, n)                      binomial proportion interval
    n_to_threshold(curve, q, ...)     N_q: experience until the competence threshold is reached (censored if never)
    speedup(n_init, n_exp, ...)       S_q = N_q(initial) / N_q(experienced) with a bootstrap interval
    aulc(curve)                       normalised area under the learning curve
    brier / ece / reliability         calibration of probability forecasts, clustered intervals
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

Z95 = 1.959963984540054


def _clean(xs: Sequence[Optional[float]]) -> List[float]:
    return [float(x) for x in xs if x is not None and math.isfinite(float(x))]


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _pct(sorted_xs: Sequence[float], q: float) -> float:
    if not sorted_xs:
        return float("nan")
    k = (len(sorted_xs) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (k - lo)


def mean_ci(xs: Sequence[Optional[float]], *, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0) -> Dict[str, Any]:
    """Mean of independent units with a percentile bootstrap interval (and the normal-theory SE for reference)."""
    v = _clean(xs)
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None, "se": None}
    m = _mean(v)
    se = (math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) / math.sqrt(n)) if n > 1 else None
    if n == 1:
        return {"n": 1, "mean": m, "lo": None, "hi": None, "se": None}
    rng = random.Random(seed)
    boots = sorted(_mean([v[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return {"n": n, "mean": m, "lo": _pct(boots, alpha / 2), "hi": _pct(boots, 1 - alpha / 2), "se": se}


def diff_ci(xs: Sequence[Optional[float]], ys: Sequence[Optional[float]], *, paired: bool = False,
            n_boot: int = 4000, n_perm: int = 10000, alpha: float = 0.05, seed: int = 0) -> Dict[str, Any]:
    """mean(xs) - mean(ys): bootstrap interval and a two-sided permutation p-value (sign flips when paired)."""
    rng = random.Random(seed)
    if paired:
        pairs = [(float(a), float(b)) for a, b in zip(xs, ys) if a is not None and b is not None
                 and math.isfinite(float(a)) and math.isfinite(float(b))]
        d = [a - b for a, b in pairs]
        out = mean_ci(d, n_boot=n_boot, alpha=alpha, seed=seed)
        out["paired"] = True
        if len(d) < 2:
            out["p"] = None
            return out
        obs = abs(_mean(d))
        if len(d) <= 16:                                    # exact sign-flip distribution
            import itertools
            tot = ge = 0
            for signs in itertools.product((1, -1), repeat=len(d)):
                tot += 1
                ge += abs(_mean([s * x for s, x in zip(signs, d)])) >= obs - 1e-12
            out["p"] = ge / tot
            out["test"] = "exact sign-flip permutation"
        else:
            ge = sum(abs(_mean([x if rng.random() < 0.5 else -x for x in d])) >= obs - 1e-12 for _ in range(n_perm))
            out["p"] = (ge + 1) / (n_perm + 1)
            out["test"] = "Monte-Carlo sign-flip permutation"
        return out
    a, b = _clean(xs), _clean(ys)
    out: Dict[str, Any] = {"paired": False, "n_x": len(a), "n_y": len(b)}
    if not a or not b:
        out.update(mean=None, lo=None, hi=None, p=None)
        return out
    obs = _mean(a) - _mean(b)
    boots = sorted(_mean([a[rng.randrange(len(a))] for _ in a]) - _mean([b[rng.randrange(len(b))] for _ in b])
                   for _ in range(n_boot))
    pool = a + b
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        ge += abs(_mean(pool[:len(a)]) - _mean(pool[len(a):])) >= abs(obs) - 1e-12
    out.update(mean=obs, lo=_pct(boots, alpha / 2), hi=_pct(boots, 1 - alpha / 2), p=(ge + 1) / (n_perm + 1),
               test="Monte-Carlo label permutation")
    return out


def wilson(k: int, n: int, z: float = Z95) -> Dict[str, Any]:
    if n <= 0:
        return {"k": k, "n": n, "p": None, "lo": None, "hi": None}
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"k": k, "n": n, "p": p, "lo": max(0.0, c - h), "hi": min(1.0, c + h)}


# ------------------------------------------------------------------------------------------------ learning curves
def rolling(xs: Sequence[float], w: int) -> List[float]:
    out = []
    for i in range(len(xs)):
        seg = xs[max(0, i - w + 1):i + 1]
        out.append(_mean(seg))
    return out


def n_to_threshold(curve: Sequence[float], q: float, *, experience: Optional[Sequence[float]] = None,
                   window: int = 1, min_points: int = 1) -> Dict[str, Any]:
    """N_q for one unit. ``curve[i]`` is the competence measured on unit of experience i (a lifetime or a decision);
    ``experience[i]`` the cumulative new experience consumed up to and including it (default i + 1).

    Reached at the first i >= min_points - 1 where the mean of the last ``window`` points is >= q; N_q is the
    experience consumed up to and including that point. Never reached: ``reached=False`` and ``n`` is the total
    experience (a right-censored observation)."""
    exp = list(experience) if experience is not None else [float(i + 1) for i in range(len(curve))]
    r = rolling(list(curve), window)
    for i, v in enumerate(r):
        if i >= min_points - 1 and v >= q:
            return {"reached": True, "n": float(exp[i]), "index": i}
    return {"reached": False, "n": float(exp[-1]) if exp else 0.0, "index": None}


def aulc(curve: Sequence[float]) -> Optional[float]:
    """Normalised area under the learning curve: the mean competence over the whole experience budget."""
    v = _clean(curve)
    return _mean(v) if v else None


def summarise_nq(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per condition: how many units reached q (Wilson interval), the restricted mean N_q (censored at the budget,
    a lower bound of the true mean when anything is censored) and the median N_q if more than half reached it."""
    n = len(results)
    k = sum(1 for r in results if r["reached"])
    ns = sorted(float(r["n"]) for r in results)
    med = None
    if k * 2 > n:
        med = _pct(ns, 0.5)
    return {"n_units": n, "reached": wilson(k, n), "restricted_mean_n": mean_ci(ns), "median_n": med,
            "never_reached": n - k, "censoring_note": "units that never reached q enter with N = experience budget"}


def speedup(n_init: Sequence[Dict[str, Any]], n_exp: Sequence[Dict[str, Any]], *, paired: bool = True,
            n_boot: int = 4000, seed: int = 0) -> Dict[str, Any]:
    """S_q = N_q(initial) / N_q(experienced) as the ratio of restricted mean N_q (censored at the budget), with a
    bootstrap interval over units (paired by unit index when ``paired``) and the log-ratio permutation test.
    When both versions are censored in a unit that unit contributes a ratio of exactly 1 (no evidence either way)."""
    a = [float(r["n"]) for r in n_init]
    b = [float(r["n"]) for r in n_exp]
    if not a or not b:
        return {"S_q": None}
    s = _mean(a) / _mean(b) if _mean(b) > 0 else None
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        if paired and len(a) == len(b):
            idx = [rng.randrange(len(a)) for _ in a]
            ma, mb = _mean([a[i] for i in idx]), _mean([b[i] for i in idx])
        else:
            ma = _mean([a[rng.randrange(len(a))] for _ in a])
            mb = _mean([b[rng.randrange(len(b))] for _ in b])
        if mb > 0:
            boots.append(ma / mb)
    boots.sort()
    logs_a = [math.log(max(x, 1e-9)) for x in a]
    logs_b = [math.log(max(x, 1e-9)) for x in b]
    test = diff_ci(logs_a, logs_b, paired=paired and len(a) == len(b), seed=seed)
    both_censored = sum(1 for x, y in zip(n_init, n_exp) if not x["reached"] and not y["reached"]) if paired else None
    return {"S_q": s, "lo": _pct(boots, 0.025) if boots else None, "hi": _pct(boots, 0.975) if boots else None,
            "log_ratio": {"mean": test.get("mean"), "lo": test.get("lo"), "hi": test.get("hi"), "p": test.get("p")},
            "n_units": len(a), "units_both_censored": both_censored,
            "reached_initial": sum(1 for r in n_init if r["reached"]),
            "reached_experienced": sum(1 for r in n_exp if r["reached"])}


# ------------------------------------------------------------------------------------------------ calibration
def brier(p: Sequence[float], y: Sequence[float]) -> float:
    return _mean([(float(a) - float(b)) ** 2 for a, b in zip(p, y)])


def ece(p: Sequence[float], y: Sequence[float], bins: int = 10) -> float:
    tot, n = 0.0, len(p)
    for k in range(bins):
        lo, hi = k / bins, (k + 1) / bins
        idx = [i for i, x in enumerate(p) if (lo <= x < hi) or (k == bins - 1 and x == 1.0)]
        if idx:
            tot += len(idx) / n * abs(_mean([p[i] for i in idx]) - _mean([y[i] for i in idx]))
    return tot


def reliability(p: Sequence[float], y: Sequence[float], bins: int = 5) -> List[Dict[str, Any]]:
    out = []
    for k in range(bins):
        lo, hi = k / bins, (k + 1) / bins
        idx = [i for i, x in enumerate(p) if (lo <= x < hi) or (k == bins - 1 and x == 1.0)]
        if idx:
            out.append({"bin": [lo, hi], "n": len(idx), "mean_forecast": _mean([p[i] for i in idx]),
                        "observed_rate": _mean([y[i] for i in idx]), "wilson": wilson(int(sum(y[i] for i in idx)),
                                                                                       len(idx))})
    return out


def clustered_ci(values_by_cluster: Dict[str, List[float]], stat: str = "mean", *, n_boot: int = 4000,
                 seed: int = 0) -> Dict[str, Any]:
    """Bootstrap over clusters (e.g. lifetimes) of the pooled mean of the values: forecasts of one lifetime are
    correlated, so the cluster is the resampling unit."""
    keys = [k for k, v in values_by_cluster.items() if v]
    pooled = [x for k in keys for x in values_by_cluster[k]]
    if not pooled:
        return {"n": 0, "clusters": 0, "mean": None, "lo": None, "hi": None}
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        xs = [x for k in pick for x in values_by_cluster[k]]
        boots.append(_mean(xs))
    boots.sort()
    return {"n": len(pooled), "clusters": len(keys), "mean": _mean(pooled), "lo": _pct(boots, 0.025),
            "hi": _pct(boots, 0.975)}


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 3:
        return None

    def ranks(v: Sequence[float]) -> List[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = _mean(rx), _mean(ry)
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy)


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 3:
        return None
    mx, my = _mean(x), _mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)
