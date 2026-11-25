"""Held-out evaluation metrics (MASTER_SPEC 5.5): log loss, Brier score, expected calibration error, ranking quality,
distributional checks for quantile models, Poisson deviance and user-cluster bootstrap intervals.

All functions are deterministic NumPy code; the bootstrap takes an explicit seed.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-7


def log_loss(y, p) -> float:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def row_log_loss(y, p) -> np.ndarray:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def calibration_table(y, p, bins: int = 15) -> list[dict]:
    """Equal-width reliability table: per bin {lo, hi, n, mean_pred, mean_obs}."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    out = []
    for i in range(bins):
        m = b == i
        if m.any():
            out.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": int(m.sum()),
                        "mean_pred": float(p[m].mean()), "mean_obs": float(y[m].mean())})
    return out


def ece(y, p, bins: int = 15) -> float:
    n = len(y)
    return float(sum(r["n"] / n * abs(r["mean_pred"] - r["mean_obs"]) for r in calibration_table(y, p, bins)))


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), float)
    # average ranks for ties
    starts = np.r_[0, np.flatnonzero(np.diff(xs)) + 1]
    ends = np.r_[starts[1:], len(xs)]
    avg = (starts + ends - 1) / 2.0 + 1.0
    ranks[order] = np.repeat(avg, ends - starts)
    return ranks


def auc(y, p) -> float:
    y = np.asarray(y, bool)
    p = np.asarray(p, float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def binary_report(y, p, bins: int = 15) -> dict:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return {"n": int(len(y)), "log_loss": log_loss(y, p), "brier": brier(y, p), "ece": ece(y, p, bins),
            "auc": auc(y, p), "mean_pred": float(p.mean()), "mean_obs": float(y.mean())}


# ---------------------------------------------------------------- ranking within impressions (MIND)
def group_ranking(y, p, groups) -> dict:
    """Mean impression AUC, MRR, nDCG@5 and nDCG@10 over groups with at least one positive and one negative."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    groups = np.asarray(groups)
    order = np.lexsort((-p, groups))
    g, ys = groups[order], y[order]
    starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
    ends = np.r_[starts[1:], len(g)]
    aucs, mrrs, n5, n10 = [], [], [], []
    ps = p[order]
    for s, e in zip(starts, ends):
        yy = ys[s:e]
        k1 = yy.sum()
        if k1 == 0 or k1 == len(yy):
            continue
        aucs.append(auc(yy, ps[s:e]))
        pos = np.arange(1, len(yy) + 1)
        mrrs.append(float((yy / pos).sum() / k1))
        for k, acc in ((5, n5), (10, n10)):
            disc = 1.0 / np.log2(pos[:k] + 1)
            dcg = float((yy[:k] * disc).sum())
            ideal = float(disc[:int(min(k1, k))].sum())
            acc.append(dcg / ideal)
    return {"impressions": len(aucs), "impression_auc": float(np.mean(aucs)), "mrr": float(np.mean(mrrs)),
            "ndcg5": float(np.mean(n5)), "ndcg10": float(np.mean(n10))}


# ---------------------------------------------------------------- distributional (quantile) models
def pinball_rows(y, Q, levels) -> np.ndarray:
    y = np.asarray(y, float)[:, None]
    d = y - np.asarray(Q, float)
    tau = np.asarray(levels, float)[None, :]
    return np.mean(np.maximum(tau * d, (tau - 1) * d), axis=1)


def pinball(y, Q, levels) -> float:
    return float(np.mean(pinball_rows(y, Q, levels)))


def pit(y, Q, levels) -> np.ndarray:
    """Probability integral transform values: the predictive CDF at y, linear between quantile levels and linear
    towards 0 / 1 outside the outer levels (within one outer-gap width)."""
    y = np.asarray(y, float)
    Q = np.asarray(Q, float)
    lv = np.asarray(levels, float)
    out = np.empty(len(y))
    for i in range(len(y)):
        q = Q[i]
        if y[i] <= q[0]:
            w = max(q[1] - q[0], 1e-9)
            out[i] = max(0.0, lv[0] * (1 - (q[0] - y[i]) / w))
        elif y[i] >= q[-1]:
            w = max(q[-1] - q[-2], 1e-9)
            out[i] = min(1.0, lv[-1] + (1 - lv[-1]) * min(1.0, (y[i] - q[-1]) / w))
        else:
            j = int(np.searchsorted(q, y[i], side="right") - 1)
            j = min(max(j, 0), len(q) - 2)
            span = q[j + 1] - q[j]
            t = 0.5 if span <= 0 else (y[i] - q[j]) / span
            out[i] = lv[j] + t * (lv[j + 1] - lv[j])
    return out


def pit_fast(y, Q, levels) -> np.ndarray:
    """Vectorised `pit` (same definition)."""
    y = np.asarray(y, float)
    Q = np.asarray(Q, float)
    lv = np.asarray(levels, float)
    n, k = Q.shape
    j = (Q <= y[:, None]).sum(axis=1) - 1                      # index of the last quantile <= y
    out = np.empty(n)
    below = j < 0
    above = j >= k - 1
    mid = ~(below | above)
    w0 = np.maximum(Q[:, 1] - Q[:, 0], 1e-9)
    out[below] = np.maximum(0.0, lv[0] * (1 - (Q[below, 0] - y[below]) / w0[below]))
    w1 = np.maximum(Q[:, -1] - Q[:, -2], 1e-9)
    out[above] = np.minimum(1.0, lv[-1] + (1 - lv[-1]) * np.minimum(1.0, (y[above] - Q[above, -1]) / w1[above]))
    jm = j[mid]
    rows = np.flatnonzero(mid)
    lo, hi = Q[rows, jm], Q[rows, jm + 1]
    span = hi - lo
    t = np.where(span > 0, (y[mid] - lo) / np.where(span > 0, span, 1), 0.5)
    out[mid] = lv[jm] + t * (lv[jm + 1] - lv[jm])
    return out


def ks_uniform(u) -> float:
    u = np.sort(np.asarray(u, float))
    n = len(u)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


def coverage(y, Q, levels, lo: float, hi: float) -> float:
    lv = list(np.round(levels, 6))
    a, b = lv.index(round(lo, 6)), lv.index(round(hi, 6))
    y = np.asarray(y, float)
    return float(np.mean((y >= Q[:, a]) & (y <= Q[:, b])))


def wasserstein1(a, b) -> float:
    """1-Wasserstein distance between two samples (equal-quantile coupling on a common grid)."""
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    g = (np.arange(2000) + 0.5) / 2000
    return float(np.mean(np.abs(np.quantile(a, g) - np.quantile(b, g))))


def ks_two_sample(a, b) -> float:
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


# ---------------------------------------------------------------- counts
def poisson_deviance(y, mu) -> float:
    y = np.asarray(y, float)
    mu = np.maximum(np.asarray(mu, float), 1e-9)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(2 * np.mean(t - (y - mu)))


def total_variation(p, q) -> float:
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    return float(0.5 * np.abs(p / p.sum() - q / q.sum()).sum())


# ---------------------------------------------------------------- uncertainty
def cluster_bootstrap(values, groups, n_boot: int = 200, seed: int = 0, weights=None) -> dict:
    """95% interval of a mean of per-row `values` when rows are clustered in `groups` (users): users are resampled
    with replacement. Returns {mean, lo, hi, n_boot}."""
    values = np.asarray(values, float)
    w = np.ones(len(values)) if weights is None else np.asarray(weights, float)
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    G = inv.max() + 1
    s = np.bincount(inv, weights=values * w, minlength=G)
    c = np.bincount(inv, weights=w, minlength=G)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        k = np.bincount(rng.integers(0, G, G), minlength=G)
        stats[b] = (k * s).sum() / (k * c).sum()
    return {"mean": float(s.sum() / c.sum()), "lo": float(np.quantile(stats, 0.025)),
            "hi": float(np.quantile(stats, 0.975)), "n_boot": n_boot, "clusters": int(G)}
