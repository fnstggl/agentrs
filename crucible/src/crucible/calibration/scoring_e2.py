"""Held-out evaluation metrics for workstream E2 (NumPy only, no dataset access).

Probabilistic: log loss, Brier score, expected calibration error (equal-mass bins), ROC AUC, average precision,
reliability tables. Uplift: Qini and uplift curves with tie handling, Qini coefficient normalised by the perfect
ranking, uplift by predicted decile, difference-in-means ATE. Counts: Poisson deviance, negative-binomial dispersion
and sampling, Spearman correlation, top-k recall, KS distance. Uncertainty: paired bootstrap of metric differences.
"""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-15


# ----------------------------------------------------------------------------------------------- probabilities

def log_loss(y, p) -> float:
    y = np.asarray(y, dtype=np.float64); p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def log_loss_rows(y, p) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64); p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def brier(y, p) -> float:
    y = np.asarray(y, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def _equal_mass_bins(p: np.ndarray, n_bins: int) -> np.ndarray:
    order = np.argsort(p, kind="stable")
    bins = np.empty(p.size, dtype=np.int64)
    bins[order] = (np.arange(p.size) * n_bins) // max(p.size, 1)
    return bins


def reliability(y, p, n_bins: int = 15) -> list[dict]:
    y = np.asarray(y, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    b = _equal_mass_bins(p, n_bins)
    n = np.bincount(b, minlength=n_bins).astype(np.float64)
    sp = np.bincount(b, weights=p, minlength=n_bins); sy = np.bincount(b, weights=y, minlength=n_bins)
    rows = []
    for i in range(n_bins):
        if n[i] > 0:
            rows.append({"bin": i, "n": int(n[i]), "mean_predicted": float(sp[i] / n[i]), "mean_observed": float(sy[i] / n[i])})
    return rows


def ece(y, p, n_bins: int = 15) -> float:
    rows = reliability(y, p, n_bins)
    total = sum(r["n"] for r in rows)
    return float(sum(r["n"] / total * abs(r["mean_observed"] - r["mean_predicted"]) for r in rows)) if total else 0.0


def rankdata(a) -> np.ndarray:
    """Average ranks (1-based) with ties averaged."""
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    boundaries = np.r_[True, s[1:] != s[:-1]]
    group = np.cumsum(boundaries) - 1
    starts = np.flatnonzero(boundaries)
    ends = np.r_[starts[1:], s.size]
    avg = (starts + ends + 1) / 2.0
    ranks = np.empty(a.size, dtype=np.float64)
    ranks[order] = avg[group]
    return ranks


def auc(y, p) -> float:
    y = np.asarray(y).astype(bool)
    n1 = int(y.sum()); n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, p) -> float:
    y = np.asarray(y).astype(np.float64); p = np.asarray(p, dtype=np.float64)
    order = np.argsort(-p, kind="stable")
    ys = y[order]; ps = p[order]
    tp = np.cumsum(ys)
    last = np.r_[ps[1:] != ps[:-1], True]          # evaluate at the end of each tie group
    tp_g = tp[last]; k_g = (np.flatnonzero(last) + 1).astype(np.float64)
    if tp[-1] == 0:
        return float("nan")
    precision = tp_g / k_g
    recall = tp_g / tp[-1]
    prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev) * precision))


def binary_report(y, p, n_bins: int = 15) -> dict:
    if len(y) == 0:
        return {"n": 0}
    return {"n": int(len(y)), "log_loss": log_loss(y, p), "brier": brier(y, p), "ece": ece(y, p, n_bins),
            "auc": auc(y, p), "mean_predicted": float(np.mean(p)), "mean_observed": float(np.mean(y))}


# ----------------------------------------------------------------------------------------------- uplift

def _uplift_cumulants(y, t, score, w=None, order=None):
    """Cumulative (weighted) treated/control counts and responders at the end of each tie group, descending score.

    `w` are row weights (bootstrap multiplicities); `order` a precomputed descending-score order."""
    y = np.asarray(y, dtype=np.float64); t = np.asarray(t).astype(bool); s = np.asarray(score, dtype=np.float64)
    if order is None:
        order = np.argsort(-s, kind="stable")
    ys, ts, ss = y[order], t[order], s[order]
    ws = np.ones(ys.size) if w is None else np.asarray(w, dtype=np.float64)[order]
    wt = ws * ts; wc = ws * ~ts
    k = np.cumsum(ws); nt = np.cumsum(wt); nc = np.cumsum(wc)
    yt = np.cumsum(ys * wt); yc = np.cumsum(ys * wc)
    last = np.r_[ss[1:] != ss[:-1], True]
    z = np.zeros(1)
    return np.r_[z, k[last]], np.r_[z, nt[last]], np.r_[z, nc[last]], np.r_[z, yt[last]], np.r_[z, yc[last]]


def qini_curve(y, t, score, w=None, order=None):
    """x = number targeted, q = Y_T(k) - Y_C(k) * N_T(k) / N_C(k); evaluated at tie-group ends, linear in between."""
    k, nt, nc, yt, yc = _uplift_cumulants(y, t, score, w, order)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = yt - np.where(nc > 0, yc * nt / np.where(nc > 0, nc, 1.0), 0.0)
    return k, q


def _area(x, yv) -> float:
    return float(np.sum((x[1:] - x[:-1]) * (yv[1:] + yv[:-1]) / 2.0))


def qini_coefficient(y, t, score, w=None, orders=None) -> dict:
    """Area between the Qini curve and the random line, raw (per targeted user) and normalised by the perfect ranking.

    `orders` = (model order, perfect order) precomputed for repeated (bootstrap) evaluation."""
    y = np.asarray(y, dtype=np.float64); t = np.asarray(t).astype(bool)
    perfect = y * t - y * (~t)                     # treated responders first, control responders last
    om, op = orders if orders is not None else (None, None)
    x, q = qini_curve(y, t, score, w, om)
    n = x[-1]
    random_area = q[-1] * n / 2.0
    model_area = _area(x, q)
    xp, qp = qini_curve(y, t, perfect, w, op)
    perfect_area = _area(xp, qp)
    denom = perfect_area - random_area
    return {"qini_area_per_user": (model_area - random_area) / (n * n) if n else 0.0,
            "qini_coefficient": (model_area - random_area) / denom if denom != 0 else float("nan")}


def qini_bootstrap(y, t, score, n_boot: int = 200, seed: int = 0) -> dict:
    """Bootstrap distribution of the normalised Qini coefficient (rows resampled with replacement)."""
    y = np.asarray(y, dtype=np.float64); t = np.asarray(t).astype(bool); s = np.asarray(score, dtype=np.float64)
    orders = (np.argsort(-s, kind="stable"), np.argsort(-(y * t - y * (~t)), kind="stable"))
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, y.size, size=y.size), minlength=y.size)
        vals[i] = qini_coefficient(y, t, s, w=w, orders=orders)["qini_coefficient"]
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return {"point": qini_coefficient(y, t, s, orders=orders)["qini_coefficient"], "se": float(vals.std(ddof=1)),
            "ci95": [float(lo), float(hi)], "n_boot": n_boot}


def auuc_normalised(y, t, score) -> float:
    """Area under the uplift curve U(k) = (Y_T/N_T - Y_C/N_C) * k minus the random line, over the perfect ranking's."""
    y = np.asarray(y, dtype=np.float64); t = np.asarray(t).astype(bool)

    def curve(sc):
        k, nt, nc, yt, yc = _uplift_cumulants(y, t, sc)
        with np.errstate(invalid="ignore", divide="ignore"):
            u = np.where((nt > 0) & (nc > 0), (yt / np.maximum(nt, 1) - yc / np.maximum(nc, 1)) * k, 0.0)
        return k, u
    x, u = curve(score)
    n = x[-1]
    rand = u[-1] * n / 2.0
    xp, up = curve(y * t - y * (~t))
    denom = _area(xp, up) - rand
    return float((_area(x, u) - rand) / denom) if denom != 0 else float("nan")


def ate(y, t) -> dict:
    y = np.asarray(y, dtype=np.float64); t = np.asarray(t).astype(bool)
    p1 = y[t].mean(); p0 = y[~t].mean()
    se = math.sqrt(p1 * (1 - p1) / t.sum() + p0 * (1 - p0) / (~t).sum())
    return {"treated_rate": float(p1), "control_rate": float(p0), "ate": float(p1 - p0), "se": se,
            "ci95": [float(p1 - p0 - 1.96 * se), float(p1 - p0 + 1.96 * se)],
            "relative_lift": float((p1 - p0) / p0) if p0 > 0 else float("nan")}


def uplift_by_decile(y, t, tau, n_bins: int = 10) -> list[dict]:
    y = np.asarray(y, dtype=np.float64); t = np.asarray(t).astype(bool); tau = np.asarray(tau, dtype=np.float64)
    b = _equal_mass_bins(tau, n_bins)
    rows = []
    for i in range(n_bins):
        m = b == i
        mt = m & t; mc = m & ~t
        if mt.sum() == 0 or mc.sum() == 0:
            continue
        p1 = y[mt].mean(); p0 = y[mc].mean()
        se = math.sqrt(p1 * (1 - p1) / mt.sum() + p0 * (1 - p0) / mc.sum())
        rows.append({"bin": i, "n": int(m.sum()), "mean_predicted_effect": float(tau[m].mean()),
                     "observed_effect": float(p1 - p0), "se": se})
    return rows


# ----------------------------------------------------------------------------------------------- counts

def poisson_deviance_rows(y, mu) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64); mu = np.maximum(np.asarray(mu, dtype=np.float64), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2.0 * (term - (y - mu))


def poisson_deviance(y, mu) -> float:
    return float(np.mean(poisson_deviance_rows(y, mu)))


def spearman(a, b) -> float:
    ra = rankdata(a); rb = rankdata(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = math.sqrt(float(np.dot(ra, ra) * np.dot(rb, rb)))
    return float(np.dot(ra, rb) / d) if d > 0 else float("nan")


def top_recall(y, mu, frac: float = 0.01) -> float:
    """Share of the actual top `frac` (by y, ties broken by order) that the prediction ranks in its top `frac`."""
    y = np.asarray(y); mu = np.asarray(mu)
    k = max(1, int(round(frac * y.size)))
    top_true = set(np.argsort(-y, kind="stable")[:k].tolist())
    top_pred = np.argsort(-mu, kind="stable")[:k]
    return float(sum(1 for i in top_pred.tolist() if i in top_true) / k)


def count_calibration(y, mu, n_bins: int = 10) -> list[dict]:
    y = np.asarray(y, dtype=np.float64); mu = np.asarray(mu, dtype=np.float64)
    b = _equal_mass_bins(mu, n_bins)
    rows = []
    for i in range(n_bins):
        m = b == i
        if m.any():
            rows.append({"bin": i, "n": int(m.sum()), "mean_predicted": float(mu[m].mean()), "mean_observed": float(y[m].mean())})
    return rows


def nb_loglik(y, mu, alpha: float) -> float:
    """Negative-binomial (NB2, variance mu + alpha mu^2) mean log-likelihood."""
    from math import lgamma
    y = np.asarray(y, dtype=np.float64); mu = np.maximum(np.asarray(mu, dtype=np.float64), 1e-12)
    r = 1.0 / alpha
    lg = np.vectorize(lgamma)
    uy, inv = np.unique(y, return_inverse=True)
    lgy = (lg(uy + r) - lg(uy + 1.0))[inv] - lgamma(r)
    return float(np.mean(lgy + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))))


def fit_nb_alpha(y, mu, lo: float = 1e-4, hi: float = 100.0, iters: int = 60) -> float:
    """Maximum-likelihood NB2 dispersion given fitted means (golden-section search on log alpha)."""
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nb_loglik(y, mu, math.exp(c)), nb_loglik(y, mu, math.exp(d))
    for _ in range(iters):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = nb_loglik(y, mu, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = nb_loglik(y, mu, math.exp(d))
    return float(math.exp((a + b) / 2))


def sample_nb(rng: np.random.Generator, mu, alpha: float) -> np.ndarray:
    mu = np.maximum(np.asarray(mu, dtype=np.float64), 1e-12)
    r = 1.0 / alpha
    lam = rng.gamma(shape=r, scale=mu / r)
    return rng.poisson(lam)


def ks_distance(a, b) -> float:
    a = np.sort(np.asarray(a, dtype=np.float64)); b = np.sort(np.asarray(b, dtype=np.float64))
    grid = np.union1d(a, b)
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def count_distribution(y) -> dict:
    y = np.asarray(y, dtype=np.float64)
    q = np.quantile(y, [0.5, 0.9, 0.99, 0.999])
    return {"mean": float(y.mean()), "share_zero": float(np.mean(y == 0)), "share_ge_10": float(np.mean(y >= 10)),
            "q50": float(q[0]), "q90": float(q[1]), "q99": float(q[2]), "q999": float(q[3])}


# ----------------------------------------------------------------------------------------------- uncertainty

def paired_bootstrap_mean_diff(d_rows: np.ndarray, n_boot: int = 200, seed: int = 0) -> dict:
    """Bootstrap of mean(d) by resampling rows (multinomial weights). Returns mean, se and the 95% interval."""
    d = np.asarray(d_rows, dtype=np.float64)
    n = d.size
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.integers(0, n, size=n), minlength=n)
        means[i] = float(np.dot(w, d) / n)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(d.mean()), "se": float(means.std(ddof=1)), "ci95": [float(lo), float(hi)], "n_boot": n_boot}


def weighted_corr(x, y, w) -> float:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64); w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    mx, my = np.dot(w, x), np.dot(w, y)
    cov = np.dot(w, (x - mx) * (y - my))
    vx, vy = np.dot(w, (x - mx) ** 2), np.dot(w, (y - my) ** 2)
    return float(cov / math.sqrt(vx * vy)) if vx > 0 and vy > 0 else float("nan")
