"""cam_live.predict.calibration — is the evaluator's stated confidence borne out by reality? (spec 6.4)

Binary metrics: Brier score, negative log-likelihood, calibration slope and intercept (logistic recalibration
``y ~ sigmoid(a + b * logit(p))``; a perfectly calibrated forecaster has a = 0, b = 1; over-confidence shows as
b < 1), reliability bins and expected calibration error. Every metric comes with a studentised confidence
interval, and reports can be grouped by objective family and horizon.

Forecast rows are rarely independent: the forecasts made at the successive prefixes of one decision or lifetime
share one outcome. Pass ``clusters`` (one label per row: decision, lifetime, experiment) wherever they are known;
then Brier / NLL intervals are over cluster means, the slope / intercept intervals use the cluster-robust (CR1)
sandwich covariance with G - 1 degrees of freedom and the ECE interval a cluster bootstrap. Without labels the
slope / intercept intervals use a HAC (Newey-West, Bartlett kernel, Andrews' AR(1) automatic bandwidth) sandwich
in the given row order: valid for independent rows and for dependence among neighbouring rows, but it cannot repair
dependence between rows far apart in the given order (``cov="iid"`` gives the old model-based Wald interval).

Only *prospective* predictions are scored: :func:`prospective_guard` refuses any prediction whose ``made_at`` is not
strictly before the time its outcome was observed, and :func:`score_predictions` applies it to stored
``PredictionRecord``s.
"""
from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..contracts import OutcomeStatus, PredictionRecord
from . import dists as D

EPS = 1e-12
DEFAULT_HORIZON_EDGES = (3600.0, 86400.0, 7 * 86400.0, 30 * 86400.0)


class RetrospectivePredictionError(ValueError):
    """A prediction that was not made strictly before its outcome was offered for scoring."""


def _np(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


@functools.lru_cache(maxsize=4096)
def t_quantile(q: float, df: float) -> float:
    return D.student_t_icdf_scalar(float(q), max(float(df), 1e-6))


def t_sf2(t: float, df: float) -> float:
    """Two-sided p-value of a t statistic."""
    t, df = float(t), max(float(df), 1e-6)
    return D.betainc_scalar(0.5 * df, 0.5, df / (df + t * t))        # = 2 * upper tail, accurate for tiny p


# ----------------------------------------------------------------------------------------------------------------
# per-item scores
# ----------------------------------------------------------------------------------------------------------------
def brier(p: Any, y: Any) -> np.ndarray:
    p, y = _np(p), _np(y)
    return (p - y) ** 2


def binary_nll(p: Any, y: Any) -> np.ndarray:
    p, y = np.clip(_np(p), EPS, 1 - EPS), _np(y)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def logit(p: Any) -> np.ndarray:
    p = np.clip(_np(p), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


# ----------------------------------------------------------------------------------------------------------------
# intervals
# ----------------------------------------------------------------------------------------------------------------
def studentised_ci(values: Any, level: float = 0.95, clusters: Optional[Sequence[Hashable]] = None
                   ) -> Dict[str, float]:
    """Mean with a Student-t interval. With ``clusters`` (one label per value, e.g. the experiment or trajectory a
    prediction belongs to) the interval is computed over cluster means, so correlated rows are not counted as
    independent evidence."""
    v = _np(values)
    if clusters is not None:
        groups: Dict[Hashable, List[float]] = {}
        for c, x in zip(clusters, v):
            groups.setdefault(c, []).append(float(x))
        v = np.asarray([np.mean(g) for g in groups.values()], dtype=np.float64)
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "se": float("nan"), "n": 0}
    mean = float(v.mean())
    if n == 1:
        return {"mean": mean, "lo": float("-inf"), "hi": float("inf"), "se": float("inf"), "n": 1}
    se = float(v.std(ddof=1) / math.sqrt(n))
    half = t_quantile(0.5 + level / 2.0, n - 1) * se
    return {"mean": mean, "lo": mean - half, "hi": mean + half, "se": se, "n": n}


# ----------------------------------------------------------------------------------------------------------------
# slope / intercept
# ----------------------------------------------------------------------------------------------------------------
def _hac_meat(sc: np.ndarray) -> Tuple[np.ndarray, int]:
    """Newey-West meat sum_l w_l (G_l + G_l^T) of score contributions ``sc`` [n, k] (Bartlett kernel), lag
    truncation from Andrews' (1991) AR(1) plug-in bandwidth 1.1447 (alpha(1) n)^(1/3)."""
    n = sc.shape[0]
    num = den = 0.0
    for j in range(sc.shape[1]):
        u = sc[:, j]
        d = float(np.dot(u[:-1], u[:-1]))
        rho = float(np.dot(u[1:], u[:-1]) / d) if d > 0 else 0.0
        rho = min(max(rho, -0.97), 0.97)
        s2 = float(np.mean((u[1:] - rho * u[:-1]) ** 2))
        num += 4.0 * rho * rho * s2 * s2 / ((1.0 - rho) ** 6 * (1.0 + rho) ** 2)
        den += s2 * s2 / (1.0 - rho) ** 4
    alpha = num / den if den > 0 else 0.0
    lag = int(min(n - 1, math.ceil(1.1447 * (alpha * n) ** (1.0 / 3.0)))) if alpha > 0 else 0
    meat = sc.T @ sc
    for l in range(1, lag + 1):
        g = sc[l:].T @ sc[:-l]
        meat = meat + (1.0 - l / (lag + 1.0)) * (g + g.T)
    return meat, lag


def logistic_recalibration(p: Any, y: Any, level: float = 0.95, ridge: float = 1e-8, iters: int = 100, *,
                           clusters: Optional[Sequence[Hashable]] = None, cov: str = "auto") -> Dict[str, float]:
    """Fit ``y ~ sigmoid(intercept + slope * logit(p))`` by Newton's method and return the estimates with t-based
    intervals. ``cov``: ``"auto"`` = cluster-robust (CR1 sandwich, G - 1 degrees of freedom) when ``clusters`` are
    given, else HAC in row order (Bartlett / Andrews bandwidth, n - 2 degrees of freedom); ``"cluster"``,
    ``"hac"``, or ``"iid"`` (model-based Wald, n - 2 df: valid only for independent rows). ``identifiable`` is
    False when the data cannot determine the fit (one class only, constant forecasts or perfect separation)."""
    x, y = logit(p), _np(y)
    n = len(x)
    out = {"slope": float("nan"), "intercept": float("nan"), "slope_lo": float("nan"), "slope_hi": float("nan"),
           "intercept_lo": float("nan"), "intercept_hi": float("nan"), "n": n, "identifiable": False}
    if clusters is not None and len(clusters) != n:
        raise ValueError("clusters must have one label per forecast")
    method = cov if cov != "auto" else ("cluster" if clusters is not None else "hac")
    if method not in ("cluster", "hac", "iid"):
        raise ValueError("cov must be 'auto', 'cluster', 'hac' or 'iid'")
    if method == "cluster" and clusters is None:
        raise ValueError("cov='cluster' needs clusters")
    out["cov"] = method
    if n < 3 or y.min() == y.max() or np.ptp(x) < 1e-9:
        return out
    design = np.stack([np.ones(n), x], 1)
    beta = np.zeros(2)
    converged = False
    for _ in range(iters):
        eta = np.clip(design @ beta, -40, 40)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-12, None)
        hess = design.T @ (design * w[:, None]) + ridge * np.eye(2)
        step = np.linalg.solve(hess, design.T @ (y - mu) - ridge * beta)
        beta = beta + step
        if np.abs(step).max() < 1e-10:
            converged = True
            break
    if not converged or np.abs(beta).max() > 1e3:
        return out
    eta = np.clip(design @ beta, -40, 40)
    mu = 1.0 / (1.0 + np.exp(-eta))
    bread = np.linalg.inv(design.T @ (design * (mu * (1 - mu))[:, None]) + ridge * np.eye(2))
    if method == "iid":
        covm, df = bread, n - 2
    else:
        sc = design * (y - mu)[:, None]                                  # per-row score contributions
        if method == "cluster":
            labels: Dict[Hashable, int] = {}
            idx = np.asarray([labels.setdefault(c, len(labels)) for c in clusters])
            g = len(labels)
            sums = np.zeros((g, 2))
            np.add.at(sums, idx, sc)
            if g < 2:
                covm, df = np.full((2, 2), np.inf), 1
            else:
                covm = bread @ (sums.T @ sums) @ bread * (g / (g - 1.0)) * ((n - 1.0) / (n - 2.0))
                df = g - 1
            out["n_clusters"] = g
        else:
            meat, lag = _hac_meat(sc)
            covm, df = bread @ meat @ bread * (n / (n - 2.0)), n - 2
            out["hac_lags"] = lag
    se = np.sqrt(np.clip(np.diag(covm), 0.0, None))
    tq = t_quantile(0.5 + level / 2.0, df)
    out.update(intercept=float(beta[0]), slope=float(beta[1]), intercept_se=float(se[0]), slope_se=float(se[1]),
               intercept_lo=float(beta[0] - tq * se[0]), intercept_hi=float(beta[0] + tq * se[0]),
               slope_lo=float(beta[1] - tq * se[1]), slope_hi=float(beta[1] + tq * se[1]), identifiable=True,
               df=float(df))
    return out


# ----------------------------------------------------------------------------------------------------------------
# reliability and ECE
# ----------------------------------------------------------------------------------------------------------------
def reliability_bins(p: Any, y: Any, n_bins: int = 10, strategy: str = "uniform") -> List[Dict[str, float]]:
    """Reliability curve. ``uniform`` uses equal-width probability bins, ``quantile`` equal-count bins. Empty bins
    are returned with n = 0 so that curves from different runs line up."""
    p, y = _np(p), _np(y)
    if strategy == "quantile" and len(p):
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        edges[0], edges[-1] = 0.0, 1.0
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        sel = idx == b
        k = int(sel.sum())
        out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": k,
                    "mean_p": float(p[sel].mean()) if k else float("nan"),
                    "frac_pos": float(y[sel].mean()) if k else float("nan")})
    return out


def _ece_and_se(p: np.ndarray, y: np.ndarray, n_bins: int) -> Tuple[float, float]:
    n = max(len(p), 1)
    ece, var = 0.0, 0.0
    for b in reliability_bins(p, y, n_bins):
        if b["n"]:
            ece += b["n"] / n * abs(b["mean_p"] - b["frac_pos"])
            f = min(max(b["frac_pos"], 0.5 / b["n"]), 1 - 0.5 / b["n"])           # continuity-corrected
            var += (b["n"] / n) ** 2 * f * (1 - f) / b["n"]
    return ece, math.sqrt(var)


def ece(p: Any, y: Any, n_bins: int = 10) -> float:
    """Expected calibration error: sum_b n_b / n * |mean forecast - observed frequency|."""
    return _ece_and_se(_np(p), _np(y), n_bins)[0]


def ece_ci(p: Any, y: Any, n_bins: int = 10, level: float = 0.95, n_boot: int = 400, seed: int = 0,
           clusters: Optional[Sequence[Hashable]] = None) -> Dict[str, float]:
    """ECE with a studentised (bootstrap-t) interval; the standard error inside each replicate is the binomial
    plug-in of the binned frequencies. With ``clusters`` the bootstrap resamples whole clusters. The lower end is
    clipped at 0."""
    p, y = _np(p), _np(y)
    est, se = _ece_and_se(p, y, n_bins)
    n = len(p)
    if n < 2 or se <= 0:
        return {"ece": est, "lo": float("nan"), "hi": float("nan"), "se": se, "n": n}
    rng = np.random.default_rng(seed)
    groups: Optional[List[np.ndarray]] = None
    if clusters is not None:
        by: Dict[Hashable, List[int]] = {}
        for k, c in enumerate(clusters):
            by.setdefault(c, []).append(k)
        groups = [np.asarray(v) for v in by.values()]
    ts = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n) if groups is None else \
            np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        e_b, se_b = _ece_and_se(p[i], y[i], n_bins)
        if se_b > 0:
            ts.append((e_b - est) / se_b)
    lo_q, hi_q = np.quantile(ts, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"ece": est, "lo": max(0.0, est - hi_q * se), "hi": est - lo_q * se, "se": se, "n": n}


# ----------------------------------------------------------------------------------------------------------------
# reports
# ----------------------------------------------------------------------------------------------------------------
def binary_report(p: Any, y: Any, n_bins: int = 10, level: float = 0.95,
                  clusters: Optional[Sequence[Hashable]] = None, seed: int = 0) -> Dict[str, Any]:
    p, y = _np(p), _np(y)
    return {"n": int(len(p)), "base_rate": float(y.mean()) if len(y) else float("nan"),
            "mean_p": float(p.mean()) if len(p) else float("nan"),
            "brier": studentised_ci(brier(p, y), level, clusters),
            "nll": studentised_ci(binary_nll(p, y), level, clusters),
            "calibration": logistic_recalibration(p, y, level, clusters=clusters),
            "ece": ece_ci(p, y, n_bins, level, seed=seed, clusters=clusters),
            "reliability": reliability_bins(p, y, n_bins)}


def horizon_bucket(horizon_s: float, edges: Sequence[float] = DEFAULT_HORIZON_EDGES) -> str:
    names = ["<=1h", "<=1d", "<=1w", "<=30d", ">30d"] if tuple(edges) == DEFAULT_HORIZON_EDGES else None
    i = int(np.searchsorted(np.asarray(edges, dtype=np.float64), horizon_s, side="left"))
    if names is not None:
        return names[i]
    return f"<={edges[i]:g}s" if i < len(edges) else f">{edges[-1]:g}s"


def grouped_binary_report(p: Any, y: Any, families: Sequence[str], horizons_s: Sequence[float], *,
                          horizon_edges: Sequence[float] = DEFAULT_HORIZON_EDGES, n_bins: int = 10,
                          level: float = 0.95, clusters: Optional[Sequence[Hashable]] = None) -> Dict[str, Any]:
    """Calibration overall, by objective family, by horizon bucket and by (family, horizon)."""
    p, y = _np(p), _np(y)
    hb = [horizon_bucket(h, horizon_edges) for h in horizons_s]

    def sub(sel: np.ndarray) -> Dict[str, Any]:
        cl = None if clusters is None else [c for c, s in zip(clusters, sel) if s]
        return binary_report(p[sel], y[sel], n_bins, level, cl)

    out: Dict[str, Any] = {"overall": binary_report(p, y, n_bins, level, clusters), "by_family": {},
                           "by_horizon": {}, "by_family_horizon": {}}
    fam = np.asarray(list(families), dtype=object)
    hba = np.asarray(hb, dtype=object)
    for f in sorted(set(families)):
        out["by_family"][f] = sub(fam == f)
    for h in sorted(set(hb)):
        out["by_horizon"][h] = sub(hba == h)
    for f in sorted(set(families)):
        for h in sorted(set(hb)):
            sel = (fam == f) & (hba == h)
            if sel.any():
                out["by_family_horizon"][f"{f}|{h}"] = sub(sel)
    return out


# ----------------------------------------------------------------------------------------------------------------
# prospective scoring of stored predictions
# ----------------------------------------------------------------------------------------------------------------
def prospective_guard(made_at: float, outcome_at: Optional[float], registered_at: Optional[float] = None) -> None:
    """Raise unless the prediction existed strictly before its outcome was observed."""
    if outcome_at is None:
        raise RetrospectivePredictionError("the prediction has no outcome time")
    if not (made_at < outcome_at):
        raise RetrospectivePredictionError(
            f"prediction made at {made_at} is not strictly before its outcome at {outcome_at}")
    if registered_at is not None and not (registered_at < outcome_at):
        raise RetrospectivePredictionError(
            f"prediction registered at {registered_at} is not strictly before its outcome at {outcome_at}")


def score_predictions(records: Sequence[PredictionRecord], *,
                      family_of: Optional[Callable[[PredictionRecord], str]] = None,
                      registered_at: Optional[Callable[[PredictionRecord], Optional[float]]] = None,
                      horizon_edges: Sequence[float] = DEFAULT_HORIZON_EDGES, strict: bool = True,
                      n_bins: int = 10, level: float = 0.95) -> Dict[str, Any]:
    """Score resolved, prospectively registered predictions.

    Unresolved / pending / unobservable records are skipped (counted in ``skipped``). A record that fails
    :func:`prospective_guard` raises when ``strict`` (default) and is otherwise dropped and counted in ``refused``;
    it is never scored. Binary predictions get the full grouped calibration report; every family gets its
    negative log-likelihood, and families with a CDF get the coverage of their central 50 % and 90 % intervals.
    Rows are clustered by experiment, else decision, else prediction (``query.experiment_id or decision_id or
    prediction_id``): the NLL intervals are over cluster means and the binary report is cluster-robust.
    """
    fam_fn = family_of or (lambda r: "")
    p_bin, y_bin, fam_bin, hor_bin, cl_bin = [], [], [], [], []
    nll: Dict[str, List[float]] = {}
    nll_cl: Dict[str, List[str]] = {}
    cover: Dict[str, List[Tuple[bool, bool]]] = {}
    skipped = refused = 0
    for r in records:
        if r.resolved_status not in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED) or r.resolved_value is None:
            skipped += 1
            continue
        try:
            prospective_guard(r.made_at, r.resolved_at, registered_at(r) if registered_at else None)
        except RetrospectivePredictionError:
            if strict:
                raise
            refused += 1
            continue
        fam = fam_fn(r)
        key = f"{r.dist.family}|{fam}|{horizon_bucket(r.query.horizon_s, horizon_edges)}"
        lp = D.record_log_prob(r.dist, r.resolved_value, r.resolved_status)
        nll.setdefault(key, []).append(-lp)
        nll_cl.setdefault(key, []).append(r.query.experiment_id or r.decision_id or r.prediction_id)
        dist = D.from_record(r.dist)
        if r.dist.family == D.FAMILY_BERNOULLI:
            p_bin.append(float(dist.mean))
            y_bin.append(float(r.resolved_value))
            fam_bin.append(fam)
            hor_bin.append(r.query.horizon_s)
            cl_bin.append(r.query.experiment_id or r.decision_id or r.prediction_id)
        elif r.dist.family in (D.FAMILY_NORMAL, D.FAMILY_STUDENT_T, D.FAMILY_POISSON, D.FAMILY_NEG_BINOMIAL) \
                and r.resolved_status == OutcomeStatus.OBSERVED:
            y = float(r.resolved_value)
            ins = []
            for lv in (0.5, 0.9):
                lo, hi = dist.interval(lv)
                ins.append(bool(float(lo) <= y <= float(hi)))
            cover.setdefault(key, []).append((ins[0], ins[1]))
    out: Dict[str, Any] = {"n_scored": sum(len(v) for v in nll.values()), "skipped": skipped, "refused": refused,
                           "nll": {k: studentised_ci(v, level, nll_cl[k]) for k, v in sorted(nll.items())},
                           "coverage": {k: {"n": len(v), "cover50": float(np.mean([a for a, _ in v])),
                                            "cover90": float(np.mean([b for _, b in v]))}
                                        for k, v in sorted(cover.items())}}
    if p_bin:
        out["binary"] = grouped_binary_report(p_bin, y_bin, fam_bin, hor_bin, horizon_edges=horizon_edges,
                                              n_bins=n_bins, level=level, clusters=cl_bin)
    return out
