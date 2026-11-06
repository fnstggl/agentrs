"""Evaluation metrics with bootstrap confidence intervals."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional, Sequence

import numpy as np

from agentrs.data.schemas import EpisodeRecord


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def episode_metrics(eps: Sequence[EpisodeRecord]) -> dict:
    if not eps:
        return {"n": 0}
    succ = [float(bool(e.success)) for e in eps]
    steps = [e.num_steps for e in eps]
    inv = sum(e.num_invalid for e in eps)
    tot_steps = sum(steps)
    with_err = [e for e in eps if e.num_invalid > 0]
    lo, hi = bootstrap_ci(succ)
    out = {
        "n": len(eps),
        "success_rate": float(np.mean(succ)),
        "success_ci95": [lo, hi],
        "mean_steps": float(np.mean(steps)),
        "invalid_action_rate": inv / max(1, tot_steps),
        "real_steps_per_success": tot_steps / max(1, sum(succ)),
        "error_recovery_rate": (sum(bool(e.success) for e in with_err) / len(with_err)) if with_err else None,
        "mean_wall_time_s": float(np.mean([e.wall_time for e in eps])),
    }
    long = [e for e in eps if e.metadata.get("difficulty", 0) >= 4]
    if long:
        out["long_horizon_success"] = float(np.mean([bool(e.success) for e in long]))
    by_fam: dict[str, list[float]] = defaultdict(list)
    for e in eps:
        by_fam[e.family].append(float(bool(e.success)))
    out["by_family"] = {k: float(np.mean(v)) for k, v in sorted(by_fam.items())}
    return out


def expected_calibration_error(probs: Iterable[float], labels: Iterable[float], n_bins: int = 10) -> float:
    p = np.asarray(list(probs), dtype=float)
    y = np.asarray(list(labels), dtype=float)
    if len(p) == 0:
        return float("nan")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def reliability_table(probs, labels, n_bins: int = 10) -> list[dict]:
    p = np.asarray(list(probs), dtype=float)
    y = np.asarray(list(labels), dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            rows.append({"bin": b, "n": int(m.sum()), "mean_pred": float(p[m].mean()), "frac_pos": float(y[m].mean())})
    return rows


def auroc(probs, labels) -> Optional[float]:
    p = np.asarray(list(probs), dtype=float)
    y = np.asarray(list(labels), dtype=float)
    pos, neg = p[y > 0.5], p[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # Mann-Whitney U
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    # average ranks for ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def bernoulli_nll(probs, labels, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(list(probs), dtype=float), eps, 1 - eps)
    y = np.asarray(list(labels), dtype=float)
    if len(p) == 0:
        return float("nan")
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(probs, labels) -> float:
    p = np.asarray(list(probs), dtype=float)
    y = np.asarray(list(labels), dtype=float)
    return float(((p - y) ** 2).mean()) if len(p) else float("nan")

