"""Two-level bootstrap over training seeds and evaluation worlds (experiment register rule; spec Part 19).

Values are a matrix ``M[s, w]``: training seed s (rows; a learner without training seeds has one row) by evaluation
world w (columns; the same worlds for every seed, so resampling worlds keeps the pairing). One bootstrap draw
resamples seeds with replacement, then worlds with replacement, and averages the selected cells (NaN cells, e.g. a
world a learner could not run, are skipped). Paired differences between two learners are formed per (seed, world)
before resampling when both have the same seeds, or per world against the other learner's seed-mean otherwise.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def two_level(M, n_boot: int = 4000, level: float = 0.95, seed: int = 0) -> dict:
    A = np.atleast_2d(np.asarray(M, dtype=np.float64))
    S, W = A.shape
    ok = np.isfinite(A)
    if not ok.any():
        return {"mean": None, "lo": None, "hi": None, "n_seeds": S, "n_worlds": W, "n_cells": 0}
    rng = np.random.default_rng(seed)
    point = float(np.nanmean(A))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        si = rng.integers(0, S, S)
        wi = rng.integers(0, W, W)
        sub = A[np.ix_(si, wi)]
        boots[i] = np.nanmean(sub) if np.isfinite(sub).any() else np.nan
    boots = boots[np.isfinite(boots)]
    a = (1 - level) / 2
    return {"mean": point, "lo": float(np.quantile(boots, a)), "hi": float(np.quantile(boots, 1 - a)),
            "n_seeds": S, "n_worlds": W, "n_cells": int(ok.sum())}


def paired(A, B, n_boot: int = 4000, level: float = 0.95, seed: int = 0) -> dict:
    """Mean of A - B with a two-level interval. A, B: [S_a, W], [S_b, W] on the same worlds."""
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    if A.shape[1] != B.shape[1]:
        raise ValueError("paired comparison needs the same evaluation worlds")
    if A.shape[0] == B.shape[0]:
        D = A - B
    else:
        D = A - np.nanmean(B, 0, keepdims=True)
    out = two_level(D, n_boot, level, seed)
    out["p_gt0"] = float(np.mean(_boot_means(D, n_boot, seed) > 0)) if out["mean"] is not None else None
    return out


def _boot_means(D: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    S, W = D.shape
    out = np.empty(n_boot)
    for i in range(n_boot):
        sub = D[np.ix_(rng.integers(0, S, S), rng.integers(0, W, W))]
        out[i] = np.nanmean(sub) if np.isfinite(sub).any() else np.nan
    return out[np.isfinite(out)]


def curve(Ms: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """{x: M[s, w]} -> {x: two_level(M)} (e.g. a metric at every support size k)."""
    return {str(x): two_level(M, n_boot, seed=seed) for x, M in Ms.items()}


def area(Ms: dict, xs: Optional[Sequence] = None) -> np.ndarray:
    """Learning-curve area per (seed, world): the mean of the metric over the grid points `xs` (default: all keys of
    `Ms`, e.g. the doubling grid k = 0, 1, 2, ..., 64, so every doubling of experience weighs the same)."""
    keys = list(xs) if xs is not None else list(Ms)
    return np.nanmean(np.stack([np.asarray(Ms[k], dtype=np.float64) for k in keys], 0), 0)
