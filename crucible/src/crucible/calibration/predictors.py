"""Portable predictors for fitted calibration models.

Every checkpoint is a NumPy `.npz` (loaded with `allow_pickle=False`) plus a JSON model card, so world code can load a
model without scikit-learn, torch or pickle. Each predictor kind has a builder `from_params(params, card)` and a
`predict(X)`; the fitting code exports to the same parameter layout and the tests check that the exported predictor
reproduces the fitted object's predictions.

Kinds:

* ``constant``            probability ``p`` for every row.
* ``logistic``            sigmoid(((X - mean) / scale) @ w + b).
* ``gbdt_binary``         gradient-boosted regression trees on the logit scale (exported from scikit-learn's
                          HistGradientBoostingClassifier); numeric features, NaN routed by the learned missing branch.
* ``quantile_net``        a monotone quantile network: X -> hidden tanh layers -> base + cumulative softplus increments,
                          giving Q non-crossing quantiles of the target per row (no hidden layer = linear quantiles).
* ``binned_quantiles``    quantiles of the target per bin of one feature (baseline for distributional models).
* ``gamma_poisson``       heterogeneous Poisson rates: individual rate ~ Gamma(shape, rate); `predict` returns the
                          posterior mean rate given (count, exposure) columns.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _softplus(z: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, z)


class Constant:
    kind = "constant"

    def __init__(self, p: float):
        self.p = float(p)

    def params(self) -> dict:
        return {"p": np.array([self.p])}

    @classmethod
    def from_params(cls, P, card=None):
        return cls(float(P["p"][0]))

    def predict(self, X):
        return np.full(len(X), self.p)


class Logistic:
    kind = "logistic"

    def __init__(self, w, b, mean, scale):
        self.w = np.asarray(w, float)
        self.b = float(b)
        self.mean = np.asarray(mean, float)
        self.scale = np.asarray(scale, float)

    def params(self) -> dict:
        return {"w": self.w, "b": np.array([self.b]), "mean": self.mean, "scale": self.scale}

    @classmethod
    def from_params(cls, P, card=None):
        return cls(P["w"], float(P["b"][0]), P["mean"], P["scale"])

    def logit(self, X):
        Z = (np.asarray(X, float) - self.mean) / self.scale
        return Z @ self.w + self.b

    def predict(self, X):
        return _sigmoid(self.logit(X))


class TreeEnsemble:
    """Sum of regression trees on the logit scale. Arrays are concatenated over trees; `roots[t]` is tree t's root."""
    kind = "gbdt_binary"

    def __init__(self, baseline, roots, feature, threshold, left, right, is_leaf, value, missing_left,
                 offset=0.0):
        self.baseline = float(baseline)
        self.roots = np.asarray(roots, np.int64)
        self.feature = np.asarray(feature, np.int64)
        self.threshold = np.asarray(threshold, float)
        self.left = np.asarray(left, np.int64)
        self.right = np.asarray(right, np.int64)
        self.is_leaf = np.asarray(is_leaf, bool)
        self.value = np.asarray(value, float)
        self.missing_left = np.asarray(missing_left, bool)
        self.offset = float(offset)          # optional logit shift (intercept recalibration)

    @classmethod
    def from_sklearn(cls, model, offset: float = 0.0) -> "TreeEnsemble":
        """Export a fitted HistGradientBoostingClassifier (binary, numeric features only)."""
        roots, feat, thr, left, right, leaf, val, miss = [], [], [], [], [], [], [], []
        base = 0
        for it in model._predictors:
            if len(it) != 1:
                raise ValueError("only binary classifiers are supported")
            nodes = it[0].nodes
            if nodes["is_categorical"].any():
                raise ValueError("categorical splits are not supported")
            roots.append(base)
            feat.append(nodes["feature_idx"].astype(np.int64))
            thr.append(nodes["num_threshold"].astype(float))
            left.append(nodes["left"].astype(np.int64) + base)
            right.append(nodes["right"].astype(np.int64) + base)
            leaf.append(nodes["is_leaf"].astype(bool))
            val.append(nodes["value"].astype(float))
            miss.append(nodes["missing_go_to_left"].astype(bool))
            base += len(nodes)
        bl = np.asarray(model._baseline_prediction).ravel()[0]
        return cls(bl, roots, np.concatenate(feat), np.concatenate(thr), np.concatenate(left), np.concatenate(right),
                   np.concatenate(leaf), np.concatenate(val), np.concatenate(miss), offset)

    def params(self) -> dict:
        return {"baseline": np.array([self.baseline]), "roots": self.roots, "feature": self.feature,
                "threshold": self.threshold, "left": self.left, "right": self.right,
                "is_leaf": self.is_leaf, "value": self.value, "missing_left": self.missing_left,
                "offset": np.array([self.offset])}

    @classmethod
    def from_params(cls, P, card=None):
        return cls(float(P["baseline"][0]), P["roots"], P["feature"], P["threshold"], P["left"], P["right"],
                   P["is_leaf"], P["value"], P["missing_left"], float(P["offset"][0]) if "offset" in P else 0.0)

    def logit(self, X):
        X = np.asarray(X, float)
        n = len(X)
        rows = np.arange(n)
        out = np.full(n, self.baseline + self.offset)
        for r in self.roots:
            node = np.full(n, r, np.int64)
            active = ~self.is_leaf[node]
            while active.any():
                nd = node[active]
                x = X[rows[active], self.feature[nd]]
                nan = np.isnan(x)
                go_left = np.where(nan, self.missing_left[nd], x <= self.threshold[nd])
                node[active] = np.where(go_left, self.left[nd], self.right[nd])
                active = ~self.is_leaf[node]
            out += self.value[node]
        return out

    def predict(self, X):
        return _sigmoid(self.logit(X))


class QuantileNet:
    """Monotone quantile regression network evaluated in NumPy (weights exported from torch)."""
    kind = "quantile_net"

    def __init__(self, mean, scale, weights: list[tuple[np.ndarray, np.ndarray]], head_w, head_b, levels,
                 y_min=None, y_max=None):
        self.mean = np.asarray(mean, float)
        self.scale = np.asarray(scale, float)
        self.weights = [(np.asarray(W, float), np.asarray(b, float)) for W, b in weights]
        self.head_w = np.asarray(head_w, float)          # [hidden, Q]
        self.head_b = np.asarray(head_b, float)          # [Q]
        self.levels = np.asarray(levels, float)
        self.y_min = None if y_min is None else float(y_min)
        self.y_max = None if y_max is None else float(y_max)

    def params(self) -> dict:
        P = {"mean": self.mean, "scale": self.scale, "head_w": self.head_w, "head_b": self.head_b,
             "levels": self.levels, "n_layers": np.array([len(self.weights)]),
             "y_bounds": np.array([np.nan if self.y_min is None else self.y_min,
                                   np.nan if self.y_max is None else self.y_max])}
        for i, (W, b) in enumerate(self.weights):
            P[f"W{i}"] = W
            P[f"b{i}"] = b
        return P

    @classmethod
    def from_params(cls, P, card=None):
        n = int(P["n_layers"][0])
        lo, hi = (float(v) for v in P["y_bounds"])
        return cls(P["mean"], P["scale"], [(P[f"W{i}"], P[f"b{i}"]) for i in range(n)], P["head_w"], P["head_b"],
                   P["levels"], None if np.isnan(lo) else lo, None if np.isnan(hi) else hi)

    def predict(self, X):
        """[n, Q] non-decreasing quantiles of the target."""
        H = (np.asarray(X, float) - self.mean) / self.scale
        for W, b in self.weights:
            H = np.tanh(H @ W + b)
        raw = H @ self.head_w + self.head_b
        q = np.concatenate([raw[:, :1], raw[:, :1] + np.cumsum(_softplus(raw[:, 1:]), axis=1)], axis=1)
        if self.y_min is not None or self.y_max is not None:
            q = np.clip(q, self.y_min if self.y_min is not None else -np.inf,
                        self.y_max if self.y_max is not None else np.inf)
        return q

    def sample(self, X, u):
        return interpolate_quantiles(self.predict(X), self.levels, u)


class BinnedQuantiles:
    """Quantiles of the target per bin of feature `col`, optionally crossed with an integer group feature
    `group_col` (e.g. consumed / skipped). `edges` are interior bin edges; `table` is [G, B, Q]."""
    kind = "binned_quantiles"

    def __init__(self, col: int, edges, table, levels, group_col: int = -1):
        self.col = int(col)
        self.group_col = int(group_col)
        self.edges = np.asarray(edges, float)
        t = np.asarray(table, float)
        self.table = t if t.ndim == 3 else t[None]
        self.levels = np.asarray(levels, float)

    def params(self) -> dict:
        return {"col": np.array([self.col]), "group_col": np.array([self.group_col]), "edges": self.edges,
                "table": self.table, "levels": self.levels}

    @classmethod
    def from_params(cls, P, card=None):
        return cls(int(P["col"][0]), P["edges"], P["table"], P["levels"],
                   int(P["group_col"][0]) if "group_col" in P else -1)

    def predict(self, X):
        X = np.asarray(X, float)
        b = np.searchsorted(self.edges, X[:, self.col], side="left") if len(self.edges) else np.zeros(len(X), int)
        g = np.zeros(len(X), int) if self.group_col < 0 else \
            np.clip(X[:, self.group_col].astype(int), 0, self.table.shape[0] - 1)
        return self.table[g, b]

    def sample(self, X, u):
        return interpolate_quantiles(self.predict(X), self.levels, u)


class GammaPoisson:
    """Individual event rates lambda_i ~ Gamma(shape, rate); counts | lambda_i ~ Poisson(lambda_i * exposure).

    X columns: [count, exposure]. `predict` gives the posterior mean rate (shape + count) / (rate + exposure);
    `draw_rates(n, rng)` draws rates for new individuals (what a simulated population uses)."""
    kind = "gamma_poisson"

    def __init__(self, shape: float, rate: float):
        self.shape = float(shape)
        self.rate = float(rate)

    def params(self) -> dict:
        return {"shape": np.array([self.shape]), "rate": np.array([self.rate])}

    @classmethod
    def from_params(cls, P, card=None):
        return cls(float(P["shape"][0]), float(P["rate"][0]))

    def predict(self, X):
        X = np.asarray(X, float)
        return (self.shape + X[:, 0]) / (self.rate + X[:, 1])

    def draw_rates(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.gamma(self.shape, 1.0 / self.rate, size=n)


def interpolate_quantiles(Q: np.ndarray, levels: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse-CDF sampling from per-row quantiles: linear between levels; below the first and above the last level
    the CDF is linear over one outer-gap width (the same convention as `metrics.pit_fast`)."""
    Q = np.asarray(Q, float)
    lv = np.asarray(levels, float)
    u = np.asarray(u, float)
    uc = np.clip(u, lv[0], lv[-1])
    j = np.clip(np.searchsorted(lv, uc, side="right") - 1, 0, len(lv) - 2)
    t = (uc - lv[j]) / (lv[j + 1] - lv[j])
    rows = np.arange(len(Q))
    y = Q[rows, j] + t * (Q[rows, j + 1] - Q[rows, j])
    lo = u < lv[0]
    hi = u > lv[-1]
    y[lo] = Q[lo, 0] - (lv[0] - u[lo]) / lv[0] * (Q[lo, 1] - Q[lo, 0])
    y[hi] = Q[hi, -1] + (u[hi] - lv[-1]) / (1 - lv[-1]) * (Q[hi, -1] - Q[hi, -2])
    return y


KINDS = {c.kind: c for c in (Constant, Logistic, TreeEnsemble, QuantileNet, BinnedQuantiles, GammaPoisson)}


def register_kind(kind: str, cls) -> None:
    """Other calibration modules add predictor kinds here (`cls.from_params(params, card)`, `.predict(X)`)."""
    if kind in KINDS and KINDS[kind] is not cls:
        raise ValueError(f"predictor kind {kind!r} already registered")
    KINDS[kind] = cls
