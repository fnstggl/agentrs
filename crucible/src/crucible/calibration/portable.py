"""Portable NumPy predictors for fitted calibration models (workstream E2).

World code loads fitted models through the registry and must not depend on scikit-learn, pickle or the datasets.
Every fitted estimator is therefore exported to plain arrays (saved in an `.npz` without pickle) and evaluated here
with NumPy only. Exporters (`export_*`) run at fit time and take the scikit-learn estimator; predictors (`Portable*`)
take the arrays. `check_equal` is used at fit time to prove that the export reproduces the estimator.

Kinds:
    constant      p = value (link-free); used by baselines
    linear        z = ((x - mean) / scale) . coef + intercept, then the link
    trees         z = baseline + sum of leaf values of a histogram gradient-boosting ensemble, then the link
    proportional  mu = a * expm1(x[:, j]) + b   (rule baseline on a log1p-count feature)
Links: 'logit' (probability), 'log' (Poisson mean), 'identity'.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def apply_link(z: np.ndarray, link: str) -> np.ndarray:
    if link == "logit":
        return _sigmoid(z)
    if link == "log":
        return np.exp(np.clip(z, -50, 50))
    if link == "identity":
        return z
    raise ValueError(f"unknown link {link!r}")


# ----------------------------------------------------------------------------------------------- exporters

def export_constant(value: float, link: str = "identity") -> dict[str, np.ndarray]:
    return {"kind": np.array("constant"), "link": np.array(link), "value": np.array([float(value)])}


def export_linear(coef, intercept, mean, scale, link: str) -> dict[str, np.ndarray]:
    return {"kind": np.array("linear"), "link": np.array(link),
            "coef": np.asarray(coef, dtype=np.float64).ravel(), "intercept": np.array([float(np.ravel(intercept)[0])]),
            "mean": np.asarray(mean, dtype=np.float64).ravel(), "scale": np.asarray(scale, dtype=np.float64).ravel()}


def export_proportional(a: float, b: float, feature_index: int) -> dict[str, np.ndarray]:
    return {"kind": np.array("proportional"), "link": np.array("identity"),
            "ab": np.array([float(a), float(b)]), "feature_index": np.array([int(feature_index)])}


def export_hgb(est, link: str) -> dict[str, np.ndarray]:
    """Flatten a fitted HistGradientBoosting{Classifier,Regressor} (binary or single-output) into arrays."""
    if getattr(est, "is_categorical_", None) is not None and np.any(est.is_categorical_):
        raise ValueError("categorical features are not supported by the portable exporter")
    feats, thr, left, right, leaf, value, miss, roots, depth = [], [], [], [], [], [], [], [], []
    offset = 0
    for it in est._predictors:
        if len(it) != 1:
            raise ValueError("only binary classification / single-output regression is supported")
        nodes = it[0].nodes
        roots.append(offset)
        feats.append(nodes["feature_idx"].astype(np.int32))
        thr.append(nodes["num_threshold"].astype(np.float64))
        left.append(nodes["left"].astype(np.int64) + offset)
        right.append(nodes["right"].astype(np.int64) + offset)
        leaf.append(nodes["is_leaf"].astype(bool))
        value.append(nodes["value"].astype(np.float64))
        miss.append(nodes["missing_go_to_left"].astype(bool))
        depth.append(int(nodes["depth"].max()) if len(nodes) else 0)
        offset += len(nodes)
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)
    return {"kind": np.array("trees"), "link": np.array(link),
            "baseline": np.array([float(np.ravel(est._baseline_prediction)[0])]),
            "feature": cat(feats, np.int32), "threshold": cat(thr, np.float64),
            "left": cat(left, np.int32), "right": cat(right, np.int32), "is_leaf": cat(leaf, bool),
            "value": cat(value, np.float64), "missing_left": cat(miss, bool),
            "roots": np.asarray(roots, dtype=np.int32), "max_depth": np.array([max(depth) if depth else 0])}


# ----------------------------------------------------------------------------------------------- predictors

def raw_score(arrays: dict[str, np.ndarray], X: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """Score before the link (log-odds, log-mean or identity)."""
    kind = str(arrays["kind"])
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-D [n, features]")
    n = X.shape[0]
    if kind == "constant":
        return np.full(n, float(arrays["value"][0]))
    if kind == "linear":
        Z = (X - arrays["mean"]) / arrays["scale"]
        return Z @ arrays["coef"] + float(arrays["intercept"][0])
    if kind == "proportional":
        a, b = arrays["ab"]
        j = int(arrays["feature_index"][0])
        return a * np.expm1(X[:, j]) + b
    if kind == "trees":
        roots = arrays["roots"].astype(np.int64)
        feature = arrays["feature"]; threshold = arrays["threshold"]; is_leaf = arrays["is_leaf"]
        left = arrays["left"]; right = arrays["right"]; value = arrays["value"]; miss = arrays["missing_left"]
        out = np.full(n, float(arrays["baseline"][0]))
        if roots.size == 0:
            return out
        steps = int(arrays["max_depth"][0]) + 1
        for s in range(0, n, chunk):
            Xc = X[s:s + chunk]
            m = Xc.shape[0]
            idx = np.broadcast_to(roots, (m, roots.size)).copy()
            rows = np.arange(m)[:, None]
            for _ in range(steps):
                leaf_now = is_leaf[idx]
                if leaf_now.all():
                    break
                x = Xc[rows, feature[idx]]
                go_left = (x <= threshold[idx]) | (np.isnan(x) & miss[idx])
                nxt = np.where(go_left, left[idx], right[idx])
                idx = np.where(leaf_now, idx, nxt)
            out[s:s + chunk] += value[idx].sum(axis=1)
        return out
    raise ValueError(f"unknown predictor kind {kind!r}")


def predict(arrays: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    return apply_link(raw_score(arrays, X), str(arrays["link"]))


def check_equal(arrays: dict[str, np.ndarray], reference: np.ndarray, X: np.ndarray) -> float:
    """Max absolute difference between the portable prediction and the estimator's own prediction."""
    return float(np.max(np.abs(predict(arrays, X) - np.asarray(reference, dtype=np.float64)))) if len(X) else 0.0


def pack(prefix: str, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {f"{prefix}/{k}": v for k, v in arrays.items()}


def unpack(prefix: str, npz) -> dict[str, np.ndarray]:
    p = prefix + "/"
    out = {k[len(p):]: npz[k] for k in npz.keys() if k.startswith(p)}
    if not out:
        raise KeyError(f"no arrays with prefix {prefix!r}")
    return out
