"""Training data and fitting for the learned recommenders (workstream C2).

Learned recommenders are fitted only on what the platform recorded: impression and consumption batches
(`audience.batches`, the blobs of the recorded `cols.batch` decisions) and the platform's exact audience sets. A
fitted model is world state: the training process records the fit through `substrate.inference.recorded` (BLAS
results can differ across thread counts, so regeneration reuses the recorded parameters instead of refitting) and
emits one `feed.model` event (internal visibility) whose applier stores the parameter arrays under
`rsm.<platform>.<recommender>.<name>`. Nothing an in-world actor can call reads them.
"""
from __future__ import annotations

import numpy as np

from .. import audience as AUD
from ...substrate import artifacts

PREFIX = "rsm."


def model_key(platform: str, rec: str, name: str) -> str:
    return f"{PREFIX}{platform}.{rec}.{name}"


def model(world, platform: str, rec: str) -> dict | None:
    """The fitted parameter arrays of recommender `rec` on `platform`, or None before its first fit."""
    pre = f"{PREFIX}{platform}.{rec}."
    st = world.state
    out = {k[len(pre):]: st.arrays[k] for k in st.arrays if k.startswith(pre)}
    return out or None


def consumption_rows(world, platform: str, since_t: int, until_t: int | None = None) -> dict:
    """Concatenated consumption rows of the platform's retained batches recorded in [since_t, until_t]."""
    vs, its, ws, ts = [], [], [], []
    for b in AUD.batches(world, kind="consume", platform=platform, since_t=since_t):
        if until_t is not None and b["t"] > until_t:
            continue
        a = artifacts.get_array_blob(world, b["ref"])
        vs.append(a["idx"].astype(np.int64))
        its.append(a["v.aud_c_item"].astype(np.int64))
        ws.append(a["v.aud_c_watch_s"].astype(np.float64))
        ts.append(np.full(a["idx"].size, b["t"], np.int64))
    if not vs:
        z = np.zeros(0, np.int64)
        return {"viewer": z, "item": z, "watch_s": z.astype(np.float64), "t": z}
    return {"viewer": np.concatenate(vs), "item": np.concatenate(its), "watch_s": np.concatenate(ws),
            "t": np.concatenate(ts)}


def impression_rows(world, platform: str, since_t: int, until_t: int) -> dict:
    vs, its, sl, ts = [], [], [], []
    for b in AUD.batches(world, kind="imp", platform=platform, since_t=since_t):
        if b["t"] > until_t:
            continue
        a = artifacts.get_array_blob(world, b["ref"])
        vs.append(a["idx"].astype(np.int64))
        its.append(a["v.aud_i_item"].astype(np.int64))
        sl.append(a["v.aud_i_slot"].astype(np.int64))
        ts.append(np.full(a["idx"].size, b["t"], np.int64))
    if not vs:
        z = np.zeros(0, np.int64)
        return {"viewer": z, "item": z, "slot": z, "t": z}
    return {"viewer": np.concatenate(vs), "item": np.concatenate(its), "slot": np.concatenate(sl),
            "t": np.concatenate(ts)}


def is_view(world, items: np.ndarray, watch_s: np.ndarray) -> np.ndarray:
    dur = AUD.item_col(world, "duration_s")[items].astype(np.float64)
    return watch_s >= AUD.view_threshold(dur)


def implicit_als(P: np.ndarray, factors: int, reg: float, alpha: float, iters: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Implicit-feedback ALS (Hu, Koren & Volinsky 2008) on a dense binary preference matrix P [users, items] with
    confidence 1 + alpha * P. Returns (user factors X, item factors Y)."""
    U, I = P.shape
    f = int(factors)
    X = rng.standard_normal((U, f)) * 0.1
    Y = rng.standard_normal((I, f)) * 0.1
    eye = np.eye(f) * reg
    P = P.astype(np.float64)
    for _ in range(int(iters)):
        for left, right, M in ((0, None, P), (1, None, P.T)):
            other = Y if left == 0 else X
            G = other.T @ other + eye
            OO = np.einsum("if,ig->ifg", other, other).reshape(other.shape[0], f * f)
            A = G[None] + alpha * (M @ OO).reshape(M.shape[0], f, f)
            b = (1.0 + alpha) * (M @ other)
            sol = np.linalg.solve(A, b[..., None])[..., 0]
            if left == 0:
                X = sol
            else:
                Y = sol
    return X, Y


def logistic_fit(F: np.ndarray, y: np.ndarray, l2: float, iters: int, w0: np.ndarray | None = None) -> np.ndarray:
    """L2-regularised logistic regression by Newton's method (IRLS)."""
    n, d = F.shape
    w = np.zeros(d) if w0 is None or w0.shape != (d,) else w0.astype(np.float64).copy()
    R = np.eye(d) * l2 * n
    R[0, 0] = 0.0                                     # no penalty on the bias
    for _ in range(int(iters)):
        z = np.clip(F @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = F.T @ (p - y) + R @ w
        H = (F * (p * (1 - p))[:, None]).T @ F + R + np.eye(d) * 1e-9
        step = np.linalg.solve(H, g)
        w = w - step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return w
