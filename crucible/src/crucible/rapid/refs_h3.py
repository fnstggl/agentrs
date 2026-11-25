"""Evaluator-side oracles and scripted public references for the H3 families (MASTER_SPEC Part 7.4).

Oracles run on forks of a world with private access (the realised noise, the hidden state); references play through
a `PublicView` only. Nothing in a world imports this module (`admission.static_isolation`); families name these
functions in their `oracle=` / `reference=` fields and `oracles.oracle` / `references.play` dispatch to them.
"""
from __future__ import annotations

import math

import numpy as np

from .oracles import _levers, _noise, eval_schedules, fork, lever_oracle, value


# ====================================================================== lever families (primitive_graph)
def _run_schedule(env, names, X) -> float:
    e = fork(env)
    for row in X:
        e.step({"kind": "levers", "values": {n: float(x) for n, x in zip(names, row)}})
    return value(e)


def lever_schedule_oracle(env, segments=(4, 8), pop: int = 64, iters: int = 10, grid: int = 11,
                          seed: int = 0) -> dict:
    """Search over lever schedules on the world's realised noise: the best constant setting on a grid, then
    cross-entropy search over piecewise-constant schedules with 4 and 8 segments started from it, then the
    4.0-style schedule oracle; the best schedule is replayed on a fork and its value returned."""
    names, lo, hi = _levers(env)
    H = env.horizon - env.t
    L = len(names)
    noise = _noise(env, H)
    axes = [np.linspace(lo[j], hi[j], grid) for j in range(L)]
    G = np.array(np.meshgrid(*axes, indexing="ij")).reshape(L, -1).T
    v = eval_schedules(env, np.repeat(G[:, None, :], H, axis=1), None, noise)
    best_c = G[int(np.argmax(v))]
    cands = [(float(v.max()), np.repeat(best_c[None, :], H, axis=0), "best constant on a grid")]
    rng = np.random.default_rng(seed)
    for S in segments:
        seg_of = np.minimum(S - 1, (np.arange(H) * S) // max(H, 1))
        mu = np.repeat(((best_c - lo) / np.maximum(hi - lo, 1e-12))[None, :], S, axis=0)
        sd = np.full((S, L), 0.3)
        best = (-math.inf, None)
        for it in range(iters):
            U = np.clip(rng.normal(mu, sd, (pop, S, L)), 0.0, 1.0)
            U[0] = mu
            X = lo + U[:, seg_of, :] * (hi - lo)
            val = eval_schedules(env, X, None, noise)
            order = np.argsort(-val)
            if val[order[0]] > best[0]:
                best = (float(val[order[0]]), X[order[0]])
            el = U[order[:max(4, pop // 8)]]
            mu, sd = el.mean(axis=0), np.maximum(el.std(axis=0), 0.02)
        cands.append((best[0], best[1], f"schedule CEM ({S} segments, from the best constant)"))
    top = max(cands, key=lambda c: c[0])
    v_top = _run_schedule(env, names, top[1])
    h0 = lever_oracle(env)
    if h0["value"] >= v_top:
        return dict(h0, candidates={c[2]: c[0] for c in cands})
    return {"value": v_top, "how": top[2], "search_value": top[0], "candidates": {c[2]: c[0] for c in cands}}
