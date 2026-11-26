"""Shared, read-only feature views for recommenders and the consumption engine (workstream C2).

Everything here reads world state (columns and exact audience sets) and computes; nothing writes. Results that
several recommenders need for the same cohort and candidate pool (membership matrices, candidate embeddings) are
memoised for the current world sequence number only, so a cache entry can never outlive the state it was read from.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .. import audience as AUD
from .. import idset
from ...substrate import columns

CHUNK = 8192


def _ck(world, *parts) -> tuple:
    h = hashlib.blake2b(digest_size=12)
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(p.dtype.str.encode())
            h.update(np.ascontiguousarray(p).tobytes())
        else:
            h.update(repr(p).encode())
    return (id(world.state), world.seq, h.hexdigest())


def _cache(world) -> dict:
    c = world.runtime.get("recsys.cache")
    if c is None or c.get("_seq") != (id(world.state), world.seq):
        c = world.runtime["recsys.cache"] = {"_seq": (id(world.state), world.seq)}
    return c


def memo(world, name, fn, *parts):
    c = _cache(world)
    k = (name, _ck(world, *parts))
    if k not in c:
        c[k] = fn()
    return c[k]


def hash_normal(seed: int, idx: np.ndarray, dims: int, salt: int) -> np.ndarray:
    """Deterministic standard normals [len(idx), dims] from a counter-based hash (Box-Muller)."""
    from ..scale import splitmix64
    idx = np.asarray(idx, dtype=np.uint64)
    out = np.empty((idx.size, dims), dtype=np.float64)
    with np.errstate(over="ignore"):
        base = splitmix64(np.uint64((seed * 0x9E3779B1 + salt * 0x85EBCA77) & 0xFFFFFFFFFFFFFFFF))
        for d in range(0, dims, 2):
            h1 = splitmix64(idx ^ (base + np.uint64(2 * d + 1)))
            h2 = splitmix64(idx ^ (base + np.uint64(2 * d + 2)))
            u1 = ((h1 >> np.uint64(11)).astype(np.float64) + 0.5) / 9007199254740992.0
            u2 = ((h2 >> np.uint64(11)).astype(np.float64) + 0.5) / 9007199254740992.0
            r = np.sqrt(-2.0 * np.log(u1))
            out[:, d] = r * np.cos(2 * np.pi * u2)
            if d + 1 < dims:
                out[:, d + 1] = r * np.sin(2 * np.pi * u2)
    return out


def interests(world, viewers: np.ndarray, *, need_weight: float = 0.0) -> np.ndarray:
    """Unit interest vectors (float32) of viewers; with need_weight > 0, shifted by today's current need (a
    deterministic per-individual, per-day perturbation): what someone looks for today is not exactly their taste."""
    u = columns.col(world, "pop", "interests")[viewers].astype(np.float32)
    if need_weight > 0:
        day = int(world.now // 86400)
        nd = hash_normal(world.spec.seed, viewers, u.shape[1], 7919 + day).astype(np.float32) / np.sqrt(u.shape[1])
        u = u + np.float32(need_weight) * nd
    n = np.linalg.norm(u, axis=1, keepdims=True)
    return u / np.maximum(n, 1e-6)


def item_emb(world, items: np.ndarray) -> np.ndarray:
    e = columns.col(world, AUD.ITEMS, "emb")[items].astype(np.float32)
    n = np.linalg.norm(e, axis=1, keepdims=True)
    return e / np.maximum(n, 1e-6)


def membership(world, kind: str, keys: np.ndarray, viewers: np.ndarray) -> np.ndarray:
    """[len(viewers), len(keys)] bool: viewers[v] in set (kind, keys[j]). kind: vw (viewed item), ex (exposed to
    item), cv (viewed creator), fo (follows creator)."""
    def run():
        out = np.zeros((viewers.size, keys.size), dtype=bool)
        st = world.state
        for j, k in enumerate(keys):
            name = idset.key(kind, int(k))
            if name in st.arrays:
                out[:, j] = idset.contains(st, name, viewers)
        return out
    return memo(world, f"member.{kind}", run, keys, viewers)


def creators_of(world, items: np.ndarray) -> np.ndarray:
    return columns.col(world, AUD.ITEMS, "creator")[items].astype(np.int64)


def creator_membership(world, kind: str, items: np.ndarray, viewers: np.ndarray) -> np.ndarray:
    """[V, C] bool: viewer v in the creator set (cv / fo) of the creator of items[c]."""
    cre = creators_of(world, items)
    uc, inv = np.unique(cre, return_inverse=True)
    m = membership(world, kind, uc, viewers)
    return m[:, inv]


def age_s(world, items: np.ndarray) -> np.ndarray:
    return (world.now - columns.col(world, AUD.ITEMS, "published_at")[items]).astype(np.float64)


def topk(scores: np.ndarray, cand: np.ndarray, k: int) -> np.ndarray:
    """Row-wise top-k candidates by score (descending, ties by candidate order); -inf scores are not returned
    (padded with -1)."""
    V, C = scores.shape
    k = int(k)
    if C == 0 or V == 0:
        return np.full((V, k), -1, dtype=np.int64)
    kk = min(k, C)
    if kk < C:
        part = np.argpartition(-scores, kk - 1, axis=1)[:, :kk]
    else:
        part = np.broadcast_to(np.arange(C), (V, C)).copy()
    ps = np.take_along_axis(scores, part, axis=1)
    o = np.argsort(-ps, axis=1, kind="stable")
    part = np.take_along_axis(part, o, axis=1)
    ps = np.take_along_axis(ps, o, axis=1)
    out = np.full((V, k), -1, dtype=np.int64)
    sel = cand[part]
    sel[~np.isfinite(ps)] = -1
    out[:, :kk] = sel
    return out
