"""Recommendation mechanisms of the social internet (workstream C2; MASTER_SPEC Part 4.5, INTERFACE_CONTRACTS 3).

A recommender ranks a platform's candidate items for a cohort of viewers:

    @runtime_checkable
class Recommender(Protocol):
        name: str; version: str
        def rank(self, world, platform, viewers, candidates, k, rng) -> np.ndarray   # [len(viewers), k] item ids

`REGISTRY` maps a name to its class; each class takes its parameters as keyword arguments. Twelve mechanisms
(`algorithms.py`), each a different algorithm over different evidence:

    chronological  following feed: newest items of creators the viewer follows
    interest       content-based retrieval: cosine of the viewer's (need-shifted) interests and item representations
    cf             collaborative filtering: implicit-feedback matrix factorisation (ALS) trained on recorded views
    engagement     learned engagement prediction: logistic model trained online on delayed, noisy outcomes
    explore        exploration: batched Thompson sampling on the platform's delayed, noisy published analytics
    social         social-graph distribution: items shared to the viewer and items the viewer's contacts liked
    diversity      diversity-aware ranking: maximal marginal relevance over interest relevance
    reputation     reputation-sensitive distribution: creators' earned completion and follow rates, blended
    search         search-driven discovery: BM25 over item text for queries drawn from the viewer's current needs
    fresh          recency and freshness: exponential age decay times recent view velocity
    trending       global popularity: recent view velocity and total views
    random         uniform random (control)

A platform blends several recommenders (`Mixture`): weighted reciprocal-rank fusion of their top-k lists, so the
final feed has k slots; an item that wins slots takes them from competitors (exposure is finite). Mixtures and
parameters are evaluator-side (`world.runtime["recsys_private"]`) and may change on a schedule during a run.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .. import audience as AUD


@runtime_checkable
class Recommender(Protocol):
    name: str
    version: str

    def rank(self, world, platform: str, viewers: np.ndarray, candidates: np.ndarray, k: int,
             rng) -> np.ndarray: ...


REGISTRY: dict[str, type] = {}


def register(cls):
    if not cls.name or cls.name in REGISTRY:
        raise ValueError(f"bad or duplicate recommender name {cls.name!r}")
    if cls.name not in AUD.SOURCE_CODE:
        raise ValueError(f"recommender {cls.name} has no impression source code")
    REGISTRY[cls.name] = cls
    return cls


def make(name: str, params: dict | None = None):
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"unknown recommender {name!r} (known: {sorted(REGISTRY)})")
    return cls(**(params or {}))


class Mixture:
    """A platform's blend of recommenders: weighted reciprocal-rank fusion (RRF) of each member's top-`depth` list.
    rank() returns the fused top-k items and, for each slot, the member that contributed most to it."""

    def __init__(self, entries: list[dict], rrf_k: float = 20.0, depth: int | None = None):
        if not entries:
            raise ValueError("a mixture needs at least one recommender")
        self.members = []
        for e in entries:
            w = float(e.get("weight", 1.0))
            if not (w > 0):
                raise ValueError("recommender weights must be > 0")
            self.members.append((make(e["name"], e.get("params")), w))
        self.rrf_k = float(rrf_k)
        self.depth = depth

    @property
    def names(self) -> list[str]:
        return [m.name for m, _ in self.members]

    def rank(self, world, platform, viewers, candidates, k, rng) -> tuple[np.ndarray, np.ndarray]:
        from .features import CHUNK
        viewers = np.asarray(viewers, dtype=np.int64)
        cand = np.unique(np.asarray(candidates, dtype=np.int64))
        V, C = viewers.size, cand.size
        if V == 0 or C == 0:
            return np.full((V, k), -1, np.int64), np.zeros((V, k), np.int8)
        if len(self.members) == 1:
            r = self.members[0][0].rank(world, platform, viewers, cand, k, rng)
            return r, np.where(r >= 0, AUD.SOURCE_CODE[self.members[0][0].name], 0).astype(np.int8)
        outs, srcs = [], []
        for s in range(0, V, CHUNK):
            o, sr = self._fuse(world, platform, viewers[s:s + CHUNK], cand, k, rng)
            outs.append(o)
            srcs.append(sr)
        return np.vstack(outs), np.vstack(srcs)

    def _fuse(self, world, platform, viewers, cand, k, rng):
        V, C = viewers.size, cand.size
        depth = int(self.depth or 2 * k)
        fused = np.zeros((V, C), dtype=np.float64)
        best = np.zeros((V, C), dtype=np.float64)
        src = np.zeros((V, C), dtype=np.int8)
        rows = np.repeat(np.arange(V), depth)
        for rec, w in self.members:
            r = rec.rank(world, platform, viewers, cand, depth, rng)
            pos = np.tile(np.arange(depth), V)
            flat = r.reshape(-1)
            ok = flat >= 0
            col = np.searchsorted(cand, flat[ok])
            contrib = w / (self.rrf_k + pos[ok] + 1.0)
            np.add.at(fused, (rows[ok], col), contrib)
            cur = best[rows[ok], col]
            upd = contrib > cur
            rr, cc = rows[ok][upd], col[upd]
            best[rr, cc] = contrib[upd]
            src[rr, cc] = AUD.SOURCE_CODE[rec.name]
        fused[fused <= 0] = -np.inf
        from .features import topk
        out = topk(fused, cand, k)
        colsel = np.searchsorted(cand, np.maximum(out, 0))
        s = np.take_along_axis(src, colsel, axis=1)
        s[out < 0] = 0
        return out, s


from . import algorithms  # noqa: E402,F401  (registers the mechanisms)
