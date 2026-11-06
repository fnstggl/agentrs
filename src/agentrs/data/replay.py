"""Experience replay with stratified, source-weighted sampling and leakage guards."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Iterable, Optional

from agentrs.data.schemas import EpisodeRecord, Source, Transition

TRAINABLE_SPLITS = frozenset({"train"})


class LeakageError(RuntimeError):
    pass


class ReplayBuffer:
    """Holds real and imagined experience in separate pools.

    * Refuses transitions from non-training splits (held-out tasks never enter).
    * Supports uniform, recent, historical, success/failure, high-error and
      category-stratified sampling, with per-source weights.
    """

    def __init__(self, capacity: int = 200_000, seed: int = 0,
                 allowed_splits: Iterable[str] = TRAINABLE_SPLITS):
        self.capacity = capacity
        self.rng = random.Random(seed)
        self.allowed_splits = frozenset(allowed_splits)
        self.real: list[Transition] = []
        self.imagined: list[Transition] = []
        self.priority: dict[int, float] = {}  # id(transition) -> e.g. world-model prediction error
        self._insert_counter = 0
        self._order: dict[int, int] = {}

    # ------------------------------------------------------------------ insert
    def add(self, t: Transition) -> None:
        if t.split not in self.allowed_splits:
            raise LeakageError(f"transition from split {t.split!r} ({t.task_id}) may not enter the training buffer")
        pool = self.imagined if t.provenance.source == Source.IMAGINED else self.real
        pool.append(t)
        self._insert_counter += 1
        self._order[id(t)] = self._insert_counter
        if len(pool) > self.capacity:
            # reservoir-style eviction that keeps some old data (prevents forgetting)
            idx = self.rng.randrange(len(pool) // 2) if self.rng.random() < 0.5 else 0
            old = pool.pop(idx)
            self.priority.pop(id(old), None)
            self._order.pop(id(old), None)

    def add_episode(self, ep: EpisodeRecord) -> None:
        for t in ep.transitions:
            self.add(t)

    def extend(self, ts: Iterable[Transition]) -> None:
        for t in ts:
            self.add(t)

    def __len__(self) -> int:
        return len(self.real)

    def set_priority(self, t: Transition, value: float) -> None:
        self.priority[id(t)] = float(value)

    # ------------------------------------------------------------------ sample
    def _pool(self, imagined: bool) -> list[Transition]:
        return self.imagined if imagined else self.real

    def sample(self, n: int, strategy: str = "uniform", imagined: bool = False,
               source_weights: Optional[dict[str, float]] = None,
               predicate: Optional[Callable[[Transition], bool]] = None,
               recent_fraction: float = 0.25) -> list[Transition]:
        pool = self._pool(imagined)
        if predicate is not None:
            pool = [t for t in pool if predicate(t)]
        if not pool:
            return []
        if strategy == "uniform":
            cand = pool
            weights = None
        elif strategy == "recent":
            cand = sorted(pool, key=lambda t: self._order.get(id(t), 0))[-max(1, int(len(pool) * recent_fraction)):]
            weights = None
        elif strategy == "historical":
            cand = sorted(pool, key=lambda t: self._order.get(id(t), 0))[: max(1, int(len(pool) * (1 - recent_fraction)))]
            weights = None
        elif strategy == "success":
            cand = [t for t in pool if t.episode_success] or pool
            weights = None
        elif strategy == "failure":
            cand = [t for t in pool if t.episode_success is False] or pool
            weights = None
        elif strategy == "prioritized":
            cand = pool
            weights = [self.priority.get(id(t), 1.0) + 1e-3 for t in pool]
        elif strategy == "stratified":
            return self._stratified(pool, n, key=lambda t: t.family or t.task_id)
        elif strategy == "balanced_old_new":
            half = n // 2
            return (self.sample(half, "recent", imagined, source_weights, predicate, recent_fraction)
                    + self.sample(n - half, "historical", imagined, source_weights, predicate, recent_fraction))
        else:
            raise ValueError(f"unknown strategy {strategy}")
        if source_weights:
            sw = [source_weights.get(t.provenance.source.value, 1.0) for t in cand]
            weights = sw if weights is None else [a * b for a, b in zip(weights, sw)]
        if weights is not None and sum(weights) <= 0:
            return []
        return self.rng.choices(cand, weights=weights, k=n)

    def _stratified(self, pool: list[Transition], n: int, key) -> list[Transition]:
        groups: dict[str, list[Transition]] = defaultdict(list)
        for t in pool:
            groups[key(t)].append(t)
        keys = sorted(groups)
        out = []
        for i in range(n):
            g = groups[keys[i % len(keys)]]
            out.append(self.rng.choice(g))
        self.rng.shuffle(out)
        return out

    def episodes(self, imagined: bool = False) -> list[list[Transition]]:
        eps: dict[str, list[Transition]] = defaultdict(list)
        for t in self._pool(imagined):
            eps[t.episode_id].append(t)
        return [sorted(v, key=lambda t: t.step) for v in eps.values()]

    def stats(self) -> dict:
        by_src: dict[str, int] = defaultdict(int)
        for t in self.real:
            by_src[t.provenance.source.value] += 1
        return {"real": len(self.real), "imagined": len(self.imagined), "by_source": dict(by_src)}
