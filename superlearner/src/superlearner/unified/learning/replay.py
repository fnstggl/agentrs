"""Persistent episode replay with per-family reservoirs.

Real experience only: nothing imagined is ever stored. Each family keeps a recent window and a true uniform reservoir
of everything it has seen, so a long stretch on one family cannot erase another's experience.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import torch

from ..interfaces import ReplaySequence

_FIELDS = ("values", "deltas", "present", "extra", "goal", "cand_tokens", "cand_features", "cand_mask", "cand_dur", "cand_bonus", "rscale", "action",
           "behavior_log_prob", "reward", "dt", "cont", "first")


class EpisodeReplay:
    def __init__(self, capacity_per_family: int = 400, recent_per_family: int = 100, holdout: float = 0.1, seed: int = 0,
                 n_ids: int = 33):
        self.capacity, self.recent_cap, self.holdout = capacity_per_family, recent_per_family, holdout
        self.rng = np.random.default_rng(seed)
        self.reservoir: dict[str, list[dict]] = defaultdict(list)
        self.recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.recent_cap))
        self.seen: dict[str, int] = defaultdict(int)
        self.held: deque = deque(maxlen=128)
        self.total_steps = 0
        self.ids = np.arange(n_ids)

    # -------------------------------------------------------------------- writing
    def add(self, episode: dict, family: str = "default") -> None:
        n = len(episode["reward"])
        if n < 2:
            return
        ep = {k: np.asarray(episode[k]) for k in _FIELDS}
        ep["family"] = family
        if self.rng.random() < self.holdout:
            self.held.append(ep)     # held-out episodes are only used to test prediction
            return
        self.total_steps += n
        self.seen[family] += 1
        self.recent[family].append(ep)
        pool = self.reservoir[family]
        if len(pool) < self.capacity:
            pool.append(ep)
        else:
            j = int(self.rng.integers(self.seen[family]))
            if j < self.capacity:
                pool[j] = ep

    def __len__(self) -> int:
        return sum(len(v) for v in self.reservoir.values())

    # -------------------------------------------------------------------- reading
    def _pick(self) -> dict:
        fams = [f for f, v in self.reservoir.items() if v]
        fam = fams[int(self.rng.integers(len(fams)))]
        pool = self.recent[fam] if self.rng.random() < 0.5 and self.recent[fam] else self.reservoir[fam]
        return pool[int(self.rng.integers(len(pool)))]

    def sample(self, batch: int, length: int) -> ReplaySequence:
        if not len(self):
            raise ValueError("replay is empty")
        return self._collate([self._cut(self._pick(), length) for _ in range(batch)], length)

    def sample_holdout(self, batch: int, length: int) -> ReplaySequence | None:
        if not self.held:
            return None
        eps = [self.held[int(self.rng.integers(len(self.held)))] for _ in range(batch)]
        return self._collate([self._cut(e, length) for e in eps], length)

    def _cut(self, ep: dict, length: int) -> tuple[dict, int, int]:
        n = len(ep["reward"])
        start = 0 if n <= length else int(self.rng.integers(n - length + 1))
        return ep, start, min(n, start + length)

    def _collate(self, cuts: list, length: int) -> ReplaySequence:
        b = len(cuts)
        first = cuts[0][0]
        arrays = {k: np.zeros((b, length, *first[k].shape[1:]), dtype=first[k].dtype) for k in _FIELDS}
        mask = np.zeros((b, length), dtype=np.float32)
        fams = []
        for i, (ep, s, e) in enumerate(cuts):
            for k in _FIELDS:
                arrays[k][i, :e - s] = ep[k][s:e]
            mask[i, :e - s] = 1.0
            fams.append(ep["family"])
        # Padding rows need a valid candidate so that masked softmax stays defined; they are excluded by `mask`.
        pad = mask == 0
        arrays["cand_mask"][pad, 0] = True
        arrays["cont"][pad] = 0.0
        arrays["rscale"][pad] = 1.0
        t = {k: torch.from_numpy(v) for k, v in arrays.items()}
        ids = torch.from_numpy(self.ids).expand(b, length, -1).clone()
        return ReplaySequence(values=t["values"].float(), deltas=t["deltas"].float(), present=t["present"].bool(), ids=ids,
                              extra=t["extra"].float(),
                              goal=t["goal"].float(), cand_tokens=t["cand_tokens"].long(),
                              cand_features=t["cand_features"].float(), cand_mask=t["cand_mask"].bool(),
                              cand_dur=t["cand_dur"].float(), cand_bonus=t["cand_bonus"].float(), rscale=t["rscale"].float(), action=t["action"].long(),
                              behavior_log_prob=t["behavior_log_prob"].float(), reward=t["reward"].float(),
                              dt=t["dt"].float(), cont=t["cont"].float(), first=t["first"].float(),
                              mask=torch.from_numpy(mask), family=fams)
