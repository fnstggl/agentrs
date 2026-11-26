"""Learner-facing vectorised Tier 1 worlds (MASTER_SPEC Part 8.12: fast CPU and batched simulation where valid).

`RapidVecEnv(family, seeds)` steps many worlds of one batchable family together through the vectorised batch path
(`crucible.rapid.batch.BatchRapid`, bit-for-bit equal to the single worlds `make(family, seed)`). It is a
participant interface: it returns only what each world's participant observes - its readings and the money it
saw move (revenue and cost of the tick) - and, in the default `observation_only` mode (Clarification 2), a reward
of exactly 0.0 for every world. Hidden parameters, the evaluator outcome and family identity never appear; worlds
are indexed by position only.

    vec = RapidVecEnv("net_routing", seeds=range(256))
    obs = vec.reset()                              # {"readings": (B, G), "revenue": (B,), "cost": (B,), "t": 0}
    obs, reward, done = vec.step(levers)           # levers (B, L) in lever units; reward == 0.0

`BatchedSearchLearner` is a learner that uses it: a population of lever settings per world, improved from the
income each world's participant observed (cross-entropy search over constant settings within an episode, one
setting per world, all worlds of the batch at once). It knows no mechanism and reads no evaluator data.

Throughput target (decision H-D3, fixed before measuring the gate): on one CPU thread, a learner rolling out a
batch of 256 worlds through `RapidVecEnv` must reach at least THROUGHPUT_TARGET = 20,000 world-steps per second
and at least SPEEDUP_TARGET = 10 times the world-steps per second of the same learner stepping the single worlds
one by one. Rationale: the single-world path runs at a few thousand steps per second; a batched path that is not an
order of magnitude faster does not change what a learner can afford (hundreds of thousands of episodes per
CPU-hour), and 20,000 steps/s leaves a margin of about five under the measured 100,000 steps/s of the bare batch
step on this machine under shared load.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..experience.modes import OBSERVATION_ONLY, check_mode
from .batch import BatchRapid

THROUGHPUT_TARGET = 20_000.0
SPEEDUP_TARGET = 10.0


class RapidVecEnv:
    def __init__(self, family: str, seeds: Sequence[int], reward_mode: str | None = None, **difficulty):
        self.reward_mode = check_mode(reward_mode)
        if self.reward_mode != OBSERVATION_ONLY:
            raise ValueError("Tier 1 worlds return no reward; only observation_only is available here")
        self._b = BatchRapid(family, seeds, **difficulty)
        self.B = self._b.B
        self.n_levers = self._b.n_levers
        self.lever_low, self.lever_high = self._b.lo.copy(), self._b.hi.copy()
        self.horizon = int(self._b.singles[0].horizon)
        self.t = 0

    def reset(self) -> dict:
        self._b.reset()
        self.t = 0
        return {"readings": np.zeros((self.B, 0)), "revenue": np.zeros(self.B), "cost": np.zeros(self.B), "t": 0}

    def step(self, levers: np.ndarray) -> tuple[dict, np.ndarray, bool]:
        out = self._b.step(levers)
        self.t += 1
        obs = {"readings": np.asarray(out["readings"], dtype=float), "revenue": np.asarray(out["revenue"]),
               "cost": np.asarray(out["cost"]), "t": self.t}
        return obs, np.zeros(self.B), self.t >= self.horizon


class BatchedSearchLearner:
    """Cross-entropy search over constant lever settings, one search per world, from observed income only: each
    round holds a candidate setting for `hold` ticks in every world, scores it by the income that world's
    participant observed over the later half of the hold, and moves each world's sampling distribution towards its
    best candidates."""

    def __init__(self, vec: RapidVecEnv, hold: int = 4, elite: float = 0.3, seed: int = 0):
        self.vec, self.hold, self.elite = vec, int(hold), float(elite)
        self.rng = np.random.default_rng(seed)
        B, L = vec.B, vec.n_levers
        self.mu = np.full((B, L), 0.5)
        self.sd = np.full((B, L), 0.3)
        self.steps = 0

    def episode(self, rounds_per_update: int = 4) -> np.ndarray:
        """Play one episode in every world; returns each world's observed net income."""
        v = self.vec
        v.reset()
        lo, span = v.lever_low, v.lever_high - v.lever_low
        total = np.zeros(v.B)
        cands, scores = [], []
        done = False
        while not done:
            u = np.clip(self.mu + self.sd * self.rng.standard_normal(self.mu.shape), 0.0, 1.0)
            got = np.zeros(v.B)
            for k in range(self.hold):
                obs, reward, done = v.step(lo + u * span)
                assert not reward.any()
                net = obs["revenue"] - obs["cost"]
                total += net
                self.steps += v.B
                if k >= self.hold // 2:
                    got += net
                if done:
                    break
            cands.append(u)
            scores.append(got)
            if len(cands) >= rounds_per_update:
                U, S = np.stack(cands, 1), np.stack(scores, 1)          # (B, R, L), (B, R)
                n = max(1, int(round(self.elite * U.shape[1])))
                top = np.argsort(-S, axis=1)[:, :n]
                el = np.take_along_axis(U, top[:, :, None], axis=1)
                self.mu = 0.5 * self.mu + 0.5 * el.mean(axis=1)
                self.sd = np.clip(0.5 * self.sd + 0.5 * el.std(axis=1), 0.03, 0.4)
                cands, scores = [], []
        return total
