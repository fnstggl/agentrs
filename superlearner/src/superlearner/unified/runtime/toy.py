"""Small controlled worlds for the mandatory integration tests. They use Crucible's observation layout."""
from __future__ import annotations

import numpy as np

from crucible.research.learner import EnvSpec


class ToyVecEnv:
    """``target``: reward peaks at hidden lever settings. ``cue``: a reading shown only on the first tick decides which
    lever direction pays on later ticks. ``cue_final``: the same cue, but only the last tick is rewarded, so no reward
    reveals the answer along the way and only memory of the first observation helps."""

    def __init__(self, n: int, mode: str = "target", horizon: int = 12, seed: int = 0, target: dict | None = None,
                 spec: EnvSpec | None = None):
        self.spec = spec or EnvSpec()
        self.n, self.mode, self.h = n, mode, horizon
        self.rng = np.random.default_rng(seed)
        self.target = target or {0: 0.6}
        self.t = np.zeros(n, dtype=np.int64)
        self.done = np.ones(n, dtype=bool)
        self.x = np.zeros((n, self.spec.max_levers), dtype=np.float32)
        self.cue = np.ones(n, dtype=np.float32)
        self.returns = np.zeros(n)
        self.finished: list[float] = []

    def _obs(self, r: np.ndarray) -> np.ndarray:
        sp = self.spec
        o = np.zeros((self.n, sp.obs_dim), dtype=np.float32)
        o[:, sp.gauge_mask.start:sp.gauge_mask.start + 2] = 1.0
        o[:, sp.lever_mask.start:sp.lever_mask.start + 6] = 1.0
        if self.mode in ("cue", "cue_final"):
            # The settings stay hidden here: a visible lever would act as an external memory of the cue.
            o[:, 0] = np.where(self.t == 0, self.cue, 0.0)
        else:
            o[:, sp.last_action] = self.x
            o[:, 1] = self.x[:, 0]
        sc = sp.scalars.start
        o[:, sc] = r
        o[:, sc + 3] = 1.0
        o[:, sc + 4] = self.t / self.h
        return o

    def reset(self, which=None) -> np.ndarray:
        idx = np.arange(self.n) if which is None else np.flatnonzero(which)
        self.t[idx] = 0
        self.done[idx] = False
        self.x[idx] = 0.0
        self.returns[idx] = 0.0
        self.cue[idx] = self.rng.choice([-1.0, 1.0], len(idx))
        return self._obs(np.zeros(self.n, dtype=np.float32))

    def _reward(self, a: np.ndarray) -> np.ndarray:
        if self.mode == "cue_final":
            last = self.t == self.h - 1
            return np.where(last, (np.sign(a[:, 0]) * self.cue > 0) * 1.0 - 0.5, 0.0).astype(np.float32)
        if self.mode == "cue":
            late = self.t >= 2
            return np.where(late, (np.sign(a[:, 0]) * self.cue > 0) * 1.0 - 0.5, 0.0).astype(np.float32)
        err = sum((a[:, k] - v) ** 2 for k, v in self.target.items()) / len(self.target)
        return (1.0 - 2.0 * err).astype(np.float32)

    def step(self, actions: np.ndarray):
        a = np.clip(actions, -1, 1)
        live = ~self.done
        r = self._reward(a) * live
        self.x[live] = a[live]
        self.t[live] += 1
        self.returns += r
        ended = live & (self.t >= self.h)
        self.done |= ended
        self.finished += [float(self.returns[i]) / self.h for i in np.flatnonzero(ended)]
        return self._obs(r), r, self.done.copy(), {}

    def drain(self) -> list[float]:
        out, self.finished = self.finished, []
        return out


def run_toy(learner, env: ToyVecEnv, episodes: int, learn: bool, explore: bool | None = None, updates: bool = True) -> float:
    """Run whole batches of episodes; return the mean per-tick return."""
    explore = learn if explore is None else explore
    was = learner.frozen
    learner.freeze(not learn)
    out: list[float] = []
    try:
        for _ in range(max(1, episodes // env.n)):
            obs = env.reset()
            learner.reset_context(env.n)
            while not env.done.all():
                a = learner.act(obs, explore=explore)
                nxt, r, d, _ = env.step(a)
                if learn:
                    learner.observe(obs, a, r, d, nxt)
                obs = nxt
            if learn and updates:
                learner.update()
            out += env.drain()
    finally:
        learner.freeze(was)
    return float(np.mean(out))
