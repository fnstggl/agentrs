"""Reference learners that do not learn: the floor every comparison needs.

`HoldLearner` never touches a lever (the idle participant), `RandomLearner` draws uniformly random settings every
tick (or holds each draw for `hold` ticks). Trained and scripted learners live in `research.learners`."""
from __future__ import annotations

import json

import numpy as np

from .learner import EnvSpec, Learner


class HoldLearner(Learner):
    """Leaves every lever where it is. Its score is 0 by construction of the normalisation."""
    name, kind = "hold", "idle"

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        return np.asarray(obs[:, self.spec.last_action], dtype=np.float32).copy()

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"learner": self.name}, fh)

    def load(self, path: str) -> None:
        pass


class RandomLearner(Learner):
    """Uniformly random settings, redrawn every `hold` ticks. Seeded; has no weights."""
    name, kind = "random", "random"

    def __init__(self, spec: EnvSpec, seed: int = 0, device: str = "cpu", hold: int = 1):
        super().__init__(spec, seed, device)
        self.hold = max(1, int(hold))
        self.rng = np.random.default_rng(seed)
        self._cur: np.ndarray | None = None
        self._age: np.ndarray | None = None

    def reset_context(self, n: int, which: np.ndarray | None = None) -> None:
        if self._cur is None or len(self._cur) != n:
            self._cur = self.rng.uniform(-1, 1, (n, self.spec.act_dim)).astype(np.float32)
            self._age = np.zeros(n, dtype=np.int64)
        elif which is not None:
            self._age[np.asarray(which, bool)] = 0
        else:
            self._age[:] = 0

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        n = obs.shape[0]
        if self._cur is None or len(self._cur) != n:
            self.reset_context(n)
        redraw = self._age % self.hold == 0
        if redraw.any():
            self._cur[redraw] = self.rng.uniform(-1, 1, (int(redraw.sum()), self.spec.act_dim)).astype(np.float32)
        self._age += 1
        return self._cur.copy()

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"learner": self.name, "seed": self.seed, "hold": self.hold}, fh)

    def load(self, path: str) -> None:
        pass
