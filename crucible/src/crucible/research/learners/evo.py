"""`evo_probe`: population search over the parameters of a procedure.

A third kind of learner next to gradient-trained policies and learned models: the *form* of the behaviour is the
scripted probe-and-commit experimenter (`scripted.ProbeExploitLearner`) and what is learned across episodes is a
handful of its parameters (how long an arm lasts, how much of the episode may be spent probing, how sure a gain
must be, whether the first tick of an arm is discarded, how long the first step of a lever is). Learning is the cross-entropy
method: every training episode runs with a parameter vector drawn from a diagonal Gaussian; when `population`
episodes have finished, the Gaussian moves toward the vectors of the best `elite_frac` of them (by episode return
in the public money unit). That is one "update". Frozen or not exploring, the mean vector is used.

Episode returns of different worlds are not on one scale, so a draw is judged by its rank among the episodes of
the same length in the population (the episode length is the only public hint of which kind of world it was);
the selection is still noisy and a population of a few dozen episodes per update averages over worlds. No
gradient, no network, ten numbers of long-term state.
"""
from __future__ import annotations

import json
import time

import numpy as np

from ..learner import EnvSpec, Learner
from .scripted import PROBE_DEFAULTS, _ProbeSlot

EVO_DEFAULTS = {"population": 48, "elite_frac": 0.25, "smooth": 0.5, "init_sd": 0.8, "min_sd": 0.1}
N_THETA = 5


def _sig(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def decode(theta: np.ndarray) -> dict:
    """Parameter vector -> options of the probe-and-commit procedure."""
    return dict(PROBE_DEFAULTS, ticks_per_arm=int(round(2 + 4 * _sig(theta[0]))),
                max_probe_frac=0.15 + 0.55 * _sig(theta[1]), accept_z=3.0 * _sig(theta[2]),
                settle=int(theta[3] > 0.0), delta=0.1 + 0.7 * _sig(theta[4]))


class EvoProbeLearner(Learner):
    name, kind = "evo_probe", "search"

    def __init__(self, spec: EnvSpec, seed: int = 0, device: str = "cpu", **kw):
        super().__init__(spec, seed, device)
        bad = sorted(set(kw) - set(EVO_DEFAULTS))
        if bad:
            raise TypeError(f"evo_probe has no option(s) {bad}")
        self.p = dict(EVO_DEFAULTS, **kw)
        self.mean = np.array([0.0, 0.0, 0.0, 0.5, 0.0])
        self.sd = np.full(N_THETA, float(self.p["init_sd"]))
        self.generation = 0
        self._usage.params = 2 * N_THETA
        self._rng = np.random.default_rng([self.seed, 611953])
        self._slots: list[_ProbeSlot] | None = None
        self._theta: list[np.ndarray | None] = []
        self._ret: np.ndarray | None = None
        self._live: np.ndarray | None = None
        self._results: list[tuple[float, np.ndarray, int]] = []
        self._len: np.ndarray | None = None
        self.last_stats: dict = {}

    def _new(self, i: int) -> _ProbeSlot:
        return _ProbeSlot(decode(self.mean), self.spec, np.random.default_rng([self.seed, 7919, i]))

    def reset_context(self, n: int, which: np.ndarray | None = None) -> None:
        if self._slots is None or len(self._slots) != n or which is None:
            self._slots = [self._new(i) for i in range(n)]
            self._theta = [None] * n
            self._ret, self._live, self._len = np.zeros(n), np.ones(n, dtype=bool), np.zeros(n, dtype=np.int64)
            return
        w = np.asarray(which)
        for i in (np.flatnonzero(w) if w.dtype == bool else w):
            i = int(i)
            self._slots[i], self._theta[i], self._ret[i], self._live[i] = self._new(i), None, 0.0, True
            self._len[i] = 0

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        c0 = time.process_time()
        n = obs.shape[0]
        if self._slots is None or len(self._slots) != n:
            self.reset_context(n)
        out = np.zeros((n, self.spec.act_dim), dtype=np.float32)
        for i, s in enumerate(self._slots):
            if s.t == 0 and explore and not self.frozen:        # a training episode: its own parameter draw
                self._theta[i] = self.mean + self.sd * self._rng.standard_normal(N_THETA)
                s.p = decode(self._theta[i])
            out[i] = s.step(obs[i])
        self._usage.infer_cpu_s += time.process_time() - c0
        return out

    def observe(self, obs, action, reward, done, next_obs) -> None:
        if self.frozen or self._ret is None:
            return
        c0 = time.process_time()
        for i in range(len(self._ret)):
            if not self._live[i]:
                continue
            self._ret[i] += float(reward[i])
            self._len[i] += 1
            self._usage.extra["observed_steps"] = self._usage.extra.get("observed_steps", 0) + 1
            if done[i]:
                self._live[i] = False
                if self._theta[i] is not None:
                    self._results.append((float(self._ret[i]), self._theta[i], int(self._len[i])))
        self._usage.train_cpu_s += time.process_time() - c0

    def update(self) -> dict:
        if self.frozen or len(self._results) < int(self.p["population"]):
            return {"updates": 0}
        c0 = time.process_time()
        res, self._results = self._results, []
        groups: dict[int, list[int]] = {}
        for i, (_, _, n) in enumerate(res):
            groups.setdefault(n, []).append(i)
        fit = np.zeros(len(res))
        for idx in groups.values():                             # centred rank among episodes of the same length
            order = np.argsort([res[i][0] for i in idx])
            for rank, j in enumerate(order):
                fit[idx[j]] = (rank + 0.5) / len(idx) - 0.5
        best = np.argsort(-fit, kind="stable")
        k = max(2, int(round(float(self.p["elite_frac"]) * len(res))))
        elite = np.stack([res[int(i)][1] for i in best[:k]])
        a = float(self.p["smooth"])
        self.mean = (1 - a) * self.mean + a * elite.mean(0)
        self.sd = np.maximum(float(self.p["min_sd"]), (1 - a) * self.sd + a * elite.std(0))
        self.generation += 1
        self._usage.updates += 1
        self.last_stats = {"generation": self.generation, "mean_return": float(np.mean([r[0] for r in res])),
                           "elite_return": float(np.mean([res[int(i)][0] for i in best[:k]])),
                           "options": decode(self.mean)}
        self._usage.train_cpu_s += time.process_time() - c0
        return {"updates": 1}

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"learner": self.name, "seed": self.seed, "params": self.p, "mean": self.mean.tolist(),
                       "sd": self.sd.tolist(), "generation": self.generation}, fh)

    def load(self, path: str) -> None:
        with open(path) as fh:
            d = json.load(fh)
        self.p.update(d.get("params", {}))
        self.mean, self.sd = np.asarray(d["mean"], dtype=float), np.asarray(d["sd"], dtype=float)
        self.generation = int(d.get("generation", 0))
        self._usage.updates = self.generation
        self._slots = None
