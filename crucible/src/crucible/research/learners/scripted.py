"""Scripted experimenters: fixed, sensible procedures with no weights.

They adapt only inside an episode, so they measure what a procedure that was written down in advance achieves on a
mechanism it knows nothing about. Both read only the observation: the lever mask, the settings in force, the last
reward, the operator's balance and the clock (`time` is tick / horizon, so the horizon is known after one tick).
Neither sees a lever name, a gauge meaning or anything private to the task.

`probe_exploit`
    1. hold the starting settings for one arm (the baseline);
    2. one lever at a time, everything else where it is: an arm is a setting held for `ticks_per_arm` ticks, the
       first `settle` ticks not counted. Three designs (`arms`):
       "line"   (default) step the lever `delta` up (down if that fails or the range ends); a step that beats the
                running level is kept at once and followed by a step twice as long the same way, until one fails.
                What is kept is already the better setting and a failed step is small, so the experiment is cheap
                where most moves lose money;
       "local"  baseline plus and minus `delta` for every lever (both inward at a bound), nothing kept until the end;
       "range"  the two of (bottom, middle, top of the range) furthest from the baseline, and with `interpolate`
                the maximum of the parabola through the three points;
    3. commit to the combination found ("local"/"range": the best arm of every lever that beat the baseline, run
       for one arm and kept only if it beats the baseline, otherwise the single best arm or the baseline);
    4. hold. If the running mean of net falls well below the level measured right after committing (what a regime
       change looks like), start again from step 1 around the current settings.
    `ticks_per_arm` is shortened (never below 2) when the whole experiment would take more than `max_probe_frac`
    of the horizon, and experimenting stops at that fraction in any case.

`hill_climb`
    Simultaneous-perturbation stochastic approximation: evaluate x + c*D and x - c*D for a random sign vector D,
    move x along D by a step proportional to the normalised difference, with a decaying gain; a step whose pair
    average is clearly worse than the previous one is undone. The best measured point is held for the last
    `exploit_frac` of the episode.

Both stop experimenting and return to the starting settings when the balance is nearly gone.
"""
from __future__ import annotations

import json
import time

import numpy as np

from ..learner import EnvSpec, Learner

PROBE_DEFAULTS = {"ticks_per_arm": 4, "settle": 1, "arms": "line", "delta": 0.4, "interpolate": True,
                  "max_probe_frac": 0.5, "accept_z": 1.0,
                  "reprobe": True, "collapse_frac": 0.5, "window": 6, "min_drop": 0.03, "min_funds": 0.15}
HILL_DEFAULTS = {"ticks_per_eval": 3, "settle": 1, "c": 0.15, "a": 0.35, "decay": 0.6, "decay_iters": 6.0,
                 "block_z": 2.0, "exploit_frac": 0.25, "min_funds": 0.15}


class _Segment:
    """A setting held for `k` ticks; the mean reward of the ticks after the first `settle`."""
    __slots__ = ("tag", "x", "k", "settle", "n", "vals")

    def __init__(self, tag, x: np.ndarray, k: int, settle: int):
        self.tag, self.x, self.k, self.settle = tag, x, int(k), int(settle)
        self.n = 0
        self.vals: list[float] = []

    def add(self, r: float) -> None:
        self.n += 1
        if self.n > min(self.settle, self.k - 1):
            self.vals.append(float(r))

    @property
    def done(self) -> bool:
        return self.n >= self.k

    @property
    def mean(self) -> float:
        return float(np.mean(self.vals)) if self.vals else 0.0

    @property
    def sd(self) -> float:
        return float(np.std(self.vals, ddof=1)) if len(self.vals) > 1 else 0.0


class _Slot:
    """Shared plumbing of one environment slot: parse the observation, learn the horizon, guard the balance."""

    def __init__(self, p: dict, spec: EnvSpec, rng: np.random.Generator):
        self.p, self.spec, self.rng = p, spec, rng
        self.t = 0
        self.h = 120                       # replaced by the true number of ticks after the first tick
        self.clock0 = 0.0
        self.x0: np.ndarray | None = None
        self.levers: np.ndarray | None = None
        self.seg: _Segment | None = None
        self.queue: list[_Segment] = []
        self.incumbent: np.ndarray | None = None
        self.safe = False

    def step(self, o: np.ndarray) -> np.ndarray:
        sp = self.spec
        sc = o[sp.scalars]
        r, funds, clock = float(sc[0]), float(sc[3]), float(sc[4])
        if self.t == 0:
            self.x0 = np.asarray(o[sp.last_action], dtype=np.float64).copy()
            self.levers = np.flatnonzero(o[sp.lever_mask] > 0.5)
            self.incumbent = self.x0.copy()
            self.clock0 = clock
            self.start()
        else:
            if self.t == 1 and clock > self.clock0:
                # ticks left when this context began (the whole horizon unless the context was reset in mid-episode)
                self.h = max(2, int(round((1.0 - self.clock0) / (clock - self.clock0))))
                self.horizon_known()
            if self.seg is not None:
                self.seg.add(r)
                if self.seg.done:
                    seg, self.seg = self.seg, None
                    self.finished(seg)
            else:
                self.holding(r)
            if not self.safe and funds < self.p["min_funds"]:
                # nearly out of money: stop experimenting for good, rest at the starting settings
                self.safe, self.seg, self.queue, self.incumbent = True, None, [], self.x0.copy()
        if self.seg is None and self.queue and not self.safe:
            self.seg = self.queue.pop(0)
        x = self.seg.x if self.seg is not None else self.incumbent
        self.t += 1
        return np.asarray(x, dtype=np.float32)

    # hooks
    def start(self) -> None: ...
    def horizon_known(self) -> None: ...
    def finished(self, seg: _Segment) -> None: ...
    def holding(self, r: float) -> None: ...


class _ProbeSlot(_Slot):
    def start(self) -> None:
        self.k = int(self.p["ticks_per_arm"])
        self.hold_vals: list[float] = []
        self.ref: tuple[float, float] | None = None
        self.probes = 0
        self.begin(self.x0)

    def arms(self, base: np.ndarray, j: int) -> list[float]:
        if self.p["arms"] == "range":           # the two of (bottom, middle, top) furthest from the baseline
            pts = sorted((-1.0, 0.0, 1.0), key=lambda v: -abs(v - base[j]))[:2]
            return [float(v) for v in pts if abs(v - base[j]) > 1e-6]
        d = float(self.p["delta"])
        out = []
        for v in (base[j] + d, base[j] - d):
            v = float(np.clip(v, -1.0, 1.0))
            if abs(v - base[j]) > 1e-6:
                out.append(v)
        if len(out) == 1:                       # at a bound: a second arm further inward
            v = float(np.clip(base[j] + 2 * (out[0] - base[j]), -1.0, 1.0))
            if abs(v - out[0]) > 1e-6:
                out.append(v)
        return out

    def best_of(self, j: int, b: float, res: list[tuple[float, float]]) -> tuple[float, float]:
        """(estimated net, value) for lever j: the best measured arm or, with `interpolate`, the maximum of the
        parabola through the baseline and the two arms, inside the range the arms span."""
        m, v = max(res)
        if self.p["interpolate"] and len(res) == 2:
            xs = np.array([self.base[j], res[0][1], res[1][1]])
            ys = np.array([b, res[0][0], res[1][0]])
            if len(set(np.round(xs, 6))) == 3:
                c = np.polyfit(xs, ys, 2)
                grid = np.linspace(xs.min(), xs.max(), 41)
                fit = np.polyval(c, grid)
                k = int(np.argmax(fit))
                if fit[k] > m:
                    return float(fit[k]), float(grid[k])
        return float(m), float(v)

    def n_segments(self) -> int:
        if self.p["arms"] == "line":
            return 1 + 2 * len(self.levers)
        return 2 + sum(len(self.arms(self.x0, int(j))) for j in self.levers)

    def begin(self, base: np.ndarray) -> None:
        self.base = np.asarray(base, dtype=np.float64).copy()
        self.stage, self.results, self.base_seg = "arms", {}, None
        self.probes += 1
        s = int(self.p["settle"])
        self.queue = [_Segment(("base",), self.base, self.k, s)]
        self.t_begin = self.t
        if self.p["arms"] == "line":
            self.line = {"todo": [int(j) for j in self.levers], "j": None, "dir": 1.0, "step": 0.0, "won": False,
                         "flipped": False}
            return
        for j in self.levers:
            for v in self.arms(self.base, int(j)):
                x = self.base.copy()
                x[j] = v
                self.queue.append(_Segment(("arm", int(j), v), x, self.k, s))

    def horizon_known(self) -> None:
        fit = int(self.p["max_probe_frac"] * self.h / max(1, self.n_segments()))
        self.k = int(np.clip(fit, 2, int(self.p["ticks_per_arm"])))
        for seg in self.queue:
            seg.k = self.k
        if self.seg is not None:
            self.seg.k = max(self.seg.n, self.k)

    def line_search(self, seg: _Segment) -> None:
        """`arms="line"`: one lever at a time, a step of `delta` up (down if that fails); a step that beats the
        running level is kept at once and followed by a step twice as long the same way, until one fails or the
        range ends. Kept steps are already the better settings, failed steps are small."""
        ln, d0 = self.line, float(self.p["delta"])
        if seg.tag[0] == "base":
            self.base_seg, self.level, self.level_sd = seg, seg.mean, seg.sd
        else:
            j, v = seg.tag[1], seg.tag[2]
            tol = float(self.p["accept_z"]) * self.level_sd * np.sqrt(2.0 / max(1, len(seg.vals)))
            if seg.mean > self.level + tol:
                self.base[j], self.level, ln["won"] = v, seg.mean, True
                ln["step"] *= 2.0
            elif not ln["won"] and not ln["flipped"]:
                ln["dir"], ln["step"], ln["flipped"] = -ln["dir"], d0, True
            else:
                ln["j"] = None
        while self.t - self.t_begin < float(self.p["max_probe_frac"]) * self.h:
            if ln["j"] is None:
                if not ln["todo"]:
                    break
                j = ln["todo"].pop(0)
                up = self.base[j] < 1.0 - 1e-6
                ln.update(j=j, step=d0, won=False, dir=1.0 if up else -1.0, flipped=not up)
            j = ln["j"]
            v = float(np.clip(self.base[j] + ln["dir"] * ln["step"], -1.0, 1.0))
            if abs(v - self.base[j]) < 1e-6:                    # the range ends here
                if not ln["won"] and not ln["flipped"]:
                    ln["dir"], ln["step"], ln["flipped"] = -ln["dir"], d0, True
                else:
                    ln["j"] = None
                continue
            x = self.base.copy()
            x[j] = v
            self.queue = [_Segment(("arm", j, v), x, self.k, int(self.p["settle"]))]
            return
        self.commit(self.base)

    def finished(self, seg: _Segment) -> None:
        if self.p["arms"] == "line":
            return self.line_search(seg)
        if seg.tag[0] == "base":
            self.base_seg = seg
        elif seg.tag[0] == "arm":
            self.results.setdefault(seg.tag[1], []).append((seg.mean, seg.tag[2]))
        if self.queue:
            return
        b = self.base_seg.mean
        if self.stage == "arms":
            tol = float(self.p["accept_z"]) * self.base_seg.sd * np.sqrt(2.0 / max(1, len(self.base_seg.vals)))
            combo, gains = self.base.copy(), []
            measured = []
            for j, res in sorted(self.results.items()):
                m, v = self.best_of(j, b, res)
                if m - b > tol:
                    combo[j] = v
                    gains.append((m - b, j, v))
                m, v = max(res)
                if m - b > tol:
                    measured.append((m - b, j, v))
            self.single = None
            if measured:
                g, j, v = max(measured)
                self.single = self.base.copy()
                self.single[j] = v
            if len(gains) >= 2 or (gains and self.p["interpolate"]):
                self.stage = "verify"
                self.queue = [_Segment(("combo",), combo, self.k, int(self.p["settle"]))]
                return
            self.commit(self.single if measured else self.base)
        else:                                   # the combination was run: keep it only if it beat the baseline
            self.commit(seg.x if seg.mean > b else (self.single if self.single is not None else self.base))

    def commit(self, x: np.ndarray) -> None:
        self.incumbent = np.asarray(x, dtype=np.float64).copy()
        self.stage, self.hold_vals, self.ref = "hold", [], None

    def holding(self, r: float) -> None:
        if self.safe or not self.p["reprobe"]:
            return
        w = int(self.p["window"])
        self.hold_vals.append(r)
        if self.ref is None:
            if len(self.hold_vals) >= w + 1:    # the first tick after a change of settings is not counted
                v = self.hold_vals[1:w + 1]
                self.ref = (float(np.mean(v)), float(np.std(v, ddof=1)))
            return
        ref, sd = self.ref
        gain = max(0.0, ref - self.base_seg.mean) if self.base_seg is not None else 0.0
        drop = max(float(self.p["collapse_frac"]) * gain, 4.0 * sd / np.sqrt(w), float(self.p["min_drop"]))
        if float(np.mean(self.hold_vals[-w:])) < ref - drop and len(self.hold_vals) >= 2 * w + 1:
            need = self.k * (self.n_segments() + 1)
            if self.h - self.t >= 1.5 * need:
                self.begin(self.incumbent)


class _HillSlot(_Slot):
    def start(self) -> None:
        self.k = int(self.p["ticks_per_eval"])
        self.x = self.x0.copy()
        self.it, self.v, self.prev = 0, None, None
        self.best = (-np.inf, self.x0.copy())
        self.noise: list[float] = []
        self.queue = [_Segment(("base",), self.x0.copy(), self.k, int(self.p["settle"]))]

    def pair(self) -> None:
        c = float(self.p["c"])
        d = np.zeros_like(self.x)
        d[self.levers] = self.rng.choice([-1.0, 1.0], size=len(self.levers))
        xp, xm = np.clip(self.x + c * d, -1.0, 1.0), np.clip(self.x - c * d, -1.0, 1.0)
        s = int(self.p["settle"])
        self.queue = [_Segment(("plus",), xp, self.k, s), _Segment(("minus",), xm, self.k, s)]

    def finished(self, seg: _Segment) -> None:
        if seg.sd > 0:
            self.noise.append(seg.sd)
        if seg.mean > self.best[0]:
            self.best = (seg.mean, seg.x.copy())
        if self.t >= (1.0 - float(self.p["exploit_frac"])) * self.h:
            self.incumbent, self.queue = self.best[1].copy(), []
            return
        if seg.tag[0] == "base":
            self.pair()
        elif seg.tag[0] == "plus":
            self.plus = seg
        else:
            yp, ym, xp, xm = self.plus.mean, seg.mean, self.plus.x, seg.x
            avg, diff = 0.5 * (yp + ym), yp - ym
            se = (float(np.median(self.noise)) if self.noise else 0.0) / np.sqrt(max(1, len(seg.vals)))
            if self.prev is not None and avg < self.prev[0] - float(self.p["block_z"]) * se:
                self.x = self.prev[1].copy()            # blocking: the last step made things clearly worse
                self.it += 2                            # and the gain shrinks faster
            else:
                self.prev = (avg, self.x.copy())
                self.v = diff * diff if self.v is None else 0.7 * self.v + 0.3 * diff * diff
                step = float(np.clip(diff / np.sqrt(self.v + 1e-12), -2.0, 2.0))
                gain = float(self.p["a"]) / (1.0 + self.it / float(self.p["decay_iters"])) ** float(self.p["decay"])
                self.x = np.clip(self.x + gain * step * (xp - xm) / (2.0 * float(self.p["c"])), -1.0, 1.0)
            self.it += 1
            self.pair()


class _ScriptedLearner(Learner):
    kind = "scripted"
    _slot_cls: type = _Slot
    _defaults: dict = {}

    def __init__(self, spec: EnvSpec, seed: int = 0, device: str = "cpu", **params):
        super().__init__(spec, seed, device)
        bad = sorted(set(params) - set(self._defaults))
        if bad:
            raise TypeError(f"{self.name} has no option(s) {bad}")
        self.params = dict(self._defaults, **params)
        self._slots: list[_Slot] | None = None

    def _new(self, i: int) -> _Slot:
        # every episode of slot i starts from the same generator state: an evaluation is reproducible whatever
        # ran before it
        return self._slot_cls(self.params, self.spec, np.random.default_rng([self.seed, 7919, i]))

    def reset_context(self, n: int, which: np.ndarray | None = None) -> None:
        if self._slots is None or len(self._slots) != n or which is None:
            self._slots = [self._new(i) for i in range(n)]
        else:
            w = np.asarray(which)
            for i in (np.flatnonzero(w) if w.dtype == bool else w):
                self._slots[int(i)] = self._new(int(i))

    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        c0 = time.process_time()
        n = obs.shape[0]
        if self._slots is None or len(self._slots) != n:
            self.reset_context(n)
        out = np.zeros((n, self.spec.act_dim), dtype=np.float32)
        for i in range(n):
            out[i] = self._slots[i].step(obs[i])
        self._usage.infer_cpu_s += time.process_time() - c0
        return out

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"learner": self.name, "seed": self.seed, "params": self.params}, fh)

    def load(self, path: str) -> None:
        with open(path) as fh:
            d = json.load(fh)
        self.seed = int(d.get("seed", self.seed))
        self.params = dict(self._defaults, **d.get("params", {}))
        self._slots = None


class ProbeExploitLearner(_ScriptedLearner):
    name = "probe_exploit"
    _slot_cls, _defaults = _ProbeSlot, PROBE_DEFAULTS


class HillClimbLearner(_ScriptedLearner):
    name = "hill_climb"
    _slot_cls, _defaults = _HillSlot, HILL_DEFAULTS
