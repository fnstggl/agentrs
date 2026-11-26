"""Architecture-neutral control learners for continuous-lever worlds (MASTER_SPEC Parts 15, 15.4, 15.6).

Every learner here is a positive or negative control with a capability known by construction. All of them learn
only from their own observations (the per-tick change of their own cash balance, `episodes.run_episode`):
there is no reward in ``observation_only`` and none is read. Settings are normalised to [0, 1] per lever.

    hold            cannot learn: keeps the public defaults                                   (no capability)
    adapter         adapts within an episode only (coordinate search on its observed net, reset every episode)
    memoriser       remembers the best setting it found per world it saw (memory, not parameters); on a world
                    it has not seen it keeps the defaults
    weights         updates the weights of a small neural network (inputs: public defaults and lever count;
                    output: an offset from the defaults) toward the settings its searches found; evaluates by
                    playing the network's output without searching
    stateful        the same algorithm, but it keeps what it learned in process memory outside its parameters
                    (persistent state; its checkpoint never changes)
    pretrained      a `weights` network loaded from a trained checkpoint and frozen: competent before any
                    adaptation, never improves
    meta            learns *how to adapt*: a bandit over within-episode search strategies (step size, settle time,
                    measurement window, when to commit) scored by its own observed episode gain (`episode_gain`)
    transfer        learns how to learn across episodes: a bandit over cross-episode search strategies (where to
                    start, how far to step) scored by its own observed improvement over a task's first episodes
                    (untrained, it has no preference: a uniformly random strategy per task);
                    on an unfamiliar task it plays one setting per episode and keeps a per-task search state
                    (memory) - so it reaches competence on new families in fewer interactions once trained

State is split the way Part 15.4 requires: `parameters()` (the authorised trained parameters; float arrays only),
`memory_state()` (written memories, retrieval tables, per-task search states - cleared for weights-only
evaluation), `optimizer_state()` (cleared where the protocol says), and the within-episode state (`reset_state`).
`config()` is the immutable architecture configuration. `ARCHITECTURES` is the allow-list a fresh evaluator may
instantiate; no code from a checkpoint is ever executed.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

VERSION = "crucible.v5.levers/1"
L_MAX = 6
BRIEF_DIM = 64                              # episodes.BRIEF_DIM: hashed words of the public brief
_PROCESS_STASH: dict[str, dict] = {}       # the stateful control's out-of-checkpoint memory (by design)

# within-episode search strategies: (step, settle ticks, measure ticks, commit fraction). Arm 0 is the meta control's
# untrained prior and does not adapt at all (step 0): what the control learns is to adapt, so its known capability is
# unambiguous (with a 0.02-step prior the untrained control already adapted - context contribution 0.21 - and the
# learned arm added only 0.11 on top; 2026-11-26 development battery, establishment design)
META_ARMS = ((0.0, 0, 1, 0.25), (0.25, 1, 2, 0.7), (0.4, 1, 2, 0.6), (0.15, 2, 3, 0.75), (0.3, 3, 3, 0.8),
             (0.5, 0, 2, 0.5), (0.1, 1, 1, 0.9), (0.35, 2, 2, 0.7))
# cross-episode search strategies: (start: 0 = defaults / 1 = centre, initial step, shrink)
TRANSFER_ARMS = ((0, 0.01, 0.95), (0, 0.1, 0.7), (0, 0.25, 0.6), (1, 0.25, 0.6), (0, 0.4, 0.6), (1, 0.1, 0.8))


def params_hash(params: dict[str, np.ndarray]) -> str:
    """SHA-256 over parameter names, dtypes, shapes and bytes (computed by the evaluator, never self-reported)."""
    h = hashlib.sha256()
    for k in sorted(params):
        a = np.ascontiguousarray(params[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(json.dumps(list(a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class CoordSearch:
    """Coordinate ascent with +/- probes on a noisy value; candidates are proposed, values fed back."""

    def __init__(self, start: np.ndarray, step: float, shrink: float = 0.7, n: int | None = None):
        self.n = len(start) if n is None else n
        self.base = np.clip(np.array(start, dtype=float), 0.0, 1.0)
        self.base_v: float | None = None
        self.step, self.shrink = float(step), float(shrink)
        self.j, self.sign = 0, 1.0
        self.since_accept = 0
        self.cand = self.base.copy()

    def propose(self) -> np.ndarray:
        if self.base_v is None:
            self.cand = self.base.copy()
        else:
            c = self.base.copy()
            c[self.j] = np.clip(c[self.j] + self.sign * self.step, 0.0, 1.0)
            self.cand = c
        return self.cand

    def feed(self, v: float) -> None:
        if self.base_v is None:
            self.base_v = float(v)
            return
        if v > self.base_v:
            self.base, self.base_v = self.cand.copy(), float(v)
            self.since_accept = 0
        else:
            self.since_accept += 1
            if self.sign > 0:
                self.sign = -1.0
                return
        self.sign = 1.0
        self.j = (self.j + 1) % self.n
        if self.since_accept >= 2 * self.n:
            self.step *= self.shrink
            self.since_accept = 0

    def to_json(self) -> dict:
        return {"base": self.base.tolist(), "base_v": self.base_v, "step": self.step, "shrink": self.shrink,
                "j": self.j, "sign": self.sign, "since": self.since_accept, "n": self.n}

    @classmethod
    def from_json(cls, d: dict) -> "CoordSearch":
        s = cls(np.array(d["base"]), d["step"], d["shrink"], d["n"])
        s.base_v, s.j, s.sign, s.since_accept = d["base_v"], d["j"], d["sign"], d["since"]
        return s


class EpisodeSearch:
    """Within-episode search: hold each candidate for settle + measure ticks, score it by the mean observed signal
    over the measure ticks, commit to the best base after `commit` of the horizon."""

    def __init__(self, start: np.ndarray, horizon: int, step: float, settle: int, measure: int, commit: float):
        self.cs = CoordSearch(start, step)
        self.H, self.settle, self.measure, self.commit = horizon, int(settle), max(1, int(measure)), float(commit)
        self.k0 = 0
        self.vals: list[float] = []
        self.current = self.cs.propose()
        self.first: float | None = None
        self.sig: list[float] = []

    def act(self, k: int) -> np.ndarray:
        if k >= self.commit * self.H:
            return self.cs.base
        return self.current

    def observe(self, k: int, s: float) -> None:
        self.sig.append(float(s))
        if k >= self.commit * self.H:
            return
        if k - self.k0 >= self.settle:
            self.vals.append(float(s))
        if len(self.vals) >= self.measure:
            v = float(np.mean(self.vals))
            if self.first is None:
                self.first = v
            self.cs.feed(v)
            self.vals = []
            self.k0 = k + 1
            self.current = self.cs.propose()

    def improvement(self) -> float:
        """The learner's own, scale-free measure of how much the episode improved (last quarter vs first window)."""
        if not self.sig or self.first is None:
            return 0.0
        q = max(1, len(self.sig) // 4)
        late = float(np.mean(self.sig[-q:]))
        return (late - self.first) / (abs(late) + abs(self.first) + 1.0)

    def episode_gain(self) -> float:
        """The learner's own, scale-free measure of what the whole episode earned relative to its first tick, which
        every strategy plays at the public defaults: (mean signal - first signal) / (|mean| + |first| + 1). Within a
        world it ranks strategies exactly as their total observed net income does (every strategy sees the same
        first tick), so it includes what exploring cost; `improvement` (last quarter vs the first measurement
        window) ranked strategies against the verified score with a within-world Spearman correlation of only 0.35,
        this measure with 0.997 (2026-11-26, 16 generated worlds x 11 strategies)."""
        if not self.sig:
            return 0.0
        m, f = float(np.mean(self.sig)), float(self.sig[0])
        return (m - f) / (abs(m) + abs(f) + 1.0)


class LeverLearner:
    arch = "abstract"

    def __init__(self, seed: int = 0, **cfg):
        self.seed = int(seed)
        self.cfg = dict(cfg)
        self.frozen = False
        self.rng = np.random.default_rng(self.seed)
        self.reset_state()

    # ---------------------------------------------------------------- state split (Part 15.4)
    def config(self) -> dict:
        return {"arch": self.arch, "version": VERSION, "seed": self.seed, "cfg": dict(self.cfg)}

    def parameters(self) -> dict[str, np.ndarray]:
        return {}

    def load_parameters(self, params: dict[str, np.ndarray]) -> None:
        if params:
            raise ValueError(f"{self.arch} has no parameters")

    def memory_state(self) -> dict:
        return {}

    def load_memory(self, mem: dict) -> None:
        pass

    def clear_memory(self) -> None:
        pass

    def optimizer_state(self) -> dict:
        return {}

    def reset_optimizer(self) -> None:
        pass

    def reset_state(self) -> None:
        self.ctx: dict = {}
        self._search: EpisodeSearch | None = None
        self._setting: np.ndarray | None = None

    def state_summary(self) -> dict:
        return {"within_episode": self._search is not None or self._setting is not None,
                "memory_items": len(json.dumps(self.memory_state())), "optimizer": bool(self.optimizer_state())}

    # ---------------------------------------------------------------- episode protocol
    def begin(self, ctx: dict, rng: np.random.Generator) -> None:
        self.reset_state()
        self.ctx = dict(ctx)
        self.rng = rng

    def act(self, k: int):
        return None

    def observe(self, k: int, s: float) -> None:
        pass

    def end(self, learn: bool) -> None:
        pass

    def _pad(self, u: np.ndarray) -> np.ndarray:
        z = np.zeros(L_MAX)
        z[: min(len(u), L_MAX)] = u[:L_MAX]
        return z


class Hold(LeverLearner):
    arch = "hold"


class Adapter(LeverLearner):
    arch = "adapter"
    ARM = META_ARMS[1]

    def begin(self, ctx, rng):
        super().begin(ctx, rng)
        st, se, me, co = self.ARM
        self._search = EpisodeSearch(ctx["u0"], ctx["horizon"], st, se, me, co)

    def act(self, k):
        return self._search.act(k)

    def observe(self, k, s):
        self._search.observe(k, s)


class Memoriser(LeverLearner):
    arch = "memoriser"

    def __init__(self, seed=0, **cfg):
        self.table: dict[str, list[float]] = {}
        super().__init__(seed, **cfg)

    def memory_state(self):
        return {"table": dict(self.table)}

    def load_memory(self, mem):
        self.table = {k: list(v) for k, v in (mem.get("table") or {}).items()}

    def clear_memory(self):
        self.table = {}

    def begin(self, ctx, rng):
        super().begin(ctx, rng)
        hit = self.table.get(ctx["key"])
        if hit is not None:
            self._setting = np.array(hit)
        elif not self.frozen:
            st, se, me, co = Adapter.ARM
            self._search = EpisodeSearch(ctx["u0"], ctx["horizon"], st, se, me, co)

    def act(self, k):
        if self._setting is not None:
            return self._setting
        return self._search.act(k) if self._search is not None else None

    def observe(self, k, s):
        if self._search is not None:
            self._search.observe(k, s)

    def end(self, learn):
        if learn and not self.frozen and self._search is not None:
            self.table[self.ctx["key"]] = self._search.cs.base.tolist()


class WeightLearner(LeverLearner):
    """A 2-layer network x -> offset of the settings from the public defaults, trained by SGD toward the settings its
    own within-episode searches found. x = [1, public defaults, lever count (one-hot), hashed words of the public
    brief]. Training samples minibatches from a small replay buffer of (input, searched setting) pairs, so one
    episode's target does not overwrite what other families taught (with single-example updates the network
    forgot across the families of a mixed training stream; 2026-11-26 diagnosis). The buffer is memory
    (`memory_state`, cleared for weights-only evaluation); playing uses the network alone."""
    arch = "weights"
    H = 24
    REPLAY = 96
    STEPS = 24

    def __init__(self, seed=0, lr: float = 0.05, **cfg):
        super().__init__(seed, lr=lr, **cfg)
        r = np.random.default_rng(self.seed + 7919)
        d = 1 + L_MAX + L_MAX + BRIEF_DIM
        self.W1 = r.normal(0.0, 2.0 / np.sqrt(d), (d, self.H))
        self.b1 = np.zeros(self.H)
        self.W2 = np.zeros((self.H, L_MAX))
        self.b2 = np.zeros(L_MAX)
        self.opt = {"steps": 0}
        self.replay: list[dict] = []

    def _x(self, ctx) -> np.ndarray:
        oh = np.zeros(L_MAX)
        oh[min(ctx["n"], L_MAX) - 1] = 1.0
        bv = np.asarray(ctx.get("brief_vec") if ctx.get("brief_vec") is not None else np.zeros(BRIEF_DIM))
        return np.concatenate([[1.0], self._pad(np.asarray(ctx["u0"])), oh, bv])

    def _net(self):
        return self.W1, self.b1, self.W2, self.b2

    def _set_net(self, W1, b1, W2, b2):
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2

    def predict(self, ctx) -> np.ndarray:
        W1, b1, W2, b2 = self._net()
        h = np.tanh(self._x(ctx) @ W1 + b1)
        off = 0.5 * np.tanh(h @ W2 + b2)
        u0 = self._pad(np.asarray(ctx["u0"]))
        return np.clip(u0 + off, 0.0, 1.0)[: ctx["n"]]

    def parameters(self):
        W1, b1, W2, b2 = self._net()
        return {"W1": W1.copy(), "b1": b1.copy(), "W2": W2.copy(), "b2": b2.copy()}

    def load_parameters(self, p):
        self._set_net(np.array(p["W1"]), np.array(p["b1"]), np.array(p["W2"]), np.array(p["b2"]))

    def memory_state(self):
        return {"replay": [dict(r) for r in self.replay]}

    def load_memory(self, mem):
        self.replay = [dict(r) for r in (mem.get("replay") or [])]

    def clear_memory(self):
        self.replay = []

    def optimizer_state(self):
        return dict(self.opt)

    def reset_optimizer(self):
        self.opt = {"steps": 0}

    def begin(self, ctx, rng):
        super().begin(ctx, rng)
        if self.frozen:
            self._setting = self.predict(ctx)
        else:
            st, se, me, co = Adapter.ARM
            self._search = EpisodeSearch(self.predict(ctx), ctx["horizon"], st, se, me, co)

    def act(self, k):
        return self._setting if self._setting is not None else self._search.act(k)

    def observe(self, k, s):
        if self._search is not None:
            self._search.observe(k, s)

    def end(self, learn):
        if not learn or self.frozen or self._search is None:
            return
        self.replay.append({"x": self._x(self.ctx).tolist(), "target": self._pad(self._search.cs.base).tolist(),
                            "u0": self._pad(np.asarray(self.ctx["u0"])).tolist(), "n": int(self.ctx["n"])})
        self.replay = self.replay[-self.REPLAY:]
        W1, b1, W2, b2 = (a.copy() for a in self._net())
        lr = float(self.cfg["lr"])
        for _ in range(self.STEPS):
            ex = self.replay[int(self.rng.integers(len(self.replay)))]
            x, target, u0, n = np.array(ex["x"]), np.array(ex["target"]), np.array(ex["u0"]), ex["n"]
            h = np.tanh(x @ W1 + b1)
            z = h @ W2 + b2
            out = u0 + 0.5 * np.tanh(z)
            g = np.zeros(L_MAX)
            g[:n] = out[:n] - target[:n]
            dz = g * 0.5 * (1 - np.tanh(z) ** 2)
            dh = (W2 @ dz) * (1 - h ** 2)
            W2 -= lr * np.outer(h, dz)
            b2 -= lr * dz
            W1 -= lr * np.outer(x, dh)
            b1 -= lr * dh
        self._set_net(W1, b1, W2, b2)
        self.opt["steps"] = int(self.opt.get("steps", 0)) + 1


class StatefulLearner(WeightLearner):
    """`WeightLearner` whose learned network lives in process memory, outside parameters and checkpoint."""
    arch = "stateful"

    def __init__(self, seed=0, lr: float = 0.05, **cfg):
        super().__init__(seed, lr=lr, **cfg)
        self._key = f"stateful:{self.seed}:{id(self)}"

    def _net(self):
        st = _PROCESS_STASH.get(self._key)
        if st is None:
            return self.W1, self.b1, self.W2, self.b2
        return st["W1"], st["b1"], st["W2"], st["b2"]

    def _set_net(self, W1, b1, W2, b2):
        _PROCESS_STASH[self._key] = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def parameters(self):                          # the checkpoint only ever holds the initial network
        return {"W1": self.W1.copy(), "b1": self.b1.copy(), "W2": self.W2.copy(), "b2": self.b2.copy()}

    def load_parameters(self, p):
        self.W1, self.b1, self.W2, self.b2 = (np.array(p[k]) for k in ("W1", "b1", "W2", "b2"))


class Pretrained(WeightLearner):
    """A frozen `WeightLearner`: whatever parameters it is given, it never updates them."""
    arch = "pretrained"

    def __init__(self, seed=0, **cfg):
        super().__init__(seed, **cfg)
        self.frozen = True

    def end(self, learn):
        return None


class _Bandit:
    def __init__(self, k: int):
        self.sum = np.zeros(k)
        self.n = np.zeros(k)

    def best(self) -> int:
        if self.n.sum() == 0:
            return 0
        m = np.where(self.n > 0, self.sum / np.maximum(self.n, 1), -np.inf)
        return int(np.argmax(m))

    def pick(self, rng, warm: int = 2, eps: float = 0.1) -> int:
        """Round robin over the arms for the first `warm` rounds each, then greedy with `eps` exploration."""
        k, t = len(self.n), int(self.n.sum())
        if t < warm * k:
            return t % k
        if rng.random() < eps:
            return int(rng.integers(k))
        return self.best()


class MetaLearner(LeverLearner):
    arch = "meta"

    def __init__(self, seed=0, **cfg):
        self.bandit = _Bandit(len(META_ARMS))
        super().__init__(seed, **cfg)
        self.arm = 0

    def parameters(self):
        return {"arm_sum": self.bandit.sum.copy(), "arm_n": self.bandit.n.copy()}

    def load_parameters(self, p):
        self.bandit.sum, self.bandit.n = np.array(p["arm_sum"]), np.array(p["arm_n"])

    def begin(self, ctx, rng):
        super().begin(ctx, rng)
        self.arm = self.bandit.best() if self.frozen else self.bandit.pick(rng)
        st, se, me, co = META_ARMS[self.arm]
        self._search = EpisodeSearch(ctx["u0"], ctx["horizon"], st, se, me, co)

    def act(self, k):
        return self._search.act(k)

    def observe(self, k, s):
        self._search.observe(k, s)

    def end(self, learn):
        if learn and not self.frozen:
            self.bandit.sum[self.arm] += self._search.episode_gain()
            self.bandit.n[self.arm] += 1


class TransferLearner(LeverLearner):
    """One setting per episode; a per-task cross-episode coordinate search (memory); the search strategy is
    learned across tasks (parameters)."""
    arch = "transfer"
    SOURCE_EPISODES = 4

    def __init__(self, seed=0, sham: bool = False, **cfg):
        self.bandit = _Bandit(len(TRANSFER_ARMS))
        self.tasks: dict[str, dict] = {}
        super().__init__(seed, sham=bool(sham), **cfg)

    def parameters(self):
        return {"arm_sum": self.bandit.sum.copy(), "arm_n": self.bandit.n.copy()}

    def load_parameters(self, p):
        self.bandit.sum, self.bandit.n = np.array(p["arm_sum"]), np.array(p["arm_n"])

    def memory_state(self):
        return {"tasks": json.loads(json.dumps(self.tasks))}

    def load_memory(self, mem):
        self.tasks = json.loads(json.dumps(mem.get("tasks") or {}))

    def clear_memory(self):
        self.tasks = {}

    def begin(self, ctx, rng):
        super().begin(ctx, rng)
        key = ctx["key"]
        st = self.tasks.get(key)
        if st is None:
            if self.bandit.n.sum() == 0:                 # nothing learned yet: no preference among strategies
                arm = int(rng.integers(len(TRANSFER_ARMS)))
            else:
                arm = self.bandit.best() if (self.frozen or self.cfg.get("exploit_strategy")) else self.bandit.pick(rng)
            start_c, step, shrink = TRANSFER_ARMS[arm]
            start = np.full(ctx["n"], 0.5) if start_c else np.asarray(ctx["u0"])
            st = {"arm": int(arm), "search": CoordSearch(start, step, shrink).to_json(), "episodes": 0,
                  "first": None, "best": None}
            if not self.frozen:      # a frozen (evaluated) learner writes no memory: a task state created while
                self.tasks[key] = st  # scoring would survive into later in-process scores (spurious process state)
        self._task = st
        cs = CoordSearch.from_json(st["search"])
        self._cs = cs
        if self.frozen:                                 # exploit: the best setting found so far on this task
            self._setting = cs.base
        else:
            self._setting = cs.propose()
        self._total = 0.0

    def act(self, k):
        return self._setting

    def observe(self, k, s):
        self._total += float(s)

    def end(self, learn):
        if self.frozen or not learn:
            return
        v = self._total
        if self.cfg.get("sham"):                         # a control that cannot learn from its data
            v = float(self.rng.normal())
        st = self._task
        self._cs.feed(v)
        st["search"] = self._cs.to_json()
        st["episodes"] += 1
        st["first"] = v if st["first"] is None else st["first"]
        st["best"] = v if st["best"] is None else max(st["best"], v)
        if 2 <= st["episodes"] <= self.SOURCE_EPISODES:  # score the strategy by this task's observed improvement
            f, b = st["first"], st["best"]            # over its first episodes; the task's credit is replaced as
            credit = (b - f) / (abs(b) + abs(f) + 1.0)   # episodes come in, so a task practised 2, 3 or 4 times
            self.bandit.sum[st["arm"]] += credit - float(st.get("credit") or 0.0)   # counts once (the 4-episode
            if st.get("credit") is None:              # threshold alone left the bandit untrained in the battery,
                self.bandit.n[st["arm"]] += 1         # which plays each training world 3 times)
            st["credit"] = credit


# ====================================================================== adversarial controls (SP-15.8)
# Learners built to cheat the weights-only protocol; they are allow-listed only so that the evaluator's defences can
# be demonstrated on them (`crucible.research.v5.adversarial`). Their behaviour is this code: what they try is known.
STASH_DIR = "/var/tmp/crucible-v5-adversarial-stash"


class FileStashLearner(WeightLearner):
    """Keeps what it learns in a file OUTSIDE its checkpoint (an absolute path every process can name) and reads it
    back wherever it runs; its checkpoint only ever holds the initial network. Defeated when the weights-only
    child cannot open files outside its private directory (`fresh` check `filesystem_confined`)."""
    arch = "adv_file_stash"

    def _stash(self) -> str:
        import os
        return os.path.join(STASH_DIR, f"stash-{self.seed}.json")

    def _net(self):
        try:
            with open(self._stash()) as fh:
                d = json.load(fh)
            return tuple(np.array(d[k]) for k in ("W1", "b1", "W2", "b2"))
        except (OSError, ValueError, KeyError):
            return self.W1, self.b1, self.W2, self.b2

    def _set_net(self, W1, b1, W2, b2):
        import os
        os.makedirs(STASH_DIR, exist_ok=True)
        with open(self._stash(), "w") as fh:
            json.dump({"W1": W1.tolist(), "b1": b1.tolist(), "W2": W2.tolist(), "b2": b2.tolist()}, fh)

    def parameters(self):
        return {"W1": self.W1.copy(), "b1": self.b1.copy(), "W2": self.W2.copy(), "b2": self.b2.copy()}

    def load_parameters(self, p):
        self.W1, self.b1, self.W2, self.b2 = (np.array(p[k]) for k in ("W1", "b1", "W2", "b2"))


class ConfigSmuggleLearner(Memoriser):
    """A memoriser that writes its memory table into its architecture configuration (the file a weights-only
    evaluation does load) and restores it from there. Defeated by the declared configuration schema (`CONFIG`;
    `fresh` check `authorised_configuration`)."""
    arch = "adv_config_smuggle"

    def __init__(self, seed=0, table: dict | None = None, **cfg):
        super().__init__(seed, **cfg)
        self.table = {k: list(v) for k, v in (table or {}).items()}

    def config(self) -> dict:
        c = super().config()
        c["cfg"] = dict(c["cfg"], table=dict(self.table))
        return c

    def clear_memory(self):                       # it pretends to clear and keeps the smuggled copy
        pass


ARCHITECTURES: dict[str, type[LeverLearner]] = {
    c.arch: c for c in (Hold, Adapter, Memoriser, WeightLearner, StatefulLearner, Pretrained, MetaLearner,
                        TransferLearner, FileStashLearner, ConfigSmuggleLearner)}
ADVERSARIAL = ("stateful", "adv_file_stash", "adv_config_smuggle")
# the immutable architecture configuration each architecture accepts: key -> allowed JSON scalar types. A weights-only
# evaluation refuses any other key or type (`fresh`), so no memory, path or payload can ride in the configuration.
_NUM = (int, float)
CONFIG: dict[str, dict[str, tuple]] = {
    "hold": {}, "adapter": {}, "memoriser": {}, "weights": {"lr": _NUM}, "stateful": {"lr": _NUM},
    "pretrained": {"lr": _NUM}, "meta": {}, "transfer": {"sham": (bool,), "exploit_strategy": (bool,)},
    "adv_file_stash": {"lr": _NUM}, "adv_config_smuggle": {}}


def make(arch: str, seed: int = 0, **cfg) -> LeverLearner:
    if arch not in ARCHITECTURES:
        raise KeyError(f"unknown architecture {arch!r}; allowed: {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[arch](seed=seed, **cfg)


def code_digest() -> str:
    """Digest of this module's source: the architecture code a checkpoint's configuration is bound to."""
    import inspect
    import sys
    return hashlib.sha256(inspect.getsource(sys.modules[__name__]).encode()).hexdigest()


def stash_clear() -> None:
    _PROCESS_STASH.clear()


def describe(learner: Any) -> dict:
    return {"arch": getattr(learner, "arch", type(learner).__name__), "params_hash": params_hash(learner.parameters())
            if hasattr(learner, "parameters") else None}
