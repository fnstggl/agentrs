"""Evaluation protocols: each one detects what it claims and nothing else.

The learners in this file each have exactly one kind of learning, so that a protocol can be shown to attribute a
gain to the right place:

    ContextLearner   adapts inside an episode (a search whose state lives in its context); no weights
    WeightLearner    a constant-settings policy improved by `update` (score-function gradient); no context
    Memoriser        stores the best action it found for each training task and replays it on recognition; nothing
                     it stores says anything about another task
    MetaLearner      an in-context climber whose step size is a weight: training changes only how fast it adapts
    HoldLearner      learns nothing (research.baselines)

Tasks are hand-written mechanisms with a known optimum (one useful lever with an interior peak, one lever that
only costs money), wrapped as `Task`s whose private reference is the true optimum, plus a few generated tasks.
"""
import ast
import json
import os
import stat
import sys
from dataclasses import dataclass, field

import numpy as np
import pytest

SCRATCH_CACHE = "/tmp/claude-0/-workspace-agentrs/bd8b0c90-42cb-4c33-8b8b-7360e51d82a1/scratchpad/cache"

from crucible.mechanisms import scm  # noqa: E402
from crucible.mechanisms.sim import MechSim, demand_income  # noqa: E402
from crucible.research import protocols as P  # noqa: E402
from crucible.research import stats as S  # noqa: E402
from crucible.research import tasks as T  # noqa: E402
from crucible.research.accounting import Cost, Meter  # noqa: E402
from crucible.research.baselines import HoldLearner, RandomLearner  # noqa: E402
from crucible.research.learner import EnvSpec, Learner  # noqa: E402

SPEC = EnvSpec()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_ENVS = 8
H = 40                       # ticks per hand-made episode
BATCH = N_ENVS * H           # environment steps per training batch


@pytest.fixture(scope="module", autouse=True)
def task_cache():
    """Validated tasks cost seconds each to build: keep them in the session's scratch cache while this module runs
    (unless a cache is already configured, or the scratch directory is gone - then the default cache is used)."""
    old = os.environ.get("CRUCIBLE_CACHE")
    if old is None and os.path.isdir(os.path.dirname(SCRATCH_CACHE)):
        os.environ["CRUCIBLE_CACHE"] = SCRATCH_CACHE
    yield
    if old is None:
        os.environ.pop("CRUCIBLE_CACHE", None)


# ====================================================================== hand-made tasks with a known optimum
@dataclass(frozen=True)
class HandTask(T.Task):
    mech: dict = field(default=None, compare=False, repr=False)

    def mechanism(self) -> dict:
        return self.mech


def peak_mech(peak: float, cost: float = 1500.0, horizon: int = H, flip_at: int | None = None) -> dict:
    """Lever `a` earns most at `peak` (inverted U); lever `c` only costs money. Defaults: both at 0. With
    `flip_at`, the response to `a` is inverted from that tick on (what was best becomes worst)."""
    term = {"kind": "invu", "src": "a", "w": 1.0, "peak": peak, "width": 0.6}
    if flip_at is not None:
        term["w2"] = -1.0
    return scm.handmade(levers=[{"name": "a"}, {"name": "c", "cost": cost}],
                        latents=[{"id": "x", "terms": [term]}],
                        outcome={"terms": [{"kind": "linear", "src": "x", "w": 1.0}], "bias": 1.0, "cap": 4.0},
                        horizon=horizon, regime_at=flip_at)


def hand_task(family: str, seed: int, mech: dict, reference, transfer: str = "", split: str = "") -> HandTask:
    """Wrap a hand-made mechanism as a task. `reference` is the known best policy (constant settings or a
    callable for `MechSim.run`); the private default and reference nets are measured on the task's own noise."""
    T._ensure()
    income, h = demand_income(mech), int(mech["config"]["horizon"])     # the one definition of a facility's income
    wseed = T.world_seed_of(family, seed)
    endow = T.ENDOWMENT_TICKS * income
    sim = MechSim(mech, wseed, "r1", income_cents_per_tick=income, funds=[endow], held=[True])
    d = float(sim.run(None, h)["net"][:, 0].mean())
    sim.reset()
    r = float(sim.run(reference, h)["net"][:, 0].mean())
    assert r > d
    return HandTask(family, seed, seed, wseed, income, endow, h, 1, True,
                    int(round(h * (d + T.GOAL_FRACTION * (r - d)))), d, r, None, "hand", (), "scm", transfer, split,
                    mech=mech)


def peak_tasks(family: str, seeds, lo: float, hi: float, transfer: str = "", cost: float = 1500.0) -> list[HandTask]:
    out = []
    for s in seeds:
        p = float(np.random.default_rng([sum(map(ord, family)), s]).uniform(lo, hi))
        out.append(hand_task(family, s, peak_mech(p, cost), {"a": p, "c": 0.0}, transfer))
    return out


@pytest.fixture(scope="module")
def fam_a():
    """Family A: the peak of the useful lever lies between 0.62 and 0.78. Disjoint training and evaluation seeds."""
    return {"train": peak_tasks("hand_a", range(0, 8), 0.62, 0.78), "eval": peak_tasks("hand_a", range(100, 112), 0.62, 0.78,
                                                                                       transfer="within")}


@pytest.fixture(scope="module")
def fam_b():
    """Family B: the peak lies between 0.17 and 0.23 - what is best in A earns far less than doing nothing here,
    and is further from B's optimum than the untouched defaults are."""
    return {"train": peak_tasks("hand_b", range(0, 8), 0.17, 0.23), "eval": peak_tasks("hand_b", range(100, 112), 0.17, 0.23,
                                                                                       transfer="structural")}


# ====================================================================== learners with one kind of learning each
def _real(obs, spec):
    return obs[:, spec.lever_mask] > 0.5


class ContextLearner(Learner):
    """Coordinate search inside the episode: tries a grid on one lever at a time, keeps the best, then stays.
    Everything it knows is in its context; `reset_context` returns it to knowing nothing. No weights."""
    name, kind = "context_only", "scripted"
    GRID = np.linspace(-1.0, 1.0, 6)

    def __init__(self, spec, seed=0, device="cpu", explore_frac: float = 1.0):
        super().__init__(spec, seed, device)
        self.explore_frac = float(explore_frac)
        self._t = None

    def reset_context(self, n, which=None):
        if self._t is None or len(self._t) != n or which is None:
            self._t = np.zeros(n, dtype=np.int64)
            self._best = -np.ones((n, self.spec.act_dim), dtype=np.float32)
            self._best_r = np.full(n, -np.inf)
            self._last = self._best.copy()
        else:
            w = np.asarray(which, bool)
            self._t[w] = 0
            self._best[w] = -1.0
            self._best_r[w] = -np.inf
            self._last[w] = -1.0

    def act(self, obs, explore=True):
        n = obs.shape[0]
        if self._t is None or len(self._t) != n:
            self.reset_context(n)
        real = _real(obs, self.spec)
        r = obs[:, self.spec.scalars][:, 0]
        out = np.zeros((n, self.spec.act_dim), dtype=np.float32)
        g = len(self.GRID)
        for i in range(n):
            t = int(self._t[i])
            if t > 0 and r[i] > self._best_r[i]:
                self._best_r[i], self._best[i] = r[i], self._last[i]
            levers = np.flatnonzero(real[i])
            a = self._best[i].copy()
            probes = int(round(self.explore_frac * g)) * len(levers) if self.explore_frac != 1.0 else g * len(levers)
            if t < probes:
                if self.explore_frac == 1.0:
                    a[levers[t // g]] = self.GRID[t % g]
                else:                                   # a long random search before committing
                    a[levers] = np.random.default_rng([self.seed, i, t]).uniform(-1, 1, len(levers))
            self._last[i] = a
            out[i] = a
            self._t[i] = t + 1
        return out

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"learner": self.name}, fh)

    def load(self, path):
        pass


class WeightLearner(Learner):
    """One weight vector = the settings it plays on every tick of every task. `update` moves it along a
    score-function gradient estimated from exploratory transitions. No context: `act` looks at nothing but the
    weights (and, exploring, its random generator)."""
    name, kind = "weights_only", "model_free"

    def __init__(self, spec, seed=0, device="cpu", lr: float = 1.0, sigma: float = 0.15, batch: int = BATCH // 2):
        super().__init__(spec, seed, device)
        self.lr, self.sigma, self.batch = float(lr), float(sigma), int(batch)
        self.theta = -np.ones(spec.act_dim)
        self.rng = np.random.default_rng(seed)
        self._buf: list[tuple[np.ndarray, float, np.ndarray]] = []
        self._usage.params = spec.act_dim

    def act(self, obs, explore=True):
        n = obs.shape[0]
        a = np.tile(self.theta, (n, 1))
        if explore and not self.frozen:
            a = a + self.sigma * self.rng.standard_normal(a.shape)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def observe(self, obs, action, reward, done, next_obs):
        assert not self.frozen, "observe must not be called on a frozen learner"
        real = _real(obs, self.spec)
        for i in range(obs.shape[0]):
            self._buf.append((np.asarray(action[i], float), float(reward[i]), real[i]))

    def update(self):
        if self.frozen or len(self._buf) < self.batch:
            return {"updates": 0}
        a = np.array([b[0] for b in self._buf])
        r = np.array([b[1] for b in self._buf])
        m = np.array([b[2] for b in self._buf], dtype=float)
        g = ((r - r.mean())[:, None] * (a - self.theta) * m).mean(axis=0) / self.sigma ** 2
        self.theta = np.clip(self.theta + self.lr * g, -1.0, 1.0)
        self._buf = []
        self._usage.updates += 1
        return {"updates": 1}

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"theta": self.theta.tolist(), "updates": self._usage.updates,
                       "buffer": [[b[0].tolist(), b[1], b[2].tolist()] for b in self._buf],
                       "rng": _rng_state(self.rng)}, fh)

    def load(self, path):
        with open(path) as fh:
            d = json.load(fh)
        self.theta = np.array(d["theta"], dtype=float)
        self._usage.updates = int(d["updates"])
        self._buf = [(np.array(a), float(r), np.array(m, dtype=bool)) for a, r, m in d["buffer"]]
        self.rng.bit_generator.state = d["rng"]


def _rng_state(rng):
    return json.loads(json.dumps(rng.bit_generator.state, default=int))


class LeakyLearner(WeightLearner):
    """Keeps learning while frozen: every `act` nudges the weights. Frozen evaluation must catch this."""
    name = "leaky"

    def act(self, obs, explore=True):
        self.theta = np.clip(self.theta + 1e-3, -1.0, 1.0)
        return super().act(obs, explore)


class Memoriser(Learner):
    """Recognises a training task by its first reading at the default settings and replays the best action it
    ever found there. While it has no memory of a task it searches at random (training) or leaves the levers alone
    (evaluation). What it stores about one task says nothing about any other. Frozen, it may not retrieve -
    unless it is built with `cheat=True`, in which case it reads its store anyway."""
    name, kind = "memoriser", "memory"

    def __init__(self, spec, seed=0, device="cpu", cheat: bool = False):
        super().__init__(spec, seed, device)
        self.cheat = bool(cheat)
        self.store: dict[str, list] = {}            # key -> [best reward, best action]
        self.rng = np.random.default_rng(seed)
        self._t = None

    def reset_context(self, n, which=None):
        if self._t is None or len(self._t) != n or which is None:
            self._t = np.zeros(n, dtype=np.int64)
            self._key = [None] * n
            self._plan = [None] * n
            self._found = [None] * n                # best (reward, action) of the running episode
        else:
            for i in np.flatnonzero(np.asarray(which, bool)):
                self._t[i], self._key[i], self._plan[i], self._found[i] = 0, None, None, None

    def act(self, obs, explore=True):
        n = obs.shape[0]
        if self._t is None or len(self._t) != n:
            self.reset_context(n)
        real = _real(obs, self.spec)
        out = -np.ones((n, self.spec.act_dim), dtype=np.float32)
        may_read = self.cheat or not self.frozen
        for i in range(n):
            t = int(self._t[i])
            if t > 0:                                # tick 0: the defaults; the first reading identifies the task
                if t == 1:
                    self._key[i] = "|".join(f"{float(v):.6f}" for v in obs[i, self.spec.gauges]) + \
                        f"|{float(obs[i, self.spec.scalars][0]):.6f}|{int(real[i].sum())}"
                    hit = self.store.get(self._key[i]) if may_read else None
                    self._plan[i] = None if hit is None else np.array(hit[1], dtype=np.float32)
                if self._plan[i] is not None:
                    out[i] = self._plan[i]
                elif explore and not self.frozen:
                    lv = np.flatnonzero(real[i])
                    out[i, lv] = self.rng.uniform(-1, 1, len(lv))
            self._t[i] = t + 1
        return out

    def observe(self, obs, action, reward, done, next_obs):
        assert not self.frozen, "observe must not be called on a frozen learner"
        for i in range(obs.shape[0]):
            if self._key[i] is None:
                continue
            if self._found[i] is None or reward[i] > self._found[i][0]:
                self._found[i] = (float(reward[i]), np.asarray(action[i], float).tolist())
            if done[i]:                              # commit at the end of the episode only
                old = self.store.get(self._key[i])
                if old is None or self._found[i][0] > old[0]:
                    self.store[self._key[i]] = list(self._found[i])
                    self._usage.updates += 1

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"store": self.store, "updates": self._usage.updates, "rng": _rng_state(self.rng)}, fh)

    def load(self, path):
        with open(path) as fh:
            d = json.load(fh)
        self.store, self._usage.updates = d["store"], int(d["updates"])
        self.rng.bit_generator.state = d["rng"]


class MetaLearner(Learner):
    """Climbs inside the episode (context: the best settings so far), one lever at a time, in steps of a size
    that is its only weight. Training tunes the step size from whole-episode returns; nothing it learns says
    where a good setting is, only how fast to look for one. Right after a context reset it knows nothing and
    plays the lowest settings."""
    name, kind = "meta_only", "meta"

    def __init__(self, spec, seed=0, device="cpu", step: float = 0.02, lr: float = 0.5):
        super().__init__(spec, seed, device)
        self.log_step, self.lr = float(np.log(step)), float(lr)
        self._pending: list[tuple[float, float]] = []
        self._t = None
        self._usage.params = 1

    def reset_context(self, n, which=None):
        if self._t is None or len(self._t) != n or which is None:
            self._t = np.zeros(n, dtype=np.int64)
            self._best = -np.ones((n, self.spec.act_dim), dtype=np.float32)
            self._best_r = np.full(n, -np.inf)
            self._last = self._best.copy()
            self._stuck = np.zeros((n, self.spec.act_dim), dtype=bool)
            self._mult = np.where(np.arange(n) % 2 == 0, 0.5, -0.5)      # exploration: half the slots step wider
            self._ret = np.zeros(n)
        else:
            w = np.asarray(which, bool)
            self._t[w], self._best[w], self._best_r[w], self._last[w] = 0, -1.0, -np.inf, -1.0
            self._stuck[w], self._ret[w] = False, 0.0

    def act(self, obs, explore=True):
        n = obs.shape[0]
        if self._t is None or len(self._t) != n:
            self.reset_context(n)
        real = _real(obs, self.spec)
        r = obs[:, self.spec.scalars][:, 0]
        out = np.zeros((n, self.spec.act_dim), dtype=np.float32)
        for i in range(n):
            t = int(self._t[i])
            step = float(np.exp(self.log_step + (self._mult[i] if explore and not self.frozen else 0.0)))
            levers = np.flatnonzero(real[i])
            if t > 0:
                j = levers[(t - 1) % len(levers)] if t > 1 else None
                if r[i] > self._best_r[i]:
                    self._best_r[i], self._best[i] = r[i], self._last[i]
                elif j is not None:
                    self._stuck[i, j] = True
            a = self._best[i].copy()
            if t > 0:
                j = levers[t % len(levers)]
                if not self._stuck[i, j]:
                    a[j] = min(1.0, a[j] + step)
            self._last[i] = a
            out[i] = a
            self._t[i] = t + 1
        return out

    def observe(self, obs, action, reward, done, next_obs):
        assert not self.frozen
        self._ret += reward
        for i in np.flatnonzero(done):
            self._pending.append((float(self._mult[i]), float(self._ret[i])))

    def update(self):
        if self.frozen or len(self._pending) < 8:
            return {"updates": 0}
        m = np.array([p[0] for p in self._pending])
        g = np.array([p[1] for p in self._pending])
        hi, lo = g[m > 0].mean(), g[m < 0].mean()
        self.log_step += self.lr * float(np.sign(hi - lo))
        self._pending = []
        self._usage.updates += 1
        return {"updates": 1}

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"log_step": self.log_step, "pending": self._pending, "updates": self._usage.updates}, fh)

    def load(self, path):
        with open(path) as fh:
            d = json.load(fh)
        self.log_step, self._pending = float(d["log_step"]), [tuple(p) for p in d["pending"]]
        self._usage.updates = int(d["updates"])


class Spender(HoldLearner):
    """Pushes every lever to its maximum."""
    name = "spender"

    def act(self, obs, explore=True):
        return np.ones((obs.shape[0], self.spec.act_dim), np.float32)


def F(cls, **kw):
    return lambda seed: cls(SPEC, seed, **kw)


class Spy(HoldLearner):
    """Leaves the levers alone and counts what the harness calls."""
    name = "spy"

    def __init__(self, spec, seed=0, device="cpu"):
        super().__init__(spec, seed, device)
        self.calls = {"observe": 0, "update": 0, "act_rows": 0, "reset": 0}
        self.freezes = []

    def reset_context(self, n, which=None):
        self.calls["reset"] += 1

    def act(self, obs, explore=True):
        self.calls["act_rows"] += obs.shape[0]
        return super().act(obs, explore)

    def observe(self, *a):
        self.calls["observe"] += 1

    def update(self):
        self.calls["update"] += 1
        return {"updates": 0}

    def freeze(self, frozen=True):
        self.freezes.append(bool(frozen))
        super().freeze(frozen)


# ====================================================================== 1. frozen evaluation
def test_frozen_evaluation_is_paired_by_task_restores_the_flag_and_never_teaches(fam_a):
    short = hand_task("hand_short", 0, peak_mech(0.7, horizon=20), {"a": 0.7, "c": 0.0})
    tasks = fam_a["eval"][:5] + [short] + fam_a["eval"][5:]              # another horizon in the middle
    spy = Spy(SPEC)
    hold = P.evaluate(spy, tasks, n_envs=4)
    assert [r["task"] for r in hold["records"]] == [t.id for t in tasks] == hold["tasks"]
    assert spy.calls["observe"] == 0 and spy.calls["update"] == 0
    assert spy.frozen is False and spy.freezes == [True, False]
    spy.freeze(True)
    P.evaluate(spy, tasks[:2])
    assert spy.frozen is True                                            # restored to what it was, not to False
    # every row a learner was asked to act on was a live episode: act rows == environment steps
    assert hold["cost"]["env_steps"] == sum(t.horizon for t in tasks) == 12 * H + 20
    assert spy.calls["act_rows"] == hold["cost"]["env_steps"] + 2 * H
    assert all(r["score"] == 0.0 and r["ticks_moved"] == 0 and r["success"] is False for r in hold["records"])

    ctx = P.evaluate(ContextLearner(SPEC), tasks, n_envs=4)
    d = S.paired_diff(P._scores(ctx), P._scores(hold))
    assert d["n"] == len(tasks) and d["lo"] > 0.5 and d["frac_better"] == 1.0
    a = ctx["aggregate"]
    assert a["score"]["n"] == len(tasks) and a["score"]["lo"] < a["score"]["mean"] < a["score"]["hi"]
    assert a["iqm"]["lo"] <= a["iqm"]["iqm"] <= a["iqm"]["hi"]
    assert a["success"]["p"] == 1.0 and 0.7 < a["success"]["lo"] < 1.0 and a["success"]["hi"] == 1.0   # Wilson
    assert hold["aggregate"]["success"]["p"] == 0.0 and 0.0 < hold["aggregate"]["success"]["hi"] < 0.3
    assert set(ctx["by_family"]) == {"hand_a", "hand_short"} and ctx["by_family"]["hand_a"]["n"] == 12
    assert set(ctx["by_transfer"]) == {"within", "unspecified"}
    # the within-episode profile: four segments that average to the episode score; late beats early
    assert len(a["segments"]) == 4 and a["adaptation"]["lo"] > 0.1
    for r in ctx["records"]:
        assert np.mean(r["segment_scores"]) == pytest.approx(r["score"])
        assert np.mean(r["segment_scores"][:2]) == pytest.approx(r["score_first_half"])
        assert r["adaptation"] == pytest.approx(r["segment_scores"][-1] - r["segment_scores"][0])
    # clearing the context every tick removes everything this learner can do; every 6 ticks, most of it
    cut1 = P.evaluate(ContextLearner(SPEC), tasks, reset_every=1)
    cut6 = P.evaluate(ContextLearner(SPEC), tasks, reset_every=6)
    assert cut1["aggregate"]["score"]["mean"] == 0.0
    assert 0.0 < cut6["aggregate"]["score"]["mean"] < a["score"]["mean"] - 0.1
    with pytest.raises(ValueError, match="distinct"):
        P.evaluate(spy, tasks + tasks[:1])
    json.dumps(P._js(ctx), allow_nan=False)


def test_frozen_evaluation_catches_a_learner_whose_weights_change(fam_a, tmp_path):
    ok = P.evaluate(WeightLearner(SPEC), fam_a["eval"][:4])
    assert ok["frozen"]["unchanged"] is True and ok["frozen"]["method"] == "structured"
    bad = P.evaluate(LeakyLearner(SPEC), fam_a["eval"][:4])
    assert bad["frozen"]["unchanged"] is False and bad["frozen"]["hash_before"] != bad["frozen"]["hash_after"]

    # the hash is of the long-term state: accounting and generator state stored in a checkpoint are not weights
    def js(name, **d):
        p = str(tmp_path / name)
        with open(p, "w") as fh:
            json.dump(d, fh)
        return P.checkpoint_digest(p)
    base = js("a.json", theta=[0.1, 0.2], usage={"infer_cpu_s": 1.0}, rng={"state": 5})
    assert js("b.json", theta=[0.1, 0.2], usage={"infer_cpu_s": 9.0}, rng={"state": 6})["hash"] == base["hash"]
    assert js("c.json", theta=[0.1, 0.2000001], usage={"infer_cpu_s": 1.0}, rng={"state": 5})["hash"] != base["hash"]
    np.savez(str(tmp_path / "w1.npz"), w=np.arange(4.0), b=np.zeros(2))
    np.savez(str(tmp_path / "w2.npz"), b=np.zeros(2), w=np.arange(4.0))
    np.savez(str(tmp_path / "w3.npz"), w=np.arange(4.0) + 1e-9, b=np.zeros(2))
    h1, h2, h3 = (P.checkpoint_digest(str(tmp_path / f"w{k}.npz")) for k in (1, 2, 3))
    assert h1["method"] == "structured" and h1["hash"] == h2["hash"] != h3["hash"]
    with open(tmp_path / "raw.bin", "wb") as fh:
        fh.write(b"\x00\x01opaque")
    assert P.checkpoint_digest(str(tmp_path / "raw.bin"))["method"] == "bytes"

    class Own(HoldLearner):
        def state_hash(self):
            return "abc"
    assert P.learner_hash(Own(SPEC)) == {"hash": "abc", "method": "state_hash", "bytes": None}


def test_torch_checkpoints_are_hashed_by_their_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    w = {"layer.weight": torch.arange(6.0).reshape(2, 3), "layer.bias": torch.zeros(2)}
    torch.save({"model": w, "usage": {"infer_cpu_s": 0.1, "env_steps": 10}}, str(tmp_path / "a.pt"))
    torch.save({"model": w, "usage": {"infer_cpu_s": 7.0, "env_steps": 99}}, str(tmp_path / "b.pt"))
    w2 = dict(w, **{"layer.bias": torch.tensor([0.0, 1e-6])})
    torch.save({"model": w2, "usage": {"infer_cpu_s": 0.1, "env_steps": 10}}, str(tmp_path / "c.pt"))
    a, b, c = (P.checkpoint_digest(str(tmp_path / f"{k}.pt")) for k in "abc")
    assert a["method"] == "structured" and a["hash"] == b["hash"] != c["hash"]


# ====================================================================== 3. the three timescales
EXPECTED = {
    "context_only": (ContextLearner, {"fast": True, "parametric": False, "meta": False}),
    "weights_only": (WeightLearner, {"fast": False, "parametric": True, "meta": False}),
    "memoriser": (Memoriser, {"fast": False, "parametric": False, "meta": False}),
    "meta_only": (MetaLearner, {"fast": True, "parametric": True, "meta": True}),
    "hold": (HoldLearner, {"fast": False, "parametric": False, "meta": False}),
}


_KEEP: dict = {}            # results worth rendering again in the report tests


@pytest.mark.parametrize("name", list(EXPECTED))
def test_timescales_attribute_each_learners_gain_to_the_right_timescale(name, fam_a):
    cls, want = EXPECTED[name]
    r = _KEEP[f"timescales/{name}"] = P.timescales(F(cls), fam_a["train"], fam_a["eval"], 6 * BATCH, n_envs=N_ENVS)
    assert r["attribution"] == want, (r["attribution"], r["p_holm"])
    assert r["disjoint"]["ok"] and r["disjoint"]["checked_training_tasks"] == 48 and r["frozen_unchanged"]
    gain = r["parametric"]["gain"]["mean_diff"]
    direct = r["parametric"]["without_context"]["mean_diff"]
    meta = r["meta"]["context_gain"]["mean_diff"]
    ctx_trained = r["fast"]["trained"]["context"]["mean_diff"]
    assert gain == pytest.approx(direct + meta, abs=1e-9)        # the split of the parametric gain is exact
    if name == "context_only":
        # all of its performance is carried by context, at both checkpoints, and training changed nothing
        assert ctx_trained > 0.8 and r["fast"]["untrained"]["context"]["mean_diff"] == ctx_trained
        assert r["fast"]["trained"]["late_minus_early"]["lo"] > 0.1
        assert gain == 0.0 and meta == 0.0 and r["training"]["updates"] == 0
        assert r["checkpoints"]["weights_changed_by_training"] is False
    elif name == "weights_only":
        # everything it gained survives without any context; inside an episode it does not adapt at all
        assert gain > 0.9 and direct == gain and meta == 0.0 and ctx_trained == 0.0
        assert r["fast"]["trained"]["late_minus_early"]["mean"] == pytest.approx(0.0, abs=1e-9)
        assert r["meta"]["rise_gain"]["mean_diff"] == pytest.approx(0.0, abs=1e-9)
        assert r["training"]["updates"] == 12 and r["parametric"]["success_gain"]["mean_diff"] == 1.0
        assert r["parametric"]["gain"]["lo"] > 0.9 and r["p_holm"]["parametric"] < 0.01
    elif name == "memoriser":
        # it did store things (its checkpoint changed), and none of it is worth anything on fresh tasks
        assert r["training"]["updates"] > 0 and r["checkpoints"]["weights_changed_by_training"] is True
        assert gain == 0.0 and direct == 0.0 and meta == 0.0 and ctx_trained == 0.0
    elif name == "meta_only":
        # training taught it nothing about where to go (no gain without context), only to adapt faster
        assert gain > 0.4 and direct == 0.0 and meta == gain
        assert r["meta"]["rise_gain"]["lo"] > 0.05           # the climb comes earlier in the episode:
        # after a quarter of an episode the trained checkpoint is where the untrained one was at the end
        assert r["meta"]["segments_to_untrained_final"]["segment"] == 1
        assert r["fast"]["trained"]["context"]["mean_diff"] > r["fast"]["untrained"]["context"]["mean_diff"] + 0.4
    else:
        assert gain == 0.0 and direct == 0.0 and meta == 0.0 and ctx_trained == 0.0
    json.dumps(r, allow_nan=False)


# ====================================================================== 4. parametric internalisation
STEP_NAMES = ["evaluate_original_weights", "learning_interactions", "update_weights", "save_checkpoint",
              "clear_task_context", "disable_retrieval", "evaluate_on_fresh_tasks",
              "compare_with_original_paired_by_task"]


def test_internalisation_runs_its_eight_steps_and_finds_what_the_weight_learner_internalised(fam_a):
    r = P.internalisation(F(WeightLearner), fam_a["train"], fam_a["eval"], 6 * BATCH, n_envs=N_ENVS)
    assert [s["step"] for s in r["steps"]] == list(range(1, 9)) and [s["name"] for s in r["steps"]] == STEP_NAMES
    assert r["verdict"] == "internalised" and r["valid"] and r["violations"] == []
    assert r["gain"]["n"] == 12 and r["gain"]["lo"] > 0.9 and r["gain"]["p_value"] < 0.01
    assert r["success_gain"]["mean_diff"] == 1.0
    assert r["update_count"] == 12 and r["env_interactions"] == 6 * BATCH
    assert r["compute"]["train_cpu_s"] > 0 and r["compute"]["infer_cpu_s"] > 0 and r["compute"]["env_sim_s"] > 0
    s = {x["name"]: x for x in r["steps"]}
    assert s["save_checkpoint"]["weights_changed_by_training"] is True
    assert s["clear_task_context"]["checkpoint_round_trips"] is True
    assert s["disable_retrieval"]["frozen_flag"] is True
    assert r["disjoint"]["ok"] and r["disjoint"]["checked_training_tasks"] == 48
    assert r["frozen_unchanged"] == {"original": True, "trained": True}
    assert s["evaluate_on_fresh_tasks"]["hash_before"] == s["evaluate_on_fresh_tasks"]["hash_after"] \
        == s["save_checkpoint"]["checkpoint_hash"]
    assert r["repeat"] == {"identical": True, "max_abs_score_diff": 0.0}
    assert r["transfer_performance"]["by_transfer"]["within"]["score"]["mean"] > 0.95
    assert r["transfer_performance"]["success"]["p"] == 1.0
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize("cls", [ContextLearner, Memoriser, HoldLearner])
def test_internalisation_reports_no_gain_when_nothing_went_into_the_weights(cls, fam_a):
    r = P.internalisation(F(cls), fam_a["train"], fam_a["eval"], 3 * BATCH, n_envs=N_ENVS)
    assert [s["name"] for s in r["steps"]] == STEP_NAMES             # reported in full although nothing improved
    assert r["valid"] and r["verdict"] == "no_gain_detected"
    assert r["gain"]["n"] == 12 and r["gain"]["mean_diff"] == 0.0 and r["gain"]["p_value"] == 1.0
    assert r["env_interactions"] == 3 * BATCH and r["disjoint"]["ok"]
    changed = r["steps"][3]["weights_changed_by_training"]
    if cls is Memoriser:
        assert r["update_count"] > 0 and changed is True             # it wrote to its store; none of it transfers
    else:
        assert r["update_count"] == 0 and changed is False
    if cls is ContextLearner:
        # the context learner is good at these tasks - before and after alike: nothing was internalised
        assert r["original"]["score"]["mean"] > 0.8 and r["trained"]["score"]["mean"] == r["original"]["score"]["mean"]


def test_a_memoriser_gains_on_what_it_stored_and_loses_that_gain_when_frozen(fam_a):
    sets = {"seen": fam_a["train"], "fresh": fam_a["eval"][:6]}
    honest = P.lifetime(Memoriser(SPEC), fam_a["train"], 3 * BATCH, BATCH, sets, n_envs=N_ENVS)
    tr = [p["train"]["score"] for p in honest["points"][1:]]
    assert tr[0] < 0.45 and min(tr[1:]) > 0.75                     # the apparent gain: replaying these very tasks
    for name in ("seen", "fresh"):
        y = [p["eval"][name]["score"]["mean"] for p in honest["points"]]
        assert max(y) == min(y) == 0.0                             # frozen, retrieval is off: the gain is gone
    assert honest["disjoint"]["seen"]["ok"] is False and len(honest["disjoint"]["seen"]["overlap"]) == 8
    assert honest["disjoint"]["fresh"]["ok"] is True
    # a memoriser that retrieves although frozen keeps its gain on the stored tasks only
    cheat = P.lifetime(Memoriser(SPEC, cheat=True), fam_a["train"], 3 * BATCH, BATCH, sets, n_envs=N_ENVS)
    seen = [p["eval"]["seen"]["score"]["mean"] for p in cheat["points"]]
    fresh = [p["eval"]["fresh"]["score"]["mean"] for p in cheat["points"]]
    assert seen[-1] > 0.75 and seen[-1] - seen[0] > 0.7 and max(fresh) == min(fresh)
    # which is why the internalisation protocol evaluates on fresh tasks and checks that they are fresh
    r = P.internalisation(F(Memoriser, cheat=True), fam_a["train"], fam_a["eval"], 3 * BATCH, n_envs=N_ENVS)
    assert r["verdict"] == "no_gain_detected" and r["gain"]["mean_diff"] == 0.0 and r["update_count"] > 0
    with pytest.raises(P.ProtocolViolation, match="overlap"):
        P.internalisation(F(Memoriser, cheat=True), fam_a["train"], fam_a["train"][:4], 2 * BATCH, n_envs=N_ENVS)


def test_overlapping_training_and_evaluation_tasks_are_rejected(fam_a):
    with pytest.raises(P.ProtocolViolation, match="overlap"):
        P.internalisation(F(WeightLearner), fam_a["train"], fam_a["eval"][:3] + fam_a["train"][:1], BATCH, n_envs=N_ENVS)
    # the same hidden mechanism under another task id is the same task
    one = hand_task("hand_x", 1, peak_mech(0.66), {"a": 0.66, "c": 0.0})
    twin = hand_task("hand_y", 2, peak_mech(0.66), {"a": 0.66, "c": 0.0})
    assert one.id != twin.id and P.mech_fingerprint(one) == P.mech_fingerprint(twin)
    assert P.overlap([one], [twin]) == [one.id] and P.overlap(fam_a["train"], fam_a["eval"]) == []
    with pytest.raises(P.ProtocolViolation, match="overlap"):
        P.internalisation(F(WeightLearner), fam_a["train"] + [one], fam_a["eval"][:3] + [twin], BATCH, n_envs=N_ENVS)
    # a stream cannot be checked in advance: the offending task is caught as it is consumed

    def stream():
        yield from fam_a["train"]
        yield from fam_a["eval"]
    with pytest.raises(P.ProtocolViolation, match="also in evaluation set"):
        P.internalisation(F(WeightLearner), stream(), fam_a["eval"], 4 * BATCH, n_envs=N_ENVS)
    with pytest.raises(P.ProtocolViolation):
        P.timescales(F(WeightLearner), fam_a["train"], fam_a["train"][:2], BATCH, n_envs=N_ENVS)
    with pytest.raises(P.ProtocolViolation):
        P.lifetime(WeightLearner(SPEC), fam_a["train"], BATCH, BATCH, fam_a["train"][:2], n_envs=N_ENVS,
                   require_disjoint=True)
    with pytest.raises(P.ProtocolViolation):
        P.transfer(F(WeightLearner), fam_a["train"], {"t": {"category": "within", "eval": fam_a["train"][:2],
                                                            "train": None}},
                   source_budget=BATCH, adapt_budget=BATCH, eval_every=BATCH, n_envs=N_ENVS)


def test_internalisation_is_invalid_when_weights_change_during_frozen_evaluation(fam_a):
    r = P.internalisation(F(LeakyLearner), fam_a["train"], fam_a["eval"], 2 * BATCH, n_envs=N_ENVS)
    assert r["valid"] is False and r["verdict"] == "invalid"
    assert any("changed during the frozen evaluation" in v for v in r["violations"])
    assert r["frozen_unchanged"]["trained"] is False and r["steps"][6]["hash_before"] != r["steps"][6]["hash_after"]
    assert r["gain"]["n"] == 12 and r["gain"]["mean_diff"] is not None      # still reported, and marked invalid


def test_retention_measures_how_much_of_the_first_family_survives_the_second(fam_a, fam_b):
    r = P.internalisation(F(WeightLearner), fam_a["train"], fam_a["eval"], 6 * BATCH, n_envs=N_ENVS,
                          retention={"train_tasks": fam_b["train"], "budget_env_steps": 6 * BATCH,
                                     "eval_tasks": fam_b["eval"][:8]})
    _KEEP["internalisation"] = r
    ret = r["retention"]
    assert r["verdict"] == "internalised" and r["valid"]
    # the one weight vector moved from A's optimum to B's: part of A is lost, B is gained
    k = ret["kept_fraction"]
    assert k["defined"] and 0.3 < k["ratio"] < 0.7 and k["hi"] < 0.75
    assert ret["forgetting"]["lo"] > 0.1 and ret["forgetting_verdict"] == "gain"
    assert ret["first_family"]["after_second_training"]["mean"] < ret["first_family"]["after_first_training"]["mean"]
    assert ret["second_family"]["gain"]["lo"] > 1.0 and ret["second_family"]["after"]["mean"] > 0.9
    assert ret["second_training"]["env_steps"] == 6 * BATCH and ret["second_training"]["updates"] == 12
    assert ret["frozen_unchanged"] is True and all(v["ok"] for v in ret["disjoint"].values())
    # control: more of the same family afterwards loses nothing
    more = peak_tasks("hand_a", range(300, 308), 0.62, 0.78)
    same = P.internalisation(F(WeightLearner), fam_a["train"], fam_a["eval"], 6 * BATCH, n_envs=N_ENVS,
                             retention={"train_tasks": more, "budget_env_steps": 2 * BATCH})["retention"]
    assert same["kept_fraction"]["defined"] and same["kept_fraction"]["ratio"] > 0.97
    assert same["forgetting_verdict"] != "gain" or same["forgetting"]["mean_diff"] < 0.02
    # a learner that gained nothing has nothing to retain, and the protocol says so instead of dividing by zero
    none = P.internalisation(F(HoldLearner), fam_a["train"], fam_a["eval"], BATCH, n_envs=N_ENVS,
                             retention={"train_tasks": fam_b["train"], "budget_env_steps": BATCH})["retention"]
    assert none["kept_fraction"]["defined"] is False and none["kept_fraction"]["ratio"] is None


# ====================================================================== 2. lifetime / learning curve
def _life(fam_a, d, seed=3, budget=6 * BATCH, **kw):
    return P.lifetime(WeightLearner(SPEC, seed=seed), fam_a["train"], budget, BATCH, {"fresh": fam_a["eval"]},
                      n_envs=N_ENVS, checkpoint_dir=d, **kw)


@pytest.fixture(scope="module")
def life_weight(fam_a, tmp_path_factory):
    return _life(fam_a, str(tmp_path_factory.mktemp("life")))


def test_lifetime_records_every_point_with_its_costs_and_derives_each_measure_separately(life_weight):
    r = life_weight
    pts = r["points"]
    assert [p["env_steps"] for p in pts] == [k * BATCH for k in range(7)]
    assert [p["updates"] for p in pts] == [2 * k for k in range(7)] and pts[-1]["updates_reported"] == 12
    for col in ("train_cpu_s", "infer_cpu_s", "env_sim_s", "sim_cost_cents", "sim_revenue_cents", "compute_s", "wall_s"):
        v = [p[col] for p in pts]
        assert v == sorted(v) and v[0] == 0.0 or col == "wall_s", col
        assert v[-1] > 0.0, col
    assert all(p["train_gpu_s"] == 0 and p["infer_tokens"] == 0 and p["search_evals"] == 0 for p in pts)
    assert pts[0]["train"] is None and pts[1]["train"]["episodes"] == N_ENVS and pts[1]["train"]["n_scored"] == N_ENVS
    assert all(os.path.exists(p["checkpoint"]) for p in pts) and len({p["weights_hash"] for p in pts}) == 7
    assert all(p["frozen_unchanged"] for p in pts) and r["frozen_unchanged"]
    y = [p["eval"]["fresh"]["score"]["mean"] for p in pts]
    ok = [p["eval"]["fresh"]["success"]["p"] for p in pts]
    assert y[0] == 0.0 and y[-1] > 0.95 and ok[0] == 0.0 and ok[-1] == 1.0 and y == sorted(y)
    # training cost and evaluation cost are kept apart
    assert r["cost"]["train"]["env_steps"] == 6 * BATCH and r["cost"]["eval"]["env_steps"] == 7 * 12 * H
    assert r["cost"]["eval"]["updates"] == 0 and r["cost"]["eval"]["train_cpu_s"] == 0.0
    assert r["complete"] and r["resumed"] == [] and r["disjoint"]["fresh"]["ok"]
    assert r["train_tasks"]["n"] == 6 * N_ENVS and r["train_tasks"]["families"] == {"hand_a": 48}
    # derived, each on its own
    x = [p["env_steps"] for p in pts]
    sc = r["derived"]["fresh"]["score"]
    assert sc["auc_steps"]["mean"] == pytest.approx(S.auc(x, y, x_max=6 * BATCH))
    assert sc["auc_steps"]["lo"] < sc["auc_steps"]["mean"] < sc["auc_steps"]["hi"] and sc["auc_steps"]["n"] == 12
    xc = [p["compute_s"] for p in pts]
    assert sc["auc_compute"]["x_max"] == xc[-1] and sc["auc_compute"]["mean"] == pytest.approx(S.auc(xc, y))
    assert set(sc["at_budget"]) == {str(int(6 * BATCH * f)) for f in (0.25, 0.5, 1.0)}
    assert sc["at_budget"][str(3 * BATCH)]["mean"] == pytest.approx(y[3]) and sc["at_budget"][str(3 * BATCH)]["reached"]
    assert sc["final"]["mean"] == pytest.approx(y[-1])
    th = sc["to_threshold"]["0.5"]
    k = next(i for i, v in enumerate(y) if v >= 0.5)
    assert x[k - 1] < th["steps"] <= x[k] and th["lo"] <= th["steps"] <= th["hi"] and th["reached_frac"] == 1.0
    assert xc[k - 1] <= th["compute_s"] <= xc[k] and th["wall_s"] > 0 and th["sustain"] == 2
    su = r["derived"]["fresh"]["success"]
    assert 0.0 < su["auc_steps"]["mean"] < 1.0 and su["final"]["mean"] == 1.0
    assert su["to_threshold"]["0.5"]["steps"] is not None
    res = r["derived"]["fresh"]["simulated_resources"]
    assert res["sim_cost_cents"] == pts[-1]["sim_cost_cents"] > 0 and res["ticks_idle"] == 0
    # the final evaluation keeps its episode records
    assert len(r["final"]["fresh"]["records"]) == 12 and r["final"]["fresh"]["records"][0]["task"] == "hand_a:100"
    json.dumps(r, allow_nan=False)
    # a threshold never reached is reported as such, not as the budget
    never = P.derive_curve(pts, "fresh", 6 * BATCH, thresholds=(2.0,), sustain=2, budgets=())
    assert never["score"]["to_threshold"]["2.0"]["steps"] is None
    assert never["score"]["to_threshold"]["2.0"]["reached_frac"] == 0.0


def test_an_interrupted_lifetime_resumes_from_its_last_checkpoint_with_the_same_curve(life_weight, fam_a, tmp_path):
    full, d = life_weight, str(tmp_path / "run")

    def interrupt(point):
        if point["i"] == 3:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        _life(fam_a, d, on_point=interrupt)
    assert len(json.load(open(os.path.join(d, "lifetime_state.json")))["points"]) == 4
    # a new process starts with a new learner: nothing survives but the directory
    again = _life(fam_a, d)
    assert again["resumed"] == [3] and again["complete"] and len(again["points"]) == 7
    for a, b in zip(full["points"], again["points"]):
        assert (a["env_steps"], a["updates"], a["episodes"]) == (b["env_steps"], b["updates"], b["episodes"])
        assert a["eval"]["fresh"]["scores"] == b["eval"]["fresh"]["scores"]
        assert a["eval"]["fresh"]["segment_scores"] == b["eval"]["fresh"]["segment_scores"]
        assert a["weights_hash"] == b["weights_hash"] and a["sim_cost_cents"] == b["sim_cost_cents"]
        assert a["train"] == b["train"]
    assert again["train_tasks"] == full["train_tasks"]
    for m in ("score", "success"):
        assert again["derived"]["fresh"][m]["auc_steps"]["mean"] == full["derived"]["fresh"][m]["auc_steps"]["mean"]
    assert again["points"][-1]["train_cpu_s"] > again["points"][3]["train_cpu_s"]     # costs carried across
    # calling again changes nothing; the directory of another run is refused; a larger budget continues this one
    assert _life(fam_a, d)["points"] == again["points"]
    with pytest.raises(ValueError, match="different run"):
        _life(fam_a, d, seed=4)
    more = _life(fam_a, d, budget=8 * BATCH)
    assert [p["env_steps"] for p in more["points"]] == [k * BATCH for k in range(9)] and more["resumed"] == [3, 6]
    assert [p["eval"]["fresh"]["scores"] for p in more["points"][:7]] == \
        [p["eval"]["fresh"]["scores"] for p in full["points"]]
    # stopping early says so
    part = P.lifetime(WeightLearner(SPEC, seed=3), fam_a["train"], 6 * BATCH, BATCH, {"fresh": fam_a["eval"][:3]},
                      n_envs=N_ENVS, max_points=2)
    assert part["complete"] is False and len(part["points"]) == 2 and part["points"][-1]["checkpoint"] is None


def test_training_batches_have_one_horizon_a_fixed_size_and_resume_exactly(fam_a):
    short = [hand_task("hand_short", s, peak_mech(0.7, horizon=20), {"a": 0.7, "c": 0.0}) for s in range(5)]
    pool = [fam_a["train"][0], short[0], fam_a["train"][1], fam_a["train"][2], short[1], fam_a["train"][3],
            short[2], fam_a["train"][4], short[3], short[4]]

    def at(i):
        return pool[i % len(pool)]
    src = P.TaskSource(at, 4)
    b1 = src.batch()
    assert len(b1) == 4 and {t.horizon for t in b1} == {H} and [t.id for t in b1] == [f"hand_a:{k}" for k in range(4)]
    st = src.state()
    assert st["consumed"] == 6 and st["pending"] == {"20": [1, 4]}       # two short tasks are waiting for company
    b2 = src.batch()
    assert len(b2) == 4 and {t.horizon for t in b2} == {20}
    for source in (at, list(pool), itertools_cycle(pool)):               # an index, a cycled set, a plain iterator
        again = P.TaskSource(source, 4, **st)
        assert [t.id for t in again.batch()] == [t.id for t in b2] and again.state() == src.state()
    # a finite iterator hands out what is left in smaller batches and then stops
    fin = P.TaskSource(iter(pool), 4)
    sizes = []
    while (b := fin.batch()) is not None:
        assert len({t.horizon for t in b}) == 1
        sizes.append(len(b))
    assert sum(sizes) == len(pool) and sizes[:2] == [4, 4] and fin.exhausted
    # a learner trained on a mixed stream only ever sees whole batches of live episodes
    spy = Spy(SPEC)
    cost = Cost()
    out = P._train(spy, P.TaskSource(at, 4), 400, cost, spec=SPEC, n_envs=4, update_every=1, explore=True, segments=4)
    assert cost.env_steps == 4 * H + 4 * 20 + 4 * H == spy.calls["act_rows"] and out["episodes"] == 12
    assert spy.calls["observe"] == H + 20 + H == spy.calls["update"] and spy.calls["reset"] == 3


def itertools_cycle(pool):
    import itertools
    return itertools.cycle(pool)


def test_accounting_separates_measured_from_reported_and_never_counts_twice():
    class Reporter(HoldLearner):
        def act(self, obs, explore=True):
            self._usage.infer_tokens += 7
            self._usage.search_evals += 3
            self._usage.infer_gpu_s += 0.5
            self._usage.infer_cpu_s += 100.0            # a claim; the harness measures its own
            return super().act(obs, explore)

        def update(self):
            self._usage.updates += 2
            self._usage.train_gpu_s += 0.25
            return {"updates": 2}
    lr = Reporter(SPEC)
    c = Cost()
    m = Meter(lr, c)
    obs = np.zeros((3, SPEC.obs_dim), np.float32)
    for _ in range(4):
        m.act(obs, False)
    m.update()
    m.close()
    m.close()                                           # idempotent
    assert (c.infer_tokens, c.search_evals, c.infer_gpu_s, c.train_gpu_s) == (28, 12, 2.0, 0.25)
    assert c.updates == 2 and c.updates_returned == 2 and c.updates_reported == 2
    assert 0.0 <= c.infer_cpu_s < 1.0 and c.reported["infer_cpu_s"] == 400.0      # measured, with the claim kept apart
    assert c.compute_s == pytest.approx(c.infer_cpu_s + c.train_cpu_s + 2.25)

    class Silent(HoldLearner):                          # reports no usage at all: the returned counters are used
        def update(self):
            return {"updates": 3}
    c2 = Cost()
    m2 = Meter(Silent(SPEC), c2)
    m2.update()
    m2.update()
    m2.close()
    assert c2.updates == 6 and c2.updates_reported == 0
    assert Cost.from_dict(c.to_dict()).to_dict() == c.to_dict() and c.copy().add(c2).updates == 8


# ====================================================================== 5. transfer
@pytest.fixture(scope="module")
def targets(fam_a, fam_b):
    return {"within": {"category": "within", "eval": fam_a["eval"][:8],
                       "train": peak_tasks("hand_a", range(200, 208), 0.62, 0.78)},
            "moved": {"category": "structural", "eval": fam_b["eval"][:8], "train": fam_b["train"]},
            "near": {"category": "compositional", "eval": peak_tasks("hand_c", range(100, 108), 0.5, 0.6),
                     "train": None}}


def test_transfer_efficiency_is_above_one_with_a_useful_prior_and_negative_transfer_is_flagged(fam_a, targets):
    r = _KEEP["transfer"] = P.transfer(F(WeightLearner, lr=0.3), fam_a["train"], targets, source_budget=12 * BATCH,
                                       adapt_budget=4 * BATCH, eval_every=BATCH, n_envs=N_ENVS)
    assert r["categories"] == {"within": ["within"], "compositional": ["near"], "structural": ["moved"],
                               "cross_impl": []}
    w = r["targets"]["within"]
    assert w["zero_shot"]["gain"]["lo"] > 0.9 and w["zero_shot"]["source_trained"]["success"]["p"] == 1.0
    ad = w["adaptation"]
    assert ad["same_training_tasks"] and ad["auc"]["ratio"]["defined"]
    assert ad["auc"]["ratio"]["ratio"] > 1.2 and ad["auc"]["ratio"]["lo"] > 1.0          # the prior is worth steps
    assert ad["auc"]["diff"]["lo"] > 0.0 and w["negative_transfer"] is False and w["negative_zero_shot"] is False
    st = ad["steps_to_threshold"]
    assert st["pretrained"] == 0.0 and st["scratch"] > 0 and st["saved_steps"] == st["scratch"] and st["unbounded"]
    assert len(ad["curves"]["pretrained"]["x"]) == 5 and ad["curves"]["scratch"]["mean"][0] == 0.0
    assert ad["curves"]["pretrained"]["mean"][0] > 0.95
    # the same prior is harmful where the optimum moved: flagged, not averaged away
    m = r["targets"]["moved"]
    assert m["negative_zero_shot"] is True and m["zero_shot"]["gain"]["hi"] < 0.0
    assert m["negative_transfer"] is True and m["adaptation"]["auc"]["diff"]["hi"] < 0.0
    assert m["adaptation"]["auc"]["pretrained"]["mean"] < m["adaptation"]["auc"]["scratch"]["mean"]
    assert "pretrained" in m["evidence"]
    assert r["negative_transfer"] == ["moved"] and r["negative_zero_shot"] == ["moved"]
    # a target without training tasks is zero-shot only
    n = r["targets"]["near"]
    assert n["adaptation"] is None and n["negative_transfer"] is None and n["zero_shot"]["gain"]["lo"] > 0.5
    assert r["source"]["runs"][0]["training"]["env_steps"] == 12 * BATCH
    json.dumps(r, allow_nan=False)


def test_transfer_efficiency_is_one_for_a_learner_with_nothing_to_transfer(fam_a, targets):
    only = {"within": targets["within"]}
    # nothing in a context-only learner survives its source training: the two adaptation curves are the same curve
    only["within"] = dict(only["within"], eval=only["within"]["eval"][:6])
    r = P.transfer(F(ContextLearner), fam_a["train"], only, source_budget=BATCH, adapt_budget=2 * BATCH,
                   eval_every=BATCH, n_envs=N_ENVS)
    ad = r["targets"]["within"]["adaptation"]
    assert ad["auc"]["ratio"]["defined"] and ad["auc"]["ratio"]["ratio"] == 1.0
    assert ad["auc"]["ratio"]["lo"] == ad["auc"]["ratio"]["hi"] == 1.0 and ad["auc"]["diff"]["mean_diff"] == 0.0
    assert r["targets"]["within"]["zero_shot"]["gain"]["mean_diff"] == 0.0
    assert r["targets"]["within"]["negative_transfer"] is False and r["negative_transfer"] == []
    # a weight learner whose source taught it nothing relevant (the useful lever did nothing there): about one
    flat = [HandTask("hand_flat", s, s, T.world_seed_of("hand_flat", s), 4001, 80020, H, 1, False,
                     engine="scm", mech=scm.handmade(
                         levers=[{"name": "a"}, {"name": "c", "cost": 1500.0}],
                         latents=[{"id": "x", "terms": [{"kind": "linear", "src": "a", "w": 0.0}]}],
                         outcome={"terms": [{"kind": "linear", "src": "x", "w": 1.0}], "bias": 1.0, "cap": 4.0},
                         horizon=H, income_cents_per_tick=4001)) for s in range(8)]
    r2 = P.transfer(F(WeightLearner, lr=0.3), flat, only, source_budget=3 * BATCH, adapt_budget=4 * BATCH,
                    eval_every=BATCH, n_envs=N_ENVS, seeds=(0, 1))
    ad2 = r2["targets"]["within"]["adaptation"]
    assert r2["targets"]["within"]["n_pairs"] == 12 and r2["seeds"] == [0, 1]
    assert ad2["auc"]["ratio"]["defined"] and 0.85 < ad2["auc"]["ratio"]["ratio"] < 1.15
    assert r2["targets"]["within"]["negative_transfer"] is False
    assert abs(r2["targets"]["within"]["zero_shot"]["gain"]["mean_diff"]) < 0.05
    s2 = ad2["steps_to_threshold"]
    assert s2["ratio"] is not None and 0.6 < s2["ratio"] < 1.6


# ====================================================================== 6. comparison and diagnosis
@pytest.fixture(scope="module")
def comparison(fam_a):
    learners = {"weights": F(WeightLearner), "context": F(ContextLearner), "twin": F(ContextLearner),
                "hold": F(HoldLearner)}
    return P.compare(learners, fam_a["train"], {"fresh": fam_a["eval"]}, 3 * BATCH, BATCH, seeds=(0, 1),
                     reference="context", n_envs=N_ENVS, keep_runs=True)


def test_comparison_under_equal_budgets_is_paired_and_corrected(comparison):
    c = comparison
    assert c["checks"] == {"same_eval_tasks": True, "same_train_tasks": True, "same_step_grid": True,
                           "same_seeds": True, "same_versions": True, "all_complete": True, "frozen_unchanged": True,
                           "equal_budgets": True}
    assert c["seeds"] == [0, 1] and c["budget_env_steps"] == 3 * BATCH and c["correction"] == "holm"
    rows = c["pairwise"]
    quantities = {"final_score@equal_steps", "final_success@equal_steps", "auc_score_vs_steps",
                  "auc_success_vs_steps", "score@equal_compute"}
    assert {r["quantity"] for r in rows} == quantities and len(rows) == 3 * 5      # three learners against the reference
    assert all(r["b"] == "context" and r["n"] == 12 and r["family_size"] == 15 for r in rows)   # paired over tasks
    raw = [r["p_value"] for r in rows]
    assert [r["p_holm"] for r in rows] == pytest.approx(S.holm(raw)) and all(r["p_holm"] >= r["p_value"] for r in rows)
    by = {(r["a"], r["quantity"]): r for r in rows}
    # identical tasks and seeds: a learner compared with its twin differs by exactly nothing
    assert by[("twin", "final_score@equal_steps")]["mean_diff"] == 0.0
    assert by[("twin", "final_score@equal_steps")]["significant"] is False
    # the trained weights end above the searcher, but spent the budget getting there: the two quantities disagree,
    # which is why neither is folded into the other
    assert by[("weights", "final_score@equal_steps")]["direction"] == "weights"
    assert by[("weights", "auc_score_vs_steps")]["direction"] == "context"
    assert by[("hold", "final_score@equal_steps")]["direction"] == "context"
    assert by[("hold", "final_success@equal_steps")]["mean_diff"] == -1.0
    # pairing is the per-task difference of means over seeds
    runs = c["runs"]
    fin = {n: np.mean([[v for v in r["points"][-1]["eval"]["fresh"]["scores"]] for r in runs[n]], axis=0)
           for n in runs}
    assert by[("weights", "final_score@equal_steps")]["mean_diff"] == pytest.approx(
        float(np.mean(fin["weights"] - fin["context"])))
    assert c["per_learner"]["weights"]["final"]["fresh"]["seed_sd"] is not None
    assert len(c["per_learner"]["weights"]["final"]["fresh"]["per_seed"]) == 2
    # equal compute, separately: everybody is read at the smallest total any learner used
    eq = c["equal_compute"]
    assert eq["measurable"] and eq["budget_s"] == pytest.approx(min(
        r["points"][-1]["compute_s"] for rs in runs.values() for r in rs))
    assert all(eq["per_learner"][n]["fresh"]["reached"] for n in runs)
    assert eq["per_learner"]["weights"]["fresh"]["env_steps_at_budget"] < 3 * BATCH    # it had not finished by then
    # runs that are not comparable are refused or flagged, never silently paired
    import copy
    bad = copy.deepcopy({k: runs[k] for k in ("weights", "hold")})
    bad["hold"][0]["config"]["eval_sets"]["fresh"] = "something else"
    with pytest.raises(P.ProtocolViolation, match="identical task sets"):
        P.compare_runs(bad)
    odd = copy.deepcopy({k: runs[k] for k in ("weights", "hold")})
    odd["hold"][1]["train_tasks"]["digest"] = "other tasks"
    assert P.compare_runs(odd)["checks"]["equal_budgets"] is False
    old = copy.deepcopy({k: runs[k] for k in ("weights", "hold")})
    old["hold"][0]["versions"]["task"] = "task/1"
    with pytest.raises(P.ProtocolViolation, match="versions"):
        P.compare_runs(old)
    json.dumps({k: v for k, v in c.items() if k != "runs"}, allow_nan=False)


def test_comparison_says_what_the_records_establish_about_failures(comparison):
    d = comparison["diagnosis"]
    assert d["hold"]["fresh"]["n_failed"] == 24 and d["hold"]["fresh"]["classes"] == {"never_touched": 24}
    assert d["hold"]["fresh"]["vs_reference"]["verdict"] == "worse than the reference"
    assert d["weights"]["fresh"]["n_failed"] == 0 and d["weights"]["fresh"]["vs_reference"]["verdict"] == \
        "better than the reference"
    assert d["twin"]["fresh"]["vs_reference"]["verdict"].startswith("no better than the reference")
    assert any("against context" in s for s in d["twin"]["fresh"]["statements"])


def test_failure_diagnosis_tells_the_ways_of_failing_apart(fam_a):
    def classes(learner, tasks):
        ev = P.evaluate(learner, tasks, n_envs=N_ENVS)
        return [P.diagnose_episode(r) for r in ev["records"]], ev
    tasks = fam_a["eval"][:4]
    got, _ = classes(HoldLearner(SPEC), tasks)
    assert {g["class"] for g in got} == {"never_touched"}
    # pushing an expensive lever to the top: the money runs out and the facility falls back to its defaults
    costly = [hand_task("hand_costly", s, peak_mech(0.7, cost=30000.0), {"a": 0.7, "c": 0.0}) for s in range(3)]
    got, ev = classes(Spender(SPEC), costly)
    assert {g["class"] for g in got} == {"went_broke"}
    assert all(r["ticks_fallback"] >= 30 and r["funds_end_cents"] < 30000 for r in ev["records"])
    assert "could not pay" in got[0]["statement"]
    # the same learner where the lever is merely wasteful: it explored nothing and never found the peak
    got, _ = classes(Spender(SPEC), tasks)
    assert {g["class"] for g in got} == {"never_found"}
    # a long random search that commits in the last quarter: found, too late
    got, ev = classes(ContextLearner(SPEC, explore_frac=2.5), tasks)
    assert {g["class"] for g in got} == {"found_late"}, [(g["class"], g["evidence"]["segment_scores"]) for g in got]
    assert all(r["success"] is False and r["segment_scores"][-1] > 0.5 for r in ev["records"])
    # the response to the lever inverts half way: what was found stops working while the settings are held
    flip = [hand_task("hand_flip", s, peak_mech(0.7, flip_at=H // 2),
                      lambda t, last: {"a": 0.7 if t < H // 2 else 0.0, "c": 0.0}) for s in range(3)]
    got, ev = classes(ContextLearner(SPEC), flip)
    assert {(g["class"], g["sub"]) for g in got} == {("found_and_lost", "held")}
    assert "world changed" in got[0]["statement"]

    class Wanderer(HoldLearner):                         # plays the optimum, then walks away from it
        def reset_context(self, n, which=None):
            self.t = 0

        def act(self, obs, explore=True):
            a = -np.ones((obs.shape[0], self.spec.act_dim), np.float32)
            if self.t < 15:
                a[:, 0] = 0.4
            self.t += 1
            return a
    got, _ = classes(Wanderer(SPEC), tasks)
    assert {(g["class"], g["sub"]) for g in got} == {("found_and_lost", "own_change")}
    # without a private reference nothing can be said about quality, and the diagnosis says that
    blind = [HandTask("hand_blind", 0, 0, 1, 4001, 80020, H, 1, False, engine="scm", mech=peak_mech(0.7))]
    got, ev = classes(Spender(SPEC), blind)
    assert got[0]["class"] == "not_determinable" and ev["records"][0]["score"] is None
    summary = P.diagnose_records(ev["records"])
    assert summary["classes"] == {"not_determinable": 1} and summary["n_failed"] == 1


# ====================================================================== generated tasks and the kernel
@pytest.fixture(scope="module")
def gen_tasks():
    """A few generated, validated tasks (cached on disk after their first build)."""
    return T.build_tasks([("scm_additive", T.VAL_SEED0), ("scm_additive", T.VAL_SEED0 + 1), ("scm_invu", T.VAL_SEED0)])


def test_protocols_run_on_generated_tasks_and_training_streams(gen_tasks, tmp_path):
    ev = P.evaluate_many({"hold": [HoldLearner(SPEC)], "random": [RandomLearner(SPEC, 1), RandomLearner(SPEC, 2)]},
                         gen_tasks, reference="hold", n_envs=4)
    assert ev["versions"]["task"] == T.TASK_VERSION and ev["families"] == ["scm_additive", "scm_invu"]
    per = ev["per_learner"]
    assert set(per["hold"]["by_transfer"]) == {"within", "structural"} and per["hold"]["by_transfer"]["within"]["n"] == 2
    assert per["random"]["by_family"]["scm_additive"]["n"] == 4 and per["random"]["seeds"] == [1, 2]
    assert abs(per["hold"]["score"]["mean"]) < 0.35 and per["hold"]["diagnosis"]["classes"] == {"never_touched": 3}
    assert per["random"]["score"]["mean"] < per["hold"]["score"]["mean"]
    assert per["random"]["diagnosis"]["vs_reference"]["reference"] == "hold"
    assert all(r["n"] == 3 and r["b"] == "hold" for r in ev["pairwise"])
    # a learning curve on an endless stream of unvalidated training tasks of a real family
    d = str(tmp_path / "gen")
    kw = dict(n_envs=4, checkpoint_dir=d)
    r = P.lifetime(WeightLearner(SPEC, batch=96), P.task_stream(["scm_additive"]), 2 * 4 * 96, 4 * 96,
                   {"val": gen_tasks}, require_disjoint=True, max_points=2, **kw)
    assert r["complete"] is False and [p["env_steps"] for p in r["points"]] == [0, 384]
    r = P.lifetime(WeightLearner(SPEC, batch=96), P.task_stream(["scm_additive"]), 2 * 4 * 96, 4 * 96,
                   {"val": gen_tasks}, require_disjoint=True, **kw)
    assert r["complete"] and r["resumed"] == [1] and [p["env_steps"] for p in r["points"]] == [0, 384, 768]
    assert r["train_tasks"] == {"n": 8, "digest": r["train_tasks"]["digest"], "families": {"scm_additive": 8}}
    assert r["points"][1]["train"]["score"] is None and r["points"][1]["train"]["reward_per_tick"] is not None
    assert r["points"][-1]["updates"] == 8 and r["disjoint"]["val"]["ok"] and r["points"][-1]["env_sim_s"] > 0
    assert set(r["final"]["val"]["by_transfer"]) == {"within", "structural"}     # new seeds of a training family


def test_a_numeric_learner_is_diagnosed_through_tool_calls_in_a_kernel_world(gen_tasks):
    t = gen_tasks[0]
    k = P.kernel_diagnosis(HoldLearner(SPEC), t)
    sim = P.evaluate(HoldLearner(SPEC), [t])["records"][0]
    assert k["score"] == pytest.approx(sim["score"]) and k["success"] == sim["success"]   # the same problem, by tools
    assert k["tool_calls"] >= 2 * t.horizon and k["classes"].get("ok_effect", 0) + k["classes"].get("ok_read", 0) > 0
    assert k["cash"]["max"] >= t.endowment and k["difficulty"] == "absent"      # the task is known to be feasible
    # holding one setting is, as tool calls, the same call repeated without new information
    assert any(f["kind"] == "loop" for f in k["findings"])
    # varied settings that miss the objective on a feasible task: a failure of strategy, and it is named
    r = P.kernel_diagnosis(RandomLearner(SPEC, 1), t)
    assert r["success"] is False and r["primary"] == "strategic_failure" and "reference succeeded" in r["primary_reason"]
    assert r["cash"]["spent_by_kind"]["rig_settle"] > k["cash"]["spent_by_kind"]["rig_settle"]
    json.dumps(P._js([k, r]), allow_nan=False)


# ====================================================================== 7. sealed evaluation
@pytest.fixture()
def sealed_dir(monkeypatch):
    """A sealed directory with a fixed salt, so that the (expensive) validated sealed tasks of this test are built
    once and found again; the registry and the log start empty every time."""
    d = os.path.join(T.cache_dir(), "sealed_test")
    os.makedirs(d, exist_ok=True)
    salt = os.path.join(d, "sealed_salt")
    if not os.path.exists(salt):
        fd = os.open(salt, os.O_CREAT | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write("5ea1ed" * 10 + "0000")
    for f in ("sealed_sets.json", "sealed_access.jsonl"):
        if os.path.exists(os.path.join(d, f)):
            os.remove(os.path.join(d, f))
    monkeypatch.setenv("CRUCIBLE_SEALED_DIR", d)
    return d


def test_the_salt_is_created_private_and_sets_are_restricted_to_test_families(tmp_path, monkeypatch):
    from crucible.research import sealed
    d = tmp_path / "vault"
    monkeypatch.setenv("CRUCIBLE_SEALED_DIR", str(d))
    with pytest.raises(sealed.SealedSetError, match="restricted to test-split"):
        sealed.create_set("leak", ["scm_additive"], 1, 1)                    # a training family
    with pytest.raises(sealed.SealedSetError, match="restricted to test-split"):
        sealed.create_set("leak", ["scm_stock_gate", "scm_invu"], 1, 1)      # a validation family among them
    info = sealed.create_set("final", ["scm_stock_gate"], n_per_family=2, budget=3, note="x")
    salt_file = d / "sealed_salt"
    assert stat.S_IMODE(os.stat(salt_file).st_mode) == 0o600 and stat.S_IMODE(os.stat(d).st_mode) == 0o700
    salt = salt_file.read_text().strip()
    assert len(salt) == 64 and int(salt, 16) >= 0
    assert info == sealed.describe("final") and info["generation"] == 1 and info["remaining"] == 3
    everything = json.dumps([info, sealed.describe(), sealed.access_log()]) + (d / "sealed_sets.json").read_text()
    assert salt not in everything and salt[:16] not in everything
    assert [e["event"] for e in sealed.access_log()] == ["salt_created", "create"]
    with pytest.raises(sealed.SealedSetError, match="already exists"):
        sealed.create_set("final", ["scm_stock_gate"], 1, 1)
    # seeds depend on the salt: another vault, other tasks; and they stay far from the public seed ranges
    a = sealed._seed(bytes.fromhex(salt), "final", 1, "scm_stock_gate", 0)
    b = sealed._seed(bytes.fromhex("00" * 32), "final", 1, "scm_stock_gate", 0)
    assert a != b and a >= sealed.SEALED_SEED0 > T.VAL_SEED0 + 10**6
    assert a != sealed._seed(bytes.fromhex(salt), "final", 2, "scm_stock_gate", 0)


def test_sealed_sets_log_every_access_refuse_beyond_the_budget_and_rotate(sealed_dir):
    from crucible.research import sealed
    sealed.create_set("t", ["scm_stock_gate"], n_per_family=1, budget=2)
    with pytest.raises(sealed.SealedSetError, match="acknowledgement"):
        sealed.open_set("t", "yes, I acknowledge", "peek")
    with pytest.raises(sealed.SealedSetError, match="purpose"):
        sealed.open_set("t", sealed.SEALED_ACK, "  ")
    assert sealed.describe("t")["accesses"] == 0                           # refusals are logged, not counted
    first = sealed.open_set("t", sealed.SEALED_ACK, "look at the tasks", caller="test-suite")
    assert len(first) == 1 and first.tag["access"] == 1 and first.tasks[0].split == "test"
    task = first.tasks[0]
    assert task.seed >= sealed.SEALED_SEED0 and task.validated and "seed" not in repr(first)
    # one access, several learners: the result is tagged, paired, and carries no seed
    res = sealed.evaluate({"hold": [HoldLearner(SPEC)], "random": [RandomLearner(SPEC, 1)]}, "t", sealed.SEALED_ACK,
                          "final table", reference="hold", n_envs=2)
    assert res["protocol"] == "sealed" and res["sealed"]["sealed"] is True
    assert (res["sealed"]["set"], res["sealed"]["generation"], res["sealed"]["access"], res["sealed"]["budget"]) == \
        ("t", 1, 2, 2)
    assert res["sealed"]["purpose"] == "final table" and res["sealed"]["task_version"] == T.TASK_VERSION
    text = json.dumps(res)
    assert str(task.seed) not in text and str(task.mech_seed) not in text and str(task.world_seed) not in text
    assert res["tasks"] == ["sealed/t/g1/scm_stock_gate/0"]
    rec = res["per_learner"]["hold"]["records"][0]
    assert rec["task"] == "sealed/t/g1/scm_stock_gate/0" and rec["seed"] is None and rec["sealed"] is True
    assert rec["score"] is not None and res["per_learner"]["hold"]["learner"]["seed"] == 0   # the learner's seed stays
    # the budget is spent: the next request is refused, the set is contaminated, and it stays refused
    with pytest.raises(sealed.SealedSetError, match="contaminated"):
        sealed.open_set("t", sealed.SEALED_ACK, "one more look")
    assert sealed.describe("t")["contaminated"] is True and sealed.describe("t")["accesses"] == 2
    with pytest.raises(sealed.SealedSetError, match="contaminated"):
        sealed.evaluate({"hold": [HoldLearner(SPEC)]}, "t", sealed.SEALED_ACK, "again")
    # a new generation: recorded, new tasks, counter back to zero
    with pytest.raises(sealed.SealedSetError):
        sealed.rotate("t", sealed.SEALED_ACK, "")
    new = sealed.rotate("t", sealed.SEALED_ACK, "generation 1 was used for the report draft")
    assert new["generation"] == 2 and new["accesses"] == 0 and new["contaminated"] is False
    assert new["history"][0]["generation"] == 1 and new["history"][0]["contaminated"] is True
    second = sealed.open_set("t", sealed.SEALED_ACK, "final table, generation 2")
    assert second.tag["generation"] == 2 and second.tasks[0].seed != task.seed
    log = sealed.access_log("t")
    assert [(e["event"], e.get("count")) for e in log] == [
        ("create", None), ("refused", None), ("refused", None), ("access", 1), ("access", 2), ("refused", 2),
        ("refused", 2), ("rotate", None), ("access", 1)]
    assert log[3]["purpose"] == "look at the tasks" and log[3]["caller"] == "test-suite"
    assert "test_research_protocols" in log[4]["caller"] and log[5]["marked_contaminated"] is True
    assert log[7]["reason"].startswith("generation 1") and log[8]["generation"] == 2


def test_sealed_sets_are_unreachable_from_learner_and_world_code(sealed_dir, fam_a):
    from crucible.research import sealed
    sealed.create_set("guarded", ["scm_stock_gate"], n_per_family=1, budget=5)

    class Sneaky(HoldLearner):                           # a learner that goes looking for the test set
        def act(self, obs, explore=True):
            import importlib
            importlib.import_module("crucible.research." + "sealed").open_set(
                "guarded", sealed.SEALED_ACK, "tune on the test set")
            return super().act(obs, explore)
    with pytest.raises(sealed.SealedSetError, match="not reachable"):
        P.evaluate(Sneaky(SPEC), fam_a["eval"][:1], verify_frozen=False)
    with pytest.raises(sealed.SealedSetError, match="not reachable"):
        Sneaky(SPEC).act(np.zeros((1, SPEC.obs_dim), np.float32))           # called directly, without any harness
    ns = {"__name__": "crucible.research.learners.fake", "sealed": sealed}
    exec("def grab():\n    return sealed.open_set('guarded', sealed.SEALED_ACK, 'x')", ns)
    with pytest.raises(sealed.SealedSetError, match="not reachable"):
        ns["grab"]()
    assert sealed.describe("guarded")["accesses"] == 0
    denied = [e for e in sealed.access_log("guarded") if e["event"] == "denied"]
    assert len(denied) == 3 and "Sneaky" in denied[1]["caller"] and "learners.fake" in denied[2]["caller"]
    # statically: nothing in the package imports the sealed module - not the learners, not any world module
    src = os.path.join(ROOT, "src", "crucible")
    offenders, scanned = [], 0
    for base, _, files in os.walk(src):
        for f in files:
            path = os.path.join(base, f)
            if not f.endswith(".py") or os.path.relpath(path, src) == os.path.join("research", "sealed.py"):
                continue
            with open(path) as fh:
                text = fh.read()
            if "sealed" not in text:                     # nothing to parse: the word does not occur at all
                continue
            scanned += 1
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.Import) and any(a.name.split(".")[-1] == "sealed" for a in node.names):
                    offenders.append(path)
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if mod.split(".")[-1] == "sealed" or (any(a.name == "sealed" for a in node.names)
                                                           and (mod.endswith("research") or (node.level and not mod))):
                        offenders.append(path)
                elif isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in \
                        ("import_module", "__import__"):
                    if any(isinstance(a, ast.Constant) and isinstance(a.value, str) and "sealed" in a.value
                           for a in node.args):
                        offenders.append(path)
    assert offenders == [] and scanned >= 1              # (files that mention the word in prose were parsed)
    assert os.path.isdir(os.path.join(src, "research", "learners")) and os.path.isdir(os.path.join(src, "mechanisms"))


# ====================================================================== 8. reports
def _check_report(out: dict, min_figures: int):
    with open(out["markdown"]) as fh:
        md = fh.read()
    assert len(out["figures"]) >= min_figures and len(set(out["figures"])) == len(out["figures"])
    for f in out["figures"]:
        assert os.path.getsize(f) > 3000 and f.endswith(".png")
        assert f"]({os.path.basename(f)})" in md                 # every figure sits in the report, next to its table
    assert md.count("![") == len(out["figures"]) and "| --- |" not in md and "|---|" in md
    low = md.lower()
    assert "combined score" not in low and "overall score" not in low and "nan" not in low
    return md


def test_reports_have_a_table_and_a_figure_per_quantity(comparison, life_weight, fam_a, tmp_path):
    from crucible.research import report
    out = str(tmp_path)
    md = _check_report(report.render({k: v for k, v in comparison.items() if k != "runs"}, out, "cmp"), 5)
    assert "final_score@equal_steps" in md and "score@equal_compute" in md and "p (Holm)" in md
    assert "never_touched: 24" in md and "worse than the reference" in md and "## Equal compute" in md
    assert {"cmp__fresh__score_vs_steps.png", "cmp__fresh__success_vs_steps.png", "cmp__fresh__score_vs_compute.png",
            "cmp__fresh__final_score.png", "cmp__fresh__auc_steps.png"} <= set(os.listdir(out))
    md = _check_report(report.render(life_weight, out, "life"), 4)
    assert "steps to 0.5 held for 2 points" in md and "area under the curve against compute" in md
    assert "simulated resources in training" in md and "| training |" in md and "| evaluation |" in md
    runs = {"weights": comparison["runs"]["weights"]}
    bundle = {"protocol": "lifetime_set", "label": "one learner, two seeds", "runs": runs,
              "aggregate": {n: P.aggregate_lifetimes(rs) for n, rs in runs.items()}}
    md = _check_report(report.render(bundle, out, "set"), 2 + 4)
    assert "sd " in md and "over 2 seeds" in md
    base = P.baseline({"context_only": F(ContextLearner), "hold": F(HoldLearner), "spender": F(Spender)},
                      fam_a["eval"], reference="hold", n_envs=N_ENVS)
    json_path = P.save_result(base, os.path.join(out, "base.json"))
    md = _check_report(report.render(json_path, out), 3)                     # from the file, as the runner does
    assert "success [Wilson 95%]" in md and "segment 4" in md and "never_found: 12" in md
    ev = P.evaluate(ContextLearner(SPEC), fam_a["eval"][:4])
    _check_report(report.render(ev, out, "ev"), 1)
    with pytest.raises(ValueError, match="no report"):
        report.render({"protocol": "nothing"}, out)
    # one learner, one colour and marker in every figure, whoever else is on the page
    a, b = report.Styles(["ppo_gru", "hold", "zeta", "alpha"]), report.Styles(["hold", "alpha", "ppo_gru"])
    assert a("ppo_gru") == b("ppo_gru") and a("hold") == b("hold") and a("alpha") == b("alpha")
    assert len({a(n)[0] for n in ("ppo_gru", "hold", "zeta", "alpha")}) == 4
    many = report.Styles([f"l{i}" for i in range(11)])
    assert many("l9")[0] == report.OTHER and len({many(f"l{i}")[0] for i in range(8)}) == 8   # no invented hues


def test_reports_of_the_timescale_internalisation_and_transfer_protocols(fam_a, fam_b, tmp_path):
    from crucible.research import report
    out = str(tmp_path)
    ts = _KEEP.get("timescales/meta_only") or P.timescales(F(MetaLearner), fam_a["train"], fam_a["eval"], 6 * BATCH,
                                                          n_envs=N_ENVS)
    md = _check_report(report.render(ts, out, "ts"), 4)
    assert "fast adaptation" in md and "parametric learning" in md and "meta-learning" in md
    assert "by segment 1 of 4" in md and md.count("| yes |") == 3
    it = _KEEP.get("internalisation") or P.internalisation(
        F(WeightLearner), fam_a["train"], fam_a["eval"], 6 * BATCH, n_envs=N_ENVS,
        retention={"train_tasks": fam_b["train"], "budget_env_steps": 6 * BATCH, "eval_tasks": fam_b["eval"][:8]})
    md = _check_report(report.render(it, out, "int"), 3)
    assert "**Verdict: internalised.**" in md and "evaluate on fresh tasks" in md and "of the first gain is kept" in md
    assert all(f"| {k} |" in md for k in range(1, 9))
    md = _check_report(report.render({"protocol": "internalisation_set", "label": "b", "results": {"weights_only": [it]}},
                                     out, "ints"), 4)
    assert "| weights_only | 0 | internalised |" in md
    bundle = {"protocol": "timescales_set", "label": "b", "results": {"meta_only": [ts]}}
    assert "(present)" in _check_report(report.render(bundle, out, "tss"), 3 + 4)
    tr = _KEEP.get("transfer")
    if tr is None:
        tr = P.transfer(F(WeightLearner, lr=0.3), fam_a["train"],
                        {"within": {"category": "within", "eval": fam_a["eval"][:8], "train": fam_a["train"]}},
                        source_budget=6 * BATCH, adapt_budget=2 * BATCH, eval_every=BATCH, n_envs=N_ENVS)
    md = _check_report(report.render(tr, out, "tr"), 3)
    assert "AUC ratio (efficiency)" in md and "zero-shot gain" in md
    if "moved" in tr["targets"]:
        assert "**Negative transfer** in adaptation: ['moved']" in md and "| NEGATIVE |" in md
    md = _check_report(report.render({"protocol": "transfer_set", "label": "b", "results": {"weights_only": tr}},
                                     out, "trs"), 3)
    assert "| weights_only | within | within |" in md


# ====================================================================== 9. the command line
def test_the_benchmark_command_runs_named_learners_resumes_and_writes_results_and_figures(tmp_path, sealed_dir):
    from crucible.research import learners as registry
    from crucible.research import sealed
    if "t_weights" not in registry.available():        # a learner of this file, reached by name like any other
        registry.register("t_weights", lambda spec, seed, device, **kw: WeightLearner(spec, seed, device, **kw),
                          kind="model_free", has_weights=True, learns_across_episodes=True,
                          defaults={"lr": 1.0, "sigma": 0.15, "batch": 96}, doc="test learner")
    sys.path.insert(0, os.path.join(ROOT, "examples"))
    try:
        import ow_benchmark
    finally:
        sys.path.pop(0)
    lines: list[str] = []
    common = ["--learners", "hold,t_weights", "--opts", '{"t_weights": {"batch": 96}}', "--seeds", "0,1",
              "--budget", "768", "--eval-every", "384", "--train-families", "scm_additive", "--eval-families",
              "scm_invu", "--n-eval", "1", "--n-envs", "4", "--train-pool", "4", "--tag", "t", "--reference", "hold",
              "--out", str(tmp_path / "exp"), "--figures", str(tmp_path / "fig"), "--work", str(tmp_path / "work")]
    out = ow_benchmark.main(["lifetime", "--no-report"] + common, log=lines.append)
    res = json.load(open(out["result"]))
    assert os.path.basename(out["result"]) == "ow_benchmark_lifetime_t.json" and res["protocol"] == "lifetime_set"
    assert set(res["runs"]) == {"hold", "t_weights"} and [r["learner"]["seed"] for r in res["runs"]["hold"]] == [0, 1]
    run = res["runs"]["t_weights"][0]
    assert run["learner"]["name"] == "t_weights" and run["complete"] and run["points"][-1]["env_steps"] == 768
    assert set(run["points"][0]["eval"]) == {"within", "structural"} and run["points"][-1]["updates"] == 8
    assert res["experiment"]["learners"] == ["hold", "t_weights"] and res["config"]["train_families"] == ["scm_additive"]
    assert sum("ran" in x for x in lines) == 4 and out["report"] is None
    # the comparison is made of the same runs: nothing is trained again
    lines.clear()
    cmp_out = ow_benchmark.main(["compare", "--kernel-diagnose", "1"] + common, log=lines.append)
    assert sum("reused" in x for x in lines) == 4 and not any(": ran" in x for x in lines)
    cmp_res = json.load(open(cmp_out["result"]))
    assert cmp_res["protocol"] == "compare" and cmp_res["checks"]["equal_budgets"] and cmp_res["reference"] == "hold"
    assert {r["set"] for r in cmp_res["pairwise"]} == {"within", "structural"}
    assert os.path.dirname(cmp_out["report"]) == str(tmp_path / "fig" / "ow_benchmark_compare_t")
    assert len(cmp_out["figures"]) >= 8 and "through tool calls on" in open(cmp_out["report"]).read()
    assert cmp_res["diagnosis"]["hold"]["within"]["kernel"][0]["tool_calls"] > 0
    base = ow_benchmark.main(["baseline", "--no-report"] + common, log=lines.append)
    assert json.load(open(base["result"]))["n_tasks"] == 2
    # the final test goes through the sealed accessor: declared, logged, tagged, seedless
    fin = ow_benchmark.main(["sealed", "--no-report"] + common + ["--sealed-set", "runner", "--sealed-families", "scm_stock_gate",
                                                   "--sealed-n", "1", "--purpose", "runner test", "--ack",
                                                   sealed.SEALED_ACK], log=lines.append)
    fres = json.load(open(fin["result"]))
    assert fres["protocol"] == "sealed" and fres["sealed"]["access"] == 1 and fres["sealed"]["set"] == "runner"
    assert fres["subjects"]["t_weights"][0]["trained"] is True and fres["subjects"]["hold"][0]["trained"] is True
    assert [e["event"] for e in sealed.access_log("runner")] == ["create", "prepare", "access"]
    with pytest.raises(SystemExit):                    # the budget of one is spent: asking again is refused
        try:
            ow_benchmark.main(["sealed"] + common + ["--sealed-set", "runner", "--purpose", "again", "--ack",
                                                     sealed.SEALED_ACK], log=lines.append)
        except sealed.SealedSetError as e:
            raise SystemExit(str(e))
    with pytest.raises(SystemExit, match="unknown learner"):
        ow_benchmark.main(["baseline", "--learners", "no_such_learner", "--tag", "x", "--out", str(tmp_path / "e2"),
                           "--figures", str(tmp_path / "f2"), "--work", str(tmp_path / "w2")], log=lines.append)
    with pytest.raises(SystemExit, match="test-split"):
        ow_benchmark.main(["baseline"] + common + ["--eval-families", "scm_stock_gate"], log=lines.append)
    with pytest.raises(SystemExit, match="--ack"):
        ow_benchmark.main(["sealed"] + common, log=lines.append)
    os.remove(cmp_out["result"])                       # (rendered above already)
    rep = ow_benchmark.main(["report", "--tag", "t", "--out", str(tmp_path / "exp"), "--figures",
                             str(tmp_path / "fig")], log=lines.append)
    assert set(rep) == {"baseline", "lifetime", "sealed"} and len(rep["lifetime"]["figures"]) >= 8
    assert "SEALED TEST SET" in open(rep["sealed"]["markdown"]).read()


def test_a_registry_learner_with_a_torch_checkpoint_passes_through_the_harness(fam_a, tmp_path):
    """The recurrent PPO learner of the registry under these protocols: its checkpoint hashes by content, frozen
    evaluation leaves it unchanged and reproduces itself, training changes it, and a learning curve resumes."""
    pytest.importorskip("torch")
    from crucible.research import learners as registry
    if "ppo_gru" not in registry.available():
        pytest.skip("the registry has no ppo_gru")
    make = P.from_registry("ppo_gru", SPEC, hidden=16, rollout_steps=BATCH)
    lr = make(0)
    ev = P.evaluate(lr, fam_a["eval"][:4], n_envs=4)
    assert ev["frozen"] == dict(ev["frozen"], unchanged=True, method="structured") and ev["learner"]["name"] == "ppo_gru"
    assert ev["learner"]["params"] > 0 and ev["cost"]["infer_cpu_s"] > 0
    again = P.evaluate(lr, fam_a["eval"][:4], n_envs=4)
    assert [r["score"] for r in again["records"]] == [r["score"] for r in ev["records"]]      # frozen: reproducible
    d = str(tmp_path / "ppo")
    kw = dict(n_envs=N_ENVS, checkpoint_dir=d)
    part = P.lifetime(make(0), fam_a["train"], 2 * BATCH, BATCH, {"fresh": fam_a["eval"][:4]}, max_points=2, **kw)
    assert part["points"][1]["updates"] > 0 and part["points"][0]["weights_hash"] != part["points"][1]["weights_hash"]
    full = P.lifetime(make(0), fam_a["train"], 2 * BATCH, BATCH, {"fresh": fam_a["eval"][:4]}, **kw)
    assert full["resumed"] == [1] and full["complete"] and full["frozen_unchanged"]
    assert full["points"][2]["updates"] > full["points"][1]["updates"] == part["points"][1]["updates"]
    assert full["points"][2]["train_cpu_s"] > 0 and full["cost"]["train"]["reported"]["train_cpu_s"] > 0
    assert full["points"][2]["updates_returned"] == full["points"][2]["updates_reported"]


def test_the_benchmark_command_starts_in_a_fresh_process_and_checks_its_arguments(tmp_path):
    """The command as a user runs it: a new interpreter in which nothing has been imported yet."""
    import subprocess
    cmd = [sys.executable, os.path.join(ROOT, "examples", "ow_benchmark.py")]
    where = ["--out", str(tmp_path / "e"), "--figures", str(tmp_path / "f"), "--work", str(tmp_path / "w")]
    env = dict(os.environ, CRUCIBLE_SEALED_DIR=str(tmp_path / "vault"))
    out = subprocess.run(cmd + ["list"], capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    listed = json.loads(out.stdout)
    assert {"hold", "random"} <= set(listed["learners"]) and "scm_additive" in listed["families"]["train"]
    assert "scm_stock_gate" in listed["families"]["test"] and listed["sealed_sets"] == {}
    assert not os.path.exists(tmp_path / "vault" / "sealed_salt")           # looking at the list opens nothing
    for extra, message in ((["--eval-families", "scm_stock_gate"], "test-split"),
                           (["--eval-families", "no_such_family"], "unknown mechanism family"),
                           (["--train-families", "scm_invu"], "training uses training families only"),
                           (["--learners", "no_such_learner"], "unknown learner")):
        bad = subprocess.run(cmd + ["baseline"] + where + extra, capture_output=True, text=True, env=env, timeout=300)
        assert bad.returncode != 0 and message in bad.stderr, (extra, bad.stderr[-500:])
    assert not os.path.exists(tmp_path / "e")                               # nothing was written for a refused command
