"""Training and evaluation loops that use the learner contract and nothing else.

`train` drives any `Learner` on a stream of tasks: `n_envs` environments run side by side, a slot that finishes its
episode gets the next task of the stream and a fresh context, every transition goes to `observe`, and `update` is
called after every tick (the learner decides when it has enough to change its weights). At regular intervals of
environment steps the learner is frozen and evaluated on a fixed task set with fresh context (`evaluate`), and
the point (steps, updates, CPU seconds, scores) is appended to the curve. Nothing here knows what kind of learner
it is driving.

`train_pool` is `research.tasks.train_stream` with a disk cache. Building an unvalidated task costs 0.3 s (scm) to
4 s (abm) of CPU, almost all of it the facility's demand-income probe, so a training run that drew every task
fresh would spend most of its time there. The pool holds the first `n_per_family` tasks of the stream for each
family (the same `Task` objects `train_stream` yields, in the same order) and `pool_stream` cycles through them in
a seeded shuffled order. The cache is a pure function of (family, seed, versions).
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Iterator, Sequence

import numpy as np

from .. import tasks as T
from ..envs import MechVecEnv, run_episodes
from ..learner import EnvSpec, Learner

POOL_VERSION = "pool/1"


# ---------------------------------------------------------------------------- training tasks
def _pool_path(family: str, seed: int) -> str:
    from ...core.canonical import digest
    from ...mechanisms.base import CONTRACT_VERSION, MECH_FAMILIES
    v = digest([POOL_VERSION, T.TASK_VERSION, CONTRACT_VERSION, getattr(MECH_FAMILIES.get(family), "engine", "")], 6)
    return os.path.join(T.cache_dir(), "train_pool", v, family, f"{int(seed)}.json")


def _pool_one(args) -> dict:
    family, seed = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    T._ensure()                                  # the cache path depends on the family's engine
    path = _pool_path(family, seed)
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        pass
    t = T.make_task(family, seed, validate=False)
    rec = {"task": t.to_dict(), "mech": t.mechanism()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, path)
    return rec


def train_pool(families: Sequence[str], n_per_family: int, start: int = 0, workers: int = 1) -> list[T.Task]:
    """The first `n_per_family * len(families)` tasks of `train_stream(families, start)`, in stream order (seed
    major), from the disk cache where possible. The mechanisms are put into the in-process mechanism cache of
    `research.tasks`, so `Task.sim()` does not regenerate them."""
    T._ensure()
    pairs = [(f, int(start) + s) for s in range(int(n_per_family)) for f in families]
    if pairs and pairs[-1][1] > T.TRAIN_SEED_MAX:
        raise RuntimeError("training seed range exhausted")
    missing = [p for p in pairs if not os.path.exists(_pool_path(*p))]
    if workers > 1 and len(missing) > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(workers, len(missing))) as pool:
            pool.map(_pool_one, missing, chunksize=4)
    out = []
    for p in pairs:
        rec = _pool_one(p)
        t = T.Task.from_dict(rec["task"])
        if len(T._MECHS) > 4000:                 # the cache in research.tasks clears itself above 4096 entries
            T._MECHS.clear()
        T._MECHS.setdefault((t.family, int(t.mech_seed)), rec["mech"])
        out.append(t)
    return out


def pool_stream(pool: Sequence[T.Task], shuffle_seed: int | None = 0) -> Iterator[T.Task]:
    """Cycle through a pool for ever; every pass is a new seeded permutation (None: pool order)."""
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    while True:
        order = rng.permutation(len(pool)) if rng is not None else range(len(pool))
        for i in order:
            yield pool[int(i)]


# ---------------------------------------------------------------------------- frozen evaluation
def evaluate(learner: Learner, tasks: Sequence[T.Task], spec: EnvSpec | None = None, batch: int = 64,
             explore: bool = False) -> list[dict]:
    """Frozen evaluation with fresh context: one episode per task, records in task order. The learner's frozen
    flag is restored afterwards; its weights cannot change in between."""
    was = learner.frozen
    learner.freeze(True)
    recs: list[dict] = []
    try:
        for a in range(0, len(tasks), batch):
            env = MechVecEnv(list(tasks[a:a + batch]), spec or learner.spec)
            recs += run_episodes(env, learner, explore=explore, learn=False)
    finally:
        learner.freeze(was)
    return recs


def summarise(recs: Sequence[dict]) -> dict:
    """Mean normalised score, success rate and the within-episode halves of a list of episode records."""
    sc = [r["score"] for r in recs if r["score"] is not None]
    ok = [bool(r["success"]) for r in recs if r["success"] is not None]
    h1 = [r["score_first_half"] for r in recs if r.get("score_first_half") is not None]
    h2 = [r["score_second_half"] for r in recs if r.get("score_second_half") is not None]
    fam: dict[str, list[float]] = {}
    for r in recs:
        if r["score"] is not None:
            fam.setdefault(r["family"], []).append(r["score"])
    return {"n": len(sc), "score": float(np.mean(sc)) if sc else None,
            "success": float(np.mean(ok)) if ok else None,
            "score_first_half": float(np.mean(h1)) if h1 else None,
            "score_second_half": float(np.mean(h2)) if h2 else None,
            "by_family": {k: float(np.mean(v)) for k, v in sorted(fam.items())}}


# ---------------------------------------------------------------------------- training
def train(learner: Learner, stream: Iterator[T.Task], total_steps: int, n_envs: int = 16,
          eval_tasks: Sequence[T.Task] | None = None, eval_every: int = 0, spec: EnvSpec | None = None,
          on_point: Callable[[dict], None] | None = None, curve: list[dict] | None = None,
          log: Callable[[str], None] | None = None) -> list[dict]:
    """Train `learner` for `total_steps` environment steps of the stream. Returns the learning curve: one point
    per evaluation, `{"steps", "episodes", "updates", "train_cpu_s", "infer_cpu_s", "wall_s", "score", "success",
    "by_family", "scores": [per task]}`; the first point is the learner before any training. `curve` (a list of
    earlier points) resumes a run: counting starts from its last point."""
    spec = spec or learner.spec
    curve = list(curve or [])
    steps0 = int(curve[-1]["steps"]) if curve else 0
    episodes = int(curve[-1]["episodes"]) if curve else 0
    wall0 = time.perf_counter() - (float(curve[-1]["wall_s"]) if curve else 0.0)

    def point(steps: int) -> None:
        recs = evaluate(learner, eval_tasks, spec) if eval_tasks else []
        u = learner.usage()
        s = summarise(recs)
        p = {"steps": int(steps), "episodes": int(episodes), "updates": int(u.updates),
             "train_cpu_s": float(u.train_cpu_s), "infer_cpu_s": float(u.infer_cpu_s),
             "wall_s": float(time.perf_counter() - wall0), "score": s["score"], "success": s["success"],
             "score_first_half": s["score_first_half"], "score_second_half": s["score_second_half"],
             "by_family": s["by_family"], "scores": [r["score"] for r in recs],
             "train_return": float(np.mean(recent)) if recent else None}
        curve.append(p)
        if log:
            log(f"steps={p['steps']} episodes={episodes} updates={p['updates']} score={p['score']} "
                f"success={p['success']} train_return={p['train_return']} wall={p['wall_s']:.0f}s")
        if on_point:
            on_point(p)

    recent: list[float] = []
    if not curve:
        point(0)
    learner.freeze(False)
    env = MechVecEnv([next(stream) for _ in range(n_envs)], spec)
    obs = env.reset()
    learner.reset_context(n_envs)
    steps, next_eval = steps0, (steps0 // eval_every + 1) * eval_every if eval_every else None
    ret = np.zeros(n_envs)
    while steps < total_steps:
        act = learner.act(obs, explore=True)
        nxt, rew, done, _ = env.step(act)
        learner.observe(obs, act, rew, done, nxt)
        steps += n_envs
        learner.usage().env_steps += n_envs
        ret += rew
        if done.any():
            idx = np.flatnonzero(done)
            episodes += len(idx)
            recent.extend(float(x) for x in ret[idx])
            del recent[:-256]
            ret[idx] = 0.0
            env.drain()
            nxt = env.reset(which=done, tasks=[next(stream) for _ in idx])
            learner.reset_context(n_envs, which=done)
        learner.update()
        obs = nxt
        if next_eval is not None and steps >= next_eval:
            point(steps)
            learner.freeze(False)
            next_eval += eval_every
            # the evaluation used the learner's context: every training slot starts a new episode
            obs = env.reset(tasks=[next(stream) for _ in range(n_envs)])
            learner.reset_context(n_envs)
            ret[:] = 0.0
    if eval_every and (not curve or curve[-1]["steps"] != steps):
        point(steps)
    return curve
