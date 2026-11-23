"""Real-environment training loop. Every executed action and observed outcome comes from the environment."""
from __future__ import annotations

import time
from typing import Callable, Iterator

import numpy as np

from crucible.research.envs import MechVecEnv
from crucible.research.tasks import Task


def train(learner, stream: Iterator[Task], env_steps: int, n_envs: int = 16, update_every: int = 16,
          on_progress: Callable[[int, dict], None] | None = None, progress_every: int = 20_000) -> dict:
    """Train for ``env_steps`` ticks on tasks drawn from ``stream``. A finished slot gets a fresh task."""
    tasks = [next(stream) for _ in range(n_envs)]
    env = MechVecEnv(tasks, learner.spec)
    obs = env.reset()
    learner.reset_context(n_envs)
    learner.set_labels([t.family for t in tasks])
    steps, since, last_prog = 0, 0, 0
    t0 = time.perf_counter()
    last: dict = {}
    while steps < env_steps:
        act = learner.act(obs, explore=True)
        nxt, rew, done, _ = env.step(act)
        learner.observe(obs, act, rew, done, nxt)
        obs = nxt
        steps += n_envs
        since += n_envs
        if done.any():
            idx = np.flatnonzero(done)
            new = [next(stream) for _ in idx]
            obs = env.reset(which=idx, tasks=new)
            for i, t in zip(idx, new):
                learner.labels[int(i)] = t.family
            learner.reset_context(n_envs, done)
        if since >= update_every:
            since = 0
            out = learner.update()
            if out.get("updates"):
                last = out
        if on_progress and steps - last_prog >= progress_every:
            last_prog = steps
            on_progress(steps, last)
    env.drain()
    learner.usage().env_steps += steps
    return {"env_steps": steps, "wall_s": time.perf_counter() - t0, "updates": learner.usage().updates, "last": last}
