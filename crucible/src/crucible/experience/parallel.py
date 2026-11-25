"""Parallel worlds for learners on the experience interface.

* `VecExperience(envs)` - several `ExperienceEnv`s stepped together in one process (one decision per environment
  per step), the vectorised form numeric learners are used to.
* Process-parallel rollouts reuse `crucible.scale.rollout.RolloutFarm` unchanged: `make_world` below builds any
  `make(tier, name, seed, reward_mode)` environment inside a worker as the farm's semi-MDP (``reset/step/tools``),
  and `make_policy` builds a `ReferenceLearner` from a published parameter blob. `FarmRun` drives a farm: it submits
  one task per world, stores every transition in an `ExperienceStore`, updates the learner from the stored
  experience and publishes new parameters, and measures how many environments really ran at the same time
  (overlapping worker intervals, distinct worker processes).
"""
from __future__ import annotations

import time
from typing import Any

from .modes import OBSERVATION_ONLY
from .store import FORMAT


class VecExperience:
    def __init__(self, envs: list):
        if not envs:
            raise ValueError("at least one environment")
        self.envs = list(envs)
        self.n = len(self.envs)
        self.agents = [e.agents()[0] for e in self.envs]

    def observe(self) -> list[dict]:
        return [e.observe(a) for e, a in zip(self.envs, self.agents)]

    def step(self, actions: list[dict | None], seconds: int | list[int]) -> tuple[list[dict | None], list[float], list[dict]]:
        """Execute one action (or None) per environment, advance each by `seconds`, return (results, rewards,
        next observations)."""
        secs = seconds if isinstance(seconds, list) else [int(seconds)] * self.n
        results = []
        for e, a, act in zip(self.envs, self.agents, actions):
            results.append(e.execute(a, act) if act is not None else None)
        for e, s in zip(self.envs, secs):
            e.advance(int(s))
        rewards = [float(e.reward(a)) for e, a in zip(self.envs, self.agents)]
        return results, rewards, self.observe()

    def close(self) -> None:
        for e in self.envs:
            e.close()


# ====================================================================== RolloutFarm hooks (run inside workers)
class FarmEnv:
    """One agent of an `ExperienceEnv` as the farm's semi-MDP: a step executes the action (or waits), advances the
    world by the action's duration plus `think_s`, and returns the next observation and the mode's reward."""

    def __init__(self, ref: dict, think_s: int | None = None):
        from .registry import make
        kw = dict(ref.get("kw") or {})
        self.env = make(int(ref["tier"]), ref["name"], int(ref.get("seed", 0)), ref.get("reward_mode",
                                                                                         OBSERVATION_ONLY), **kw)
        self.actor = self.env.agents()[0]
        self.think_s = int(think_s or ref.get("think_s") or getattr(self.env, "tick_s", 600))

    def reset(self) -> dict:
        return self.env.observe(self.actor)

    def tools(self) -> list[dict]:
        return self.env.tools(self.actor)

    def step(self, action: dict | None):
        t0 = self.env.now
        res = self.env.execute(self.actor, action) if action is not None else None
        self.env.advance(self.think_s)
        r = float(self.env.reward(self.actor))
        done = bool(getattr(self.env, "done", False))
        return self.env.observe(self.actor), r, done, {"sim_dt": self.env.now - t0, "result": res, "t": self.env.now}

    def close(self) -> None:
        self.env.close()


def make_world(ref: dict, **kwargs) -> FarmEnv:
    return FarmEnv(ref, **kwargs)


def make_policy(params: bytes, **kwargs):
    from .reference import ReferenceLearner
    L = ReferenceLearner(**kwargs)
    L.load_params(params)
    return L


# ====================================================================== the consumer side
class FarmRun:
    """Process-parallel experience collection and learning on any tier through `RolloutFarm`.

    ``FarmRun(learner, worlds, n_workers=4, steps_per_task=32, store=store).run(rounds=2)`` submits one task per
    world per round (every world continues in its worker), records every transition, updates `learner` from the
    stored experience after each round and publishes the new parameters."""

    def __init__(self, learner, worlds: list[dict], *, n_workers: int = 4, steps_per_task: int = 32, store=None,
                 policy_kwargs: dict | None = None, think_s: int | None = None, start_method: str = "spawn",
                 timeout_s: float = 600.0):
        from ..scale.rollout import RolloutFarm
        self.timeout_s = float(timeout_s)
        self.learner = learner
        self.worlds = [dict(w, key=w.get("key") or f"w{i}") for i, w in enumerate(worlds)]
        self.store = store
        self.steps_per_task = int(steps_per_task)
        kw = dict(policy_kwargs or {})
        kw.setdefault("reward_mode", getattr(learner, "reward_mode", OBSERVATION_ONLY))
        kw.setdefault("seed", getattr(learner, "seed", 0))
        self.farm = RolloutFarm(n_workers, make_world="crucible.experience.parallel:make_world",
                                make_policy="crucible.experience.parallel:make_policy",
                                world_kwargs={"think_s": think_s} if think_s else {}, policy_kwargs=kw,
                                params=learner.params_blob(), start_method=start_method, policy_scope="world",
                                max_live_worlds=max(4, len(self.worlds)), name="experience")
        self.intervals: list[tuple[float, float, str, int]] = []
        self.rollouts = 0
        self.steps = 0
        self.failures: list[dict] = []
        self.update_log: list[dict] = []

    def run(self, rounds: int = 1) -> dict:
        for _ in range(int(rounds)):
            ids = [self.farm.submit(w, steps=self.steps_per_task, seed=int(w.get("seed", 0))) for w in self.worlds]
            got = []
            for _ in ids:
                r = self.farm.get(timeout=self.timeout_s)
                arrive = time.monotonic()
                if not r.ok:
                    self.failures.append({"kind": r.kind, "error": r.error, "key": r.world_key})
                    continue
                self.intervals.append((arrive - r.wall_s, arrive, r.world_key, r.worker_pid))
                got.append(r)
            records = []
            for r in got:
                ref = next(w for w in self.worlds if w["key"] == r.world_key)
                recs = rollout_records(r, ref)
                records.extend(recs)
                if self.store is not None:
                    for rec in recs:
                        self.store.append(rec)
                self.rollouts += 1
                self.steps += r.steps
            if records and hasattr(self.learner, "learn_from_records"):
                u = self.learner.learn_from_records(records)
                self.update_log.append(u)
                self.farm.publish(self.learner.params_blob())
        return self.summary()

    def concurrency(self) -> int:
        """The largest number of environments whose worker intervals overlap (they ran at the same time)."""
        pts = sorted([(a, 1) for a, _, _, _ in self.intervals] + [(b, -1) for _, b, _, _ in self.intervals],
                     key=lambda x: (x[0], x[1]))
        cur = best = 0
        for _, d in pts:
            cur += d
            best = max(best, cur)
        return best

    def summary(self) -> dict:
        return {"worlds": len(self.worlds), "rollouts": self.rollouts, "steps": self.steps,
                "max_concurrent_environments": self.concurrency(),
                "worker_pids": sorted({p for _, _, _, p in self.intervals}), "failures": self.failures,
                "updates": int(sum(u.get("updates", 0) for u in self.update_log)),
                "params_versions": self.farm.version}

    def wait_ready(self, timeout: float = 180.0) -> float:
        return self.farm.wait_ready(timeout)

    def close(self) -> None:
        self.farm.close()


def rollout_records(r, ref: dict) -> list[dict]:
    """Experience records (``crucible-exp/1``) of one farm rollout."""
    out = []
    env = {"tier": int(ref["tier"]), "kind": "farm", "name": ref["name"], "seed": int(ref.get("seed", 0)),
           "world_id": ref["key"], "spec_hash": None}
    for i in range(r.steps):
        obs = r.obs[i] if r.obs else None
        out.append({"format": FORMAT, "env": env, "reward_mode": ref.get("reward_mode", OBSERVATION_ONLY),
                    "agent": (obs or {}).get("agent"), "step": i, "t": int((obs or {}).get("t", 0)),
                    "observation": obs, "action": r.actions[i], "result": r.results[i], "reward": float(r.rewards[i]),
                    "next_cursor": None, "learner": {"name": "reference", "params_hash": r.params_hash,
                                                     "param_version": r.param_version}})
    return out
