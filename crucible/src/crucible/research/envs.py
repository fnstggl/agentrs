"""Batched numeric environments over hidden-mechanism tasks.

`MechVecEnv(tasks)` runs one kernel-free simulator per task (the simulator is bit-identical to the kernel-hosted
facility, so nothing here is a different environment from the one a tool-using agent sees; `research.bridge`
shows that by running the same learner through `rig.*` tool calls). Observations and actions follow
`research.learner` (padded and masked to one size for every mechanism).

Reward is the tick's net income in units of the task's public money scale, which is the operator's own starting
balance divided by `tasks.ENDOWMENT_TICKS`: a dimensionless number computed only from what the operator's own
account shows. Nothing private enters an observation or a reward. Episode summaries (`MechVecEnv.finished`) are the operator-side record: total net, the
normalised score against the task's private reference, whether the public objective was reached, and how many
ticks ran at fallback settings or idle because the operator could not pay.

An episode is one pass over the task's horizon. `step` reports `done` when the horizon is reached; the caller then
calls `reset(which=done)` (optionally with new tasks for those slots) before stepping those slots again.

Reward modes (Crucible 5.0, `crucible.experience.modes`; Clarification 2): the default is ``"observation_only"``,
which removes every reward path: `step` returns zeros and the reward
slot of the observation (scalars[0]) stays 0; revenue, cost, funds and time (scalars[1:5]) are unchanged, because
they are what the operator's own account shows. ``reward_mode="legacy_shaped_reward"``, selected explicitly, is the
4.0 behaviour byte for byte (historical baselines). ``"verified_terminal_outcome"`` returns 0 on every tick and, on the
tick that ends an episode, 1.0 if the public objective was reached (`Task.success`) and 0.0 otherwise; the reward
slot of the observation stays 0 as well.
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from ..experience.modes import LEGACY, VERIFIED, check_mode
from .learner import EnvSpec
from .tasks import ENDOWMENT_TICKS, Task


def symlog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def money_scale(task: Task) -> float:
    """The public unit of money for a task: its starting balance / ENDOWMENT_TICKS (cents)."""
    return max(1.0, float(task.endowment) / ENDOWMENT_TICKS)


class MechVecEnv:
    def __init__(self, tasks: Sequence[Task], spec: EnvSpec | None = None, seat: int = 0, money: bool = True,
                 reward_mode: str | None = None):
        self.reward_mode = check_mode(reward_mode)
        self.spec = spec or EnvSpec()
        self.seat = int(seat)
        self.money = bool(money)
        self.n = len(tasks)
        self.tasks: list[Task] = list(tasks)
        self.sims = [None] * self.n
        self.finished: list[dict] = []            # operator-side episode records, appended as episodes end
        self.sim_seconds = 0.0                    # wall-clock spent inside the simulators (environment cost)
        self.steps = 0
        sp = self.spec
        self._lo = np.zeros((self.n, sp.max_levers))
        self._hi = np.ones((self.n, sp.max_levers))
        self._lmask = np.zeros((self.n, sp.max_levers), dtype=np.float32)
        self._gmask = np.zeros((self.n, sp.max_gauges), dtype=np.float32)
        self._names: list[list[str]] = [[] for _ in range(self.n)]
        self._gnames: list[list[str]] = [[] for _ in range(self.n)]
        self._scale = np.ones(self.n)
        self._h = np.ones(self.n, dtype=np.int64)
        self._ep = [None] * self.n
        self._obs = np.zeros((self.n, sp.obs_dim), dtype=np.float32)
        self.done = np.ones(self.n, dtype=bool)   # every slot needs a reset before its first step
        for i, t in enumerate(self.tasks):
            self._install(i, t)

    # ------------------------------------------------------------------ construction
    def _install(self, i: int, task: Task) -> None:
        sp = self.spec
        sim = task.sim(money=self.money)
        levers = sim.engine.levers(self.seat)
        gauges = sim.engine.gauges(self.seat)
        if len(levers) > sp.max_levers or len(gauges) > sp.max_gauges:
            raise ValueError(f"task {task.id} has {len(levers)} levers and {len(gauges)} gauges; the EnvSpec allows "
                             f"{sp.max_levers} and {sp.max_gauges}")
        self.tasks[i], self.sims[i] = task, sim
        self._names[i] = [lv.name for lv in levers]
        self._gnames[i] = [g.name for g in gauges]
        self._lo[i], self._hi[i], self._lmask[i], self._gmask[i] = 0.0, 1.0, 0.0, 0.0
        for j, lv in enumerate(levers):
            self._lo[i, j], self._hi[i, j], self._lmask[i, j] = lv.lo, lv.hi, 1.0
        self._gmask[i, :len(gauges)] = 1.0
        self._scale[i] = money_scale(task)
        self._h[i] = int(task.horizon)

    def _norm_settings(self, i: int) -> np.ndarray:
        out = np.zeros(self.spec.max_levers, dtype=np.float32)
        s = self.sims[i].settings[self.seat]
        for j, name in enumerate(self._names[i]):
            span = self._hi[i, j] - self._lo[i, j]
            out[j] = 0.0 if span <= 0 else 2.0 * (s[name] - self._lo[i, j]) / span - 1.0
        return out

    def _write_obs(self, i: int, readings: dict | None, reward: float, rev: float, cost: float) -> None:
        sp, o = self.spec, self._obs[i]
        o[:] = 0.0
        if readings is not None:
            g = np.array([readings[name] for name in self._gnames[i]], dtype=np.float64)
            o[sp.gauges][:len(g)] = symlog(g)
        o[sp.gauge_mask] = self._gmask[i]
        o[sp.last_action] = self._norm_settings(i)
        o[sp.lever_mask] = self._lmask[i]
        sim, task = self.sims[i], self.tasks[i]
        funds = sim.funds[self.seat]
        sc = o[sp.scalars]
        sc[0], sc[1], sc[2] = reward, rev / self._scale[i], cost / self._scale[i]
        sc[3] = 1.0 if funds is None else float(funds) / max(1.0, float(task.endowment))
        sc[4] = sim.t / float(self._h[i])

    # ------------------------------------------------------------------ gym-like API
    def reset(self, which: np.ndarray | Sequence[int] | None = None, tasks: Sequence[Task] | None = None) -> np.ndarray:
        """Start a new episode in every slot, or in the slots selected by `which` (a boolean mask or indices).
        `tasks` (one per selected slot) replaces the tasks of those slots; otherwise the same task is replayed
        from its start with the same noise."""
        if which is None:
            idx = list(range(self.n))
        else:
            w = np.asarray(which)
            idx = [int(i) for i in (np.flatnonzero(w) if w.dtype == bool else w)]
        if tasks is not None and len(tasks) != len(idx):
            raise ValueError("one task per reset slot")
        for k, i in enumerate(idx):
            if tasks is not None:
                self._install(i, tasks[k])
            else:
                self.sims[i].reset()
            self._ep[i] = {"net": 0.0, "revenue": 0.0, "cost": 0.0, "score": 0.0, "fallback": 0, "idle": 0,
                           "nets": []}
            self.done[i] = False
            self._write_obs(i, None, 0.0, 0.0, 0.0)
        return self._obs.copy()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """`actions`: [n, max_levers] in [-1, 1]. Slots whose episode is over are not stepped (reward 0, done)."""
        a = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        if a.shape != (self.n, self.spec.max_levers):
            raise ValueError(f"actions must have shape {(self.n, self.spec.max_levers)}, got {a.shape}")
        values = self._lo + (a + 1.0) * 0.5 * (self._hi - self._lo)
        return self.step_values(values)

    def step_values(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """`step` with lever values in their own units ([n, max_levers]; the simulator clips them to range, as
        `rig.set` does). `crucible.experience` uses it so that a setting chosen through a tool runs exactly."""
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.n, self.spec.max_levers):
            raise ValueError(f"values must have shape {(self.n, self.spec.max_levers)}, got {values.shape}")
        reward = np.zeros(self.n, dtype=np.float32)
        net = np.zeros(self.n)
        t0 = time.perf_counter()
        for i in range(self.n):
            if self.done[i]:
                continue
            sim = self.sims[i]
            x = {name: float(values[i, j]) for j, name in enumerate(self._names[i])}
            res = sim.step(x if sim.n_seats == 1 else [x if s == self.seat else None for s in range(sim.n_seats)])
            rev, cost = float(res.revenue[self.seat]), float(res.cost[self.seat])
            net[i] = rev - cost
            legacy = self.reward_mode == LEGACY
            if legacy:
                reward[i] = net[i] / self._scale[i]
            ep = self._ep[i]
            ep["net"] += net[i]
            ep["revenue"] += rev
            ep["cost"] += cost
            ep["score"] += float(res.score[self.seat])
            ep["nets"].append(net[i])
            status = sim.last[self.seat]["status"]
            if status in ("fallback", "idle"):
                ep[status] += 1
            self._write_obs(i, res.readings[self.seat], float(reward[i]) if legacy else 0.0, rev, cost)
            self.steps += 1
            if sim.t >= self._h[i]:
                self.done[i] = True
                self._finish(i)
                if self.reward_mode == VERIFIED:
                    reward[i] = 1.0 if self.finished[-1]["success"] else 0.0
        self.sim_seconds += time.perf_counter() - t0
        return self._obs.copy(), reward, self.done.copy(), {"net_cents": net}

    def _finish(self, i: int) -> None:
        task, ep = self.tasks[i], self._ep[i]
        h = int(self._h[i])
        nets = np.asarray(ep["nets"], dtype=float)
        half = h // 2
        rec = {"task": task.id, "family": task.family, "seed": task.seed, "engine": task.engine,
               "transfer": task.transfer, "split": task.split, "ticks": h, "net_cents": float(ep["net"]),
               "net_per_tick": float(ep["net"] / h), "revenue_cents": float(ep["revenue"]),
               "cost_cents": float(ep["cost"]), "objective_quantity": float(ep["score"]),
               "ticks_fallback": int(ep["fallback"]), "ticks_idle": int(ep["idle"]),
               "funds_end_cents": self.sims[i].funds[self.seat],
               "score": task.score(float(ep["net"] / h)), "success": task.success(float(ep["net"])),
               # within-episode adaptation: the same normalisation applied to each half of the episode
               "score_first_half": task.score(float(nets[:half].mean())) if half else None,
               "score_second_half": task.score(float(nets[half:].mean())) if h - half else None,
               "random_score": None if task.random_net is None else task.score(task.random_net)}
        self.finished.append(rec)

    def drain(self) -> list[dict]:
        """Episode records finished since the last call."""
        out, self.finished = self.finished, []
        return out

    # ------------------------------------------------------------------ public facts a learner may use
    def lever_mask(self) -> np.ndarray:
        return self._lmask.copy()

    def public(self) -> list[dict]:
        return [t.public() for t in self.tasks]


def run_episodes(env: MechVecEnv, learner, explore: bool = False, learn: bool = False) -> list[dict]:
    """Run every slot of `env` for one episode with `learner` and return the episode records (in slot order).
    `learn=True` feeds the experience to `learner.observe` (it still decides when to `update`)."""
    obs = env.reset()
    learner.reset_context(env.n)
    t0 = time.perf_counter()
    while not env.done.all():
        act = learner.act(obs, explore=explore)
        nxt, rew, done, _ = env.step(act)
        if learn and not learner.frozen:
            learner.observe(obs, act, rew, done, nxt)
        obs = nxt
    u = learner.usage()
    u.env_steps += int(sum(int(env._h[i]) for i in range(env.n)))
    wall = time.perf_counter() - t0
    recs = env.drain()
    order = {t.id: k for k, t in enumerate(env.tasks)}
    recs.sort(key=lambda r: order[r["task"]])
    for r in recs:
        r["wall_s_batch"] = wall
    return recs
