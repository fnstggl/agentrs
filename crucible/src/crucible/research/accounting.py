"""Cost accounting: what a learner consumed to get where it is, counted by the evaluator.

Every quantity is kept separately and none is ever folded into a score. Two buckets are kept apart for every run:
`train` (interaction the learner may learn from) and `eval` (frozen evaluation, which is the experimenter's cost,
not the learner's experience). A `Cost` holds, for one bucket:

    env_steps          environment ticks stepped (one per live slot per tick), counted by the harness
    episodes           episodes finished
    updates            optimiser steps: the change of `learner.usage().updates` over the window if the learner
                       maintains it, otherwise the sum of the `"updates"` counters its `update()` calls returned
                       (both are stored: `updates_reported`, `updates_returned`)
    train_cpu_s        process CPU seconds spent inside `observe` and `update`   (measured here)
    infer_cpu_s        process CPU seconds spent inside `act`                    (measured here)
    train_wall_s, infer_wall_s   the same intervals on the wall clock
    train_gpu_s, infer_gpu_s     GPU seconds: only a learner can know these, so they are the change of its
                       `usage()` over the window (0 for a learner that reports none)
    env_sim_s          seconds inside the simulators (the environment's own timer)
    infer_tokens, search_evals   language-model tokens and inference-time search evaluations (`usage()` deltas)
    sim_cost_cents, sim_revenue_cents, ticks_fallback, ticks_idle
                       simulated resources: money the operator spent and earned in the worlds, and ticks it could
                       not pay for its own settings

CPU time is `time.process_time()`: all threads of this process, not children. A learner that computes in child
processes or on a GPU must report it through `usage()`; what it reports is kept verbatim under `reported` next to
what was measured, and never substituted for it.

`compute_s` is the learner-side compute used as the x-axis of "performance against compute":
train_cpu_s + infer_cpu_s + train_gpu_s + infer_gpu_s. Simulator time is not part of it (it is the same for every
learner at equal environment steps) and is reported on its own.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields

import numpy as np

_USAGE_FIELDS = ("env_steps", "updates", "train_cpu_s", "train_gpu_s", "infer_cpu_s", "infer_gpu_s", "infer_tokens",
                 "search_evals")


@dataclass
class Cost:
    env_steps: int = 0
    episodes: int = 0
    updates: int = 0
    updates_returned: int = 0
    updates_reported: int = 0
    train_cpu_s: float = 0.0
    train_wall_s: float = 0.0
    train_gpu_s: float = 0.0
    infer_cpu_s: float = 0.0
    infer_wall_s: float = 0.0
    infer_gpu_s: float = 0.0
    env_sim_s: float = 0.0
    infer_tokens: int = 0
    search_evals: int = 0
    sim_cost_cents: float = 0.0
    sim_revenue_cents: float = 0.0
    ticks_fallback: int = 0
    ticks_idle: int = 0
    reported: dict = field(default_factory=dict)      # the learner's own usage() deltas over the same windows

    @property
    def compute_s(self) -> float:
        return float(self.train_cpu_s + self.infer_cpu_s + self.train_gpu_s + self.infer_gpu_s)

    @property
    def sim_net_cents(self) -> float:
        return float(self.sim_revenue_cents - self.sim_cost_cents)

    def add(self, other: "Cost") -> "Cost":
        for f in fields(self):
            if f.name == "reported":
                for k, v in other.reported.items():
                    self.reported[k] = self.reported.get(k, 0) + v
            else:
                setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["compute_s"] = self.compute_s
        d["sim_net_cents"] = self.sim_net_cents
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Cost":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def copy(self) -> "Cost":
        return Cost.from_dict(self.to_dict())


def usage_snapshot(learner) -> dict:
    """The learner's own counters right now (missing or malformed fields read as 0)."""
    try:
        u = learner.usage()
    except Exception:                                    # noqa: BLE001 - accounting never takes a run down
        return {k: 0 for k in _USAGE_FIELDS}
    out = {}
    for k in _USAGE_FIELDS:
        v = getattr(u, k, 0)
        try:
            out[k] = float(v) if np.isfinite(float(v)) else 0.0
        except (TypeError, ValueError):
            out[k] = 0.0
    return out


def params_of(learner) -> int:
    try:
        return int(getattr(learner.usage(), "params", 0) or 0)
    except Exception:                                    # noqa: BLE001
        return 0


class Meter:
    """Routes every call the harness makes to a learner and charges it to one `Cost`.

        m = Meter(learner, cost)
        a = m.act(obs, explore)
        m.observe(obs, a, rew, done, nxt); m.update()
        m.close()          # folds the learner's own usage() deltas (updates, GPU seconds, tokens, search) in

    One meter covers one window; `close` must be called once (it is idempotent)."""

    def __init__(self, learner, cost: Cost):
        self.learner, self.cost = learner, cost
        self._u0 = usage_snapshot(learner)
        self._returned = 0
        self._open = True

    def act(self, obs: np.ndarray, explore: bool) -> np.ndarray:
        c0, w0 = time.process_time(), time.perf_counter()
        a = self.learner.act(obs, explore=explore)
        self.cost.infer_cpu_s += time.process_time() - c0
        self.cost.infer_wall_s += time.perf_counter() - w0
        return a

    def observe(self, obs, action, reward, done, next_obs) -> None:
        c0, w0 = time.process_time(), time.perf_counter()
        self.learner.observe(obs, action, reward, done, next_obs)
        self.cost.train_cpu_s += time.process_time() - c0
        self.cost.train_wall_s += time.perf_counter() - w0

    def update(self) -> int:
        c0, w0 = time.process_time(), time.perf_counter()
        out = self.learner.update()
        self.cost.train_cpu_s += time.process_time() - c0
        self.cost.train_wall_s += time.perf_counter() - w0
        try:
            n = int((out or {}).get("updates", 0))
        except (TypeError, ValueError, AttributeError):
            n = 0
        self._returned += max(0, n)
        return n

    def close(self) -> Cost:
        if not self._open:
            return self.cost
        self._open = False
        u1 = usage_snapshot(self.learner)
        d = {k: u1[k] - self._u0[k] for k in _USAGE_FIELDS}
        c = self.cost
        reported = int(round(max(0.0, d["updates"])))
        c.updates_returned += self._returned
        c.updates_reported += reported
        c.updates += reported if reported > 0 else self._returned
        c.train_gpu_s += max(0.0, d["train_gpu_s"])
        c.infer_gpu_s += max(0.0, d["infer_gpu_s"])
        c.infer_tokens += int(round(max(0.0, d["infer_tokens"])))
        c.search_evals += int(round(max(0.0, d["search_evals"])))
        for k in ("train_cpu_s", "infer_cpu_s", "env_steps"):
            if d[k]:
                c.reported[k] = c.reported.get(k, 0) + d[k]
        return c


def charge_episodes(cost: Cost, records: list[dict], sim_seconds: float) -> None:
    """Add what a batch of finished episodes consumed in the worlds."""
    cost.env_sim_s += float(sim_seconds)
    for r in records:
        cost.episodes += 1
        cost.env_steps += int(r["ticks"])
        cost.sim_cost_cents += float(r["cost_cents"])
        cost.sim_revenue_cents += float(r["revenue_cents"])
        cost.ticks_fallback += int(r["ticks_fallback"])
        cost.ticks_idle += int(r["ticks_idle"])


COST_COLUMNS = ("env_steps", "episodes", "updates", "train_cpu_s", "infer_cpu_s", "train_gpu_s", "infer_gpu_s",
                "env_sim_s", "infer_tokens", "search_evals", "compute_s", "sim_cost_cents", "sim_revenue_cents",
                "sim_net_cents", "ticks_fallback", "ticks_idle")


def cost_row(cost: Cost | dict) -> dict:
    """The reporting columns of a cost, in a fixed order, followed by the provenance of the update count and
    by what the learner reported about itself over the same windows (`reported`, kept apart from what was
    measured)."""
    d = cost.to_dict() if isinstance(cost, Cost) else dict(cost)
    row = {k: d.get(k, 0) for k in COST_COLUMNS}
    row.update(updates_returned=d.get("updates_returned", 0), updates_reported=d.get("updates_reported", 0),
               reported=dict(d.get("reported") or {}))
    return row
