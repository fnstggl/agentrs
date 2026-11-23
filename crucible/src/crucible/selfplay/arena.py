"""Multi-seat arenas: several independent individuals operating the seats of one hidden mechanism.

`SelfPlayEnv(tasks, spec)` runs one kernel-free simulator per task with **every seat operated**. Each seat gets its
own observation in the `research.learner` layout, built only from what that seat's operator can see through
`rig.read` and `bank.accounts`: its own gauges, its own settings, its own revenue, cost and balance, and the clock.
Each seat gets its own reward: its own net income of the tick in the task's public money unit. Nothing of another
seat's private view (its gauges, settings, money, reward) enters an observation; whatever one seat learns about
another it learns through the mechanism (for instance the rival's lagged price that a market shows on a gauge).
The engine's private diagnostics are kept on the operator side (`SelfPlayEnv.finished`, `SelfPlayEnv.trace`) for
analysis and audit; they never reach a learner.

`KernelSelfPlayEnv(task, spec)` is the same match played in a kernel world: the participants are admitted with
`admin.join`, hold their seats, and every setting goes through `rig.set`, every reading comes from `rig.read`,
every balance from `bank.accounts`. `tests/test_selfplay.py` requires both seats' observations and rewards to be
equal to `SelfPlayEnv` tick for tick, so a self-play result is a statement about recorded actions in a shared
world, not about a private simulator.

An `Individual` is one self-play agent, and owns separately everything that makes it one: its learner (policy,
weights, recurrent or episodic state, replay or rollout buffer), its experience history, its resource budget (its
own funds; an individual that goes broke idles by the facility's money rules), its persistent identity and lineage,
and its own objective. `play` runs individuals against each other; nothing is shared between them but the world.

One individual lives in one persistent world: the hidden mechanism of a task. Matches in that world differ in
their per-tick noise (`with_noise`), never in the mechanism, so what an individual has learned stays meaningful
from match to match. Claims are made over several worlds and learner seeds.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from typing import Any, Callable, Sequence

import numpy as np

from ..core.canonical import digest
from ..research.envs import money_scale
from ..research.learner import EnvSpec, Learner
from ..research.tasks import Task

OBJECTIVES = ("own", "joint", "score", "mixed")


# ====================================================================== tasks
def with_noise(task: Task, k: int) -> Task:
    """The same hidden mechanism in a different world: noise index `k` (0 is the task itself). Per-tick noise is
    a function of the world seed, so two matches with the same `k` share every random number of the mechanism
    (paired comparisons), and different `k` are independent replications."""
    if int(k) == 0:
        return task
    seed = int(digest(["selfplay-noise", task.family, int(task.seed), int(k)])[:8], 16)
    return dataclasses.replace(task, world_seed=seed)


def arena_task(family: str, seed: int) -> Task:
    """An unvalidated multi-seat task (self-play results are comparisons between conditions on the same task, so
    the single-seat reference normalisation of validated tasks is not needed)."""
    from ..research.tasks import make_task
    return make_task(family, int(seed), validate=False)


# ====================================================================== private diagnostics (operator side)
_TECH_KEYS = ("arrivals", "completed", "late", "dropped", "shed", "value_arrived", "value_completed", "value_lost",
              "pool_util", "slowdown")
_ABM_SEAT = ("units", "orders", "price", "quality", "active")
_ABM_TOTAL = ("units_seats", "units_rivals", "units_market", "paid_seats", "paid_market", "income_credited", "pool")


def private_names(engine: str, info: dict, n_seats: int) -> list[str]:
    """Names of the quantities `private_values` returns for this engine, in order."""
    if engine == "abm":
        return [f"{k}_{s}" for s in range(n_seats) for k in _ABM_SEAT] + list(_ABM_TOTAL)
    if engine == "techsys":
        nl = len(info.get("lines", []))
        return list(_TECH_KEYS) + [f"completed_line_{i}" for i in range(nl)] + [f"arrivals_line_{i}" for i in range(nl)]
    return [k for k in sorted(info) if isinstance(info[k], (int, float)) and not isinstance(info[k], bool)]


def private_values(engine: str, info: dict, n_seats: int) -> list[float]:
    """The real quantities behind a tick, from the engine's private diagnostics: what customers were served, what
    they paid, what work was completed. Operator side only (audit and analysis); never part of an observation."""
    if engine == "abm":
        served, price, orders, qual, active = info["served"], info["price"], info["orders"], info["quality"], info["active"]
        out = []
        for s in range(n_seats):
            out += [served[s], orders[s], price[s], qual[s], active[s]]
        rev = info["revenue_all"]
        us, um = sum(served[:n_seats]), sum(served)
        out += [us, um - us, um, sum(rev[:n_seats]), sum(rev), info.get("income_credited", 0.0), info.get("pool", 1.0)]
        return out
    if engine == "techsys":
        lines = info.get("lines", [])
        return [float(info.get(k) or 0.0) for k in _TECH_KEYS] + [ln["completed"] for ln in lines] + \
            [ln["arrivals"] for ln in lines]
    return [float(info[k]) for k in sorted(info) if isinstance(info[k], (int, float)) and not isinstance(info[k], bool)]


def private_quantities(engine: str, info: dict, n_seats: int) -> dict[str, float]:
    """`private_values` under their names."""
    return {k: float(v) for k, v in zip(private_names(engine, info, n_seats), private_values(engine, info, n_seats))}


_LEVELS = ("price_", "quality_", "active_", "pool", "pool_util", "slowdown")     # averaged; everything else summed


def _is_level(key: str) -> bool:
    return key.startswith(_LEVELS) or key in _LEVELS


def _symlog(v: float) -> float:
    return math.copysign(math.log1p(abs(v)), v)


# ====================================================================== the kernel-free arena
_ENGINES: dict[tuple[str, int], Any] = {}


def _sim(task: Task, funds: Sequence[int | None]):
    """`task.sim(held=all, funds=funds)` with the compiled engine shared between the matches of one world (an
    engine instance holds lookup tables compiled from its config and no state, so sharing it changes nothing;
    compiling an agent-based market costs several milliseconds)."""
    from ..mechanisms.base import build_engine
    from ..mechanisms.sim import MechSim
    key = (task.family, int(task.mech_seed))
    eng = _ENGINES.get(key)
    mech = task.mechanism()
    if eng is None:
        if len(_ENGINES) > 256:
            _ENGINES.clear()
        eng = _ENGINES[key] = build_engine(mech)
    return MechSim(mech, task.world_seed, "r1", income_cents_per_tick=task.income, funds=list(funds),
                   held=[True] * int(task.n_seats), engine=eng)


class SelfPlayEnv:
    """`n = len(tasks)` parallel matches. All tasks must have the same number of seats.

    `reset(...)` returns one observation array `[n, obs_dim]` per seat; `step(actions)` takes one action array
    `[n, max_levers]` in [-1, 1] per seat (None: that seat keeps its settings) and returns
    `(observations per seat, rewards [seats, n], done [n], info)`.
    """

    def __init__(self, tasks: Sequence[Task], spec: EnvSpec | None = None, funds: Sequence[Sequence[int | None]] | None = None,
                 trace: bool = False):
        self.spec = spec or EnvSpec()
        self.n = len(tasks)
        if self.n == 0:
            raise ValueError("at least one task")
        seats = {int(t.n_seats) for t in tasks}
        if len(seats) != 1:
            raise ValueError(f"all tasks of an arena need the same number of seats, got {sorted(seats)}")
        self.n_seats = seats.pop()
        self.keep_trace = bool(trace)
        sp, n, S = self.spec, self.n, self.n_seats
        self.tasks: list[Task] = list(tasks)
        self.sims: list[Any] = [None] * n
        self.finished: list[dict] = []
        self.trace: list[list[dict]] = [[] for _ in range(n)]
        self.sim_seconds = 0.0
        self.steps = 0
        self._lo = np.zeros((S, n, sp.max_levers))
        self._hi = np.ones((S, n, sp.max_levers))
        self._lmask = np.zeros((S, n, sp.max_levers), dtype=np.float32)
        self._gmask = np.zeros((S, n, sp.max_gauges), dtype=np.float32)
        self._names: list[list[list[str]]] = [[[] for _ in range(n)] for _ in range(S)]
        self._gnames: list[list[list[str]]] = [[[] for _ in range(n)] for _ in range(S)]
        self._scale = [1.0] * n
        self._h = np.ones(n, dtype=np.int64)
        self._hf = [1.0] * n
        self._endow = [1.0] * n
        self._engine = [""] * n
        self._lv: list[list[list]] = [[[] for _ in range(n)] for _ in range(S)]
        self._row: list[list[list]] = [[[] for _ in range(n)] for _ in range(S)]
        self._pnames: list[list[str] | None] = [None] * n
        self._funds0: list[list[int | None]] = [[None] * S for _ in range(n)]
        self._ep: list[dict | None] = [None] * n
        self._obs = [np.zeros((n, sp.obs_dim), dtype=np.float32) for _ in range(S)]
        self.done = np.ones(n, dtype=bool)
        for i, t in enumerate(self.tasks):
            self._install(i, t, None if funds is None else funds[i])

    # ------------------------------------------------------------------ construction
    def _install(self, i: int, task: Task, funds: Sequence[int | None] | None) -> None:
        sp, S = self.spec, self.n_seats
        if int(task.n_seats) != S:
            raise ValueError(f"task {task.id} has {task.n_seats} seats, this arena has {S}")
        f0 = [int(task.endowment)] * S if funds is None else [None if f is None else int(f) for f in funds]
        if len(f0) != S:
            raise ValueError("one starting balance per seat")
        sim = _sim(task, f0)
        self.tasks[i], self.sims[i], self._funds0[i] = task, sim, f0
        self._engine[i] = task.engine or sim.mech["engine"]
        for s in range(S):
            levers, gauges = sim.engine.levers(s), sim.engine.gauges(s)
            if len(levers) > sp.max_levers or len(gauges) > sp.max_gauges:
                raise ValueError(f"task {task.id} seat {s} has {len(levers)} levers and {len(gauges)} gauges; the "
                                 f"EnvSpec allows {sp.max_levers} and {sp.max_gauges}")
            self._names[s][i] = [lv.name for lv in levers]
            self._gnames[s][i] = [g.name for g in gauges]
            self._lo[s, i], self._hi[s, i], self._lmask[s, i], self._gmask[s, i] = 0.0, 1.0, 0.0, 0.0
            for j, lv in enumerate(levers):
                self._lo[s, i, j], self._hi[s, i, j], self._lmask[s, i, j] = lv.lo, lv.hi, 1.0
            self._gmask[s, i, :len(gauges)] = 1.0
            self._lv[s][i] = [(lv.name, float(lv.lo), float(lv.hi) - float(lv.lo)) for lv in levers]
            row = [0.0] * sp.obs_dim
            row[sp.gauge_mask] = [float(v) for v in self._gmask[s, i]]
            row[sp.lever_mask] = [float(v) for v in self._lmask[s, i]]
            self._row[s][i] = row
        self._scale[i] = money_scale(task)
        self._h[i] = int(task.horizon)
        self._hf[i] = float(task.horizon)
        self._endow[i] = max(1.0, float(task.endowment))
        self._pnames[i] = None

    def _norm_settings(self, s: int, i: int) -> list[float]:
        cur = self.sims[i].settings[s]
        return [(2.0 * (cur[name] - lo) / span - 1.0) if span > 0 else 0.0 for name, lo, span in self._lv[s][i]]

    def _write_obs(self, s: int, i: int, readings: dict | None, reward: float, rev: float, cost: float,
                   norm: list[float] | None = None) -> None:
        """Seat `s` of slot `i`: built from that seat's own record only (the arguments and the simulator's
        per-seat settings and balance). No other seat's reading, setting or money is read here."""
        sp = self.spec
        row = self._row[s][i]                       # masks are prefilled; the rest is overwritten below
        names = self._gnames[s][i]
        if readings is None:
            for j in range(len(names)):
                row[j] = 0.0
        else:
            for j, name in enumerate(names):
                row[j] = _symlog(readings[name])
        base = 2 * sp.max_gauges
        for j, v in enumerate(norm if norm is not None else self._norm_settings(s, i)):
            row[base + j] = v
        sim = self.sims[i]
        funds = sim.funds[s]
        k = base + 2 * sp.max_levers
        scale = self._scale[i]
        row[k], row[k + 1], row[k + 2] = reward, rev / scale, cost / scale
        row[k + 3] = 1.0 if funds is None else float(funds) / self._endow[i]
        row[k + 4] = sim.t / self._hf[i]
        self._obs[s][i] = row

    # ------------------------------------------------------------------ gym-like API
    def reset(self, which: np.ndarray | Sequence[int] | None = None, tasks: Sequence[Task] | None = None,
              funds: Sequence[Sequence[int | None]] | None = None) -> list[np.ndarray]:
        """Start a new match in every slot, or in the slots selected by `which`. `tasks` (one per selected slot)
        replaces the tasks of those slots; `funds` (per selected slot, one starting balance per seat) replaces
        the starting balances. Without `tasks` the same task is replayed with the same noise."""
        if which is None:
            idx = list(range(self.n))
        else:
            w = np.asarray(which)
            idx = [int(i) for i in (np.flatnonzero(w) if w.dtype == bool else w)]
        if tasks is not None and len(tasks) != len(idx):
            raise ValueError("one task per reset slot")
        if funds is not None and len(funds) != len(idx):
            raise ValueError("one list of starting balances per reset slot")
        for k, i in enumerate(idx):
            if tasks is not None or funds is not None:
                self._install(i, tasks[k] if tasks is not None else self.tasks[i],
                              funds[k] if funds is not None else self._funds0[i])
            else:
                self.sims[i].reset()
            S = self.n_seats
            self._ep[i] = {"net": [0.0] * S, "revenue": [0.0] * S, "cost": [0.0] * S, "score": [0.0] * S,
                           "fallback": [0] * S, "idle": [0] * S,
                           "act": [[0.0] * len(self._names[s][i]) for s in range(S)], "private": None, "paid": 0.0}
            self.trace[i] = []
            self.done[i] = False
            for s in range(S):
                self._write_obs(s, i, None, 0.0, 0.0, 0.0)
        return [o.copy() for o in self._obs]

    def step(self, actions: Sequence[np.ndarray | None]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, dict]:
        S, n, sp = self.n_seats, self.n, self.spec
        if len(actions) != S:
            raise ValueError(f"one action array per seat ({S}), got {len(actions)}")
        values: list[np.ndarray | None] = []
        for s, a in enumerate(actions):
            if a is None:
                values.append(None)
                continue
            a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
            if a.shape != (n, sp.max_levers):
                raise ValueError(f"seat {s}: actions must have shape {(n, sp.max_levers)}, got {a.shape}")
            values.append(self._lo[s] + (a + 1.0) * 0.5 * (self._hi[s] - self._lo[s]))
        reward = np.zeros((S, n), dtype=np.float32)
        net, rev_a, cost_a, score = np.zeros((S, n)), np.zeros((S, n)), np.zeros((S, n)), np.zeros((S, n))
        status: list[list[str]] = [[""] * n for _ in range(S)]
        vals = [None if v is None else v.tolist() for v in values]
        t0 = time.perf_counter()
        for i in range(n):
            if self.done[i]:
                continue
            sim = self.sims[i]
            x = [None if vals[s] is None else dict(zip(self._names[s][i], vals[s][i])) for s in range(S)]
            res = sim.step(x)
            ep = self._ep[i]
            scale = self._scale[i]
            pv = private_values(self._engine[i], sim.info, S)
            if ep["private"] is None:
                self._pnames[i] = private_names(self._engine[i], sim.info, S)
                ep["private"] = [0.0] * len(pv)
            acc = ep["private"]
            for j, v in enumerate(pv):
                acc[j] += v
            last = sim.last
            for s in range(S):
                rev, cost = res.revenue[s], res.cost[s]
                d = rev - cost
                net[s, i], rev_a[s, i], cost_a[s, i], score[s, i] = d, rev, cost, res.score[s]
                reward[s, i] = d / scale
                ep["net"][s] += d
                ep["revenue"][s] += rev
                ep["cost"][s] += cost
                ep["score"][s] += res.score[s]
                st = last[s]["status"]
                status[s][i] = st
                if st != "ok" and st in ("fallback", "idle"):
                    ep[st][s] += 1
                norm = self._norm_settings(s, i)
                ea = ep["act"][s]
                for j, v in enumerate(norm):
                    ea[j] += v
                self._write_obs(s, i, res.readings[s], float(reward[s, i]), rev, cost, norm)
            ep["paid"] += sum(res.revenue)
            if self.keep_trace:
                self.trace[i].append({"t": sim.t - 1, "net": [float(net[s, i]) for s in range(S)],
                                      "revenue": [float(rev_a[s, i]) for s in range(S)],
                                      "cost": [float(cost_a[s, i]) for s in range(S)],
                                      "status": [status[s][i] for s in range(S)],
                                      "settings": [dict(last[s]["settings"]) for s in range(S)],
                                      "funds": list(sim.funds), "private": dict(zip(self._pnames[i], pv))})
            self.steps += 1
            if sim.t >= self._h[i]:
                self.done[i] = True
                self._finish(i)
        self.sim_seconds += time.perf_counter() - t0
        return ([o.copy() for o in self._obs], reward, self.done.copy(),
                {"net_cents": net, "revenue_cents": rev_a, "cost_cents": cost_a, "score": score, "status": status})

    def _finish(self, i: int) -> None:
        task, ep, S = self.tasks[i], self._ep[i], self.n_seats
        h = int(self._h[i])
        sim = self.sims[i]
        private = {}
        for key, v in zip(self._pnames[i] or [], ep["private"] or []):
            private[key] = float(v / h if _is_level(key) else v)
        seats = []
        for s in range(S):
            seats.append({"net_cents": float(ep["net"][s]), "net_per_tick": float(ep["net"][s] / h),
                          "revenue_cents": float(ep["revenue"][s]), "cost_cents": float(ep["cost"][s]),
                          "objective_quantity": float(ep["score"][s]), "ticks_fallback": int(ep["fallback"][s]),
                          "ticks_idle": int(ep["idle"][s]), "funds_start": self._funds0[i][s],
                          "funds_end": sim.funds[s],
                          "mean_action": [float(v / h) for v in ep["act"][s]]})
        bound = getattr(sim.engine, "revenue_bound", None)      # the engine's own statement of what its demand
        self.finished.append({"task": task.id, "family": task.family, "seed": task.seed, "world_seed": task.world_seed,
                              "engine": self._engine[i], "split": task.split, "ticks": h, "slot": i, "seats": seats,
                              "joint_net_per_tick": float(sum(ep["net"]) / h),
                              "paid_by_demand_cents": float(ep["paid"]),
                              "demand_income_cents": float(task.income) * h,
                              "revenue_bound_cents": float(bound(h)) if callable(bound) else None,
                              "private": private})

    def drain(self) -> list[dict]:
        """Operator-side match records finished since the last call (slot order)."""
        out, self.finished = sorted(self.finished, key=lambda r: r["slot"]), []
        return out

    # ------------------------------------------------------------------ public facts
    def lever_mask(self, seat: int) -> np.ndarray:
        return self._lmask[seat].copy()

    def normalise(self, seat: int, slot: int, settings: dict[str, float]) -> np.ndarray:
        """A complete or partial settings mapping of a seat as an action vector in [-1, 1] (unmentioned levers
        at their current value)."""
        out = np.zeros(self.spec.max_levers, dtype=np.float32)
        out[:len(self._names[seat][slot])] = self._norm_settings(seat, slot)
        for j, name in enumerate(self._names[seat][slot]):
            if name in settings:
                span = self._hi[seat, slot, j] - self._lo[seat, slot, j]
                out[j] = 0.0 if span <= 0 else 2.0 * (float(settings[name]) - self._lo[seat, slot, j]) / span - 1.0
        return np.clip(out, -1.0, 1.0)


# ====================================================================== the same match in a kernel world
class KernelSelfPlayEnv:
    """One match in a kernel world: every seat is held by an admitted participant who operates it with `rig.set`
    and `rig.read` and reads its balance with `bank.accounts`. Same observations, actions and rewards as one slot
    of `SelfPlayEnv` (tested tick for tick). The run is an ordinary recorded world: `calls` lists every tool call
    per participant, and the event log replays."""
    n = 1

    def __init__(self, task: Task, spec: EnvSpec | None = None, handles: Sequence[str] | None = None,
                 funds: Sequence[int] | None = None, store=None, strict: bool = True, extra_handles: Sequence[str] = ()):
        import crucible
        crucible.load_all_modules()
        from ..core.kernel import World
        from ..worldgen import generate
        self.task, self.spec = task, spec or EnvSpec()
        self.n_seats = S = int(task.n_seats)
        self.handles = list(handles) if handles is not None else [f"seat{s + 1}" for s in range(S)]
        if len(self.handles) != S or len(set(self.handles)) != S:
            raise ValueError("one distinct handle per seat")
        cfg, seed = task.world_config(holders=self.handles)
        self.world = World.create(generate(cfg, seed), world_id=f"selfplay-{task.family}-{task.seed}-{task.world_seed}",
                                  store=store, strict=strict)
        self.funds0 = [int(task.endowment)] * S if funds is None else [int(f) for f in funds]
        self.actors = []
        for h, f in zip(self.handles, self.funds0):
            self.actors.append(self.world.system_call("admin.join", {"handle": h, "endowment": f})["principal"])
        self.extra = {h: self.world.system_call("admin.join", {"handle": h, "endowment": int(task.endowment)})["principal"]
                      for h in extra_handles}
        self.calls: list[list[dict]] = [[] for _ in range(S)]
        self.levers, self.gauges = [], []
        self.tick_s = 3600
        for s in range(S):
            sp = self._call(s, "rig.spec", {})
            if sp["seat"] != s:
                raise RuntimeError(f"{self.handles[s]} was expected to hold seat {s}, holds {sp['seat']}")
            self.tick_s = int(sp["tick_s"])
            self.levers.append(sp["levers"])
            self.gauges.append([g["name"] for g in sp["gauges"]])
            if len(sp["levers"]) > self.spec.max_levers or len(sp["gauges"]) > self.spec.max_gauges:
                raise ValueError("task does not fit the EnvSpec")
        self.horizon = int(task.horizon)
        self.scale = money_scale(task)
        self.t = 0
        self.net = [0.0] * S
        self.done = np.ones(1, dtype=bool)

    # ------------------------------------------------------------------ tool calls
    def _call(self, seat: int, tool: str, args: dict):
        res = self.world.execute(self.actors[seat], {"tool": tool, "args": args})
        self.calls[seat].append({"t": self.world.now, "tool": tool, "args": args, "ok": res.ok,
                                 "error": None if res.ok else res.error})
        if not res.ok:
            raise RuntimeError(f"{tool} by {self.actors[seat]} failed: {res.error}")
        return res.result

    def balance(self, seat: int) -> int:
        return int(sum(a["balance"] for a in self._call(seat, "bank.accounts", {}) if a["owner"] == self.actors[seat]))

    def account(self, seat: int) -> str:
        return [a["account"] for a in self._call(seat, "bank.accounts", {}) if a["owner"] == self.actors[seat]][0]

    def transfer(self, seat: int, to: str, amount: int, memo: str = "") -> dict:
        """Seat `seat`'s holder pays `amount` cents to `to` (a principal, handle or account) with `bank.transfer`."""
        return self._call(seat, "bank.transfer", {"from_account": self.account(seat), "to": to,
                                                  "amount": int(amount), "memo": memo})

    def rig_net(self, seat: int, window_s: int = 0) -> float:
        """The self-play objective of a seat's holder, from the registered metric `rig_net`."""
        from ..goals.metrics import evaluate_metric
        return float(evaluate_metric(self.world, "rig_net", {"principal": self.actors[seat], "rig": "r1",
                                                             "window_s": int(window_s)}))

    def _obs(self, seat: int, report: dict | None, settings: dict) -> np.ndarray:
        sp = self.spec
        o = np.zeros((1, sp.obs_dim), dtype=np.float32)
        names = self.gauges[seat]
        if report is not None:
            o[0, sp.gauges][:len(names)] = [_symlog(report["readings"][name]) for name in names]
        o[0, sp.gauge_mask][:len(names)] = 1.0
        for j, lv in enumerate(self.levers[seat]):
            span = lv["max"] - lv["min"]
            o[0, sp.last_action][j] = 0.0 if span <= 0 else 2.0 * (settings[lv["name"]] - lv["min"]) / span - 1.0
        o[0, sp.lever_mask][:len(self.levers[seat])] = 1.0
        sc = o[0, sp.scalars]
        if report is not None:
            sc[0] = report["net_cents"] / self.scale
            sc[1] = report["revenue_cents"] / self.scale
            sc[2] = report["cost_cents"] / self.scale
        sc[3] = self.balance(seat) / max(1.0, float(self.task.endowment))
        sc[4] = self.t / float(self.horizon)
        return o

    # ------------------------------------------------------------------ gym-like API (one slot)
    def reset(self) -> list[np.ndarray]:
        if self.t != 0:
            raise RuntimeError("a kernel world is not rewound: build a new KernelSelfPlayEnv (or fork the world)")
        self.done[:] = False
        return [self._obs(s, None, self._call(s, "rig.read", {})["next_settings"]) for s in range(self.n_seats)]

    def step(self, actions: Sequence[np.ndarray | None]):
        S = self.n_seats
        for s in range(S):
            if actions[s] is None:
                continue
            a = np.clip(np.asarray(actions[s], dtype=np.float64).reshape(-1), -1.0, 1.0)
            settings = {lv["name"]: float(lv["min"] + (a[j] + 1.0) * 0.5 * (lv["max"] - lv["min"]))
                        for j, lv in enumerate(self.levers[s])}
            self._call(s, "rig.set", {"settings": settings})
        self.world.advance_time(self.world.now + self.tick_s)
        self.t += 1
        self.done[:] = self.t >= self.horizon
        obs, reward = [], np.zeros((S, 1), dtype=np.float32)
        net, status = np.zeros((S, 1)), []
        for s in range(S):
            rd = self._call(s, "rig.read", {})
            rep = rd["report"]
            net[s, 0] = float(rep["net_cents"])
            reward[s, 0] = rep["net_cents"] / self.scale
            self.net[s] += rep["net_cents"]
            status.append([rep["status"]])
            obs.append(self._obs(s, rep, rd["next_settings"]))
        return obs, reward, self.done.copy(), {"net_cents": net, "status": status}

    def summary(self) -> dict:
        return {"task": self.task.id, "ticks": self.t, "net_cents": [float(x) for x in self.net],
                "rig_net": [self.rig_net(s) for s in range(self.n_seats)],
                "tool_calls": [len(c) for c in self.calls], "world_digest": self.world.digest()}


# ====================================================================== objectives
@dataclasses.dataclass(frozen=True)
class Objective:
    """What an individual is trained to maximise, as a function of the per-seat rewards of a tick.

    own     its own net income (the facility metric `rig_net`): revenue from independent demand minus own cost
    joint   the mean of all seats' net incomes (a team objective: every seat is credited the same number)
    score   the mechanism's own objective quantity for the seat (the facility metric `rig_score`)
    mixed   own + `weight` x the mean of the other seats' nets

    Only the *training signal* changes with the objective. No money moves between seats for it, the observation
    keeps showing the seat's own net, and every result table reports own and joint outcomes separately.
    """
    kind: str = "own"
    weight: float = 0.0

    def __post_init__(self):
        if self.kind not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")

    def reward(self, seat: int, reward: np.ndarray, score: np.ndarray | None = None) -> np.ndarray:
        """`reward`: [seats, n] own nets in money units; `score`: [seats, n] objective quantities in money units."""
        if self.kind == "own":
            return reward[seat]
        if self.kind == "joint":
            return reward.mean(axis=0)
        if self.kind == "score":
            if score is None:
                raise ValueError("the 'score' objective needs the per-seat objective quantity")
            return score[seat]
        others = [s for s in range(reward.shape[0]) if s != seat]
        return reward[seat] + self.weight * (reward[others].mean(axis=0) if others else 0.0)

    def describe(self) -> str:
        return {"own": "own net income (rig_net)", "joint": "mean net income of all seats (team objective)",
                "score": "the mechanism's objective quantity for the seat (rig_score)",
                "mixed": f"own net + {self.weight:g} x mean net of the other seats"}[self.kind]

    @classmethod
    def of(cls, x: "Objective | str | dict | None") -> "Objective":
        if isinstance(x, Objective):
            return x
        if x is None:
            return cls()
        if isinstance(x, dict):
            return cls(str(x.get("kind", "own")), float(x.get("weight", 0.0)))
        return cls(str(x))


# ====================================================================== individuals
class Individual:
    """One self-play agent. Everything listed here belongs to this individual alone:

    learner      its policy and learning rule (`research.learner.Learner`): weights, optimiser state, recurrent or
                 episodic context, replay or rollout buffer all live inside this object and nowhere else
    objective    what it is trained to maximise (`Objective`)
    seat         the seat it is bound to (None: it can sit anywhere the lever layout is the same)
    funds_scale  its starting balance in a match, as a multiple of the task's public starting balance
    bankroll     optional persistent funds in cents carried from match to match (None: every match is funded
                 separately). With a bankroll, a batch of n matches stakes `min(funds_scale x endowment,
                 bankroll // n)` on each; an individual whose bankroll is gone enters with nothing and, by the
                 facility's money rules, falls back to defaults or idles
    history      its own experience history: one row per batch of matches (who it met, what it earned)
    id, parent, generation, tags   persistent identity and lineage

    `privileged` marks operator-side fixtures built from the hidden mechanism (scripted profiles). They are legal
    opponents and baselines, never counted as learning systems.
    """

    def __init__(self, id: str, learner: Learner, objective: Objective | str | dict | None = "own",
                 seat: int | None = None, funds_scale: float = 1.0, bankroll: int | None = None,
                 parent: str | None = None, generation: int = 0, tags: Sequence[str] = (), privileged: bool = False,
                 world: str | None = None, config: dict | None = None):
        self.id = str(id)
        self.learner = learner
        self.objective = Objective.of(objective)
        self.seat = None if seat is None else int(seat)
        self.funds_scale = float(funds_scale)
        self.bankroll = None if bankroll is None else int(bankroll)
        self.parent, self.generation, self.tags = parent, int(generation), tuple(tags)
        self.privileged = bool(privileged)
        self.world = world                       # the task id of the persistent world it lives in
        self.config = dict(config or {})         # how to rebuild its learner (`learners.make_learner`)
        self.history: list[dict] = []
        self.matches = 0
        self.env_steps = 0
        self.lifetime_net_cents = 0.0

    # ---- identity ----------------------------------------------------------------------------------
    @property
    def arch(self) -> str:
        return str(getattr(self.learner, "name", type(self.learner).__name__))

    @property
    def learns(self) -> bool:
        return not self.learner.frozen and getattr(self.learner, "kind", "") not in ("idle", "random", "fixed")

    def stake(self, task: Task, n: int = 1) -> int:
        """Starting balance in each of `n` simultaneous matches of `task`."""
        want = int(round(self.funds_scale * task.endowment))
        if self.bankroll is None:
            return want
        return max(0, min(want, self.bankroll // max(1, n)))

    # ---- persistence -------------------------------------------------------------------------------
    def meta(self) -> dict:
        return {"format": "crucible-selfplay-individual/1", "id": self.id, "arch": self.arch,
                "kind": getattr(self.learner, "kind", ""), "objective": dataclasses.asdict(self.objective),
                "seat": self.seat, "funds_scale": self.funds_scale, "bankroll": self.bankroll,
                "parent": self.parent, "generation": self.generation, "tags": list(self.tags),
                "privileged": self.privileged, "world": self.world, "config": self.config,
                "frozen": bool(self.learner.frozen), "matches": self.matches, "env_steps": self.env_steps,
                "lifetime_net_cents": self.lifetime_net_cents, "history": self.history}

    def save(self, directory: str) -> str:
        """Write the individual to `directory`: `meta.json` (identity, lineage, objective, budget, history) and
        `learner.ckpt` (whatever the learner's own `save` writes). Episodic context is not saved."""
        os.makedirs(directory, exist_ok=True)
        self.learner.save(os.path.join(directory, "learner.ckpt"))
        tmp = os.path.join(directory, f"meta.json.{os.getpid()}.tmp")
        with open(tmp, "w") as fh:
            json.dump(self.meta(), fh)
        os.replace(tmp, os.path.join(directory, "meta.json"))
        return directory

    @classmethod
    def load(cls, directory: str, spec: EnvSpec | None = None, learner: Learner | None = None,
             factory: Callable[[dict], Learner] | None = None) -> "Individual":
        """Restore an individual. The learner is rebuilt from the saved `config` with `learners.make_learner`
        (or by `factory(config)`, or taken from `learner`) and then loads its checkpoint."""
        with open(os.path.join(directory, "meta.json")) as fh:
            m = json.load(fh)
        if learner is None:
            if factory is not None:
                learner = factory(m["config"])
            else:
                from .learners import make_learner
                learner = make_learner(spec=spec or EnvSpec(), **m["config"])
        learner.load(os.path.join(directory, "learner.ckpt"))
        ind = cls(m["id"], learner, m["objective"], m["seat"], m["funds_scale"], m["bankroll"], m["parent"],
                  m["generation"], m["tags"], m["privileged"], m.get("world"), m["config"])
        ind.history, ind.matches, ind.env_steps = list(m["history"]), int(m["matches"]), int(m["env_steps"])
        ind.lifetime_net_cents = float(m["lifetime_net_cents"])
        if m.get("frozen"):
            learner.freeze(True)
        return ind


# ====================================================================== playing
def play(env: SelfPlayEnv, individuals: Sequence[Individual | None], *, noise: Sequence[int] | int | None = None,
         learn: bool | Sequence[bool] = True, explore: bool | Sequence[bool] | None = None,
         tasks: Sequence[Task] | None = None, tag: str = "") -> list[dict]:
    """One batch of `env.n` simultaneous matches, one full episode each, `individuals[s]` in seat s (None: the
    seat is held but nobody operates it, it rests at its defaults).

    `noise`: noise index per slot (an int k gives slots k, k+1, ...; None replays the installed worlds).
    `learn`: per seat, whether the individual observes the experience and updates afterwards (never for a frozen
    learner). `explore`: per seat, whether it acts with exploration (default: exactly when it learns).

    Each individual acts on its own observation, is credited its own objective, and updates from its own
    experience. Returns the operator-side match records; each row gains `individuals` (ids per seat) and `tag`.
    """
    S, n = env.n_seats, env.n
    if len(individuals) != S:
        raise ValueError(f"one individual (or None) per seat ({S})")
    ids = [None if ind is None else ind.id for ind in individuals]
    if len({i for i in ids if i is not None}) != len([i for i in ids if i is not None]):
        raise ValueError("an individual cannot occupy two seats of one match")
    for s, ind in enumerate(individuals):
        if ind is not None and ind.seat is not None and ind.seat != s:
            raise ValueError(f"individual {ind.id} is bound to seat {ind.seat}, not {s}")
    learn_s = [bool(learn)] * S if isinstance(learn, (bool, np.bool_)) else [bool(x) for x in learn]
    learn_s = [learn_s[s] and individuals[s] is not None and not individuals[s].learner.frozen for s in range(S)]
    if explore is None:
        explore_s = list(learn_s)
    else:
        explore_s = [bool(explore)] * S if isinstance(explore, (bool, np.bool_)) else [bool(x) for x in explore]
    base = list(tasks) if tasks is not None else list(env.tasks)
    if noise is not None:
        ks = [int(noise) + i for i in range(n)] if isinstance(noise, (int, np.integer)) else [int(k) for k in noise]
        if len(ks) != n:
            raise ValueError("one noise index per slot")
        base = [with_noise(_root_task(t), k) for t, k in zip(base, ks)]
    else:
        ks = [None] * n
    funds = [[int(t.endowment) if ind is None else ind.stake(t, n) for ind in individuals] for t in base]
    obs = env.reset(tasks=base, funds=funds)
    for ind in individuals:
        if ind is not None:
            ind.learner.reset_context(n)
    scale_score = np.asarray(env._scale)
    while not env.done.all():
        acts: list[np.ndarray | None] = []
        for s, ind in enumerate(individuals):
            acts.append(None if ind is None else
                        np.asarray(ind.learner.act(obs[s], explore=explore_s[s]), dtype=np.float32))
        nxt, rew, done, info = env.step(acts)
        score = (info["score"] / scale_score).astype(np.float32)
        for s, ind in enumerate(individuals):
            if ind is not None and learn_s[s]:
                ind.learner.observe(obs[s], acts[s], ind.objective.reward(s, rew, score).astype(np.float32), done,
                                    nxt[s])
        obs = nxt
    counters = [ind.learner.update() if ind is not None and learn_s[s] else None
                for s, ind in enumerate(individuals)]
    recs = env.drain()
    for r, k in zip(recs, ks):
        r["individuals"], r["noise"], r["tag"] = ids, k, tag
        r["learned"] = list(learn_s)
    for s, ind in enumerate(individuals):
        if ind is None:
            continue
        nets = [r["seats"][s]["net_cents"] for r in recs]
        ind.matches += n
        ind.env_steps += int(sum(r["ticks"] for r in recs))
        ind.learner.usage().env_steps += int(sum(r["ticks"] for r in recs))
        ind.lifetime_net_cents += float(sum(nets))
        if ind.bankroll is not None:
            ind.bankroll += int(round(sum(r["seats"][s]["funds_end"] - r["seats"][s]["funds_start"] for r in recs)))
        ind.history.append({"n": n, "seat": s, "task": recs[0]["task"], "with": [i for j, i in enumerate(ids) if j != s],
                            "learned": learn_s[s], "tag": tag,
                            "net_per_tick": float(np.mean([r["seats"][s]["net_per_tick"] for r in recs])),
                            "joint_net_per_tick": float(np.mean([r["joint_net_per_tick"] for r in recs])),
                            "idle_ticks": int(sum(r["seats"][s]["ticks_idle"] for r in recs)),
                            "updates": None if counters[s] is None else int(counters[s].get("updates", 0))})
    return recs


def _root_task(task: Task) -> Task:
    """The noise-0 world of a task (its world seed is a function of family and seed)."""
    from ..research.tasks import world_seed_of
    ws = world_seed_of(task.family, task.seed)
    return task if task.world_seed == ws else dataclasses.replace(task, world_seed=ws)


def summarise(recs: Sequence[dict]) -> dict:
    """Mean outcomes over match records: per-seat net per tick, joint net per tick, and the private quantities."""
    if not recs:
        return {"n": 0}
    S = len(recs[0]["seats"])
    out: dict[str, Any] = {"n": len(recs),
                           "net_per_tick": [float(np.mean([r["seats"][s]["net_per_tick"] for r in recs])) for s in range(S)],
                           "joint_net_per_tick": float(np.mean([r["joint_net_per_tick"] for r in recs])),
                           "ticks_idle": [int(sum(r["seats"][s]["ticks_idle"] for r in recs)) for s in range(S)],
                           "ticks_fallback": [int(sum(r["seats"][s]["ticks_fallback"] for r in recs)) for s in range(S)],
                           "mean_action": [np.mean([r["seats"][s]["mean_action"] for r in recs], axis=0).tolist()
                                           for s in range(S)]}
    keys = sorted(set().union(*[r["private"].keys() for r in recs]))
    out["private"] = {k: float(np.mean([r["private"].get(k, 0.0) for r in recs])) for k in keys}
    return out
