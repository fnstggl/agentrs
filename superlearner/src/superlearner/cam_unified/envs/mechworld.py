"""Hidden-mechanism worlds behind the same contract as OpenWorld: numerical worlds are one input format, not the
architecture.

`MechWorldAdapter` (agent side, `EnvAdapter`) runs `crucible.research.envs.MechVecEnv` (one size-1 env per batch row,
so every row has its own clock) and presents it as entities, measurements and tools:

* levers and gauges are **entities** with per-episode neutral random names (`worldgen.Surface`), in a per-episode
  order; their public spec (range, default, integer flag) arrives once at reset as CONSTRAINT observations;
* each tick (every `tick_s` simulated seconds of the operator clock) delivers a report EVENT plus NUMERIC
  observations: every gauge reading (in the surface's unit), every lever setting in force (with its public range in
  meta), revenue / cost / net cents and the operator's funds in cents;
* tools: ``set_<lever>`` with one typed numeric argument bounded by the public range (integer levers take
  integers), and the harness ``wait``; the contract's ``noop`` (nothing, duration 0) and ``advance_time``
  (``{"seconds"}`` or ``{"days"}``, = wait) are honoured. A setting applies from the next tick (as `rig.set`).

Money, settings and gauge readings are numbers exactly as the simulator states them (contract: no unit conversion).
Only a *declared invariance condition* (`worldgen.invariance_pair`, ``Surface(scale_units=True)``) presents gauges
at a different scale, as a different environment would - the unit label then names the scale (e.g.
``milli-reading``); the default surface never rescales. Everything private - the task's default/reference/random
nets, the family, the seed, the engine's truth and diagnostics - stays in `MechWorldOperator` /
`MechWorldEvaluator`: the verified return is the tick's net cents over the task's public money scale (exactly
`MechVecEnv`'s reward; recorded in ``GoalSpec.meta["reward_scale_cents"]``), the independent result is
`Task.score` / `Task.success` on the episode's final net and the episode record of `MechVecEnv.drain()`; success
is decided at the end of the episode (the net can fall back after crossing the goal - the first crossing is
reported separately as ``reached_day``).

Also here: `RigTask`, the dense-reward rig task *inside the real OpenWorld* (``ow_works``: ``rig.list / take / spec /
set / read`` through `OpenWorldAdapter`), with `RigNetObjective` (verified net cents from the facility's own
records each step) and `RigOperatorControl`, a non-privileged scripted control (take a seat, then Crucible's
`RigExperimenter`).
"""
from __future__ import annotations

import bisect
import copy
import json
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..interfaces import (Action, EvidenceKind, ExecutionResult, GoalSpec, IndependentGoalResult, Observation,
                          ObservationBatch, ObsKind, ToolSchema, new_id)
from .openworld import ADVANCE_TOOL, NOOP_TOOL, WAIT_TOOL, noop_result, wait_result, wait_tool
from .worldgen import Surface

DAY = 86400


@dataclass
class _MEnv:
    index: int
    task: Any = None
    env: Any = None                       # MechVecEnv of one task
    surface: Any = None
    env_id: str = ""
    lever_engine: list = field(default_factory=list)       # engine lever names, engine order
    lever_shown: list = field(default_factory=list)        # surface names, engine order
    lever_spec: list = field(default_factory=list)         # Lever objects, engine order
    gauge_engine: list = field(default_factory=list)
    gauge_shown: list = field(default_factory=list)
    gauge_unit: tuple = ("reading", 1.0)
    lever_order: list = field(default_factory=list)        # presentation order (indices into engine order)
    gauge_order: list = field(default_factory=list)
    facility: str = ""
    account: str = ""
    raw: list = field(default_factory=list)                # current raw lever values (engine order)
    t_s: int = 0
    ticks: int = 0
    seq: int = 0
    buffer: list = field(default_factory=list)
    delivered: int = 0
    first: list = field(default_factory=list)
    actions: list = field(default_factory=list)            # normalized action vector used at each tick (audit)
    net_total: float = 0.0
    goal: Optional[GoalSpec] = None
    record: Optional[dict] = None


class MechWorldOperator:
    """Operator side of a batch of hidden-mechanism worlds (holds the private `Task`s)."""

    def __init__(self, tasks: Sequence[Any] | Callable[[int, int], Any], n_envs: Optional[int] = None, *,
                 tick_s: int = 3600, set_cost_s: int = 60, max_wait_s: int = 86400,
                 surface: Optional[Callable[[int, int], Surface]] = None, seed: int = 0, spec=None):
        from crucible.mechanisms import load_engines
        from crucible.research.learner import EnvSpec
        load_engines()
        from .worldgen import register_families
        register_families()
        if callable(tasks):
            if n_envs is None:
                raise ValueError("n_envs is required with a task factory")
            self._task_fn, self.n = tasks, int(n_envs)
        else:
            tl = list(tasks)
            self.n = len(tl) if n_envs is None else int(n_envs)
            self._task_fn = lambda b, seed, _t=tl: _t[b if len(_t) > 1 else 0] if seed is None else \
                _reseed(_t[b if len(_t) > 1 else 0], seed)
        self.tick_s, self.set_cost_s, self.max_wait_s = int(tick_s), int(set_cost_s), int(max_wait_s)
        self.rng = random.Random(seed)
        self._surface = surface or (lambda b, ep: Surface(self.rng.randrange(2 ** 31)))
        self.spec = spec or EnvSpec(max_levers=32, max_gauges=32)
        self.envs = [_MEnv(i) for i in range(self.n)]
        self.episode = 0
        self.tick_hooks: list[Callable[[int, float, float, dict], None]] = []    # (b, net_cents, reward, last)
        self.episode_hooks: list[Callable[[int], None]] = []
        self.clock_hooks: list[Callable[[int], None]] = []

    # ------------------------------------------------------------------ episodes
    def task(self, b: int):
        return self.envs[b].task

    def reset(self, seeds: Optional[Sequence[int]] = None) -> list:
        from crucible.research.envs import MechVecEnv, money_scale
        self.episode += 1
        for b in range(self.n):
            s = None if seeds is None else int(seeds[b] if len(seeds) > 1 else seeds[0])
            task = self._task_fn(b, s)
            e = _MEnv(b, task)
            e.env = MechVecEnv([task], self.spec)
            e.env.reset()
            sim = e.env.sims[0]
            eng = sim.engine
            e.surface = self._surface(b, self.episode)
            e.lever_spec = list(eng.levers(0))
            e.lever_engine = [lv.name for lv in e.lever_spec]
            e.gauge_engine = [g.name for g in eng.gauges(0)]
            shown = e.surface.names(e.lever_engine + e.gauge_engine)
            e.lever_shown, e.gauge_shown = shown[:len(e.lever_engine)], shown[len(e.lever_engine):]
            e.lever_order = e.surface.order(len(e.lever_engine))
            e.gauge_order = e.surface.order(len(e.gauge_engine))
            e.gauge_unit = e.surface.unit()
            fac = e.surface.names(["facility", "account"]) if e.surface.rename else ["facility", "account"]
            e.facility, e.account = f"facility:{fac[0]}", f"account:{fac[1]}"
            e.raw = [float(sim.settings[0][n]) for n in e.lever_engine]
            e.env_id = f"mw{b}-{new_id()[:10]}"
            e.goal = self._goal(e, money_scale(task))
            e.first = self._intro(e)
            e.buffer, e.delivered = [], 0
            self.envs[b] = e
        for b in range(self.n):
            for h in self.episode_hooks:
                h(b)
        return [self._deliver(e) for e in self.envs]

    def _goal(self, e: _MEnv, scale: float) -> GoalSpec:
        t = e.task
        if t.goal_net_cents is not None:
            text = e.surface.phrase("goal", horizon=t.horizon, goal=int(t.goal_net_cents))
            metrics = {"net_cents": float(t.goal_net_cents)}
        else:
            text = e.surface.phrase("goal_open", horizon=t.horizon)
            metrics = {}
        metrics["horizon_ticks"] = float(t.horizon)
        return GoalSpec(goal_id=new_id("g_"), text=text, metrics=metrics,
                        deadline_days=t.horizon * self.tick_s / DAY, family="mechworld",
                        meta={"starting_balance_cents": int(t.endowment), "tick_s": self.tick_s,
                              "reward_scale_cents": float(scale)})

    def _brief(self, e: _MEnv) -> str:
        import re
        txt = e.env.sims[0].engine.brief(0)
        for a, b in zip(e.lever_engine + e.gauge_engine, e.lever_shown + e.gauge_shown):
            txt = re.sub(rf"\b{re.escape(a)}\b", b, txt)
        return txt

    def _intro(self, e: _MEnv) -> list:
        out = [Observation(ObsKind.TEXT, content=self._brief(e), entity=e.facility, entity_type="facility", t=0.0,
                           source="facility", meta={"brief": True}),
               Observation(ObsKind.NUMERIC, value=float(e.task.endowment), unit="cents", entity=e.account,
                           entity_type="account", t=0.0, source="account", meta={"field": "funds"})]
        for j in e.lever_order:
            lv = e.lever_spec[j]
            out.append(Observation(ObsKind.CONSTRAINT, content=self._lever_doc(e, j), entity=f"lever:{e.lever_shown[j]}",
                                   entity_type="lever", t=0.0, source="facility",
                                   meta={"min": lv.lo, "max": lv.hi, "default": lv.default, "integer": lv.integer}))
            out.append(Observation(ObsKind.NUMERIC, value=e.raw[j], unit="setting", entity=f"lever:{e.lever_shown[j]}",
                                   entity_type="lever", t=0.0, source="facility",
                                   meta={"field": "setting", "min": lv.lo, "max": lv.hi, "default": lv.default,
                                         "integer": lv.integer}))
        for k in e.gauge_order:
            out.append(Observation(ObsKind.RELATION, entity=f"gauge:{e.gauge_shown[k]}", entity_type="gauge",
                                   target=e.facility, relation="measures", t=0.0, source="facility",
                                   meta={"unit": e.gauge_unit[0]}))
        return out

    def _lever_doc(self, e: _MEnv, j: int) -> str:
        lv = e.lever_spec[j]
        return e.surface.phrase("lever", name=e.lever_shown[j], lo=lv.lo, hi=lv.hi, default=lv.default,
                                integer=", integer" if lv.integer else "")

    # ------------------------------------------------------------------ tools / actions
    def tools(self, b: int) -> list:
        e = self.envs[b]
        out = []
        for j in e.lever_order:
            lv = e.lever_spec[j]
            ty = "integer" if lv.integer else "number"
            lo, hi = (int(round(lv.lo)), int(round(lv.hi))) if lv.integer else (float(lv.lo), float(lv.hi))
            out.append(ToolSchema(f"set_{e.lever_shown[j]}", self._lever_doc(e, j),
                                  {"type": "object", "properties": {"value": {"type": ty, "minimum": lo, "maximum": hi}},
                                   "required": ["value"], "additionalProperties": False},
                                  returns="{lever, value, applies_at_tick}", duration_hint=self.set_cost_s / DAY,
                                  meta={"lever": True}))
        return out + [wait_tool(self.max_wait_s)]

    def execute(self, b: int, action: Action) -> ExecutionResult:
        from crucible.core import schema as S
        e = self.envs[b]
        t = e.t_s / DAY
        if action.tool in (WAIT_TOOL, ADVANCE_TOOL):
            return wait_result(action, e.t_s, self.max_wait_s)
        if action.tool == NOOP_TOOL:
            return noop_result(action, t)
        name = action.tool[4:] if action.tool.startswith("set_") else None
        if name not in e.lever_shown:
            return self._err(action, t, "unknown_tool", f"unknown tool {action.tool!r}")
        j = e.lever_shown.index(name)
        schema = next(ts.parameters for ts in self.tools(b) if ts.name == action.tool)
        errs = S.validate(schema, action.args or {})
        if errs:
            return self._err(action, t, "invalid_action", "; ".join(errs[:4]))
        lv = e.lever_spec[j]
        e.raw[j] = lv.clip(action.args["value"])
        body = {"lever": name, "value": e.raw[j], "applies_at_tick": e.ticks}
        obs = [Observation(ObsKind.TOOL_RESULT, json.dumps(body), entity=f"lever:{name}", entity_type="lever", t=t,
                           source=action.tool, evidence=EvidenceKind.CONTROLLED, caused_by=action.action_id,
                           meta={"tool": action.tool}),
               Observation(ObsKind.NUMERIC, value=e.raw[j], unit="setting", entity=f"lever:{name}", entity_type="lever",
                           t=t, source=action.tool, evidence=EvidenceKind.CONTROLLED, caused_by=action.action_id,
                           meta={"field": "setting", "min": lv.lo, "max": lv.hi})]
        return ExecutionResult(True, obs, None, 0.0, self.set_cost_s / DAY, t, t + self.set_cost_s / DAY,
                               meta={"tool": action.tool})

    @staticmethod
    def _err(action: Action, t: float, code: str, msg: str) -> ExecutionResult:
        return ExecutionResult(False, [Observation(ObsKind.ERROR, f"{code}: {msg}", t=t, source=action.tool,
                                                   evidence=EvidenceKind.CONTROLLED, caused_by=action.action_id,
                                                   meta={"tool": action.tool, "code": code, "message": msg,
                                                         "details": None})],
                               code, started_t=t, finished_t=t, meta={"tool": action.tool})

    # ------------------------------------------------------------------ the clock
    def normalized(self, e: _MEnv) -> np.ndarray:
        """The MechVecEnv action for the settings in force: 2 (v - lo) / (hi - lo) - 1 per lever."""
        a = np.zeros((1, self.spec.max_levers))
        for j, lv in enumerate(e.lever_spec):
            span = lv.hi - lv.lo
            a[0, j] = 0.0 if span <= 0 else 2.0 * (e.raw[j] - lv.lo) / span - 1.0
        return a

    def _tick(self, e: _MEnv) -> None:
        a = self.normalized(e)
        _, rew, done, info = e.env.step(a)
        e.actions.append(a[0, :len(e.lever_spec)].copy())
        last = e.env.sims[0].last[0]
        # settings actually in force (the simulator clips/rounds) become the shown raw values
        used = last.get("settings") or {}
        e.raw = [float(e.env.sims[0].settings[0][n]) for n in e.lever_engine]
        net = float(info["net_cents"][0])
        e.net_total += net
        e.ticks += 1
        e.seq += 1
        t = e.t_s / DAY
        scale_lab, scale = e.gauge_unit
        readings = {e.gauge_shown[k]: float(last["readings"][e.gauge_engine[k]]) * scale for k in e.gauge_order}
        settings = {e.lever_shown[j]: e.raw[j] for j in e.lever_order}
        rep = {"tick": e.ticks - 1, "readings": readings, "settings": settings, "revenue_cents": int(last["revenue"]),
               "cost_cents": int(last["cost"]), "net_cents": net, "status": last["status"],
               "ran_at": {e.lever_shown[j]: float(used.get(e.lever_engine[j], e.raw[j])) for j in e.lever_order}}
        obs = [Observation(ObsKind.EVENT, json.dumps({"type": "facility.report", "text": e.surface.phrase(
            "report", tick=e.ticks - 1), "payload": rep}), entity=e.facility, entity_type="facility", t=t, seq=e.seq,
            source="facility", meta={"tick": e.ticks - 1})]
        for k in e.gauge_order:
            obs.append(Observation(ObsKind.NUMERIC, value=readings[e.gauge_shown[k]], unit=scale_lab,
                                   entity=f"gauge:{e.gauge_shown[k]}", entity_type="gauge", t=t, seq=e.seq,
                                   source="facility", meta={"field": "reading", "tick": e.ticks - 1}))
        for j in e.lever_order:
            lv = e.lever_spec[j]
            obs.append(Observation(ObsKind.NUMERIC, value=e.raw[j], unit="setting", entity=f"lever:{e.lever_shown[j]}",
                                   entity_type="lever", t=t, seq=e.seq, source="facility",
                                   meta={"field": "setting", "min": lv.lo, "max": lv.hi, "tick": e.ticks - 1}))
        for f in ("revenue_cents", "cost_cents", "net_cents"):
            obs.append(Observation(ObsKind.NUMERIC, value=float(rep[f]), unit="cents", entity=e.facility,
                                   entity_type="facility", t=t, seq=e.seq, source="facility",
                                   meta={"field": f, "tick": e.ticks - 1}))
        funds = e.env.sims[0].funds[0]
        if funds is not None:
            obs.append(Observation(ObsKind.NUMERIC, value=float(funds), unit="cents", entity=e.account,
                                   entity_type="account", t=t, seq=e.seq, source="account",
                                   meta={"field": "funds", "tick": e.ticks - 1}))
        e.buffer.extend(obs)
        for h in self.tick_hooks:
            h(e.index, net, float(rew[0]), last)
        if bool(done[0]):
            recs = e.env.drain()
            e.record = recs[-1] if recs else None

    def _advance_env(self, e: _MEnv, seconds: int) -> None:
        end = e.t_s + int(seconds)
        while not bool(e.env.done[0]):
            nxt = (e.t_s // self.tick_s + 1) * self.tick_s
            if nxt > end:
                break
            e.t_s = nxt
            self._tick(e)
        e.t_s = max(e.t_s, end) if not bool(e.env.done[0]) else e.t_s
        for h in self.clock_hooks:
            h(e.index)

    def advance(self, durations_days: Sequence[float]) -> list:
        for b, d in enumerate(durations_days):
            self._advance_env(self.envs[b], max(0, int(round(float(d) * DAY))))
        return [self._deliver(e) for e in self.envs]

    # ------------------------------------------------------------------ delivery
    def _deliver(self, e: _MEnv) -> list:
        out = list(e.first)
        e.first = []
        i = bisect.bisect_right(e.buffer, e.delivered, key=lambda o: o.seq)
        out += e.buffer[i:]
        e.delivered = e.seq
        return out

    def observe(self, b: int) -> list:
        return self._deliver(self.envs[b])

    def collect(self, b: int, since: int) -> tuple:
        e = self.envs[b]
        i = bisect.bisect_right(e.buffer, int(since), key=lambda o: o.seq)
        return e.buffer[i:], e.seq

    def now_days(self, b: int) -> float:
        return self.envs[b].t_s / DAY

    def done(self, b: int) -> bool:
        return bool(self.envs[b].env.done[0])

    def checkpoint(self, path: str) -> dict:
        state = {"episode": self.episode, "rng": self.rng.getstate(),
                 "envs": [{k: v for k, v in e.__dict__.items() if k != "env"} |
                          {"sim": e.env.sims[0].snapshot(), "ep": copy.deepcopy(e.env._ep[0]),
                           "done": bool(e.env.done[0])} for e in self.envs]}
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return {"path": path}

    def restore(self, path: str) -> None:
        from crucible.research.envs import MechVecEnv
        with open(path, "rb") as f:
            st = pickle.load(f)
        self.episode = st["episode"]
        self.rng.setstate(st["rng"])
        for i, d in enumerate(st["envs"]):
            e = _MEnv(i)
            for k, v in d.items():
                if k not in ("sim", "ep", "done"):
                    setattr(e, k, v)
            e.env = MechVecEnv([e.task], self.spec)
            e.env.reset()
            e.env.sims[0].restore(d["sim"])
            e.env._ep[0] = d["ep"]
            e.env.done[0] = d["done"]
            self.envs[i] = e

    def close(self) -> None:
        pass


def _reseed(task: Any, seed: int):
    from crucible.research.tasks import make_task
    return make_task(task.family, int(seed), validate=task.validated)


class _MPort:
    __slots__ = ("reset", "observe", "tools", "execute", "collect", "advance", "goals", "now", "done", "env_ids",
                 "close")


class MechWorldAdapter:
    """Agent-side `EnvAdapter` over hidden-mechanism worlds (see module docstring)."""

    def __init__(self, port: _MPort, n: int):
        self._port, self.n = port, int(n)

    def _batch(self, obs):
        return ObservationBatch(self._port.env_ids(), obs, [self._port.now(b) for b in range(self.n)],
                                [self._port.done(b) for b in range(self.n)])

    def reset(self, seeds=None) -> ObservationBatch:
        return self._batch(self._port.reset(seeds))

    def observe(self) -> ObservationBatch:
        return self._batch([self._port.observe(b) for b in range(self.n)])

    def available_tools(self) -> list:
        return [self._port.tools(b) for b in range(self.n)]

    def execute(self, actions) -> list:
        if len(actions) != self.n:
            raise ValueError(f"expected {self.n} actions")
        return [None if a is None else self._port.execute(b, a) for b, a in enumerate(actions)]

    def collect_events(self, since) -> tuple:
        got = [self._port.collect(b, int(since[b])) for b in range(self.n)]
        return [g[0] for g in got], [g[1] for g in got]

    def advance_time(self, duration) -> ObservationBatch:
        return self._batch(self._port.advance(list(duration)))

    def goals(self) -> list:
        return self._port.goals()

    def close(self) -> None:
        self._port.close()


def mech_world(tasks, n_envs: Optional[int] = None, **kw) -> tuple:
    """(MechWorldAdapter, MechWorldOperator) over `tasks` (Crucible research `Task`s, one per env or one for all,
    or a factory ``(env_index, seed | None) -> Task``). Keep the operator on the harness side."""
    op = MechWorldOperator(tasks, n_envs, **kw)
    p = _MPort()
    p.reset = lambda seeds: op.reset(seeds)
    p.observe = lambda b: op.observe(b)
    p.tools = lambda b: op.tools(b)
    p.execute = lambda b, a: op.execute(b, a)
    p.collect = lambda b, s: op.collect(b, s)
    p.advance = lambda d: op.advance(d)
    p.goals = lambda: [copy.deepcopy(e.goal) for e in op.envs]
    p.now = lambda b: op.now_days(b)
    p.done = lambda b: op.done(b)
    p.env_ids = lambda: [e.env_id for e in op.envs]
    p.close = lambda: op.close()
    return MechWorldAdapter(p, op.n), op


class MechWorldEvaluator:
    """`GoalEvaluator` for `MechWorldOperator` rows: verified returns are the tick nets over the public money scale
    (MechVecEnv's reward); results use the task's private normalisation (operator side only)."""

    name = "crucible.research.Task"

    def __init__(self, operator: MechWorldOperator):
        self._op = operator
        n = operator.n
        self.pending, self.total = [0.0] * n, [0.0] * n
        self.success_t: list = [None] * n
        self.reached_t: list = [None] * n
        operator.tick_hooks.append(self._on_tick)
        operator.episode_hooks.append(self._begin)

    def _begin(self, b: int) -> None:
        self.pending[b] = self.total[b] = 0.0
        self.success_t[b] = None
        self.reached_t[b] = None

    def _on_tick(self, b: int, net: float, reward: float, last: dict) -> None:
        self.pending[b] += reward
        self.total[b] += reward
        e = self._op.envs[b]
        if self.reached_t[b] is None and e.task.goal_net_cents is not None and e.net_total >= e.task.goal_net_cents:
            self.reached_t[b] = e.t_s           # first crossing (the net may still fall back)
        if bool(e.env.done[0]) and self.success_t[b] is None and e.task.success(e.net_total):
            self.success_t[b] = e.t_s           # the outcome: Task.success on the final net, at the episode's end

    def take_reward(self, b: int) -> float:
        r, self.pending[b] = self.pending[b], 0.0
        return r

    def take_rewards(self) -> list:
        return [self.take_reward(b) for b in range(self._op.n)]

    def evaluate_goal(self, env_index: int) -> IndependentGoalResult:
        e = self._op.envs[int(env_index)]
        t = e.task
        ticks = max(1, e.ticks)
        score = t.score(e.net_total / ticks)
        ok = bool(e.env.done[0]) and t.success(e.net_total)      # decided on the final net of the episode
        rec = e.record or {}
        return IndependentGoalResult(
            goal_id=e.goal.goal_id, achieved=bool(ok), score=0.0 if score is None else float(score),
            metrics={"net_cents": e.net_total, "ticks": e.ticks, "net_per_tick": e.net_total / ticks},
            t=e.t_s / DAY, evaluator=self.name,
            details={"family": t.family, "seed": t.seed, "default_net": t.default_net,
                     "reference_net": t.reference_net, "random_net": t.random_net, "validated": t.validated,
                     "episode_record": rec, "normalised_score": score, "episode_done": bool(e.env.done[0]),
                     "success_t": self.success_t[int(env_index)],
                     "reached_day": None if self.reached_t[int(env_index)] is None
                     else self.reached_t[int(env_index)] / DAY})

    def state_dict(self) -> bytes:
        return pickle.dumps({"pending": self.pending, "total": self.total, "success_t": self.success_t,
                             "reached_t": self.reached_t})

    def load_state(self, blob: bytes) -> None:
        d = pickle.loads(blob)
        self.pending, self.total, self.success_t = d["pending"], d["total"], d["success_t"]
        self.reached_t = d.get("reached_t", [None] * len(self.success_t))


# ============================================================================================== rig in OpenWorld
DEFAULT_RIG = "works"


@dataclass
class RigTask:
    """The ow_works hidden-mechanism facility as a dense-reward OpenWorld task (operator-side object). Success:
    verified net income of the actor's rig seat over the run (decided at the horizon) >= `goal_net_cents` (facility
    records, `rig_net`). Reward unit: `reward_scale_cents` cents (default Crucible's public money scale, the
    starting balance / `research.tasks.ENDOWMENT_TICKS`, as `MechVecEnv` uses), recorded in the GoalSpec."""
    world: dict
    endowment: int = 1_000_000
    horizon_s: int = 4 * DAY
    goal_net_cents: int = 0
    rig: str = DEFAULT_RIG
    protocol: str = "ow_rig"
    sustain_checks: int = 1
    reward_scale: Optional[float] = None    # cents per reward unit; None = endowment / ENDOWMENT_TICKS

    @property
    def reward_scale_cents(self) -> float:
        if self.reward_scale:
            return float(self.reward_scale)
        from crucible.research.tasks import ENDOWMENT_TICKS
        return max(1.0, float(self.endowment) / ENDOWMENT_TICKS)

    @property
    def description(self) -> str:
        return (f"A production facility is available in this economy. Take a seat on it and operate it so that its "
                f"net income (revenue minus operating cost, in cents) over the next {self.horizon_s // DAY} days is "
                f"at least {self.goal_net_cents} cents.")

    @property
    def success(self) -> dict:
        return {"metric": "rig_net", "args": {"principal": "$self", "rig": self.rig, "window_s": 0}, "op": ">=",
                "value": int(self.goal_net_cents)}

    @property
    def id(self) -> str:
        return f"rig-{self.rig}-{self.goal_net_cents}"

    def make_objective(self):
        return RigNetObjective(self)


class RigNetObjective:
    """Dense verified reward: the change of the actor's realised facility net (operator metric `rig_net`, from the
    facility's own records) since the previous step, in units of the task's `reward_scale_cents`. Success is the
    net at the horizon (first crossing reported as ``reached_t``)."""

    def __init__(self, task: RigTask):
        self.task = task
        self.prev: dict = {}
        self.start: dict = {}
        self.success_t: dict = {}
        self.reached_t: dict = {}

    def _net(self, world, actor) -> float:
        from crucible.goals.metrics import evaluate_metric
        return float(evaluate_metric(world, "rig_net", {"principal": actor, "rig": self.task.rig, "window_s": 0}))

    def begin(self, world, actor) -> None:
        self.start[actor] = world.now
        self.prev[actor] = self._net(world, actor)

    def step_reward(self, world, actor) -> float:
        v = self._net(world, actor)
        r = (v - self.prev[actor]) / self.task.reward_scale_cents
        self.prev[actor] = v
        reached = v >= self.task.goal_net_cents and v != 0
        if actor not in self.reached_t and reached:
            self.reached_t[actor] = world.now
        if actor not in self.success_t and reached and world.now - self.start[actor] >= self.task.horizon_s:
            self.success_t[actor] = world.now
        return float(r)

    def status(self, world, actor) -> dict:
        from crucible.goals.metrics import evaluate_metric
        v = self._net(world, actor)
        g = max(1.0, float(self.task.goal_net_cents))
        ticks = evaluate_metric(world, "rig_ticks", {"rig": self.task.rig})
        el = world.now - self.start.get(actor, 0)
        return {"success": actor in self.success_t, "failed": actor not in self.success_t and el >= self.task.horizon_s,
                "progress": float(np.clip(v / g, 0.0, 1.0)) if self.task.goal_net_cents > 0 else 0.0,
                "success_t": self.success_t.get(actor), "reached_t": self.reached_t.get(actor), "elapsed_s": el,
                "reward_scale_cents": self.task.reward_scale_cents, "metrics": {"rig_net": v, "rig_ticks": ticks}}


def _rig_control_base():
    from crucible.research.llm_eval import RigExperimenter
    return RigExperimenter


class RigOperatorControl:
    """Non-privileged scripted control for `RigTask`: rig.list -> rig.take, then Crucible's `RigExperimenter`
    (one-lever-at-a-time screening, combination, exploitation) through ordinary tool calls."""

    def __new__(cls, rig: str = DEFAULT_RIG, horizon: Optional[int] = None, **kw):
        from crucible.agents.scripted import call, ok
        base = _rig_control_base()

        class _Ctl(base):
            name = "rig_operator_control"

            def script(self):
                r = yield call("rig.take", rig=rig)
                if not ok(r):
                    return
                yield from base.script(self)
        return _Ctl(horizon=horizon, **kw)
