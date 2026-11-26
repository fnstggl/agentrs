"""Tier 1 adapters: the 4.0 hidden-mechanism simulators (`MechVecEnv`, `SelfPlayEnv`) and the v5 `RapidEnv`
behind the one `ExperienceEnv` interface.

`MechExperienceEnv(task)` runs one facility of a hidden-mechanism task. Its tools are the kernel facility's own
(``rig.list``, ``rig.spec``, ``rig.set``, ``rig.read``, ``rig.history``, ``bank.accounts``), built from the same
endpoint definitions, schemas and descriptions as `crucible.mechanisms.module.Rig`, and its principals are those
of the kernel twin (``p:agent``; ``p:seat1``, ``p:seat2``, ... for multi-seat tasks). Settings apply from the next
tick; a tick happens every ``tick_s`` seconds of `advance`. So a learner that operates this environment operates
the kernel-hosted facility (`research.bridge.KernelMechEnv`, `KernelExperienceEnv` on ``mech_<family>``) with the
same actions. Underneath, the 4.0 numeric environment does the work, in the same reward mode, so its numeric view
(`numeric()`) keeps serving `research.learner.Learner` implementations unchanged.

Observations carry the exact report of the last tick (readings, revenue, cost, net, shortfall as the facility
reports them to its operator), the settings in force, the balance, the brief, and `rig.report` events whose `cause`
points at the `rig.set` that chose the settings they ran at. Hidden: how levers act; stated in `missing`. The task's
private reference, the episode's normalised score and the mechanism's creator-defined per-tick objective quantity
(`score`) are evaluator data (`verify()`); `score` appears in what the operator sees only in an explicitly selected
``legacy_shaped_reward`` run (the 4.0 contract).

Reward mode: ``observation_only`` unless another mode is named (Clarification 2). Time: `t` and every timestamp are
simulated seconds in every adapter (`TIME_UNIT`).
"""
from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from typing import Any

import numpy as np

from ..core import schema as S
from ..core.canonical import digest as _digest
from .env import BaseExperienceEnv, error_result
from .features import annotate_tools, flatten
from .modes import LEGACY, OBSERVATION_ONLY, VERIFIED, check_mode
from .observation import TIME_UNIT, Observation, json_copy

RIG_ID = "r1"


def _endpoint_spec(cls: type, attr: str, host: str) -> tuple[dict, dict]:
    meta = getattr(cls, attr).__crucible_endpoint__
    spec = {"name": f"{host}.{meta['name']}", "description": meta["doc"], "inputSchema": meta["params"],
            "annotations": {"readOnlyHint": bool(meta["read_only"]), "costSeconds": int(meta["cost_s"]),
                            "host": host, "builtin": True, "version": str(meta.get("version") or cls.version)}}
    return spec, meta


def _facility_tools() -> dict[str, tuple[dict, dict]]:
    from ..mechanisms.module import Rig
    from ..services.bank import Bank
    out = {}
    for attr in ("ep_list", "ep_spec", "ep_set", "ep_read", "ep_history"):
        spec, meta = _endpoint_spec(Rig, attr, "rig")
        out[spec["name"]] = (spec, meta)
    spec, meta = _endpoint_spec(Bank, "ep_accounts", "bank")
    out[spec["name"]] = (spec, meta)
    return out


class MechExperienceEnv(BaseExperienceEnv):
    tier = 1
    kind = "mech"

    def __init__(self, task, *, reward_mode: str | None = None, handles: list[str] | None = None,
                 tick_s: int | None = None, max_events: int = 1000, name: str | None = None):
        from ..mechanisms.module import MONEY_RULES, SCORE_TARGETS
        from ..research.envs import MechVecEnv
        from ..selfplay.arena import SelfPlayEnv
        super().__init__(check_mode(reward_mode), name or f"mech:{task.family}", task.seed)
        self.task = task
        self.n_seats = int(task.n_seats)
        self.multi = self.n_seats > 1
        self.vec = SelfPlayEnv([task], reward_mode=self.reward_mode) if self.multi else \
            MechVecEnv([task], reward_mode=self.reward_mode)
        self.vec.reset()
        self.handles = list(handles) if handles else (["agent"] if not self.multi else
                                                     [f"seat{s + 1}" for s in range(self.n_seats)])
        if len(self.handles) != self.n_seats:
            raise ValueError("one handle per seat")
        self._agents = [f"p:{h}" for h in self.handles]
        self._seat = {a: s for s, a in enumerate(self._agents)}
        self._acct = {a: f"acct_t1_{s:02d}" for s, a in enumerate(self._agents)}
        self.tick_s = int(tick_s or self._tick_s_of(task))
        self.max_events = int(max_events)
        self._money_rules, self._targets = dict(MONEY_RULES, lease_cents=0), list(SCORE_TARGETS)
        self._tools = _facility_tools()
        self._annotated = annotate_tools([v[0] for v in self._tools.values()])
        self._log: list[dict] = []          # this environment's event log (seq = index + 1)
        self._vis: dict[str, list[int]] = {a: [] for a in self._agents}
        self._cursor = {a: 0 for a in self._agents}
        self._last_result: dict[str, dict | None] = {a: None for a in self._agents}
        self._pending = {a: 0.0 for a in self._agents}
        self._actions = {a: 0 for a in self._agents}
        self._time_used = {a: 0 for a in self._agents}
        self._last_set: dict[str, dict | None] = {a: None for a in self._agents}
        self._reports: dict[str, list[dict]] = {a: [] for a in self._agents}
        self._now = 0
        self._t0 = 0                         # clock at the start of the current episode
        self.episode = 0
        self._begin_episode()

    # ------------------------------------------------------------------ construction
    @staticmethod
    def _tick_s_of(task) -> int:
        from ..worldgen import generate
        import crucible
        crucible.load_all_modules()
        cfg, seed = task.world_config()
        for m in generate(cfg, seed).modules:
            if m["name"] == "rig":
                return int(m["config"]["rigs"][0].get("tick_s", 3600))
        return 3600

    def _begin_episode(self) -> None:
        sim = self.sims
        self._levers = [list(sim.engine.levers(s)) for s in range(self.n_seats)]
        self._gauges = [list(sim.engine.gauges(s)) for s in range(self.n_seats)]
        self._desired = [dict(sim.settings[s]) for s in range(self.n_seats)]
        for a in self._agents:
            self._reports[a] = []
            self._last_set[a] = None
        self._t0 = self._now
        self._emit("rig.episode", {"rig": RIG_ID, "episode": self.episode, "task": self.task.id,
                                   "horizon_ticks": int(self.task.horizon), "tick_s": self.tick_s}, "public")

    @property
    def sims(self):
        return self.vec.sims[0]

    # ------------------------------------------------------------------ log
    def _emit(self, typ: str, payload: dict, vis, actor: str = "sys", cause: dict | None = None) -> int:
        seq = len(self._log) + 1
        ev = {"seq": seq, "t": int(self._now), "type": typ, "actor": actor, "payload": payload, "vis": vis,
              "cause": cause}
        self._log.append(ev)
        for a in self._agents:
            if vis == "public" or (isinstance(vis, (list, tuple)) and a in vis):
                if typ == "cmd" and actor == a:
                    continue                 # an agent's own command is not news to it
                self._vis[a].append(seq)
        return seq

    def _event_view(self, agent: str, ev: dict) -> dict:
        c = ev["cause"]
        cause = None
        if c is not None and c.get("agent") == agent:
            cause = {"action_seq": c["action_seq"], "tool": c["tool"], "t": c["t"], "delay_s": int(ev["t"] - c["t"]),
                     "via": c["via"]}
        return {"seq": ev["seq"], "t": ev["t"], "type": ev["type"], "actor": ev["actor"],
                "payload": json_copy(ev["payload"]), "cause": cause}

    # ------------------------------------------------------------------ protocol
    def agents(self) -> list[str]:
        return list(self._agents)

    @property
    def now(self) -> int:
        return int(self._now)

    @property
    def ticks_done(self) -> int:
        return int(self.sims.t)

    @property
    def done(self) -> bool:
        return bool(self.vec.done[0])

    def next_tick_at(self) -> int:
        return int(self._t0 + (self.ticks_done + 1) * self.tick_s)

    def tools(self, agent: str) -> list[dict]:
        self._check(agent)
        return [dict(t, annotations=dict(t["annotations"])) for t in self._annotated]

    def _check(self, agent: str) -> None:
        if agent not in self._seat:
            raise KeyError(f"unknown agent {agent!r}; agents: {self._agents}")

    def execute(self, agent: str, action: dict) -> dict:
        self._check(agent)
        self.last_changed = False
        tool = action.get("tool") if isinstance(action, dict) else None
        args = action.get("args", {}) if isinstance(action, dict) else None
        if not isinstance(tool, str) or not isinstance(args, dict):
            res = error_result(self._now, tool if isinstance(tool, str) else "", "invalid_action",
                               "action must be {'tool': str, 'args': object}")
            self._last_result[agent] = res
            return res
        ent = self._tools.get(tool)
        if ent is None:
            res = error_result(self._now, tool, "unknown_tool", f"unknown tool {tool!r}; tools: {sorted(self._tools)}")
            self._last_result[agent] = res
            return res
        spec, meta = ent
        args = S.with_defaults(meta["params"], args)
        errs = S.validate(meta["params"], args)
        if errs:
            res = error_result(self._now, tool, "invalid_action", "; ".join(errs[:6]))
            self._last_result[agent] = res
            return res
        if args.get("rig") not in (None, RIG_ID):
            res = error_result(self._now, tool, "not_found", f"no facility {args.get('rig')!r} (this world has {RIG_ID})")
            self._last_result[agent] = res
            return res
        seq = self._emit("cmd", {"tool": tool, "args": json_copy(args), "cost_s": int(meta["cost_s"])}, (agent,),
                         actor=agent)
        s = self._seat[agent]
        before = dict(self._desired[s])
        try:
            out = getattr(self, "_t_" + tool.replace(".", "_"))(agent, s, args, seq)
            res = {"ok": True, "t": int(self._now), "tool": tool, "cost_s": int(meta["cost_s"]), "result": out}
        except _Refused as e:
            res = {"ok": False, "t": int(self._now), "tool": tool, "cost_s": int(meta["cost_s"]),
                   "error": {"code": e.code, "message": str(e)}}
        res["action_seq"] = seq                                # what `cause.action_seq` of its consequences names
        self._actions[agent] += 1
        self._time_used[agent] += int(meta["cost_s"])
        self._last_result[agent] = res
        self.steps += 1
        self.last_changed = self._desired[s] != before        # the facility will run differently
        self.changes += int(self.last_changed)
        return json_copy(res)

    # tools ---------------------------------------------------------------------------------------------
    def _report_view(self, rep: dict) -> dict:
        keys = ("t", "time", "status", "readings", "revenue_cents", "cost_cents", "net_cents", "shortfall_cents") + \
            (("score",) if self.reward_mode == LEGACY else ()) + ("settings",)
        return {k: json_copy(rep[k]) for k in keys}

    def _t_rig_list(self, agent, s, args, seq):
        return [{"rig": RIG_ID, "tick_s": self.tick_s, "ticks_done": self.ticks_done, "next_tick_at": self.next_tick_at(),
                 "lease_cents": 0, "seats": [{"seat": k, "status": "held", "holder": a, "yours": a == agent}
                                             for k, a in enumerate(self._agents)]}]

    def _t_rig_spec(self, agent, s, args, seq):
        seat = s if args.get("seat") is None else int(args["seat"])
        if seat >= self.n_seats:
            raise _Refused("invalid_action", f"facility {RIG_ID} has seats 0..{self.n_seats - 1}")
        pub = self.sims.engine.public_spec(seat)
        return json_copy({"rig": RIG_ID, "seat": seat, "tick_s": self.tick_s, **pub, "money": self._money_rules,
                          "targets": [g.name for g in self._gauges[seat]] + self._targets})

    def _t_rig_set(self, agent, s, args, seq):
        from ..mechanisms.sim import clip_value
        settings = args["settings"]
        by = {lv.name: lv for lv in self._levers[s]}
        unknown = sorted(k for k in settings if k not in by)
        if unknown:
            raise _Refused("invalid_action", f"unknown lever name(s) {unknown} for facility {RIG_ID} seat {s}; "
                                             f"valid names: {sorted(by)}")
        if self.done:
            raise _Refused("conflict", "the episode is over: the facility runs no more ticks")
        cur = dict(self._desired[s])
        clipped = []
        for k in sorted(settings):
            lv = by[k]
            v = clip_value(settings[k], lv.lo, lv.hi, lv.default, lv.integer)
            if v != settings[k]:
                clipped.append(k)
            cur[k] = v
        self._desired[s] = cur
        cause = {"agent": agent, "action_seq": seq, "tool": "rig.set", "t": int(self._now)}
        if settings:
            self._emit("rig.set", {"rig": RIG_ID, "seat": s, "settings": dict(cur), "by": agent}, (agent,),
                       actor=agent, cause=dict(cause, via="direct"))
            self._last_set[agent] = cause
        return {"rig": RIG_ID, "seat": s, "settings": dict(cur), "clipped": clipped, "applies_at_tick": self.ticks_done,
                "applies_at": self.next_tick_at()}

    def _t_rig_read(self, agent, s, args, seq):
        reps = self._reports[agent]
        return {"rig": RIG_ID, "seat": s, "ticks_done": self.ticks_done, "next_tick_at": self.next_tick_at(),
                "next_settings": dict(self._desired[s]), "report": self._report_view(reps[-1]) if reps else None}

    def _t_rig_history(self, agent, s, args, seq):
        n = int(args.get("n") or 20)
        return {"rig": RIG_ID, "seat": s, "reports": [self._report_view(r) for r in self._reports[agent][-n:]]}

    def _t_bank_accounts(self, agent, s, args, seq):
        bal = self._balance(agent)
        return [{"account": self._acct[agent], "owner": agent, "name": "personal", "balance": bal,
                 "balance_fmt": f"${bal / 100:,.2f}", "overdraft": 0, "status": "open"}]

    def _balance(self, agent: str) -> int:
        f = self.sims.funds[self._seat[agent]]
        return 0 if f is None else int(round(float(f)))

    # time ----------------------------------------------------------------------------------------------
    def advance(self, seconds: int) -> None:
        target = self._now + max(0, int(seconds))
        while not self.done and self.next_tick_at() <= target:
            self._now = self.next_tick_at()
            self._tick()
        self._now = target

    def _tick(self) -> None:
        sp = self.vec.spec
        vals = []
        for s in range(self.n_seats):
            v = np.zeros((1, sp.max_levers), dtype=np.float64)
            for j, lv in enumerate(self._levers[s]):
                v[0, j] = float(self._desired[s][lv.name])     # exact: the value chosen through rig.set
            vals.append(v)
        if self.multi:
            _, rew, _, _ = self.vec.step_values(vals)
            per_seat = [float(rew[s, 0]) for s in range(self.n_seats)]
        else:
            _, rew, _, _ = self.vec.step_values(vals[0])
            per_seat = [float(rew[0])]
        sim = self.sims
        t = sim.t - 1
        for s, a in enumerate(self._agents):
            self._pending[a] += per_seat[s]
            o = sim.last[s]
            rep = {"rig": RIG_ID, "seat": s, "t": int(t), "time": int(self._now), "holder": a, "status": o["status"],
                   "readings": {k: float(v) for k, v in o["readings"].items()},
                   "revenue_cents": int(round(float(o["revenue"]))), "cost_cents": int(round(float(o["cost"]))),
                   "shortfall_cents": int(round(float(o["shortfall"]))), "score": float(o["score"]),
                   "settings": {k: float(v) for k, v in o["settings"].items()}}
            rep["net_cents"] = rep["revenue_cents"] - rep["cost_cents"]
            self._reports[a].append(rep)
            c = self._last_set[a]
            hide = ("time",) if self.reward_mode == LEGACY else ("time", "score")
            self._emit("rig.report", {k: v for k, v in rep.items() if k not in hide}, (a,), cause=None if c is None
                       else dict(c, via="settings_in_force"))
        if self.done:
            self._emit("rig.episode_end", {"rig": RIG_ID, "episode": self.episode, "ticks": int(sim.t)}, "public")

    # observation ---------------------------------------------------------------------------------------
    def observe(self, agent: str) -> Observation:
        self._check(agent)
        s = self._seat[agent]
        vis = self._vis[agent]
        start = bisect_right(vis, self._cursor[agent])
        pending = vis[start:]
        deliver = pending[:self.max_events]
        cursor = deliver[-1] if len(deliver) < len(pending) else len(self._log)
        events = [self._event_view(agent, self._log[q - 1]) for q in deliver]
        self._cursor[agent] = cursor
        pre = f"rig.{RIG_ID}"
        # the same paths as the kernel facility's own summary (`Rig.observe` flattened: rig.seats.r1.*), plus what
        # rig.read would return (revenue, cost, shortfall, next settings) and the facility's public timing
        seat_view: dict[str, Any] = {"seat": s, "ticks_done": self.ticks_done}
        reps = self._reports[agent]
        if reps:
            r = reps[-1]
            seat_view.update(t=r["t"], readings=dict(r["readings"]), net_cents=r["net_cents"],
                             revenue_cents=r["revenue_cents"], cost_cents=r["cost_cents"],
                             shortfall_cents=r["shortfall_cents"])
            if self.reward_mode == LEGACY:
                seat_view["score"] = r["score"]         # the 4.0 operator view (evaluator data in v5 modes)
        seat_view["next_settings"] = {lv.name: float(self._desired[s][lv.name]) for lv in self._levers[s]}
        seat_view.update(tick_s=self.tick_s, next_tick_at=self.next_tick_at(), horizon_ticks=int(self.task.horizon))
        numbers, arrays = flatten({"rig": {"seats": {RIG_ID: seat_view}},
                                   "bank": {"balances": {self._acct[agent]: self._balance(agent)}}})
        numbers["world.time_s"] = int(self._now)
        numbers["world.day"] = int(self._now // 86400)
        numbers["world.tick_s"] = int(self.tick_s)
        if reps:
            arrays[f"{pre}.readings"] = [reps[-1]["readings"][g.name] for g in self._gauges[s]]
        arrays[f"{pre}.settings"] = [float(self._desired[s][lv.name]) for lv in self._levers[s]]
        arrays[f"{pre}.lever_min"] = [float(lv.lo) for lv in self._levers[s]]
        arrays[f"{pre}.lever_max"] = [float(lv.hi) for lv in self._levers[s]]
        missing = [f"{pre}: how the levers act on the gauges, revenue and cost is not observable (find out by acting)"]
        if self.done:
            missing.append(f"{pre}: the episode is over after {self.ticks_done} ticks; no further reports")
        else:
            missing.append(f"{pre}.report: the next report arrives at t={self.next_tick_at()}")
        if not reps:
            missing.append(f"{pre}.report: no report yet (the first tick has not run)")
        if len(deliver) < len(pending):
            missing.append(f"events: {len(pending) - len(deliver)} more visible events after cursor {cursor}; "
                           f"read them with observe_since(agent, {cursor})")
        me = {"id": agent, "kind": "person", "attrs": {}, "relations": [{"rel": "operates_seat", "to": RIG_ID},
                                                                        {"rel": "holds", "to": self._acct[agent]}]}
        ents = [me, {"id": RIG_ID, "kind": "facility", "attrs": {"seats": self.n_seats, "tick_s": self.tick_s},
                     "relations": [{"rel": "operated_by", "to": agent}]},
                {"id": self._acct[agent], "kind": "account", "attrs": {"balance_cents": self._balance(agent)},
                 "relations": [{"rel": "owned_by", "to": agent}]}]
        pub = self.task.public()
        obs = Observation(
            t=int(self._now), agent=agent, numbers=numbers, arrays=arrays,
            text=[f"{pre}.brief: {self.sims.engine.brief(s)}"], artifacts=[], tools=self.tools(agent),
            events=events, entities=ents,
            resources={"time_unit": TIME_UNIT, "actions": self._actions[agent], "action_time_s": self._time_used[agent],
                       "balance_cents": self._balance(agent),
                       "ticks_left": max(0, int(self.task.horizon) - self.ticks_done),
                       "levers": len(self._levers[s])},
            analytics={f"{pre}.ticks_done": self.ticks_done, f"{pre}.seats": self.n_seats,
                       f"{pre}.seats_held": self.n_seats},
            last_result=json_copy(self._last_result[agent]), missing=missing, cursor=int(cursor),
            objective=pub.get("objective"))
        return obs

    def observe_since(self, agent: str, cursor: int) -> tuple[list[dict], int]:
        self._check(agent)
        vis = self._vis[agent]
        start = bisect_right(vis, int(cursor))
        return [self._event_view(agent, self._log[q - 1]) for q in vis[start:]], len(self._log)

    def reward(self, agent: str) -> float:
        self._check(agent)
        r, self._pending[agent] = self._pending[agent], 0.0
        return 0.0 if self.reward_mode == OBSERVATION_ONLY else float(r)

    def verify(self) -> dict:
        """Evaluator side: the operator records of finished episodes (normalised score against the task's private
        reference, success against the public objective) and the running totals of the current one."""
        self.verifications += 1
        recs = [dict(r) for r in self.vec.finished]
        cur = {}
        if not self.done:
            ep = self.vec._ep[0]
            if self.multi:
                cur = {"ticks": self.ticks_done, "net_cents": [float(x) for x in ep["net"]]}
            else:
                cur = {"ticks": self.ticks_done, "net_cents": float(ep["net"]),
                       "success_so_far": self.task.success(float(ep["net"]))}
        return {"verifier": "research.tasks.Task (operator-side episode records)", "task": self.task.id,
                "validated": bool(self.task.validated), "episodes": recs, "current": cur,
                "agents": {a: {"success": (recs[-1]["success"] if recs and not self.multi else
                                           (self.task.success(recs[-1]["seats"][s]["net_cents"]) if recs else None))}
                           for s, a in enumerate(self._agents)}}

    # operator side -------------------------------------------------------------------------------------
    def reset(self, task=None) -> None:
        """A new episode: the same task replayed (same noise) or `task` installed. The clock keeps running."""
        if task is not None:
            if int(task.n_seats) != self.n_seats:
                raise ValueError("a replacement task needs the same number of seats")
            self.task = task
            self.vec.reset(tasks=[task])
        else:
            self.vec.reset()
        self.episode += 1
        self._begin_episode()

    def numeric(self):
        """The underlying 4.0 numeric environment (same reward mode), for `research.learner.Learner`s."""
        return self.vec

    def digest(self) -> str:
        sim = self.sims
        st = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in sorted(sim.state.items())}
        return _digest([int(sim.t), [dict(x) for x in sim.settings], [None if f is None else float(f) for f in sim.funds],
                        st, [dict(d) for d in self._desired]])

    def describe(self) -> dict:
        d = super().describe()
        d.update(time_unit=TIME_UNIT, task=self.task.id, world_id=f"t1-{self.task.family}-{self.task.seed}", spec_hash=_digest(
            json.loads(json.dumps(self.task.public(), sort_keys=True))), agents=list(self._agents),
            tick_s=self.tick_s, seats=self.n_seats)
        return d


class _Refused(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(msg)
        self.code = code


# ====================================================================== v5 Tier 1 (workstream H's RapidEnv)
class RapidExperienceEnv(BaseExperienceEnv):
    """`crucible.rapid.make(family, seed, **difficulty)` behind the ExperienceEnv interface, for one or several
    participants (INTERFACE_CONTRACTS section 8: ``reset()``, ``step({agent: action})``, ``outcome()``).

    Time. A rapid world advances in ticks; the adapter states every time in simulated seconds (`TIME_UNIT`): a tick
    lasts `tick_s` seconds (the family's ``tick_s`` if it has one, else ``DEFAULT_TICK_S``), `t` = ticks done x
    `tick_s`, and every action reports ``cost_s = tick_s``: it occupies the tick it runs in. A tick that an action
    ran counts against the next `advance` (execute-then-advance(tick_s) is one tick, as is a bare advance(tick_s)), so
    a driver that acts or waits once per decision moves the world one tick per decision either way.

    Actions. ``{"tool": "rapid.<kind>", "args": {...}}`` (or the rapid form ``{"kind": ...}``). With one participant
    an action runs at once: one tick passes and `execute` returns its result. With several, actions are *joint*:
    `execute` validates the action and queues it for the next tick (``result = {"queued": true, "tick": k}``); the
    tick runs when every participant has queued an action, or at `advance` (participants without an action wait).
    Each participant's own result of that tick is then its `last_result`, and an ``action.result`` event visible
    only to it whose `cause` names the action (``via: "direct"``). Coordination plans that every named party queued
    for the same tick are carried out together, as the rapid world's own rule says.

    Events. Each participant receives exactly the rapid world's events it may see (the world's own visibility rule:
    a message is seen by its sender and its recipient only), once, numbered by their position in the world's log
    (`seq`), with `t` in seconds. A ``rapid.observation`` event per tick carries the participant's own exact numbers
    and resources of that tick; its `cause` points at the participant's latest action in force (``via: "in_force"``,
    with `delay_s`), so a learner can relate later outcomes to what it did. Nothing else is attributed: no event names
    a cause the participant did not take.

    Reward modes: ``observation_only`` (default) pays 0.0; ``verified_terminal_outcome`` pays 1.0 once, at the end of
    the episode, when `outcome()["success"]` holds; ``legacy_shaped_reward`` pays the ``reward`` of a rapid step
    result if a family has one (v5 families have none). `outcome()` stays on the evaluator side (`verify`).
    `rapid_env=` installs an already built `RapidEnv` (a world drawn by `crucible.rapid.generator`)."""
    tier = 1
    kind = "rapid"
    DEFAULT_TICK_S = 3600
    QUIET_KINDS = ("query",)                # kinds that only read: they never change the world
    # identity of the generated world in a `query spec` result: a memorisation key and an evaluation-partition
    # label, not an observation (kept only in an explicitly selected legacy run)
    SPEC_IDENTITY_KEYS = ("family", "seed", "category", "split")

    def __init__(self, family: str, seed: int = 0, *, reward_mode: str | None = None, rapid_env=None,
                 tick_s: int | None = None, **difficulty):
        from .. import rapid
        if not hasattr(rapid, "make"):
            raise ImportError("crucible.rapid.make is not available (workstream H not merged)")
        super().__init__(check_mode(reward_mode), f"rapid:{family}", seed)
        self.env = rapid_env if rapid_env is not None else rapid.make(family, int(seed), **difficulty)
        cfg = getattr(self.env, "_config", {}) or {}
        self.tick_s = int(tick_s or cfg.get("tick_s") or self.DEFAULT_TICK_S)
        first = self.env.reset()
        ags = getattr(self.env, "agents", None)
        ags = ags() if callable(ags) else ags
        self._agents = [str(a) for a in (ags or [first.get("agent", "a0")])]
        self.multi = len(self._agents) > 1
        self.episode = 0
        self._begin()

    # ------------------------------------------------------------------ bookkeeping
    def _begin(self) -> None:
        self._log: list[dict] = []          # the adapter's event log: one entry per raw world event (seq = index+1)
        self._raw_seen = 0                  # raw world events already converted
        self._own: dict[str, list[int]] = {a: [] for a in self._agents}   # adapter-made events per agent
        self._vis: dict[str, list[int]] = {a: [] for a in self._agents}
        self._cursor = {a: 0 for a in self._agents}
        self._queue: dict[str, tuple[dict, int, str]] = {}
        self._last: dict[str, dict | None] = {a: None for a in self._agents}
        self._in_force: dict[str, dict | None] = {a: None for a in self._agents}
        self._pending = {a: 0.0 for a in self._agents}
        self._paid: set[str] = set()
        self._counts = {a: {"executes": 0, "rejected": 0, "cost_s": 0, "cost_cents": 0.0} for a in self._agents}
        self._wait = 0                      # seconds advanced that do not yet fill a tick
        self._credit = 0                    # seconds of the next advance already spent by a tick an action ran
        self._obs = {a: self._raw_obs(a) for a in self._agents}
        self._sync()

    def _raw_obs(self, agent: str) -> dict:
        fn = getattr(self.env, "observe", None)
        return fn(agent) if callable(fn) else {}

    def _raw_events(self) -> list[dict]:
        """The rapid world's log (with visibility). `RapidEnv` keeps it in `_events`; an env without it offers only
        `observe_since`, which the adapter then reads per participant."""
        evs = getattr(self.env, "_events", None)
        if evs is None:
            raise TypeError("this RapidEnv exposes no event log; RapidExperienceEnv needs Env._events")
        return evs

    def _visible(self, e: dict, agent: str) -> bool:
        v = e.get("vis", "public")
        return v == "public" or (isinstance(v, (list, tuple)) and agent in v)

    def _emit(self, typ: str, payload: dict, vis, actor: str = "world", cause: dict | None = None,
              t: int | None = None) -> int:
        seq = len(self._log) + 1
        ev = {"seq": seq, "t": int(self.now if t is None else t), "type": typ, "actor": actor, "payload": payload,
              "vis": vis, "cause": cause}
        self._log.append(ev)
        for a in self._agents:
            if self._visible(ev, a):
                self._vis[a].append(seq)
        return seq

    def _sync(self) -> None:
        """Convert the rapid world's new raw events into the adapter's log (each once, with its own visibility)."""
        raw = self._raw_events()
        tick = int(getattr(self.env, "t", 0))
        for e in raw[self._raw_seen:]:
            e = dict(e)
            vis = e.pop("vis", "public")
            typ = str(e.pop("type", "event"))
            et = e.pop("t", None)
            actor = str(e.pop("from", "world"))
            t = (int(et) if isinstance(et, (int, float)) else max(0, tick - 1)) * self.tick_s
            self._emit(typ, json_copy(e), vis if vis == "public" else list(vis), actor=actor, t=t)
        self._raw_seen = len(raw)

    # ------------------------------------------------------------------ protocol
    def agents(self) -> list[str]:
        return list(self._agents)

    @property
    def now(self) -> int:
        return int(getattr(self.env, "t", 0)) * self.tick_s

    @property
    def done(self) -> bool:
        return bool(getattr(self.env, "done", False))

    @property
    def horizon(self) -> int:
        return int(getattr(self.env, "horizon", 0))

    def _check(self, agent: str) -> None:
        if agent not in self._vis:
            raise KeyError(f"unknown agent {agent!r}; agents: {self._agents}")

    def tools(self, agent: str) -> list[dict]:
        """The world's action kinds as interface tools ``rapid.<kind>``; each tool's input schema is the kind's
        action schema."""
        self._check(agent)
        fn = getattr(self.env, "tools", None)
        tools = fn(agent) if callable(fn) else (self._obs[agent].get("tools") or [])
        out = []
        for t in json_copy(list(tools)):
            name = str(t.get("name", ""))
            t["name"] = name if "." in name else f"rapid.{name}"
            ann = dict(t.get("annotations") or {})
            ann.setdefault("host", "rapid")
            ann.setdefault("readOnlyHint", name in self.QUIET_KINDS)
            ann.setdefault("costSeconds", self.tick_s)
            t["annotations"] = ann
            out.append(t)
        return annotate_tools(out)

    @staticmethod
    def _rapid_action(action: Any) -> Any:
        if not isinstance(action, dict) or "kind" in action or "tool" not in action:
            return action
        kind = str(action["tool"])
        kind = kind[len("rapid."):] if kind.startswith("rapid.") else kind
        args = action.get("args")
        args = dict(args) if isinstance(args, dict) else {}
        args.setdefault("kind", kind)
        return args

    def execute(self, agent: str, action: dict) -> dict:
        self._check(agent)
        self.last_changed = False
        tool = action.get("tool", action.get("kind", "")) if isinstance(action, dict) else ""
        tool = str(tool) if "." in str(tool) or not tool else f"rapid.{tool}"
        if self.done:
            return self._finish(agent, error_result(self.now, tool, "conflict", "the episode is over (horizon "
                                                                                 "reached)"))
        act = self._rapid_action(action)
        space = self.env.action_space(agent) if hasattr(self.env, "action_space") else None
        errs = space.validate(act) if space is not None else []
        if errs:
            return self._finish(agent, error_result(self.now, tool, "invalid_action", "; ".join(errs[:5])))
        seq = self._emit("cmd", {"tool": tool, "args": json_copy(act)}, [agent], actor=agent)
        self._own[agent].append(seq)
        kind = act.get("kind") if isinstance(act, dict) else None
        if not self.multi:
            self._queue[agent] = (act, seq, tool)
            res = self._run_tick()[agent]
            self._credit = self.tick_s
            return self._finish(agent, res, changed=bool(res.get("ok")) and kind not in self.QUIET_KINDS)
        if agent in self._queue:
            self._own[agent].pop()
            return self._finish(agent, error_result(self.now, tool, "conflict",
                                                    "you already queued an action for this tick; it runs when every "
                                                    "participant has acted or time advances"))
        self._queue[agent] = (act, seq, tool)
        res = {"ok": True, "t": self.now, "tool": tool, "cost_s": self.tick_s, "action_seq": seq,
               "result": {"queued": True, "tick": int(getattr(self.env, "t", 0)), "runs_at": self.now + self.tick_s,
                          "note": "joint action: the result arrives as last_result and an action.result event"}}
        out = self._finish(agent, res, changed=False, counted_cost=True)
        if len(self._queue) == len(self._agents):
            self._run_tick()
            self._credit = self.tick_s
        return out

    def _finish(self, agent: str, res: dict, changed: bool = False, counted_cost: bool = True) -> dict:
        cnt = self._counts[agent]
        cnt["executes"] += 1
        cnt["rejected"] += 0 if res.get("ok") else 1
        cnt["cost_s"] += int(res.get("cost_s") or 0)
        self.steps += 1
        self.last_changed = bool(changed)
        self.changes += int(self.last_changed)
        if not self.multi or not res.get("ok"):
            self._last[agent] = json_copy(res)
        return json_copy(res)

    def _run_tick(self) -> dict[str, dict]:
        """One tick of the rapid world with every queued action; returns each actor's interface result."""
        queued = dict(self._queue)
        self._queue.clear()
        t_before = self.now
        cash0 = {a: self._cash(a) for a in self._agents}
        out = self.env.step({a: q[0] for a, q in queued.items()} if queued else None)
        results = out.get("results")
        if not isinstance(results, dict):
            r0 = dict(out.get("result") or {})
            results = {self._agents[0]: r0}
        self._sync()
        ret = {}
        for a in self._agents:
            self._obs[a] = self._raw_obs(a)
            cost_cents = max(0.0, cash0[a] - self._cash(a)) if cash0[a] is not None else None
            if a not in queued:
                continue
            act, seq, tool = queued[a]
            raw = dict(results.get(a) or {})
            if self.reward_mode == LEGACY and isinstance(raw.get("reward"), (int, float)):
                self._pending[a] += float(raw["reward"])
            raw.pop("reward", None)
            r = {"ok": bool(raw.get("ok", False)), "t": int(t_before), "tool": tool, "cost_s": self.tick_s,
                 "action_seq": seq}
            if r["ok"]:
                r["result"] = self._participant_result(json_copy(raw.get("result", raw.get("results"))))
            else:
                err = raw.get("error")
                r["error"] = err if isinstance(err, dict) else {"code": "refused", "message": str(err)}
            self._counts[a]["cost_cents"] += float(cost_cents or 0.0)
            r["cost_cents"] = None if cost_cents is None else float(cost_cents)
            cause = {"agent": a, "action_seq": seq, "tool": tool, "t": int(t_before)}
            self._emit("action.result", json_copy({k: v for k, v in r.items() if k != "action_seq"}), [a], actor=a,
                       cause=dict(cause, via="direct"))
            if r["ok"] and (act.get("kind") if isinstance(act, dict) else None) not in self.QUIET_KINDS:
                self._in_force[a] = cause
            if self.multi:
                self._last[a] = json_copy(r)
                if r["ok"] and (act.get("kind") if isinstance(act, dict) else None) not in self.QUIET_KINDS:
                    self.changes += 1                  # a joint action carried out (operator-side count)
            ret[a] = r
        for a in self._agents:
            ob = self._obs[a]
            c = self._in_force.get(a)
            self._emit("rapid.observation", {"tick": int(getattr(self.env, "t", 0)) - 1,
                                              "numbers": json_copy(ob.get("numbers") or {}),
                                              "resources": json_copy(ob.get("resources") or {})}, [a],
                       cause=None if c is None else dict(c, via="in_force"))
        if self.done and self.reward_mode == VERIFIED:
            oc = self.env.outcome()
            for a in self._agents:
                if oc.get("success") and a not in self._paid:
                    self._paid.add(a)
                    self._pending[a] += 1.0
        return ret

    def _participant_result(self, res: Any) -> Any:
        if self.reward_mode == LEGACY:
            return res
        if isinstance(res, list):
            return [self._participant_result(x) for x in res]
        if isinstance(res, dict) and isinstance(res.get("spec"), dict):
            res = dict(res, spec={k: v for k, v in res["spec"].items() if k not in self.SPEC_IDENTITY_KEYS})
        return res

    @staticmethod
    def _entities(raw: list) -> list[dict]:
        """Rapid entities in the contract's shape ``{id, kind, relations, attrs}`` (a message entity
        ``{kind, from, content}`` becomes ``{id: "message:<n>", kind: "message", relations: [{rel: "from", to}],
        attrs: {content}}``); valid entities pass unchanged."""
        out = []
        for n, e in enumerate(raw or []):
            if not isinstance(e, dict):
                continue
            if isinstance(e.get("id"), str) and isinstance(e.get("kind"), str) and isinstance(e.get("relations", []),
                                                                                              list):
                out.append(e)
                continue
            kind = str(e.get("kind", "entity"))
            rel = [{"rel": k, "to": e[k]} for k in ("from", "to") if isinstance(e.get(k), str)]
            attrs = {k: v for k, v in e.items() if k not in ("id", "kind", "from", "to", "relations")}
            out.append({"id": str(e.get("id") or f"{kind}:{n}"), "kind": kind, "relations": rel, "attrs": attrs})
        return out

    def _cash(self, agent: str) -> float | None:
        res = (self._obs.get(agent) or {}).get("resources") or {}
        v = res.get("cash_cents")
        return float(v) if isinstance(v, (int, float)) else None

    def advance(self, seconds: int) -> None:
        """Time passes: every whole tick inside `seconds` runs (queued actions in the first; everyone else waits).
        Shorter waits accumulate until a tick is due."""
        sec = max(0, int(seconds))
        used = min(self._credit, sec)
        self._credit -= used
        self._wait += sec - used
        while self._wait >= self.tick_s and not self.done:
            self._wait -= self.tick_s
            self._run_tick()
        if self.done:
            self._wait = 0

    def _event_view(self, agent: str, ev: dict) -> dict:
        c = ev["cause"]
        cause = None
        if c is not None and c.get("agent") == agent:
            cause = {"action_seq": c["action_seq"], "tool": c["tool"], "t": c["t"], "delay_s": int(ev["t"] - c["t"]),
                     "via": c["via"]}
        return {"seq": ev["seq"], "t": ev["t"], "type": ev["type"], "actor": ev["actor"],
                "payload": json_copy(ev["payload"]), "cause": cause}

    def _visible_seqs(self, agent: str) -> list[int]:
        own = set(self._own[agent])          # an agent's own command is not news to it
        return [q for q in self._vis[agent] if q not in own]

    def observe(self, agent: str) -> Observation:
        self._check(agent)
        raw = json_copy(self._obs[agent])
        vis = self._visible_seqs(agent)
        i = bisect_right(vis, self._cursor[agent])
        events = [self._event_view(agent, self._log[q - 1]) for q in vis[i:]]
        cursor = len(self._log)
        self._cursor[agent] = cursor
        numbers = dict(raw.get("numbers") or {})
        tick = int(getattr(self.env, "t", 0))
        numbers.update({"world.time_s": self.now, "world.tick": tick, "world.tick_s": self.tick_s,
                        "world.ticks_left": max(0, self.horizon - tick)})
        cnt = self._counts[agent]
        resources = dict(raw.get("resources") or {})
        resources.update({"time_unit": TIME_UNIT, "executes": cnt["executes"], "rejected": cnt["rejected"],
                          "cost_s_total": cnt["cost_s"], "cost_cents_total": round(cnt["cost_cents"], 6),
                          "queued_action": agent in self._queue})
        missing = list(raw.get("missing") or [])
        if self.multi:
            missing.append("rapid: other participants' private messages, settings and readings are not observable")
        if self.done:
            missing.append("rapid: the episode is over (horizon reached)")
        return Observation(t=self.now, agent=agent, numbers=numbers, arrays=raw.get("arrays") or {},
                           text=list(raw.get("text") or []), artifacts=list(raw.get("artifacts") or []),
                           tools=self.tools(agent), events=events, entities=self._entities(raw.get("entities")),
                           resources=resources, analytics=dict(raw.get("analytics") or {}),
                           last_result=json_copy(self._last[agent]), missing=missing, cursor=int(cursor),
                           objective=raw.get("objective"))

    def observe_since(self, agent: str, cursor: int) -> tuple[list[dict], int]:
        self._check(agent)
        vis = self._visible_seqs(agent)
        i = bisect_right(vis, int(cursor))
        return [self._event_view(agent, self._log[q - 1]) for q in vis[i:]], len(self._log)

    def reset(self) -> None:
        """A new episode of the same world (same seed, same noise)."""
        self.env.reset()
        self.episode += 1
        self._begin()

    def reward(self, agent: str) -> float:
        self._check(agent)
        r, self._pending[agent] = self._pending[agent], 0.0
        return 0.0 if self.reward_mode == OBSERVATION_ONLY else float(r)

    def verify(self) -> dict:
        """Evaluator side: the rapid world's own outcome measurement."""
        self.verifications += 1
        oc = self.env.outcome()
        return {"verifier": "crucible.rapid RapidEnv.outcome", "outcome": oc,
                "agents": {a: {"success": oc.get("success")} for a in self._agents}}

    def digest(self) -> str:
        """The rapid world's state: tick, hidden state arrays, ledgers, and the adapter's event log."""
        st = getattr(self.env, "_state", None)
        arrays = {}
        if isinstance(st, dict):
            for k, v in sorted(st.items()):
                arrays[k] = hashlib.sha256(np.ascontiguousarray(np.asarray(v)).tobytes()).hexdigest() \
                    if isinstance(v, np.ndarray) else json.dumps(v, sort_keys=True, default=str)
        led = getattr(self.env, "_ledger", None)
        return _digest([self.name, self.seed, self.episode, int(getattr(self.env, "t", 0)), arrays,
                        json.loads(json.dumps(led, sort_keys=True, default=str)) if led is not None else None,
                        [[e["seq"], e["t"], e["type"]] for e in self._log]])

    def describe(self) -> dict:
        d = super().describe()
        d.update(time_unit=TIME_UNIT, tick_s=self.tick_s, agents=list(self._agents), joint_actions=self.multi,
                 family=getattr(self.env, "family", self.name), horizon_ticks=self.horizon)
        return d
