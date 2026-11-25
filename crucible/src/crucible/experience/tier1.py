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

Observations carry the exact report of the last tick (readings, revenue, cost, net, shortfall, score as the
facility reports them to its operator), the settings in force, the balance, the brief, and `rig.report` events
whose `cause` points at the `rig.set` that chose the settings they ran at. Hidden: how levers act; stated in
`missing`. The task's private reference and the episode's normalised score are evaluator data (`verify()`).
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
from .modes import LEGACY, OBSERVATION_ONLY, check_mode
from .observation import Observation, json_copy

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

    def __init__(self, task, *, reward_mode: str = LEGACY, handles: list[str] | None = None, tick_s: int | None = None,
                 max_events: int = 1000, name: str | None = None):
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
        return {k: json_copy(rep[k]) for k in ("t", "time", "status", "readings", "revenue_cents", "cost_cents",
                                                "net_cents", "shortfall_cents", "score", "settings")}

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
            self._emit("rig.report", {k: v for k, v in rep.items() if k != "time"}, (a,), cause=None if c is None
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
            seat_view.update(t=r["t"], readings=dict(r["readings"]), net_cents=r["net_cents"], score=r["score"],
                             revenue_cents=r["revenue_cents"], cost_cents=r["cost_cents"],
                             shortfall_cents=r["shortfall_cents"])
        seat_view["next_settings"] = {lv.name: float(self._desired[s][lv.name]) for lv in self._levers[s]}
        seat_view.update(tick_s=self.tick_s, next_tick_at=self.next_tick_at(), horizon_ticks=int(self.task.horizon))
        numbers, arrays = flatten({"rig": {"seats": {RIG_ID: seat_view}},
                                   "bank": {"balances": {self._acct[agent]: self._balance(agent)}}})
        numbers["world.time_s"] = int(self._now)
        numbers["world.day"] = int(self._now // 86400)
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
            resources={"actions": self._actions[agent], "action_time_s": self._time_used[agent],
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
        d.update(task=self.task.id, world_id=f"t1-{self.task.family}-{self.task.seed}", spec_hash=_digest(
            json.loads(json.dumps(self.task.public(), sort_keys=True))), agents=list(self._agents),
            tick_s=self.tick_s, seats=self.n_seats)
        return d


class _Refused(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(msg)
        self.code = code


# ====================================================================== v5 Tier 1 (workstream H's RapidEnv)
class RapidExperienceEnv(BaseExperienceEnv):
    """`crucible.rapid.make(family, seed, **difficulty)` behind the ExperienceEnv interface (INTERFACE_CONTRACTS
    section 8: ``reset() -> Observation``, ``step(actions) -> {observation, result, events}``, ``outcome()``).

    `RapidEnv.step` takes ``{agent: action}``; the adapter calls it once per `execute`, so every executed action is
    one tick of the world. An action is either a rapid action (``{"kind": ..., ...}``) or the interface's tool form
    ``{"tool": <kind>, "args": {...}}`` (the tools are the action kinds, whose schemas name the ``kind`` field); the
    adapter passes the rapid action on. Waiting: ``advance(seconds > 0)`` lets one tick pass with no action when no
    action was executed since the last tick, so a driver that executes or waits once per decision moves the world
    one tick per decision either way. The legacy reward of a rapid family is the ``reward`` its `step` result
    carries, if any (Tier 1 v5 families need not have one); the verified outcome is `outcome()["success"]`;
    `outcome()` itself stays on the evaluator side (`verify`). Observations always list the current tools (the
    rapid world sends them only when they change). `rapid_env=` installs an already built `RapidEnv` (a world drawn
    by `crucible.rapid.generator` outside the family registry)."""
    tier = 1
    kind = "rapid"

    def __init__(self, family: str, seed: int = 0, *, reward_mode: str = LEGACY, rapid_env=None, **difficulty):
        from .. import rapid
        if not hasattr(rapid, "make"):
            raise ImportError("crucible.rapid.make is not available (workstream H not merged)")
        super().__init__(check_mode(reward_mode), f"rapid:{family}", seed)
        self.env = rapid_env if rapid_env is not None else rapid.make(family, int(seed), **difficulty)
        first = self.env.reset()
        ags = getattr(self.env, "agents", None)
        ags = ags() if callable(ags) else ags
        self._agents = [str(a) for a in (ags or [first.get("agent", "agent")])]
        self._events: list[dict] = []
        self._begin(first)

    def _begin(self, first: dict) -> None:
        self._obs = {a: (dict(first, agent=a) if a != first.get("agent") else first) for a in self._agents}
        if hasattr(self.env, "observe"):
            for a in self._agents:
                self._obs[a] = self.env.observe(a)
        self._pending = {a: 0.0 for a in self._agents}
        self._paid: set[str] = set()
        self._last: dict[str, dict | None] = {a: None for a in self._agents}
        self._acted = False                  # an action was executed since the last tick
        for a in self._agents:
            self._obs[a] = self._contract_events(dict(self._obs[a]))

    def agents(self) -> list[str]:
        return list(self._agents)

    @property
    def done(self) -> bool:
        return bool(getattr(self.env, "done", False))

    @property
    def horizon(self) -> int:
        return int(getattr(self.env, "horizon", 0))

    def observe(self, agent: str) -> Observation:
        obs = json_copy(self._obs[agent])
        obs["tools"] = self.tools(agent)
        obs["last_result"] = json_copy(self._last[agent])
        return obs

    def _contract_events(self, obs: dict) -> dict:
        """A rapid observation with its events in the interface's event form (seq, t, type, actor, payload,
        cause), numbered in this adapter's log."""
        evs = []
        for e in obs.get("events") or []:
            if isinstance(e, dict) and {"seq", "t", "type", "actor", "payload", "cause"} <= set(e):
                evs.append(e)
                continue
            e = dict(e) if isinstance(e, dict) else {"type": "event", "value": e}
            ev = {"seq": len(self._events) + 1, "t": int(e.pop("t", obs.get("t", 0)) or 0),
                  "type": str(e.pop("type", "event")), "actor": str(e.pop("from", "world")), "payload": e,
                  "cause": None}
            self._events.append(ev)
            evs.append(ev)
        obs["events"] = evs
        return obs

    def tools(self, agent: str) -> list[dict]:
        """The world's action kinds as interface tools ``rapid.<kind>`` (the ``<host>.<endpoint>`` naming of the
        Observation contract); each tool's input schema is the kind's action schema."""
        fn = getattr(self.env, "tools", None)
        tools = fn(agent) if callable(fn) else (self._obs[agent].get("tools") or [])
        out = []
        for t in json_copy(list(tools)):
            name = str(t.get("name", ""))
            t["name"] = name if "." in name else f"rapid.{name}"
            ann = dict(t.get("annotations") or {})
            ann.setdefault("host", "rapid")
            ann.setdefault("readOnlyHint", name == "query")
            t["annotations"] = ann
            out.append(t)
        return annotate_tools(out)

    @staticmethod
    def _rapid_action(action: Any) -> Any:
        """A rapid action from either form: ``{"kind": ...}`` as is, ``{"tool": "rapid.<kind>" | "<kind>", "args"}``
        as its arguments with the kind filled in."""
        if not isinstance(action, dict) or "kind" in action or "tool" not in action:
            return action
        kind = str(action["tool"])
        kind = kind[len("rapid."):] if kind.startswith("rapid.") else kind
        args = action.get("args")
        args = dict(args) if isinstance(args, dict) else {}
        args.setdefault("kind", kind)
        return args

    def _absorb(self, out: dict, agent: str | None) -> None:
        got = {}
        obs_all = out.get("observations")
        if isinstance(obs_all, dict):
            got.update({a: o for a, o in obs_all.items() if a in self._obs and isinstance(o, dict)})
        obs = out.get("observation")
        if isinstance(obs, dict):
            got[str(obs.get("agent") or agent or self._agents[0])] = obs
        for a, o in got.items():
            if a in self._obs:
                self._obs[a] = self._contract_events(dict(o))

    def execute(self, agent: str, action: dict) -> dict:
        out = self.env.step({agent: self._rapid_action(action)})
        self._absorb(out, agent)
        obs = out.get("observation")
        self._acted = True
        res = dict(out.get("result") or {})
        if self.reward_mode == LEGACY and isinstance(res.get("reward"), (int, float)):
            self._pending[agent] += float(res["reward"])
        res.pop("reward", None)
        self.steps += 1
        tool = action.get("tool", action.get("kind", "")) if isinstance(action, dict) else ""
        r = {"ok": bool(res.get("ok", False)), "t": int((obs or {}).get("t", 0)), "tool": str(tool),
             "cost_s": int(res.get("cost_s", 0))}
        if r["ok"]:
            r["result"] = res.get("result", res.get("results"))
        else:
            r["error"] = res.get("error") if isinstance(res.get("error"), dict) else {"code": "error",
                                                                                      "message": str(res.get("error"))}
        if self.reward_mode != LEGACY:
            oc = self.env.outcome() if self.reward_mode != OBSERVATION_ONLY else {}
            if oc.get("success") and agent not in self._paid:
                self._paid.add(agent)
                self._pending[agent] += 1.0
        self._last[agent] = json_copy(r)
        return r

    def observe_since(self, agent: str, cursor: int) -> tuple[list[dict], int]:
        evs = [e for e in self._events if int(e.get("seq", 0)) > int(cursor)]
        return json_copy(evs), max([int(e.get("seq", 0)) for e in self._events] + [int(cursor)])

    def advance(self, seconds: int) -> None:
        """One tick with no action when time passes and no action was executed since the last tick (a wait)."""
        if int(seconds) > 0 and not self._acted and not self.done:
            self._absorb(self.env.step(None), None)
        self._acted = False

    def reset(self) -> None:
        """A new episode of the same world (same seed, same noise)."""
        self._begin(self.env.reset())

    def reward(self, agent: str) -> float:
        r, self._pending[agent] = self._pending[agent], 0.0
        return 0.0 if self.reward_mode == OBSERVATION_ONLY else float(r)

    def verify(self) -> dict:
        self.verifications += 1
        return {"verifier": "crucible.rapid RapidEnv.outcome", "outcome": self.env.outcome()}

    def digest(self) -> str:
        return _digest([self.name, self.seed, self.steps, [self._obs[a].get("cursor") for a in self._agents]])
