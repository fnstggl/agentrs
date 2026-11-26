"""`RapidEnv`: one Tier 1 world behind the structured action interface (INTERFACE_CONTRACTS section 8).

    env = make("net_routing", seed=3, horizon=80, observation_noise=0.2)
    obs = env.reset()                               # a section-11 Observation for agent a0
    r = env.step({"kind": "levers", "values": {"ka31": 0.7}})
    r["observation"], r["result"], r["events"]
    env.outcome()                                   # evaluator side only

`step` takes one action (a dict with `kind`) for the first agent, or a mapping agent -> action in worlds with
several participants. Every call is one tick: the actions are carried out in order (a `sequence` expands into its
steps), joint plans that every named party submitted identically are carried out, then the world advances. An
invalid action is refused with an error and the tick still passes (the world does not wait).

Randomness: the world's dynamics draw from `tick_rng(world_seed, env_id, t)` - the same derivation a kernel
facility uses - and actions (queries, experiments) from a separate stream, so the world noise does not depend on
what the participant asked. Messages to `journal` file claims or predictions in the world; they are read only by
the evaluator. `program` actions run an agent-written controller in the sandbox (`crucible.rapid.program`).

Nothing an Observation contains is evaluator data. `outcome()` is the evaluator's measure and is never part of an
observation.
"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from ..core.canonical import digest
from ..experience.modes import OBSERVATION_ONLY, check_mode
from ..mechanisms.base import tick_rng
from ..mechanisms.sim import TickRng
from . import observation as O
from .actions import ActionSpace
from .engines import build
from .engines.base import ActionError, RapidEngine


class RapidEnv(Protocol):
    family: str
    seed: int
    spec: dict

    def reset(self) -> dict: ...
    def step(self, actions: dict[str, Any]) -> dict: ...
    def outcome(self) -> dict: ...


def world_seed_of(family: str, seed: int) -> int:
    return int(digest(["rapid-world", family, int(seed)])[:8], 16)


class Env:
    """The RapidEnv implementation shared by every engine."""

    def __init__(self, family: str, seed: int, config: dict, difficulty: dict | None = None,
                 world_seed: int | None = None, env_id: str = "r1", meta: dict | None = None,
                 reward_mode: str | None = None, objective: Any = "assigned"):
        self.family, self.seed = family, int(seed)
        self._config = config
        self._engine: RapidEngine = build(config)
        self.world_seed = world_seed_of(family, seed) if world_seed is None else int(world_seed)
        self.env_id = env_id
        self.difficulty = dict(difficulty or {})
        # Clarification 2: observation-only is the default; no mode adds a reward to anything this world returns
        # (legacy and verified modes are interpreted by the experience adapters, never here)
        self.reward_mode = OBSERVATION_ONLY if reward_mode is None else check_mode(reward_mode)
        self._meta = dict(meta or {}, family=family, seed=self.seed, engine=config.get("engine"),
                          difficulty=self.difficulty)                       # evaluator side, never in `spec`
        if objective is None or objective == "none":
            self.objective_mode, self.goal = "none", None
        elif objective == "assigned":
            self.objective_mode, self.goal = "assigned", None
        elif isinstance(objective, dict) and objective.get("text"):
            self.objective_mode, self.goal = "generated", dict(objective)
        else:
            raise ValueError("objective is 'assigned', None (objective-free) or a generated goal dict with 'text'")
        eng = self._engine
        # the participant-visible description: what the world is and what can be done in it. It names no family,
        # seed, engine, category, split or difficulty setting (a family name describes its mechanism and the split
        # marks evaluation worlds)
        self.spec = {"agents": eng.agents, "horizon_ticks": eng.horizon,
                     "brief": {a: eng.brief(a) for a in eng.agents},
                     "objective": {a: self.objective_text(a) for a in eng.agents},
                     "objective_mode": self.objective_mode, "experience_mode": self.reward_mode,
                     "action_kinds": {a: sorted(eng.action_space(a).kinds) for a in eng.agents}}
        self._rng = TickRng(self.world_seed, env_id)
        self._arng = TickRng(self.world_seed, env_id + "/act")
        self.reset()

    def objective_text(self, agent: str) -> str | None:
        """The objective a participant is told: the family's own (assigned), a generated goal's text, or nothing
        (objective-free operation)."""
        if self.objective_mode == "none":
            return None
        if self.objective_mode == "generated":
            return str(self.goal["text"])
        return self._engine.objective_text(agent)

    # ------------------------------------------------------------------ protocol
    @property
    def horizon(self) -> int:
        return self._engine.horizon

    def agents(self) -> list[str]:
        return list(self._engine.agents)

    @property
    def done(self) -> bool:
        return self.t >= self._engine.horizon

    def reset(self) -> dict:
        eng = self._engine
        self.t = 0
        self._state = eng.init_state(tick_rng(self.world_seed, self.env_id, "init"))
        start = int(self._config.get("endowment", 0))
        self._ledger = {"revenue": {a: 0.0 for a in eng.agents}, "cost": {a: 0.0 for a in eng.agents},
                        "cash": {a: float(start) for a in eng.agents}, "journal": [], "messages": [],
                        "actions": {a: 0 for a in eng.agents}, "errors": {a: 0 for a in eng.agents}}
        self._events: list[dict] = []
        self._cursor = {a: 0 for a in eng.agents}
        self._inbox: dict[str, list] = {a: [] for a in eng.agents}
        self._spaces = {a: eng.action_space(a) for a in eng.agents}
        self._last_numbers: dict[str, list] = {a: [] for a in eng.agents}
        self._obs: dict[str, dict] = {}
        hist = eng.history(self._state)
        for a in eng.agents:
            self._obs[a] = self._make_obs(a, {"numbers": {}, "arrays": hist.get("arrays", {}),
                                              "text": [eng.brief(a)] + hist.get("text", [])},
                                          None, tools=True, rev=0.0, cost=0.0)
        return self._obs[eng.agents[0]]

    def observe(self, agent: str | None = None) -> dict:
        return self._obs[agent or self._engine.agents[0]]

    def tools(self, agent: str | None = None) -> list[dict]:
        return self._spaces[agent or self._engine.agents[0]].tools()

    def action_space(self, agent: str | None = None) -> ActionSpace:
        return self._spaces[agent or self._engine.agents[0]]

    def observe_since(self, agent: str, cursor: int) -> tuple[list[dict], int]:
        evs = [e for e in self._events[cursor:] if self._visible(e, agent)]
        return evs, len(self._events)

    def step(self, actions: dict[str, Any] | None) -> dict:
        eng = self._engine
        a0 = eng.agents[0]
        if actions is None:
            per = {}
        elif isinstance(actions, dict) and "kind" in actions:
            per = {a0: actions}
        elif isinstance(actions, dict) and all(k in eng.agents for k in actions):
            per = dict(actions)
        else:
            per = {a0: actions}
        if self.done:
            res = {"ok": False, "error": "the episode is over (horizon reached)"}
            return {"observation": self._obs[a0], "result": res, "events": []}
        prog = [a for a, act in per.items() if isinstance(act, dict) and act.get("kind") == "program"]
        if prog and not getattr(eng, "handles_programs", False):     # engines that run programs themselves get them
            from .program import run_controller
            return run_controller(self, prog[0], per[prog[0]])
        return self._tick(per)

    def _tick(self, per: dict) -> dict:
        eng = self._engine
        a0 = eng.agents[0]
        arng = self._arng(self.t)
        results: dict[str, dict] = {}
        plans: dict[str, dict] = {}
        for a in eng.agents:
            act = per.get(a)
            if act is None:
                results[a] = {"ok": True, "result": {"waited": True}}
                continue
            self._ledger["actions"][a] += 1
            space = self._spaces[a]
            errs = space.validate(act)
            if errs:
                self._ledger["errors"][a] += 1
                results[a] = {"ok": False, "error": "; ".join(errs[:5])}
                continue
            outs, err = [], None
            for st in space.expand(act):
                try:
                    if st["kind"] == "coordinate":
                        plans[a] = st
                        outs.append({"proposed": st["plan"], "with": st["with"]})
                    else:
                        outs.append(self._apply(a, st, arng))
                except ActionError as e:
                    err = str(e)
                    break
            if err is not None:
                self._ledger["errors"][a] += 1
                results[a] = {"ok": False, "error": err, "done_steps": outs}
            else:
                results[a] = {"ok": True, "result": outs[0] if len(outs) == 1 else outs}
        if plans:
            self._coordinate(plans, results, arng)
        before = eng.copy_state(self._state) if self._config.get("_audit") else None
        out = eng.advance(self._state, self._rng(self.t), self.t)
        for a in eng.agents:
            self._ledger["revenue"][a] += out["revenue"].get(a, 0.0)
            self._ledger["cost"][a] += out["cost"].get(a, 0.0)
            self._ledger["cash"][a] += out["revenue"].get(a, 0.0) - out["cost"].get(a, 0.0)
        if before is not None:
            self._audit = getattr(self, "_audit", [])
            self._audit.extend(eng.conservation(before, self._state, out))
        self._events.append({"t": self.t, "type": "tick", "vis": "public"})
        for e in out.get("events", []):
            self._events.append(dict(e, t=self.t))
        self.t += 1
        if getattr(eng, "dynamic_space", False):
            self._spaces = {a: eng.action_space(a, self._state) for a in eng.agents}
            changed = True
        else:
            changed = False
        for a in eng.agents:
            ob = out["obs"].get(a, {})
            txt = list(ob.get("text", [])) + [f"message from {m['from']}: {m.get('text', '')}" for m in self._inbox[a]]
            ob = dict(ob, text=txt)
            if self._inbox[a]:
                ob["entities"] = list(ob.get("entities", [])) + [{"kind": "message", "from": m["from"],
                                                                  "content": m.get("content")} for m in self._inbox[a]]
            self._inbox[a] = []
            self._obs[a] = self._make_obs(a, ob, results.get(a), tools=changed,
                                          rev=out["revenue"].get(a, 0.0), cost=out["cost"].get(a, 0.0))
        r0 = results.get(a0, {"ok": True})
        res = {"ok": bool(r0["ok"])}
        if not r0["ok"]:
            res["error"] = r0["error"]
        res["results"] = r0.get("result")
        ret = {"observation": self._obs[a0], "result": res, "events": self._obs[a0]["events"]}
        if len(eng.agents) > 1:
            ret["observations"] = dict(self._obs)
            ret["results"] = results
        return ret

    def _apply(self, agent: str, st: dict, arng: np.random.Generator) -> dict:
        eng = self._engine
        k = st["kind"]
        if k == "message":
            msg = {"from": agent, "to": st["to"], "t": self.t, "text": st.get("text", ""),
                   "content": st.get("content")}
            if st["to"] == "journal":
                self._ledger["journal"].append(msg)
                return {"filed": True, "entries": len(self._ledger["journal"])}
            if st["to"] not in eng.agents:
                raise ActionError(f"no participant {st['to']}")
            self._inbox[st["to"]].append(msg)
            self._ledger["messages"].append(msg)
            self._events.append({"type": "message", "from": agent, "to": st["to"], "vis": [agent, st["to"]]})
            return {"delivered": True}
        if k == "query" and st["target"] in ("tools", "history", "spec"):
            if st["target"] == "tools":
                return {"tools": self._spaces[agent].tools()}
            if st["target"] == "history":
                n = int((st.get("args") or {}).get("n", 10))
                return {"history": self._last_numbers[agent][-max(1, min(n, 200)):]}
            return {"spec": self.spec}
        return eng.apply(self._state, agent, st, arng, self.t)

    def _coordinate(self, plans: dict, results: dict, arng) -> None:
        eng = self._engine
        done = set()
        for a, st in plans.items():
            if a in done:
                continue
            parties = sorted(set(st["with"]) | {a})
            if all(p in plans and plans[p]["plan"] == st["plan"] and set(plans[p]["with"]) | {p} == set(parties)
                   for p in parties):
                try:
                    r = eng.coordinate(self._state, st["plan"], parties, arng, self.t)
                    for p in parties:
                        results[p] = {"ok": True, "result": {"agreed": True, "plan": st["plan"], **r}}
                except ActionError as e:
                    for p in parties:
                        results[p] = {"ok": False, "error": str(e)}
                done |= set(parties)
            else:
                results[a] = {"ok": True, "result": {"agreed": False, "plan": st["plan"],
                                                     "note": "not every named party submitted this plan this tick"}}

    # ------------------------------------------------------------------ observations
    def _visible(self, e: dict, agent: str) -> bool:
        v = e.get("vis", "public")
        return v == "public" or (isinstance(v, list) and agent in v)

    def _make_obs(self, agent: str, ob: dict, last: dict | None, tools: bool, rev: float, cost: float) -> dict:
        evs, cur = self.observe_since(agent, self._cursor[agent])
        self._cursor[agent] = cur
        evs = [{k: v for k, v in e.items() if k != "vis"} for e in evs]
        res = {"cash_cents": float(self._ledger["cash"][agent]), "revenue_cents_last_tick": float(rev),
               "cost_cents_last_tick": float(cost)} if self._engine.config.get("money") or \
            self._engine.config.get("endowment") else {}
        res.update(ob.get("resources", {}))
        nums = dict(ob.get("numbers", {}))
        self._last_numbers[agent].append({"t": self.t, **{k: v for k, v in nums.items()
                                                           if isinstance(v, float) and v == v}})
        return O.make(t=self.t, agent=agent, numbers=nums, arrays=ob.get("arrays"), text=ob.get("text"),
                      tools=self._spaces[agent].tools() if tools else [], events=evs, entities=ob.get("entities"),
                      resources=res, analytics=ob.get("analytics"), last_result=last, missing=ob.get("missing"),
                      cursor=cur, objective=self.objective_text(agent))

    # ------------------------------------------------------------------ evaluator side
    def outcome(self) -> dict:
        """Evaluator side: the objective quantity and what it was computed from. Never part of an observation."""
        out = self._engine.outcome(self._state, self._ledger)
        obj = self._config.get("objective", {})
        out.update({"family": self.family, "seed": self.seed, "t": self.t, "horizon": self.horizon,
                    "done": self.done, "objective": obj.get("type"), "objective_mode": self.objective_mode,
                    "actions": dict(self._ledger["actions"]), "invalid_actions": dict(self._ledger["errors"])})
        thr = obj.get("target")
        out["success"] = None if thr is None else bool(out["value"] >= thr)
        return out
