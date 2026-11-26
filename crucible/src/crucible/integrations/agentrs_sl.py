"""AGENTRS superlearner (``agentrs.sl``) `World` adapter for Crucible goals.

Families ``crucible/<goal family>`` (``revenue``, ``recruit``, ``service``, ``solvency``) start from the goal's own
initial conditions; ``crucible/<goal family>/deployed_service`` start after the documented scripted start-state prefix
(`crucible.integrations.deployed_service_prefix`). They are registered with ``agentrs.sl.worlds.register`` when this
module is imported.

* **Task -> world.** ``Task(seed, split)`` -> ``generate_goal(goal family, world family, base[split] + seed, level)``
  + ``instantiate`` (train / val / test use disjoint seed ranges). `reset_world` restarts the SAME task from its start
  snapshot (same hidden parameters: population, firms, prices), the unit of an sl trial.
* **Actions** (``ActionSpace(kind="sequential")``): ``host`` -> ``endpoint`` -> one label per argument slot -> END, or
  ``wait``. Slots and options come from `crucible.integrations.ActionTemplates`: identifiers the actor has seen under
  the argument's key (most recent first) or recent identifiers, enum values, booleans, a fixed integer grid, "omit"
  for optional arguments. Labels describe options by role, not by raw id (``listing:key:lst:0`` = the most recently
  seen ``listing``-keyed id of kind ``lst``), so they mean the same thing across worlds. Limitation: arguments that
  need free-form content (``code.commit.files``, contract clauses, prose) cannot be produced by this structured
  action space; tools that require them are not offered (an LLM-backed agent - `crucible.agents.llm`, CAM 2.0 - is
  needed for those). At most 24 options per slot (the sl pointer head's candidate limit).
* **Observation** (``Obs``): tokens (visible event types/kinds, last outcome, summary strings), numbers (numeric
  summary leaves with ids replaced by kinds, decisions left), entities (remembered identifiers with recency / source
  features), text (rendered observation). The actor's own view only.
* **Time.** One decision per sl step, then ``advance_time`` moves the world by ``dt`` seconds (``Consequence.dt`` is in
  hours). Episodes end at the goal horizon or after ``max_decisions`` (and, in the explicitly selected legacy and
  verified modes, on the verifier's outcome).
* **Reward modes** (Crucible 5.0; Clarification 2). ``reward_mode="observation_only"`` (THE DEFAULT): the native
  reward is always 0.0 (no action-quality penalty), the episode never ends because of the verifier's outcome,
  `goal()` carries the public objective text and horizon only (not the evaluator's metric name or target), the world
  hides evaluator tools and fields from the actor, and ``evaluate_goal`` (sl's evaluator side) reports only the
  verified success label. ``"verified_terminal_outcome"``: 1.0 once at the terminal outcome when the verifier confirms
  success, else 0.0. ``"legacy_shaped_reward"`` (explicit): the 4.0 adapter - a small native cost for schema-invalid
  calls, episodes end on verified success, `goal()` names the progress metric and target, ``evaluate_goal`` reads
  Crucible's `GoalVerifier` progress (sl's own RewardContract adds potential shaping on it).
"""
from __future__ import annotations

import copy
import pickle
from typing import Optional

from agentrs.sl import worlds as W
from agentrs.sl.world import ActionSpace, Consequence, GoalStatus, Obs, Task, World

from ..api import render as R
from . import DAY, OMIT, ActionTemplates, ActorView, EntityMemory, id_kind, make_goal_world, symlog

SPLIT_BASE = {"train": 10_000_000, "val": 1_000_000, "test": 0}
MAX_OPTIONS = 24
INVALID_COST = 0.01
SKIP = ("time", "principal", "events", "events_dropped")


def _flat_numbers(x, path: str, out: dict, depth: int = 0) -> None:
    if depth > 4:
        return
    if isinstance(x, dict):
        for k in sorted(x, key=str):
            kk = str(k)
            kn = f"<{id_kind(kk)}>" if id_kind(kk) != "word" else kk
            _flat_numbers(x[k], f"{path}.{kn}", out, depth + 1)
    elif isinstance(x, bool):
        out[path] = out.get(path, 0.0) + float(x)
    elif isinstance(x, (int, float)):
        out[path] = out.get(path, 0.0) + float(x)


class CrucibleSLWorld(World):
    episodic = True
    agent_controls_length = False      # episodes end by verified outcome / horizon / decision cap, not by an action

    def __init__(self, goal_family: str = "service", world_family: str = "tiny", level: int = 0, dt: int = 3600,
                 max_decisions: int = 48, prefix: Optional[str] = None, prefix_kw: Optional[dict] = None,
                 reward_mode: Optional[str] = None):
        from ..experience.modes import check_mode
        self.reward_mode = check_mode(reward_mode)
        self._paid = False
        self.goal_family, self.world_family, self.level = goal_family, world_family, level
        self.dt, self.max_decisions, self.prefix, self.prefix_kw = int(dt), int(max_decisions), prefix, prefix_kw or {}
        self.family = f"crucible/{goal_family}" + (f"/{prefix}" if prefix else "")
        self.world = None

    # ------------------------------------------------------------------ lifecycle
    def initialize_world(self, task: Task) -> None:
        self.task = task
        seed = SPLIT_BASE.get(task.split, 0) + int(task.seed)
        self.goal_contract, w, self.actor, pinfo = make_goal_world(self.goal_family, self.world_family, seed, self.level,
                                                          prefix=self.prefix, prefix_kw=self.prefix_kw)
        self.mem0 = EntityMemory()
        for a, r in pinfo.get("history", []):
            self.mem0.observe(None, a, r)
        self.base = w.snapshot_blob()
        self.world_id = w.world_id
        self.n_episode = 0
        self.reset_world()

    def reset_world(self) -> None:
        from ..core.kernel import World as CW
        from ..experience.modes import prepare_world
        from ..goals.contracts import GoalVerifier
        self.world = prepare_world(CW.from_blob(self.base, world_id=f"{self.world_id}-e{self.n_episode}"),
                                   self.reward_mode)
        self._paid = False
        self.n_episode += 1
        self.verifier = GoalVerifier(self.goal_contract, self.actor)
        self.verifier.begin(self.world)
        self.next_check = self.world.now + DAY
        self.view = ActorView(self.world, self.actor)
        self.mem = copy.deepcopy(self.mem0)
        self.templates = ActionTemplates(self.mem, self.actor)
        self.n_dec = 0
        self.last = None
        self.last_invalid = False
        self._obs = self.view.observe()
        self.mem.observe(self._obs)
        self._space = None

    def agents(self) -> list:
        return ["0"]

    # ------------------------------------------------------------------ agent view
    def goal(self, agent: str = "0") -> Obs:
        g = self.goal_contract
        if self.reward_mode == "legacy_shaped_reward":
            return Obs(tokens=[f"goal:{g.family}", f"level:{self.level}"] + [f"metric:{g.progress['metric']}"],
                       numbers={"target": symlog(float(g.progress.get("target", 0))),
                                "horizon_days": g.horizon_s / DAY}, text=g.description)
        # the public objective only: its text and horizon (the evaluator's metric and target stay evaluator-side)
        return Obs(tokens=[f"goal:{g.family}", f"level:{self.level}"], numbers={"horizon_days": g.horizon_s / DAY},
                   text=g.description)

    def observe(self, agent: str = "0") -> Obs:
        o = self._obs
        toks = []
        for e in (o.get("events") or [])[-30:]:
            toks.append(f"ev:{e['type']}")
            pl = e.get("payload") or {}
            if isinstance(pl, dict) and isinstance(pl.get("kind"), str):
                toks.append(f"ev:{e['type']}:{pl['kind']}")
        if self.last is not None:
            toks.append(f"res:{self.last.get('tool')}:" + ("ok" if self.last.get("ok") else
                                                           (self.last.get("error") or {}).get("code", "error")))
        nums: dict = {}
        for k in sorted(o):
            if k not in SKIP:
                _flat_numbers(o[k], k, nums)
                for kk, vv in (o[k].items() if isinstance(o[k], dict) else []):
                    if isinstance(vv, dict):
                        for k3, v3 in vv.items():
                            if isinstance(v3, str) and len(v3) < 24:
                                toks.append(f"{k}.{kk}={v3}")
        numbers = {k: symlog(v) for k, v in list(nums.items())[:40]}
        numbers["decisions_left"] = (self.max_decisions - self.n_dec) / self.max_decisions
        numbers["day"] = self.world.now / DAY
        ents = []
        for v, (tick, key) in list(reversed(self.mem.recent.items()))[:40]:
            ents.append((id_kind(v), [min(1.0, (self.mem.tick - tick) / 10.0), float(v == self.actor),
                                      float(key in ("listing", "host", "account", "org", "repo", "instance")), 1.0]))
        text = R.render_observation({"time": o.get("time", 0), "principal": self.actor, "last_result": self.last,
                                     "summary": {k: v for k, v in o.items() if k not in SKIP},
                                     "events": o.get("events") or [], "dropped": o.get("events_dropped", 0)},
                                    max_events=12, show_tools=False)
        return Obs(tokens=toks[:64], numbers=numbers, entities=ents, text=text)

    def _build_space(self) -> dict:
        tools = self.view.tools()
        tree: dict = {}
        for t in tools:
            slots = self.templates.slots(t)
            if slots is None:
                continue
            host, _, ep = t["name"].partition(".")
            tree.setdefault(host, {})[ep] = (t, [s for s in slots])
        return tree

    @staticmethod
    def option_label(slot, opt) -> str:
        if opt.source in ("key", "recent", "self") and isinstance(opt.value, str):
            return f"{slot.param}:{opt.source}:{id_kind(opt.value)}:{opt.rank}"
        if opt.source == "omit":
            return f"{slot.param}:omit"
        return f"{slot.param}:{opt.source}:{opt.label}"[:60]

    def available_actions(self, agent: str = "0") -> ActionSpace:
        tree = self._space = self._build_space()
        hosts = sorted(tree)[: MAX_OPTIONS - 1]

        def allowed(prefix: list) -> list:
            if not prefix:
                return ["wait"] + hosts
            if prefix[0] == "wait":
                return ["END"]
            eps = tree.get(prefix[0], {})
            if len(prefix) == 1:
                return sorted(eps)[:MAX_OPTIONS]
            t, slots = eps[prefix[1]]
            j = len(prefix) - 2
            if j >= len(slots):
                return ["END"]
            labs = [self.option_label(slots[j], o) for o in slots[j].options[:MAX_OPTIONS]]
            return list(dict.fromkeys(labs))

        return ActionSpace(kind="sequential", allowed=allowed, max_len=2 + self.templates.max_slots + 1)

    def decode(self, action: list) -> Optional[dict]:
        toks = [a for a in action if a != "END"]
        if not toks or toks[0] == "wait":
            return None
        tree = self._space if self._space is not None else self._build_space()
        t, slots = tree[toks[0]][toks[1]]
        choice = []
        for j, s in enumerate(slots):
            labs = [self.option_label(s, o) for o in s.options[:MAX_OPTIONS]]
            lab = toks[2 + j] if 2 + j < len(toks) else None
            choice.append(labs.index(lab) if lab in labs else (0 if s.required else
                                                                next(i for i, o in enumerate(s.options) if o.value == OMIT)))
        return ActionTemplates.build(t, slots, choice)

    # ------------------------------------------------------------------ dynamics
    def execute(self, agent, action) -> None:
        call = self.decode(list(action))
        self.n_dec += 1
        self.last_invalid = False
        if call is None:
            self.last = None
            return
        r = self.world.execute(self.actor, call)
        self.last = r.to_dict()
        self.last_invalid = (not r.ok) and (r.error or {}).get("code") in ("invalid_action", "unknown_tool")
        self.mem.observe(None, call, self.last)

    def advance_time(self) -> None:
        w = self.world
        w.advance_time(w.now + self.dt)
        if w.now < self.next_check:
            self.verifier.sample(w)        # sustained point-in-time predicates are watched between daily checks
        while w.now >= self.next_check:
            self.verifier.daily_check(w)
            self.next_check += DAY
        self._obs = self.view.observe()
        self.mem.observe(self._obs)
        self._space = None

    def _ended(self) -> bool:
        st = self.verifier.status(self.world)
        if self.reward_mode == "observation_only":     # time and decision budget only, never the verifier's outcome
            return bool(self.n_dec >= self.max_decisions or st["elapsed_s"] >= self.goal_contract.horizon_s)
        return bool(st["success"] or st["failed"] or self.n_dec >= self.max_decisions)

    def observe_consequences(self, agent: str = "0") -> Consequence:
        ev = []
        if self.last is not None:
            ev.append("ok" if self.last.get("ok") else (self.last.get("error") or {}).get("code", "error"))
        if self.last_invalid:
            ev.append("invalid")
        done = self._ended()
        if self.reward_mode == "legacy_shaped_reward":
            reward = -INVALID_COST if self.last_invalid else 0.0
        elif self.reward_mode == "verified_terminal_outcome" and done and not self._paid and \
                self.verifier.status(self.world)["success"]:
            self._paid, reward = True, 1.0
        else:
            reward = 0.0
        return Consequence(reward=reward, dt=self.dt / 3600.0, done=done, events=ev)

    def evaluate_goal(self, agent: str = "0") -> GoalStatus:
        st = self.verifier.status(self.world)
        left = max(0.0, 1.0 - max(self.n_dec / self.max_decisions, st["elapsed_s"] / self.goal_contract.horizon_s))
        if self.reward_mode != "legacy_shaped_reward":      # the verified outcome label only
            ok = float(bool(st["success"]))
            return GoalStatus(success=bool(st["success"]), progress=ok, violated=None, score=ok, time_left=left)
        return GoalStatus(success=bool(st["success"]), progress=float(st["progress"]), violated=None,
                          score=float(st["progress"]), time_left=left)

    # ------------------------------------------------------------------ state
    def save_world_state(self):
        return {"blob": self.world.snapshot_blob(), "wid": self.world.world_id, "verifier": pickle.dumps(self.verifier),
                "next_check": self.next_check, "mem": self.mem.state_dict(), "n_dec": self.n_dec,
                "last": copy.deepcopy(self.last), "last_invalid": self.last_invalid, "cursor": self.view.cursor,
                "obs": copy.deepcopy(self._obs)}

    def restore_world_state(self, s) -> None:
        from ..core.kernel import World as CW
        from ..experience.modes import prepare_world
        self.world = prepare_world(CW.from_blob(s["blob"], world_id=s["wid"]), self.reward_mode)
        self.view.rebind(self.world)
        self.view.cursor = s["cursor"]
        self.verifier = pickle.loads(s["verifier"])
        self.next_check, self.n_dec, self.last = s["next_check"], s["n_dec"], copy.deepcopy(s["last"])
        self.last_invalid, self._obs = s["last_invalid"], copy.deepcopy(s["obs"])
        self.mem = EntityMemory()
        self.mem.load_state_dict(s["mem"])
        self.templates = ActionTemplates(self.mem, self.actor)
        self._space = None

    def close(self) -> None:
        if self.world is not None:
            self.world.close()


def register_families(world_family: str = "tiny", level: int = 0, max_decisions: int = 48, dt: int = 3600,
                      reward_mode: Optional[str] = None, prefix_name: str = "crucible") -> list:
    """Register the goal families with ``agentrs.sl.worlds``: ``crucible/<goal family>`` in the default
    (observation-only) mode, and ``crucible-legacy/<goal family>`` in the explicitly selected 4.0 mode for historical
    baselines (`FAMILY_NAMES`, `LEGACY_FAMILY_NAMES`)."""
    from ..goals.contracts import GOAL_FAMILIES, HELDOUT_GOAL_FAMILIES
    names = []
    for gf in sorted(set(GOAL_FAMILIES) - set(HELDOUT_GOAL_FAMILIES)):   # held-out goal families are never registered
        name = f"{prefix_name}/{gf}"
        W.register(name, lambda gf=gf: CrucibleSLWorld(gf, world_family, level, dt, max_decisions,
                                                       reward_mode=reward_mode),
                   meta={"source": "Crucible (crucible/)", "agents": 1, "goal_family": gf, "world_family": world_family,
                         "reward_mode": reward_mode or "observation_only"})
        names.append(name)
    for gf in ("revenue", "service"):
        name = f"{prefix_name}/{gf}/deployed_service"
        W.register(name, lambda gf=gf: CrucibleSLWorld(gf, world_family, level, dt, max_decisions, "deployed_service",
                                                       reward_mode=reward_mode),
                   meta={"source": "Crucible (crucible/)", "agents": 1, "goal_family": gf, "prefix": "deployed_service",
                         "reward_mode": reward_mode or "observation_only"})
        names.append(name)
    return names


FAMILY_NAMES = register_families()
LEGACY_FAMILY_NAMES = register_families(reward_mode="legacy_shaped_reward", prefix_name="crucible-legacy")
