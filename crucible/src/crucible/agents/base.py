"""Policy interface and a multi-agent runner with trajectory recording.

A policy sees only what its actor sees: the observation (World.observe + visible events since its last turn) and
the tool list (World.available_tools). It returns one action `{"tool", "args"}` or None (wait). Policies are fully
separate from world mechanics and may be anything: scripted, random, an LLM, a neural network, a human.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from ..core.kernel import World


class Policy:
    name = "policy"
    kind = "abstract"          # scripted | random | heuristic | llm | rl | human
    version = "1"

    def reset(self, actor: str, rng: np.random.Generator) -> None:
        self.actor = actor
        self.rng = rng

    def act(self, obs: dict, tools: list[dict]) -> dict | None:
        raise NotImplementedError

    def observe_result(self, action: dict, result: dict) -> None:
        """Called with the ActionResult dict of the policy's last action."""

    def usage(self) -> dict:
        """Resources consumed by the policy itself (e.g. LLM tokens)."""
        return {}


@dataclass
class Step:
    t: int
    actor: str
    policy: str
    obs: dict
    n_tools: int
    action: dict | None
    result: dict | None
    events: list[int]
    reward: float = 0.0
    wall_ms: float = 0.0


@dataclass
class Trajectory:
    actor: str
    policy: str
    world_id: str
    world_spec_hash: str
    steps: list[Step] = field(default_factory=list)

    def to_records(self) -> list[dict]:
        return [{"t": s.t, "actor": s.actor, "policy": s.policy, "obs": s.obs, "n_tools": s.n_tools,
                 "action": s.action, "result": s.result, "events": s.events, "reward": s.reward,
                 "wall_ms": s.wall_ms, "world": self.world_id, "spec": self.world_spec_hash} for s in self.steps]


class Runner:
    """Drives several (actor, policy) pairs in one world: every agent acts once per round (seeded order), then the
    world clock advances by `dt` seconds (running all due processes: demand, billing, NPC decisions...)."""

    def __init__(self, world: World, agents: Iterable[tuple[str, Policy]], dt: int = 600, seed: int = 0,
                 objectives: dict | None = None, record: bool = True, max_events: int = 30):
        self.world = world
        self.agents = list(agents)
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.cursor = {a: world.seq for a, _ in self.agents}
        self.objectives = objectives or {}
        self.record = record
        self.max_events = max_events
        self.trajs = {a: Trajectory(a, p.name, world.world_id, world.spec.hash()) for a, p in self.agents}
        self._tools_cache: dict[str, tuple[int, list]] = {}
        for i, (a, p) in enumerate(self.agents):
            p.reset(a, np.random.default_rng([seed, i]))
            obj = self.objectives.get(a)
            if obj is not None:
                obj.begin(world, a)

    def tools(self, actor: str) -> list[dict]:
        n = len(self.world.state.find_ids("knowledge", "principal", actor))
        c = self._tools_cache.get(actor)
        if c is None or c[0] != n:
            c = (n, self.world.available_tools(actor))
            self._tools_cache[actor] = c
        return c[1]

    def observe(self, actor: str) -> dict:
        obs = self.world.observe(actor)
        evs, dropped = self.world.visible_events(actor, self.cursor[actor], limit=self.max_events)
        self.cursor[actor] = self.world.seq
        obs["events"] = [{"seq": e.seq, "t": e.t, "type": e.type, "actor": e.actor, "payload": e.payload}
                         for e in evs if not (e.type == "cmd" and e.actor == actor)]
        obs["events_dropped"] = dropped
        return obs

    def round(self) -> None:
        order = self.rng.permutation(len(self.agents))
        for i in order:
            actor, pol = self.agents[i]
            t0 = time.perf_counter()
            obs = self.observe(actor)
            tools = self.tools(actor)
            action = pol.act(obs, tools)
            res = None
            evs: list[int] = []
            if action is not None:
                r = self.world.execute(actor, action)
                res = r.to_dict()
                evs = r.events
                pol.observe_result(action, res)
            reward = 0.0
            obj = self.objectives.get(actor)
            if obj is not None:
                reward = obj.step_reward(self.world, actor)
            if self.record:
                self.trajs[actor].steps.append(Step(self.world.now, actor, pol.name, _compact(obs), len(tools), action,
                                                    res, evs, reward, (time.perf_counter() - t0) * 1000))
        self.world.advance_time(self.world.now + self.dt)

    def run(self, rounds: int) -> dict:
        for _ in range(rounds):
            self.round()
        return self.trajs


def _compact(obs: dict) -> dict:
    out = dict(obs)
    if len(out.get("events", [])) > 30:
        out["events"] = out["events"][-30:]
    return out
