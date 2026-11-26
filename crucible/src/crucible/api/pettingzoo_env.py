"""PettingZoo parallel environment: several learning agents (actors) in one persistent world.

Each `step(actions)` executes every live agent's action in a random order drawn from a per-step seeded generator
(``default_rng([order_seed, episode, step])``; the order is reported in ``infos[agent]["order"]``), then advances
the world clock once by `dt`. Rewards/termination come from optional per-agent objectives (same protocol as
`CrucibleEnv`); agents may have different observation renderers. Episode semantics are those of `CrucibleEnv`:
``reset_mode="continue"`` never resets the world, ``reset_mode="fork"`` forks a base snapshot per episode.

``reward_mode`` (Crucible 5.0) is that of `CrucibleEnv`: ``observation_only`` (the default, Clarification 2),
``legacy_shaped_reward`` (explicitly selected: the 4.0 behaviour),
``verified_terminal_outcome`` (reward 1.0 once per agent at its first verified success; ``infos[a]["objective"]``
holds only ``success``/``failed``) and ``observation_only`` (rewards 0.0, no ``objective`` in the infos, termination
only by truncation; `verify()` returns every agent's status to the evaluator).
"""
from __future__ import annotations

import functools
from typing import Any, Callable

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from ..experience.modes import (LEGACY, OBSERVATION_ONLY, VERIFIED, check_mode, info_objective, prepare_spec,
                                prepare_world)
from . import render as R
from .gym_env import CHARSET
from .local import API_VERSION, ActorIdentity, Observation, WorldServer


class CrucibleParallelEnv(ParallelEnv):
    metadata = {"name": "crucible_parallel_v1", "render_modes": ["ansi"], "is_parallelizable": True,
                "api_version": API_VERSION}

    def __init__(self, server: WorldServer | None = None, world_id: str | None = None,
                 actors: dict[str, ActorIdentity | str] | None = None, *, spec: Any = None, seed: int | None = None,
                 agents: list[str] | None = None, n_agents: int = 2, endowment: int = 0,
                 objectives: dict[str, Any] | None = None,
                 renderers: dict[str, Callable[[Observation], str]] | None = None, reset_mode: str = "continue",
                 base_snapshot: str | None = None, ephemeral_forks: bool = True, dt: int = 60,
                 max_steps: int | None = 100, max_events: int = 30, max_text_length: int = 65536,
                 max_action_length: int = 8192, order_seed: int = 0,
                 driver: Callable[[WorldServer, str], None] | None = None, render_mode: str | None = None,
                 reward_mode: str | None = None):
        self.reward_mode = check_mode(reward_mode)
        self._paid: set[str] = set()
        self._statuses: dict[str, dict] = {}
        if reset_mode not in ("continue", "fork"):
            raise ValueError("reset_mode must be 'continue' or 'fork'")
        self._owns_server = server is None
        self.server = server or WorldServer()
        if world_id is None:
            if spec is None:
                raise ValueError("CrucibleParallelEnv needs world_id or spec")
            world_id = self.server.create_world(prepare_spec(self.server._spec_from(spec, seed), self.reward_mode))
        else:
            prepare_world(self.server.world(world_id), self.reward_mode)
        self.world_id = world_id
        if actors is None:
            names = agents or [f"agent-{i}" for i in range(n_agents)]
            actors = {n: self.server.join_world(world_id, n, endowment=endowment) for n in names}
        self.identities: dict[str, ActorIdentity] = {
            n: self.server.authenticate(world_id, a) for n, a in actors.items()}
        self.possible_agents = list(self.identities)
        self.agents: list[str] = []
        self.objectives = dict(objectives or {})
        self.renderers = dict(renderers or {})
        self.reset_mode = reset_mode
        self.ephemeral_forks = ephemeral_forks
        self.base_snapshot = base_snapshot
        if reset_mode == "fork" and base_snapshot is None:
            self.base_snapshot = self.server.snapshot_world(world_id)
        self.dt = int(dt)
        self.max_steps = max_steps
        self.max_events = max_events
        self.max_text_length = max_text_length
        self.max_action_length = max_action_length
        self.order_seed = int(order_seed)
        self.driver = driver
        self.render_mode = render_mode
        self.episode = 0
        self.steps = 0
        self._wid: str | None = None
        self._fork: str | None = None
        self._last_text: dict[str, str] = {}

    # ------------------------------------------------------------------ spaces
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return spaces.Dict({
            "text": spaces.Text(max_length=self.max_text_length, min_length=0, charset=CHARSET),
            "time": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return spaces.Text(max_length=self.max_action_length, min_length=0, charset=CHARSET)

    # ------------------------------------------------------------------ helpers
    @property
    def current_world_id(self) -> str | None:
        return self._wid

    def _actor(self, agent: str) -> ActorIdentity:
        i = self.identities[agent]
        return ActorIdentity(self._wid, i.principal, i.token, i.account)

    def _observe(self, agent: str) -> tuple[dict, Observation]:
        o = self.server.observe(self._wid, self._actor(agent), max_events=self.max_events)
        fn = self.renderers.get(agent)
        text = R.sanitize_text(fn(o) if fn else o.text, CHARSET, self.max_text_length)
        self._last_text[agent] = text
        return {"text": text, "time": np.array([float(o.time)], dtype=np.float64)}, o

    def _status(self, agent: str) -> tuple[float, dict]:
        obj = self.objectives.get(agent)
        if obj is None:
            return 0.0, {}
        w = self.server.world(self._wid)
        p = self.identities[agent].principal
        return float(obj.step_reward(w, p)), dict(obj.status(w, p) or {})

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None, options: dict | None = None):
        options = options or {}
        if seed is not None:
            self.order_seed = int(seed)
        self.episode += 1
        self.steps = 0
        mode = options.get("reset_mode", self.reset_mode)
        if mode == "fork":
            snap = options.get("snapshot") or self.base_snapshot or self.server.snapshot_world(self.world_id)
            old = self._fork
            self._fork = self.server.fork_world(snap, ephemeral=self.ephemeral_forks)
            if old is not None and self.ephemeral_forks:
                self.server.close_world(old)
            self._wid = self._fork
        else:
            self._wid = self.server.resolve(self.world_id)
        self.agents = list(self.possible_agents)
        self._paid = set()
        w = self.server.world(self._wid)
        obs, infos = {}, {}
        for a in self.agents:
            obj = self.objectives.get(a)
            if obj is not None:
                obj.begin(w, self.identities[a].principal)
            obs[a], o = self._observe(a)
            infos[a] = {"world_id": o.world_id, "principal": o.principal, "episode": self.episode, "step": 0,
                        "time": o.time, "events": o.events, "tools_changed": o.tools_changed,
                        "available_tools": o.tools, "reset_mode": mode}
        return obs, infos

    def step(self, actions: dict[str, Any]):
        if self._wid is None:
            raise RuntimeError("call reset() before step()")
        rng = np.random.default_rng([self.order_seed, self.episode, self.steps])
        acting = [a for a in self.agents if a in actions]
        order = [acting[i] for i in rng.permutation(len(acting))]
        results = {a: self.server.execute_action(self._wid, self._actor(a), actions[a]) for a in order}
        if self.driver is not None:
            self.driver(self.server, self._wid)
        if self.dt > 0:
            self.server.advance_time(self._wid, dt=self.dt)
        self.steps += 1
        obs, rew, term, trunc, infos = {}, {}, {}, {}, {}
        for a in self.agents:
            r, status = self._status(a)
            self._statuses[a] = status
            obs[a], o = self._observe(a)
            if self.reward_mode == LEGACY:
                rew[a] = r
                term[a] = bool(status.get("success") or status.get("failed"))
            elif self.reward_mode == VERIFIED:
                rew[a] = 0.0
                if status.get("success") and a not in self._paid:
                    self._paid.add(a)
                    rew[a] = 1.0
                term[a] = bool(status.get("success") or status.get("failed"))
            else:                               # observation-only: no reward, no objective, no outcome termination
                rew[a], term[a] = 0.0, False
            trunc[a] = bool(self.max_steps is not None and self.steps >= self.max_steps and not term[a])
            infos[a] = {"world_id": o.world_id, "principal": o.principal, "episode": self.episode,
                        "step": self.steps, "time": o.time, "order": list(order), "action_result": results.get(a),
                        "events": o.events, "dropped_events": o.dropped, "tools_changed": o.tools_changed}
            if self.reward_mode == LEGACY:
                infos[a]["objective"] = status
            elif self.reward_mode != OBSERVATION_ONLY:
                infos[a]["objective"] = info_objective(self.reward_mode, status)
            if o.tools_changed:
                infos[a]["available_tools"] = o.tools
        self.agents = [a for a in self.agents if not (term[a] or trunc[a])]
        return obs, rew, term, trunc, infos

    def verify(self) -> dict:
        """Evaluator side: every agent's latest objective status."""
        return {a: dict(s) for a, s in self._statuses.items()}

    def render(self):
        if self.render_mode == "ansi":
            return "\n\n".join(f"== {a}\n{t}" for a, t in sorted(self._last_text.items()))
        return None

    def close(self):
        if self._fork is not None and self.ephemeral_forks:
            try:
                self.server.close_world(self._fork)
            except Exception:  # noqa: BLE001
                pass
            self._fork = None
        if self._owns_server:
            self.server.close()
