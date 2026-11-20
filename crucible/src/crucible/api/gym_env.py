"""Gymnasium environment: one learning agent acting in a (possibly multi-actor) persistent world.

The world is *not* the episode. An episode is a bounded stretch of one actor's life used for credit assignment;
the world it lives in keeps running across episodes:

* ``reset_mode="continue"`` (default): `reset()` only marks an episode boundary. Nothing in the world is reset:
  money, messages, reputation and the clock carry over.
* ``reset_mode="fork"``: every `reset()` forks a fresh copy of a base snapshot (default: the world right after the
  env was constructed), so episodes start from one fixed state (episodic training). The persistent world is never
  modified by these episodes; forks are ephemeral by default and discarded at the next reset.

Observation: ``{"text": <rendered observation>, "time": Box([t])}``; action: a JSON tool call as text (any shape
`render.parse_action` accepts) or a dict ``{"tool": ..., "args": {...}}``. After the action the world clock moves
by `dt` seconds (``time_mode="fixed"``), by the action's cost (``"cost"``) or not at all (``"none"``: an external
driver owns the clock). Rewards and termination come from an optional *objective* object::

    class Objective:
        def begin(self, world, actor): ...                      # episode start (world = kernel World)
        def step_reward(self, world, actor) -> float: ...       # after every step
        def status(self, world, actor) -> dict: ...             # {"success": bool, "failed": bool, "progress": float}

Without one the reward is 0.0 and episodes end only by truncation (`max_steps`).
"""
from __future__ import annotations

import string
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import render as R
from .local import API_VERSION, ActorIdentity, Observation, WorldServer

CHARSET = frozenset(string.printable)  # ASCII printable incl. whitespace; renderers escape everything else


def _resolve_actor(server: WorldServer, world_id: str, actor: Any, handle: str, endowment: int) -> ActorIdentity:
    if actor is None:
        return server.join_world(world_id, handle, endowment=endowment)
    if isinstance(actor, ActorIdentity):
        return server.authenticate(world_id, actor)
    return server.authenticate(world_id, str(actor))


class CrucibleEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "api_version": API_VERSION}

    def __init__(self, server: WorldServer | None = None, world_id: str | None = None,
                 actor: ActorIdentity | str | None = None, *, spec: Any = None, seed: int | None = None,
                 handle: str = "agent", endowment: int = 0, objective: Any = None, reset_mode: str = "continue",
                 base_snapshot: str | None = None, ephemeral_forks: bool = True, dt: int = 60,
                 time_mode: str = "fixed", min_dt: int = 1, max_steps: int | None = 100, max_events: int = 30,
                 max_text_length: int = 65536, max_action_length: int = 8192,
                 renderer: Callable[[Observation], str] | None = None, include_tools: str = "changed",
                 driver: Callable[[WorldServer, str], None] | None = None, render_mode: str | None = None):
        if reset_mode not in ("continue", "fork"):
            raise ValueError("reset_mode must be 'continue' or 'fork'")
        if time_mode not in ("fixed", "cost", "none"):
            raise ValueError("time_mode must be 'fixed', 'cost' or 'none'")
        self._owns_server = server is None
        self.server = server or WorldServer()
        if world_id is None:
            if spec is None:
                raise ValueError("CrucibleEnv needs world_id or spec")
            world_id = self.server.create_world(spec, seed)
        self.world_id = world_id
        self.identity = _resolve_actor(self.server, world_id, actor, handle, endowment)
        self.principal = self.identity.principal
        self.objective = objective
        self.reset_mode = reset_mode
        self.ephemeral_forks = ephemeral_forks
        self.base_snapshot = base_snapshot
        if reset_mode == "fork" and base_snapshot is None:
            self.base_snapshot = self.server.snapshot_world(world_id)
        self.dt, self.time_mode, self.min_dt = int(dt), time_mode, int(min_dt)
        self.max_steps = max_steps
        self.max_events = max_events
        self.max_text_length = max_text_length
        self.renderer = renderer
        self.include_tools = include_tools
        self.driver = driver
        self.render_mode = render_mode
        self.observation_space = spaces.Dict({
            "text": spaces.Text(max_length=max_text_length, min_length=0, charset=CHARSET),
            "time": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64),
        })
        self.action_space = spaces.Text(max_length=max_action_length, min_length=0, charset=CHARSET)
        self.episode = 0
        self.steps = 0
        self._wid: str | None = None
        self._fork: str | None = None
        self._last_text = ""
        self._last_obs: Observation | None = None

    # ------------------------------------------------------------------ helpers
    @property
    def current_world_id(self) -> str | None:
        """World the current episode runs in (the persistent world, or this episode's fork)."""
        return self._wid

    @property
    def world(self):
        """Kernel `World` of the current episode (for objectives and analysis; read it, don't mutate it)."""
        return self.server.world(self._wid or self.world_id)

    def _actor(self) -> ActorIdentity:
        return ActorIdentity(self._wid, self.principal, self.identity.token, self.identity.account)

    def _observe(self) -> tuple[dict, Observation]:
        o = self.server.observe(self._wid, self._actor(), max_events=self.max_events,
                                include_tools=self.include_tools)
        text = self.renderer(o) if self.renderer else o.text
        text = R.sanitize_text(text, CHARSET, self.max_text_length)
        self._last_text, self._last_obs = text, o
        return {"text": text, "time": np.array([float(o.time)], dtype=np.float64)}, o

    def _info(self, o: Observation, **extra) -> dict:
        info = {"world_id": o.world_id, "principal": o.principal, "episode": self.episode, "step": self.steps,
                "time": o.time, "seq": o.seq, "events": o.events, "dropped_events": o.dropped,
                "tools_changed": o.tools_changed}
        if o.tools_changed:
            info["available_tools"] = o.tools if o.tools is not None else \
                self.server.available_tools(self._wid, self._actor())
        info.update(extra)
        return info

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        mode = options.get("reset_mode", self.reset_mode)
        self.episode += 1
        self.steps = 0
        if mode == "fork":
            snap = options.get("snapshot") or self.base_snapshot or self.server.snapshot_world(self.world_id)
            old = self._fork
            self._fork = self.server.fork_world(snap, ephemeral=self.ephemeral_forks)
            if old is not None and self.ephemeral_forks:
                self.server.close_world(old)
            self._wid = self._fork
        else:
            self._wid = self.server.resolve(self.world_id)
        status = {}
        if self.objective is not None:
            w = self.server.world(self._wid)
            self.objective.begin(w, self.principal)
            status = dict(self.objective.status(w, self.principal) or {})
        obs, o = self._observe()
        return obs, self._info(o, objective=status, reset_mode=mode)

    def step(self, action: Any):
        if self._wid is None:
            raise gym.error.ResetNeeded("call reset() before step()")
        res = self.server.execute_action(self._wid, self._actor(), action)
        if self.driver is not None:
            self.driver(self.server, self._wid)
        if self.time_mode == "fixed" and self.dt > 0:
            self.server.advance_time(self._wid, dt=self.dt)
        elif self.time_mode == "cost":
            self.server.advance_time(self._wid, dt=max(int(res.get("cost_s") or 0), self.min_dt))
        self.steps += 1
        reward, status = 0.0, {}
        if self.objective is not None:
            w = self.server.world(self._wid)
            reward = float(self.objective.step_reward(w, self.principal))
            status = dict(self.objective.status(w, self.principal) or {})
        terminated = bool(status.get("success") or status.get("failed"))
        truncated = bool(self.max_steps is not None and self.steps >= self.max_steps and not terminated)
        obs, o = self._observe()
        return obs, reward, terminated, truncated, self._info(o, action_result=res, objective=status)

    def render(self):
        if self.render_mode == "ansi":
            return self._last_text
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
        super().close()
