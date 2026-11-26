"""Gymnasium and PettingZoo wrappers over any `ExperienceEnv` (every tier: Tier 1 mechanism and rapid worlds, Tier 2
domains, Tier 3 civilisations, 4.0 kernel worlds), with structured observations that keep exact numbers
(MASTER_SPEC 11.1: precise financial and technical measurements are not forced into lossy text).

    env = ExperienceGymEnv(1, "rapid:net_routing", seed=3)          # reward_mode defaults to observation_only
    obs, info = env.reset()
    obs["numbers"], obs["numbers_mask"]        # float64 vector of the observation's exact numbers (slot table below)
    obs["json"]                                # the whole `Observation` as JSON text: lossless, exact
    obs, reward, terminated, truncated, info = env.step({"tool": "rapid.levers", "args": {...}})
    env.verify()                               # evaluator side; never part of an observation

Observation space (`spaces.Dict`):

    t             Box(1,) float64      world time in seconds
    numbers       Box(max_numbers,)    the observation's `numbers`, float64, placed in slots; a slot is assigned to a
                  float64              path the first time it appears and keeps it for the life of the wrapper
                                       (``info["numbers_keys"]`` lists the slot table whenever it grows)
    numbers_mask  MultiBinary          1 where the slot holds a value of this observation
    json          Text                 `json.dumps(observation)` - every field of the universal contract, exact

Actions: a ``{"tool", "args"}`` dict, its JSON text, or None (wait). Text that does not parse is not executed; the
step then reports the parse error in ``info["action_result"]`` and time still passes.

Reward: whatever the wrapped environment's reward mode gives - in the default ``observation_only`` mode always 0.0,
which Gymnasium requires as a scalar and which is **not a learning signal** (Clarification 2). ``terminated`` is
never set by an outcome; ``truncated`` is set by ``max_steps`` and by the end of a Tier 1 episode (its horizon, a
time limit). The learner computes its own training signal from the observations if it wants one.
"""
from __future__ import annotations

import json
import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .modes import OBSERVATION_ONLY

CHARSET = "".join(chr(c) for c in range(32, 127)) + "\n\t"


def _env(tier, name, seed, reward_mode, env, kw):
    if env is not None:
        return env
    from .registry import make
    return make(int(tier), str(name), int(seed), reward_mode, **kw)


def _ascii(s: str) -> str:
    return s.encode("ascii", "backslashreplace").decode("ascii")


class _Slots:
    def __init__(self, n: int):
        self.n = int(n)
        self.keys: list[str] = []
        self.index: dict[str, int] = {}
        self.overflow = 0

    def encode(self, numbers: dict) -> tuple[np.ndarray, np.ndarray, bool]:
        vec = np.zeros(self.n, dtype=np.float64)
        mask = np.zeros(self.n, dtype=np.int8)
        grew = False
        for k in sorted(numbers):
            v = numbers[k]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
                continue
            i = self.index.get(k)
            if i is None:
                if len(self.keys) >= self.n:
                    self.overflow += 1
                    continue
                i = self.index[k] = len(self.keys)
                self.keys.append(k)
                grew = True
            vec[i] = float(v)
            mask[i] = 1
        return vec, mask, grew


def _space(max_numbers: int, max_json: int) -> spaces.Dict:
    return spaces.Dict({
        "t": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64),
        "numbers": spaces.Box(low=-np.inf, high=np.inf, shape=(int(max_numbers),), dtype=np.float64),
        "numbers_mask": spaces.MultiBinary(int(max_numbers)),
        "json": spaces.Text(max_length=int(max_json), min_length=0, charset=CHARSET),
    })


def _parse(action: Any) -> tuple[dict | None, dict | None]:
    if action is None or isinstance(action, dict):
        return action, None
    if isinstance(action, (bytes, str)):
        try:
            a = json.loads(action)
        except (TypeError, ValueError) as e:
            return None, {"ok": False, "t": 0, "tool": "", "cost_s": 0,
                          "error": {"code": "parse_error", "message": f"action is not JSON: {e}"}}
        if a is None or isinstance(a, dict):
            return a, None
    return None, {"ok": False, "t": 0, "tool": "", "cost_s": 0,
                  "error": {"code": "parse_error", "message": "an action is {'tool': str, 'args': {...}} or None"}}


class ExperienceGymEnv(gym.Env):
    """One agent of an `ExperienceEnv` as a Gymnasium environment (see the module docstring)."""
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, tier: int = 1, name: str = "mech:scm_additive", seed: int = 0, reward_mode: str | None = None,
                 *, env=None, agent: str | None = None, think_s: int | None = None, max_steps: int | None = 200,
                 max_numbers: int = 256, max_json: int = 1 << 20, render_mode: str | None = None, **kw):
        self.env = _env(tier, name, seed, reward_mode, env, kw)
        self.reward_mode = self.env.reward_mode
        self.agent = agent or self.env.agents()[0]
        self.think_s = int(think_s if think_s is not None else getattr(self.env, "tick_s", 600))
        self.max_steps = max_steps
        self.max_json = int(max_json)
        self.render_mode = render_mode
        self.slots = _Slots(max_numbers)
        self.observation_space = _space(max_numbers, max_json)
        self.action_space = spaces.Text(max_length=8192, min_length=0, charset=CHARSET)
        self.steps = 0
        self.episode = 0
        self._last = None

    def _obs(self, o: dict) -> tuple[dict, dict]:
        vec, mask, grew = self.slots.encode(o.get("numbers") or {})
        text = _ascii(json.dumps(o, sort_keys=True))
        if len(text) > self.max_json:
            raise ValueError(f"observation JSON of {len(text)} characters exceeds max_json={self.max_json}")
        self._last = o
        info = {"agent": self.agent, "reward_mode": self.reward_mode,
                "reward_is_learning_signal": self.reward_mode != OBSERVATION_ONLY, "cursor": o.get("cursor"),
                "numbers_overflow": self.slots.overflow}
        if grew:
            info["numbers_keys"] = list(self.slots.keys)
        return {"t": np.array([float(o["t"])], dtype=np.float64), "numbers": vec, "numbers_mask": mask,
                "json": text}, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.episode > 0 and getattr(self.env, "done", False) and hasattr(self.env, "reset"):
            self.env.reset()
        self.episode += 1
        self.steps = 0
        obs, info = self._obs(self.env.observe(self.agent))
        info["numbers_keys"] = list(self.slots.keys)
        return obs, info

    def step(self, action: Any):
        act, err = _parse(action)
        res = err
        if act is not None:
            res = self.env.execute(self.agent, act)
        self.env.advance(self.think_s)
        self.steps += 1
        reward = float(self.env.reward(self.agent))
        obs, info = self._obs(self.env.observe(self.agent))
        info["action_result"] = res
        truncated = bool(getattr(self.env, "done", False)) or \
            bool(self.max_steps is not None and self.steps >= self.max_steps)
        return obs, reward, False, truncated, info

    def verify(self) -> dict:
        """Evaluator side: the wrapped environment's independent verification."""
        return self.env.verify()

    def render(self):
        if self.render_mode == "ansi" and self._last is not None:
            from ..api import render as R
            return json.dumps(self._last, indent=1)[:65536] if not hasattr(R, "render_experience") else \
                R.render_experience(self._last)
        return None

    def close(self):
        try:
            self.env.close()
        finally:
            super().close()


def _parallel_base():
    from pettingzoo import ParallelEnv
    return ParallelEnv


class ExperienceParallelEnv(_parallel_base()):
    """Every agent of an `ExperienceEnv` as a PettingZoo parallel environment: `step(actions)` executes every live
    agent's action (a joint tick in Tier 1 rapid worlds), then time passes once. Rewards are the wrapped
    environment's (0.0 in the default observation-only mode); terminations are never outcome-based."""
    metadata = {"name": "crucible_experience_parallel_v1", "render_modes": []}

    def __init__(self, tier: int = 1, name: str = "rapid:team_commons", seed: int = 0, reward_mode: str | None = None,
                 *, env=None, think_s: int | None = None, max_steps: int | None = 200, max_numbers: int = 256,
                 max_json: int = 1 << 20, **kw):
        self.env = _env(tier, name, seed, reward_mode, env, kw)
        self.reward_mode = self.env.reward_mode
        self.possible_agents = list(self.env.agents())
        self.agents: list[str] = []
        self.think_s = int(think_s if think_s is not None else getattr(self.env, "tick_s", 600))
        self.max_steps = max_steps
        self.max_json = int(max_json)
        self.slots = {a: _Slots(max_numbers) for a in self.possible_agents}
        self._spaces = {a: _space(max_numbers, max_json) for a in self.possible_agents}
        self._act = {a: spaces.Text(max_length=8192, min_length=0, charset=CHARSET) for a in self.possible_agents}
        self.steps = 0

    def observation_space(self, agent):
        return self._spaces[agent]

    def action_space(self, agent):
        return self._act[agent]

    def _obs(self, a: str) -> tuple[dict, dict]:
        o = self.env.observe(a)
        vec, mask, grew = self.slots[a].encode(o.get("numbers") or {})
        info = {"reward_mode": self.reward_mode, "reward_is_learning_signal": self.reward_mode != OBSERVATION_ONLY}
        if grew:
            info["numbers_keys"] = list(self.slots[a].keys)
        return {"t": np.array([float(o["t"])], dtype=np.float64), "numbers": vec, "numbers_mask": mask,
                "json": _ascii(json.dumps(o, sort_keys=True))[:self.max_json]}, info

    def reset(self, seed: int | None = None, options: dict | None = None):
        if self.steps and getattr(self.env, "done", False) and hasattr(self.env, "reset"):
            self.env.reset()
        self.agents = list(self.possible_agents)
        self.steps = 0
        obs, infos = {}, {}
        for a in self.agents:
            obs[a], infos[a] = self._obs(a)
            infos[a]["numbers_keys"] = list(self.slots[a].keys)
        return obs, infos

    def step(self, actions: dict):
        results = {}
        for a in self.agents:
            act, err = _parse(actions.get(a))
            results[a] = err if act is None else self.env.execute(a, act)
        self.env.advance(self.think_s)
        self.steps += 1
        done = bool(getattr(self.env, "done", False)) or bool(self.max_steps is not None and
                                                               self.steps >= self.max_steps)
        obs, rew, term, trunc, infos = {}, {}, {}, {}, {}
        for a in self.agents:
            obs[a], infos[a] = self._obs(a)
            infos[a]["action_result"] = results.get(a)
            rew[a] = float(self.env.reward(a))
            term[a], trunc[a] = False, done
        if done:
            self.agents = []
        return obs, rew, term, trunc, infos

    def verify(self) -> dict:
        return self.env.verify()

    def close(self):
        self.env.close()
