"""Gymnasium view of any ``agentrs`` Environment.

Observations are the canonical observation text (goal prepended); actions are canonical
action strings parsed by the strict action grammar (unparseable strings become an
observable error, never code execution). ``info`` carries the structured observation,
the candidate action mask and, at episode end, the verification result.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import gymnasium as gym
from gymnasium import spaces

from agentrs.actions.schema import try_parse_action
from agentrs.environments.base import Environment, TaskSpec

_CHARSET = "".join(chr(i) for i in range(32, 127)) + "\n"


class GymnasiumEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, env: Environment, tasks: Sequence[TaskSpec], max_obs_chars: int = 8000,
                 render_mode: Optional[str] = None):
        self.env = env
        self.tasks = list(tasks)
        self.max_obs_chars = max_obs_chars
        self.render_mode = render_mode
        self.observation_space = spaces.Text(max_length=max_obs_chars, charset=_CHARSET)
        self.action_space = spaces.Text(max_length=300, charset=_CHARSET)
        self._task: Optional[TaskSpec] = None
        self._last = ""

    def _fmt(self, obs) -> str:
        txt = f"GOAL: {self._task.goal}\n" + obs.to_text()
        txt = "".join(c if c in _CHARSET else "?" for c in txt)
        return txt[: self.max_obs_chars]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        idx = (options or {}).get("task_index")
        if idx is None:
            idx = int(self.np_random.integers(len(self.tasks)))
        self._task = self.tasks[idx]
        obs = self.env.reset(self._task)
        self._last = self._fmt(obs)
        return self._last, {"observation": obs, "action_mask": [a.to_str() for a in self.env.available_actions()],
                            "task_id": self._task.task_id}

    def step(self, action: str):
        a = try_parse_action(action) if isinstance(action, str) else action
        if a is None:
            obs = self.env.observe().model_copy(update={"last_action_error": f"unparseable action {action!r}"})
            self._last = self._fmt(obs)
            return self._last, 0.0, False, False, {"observation": obs, "action_valid": False}
        r = self.env.step(a)
        info: dict[str, Any] = {"observation": r.observation, "action_valid": r.action_valid, "error": r.error,
                                "action_mask": [x.to_str() for x in self.env.available_actions()]}
        if r.terminated or r.truncated:
            info["verification"] = self.env.verify_outcome().model_dump()
        self._last = self._fmt(r.observation)
        return self._last, float(r.reward), bool(r.terminated), bool(r.truncated), info

    def render(self):
        return self._last

    def close(self):
        self.env.close()
