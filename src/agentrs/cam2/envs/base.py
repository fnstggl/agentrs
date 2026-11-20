"""Unified environment interface for CAM 2.0.

``EnvironmentAdapter`` is adapted to each benchmark's real API — it never replaces the benchmark's own task logic
or evaluator. Methods a benchmark does not support natively are implemented honestly (e.g. ``snapshot`` as
"task + seed + action history" with replay-based ``restore``) or raise ``NotSupported``.

An environment whose runtime dependencies are missing raises ``DependencyUnavailable`` with the exact setup
procedure; nothing silently substitutes a different environment.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from agentrs.cam2.tools import Observation, ToolCall, ToolSpec


class DependencyUnavailable(RuntimeError):
    """A benchmark's runtime dependency is missing. The message carries the setup procedure."""


class NotSupported(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskRef:
    env: str          # adapter name ("miniwob", "appworld", ...)
    family: str       # task family (MiniWoB task name, AppWorld scenario id, ...)
    seed: int = 0     # instance seed (MiniWoB) / unused (AppWorld task ids are instances)
    instance: str = ""  # benchmark-native instance id (AppWorld task_id)

    def key(self) -> str:
        return f"{self.env}:{self.family}:{self.instance or self.seed}"


@dataclass
class StepResult:
    obs: Observation
    reward: float
    terminated: bool          # the task reached a terminal state (success or failure)
    truncated: bool           # stopped by a step/time limit: bootstrap, do not treat as terminal
    info: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    success: bool
    score: float              # benchmark-native score in [0, 1] (MiniWoB: success; AppWorld: passed-test fraction)
    details: dict = field(default_factory=dict)


class EnvironmentAdapter(ABC):
    name = "env"

    @abstractmethod
    def reset(self, task: TaskRef) -> Observation: ...

    @abstractmethod
    def observe(self) -> Observation: ...

    @abstractmethod
    def available_tools(self) -> list[ToolSpec]: ...

    @abstractmethod
    def step(self, action: ToolCall) -> StepResult: ...

    @abstractmethod
    def evaluate(self) -> EvalResult: ...

    def snapshot(self) -> Any:
        raise NotSupported(f"{self.name} has no snapshot support")

    def restore(self, snap: Any) -> Observation:
        raise NotSupported(f"{self.name} has no restore support")

    def close(self) -> None:
        pass

    def describe(self) -> dict:
        return {"name": self.name}
