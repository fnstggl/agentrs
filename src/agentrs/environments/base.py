"""Environment-agnostic interface.

Every environment (simulated web apps, real browsers via Playwright, BrowserGym,
later terminals / APIs) implements ``Environment``. The agent only ever sees
``Observation`` objects and the *public* part of a task (``PublicTask``). Task
answers and verifier configuration live in ``TaskSpec.private`` and never reach
the agent.
"""

from __future__ import annotations

import abc
import hashlib
from typing import Any, Optional

from pydantic import BaseModel, Field

from agentrs.actions.schema import Action


class ElementState(BaseModel):
    """One actionable or informative element in a structured observation."""

    eid: str
    role: str  # button, link, textbox, checkbox, combobox, menuitem, heading, text, alert, dialog, ...
    name: str = ""
    value: Optional[str] = None
    checked: Optional[bool] = None
    expanded: Optional[bool] = None
    disabled: bool = False
    focused: bool = False
    options: Optional[list[str]] = None

    def to_line(self) -> str:
        parts = [f"[{self.eid}] {self.role}"]
        if self.name:
            parts.append(_q(self.name))
        if self.value is not None:
            parts.append(f"value={_q(self.value)}")
        if self.checked is not None:
            parts.append("checked" if self.checked else "unchecked")
        if self.expanded is not None:
            parts.append("expanded" if self.expanded else "collapsed")
        if self.disabled:
            parts.append("disabled")
        if self.focused:
            parts.append("focused")
        if self.options:
            parts.append("options=[" + ", ".join(_q(o) for o in self.options) + "]")
        return " ".join(parts)

    def content_key(self) -> tuple:
        """Identity-free content signature (used for diffs across id changes)."""
        return (self.role, self.name)


def _q(s: str) -> str:
    s = s.replace("\n", " ")
    if len(s) > 80:
        s = s[:77] + "..."
    return '"' + s.replace('"', "'") + '"'


class Observation(BaseModel):
    """What the agent observes after a step. Never contains hidden simulator state."""

    env_id: str
    url: str = ""
    title: str = ""
    elements: list[ElementState] = Field(default_factory=list)
    text: Optional[str] = None  # free-form text (terminal output, tool output)
    last_action_error: Optional[str] = None  # observable error from the last action
    scroll_info: Optional[str] = None  # e.g. "more content below"
    screenshot_ref: Optional[str] = None  # content hash of a stored screenshot
    step: int = 0

    def element(self, eid: str) -> Optional[ElementState]:
        for e in self.elements:
            if e.eid == eid:
                return e
        return None

    def element_ids(self) -> set[str]:
        return {e.eid for e in self.elements}

    def to_text(self, max_elements: int = 60) -> str:
        lines = [f"URL: {self.url}", f"TITLE: {self.title}"]
        if self.last_action_error:
            lines.append(f"LAST_ERROR: {self.last_action_error}")
        for e in self.elements[:max_elements]:
            lines.append(e.to_line())
        if len(self.elements) > max_elements:
            lines.append(f"... ({len(self.elements) - max_elements} more elements)")
        if self.scroll_info:
            lines.append(f"SCROLL: {self.scroll_info}")
        if self.text:
            lines.append("TEXT: " + self.text[:500])
        return "\n".join(lines)

    def fingerprint(self) -> str:
        return hashlib.sha1(self.to_text(10_000).encode()).hexdigest()[:16]


class PublicTask(BaseModel):
    """The part of a task the agent is allowed to see."""

    task_id: str
    env_id: str
    family: str
    goal: str
    max_steps: int = 20
    category: str = ""  # coarse category, e.g. "navigation", "form", "retrieval"


class TaskSpec(BaseModel):
    """Full task specification. ``private`` holds answers/verifier config."""

    task_id: str
    env_id: str
    family: str
    goal: str
    seed: int = 0
    split: str = "train"
    max_steps: int = 20
    category: str = ""
    layout: str = "default"
    difficulty: int = 1
    private: dict[str, Any] = Field(default_factory=dict, repr=False)

    def public(self) -> PublicTask:
        return PublicTask(
            task_id=self.task_id,
            env_id=self.env_id,
            family=self.family,
            goal=self.goal,
            max_steps=self.max_steps,
            category=self.category,
        )


class Verification(BaseModel):
    """Independent outcome verification, computed from environment-side state."""

    success: bool
    score: float = 0.0
    progress: float = 0.0  # objectively verifiable partial progress in [0, 1]
    verifier_id: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    observation: Observation
    reward: float = 0.0
    terminated: bool = False  # true task termination (success/failure decided)
    truncated: bool = False  # cut off by budget/timeout; not a true terminal state
    action_valid: bool = True
    error: Optional[str] = None
    info: dict[str, Any] = Field(default_factory=dict)


class EnvironmentError_(RuntimeError):
    """Raised for unrecoverable environment failures (crash, reset failure)."""


class Environment(abc.ABC):
    """Common environment interface. Implementations must be safe to call
    ``step`` with *any* typed action: invalid actions produce an observable
    error, never a crash."""

    env_id: str = "abstract"
    version: str = "0"

    @abc.abstractmethod
    def reset(self, task: TaskSpec) -> Observation: ...

    @abc.abstractmethod
    def observe(self) -> Observation: ...

    @abc.abstractmethod
    def available_actions(self) -> list[Action]:
        """A finite set of currently valid candidate actions (an action mask).
        Free-text arguments are proposals; environments may accept others."""

    @abc.abstractmethod
    def step(self, action: Action) -> StepResult: ...

    @abc.abstractmethod
    def verify_outcome(self) -> Verification: ...

    @abc.abstractmethod
    def get_task_metadata(self) -> PublicTask: ...

    def close(self) -> None:  # pragma: no cover - default no-op
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
