"""Belief state for a partially observable environment.

The belief state explicitly separates
* **observed facts**: the current observation and the bounded action/outcome
  history (exactly what the environment reported), from
* **inferred state**: salient values extracted from earlier observations that
  are no longer visible (e.g. an order number seen on a previous page),
  retrieved episodic-memory hints, and an optional learned latent vector.

The serialized prompt marks the two sections separately so the model is never
told that inferred content is an observed fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from agentrs.actions.schema import Action, ActionType
from agentrs.environments.base import Observation, PublicTask, StepResult

_SALIENT_RE = re.compile(r"(\$\d+(?:\.\d\d)?|#[A-Z0-9]{3,8}|\b[A-Z]{2,3}-\d{3,6}\b)")


@dataclass
class HistoryItem:
    action: str
    outcome: str  # "ok" | "error: ..." | "navigated to /x"
    url: str = ""


@dataclass
class BeliefState:
    task: PublicTask
    max_history: int = 6
    max_obs_elements: int = 40
    observation: Optional[Observation] = None
    history: list[HistoryItem] = field(default_factory=list)
    salient: list[tuple[str, str]] = field(default_factory=list)  # (value, context) inferred from past obs
    memory_hints: list[str] = field(default_factory=list)
    latent: Optional[Any] = None  # optional learned recurrent state (torch tensor)
    steps: int = 0

    def reset(self, obs: Observation) -> None:
        self.observation = obs
        self.history = []
        self.salient = []
        self.steps = 0
        self._harvest(obs)

    def _harvest(self, obs: Observation) -> None:
        texts = [e.name for e in obs.elements] + ([obs.text] if obs.text else [])
        for t in texts:
            for m in _SALIENT_RE.findall(t or ""):
                if m not in [s for s, _ in self.salient]:
                    self.salient.append((m, t[:60]))
        self.salient = self.salient[-12:]

    def update(self, action: Action, result: StepResult) -> None:
        prev_url = self.observation.url if self.observation else ""
        if result.error:
            outcome = "error: " + result.error[:80]
        elif result.observation.url != prev_url:
            outcome = f"navigated to {result.observation.url}"
        else:
            outcome = "ok"
        self.history.append(HistoryItem(action=action.to_str(), outcome=outcome, url=prev_url))
        self.observation = result.observation
        self.steps += 1
        self._harvest(result.observation)

    # ------------------------------------------------------------------ prompt
    def prompt(self, observation: Optional[Observation] = None, history: Optional[list[HistoryItem]] = None) -> str:
        obs = observation if observation is not None else self.observation
        hist = self.history if history is None else history
        assert obs is not None
        lines = [f"GOAL: {self.task.goal}"]
        inferred = []
        visible = obs.to_text(10_000)
        for v, ctx in self.salient:
            if v not in visible:
                inferred.append(f"seen earlier: {ctx}")
        inferred += [f"memory: {h}" for h in self.memory_hints[:3]]
        if inferred:
            lines.append("INFERRED (not currently visible):")
            lines += ["- " + s for s in inferred[-5:]]
        if hist:
            lines.append("HISTORY:")
            for i, h in enumerate(hist[-self.max_history:], start=max(1, len(hist) - self.max_history + 1)):
                lines.append(f"{i}. {h.action} -> {h.outcome}")
        lines.append("OBSERVATION:")
        lines.append(obs.to_text(self.max_obs_elements))
        lines.append("ACTION:")
        return "\n".join(lines)

    # ------------------------------------------------------------------ candidates
    def augment_candidates(self, env_candidates: list[Action], obs: Optional[Observation] = None) -> list[Action]:
        """Add generic candidates derived from inferred state: values seen on
        earlier pages can be typed into editable fields or given as answers."""
        obs = obs if obs is not None else self.observation
        out = list(env_candidates)
        seen = {a.to_str() for a in out}
        values = [v for v, _ in self.salient]
        if obs is not None and values:
            for e in obs.elements:
                if e.role in ("textbox", "searchbox") and not e.disabled:
                    for v in values[-4:]:
                        a = Action.type_text(e.eid, v)
                        if a.to_str() not in seen:
                            out.append(a)
                            seen.add(a.to_str())
            for v in values[-4:]:
                a = Action.finish(v)
                if a.to_str() not in seen:
                    out.append(a)
                    seen.add(a.to_str())
        return out

    def snapshot(self) -> dict:
        return {"steps": self.steps, "history": [h.__dict__ for h in self.history],
                "salient": list(self.salient), "memory_hints": list(self.memory_hints)}


def action_text_for_lm(a: Action | str) -> str:
    """Continuation string the language model scores for an action."""
    s = a if isinstance(a, str) else a.to_str()
    return " " + s + "\n"


def is_terminal_action(a: Action) -> bool:
    return a.type == ActionType.FINISH
