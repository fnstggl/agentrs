"""Reusable skills (options) and hierarchical execution.

A ``Skill`` is a named, parameterised multi-step behaviour with preconditions,
a routine that maps the current observation to the next *primitive* action (or
``None`` when done), termination and success criteria, environment
compatibility and a learned reliability estimate (Beta posterior over verified
outcomes of episodes in which it was used).

``SkillExecutor`` wraps any actor. When the inner actor selects a skill action
(``skill(name, "args")``), the executor expands it step by step into primitive
actions. Every primitive action still goes through ``env.step`` (and therefore
through ``GuardedEnvironment``), so skills cannot bypass permissions.

``discover_skills`` mines frequent id-free action patterns from successful
trajectories as candidate skills (an initial, simple form of skill discovery).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from agentrs.actions.schema import Action, ActionType
from agentrs.agent.belief import BeliefState
from agentrs.agent.runtime import Decision
from agentrs.environments.base import Observation

Routine = Callable[[Observation, str, dict], Optional[Action]]


@dataclass
class Skill:
    name: str
    description: str
    parameters: list[str]
    precondition: Callable[[Observation, str], bool]
    routine: Routine
    max_steps: int = 8
    success_criteria: str = ""
    env_compat: tuple[str, ...] = ("miniweb", "localweb")
    successes: int = 0
    failures: int = 0

    @property
    def reliability(self) -> float:
        """Posterior mean of a Beta(1,1) prior updated with verified outcomes."""
        return (self.successes + 1) / (self.successes + self.failures + 2)

    def record(self, success: bool) -> None:
        if success:
            self.successes += 1
        else:
            self.failures += 1


def _quoted(goal: str) -> list[str]:
    return re.findall(r"[\"“]([^\"”]{1,64})[\"”]", goal)


def _fill_and_submit(obs: Observation, goal: str, state: dict) -> Optional[Action]:
    """Type each quoted goal value into the field whose label is mentioned next to it, then submit."""
    vals = _quoted(goal)
    fields = [e for e in obs.elements if e.role in ("textbox", "searchbox") and not e.disabled]
    used = state.setdefault("filled", set())
    for f in fields:
        if f.eid in used:
            continue
        lab = f.name.lower()
        best = None
        for v in vals:
            i = goal.find(f'"{v}"')
            ctx = goal[max(0, i - 40): i + len(v) + 40].lower()
            if any(w in ctx for w in lab.split() if len(w) > 2):
                best = v
                break
        used.add(f.eid)
        if best is not None and (f.value or "") != best:
            return Action.type_text(f.eid, best)
    if state.get("submitted"):
        return None
    for e in obs.elements:
        if e.role == "button" and e.name.lower() in ("submit", "sign in", "create account", "place order", "save",
                                                     "search"):
            state["submitted"] = True
            return Action.click(e.eid)
    return None


def _open_menu_item(obs: Observation, goal: str, state: dict) -> Optional[Action]:
    vals = _quoted(goal)
    for e in obs.elements:
        if e.role == "menuitem" and e.name in vals:
            return Action.click(e.eid)
    if state.get("opened"):
        return None
    for e in obs.elements:
        if e.role == "menubutton" and not e.expanded:
            state["opened"] = True
            return Action.click(e.eid)
    return None


def default_skills() -> list[Skill]:
    return [
        Skill("fill_and_submit", "Fill labelled form fields with quoted goal values, then submit.", [],
              precondition=lambda o, g: any(e.role in ("textbox", "searchbox") for e in o.elements),
              routine=_fill_and_submit, success_criteria="form submitted with goal values"),
        Skill("open_menu_item", "Open a collapsed menu and click the quoted menu item.", [],
              precondition=lambda o, g: any(e.role == "menubutton" for e in o.elements),
              routine=_open_menu_item, success_criteria="menu item clicked"),
    ]


class SkillLibrary:
    def __init__(self, skills: Optional[Sequence[Skill]] = None):
        self.skills = {s.name: s for s in (skills or default_skills())}

    def applicable(self, obs: Observation, goal: str, env_id: str = "miniweb") -> list[Skill]:
        return [s for s in self.skills.values() if env_id in s.env_compat and s.precondition(obs, goal)]

    def candidate_actions(self, obs: Observation, goal: str, env_id: str = "miniweb") -> list[Action]:
        return [Action(type=ActionType.SKILL, name=s.name) for s in self.applicable(obs, goal, env_id)]


class SkillExecutor:
    """Actor wrapper implementing hierarchical (option-style) execution."""

    def __init__(self, inner, library: SkillLibrary, offer_skills: bool = True):
        self.inner, self.lib, self.offer = inner, library, offer_skills
        self.name = f"skills+{getattr(inner, 'name', 'actor')}"
        self.active: Optional[Skill] = None
        self.state: dict = {}
        self.steps = 0
        self.used: list[str] = []

    def begin_episode(self, belief: BeliefState):
        self.active, self.state, self.steps, self.used = None, {}, 0, []
        self.inner.begin_episode(belief)

    def decide(self, belief: BeliefState, candidates: list[Action]) -> Decision:
        obs = belief.observation
        assert obs is not None
        if self.active is not None:
            a = self.active.routine(obs, belief.task.goal, self.state) if self.steps < self.active.max_steps else None
            if a is not None and a.to_str() in {c.to_str() for c in candidates} | {a.to_str()}:
                self.steps += 1
                return Decision(action=a, candidates=[c.to_str() for c in candidates], selection=f"skill:{self.active.name}")
            self.active = None  # terminated
        cands = list(candidates)
        if self.offer:
            cands += self.lib.candidate_actions(obs, belief.task.goal)
        dec = self.inner.decide(belief, cands)
        if dec.action.type == ActionType.SKILL and dec.action.name in self.lib.skills:
            self.active = self.lib.skills[dec.action.name]
            self.state, self.steps = {}, 0
            self.used.append(self.active.name)
            return self.decide(belief, candidates)
        return dec

    def end_episode(self, success: bool) -> None:
        for n in set(self.used):
            self.lib.skills[n].record(success)


def discover_skills(trajectories: Sequence[Sequence[str]], min_count: int = 5, length: int = 2) -> list[tuple[tuple, int]]:
    """Frequent id-free action-type n-grams in successful trajectories (candidate skills)."""
    cnt: Counter = Counter()
    for traj in trajectories:
        types = [re.sub(r"\(.*", "", a) for a in traj]
        for i in range(len(types) - length + 1):
            cnt[tuple(types[i : i + length])] += 1
    return [(k, v) for k, v in cnt.most_common() if v >= min_count]

