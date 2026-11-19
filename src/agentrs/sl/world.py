"""Unified world interface for the superlearner.

Every environment family (the original CAM simulators, Melting Pot substrates, external business simulators,
the economic crucible) is wrapped as a ``World`` exposing the same operations:

    initialize_world(task)  observe(agent)  available_actions(agent)  execute(agent, action)  advance_time()
    observe_consequences(agent)  evaluate_goal(agent)  save_world_state()  restore_world_state(s)  reset_world()

Three kinds of knowledge are kept apart:
* what the AGENT observes: ``observe`` (``Obs``) and ``available_actions`` (``ActionSpace``) only;
* what the SIMULATOR knows: the World's internal state (never returned by the agent-facing calls);
* what the VERIFIER knows: ``evaluate_goal`` returns verified outcome numbers (success, progress, violations,
  score) computed from simulator state. They feed the reward contract and evaluation, never the observation.

``Obs`` is deliberately generic (symbols, named numbers, an egocentric image, entity sets, optional text) so
different worlds need no shared hand-authored vocabulary: the neural core hashes symbols and field names into
learned embeddings. Actions are either a list of labelled discrete choices or a sequential grammar of labelled
slot choices; the policy scores candidate labels with a pointer head, so new worlds need no new output layer.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class Obs:
    tokens: list = field(default_factory=list)          # discrete symbols visible to the agent
    numbers: dict = field(default_factory=dict)         # named scalar readings visible to the agent
    image: Optional[np.ndarray] = None                  # egocentric image (H, W, C) uint8, agent's own view only
    entities: list = field(default_factory=list)        # [(type, [features...]), ...] visible entities
    text: Optional[str] = None                          # natural-language rendering (semantic-bootstrap track)


@dataclass
class ActionSpace:
    kind: str                                           # "discrete" | "sequential"
    labels: list = field(default_factory=list)          # discrete: one label per action index
    allowed: Optional[Callable] = None                  # sequential: allowed(prefix) -> labels ("END" terminates)
    max_len: int = 8


@dataclass
class Consequence:
    reward: float = 0.0                                 # the world's native reward for this agent (if any)
    dt: float = 1.0                                     # world time consumed (semi-Markov)
    done: bool = False                                  # episode over for this agent
    events: list = field(default_factory=list)          # observable event symbols


@dataclass
class GoalStatus:
    success: bool = False                               # verified goal achievement (only meaningful when done)
    progress: float = 0.0                               # verified progress in [0, 1]
    violated: Optional[str] = None                      # hard-constraint violation (unacceptable terminal state)
    score: float = 0.0                                  # verified outcome metric (net worth, collective return, utility)
    time_left: float = 1.0                              # fraction of the horizon left


@dataclass
class Task:
    family: str
    seed: int
    split: str = "train"                                # train | val | test
    params: dict = field(default_factory=dict)          # variant parameters (rules, opponents, resources)


class World:
    """Base class. Single-agent worlds use agent id "0"; multi-agent worlds list several ids, of which the
    learner may control some (``controlled``) while the others are driven by background policies."""

    family = "world"
    episodic = True

    def initialize_world(self, task: Task) -> None:
        raise NotImplementedError

    def agents(self) -> list:
        return ["0"]

    def observe(self, agent: str) -> Obs:
        raise NotImplementedError

    def goal(self, agent: str) -> Obs:
        return Obs()

    def available_actions(self, agent: str) -> ActionSpace:
        raise NotImplementedError

    def execute(self, agent: str, action) -> None:
        raise NotImplementedError

    def advance_time(self) -> None:
        """Simultaneous-move worlds step here once every agent has executed; turn-based worlds may no-op."""

    def observe_consequences(self, agent: str) -> Consequence:
        raise NotImplementedError

    def evaluate_goal(self, agent: str) -> GoalStatus:
        raise NotImplementedError

    def save_world_state(self):
        return copy.deepcopy(self.__dict__)

    def restore_world_state(self, state) -> None:
        self.__dict__.update(copy.deepcopy(state))

    def reset_world(self) -> None:
        """New episode of the SAME task (same hidden rules / counterparties / map seed): the unit of a trial."""
        raise NotImplementedError

    def close(self) -> None:
        pass


@dataclass
class RewardSpec:
    native: float = 1.0          # weight of the world's native reward (e.g. Melting Pot)
    success: float = 1.0         # terminal bonus for verified success
    shaping: float = 0.5         # potential-based shaping on verified progress (Phi(terminal) = 0)
    time_cost: float = 0.0       # per unit of world time
    violation: float = 1.0       # terminal cost of a hard-constraint violation
    gamma: float = 0.99


class RewardContract:
    """Reward = native + potential shaping on verified progress (policy-invariant; Phi = 0 at termination, so
    ending early or destroying resources never pays) + verified success bonus - time cost - violation cost.
    The verifier's progress/success come from ``evaluate_goal`` (simulator state), never from the agent."""

    def __init__(self, spec: Optional[RewardSpec] = None):
        self.s = spec or RewardSpec()

    def components(self, before: GoalStatus, after: GoalStatus, cons: Consequence, disc: float) -> dict:
        """``disc`` must be the discount the learner applies to this transition (the runner's semi-Markov discount
        on normalised elapsed time). Shaping is potential-based only under the learner's own discount: with any
        other discount it no longer telescopes. sl-exp-v1..v10 discounted shaping by raw elapsed hours (gamma**dt),
        which made holding verified progress cost (0.99 - 0.99**dt) * shaping * progress per step: ~0.15 x progress
        per bizdev step (dt ~ 38 h) and ~0.5 x progress per product step (dt ~ 400 h). Fixed in sl-exp-v11."""
        s = self.s
        phi1 = 0.0 if cons.done else after.progress
        out = {"native": s.native * cons.reward, "shaping": s.shaping * (disc * phi1 - before.progress),
               "time": -s.time_cost * cons.dt, "terminal": 0.0}
        if cons.done:
            out["terminal"] = s.success * float(after.success) - s.violation * float(after.violated is not None)
        return out

    def __call__(self, before: GoalStatus, after: GoalStatus, cons: Consequence, disc: float) -> float:
        return float(sum(self.components(before, after, cons, disc).values()))
