"""Common interface of CAM environments.

Every environment is an executable simulator with hidden parameters that the agent never sees:
the agent receives only ``goal_tokens`` / ``obs_tokens`` / ``features`` (observable), acts with a
token sequence that must follow the environment's action grammar (tool + complete arguments), and
the environment — not any model — computes the consequences, events, verified progress and the final
verified success. ``split``: ``train``/``val`` draw hidden parameters from the training distribution
(val uses disjoint seeds); ``test`` draws them from held-out distributions (unseen counterparty
policies, unseen preference / market structures).
"""
from __future__ import annotations

import copy
import random
import zlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

NBINS = 21  # numeric tokens N0..N20


def nb(x: float, lo: float = 0.0, hi: float = 1.0) -> str:
    """Bin a number into one of the shared numeric tokens."""
    t = 0.0 if hi == lo else (x - lo) / (hi - lo)
    return f"N{int(round(min(1.0, max(0.0, t)) * (NBINS - 1)))}"


def logb(x: float, hi: float) -> str:
    return nb(np.log1p(max(0.0, x)) / np.log1p(hi))


@dataclass
class StepResult:
    elapsed_h: float                       # real (simulated) time the action took until its response
    events: dict = field(default_factory=dict)
    done: bool = False
    success: Optional[bool] = None         # set by the environment's verifier when done
    valid: bool = True
    info: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Contract:
    """Externally specified, environment-verified task contract (independent of any model).

    ``clock``/``horizon``: the unit the deadline is measured in (weeks, rounds, steps) and the deadline;
    every action reports how much of that clock it consumed (semi-Markov time). ``resources``: verified
    resources reported as fractions of their initial amount / limit. ``constraints``: hard constraints;
    violating one is an *unacceptable terminal state* (distinct from failing to reach the goal in time).
    ``milestones``: verified intermediate achievements — reported and learned from, never counted as
    success. ``failure_causes``: the enumerated ways an episode can end without success."""
    clock: str
    horizon: float
    resources: tuple = ()
    constraints: tuple = ()
    milestones: tuple = ()
    failure_causes: tuple = ("deadline",)


@dataclass
class Status:
    """Verified status at one point in time (computed by the environment, observable by the agent; ``env.task_status()``)."""
    progress: float
    clock: float
    time_left: float
    resources: dict = field(default_factory=dict)
    milestones: dict = field(default_factory=dict)
    violated: Optional[str] = None
    costs: dict = field(default_factory=dict)


N_STATUS = 8  # generic status vector: progress, time left, 3 resources, milestone fraction, #constraints, #resources


def status_vector(c: Contract, s: Status) -> np.ndarray:
    v = np.zeros(N_STATUS, dtype=np.float32)
    v[0], v[1] = s.progress, s.time_left
    for i, name in enumerate(c.resources[:3]):
        v[2 + i] = min(2.0, max(0.0, s.resources.get(name, 0.0)))
    v[5] = (sum(bool(x) for x in s.milestones.values()) / len(c.milestones)) if c.milestones else 0.0
    v[6] = len(c.constraints) / 2.0
    v[7] = len(c.resources) / 3.0
    return v


class Grammar:
    """Slot grammar: an action is TYPE followed by one value per slot of that type, then END.
    ``allowed(prefix)`` returns the tokens that may come next (used as the decoding mask, both when
    sampling and when re-scoring for training, so the recorded probabilities are exact)."""

    def __init__(self, spec: dict[str, list[list[str]]], name: str = ""):
        self.spec = spec
        self.name = name

    def allowed(self, prefix: list[str]) -> list[str]:
        if not prefix:
            return list(self.spec)
        slots = self.spec[prefix[0]]
        k = len(prefix) - 1
        return slots[k] if k < len(slots) else ["END"]

    def complete(self, prefix: list[str]) -> bool:
        return bool(prefix) and prefix[-1] == "END"

    def tokens(self) -> set[str]:
        out = set(self.spec) | {"END"}
        for slots in self.spec.values():
            for s in slots:
                out |= set(s)
        return out


class CamEnv:
    family = "base"
    keywords: list[str] = []
    events: list[str] = []
    n_features = 0
    max_steps = 20
    # Training-only goal difficulty in (0, 1] set by the curriculum before ``reset``; environments that
    # support it scale their goal on the ``train`` split only, so validation / test goals never change.
    difficulty = 1.0

    def __init__(self):
        self.rng = random.Random(0)
        self.np = np.random.default_rng(0)
        self.split = "train"
        self.t = 0
        self.done = False
        self.success: Optional[bool] = None

    # ---- to implement
    def reset(self, seed: int, split: str = "train") -> None:
        raise NotImplementedError

    def goal_tokens(self) -> list[str]:
        raise NotImplementedError

    def obs_tokens(self) -> list[str]:
        raise NotImplementedError

    def features(self) -> np.ndarray:
        raise NotImplementedError

    def grammar(self) -> Grammar:
        raise NotImplementedError

    def step(self, action: list[str]) -> StepResult:
        raise NotImplementedError

    def progress(self) -> float:
        """Verified, environment-measured progress toward the goal in [0, 1] (not a model output)."""
        raise NotImplementedError

    def hidden(self) -> dict:
        """Hidden parameters (for analysis only; never given to the agent)."""
        return {}

    # ---- task contract (generic interface used by the reward contract, audits, archive, hindsight)
    _shared_attrs: tuple = ()  # large read-only arrays shared (not copied) by snapshots

    def contract(self) -> Contract:
        return Contract(clock="step", horizon=float(self.max_steps))

    def task_status(self) -> Status:
        p = self.progress()
        return Status(progress=p, clock=float(self.t), time_left=max(0.0, 1 - self.t / self.max_steps))

    def failure_cause(self) -> Optional[str]:
        """Why a finished episode did not succeed (one of ``contract().failure_causes``)."""
        if not self.done or self.success:
            return None
        return "deadline"

    def hindsight_goals(self) -> list:
        """Goals of the *same schema* that this finished episode verifiably achieved although the given
        goal was not: [(goal tokens, number of agent steps until achieved)]. Default: none."""
        return []

    def retarget(self, difficulty: float) -> None:
        """Change the (training-only) goal difficulty of a restored state. Default: no-op."""

    def snapshot(self) -> "CamEnv":
        memo = {id(getattr(self, a)): getattr(self, a) for a in self._shared_attrs if hasattr(self, a)}
        return copy.deepcopy(self, memo)

    def reseed(self, seed: int) -> None:
        """Fresh exogenous randomness for a restored state (hidden parameters unchanged), so exploring
        from an archived state cannot replay the same future noise."""
        self.rng = random.Random(f"{self.family}:fork:{seed}")
        self.np = np.random.default_rng(zlib.crc32(f"{self.family}:fork:{seed}".encode()))

    def _seed(self, seed: int, split: str) -> None:
        self.rng = random.Random(f"{self.family}:{split}:{seed}")
        self.np = np.random.default_rng(zlib.crc32(f"{self.family}:{split}:{seed}".encode()))
        self.split = split
        self.t = 0
        self.done = False
        self.success = None
