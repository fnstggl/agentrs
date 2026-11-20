"""Autocurriculum: choose *situations* (goal family x difficulty x world family x seed), never strategies.

`GoalSampler` keeps, per arm (goal family, level, world family), a Beta posterior over success and a history of
progress outcomes. Its score combines
  * learning progress: |mean progress of the recent window - mean of the window before| (absolute LP),
  * uncertainty: posterior std of the success probability,
  * a novelty bonus that decays with visits,
and it filters arms whose estimated success is outside a useful band (too easy / hopeless), except that a fixed
fraction `p_frontier` of samples always goes to long-horizon hard arms so ambitious goals stay in the distribution.
All generated goals are validated (`GoalContract.validate`) and checked not to be satisfied at their start.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .contracts import GOAL_FAMILIES, GoalContract, generate_goal


@dataclass
class Arm:
    goal_family: str
    level: int
    world_family: str
    a: float = 1.0
    b: float = 1.0
    progress: list = field(default_factory=list)
    visits: int = 0

    @property
    def key(self):
        return (self.goal_family, self.level, self.world_family)

    def p_success(self):
        return self.a / (self.a + self.b)

    def uncertainty(self):
        a, b = self.a, self.b
        return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    def learning_progress(self, w=5):
        p = self.progress
        if len(p) < 2 * w:
            return 0.5 if len(p) < 2 else abs(np.mean(p[-(len(p) // 2):]) - np.mean(p[:len(p) // 2]))
        return abs(float(np.mean(p[-w:])) - float(np.mean(p[-2 * w:-w])))


class GoalSampler:
    def __init__(self, world_families: list[str], goal_families: list[str] | None = None, levels=(0, 1, 2, 3),
                 seed: int = 0, band=(0.05, 0.95), p_frontier: float = 0.1, c_unc: float = 0.5, c_nov: float = 0.3,
                 split: str = "train"):
        self.rng = np.random.default_rng(seed)
        self.arms = {(g, l, w): Arm(g, l, w) for g in (goal_families or sorted(GOAL_FAMILIES))
                     for l in levels for w in world_families}
        self.band = band
        self.p_frontier = p_frontier
        self.c_unc = c_unc
        self.c_nov = c_nov
        self.split = split
        self.n = 0
        self.log: list[dict] = []

    def score(self, arm: Arm) -> float:
        return arm.learning_progress() + self.c_unc * arm.uncertainty() + self.c_nov / math.sqrt(1 + arm.visits)

    def sample(self) -> GoalContract:
        self.n += 1
        arms = list(self.arms.values())
        if self.rng.random() < self.p_frontier:
            hard = [a for a in arms if a.level == max(x.level for x in arms)]
            arm = hard[int(self.rng.integers(0, len(hard)))]
            why = "frontier"
        else:
            ok = [a for a in arms if self.band[0] <= a.p_success() <= self.band[1]] or arms
            s = np.array([self.score(a) for a in ok])
            p = np.exp((s - s.max()) / 0.1)
            arm = ok[int(self.rng.choice(len(ok), p=p / p.sum()))]
            why = "score"
        arm.visits += 1
        seed = int(self.rng.integers(0, 2**31 - 1))
        g = generate_goal(arm.goal_family, arm.world_family, seed, arm.level, self.split)
        errs = g.validate()
        if errs:
            raise ValueError(f"generated goal invalid: {errs}")
        self.log.append({"n": self.n, "arm": arm.key, "why": why, "goal": g.id})
        return g

    def update(self, goal: GoalContract, success: bool, progress: float) -> None:
        arm = self.arms[(goal.family, int(goal.params.get("level", 0)), goal.world["family"])]
        arm.a += float(success)
        arm.b += float(not success)
        arm.progress.append(float(progress))

    def table(self) -> list[dict]:
        return [{"arm": k, "p_success": round(a.p_success(), 3), "lp": round(a.learning_progress(), 3),
                 "uncertainty": round(a.uncertainty(), 3), "visits": a.visits} for k, a in sorted(self.arms.items())]
