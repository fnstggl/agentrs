"""cam_live.reward.selfgoals — budgeted goal-free learning (spec 11.4): the model-independent machinery.

When no objective is assigned CAM may choose learning goals of its own, inside a fixed resource budget:

* candidates are scored from epistemic uncertainty, the predicted *achievable* learning progress and the expected
  relevance elsewhere, per unit of estimated cost — never from raw unpredictability (no achievable progress, no
  score), and with repeats and trivial goals discounted (``integrity.SelfGoalDiscount``, ``NoisyTVGate``);
* every self-chosen goal is marked ``Trust.SELF``: its outcome can never enter the outer objective
  (``integrity.screen_verified`` rejects it) — there is no self-generated incentive beyond the bounded,
  discounted bootstrap signal;
* the budget is finite by construction and nothing is selected once it is spent;
* whether autonomous learning was *useful* is judged only afterwards, on independently assigned objectives with
  verified outcomes: verified utility minus what had been predicted for that objective before the goal-free period.

The candidates themselves (what could be learned, and the numbers attached to them) come from the model; this
module holds no list of activities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..contracts import Cost, Trust
from .integrity import LedgerEntry, NoisyTVGate, SelfGoalDiscount

__all__ = ["GoalFreeBudget", "LearningGoalCandidate", "SelectedGoal", "SelfGoalSelector"]


@dataclass
class GoalFreeBudget:
    """A fixed budget for one goal-free period. Every limit must be finite."""
    seconds: float
    decisions: int
    money: float = 0.0
    compute_s: float = 0.0
    spent: Cost = field(default_factory=Cost)
    n_decisions: int = 0

    def __post_init__(self) -> None:
        for name in ("seconds", "decisions", "money", "compute_s"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"goal-free budget '{name}' must be finite and non-negative")

    def can_afford(self, cost: Cost) -> bool:
        return (self.n_decisions + 1 <= self.decisions and self.spent.seconds + cost.seconds <= self.seconds
                and self.spent.money + cost.money <= self.money
                and self.spent.compute_s + cost.compute_s <= self.compute_s)

    def charge(self, cost: Cost) -> None:
        """Record what an executed self-chosen activity actually consumed."""
        self.spent = Cost(seconds=self.spent.seconds + cost.seconds, money=self.spent.money + cost.money,
                          compute_s=self.spent.compute_s + cost.compute_s,
                          primitive_ops=self.spent.primitive_ops + cost.primitive_ops,
                          tokens=self.spent.tokens + cost.tokens)
        self.n_decisions += 1

    @property
    def exhausted(self) -> bool:
        return (self.n_decisions >= self.decisions or self.spent.seconds >= self.seconds
                or self.spent.money > self.money or self.spent.compute_s > self.compute_s)

    def remaining(self) -> Dict[str, float]:
        return {"seconds": self.seconds - self.spent.seconds, "decisions": self.decisions - self.n_decisions,
                "money": self.money - self.spent.money, "compute_s": self.compute_s - self.spent.compute_s}


@dataclass
class LearningGoalCandidate:
    key: str                         # stable id of what would be learned (context / query family)
    description: str                 # model-written text of the goal (data)
    epistemic: float                 # current epistemic uncertainty about it
    achievable_progress: float       # predicted learning progress if pursued (>= 0)
    p_success: float = 0.5           # predicted probability of achieving it
    transfer: float = 1.0            # expected relevance to other contexts, in [0, 1]
    est_cost: Cost = field(default_factory=Cost)


@dataclass
class SelectedGoal:
    candidate: LearningGoalCandidate
    score: float
    discount: float
    origin: str = Trust.SELF         # never an independently assigned objective


class SelfGoalSelector:
    def __init__(self, budget: GoalFreeBudget, discount: Optional[SelfGoalDiscount] = None,
                 gate: Optional[NoisyTVGate] = None, min_objectives: int = 5):
        self.budget = budget
        self.discount = discount or SelfGoalDiscount()
        self.gate = gate or NoisyTVGate()
        self.min_objectives = int(min_objectives)
        self.history: List[Dict[str, Any]] = []            # realised vs predicted progress of pursued goals
        self._later: List[Dict[str, Any]] = []             # later independent objectives

    # --------------------------------------------------------------------------------------- selection
    def score(self, c: LearningGoalCandidate) -> float:
        """uncertainty x achievable progress x transfer x (repeat / triviality discount) x noisy-TV gate, per
        unit of estimated time. Zero achievable progress means zero, whatever the uncertainty."""
        if c.epistemic <= 0 or c.achievable_progress <= 0:
            return 0.0
        w = self.discount.weight(c.description, c.p_success, register=False)
        base = c.epistemic * c.achievable_progress * min(max(c.transfer, 0.0), 1.0) * w * self.gate.gate(c.key)
        return base / (1.0 + max(c.est_cost.seconds, 0.0))

    def select(self, candidates: Sequence[LearningGoalCandidate], k: int = 1, *,
               objective_assigned: bool = False) -> List[SelectedGoal]:
        """Up to ``k`` affordable goals, best first. Empty when an objective is assigned (the mode is off), when
        the budget is spent, or when nothing has positive expected learning value. Selection registers the goal
        with the repeat discount; the caller charges the budget with the *actual* cost after execution."""
        if objective_assigned or self.budget.exhausted:
            return []
        scored = [(self.score(c), i, c) for i, c in enumerate(candidates) if self.budget.can_afford(c.est_cost)]
        scored = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))[:max(0, k)]
        out = []
        for s, _, c in scored:
            out.append(SelectedGoal(c, s, self.discount.weight(c.description, c.p_success, register=True)))
        return out

    def record_result(self, goal: SelectedGoal, actual_cost: Cost, *, epistemic_after: float,
                      realised_progress: float) -> None:
        """After executing a self-chosen activity: charge the budget, feed the noisy-TV gate and keep the pair
        (predicted, realised) progress so the progress estimator can be scored."""
        self.budget.charge(actual_cost)
        self.gate.update(goal.candidate.key, epistemic_after)
        self.history.append({"key": goal.candidate.key, "predicted_progress": goal.candidate.achievable_progress,
                             "realised_progress": float(realised_progress), "score": goal.score,
                             "discount": goal.discount, "seconds": actual_cost.seconds})

    def progress_estimator_error(self) -> Optional[float]:
        """Mean absolute error of predicted vs realised learning progress over pursued goals."""
        if not self.history:
            return None
        return sum(abs(h["predicted_progress"] - h["realised_progress"]) for h in self.history) / len(self.history)

    # --------------------------------------------------------------------------------------- usefulness
    def record_later_objective(self, entry: LedgerEntry, predicted_without: float) -> None:
        """An independently assigned objective evaluated *after* the goal-free period. ``entry`` must come from
        the ``OutcomeLedger`` (signed, de-duplicated); ``predicted_without`` is the evaluator's prediction of its
        utility registered before the goal-free period began."""
        if entry.kind != "objective":
            raise ValueError("usefulness is judged on objective outcomes admitted to the ledger")
        self._later.append({"objective_id": entry.objective_id, "key": entry.key, "utility": entry.utility,
                            "predicted_without": float(predicted_without)})

    def usefulness(self) -> Dict[str, Any]:
        """Mean verified gain over the earlier prediction on later independent objectives, with a standard error.
        ``established`` is False until ``min_objectives`` independent objectives have been evaluated."""
        seen, gains = set(), []
        for r in self._later:
            if r["key"] not in seen:
                seen.add(r["key"])
                gains.append(r["utility"] - r["predicted_without"])
        n = len(gains)
        if n == 0:
            return {"n": 0, "established": False, "mean_gain": None, "se": None}
        mean = sum(gains) / n
        se = math.sqrt(sum((g - mean) ** 2 for g in gains) / (n - 1) / n) if n > 1 else None
        return {"n": n, "established": n >= self.min_objectives, "mean_gain": mean, "se": se}

    def state_dict(self) -> Dict[str, Any]:
        return {"spent": self.budget.spent.to_dict(), "n_decisions": self.budget.n_decisions,
                "discount": self.discount.state_dict(), "gate": self.gate.state_dict(),
                "history": list(self.history), "later": list(self._later)}

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        self.budget.spent, self.budget.n_decisions = Cost(**d["spent"]), int(d["n_decisions"])
        self.discount.load_state_dict(d["discount"])
        self.gate.load_state_dict(d["gate"])
        self.history, self._later = list(d["history"]), list(d["later"])
