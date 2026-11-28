"""cam_live.reward.selfgoals — budgeted goal-free learning (spec 11.4).

``SelfGoalSelector`` is the model-independent machinery; :class:`GoalFreeMode` wires it onto the real model and
its Bayesian fast layer (proposal from epistemic uncertainty and achievable progress, achievement decided by
observation, fixed budget, usefulness judged on later independently assigned objectives).

When no objective is assigned CAM may choose learning goals of its own, inside a fixed resource budget:

* candidates are scored from epistemic uncertainty, the predicted *achievable* learning progress and the expected
  relevance elsewhere, per unit of estimated cost — never from raw unpredictability (no achievable progress, no
  score), and with repeats and trivial goals discounted (``integrity.SelfGoalDiscount``, ``NoisyTVGate``);
* every self-chosen goal is marked ``Trust.SELF``: its outcome can never enter the outer objective (the
  ``integrity.OutcomeLedger`` admits only objectives whose recorded origin is the operator or the verifier; a
  ``TaskSpec`` with ``origin=Trust.SELF`` or one that was never registered is refused) — there is no
  self-generated incentive beyond the bounded, discounted bootstrap signal;
* the budget is finite by construction and nothing is selected once it is spent; a selected goal *reserves* one
  decision and its estimated cost until its result is recorded, so goals that are still pending count against the
  budget (no number of ``select`` calls can commit more than the budget);
* whether autonomous learning was *useful* is judged only afterwards, on independently assigned objectives with
  verified outcomes: verified utility minus what had been predicted for that objective before the goal-free period.

The candidates themselves (what could be learned, and the numbers attached to them) come from the model; this
module holds no list of activities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from typing import Tuple

from ..contracts import ActionSpec, Clock, Cost, GoalStatus, OutcomeKind, OutcomeQuery, Trust, new_id
from .integrity import LedgerEntry, NoisyTVGate, SelfGoalDiscount

__all__ = ["GoalFreeBudget", "LearningGoalCandidate", "SelectedGoal", "SelfGoalSelector", "SelfGoal", "GoalFreeMode",
           "TransferEstimator"]


@dataclass
class GoalFreeBudget:
    """A fixed budget for one goal-free period. Every limit must be finite.

    ``spent`` / ``n_decisions``: what executed self-chosen activities actually consumed. ``reserved`` /
    ``n_reserved``: what selected but not yet resolved goals hold (their estimated cost, one decision each).
    :meth:`can_afford` counts both, so pending goals can never push the committed total over a limit."""
    seconds: float
    decisions: int
    money: float = 0.0
    compute_s: float = 0.0
    spent: Cost = field(default_factory=Cost)
    n_decisions: int = 0
    reserved: Cost = field(default_factory=Cost)
    n_reserved: int = 0

    def __post_init__(self) -> None:
        for name in ("seconds", "decisions", "money", "compute_s"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"goal-free budget '{name}' must be finite and non-negative")

    def can_afford(self, cost: Cost) -> bool:
        """Would one more activity of estimated ``cost`` fit, counting what pending goals have reserved?"""
        s, r = self.spent, self.reserved
        return (self.n_decisions + self.n_reserved + 1 <= self.decisions
                and s.seconds + r.seconds + cost.seconds <= self.seconds
                and s.money + r.money + cost.money <= self.money
                and s.compute_s + r.compute_s + cost.compute_s <= self.compute_s)

    @staticmethod
    def _add(a: Cost, b: Cost, sign: float = 1.0) -> Cost:
        return Cost(seconds=max(0.0, a.seconds + sign * b.seconds), money=max(0.0, a.money + sign * b.money),
                    compute_s=max(0.0, a.compute_s + sign * b.compute_s),
                    primitive_ops=max(0, a.primitive_ops + int(sign) * b.primitive_ops),
                    tokens=max(0, a.tokens + int(sign) * b.tokens))

    def reserve(self, cost: Cost) -> bool:
        """Hold one decision and ``cost`` for a selected goal; False (nothing held) when it does not fit."""
        if not self.can_afford(cost):
            return False
        self.reserved = self._add(self.reserved, cost)
        self.n_reserved += 1
        return True

    def release(self, cost: Cost) -> None:
        """Give back a reservation (the goal was resolved or abandoned)."""
        self.reserved = self._add(self.reserved, cost, -1.0)
        self.n_reserved = max(0, self.n_reserved - 1)

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

    @property
    def committed(self) -> bool:
        """No further goal can be selected now: spent plus reserved leaves no room for one more decision."""
        return self.exhausted or self.n_decisions + self.n_reserved >= self.decisions

    def remaining(self) -> Dict[str, float]:
        return {"seconds": self.seconds - self.spent.seconds, "decisions": self.decisions - self.n_decisions,
                "money": self.money - self.spent.money, "compute_s": self.compute_s - self.spent.compute_s,
                "reserved_decisions": self.n_reserved, "reserved_seconds": self.reserved.seconds,
                "uncommitted_decisions": self.decisions - self.n_decisions - self.n_reserved}


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
    reserved: Optional[Cost] = None  # budget held for this goal until its result is recorded (None: released)


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
        the budget is spent or fully reserved by pending goals, or when nothing has positive expected learning
        value. Each selected goal reserves one decision and its estimated cost (checked one after the other, so a
        single call never exceeds the budget either); :meth:`record_result` releases the reservation and charges
        the *actual* cost, :meth:`abandon` releases it for a goal that was not executed. Selection registers the
        goal with the repeat discount."""
        if objective_assigned or self.budget.committed:
            return []
        scored = [(self.score(c), i, c) for i, c in enumerate(candidates) if self.budget.can_afford(c.est_cost)]
        scored = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))
        out = []
        for s, _, c in scored:
            if len(out) >= max(0, k):
                break
            if not self.budget.reserve(c.est_cost):
                continue
            out.append(SelectedGoal(c, s, self.discount.weight(c.description, c.p_success, register=True),
                                    reserved=c.est_cost))
        return out

    def abandon(self, goal: SelectedGoal) -> None:
        """A selected goal will not be executed: give its reservation back (nothing is charged)."""
        if goal.reserved is not None:
            self.budget.release(goal.reserved)
            goal.reserved = None

    def record_result(self, goal: SelectedGoal, actual_cost: Cost, *, epistemic_after: float,
                      realised_progress: float) -> None:
        """After executing a self-chosen activity: charge the budget, feed the noisy-TV gate and keep the pair
        (predicted, realised) progress so the progress estimator can be scored."""
        if goal.reserved is not None:
            self.budget.release(goal.reserved)
            goal.reserved = None
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
                "reserved": self.budget.reserved.to_dict(), "n_reserved": self.budget.n_reserved,
                "discount": self.discount.state_dict(), "gate": self.gate.state_dict(),
                "history": list(self.history), "later": list(self._later)}

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        """Reservations of goals that were pending at the checkpoint stay held (fail closed): resolve them, or
        :meth:`abandon` the restored goals, to give the budget back."""
        self.budget.spent, self.budget.n_decisions = Cost(**d["spent"]), int(d["n_decisions"])
        self.budget.reserved = Cost(**d["reserved"]) if "reserved" in d else Cost()
        self.budget.n_reserved = int(d.get("n_reserved", 0))
        self.discount.load_state_dict(d["discount"])
        self.gate.load_state_dict(d["gate"])
        self.history, self._later = list(d["history"]), list(d["later"])


# ================================================================================================ transfer (learned)
class TransferEstimator:
    """Expected transferable competence of a learning goal (spec 11.4: "generate learning goals based on ...
    expected transferable competence"), learned from what goal-free learning was later worth — replacing the constant
    1.0 that every candidate had before (AE-04 partial, KF-E21).

    Evidence: every outcome observed during a later *independently assigned* objective is scored twice on the same
    data — by the goal-free posterior (the fast layer that absorbed the goal-free observations) and by the reference
    prior that posterior started from; ``gain = NLL_reference - NLL_goal_free`` (nats) is how much the knowledge
    acquired without an objective improved the prediction of what happened under the objective. A soft success
    ``u = (1 + clip(gain / scale, -1, 1)) / 2`` enters a Beta posterior.

    Estimate for a candidate (key = operation|metric, features phi): Beta(prior_strength * prior + sum w_i u_i,
    prior_strength * (1 - prior) + sum w_i (1 - u_i)) over the evidence points i, with similarity weights
    w_i = max(exp((cos(phi, phi_i) - 1) / tau), 1 for the same key, 1/2 for the same operation). Without evidence
    the estimate is the prior (1/2: uninformative), and it moves only with measured later gains."""

    def __init__(self, *, prior: float = 0.5, prior_strength: float = 2.0, tau: float = 0.1, scale: float = 0.25,
                 max_points: int = 512, min_abs_gain: float = 1e-4):
        self.prior, self.prior_strength, self.tau, self.scale = float(prior), float(prior_strength), float(tau), float(scale)
        self.max_points, self.min_abs_gain = int(max_points), float(min_abs_gain)
        self.points: List[Dict[str, Any]] = []            # {key, phi (unit list), gain, objective_id}

    @staticmethod
    def _unit(phi: Any) -> List[float]:
        v = [float(x) for x in (phi.detach().cpu().double().reshape(-1).tolist() if hasattr(phi, "detach")
                                else list(phi))]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def observe(self, key: str, phi: Any, gain: float, objective_id: str = "") -> bool:
        """One later outcome's measured gain; |gain| below ``min_abs_gain`` says nothing (False, not kept)."""
        g = float(gain)
        if not math.isfinite(g) or abs(g) < self.min_abs_gain:
            return False
        self.points.append({"key": str(key), "phi": self._unit(phi), "gain": g, "objective_id": str(objective_id)})
        del self.points[:-self.max_points]
        return True

    def _weight(self, key: str, u: List[float], p: Dict[str, Any]) -> float:
        cos = sum(a * b for a, b in zip(u, p["phi"])) if len(u) == len(p["phi"]) else -1.0
        w = math.exp((min(cos, 1.0) - 1.0) / max(self.tau, 1e-6))
        if p["key"] == key:
            w = max(w, 1.0)
        elif p["key"].split("|")[0] == key.split("|")[0]:
            w = max(w, 0.5)
        return w

    def estimate(self, key: str, phi: Any = None) -> Dict[str, float]:
        u = self._unit(phi) if phi is not None else []
        a = self.prior_strength * self.prior
        b = self.prior_strength * (1.0 - self.prior)
        n_eff, gain = 0.0, 0.0
        for p in self.points:
            w = self._weight(key, u, p) if u else (1.0 if p["key"] == key else 0.0)
            if w <= 1e-6:
                continue
            s = 0.5 * (1.0 + max(-1.0, min(1.0, p["gain"] / self.scale)))
            a, b, n_eff, gain = a + w * s, b + w * (1.0 - s), n_eff + w, gain + w * p["gain"]
        return {"mean": a / (a + b), "n_eff": n_eff, "mean_gain": (gain / n_eff) if n_eff else None,
                "sd": math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1.0)))}

    def by_key(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for p in self.points:
            d = out.setdefault(p["key"], {"n": 0, "gain_sum": 0.0, "positive": 0})
            d["n"] += 1
            d["gain_sum"] += p["gain"]
            d["positive"] += int(p["gain"] > 0)
        for k, d in out.items():
            d["mean_gain"] = d["gain_sum"] / d["n"]
            d["estimate"] = self.estimate(k)["mean"]
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {"prior": self.prior, "prior_strength": self.prior_strength, "tau": self.tau, "scale": self.scale,
                "points": [dict(p) for p in self.points]}

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        self.points = [dict(p) for p in d.get("points", [])][-self.max_points:]


# ================================================================================================ on the real model
@dataclass
class SelfGoal:
    """One self-chosen learning goal: "predict ``query`` under ``action`` better than now". Never an objective:
    ``origin`` is ``Trust.SELF`` and its outcome cannot enter the outer objective of the reward."""
    goal_id: str
    key: str
    description: str
    action: ActionSpec
    query: OutcomeQuery
    phi: Any                           # [F] features of (action, query) under the belief at proposal time
    epistemic_before: float            # epistemic std of the prediction at proposal
    predicted_progress: float          # calibrated achievable progress (nats)
    required_progress: float           # progress the observation must show for the goal to count as achieved
    score: float
    discount: float
    created_at: float = 0.0
    status: str = GoalStatus.PENDING
    realised_progress: Optional[float] = None
    epistemic_after: Optional[float] = None
    origin: str = Trust.SELF
    selected: Optional[SelectedGoal] = None


class GoalFreeMode:
    """Autonomous learning when no objective is assigned (spec 11.4), on ``CamLiveModel`` + ``FastBeliefState``.

    propose   for candidate (action, outcome query) pairs supplied by the candidate generators: epistemic
              uncertainty from the fast posterior, achievable progress = expected information gain of one
              observation, 1/2 log(1 + epistemic_var / noise), scaled by how much of the progress predicted for
              that context was actually realised before (so unpredictable contexts stop looking learnable), with
              the repeat / triviality discount, the noisy-TV gate and the fixed budget of ``SelfGoalSelector``
    resolve   the *observed* outcome is absorbed by the fast layer; realised progress = improvement of the
              predictive NLL on the observations of that context (earlier ones and the new one), the same targets
              before and after; the goal is ACHIEVED only if that reaches ``required_progress``. No observation,
              no achievement. The budget is charged with the actual cost either way.
    Nothing here produces a reward for the outer objective, and the mode is off whenever an objective is assigned.
    """

    def __init__(self, model: Any, fast: Any, budget: GoalFreeBudget, *, selector: Optional[SelfGoalSelector] = None,
                 achieve_fraction: float = 0.25, calibration_rate: float = 0.3, replay_per_key: int = 8,
                 min_progress: float = 0.02, clock: Optional[Clock] = None):
        self.model, self.fast = model, fast
        self.selector = selector or SelfGoalSelector(budget)
        self.budget = self.selector.budget
        self.achieve_fraction, self.calibration_rate = float(achieve_fraction), float(calibration_rate)
        self.replay_per_key = int(replay_per_key)
        self.min_progress = float(min_progress)                # nats: below this a context is not worth a decision
        self.clock = clock
        self.calibration: Dict[str, float] = {}                # key -> realised / predicted progress (ratio of EMAs)
        self._cal: Dict[str, Tuple[float, float]] = {}         # key -> (EMA realised, EMA predicted)
        self.achieved: Dict[str, Tuple[int, int]] = {}         # key -> (achieved, resolved)
        self._replay: Dict[str, List[Tuple[Any, float, float]]] = {}   # key -> [(phi, y, exposure)]
        self.goals: List[SelfGoal] = []
        #: learned expected transfer (SG): evidence from later independently assigned objectives, scored against the
        #: prior the goal-free posterior started from (kept as ``reference``)
        self.transfer = TransferEstimator()
        self.reference = fast.clone() if hasattr(fast, "clone") else None
        self.transfer_log: List[Dict[str, Any]] = []
        self.restore_notes: List[str] = []

    # ---------------------------------------------------------------------------------------- proposal
    @staticmethod
    def key_of(action: ActionSpec, query: OutcomeQuery) -> str:
        return f"{action.operation}|{query.metric}"

    def active(self, objective_assigned: bool) -> bool:
        return not objective_assigned and not self.budget.exhausted

    def p_success(self, key: str) -> float:
        a, n = self.achieved.get(key, (0, 0))
        return (a + 1.0) / (n + 2.0)

    def _features(self, b: Any, pairs: Sequence[Tuple[ActionSpec, OutcomeQuery]], goal_text: str = "") -> Any:
        import torch
        with torch.no_grad():
            e_a, _ = self.model.encode_actions([[a for a, _ in pairs]])
            e_g = self.model.encode_goals([goal_text])
            e_q, _, _, _ = self.model.encode_queries([[q for _, q in pairs]])
            phi = self.model.features(self.model.decision_latent(b, e_a, e_g), e_q)[0]       # [N, N, F]
        idx = torch.arange(len(pairs), device=phi.device)
        return phi[idx, idx]

    def candidates(self, b: Any, pairs: Sequence[Tuple[ActionSpec, OutcomeQuery]]
                   ) -> List[Tuple[LearningGoalCandidate, Any, float]]:
        """(candidate, phi, raw predicted progress) for every pair whose outcome kind the fast layer supports."""
        if not pairs:
            return []
        phi = self._features(b, pairs)
        out = []
        for i, (a, q) in enumerate(pairs):
            if not self.fast.supports(q.kind):
                continue
            ex = q.exposure if q.kind != OutcomeKind.REAL else None
            _, v, noise = self.fast.predict(phi[i], q.kind, ex)
            v, noise = max(float(v), 0.0), max(float(noise), 1e-12)
            raw = 0.5 * math.log1p(v / noise)
            key = self.key_of(a, q)
            achievable = raw * self.calibration.get(key, 1.0)
            cand = LearningGoalCandidate(
                key=key, description=f"predict {q.metric} after {a.operation}", epistemic=math.sqrt(v),
                achievable_progress=achievable if achievable >= self.min_progress else 0.0,
                p_success=self.p_success(key), transfer=self.transfer.estimate(key, phi[i])["mean"],
                est_cost=a.est_cost)
            out.append((cand, phi[i].detach().cpu(), raw))
        return out

    def propose(self, b: Any, pairs: Sequence[Tuple[ActionSpec, OutcomeQuery]], *, k: int = 1,
                objective_assigned: bool = False) -> List[SelfGoal]:
        """Up to ``k`` self-chosen learning goals for belief ``b`` [1, D]; empty when an objective is assigned,
        the budget is spent or nothing has positive expected learning value."""
        if not self.active(objective_assigned):
            return []
        cands = self.candidates(b, pairs)
        by = {id(c): (i, phi, raw) for i, (c, phi, raw) in enumerate(cands)}
        supported = [(a, q) for a, q in pairs if self.fast.supports(q.kind)]
        picked = self.selector.select([c for c, _, _ in cands], k)
        now = float(self.clock.now()) if self.clock is not None else 0.0
        out = []
        for sel in picked:
            i, phi, raw = by[id(sel.candidate)]
            a, q = supported[i]
            g = SelfGoal(goal_id=new_id("sgoal"), key=sel.candidate.key, description=sel.candidate.description,
                         action=a, query=q, phi=phi, epistemic_before=sel.candidate.epistemic,
                         predicted_progress=sel.candidate.achievable_progress,
                         required_progress=self.achieve_fraction * raw, score=sel.score, discount=sel.discount,
                         created_at=now, selected=sel)
            self.goals.append(g)
            out.append(g)
        return out

    # ---------------------------------------------------------------------------------------- resolution
    def abandon(self, goal: SelfGoal) -> SelfGoal:
        """A proposed goal that will not be executed (another candidate was chosen, or it was refused before
        execution): its budget reservation is given back, nothing is charged, status becomes ``"abandoned"``.
        Proposed goals that are neither resolved nor abandoned keep their reservation (the budget fails closed)."""
        if goal.status != GoalStatus.PENDING:
            raise ValueError("only a pending goal can be abandoned")
        if goal.selected is not None:
            self.selector.abandon(goal.selected)
        goal.status = "abandoned"
        return goal

    def abandon_pending(self) -> int:
        """Abandon every pending goal of this process and release reservations no live goal holds (e.g. of goals
        that were pending when the checkpoint was written: after ``load_state_dict`` they are unknown here).
        Returns the number of reservations released."""
        n = 0
        for g in self.goals:
            if g.status == GoalStatus.PENDING:
                self.abandon(g)
                n += 1
        held = sum(1 for g in self.goals if g.status == GoalStatus.PENDING and g.selected is not None
                   and g.selected.reserved is not None)
        if held == 0 and self.budget.n_reserved:
            n += self.budget.n_reserved
            self.budget.reserved, self.budget.n_reserved = Cost(), 0
        return n

    def resolve(self, goal: SelfGoal, value: Optional[float], cost: Cost, *, censored: Optional[bool] = None
                ) -> SelfGoal:
        """Close a goal with what was actually observed (``value=None``: nothing was observed)."""
        if goal.status != GoalStatus.PENDING:
            raise ValueError("this goal was already resolved")
        q = goal.query
        ex = q.exposure if q.kind != OutcomeKind.REAL else None
        raw = goal.required_progress / max(self.achieve_fraction, 1e-12)
        if value is None or not math.isfinite(float(value)):
            realised, eps_after = 0.0, goal.epistemic_before
        else:
            targets = self._replay.get(goal.key, []) + [(goal.phi, float(value), q.exposure)]
            nll = lambda: [float(self.fast.nll(p, y, q.kind, e if q.kind != OutcomeKind.REAL else None,
                                               censored if p is goal.phi else None)) for p, y, e in targets]
            before = nll()
            self.fast.update(goal.phi, float(value), q.kind, exposure=ex, censored=censored)
            after = nll()
            realised = sum(b - a for b, a in zip(before, after)) / len(targets)
            eps_after = math.sqrt(max(float(self.fast.predict(goal.phi, q.kind, ex)[1]), 0.0))
            self._replay[goal.key] = targets[-self.replay_per_key:]
        goal.realised_progress, goal.epistemic_after = realised, eps_after
        goal.status = GoalStatus.ACHIEVED if (value is not None and realised >= max(goal.required_progress,
                                                                                    self.min_progress)) \
            else GoalStatus.FAILED
        a, n = self.achieved.get(goal.key, (0, 0))
        self.achieved[goal.key] = (a + int(goal.status == GoalStatus.ACHIEVED), n + 1)
        r = self.calibration_rate
        num, den = self._cal.get(goal.key, (raw, raw))
        num, den = (1.0 - r) * num + r * min(realised, 2.0 * raw), (1.0 - r) * den + r * raw
        self._cal[goal.key] = (num, den)
        self.calibration[goal.key] = min(2.0, max(0.0, num / den)) if den > 0 else 0.0
        self.selector.record_result(goal.selected, cost, epistemic_after=eps_after, realised_progress=realised)
        return goal

    # ---------------------------------------------------------------------------------------- transfer
    def observe_assigned(self, key: str, phi: Any, kind: str, value: float, exposure: float = 1.0, *,
                         censored: Optional[bool] = None, objective_id: str = "") -> Optional[float]:
        """An outcome observed under a later independently assigned objective (never during the goal-free period):
        how much better the goal-free posterior predicts it than the prior it started from (nats; None when nothing
        can be said: no goal-free learning yet, a kind the fast layer does not model, no reference)."""
        if self.reference is None or not self.fast.supports(kind) or not any(
                g.status in (GoalStatus.ACHIEVED, GoalStatus.FAILED) for g in self.goals) and not self.selector.history:
            return None
        if value is None or not math.isfinite(float(value)):
            return None
        ex = exposure if kind != OutcomeKind.REAL else None
        try:
            gain = float(self.reference.nll(phi, float(value), kind, ex, censored)) - \
                float(self.fast.nll(phi, float(value), kind, ex, censored))
        except Exception:  # noqa: BLE001 — a malformed observation teaches nothing about transfer
            return None
        kept = self.transfer.observe(key, phi, gain, objective_id)
        self.transfer_log = (self.transfer_log + [{"key": key, "gain": gain, "kept": kept,
                                                   "objective_id": objective_id}])[-256:]
        return gain

    # ---------------------------------------------------------------------------------------- usefulness
    def record_later_objective(self, entry: LedgerEntry, predicted_without: float) -> None:
        self.selector.record_later_objective(entry, predicted_without)

    def usefulness(self) -> Dict[str, Any]:
        return self.selector.usefulness()

    def report(self) -> Dict[str, Any]:
        done = [g for g in self.goals if g.status in (GoalStatus.ACHIEVED, GoalStatus.FAILED)]
        by_key: Dict[str, Dict[str, float]] = {}
        for g in done:
            d = by_key.setdefault(g.key, {"n": 0, "achieved": 0, "realised": 0.0})
            d["n"] += 1
            d["achieved"] += int(g.status == GoalStatus.ACHIEVED)
            d["realised"] += float(g.realised_progress or 0.0)
        return {"goals": len(self.goals), "resolved": len(done),
                "abandoned": sum(g.status == "abandoned" for g in self.goals),
                "pending": sum(g.status == GoalStatus.PENDING for g in self.goals),
                "achieved": sum(g.status == GoalStatus.ACHIEVED for g in done), "by_key": by_key,
                "budget_remaining": self.budget.remaining(), "budget_exhausted": self.budget.exhausted,
                "calibration": dict(self.calibration), "usefulness": self.usefulness(),
                "progress_estimator_error": self.selector.progress_estimator_error(),
                "transfer": {"points": len(self.transfer.points), "by_key": self.transfer.by_key(),
                             "restore_notes": list(self.restore_notes)}}

    def state_dict(self) -> Dict[str, Any]:
        return {"selector": self.selector.state_dict(), "calibration": dict(self.calibration),
                "cal": {k: list(v) for k, v in self._cal.items()},
                "achieved": {k: list(v) for k, v in self.achieved.items()},
                "transfer": self.transfer.state_dict(),
                "reference": self.reference.state_dict() if self.reference is not None else None}

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        """Checkpoints written before SG have no transfer evidence and no reference prior: the estimator starts
        empty (every candidate at the uninformative prior) and the reference is the fast state as restored, so gains
        are measured only for goal-free learning after the restore (recorded in ``restore_notes``)."""
        self.selector.load_state_dict(d["selector"])
        self.calibration = dict(d["calibration"])
        self._cal = {k: (float(v[0]), float(v[1])) for k, v in d.get("cal", {}).items()}
        self.achieved = {k: (int(v[0]), int(v[1])) for k, v in d["achieved"].items()}
        if d.get("transfer"):
            self.transfer.load_state_dict(d["transfer"])
        else:
            self.restore_notes.append("transfer estimator initialised fresh (checkpoint without transfer evidence)")
        self._reference_sd = d.get("reference")

    def after_fast_restore(self) -> None:
        """Call after the goal-free fast state itself was restored: the reference prior is restored from the
        checkpoint, or (older checkpoints) taken as the restored state."""
        sd = getattr(self, "_reference_sd", None)
        if sd and self.reference is not None:
            self.reference.load_state_dict(sd)
        elif hasattr(self.fast, "clone"):
            self.reference = self.fast.clone()
            self.restore_notes.append("reference prior = restored goal-free fast state (checkpoint without one)")
