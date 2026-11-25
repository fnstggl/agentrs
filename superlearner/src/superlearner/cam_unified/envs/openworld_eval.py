"""Operator-side goal evaluation for OpenWorld environments (implements `interfaces.GoalEvaluator`).

For every environment the evaluator holds an *objective* (the Crucible objective protocol: ``begin(world, actor)``,
``step_reward(world, actor) -> float``, ``status(world, actor) -> dict``). By default it is
`crucible.goals.contracts.RewardContract(task)`: potential-based shaping on the task's verified progress metric plus
a one-time success bonus, with sustain streaks counted by `GoalVerifier` at daily checks. Tasks that need another
objective (OW-4 recovery, OW-7 joint utility, rig net income ...) provide ``make_objective()``.

The evaluator registers two hooks on the `OpenWorldOperator`: at every episode start it begins the objective, and at
every clock stop (each daily check time and the end of each advance) it calls ``step_reward``; the increments
accumulate until the harness takes them (`take_reward`). Those scalars are the only thing derived from evaluation
that reaches a learner. `evaluate_goal` returns the independent result with evaluator-only `details` (streaks,
bindings, the contract id, integrity diagnostics) that the harness keeps out of learner inputs.
"""
from __future__ import annotations

import pickle
from typing import Any, Callable, Optional

from ..interfaces import IndependentGoalResult

DAY = 86400


def leaf_values(world, actor: str, predicate: dict) -> dict:
    """Verified value of every metric leaf of a success predicate, under the actor's bindings ($self/$org/$host),
    maximised over the organisations the actor owns (as `GoalVerifier.predicate` does)."""
    from crucible.goals.contracts import _bind, bindings
    from crucible.goals.metrics import evaluate_metric, predicate_leaves
    env = bindings(world, actor)
    out: dict = {}
    for lf in predicate_leaves(predicate):
        best = None
        for org in env["$orgs"] or ["o:none"]:
            try:
                v = evaluate_metric(world, lf["metric"], _bind(lf.get("args", {}), dict(env, **{"$org": org})))
            except Exception:  # noqa: BLE001 - missing entity: no value through this organisation
                continue
            best = v if best is None else max(best, v)
        key = lf["metric"]
        out[key] = 0.0 if best is None else float(best)
    return out


class OpenWorldEvaluator:
    """`GoalEvaluator` for the environments of one `OpenWorldOperator`. Construct on the operator side only."""

    name = "crucible.goals.GoalVerifier"

    def __init__(self, operator, shaping: float = 1.0, success_bonus: float = 1.0,
                 objective_factory: Optional[Callable[[Any], Any]] = None, integrity: bool = False):
        from crucible.goals.contracts import RewardContract
        self._op = operator
        self.shaping, self.success_bonus = float(shaping), float(success_bonus)
        self._factory = objective_factory or (lambda task: task.make_objective() if hasattr(task, "make_objective")
                                              else RewardContract(task, self.shaping, self.success_bonus))
        self.integrity = bool(integrity)
        n = operator.n
        self.objectives: list = [None] * n
        self.pending = [0.0] * n
        self.total = [0.0] * n
        self.success_t: list = [None] * n
        operator.episode_hooks.append(self._begin)
        operator.clock_hooks.append(self._tick)

    # ------------------------------------------------------------------ hooks (called by the operator)
    def _begin(self, b: int) -> None:
        obj = self._factory(self._op.task(b))
        obj.begin(self._op.world(b), self._op.principal(b))
        self.objectives[b] = obj
        self.pending[b] = 0.0
        self.total[b] = 0.0
        self.success_t[b] = None

    def _tick(self, b: int) -> None:
        obj = self.objectives[b]
        if obj is None:
            return
        w, p = self._op.world(b), self._op.principal(b)
        r = float(obj.step_reward(w, p))
        self.pending[b] += r
        self.total[b] += r
        if self.success_t[b] is None:
            st = obj.status(w, p)
            if st.get("success"):
                self.success_t[b] = st.get("success_t", w.now)

    # ------------------------------------------------------------------ harness side
    def take_reward(self, b: int) -> float:
        """Verified return increment accumulated since the last call (the learner's reward)."""
        r, self.pending[b] = self.pending[b], 0.0
        return r

    def take_rewards(self) -> list:
        return [self.take_reward(b) for b in range(self._op.n)]

    def evaluate_goal(self, env_index: int) -> IndependentGoalResult:
        b = int(env_index)
        w, p = self._op.world(b), self._op.principal(b)
        obj = self.objectives[b]
        task = self._op.task(b)
        st = obj.status(w, p)
        metrics = dict(st.get("metrics", {}))
        pred = getattr(task, "success", None)
        if isinstance(pred, dict):
            metrics.update(leaf_values(w, p, pred))
        details = {k: v for k, v in st.items() if k not in ("metrics",)}
        details.update(goal_contract=getattr(task, "id", None), world=dict(getattr(task, "world", {})),
                       principal=p, return_total=self.total[b])
        if isinstance(pred, dict):
            from crucible.goals.contracts import bindings
            env = bindings(w, p)
            details["bindings"] = {"$org": env["$org"], "$host": env["$host"], "$orgs": env["$orgs"]}
        if self.integrity:
            from crucible.evaluation.audit import integrity_report
            details["integrity"] = integrity_report(w, p)
        return IndependentGoalResult(goal_id=self._op.envs[b].goal.goal_id if self._op.envs[b].goal else "",
                                     achieved=bool(st.get("success")), score=float(st.get("progress", 0.0)),
                                     metrics=metrics, t=w.now / DAY, evaluator=self.name, details=details)

    def state_dict(self) -> bytes:
        return pickle.dumps({"objectives": self.objectives, "pending": self.pending, "total": self.total,
                             "success_t": self.success_t})

    def load_state(self, blob: bytes) -> None:
        d = pickle.loads(blob)
        self.objectives, self.pending, self.total, self.success_t = (d["objectives"], d["pending"], d["total"],
                                                                     d["success_t"])
