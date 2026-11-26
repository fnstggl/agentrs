"""Operator-side goal evaluation for OpenWorld environments (implements `interfaces.GoalEvaluator`).

For every environment the evaluator holds an *objective* (the Crucible objective protocol: ``begin(world, actor)``,
``step_reward(world, actor) -> float``, ``status(world, actor) -> dict``). By default it is
`crucible.goals.contracts.RewardContract(task)`: potential-based shaping on the task's verified progress metric plus
a one-time success bonus, with sustain streaks counted by `GoalVerifier` at daily checks. Tasks that need another
objective (OW-4 recovery, OW-7 joint utility, rig net income ...) provide ``make_objective()``.

The evaluator registers two hooks on the `OpenWorldOperator`: at every episode start it begins the objective, and at
every clock stop (each daily check time and the end of each advance) it calls ``step_reward``. `evaluate_goal`
returns the independent result with evaluator-only `details` (streaks, bindings, the contract id, integrity
diagnostics) that the harness keeps out of learner inputs.

Two channels leave the evaluator (redesign plan 2026-11-26 §1 "OpenWorld evaluator: REDESIGN", W2):

* **Gated reward** (`take_reward(s)`): the objective's increments go into a Crucible 5.0
  `experience.modes.RewardGate` in the evaluator's ``reward_mode`` (default: the operator's, itself default
  ``legacy_shaped_reward``): legacy = the shaped increments as before; ``verified_terminal_outcome`` = 1.0 once at
  the first verified success, else 0.0; ``observation_only`` = 0.0 always (the objective still runs on the
  evaluator side, so labels and `evaluate_goal` are unchanged). `total` keeps the legacy return as the operator's
  record in every mode.
* **Verified labels** (`take_labels()`, `interfaces.LabelSource`): one `labels.LabelRecord` per (env, episode, goal)
  at verification time only - ``terminal_success`` at the clock stop where the verifier first reports success
  (``t_resolved_days`` = success time - episode start), ``terminal_failure`` when it reports failure, i.e. the
  horizon passed without success. A label is never emitted before its outcome was verified; a run that stops early
  leaves the episode pending (no label). Labels carry the actor-visible opaque env id and goal id, utility 1 / 0,
  the primary verified metric value in its unit (``value``), the verified leaf metric values and the resources used
  (``meta``); never the world family, seed, bindings, progress or streak.
"""
from __future__ import annotations

import pickle
from typing import Any, Callable, Optional

from ..interfaces import IndependentGoalResult
from .labels import EVAL_STREAM, FORBIDDEN_KEYS, TRAIN_STREAM, LabelRecord

DAY = 86400
LEGACY_MODE = "legacy_shaped_reward"


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
    """`GoalEvaluator` and `interfaces.LabelSource` for the environments of one `OpenWorldOperator`. Construct on the
    operator side only. `reward_mode`: None = the operator's (see the module docstring); `stream`: ``"train"`` or
    ``"eval"`` (labels of an evaluation stream are tagged so a training LabelStore refuses them)."""

    name = "crucible.goals.GoalVerifier"

    def __init__(self, operator, shaping: float = 1.0, success_bonus: float = 1.0,
                 objective_factory: Optional[Callable[[Any], Any]] = None, integrity: bool = False,
                 reward_mode: Optional[str] = None, stream: str = TRAIN_STREAM):
        from crucible.experience.modes import RewardGate, check_mode
        from crucible.goals.contracts import RewardContract
        self.reward_mode = check_mode(reward_mode if reward_mode is not None
                                      else getattr(operator, "reward_mode", LEGACY_MODE))
        if stream not in (TRAIN_STREAM, EVAL_STREAM):
            raise ValueError(f"stream must be {TRAIN_STREAM!r} or {EVAL_STREAM!r}")
        self.stream = stream
        self.gate = RewardGate(self.reward_mode)
        self._op = operator
        self.shaping, self.success_bonus = float(shaping), float(success_bonus)
        self._factory = objective_factory or (lambda task: task.make_objective() if hasattr(task, "make_objective")
                                              else RewardContract(task, self.shaping, self.success_bonus))
        self.integrity = bool(integrity)
        n = operator.n
        self.objectives: list = [None] * n
        self.total = [0.0] * n
        self.success_t: list = [None] * n
        self.labelled: list = [False] * n                 # the current episode's terminal label was emitted
        self._labels: list = []                            # emitted, not yet taken
        operator.episode_hooks.append(self._begin)
        operator.clock_hooks.append(self._tick)

    # ------------------------------------------------------------------ hooks (called by the operator)
    def _begin(self, b: int) -> None:
        obj = self._factory(self._op.task(b))
        obj.begin(self._op.world(b), self._op.principal(b))
        self.objectives[b] = obj
        self.total[b] = 0.0                                # the gate is keyed per episode: nothing carries over
        self.success_t[b] = None
        self.labelled[b] = False

    def _key(self, b: int) -> str:
        return f"env{int(b)}#ep{int(getattr(self._op, 'episode', 0))}"

    def _tick(self, b: int) -> None:
        obj = self.objectives[b]
        if obj is None:
            return
        w, p = self._op.world(b), self._op.principal(b)
        r = float(obj.step_reward(w, p))
        self.gate.add_legacy(self._key(b), r)
        self.total[b] += r
        if self.success_t[b] is None or not self.labelled[b]:
            st = obj.status(w, p)
            if self.success_t[b] is None and st.get("success"):
                self.success_t[b] = st.get("success_t", w.now)
                self.gate.verified(self._key(b), True)
            if not self.labelled[b]:
                if st.get("success"):
                    self._emit(b, st, "success", self.success_t[b])
                elif st.get("failed"):
                    self._emit(b, st, "failure", w.now)

    def _emit(self, b: int, st: dict, status: str, t_s) -> None:
        e = self._op.envs[b]
        w, p = self._op.world(b), self._op.principal(b)
        pred = getattr(self._op.task(b), "success", None)
        verified = {}
        if isinstance(pred, dict):
            verified = leaf_values(w, p, pred)
        for k, v in (st.get("metrics") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                verified.setdefault(str(k), float(v))
        verified = {k: float(v) for k, v in verified.items() if k not in FORBIDDEN_KEYS}
        value = next(iter(verified.values()), None)
        t0 = int(getattr(e, "t0", 0))
        t_res = int(w.now if t_s is None else t_s)
        meta = {"label": f"terminal_{status}", "verified": verified,
                "resources": {"sim_days": (w.now - t0) / DAY, "actions": int(e.stats.get("actions", 0)),
                              "rejected": int(e.stats.get("rejected", 0)), "waits": int(e.stats.get("waits", 0))}}
        if self.stream == EVAL_STREAM:
            meta["stream"] = EVAL_STREAM
        self._labels.append(LabelRecord(
            env_id=e.env_id, episode=int(self._op.episode), status=status, utility=1.0 if status == "success" else 0.0,
            value=value, t_resolved_days=(t_res - t0) / DAY, evaluator=self.name,
            goal_id=e.goal.goal_id if e.goal is not None else "", kind="terminal", meta=meta,
            t_start_days=t0 / DAY, t_end_days=w.now / DAY))
        self.labelled[b] = True

    # ------------------------------------------------------------------ harness side
    def take_reward(self, b: int) -> float:
        """What the learner may receive since the last call, through the RewardGate of `reward_mode` (legacy: the
        shaped verified increments; verified: 1.0 once at success; observation_only: 0.0)."""
        return float(self.gate.take(self._key(b)))

    def take_rewards(self) -> list:
        return [self.take_reward(b) for b in range(self._op.n)]

    def take_labels(self) -> list:
        """`interfaces.LabelSource`: the verified labels emitted since the previous call, each exactly once."""
        out, self._labels = self._labels, []
        return out

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
        return pickle.dumps({"objectives": self.objectives, "total": self.total, "success_t": self.success_t,
                             "gate": self.gate, "labelled": self.labelled, "labels": self._labels,
                             "reward_mode": self.reward_mode})

    def load_state(self, blob: bytes) -> None:
        d = pickle.loads(blob)
        self.objectives, self.total, self.success_t = d["objectives"], d["total"], d["success_t"]
        if "gate" in d:
            mode = d.get("reward_mode", self.reward_mode)
            if mode != self.reward_mode:
                raise ValueError(f"checkpoint evaluator reward_mode {mode!r} differs from {self.reward_mode!r}")
            self.gate, self.labelled, self._labels = d["gate"], list(d["labelled"]), list(d["labels"])
        else:                                              # a pre-W2 checkpoint: ungated pending increments
            for b, r in enumerate(d.get("pending", [])):
                self.gate.add_legacy(self._key(b), float(r))
            self.labelled = [s is not None for s in self.success_t]
