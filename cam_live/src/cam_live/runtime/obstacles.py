"""cam_live.runtime.obstacles — obstacle records (spec "autonomous obstacle resolution").

When an action fails, is refused, or does not verify, the runtime writes an :class:`ObstacleRecord` and then keeps
it linked to what actually followed: the alternatives the learner considered at its next decision, the one it
chose, how that turned out, and the learning updates recorded afterwards. There is no recovery logic here: the
failure flows back to the learner as an observation and the ordinary loop decides what happens next.
The file is append-only; each change appends a new revision of the record.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..contracts import (ActionResult, ActionSpec, Clock, Decision, FailureClass, LearningUpdateRecord,
                         ObstacleRecord, ResultStatus, new_id)
from .logs import RunLog

#: descriptive category of a failure class (spec item 3: technical / informational / strategic / resource / permission)
CATEGORY_BY_FAILURE: Dict[str, str] = {
    FailureClass.INVALID_ARGUMENTS: "technical", FailureClass.STALE_OBSERVATION: "technical",
    FailureClass.VISUAL_GROUNDING: "technical", FailureClass.PAGE_CHANGED: "technical",
    FailureClass.TRANSIENT_NETWORK: "technical", FailureClass.WRONG_SITE_ASSUMPTION: "informational",
    FailureClass.MISSING_PERMISSION: "permission", FailureClass.PLATFORM_RESTRICTION: "permission",
    FailureClass.RESOURCE_LIMIT: "resource", FailureClass.STRATEGIC: "strategic", FailureClass.UNKNOWN: "",
}


def is_obstacle(result: ActionResult) -> bool:
    if result.status == ResultStatus.NEEDS_APPROVAL:
        return False                                  # queued for the operator: not a failure
    return result.status != ResultStatus.OK or result.verified is False


class ObstacleTracker:
    def __init__(self, log: RunLog, clock: Clock, max_open: int = 64):
        self.log, self.clock, self.max_open = log, clock, max_open
        self._open: Dict[str, Dict[str, Any]] = {}        # obstacle id -> {"rec": ObstacleRecord, "rev", "op", "sig"}

    def _write(self, entry: Dict[str, Any]) -> str:
        rid = self.log.obstacle(entry["rec"], entry["rev"])
        entry["rev"] += 1
        return rid

    def open(self, action: ActionSpec, result: ActionResult, agent: Any = None) -> ObstacleRecord:
        fc = result.failure_class if result.failure_class != FailureClass.NONE else FailureClass.UNKNOWN
        desc = result.failure_detail or (f"action finished with status {result.status}"
                                         if result.status != ResultStatus.OK else
                                         "the verification condition did not hold in the resulting state")
        rec = ObstacleRecord(obstacle_id=new_id("obst"), ts=self.clock.now(), action_id=action.action_id,
                             failure_class=fc, description=f"{action.operation}: {desc}"[:2000],
                             category=CATEGORY_BY_FAILURE.get(fc, ""))
        explain = getattr(agent, "explain_obstacle", None)
        if callable(explain):
            try:
                rec.explanations = [dict(e) for e in (explain(rec) or [])]
            except Exception as e:  # noqa: BLE001 — a failing diagnostic hook must not hide the obstacle itself
                rec.outcome = f"explanation hook failed: {type(e).__name__}"
        entry = {"rec": rec, "rev": 0, "op": action.operation}
        self._open[rec.obstacle_id] = entry
        self._write(entry)
        while len(self._open) > self.max_open:
            self._open.pop(next(iter(self._open)))
        return rec

    def on_decision(self, decision: Decision) -> None:
        """The first decision after an obstacle: record what was considered and what was chosen instead."""
        for entry in self._open.values():
            rec: ObstacleRecord = entry["rec"]
            if rec.chosen_alternative or decision.chosen.action_id == rec.action_id:
                continue
            rec.alternatives = [c.action.action_id for c in decision.candidates][:64]
            rec.chosen_alternative = decision.chosen.action_id
            rec.strategy_changed = decision.chosen.operation != entry["op"]
            self._write(entry)

    def on_result(self, action: ActionSpec, result: ActionResult) -> None:
        for entry in self._open.values():
            rec: ObstacleRecord = entry["rec"]
            if rec.chosen_alternative == action.action_id and not entry.get("settled"):
                rec.outcome = (f"alternative {action.operation} finished with status {result.status}"
                               + (f" ({result.failure_class})" if result.failure_class != FailureClass.NONE else ""))
                entry["settled"] = True
                self._write(entry)

    def on_learning(self, updates: Sequence[LearningUpdateRecord]) -> None:
        """Link the learning updates that followed; an obstacle is closed once its alternative has an outcome and
        a learning step has run after it."""
        ids = [u.update_id for u in updates]
        for oid in list(self._open):
            entry = self._open[oid]
            if ids:
                entry["rec"].learning_update_ids.extend(ids)
                self._write(entry)
            if entry.get("settled") and (ids or entry.get("learn_calls", 0) >= 1):
                del self._open[oid]
            elif entry.get("settled"):
                entry["learn_calls"] = entry.get("learn_calls", 0) + 1

    def open_count(self) -> int:
        return len(self._open)

    def to_state(self) -> List[Dict[str, Any]]:
        return [{"rec": e["rec"].to_dict(), "rev": e["rev"], "op": e["op"], "settled": bool(e.get("settled")),
                 "learn_calls": e.get("learn_calls", 0)} for e in self._open.values()]

    def load_state(self, st: Optional[List[Dict[str, Any]]]) -> None:
        self._open = {}
        for d in st or []:
            rec = ObstacleRecord.from_dict(d["rec"])
            self._open[rec.obstacle_id] = {"rec": rec, "rev": int(d.get("rev", 1)), "op": d.get("op", ""),
                                           "settled": bool(d.get("settled")), "learn_calls": d.get("learn_calls", 0)}
