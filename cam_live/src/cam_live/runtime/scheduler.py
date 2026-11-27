"""cam_live.runtime.scheduler — pending-outcome queue and wake scheduling (spec 15.1).

The runtime never fills time with activity. Between decisions it sleeps until the earliest of: the time the
learner asked to be woken, the next matured outcome, the next evidence poll (if one is configured), the experiment
deadline — or an operator command, which interrupts the sleep.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..contracts import Clock, OutcomeObservation, OutcomeQuery, OutcomeStatus


@dataclass
class PendingItem:
    query: OutcomeQuery
    action_id: str = ""
    decision_id: str = ""
    experiment_id: str = ""
    not_before: float = 0.0          # retry time after the environment answered "still pending"
    attempts: int = 0
    origin: str = "environment"      # "environment": issued by the environment's result; "learner": a query the
                                     # learner registered with a prediction for an executed action

    @property
    def ready_at(self) -> float:
        return max(self.query.due_at, self.not_before)

    def to_dict(self) -> Dict[str, Any]:
        return {"query": self.query.to_dict(), "action_id": self.action_id, "decision_id": self.decision_id,
                "experiment_id": self.experiment_id, "not_before": self.not_before, "attempts": self.attempts,
                "origin": self.origin}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PendingItem":
        return cls(OutcomeQuery.from_dict(d["query"]), d.get("action_id", ""), d.get("decision_id", ""),
                   d.get("experiment_id", ""), float(d.get("not_before", 0.0)), int(d.get("attempts", 0)),
                   str(d.get("origin", "environment")))


class PendingOutcomes:
    """Outcome queries waiting to mature. An outcome is reconciled exactly once: a second observation for the same
    query id, or with the same ``dedup_key``, is dropped."""

    def __init__(self) -> None:
        self._items: Dict[str, PendingItem] = {}
        self._resolved: Dict[str, float] = {}        # query id -> observed_at
        self._dedup: Dict[str, str] = {}             # dedup key -> query id

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> List[PendingItem]:
        return sorted(self._items.values(), key=lambda i: (i.ready_at, i.query.query_id))

    def add(self, query: OutcomeQuery, *, action_id: str = "", decision_id: str = "", experiment_id: str = "",
            origin: str = "environment") -> bool:
        if query.query_id in self._items or query.query_id in self._resolved:
            return False
        self._items[query.query_id] = PendingItem(query, action_id, decision_id or query.decision_id,
                                                  experiment_id or query.experiment_id, origin=origin)
        return True

    def has(self, query_id: str) -> bool:
        return query_id in self._items or query_id in self._resolved

    def count_origin(self, origin: str) -> int:
        return sum(1 for i in self._items.values() if i.origin == origin)

    def due(self, now: float) -> List[PendingItem]:
        return [i for i in self.items() if i.ready_at <= now]

    def next_ready(self) -> Optional[float]:
        return min((i.ready_at for i in self._items.values()), default=None)

    def defer(self, query_id: str, until: float) -> None:
        it = self._items.get(query_id)
        if it is not None:
            it.not_before, it.attempts = until, it.attempts + 1

    def resolve(self, obs: OutcomeObservation) -> Optional[PendingItem]:
        """Take the item an observation settles. None when the query is unknown, still pending, already resolved,
        or the observation duplicates an event already counted."""
        if obs.status == OutcomeStatus.PENDING or obs.query_id in self._resolved:
            return None
        it = self._items.get(obs.query_id)
        if it is None:
            return None
        if obs.dedup_key and self._dedup.get(obs.dedup_key, obs.query_id) != obs.query_id:
            del self._items[obs.query_id]            # same event as an earlier observation: settle without recounting
            self._resolved[obs.query_id] = obs.observed_at
            return None
        del self._items[obs.query_id]
        self._resolved[obs.query_id] = obs.observed_at
        if obs.dedup_key:
            self._dedup[obs.dedup_key] = obs.query_id
        return it

    def was_resolved(self, query_id: str) -> bool:
        return query_id in self._resolved

    def to_state(self) -> Dict[str, Any]:
        return {"items": [i.to_dict() for i in self.items()], "resolved": dict(self._resolved),
                "dedup": dict(self._dedup)}

    @classmethod
    def from_state(cls, st: Optional[Dict[str, Any]]) -> "PendingOutcomes":
        p = cls()
        for d in (st or {}).get("items", []):
            it = PendingItem.from_dict(d)
            p._items[it.query.query_id] = it
        p._resolved = {k: float(v) for k, v in (st or {}).get("resolved", {}).items()}
        p._dedup = dict((st or {}).get("dedup", {}))
        return p


def plan_wake(candidates: Dict[str, Optional[float]]) -> Tuple[Optional[float], str]:
    """Earliest wake time among named candidates (None entries ignored) and its reason."""
    best: Tuple[Optional[float], str] = (None, "")
    for reason, t in candidates.items():
        if t is not None and (best[0] is None or t < best[0]):
            best = (float(t), reason)
    return best


class SimSleeper:
    """Sleeping in simulated time: advance the shared SimClock. Cannot block for an operator command."""
    blocking = False

    def __init__(self, clock: Any):
        self.clock = clock

    def sleep_until(self, t: Optional[float], interrupted: Callable[[], bool]) -> str:
        if t is None:
            return "nothing_scheduled"
        if interrupted():
            return "interrupted"
        now = self.clock.now()
        if t > now:
            self.clock.advance(t - now)
        return "time"


class WallSleeper:
    """Real sleeping, interruptible: ``wake`` is set by a signal or by the control channel; the control file is
    also checked every ``poll_s`` seconds (one ``stat`` call — no network activity, no inference)."""
    blocking = True

    def __init__(self, clock: Clock, wake: Optional[threading.Event] = None, poll_s: float = 1.0):
        self.clock, self.wake, self.poll_s = clock, wake or threading.Event(), max(0.05, float(poll_s))

    def sleep_until(self, t: Optional[float], interrupted: Callable[[], bool]) -> str:
        while True:
            if self.wake.is_set():
                self.wake.clear()
                return "interrupted"
            if interrupted():
                return "interrupted"
            remaining = None if t is None else t - self.clock.now()
            if remaining is not None and remaining <= 0:
                return "time"
            self.wake.wait(self.poll_s if remaining is None else min(self.poll_s, remaining))


def make_sleeper(clock: Clock, wake: Optional[threading.Event] = None, poll_s: float = 1.0):
    """SimSleeper for clocks that can be advanced (``SimClock``), WallSleeper otherwise."""
    return SimSleeper(clock) if hasattr(clock, "advance") else WallSleeper(clock, wake, poll_s)
