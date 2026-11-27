"""cam_live.diagnostics.memstore — an in-memory ``contracts.ExperienceStore``. MASTER-OWNED.

For mathematical tests that do not need durability. The durable store is ``cam_live.credit.replay``.
Implements the whole protocol of contract 1.1 (change requests C-5, D-1, D-3, D-10): outcomes are attached with
de-duplication by ``dedup_key``, items are immutable once stored, objective outcomes have a place.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from ..contracts import ExperienceItem, ExperienceRecord, OutcomeObservation, OutcomeQuery, OutcomeStatus


@dataclass
class MemAddOutcome:
    accepted: bool
    exp_id: str = ""
    duplicate: bool = False
    reason: str = ""


class MemoryStore:
    def __init__(self):
        self._recs: Dict[str, ExperienceRecord] = {}
        self._order: List[str] = []
        self._by_dec: Dict[str, str] = {}
        self._items: Dict[str, ExperienceItem] = {}
        self._objectives: Dict[str, Dict[str, Any]] = {}
        self._seen_outcomes: set = set()
        self._sampled = 0

    def append(self, rec: ExperienceRecord) -> None:
        if rec.exp_id in self._recs:
            raise KeyError(f"duplicate experience {rec.exp_id}")
        self._recs[rec.exp_id] = rec
        self._order.append(rec.exp_id)
        self._by_dec[rec.decision.decision_id] = rec.exp_id

    def update(self, rec: ExperienceRecord) -> None:
        if rec.exp_id not in self._recs:
            raise KeyError(rec.exp_id)
        self._recs[rec.exp_id] = rec

    def get(self, exp_id: str) -> Optional[ExperienceRecord]:
        return self._recs.get(exp_id)

    def by_decision(self, decision_id: str) -> Optional[ExperienceRecord]:
        return self._recs.get(self._by_dec.get(decision_id, ""))

    def put_item(self, item: ExperienceItem) -> None:
        # history is immutable: a new version of an item needs a new ref (an overwrite would change the context
        # H_t of every earlier decision that references it) [D-3]
        self._items.setdefault(item.ref, item)

    def items(self, refs: Sequence[str]) -> List[ExperienceItem]:
        return [self._items[r] for r in refs if r in self._items]

    def pending(self, now: float) -> List[Tuple[ExperienceRecord, OutcomeQuery]]:
        return [(r, q) for r in self._recs.values() for q in r.pending_queries if q.due_at <= now]

    def next_due(self) -> Optional[float]:
        dues = [q.due_at for r in self._recs.values() for q in r.pending_queries]
        return min(dues) if dues else None

    def sample(self, n: int, *, loss: str, strata: Optional[Dict[str, float]] = None,
               seed: Optional[int] = None, provenance: Optional[Sequence[str]] = None) -> List[ExperienceRecord]:
        pool = [r for r in self._recs.values() if r.eligible.get(loss, True)
                and (provenance is None or r.provenance in set(provenance))]
        if seed is None:
            seed, self._sampled = self._sampled, self._sampled + 1
        rng = random.Random(seed)
        return [] if not pool else [rng.choice(pool) for _ in range(n)] if n > len(pool) else rng.sample(pool, n)

    def add_outcome(self, obs: OutcomeObservation, exp_id: str = "") -> MemAddOutcome:
        """Attach a measurement to the record that asked ``obs.query_id``; the same event (dedup key) once."""
        if obs.status == OutcomeStatus.PENDING:
            return MemAddOutcome(False, reason="pending is not an outcome")
        rec = self._recs.get(exp_id) if exp_id else None
        if rec is None:
            rec = next((r for r in self._recs.values() if any(q.query_id == obs.query_id for q in
                                                              list(r.pending_queries) + list(r.queries))), None)
        if rec is None:
            return MemAddOutcome(False, reason="no record asked this query")
        key = (obs.dedup_key or f"{obs.query_id}@{obs.observed_at!r}", obs.query_id)
        if key in self._seen_outcomes:
            return MemAddOutcome(False, rec.exp_id, duplicate=True, reason="duplicate")
        self._seen_outcomes.add(key)
        rec.outcomes.append(obs)
        rec.pending_queries = [q for q in rec.pending_queries if q.query_id != obs.query_id]
        return MemAddOutcome(True, rec.exp_id)

    def iter_records(self, *, since_seq: int = 0, objective_id: str = "") -> Iterator[ExperienceRecord]:
        for i in self._order:
            r = self._recs[i]
            if r.seq >= since_seq and (not objective_id or r.objective_id == objective_id):
                yield r

    def put_objective_outcome(self, objective_id: str, *, status: str, value: float = 0.0, evaluated_at: float = 0.0,
                              verified: bool = False, body: Optional[Dict[str, Any]] = None) -> None:
        self._objectives[objective_id] = {"objective_id": objective_id, "status": status, "value": float(value),
                                          "evaluated_at": float(evaluated_at), "verified": bool(verified),
                                          "body": dict(body or {})}

    def objective_outcome(self, objective_id: str) -> Optional[Dict[str, Any]]:
        o = self._objectives.get(objective_id)
        return None if o is None else dict(o)

    def state_dict(self) -> Dict[str, Any]:
        return {"sampled": self._sampled}

    def load_state_dict(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self._sampled = int(state.get("sampled", 0))
        return {}

    def window(self, window_id: str) -> List[ExperienceRecord]:
        return [self._recs[i] for i in self._order if self._recs[i].window_id == window_id]

    def all(self) -> List[ExperienceRecord]:
        return [self._recs[i] for i in self._order]

    def count(self) -> int:
        return len(self._recs)

    def flush(self) -> None:
        return None

    def clone(self) -> "MemoryStore":
        return copy.deepcopy(self)
