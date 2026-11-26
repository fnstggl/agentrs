"""cam_live.diagnostics.memstore — an in-memory ``contracts.ExperienceStore``. MASTER-OWNED.

For mathematical tests that do not need durability. The durable store is ``cam_live.credit.replay``.
"""
from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Sequence, Tuple

from ..contracts import ExperienceItem, ExperienceRecord, OutcomeQuery


class MemoryStore:
    def __init__(self):
        self._recs: Dict[str, ExperienceRecord] = {}
        self._order: List[str] = []
        self._by_dec: Dict[str, str] = {}
        self._items: Dict[str, ExperienceItem] = {}

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
        self._items[item.ref] = item

    def items(self, refs: Sequence[str]) -> List[ExperienceItem]:
        return [self._items[r] for r in refs if r in self._items]

    def pending(self, now: float) -> List[Tuple[ExperienceRecord, OutcomeQuery]]:
        return [(r, q) for r in self._recs.values() for q in r.pending_queries if q.due_at <= now]

    def sample(self, n: int, *, loss: str, strata: Optional[Dict[str, float]] = None,
               seed: Optional[int] = None) -> List[ExperienceRecord]:
        pool = [r for r in self._recs.values() if r.eligible.get(loss, True)]
        rng = random.Random(seed)
        return [] if not pool else [rng.choice(pool) for _ in range(n)] if n > len(pool) else rng.sample(pool, n)

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
