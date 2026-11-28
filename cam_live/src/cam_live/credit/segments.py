"""cam_live.credit.segments — credit units inside one long, still open objective (CR, amendment item 4).

Learned credit (RUDDER, COCOA) needs a *final* return, and the held-out validation of the estimators needs units
that nothing trained on. Both were defined per lifetime (objective): a single extended objective - CAM Live's
real setting - produced no learned credit until it closed and no held-out unit ever (the learner trains on it).

A **segment** is a block of ``K`` consecutive top-level decisions of an objective (with their lower-level records):
segment k holds the decisions with top-level index in [kK, (k+1)K). Its return consists of the observed answers to
the goal-relevant queries *its own decisions declared* (whenever observed), each outcome event once; it is final when
the block is complete (the objective has >= (k+1)K top-level decisions) and none of those queries is pending. While
an objective is open, its complete segments are the credit units: RUDDER and COCOA train on matured segments and
credit their decisions, the estimators are compared with randomised decisions of *held-out* segments, and earlier
credit is revised version by version as segments mature and, at closure, over the whole lifetime. A closed
objective is one unit, exactly as before (nothing changes for lifetimes).

Truncation (documented, not hidden): an outcome measured by a query that a *later* segment's decision declared is
not part of an earlier segment's return, even if an earlier decision caused it; K bounds the horizon of credit
inside an open objective (until the objective closes and the lifetime is re-credited as a whole).

Held-out split by segment (blocked cross-validation for temporally dependent data; Roberts et al. 2017, Ecography
40:913): the learner's ``val_split="segment"`` holds out every 1/f-th segment of each objective (stratified as the
record split of INT3A-5), so a randomised decision of a held-out segment is a unit whose own outcomes no learner
and no estimator trained on.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..contracts import ExperienceRecord

SEP = "#seg"


def segment_id(objective_id: str, k: int) -> str:
    return f"{objective_id}{SEP}{int(k):05d}"


def parse_unit(unit: str) -> Tuple[str, Optional[int]]:
    """(objective id, segment index or None for a whole lifetime)."""
    if SEP in unit:
        oid, k = unit.rsplit(SEP, 1)
        try:
            return oid, int(k)
        except ValueError:
            return unit, None
    return unit, None


def is_segment(unit: str) -> bool:
    return parse_unit(unit)[1] is not None


class CreditUnits:
    """Maps records to credit units and resolves a unit to its records.

    decisions   K, top-level decisions per segment; 0 disables segments (every objective is one unit)

    The top-level index of a stored record never changes (append-only history), so it is cached per objective and
    extended when records are appended; the objective's trajectory is re-read only when the store changed."""

    def __init__(self, store: Any, decisions: int = 16):
        self.store = store
        self.decisions = max(0, int(decisions))
        self._top: Dict[str, Dict[str, int]] = {}          # objective -> exp_id -> top-level index
        self._n_top: Dict[str, int] = {}                   # objective -> top-level decisions seen
        self._traj: Dict[str, Tuple[Any, List[ExperienceRecord]]] = {}
        self._closed: Dict[str, bool] = {}

    @property
    def enabled(self) -> bool:
        return self.decisions > 0

    # ---- store reads ----------------------------------------------------------------------------------------
    def _gen(self) -> Any:
        st = self.store
        g = getattr(st, "_gen", None)
        return ("gen", g) if g is not None else ("head", st.head_seq() if hasattr(st, "head_seq") else None)

    def trajectory(self, objective_id: str) -> List[ExperienceRecord]:
        g = self._gen()
        hit = self._traj.get(objective_id)
        if hit is not None and hit[0] == g:
            return hit[1]
        recs = self.store.trajectory(objective_id)
        if len(self._traj) > 64:
            self._traj.clear()
        self._traj[objective_id] = (g, recs)
        self._index(objective_id, recs)
        return recs

    def _index(self, objective_id: str, recs: Sequence[ExperienceRecord]) -> None:
        from .providers import top_index, is_top_level
        idx = top_index(recs)
        self._top[objective_id] = {e: i for e, i in idx.items() if i >= 0}
        self._n_top[objective_id] = sum(1 for r in recs if is_top_level(r))

    def closed(self, objective_id: str) -> bool:
        if self._closed.get(objective_id):
            return True                                   # an objective outcome is never removed
        c = bool(objective_id) and self.store.objective_outcome(objective_id) is not None
        if c:
            self._closed[objective_id] = True
        return c

    def top_position(self, rec: ExperienceRecord) -> Optional[int]:
        oid = rec.objective_id
        if not oid:
            return None
        pos = self._top.get(oid, {}).get(rec.exp_id)
        if pos is None:
            self.trajectory(oid)
            pos = self._top.get(oid, {}).get(rec.exp_id)
        return pos

    # ---- units ----------------------------------------------------------------------------------------------
    def segment_index(self, rec: ExperienceRecord) -> Optional[int]:
        """Segment of a record (by its top-level decision), or None when segments are off / the record is unknown."""
        if not self.enabled:
            return None
        pos = self.top_position(rec)
        return None if pos is None else pos // self.decisions

    def unit_of(self, rec: ExperienceRecord) -> str:
        """The credit unit of a record: its objective when segments are off or the objective is closed, else the
        segment of its top-level decision."""
        oid = rec.objective_id
        if not self.enabled or not oid or self.closed(oid):
            return oid
        k = self.segment_index(rec)
        return oid if k is None else segment_id(oid, k)

    def units_of(self, records: Sequence[ExperienceRecord]) -> List[str]:
        return list(dict.fromkeys(u for u in (self.unit_of(r) for r in records) if u))

    def records(self, unit: str) -> List[ExperienceRecord]:
        oid, k = parse_unit(unit)
        recs = self.trajectory(oid)
        if k is None:
            return recs
        top = self._top.get(oid, {})
        K = self.decisions
        return [r for r in recs if top.get(r.exp_id, -1) // K == k and top.get(r.exp_id, -1) >= 0] if K else []

    def complete(self, unit: str) -> bool:
        """A segment is complete once the objective has decisions beyond it (no decision will be added to it)."""
        oid, k = parse_unit(unit)
        if k is None:
            return True
        self.trajectory(oid)
        return self._n_top.get(oid, 0) >= (k + 1) * self.decisions

    def held_out_segment(self, objective_id: str, k: int, fraction: float, seed: int) -> bool:
        """Stratified block split: segment k of an objective is held out iff floor(u + (k+1) f) > floor(u + k f),
        u = seeded hash of the objective (every 1/f-th segment; the same rule as the learner's record split)."""
        import hashlib
        f = float(fraction)
        if f <= 0.0:
            return False
        if f >= 1.0:
            return True
        u = int(hashlib.sha1(f"{seed}|segment-split|{objective_id}".encode()).hexdigest()[:8], 16) / 2.0 ** 32
        return math.floor(u + (k + 1) * f) > math.floor(u + k * f)
