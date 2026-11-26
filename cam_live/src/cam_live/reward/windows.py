"""cam_live.reward.windows — support and query windows cut from one long lifetime (spec 7.5).

One lifetime is not thousands of lifetimes, so meta-learning uses consecutive, separated windows of it:

    support window   the decisions whose transitions feed the inner update (rewarded by R_psi)
    query window     later decisions whose independently verified outcomes evaluate the updated learner

Rules enforced here
* **Temporal separation.** Every query decision starts after every support transition has ended, plus a
  configurable gap; an outcome observed before its own decision never counts.
* **Independence is counted per group, not per row.** Rows are grouped by experiment, else by parent decision,
  else by their own decision; records that share a verified outcome event (same ``dedup_key``) are merged into one
  group. ``n_independent`` is the number of groups with at least one admissible verified outcome.
* **No straddling.** A group that began before the query window started is correlated with support data: its
  rows are excluded from the query and its outcome is not counted there.
* **Eligibility.** Same provenance only (live / staging / diagnostic never mixed), an executed action, optional
  logged propensities; outcomes must be verified and observed or censored (pending / unobservable are masked).
* **Minimum evidence.** A pair is eligible for a meta-update only with at least ``min_independent_outcomes``
  independent verified query outcomes; otherwise it is returned with the reason so the skip can be logged.
* **Deterministic ids.** Window and pair ids are hashes of their content.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from ..contracts import ExperienceRecord, OutcomeObservation, OutcomeStatus

__all__ = ["independence_key", "verified_outcomes", "MetaWindow", "WindowPair", "WindowBuilder", "held_out_groups"]

OutcomeFilter = Callable[[ExperienceRecord, OutcomeObservation], bool]


def independence_key(rec: ExperienceRecord) -> str:
    """Experiment, else parent decision, else the decision itself."""
    d = rec.decision
    if d.experiment_id:
        return "exp:" + d.experiment_id
    parent = d.diagnostics.get("parent_decision_id") if isinstance(d.diagnostics, dict) else None
    return "dec:" + str(parent or d.decision_id)


def _default_filter(rec: ExperienceRecord, o: OutcomeObservation) -> bool:
    return bool(o.verified)


def verified_outcomes(rec: ExperienceRecord, outcome_filter: Optional[OutcomeFilter] = None
                      ) -> List[OutcomeObservation]:
    """Admissible verified outcomes of a record: verified, observed or censored, observed after the decision,
    de-duplicated by ``dedup_key``. ``outcome_filter`` adds the signature check (e.g. membership in the
    ``integrity.OutcomeLedger``)."""
    flt = outcome_filter or _default_filter
    out, seen = [], set()
    for o in rec.outcomes:
        key = o.dedup_key or f"{o.query_id}|{o.observed_at!r}|{o.value!r}"
        if key in seen or o.status not in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED):
            continue
        if not o.verified or o.observed_at < rec.decision.ts or not flt(rec, o):
            continue
        seen.add(key)
        out.append(o)
    return out


def _wid(role: str, exp_ids: Sequence[str], t0: float, t1: float) -> str:
    h = hashlib.sha256(json.dumps([role, sorted(exp_ids), repr(float(t0)), repr(float(t1))]).encode())
    return f"w{role[0]}_{h.hexdigest()[:16]}"


@dataclass(frozen=True)
class MetaWindow:
    window_id: str
    role: str                        # "support" | "query"
    exp_ids: Tuple[str, ...]         # in lifetime order
    t_start: float                   # first decision time
    t_end: float                     # last transition end (decision time + duration)
    groups: Tuple[str, ...]          # independent groups with >= 1 admissible verified outcome
    row_groups: Tuple[str, ...]      # group of each row (aligned with exp_ids)

    @property
    def n_rows(self) -> int:
        return len(self.exp_ids)

    @property
    def n_independent(self) -> int:
        return len(self.groups)


@dataclass(frozen=True)
class WindowPair:
    pair_id: str
    support: MetaWindow
    query: MetaWindow
    eligible: bool
    reason: str                      # "" when eligible, else why a meta-update must be skipped
    gap_s: float
    dropped: Tuple[Tuple[str, int], ...] = ()      # (reason, rows) excluded from the query

    @property
    def n_independent_query(self) -> int:
        return self.query.n_independent


class WindowBuilder:
    def __init__(self, min_independent_outcomes: int = 8, min_gap_s: float = 0.0, provenance: str = "live",
                 require_propensities: bool = False, outcome_filter: Optional[OutcomeFilter] = None):
        self.min_independent = int(min_independent_outcomes)
        self.min_gap_s, self.provenance = float(min_gap_s), provenance
        self.require_propensities = require_propensities
        self.outcome_filter = outcome_filter

    # ------------------------------------------------------------------------------------------ rows
    def row_eligible(self, rec: ExperienceRecord) -> Tuple[bool, str]:
        if rec.provenance != self.provenance:
            return False, f"provenance_{rec.provenance}"
        if rec.result is None:
            return False, "no_result"
        if rec.eligible.get("meta", True) is False:
            return False, "marked_ineligible"
        if self.require_propensities and not rec.decision.probabilities:
            return False, "no_propensities"
        return True, ""

    def _prepare(self, records: Sequence[ExperienceRecord]) -> Tuple[List[ExperienceRecord], Dict[str, str],
                                                                      Dict[str, int]]:
        """(eligible rows in lifetime order, exp_id -> merged group, dropped-reason counts)."""
        dropped: Dict[str, int] = {}
        rows = []
        for r in records:
            ok, why = self.row_eligible(r)
            if ok:
                rows.append(r)
            else:
                dropped[why] = dropped.get(why, 0) + 1
        rows.sort(key=lambda r: (r.decision.ts, r.seq, r.exp_id))
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        for r in rows:
            k = independence_key(r)
            find(k)
            for o in verified_outcomes(r, self.outcome_filter):
                union(k, "evt:" + (o.dedup_key or f"{o.query_id}|{o.observed_at!r}|{o.value!r}"))
        return rows, {r.exp_id: find(independence_key(r)) for r in rows}, dropped

    def _window(self, role: str, rows: Sequence[ExperienceRecord], group_of: Dict[str, str]) -> MetaWindow:
        ids = tuple(r.exp_id for r in rows)
        t0 = rows[0].decision.ts if rows else 0.0
        t1 = max((r.decision.ts + max(r.duration_s, 0.0) for r in rows), default=0.0)
        seen: List[str] = []
        for r in rows:
            g = group_of[r.exp_id]
            if g not in seen and verified_outcomes(r, self.outcome_filter):
                seen.append(g)
        return MetaWindow(_wid(role, ids, t0, t1), role, ids, t0, t1, tuple(seen),
                          tuple(group_of[r.exp_id] for r in rows))

    def _pair(self, support_rows: Sequence[ExperienceRecord], later_rows: Sequence[ExperienceRecord],
              group_of: Dict[str, str], earlier_groups: Set[str], dropped: Dict[str, int]) -> WindowPair:
        support = self._window("support", support_rows, group_of)
        drop = dict(dropped)
        q_rows = []
        for r in later_rows:
            if r.decision.ts < support.t_end + self.min_gap_s:
                drop["inside_gap"] = drop.get("inside_gap", 0) + 1
            elif group_of[r.exp_id] in earlier_groups:
                drop["straddles_boundary"] = drop.get("straddles_boundary", 0) + 1
            else:
                q_rows.append(r)
        query = self._window("query", q_rows, group_of)
        gap = (query.t_start - support.t_end) if q_rows and support_rows else 0.0
        if not support_rows:
            ok, reason = False, "empty_support_window"
        elif not q_rows:
            ok, reason = False, "empty_query_window"
        elif query.n_independent < self.min_independent:
            ok, reason = False, f"insufficient_independent_outcomes:{query.n_independent}<{self.min_independent}"
        else:
            ok, reason = True, ""
        pid = "p_" + hashlib.sha256((support.window_id + "|" + query.window_id).encode()).hexdigest()[:16]
        return WindowPair(pid, support, query, ok, reason, gap, tuple(sorted(drop.items())))

    # ------------------------------------------------------------------------------------------ cutting
    def cut(self, records: Sequence[ExperienceRecord], boundary_ts: float,
            start_ts: float = float("-inf"), end_ts: float = float("inf")) -> WindowPair:
        """Support = eligible decisions in [start_ts, boundary_ts); query = those in [boundary_ts, end_ts)."""
        rows, group_of, dropped = self._prepare(records)
        sup = [r for r in rows if start_ts <= r.decision.ts < boundary_ts]
        later = [r for r in rows if boundary_ts <= r.decision.ts < end_ts]
        earlier = {group_of[r.exp_id] for r in rows if r.decision.ts < boundary_ts}
        return self._pair(sup, later, group_of, earlier, dropped)

    def blocks(self, records: Sequence[ExperienceRecord]) -> List[List[ExperienceRecord]]:
        """Consecutive blocks of the lifetime, each closed as soon as it holds ``min_independent_outcomes``
        independent verified groups that started inside it. The last block may be short."""
        rows, group_of, _ = self._prepare(records)
        out: List[List[ExperienceRecord]] = []
        cur: List[ExperienceRecord] = []
        before: Set[str] = set()
        counted: Set[str] = set()
        cur_groups: Set[str] = set()
        for r in rows:
            cur.append(r)
            g = group_of[r.exp_id]
            cur_groups.add(g)
            if g not in before and verified_outcomes(r, self.outcome_filter):
                counted.add(g)
            if len(counted) >= self.min_independent:
                out.append(cur)
                before |= cur_groups
                cur, counted, cur_groups = [], set(), set()
        if cur:
            out.append(cur)
        return out

    def pairs(self, records: Sequence[ExperienceRecord]) -> List[WindowPair]:
        """(support = block j, query = block j+1) for every consecutive pair of blocks. Each block is the query of
        the update before it and the support of the update after it. Pairs that lack independent verified query
        outcomes are returned with ``eligible=False`` and the reason."""
        rows, group_of, dropped = self._prepare(records)
        blocks = self.blocks(records)
        out, earlier = [], set()
        for j in range(len(blocks) - 1):
            earlier |= {group_of[r.exp_id] for r in blocks[j]}
            out.append(self._pair(blocks[j], blocks[j + 1], group_of, set(earlier), dropped))
        return out


def held_out_groups(groups: Sequence[str], fraction: float = 0.25, salt: str = "") -> Set[str]:
    """Deterministic subset of independent groups reserved for promotion evidence (never used by the meta-update
    they evaluate). Membership depends only on the group key and ``salt``."""
    out = set()
    for g in groups:
        h = int(hashlib.sha256((salt + "|" + g).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if h < fraction:
            out.add(g)
    return out
