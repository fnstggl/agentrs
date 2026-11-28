"""cam_live.credit.links — attaching delayed outcomes and events to the earlier decisions that evidence points to
(spec 9.1; manifest RC-01, RC-06, RC-07).

Rules, each enforced in code and tested:

* **Evidence only.** A link exists because the outcome itself names the decision (its declared query, an attribution
  key such as ``by_action`` in the event data, an artifact the decision created, the arm of a registered experiment)
  or because the decision declared an attribution window for that metric. Without evidence there is no link.
* **Ambiguity is kept.** When several decisions are supported equally the outcome is split into fractional links,
  each with a stated standard deviation; the weights of one outcome never sum to more than 1 (``_make`` enforces
  it), so the credit given for an outcome never exceeds the outcome.
* **Amounts are validated.** An attribution map (``by_action`` / ``by_decision``) is used as shares only when every
  amount is a finite number >= 0; a map with a negative, non-finite or non-numeric amount (e.g. a net change with
  departures) is treated as ambiguous: the named decisions share the outcome equally, with uncertainty.
* **Never "the latest action did it".** Recency is not evidence: equal support gives equal weights.
* **No future leakage.** A decision made after the outcome time is refused as a cause, and linking never touches the
  decision's context ``H_t``; each link records what was available at decision time.
* **Retrospective updates.** When an outcome matures, the records it is linked to are queued for re-training.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..contracts import (Clock, ExperienceRecord, Observation, OutcomeObservation, OutcomeStatus, Record, WallClock,
                         new_id)
from .replay import SqliteExperienceStore, outcome_key

EV_QUERY = "declared_query"              # the decision registered the query this observation answers
EV_KEY = "attribution_key"               # the event data names the action / decision
EV_PARENT = "parent_action"              # the event was produced by executing that action
EV_ARTIFACT = "artifact"                 # the event refers to an artifact the decision created
EV_ARM = "experiment_arm"                # the event belongs to an arm of a registered experiment
EV_WINDOW = "attribution_window"         # the decision declared an attribution window for this metric
EV_SHARED = "shared_event"               # one event answers the declared queries of several decisions
#: evidence that identifies the decision exactly (weight uncertainty 0 when it names a single decision)
EXACT = (EV_QUERY, EV_KEY, EV_PARENT)


@dataclass
class OutcomeLink(Record):
    link_id: str
    outcome_key: str
    exp_id: str
    decision_id: str
    weight: float                    # fraction of the outcome attributed to this decision
    uncertainty: float               # standard deviation of that fraction (0 = the evidence names the decision)
    evidence: str
    outcome_ts: float
    decision_ts: float
    value: Optional[float] = None    # the measured quantity of the whole outcome (None = occurrence only)
    metric: str = ""
    verified: bool = False
    goal_relevant: bool = True
    created_at: float = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)
    available_at_decision: Dict[str, Any] = field(default_factory=dict)

    @property
    def amount(self) -> Optional[float]:
        return None if self.value is None else self.weight * self.value


@dataclass
class LinkResult:
    outcome_key: str
    evidence: str = ""
    links: List[OutcomeLink] = field(default_factory=list)
    refused: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    @property
    def unattributed(self) -> float:
        return max(0.0, 1.0 - sum(l.weight for l in self.links))


@dataclass
class ReconcileReport:
    outcomes_accepted: int = 0
    duplicates: int = 0
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    results: List[LinkResult] = field(default_factory=list)
    retrain_marked: List[str] = field(default_factory=list)

    @property
    def links(self) -> List[OutcomeLink]:
        return [l for r in self.results for l in r.links]


def attribution_amounts(m: Dict[Any, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Validate an attribution map ``{action or decision id: amount}``. Returns (valid amounts, malformed entries).
    An amount is valid when it is a finite number >= 0 (booleans and strings are not amounts)."""
    ok: Dict[str, float] = {}
    bad: Dict[str, Any] = {}
    for k, v in m.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            bad[str(k)] = repr(v)
            continue
        x = float(v)
        if not math.isfinite(x) or x < 0.0:
            bad[str(k)] = x if math.isfinite(x) else repr(v)
        else:
            ok[str(k)] = x
    return ok, bad


def split_uncertainty(w: float) -> float:
    """Standard deviation of the indicator 'this decision is the cause' when its probability is ``w``."""
    return math.sqrt(max(w * (1.0 - w), 0.0))


class OutcomeLinker:
    """Evidence-based linker over a :class:`SqliteExperienceStore`.

    retro_predecessors   also queue this many earlier decisions of the same objective when an outcome matures (their
                         return-prediction targets changed); 0 = only the linked decisions
    window_half_life_s   optional *declared* decay of an attribution window (part of the measurement protocol). With
                         ``None`` (default) the decisions whose windows cover an outcome share it equally.
    """

    def __init__(self, store: SqliteExperienceStore, *, clock: Optional[Clock] = None, retro_predecessors: int = 0,
                 window_half_life_s: Optional[float] = None):
        self.store = store
        self.clock = clock or WallClock()
        self.retro_predecessors = int(retro_predecessors)
        self.window_half_life_s = window_half_life_s
        self._artifact_seq = 0
        self._artifacts: Dict[str, List[str]] = {}

    # ---------------------------------------------------------------------------------------- decision-time info
    def available_at_decision(self, rec: ExperienceRecord) -> Dict[str, Any]:
        """What the decision could know: its context items and their times. ``leak_refs`` lists context items dated
        after the decision (there must be none)."""
        times = self.store.item_times(rec.context_item_refs)
        ts = rec.decision.ts
        leak = sorted(r for r, t in times.items() if t > ts + 1e-9)
        return {"decision_ts": ts, "n_context_items": len(rec.context_item_refs),
                "latest_context_ts": max(times.values()) if times else None,
                "context_obs_ids": list(rec.decision.context_obs_ids), "model_version": rec.decision.model_version,
                "n_outcomes_known": sum(1 for o in rec.outcomes if (o.observed_at or 0.0) <= ts), "leak_refs": leak}

    def audit_leakage(self, exp_ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        """Records whose stored context contains information dated after the decision. Empty = no leakage."""
        recs = self.store.get_many(exp_ids) if exp_ids is not None else self.store.iter_records()
        out = []
        for rec in recs:
            info = self.available_at_decision(rec)
            if info["leak_refs"]:
                out.append({"exp_id": rec.exp_id, "decision_ts": rec.decision.ts, "leak_refs": info["leak_refs"]})
        return out

    # ---------------------------------------------------------------------------------------- building links
    def _make(self, key: str, shares: Sequence[Tuple[ExperienceRecord, float, Optional[float]]], evidence: str,
              outcome_ts: float, value: Optional[float], metric: str, verified: bool, goal_relevant: bool,
              detail: Dict[str, Any], refused: List[Dict[str, Any]]) -> LinkResult:
        """shares: (record, weight, uncertainty or None). Decisions made after the outcome are refused here, and
        their share is left unattributed (never handed to another decision)."""
        res = LinkResult(outcome_key=key, evidence=evidence, refused=list(refused))
        now = self.clock.now()
        bad = [w for _, w, _ in shares if not (isinstance(w, (int, float)) and math.isfinite(w) and w >= 0.0)]
        if bad:
            raise ValueError(f"attribution weights must be finite and non-negative, got {bad[:4]}")
        total = sum(w for _, w, _ in shares)
        if total > 1.0 + 1e-9:                   # a share of one outcome can never exceed the outcome
            shares = [(r, w / total, u) for r, w, u in shares]
            detail = {**detail, "weights_renormalised_from": total}
        if value is not None and not math.isfinite(float(value)):
            res.refused.append({"reason": "non_finite_outcome_value", "value": value})
            res.reason = "the outcome value is not finite"
            return res
        for rec, w, unc in shares:
            if rec.decision.ts > outcome_ts + 1e-9:
                res.refused.append({"exp_id": rec.exp_id, "reason": "decision_after_outcome", "weight": w,
                                    "decision_ts": rec.decision.ts, "outcome_ts": outcome_ts})
                continue
            if w <= 0:
                continue
            res.links.append(OutcomeLink(
                link_id=new_id("link"), outcome_key=key, exp_id=rec.exp_id, decision_id=rec.decision.decision_id,
                weight=float(w), uncertainty=float(split_uncertainty(w) if unc is None else unc), evidence=evidence,
                outcome_ts=float(outcome_ts), decision_ts=float(rec.decision.ts), value=value, metric=metric,
                verified=verified, goal_relevant=goal_relevant, created_at=now, detail=dict(detail),
                available_at_decision=self.available_at_decision(rec)))
        if not res.links:
            res.reason = res.reason or ("all candidate decisions were refused" if res.refused else "no evidence")
        return res

    def link_outcome(self, obs: OutcomeObservation) -> LinkResult:
        """Link the answer to a declared query to the decision that registered it. When the same event (dedup key)
        answers queries of several decisions it is shared between them, not counted once per decision."""
        key = outcome_key(obs)
        if obs.status not in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED):
            return LinkResult(key, reason=f"status {obs.status} is not a matured outcome")
        q = self.store.query(obs.query_id)
        owners = [e for (e,) in self.store._q("SELECT DISTINCT exp_id FROM outcomes WHERE dedup_key=? ORDER BY exp_id",
                                              (key,))]
        if q is None or not owners:
            return LinkResult(key, reason="no decision registered this query")
        recs = self.store.get_many(owners)
        at = obs.observed_at or self.clock.now()         # an observation without a time was received now
        ok = [r for r in recs if r.decision.ts <= at + 1e-9]
        w = 1.0 / len(ok) if ok else 0.0
        evidence = EV_QUERY if len(recs) == 1 else EV_SHARED
        return self._make(key, [(r, w if r in ok else 1.0 / len(recs), 0.0 if len(ok) == 1 else None) for r in recs],
                          evidence, at, obs.value, q.metric, obs.verified, q.goal_relevant,
                          {"query_id": obs.query_id, "status": obs.status, "source": obs.source,
                           "measurement_uncertainty": obs.uncertainty}, [])

    def _artifact_owners(self, artifact: str) -> List[str]:
        head = self.store.head_seq()
        if head > self._artifact_seq:
            for rec in self.store.iter_records(since_seq=self._artifact_seq):
                if rec.result is not None:
                    for ref in rec.result.artifacts.values():
                        self._artifacts.setdefault(ref, []).append(rec.exp_id)
            self._artifact_seq = head
        return list(self._artifacts.get(artifact, []))

    def link_event(self, event: Observation) -> LinkResult:
        """Link an environment event to earlier decisions using the strongest evidence it carries, in this order:
        attribution keys in ``event.data`` (``by_action`` / ``by_decision`` maps, ``action_id`` / ``decision_id``),
        the action that produced it (``parent_action_id``), artifact references, an experiment arm, and finally
        the attribution windows that decisions declared for the event's metric. No evidence = no link."""
        d = event.data or {}
        key = str(d.get("dedup_key") or f"ev:{event.obs_id}")
        ts = float(event.ts)
        qty = {q.name: q.value for q in event.quantities}
        metric = str(d.get("metric") or (event.quantities[0].name if event.quantities else ""))
        value = d.get("value", qty.get(metric)) if metric else d.get("value")
        try:
            value = None if value is None else float(value)
        except (TypeError, ValueError):
            value = float("nan")                 # refused in _make: a malformed value is not an outcome
        verified = event.provenance.trust == "verifier"
        st, refused = self.store, []

        def find(kind: str, ident: str) -> Optional[ExperienceRecord]:
            rec = st.by_action(ident) if kind == "action" else st.by_decision(ident)
            if rec is None:
                refused.append({kind + "_id": ident, "reason": f"unknown_{kind}"})
            return rec

        # 1. attribution keys
        for field_name, kind in (("by_action", "action"), ("by_decision", "decision")):
            m = d.get(field_name)
            if isinstance(m, dict) and m:
                amounts, malformed = attribution_amounts(m)
                if malformed:
                    # negative / non-finite / non-numeric amounts are not shares of this outcome: the map only says
                    # which decisions are involved, so they split it equally, with the uncertainty of ambiguity
                    recs = [r for r in (find(kind, str(i)) for i in sorted(m)) if r is not None]
                    if not recs:
                        continue
                    return self._make(key, [(r, 1.0 / len(recs), None) for r in recs], EV_KEY, ts, value, metric,
                                      verified, True, {"field": field_name, "malformed_amounts": malformed,
                                                       "treated_as": "ambiguous: equal shares"}, refused)
                total = sum(amounts.values())
                if total <= 0:
                    continue
                if math.isfinite(total):
                    scaled, denom = amounts, total
                else:
                    # amounts whose sum overflows (1e308 + 1e308) are shares relative to the largest one (G2L-3);
                    # the sum itself is the outcome value only when the event states none, and an infinite sum is
                    # then refused explicitly by _make ("the outcome value is not finite")
                    peak = max(amounts.values())
                    scaled = {i: a / peak for i, a in amounts.items()}
                    denom = math.fsum(scaled.values())
                shares = []
                for ident, amount in sorted(scaled.items()):
                    rec = find(kind, str(ident))
                    if rec is not None and amount > 0:
                        shares.append((rec, amount / denom, 0.0))
                return self._make(key, shares, EV_KEY, ts, total if value is None else value, metric, verified, True,
                                  {"field": field_name, "amounts": amounts}, refused)
        for field_name, kind in (("action_id", "action"), ("decision_id", "decision")):
            if d.get(field_name):
                rec = find(kind, str(d[field_name]))
                return self._make(key, [(rec, 1.0, 0.0)] if rec else [], EV_KEY, ts, value, metric, verified, True,
                                  {"field": field_name}, refused)
        # 2. the action whose execution produced the event
        if event.parent_action_id:
            rec = find("action", event.parent_action_id)
            return self._make(key, [(rec, 1.0, 0.0)] if rec else [], EV_PARENT, ts, value, metric, verified, True,
                              {"parent_action_id": event.parent_action_id}, refused)
        # 3. artifacts
        arts = list(d.get("artifact_ids") or []) + ([d["artifact_id"]] if d.get("artifact_id") else []) + \
            list(event.refs.values())
        owners: List[str] = []
        for a in arts:
            for e in self._artifact_owners(str(a)):
                if e not in owners:
                    owners.append(e)
        if owners:
            recs = [r for r in st.get_many(owners)]
            ok = [r for r in recs if r.decision.ts <= ts + 1e-9]
            w = 1.0 / len(ok) if ok else 0.0
            return self._make(key, [(r, w if r in ok else 1.0 / len(recs), 0.0 if len(ok) == 1 else None) for r in recs],
                              EV_ARTIFACT, ts, value, metric, verified, True, {"artifacts": [str(a) for a in arts]}, refused)
        # 4. experiment arm
        if d.get("experiment_id") and d.get("arm") is not None:
            arm = str(d["arm"])
            recs = [r for r in st.experiment_records(str(d["experiment_id"])) if record_arm(r) == arm]
            ok = [r for r in recs if r.decision.ts <= ts + 1e-9]
            w = 1.0 / len(ok) if ok else 0.0
            return self._make(key, [(r, w if r in ok else 1.0 / max(len(recs), 1), 0.0 if len(ok) == 1 else None)
                                    for r in recs], EV_ARM, ts, value, metric, verified, True,
                              {"experiment_id": str(d["experiment_id"]), "arm": arm}, refused)
        # 5. declared attribution windows for this metric
        if metric:
            cands = st.queries_for_metric(metric, ts)
            recs = {r.exp_id: r for r in st.get_many([e for e, _ in cands])}
            pairs = [(recs[e], q) for e, q in cands if e in recs and recs[e].decision.ts <= ts + 1e-9]
            if pairs:
                if self.window_half_life_s:
                    k = [2.0 ** (-(ts - q.issued_at) / self.window_half_life_s) for _, q in pairs]
                else:
                    k = [1.0] * len(pairs)       # equal support -> equal shares; recency is not evidence
                tot = sum(k)
                return self._make(key, [(r, ki / tot, None) for (r, _), ki in zip(pairs, k)], EV_WINDOW, ts, value,
                                  metric, verified, True,
                                  {"windows": [[q.query_id, q.issued_at, q.due_at] for _, q in pairs],
                                   "half_life_s": self.window_half_life_s}, refused)
        return LinkResult(key, refused=refused, reason="no evidence links this event to an earlier decision")

    # ---------------------------------------------------------------------------------------- reconciliation
    def _persist(self, res: LinkResult, report: ReconcileReport, reason: str) -> None:
        st = self.store
        with st.batch():
            st._db.execute("DELETE FROM links WHERE outcome_key=?", (res.outcome_key,))     # re-split replaces
            st.put_links([l.to_dict() for l in res.links])
            affected: List[str] = []
            for l in res.links:
                for e in [l.exp_id] + st.predecessor_ids(l.exp_id, self.retro_predecessors):
                    if e not in affected:
                        affected.append(e)
            linked = {l.exp_id for l in res.links}
            now = self.clock.now()
            st.mark_for_retraining([e for e in affected if e in linked], reason, outcome_key=res.outcome_key, now=now)
            st.mark_for_retraining([e for e in affected if e not in linked], "return_changed",
                                   outcome_key=res.outcome_key, now=now)
        report.results.append(res)
        report.retrain_marked += [e for e in affected if e not in report.retrain_marked]

    def reconcile(self, outcomes: Sequence[OutcomeObservation] = (), events: Sequence[Observation] = ()
                  ) -> ReconcileReport:
        """Store matured outcomes (de-duplicated), link outcomes and events to earlier decisions, persist the links
        and queue the affected earlier records for re-training. Idempotent for repeated outcomes / events."""
        rep = ReconcileReport()
        for obs in outcomes:
            r = self.store.add_outcome(obs)
            if r.duplicate:
                rep.duplicates += 1
                continue
            if not r.accepted:
                rep.rejected.append({"query_id": obs.query_id, "reason": r.reason})
                continue
            rep.outcomes_accepted += 1
            if obs.status in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED):
                self._persist(self.link_outcome(obs), rep, "outcome_matured")
        for ev in events:
            key = str((ev.data or {}).get("dedup_key") or f"ev:{ev.obs_id}")
            if self.store.links(outcome_key=key):
                rep.duplicates += 1
                continue
            self._persist(self.link_event(ev), rep, "event_linked")
        return rep


def record_arm(rec: ExperienceRecord) -> Optional[str]:
    """The experiment arm a decision was assigned to: ``decision.diagnostics['arm']``, else the executed action's
    ``arguments['arm']``."""
    arm = rec.decision.diagnostics.get("arm", rec.decision.chosen.arguments.get("arm"))
    return None if arm is None or arm == "" else str(arm)
