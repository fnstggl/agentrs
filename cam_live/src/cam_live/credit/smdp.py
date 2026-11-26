"""cam_live.credit.smdp — temporally extended actions (spec 12A.9; manifest CU-20).

Three timescales of action (``contracts.Level``): primitive operations, executable procedures, consequential
interventions. They are treated as a semi-Markov decision process:

* An intervention-level experience record keeps the procedure and primitive traces underneath it
  (:func:`attach_trace`), so low-level data stays available for procedural learning.
* Discounting is by elapsed time, ``exp(-beta * dt)``, so the value of a delayed outcome is the same whether the
  time is accounted for as one intervention or as the operations inside it (:func:`smdp_target` composes).
* **An outcome is credited once.** Whatever level the evidence points at, the counted credit of an outcome goes to
  the intervention at the top of that hierarchy; the procedures and primitives below it *inherit* it for
  procedural learning without being counted as further independent experiences (:class:`CreditLedger`). Fifty
  browser operations inside one intervention are one consequential experience, not fifty.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..contracts import (ActionResult, ActionSpec, Cost, ExperienceRecord, Level, OutcomeStatus, Record, ResultStatus)
from .links import OutcomeLink
from .replay import SqliteExperienceStore

NAME = "smdp"


# ------------------------------------------------------------------------------------------------ discounting
def discount(dt: float, beta: float) -> float:
    """Duration-aware discount exp(-beta * dt)."""
    return math.exp(-float(beta) * max(float(dt), 0.0))


def smdp_return(rewards: Sequence[float], times: Sequence[float], beta: float, t0: Optional[float] = None) -> float:
    """sum_k exp(-beta * (t_k - t0)) r_k for rewards realised at absolute times ``times`` (t0 = first time)."""
    if not rewards:
        return 0.0
    t0 = times[0] if t0 is None else t0
    return sum(discount(t - t0, beta) * r for r, t in zip(rewards, times))


def smdp_target(reward: float, duration_s: float, next_value: float, beta: float) -> float:
    """One semi-Markov backup: reward at the start of an action that lasts ``duration_s``, then the next value."""
    return reward + discount(duration_s, beta) * next_value


# ------------------------------------------------------------------------------------------------ traces
@dataclass
class TraceNode(Record):
    """One node of an extended action: the intervention, a procedure inside it, or a single operation."""
    level: str
    action_id: str = ""
    operation: str = ""
    modality: str = ""
    started: float = 0.0
    ended: float = 0.0
    status: str = ""
    termination: str = ""            # why it ended: "ok", a failure class, "timeout", ...
    n_primitive: int = 0             # primitive operations at or below this node
    op_indices: List[int] = field(default_factory=list)       # indices into ActionResult.ops owned directly
    children: List["TraceNode"] = field(default_factory=list)

    @property
    def elapsed_s(self) -> float:
        return max(self.ended - self.started, 0.0)

    def walk(self) -> List["TraceNode"]:
        out = [self]
        for c in self.children:
            out += c.walk()
        return out


def _termination(result: Optional[ActionResult]) -> Tuple[str, str]:
    if result is None:
        return "", ""
    if result.status == ResultStatus.OK:
        return result.status, "ok" if result.verified is not False else "postcondition_failed"
    return result.status, result.failure_class if result.failure_class != "none" else result.status


def build_trace(action: ActionSpec, result: Optional[ActionResult],
                step_results: Optional[Dict[str, ActionResult]] = None) -> TraceNode:
    """Trace of an executed (possibly compositional) action. ``step_results`` maps the ``action_id`` of a step to
    its own result; primitive operations are then assigned to the innermost step whose execution interval contains
    them. Without step results every operation is owned by the root (nothing is guessed)."""
    step_results = step_results or {}
    ops = list(result.ops) if result is not None else []

    def node(spec: ActionSpec, res: Optional[ActionResult]) -> TraceNode:
        status, term = _termination(res)
        n = TraceNode(level=spec.level, action_id=spec.action_id, operation=spec.operation, modality=spec.modality,
                      started=res.started if res else 0.0, ended=res.ended if res else 0.0, status=status,
                      termination=term)
        n.children = [node(s, step_results.get(s.action_id)) for s in spec.steps]
        return n

    root = node(action, result)
    owned = set()

    def assign(n: TraceNode) -> None:
        for c in n.children:
            assign(c)
        if n is not root and n.ended > n.started:
            for i, op in enumerate(ops):
                if i not in owned and n.started - 1e-9 <= op.started and op.ended <= n.ended + 1e-9:
                    n.op_indices.append(i)
                    owned.add(i)

    assign(root)
    root.op_indices = [i for i in range(len(ops)) if i not in owned]

    def count(n: TraceNode) -> int:
        n.n_primitive = len(n.op_indices) + sum(count(c) for c in n.children)
        return n.n_primitive

    count(root)
    return root


def attach_trace(rec: ExperienceRecord, step_results: Optional[Dict[str, ActionResult]] = None) -> TraceNode:
    """Store the procedure / primitive trace of ``rec``'s executed action in ``rec.decision.diagnostics['trace']``
    (JSON) and set ``rec.duration_s`` from the measured execution interval. The raw operations stay in
    ``rec.result.ops``."""
    trace = build_trace(rec.decision.chosen, rec.result, step_results)
    rec.decision.diagnostics["trace"] = trace.to_dict()
    if rec.result is not None and rec.result.ended >= rec.result.started:
        rec.duration_s = rec.result.ended - rec.result.started
    return trace


def get_trace(rec: ExperienceRecord) -> Optional[TraceNode]:
    d = rec.decision.diagnostics.get("trace")
    return TraceNode.from_dict(d) if d else None


def validate_intervention_record(rec: ExperienceRecord) -> List[str]:
    """Problems that make ``rec`` unusable as an intervention-level SMDP transition ([] = fine)."""
    out = []
    if rec.decision.level != Level.INTERVENTION:
        out.append(f"decision level is {rec.decision.level!r}, not intervention")
    if rec.result is None:
        return out + ["no result"]
    tr = get_trace(rec)
    if tr is None:
        return out + ["no trace attached"]
    if tr.n_primitive != len(rec.result.ops):
        out.append(f"trace covers {tr.n_primitive} primitive operations, the result has {len(rec.result.ops)}")
    spec_steps = {s.action_id for s in rec.decision.chosen.steps}
    if spec_steps - {c.action_id for c in tr.children}:
        out.append("procedure steps missing from the trace")
    if not rec.decision.probabilities:
        out.append("no policy probabilities recorded")
    if rec.duration_s < 0:
        out.append("negative duration")
    return out


@dataclass
class SMDPTransition:
    """What 12A.9 asks to be recorded for an extended action, read from one experience record."""
    exp_id: str
    level: str
    start_refs: List[str]            # start state: the context items H_t
    action: ActionSpec               # selected action or skill
    n_primitive: int                 # underlying operations
    n_procedures: int
    elapsed_s: float
    cost: Cost                       # resource consumption
    termination: str                 # termination condition
    status: str                      # actual result
    n_delayed_outcomes: int          # delayed consequences that have matured
    n_pending: int
    probabilities: List[float]       # relevant policy probabilities
    propensity: float

    def discount(self, beta: float) -> float:
        return discount(self.elapsed_s, beta)


def smdp_transition(rec: ExperienceRecord) -> SMDPTransition:
    tr = get_trace(rec) or build_trace(rec.decision.chosen, rec.result)
    res = rec.result
    matured = [o for o in rec.outcomes if o.status in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED)]
    resolved = {o.query_id for o in rec.outcomes if o.status != OutcomeStatus.PENDING}
    return SMDPTransition(
        exp_id=rec.exp_id, level=rec.decision.level, start_refs=list(rec.context_item_refs), action=rec.decision.chosen,
        n_primitive=tr.n_primitive, n_procedures=sum(1 for n in tr.walk() if n.level == Level.PROCEDURE and n is not tr),
        elapsed_s=rec.duration_s, cost=res.cost if res else Cost(), termination=tr.termination,
        status=res.status if res else "", n_delayed_outcomes=len(matured),
        n_pending=sum(1 for q in rec.pending_queries if q.query_id not in resolved),
        probabilities=list(rec.decision.probabilities), propensity=rec.decision.propensity)


# ------------------------------------------------------------------------------------------------ credit once
class DoubleCreditError(ValueError):
    """An outcome would be credited more than once."""


@dataclass
class LedgerEntry:
    outcome_key: str
    exp_id: str
    level: str
    weight: float
    counted: bool
    credit: Optional[float] = None


class CreditLedger:
    """Guarantees that one outcome is credited once across the three timescales.

    Decisions form a hierarchy through ``ActionSpec.parent_decision_id`` (a procedure chosen inside an
    intervention, an operation chosen inside a procedure). For every evidence link of an outcome the *counted*
    weight goes to the top ancestor of the linked decision; the linked decision and the ancestors in between hold
    *inherited*, uncounted entries. The store refuses any assignment that would raise the counted weight of an
    outcome above 1."""

    def __init__(self, store: SqliteExperienceStore, max_depth: int = 8):
        self.store = store
        self.max_depth = max_depth

    def chain(self, rec: ExperienceRecord) -> List[ExperienceRecord]:
        """``rec`` and its ancestors, bottom-up; the last element is the top (counted) decision."""
        out, seen = [rec], {rec.exp_id}
        cur = rec
        for _ in range(self.max_depth):
            pid = cur.decision.chosen.parent_decision_id
            parent = self.store.by_decision(pid) if pid else None
            if parent is None or parent.exp_id in seen:
                break
            out.append(parent)
            seen.add(parent.exp_id)
            cur = parent
        return out

    def assign(self, links: Sequence[OutcomeLink]) -> List[LedgerEntry]:
        """Enter the links of one or more outcomes. Idempotent for the same links. Raises
        :class:`DoubleCreditError` if an outcome's counted weight would exceed 1."""
        counted: Dict[Tuple[str, str], LedgerEntry] = {}
        inherited: Dict[Tuple[str, str], LedgerEntry] = {}
        for l in links:
            rec = self.store.get(l.exp_id)
            if rec is None:
                continue
            chain = self.chain(rec)
            top = chain[-1]
            k = (l.outcome_key, top.exp_id)
            e = counted.setdefault(k, LedgerEntry(l.outcome_key, top.exp_id, top.decision.level, 0.0, True, None))
            e.weight += l.weight
            if l.value is not None:
                e.credit = (e.credit or 0.0) + l.weight * l.value
            for r in chain[:-1]:
                ki = (l.outcome_key, r.exp_id)
                ie = inherited.setdefault(ki, LedgerEntry(l.outcome_key, r.exp_id, r.decision.level, 0.0, False, None))
                ie.weight += l.weight
                if l.value is not None:
                    ie.credit = (ie.credit or 0.0) + l.weight * l.value
        entries = list(counted.values()) + [e for k, e in inherited.items() if k not in counted]
        with self.store.batch():
            for e in entries:
                try:
                    self.store.ledger_assign(e.outcome_key, e.exp_id, e.level, e.weight, counted=e.counted,
                                             note="" if e.counted else "inherited")
                except ValueError as err:
                    raise DoubleCreditError(str(err)) from err
        return entries

    def assign_direct(self, outcome_key: str, exp_id: str, weight: float, *, counted: bool = True) -> None:
        """Enter one assignment by hand (same guarantee)."""
        rec = self.store.get(exp_id)
        if rec is None:
            raise KeyError(exp_id)
        try:
            self.store.ledger_assign(outcome_key, exp_id, rec.decision.level, weight, counted=counted,
                                     note="" if counted else "inherited")
        except ValueError as err:
            raise DoubleCreditError(str(err)) from err

    def counted_weight(self, outcome_key: str) -> float:
        return sum(e["weight"] for e in self.store.ledger(outcome_key) if e["counted"])

    def by_level(self, outcome_key: str) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for e in self.store.ledger(outcome_key):
            d = out.setdefault(e["level"], {"counted": 0.0, "inherited": 0.0})
            d["counted" if e["counted"] else "inherited"] += e["weight"]
        return out

    def independent_experiences(self, outcome_keys: Optional[Sequence[str]] = None) -> int:
        """Number of independent consequential experiences behind the ledger: distinct (outcome, top decision)
        pairs with counted weight — not the number of records or operations."""
        rows = [e for e in self.store.ledger() if e["counted"] and (outcome_keys is None or e["outcome_key"] in outcome_keys)]
        return len({(e["outcome_key"], e["exp_id"]) for e in rows})

    def apply(self, entries: Sequence[LedgerEntry], name: str = NAME) -> int:
        """Write credit to the records: counted credit under ``name``, inherited credit under
        ``name + '_inherited'`` (a signal for procedural learning that is never summed with the counted one)."""
        tot: Dict[Tuple[str, bool], float] = {}
        for e in entries:
            if e.credit is not None:
                tot[(e.exp_id, e.counted)] = tot.get((e.exp_id, e.counted), 0.0) + e.credit
        with self.store.batch():
            for (exp_id, counted), v in tot.items():
                self.store.set_credit(exp_id, name if counted else name + "_inherited", v)
        return len(tot)
