"""Which metrics an in-world agreement may read (charter conditions, commitment predicates).

A predicate written by participants is checked by the world, and its outcome changes what happens (a rule applies,
money is released). If it could read an evaluator metric, the outcome would reveal the evaluator's hidden analysis
(related parties, circular flows, independence) - an in-world oracle (audit F, probe_oracle). So an in-world
predicate may read only metrics that

* declare whose data they read (`goals.metrics.METRIC_SUBJECTS`) and name a party of the agreement
  (`check_predicate_scope`), and
* are not computed with hidden evaluator analysis: `EVALUATOR_ONLY` below (the F1 recurring-revenue metrics carry
  no subject at all; the 4.0 metrics listed here carry one for 4.0 contract deliverables but embed the 4.0
  relatedness and circular-flow analysis).

Booked metrics (`booked_mrr_cents`, `booked_subscribers`, `booked_revenue_cents`) are the merchant's own records and
stay readable. Workstream L marks evaluator-only tools and fields in the kernel registry (D-19); this list is the
commerce side of that marking and is merged into it (integration note).
"""
from __future__ import annotations

from .revenue import EVALUATOR_ONLY as _F1

EVALUATOR_ONLY = frozenset(_F1 | {"independent_revenue", "revenue_from", "capability_adopters"})


def in_world_errors(world, pred, parties) -> list[str]:
    """Errors if an in-world predicate reads an evaluator metric or anything that is not its parties' business."""
    from ..goals.metrics import check_predicate_scope, predicate_leaves
    errs = [f"metric {lf.get('metric')!r} is evaluator-only and cannot be read in the world"
            for lf in predicate_leaves(pred) if lf.get("metric") in EVALUATOR_ONLY]
    return errs + check_predicate_scope(world, pred, parties)
