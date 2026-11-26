"""cam_live.credit.simple — the delayed-outcome attribution used first in live operation (spec 9.4).

Three plain tools, no learned model:

* :func:`credit_from_links` — a decision's credit is the sum of the outcome amounts that evidence links to it
  (``credit.links``), with duration-aware discounting ``exp(-beta * dt)`` and the link uncertainty carried along.
* :func:`time_decayed_attribution` — when all that is known is that an outcome fell inside a window, it is spread
  over the decisions in the window with a declared decay and an explicit share left unattributed.
* :func:`arm_difference` — the difference between randomised arms with a Welch confidence interval. This is the only
  quantity in the credit package that is allowed to be called an experimentally supported effect.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..contracts import ExperienceRecord, OutcomeStatus, Record
from .links import OutcomeLink, record_arm
from .replay import SqliteExperienceStore, outcome_key

NAME = "simple"


def discount(dt: float, beta: float) -> float:
    """Duration-aware discount ``exp(-beta * dt)`` (dt in seconds, never negative)."""
    return math.exp(-beta * max(float(dt), 0.0))


@dataclass
class CreditValue:
    value: float
    uncertainty: float               # standard deviation from the link uncertainties (and measurement error)
    n_links: int = 0
    by_evidence: Dict[str, float] = field(default_factory=dict)   # every linked amount, including the ones not summed
    provisional: bool = False        # True while the value rests on event links only (no declared measurement yet)


_DECLARED = ("declared_query", "shared_event")


def credit_from_links(links: Sequence[OutcomeLink], *, beta: float = 0.0, goal_relevant_only: bool = True,
                      value_fn: Optional[Callable[[OutcomeLink], Optional[float]]] = None,
                      combine: str = "declared_first") -> Dict[str, CreditValue]:
    """exp_id -> credit. Each link contributes ``weight * value * exp(-beta * (t_outcome - t_decision))``; its
    variance is ``(value * discount * uncertainty)^2`` plus the measurement variance. Links without a measured value
    contribute nothing (an occurrence is not a number).

    combine="declared_first" (default): once a decision's own declared queries have been answered, those answers
    are its credit and event links are only reported in ``by_evidence`` — events are usually early partial views of
    the same consequences, and summing both would count them twice. Before that, event links give a *provisional*
    value. combine="sum" adds everything (only correct when events and queries measure different things)."""
    if combine not in ("declared_first", "sum"):
        raise ValueError(combine)
    per: Dict[str, List[Tuple[OutcomeLink, float, float]]] = {}
    for l in links:
        if goal_relevant_only and not l.goal_relevant:
            continue
        v = value_fn(l) if value_fn is not None else l.value
        if v is None:
            continue
        per.setdefault(l.exp_id, []).append((l, float(v), discount(l.outcome_ts - l.decision_ts, beta)))
    out: Dict[str, CreditValue] = {}
    for e, rows in per.items():
        declared = any(l.evidence in _DECLARED for l, _, _ in rows)
        cv = CreditValue(0.0, 0.0, provisional=not declared)
        var = 0.0
        for l, v, g in rows:
            cv.by_evidence[l.evidence] = cv.by_evidence.get(l.evidence, 0.0) + l.weight * v * g
            if combine == "declared_first" and declared and l.evidence not in _DECLARED:
                continue
            cv.value += l.weight * v * g
            cv.n_links += 1
            meas = float(l.detail.get("measurement_uncertainty", 0.0) or 0.0)
            var += (v * g * l.uncertainty) ** 2 + (l.weight * g * meas) ** 2
        cv.uncertainty = math.sqrt(var)
        out[e] = cv
    return out


def time_decayed_attribution(outcome_ts: float, candidates: Sequence[Tuple[str, float]], *, window_s: float,
                             half_life_s: float, background: float = 0.0) -> Dict[str, Tuple[float, float]]:
    """Spread one outcome over the decisions ``(exp_id, decision_ts)`` made within ``window_s`` before it.

    weight_i = k_i / (sum_j k_j + background), k_i = 2^(-dt_i / half_life_s). ``background`` is the kernel mass given
    to 'no decision caused it', so the weights sum to less than 1 when it is positive. Decisions after the outcome
    or outside the window get nothing. Returns exp_id -> (weight, standard deviation of the weight)."""
    k = {}
    for exp_id, t in candidates:
        dt = outcome_ts - t
        if 0.0 <= dt <= window_s:
            k[exp_id] = 2.0 ** (-dt / half_life_s)
    tot = sum(k.values()) + max(background, 0.0)
    if tot <= 0:
        return {}
    return {e: (v / tot, math.sqrt(v / tot * (1 - v / tot))) for e, v in k.items()}


# ------------------------------------------------------------------------------------------------ randomised arms
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularised incomplete beta function (Lentz)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(ln) * _betacf(a, b, x) / a
    return 1.0 - math.exp(ln) * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    p = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - p if t > 0 else p


def student_t_ppf(p: float, df: float) -> float:
    """Quantile of Student's t by bisection on the exact CDF (no scipy at run time)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    lo, hi = -1.0, 1.0
    while student_t_cdf(lo, df) > p:
        lo *= 2.0
    while student_t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class ArmDifference(Record):
    """Difference in mean outcome between two arms of an experiment (treatment - control), Welch interval."""
    treatment: str
    control: str
    diff: float
    se: float
    ci_low: float
    ci_high: float
    confidence: float
    n_treatment: int
    n_control: int
    df: float
    mean_treatment: float
    mean_control: float
    randomized: bool = False         # True only when assignment to arms was randomised (then: experimental evidence)
    experiment_id: str = ""
    metric: str = ""

    @property
    def significant(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


def arm_difference(treatment: Sequence[float], control: Sequence[float], *, confidence: float = 0.95,
                   randomized: bool = False, treatment_name: str = "treatment", control_name: str = "control",
                   experiment_id: str = "", metric: str = "") -> Optional[ArmDifference]:
    """Welch's unequal-variance interval for mean(treatment) - mean(control). ``None`` when either arm has fewer than
    two independent units (no interval can be stated)."""
    a, b = [float(x) for x in treatment], [float(x) for x in control]
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    sa, sb = va / len(a), vb / len(b)
    se = math.sqrt(sa + sb)
    if se == 0.0:
        df, half = float(len(a) + len(b) - 2), 0.0
    else:
        df = (sa + sb) ** 2 / (sa ** 2 / (len(a) - 1) + sb ** 2 / (len(b) - 1))
        half = student_t_ppf(0.5 + confidence / 2.0, df) * se
    d = ma - mb
    return ArmDifference(treatment_name, control_name, d, se, d - half, d + half, confidence, len(a), len(b), df, ma, mb,
                         randomized, experiment_id, metric)


def record_outcome_value(rec: ExperienceRecord, *, metric_prefix: str = "", goal_relevant_only: bool = True
                         ) -> Optional[float]:
    """Sum of a record's matured (observed) outcome values, each event counted once; ``None`` if nothing matured."""
    queries = {q.query_id: q for q in rec.pending_queries}
    if rec.result is not None:
        for q in rec.result.expected_outcomes:
            queries.setdefault(q.query_id, q)
    tot, seen, any_ = 0.0, set(), False
    for o in rec.outcomes:
        q = queries.get(o.query_id)
        if o.status != OutcomeStatus.OBSERVED or o.value is None or q is None or outcome_key(o) in seen:
            continue
        if (goal_relevant_only and not q.goal_relevant) or not q.metric.startswith(metric_prefix):
            continue
        seen.add(outcome_key(o))
        tot += float(o.value)
        any_ = True
    return tot if any_ else None


def experiment_effects(store: SqliteExperienceStore, experiment_id: str, *, control: Optional[str] = None,
                       confidence: float = 0.95, metric_prefix: str = "",
                       value_fn: Optional[Callable[[ExperienceRecord], Optional[float]]] = None) -> List[ArmDifference]:
    """Arm differences of a registered experiment from the matured outcomes of its decisions. One unit = one
    decision of the experiment. ``randomized`` is True only if the experiment record says so *and* every decision
    was taken by explicit randomised assignment with a logged propensity."""
    recs = store.experiment_records(experiment_id)
    exp = store.experiment(experiment_id)
    by_arm: Dict[str, List[float]] = {}
    randomized = exp is not None and exp.assignment == "randomized"
    for r in recs:
        arm = record_arm(r)
        v = value_fn(r) if value_fn is not None else record_outcome_value(r, metric_prefix=metric_prefix)
        if arm is None or v is None:
            continue
        by_arm.setdefault(arm, []).append(v)
        randomized = randomized and r.decision.exploration == "randomized" and bool(r.decision.probabilities)
    arms = sorted(by_arm)
    if len(arms) < 2:
        return []
    control = control if control is not None else (exp.comparison if exp is not None and exp.comparison in by_arm
                                                   else arms[0])
    out = []
    for arm in arms:
        if arm == control:
            continue
        d = arm_difference(by_arm[arm], by_arm[control], confidence=confidence, randomized=randomized,
                           treatment_name=arm, control_name=control, experiment_id=experiment_id, metric=metric_prefix)
        if d is not None:
            out.append(d)
    return out


class SimpleAttribution:
    """The default live estimator: evidence links -> credit, written to ``record.credit['simple']``.

    A record without any linked outcome gets *no* entry (absence of evidence is not zero credit)."""
    name = NAME

    def __init__(self, store: SqliteExperienceStore, *, beta: float = 0.0, goal_relevant_only: bool = True,
                 value_fn: Optional[Callable[[OutcomeLink], Optional[float]]] = None, combine: str = "declared_first"):
        self.store, self.beta, self.goal_relevant_only, self.value_fn = store, float(beta), goal_relevant_only, value_fn
        self.combine = combine

    def credit(self, exp_ids: Optional[Sequence[str]] = None) -> Dict[str, CreditValue]:
        if exp_ids is None:
            raw = self.store.links()
        else:
            raw = [l for e in exp_ids for l in self.store.links(exp_id=e)]
        return credit_from_links([OutcomeLink.from_dict(l) for l in raw], beta=self.beta,
                                 goal_relevant_only=self.goal_relevant_only, value_fn=self.value_fn,
                                 combine=self.combine)

    def apply(self, exp_ids: Optional[Sequence[str]] = None) -> Dict[str, CreditValue]:
        """Compute and store the credit. Returns what was written."""
        cr = self.credit(exp_ids)
        with self.store.batch():
            for e, cv in cr.items():
                self.store.set_credit(e, self.name, cv.value, cv.uncertainty)
        return cr
