"""cam_live.credit.select — the training-default credit estimator is chosen by evidence (spec 9.4; RC-08).

Live operation starts with the simple delayed-outcome attribution. A learned estimator (RUDDER, COCOA) becomes the
default only after its estimates of arm differences have been compared with the differences actually measured in
*randomised* experiments and found to agree; if later evidence contradicts it, the default falls back. Every
selection, with the evidence it rests on, is appended to a JSON-lines log.

Validity of the comparison (fix of G-cl M1):

* **Held out by lifetime.** Only evidence whose units (lifetimes) were not used to train the estimators is scored
  (``ExperimentEvidence.held_out``; ``CreditAssigner.compare_with_experiment`` excludes every lifetime any estimator
  or the learner trained on and records how many). Evidence of unknown provenance (``held_out`` None) or in-sample
  evidence is kept for the record but never scored (``SelectionRecord.excluded_not_held_out``).
* **Paired uncertainty.** The estimate and the measured difference are computed on the same units, so their noise is
  shared. The z-score uses the standard error of the *paired* per-unit contrast (``paired_comparison``: estimate
  minus measurement as a sum of independent per-unit terms, variance estimated within each arm). Because the
  estimator is a fixed function of independent training lifetimes, its own training variance is not in that
  standard error: the test is conservative (an estimator that varies between training sets fails more often).
* **No silent zero.** A comparison whose standard error is zero or undefined cannot be scored: it counts as
  ``n_refused`` and an estimator with any refused comparison does not pass.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..contracts import Clock, Record, WallClock
from .simple import NAME as SIMPLE, ArmDifference, arm_difference, experiment_effects, record_outcome_value
from .links import record_arm


@dataclass
class ExperimentEvidence(Record):
    """One randomised comparison and what each estimator said about it."""
    experiment_id: str
    treatment: str
    control: str
    diff: float                       # measured difference (treatment - control)
    se: float
    ci_low: float
    ci_high: float
    n_treatment: int
    n_control: int
    randomized: bool
    estimates: Dict[str, float] = field(default_factory=dict)       # estimator -> its estimate of the difference
    estimate_se: Dict[str, float] = field(default_factory=dict)     # spread of the estimate across units (info only)
    held_out: Optional[bool] = None   # True: no unit was used to train the estimators; None: unknown (never scored)
    n_excluded_trained: int = 0       # units left out because an estimator / the learner trained on their lifetime
    paired_diff: Dict[str, float] = field(default_factory=dict)     # estimator -> estimate - measured, same units
    paired_se: Dict[str, float] = field(default_factory=dict)       # standard error of that paired difference
    paired_n: Dict[str, int] = field(default_factory=dict)          # units the paired comparison used
    holdout: Dict[str, Any] = field(default_factory=dict)           # how the split was made (recorded, not scored)

    @property
    def key(self) -> str:
        return f"{self.experiment_id}:{self.treatment}-{self.control}"

    @property
    def significant(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @classmethod
    def from_difference(cls, d: ArmDifference, estimates: Optional[Dict[str, float]] = None,
                        estimate_se: Optional[Dict[str, float]] = None, **kw: Any) -> "ExperimentEvidence":
        return cls(d.experiment_id, d.treatment, d.control, d.diff, d.se, d.ci_low, d.ci_high, d.n_treatment,
                   d.n_control, d.randomized, dict(estimates or {}), dict(estimate_se or {}), **kw)


ARM_DIFFERENCE, UNIT_MEAN = "arm_difference", "unit_mean"


def paired_comparison(units: Sequence[Tuple[str, float, float]], treatment: str, control: str, *,
                      kind: str = ARM_DIFFERENCE) -> Optional[Tuple[float, float, int]]:
    """Estimate minus measured arm difference on the same units, with its standard error.

    ``units`` = (arm, estimator value e_i, measured value y_i), one per independent unit (lifetime). The measured
    difference is mean_T(y) - mean_C(y); the estimate is mean_T(e) - mean_C(e) (``kind="arm_difference"``, e.g.
    per-decision credit) or mean(e) over all units (``kind="unit_mean"``, e.g. a per-unit counterfactual contrast).
    D = sum_i q_i with q_i = w_arm e_i - v_arm y_i; units are independent and identically distributed within an arm,
    so Var(D) = sum_arm n_arm Var_arm(q). Returns (D, se(D), n) or None when an arm has fewer than two units."""
    t = [(e, y) for a, e, y in units if a == treatment and math.isfinite(e) and math.isfinite(y)]
    c = [(e, y) for a, e, y in units if a == control and math.isfinite(e) and math.isfinite(y)]
    if len(t) < 2 or len(c) < 2:
        return None
    nt, nc, n = len(t), len(c), len(t) + len(c)
    if kind == ARM_DIFFERENCE:
        wt, wc = 1.0 / nt, -1.0 / nc
    elif kind == UNIT_MEAN:
        wt = wc = 1.0 / n
    else:
        raise ValueError(kind)
    var, D = 0.0, 0.0
    for rows, w, v in ((t, wt, 1.0 / nt), (c, wc, -1.0 / nc)):
        q = [w * e - v * y for e, y in rows]
        m = sum(q) / len(q)
        D += sum(q)
        var += len(q) * sum((x - m) ** 2 for x in q) / (len(q) - 1)
    return D, math.sqrt(max(var, 0.0)), n


@dataclass
class EstimatorScore(Record):
    name: str
    n: int = 0                        # randomised experiments this estimator gave an estimate for
    rmse: float = float("nan")        # error against the measured differences
    mean_sq_z: float = float("nan")   # squared error in units of the combined standard error (about 1 if calibrated)
    sign_agreement: float = float("nan")   # on experiments whose measured effect is significant
    n_significant: int = 0
    coverage: float = float("nan")    # share of estimates inside the experimental confidence interval
    passes: bool = False
    reasons: List[str] = field(default_factory=list)
    n_refused: int = 0                # comparisons that could not be scored (zero / undefined standard error)
    n_paired: int = 0                 # comparisons scored with the paired standard error


@dataclass
class SelectionRecord(Record):
    ts: float
    default_before: str
    default_after: str
    changed: bool
    n_randomized_experiments: int
    scores: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    gates: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    evidence_keys: List[str] = field(default_factory=list)
    excluded_non_randomized: List[str] = field(default_factory=list)
    excluded_not_held_out: List[str] = field(default_factory=list)  # in-sample or unknown split: never scored
    holdout: Dict[str, Any] = field(default_factory=dict)           # per scored evidence: units used / excluded


class EstimatorSelector:
    """Compares estimators with randomised-experiment differences and promotes one only when it is validated.

    baseline              the estimator used before any validation (the simple attribution)
    min_experiments       randomised experiments an estimator must have been compared on
    max_mean_sq_z         calibration gate: mean squared z of (estimate - measured)
    min_sign_agreement    on significant measured effects, the share whose sign the estimator gets right
    min_improvement       required relative reduction of RMSE against the current default on common experiments
    """

    def __init__(self, *, baseline: str = SIMPLE, min_experiments: int = 5, max_mean_sq_z: float = 3.0,
                 min_sign_agreement: float = 0.8, min_improvement: float = 0.1, log_path: Optional[str] = None,
                 clock: Optional[Clock] = None, require_held_out: bool = True):
        self.baseline = baseline
        self.require_held_out = bool(require_held_out)
        self.default = baseline
        self.min_experiments, self.max_mean_sq_z = int(min_experiments), float(max_mean_sq_z)
        self.min_sign_agreement, self.min_improvement = float(min_sign_agreement), float(min_improvement)
        self.log_path = log_path
        self.clock = clock or WallClock()
        self.evidence: Dict[str, ExperimentEvidence] = {}
        self.history: List[SelectionRecord] = []

    # ---------------------------------------------------------------------------------------------- evidence
    def add_evidence(self, ev: ExperimentEvidence) -> None:
        """Add or replace (same experiment and arm pair) one comparison. Non-randomised comparisons are kept for
        the record but never used for scoring."""
        self.evidence[ev.key] = ev

    def add_from_store(self, store, experiment_id: str, estimators: Sequence[str], *, control: Optional[str] = None,
                       confidence: float = 0.95, extra: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
                       exclude_objectives: Optional[Sequence[str]] = None) -> List[ExperimentEvidence]:
        """Build evidence for a registered experiment from the store: the measured arm differences
        (``simple.experiment_effects``) and, per estimator, the difference of the mean stored per-decision credit
        ``record.credit[name]`` between the arms, with the paired standard error. ``extra[name][treatment] =
        (estimate, se)`` supplies estimates that are not per-decision credits (e.g. ``cocoa.arm_contrast``; they
        have no paired standard error).

        ``exclude_objectives``: the lifetimes the estimators were trained on. When given, their units are left out
        and the evidence is marked ``held_out``; when omitted the split is unknown and the selector does not score
        the evidence (``require_held_out``)."""
        out = []
        skip = set(exclude_objectives or ())
        recs = [r for r in store.experiment_records(experiment_id) if r.objective_id not in skip]
        n_skip = len(store.experiment_records(experiment_id)) - len(recs) if skip else 0
        y = {r.exp_id: record_outcome_value(r) for r in recs}
        held = {r.exp_id for r in recs}
        for d in experiment_effects(store, experiment_id, control=control, confidence=confidence,
                                    value_fn=(lambda r: y.get(r.exp_id)) if skip else None):
            est, est_se, pd, ps, pn = {}, {}, {}, {}, {}
            for name in estimators:
                by_arm = {d.treatment: [], d.control: []}
                units = []
                for r in recs:
                    arm = record_arm(r)
                    if arm in by_arm and name in r.credit:
                        by_arm[arm].append(r.credit[name])
                        if y.get(r.exp_id) is not None:
                            units.append((arm, float(r.credit[name]), float(y[r.exp_id])))
                cd = arm_difference(by_arm[d.treatment], by_arm[d.control])
                if cd is not None:
                    est[name], est_se[name] = cd.diff, cd.se
                pc = paired_comparison(units, d.treatment, d.control)
                if pc is not None:
                    pd[name], ps[name], pn[name] = pc
            for name, per_arm in (extra or {}).items():
                if d.treatment in per_arm:
                    est[name], est_se[name] = per_arm[d.treatment]
            ev = ExperimentEvidence.from_difference(
                d, est, est_se, held_out=True if exclude_objectives is not None else None, n_excluded_trained=n_skip,
                paired_diff=pd, paired_se=ps, paired_n=pn,
                holdout={"source": "add_from_store", "units": len(held), "excluded_trained": n_skip})
            self.add_evidence(ev)
            out.append(ev)
        return out

    # ---------------------------------------------------------------------------------------------- scoring
    def _held_out(self, e: ExperimentEvidence) -> bool:
        return e.held_out is True or not self.require_held_out

    def _used(self) -> List[ExperimentEvidence]:
        return [e for e in self.evidence.values() if e.randomized and self._held_out(e)]

    @staticmethod
    def _z(name: str, e: ExperimentEvidence) -> Tuple[float, Optional[float], bool]:
        """(error, squared z or None when it cannot be scored, paired) of one estimator on one comparison."""
        if name in e.paired_diff and name in e.paired_se:
            err, se, paired = float(e.paired_diff[name]), float(e.paired_se[name]), True
        else:                                  # no paired statistic: both uncertainties must have been stated
            err, paired = float(e.estimates[name]) - float(e.diff), False
            est_se = e.estimate_se.get(name)
            se = math.sqrt(e.se ** 2 + est_se ** 2) if est_se is not None and est_se > 0 and math.isfinite(est_se) \
                else 0.0
        if not (math.isfinite(se) and se > 0.0 and math.isfinite(err)):
            return err, None, paired
        return err, (err / se) ** 2, paired

    def score(self, name: str, only: Optional[Sequence[str]] = None) -> EstimatorScore:
        evs = [e for e in self._used() if name in e.estimates and (only is None or e.key in only)]
        s = EstimatorScore(name=name, n=len(evs))
        if not evs:
            s.reasons.append("no randomised experiment to compare with (held out by lifetime)")
            return s
        zs = [self._z(name, e) for e in evs]
        err = [x[0] for x in zs]
        z2 = [x[1] for x in zs if x[1] is not None]
        s.n_refused = sum(1 for x in zs if x[1] is None)
        s.n_paired = sum(1 for x in zs if x[1] is not None and x[2])
        s.rmse = math.sqrt(sum(x * x for x in err) / len(err))
        s.mean_sq_z = sum(z2) / len(z2) if z2 else float("nan")
        if s.n_refused:
            s.reasons.append(f"{s.n_refused} comparisons have a zero or undefined standard error (cannot be scored)")
        sig = [e for e in evs if e.significant]
        s.n_significant = len(sig)
        if sig:
            s.sign_agreement = sum((e.estimates[name] > 0) == (e.diff > 0) for e in sig) / len(sig)
        s.coverage = sum(e.ci_low <= e.estimates[name] <= e.ci_high for e in evs) / len(evs)
        if len(z2) < self.min_experiments:
            s.reasons.append(f"only {len(z2)} scored randomised experiments (< {self.min_experiments})")
        if z2 and s.mean_sq_z > self.max_mean_sq_z:
            s.reasons.append(f"disagrees with measured differences: mean z^2 {s.mean_sq_z:.2f} > {self.max_mean_sq_z}")
        if s.n_significant == 0:
            s.reasons.append("no significant measured effect to validate against")
        elif s.sign_agreement < self.min_sign_agreement:
            s.reasons.append(f"sign agreement {s.sign_agreement:.2f} < {self.min_sign_agreement}")
        s.passes = not s.reasons
        return s

    def scores(self) -> Dict[str, EstimatorScore]:
        names = sorted({n for e in self._used() for n in e.estimates} | {self.baseline, self.default})
        return {n: self.score(n) for n in names}

    # ---------------------------------------------------------------------------------------------- selection
    def log_row(self, row: Dict[str, Any]) -> None:
        """Append one JSON row to the selection log (durable on return); no-op without ``log_path``."""
        if not self.log_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def select(self, context: Optional[Dict[str, Any]] = None) -> SelectionRecord:
        """Re-evaluate the default. A candidate is promoted only if it passes every gate *and* its error against
        the measured differences is at least ``min_improvement`` lower than the current default's on the
        experiments both were compared on. A learned default that no longer passes falls back to the baseline.
        ``context`` (e.g. ``CreditAssigner.evaluate_experiments``' per-experiment report) is logged with the row."""
        scores = self.scores()
        before, reasons = self.default, []
        current = self.default
        if current != self.baseline and not scores[current].passes:
            reasons.append(f"{current} no longer validated ({'; '.join(scores[current].reasons)}): back to {self.baseline}")
            current = self.baseline
        best, best_rmse = None, float("inf")
        for name, sc in scores.items():
            if name == current:
                continue
            if not sc.passes:
                if name != self.baseline:
                    reasons.append(f"{name} not promoted: {'; '.join(sc.reasons)}")
                continue
            common = [e.key for e in self._used() if name in e.estimates and current in e.estimates]
            if common:
                mine, theirs = self.score(name, common).rmse, self.score(current, common).rmse
                if not mine <= (1.0 - self.min_improvement) * theirs:
                    reasons.append(f"{name} not promoted: rmse {mine:.4g} is not {self.min_improvement:.0%} below "
                                   f"{current}'s {theirs:.4g} on {len(common)} common experiments")
                    continue
                reasons.append(f"{name}: rmse {mine:.4g} vs {current} {theirs:.4g} on {len(common)} common experiments")
            else:
                reasons.append(f"{name}: validated on {sc.n} experiments; {current} has no estimate to compare")
            if sc.rmse < best_rmse:
                best, best_rmse = name, sc.rmse
        if best is not None:
            reasons.append(f"promoted {best} to training default")
            current = best
        elif not self._used():
            reasons.append(f"no randomised experiment yet (held out by lifetime): default stays {current}")
        self.default = current
        rec = SelectionRecord(
            ts=self.clock.now(), default_before=before, default_after=current, changed=before != current,
            n_randomized_experiments=len(self._used()), scores={n: s.to_dict() for n, s in scores.items()},
            gates={"min_experiments": self.min_experiments, "max_mean_sq_z": self.max_mean_sq_z,
                   "min_sign_agreement": self.min_sign_agreement, "min_improvement": self.min_improvement},
            reasons=reasons, evidence_keys=sorted(e.key for e in self._used()),
            excluded_non_randomized=sorted(e.key for e in self.evidence.values() if not e.randomized),
            excluded_not_held_out=sorted(e.key for e in self.evidence.values() if e.randomized and not self._held_out(e)),
            holdout={e.key: {"held_out": e.held_out, "n_excluded_trained": e.n_excluded_trained,
                             "paired_n": dict(e.paired_n), **e.holdout} for e in self._used()})
        self.history.append(rec)
        row: Dict[str, Any] = {"selection": rec.to_dict(), "evidence": [e.to_dict() for e in self.evidence.values()]}
        if context is not None:
            row["evaluation"] = context
        self.log_row(row)
        return rec

    # ---------------------------------------------------------------------------------------------- persistence
    def state_dict(self) -> Dict[str, Any]:
        return {"default": self.default, "baseline": self.baseline,
                "evidence": [e.to_dict() for e in self.evidence.values()]}

    def load_state_dict(self, st: Dict[str, Any]) -> None:
        self.default = st.get("default", self.baseline)
        self.evidence = {}
        for d in st.get("evidence", []):
            self.add_evidence(ExperimentEvidence.from_dict(d))
