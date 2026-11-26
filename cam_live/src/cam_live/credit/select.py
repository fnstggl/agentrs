"""cam_live.credit.select — the training-default credit estimator is chosen by evidence (spec 9.4; RC-08).

Live operation starts with the simple delayed-outcome attribution. A learned estimator (RUDDER, COCOA) becomes the
default only after its estimates of arm differences have been compared with the differences actually measured in
*randomised* experiments and found to agree; if later evidence contradicts it, the default falls back. Every
selection, with the evidence it rests on, is appended to a JSON-lines log.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..contracts import Clock, Record, WallClock
from .simple import NAME as SIMPLE, ArmDifference, arm_difference, experiment_effects
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
    estimate_se: Dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.experiment_id}:{self.treatment}-{self.control}"

    @property
    def significant(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @classmethod
    def from_difference(cls, d: ArmDifference, estimates: Optional[Dict[str, float]] = None,
                        estimate_se: Optional[Dict[str, float]] = None) -> "ExperimentEvidence":
        return cls(d.experiment_id, d.treatment, d.control, d.diff, d.se, d.ci_low, d.ci_high, d.n_treatment,
                   d.n_control, d.randomized, dict(estimates or {}), dict(estimate_se or {}))


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
                 clock: Optional[Clock] = None):
        self.baseline = baseline
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
                       confidence: float = 0.95, extra: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
                       ) -> List[ExperimentEvidence]:
        """Build evidence for a registered experiment from the store: the measured arm differences
        (``simple.experiment_effects``) and, per estimator, the difference of the mean stored per-decision credit
        ``record.credit[name]`` between the arms. ``extra[name][treatment] = (estimate, se)`` supplies estimates
        that are not per-decision credits (e.g. ``cocoa.arm_contrast``)."""
        out = []
        recs = store.experiment_records(experiment_id)
        for d in experiment_effects(store, experiment_id, control=control, confidence=confidence):
            est, est_se = {}, {}
            for name in estimators:
                by_arm = {d.treatment: [], d.control: []}
                for r in recs:
                    arm = record_arm(r)
                    if arm in by_arm and name in r.credit:
                        by_arm[arm].append(r.credit[name])
                cd = arm_difference(by_arm[d.treatment], by_arm[d.control])
                if cd is not None:
                    est[name], est_se[name] = cd.diff, cd.se
            for name, per_arm in (extra or {}).items():
                if d.treatment in per_arm:
                    est[name], est_se[name] = per_arm[d.treatment]
            ev = ExperimentEvidence.from_difference(d, est, est_se)
            self.add_evidence(ev)
            out.append(ev)
        return out

    # ---------------------------------------------------------------------------------------------- scoring
    def _used(self) -> List[ExperimentEvidence]:
        return [e for e in self.evidence.values() if e.randomized]

    def score(self, name: str, only: Optional[Sequence[str]] = None) -> EstimatorScore:
        evs = [e for e in self._used() if name in e.estimates and (only is None or e.key in only)]
        s = EstimatorScore(name=name, n=len(evs))
        if not evs:
            s.reasons.append("no randomised experiment to compare with")
            return s
        err = [e.estimates[name] - e.diff for e in evs]
        z2 = [(x ** 2) / max(e.se ** 2 + e.estimate_se.get(name, 0.0) ** 2, 1e-300) for x, e in zip(err, evs)]
        s.rmse = math.sqrt(sum(x * x for x in err) / len(err))
        s.mean_sq_z = sum(z2) / len(z2)
        sig = [e for e in evs if e.significant]
        s.n_significant = len(sig)
        if sig:
            s.sign_agreement = sum((e.estimates[name] > 0) == (e.diff > 0) for e in sig) / len(sig)
        s.coverage = sum(e.ci_low <= e.estimates[name] <= e.ci_high for e in evs) / len(evs)
        if s.n < self.min_experiments:
            s.reasons.append(f"only {s.n} randomised experiments (< {self.min_experiments})")
        if s.mean_sq_z > self.max_mean_sq_z:
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
    def select(self) -> SelectionRecord:
        """Re-evaluate the default. A candidate is promoted only if it passes every gate *and* its error against
        the measured differences is at least ``min_improvement`` lower than the current default's on the
        experiments both were compared on. A learned default that no longer passes falls back to the baseline."""
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
            reasons.append(f"no randomised experiment yet: default stays {current}")
        self.default = current
        rec = SelectionRecord(
            ts=self.clock.now(), default_before=before, default_after=current, changed=before != current,
            n_randomized_experiments=len(self._used()), scores={n: s.to_dict() for n, s in scores.items()},
            gates={"min_experiments": self.min_experiments, "max_mean_sq_z": self.max_mean_sq_z,
                   "min_sign_agreement": self.min_sign_agreement, "min_improvement": self.min_improvement},
            reasons=reasons, evidence_keys=sorted(e.key for e in self._used()),
            excluded_non_randomized=sorted(e.key for e in self.evidence.values() if not e.randomized))
        self.history.append(rec)
        if self.log_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps({"selection": rec.to_dict(),
                                     "evidence": [e.to_dict() for e in self.evidence.values()]},
                                    sort_keys=True, default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
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
