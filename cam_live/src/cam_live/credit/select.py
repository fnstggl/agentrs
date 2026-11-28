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


# ------------------------------------------------------------------------------------------------ unit level (CR)
#: kinds of per-unit estimate: a per-decision credit of the executed action (its contrast between arms is formed by
#: the inverse-propensity weight) or a per-unit estimate of the arm contrast itself (e.g. COCOA's counterfactual
#: contrast of the unit)
PER_DECISION, PER_UNIT_CONTRAST = "per_decision", "per_unit_contrast"


@dataclass
class UnitContrast(Record):
    """One held-out randomised unit (amendment item 4, CR): a randomised decision whose return is final, whose lifetime
    or segment no estimator and no learner trained on, with its logged assignment probability, the measured
    return-to-go and each estimator's value for it. Units of different experiments - also of experiments with a
    single unit - are pooled: every unit carries its own arms and propensity (``pooled_unit_score``)."""
    experiment_id: str
    exp_id: str
    unit: str                         # lifetime (objective id) or matured segment id the return refers to
    treatment: str
    control: str
    arm: str                          # executed arm (treatment or control)
    p_treatment: float                # P(treatment | treatment or control), as logged at assignment
    y: float                          # measured matured return-to-go of the decision
    values: Dict[str, float] = field(default_factory=dict)     # estimator -> its value for this unit
    kinds: Dict[str, str] = field(default_factory=dict)        # estimator -> PER_DECISION | PER_UNIT_CONTRAST

    @property
    def key(self) -> str:
        return f"{self.experiment_id}:{self.exp_id}"

    @property
    def weight(self) -> float:
        """Inverse-propensity contrast weight w = 1/p (treatment) or -1/(1-p) (control): E[w f(A)] = f(T) - f(C)."""
        p = min(max(float(self.p_treatment), 1e-12), 1.0 - 1e-12)
        return 1.0 / p if self.arm == self.treatment else -1.0 / (1.0 - p)


def chi2_cdf(x: float, k: float) -> float:
    """Regularised lower incomplete gamma P(k/2, x/2) (series / continued fraction; no scipy at run time)."""
    if x <= 0.0:
        return 0.0
    a, z = 0.5 * k, 0.5 * x
    gln = math.lgamma(a)
    if z < a + 1.0:
        term = s = 1.0 / a
        ap = a
        for _ in range(1000):
            ap += 1.0
            term *= z / ap
            s += term
            if abs(term) < abs(s) * 1e-15:
                break
        return s * math.exp(-z + a * math.log(z) - gln)
    b, c, d = z + 1.0 - a, 1.0 / 1e-300, 1.0 / (z + 1.0 - a)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return 1.0 - math.exp(-z + a * math.log(z) - gln) * h


def chi2_ppf(p: float, k: float) -> float:
    lo, hi = 0.0, max(1.0, 2.0 * k)
    while chi2_cdf(hi, k) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_cdf(mid, k) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class PooledScore(Record):
    """Estimand-matched validation of one estimator on pooled held-out randomised units (CR).

    Per unit i with contrast weight w_i (``UnitContrast.weight``) the paired error term is
        d_i = (e_i - y_i - c_i) w_i                for a per-decision credit e_i of the executed action,
        d_i = x_i - (y_i - c_i) w_i                 for a per-unit contrast estimate x_i,
    with c_i the leave-one-out mean of (e - y) (resp. y) over the other units (a baseline that does not depend on
    unit i's assignment, so it changes the variance only). Under randomisation E[d_i | h_i] = (estimated contrast
    at h_i) - (true contrast at h_i): the mean d is the estimator's average contrast bias *at the information
    states of the units* - the estimand the estimator targets - whatever hidden mechanism generated each unit and
    however many units its experiment has. Validated (``passes``) when the bias is statistically equivalent to zero
    within a margin delta = ``margin_sd`` x sd(y) (two one-sided tests at level ``alpha``: the 1 - 2 alpha interval
    lies inside (-delta, delta); Schuirmann 1987) with at least ``min_units`` units."""
    name: str
    n: int = 0
    bias: float = float("nan")
    se: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    margin: float = float("nan")
    y_sd: float = float("nan")
    z: float = float("nan")
    rmse_terms: float = float("nan")  # root mean square of d (efficiency of the comparison; not a gate)
    n_experiments: int = 0
    passes: bool = False
    reasons: List[str] = field(default_factory=list)


def pooled_unit_score(units: Sequence[UnitContrast], name: str, *, min_units: int = 8, margin_sd: float = 0.5,
                      alpha: float = 0.05) -> PooledScore:
    from .simple import student_t_ppf
    rows = [u for u in units if name in u.values and math.isfinite(float(u.values[name])) and math.isfinite(u.y)]
    s = PooledScore(name=name, n=len(rows), n_experiments=len({u.experiment_id for u in rows}))
    ys = [u.y for u in units if math.isfinite(u.y)]
    if len(ys) >= 2:
        my = sum(ys) / len(ys)
        s.y_sd = math.sqrt(sum((y - my) ** 2 for y in ys) / (len(ys) - 1))
    if len(rows) < 2:
        s.reasons.append(f"only {len(rows)} held-out randomised units with an estimate (< {max(2, min_units)})")
        return s
    per_dec = [u.kinds.get(name, PER_DECISION) == PER_DECISION for u in rows]
    r = [float(u.values[name]) - u.y if pd else u.y for u, pd in zip(rows, per_dec)]
    tot, n = sum(r), len(rows)
    d = []
    for u, pd, ri in zip(rows, per_dec, r):
        c = (tot - ri) / (n - 1)
        d.append((ri - c) * u.weight if pd else float(u.values[name]) - (ri - c) * u.weight)
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    s.bias, s.se = m, math.sqrt(var / n)
    s.rmse_terms = math.sqrt(sum(x * x for x in d) / n)
    tq = student_t_ppf(1.0 - alpha, n - 1)
    s.ci_low, s.ci_high = m - tq * s.se, m + tq * s.se
    s.z = m / s.se if s.se > 0 else float("nan")
    s.margin = margin_sd * s.y_sd if math.isfinite(s.y_sd) else float("nan")
    if n < min_units:
        s.reasons.append(f"only {n} held-out randomised units with an estimate (< {min_units})")
    if not (s.se > 0 and math.isfinite(s.se)):
        s.reasons.append("zero or undefined standard error of the pooled bias (cannot be scored)")
    elif not (math.isfinite(s.margin) and s.margin > 0):
        s.reasons.append("no outcome scale for the equivalence margin (sd of the measured returns is 0)")
    elif not (-s.margin < s.ci_low and s.ci_high < s.margin):
        s.reasons.append(f"average contrast bias {m:.3g} (90% CI [{s.ci_low:.3g}, {s.ci_high:.3g}]) is not within "
                         f"+-{s.margin:.3g} ({margin_sd} sd of the measured return)")
    s.passes = not s.reasons
    return s


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
    mode: str = "per_experiment"      # "per_experiment" (no unit evidence) | "pooled_units" (CR)
    pooled: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # estimator -> PooledScore + calibration check
    n_units: int = 0


class EstimatorSelector:
    """Compares estimators with randomised-experiment differences and promotes one only when it is validated.

    baseline              the estimator used before any validation (the simple attribution)
    min_experiments       randomised experiments an estimator must have been compared on
    max_mean_sq_z         calibration gate: mean squared z of (estimate - measured)
    min_sign_agreement    on significant measured effects, the share whose sign the estimator gets right
    min_improvement       required relative reduction of RMSE against the current default on common experiments

    With unit evidence (``add_units``: held-out randomised units, pooled over experiments - CR) the selection uses the
    estimand-matched rule instead (``mode="pooled_units"``): an estimator is validated when (a) its average contrast
    bias over the pooled held-out units is equivalent to zero within ``margin_sd`` sd of the measured return
    (``pooled_unit_score``: two one-sided tests at ``alpha``, at least ``min_units`` units), and (b) on every
    experiment with at least ``calib_min_per_arm`` held-out units per arm its per-experiment paired z^2 values are
    not jointly rejected (sum z^2 <= chi^2_K quantile ``1 - calib_alpha``). (a) is valid for experiments of any size
    (one unit included) and for segments of one open objective; (b) keeps the per-experiment calibration wherever the
    data can test it (with fewer units per arm the paired standard error rests on too few degrees of freedom:
    ``cr_diagnose power``). The validated estimator with the tightest bias bound becomes the default; a learned
    default that stops passing falls back to the baseline."""

    def __init__(self, *, baseline: str = SIMPLE, min_experiments: int = 5, max_mean_sq_z: float = 3.0,
                 min_sign_agreement: float = 0.8, min_improvement: float = 0.1, log_path: Optional[str] = None,
                 clock: Optional[Clock] = None, require_held_out: bool = True, min_units: int = 8,
                 margin_sd: float = 0.5, alpha: float = 0.05, calib_min_per_arm: int = 5, calib_alpha: float = 0.01):
        self.min_units, self.margin_sd, self.alpha = int(min_units), float(margin_sd), float(alpha)
        self.calib_min_per_arm, self.calib_alpha = int(calib_min_per_arm), float(calib_alpha)
        self.units: Dict[str, UnitContrast] = {}
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
    def add_units(self, units: Sequence[UnitContrast]) -> int:
        """Add or replace (same experiment and decision) held-out randomised units; returns how many changed."""
        n = 0
        for u in units:
            old = self.units.get(u.key)
            if old is None or old.to_dict() != u.to_dict():
                self.units[u.key] = u
                n += 1
        return n

    def pooled_scores(self) -> Dict[str, Dict[str, Any]]:
        """Estimand-matched scores (see the class doc) of every estimator with a value on a held-out unit."""
        units = list(self.units.values())
        names = sorted({n for u in units for n in u.values} | {self.baseline, self.default})
        out: Dict[str, Dict[str, Any]] = {}
        for name in names:
            ps = pooled_unit_score(units, name, min_units=self.min_units, margin_sd=self.margin_sd, alpha=self.alpha)
            d = ps.to_dict()
            z2 = {}
            for e in self._used():
                if min(e.n_treatment, e.n_control) < self.calib_min_per_arm or name not in e.estimates:
                    continue
                err, z, _ = self._z(name, e)
                if z is not None:
                    z2[e.key] = z
            d["calibration"] = {"experiments": len(z2), "sum_z2": sum(z2.values()),
                                "chi2_quantile": chi2_ppf(1.0 - self.calib_alpha, len(z2)) if z2 else None,
                                "z2": z2, "min_units_per_arm": self.calib_min_per_arm}
            rejected = bool(z2) and sum(z2.values()) > d["calibration"]["chi2_quantile"]
            d["calibration"]["rejected"] = rejected
            if rejected:
                d["reasons"] = list(d["reasons"]) + [
                    f"per-experiment calibration rejected on {len(z2)} experiments with >= {self.calib_min_per_arm} "
                    f"units per arm: sum z^2 {sum(z2.values()):.2f} > chi^2 {d['calibration']['chi2_quantile']:.2f}"]
                d["passes"] = False
            hw = max(abs(ps.ci_low), abs(ps.ci_high)) if math.isfinite(ps.ci_low) else float("inf")
            d["bias_bound"] = hw
            out[name] = d
        return out

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

    def _select_pooled(self, context: Optional[Dict[str, Any]]) -> SelectionRecord:
        sc = self.pooled_scores()
        before, reasons = self.default, []
        current = self.default
        if current != self.baseline and not sc.get(current, {}).get("passes", False):
            why = "; ".join(sc.get(current, {}).get("reasons", [])) or "no estimate on the held-out units"
            reasons.append(f"{current} no longer validated ({why}): back to {self.baseline}")
            current = self.baseline
        best, best_bound = None, float("inf")
        for name, d in sc.items():
            if name == current:
                continue
            if not d["passes"]:
                if name != self.baseline:
                    reasons.append(f"{name} not promoted: {'; '.join(d['reasons'])}")
                continue
            if current != self.baseline and sc.get(current, {}).get("passes") and \
                    not d["bias_bound"] <= (1.0 - self.min_improvement) * sc[current]["bias_bound"]:
                reasons.append(f"{name} not promoted: bias bound {d['bias_bound']:.4g} is not {self.min_improvement:.0%} "
                               f"below {current}'s {sc[current]['bias_bound']:.4g}")
                continue
            reasons.append(f"{name}: validated on {d['n']} held-out units of {d['n_experiments']} experiments, "
                           f"average contrast bias {d['bias']:.4g} (90% CI [{d['ci_low']:.4g}, {d['ci_high']:.4g}], "
                           f"margin {d['margin']:.4g})")
            if d["bias_bound"] < best_bound:
                best, best_bound = name, d["bias_bound"]
        if best is not None:
            reasons.append(f"promoted {best} to training default")
            current = best
        self.default = current
        used = self._used()
        rec = SelectionRecord(
            ts=self.clock.now(), default_before=before, default_after=current, changed=before != current,
            n_randomized_experiments=len({u.experiment_id for u in self.units.values()}),
            scores={n: s.to_dict() for n, s in self.scores().items()},
            gates={"min_units": self.min_units, "margin_sd": self.margin_sd, "alpha": self.alpha,
                   "calib_min_per_arm": self.calib_min_per_arm, "calib_alpha": self.calib_alpha,
                   "min_improvement": self.min_improvement},
            reasons=reasons, evidence_keys=sorted(e.key for e in used),
            excluded_non_randomized=sorted(e.key for e in self.evidence.values() if not e.randomized),
            excluded_not_held_out=sorted(e.key for e in self.evidence.values() if e.randomized and not self._held_out(e)),
            holdout={e.key: {"held_out": e.held_out, "n_excluded_trained": e.n_excluded_trained,
                             "paired_n": dict(e.paired_n), **e.holdout} for e in used},
            mode="pooled_units", pooled=sc, n_units=len(self.units))
        self.history.append(rec)
        row: Dict[str, Any] = {"selection": rec.to_dict(), "evidence": [e.to_dict() for e in self.evidence.values()],
                               "units": [u.to_dict() for u in self.units.values()]}
        if context is not None:
            row["evaluation"] = context
        self.log_row(row)
        return rec

    def select(self, context: Optional[Dict[str, Any]] = None) -> SelectionRecord:
        """Re-evaluate the default (estimand-matched pooled rule when unit evidence exists, see the class doc). A candidate is promoted only if it passes every gate *and* its error against
        the measured differences is at least ``min_improvement`` lower than the current default's on the
        experiments both were compared on. A learned default that no longer passes falls back to the baseline.
        ``context`` (e.g. ``CreditAssigner.evaluate_experiments``' per-experiment report) is logged with the row."""
        if self.units:
            return self._select_pooled(context)
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
                "evidence": [e.to_dict() for e in self.evidence.values()],
                "units": [u.to_dict() for u in self.units.values()]}

    def load_state_dict(self, st: Dict[str, Any]) -> None:
        self.default = st.get("default", self.baseline)
        self.evidence = {}
        for d in st.get("evidence", []):
            self.add_evidence(ExperimentEvidence.from_dict(d))
        self.units = {}
        for d in st.get("units", []):                  # absent in checkpoints written before CR: no unit evidence
            u = UnitContrast.from_dict(d)
            self.units[u.key] = u
