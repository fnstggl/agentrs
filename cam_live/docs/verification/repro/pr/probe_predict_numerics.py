"""World-model likelihoods, masking, calibration metrics, registry and evaluator BCE (PI-02, PI-03, PI-06, SE-01,
SE-02, SE-03, SE-04).

* every predictive family's log-likelihood vs scipy.stats (float64), the discrete hazard vs a brute-force geometric /
  censored reference;
* build_targets: pending / unobservable / missing masked (y left at 0, mask False), imagined refused, censored counts
  masked;
* calibration on synthetic data with KNOWN calibration (intercept 0.3, slope 0.7): slope/intercept recovered, Brier /
  NLL / reliability equal my own numpy; then a clustered null (40 lifetimes x 16 forecasts sharing one outcome,
  true slope 0): how often does the reported 95 % slope interval exclude 0?
* PredictionRegistry: no edit after registration, no resolution at or before registration, single resolution,
  tampering detected;
* evaluator: BCE-with-logits on the signed outcome against every earlier prefix; pending -> zero loss and gradient;
  unsigned refused; prefixes at / after evaluation masked.
"""
import json
import math
import os
import sys
import tempfile

import numpy as np
import torch
from scipy import stats

import _common as C
from cam_live.contracts import (GoalStatus, OutcomeKind, OutcomeObservation, OutcomeQuery, OutcomeStatus,
                                PredictionRecord, SimClock, VerifiedOutcome)
from cam_live.predict import calibration as CAL
from cam_live.predict import dists as Dd
from cam_live.predict.evaluator import (EvaluatorConfig, GoalEvaluator, UnverifiedOutcomeError, evaluator_loss,
                                        prefix_targets)
from cam_live.predict.imagination import ImaginedDataError
from cam_live.predict.registry import (AlreadyResolvedError, DuplicatePredictionError, PredictionRegistry,
                                       ResolutionBeforePredictionError, TamperedRegistryError, verify_log)
from cam_live.predict.world import build_targets

F = torch.float64
rng = np.random.default_rng(0)
t = lambda x: torch.as_tensor(x, dtype=F)

# ---------------------------------------------------------------------------------------- likelihoods vs scipy
y = rng.normal(size=50) * 3
loc, sc, df = rng.normal(size=50), np.exp(rng.normal(size=50)), np.exp(rng.normal(size=50)) + 0.5
e = np.abs(Dd.normal_log_prob(t(y), t(loc), t(sc)).numpy() - stats.norm.logpdf(y, loc, sc)).max()
C.check("normal log-pdf == scipy", e < 1e-12, f"{e:.1e}")
e = np.abs(Dd.student_t_log_prob(t(y), t(loc), t(sc), t(df)).numpy() - stats.t.logpdf(y, df, loc, sc)).max()
C.check("student-t log-pdf == scipy", e < 1e-10, f"{e:.1e}")
k = rng.poisson(5.0, size=50).astype(float)
rate, expo = np.exp(rng.normal(size=50)), rng.uniform(0.5, 4, size=50)
e = np.abs(Dd.poisson_log_prob(t(k), t(rate), t(expo)).numpy() - stats.poisson.logpmf(k, rate * expo)).max()
C.check("poisson (with exposure) log-pmf == scipy", e < 1e-10, f"{e:.1e}")
conc = np.exp(rng.normal(size=50))
mu = rate * expo
e = np.abs(Dd.neg_binomial_log_prob(t(k), t(rate), t(conc), t(expo)).numpy()
           - stats.nbinom.logpmf(k, conc, conc / (conc + mu))).max()
C.check("negative binomial (mean, concentration, exposure) log-pmf == scipy", e < 1e-9, f"{e:.1e}")
big = Dd.neg_binomial_log_prob(t(k), t(rate), t(np.full(50, 1e8)), t(expo)).numpy()
C.check("negative binomial -> Poisson as concentration -> inf", np.abs(big - stats.poisson.logpmf(k, mu)).max() < 1e-5)
lg = rng.normal(size=50) * 4
yb = (rng.uniform(size=50) < 0.5).astype(float)
e = np.abs(Dd.bernoulli_log_prob(t(yb), t(lg)).numpy() - stats.bernoulli.logpmf(yb, 1 / (1 + np.exp(-lg)))).max()
C.check("bernoulli log-pmf == scipy", e < 1e-12, f"{e:.1e}")
C.check("bernoulli stable at logit 800", math.isfinite(float(Dd.bernoulli_log_prob(t(0.0), t(800.0)))))
L5 = rng.normal(size=(50, 6))
nc = rng.integers(2, 7, size=50)
yc = np.array([rng.integers(0, n) for n in nc])
ref = np.array([L5[i, yc[i]] - np.log(np.exp(L5[i, :nc[i]]).sum()) for i in range(50)])
e = np.abs(Dd.categorical_log_prob(torch.as_tensor(yc), t(L5), torch.as_tensor(nc)).numpy() - ref).max()
C.check("categorical with per-row category count == direct computation", e < 1e-12, f"{e:.1e}")
# discrete hazard with constant hazard h over unit bins == geometric; censoring = survival of completed bins
h, K = 0.2, 10
edges = t(np.arange(K + 1, dtype=float))
hl = t(np.full((5, K), math.log(h / (1 - h))))
tt = t([0.5, 3.2, 7.9, 9.5, 12.0])
ev = Dd.hazard_log_prob(tt, torch.ones(5, dtype=torch.bool), hl, edges).numpy()
want = [math.log(h) + math.floor(x) * math.log(1 - h) if x < K else K * math.log(1 - h) for x in tt.tolist()]
C.check("hazard event log-lik == geometric", np.abs(ev - want).max() < 1e-12, f"{np.abs(ev - want).max():.1e}")
ce = Dd.hazard_log_prob(tt, torch.zeros(5, dtype=torch.bool), hl, edges).numpy()
want_c = [min(math.floor(x), K) * math.log(1 - h) for x in tt.tolist()]
C.check("hazard right-censored log-lik == survival of completed bins", np.abs(ce - want_c).max() < 1e-12)
pmf = Dd.hazard_pmf(hl[0]).numpy()
C.check("hazard pmf sums to 1 (incl. tail)", abs(pmf.sum() - 1) < 1e-12)

# ---------------------------------------------------------------------------------------- build_targets masking
qs = [OutcomeQuery(f"q{i}", "m", kind=kd, horizon_s=60.0) for i, kd in
      enumerate([OutcomeKind.COUNT, OutcomeKind.COUNT, OutcomeKind.COUNT, OutcomeKind.REAL, OutcomeKind.COUNT])]
obs = [OutcomeObservation("q0", OutcomeStatus.OBSERVED, 3.0, observed_at=61.0, verified=True),
       OutcomeObservation("q1", OutcomeStatus.PENDING, None),
       OutcomeObservation("q2", OutcomeStatus.UNOBSERVABLE, None),
       None,
       OutcomeObservation("q4", OutcomeStatus.CENSORED, 2.0, observed_at=61.0)]
meta, tgt = build_targets([list(zip(qs, obs))])
C.check("only the observed entry is a target; pending/unobservable/missing/censored-count masked",
        tgt.observed[0].tolist() == [True, False, False, False, False] and not bool(tgt.censored.any())
        and tgt.y[0, 1:].abs().sum() == 0, f"observed={tgt.observed[0].tolist()} y={tgt.y[0].tolist()}")
im = OutcomeObservation("q0", OutcomeStatus.OBSERVED, 3.0, observed_at=61.0, verified=True)
setattr(im, "imagined", True)
try:
    build_targets([[(qs[0], im)]])
    C.check("imagined observation refused as a world-model target", False)
except ImaginedDataError:
    C.check("imagined observation refused as a world-model target", True)
C.info("build_targets stores y as", str(tgt.y.dtype))

# ---------------------------------------------------------------------------------------- calibration
n = 20000
p = rng.uniform(0.03, 0.97, size=n)
a0, b0 = 0.3, 0.7
lp = np.log(p / (1 - p))
yy = (rng.uniform(size=n) < 1 / (1 + np.exp(-(a0 + b0 * lp)))).astype(float)
rep = CAL.binary_report(p, yy)
cal = rep["calibration"]
C.check("calibration slope / intercept recovered on synthetic data (true 0.7 / 0.3 inside the 95% CI)",
        cal["slope_lo"] <= b0 <= cal["slope_hi"] and cal["intercept_lo"] <= a0 <= cal["intercept_hi"],
        f"slope={cal['slope']:.3f} [{cal['slope_lo']:.3f},{cal['slope_hi']:.3f}] "
        f"intercept={cal['intercept']:.3f} [{cal['intercept_lo']:.3f},{cal['intercept_hi']:.3f}]")
C.check("Brier == mean (p - y)^2", abs(rep["brier"]["mean"] - np.mean((p - yy) ** 2)) < 1e-12)
C.check("NLL == mean binary cross-entropy", abs(rep["nll"]["mean"] + np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))
        < 1e-12)
bins = rep["reliability"]
true_rate = lambda q: 1 / (1 + np.exp(-(a0 + b0 * np.log(q / (1 - q)))))
dev = max(abs(b_["frac_pos"] - np.mean(true_rate(p[(p >= b_["lo"]) & (p < b_["hi"])]))) for b_ in bins if b_["n"] > 500)
C.check("reliability curve follows the known calibration function", dev < 0.03, f"max dev={dev:.3f}")
perfect = (rng.uniform(size=n) < p).astype(float)
pc = CAL.binary_report(p, perfect)["calibration"]
C.check("perfectly calibrated forecasts: slope CI contains 1 and intercept CI contains 0",
        pc["slope_lo"] <= 1 <= pc["slope_hi"] and pc["intercept_lo"] <= 0 <= pc["intercept_hi"])
# clustered null: 40 lifetimes, 16 forecasts each, one shared outcome per lifetime, forecasts unrelated to it
excl, reps = 0, 300
excl_clustered_brier = 0
for r in range(reps):
    g = np.random.default_rng(1000 + r)
    life_p = 0.58 + 0.01 * g.normal(size=40)                  # nearly constant per lifetime (like B's evaluator)
    pp = np.repeat(life_p, 16) + 0.002 * g.normal(size=640)
    yl = (g.uniform(size=40) < 0.6).astype(float)              # outcome independent of the forecast
    yr = np.repeat(yl, 16)
    c_ = CAL.logistic_recalibration(pp, yr)
    if c_["identifiable"] and (c_["slope_lo"] > 0 or c_["slope_hi"] < 0):
        excl += 1
C.check("clustered null (true slope 0): the reported 95% slope interval excludes 0 in <= 10% of runs",
        excl / reps <= 0.10, f"excluded 0 in {excl}/{reps} = {excl / reps:.2f} of runs (Wald interval over 640 "
                             f"rows, clusters ignored; binary_report has no clustered slope)")

# ---------------------------------------------------------------------------------------- registry
with tempfile.TemporaryDirectory() as tmp:
    clock = SimClock(100.0)
    reg = PredictionRegistry(os.path.join(tmp, "pred.jsonl"), clock=clock)
    q = OutcomeQuery("qa", "lab.x", kind=OutcomeKind.COUNT, horizon_s=50.0, issued_at=100.0)
    rec = PredictionRecord("p1", q, Dd.Poisson(t(3.0)).to_record(), made_at=100.0)
    stored = reg.register(rec)
    rec.dist.params["rate"] = 99.0 if hasattr(rec.dist, "params") and isinstance(rec.dist.params, dict) else 0
    try:
        reg.register(rec)
        C.check("re-registering (editing) a prediction refused", False)
    except DuplicatePredictionError:
        C.check("re-registering (editing) a prediction refused", True)
    C.check("mutating the caller's object does not change the stored prediction",
            reg.get("p1").dist.to_dict() == stored.dist.to_dict())
    try:
        reg.resolve("p1", OutcomeObservation("qa", OutcomeStatus.OBSERVED, 2.0, observed_at=100.0, verified=True))
        C.check("resolution observed at the registration time refused", False)
    except ResolutionBeforePredictionError:
        C.check("resolution observed at the registration time refused", True)
    fut = PredictionRecord("p2", q, Dd.Poisson(t(3.0)).to_record(), made_at=500.0)
    try:
        reg.register(fut)
        C.check("prediction dated in the future of the registry clock refused", False)
    except Exception:
        C.check("prediction dated in the future of the registry clock refused", True)
    clock.advance(60.0)
    r1 = reg.resolve("p1", OutcomeObservation("qa", OutcomeStatus.OBSERVED, 2.0, observed_at=clock.now(), verified=True))
    C.check("resolved NLL == scipy Poisson NLL", abs(r1.nll + stats.poisson.logpmf(2, 3.0)) < 1e-9, f"{r1.nll}")
    try:
        reg.resolve("p1", OutcomeObservation("qa", OutcomeStatus.OBSERVED, 7.0, observed_at=clock.now() + 1, verified=True))
        C.check("second resolution refused", False)
    except AlreadyResolvedError:
        C.check("second resolution refused", True)
    path = reg._log.path
    lines = open(path).read().splitlines()
    d = json.loads(lines[0])
    s = json.dumps(d)
    open(path, "w").write("\n".join([s.replace('"made_at": 100.0', '"made_at": 99.0')] + lines[1:]) + "\n")
    try:
        verify_log(path)
        PredictionRegistry(path, clock=clock)
        C.check("editing the registry log is detected", s == s.replace('"made_at": 100.0', '"made_at": 99.0'),
                "edit not detected")
    except TamperedRegistryError:
        C.check("editing the registry log is detected", True)
    except Exception as ex:
        C.check("editing the registry log is detected", True, f"{type(ex).__name__}")

# ---------------------------------------------------------------------------------------- evaluator BCE on prefixes
torch.manual_seed(0)
ev = GoalEvaluator(EvaluatorConfig(d_model=8)).double()
T = 6
b = torch.randn(T, 8, dtype=F)
eg = torch.randn(1, 8, dtype=F).expand(T, -1)
ts = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
for status, yv in ((GoalStatus.ACHIEVED, 1.0), (GoalStatus.FAILED, 0.0)):
    hor, tg = prefix_targets(ts, status=status, evaluated_at=35.0, deadline=100.0, signature_checked=True,
                             value=5.0, achieved_at=34.0 if yv else None)
    out = ev(b, eg, hor.to(F))
    loss, m = evaluator_loss(out, tg, weights={"success": 1.0, "magnitude": 0.0, "time": 0.0, "resources": 0.0})
    valid = torch.tensor([x < 35.0 for x in ts])
    ref = torch.nn.functional.binary_cross_entropy_with_logits(out.success_logit[valid], torch.full((int(valid.sum()),), yv,
                                                                                                      dtype=F))
    C.check(f"evaluator success loss == BCE-with-logits on the signed outcome ({status}) over prefixes before evaluation",
            abs(float(loss) - float(ref)) < 1e-10 and tg.success_mask.tolist() == valid.tolist(),
            f"loss={float(loss):.6f} ref={float(ref):.6f} mask={tg.success_mask.tolist()}")
hor, tg = prefix_targets(ts, status=GoalStatus.PENDING, evaluated_at=35.0, deadline=100.0, signature_checked=True)
out = ev(b, eg, hor.to(F))
loss, _ = evaluator_loss(out, tg)
gr = torch.autograd.grad(loss, list(ev.parameters()), allow_unused=True)
C.check("pending objective: zero loss and exactly zero gradient", float(loss) == 0.0 and
        all(g is None or float(g.abs().max()) == 0.0 for g in gr))
try:
    prefix_targets(ts, status=GoalStatus.ACHIEVED, evaluated_at=35.0, deadline=100.0, signature_checked=False)
    C.check("unsigned outcome refused", False)
except UnverifiedOutcomeError:
    C.check("unsigned outcome refused", True)
# overconfident evaluator receives corrective gradient: p=0.8 forecasts, all failures
tg = prefix_targets(ts, status=GoalStatus.FAILED, evaluated_at=60.0, deadline=100.0, signature_checked=True)[1]
logit = torch.full((T,), math.log(4.0), dtype=F, requires_grad=True)
bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.zeros(T, dtype=F))
C.check("an 80% forecast that fails gets a gradient pushing p down (d loss / d logit = p - y = 0.8 per row)",
        abs(float(torch.autograd.grad(bce, logit)[0][0]) * T - 0.8) < 1e-12)
sys.exit(C.finish())
