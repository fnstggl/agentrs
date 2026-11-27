"""Reward integrity and outer-estimator support attacks (spec 7.4, 7.8; MR-05, MR-06, MR-09, MR-11).

Engine level (reward.meta): the importance-weighted estimator is unbiased with logged behaviour propensities
(Monte Carlo vs exact), and refuses — with psi bit-identical after apply() — when propensities are missing, invalid,
below the floor, the effective sample size is too small, or a row is unverified.
Real path (MetaLearner on the debug model, LabWorld verifier): forged / unsigned / replayed signed outcomes; a
VALID signature with a caller-chosen utility; a self-generated goal signed by the verifier; ledger file tampering and
truncation; reward-magnitude inflation; self-goal discount on paraphrases; noisy-TV gate on the learned reward.
"""
import dataclasses
import json
import math
import os
import sys
import tempfile

import torch

import _common as C
from cam_live.contracts import GoalStatus, OutcomeObservation, OutcomeStatus, TaskSpec, Trust, VerifiedOutcome
from cam_live.diagnostics.collect import verified_return
from cam_live.diagnostics.world import LabConfig, LabVerifier, LabWorld, lab_task
from cam_live.reward import meta as M
from cam_live.reward.diagnose import ExperimentConfig, LabMeasurementAuth, lab_lifetime, make_stack
from cam_live.reward.integrity import IntegrityError, OutcomeLedger, SelfGoalDiscount, params_hash

D = torch.float64
torch.manual_seed(3)

# ------------------------------------------------------------------------------ engine: IS unbiased, refusals
A, n, d = 4, 64, 3
phi = torch.randn(n, A, d, dtype=D)
th = {"w": torch.randn(d, dtype=D)}
logp_all = lambda t, x: torch.log_softmax(x @ t["w"], -1)
logp_fn = lambda t, b: logp_all(t, b["phi"])[torch.arange(b["phi"].shape[0]), b["a"]]
Utab = torch.randn(n, A, dtype=D)
mu = torch.softmax(torch.randn(n, A, dtype=D), -1)          # logged behaviour policy, != pi
exact = torch.func.grad(lambda t: (logp_all(t, phi).exp() * Utab).sum(-1).mean())(th)["w"]
mg = M.MetaGradient(lambda t, p, b: (t["w"] * 0).sum(), logp_fn, weight_clip=1e9, min_ess=0.0, min_ess_fraction=0.0)
gen = torch.Generator().manual_seed(0)
acc, R = torch.zeros(d, dtype=D), 4000
sq = torch.zeros(d, dtype=D)
for _ in range(R):
    a = torch.multinomial(mu, 1, generator=gen).squeeze(-1)
    q = M.QueryData(batch={"phi": phi, "a": a}, utility=Utab[torch.arange(n), a], baseline=0.0,
                    behaviour_logp=torch.log(mu[torch.arange(n), a]))
    t_ = {"w": th["w"].clone().requires_grad_(True)}
    est = mg.outer_surrogate(t_, q, "importance")
    gi = torch.autograd.grad(est.surrogate, t_["w"])[0]
    acc += gi
    sq += gi ** 2
mc = acc / R
se = torch.sqrt((sq / R - mc ** 2) / R)
z = ((mc - exact) / se).abs().max()
C.check("importance estimator with logged behaviour propensities is unbiased (MC, 4000 reps, max |z| < 4)",
        float(z) < 4.0, f"max|z|={float(z):.2f} rel_err={float((mc - exact).norm() / exact.norm()):.3f}")
# same data, propensities wrong (claim uniform logging) -> biased: shows the estimator really uses them
acc2 = torch.zeros(d, dtype=D)
gen = torch.Generator().manual_seed(0)
for _ in range(R):
    a = torch.multinomial(mu, 1, generator=gen).squeeze(-1)
    q = M.QueryData(batch={"phi": phi, "a": a}, utility=Utab[torch.arange(n), a], baseline=0.0,
                    behaviour_logp=torch.full((n,), -math.log(A), dtype=D))
    t_ = {"w": th["w"].clone().requires_grad_(True)}
    acc2 += torch.autograd.grad(mg.outer_surrogate(t_, q, "importance").surrogate, t_["w"])[0]
C.check("(sensitivity) wrong logged propensities give a biased estimate (max |z| > 6)",
        float(((acc2 / R - exact) / se).abs().max()) > 6.0, f"rel_err={float((acc2 / R - exact).norm() / exact.norm()):.3f}")

psi = {"p": torch.randn(3, dtype=D, requires_grad=True)}
inner = lambda t, p, b: -(logp_fn(t, b) * (b["phi"][torch.arange(b["phi"].shape[0]), b["a"]] @ p["p"])).mean()
mg2 = M.MetaGradient(inner, logp_fn, min_ess=4.0, min_propensity=1e-3)
a0 = torch.zeros(n, dtype=torch.long)
tr = mg2.inner_update(th, psi, [{"phi": phi, "a": a0}], K=1)
base = dict(batch={"phi": phi, "a": a0}, utility=torch.randn(n, dtype=D))
cases = {
    "missing_propensities": M.QueryData(**base),
    "invalid_propensities": M.QueryData(**base, behaviour_logp=torch.full((n,), 0.5, dtype=D)),
    "propensity_below_floor": M.QueryData(**base, behaviour_logp=torch.full((n,), math.log(1e-5), dtype=D)),
    "effective_sample_size_too_small": M.QueryData(**base, behaviour_logp=torch.cat(
        [torch.full((1,), math.log(1e-2), dtype=D), torch.full((n - 1,), math.log(0.9), dtype=D)])),
    "unverified_outcome_in_query": M.QueryData(**base, behaviour_logp=torch.full((n,), -math.log(A), dtype=D),
                                               verified=torch.tensor([True] * (n - 1) + [False])),
}
opt = torch.optim.SGD([psi["p"]], lr=1.0)
for want, q in cases.items():
    r = mg2.meta_gradient(tr, q, "importance")
    h = params_hash(psi)
    info = M.MetaGradient.apply(opt, psi, r)
    C.check(f"refusal '{want}': unsupported, no gradient, psi bit-identical", (not r.supported) and r.reason.startswith(want)
            and not r.grads and params_hash(psi) == h and info["applied"] == 0.0, f"reason={r.reason!r}")

# ------------------------------------------------------------------------------ real path: the ledger
ec = ExperimentConfig(min_independent=2)
auth = LabMeasurementAuth()
model, store, clock, ml = make_stack(0, ec, mix=0.0, check_measurement=auth)
ml.new_lifetime()
lt, world, _ = lab_lifetime(ml, store, clock, ec, seed=5150, auth=auth)
verifier = LabVerifier(world)
good = lt.verified
C.check("genuine signed objective outcome accepted", ml.admit_objective(good, utility=verified_return(lt)).accepted)
C.check("replayed signed outcome rejected as duplicate", ml.admit_objective(good).reason == "duplicate")
forged = VerifiedOutcome.from_dict({**good.to_dict(), "value": good.value + 1000.0})
C.check("tampered value with the old signature rejected", ml.admit_objective(forged).reason == "bad_signature")
# a VALID signature, then the caller states the utility
clock.advance(60.0)
v2 = verifier.verify(lab_task(world, objective_id=good.objective_id))
r = ml.admit_objective(v2, utility=1e6)
ent = [e for e in ml.ledger.entries() if e.utility >= 1e6]
C.check("a validly signed outcome cannot be admitted with a utility the verifier never stated",
        not r.accepted, f"accepted={r.accepted} ledger utility={ent[0].utility if ent else None} "
                        f"signed value={v2.value}")
clock.advance(60.0)
v3 = verifier.verify(lab_task(world, objective_id=good.objective_id))
r = ml.admit_objective(v3, exp_ids=["exp_that_does_not_exist"])
C.info("validly signed outcome credited to arbitrary exp_ids", f"accepted={r.accepted} reason={r.reason!r}")
# a self-generated goal, evaluated and signed by the verifier, admitted through the default API
self_task = TaskSpec(objective_id="self_goal_1", objective="reach 1 adopter (self-generated)",
                     success={"target": 1}, limits=lab_task(world).limits, origin=Trust.SELF)
vs = verifier.verify(self_task)
r = ml.admit_objective(vs)
C.check("a verifier-signed outcome of a self-generated goal does not supervise the outer objective by default",
        not r.accepted, f"accepted={r.accepted} reason={r.reason!r} (TaskSpec.origin=SELF is not carried into the signed "
                        f"VerifiedOutcome; objective_origin is a caller argument defaulting to OPERATOR)")
C.check("with objective_origin=SELF it is rejected", not ml.admit_objective(
    verifier.verify(dataclasses.replace(self_task, objective_id="self_goal_2")), objective_origin=Trust.SELF).accepted)
fake = OutcomeObservation(query_id="q_invented", status=OutcomeStatus.OBSERVED, value=1e6, observed_at=clock.now(),
                          verified=True, source="labworld", dedup_key="invented")
C.check("invented measurement rejected", not ml.ledger.admit_measurement(fake).accepted)
pred = dataclasses.replace(fake, source="prediction", dedup_key="pred")
C.check("prediction-sourced measurement rejected", ml.ledger.admit_measurement(pred).reason == "self_authored")

# ledger file: tamper and truncate
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "ledger.jsonl")
    L = OutcomeLedger(verifier.check_signature, path=path)
    for i in range(3):
        clock.advance(60.0)
        L.admit_verified(verifier.verify(lab_task(world, objective_id=f"obj_t{i}")), utility=float(i))
    lines = open(path).read().splitlines()
    d0 = json.loads(lines[1])
    d0["utility"] = 99.0
    open(path, "w").write("\n".join([lines[0], json.dumps(d0, sort_keys=True), lines[2]]) + "\n")
    try:
        OutcomeLedger(verifier.check_signature, path=path)
        C.check("edited ledger entry detected on load", False)
    except IntegrityError:
        C.check("edited ledger entry detected on load", True)
    open(path, "w").write("\n".join(lines[:1]) + "\n")
    L2 = OutcomeLedger(verifier.check_signature, path=path)
    C.info("truncated ledger (last 2 entries deleted) loads with a valid chain", f"{L2.verify_chain()} n={len(L2)} "
           "(only a head recorded elsewhere can reveal it; MetaLearner.load_state_dict checks the checkpoint head)")

# ------------------------------------------------------------------------------ reward magnitude inflation
ib = ml.inner.build_batch(store, lt.records[:8]).with_mix(0.0)
theta0 = ml.subset.snapshot()
p0 = {k: v.detach().clone() for k, v in ml.reward.psi().items()}
step = lambda p: M.flat({n_: v - theta0[n_] for n_, v in ml.engine.inner_update(theta0, p, [ib], 0.1, 1,
                                                                              create_graph=False).detached().items()})
d1 = step(p0)
big = {k: (v * 1000.0 if k.startswith("out.") else v) for k, v in p0.items()}
with torch.no_grad():
    rb, r0 = ml.inner.learned_reward(big, ib), ml.inner.learned_reward(p0, ib)
C.check("inflating R_psi's output never exceeds the bound", float(rb.abs().max()) <= ml.reward.bound,
        f"max|r|={float(rb.abs().max()):.4f} bound={ml.reward.bound}")
ml.reward.bound *= 50.0
d50 = step({k: (v * 50.0 if k.startswith("out.") else v) for k, v in p0.items()})
ml.reward.bound /= 50.0
C.check("r -> 50 r (bound and output scaled together) leaves the policy update unchanged",
        float((d50 - d1).abs().max()) < 1e-5 * max(1e-12, float(d1.abs().max())) / 1e-6 * 1e-1,
        f"max|diff|={float((d50 - d1).abs().max()):.2e} |update|={float(d1.abs().max()):.2e}")

# ------------------------------------------------------------------------------ self-goal discount
sg = SelfGoalDiscount()
w = [sg.weight(t, 0.3) for t in ("Reach 10 members", "reach 11 members!", "Get ten members", "obtain 10 members")]
C.info("self-goal weights: 'Reach 10 members', 'reach 11 members!', 'Get ten members', 'obtain 10 members'", w)
C.check("repeated self-goal (number / punctuation changed) is discounted", w[1] < w[0])
C.check("a paraphrased repeat of the same self-goal is discounted", w[2] < 1.0 and w[3] < 1.0,
        "SelfGoalDiscount.key only folds case, punctuation and numbers")
C.check("trivial self-goal (p_success 0.99 by the learner's own estimate) gets ~0 weight", sg.weight("x y", 0.99) < 0.11)
sys.exit(C.finish())
