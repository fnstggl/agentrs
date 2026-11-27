"""Decision rules: recorded propensities vs empirical selection frequencies; randomised experiments; VOI reference
(PI-08, SE-06, AE-01).

DecisionMaker on the debug model with hand-built CandidateEvaluations (one infeasible candidate). For every rule the
decision is drawn 20000 times; the recorded Decision.probabilities must (a) be what the index was drawn from
(chi-square of empirical counts), (b) put exactly 0 on the infeasible candidate, (c) keep the eps floor. Thompson
from normal marginals: recorded probabilities are a Monte-Carlo estimate redrawn per call — the decision must still
be drawn from the recorded vector, and its average must approach the true P(best). Experiments: registration
before assignment, assignment frequencies match the recorded arm probabilities. VOI: E[max affine] vs Monte Carlo,
the knowledge gradient vs Monte Carlo, an irrelevant noisy observation has VOI exactly 0.
"""
import math
import os
import sys
import tempfile

import numpy as np
import torch
from scipy import stats

import _common as C
from cam_live.config import CamLiveConfig
from cam_live.contracts import ActionSpec, CandidateEvaluation, Cost, SimClock
from cam_live.nn.model import build_model
from cam_live.predict import voi as V
from cam_live.predict.policy import DecisionMaker
from cam_live.predict.registry import ExperimentRegistry, PredictionRegistry

torch.manual_seed(0)
model = build_model(CamLiveConfig.debug(seed=0))
tmp = tempfile.mkdtemp()
clock = SimClock(1000.0)
preds = PredictionRegistry(os.path.join(tmp, "p.jsonl"), clock=clock)
exps = ExperimentRegistry(os.path.join(tmp, "e.jsonl"), preds, clock=clock)
dm = DecisionMaker(model, clock=clock, experiments=exps, seed=1)
eps = dm.cfg.eps_floor
q = [0.30, 0.25, 0.10, 0.28, 0.50]
sd = [0.05, 0.20, 0.40, 0.10, 0.30]
feas = [True, True, True, True, False]


def evals(with_samples):
    out = []
    g = torch.Generator().manual_seed(7)
    S = torch.randn(32, 5, generator=g, dtype=torch.float64)
    for i in range(5):
        ex = {"q_goal": q[i], "q_goal_epistemic_std": sd[i], "net_cost": 0.0}
        if with_samples:
            ex.update({f"ts_{s}": float(q[i] + sd[i] * S[s, i]) for s in range(32)})
        out.append(CandidateEvaluation(action=ActionSpec(action_id=f"a{i}", operation=f"op{i}"), value=[0.1, 0.5, -0.2, 0.3, 2.0][i],
                                       voi=[0.0, 0.05, 0.2, 0.01, 0.3][i], uncertainty=sd[i], feasible=feas[i],
                                       p_success=0.5, extras=ex))
    return out


M = 20000
for rule, ws in (("policy", True), ("greedy", True), ("ucb", True), ("voi", True), ("thompson", True),
                 ("thompson", False)):
    ev = evals(ws)
    gen = torch.Generator().manual_seed(123)
    counts = np.zeros(5)
    psum = np.zeros(5)
    mism = 0
    for _ in range(M):
        d = dm.select(ev, rule, generator=gen, ts=clock.now())
        counts[d.chosen_index] += 1
        pr = np.asarray(d.probabilities)
        psum += pr
    pbar = psum / M
    nz = pbar > 0
    chi = ((counts[nz] - M * pbar[nz]) ** 2 / (M * pbar[nz])).sum()
    pv = 1 - stats.chi2.cdf(chi, nz.sum() - 1)
    tag = f"{rule}{'' if ws else ' (normal marginals)'}"
    C.check(f"{tag}: empirical selection frequencies match the recorded propensities (chi2 p={pv:.3f})", pv > 1e-3,
            f"freq={np.round(counts / M, 4).tolist()} recorded={np.round(pbar, 4).tolist()}")
    C.check(f"{tag}: infeasible candidate has probability exactly 0 and was never chosen",
            pbar[4] == 0.0 and counts[4] == 0)
    C.check(f"{tag}: every feasible candidate keeps the eps floor {eps}/{sum(feas)}",
            float(pbar[:4].min()) >= eps / 4 - 1e-12, f"min={pbar[:4].min():.4f}")
    if rule == "greedy":
        C.check("greedy puts 1 - 3/4 eps on argmax q", abs(pbar[0] - (1 - eps + eps / 4)) < 1e-12)
    if rule == "ucb":
        best = int(np.argmax([q[i] + dm.cfg.ucb_c * sd[i] if feas[i] else -9 for i in range(5)]))
        C.check("ucb puts the mass on argmax q + c sd", int(np.argmax(pbar)) == best, f"argmax={int(np.argmax(pbar))}")
    if rule == "thompson" and not ws:
        g = np.random.default_rng(0)
        draws = np.asarray(q[:4])[None] + np.asarray(sd[:4])[None] * g.normal(size=(400000, 4))
        true = np.bincount(draws.argmax(1), minlength=4) / 400000
        true = (1 - eps) * true + eps / 4
        C.check("thompson (normal marginals): mean recorded propensity -> true P(best) with eps floor",
                np.abs(pbar[:4] - true).max() < 0.01, f"mean recorded={np.round(pbar[:4], 3).tolist()} "
                                                       f"true={np.round(true, 3).tolist()}")

# ---------------------------------------------------------------------------------------- randomised experiments
ev = evals(True)
from cam_live.contracts import OutcomeKind, OutcomeQuery, PredictionRecord
from cam_live.predict import dists as Dd
for i in (1, 3):
    pq = OutcomeQuery(f"qx{i}", f"lab.adopters.from.a{i}", kind=OutcomeKind.COUNT, horizon_s=3600.0,
                      issued_at=clock.now(), goal_relevant=True)
    ev[i].predictions = [preds.register(PredictionRecord(f"px{i}", pq, Dd.Poisson(torch.tensor(2.0 + i,
                         dtype=torch.float64)).to_record(), made_at=clock.now()))]
dec, rec = dm.select_experiment(ev, {"op1": 1, "op3": 3}, question="q", hypothesis="h", arm_probs=[0.3, 0.7],
                                ts=clock.now())
eid = rec.experiment_id
ops = [l for l in open(os.path.join(tmp, "e.jsonl")).read().splitlines()]
first_assign = next(i for i, l in enumerate(ops) if '"assign"' in l)
first_reg = next(i for i, l in enumerate(ops) if '"register"' in l)
C.check("experiment registered (with hypothesis and predictions) before the first assignment", first_reg < first_assign)
cnt = {1: 0, 3: 0}
for u in range(4000):
    d, _ = dm.select_experiment(ev, {"op1": 1, "op3": 3}, experiment_id=eid, unit_id=f"unit{u}", ts=clock.now())
    cnt[d.chosen_index] += 1
    if u == 0:
        C.check("experimental decision records the assignment probabilities (0 for non-arms)",
                [round(x, 6) for x in d.probabilities] == [0.0, 0.3, 0.0, 0.7, 0.0] and d.exploration == "randomized",
                f"{d.probabilities}")
f3 = cnt[3] / 4000
C.check("assignment frequency matches the recorded arm probability 0.7 (4 se)", abs(f3 - 0.7) < 4 * math.sqrt(0.21 / 4000),
        f"{f3:.4f}")
try:
    dm.select_experiment(ev, {"op1": 1, "op4": 4}, question="x", hypothesis="y", ts=clock.now())
    C.check("an infeasible candidate cannot be an experiment arm", False)
except ValueError:
    C.check("an infeasible candidate cannot be an experiment arm", True)
prop = dm.propose_experiment(ev)
C.info("self-proposed experiment", {k: prop[k] for k in ("arms", "question", "expected_flip_probability")})

# ---------------------------------------------------------------------------------------- VOI reference estimators
g = np.random.default_rng(1)
a, b = g.normal(size=6), g.normal(size=6)
z = g.normal(size=2_000_000)
mc = (a[None] + b[None] * z[:, None]).max(1).mean()
C.check("E[max_i a_i + b_i Z] exact == Monte Carlo", abs(V.expected_max_affine(a, b) - mc) < 3e-3,
        f"exact={V.expected_max_affine(a, b):.5f} mc={mc:.5f}")
mu, s2, n2 = torch.tensor([0.1, 0.3, 0.25], dtype=torch.float64), torch.tensor([0.2, 0.05, 0.4], dtype=torch.float64), 0.3
kg = V.knowledge_gradient_gaussian(mu, s2, n2)
ref = []
for i in range(3):
    st = math.sqrt(float(s2[i]) ** 2 / (float(s2[i]) + n2))
    m2 = np.tile(mu.numpy(), (2_000_000, 1))
    m2[:, i] += st * g.normal(size=2_000_000)
    ref.append(m2.max(1).mean() - float(mu.max()))
C.check("knowledge gradient (= E[max after] - max before) == Monte Carlo", np.abs(kg.numpy() - ref).max() < 2e-3,
        f"kg={np.round(kg.numpy(), 4).tolist()} mc={np.round(ref, 4).tolist()}")
post = V.LinearGaussianPosterior(torch.zeros(3, dtype=torch.float64), torch.diag(torch.tensor([1.0, 1.0, 1.0],
                                                                                              dtype=torch.float64)), 0.5)
phi_dec = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)   # decision depends on w0, w1 only
phi_probe = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)  # probe w0 / noisy-TV on w2
r = V.voi_linear_gaussian(post, phi_probe, phi_dec, noise_var=torch.tensor([0.5, 0.5], dtype=torch.float64))
C.check("decision-irrelevant observation (large surprise) has VOI exactly 0; relevant one > 0",
        float(r["gross"][1]) == 0.0 and float(r["gross"][0]) > 0.01 and float(r["expected_surprise"][1]) > 0,
        f"gross={r['gross'].tolist()} surprise={r['expected_surprise'].tolist()}")
sys.exit(C.finish())
