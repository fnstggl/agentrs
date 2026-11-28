"""Selection rule thompson_policy (PI-08, LRN S-5; predict/policy.py) and its calibration (SelectionCalibration).

Checks (my own numbers throughout):
  1. DecisionMaker.select samples the index from exactly the probability vector it records (generator replay).
  2. Over 20,000 decisions the frequency of each chosen candidate equals the mean recorded probability (|z| < 4):
     the recorded propensity is the true conditional probability of the choice.
  3. Infeasible candidates get exactly 0; every feasible one >= eps / n_feasible.
  4. Units drop out: q -> a q + b (a > 0) on means and samples leaves the distribution unchanged (same draws).
  5. The policy's preference counts only beyond the world model: logits -> logits + k * z(mean) changes nothing;
     a uniform policy gives the value-only distribution.
  6. My Thompson reference: p_i = share of samples s with i = argmax_j z_sj + lambda_s c_j, with my own z, c, lambda
     from the same coefficient draws == the code's vector.
  7. SelectionCalibration posterior == my weighted Bayesian linear regression (same documented model); it recovers
     known coefficients from synthetic data; a noise preference ends with lambda ~ 0 (mean lambda < 0.05).
  8. INFO: the prior alone already gives a peaked policy weight in the selection (before any verified outcome).
"""
import math
import sys

import numpy as np
import torch

import _g2d as C
from cam_live.config import CamLiveConfig
from cam_live.contracts import ActionSpec, CandidateEvaluation
from cam_live.nn.model import build_model
from cam_live.predict import policy as P

torch.manual_seed(0)
model = build_model(CamLiveConfig.debug(seed=0))
dm = P.DecisionMaker(model, seed=3)
cal = dm.calibration
S_TS = 16


def evals(n=6, seed=0, a=1.0, b=0.0, logits=None, infeasible=(4,), sdscale=1.0):
    g = torch.Generator().manual_seed(seed)
    mean = torch.randn(n, generator=g, dtype=torch.float64)
    sd = (0.3 + torch.rand(n, generator=g, dtype=torch.float64)) * sdscale
    samp = mean[None] + sd[None] * torch.randn(S_TS, n, generator=g, dtype=torch.float64)
    lg = torch.randn(n, generator=g, dtype=torch.float64) * 2.0 if logits is None else logits
    out = []
    for i in range(n):
        ex = {"q_goal": float(a * mean[i] + b), "q_goal_epistemic_std": float(a * sd[i]), "net_cost": 0.0,
              "policy_logit": float(lg[i])}
        ex.update({f"ts_{k}": float(a * samp[k, i] + b) for k in range(S_TS)})
        out.append(CandidateEvaluation(action=ActionSpec(action_id=f"a{i}", operation=f"op{i}"), value=float(lg[i]),
                                       feasible=i not in infeasible, extras=ex))
    return out


def probs_with_state(ev, state):
    dm.generator.set_state(state)
    p, sc = dm.probabilities(ev, P.RULE_THOMPSON_POLICY)
    return p, sc


# give the calibration some data so the policy has a non-trivial weight (beta_pi > 0 with spread)
rng = np.random.default_rng(1)
for k in range(60):
    z, c = rng.normal(), rng.normal()
    cal.observe(f"k{k}", z, c, 0.6 * z + 0.5 * c + rng.normal() * 0.8)
C.info("calibration summary", cal.summary())

# 1. replay
ev = evals()
st = dm.generator.get_state()
dec = dm.select(ev, P.RULE_THOMPSON_POLICY, ts=0.0)
dm.generator.set_state(st)
p2, _ = dm.probabilities(ev, P.RULE_THOMPSON_POLICY)
idx2 = int(torch.multinomial(p2, 1, generator=dm.generator))
C.check("select draws its index from exactly the recorded vector (generator replay)",
        [float(x) for x in p2] == dec.probabilities and idx2 == dec.chosen_index,
        f"chosen {dec.chosen_index} / {idx2}")

# 2. frequency == mean recorded probability (a decision whose candidates overlap: wide posteriors)
ev = evals(sdscale=4.0)
N = 20000
cnt = np.zeros(6)
psum = np.zeros(6)
p2sum = np.zeros(6)
for t in range(N):
    d = dm.select(ev, P.RULE_THOMPSON_POLICY, ts=0.0)
    cnt[d.chosen_index] += 1
    pv = np.array(d.probabilities)
    psum += pv
    p2sum += pv * (1 - pv)
freq, pm = cnt / N, psum / N
se = np.sqrt(np.maximum(p2sum / N, 1e-12) / N + 1e-12)
z = (freq - pm) / np.sqrt(np.maximum(pm * (1 - pm), 1e-12) / N)
C.check("chosen-index frequencies == mean recorded probabilities over 20,000 decisions (|z| < 4)",
        bool(np.all(np.abs(z[pm > 0]) < 4)), f"freq={np.round(freq, 4).tolist()} mean_p={np.round(pm, 4).tolist()} "
        f"max|z|={np.abs(z[pm > 0]).max():.2f}")

# 3. infeasible / floor
p, sc = dm.probabilities(ev, P.RULE_THOMPSON_POLICY)
nf = sum(1 for e in ev if e.feasible)
C.check("infeasible candidate has probability exactly 0; feasible >= eps/n_feasible",
        float(p[4]) == 0.0 and all(float(p[i]) >= dm.cfg.eps_floor / nf - 1e-15 for i in range(6) if i != 4)
        and abs(float(p.sum()) - 1) < 1e-12, [round(float(x), 4) for x in p])

# 4. affine invariance of the values
st = dm.generator.get_state()
pa, _ = probs_with_state(evals(seed=5), st)
pb, _ = probs_with_state(evals(seed=5, a=37.0, b=-1234.5), st)
C.check("q -> a q + b (a = 37, b = -1234.5) on means and samples leaves the distribution unchanged",
        float((pa - pb).abs().max()) < 1e-12, f"max diff {float((pa - pb).abs().max()):.2e}")

# 5. policy preference beyond the world model
base = evals(seed=7)
qm = torch.tensor([[e.extras["q_goal"] for e in base]], dtype=torch.float64)
mask = torch.tensor([[e.feasible for e in base]])
zref = P.standardised_values(qm[None], qm, mask)[0, 0]
lg0 = torch.tensor([e.extras["policy_logit"] for e in base], dtype=torch.float64)
pc, _ = probs_with_state(base, st)
pd, _ = probs_with_state(evals(seed=7, logits=lg0 + 3.7 * zref), st)
pu, scu = probs_with_state(evals(seed=7, logits=torch.zeros(6, dtype=torch.float64)), st)
vo = torch.tensor(scu["selection"]["value_only_probabilities"], dtype=torch.float64)
vo = P.apply_floor(vo[None], mask, dm.cfg.eps_floor)[0]
C.check("logits + k * standardised predicted value (k = 3.7) leaves the distribution unchanged",
        float((pc - pd).abs().max()) < 1e-9, f"max diff {float((pc - pd).abs().max()):.2e}")
C.check("a uniform policy gives exactly the value-only (Thompson over the world model) distribution",
        float((pu - vo).abs().max()) < 1e-12, f"max diff {float((pu - vo).abs().max()):.2e}")

# 6. my own Thompson reference from the same coefficient draws
ev6 = evals(seed=11)
dm.generator.set_state(st)
sc6 = dm.scores(ev6)
coef = cal.sample(S_TS, dm.generator)              # the draws thompson_policy will take with this state
dm.generator.set_state(st)
p6, _ = dm.probabilities(ev6, P.RULE_THOMPSON_POLICY)
feas = [e.feasible for e in ev6]
qmean = np.array([e.extras["q_goal"] for e in ev6])
fq = qmean[feas]
mu_, sd_ = fq.mean(), fq.std()
Z = np.array([[(e.extras[f"ts_{k}"] - mu_) / sd_ for e in ev6] for k in range(S_TS)])
zm = (qmean - mu_) / sd_
lgt = np.array([e.extras["policy_logit"] for e in ev6])
lp = lgt - np.log(np.exp(lgt[feas] - lgt[feas].max()).sum()) - lgt[feas].max()
cc = np.where(feas, lp - lp[feas].mean(), 0.0)
zz = np.where(feas, zm, 0.0)
cc = np.where(feas, cc - (cc @ zz) / (zz @ zz) * zz, 0.0)
cc = cc / max(1.0, math.sqrt((cc[feas] ** 2).mean()))
bq, bp = coef[:, 0].numpy(), coef[:, 1].numpy()
lam = np.clip(np.maximum(bp, 0) / np.maximum(bq, 0.05), 0, 10)
mine = np.zeros(len(ev6))
for s in range(S_TS):
    sc_ = np.where(feas, Z[s] + lam[s] * cc, -np.inf)
    mx = sc_.max()
    hit = np.where(feas & (sc_ >= mx - 1e-12 * (1 + abs(mx))))[0]
    mine[hit] += 1.0 / len(hit) / S_TS
nfe = sum(feas)
mine = (1 - dm.cfg.eps_floor) * mine + dm.cfg.eps_floor * np.array(feas, dtype=float) / nfe
C.check("my Thompson-with-policy reference (own z, c, lambda) == the code's vector",
        float(np.abs(mine - p6.numpy()).max()) < 1e-12, f"max diff {float(np.abs(mine - p6.numpy()).max()):.2e}")

# 7. calibration posterior vs my regression
post = cal.posterior()
obs = np.array(cal.obs)
w = np.array([cal.forget ** (len(obs) - 1 - k) for k in range(len(obs))])
s = math.sqrt((w * obs[:, 2] ** 2).sum() / w.sum())
X = np.stack([np.ones(len(obs)), obs[:, 0], obs[:, 1]], 1)
y = obs[:, 2] / s
P0 = np.diag([1.0, cal.prior_sd_q ** -2, cal.prior_sd_pi ** -2])
m0 = np.array([0.0, 1.0, 0.0])
nv = 1.0
for _ in range(3):
    A = P0 + (X * w[:, None]).T @ X / nv
    cov = np.linalg.inv(A)
    mean = cov @ (P0 @ m0 + (X * w[:, None]).T @ y / nv)
    r = y - X @ mean
    nv = max((cal.noise_dof + (w * r ** 2).sum()) / (cal.noise_dof + w.sum()), 1e-6)
C.check("SelectionCalibration posterior == my weighted Bayesian linear regression",
        np.allclose(mean, post["mean"], atol=1e-10) and np.allclose(cov, post["cov"], atol=1e-10),
        f"mine {np.round(mean, 4).tolist()} code {np.round(post['mean'], 4).tolist()}")

cal2 = P.SelectionCalibration(forget=1.0, max_obs=5000)
for k in range(3000):
    z_, c_ = rng.normal(), rng.normal()
    cal2.observe(f"r{k}", z_, c_, 2.0 * (0.4 * z_ + 0.7 * c_) + rng.normal() * 0.5)
m2 = cal2.posterior()["mean"]
ratio = m2[2] / m2[1]
C.check("calibration recovers the ratio beta_pi / beta_q of synthetic data (true 1.75)", abs(ratio - 1.75) < 0.1,
        f"posterior {np.round(m2, 3).tolist()} ratio {ratio:.3f}")
cal3 = P.SelectionCalibration(forget=1.0, max_obs=5000)
for k in range(3000):
    z_, c_ = rng.normal(), rng.normal()
    cal3.observe(f"n{k}", z_, c_, 0.5 * z_ + rng.normal() * 0.5)
lam3 = P.policy_weight(cal3.sample(4000, torch.Generator().manual_seed(0)))
C.check("a preference that does not predict verified progress ends with a negligible weight (mean lambda < 0.05)",
        float(lam3.mean()) < 0.05, f"mean lambda {float(lam3.mean()):.4f}, share > 0 {float((lam3 > 0).double().mean()):.2f}")

# 8. prior-only influence (INFO)
cal0 = P.SelectionCalibration()
lam0 = P.policy_weight(cal0.sample(20000, torch.Generator().manual_seed(1)))
C.info("prior only (no verified outcome): lambda mean / share > 0 / P(lambda > 1)",
       (round(float(lam0.mean()), 3), round(float((lam0 > 0).double().mean()), 3),
        round(float((lam0 > 1).double().mean()), 3)))
dm0 = P.DecisionMaker(build_model(CamLiveConfig.debug(seed=1)), seed=4)
evp = evals(seed=13, logits=torch.tensor([6.0, 0, 0, 0, 0, 0], dtype=torch.float64))
tv = []
for t in range(200):
    _, scp = dm0.probabilities(evp, P.RULE_THOMPSON_POLICY)
    tv.append(scp["selection"]["policy_share_tv"])
C.info("prior only: mean total variation the policy's preference adds to the decision (peaked policy, 6 cands)",
       round(float(np.mean(tv)), 4))
sys.exit(C.finish())
