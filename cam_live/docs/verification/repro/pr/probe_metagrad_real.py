"""Meta-gradient on the REAL adaptable subset of CamLiveModel (debug preset), independent reference (MR-03..MR-07).

Data: stored LabWorld decisions collected with C's own rig (reward.diagnose.make_stack / lab_lifetime), i.e. the real
experience-store / encode_decisions / R_psi path. Everything below the data is my own:
  * theta = live policy head + decision trunk (functional_call on the real modules), float64;
  * L_inner from the documented formula: discounted returns of r = (1-mix) R_psi + mix r_boot inside one lifetime,
    standardised; advantage against V evaluated on theta_k held fixed for the inner derivative; value loss; entropy;
    per-group (policy./trunk.) norm clip of the inner gradient; theta <- theta - alpha g;
  * R_psi is called only as the function psi -> rewards (C's IntrinsicReward via PolicyImprovementLoss.learned_reward);
  * outer J(psi) = exact expected utility of pi_{theta_K(psi)} over ALL feasible candidates of the query decisions
    (fixed random utility table): mean_i sum_a pi(a|x_i) U_i(a).
Checks for K = 1, 2, 4, 8: my torch.func gradient vs central FD on sampled psi coordinates and a random direction;
C's engine (MetaLearner.engine configuration: C's inner_loss + GroupClip) with importance weights against logged
uniform behaviour over the enumerated candidates == my gradient; every psi tensor receives gradient; detached reward
-> zero. Then the live path: MetaLearner.policy_update == my K-step SGD on the live parameters, psi bit-identical.
"""
import dataclasses
import math
import sys

import torch
from torch.func import functional_call, grad

import _common as C
from cam_live.reward import meta as M
from cam_live.reward.diagnose import ExperimentConfig, lab_lifetime, make_stack
from cam_live.reward.integrity import params_hash

D = torch.float64
torch.manual_seed(0)
ec = ExperimentConfig()
model, store, clock, ml = make_stack(0, ec, mix=0.0)
lt, _, _ = lab_lifetime(ml, store, clock, ec, seed=31337, update_every=0, goal_free_rows=(1, 4))
sup, qry = lt.records[:8], lt.records[8:]
ib = ml.inner.build_batch(store, sup).to(D)
ibq = ml.inner.build_batch(store, qry, policy_only=True).to(D)
theta0 = ml.subset.snapshot(D)
psi0 = {k: v.detach().clone().to(D) for k, v in ml.reward.psi().items()}
ALPHA = 0.1
CP, CT = ml.mc.clip_policy, ml.mc.clip_trunk
cfg = ml.inner.cfg
trunk, policy = ml.subset.trunk, ml.subset.policy
C.info("n_theta", sum(v.numel() for v in theta0.values()))
C.info("n_psi", sum(v.numel() for v in psi0.values()))
C.info("support rows / goal-free rows / query rows", (ib.n, int((~ib.has_goal).sum()), ibq.n))


def forward(th, lat):
    tr = {k[6:]: v for k, v in th.items() if k.startswith("trunk.")}
    pol = {k[7:]: v for k, v in th.items() if k.startswith("policy.")}
    z = functional_call(trunk, tr, (lat.b, lat.e_a, lat.e_g))
    out = functional_call(policy, pol, (z, lat.a_mask, lat.b))
    lg = torch.where(lat.a_mask, out.raw_logits, torch.full_like(out.raw_logits, -1e300))
    return torch.log_softmax(lg, -1), out.value


def my_loss(th, th_sg, psi, lat, mix):
    lpa, v = forward(th, lat)
    _, vb = forward(th_sg, lat)
    rl = ml.inner.learned_reward(psi, lat)
    r = mix * lat.r_boot.to(D) + (1.0 - mix) * rl
    G, acc = [None] * lat.n, torch.zeros((), dtype=D)
    gam = lat.gamma.to(D)
    for i in reversed(range(lat.n)):
        acc = r[i] + gam[i] * acc
        G[i] = acc
    G = torch.stack(G)
    dG = G - G.mean()
    Gn = dG / torch.sqrt((dG ** 2).mean() + 1e-12)
    lp = lpa[torch.arange(lat.n), lat.chosen]
    p = lpa.exp() * lat.a_mask
    ent = -(p * torch.where(lat.a_mask, lpa, torch.zeros_like(lpa))).sum(-1)
    return -(lp * (Gn - vb)).mean() + cfg.value_coef * (0.5 * (v - Gn) ** 2).mean() - cfg.entropy_coef * ent.mean()


def clip(g):
    out = dict(g)
    for pre, mx in (("policy.", CP), ("trunk.", CT)):
        ks = [k for k in g if k.startswith(pre)]
        nrm = torch.sqrt(sum((g[k] ** 2).sum() for k in ks) + 1e-30)
        s = torch.clamp(mx / nrm, max=1.0)
        for k in ks:
            out[k] = g[k] * s
    return out


def my_unroll(psi, th, lat, K, mix=0.0, alpha=ALPHA):
    for _ in range(K):
        g = clip(grad(my_loss, argnums=0)(th, th, psi, lat, mix))
        th = {n: th[n] - alpha * g[n] for n in th}
    return th


feas = ibq.a_mask
Utab = torch.randn(feas.shape, dtype=D, generator=torch.Generator().manual_seed(5)) * feas


def J(psi, K):
    lpa, _ = forward(my_unroll(psi, theta0, ib, K), ibq)
    return (lpa.exp() * feas * Utab).sum(-1).mean()


# enumerated query for C's engine: one row per feasible (decision, candidate)
rows = [(i, a) for i in range(ibq.n) for a in range(feas.shape[1]) if bool(feas[i, a])]
ri = torch.tensor([i for i, _ in rows])
ra = torch.tensor([a for _, a in rows])
nf = feas.sum(-1).to(D)
base_q = ibq.select(ri.tolist())
qb = dataclasses.replace(base_q, chosen=ra)
q = M.QueryData(batch=qb, utility=Utab[ri, ra], baseline=0.0, behaviour_logp=-torch.log(nf[ri]),
                weights=1.0 / (ibq.n * nf[ri]), groups=torch.arange(len(rows)),
                verified=torch.ones(len(rows), dtype=torch.bool))

names = list(psi0)
sizes = [psi0[n].numel() for n in names]
offs = [0]
for s in sizes:
    offs.append(offs[-1] + s)
g_rng = torch.Generator().manual_seed(11)
coords = [(n, int(j)) for n, s in zip(names, sizes) for j in torch.randperm(s, generator=g_rng)[:3].tolist()]
direction = {n: torch.randn(psi0[n].shape, dtype=D, generator=g_rng) for n in names}
dn = math.sqrt(sum(float((v ** 2).sum()) for v in direction.values()))
direction = {n: v / dn for n, v in direction.items()}
EPS = 1e-5

for K in (1, 2, 4, 8):
    f = lambda p, K=K: J(p, K)
    g_mine = grad(f)(psi0)
    fd, an = [], []
    for n, j in coords:
        pp = {k: v.clone() for k, v in psi0.items()}
        pp[n].view(-1)[j] += EPS
        jp = float(f(pp))
        pp[n].view(-1)[j] -= 2 * EPS
        jm = float(f(pp))
        fd.append((jp - jm) / (2 * EPS))
        an.append(float(g_mine[n].view(-1)[j]))
    fd_t, an_t = torch.tensor(fd, dtype=D), torch.tensor(an, dtype=D)
    e = float((fd_t - an_t).norm() / an_t.norm())
    C.check(f"K={K}: my dJ/dpsi vs central FD on {len(coords)} sampled psi coords (real subset, float64)",
            e < 1e-4, f"rel_err={e:.2e}")
    jp = float(f({n: psi0[n] + EPS * direction[n] for n in names}))
    jm = float(f({n: psi0[n] - EPS * direction[n] for n in names}))
    dd_fd, dd = (jp - jm) / (2 * EPS), sum(float((g_mine[n] * direction[n]).sum()) for n in names)
    C.check(f"K={K}: directional derivative vs FD", abs(dd - dd_fd) / max(abs(dd_fd), 1e-300) < 1e-5,
            f"autograd={dd:.6e} fd={dd_fd:.6e}")
    mg = M.MetaGradient(ml.inner.inner_loss, ml.inner.logp, inner_lr=ALPHA, K=K, weight_clip=1e9, min_ess=0.0,
                        min_ess_fraction=0.0, grad_transform=ml.engine.grad_transform)
    psi_req = {k: v.clone().requires_grad_(True) for k, v in psi0.items()}
    tr = mg.inner_update(theta0, psi_req, [ib.with_mix(0.0)], K=K)
    res = mg.meta_gradient(tr, q, "importance")
    ec_ = float(M.flat({n: res.grads[n] - g_mine[n] for n in names}).norm() / M.flat(g_mine).norm())
    C.check(f"K={K}: C engine (MetaLearner configuration) == my independent gradient", res.supported and ec_ < 1e-8,
            f"rel_err={ec_:.2e} supported={res.supported} {res.reason}")
    zero = [n for n in names if float(g_mine[n].abs().max()) == 0.0]
    C.check(f"K={K}: every psi tensor receives meta-gradient", not zero, f"zero tensors: {zero}")
    mgd = M.MetaGradient(lambda th, p, b, theta_sg=None: ml.inner.functional(th, p, b, detach_reward=True,
                                                                            theta_sg=theta_sg),
                         ml.inner.logp, inner_lr=ALPHA, K=K, weight_clip=1e9, min_ess=0.0, min_ess_fraction=0.0,
                         grad_transform=ml.engine.grad_transform)
    mgd.stopgrad_aware = True
    rd = mgd.meta_gradient(mgd.inner_update(theta0, psi_req, [ib.with_mix(0.0)], K=K), q, "importance")
    C.check(f"K={K}: R_psi detached inside the inner loss -> meta-gradient exactly 0", rd.grad_norm == 0.0,
            f"|g|={rd.grad_norm} vs {float(M.flat(g_mine).norm()):.3e}")

# ---- the live inner update equals my SGD on the documented loss, psi untouched -----------------------------------
recs = lt.records[:8]
ib32 = ml.inner.build_batch(store, recs)
th_before = ml.subset.snapshot(D)
h_psi = params_hash(ml.reward)
ml.mc.inner_lr = 0.1
rec = ml.policy_update(recs, steps=2, inner_lr=0.1)
th_live = ml.subset.snapshot(D)
mine = my_unroll({k: v.detach() for k, v in psi0.items()}, th_before, ib32.to(D), 2, mix=ib32.mix, alpha=0.1)
err = max(float((th_live[n] - mine[n]).abs().max()) for n in mine)
moved = max(float((th_live[n] - th_before[n]).abs().max()) for n in mine)
C.check("MetaLearner.policy_update == my K=2 SGD on the documented loss (float32 vs float64 reference)",
        err < 1e-4 * max(moved, 1e-6) / 1e-3 and moved > 0, f"max|live-mine|={err:.2e} max|update|={moved:.2e}")
C.check("psi bit-identical after the live policy update", params_hash(ml.reward) == h_psi)
C.info("policy_update record losses / KL", (rec.losses, rec.prediction_change))
sys.exit(C.finish())
