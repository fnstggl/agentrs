"""Independent meta-gradient reference on a small configuration (spec 7.3, 7.4, 7.7; MR-04, MR-05, MR-07).

My own implementation, written from the spec and the documented inner loss (docs/C/API.md), not from C's engine:
  * theta = {W1, b1, w2, vw}: softmax policy over A candidate actions + linear state value; psi = a tanh-bounded MLP
    reward R_psi(phi(x, a), mean_a phi(x, .)).
  * L_inner = -mean log pi(a|x) * (std(R_psi) - V_sg) + 0.5 * 0.5 * mean (V - std(R_psi))^2 - 0.01 * mean H
    (V_sg: the value evaluated on theta_k as a constant *for the inner derivative only*).
  * theta_{k+1} = theta_k - alpha * grad_theta L, K = 1, 2, 4, 8, nested torch.func.grad (no torch.autograd.grad).
  * J(psi) = exact expected verified utility of pi_{theta_K(psi)} on query contexts: mean_x sum_a pi(a|x) U(x, a).
Checks: (1) my torch.func dJ/dpsi vs central finite differences on ALL psi coordinates (float64);
(2) C's MetaGradient (unroll / checkpoint; importance estimator over an enumerated query with logged uniform
behaviour; on-policy estimator with pi-weights) with C's own policy_gradient_inner_loss equals my gradient;
(3) reward detached / unroll stopped -> exactly zero; (4) a plain v.detach() baseline loses the path for K >= 2;
(5) the inner-data score term: E over all on-policy support samples of C's estimator equals the exact gradient
of the expected objective (enumerated), and dropping the score term does not.
"""
import itertools
import math
import sys

import torch
from torch.func import grad

import _common as C
from cam_live.reward import meta as M

D = torch.float64
torch.manual_seed(1234)
nS, nQ, A, d, h, h2 = 10, 6, 4, 5, 8, 6
ALPHA, ENT, VC = 0.3, 0.01, 0.5


def logits(th, phi):
    return torch.tanh(phi @ th["W1"] + th["b1"]) @ th["w2"]


def value(th, phi):
    return phi.mean(1) @ th["vw"]


def reward(psi, phi, a):
    pa = phi[torch.arange(phi.shape[0]), a]
    x = torch.cat([pa, phi.mean(1)], -1)
    return torch.tanh(torch.tanh(x @ psi["P1"] + psi["c1"]) @ psi["p2"] + psi["c2"])


def stdz(r):
    d_ = r - r.mean()
    return d_ / torch.sqrt((d_ ** 2).mean() + 1e-12)


def my_inner_loss(th, th_sg, psi, phi, a):
    lpa = torch.log_softmax(logits(th, phi), -1)
    lp = lpa[torch.arange(phi.shape[0]), a]
    G = stdz(reward(psi, phi, a))
    v, vb = value(th, phi), value(th_sg, phi)
    ent = -(lpa.exp() * lpa).sum(-1)
    return -(lp * (G - vb)).mean() + VC * (0.5 * (v - G) ** 2).mean() - ENT * ent.mean()


def my_unroll(psi, theta0, batches, K):
    th = theta0
    for k in range(K):
        phi, a = batches[k % len(batches)]
        g = grad(my_inner_loss, argnums=0)(th, th, psi, phi, a)     # th passed twice: 2nd is constant at this level
        th = {n: th[n] - ALPHA * g[n] for n in th}
    return th


def J_exact(th, phi_q, U):
    return (torch.softmax(logits(th, phi_q), -1) * U).sum(-1).mean()


def fd_all(f, psi, eps=1e-6):
    out = {}
    for n, v in psi.items():
        g = torch.zeros_like(v)
        for i in range(v.numel()):
            p = {k: x.clone() for k, x in psi.items()}
            p[n].view(-1)[i] += eps
            jp = float(f(p))
            p[n].view(-1)[i] -= 2 * eps
            jm = float(f(p))
            g.view(-1)[i] = (jp - jm) / (2 * eps)
        out[n] = g
    return out


def rel(a, b):
    fa, fb = M.flat(a), M.flat(b)
    return float((fa - fb).norm() / fb.norm().clamp_min(1e-300))


theta0 = {"W1": torch.randn(d, h, dtype=D) * 0.6, "b1": torch.randn(h, dtype=D) * 0.1,
          "w2": torch.randn(h, dtype=D) * 0.6, "vw": torch.randn(d, dtype=D) * 0.3}
psi0 = {"P1": torch.randn(2 * d, h2, dtype=D) * 0.6, "c1": torch.randn(h2, dtype=D) * 0.1,
        "p2": torch.randn(h2, dtype=D) * 0.8, "c2": torch.zeros((), dtype=D)}
batches = [(torch.randn(nS, A, d, dtype=D), torch.randint(A, (nS,))) for _ in range(3)]
phi_q, U = torch.randn(nQ, A, d, dtype=D), torch.randn(nQ, A, dtype=D)

# ---- engine side: C's policy_gradient_inner_loss on the same functions ----------------------------------------------
logp_fn = lambda th, b: torch.log_softmax(logits(th, b["phi"]), -1)[torch.arange(b["phi"].shape[0]), b["a"]]
reward_fn = lambda psi, b: reward(psi, b["phi"], b["a"])
value_fn = lambda th, b: value(th, b["phi"])
ent_fn = lambda th, b: (lambda l: -(l.exp() * l).sum(-1))(torch.log_softmax(logits(th, b["phi"]), -1))
c_loss = M.policy_gradient_inner_loss(logp_fn, reward_fn, value_fn=value_fn, value_coef=VC, entropy_fn=ent_fn,
                                      entropy_coef=ENT, normalise=True)
eb = [{"phi": p, "a": a} for p, a in batches]
th_test = {k: v.clone() for k, v in theta0.items()}
C.check("C policy_gradient_inner_loss equals the documented formula (value)",
        abs(float(c_loss(th_test, psi0, eb[0], theta_sg=th_test)) - float(my_inner_loss(th_test, th_test, psi0, *batches[0]))) < 1e-12)

rows_phi = phi_q.repeat_interleave(A, 0)
rows_a = torch.arange(A).repeat(nQ)
q_imp = M.QueryData(batch={"phi": rows_phi, "a": rows_a}, utility=U.reshape(-1), baseline=0.0,
                    behaviour_logp=torch.full((nQ * A,), -math.log(A), dtype=D),
                    groups=torch.arange(nQ * A), verified=torch.ones(nQ * A, dtype=torch.bool))

worst = {}
for K in (1, 2, 4, 8):
    J = lambda psi, K=K: J_exact(my_unroll(psi, theta0, batches, K), phi_q, U)
    g_mine = grad(J)(psi0)
    g_fd = fd_all(J, psi0)
    e_fd = rel(g_mine, g_fd)
    C.check(f"K={K}: my torch.func dJ/dpsi vs central FD on all {sum(v.numel() for v in psi0.values())} psi coords",
            e_fd < 1e-6, f"rel_err={e_fd:.2e} |g|={float(M.flat(g_mine).norm()):.3e}")
    mg = M.MetaGradient(c_loss, logp_fn, inner_lr=ALPHA, K=K, weight_clip=1e9, min_ess=0.0, min_ess_fraction=0.0)
    psi_req = {k: v.clone().requires_grad_(True) for k, v in psi0.items()}
    for mode in ("unroll", "checkpoint"):
        tr = mg.inner_update(theta0, psi_req, eb, K=K, mode=mode)
        res = mg.meta_gradient(tr, q_imp, "importance")
        e = rel(res.grads, g_mine) if res.supported else float("nan")
        worst[(K, mode)] = e
        C.check(f"K={K} {mode}: C engine (importance, enumerated query, logged uniform mu) == my gradient",
                res.supported and e < 1e-8, f"rel_err={e:.2e} reason={res.reason!r}")
        jK = float(J_exact(tr.detached(), phi_q, U))
        C.check(f"K={K} {mode}: C engine objective == exact J(theta_K)", abs(res.objective - jK) < 1e-12,
                f"{res.objective:.12f} vs {jK:.12f}")
    # on-policy estimator with pi_K weights (the query is "sampled" from pi_K in expectation)
    tr = mg.inner_update(theta0, psi_req, eb, K=K)
    pK = torch.softmax(logits(tr.detached(), phi_q), -1).reshape(-1).detach()
    q_on = M.QueryData(batch={"phi": rows_phi, "a": rows_a}, utility=U.reshape(-1), baseline=0.0,
                       weights=pK / nQ, groups=torch.arange(nQ * A))
    res = mg.meta_gradient(tr, q_on, "on_policy")
    C.check(f"K={K}: C engine on-policy estimator with pi_K weights == my gradient", rel(res.grads, g_mine) < 1e-8,
            f"rel_err={rel(res.grads, g_mine):.2e}")
    # first-order approximation (documented as biased)
    trf = mg.inner_update(theta0, psi_req, eb, K=K, mode="first_order")
    C.info(f"K={K} first_order rel_err vs exact", rel(mg.meta_gradient(trf, q_imp, "importance").grads, g_mine))
    # ablations: reward detached, unroll stopped
    det_loss = M.policy_gradient_inner_loss(logp_fn, lambda p, b: reward_fn({k: v.detach() for k, v in p.items()}, b),
                                            value_fn=value_fn, value_coef=VC, entropy_fn=ent_fn, entropy_coef=ENT)
    mgd = M.MetaGradient(det_loss, logp_fn, inner_lr=ALPHA, K=K, weight_clip=1e9, min_ess=0.0, min_ess_fraction=0.0)
    rd = mgd.meta_gradient(mgd.inner_update(theta0, psi_req, eb, K=K), q_imp, "importance")
    C.check(f"K={K}: reward detached -> meta-gradient exactly 0 (and non-zero otherwise)",
            rd.grad_norm == 0.0 and float(M.flat(g_mine).norm()) > 1e-4, f"|g_det|={rd.grad_norm}")
    est_stop = mg.outer_surrogate(tr.detached(), q_imp, "importance")
    C.check(f"K={K}: theta_K detached -> surrogate carries no graph to psi", not est_stop.surrogate.requires_grad)
    # a plain v.detach() baseline (not stop-gradient aware) drops the theta_k(psi) -> V path for K >= 2
    plain = M.policy_gradient_inner_loss(logp_fn, reward_fn, value_fn=value_fn, value_coef=VC, entropy_fn=ent_fn,
                                         entropy_coef=ENT)
    plain.stopgrad_aware = False
    mgp = M.MetaGradient(plain, logp_fn, inner_lr=ALPHA, K=K, weight_clip=1e9, min_ess=0.0, min_ess_fraction=0.0)
    ep = rel(mgp.meta_gradient(mgp.inner_update(theta0, psi_req, eb, K=K), q_imp, "importance").grads, g_mine)
    C.info(f"K={K} rel_err of a plain v.detach() baseline (C's documented stage-1 bug)", ep)
    if K == 1:
        C.check("K=1: plain v.detach() baseline is exact at K=1", ep < 1e-10, f"{ep:.2e}")
    else:
        C.check(f"K={K}: plain v.detach() baseline is measurably biased (the guarded path matters)", ep > 1e-6,
                f"{ep:.2e}")

# ---- inner-data score term (on-policy support at step 1), exact enumeration ------------------------------------------
n1 = 2
phi_s1 = torch.randn(n1, A - 1, d, dtype=D)            # 3 candidates: 3^2 = 9 support samples
phi_q3, U3 = phi_q[:, :A - 1], U[:, :A - 1]
b0 = batches[0][0][:, :A - 1], batches[0][1].clamp(max=A - 2)


def unroll2(psi, a1):
    th1 = my_unroll(psi, theta0, [b0], 1)
    return th1, my_unroll(psi, th1, [(phi_s1, a1)], 1)


def expected_J(psi):
    th1 = my_unroll(psi, theta0, [b0], 1)
    lp1 = torch.log_softmax(logits(th1, phi_s1), -1)
    tot = 0.0
    for combo in itertools.product(range(A - 1), repeat=n1):
        a1 = torch.tensor(combo)
        p = torch.exp(lp1[torch.arange(n1), a1].sum())
        th2 = my_unroll(psi, th1, [(phi_s1, a1)], 1)
        tot = tot + p * J_exact(th2, phi_q3, U3)
    return tot


g_exact = grad(expected_J)(psi0)
mg = M.MetaGradient(c_loss, logp_fn, inner_lr=ALPHA, K=2, weight_clip=1e9, min_ess=0.0, min_ess_fraction=0.0)
psi_req = {k: v.clone().requires_grad_(True) for k, v in psi0.items()}
rows_phi3, rows_a3 = phi_q3.repeat_interleave(A - 1, 0), torch.arange(A - 1).repeat(nQ)
q3 = M.QueryData(batch={"phi": rows_phi3, "a": rows_a3}, utility=U3.reshape(-1), baseline=0.0,
                 behaviour_logp=torch.full((nQ * (A - 1),), -math.log(A - 1), dtype=D),
                 groups=torch.arange(nQ * (A - 1)))
th1 = my_unroll(psi0, theta0, [b0], 1)
p1 = torch.softmax(logits(th1, phi_s1), -1).detach()
avg_with, avg_without = None, None
for combo in itertools.product(range(A - 1), repeat=n1):
    a1 = torch.tensor(combo)
    w = float(p1[torch.arange(n1), a1].prod())
    sup = [{"phi": b0[0], "a": b0[1]}, {"phi": phi_s1, "a": a1}]
    for flag, acc in ((True, "with"), (False, "without")):
        r = mg.meta_gradient(mg.inner_update(theta0, psi_req, sup, K=2, on_policy=[False, flag]), q3, "importance",
                             score_baseline=0.0)
        g = {k: w * v for k, v in r.grads.items()}
        if acc == "with":
            avg_with = g if avg_with is None else {k: avg_with[k] + g[k] for k in g}
        else:
            avg_without = g if avg_without is None else {k: avg_without[k] + g[k] for k in g}
e_with, e_without = rel(avg_with, g_exact), rel(avg_without, g_exact)
C.check("score term: E_support[C estimator] == exact gradient of the expected objective (enumerated, K=2)",
        e_with < 1e-8, f"rel_err={e_with:.2e}")
C.check("score term: the test is sensitive (dropping the score term is biased)", e_without > 1e-3,
        f"rel_err_without={e_without:.2e}")
g_fd_e = fd_all(expected_J, psi0)
C.check("score term: my enumerated exact gradient vs central FD", rel(g_exact, g_fd_e) < 1e-6,
        f"rel_err={rel(g_exact, g_fd_e):.2e}")
sys.exit(C.finish())
