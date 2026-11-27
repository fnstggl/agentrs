"""G-nn probe: the Bayesian fast layer against independent float64 references (numpy / scipy only on the reference
side).

    .venv/bin/python cam_live/docs/verification/repro/nn/probe_fast.py

Sections: conjugate Gaussian / NIG closed forms; GLM Laplace updates against a reference Newton solve on the full
weight vector (moderate and extreme observations; the stateful path the agent runs); the shared dispersion; order
invariance; forgetting; predictive variance; calibration; invalid observations.
"""
import math
import sys

import _common  # noqa: F401
from _common import check, finish, info

import numpy as np
import torch
from scipy import optimize, special, stats

from cam_live.contracts import OutcomeKind as K
from cam_live.nn import fast
from cam_live.nn.fast import F64, FastBeliefState, FastPrior

rng = np.random.default_rng(0)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def T(x):
    return torch.tensor(np.asarray(x), dtype=F64)


# =============================================================================================== 1. Gaussian, known noise
Fd, n = 7, 80
prior = FastPrior(Fd, noise=0.3, noise_mode="fixed")
with torch.no_grad():                                        # a non-trivial learned prior (full covariance, mean)
    prior.mu0.copy_(torch.randn(prior.mu0.shape, generator=torch.Generator().manual_seed(1)) * 0.5)
    prior.chol_lower.copy_(torch.randn(prior.chol_lower.shape, generator=torch.Generator().manual_seed(2)) * 0.3)
p0 = {k: v.detach().numpy() for k, v in prior.prior(K.REAL).items()}
noise = float(p0["noise"])
Phi = unit(rng.standard_normal((n, Fd)))
w_true = rng.standard_normal(Fd)
y = Phi @ w_true + math.sqrt(noise) * rng.standard_normal(n)
st = FastBeliefState(prior, forgetting=1.0, change_h=0.0)
out = st.update(T(Phi), T(y), K.REAL)
lam0 = np.linalg.inv(p0["cov"])
lam = lam0 + Phi.T @ Phi / noise
cov_ref = np.linalg.inv(lam)
mu_ref = cov_ref @ (lam0 @ p0["mu"] + Phi.T @ y / noise)
post = st.posterior(K.REAL)
e_mu = float(np.abs(post["mu"].numpy() - mu_ref).max())
e_cov = float(np.abs(post["cov"].numpy() - cov_ref).max())
check("gaussian (known noise): sequential posterior == closed form (1e-10)", e_mu < 1e-10 and e_cov < 1e-10,
      f"|dmu|={e_mu:.2e} |dcov|={e_cov:.2e}")
S = Phi @ p0["cov"] @ Phi.T + noise * np.eye(n)
r = y - Phi @ p0["mu"]
ml = 0.5 * (n * math.log(2 * math.pi) + np.linalg.slogdet(S)[1] + r @ np.linalg.solve(S, r))
check("gaussian: sum of one-step surprises == -log marginal likelihood (1e-8)", abs(out["surprise"] - ml) < 1e-8,
      f"diff={abs(out['surprise'] - ml):.2e}")
# total KL from prior
kl_ref = 0.5 * (np.trace(lam0 @ cov_ref) - Fd + (mu_ref - p0["mu"]) @ lam0 @ (mu_ref - p0["mu"])
                + np.linalg.slogdet(p0["cov"])[1] - np.linalg.slogdet(cov_ref)[1])
check("gaussian: kl_from_prior == closed-form KL(posterior || prior) (1e-8)", abs(st.kl_from_prior(K.REAL) - kl_ref) < 1e-8,
      f"diff={abs(st.kl_from_prior(K.REAL) - kl_ref):.2e}")
# order invariance
perm = rng.permutation(n)
st2 = FastBeliefState(prior, forgetting=1.0, change_h=0.0)
st2.update(T(Phi[perm]), T(y[perm]), K.REAL)
d = float((st2.posterior(K.REAL)["mu"] - post["mu"]).abs().max())
dc = float((st2.posterior(K.REAL)["cov"] - post["cov"]).abs().max())
check("gaussian: order invariance of the posterior (1e-10)", max(d, dc) < 1e-10, f"|dmu|={d:.2e} |dcov|={dc:.2e}")
# predictive variance decreases monotonically with evidence at a fixed query
q = T(unit(rng.standard_normal(Fd)))
st3 = FastBeliefState(prior, forgetting=1.0, change_h=0.0)
vars_ = [float(st3.predict(q.double(), K.REAL)[1])]
for i in range(n):
    st3.update(T(Phi[i]), float(y[i]), K.REAL)
    vars_.append(float(st3.predict(q.double(), K.REAL)[1]))
check("gaussian: epistemic variance at a fixed query is non-increasing in the number of observations",
      all(b <= a + 1e-12 for a, b in zip(vars_, vars_[1:])), f"{vars_[0]:.4f} -> {vars_[-1]:.5f}")
# many near-collinear updates keep symmetry / PSD
st4 = FastBeliefState(prior, forgetting=1.0, change_h=0.0)
ph = unit(np.ones(Fd) + 1e-6 * rng.standard_normal((50000, Fd)))
st4.update(T(ph), T(1.0 + 0.1 * rng.standard_normal(50000)), K.REAL)
c4 = st4.posterior(K.REAL)["cov"]
lam4 = lam0 + ph.T @ ph / noise
e4 = float(np.abs(c4.numpy() - np.linalg.inv(lam4)).max())
check("gaussian: 50k near-collinear updates stay symmetric PSD and equal to the closed form (1e-8)",
      bool(torch.equal(c4, c4.T)) and float(torch.linalg.eigvalsh(c4).min()) > 0 and e4 < 1e-8, f"|dcov|={e4:.2e}")

# ========================================================================================================= 2. NIG
prior_n = FastPrior(Fd, noise=0.5, noise_mode="nig")
with torch.no_grad():
    prior_n.mu0.copy_(prior.mu0)
    prior_n.chol_lower.copy_(prior.chol_lower)
pn = {k: v.detach().numpy() for k, v in prior_n.prior(K.REAL).items()}
true_noise = 2.5
y2 = Phi @ w_true + math.sqrt(true_noise) * rng.standard_normal(n)
sn = FastBeliefState(prior_n, forgetting=1.0, change_h=0.0)
out_n = sn.update(T(Phi), T(y2), K.REAL)
L0 = np.linalg.inv(pn["V"])
Ln = L0 + Phi.T @ Phi
Vn = np.linalg.inv(Ln)
mun = Vn @ (L0 @ pn["mu"] + Phi.T @ y2)
an = float(pn["a"]) + n / 2
bn = float(pn["b"]) + 0.5 * (y2 @ y2 + pn["mu"] @ L0 @ pn["mu"] - mun @ Ln @ mun)
pp = sn.posterior(K.REAL)
errs = (float(np.abs(pp["mu"].numpy() - mun).max()), abs(float(pp["a"]) - an), abs(float(pp["b"]) - bn),
        float(np.abs(pp["cov"].numpy() - (bn / an) * Vn).max()))
check("NIG: sequential posterior (mu, a, b, (b/a)V) == closed form (1e-9)", max(errs) < 1e-9, f"errors={['%.1e' % e for e in errs]}")
# log marginal likelihood of the NIG model = multivariate Student-t
dof = 2 * float(pn["a"])
Sig = (float(pn["b"]) / float(pn["a"])) * (np.eye(n) + Phi @ pn["V"] @ Phi.T)
ml_t = -stats.multivariate_t(loc=Phi @ pn["mu"], shape=Sig, df=dof).logpdf(y2)
check("NIG: sum of Student-t surprises == -log marginal likelihood of the multivariate t (1e-7)", abs(out_n["surprise"] - ml_t) < 1e-7,
      f"diff={abs(out_n['surprise'] - ml_t):.2e}")
sn2 = FastBeliefState(prior_n, forgetting=1.0, change_h=0.0)
sn2.update(T(Phi[perm]), T(y2[perm]), K.REAL)
d = max(float((sn2.posterior(K.REAL)["mu"] - pp["mu"]).abs().max()), abs(float(sn2.posterior(K.REAL)["b"]) - float(pp["b"])))
check("NIG: order invariance (1e-9)", d < 1e-9, f"max diff={d:.2e}")
# predict() of the NIG mode: is (epistemic + noise) the predictive variance?
s0 = FastBeliefState(prior_n, forgetting=1.0, change_h=0.0)
m0, e0, n0 = (float(v) for v in s0.predict(q.double(), K.REAL))
a0, b0 = float(pn["a"]), float(pn["b"])
qv = float(q.numpy() @ pn["V"] @ q.numpy())
true_var = b0 / (a0 - 1.0) * (1.0 + qv)                       # variance of the Student-t predictive (dof 2a > 2)
info("NIG predict() at the prior: epistemic + noise", round(e0 + n0, 5))
info("NIG exact predictive variance of the Student-t (b/(a-1))(1+q)", round(true_var, 5))
check("NIG: predict() variance (epi + noise) equals the exact predictive variance of its own Student-t (5 %)",
      abs((e0 + n0) / true_var - 1.0) < 0.05,
      f"reported/true = {(e0 + n0) / true_var:.3f} (plug-in b/a instead of b/(a-1): under-states the variance by a/(a-1) = {a0 / (a0 - 1):.2f} at the prior)")

# =========================================================================================== 3. GLM Laplace updates


def ref_laplace(kind, mu, cov, phi, y, exposure=1.0, censored=False, disp=1.0):
    """Reference: Newton on the full weight vector (float64, to convergence) for the tempered log posterior."""
    lam = np.linalg.inv(cov)
    w = mu.copy()

    def gh(eta):
        if kind == K.COUNT:
            m = exposure * math.exp(eta)
            return (y - m), m
        if kind == K.TIME_TO_EVENT:
            m = (y / exposure) * math.exp(eta)
            return ((0.0 if censored else 1.0) - m), m
        p = special.expit(eta)
        return (y - exposure * p), exposure * p * (1 - p)

    def f(eta):                                       # scalar problem along Sigma phi (exactly equivalent, concave)
        g, _ = gh(eta)
        return g / disp - (eta - eta0) / s

    eta0, s = float(phi @ mu), float(phi @ cov @ phi)
    lo, hi = eta0 - 200.0, eta0 + 200.0
    eta = optimize.brentq(f, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=500)
    g, h = gh(eta)
    w = mu + cov @ phi * (eta - eta0) / s
    cov_n = np.linalg.inv(lam + (h / disp) * np.outer(phi, phi))
    return w, cov_n, eta


F2 = 5
A = rng.standard_normal((F2, F2))
cov0 = A @ A.T / F2 + 0.5 * np.eye(F2)
mu0 = 0.3 * rng.standard_normal(F2)
worst = {K.COUNT: 0.0, K.BINARY: 0.0, K.TIME_TO_EVENT: 0.0}
worst_c = dict(worst)
for trial in range(200):
    phi = unit(rng.standard_normal(F2))
    for kind in (K.COUNT, K.BINARY, K.TIME_TO_EVENT):
        if kind == K.COUNT:
            ex, yv, ce = float(rng.uniform(0.5, 4)), float(rng.poisson(3.0)), False
        elif kind == K.BINARY:
            ex, yv, ce = 1.0, float(rng.integers(0, 2)), False
        else:
            ex, yv, ce = 1.0, float(rng.exponential(1.0)), bool(rng.integers(0, 2))
        mu_n, cov_n, _ = fast.glm_update(T(mu0), T(cov0), T(phi), yv, kind, exposure=None if kind != K.COUNT else ex,
                                         censored=ce if kind == K.TIME_TO_EVENT else None, n_newton=12)
        w_ref, c_ref, _ = ref_laplace(kind, mu0, cov0, phi, yv, ex, ce)
        worst[kind] = max(worst[kind], float(np.abs(mu_n.numpy() - w_ref).max()))
        worst_c[kind] = max(worst_c[kind], float(np.abs(cov_n.numpy() - c_ref).max()))
for kind in worst:
    check(f"GLM functional update ({kind}, moderate data, dispersion 1, 12 Newton steps) == reference Laplace (1e-8)",
          worst[kind] < 1e-8 and worst_c[kind] < 1e-8, f"|dmu|={worst[kind]:.2e} |dcov|={worst_c[kind]:.2e}")

# ---- the stateful path the agent runs: FastBeliefState.update with the default prior ---------------------------------
pr = FastPrior(F2)                                            # defaults: prior_var 1, dispersion = 1 + softplus(-4)
disp = float(pr.dispersion().detach())
info("default learned dispersion applied to every GLM kind", round(disp, 6))
phi = unit(np.ones(F2))
for kind, yv, kw in ((K.BINARY, 1.0, {}), (K.TIME_TO_EVENT, 2.0, {"censored": True}), (K.COUNT, 4.0, {})):
    s = FastBeliefState(pr, forgetting=1.0, change_h=0.0)
    s.update(T(phi), yv, kind, **kw)
    got = s.posterior(kind)["mu"].numpy()
    exact, _, _ = ref_laplace(kind, np.zeros(F2), np.eye(F2), phi, yv, 1.0, bool(kw.get("censored", False)), disp=1.0)
    temp, _, _ = ref_laplace(kind, np.zeros(F2), np.eye(F2), phi, yv, 1.0, bool(kw.get("censored", False)), disp=disp)
    e_exact, e_temp = float(np.abs(got - exact).max()), float(np.abs(got - temp).max())
    ok = e_exact < 1e-8 if kind != K.COUNT else True
    check(f"stateful update ({kind}) == exact (untempered) Laplace update of its stated likelihood (1e-8)", ok,
          f"|dmu| vs exact={e_exact:.2e}, vs dispersion-tempered={e_temp:.2e}")
# what happens once the count dispersion has been trained (over-dispersed counts): binary / time-to-event are tempered too
with torch.no_grad():
    pr.disp_raw.fill_(2.0)
disp2 = float(pr.dispersion().detach())
for kind, yv, kw in ((K.BINARY, 1.0, {}), (K.TIME_TO_EVENT, 2.0, {"censored": True})):
    s = FastBeliefState(pr, forgetting=1.0, change_h=0.0)
    for _ in range(50):
        s.update(T(phi), yv, kind, **kw)
    eta = float(phi @ s.posterior(kind)["mu"].numpy())
    var = float(phi @ s.posterior(kind)["cov"].numpy() @ phi)
    # exact sequential Laplace with dispersion 1
    mu_e, cov_e = np.zeros(F2), np.eye(F2)
    for _ in range(50):
        mu_e, cov_e, _ = ref_laplace(kind, mu_e, cov_e, phi, yv, 1.0, bool(kw.get("censored", False)), disp=1.0)
    eta_e, var_e = float(phi @ mu_e), float(phi @ cov_e @ phi)
    check(f"count dispersion {disp2:.2f} (as after training on over-dispersed counts) leaves {kind} updates untouched",
          abs(eta - eta_e) < 1e-6 and abs(var / var_e - 1) < 1e-6,
          f"after 50 identical observations: eta {eta:.3f} vs exact {eta_e:.3f}; posterior variance {var:.4f} vs exact {var_e:.4f} "
          f"(x{var / var_e:.2f})")

# ---- extreme observations: is the Newton iteration converged? ------------------------------------------------------------
pr = FastPrior(F2)                                            # fresh default prior again
cases = [
    ("count y=0, exposure 1e6", K.COUNT, 0.0, {"exposure": 1e6}),
    ("count y=0, exposure 1e9", K.COUNT, 0.0, {"exposure": 1e9}),
    ("count y=3, exposure 1e7 (3 events in 1e7 impressions)", K.COUNT, 3.0, {"exposure": 1e7}),
    ("count y=1e6 (page views), exposure 1", K.COUNT, 1e6, {}),
    ("count y=1e9, exposure 1", K.COUNT, 1e9, {}),
    ("time-to-event censored at 86400 s (1 day), unit 1 s", K.TIME_TO_EVENT, 86400.0, {"censored": True}),
    ("time-to-event censored at 2.6e6 s (30 days), unit 1 s", K.TIME_TO_EVENT, 2.6e6, {"censored": True}),
    ("time-to-event event at 2.6e6 s (30 days), unit 1 s", K.TIME_TO_EVENT, 2.6e6, {"censored": False}),
    ("time-to-event censored at 3.15e7 s (1 year), unit 1 s", K.TIME_TO_EVENT, 3.15e7, {"censored": True}),
    ("binary 0 of 1e6 trials", K.BINARY, 0.0, {"exposure": 1e6}),
    ("binary 1e6 of 1e6 trials", K.BINARY, 1e6, {"exposure": 1e6}),
]
rows = []
for name, kind, yv, kw in cases:
    s = FastBeliefState(pr, forgetting=1.0, change_h=0.0)       # n_newton = 12 (the default the agent uses)
    res = s.update(T(phi), yv, kind, **kw)
    eta = float(phi @ s.posterior(kind)["mu"].numpy())
    ex = float(kw.get("exposure", 1.0)) if kind != K.TIME_TO_EVENT else 1.0
    _, _, eta_ref = ref_laplace(kind, np.zeros(F2), np.eye(F2), phi, yv, ex, bool(kw.get("censored", False)), disp=disp)
    # the functional default used for training the prior (online_nll: n_newton = 6)
    mu6, _, _ = fast.glm_update(T(np.zeros(F2)), T(np.eye(F2)), T(phi), yv, kind,
                                exposure=kw.get("exposure") if kind != K.TIME_TO_EVENT else None,
                                censored=kw.get("censored"), dispersion=disp, n_newton=6)
    eta6 = float(phi @ mu6.numpy())
    # brute-force reference of the predictive NLL the update reports as "surprise" (prior: eta ~ N(0, 1))
    grid = np.linspace(-45.0, 45.0, 900001)
    dens = stats.norm.pdf(grid, 0.0, 1.0)
    if kind == K.COUNT:
        loglik = stats.poisson.logpmf(yv, ex * np.exp(grid))
    elif kind == K.TIME_TO_EVENT:
        loglik = -np.exp(grid) * yv + (0.0 if kw.get("censored") else grid)
    else:
        loglik = stats.binom.logpmf(yv, ex, special.expit(grid))
    nll_ref = -math.log(np.trapezoid(dens * np.exp(loglik), grid))
    rows.append((name, eta_ref, eta, eta6, res["surprise"], nll_ref))
    check(f"Laplace mode reached, stateful n_newton=12: {name}", abs(eta - eta_ref) < 1e-2,
          f"eta_ref={eta_ref:.3f} eta_state={eta:.3f} (error {eta - eta_ref:+.3f} nats = x{math.exp(abs(eta - eta_ref)):.3g} in rate); "
          f"online_nll default n_newton=6: eta={eta6:.3f} (error {eta6 - eta_ref:+.3f}); "
          f"surprise reported {res['surprise']:.4f} vs brute force {nll_ref:.4f}")

# ---- surprise (adaptive Gauss-Hermite) against brute-force quadrature ---------------------------------------------------
worst = 0.0
for trial in range(60):
    phi_t = unit(rng.standard_normal(F2))
    eta0, s = float(phi_t @ mu0), float(phi_t @ cov0 @ phi_t)
    grid = np.linspace(eta0 - 14 * math.sqrt(s), eta0 + 14 * math.sqrt(s), 400001)
    dens = stats.norm.pdf(grid, eta0, math.sqrt(s))
    kind = (K.COUNT, K.BINARY, K.TIME_TO_EVENT)[trial % 3]
    if kind == K.COUNT:
        yv, ex = float(rng.poisson(4.0)), float(rng.uniform(0.5, 3))
        lik = stats.poisson.pmf(yv, ex * np.exp(grid))
        got = fast.nll_glm(T(mu0), T(cov0), T(phi_t), yv, kind, exposure=ex)
    elif kind == K.BINARY:
        yv = float(rng.integers(0, 2))
        lik = stats.bernoulli.pmf(yv, special.expit(grid))
        got = fast.nll_glm(T(mu0), T(cov0), T(phi_t), yv, kind)
    else:
        yv, ce = float(rng.exponential(1.0)), bool(rng.integers(0, 2))
        rate = np.exp(grid)
        lik = np.exp(-rate * yv) * (1.0 if ce else rate)
        got = fast.nll_glm(T(mu0), T(cov0), T(phi_t), yv, kind, censored=ce)
    ref = -math.log(np.trapezoid(dens * lik, grid))
    worst = max(worst, abs(ref - float(got)))
check("GLM predictive NLL (adaptive Gauss-Hermite) == brute-force integral (1e-5)", worst < 1e-5, f"max diff={worst:.2e}")

# ---- GLM predictive moments -------------------------------------------------------------------------------------------
st_c = FastBeliefState(FastPrior(F2), forgetting=1.0, change_h=0.0)
mean, epi, alea = (float(v) for v in st_c.predict(T(phi), K.COUNT, exposure=2.0))
mc = rng.standard_normal(2_000_000)                            # eta ~ N(0, 1) at the default prior with unit phi
lam_mc = 2.0 * np.exp(mc)
check("count predict(): mean / epistemic variance == Monte-Carlo lognormal-Poisson moments (3 %)",
      abs(mean / lam_mc.mean() - 1) < 0.03 and abs(epi / lam_mc.var() - 1) < 0.06,
      f"mean {mean:.3f} vs {lam_mc.mean():.3f}; epi {epi:.3f} vs {lam_mc.var():.3f}")
mean, epi, alea = (float(v) for v in st_c.predict(T(phi), K.BINARY))
p_mc = special.expit(mc)
check("binary predict(): mean / epistemic variance == Monte-Carlo (1 %)", abs(mean - p_mc.mean()) < 0.01 and abs(epi / p_mc.var() - 1) < 0.02,
      f"mean {mean:.4f} vs {p_mc.mean():.4f}; epi {epi:.4f} vs {p_mc.var():.4f}")

# =============================================================================== 4. forgetting, changing mechanisms
g1 = FastBeliefState(prior, forgetting=1.0, change_h=0.0)
g1.update(T(Phi), T(y), K.REAL)
mu_f, cov_f = fast.forget(g1._post[K.REAL]["mu"], g1._post[K.REAL]["cov"], T(p0["mu"]), T(p0["cov"]), 0.7)
lam_ref = 0.7 * np.linalg.inv(cov_ref) + 0.3 * lam0
mu_fr = np.linalg.solve(lam_ref, 0.7 * np.linalg.inv(cov_ref) @ mu_ref + 0.3 * lam0 @ p0["mu"])
check("forget(): precision / natural-mean interpolation towards the prior (1e-9)",
      float(np.abs(cov_f.numpy() - np.linalg.inv(lam_ref)).max()) < 1e-9 and float(np.abs(mu_f.numpy() - mu_fr).max()) < 1e-9)
check("forget(gamma=1) is a no-op and forget(gamma=0) returns the prior (1e-9)",
      float((fast.forget(g1._post[K.REAL]["mu"], g1._post[K.REAL]["cov"], T(p0["mu"]), T(p0["cov"]), 1.0)[1] - g1._post[K.REAL]["cov"]).abs().max()) < 1e-9
      and float((fast.forget(g1._post[K.REAL]["mu"], g1._post[K.REAL]["cov"], T(p0["mu"]), T(p0["cov"]), 0.0)[0] - T(p0["mu"])).abs().max()) < 1e-9)
# forgetting keeps uncertainty from collapsing: variance in an excited direction has a floor
gf = FastBeliefState(prior, forgetting=0.95, change_h=0.0)
gs = FastBeliefState(prior, forgetting=1.0, change_h=0.0)
ph1 = T(unit(np.ones(Fd)))
for _ in range(2000):
    gf.update(ph1, 1.0, K.REAL)
    gs.update(ph1, 1.0, K.REAL)
vf, vs = float(gf.predict(ph1, K.REAL)[1]), float(gs.predict(ph1, K.REAL)[1])
check("forgetting keeps a variance floor (static posterior keeps shrinking)", vf > 20 * vs, f"with forgetting {vf:.5f}, without {vs:.6f}")
# tracking after an abrupt change
w1, w2 = rng.standard_normal(Fd), -rng.standard_normal(Fd)
Pc = unit(rng.standard_normal((600, Fd)))
yc = np.concatenate([Pc[:300] @ w1, Pc[300:] @ w2]) + math.sqrt(noise) * rng.standard_normal(600)
errs = {}
for name, kw in (("static", dict(forgetting=1.0, change_h=0.0)), ("forgetting", dict(forgetting=0.97, change_h=0.0)),
                 ("cusum", dict(forgetting=1.0, change_h=12.0))):
    s = FastBeliefState(prior, **kw)
    s.update(T(Pc[:340]), T(yc[:340]), K.REAL)
    errs[name] = float(np.linalg.norm(s.posterior(K.REAL)["mu"].numpy() - w2))
    if name == "cusum":
        n_changes = s.posterior(K.REAL)["n_changes"]
check("after an abrupt mechanism change forgetting / CUSUM track the new weights better than the static posterior",
      errs["forgetting"] < 0.6 * errs["static"] and errs["cusum"] < 0.6 * errs["static"] and n_changes >= 1, str({k: round(v, 3) for k, v in errs.items()}))

# ====================================================================================================== 5. calibration
# data from the model's own prior (fixed noise): PIT uniform, coverage of the 90 % interval, z variance
cover, z2 = [], []
pcal = FastPrior(6, noise=0.4, noise_mode="fixed")
pc = {k: v.detach().numpy() for k, v in pcal.prior(K.REAL).items()}
for task in range(300):
    w = rng.multivariate_normal(pc["mu"], pc["cov"])
    Pt = unit(rng.standard_normal((25, 6)))
    yt = Pt @ w + math.sqrt(float(pc["noise"])) * rng.standard_normal(25)
    s = FastBeliefState(pcal, forgetting=1.0, change_h=0.0)
    for i in range(25):
        m, e, nz = (float(v) for v in s.predict(T(Pt[i]), K.REAL))
        zz = (yt[i] - m) / math.sqrt(e + nz)
        z2.append(zz * zz)
        cover.append(abs(zz) < 1.6449)
        s.update(T(Pt[i]), float(yt[i]), K.REAL)
check("calibration (known noise, data from the prior): 90 % interval coverage in [0.885, 0.915], E z^2 in [0.95, 1.05]",
      0.885 < np.mean(cover) < 0.915 and 0.95 < np.mean(z2) < 1.05, f"coverage={np.mean(cover):.4f} E z^2={np.mean(z2):.4f} (n={len(z2)})")
# NIG: data from the NIG prior; Student-t PIT coverage, and the Gaussian reading of predict()
cover_t, cover_g, by_k = [], [], {0: [], 1: [], 5: [], 24: []}
pnig = FastPrior(6, noise=0.4, noise_mode="nig")
pg = {k: v.detach().numpy() for k, v in pnig.prior(K.REAL).items()}
for task in range(400):
    s2 = 1.0 / rng.gamma(float(pg["a"]), 1.0 / float(pg["b"]))
    w = rng.multivariate_normal(pg["mu"], s2 * pg["V"])
    Pt = unit(rng.standard_normal((25, 6)))
    yt = Pt @ w + math.sqrt(s2) * rng.standard_normal(25)
    s = FastBeliefState(pnig, forgetting=1.0, change_h=0.0)
    for i in range(25):
        m, e, nz = (float(v) for v in s.predict(T(Pt[i]), K.REAL))
        post_i = s._post[K.REAL]
        dof_i = 2.0 * float(post_i["a"])
        zz = (yt[i] - m) / math.sqrt(e + nz)
        cover_t.append(abs(zz) < stats.t.ppf(0.95, dof_i))
        cover_g.append(abs(zz) < 1.6449)
        if i in by_k:
            by_k[i].append(abs(zz) < 1.6449)
        s.update(T(Pt[i]), float(yt[i]), K.REAL)
check("calibration (NIG, data from the NIG prior): Student-t 90 % interval coverage in [0.885, 0.915]", 0.885 < np.mean(cover_t) < 0.915,
      f"coverage={np.mean(cover_t):.4f}")
info("NIG predict() read as mean +- 1.645 sqrt(epi + noise): 90 % interval coverage by support size",
     {k: round(float(np.mean(v)), 3) for k, v in by_k.items()})
# GLM calibration: count predictive mean unbiased under the model after many updates
bias = []
for task in range(40):
    w = rng.standard_normal(4) * 0.7
    Pt = unit(rng.standard_normal((400, 4)))
    ex = rng.uniform(0.5, 4, 400)
    yt = rng.poisson(ex * np.exp(Pt @ w))
    s = FastBeliefState(FastPrior(4), forgetting=1.0, change_h=0.0)
    s.update(T(Pt[:300]), T(yt[:300].astype(float)), K.COUNT, exposure=T(ex[:300]))
    m, _, _ = s.predict(T(Pt[300:]), K.COUNT, exposure=T(ex[300:]))
    bias.append(float((m.numpy() - ex[300:] * np.exp(Pt[300:] @ w)).mean() / (ex[300:] * np.exp(Pt[300:] @ w)).mean()))
check("count: predictive mean after 300 observations is within 5 % of the true rate on average", abs(np.mean(bias)) < 0.05,
      f"mean relative bias={np.mean(bias):+.4f}")

# ================================================================================================ 6. invalid observations
s = FastBeliefState(FastPrior(F2), forgetting=1.0, change_h=0.0)
bad = []
for name, kind, yv, kw in (("negative count", K.COUNT, -5.0, {}), ("binary y=2 of 1 trial", K.BINARY, 2.0, {}),
                           ("negative time", K.TIME_TO_EVENT, -3.0, {})):
    before = s.posterior(kind)["mu"].clone()
    try:
        res = s.update(T(phi), yv, kind, **kw)
        moved = float((s.posterior(kind)["mu"] - before).abs().max())
        bad.append(f"{name}: accepted, surprise={res['surprise']}, |dmu|={moved:.3f}")
    except ValueError as e:
        pass
check("impossible observations (negative count / successes > trials / negative time) are rejected", not bad, "; ".join(bad))
for name, kind, yv in (("nan", K.REAL, float("nan")), ("inf", K.COUNT, float("inf"))):
    try:
        s.update(T(phi), yv, kind)
        check(f"non-finite outcome ({name}) rejected", False)
    except ValueError:
        check(f"non-finite outcome ({name}) rejected", True)
try:
    s.update(T(phi), 1.0, K.CATEGORICAL)
    check("categorical outcome rejected by the fast layer", False)
except ValueError:
    check("categorical outcome rejected by the fast layer (documented: change request A-5)", True)

# state round trip
import io
s = FastBeliefState(prior_n, forgetting=0.99)
s.update(T(Phi[:20]), T(y2[:20]), K.REAL)
s.update(T(Phi[:5]), T(np.arange(5.0)), K.COUNT)
buf = io.BytesIO()
torch.save(s.state_dict(), buf)
buf.seek(0)
r = FastBeliefState(FastPrior(Fd, noise_mode="nig"))
r.load_state_dict(torch.load(buf, weights_only=False))
a_, b_ = s.update(T(Phi[20]), 0.3, K.REAL), r.update(T(Phi[20]), 0.3, K.REAL)
check("fast state: state_dict -> torch.save -> load_state_dict -> next update is bit-exact",
      a_ == b_ and bool(torch.equal(s.posterior(K.REAL)["cov"], r.posterior(K.REAL)["cov"]))
      and bool(torch.equal(s.posterior(K.COUNT)["mu"], r.posterior(K.COUNT)["mu"])))

sys.exit(finish())
