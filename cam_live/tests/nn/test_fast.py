"""Fast Bayesian state: numerics against closed form, GLM convergence, forgetting, censoring (NA-07)."""
import math

import numpy as np
import pytest
import torch

from cam_live.contracts import OutcomeKind
from cam_live.nn import fast
from cam_live.nn.fast import F64, FastBeliefState, FastPrior

K = OutcomeKind


def _unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def _closed_form(Phi, y, mu0, cov0, noise):
    """Independent reference: normal equations solved with numpy in float64."""
    P, yv = Phi.numpy(), y.numpy()
    lam0 = np.linalg.inv(cov0.numpy())
    lam = lam0 + P.T @ P / noise
    cov = np.linalg.inv(lam)
    return cov @ (lam0 @ mu0.numpy() + P.T @ yv / noise), cov


def test_gaussian_fixed_noise_matches_closed_form():
    g = torch.Generator().manual_seed(1)
    F, T, noise = 8, 60, 0.3
    prior = FastPrior(F, noise=noise, noise_mode="fixed")
    noise = float(prior.noise().detach())                 # the parameter is stored in float32
    st = FastBeliefState(prior, change_h=0.0)
    Phi = _unit(torch.randn(T, F, generator=g, dtype=F64))
    w = torch.randn(F, generator=g, dtype=F64)
    y = Phi @ w + math.sqrt(noise) * torch.randn(T, generator=g, dtype=F64)
    out = st.update(Phi, y, K.REAL)
    p0 = prior.prior(K.REAL)
    mu_ref, cov_ref = _closed_form(Phi, y, p0["mu"].detach(), p0["cov"].detach(), noise)
    post = st.posterior(K.REAL)
    assert np.allclose(post["mu"].numpy(), mu_ref, atol=1e-10)
    assert np.allclose(post["cov"].numpy(), cov_ref, atol=1e-10)
    assert out["n"] == T and out["kl"] > 0 and math.isfinite(out["surprise"])
    # the batch (Cholesky) form agrees with the recursion
    mu_b, cov_b = fast.gaussian_fit(Phi, y, None, p0["mu"].detach(), p0["cov"].detach(), noise)
    assert torch.allclose(mu_b, post["mu"], atol=1e-10) and torch.allclose(cov_b, post["cov"], atol=1e-10)
    # the summed one-step predictive NLLs equal the exact log marginal likelihood
    S = Phi @ p0["cov"].detach() @ Phi.T + noise * torch.eye(T, dtype=F64)
    r = y - Phi @ p0["mu"].detach()
    ml = 0.5 * (T * math.log(2 * math.pi) + torch.logdet(S) + r @ torch.linalg.solve(S, r))
    assert abs(out["surprise"] - float(ml)) < 1e-8


def test_gaussian_many_collinear_updates_stay_exact_and_psd():
    F, noise, n = 6, 0.5, 20000
    prior = FastPrior(F, noise=noise, noise_mode="fixed")
    noise = float(prior.noise().detach())
    st = FastBeliefState(prior, change_h=0.0)
    phi = _unit(torch.tensor([1.0, -2.0, 0.5, 0.0, 3.0, 1.0], dtype=F64))
    g = torch.Generator().manual_seed(2)
    y = 1.7 + math.sqrt(noise) * torch.randn(n, generator=g, dtype=F64)
    st.update(phi.expand(n, F), y, K.REAL)
    post = st.posterior(K.REAL)
    p0 = prior.prior(K.REAL)
    mu_ref, cov_ref = _closed_form(phi.expand(n, F), y, p0["mu"].detach(), p0["cov"].detach(), noise)
    assert np.allclose(post["mu"].numpy(), mu_ref, atol=1e-9)
    assert np.allclose(post["cov"].numpy(), cov_ref, atol=1e-9)
    ev = torch.linalg.eigvalsh(post["cov"])
    assert float(ev.min()) > 0 and torch.equal(post["cov"], post["cov"].T)
    # the excited direction is known to ~noise/n, the orthogonal ones keep the prior variance
    v0 = float(p0["cov"][0, 0])                 # prior variance (1.0 up to float32 parameter storage)
    assert abs(float(phi @ post["cov"] @ phi) - 1.0 / (1.0 / v0 + n / noise)) < 1e-12
    assert abs(float(ev.max()) - v0) < 1e-9


def test_nig_matches_closed_form_and_learns_noise():
    g = torch.Generator().manual_seed(3)
    F, T, true_noise = 5, 400, 4.0
    prior = FastPrior(F, noise=0.5, noise_mode="nig")
    st = FastBeliefState(prior, change_h=0.0)
    Phi = _unit(torch.randn(T, F, generator=g, dtype=F64))
    w = torch.randn(F, generator=g, dtype=F64)
    y = Phi @ w + math.sqrt(true_noise) * torch.randn(T, generator=g, dtype=F64)
    st.update(Phi, y, K.REAL)
    p0 = {k: v.detach() for k, v in prior.prior(K.REAL).items()}
    # independent closed form of the normal-inverse-gamma posterior
    P, yv = Phi.numpy(), y.numpy()
    lam0 = np.linalg.inv(p0["V"].numpy())
    lam = lam0 + P.T @ P
    V = np.linalg.inv(lam)
    mu = V @ (lam0 @ p0["mu"].numpy() + P.T @ yv)
    a = float(p0["a"]) + T / 2
    b = float(p0["b"]) + 0.5 * (yv @ yv + p0["mu"].numpy() @ lam0 @ p0["mu"].numpy() - mu @ lam @ mu)
    post = st.posterior(K.REAL)
    assert np.allclose(post["mu"].numpy(), mu, atol=1e-9)
    assert abs(float(post["a"]) - a) < 1e-9 and abs(float(post["b"]) - b) < 1e-7
    assert np.allclose(post["cov"].numpy(), (b / a) * V, atol=1e-9)
    assert abs(float(post["noise"]) - true_noise) / true_noise < 0.2       # the noise scale is inferred
    mu_b, V_b, a_b, b_b = fast.nig_fit(Phi, y, None, p0["mu"], p0["V"], p0["a"], p0["b"])
    assert torch.allclose(mu_b, post["mu"], atol=1e-9) and abs(float(b_b) - b) < 1e-7


@pytest.mark.parametrize("mode", ["fixed", "nig"])
def test_predict_returns_mean_epistemic_noise(mode):
    F = 4
    st = FastBeliefState(FastPrior(F, noise=0.5, noise_mode=mode), change_h=0.0)
    phi = _unit(torch.ones(3, F))
    mean, epi, noise = st.predict(phi, K.REAL)
    assert mean.shape == epi.shape == noise.shape == (3,) and mean.dtype == torch.float32
    if mode == "fixed":
        assert torch.allclose(epi, torch.ones(3), atol=1e-5) and torch.allclose(noise, torch.full((3,), 0.5), atol=1e-5)
    else:
        # prior: a0 = 2, b0 = a0 * 0.5 = 1, V = I / 0.5 -> q = 2; E[s2] = b / (a - 1) = 1
        assert torch.allclose(noise, torch.ones(3), atol=1e-5) and torch.allclose(epi, torch.full((3,), 2.0), atol=1e-5)
    for _ in range(50):
        st.update(phi[0], 2.0, K.REAL)
    mean2, epi2, _ = st.predict(phi, K.REAL)
    assert float(epi2[0]) < 0.05 and abs(float(mean2[0]) - 2.0) < 0.1     # epistemic part shrinks with evidence


def test_nig_predict_variance_is_the_student_t_variance():
    """epistemic + noise of the inferred-noise Gaussian is the variance of its own Student-t predictive (the old
    plug-in b / a under-stated it by a / (a - 1), a factor 2 at the default prior)."""
    g = torch.Generator().manual_seed(21)
    F = 6
    prior = FastPrior(F, noise=0.4, noise_mode="nig")
    st = FastBeliefState(prior, change_h=0.0)
    Phi = _unit(torch.randn(40, F, generator=g, dtype=F64))
    y = Phi @ torch.randn(F, generator=g, dtype=F64) + 0.7 * torch.randn(40, generator=g, dtype=F64)
    q = _unit(torch.randn(5, F, generator=g, dtype=F64))
    for k in (0, 1, 3, 40):
        st.reset()
        if k:
            st.update(Phi[:k], y[:k], K.REAL)
        m, epi, noise = st.predict(q, K.REAL)
        s = st._post[K.REAL]
        loc, scale2, dof = fast.predict_nig(s["mu"], s["cov"], s["a"], s["b"], q)
        ref = torch.distributions.StudentT(dof, loc, scale2.sqrt()).variance        # independent: torch's Student-t
        assert torch.allclose((epi + noise).double(), ref, rtol=1e-6), (k, epi + noise, ref)
        assert torch.allclose(m.double(), loc, atol=1e-12)
        # the epistemic part is the posterior variance of the mean phi^T w (law of total variance)
        _, lin_var = st.linear(q, K.REAL)
        assert torch.allclose(lin_var, epi, rtol=1e-6)
    # Monte-Carlo: the variance of phi^T w under joint posterior draws of (w, s2) is the reported epistemic part
    w = st.sample_weights(200_000, K.REAL, torch.Generator().manual_seed(3))
    assert abs(float((w @ q[0]).var()) / float(epi[0]) - 1.0) < 0.03


def _glm_data(kind, w, n, g, F):
    Phi = _unit(torch.randn(n, F, generator=g, dtype=F64)) * 1.0
    eta = Phi @ w
    if kind == K.COUNT:
        ex = torch.rand(n, generator=g, dtype=F64) * 4 + 0.5
        y = torch.poisson(ex * eta.exp(), generator=g)
        return Phi, y, ex, None
    if kind == K.BINARY:
        y = torch.bernoulli(torch.sigmoid(eta), generator=g)
        return Phi, y, None, None
    t = -torch.log(torch.rand(n, generator=g, dtype=F64)) / eta.exp()       # exponential with rate exp(eta)
    c = -torch.log(torch.rand(n, generator=g, dtype=F64)) / 0.7             # independent censoring times
    return Phi, torch.minimum(t, c), None, (c < t)


@pytest.mark.parametrize("kind,n,tol", [(K.COUNT, 3000, 0.15), (K.BINARY, 6000, 0.3), (K.TIME_TO_EVENT, 4000, 0.2)])
def test_glm_updates_converge_to_generating_weights(kind, n, tol):
    g = torch.Generator().manual_seed(11)
    F = 5
    w = torch.tensor([1.0, -0.8, 0.5, 0.0, 1.2], dtype=F64)
    Phi, y, ex, cens = _glm_data(kind, w, n, g, F)
    prior = FastPrior(F, prior_var=4.0)
    st = FastBeliefState(prior, change_h=0.0)
    before = float((st.posterior(kind)["mu"] - w).norm())
    out = st.update(Phi, y, kind, exposure=ex, censored=cens)
    post = st.posterior(kind)
    err = float((post["mu"] - w).norm())
    assert err < tol < before, (err, before)
    assert float(torch.linalg.eigvalsh(post["cov"]).min()) > 0
    assert float(torch.diagonal(post["cov"]).max()) < 0.2            # the posterior contracted
    assert out["kl"] > 0 and all(math.isfinite(s) for s in out["surprises"])


def test_censoring_is_used_not_ignored():
    """Treating censored times as events biases the rate upwards; the censoring-aware update does not."""
    g = torch.Generator().manual_seed(5)
    F, n = 3, 4000
    w = torch.tensor([0.5, -1.0, 0.8], dtype=F64)
    Phi, t, _, cens = _glm_data(K.TIME_TO_EVENT, w, n, g, F)
    assert 0.2 < float(cens.double().mean()) < 0.8
    ok = FastBeliefState(FastPrior(F, prior_var=4.0), change_h=0.0)
    naive = FastBeliefState(FastPrior(F, prior_var=4.0), change_h=0.0)
    ok.update(Phi, t, K.TIME_TO_EVENT, censored=cens)
    naive.update(Phi, t, K.TIME_TO_EVENT, censored=torch.zeros_like(cens))
    e_ok = float((ok.posterior(K.TIME_TO_EVENT)["mu"] - w).norm())
    e_naive = float((naive.posterior(K.TIME_TO_EVENT)["mu"] - w).norm())
    assert e_ok < 0.2 and e_naive > 2 * e_ok
    # one fully censored observation only lowers the rate, and its surprise is -log S(t)
    st = FastBeliefState(FastPrior(F), change_h=0.0)
    phi = _unit(torch.ones(F, dtype=F64))
    m0, _, _ = st.predict(phi, K.TIME_TO_EVENT)
    st.update(phi, 5.0, K.TIME_TO_EVENT, censored=True)
    m1, _, _ = st.predict(phi, K.TIME_TO_EVENT)
    assert float(m1) > float(m0)
    # the time unit (exposure) rescales consistently
    a, b = FastBeliefState(FastPrior(F), change_h=0.0), FastBeliefState(FastPrior(F), change_h=0.0)
    a.update(phi, 2.0, K.TIME_TO_EVENT)
    b.update(phi, 7200.0, K.TIME_TO_EVENT, exposure=3600.0)
    assert torch.allclose(a.posterior(K.TIME_TO_EVENT)["mu"], b.posterior(K.TIME_TO_EVENT)["mu"], atol=1e-12)


def _ref_glm_map(kind, mu, cov, phi, y, exposure=1.0, censored=False, disp=1.0):
    """Independent float64 reference (numpy / scipy only): the exact MAP of the full weight vector lies on
    mu + Sigma phi t, so it is found by scipy's bracketing root finder (brentq) on the scalar stationarity
    condition; the Laplace covariance is formed at that MAP. Returns (w, cov, eta, kkt) where ``kkt(w)`` is the
    length of the full-vector Newton step of the negative log posterior from ``w`` (its distance to the MAP)."""
    from scipy import optimize, special
    mu, cov, phi = (np.asarray(t, dtype=np.float64) for t in (mu, cov, phi))
    lam = np.linalg.inv(cov)

    def terms(eta):                         # loglik'(eta), -loglik''(eta) of the stated likelihood
        if kind == K.COUNT:
            m = exposure * math.exp(eta)
            return y - m, m
        if kind == K.TIME_TO_EVENT:
            m = (y / exposure) * math.exp(eta)
            return (0.0 if censored else 1.0) - m, m
        p, q = special.expit(eta), special.expit(-eta)
        return (y - exposure + exposure * q) if eta > 0 else (y - exposure * p), exposure * p * q

    eta0, s = float(phi @ mu), float(phi @ cov @ phi)
    f = lambda e: terms(e)[0] / disp - (e - eta0) / s
    lo, hi = eta0 - 1.0, eta0 + 1.0
    while f(lo) < 0:
        lo = eta0 - 2 * (eta0 - lo)
    while f(hi) > 0:
        hi = eta0 + 2 * (hi - eta0)
    eta = optimize.brentq(f, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=1000)
    w = mu + cov @ phi * (eta - eta0) / s

    def kkt(wx):                            # length of the full-vector Newton step from wx (distance to the MAP)
        wx = np.asarray(wx, dtype=np.float64)
        gr, h = terms(float(phi @ wx))
        grad = lam @ (wx - mu) - gr / disp * phi
        return float(np.linalg.norm(np.linalg.solve(lam + h / disp * np.outer(phi, phi), grad)))

    h = terms(eta)[1] / disp                # Laplace covariance (lam + h phi phi^T)^-1 by Sherman-Morrison
    sp = cov @ phi
    return w, cov - (h / (1.0 + h * s)) * np.outer(sp, sp), eta, kkt


#: observations far from the prior predictor, in the units the agent uses (seconds, raw exposures)
EXTREME = [
    ("time to event censored at 30 days, unit 1 s", K.TIME_TO_EVENT, 2.592e6, dict(censored=True)),
    ("time to event censored at one year, unit 1 s", K.TIME_TO_EVENT, 3.1536e7, dict(censored=True)),
    ("event after 30 days, unit 1 s", K.TIME_TO_EVENT, 2.592e6, dict(censored=False)),
    ("event after 1 ms, unit 1 s", K.TIME_TO_EVENT, 1e-3, dict(censored=False)),
    ("3 events in 1e7 impressions", K.COUNT, 3.0, dict(exposure=1e7)),
    ("0 events at exposure 1e9", K.COUNT, 0.0, dict(exposure=1e9)),
    ("1e9 events at exposure 1", K.COUNT, 1e9, {}),
    ("5 events at exposure 1e-12", K.COUNT, 5.0, dict(exposure=1e-12)),
    ("0 successes of 1e6 trials", K.BINARY, 0.0, dict(exposure=1e6)),
    ("1e6 successes of 1e6 trials", K.BINARY, 1e6, dict(exposure=1e6)),
    ("moderate count", K.COUNT, 4.0, dict(exposure=2.0)),
    ("one success", K.BINARY, 1.0, {}),
]


@pytest.mark.parametrize("name,kind,y,kw", EXTREME, ids=[e[0] for e in EXTREME])
def test_laplace_mode_is_exact_far_from_the_prior(name, kind, y, kw):
    """The stateful update, the functional update and the training path (online_nll) all land on the exact Laplace
    mode of the full weight vector (independent scipy optimiser), with a non-trivial prior. The previous fixed
    number of clipped Newton steps stopped short by up to 6 nats (a rate wrong by x339) for these inputs."""
    F = 5
    g = torch.Generator().manual_seed(17)
    A = torch.randn(F, F, generator=g, dtype=F64)
    for scale in (1.0, 25.0):                                   # prior variance of the linear predictor ~1 and ~25
        prior = FastPrior(F, prior_var=scale)
        with torch.no_grad():
            prior.mu0[prior.index(kind)].copy_(0.3 * torch.randn(F, generator=g))
            prior.chol_lower[prior.index(kind)].copy_(0.3 * torch.randn(F * (F - 1) // 2, generator=g))
        p = {k: v.detach() for k, v in prior.prior(kind).items()}
        phi = _unit(torch.randn(F, generator=g, dtype=F64) + A[0])
        st = FastBeliefState(prior, change_h=0.0)
        out = st.update(phi, y, kind, **kw)
        post = st.posterior(kind)
        ex = float(kw.get("exposure", 1.0))
        w_ref, c_ref, eta_ref, kkt = _ref_glm_map(kind, p["mu"], p["cov"], phi, y, ex, bool(kw.get("censored", False)))
        eta = float(phi @ post["mu"])
        assert abs(eta - eta_ref) < 1e-8 * (1 + abs(eta_ref)), (name, scale, eta, eta_ref)
        assert np.allclose(post["mu"].numpy(), w_ref, rtol=1e-8, atol=1e-8)
        assert kkt(post["mu"].numpy()) < 1e-8 * (1 + float(post["mu"].norm()))     # stationary in weight space
        assert np.allclose(post["cov"].numpy(), c_ref, rtol=1e-9, atol=1e-10)
        assert out["converged"] and 1 <= out["newton_iters"] <= 20, out
        # the functional update and the prequential training path use the same converged mode
        mu_f, cov_f, info = fast.glm_update(p["mu"], p["cov"], phi, y, kind, kw.get("exposure"), kw.get("censored"))
        assert torch.allclose(mu_f, post["mu"], rtol=0, atol=1e-12) and bool(info["converged"])
        ex_t = None if "exposure" not in kw else torch.tensor([[ex]], dtype=F64)
        ce_t = None if "censored" not in kw else torch.tensor([[float(kw["censored"])]], dtype=F64)
        two = fast.online_nll(prior.prior(kind), phi.expand(1, 2, F), torch.full((1, 2), y, dtype=F64), kind,
                              exposure=None if ex_t is None else ex_t.expand(1, 2),
                              censored=None if ce_t is None else ce_t.expand(1, 2))
        st2 = FastBeliefState(prior, change_h=0.0)
        ref2 = st2.update(phi.expand(2, F), torch.full((2,), y, dtype=F64), kind, **kw)["surprises"]
        # log-likelihood terms of size y * |eta| cancel in the surprise: float64 resolves it to ~1e-16 of that size
        tol = 1e-9 + 1e-14 * abs(y) * (1 + abs(eta_ref))
        assert torch.allclose(two[0].detach(), torch.tensor(ref2, dtype=F64), rtol=1e-10, atol=tol), (two, ref2)


def test_glm_surprise_matches_accurate_quadrature_for_extreme_observations():
    """The reported surprise is the predictive NLL -log int p(y | eta) N(eta; eta0, s) d eta (scipy quad around
    the integrand's peak as reference)."""
    from scipy import integrate, special
    phi = _unit(torch.ones(4, dtype=F64))
    for name, kind, y, kw in EXTREME:
        st = FastBeliefState(FastPrior(4), change_h=0.0)
        got = st.update(phi, y, kind, **kw)["surprise"]
        ex, cens = float(kw.get("exposure", 1.0)), bool(kw.get("censored", False))
        if kind == K.COUNT:
            ll = lambda e: y * (math.log(ex) + e) - ex * math.exp(e) - math.lgamma(y + 1)
        elif kind == K.TIME_TO_EVENT:
            ll = lambda e: (0.0 if cens else e) - y * math.exp(e)
        else:
            ll = lambda e: (y * e - ex * np.logaddexp(0.0, e) + math.lgamma(ex + 1) - math.lgamma(y + 1)
                            - math.lgamma(ex - y + 1))
        lp = lambda e: ll(e) - 0.5 * e * e - 0.5 * math.log(2 * math.pi)          # prior eta ~ N(0, 1)
        grid = np.linspace(-60, 60, 120001)
        vals = np.array([lp(e) for e in grid])
        peak = float(grid[int(np.argmax(vals))])
        top = float(vals.max())
        width = 1.0 / math.sqrt(max(1.0 + abs(float(np.gradient(np.gradient(vals, grid), grid)[int(np.argmax(vals))])), 1.0))
        val, _ = integrate.quad(lambda e: math.exp(lp(e) - top), peak - 40 * width, peak + 40 * width,
                                points=[peak], limit=500, epsabs=0, epsrel=1e-13)
        ref = -(top + math.log(val))
        assert abs(got - ref) < 1e-6 * (1 + abs(ref)), (name, got, ref)


def test_dispersion_tempers_counts_only():
    """A trained count dispersion must not change binary or time-to-event updates (it tempered all three kinds)."""
    F = 4
    phi = _unit(torch.ones(F, dtype=F64))
    prior = FastPrior(F)
    assert float(prior.dispersion()) == 1.0                    # exactly Poisson at initialisation
    with torch.no_grad():
        prior.disp_raw.fill_(math.log(3.0))
    p = {k: v.detach() for k, v in prior.prior(K.BINARY).items()}
    for kind, y, kw in ((K.BINARY, 1.0, {}), (K.TIME_TO_EVENT, 2.0, {"censored": True}),
                        (K.TIME_TO_EVENT, 0.5, {}), (K.COUNT, 7.0, {"exposure": 2.0})):
        st = FastBeliefState(prior, change_h=0.0)
        for _ in range(20):
            st.update(phi, y, kind, **kw)
        mu_e, cov_e = p["mu"].numpy(), p["cov"].numpy()
        disp = 3.0 if kind == K.COUNT else 1.0
        for _ in range(20):
            mu_e, cov_e, _, _ = _ref_glm_map(kind, mu_e, cov_e, phi.numpy(), y, float(kw.get("exposure", 1.0)),
                                             bool(kw.get("censored", False)), disp=disp)
        post = st.posterior(kind)
        assert np.allclose(post["mu"].numpy(), mu_e, atol=1e-8), (kind, post["mu"], mu_e)
        assert np.allclose(post["cov"].numpy(), cov_e, atol=1e-9), kind
    # the training path: binary / time-to-event NLLs do not depend on the dispersion, counts do
    g = torch.Generator().manual_seed(4)
    Phi = _unit(torch.randn(2, 6, F, generator=g, dtype=F64))
    for kind, y in ((K.BINARY, torch.bernoulli(torch.full((2, 6), 0.4, dtype=F64), generator=g)),
                    (K.TIME_TO_EVENT, torch.rand(2, 6, generator=g, dtype=F64) * 3),
                    (K.COUNT, torch.poisson(torch.full((2, 6), 3.0, dtype=F64), generator=g))):
        a = fast.online_nll(prior.prior(kind), Phi, y, kind)
        with torch.no_grad():
            prior.disp_raw.fill_(0.0)
        b = fast.online_nll(prior.prior(kind), Phi, y, kind)
        with torch.no_grad():
            prior.disp_raw.fill_(math.log(3.0))
        same = torch.allclose(a, b, rtol=0, atol=1e-12)
        assert same == (kind != K.COUNT), kind


def test_mode_search_reports_iterations_and_non_convergence():
    phi = _unit(torch.ones(3, dtype=F64))
    mu, cov = torch.zeros(3, dtype=F64), torch.eye(3, dtype=F64)
    _, _, ok = fast.glm_update(mu, cov, phi, 3.15e7, K.TIME_TO_EVENT, censored=True)
    assert bool(ok["converged"]) and int(ok["newton_iters"]) <= 10
    _, _, capped = fast.glm_update(mu, cov, phi, 3.15e7, K.TIME_TO_EVENT, censored=True, n_newton=1)
    assert not bool(capped["converged"]) and int(capped["newton_iters"]) == 1
    st = FastBeliefState(FastPrior(3), change_h=0.0, n_newton=1)
    out = st.update(phi, 3.15e7, K.TIME_TO_EVENT, censored=True)
    assert out["converged"] is False and out["newton_iters"] == 1 and st.stats["unconverged_updates"] == 1
    st = FastBeliefState(FastPrior(3), change_h=0.0)
    out = st.update(phi.expand(3, 3), torch.tensor([3.15e7, 1.0, 0.0]), K.TIME_TO_EVENT,
                    censored=torch.tensor([True, False, True]))
    assert out["converged"] is True and 1 <= out["newton_iters"] <= 10 and st.stats["unconverged_updates"] == 0
    assert "newton_iters" not in st.update(phi, 1.0, K.REAL)


@pytest.mark.parametrize("kind,y,kw,why", [
    (K.COUNT, -5.0, {}, "negative"), (K.COUNT, 2.0, {"exposure": 0.0}, "zero exposure"),
    (K.COUNT, 2.0, {"exposure": -1.0}, "negative"), (K.BINARY, 2.0, {}, "successes"),
    (K.BINARY, -1.0, {}, "successes"), (K.BINARY, 1.0, {"exposure": -3.0}, "trials"),
    (K.TIME_TO_EVENT, -3.0, {}, "negative"), (K.TIME_TO_EVENT, 3.0, {"exposure": 0.0}, "time unit"),
    (K.TIME_TO_EVENT, 3.0, {"censored": 0.5}, "censored"), (K.COUNT, float("inf"), {}, "finite")])
def test_impossible_observations_are_rejected_without_touching_the_state(kind, y, kw, why):
    phi = _unit(torch.ones(4, dtype=F64))
    st = FastBeliefState(FastPrior(4), change_h=0.0)
    st.update(phi, 1.0, kind)
    before = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in st._post[kind].items()}
    with pytest.raises(ValueError, match=why):
        st.update(phi.expand(2, 4), torch.tensor([1.0, y], dtype=F64), kind, **kw)
    for k, v in st._post[kind].items():
        assert (torch.equal(v, before[k]) if isinstance(v, torch.Tensor) else v == before[k]), k
    # the training path rejects the same rows, but ignores rows that are masked out
    yy = torch.tensor([[1.0, y]], dtype=F64)
    kws = {n: torch.tensor([[1.0, float(v)]], dtype=F64) for n, v in kw.items()}
    with pytest.raises(ValueError, match=why):
        fast.online_nll(FastPrior(4).prior(kind), phi.expand(1, 2, 4), yy, kind, **kws)
    if math.isfinite(y):
        nll = fast.online_nll(FastPrior(4).prior(kind), phi.expand(1, 2, 4), yy, kind,
                              mask=torch.tensor([[True, False]]), **kws)
        assert torch.isfinite(nll).all() and float(nll[0, 1]) == 0.0


def test_forgetting_and_change_detection_track_a_mechanism_change():
    g = torch.Generator().manual_seed(4)
    F, n, noise = 4, 400, 0.1
    w1 = torch.tensor([1.0, 0.0, -1.0, 0.5], dtype=F64)
    w2 = torch.tensor([-1.0, 1.5, 1.0, -0.5], dtype=F64)
    Phi = _unit(torch.randn(2 * n, F, generator=g, dtype=F64))
    y = torch.cat([Phi[:n] @ w1, Phi[n:] @ w2]) + math.sqrt(noise) * torch.randn(2 * n, generator=g, dtype=F64)
    prior = FastPrior(F, noise=noise, noise_mode="fixed")
    static = FastBeliefState(prior, forgetting=1.0, change_h=0.0)
    forget = FastBeliefState(prior, forgetting=0.97, change_h=0.0)
    detect = FastBeliefState(prior, forgetting=1.0, change_h=12.0)
    for st in (static, forget, detect):
        st.update(Phi[:n], y[:n], K.REAL)
        assert float((st.posterior(K.REAL)["mu"] - w1).norm()) < 0.25
    assert detect.posterior(K.REAL)["n_changes"] == 0                 # no false alarm on the stationary half
    k = 40                                                            # 40 observations after the change
    outs = [st.update(Phi[n:n + k], y[n:n + k], K.REAL) for st in (static, forget, detect)]
    e_static, e_forget, e_detect = [float((st.posterior(K.REAL)["mu"] - w2).norm()) for st in (static, forget, detect)]
    assert e_forget < 0.5 * e_static and e_detect < 0.5 * e_static, (e_static, e_forget, e_detect)
    assert outs[2]["change_detected"] and detect.posterior(K.REAL)["n_changes"] >= 1
    # stabilised forgetting is bounded by the prior: an unexcited state returns to it and never beyond
    idle = FastBeliefState(prior, forgetting=0.9, change_h=0.0)
    phi = _unit(torch.tensor([1.0, 0, 0, 0], dtype=F64))
    for _ in range(300):
        idle.update(phi, 1.0, K.REAL)
    cov = idle.posterior(K.REAL)["cov"]
    v0 = float(prior.prior(K.REAL)["cov"][0, 0])
    assert float(torch.linalg.eigvalsh(cov).max()) <= v0 + 1e-9
    assert abs(float(cov[1, 1]) - v0) < 1e-9


def test_change_detector_false_alarm_rate_is_low_under_the_model():
    g = torch.Generator().manual_seed(9)
    F, n, noise = 6, 5000, 0.5
    w = torch.randn(F, generator=g, dtype=F64)
    Phi = _unit(torch.randn(n, F, generator=g, dtype=F64))
    y = Phi @ w + math.sqrt(noise) * torch.randn(n, generator=g, dtype=F64)
    st = FastBeliefState(FastPrior(F, noise=noise, noise_mode="fixed"))
    st.update(Phi, y, K.REAL)
    assert st.posterior(K.REAL)["n_changes"] <= 2


def test_sample_weights_clone_reset_and_state_dict():
    F = 5
    prior = FastPrior(F)
    st = FastBeliefState(prior, change_h=0.0)
    g = torch.Generator().manual_seed(0)
    Phi = _unit(torch.randn(30, F, generator=g, dtype=F64))
    for kind, y in ((K.REAL, Phi[:, 0] * 3), (K.COUNT, torch.ones(30) * 2), (K.BINARY, (Phi[:, 1] > 0).double())):
        st.update(Phi, y, kind)
    for kind in (K.REAL, K.COUNT, K.BINARY, K.TIME_TO_EVENT):
        a = st.sample_weights(4000, kind, torch.Generator().manual_seed(1))
        b = st.sample_weights(4000, kind, torch.Generator().manual_seed(1))
        assert a.shape == (4000, F) and torch.equal(a, b)
        post = st.posterior(kind)
        assert torch.allclose(a.mean(0), post["mu"], atol=0.12)
        if kind != K.REAL:                      # Gaussian marginal; the real kind is Student-t (wider)
            assert torch.allclose(torch.cov(a.T), post["cov"], atol=0.12)
    c = st.clone()
    c.update(Phi[0], 100.0, K.REAL)
    assert st.n_updates(K.REAL) == 30 and c.n_updates(K.REAL) == 31
    assert not torch.equal(st.posterior(K.REAL)["mu"], c.posterior(K.REAL)["mu"])
    sd = st.state_dict()
    other = FastBeliefState(FastPrior(F))
    other.load_state_dict(sd)
    for kind in (K.REAL, K.COUNT, K.BINARY):
        assert torch.equal(other.posterior(kind)["mu"], st.posterior(kind)["mu"])
        assert torch.equal(other.posterior(kind)["cov"], st.posterior(kind)["cov"])
    assert st.kl_from_prior(K.COUNT) > 0
    st.reset(K.COUNT)
    assert st.n_updates(K.COUNT) == 0 and abs(st.kl_from_prior(K.COUNT)) < 1e-9 and st.n_updates(K.REAL) == 30
    st.reset()
    assert st.n_updates(K.REAL) == 0
    with pytest.raises(ValueError):
        st.update(Phi[0], 1.0, K.CATEGORICAL)
    with pytest.raises(ValueError):
        st.update(Phi[0], float("nan"), K.REAL)
    assert not st.supports(K.CATEGORICAL) and st.supports(K.COUNT)


def test_kl_and_surprise_are_the_stated_quantities():
    F = 3
    g = torch.Generator().manual_seed(8)
    A = torch.randn(F, F, generator=g, dtype=F64)
    cov = A @ A.T + torch.eye(F, dtype=F64)
    mu = torch.randn(F, generator=g, dtype=F64)
    phi = _unit(torch.randn(F, generator=g, dtype=F64))
    mu_n, cov_n, info = fast.gaussian_update(mu, cov, phi, 1.3, 0.4)
    p = torch.distributions.MultivariateNormal(mu_n, cov_n)
    q = torch.distributions.MultivariateNormal(mu, cov)
    assert abs(float(torch.distributions.kl_divergence(p, q)) - float(info["kl"])) < 1e-10
    pred = torch.distributions.Normal(phi @ mu, (phi @ cov @ phi + 0.4).sqrt())
    assert abs(float(-pred.log_prob(torch.tensor(1.3, dtype=F64))) - float(info["surprise"])) < 1e-12
    # GLM: KL of the Gaussian approximation, and surprise against brute-force numerical integration
    mu_g, cov_g, info_g = fast.glm_update(mu, cov, phi, 4.0, K.COUNT, exposure=1.5, n_newton=20)
    p = torch.distributions.MultivariateNormal(mu_g, cov_g)
    assert abs(float(torch.distributions.kl_divergence(p, q)) - float(info_g["kl"])) < 1e-8
    eta0, s = float(phi @ mu), float(phi @ cov @ phi)
    grid = torch.linspace(eta0 - 12 * math.sqrt(s), eta0 + 12 * math.sqrt(s), 200001, dtype=F64)
    dens = torch.exp(-0.5 * (grid - eta0) ** 2 / s) / math.sqrt(2 * math.pi * s)
    pmf = torch.exp(4.0 * (math.log(1.5) + grid) - 1.5 * grid.exp() - math.lgamma(5.0))
    ref = -math.log(float(torch.trapezoid(dens * pmf, grid)))
    assert abs(ref - float(info_g["surprise"])) < 1e-6


def test_functional_form_is_differentiable_alpaca_style():
    """Gradients of the prequential NLL reach the features and every prior parameter, for each likelihood."""
    F, T, B = 4, 6, 3
    g = torch.Generator().manual_seed(6)
    for mode in ("nig", "fixed"):
        prior = FastPrior(F, noise_mode=mode)
        for kind in (K.REAL, K.COUNT, K.BINARY, K.TIME_TO_EVENT):
            raw = torch.randn(B, T, F, generator=g, dtype=F64, requires_grad=True)
            phi = raw / raw.norm(dim=-1, keepdim=True)
            y = torch.rand(B, T, generator=g, dtype=F64).round() if kind == K.BINARY else \
                torch.rand(B, T, generator=g, dtype=F64) * 3
            if kind == K.COUNT:
                y = y.round()
            mask = torch.ones(B, T, dtype=torch.bool)
            mask[0, -2:] = False
            nll = fast.online_nll(prior.prior(kind), phi, y, kind, mask=mask, noise_mode=mode)
            assert nll.shape == (B, T) and float(nll[0, -1]) == 0.0 and torch.isfinite(nll).all()
            prior.zero_grad()
            nll.sum().backward()
            assert raw.grad is not None and float(raw.grad.abs().sum()) > 0
            assert float(prior.mu0.grad.abs().sum()) > 0 and float(prior.chol_diag_raw.grad.abs().sum()) > 0
            assert float(prior.chol_lower.grad.abs().sum()) > 0
            if kind == K.REAL:
                assert float(prior.log_noise.grad.abs()) > 0
            if kind == K.COUNT:
                assert float(prior.disp_raw.grad.abs()) > 0
    # update_mask: query rows are scored but not learned from
    prior = FastPrior(F, noise_mode="fixed")
    phi = _unit(torch.randn(1, T, F, generator=g, dtype=F64))
    y = torch.randn(1, T, generator=g, dtype=F64)
    sup = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    a = fast.online_nll(prior.prior(K.REAL), phi, y, K.REAL, update_mask=sup, noise_mode="fixed")
    y2 = y.clone()
    y2[0, 3] = 50.0
    b = fast.online_nll(prior.prior(K.REAL), phi, y2, K.REAL, update_mask=sup, noise_mode="fixed")
    assert torch.allclose(a[0, 4:], b[0, 4:]) and not torch.allclose(a[0, 3], b[0, 3])


def test_gradcheck_of_the_functional_updates():
    F = 3
    g = torch.Generator().manual_seed(12)
    A = torch.randn(F, F, generator=g, dtype=F64)
    cov = (A @ A.T + torch.eye(F, dtype=F64)).requires_grad_(True)
    mu = torch.randn(F, generator=g, dtype=F64, requires_grad=True)
    phi = torch.randn(F, generator=g, dtype=F64, requires_grad=True)

    def f_gauss(mu, cov, phi):
        m, c, info = fast.gaussian_update(mu, fast._sym(cov), phi, 0.7, 0.3)
        return m.sum() + c.sum() + info["surprise"] + info["kl"]

    def f_glm(mu, cov, phi):
        m, c, info = fast.glm_update(mu, fast._sym(cov), phi, 3.0, K.COUNT, exposure=2.0, n_newton=6)
        return m.sum() + c.sum() + info["surprise"] + info["kl"]

    assert torch.autograd.gradcheck(f_gauss, (mu, cov, phi), atol=1e-6)
    assert torch.autograd.gradcheck(f_glm, (mu, cov, phi), atol=1e-5)

    # far from the prior predictor (one year of censoring in seconds; 0 of 1e6 trials): the gradient is the
    # implicit-function gradient of the converged mode
    def f_tte(mu, cov, phi):
        m, c, info = fast.glm_update(mu, fast._sym(cov), phi, 3.15e7, K.TIME_TO_EVENT, censored=True)
        return m.sum() + c.sum() + 1e-2 * info["surprise"] + info["kl"]

    def f_bin(mu, cov, phi):
        m, c, info = fast.glm_update(mu, fast._sym(cov), phi, 0.0, K.BINARY, exposure=1e6)
        return m.sum() + c.sum() + 1e-2 * info["surprise"] + info["kl"]

    assert torch.autograd.gradcheck(f_tte, (mu, cov, phi), atol=1e-5, rtol=1e-4)
    assert torch.autograd.gradcheck(f_bin, (mu, cov, phi), atol=1e-5, rtol=1e-4)
