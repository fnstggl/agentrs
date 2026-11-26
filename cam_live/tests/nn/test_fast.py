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
    assert torch.allclose(epi, torch.ones(3), atol=1e-5) and torch.allclose(noise, torch.full((3,), 0.5), atol=1e-5)
    for _ in range(50):
        st.update(phi[0], 2.0, K.REAL)
    mean2, epi2, _ = st.predict(phi, K.REAL)
    assert float(epi2[0]) < 0.05 and abs(float(mean2[0]) - 2.0) < 0.1     # epistemic part shrinks with evidence


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


def test_laplace_update_is_the_exact_posterior_mode():
    """For one Poisson observation the update must land on the true MAP (checked by a generic optimiser)."""
    F = 4
    g = torch.Generator().manual_seed(7)
    A = torch.randn(F, F, generator=g, dtype=F64)
    cov = A @ A.T + 0.5 * torch.eye(F, dtype=F64)
    mu = torch.randn(F, generator=g, dtype=F64)
    phi = _unit(torch.randn(F, generator=g, dtype=F64))
    y, ex = 9.0, 2.0
    mu_n, cov_n, info = fast.glm_update(mu, cov, phi, y, K.COUNT, exposure=ex, n_newton=30)
    w = mu.clone().requires_grad_(True)
    lam = torch.linalg.inv(cov)
    opt = torch.optim.LBFGS([w], lr=1.0, max_iter=500, tolerance_grad=1e-14, tolerance_change=1e-16,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        eta = phi @ w
        loss = -(y * eta - ex * eta.exp()) + 0.5 * (w - mu) @ lam @ (w - mu)
        loss.backward()
        return loss

    opt.step(closure)
    assert torch.allclose(mu_n, w.detach(), atol=1e-6)
    h = ex * float((phi @ mu_n).exp())
    assert torch.allclose(cov_n, torch.linalg.inv(lam + h * torch.outer(phi, phi)), atol=1e-9)
    assert float(info["kl"]) > 0


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
