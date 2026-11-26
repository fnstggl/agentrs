"""Value of information: exact linear-Gaussian VOI, knowledge gradient, GLM sampling, noisy TV, realised regret."""
import math

import numpy as np
import pytest
import scipy.stats as st
import torch

from cam_live.predict import voi as V
from cam_live.predict.voi import (DownstreamValueEstimator, GLMPosterior, LinearGaussianPosterior, VOICosts,
                                  downstream_loss, expected_max_affine, knowledge_gradient_gaussian,
                                  score_downstream, spearman, voi_glm, voi_linear_gaussian, voi_sampling)

F64 = torch.float64


def test_expected_max_affine_matches_quadrature():
    rng = np.random.default_rng(0)
    z = np.linspace(-9, 9, 400001)
    w = st.norm.pdf(z) * (z[1] - z[0])
    for k in (1, 2, 5, 12):
        a, b = rng.normal(size=k), rng.normal(size=k)
        ref = float(np.sum(np.max(a[:, None] + b[:, None] * z[None], 0) * w))
        np.testing.assert_allclose(expected_max_affine(a, b), ref, rtol=1e-7, atol=1e-9)
    # equal slopes, dominated lines, zero slopes
    np.testing.assert_allclose(expected_max_affine([0.0, 1.0, -3.0], [0.0, 0.0, 0.0]), 1.0)
    np.testing.assert_allclose(expected_max_affine([0.0, 0.0], [1.0, -1.0]), math.sqrt(2 / math.pi), rtol=1e-12)
    assert expected_max_affine([0.0, -50.0], [0.0, 1.0]) < 1e-12        # a line that practically never wins


def test_conjugate_update_matches_batch_bayes(gen):
    f = 5
    phi = torch.randn(40, f, generator=gen, dtype=F64)
    w = torch.randn(f, generator=gen, dtype=F64)
    y = phi @ w + 0.7 * torch.randn(40, generator=gen, dtype=F64)
    post = LinearGaussianPosterior.prior(f, scale=2.0, noise_var=0.49)
    for i in range(40):
        post = post.update(phi[i], y[i])
    prec = torch.eye(f, dtype=F64) / 4.0 + phi.t() @ phi / 0.49
    cov = torch.linalg.inv(prec)
    np.testing.assert_allclose(post.cov, cov, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(post.mean, cov @ (phi.t() @ y) / 0.49, rtol=1e-8, atol=1e-10)
    m, v = post.predict(phi[:3])
    np.testing.assert_allclose(v, torch.einsum("pi,ij,pj->p", phi[:3], cov, phi[:3]), rtol=1e-8)
    s = post.sample_weights(40000, gen)
    np.testing.assert_allclose(s.mean(0), post.mean, atol=0.01)
    np.testing.assert_allclose(torch.cov(s.t()), post.cov, atol=0.01)


def test_linear_gaussian_voi_equals_knowledge_gradient_and_sampling(gen):
    k = 6
    mu = torch.tensor([0.2, 0.0, -0.3, 0.25, 0.1, -1.0], dtype=F64)
    s2 = torch.tensor([0.5, 1.0, 2.0, 0.05, 0.3, 0.2], dtype=F64)
    noise = torch.tensor([0.5, 0.5, 1.0, 0.1, 2.0, 0.3], dtype=F64)
    post = LinearGaussianPosterior(mu, torch.diag(s2))
    eye = torch.eye(k, dtype=F64)                                        # independent arms: probe i observes arm i
    out = voi_linear_gaussian(post, eye, eye, noise)
    kg = knowledge_gradient_gaussian(mu, s2, noise)
    np.testing.assert_allclose(out["gross"], kg, rtol=1e-9, atol=1e-12)
    assert bool((out["gross"] >= 0).all()) and float(out["base"]) == 0.25
    # the generic sampling estimator with the real posterior update agrees within Monte-Carlo error
    def sample(a, n):
        m, v = post.predict(eye[a])
        return list(m + torch.sqrt(v + noise[a]) * torch.randn(n, generator=gen, dtype=F64))

    mc = voi_sampling(range(k), sample, lambda a, y: post.update(eye[a], y, float(noise[a])),
                      lambda b: b.mean, post, n=1500)
    assert bool(((mc["gross"] - kg).abs() < 4 * mc["stderr"] + 1e-3).all())
    # correlated beliefs: observing one arm teaches about another; exact value vs brute-force simulation
    a_mat = torch.randn(4, 4, generator=gen, dtype=F64)
    post_c = LinearGaussianPosterior(torch.randn(4, generator=gen, dtype=F64) * 0.2, a_mat @ a_mat.t() / 4)
    phi_p = torch.randn(3, 4, generator=gen, dtype=F64)
    phi_d = torch.randn(5, 4, generator=gen, dtype=F64)
    exact = voi_linear_gaussian(post_c, phi_p, phi_d, 0.4)["gross"]
    for j in range(3):
        m, v = post_c.predict(phi_p[j])
        ys = m + torch.sqrt(v + 0.4) * torch.randn(1500, generator=gen, dtype=F64)
        vals = torch.stack([(phi_d @ post_c.update(phi_p[j], y, 0.4).mean).max() for y in ys])
        se = vals.std() / math.sqrt(1500)
        assert abs(float(vals.mean() - (phi_d @ post_c.mean).max() - exact[j])) < 4 * float(se) + 1e-3


def _noisy_tv_setup():
    """6 weights. Decision candidates only involve weights 0-2. Probe 0 measures an uncertain decision-relevant
    direction; probe 1 is a 'noisy TV': it reads weight 5 (very uncertain, irrelevant to every decision) through
    enormous observation noise; probe 2 returns pure noise."""
    mean = torch.tensor([0.1, 0.0, -0.1, 0.0, 0.0, 0.0], dtype=F64)
    cov = torch.diag(torch.tensor([1.0, 1.0, 1.0, 0.1, 0.1, 400.0], dtype=F64))
    post = LinearGaussianPosterior(mean, cov, noise_var=0.25)
    phi_dec = torch.tensor([[1.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0], [0, 0, 1.0, 0, 0, 0]], dtype=F64)
    phi_probe = torch.tensor([[0, 1.0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1.0], [0, 0, 0, 0, 0, 0]], dtype=F64)
    noise = torch.tensor([0.25, 900.0, 2500.0], dtype=F64)
    return post, phi_probe, phi_dec, noise


def test_noisy_irrelevant_observation_has_zero_voi_but_high_surprise(gen):
    post, phi_probe, phi_dec, noise = _noisy_tv_setup()
    out = voi_linear_gaussian(post, phi_probe, phi_dec, noise)
    gross, surprise = out["gross"], out["expected_surprise"]
    assert float(gross[0]) > 0.2                                         # the informative probe is worth something
    assert float(gross[1]) < 1e-12 and float(gross[2]) < 1e-12           # the TV and the noise are worth nothing
    assert float(surprise[1]) > float(surprise[0]) + 2.0                 # ... although they are far more surprising
    assert float(surprise[2]) > float(surprise[0]) + 2.0
    assert float(out["info_gain_nats"][1]) > 0.15                        # the TV even reduces (irrelevant) uncertainty
    assert int(gross.argmax()) == 0 and int(surprise.argmin()) == 0      # ranking by surprise is exactly backwards
    # realised: the mean squared prediction error on the TV is orders of magnitude above the informative probe
    w = post.sample_weights(4000, gen)
    y = w @ phi_probe.t() + torch.sqrt(noise) * torch.randn(4000, 3, generator=gen, dtype=F64)
    sq_err = ((y - phi_probe @ post.mean) ** 2).mean(0)
    assert float(sq_err[1]) > 500 * float(sq_err[0])
    # and the decision is never changed by it
    best0 = int((phi_dec @ post.mean).argmax())
    for i in range(50):
        assert int((phi_dec @ post.update(phi_probe[1], y[i, 1], float(noise[1])).mean).argmax()) == best0


def test_costs_time_and_opportunity_cost_are_netted():
    post, phi_probe, phi_dec, noise = _noisy_tv_setup()
    gross = voi_linear_gaussian(post, phi_probe, phi_dec, noise)["gross"]
    costs = VOICosts(cost=torch.tensor([0.05, 0.0, 0.0]), elapsed_s=torch.tensor([600.0, 600.0, 0.0]),
                     opportunity_rate=1e-4)
    out = voi_linear_gaussian(post, phi_probe, phi_dec, noise, costs)
    np.testing.assert_allclose(out["net"], gross - torch.tensor([0.05 + 0.06, 0.06, 0.0]), atol=1e-12)
    assert float(out["net"][1]) < 0 == float(out["net"][2])              # a worthless probe that takes time is a loss
    # discounting applies to the value obtained after waiting; a direct effect is credited
    c2 = VOICosts(elapsed_s=1000.0, discount_rate=1e-3, direct_value=0.3)
    base = float(out["base"])
    net2 = voi_linear_gaussian(post, phi_probe, phi_dec, noise, c2)["net"]
    np.testing.assert_allclose(net2, math.exp(-1.0) * (base + gross) - base + 0.3, atol=1e-12)
    # an expensive informative probe can rank below doing nothing
    pricey = voi_linear_gaussian(post, phi_probe, phi_dec, noise, VOICosts(cost=torch.tensor([5.0, 0.0, 0.0])))
    assert int(pricey["net"].argmax()) != 0
    # masked decision candidates are excluded
    masked = voi_linear_gaussian(post, phi_probe, phi_dec, noise, dec_mask=torch.tensor([True, False, True]))
    assert float(masked["gross"][0]) < 1e-12                             # probe 0 only informs the excluded candidate


def _realised_gain(post, phi_p, phi_d, noise, gen, n=20000):
    """Average, over worlds drawn from the prior, of (true value of the arm chosen after the probe) - (true value
    of the arm chosen before): the realised reduction of regret at the next decision."""
    w = post.sample_weights(n, gen)                                      # [n, F] true worlds
    q_true = w @ phi_d.t()
    a0 = int((phi_d @ post.mean).argmax())
    m, v = post.predict(phi_p)
    y = w @ phi_p + math.sqrt(noise) * torch.randn(n, generator=gen, dtype=F64)
    gain_vec = post.cov @ phi_p / (float(v) + noise)
    mean_after = post.mean[None] + (y - m)[:, None] * gain_vec[None]     # exact posterior means, vectorised
    a1 = (mean_after @ phi_d.t()).argmax(-1)
    gain = q_true[torch.arange(n), a1] - q_true[:, a0]
    return float(gain.mean()), float(gain.std() / math.sqrt(n))


def test_voi_ranks_probes_by_realised_downstream_regret_reduction(gen, measure):
    rho_voi, rho_surprise, rho_ig = [], [], []
    for world in range(6):
        f, n_probe, n_arm = 6, 12, 8
        a_mat = torch.randn(f, f, generator=gen, dtype=F64)
        post = LinearGaussianPosterior(0.3 * torch.randn(f, generator=gen, dtype=F64), a_mat @ a_mat.t() / f)
        phi_d = torch.randn(n_arm, f, generator=gen, dtype=F64)
        phi_p = torch.randn(n_probe, f, generator=gen, dtype=F64) * torch.logspace(-1, 0.5, n_probe, dtype=F64)[:, None]
        noise = torch.logspace(-1.5, 2.0, n_probe, dtype=F64)[torch.randperm(n_probe, generator=gen)]
        if world == 0:                                                   # vectorised simulation == real update
            m, v = post.predict(phi_p[0])
            upd = post.update(phi_p[0], 1.3, float(noise[0]))
            np.testing.assert_allclose(upd.mean, post.mean + post.cov @ phi_p[0] / (v + noise[0]) * (1.3 - m), rtol=1e-9)
        out = voi_linear_gaussian(post, phi_p, phi_d, noise)
        sims = [_realised_gain(post, phi_p[j], phi_d, float(noise[j]), gen) for j in range(n_probe)]
        realised, se = torch.tensor([m for m, _ in sims], dtype=F64), torch.tensor([e for _, e in sims], dtype=F64)
        rho_voi.append(spearman(out["gross"], realised))
        rho_surprise.append(spearman(out["expected_surprise"], realised))
        rho_ig.append(spearman(out["info_gain_nats"], realised))
        # VOI *is* the expected regret reduction: agreement within Monte-Carlo error of the simulation
        assert bool(((out["gross"] - realised).abs() < 4.5 * se + 1e-3).all())
    measure("voi_spearman_vs_realised_regret_reduction_mean", float(np.mean(rho_voi)))
    measure("surprise_spearman_vs_realised_regret_reduction_mean", float(np.mean(rho_surprise)))
    measure("info_gain_spearman_vs_realised_regret_reduction_mean", float(np.mean(rho_ig)))
    assert min(rho_voi) > 0.85 and np.mean(rho_voi) > 0.93
    assert np.mean(rho_surprise) < np.mean(rho_voi) - 0.4                # raw surprise does not rank information value


def test_glm_posterior_update_is_consistent(gen):
    f = 3
    w = torch.tensor([0.8, -0.5, 0.3], dtype=F64)
    for family in ("bernoulli", "poisson"):
        post = GLMPosterior(torch.zeros(f, dtype=F64), torch.eye(f, dtype=F64) * 4.0, family)
        v0 = float(post.cov.trace())
        for _ in range(1500):
            phi = torch.randn(f, generator=gen, dtype=F64)
            eta = float(phi @ w)
            y = float(torch.bernoulli(torch.sigmoid(torch.tensor(eta)), generator=gen)) if family == "bernoulli" \
                else float(torch.poisson(torch.tensor(2.0 * math.exp(eta)), generator=gen))
            post = post.update(phi, y, exposure=2.0)
        assert float((post.mean - w).abs().max()) < (0.25 if family == "bernoulli" else 0.12), family
        assert float(post.cov.trace()) < 0.05 * v0
        assert float(torch.linalg.eigvalsh(post.cov).min()) > 0
    # one binary observation: Laplace update close to the exact one-dimensional posterior (numerical integration)
    post = GLMPosterior(torch.tensor([0.3], dtype=F64), torch.tensor([[1.5]], dtype=F64), "bernoulli")
    up = post.update(torch.tensor([1.0], dtype=F64), 1.0)
    grid = np.linspace(-12, 12, 200001)
    dens = st.norm.pdf(grid, 0.3, math.sqrt(1.5)) / (1 + np.exp(-grid))
    dens /= dens.sum()
    mean_exact = float((grid * dens).sum())
    var_exact = float(((grid - mean_exact) ** 2 * dens).sum())
    assert abs(float(up.mean) - mean_exact) < 0.12 and abs(float(up.cov) - var_exact) < 0.2
    assert float(up.mean) > 0.3 and float(up.cov) < 1.5
    with pytest.raises(ValueError):
        GLMPosterior(torch.zeros(1), torch.eye(1), "gamma")


@pytest.mark.parametrize("family", ["bernoulli", "poisson"])
def test_glm_voi_is_decision_relevant(family, gen, measure):
    f = 5
    mean = torch.tensor([0.0, 0.1, -0.1, 0.0, 0.0], dtype=F64)
    cov = torch.diag(torch.tensor([1.0, 1.0, 1.0, 0.2, 9.0], dtype=F64))
    post = GLMPosterior(mean, cov, family)
    phi_dec = torch.eye(f, dtype=F64)[:3]
    phi_probe = torch.stack([torch.eye(f, dtype=F64)[0], torch.eye(f, dtype=F64)[1] * 0.3,
                             torch.eye(f, dtype=F64)[4]])                 # informative, weakly informative, irrelevant
    out = voi_glm(post, phi_probe, phi_dec, n_samples=600, generator=gen)
    g = out["gross"]
    assert float(g[0]) > 5 * float(out["stderr"][0]) > 0                  # clearly positive
    assert float(g[0]) > float(g[1]) > -1e-9
    assert abs(float(g[2])) < 1e-12                                       # cannot change any decision value
    # the irrelevant probe is the most unpredictable one (largest predictive variance of its linear predictor)
    assert float(post.eta(phi_probe[2])[1]) > 5 * float(post.eta(phi_probe[0])[1])
    # realised: decide after one real observation, in worlds drawn from the prior
    n = 3000
    chol = torch.linalg.cholesky(cov)
    w = mean + torch.randn(n, f, generator=gen, dtype=F64) @ chol.t()
    resp = (lambda e: torch.sigmoid(e)) if family == "bernoulli" else (lambda e: torch.exp(e))
    q_true = resp(w @ phi_dec.t())
    a0 = int(post.predict_mean(phi_dec).argmax())
    realised = []
    for j in range(3):
        eta = w @ phi_probe[j]
        y = torch.bernoulli(torch.sigmoid(eta), generator=gen) if family == "bernoulli" \
            else torch.poisson(torch.exp(eta), generator=gen)
        best = {float(v): int(post.update(phi_probe[j], float(v)).predict_mean(phi_dec).argmax()) for v in y.unique()}
        a1 = torch.tensor([best[float(v)] for v in y])
        realised.append(float((q_true[torch.arange(n), a1] - q_true[:, a0]).mean()))
    measure(f"glm_{family}_voi", [float(x) for x in g])
    measure(f"glm_{family}_realised_gain", realised)
    assert realised[0] > realised[1] > -0.02 and abs(realised[2]) < 1e-12
    assert realised[0] > 0.02


def test_spearman_matches_scipy_and_handles_ties():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=50), rng.normal(size=50)
    np.testing.assert_allclose(spearman(x, y), st.spearmanr(x, y).statistic, rtol=1e-12)
    xt = np.round(x, 0)
    np.testing.assert_allclose(spearman(xt, y), st.spearmanr(xt, y).statistic, rtol=1e-12)
    assert math.isnan(spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


def test_downstream_estimator_learns_and_must_beat_the_computed_reference(gen, measure):
    """Realised downstream usefulness of an action = a nonlinear function of its latent + noise. The learned
    estimator is scored on held-out actions against (a) an accurate computed VOI and (b) a mis-specified one."""
    d = 12
    w1, w2 = torch.randn(d, generator=gen) / math.sqrt(d), torch.randn(d, generator=gen) / math.sqrt(d)

    def truth(z):
        return torch.relu(z @ w1) * torch.sigmoid(2 * (z @ w2))

    def make(n):
        z = torch.randn(n, d, generator=gen)
        clean = truth(z)
        accurate_voi = clean + 0.02 * torch.randn(n, generator=gen)       # a computed VOI from a well-specified model
        wrong_voi = torch.relu(-(z @ w1)) + 0.3 * torch.randn(n, generator=gen)   # ... and from a mis-specified one
        realised = clean + 0.15 * torch.randn(n, generator=gen)
        return z, torch.stack([wrong_voi, (z @ w2)], -1), realised, accurate_voi, wrong_voi

    z, sig, y, _, _ = make(3000)
    est = DownstreamValueEstimator(d, n_signals=2, hidden=64)
    opt = torch.optim.Adam(est.parameters(), lr=3e-3)
    measured = torch.rand(3000, generator=gen) < 0.8                      # usefulness of 20 % was never measured
    y_seen = torch.where(measured, y, torch.full_like(y, float("nan")))
    first = None
    for _ in range(500):
        idx = torch.randint(0, 3000, (128,), generator=gen)
        loss = downstream_loss(est(z[idx], sig[idx]), y_seen[idx], measured[idx])
        first = float(loss.detach()) if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss.detach()) < first - 0.5
    zt, st_, yt, good_ref, bad_ref = make(600)
    with torch.no_grad():
        pred = est(zt, st_)
    vs_good = score_downstream(pred.mean, yt, reference=good_ref, n_boot=200)
    vs_bad = score_downstream(pred.mean, yt, reference=bad_ref, n_boot=200)
    measure("downstream_estimator_spearman_heldout", vs_good["spearman"])
    measure("downstream_reference_accurate_spearman", vs_good["reference_spearman"])
    measure("downstream_reference_misspecified_spearman", vs_bad["reference_spearman"])
    assert vs_good["spearman"] > 0.6 and vs_good["spearman_lo"] > 0.5
    assert vs_good["mse"] < float(yt.var())
    # against an accurate computed VOI the learned estimator has NOT proved itself, and the scorer says so
    assert vs_good["learned_beats_reference"] is False and vs_good["delta_hi"] < 0.1
    # against a mis-specified one it does
    assert vs_bad["learned_beats_reference"] is True and vs_bad["delta_lo"] > 0.2
    # its own uncertainty is explicit and sensible
    assert 0.08 < float(pred.dist().scale.mean()) < 0.4
    with pytest.raises(ValueError):
        est(zt)
    # an all-unmeasured batch trains nothing
    est.zero_grad()
    l0 = downstream_loss(est(zt[:8], st_[:8]), torch.full((8,), float("nan")), torch.zeros(8, dtype=torch.bool))
    l0.backward()
    assert float(l0.detach()) == 0.0 and all(float(p.grad.abs().max()) == 0.0 for p in est.parameters())
