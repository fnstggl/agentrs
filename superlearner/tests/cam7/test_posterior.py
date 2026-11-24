"""The fast Bayesian layer: exactness, the Kalman equivalence the specification states, stability, and the value of
information."""
import math

import numpy as np
import torch

from superlearner.cam7 import posterior as P

F64 = torch.float64


def _data(B=3, T=20, D=5, K=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(B, T, D, generator=g, dtype=F64)
    w = torch.randn(B, K, D, generator=g, dtype=F64)
    y = torch.einsum("btd,bkd->btk", phi, w) + 0.1 * torch.randn(B, T, K, generator=g, dtype=F64)
    return phi, y


def test_posterior_matches_closed_form():
    phi, y = _data()
    B, T, D = phi.shape
    K = y.shape[-1]
    om = torch.rand(B, T, dtype=F64)
    mu0 = torch.randn(K, D, dtype=F64)
    lam0 = torch.rand(K, D, dtype=F64) + 0.5
    a0, b0 = torch.tensor([2.0, 3.0], dtype=F64), torch.tensor([0.5, 0.2], dtype=F64)
    post = P.fit(phi, y, om, mu0, lam0, a0, b0)
    for b in range(B):
        for k in range(K):
            Om = torch.diag(om[b])
            Ln = torch.diag(lam0[k]) + phi[b].T @ Om @ phi[b]
            mun = torch.linalg.solve(Ln, torch.diag(lam0[k]) @ mu0[k] + phi[b].T @ Om @ y[b, :, k])
            bn = b0[k] + 0.5 * (y[b, :, k] @ Om @ y[b, :, k] + mu0[k] @ torch.diag(lam0[k]) @ mu0[k] - mun @ Ln @ mun)
            assert torch.allclose(post.mu[b, k], mun, atol=1e-8)
            assert torch.allclose(post.b[b, k], bn, atol=1e-8)
            assert torch.allclose(post.a[b, k], a0[k] + 0.5 * om[b].sum())


def test_batch_update_equals_kalman_recursion_with_known_noise():
    """With the noise variance fixed, the batch posterior is the spec's recursion: innovation, gain, mean, covariance."""
    phi, y = _data(B=1, T=15, D=4, K=1, seed=1)
    s2 = 0.3
    lam0 = torch.full((1, 4), 2.0, dtype=F64)
    mu0 = torch.zeros(1, 4, dtype=F64)
    mu, Sig = mu0[0].clone(), torch.diag(1.0 / lam0[0]) * s2
    for t in range(15):
        f = phi[0, t]
        innov = y[0, t, 0] - f @ mu
        Kt = Sig @ f / (s2 + f @ Sig @ f)
        mu = mu + Kt * innov
        Sig = Sig - torch.outer(Kt, f @ Sig)
    post = P.fit(phi, y, torch.ones(1, 15, dtype=F64), mu0, lam0, torch.tensor([1.0]), torch.tensor([1.0]))
    assert torch.allclose(post.mu[0, 0], mu, atol=1e-9)
    prec = post.chol[0, 0] @ post.chol[0, 0].T
    assert torch.allclose(torch.linalg.inv(prec) * s2, Sig, atol=1e-9)


def test_predictive_is_student_t_and_calibrated():
    g = torch.Generator().manual_seed(3)
    B, T, D = 400, 30, 3
    phi = torch.randn(B, T + 1, D, generator=g, dtype=F64)
    w = torch.randn(B, 1, D, generator=g, dtype=F64) * 0.7
    y = torch.einsum("btd,bkd->btk", phi, w) + 0.5 * torch.randn(B, T + 1, 1, generator=g, dtype=F64)
    om = torch.ones(B, T + 1, dtype=F64)
    om[:, -1] = 0
    post = P.fit(phi, y, om, torch.zeros(1, D, dtype=F64), torch.full((1, D), 0.5 / 0.49, dtype=F64),
                 torch.tensor([2.0]), torch.tensor([0.5]))
    loc, sc2, dof = P.predict(post, phi[:, -1:])
    z = ((y[:, -1:] - loc) / sc2.sqrt()).abs().squeeze()
    nu = float(dof[0, 0])                                         # same for every task here
    torch.manual_seed(0)
    q90 = torch.distributions.StudentT(torch.tensor(nu, dtype=F64)).sample((400000,)).quantile(0.9)
    cover = (z < q90).double().mean()
    assert abs(float(cover) - 0.8) < 0.07, cover
    nll = P.t_nll(y[:, -1:], loc, sc2, dof)
    ref = -torch.distributions.StudentT(dof[:, 0], loc[:, 0, 0], sc2[:, 0, 0].sqrt()).log_prob(y[:, -1, 0])
    assert torch.allclose(nll[:, 0, 0], ref, atol=1e-8)


def test_ill_conditioned_features_stay_finite_and_differentiable():
    B, T, D = 4, 40, 30
    base = torch.randn(B, T, 2, dtype=F64)
    phi = torch.cat([base.repeat(1, 1, 15)], -1) + 1e-9 * torch.randn(B, T, D, dtype=F64)   # rank 2 of 30
    phi.requires_grad_(True)
    y = torch.randn(B, T, 2, dtype=F64) * 1e-3
    lam = torch.full((2, D), 1e-6, dtype=F64, requires_grad=True)
    post = P.fit(phi, y, torch.ones(B, T, dtype=F64), torch.zeros(2, D, dtype=F64), lam, torch.tensor([2.0, 2.0]),
                 torch.tensor([1e-6, 1e-6]))
    loc, sc2, dof = P.predict(post, phi[:, :5])
    loss = P.t_nll(y[:, :5], loc, sc2, dof).mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(phi.grad).all() and torch.isfinite(lam.grad).all()


def test_knowledge_gradient_values_decision_relevant_information():
    # two candidates: 0 known exactly, 1 uncertain. Uncertain one has positive KG; KG of a certain one is 0.
    mean = torch.tensor([[1.0, 0.9]], dtype=F64)
    cov = torch.tensor([[[0.0, 0.0], [0.0, 1.0]]], dtype=F64)
    kg = P.knowledge_gradient(mean, cov, torch.full((1, 2), 0.1, dtype=F64), n_points=2000)
    sd = 1.0 / math.sqrt(1.1)                    # sd of the change in candidate 1's mean
    zc = (1.0 - 0.9) / sd
    pdf, cdf = math.exp(-zc * zc / 2) / math.sqrt(2 * math.pi), 0.5 * (1 + math.erf(zc / math.sqrt(2)))
    exact = sd * (pdf - zc * (1 - cdf))                       # E[max(0, 0.9 + sd Z - 1)]
    assert abs(float(kg[0, 1]) - exact) < 2e-3 and float(kg[0, 0]) == 0.0
    # a candidate whose observation cannot change the decision is worth nothing: very uncertain but hopeless
    mean2 = torch.tensor([[1.0, -100.0]], dtype=F64)
    kg2 = P.knowledge_gradient(mean2, cov, torch.full((1, 2), 0.1, dtype=F64))
    assert float(kg2[0, 1]) < 1e-12
    # irrelevant novelty: information gain is high for it anyway
    assert float(P.info_gain(cov, torch.full((1, 2), 0.1, dtype=F64))[0, 1]) > 1.0


def test_sample_weights_moments():
    phi, y = _data(B=1, T=50, D=3, K=1, seed=4)
    post = P.fit(phi, y, torch.ones(1, 50, dtype=F64), torch.zeros(1, 3, dtype=F64), torch.ones(1, 3, dtype=F64),
                 torch.tensor([2.0]), torch.tensor([0.1]))
    torch.manual_seed(0)
    w = P.sample_weights(post, 20000)[0, :, 0]
    assert torch.allclose(w.mean(0), post.mu[0, 0], atol=0.01)
    cov_s = torch.cov(w.T)
    prec = post.chol[0, 0] @ post.chol[0, 0].T
    cov_t = torch.linalg.inv(prec) * post.b[0, 0] / (post.a[0, 0] - 1)        # Student-t covariance of w
    assert torch.allclose(cov_s, cov_t, rtol=0.1, atol=1e-5)
