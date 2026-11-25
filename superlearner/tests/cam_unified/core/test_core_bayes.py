"""Bayesian last layer (A05): the Kalman recursion equals closed-form Bayesian linear regression (known noise) and the
exact normal-inverse-gamma posterior of CAM 7.0 (unknown noise); channels bound by outcome key; Joseph-form
stability; new observations change predictions; learned features/prior/noise train through the predictive."""
import math

import pytest
import torch

from superlearner.cam7 import posterior as P7
from superlearner.cam_unified.core.bayes import BayesLastLayer
from superlearner.cam_unified.core.beliefs import BeliefModule
from superlearner.cam_unified.interfaces import Likelihood, OutcomeBatch, TokenKind, TokenSet
from superlearner.cam_unified.testing import tiny_config

F64 = torch.float64


def ob(y, keys=None, lik=None, mask=None, D=8):
    B, Q = y.shape
    q = TokenSet(torch.zeros(B, Q, D), torch.ones(B, Q, dtype=torch.bool), torch.full((B, Q), int(TokenKind.QUERY)),
                 torch.zeros(B, Q), torch.full((B, Q), -1), None)
    lik = torch.full((B, Q), int(Likelihood.GAUSSIAN)) if lik is None else lik
    return OutcomeBatch(q, lik, torch.zeros(B, Q, dtype=torch.long), y, torch.zeros(B, Q, dtype=torch.bool),
                        torch.ones(B, Q, dtype=torch.bool) if mask is None else mask, torch.full((B, Q), -1),
                        keys if keys is not None else [[f"c{q}" for q in range(Q)] for _ in range(B)])


def _layer(mode, F=5, prior_var=0.7, noise=0.3):
    cfg = tiny_config(bayes_feat=F, bayes_prior_var_init=prior_var, bayes_obs_var_init=noise)
    bl = BayesLastLayer(cfg, noise_mode=mode)
    bm = BeliefModule(cfg, n_channels=4)
    return cfg, bl, bm


def _data(B, T, F, seed=0, noise=0.3):
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(B, T, F, generator=g, dtype=F64)
    w = torch.randn(B, F, generator=g, dtype=F64)
    y = (phi * w[:, None]).sum(-1) + math.sqrt(noise) * torch.randn(B, T, generator=g, dtype=F64)
    return phi, y


def _run(bl, fs, phi, y, head_mean=None):
    for t in range(phi.shape[1]):
        hm = None if head_mean is None else head_mean[:, t:t + 1]
        fs = bl.update(fs, phi[:, t:t + 1], ob(y[:, t:t + 1]), head_mean=hm)
    return fs


def test_kalman_recursion_equals_closed_form_known_noise():
    cfg, bl, bm = _layer("fixed")
    B, T, F, s2, pv = 3, 25, 5, 0.3, 0.7
    phi, y = _data(B, T, F)
    head_mean = torch.randn(B, T, dtype=F64)
    fs = _run(bl, bm.init_fast(B), phi, y + head_mean, head_mean)       # residual model: target y - head_mean
    for b in range(B):
        S0inv = torch.eye(F, dtype=F64) / pv
        Sn = torch.linalg.inv(S0inv + phi[b].T @ phi[b] / s2)
        mun = Sn @ (phi[b].T @ y[b] / s2)
        assert torch.allclose(fs.bayes_mu[b, 0], mun, atol=1e-9)
        assert torch.allclose(fs.bayes_P[b, 0], Sn, atol=1e-9)
        assert float(fs.bayes_n[b, 0]) == T
    assert fs.channels[0] == {"c0": 0}
    # predictive of a new point = closed-form Bayesian linear regression predictive
    f_new = torch.randn(B, 1, F, dtype=F64)
    mean, var = bl.predict(fs, f_new, ob(torch.zeros(B, 1)))
    for b in range(B):
        Sn = fs.bayes_P[b, 0]
        assert torch.allclose(mean[b, 0], f_new[b, 0] @ fs.bayes_mu[b, 0], atol=1e-9)
        assert torch.allclose(var[b, 0], s2 + f_new[b, 0] @ Sn @ f_new[b, 0], atol=1e-9)


def test_several_queries_per_step_are_applied_sequentially_and_exactly():
    cfg, bl, bm = _layer("fixed")
    B, T, F = 2, 12, 5
    phi, y = _data(B, T, F, seed=1)
    fs = bl.update(bm.init_fast(B), phi, ob(y, keys=[["same"] * T for _ in range(B)]))   # one step, T queries
    for b in range(B):
        Sn = torch.linalg.inv(torch.eye(F, dtype=F64) / 0.7 + phi[b].T @ phi[b] / 0.3)
        assert torch.allclose(fs.bayes_mu[b, 0], Sn @ phi[b].T @ y[b] / 0.3, atol=1e-9)
    assert fs.channels[0] == {"same": 0}


def test_nig_recursion_equals_cam7_closed_form_posterior():
    cfg, bl, bm = _layer("nig", noise=0.4, prior_var=0.8)
    B, T, F = 3, 30, 5
    phi, y = _data(B, T, F, seed=2, noise=0.9)
    fs = _run(bl, bm.init_fast(B), phi, y)
    s20, pv, a0 = 0.4, 0.8, bl.a0
    lam0 = torch.full((F,), s20 / pv, dtype=F64)      # prior w | s2 ~ N(0, s2 V0), V0 = pv / s20
    post = P7.fit(phi, y.unsqueeze(-1), torch.ones(B, T, dtype=F64), torch.zeros(1, F, dtype=F64), lam0.unsqueeze(0),
                  torch.tensor([a0], dtype=F64), torch.tensor([a0 * s20], dtype=F64))
    assert torch.allclose(fs.bayes_mu[:, 0], post.mu[:, 0], atol=1e-8)
    assert torch.allclose(fs.bayes_noise[:, 0], post.s2[:, 0], atol=1e-8)
    prec = post.chol[:, 0] @ post.chol[:, 0].transpose(-1, -2)
    assert torch.allclose(fs.bayes_P[:, 0], torch.linalg.inv(prec) * post.s2[:, 0, None, None], atol=1e-8)
    # predictive equals the closed-form Student-t
    f_new = torch.randn(B, 1, F, dtype=F64)
    p = bl.predictive(fs, f_new, ob(torch.zeros(B, 1)))
    loc, scale2, dof = P7.predict(post, f_new)
    assert torch.allclose(p["loc"], loc[..., 0], atol=1e-8) and torch.allclose(p["scale2"], scale2[..., 0], atol=1e-8)
    assert torch.allclose(p["dof"], dof, atol=1e-8)
    y_new = torch.randn(B, 1, dtype=F64)
    lp = bl.log_prob(fs, f_new, ob(y_new))
    assert torch.allclose(lp, -P7.t_nll(y_new.unsqueeze(-1), loc, scale2, dof)[..., 0], atol=1e-8)


def test_channels_by_key_independent_and_only_continuous_observed_outcomes():
    cfg, bl, bm = _layer("fixed")
    B, F = 2, 5
    g = torch.Generator().manual_seed(3)
    fs = bm.init_fast(B)
    keys = [["revenue@prod_1", "build_ok@repo_1", "latency@svc_1"]] * B
    lik = torch.tensor([[Likelihood.GAUSSIAN, Likelihood.BERNOULLI, Likelihood.STUDENT_T]] * B)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    phis = torch.randn(B, 8, 3, F, generator=g, dtype=F64)
    ys = torch.randn(B, 8, 3, generator=g, dtype=F64)
    for t in range(8):
        fs = bl.update(fs, phis[:, t], ob(ys[:, t], keys=keys, lik=lik, mask=mask))
    assert fs.channels[0] == {"revenue@prod_1": 0, "latency@svc_1": 1}, "Bernoulli outcomes get no Kalman channel"
    for c, q in ((0, 0), (1, 2)):
        Sn = torch.linalg.inv(torch.eye(F, dtype=F64) / 0.7 + phis[0, :, q].T @ phis[0, :, q] / 0.3)
        assert torch.allclose(fs.bayes_mu[0, c], Sn @ phis[0, :, q].T @ ys[0, :, q] / 0.3, atol=1e-9)
    assert float(fs.bayes_n[1, 1]) == 0, "unobserved queries do not update"
    m, v = bl.predict(fs, phis[:, 0], ob(ys[:, 0], keys=keys, lik=lik))
    assert m[0, 1] == 0 and v[0, 1] == 0, "unbound queries defer to the head"
    hm, hv = torch.ones(B, 3, dtype=F64), torch.full((B, 3), 2.0, dtype=F64)
    bound = bl.predictive(fs, phis[:, 0], ob(ys[:, 0], keys=keys, lik=lik))["bound"]
    cm, cv = BayesLastLayer.combine(hm, hv, m, v, bound)
    assert torch.allclose(cm[:, 1], hm[:, 1]) and torch.allclose(cv[0, 0], v[0, 0]) and cm[0, 0] == 1 + m[0, 0]
    # capacity: more keys than slots -> counted overflow, never silently reusing a slot
    many = [[f"k{i}" for i in range(6)]] * B
    fs2 = bl.update(bm.init_fast(B), torch.randn(B, 6, F, dtype=F64), ob(torch.randn(B, 6, dtype=F64), keys=many))
    assert len(fs2.channels[0]) == 4 and bl.stats["channel_overflow"] == 2 * B


def test_joseph_form_stays_symmetric_psd_under_ill_conditioning():
    cfg, bl, bm = _layer("nig", F=6, noise=1e-6, prior_var=1e4)
    B, T, F = 2, 3000, 6
    g = torch.Generator().manual_seed(4)
    base = torch.randn(B, T, 1, generator=g, dtype=F64)
    phi = base.expand(B, T, F) + 1e-4 * torch.randn(B, T, F, generator=g, dtype=F64)   # nearly collinear features
    y = base[..., 0] * 3 + 1e-3 * torch.randn(B, T, generator=g, dtype=F64)
    fs = bm.init_fast(B)
    for t in range(0, T, 10):
        fs = bl.update(fs, phi[:, t:t + 10], ob(y[:, t:t + 10], keys=[["k"] * 10] * B))
    Pm = fs.bayes_P[:, 0]
    assert torch.isfinite(Pm).all() and torch.allclose(Pm, Pm.transpose(-1, -2))
    assert torch.linalg.eigvalsh(Pm).min() > -1e-12 * torch.linalg.eigvalsh(Pm).max()
    pred = (phi[:, -1] * fs.bayes_mu[:, 0]).sum(-1)
    assert torch.allclose(pred, y[:, -1], atol=0.05)


def test_new_observations_alter_predictions_fast_adaptation():
    """A05: a consistent offset between outcomes and the head's mean is absorbed within a few observations; the
    predicted correction moves toward it and its uncertainty shrinks."""
    cfg = tiny_config(bayes_feat=8)
    bl = BayesLastLayer(cfg)
    bm = BeliefModule(cfg)
    B, Q = 4, 1
    feat = torch.randn(B, Q, cfg.d_model)
    with torch.no_grad():
        phi = bl.features(feat, None)
    fs = bm.init_fast(B)
    head_mean = torch.zeros(B, Q)
    offset = torch.tensor([[2.0], [-1.0], [0.5], [3.0]])
    m0, v0 = bl.predict(bl.update(fs, phi, ob(offset, mask=torch.zeros(B, Q, dtype=torch.bool))), phi, ob(offset))
    assert torch.all(m0 == 0) and torch.all(v0 == 0), "nothing bound before any observation"
    fs1 = bl.update(fs, phi, ob(offset), head_mean=head_mean)
    m1, v1 = bl.predict(fs1, phi, ob(offset))
    for _ in range(9):
        fs1 = bl.update(fs1, phi, ob(offset), head_mean=head_mean)
    m10, v10 = bl.predict(fs1, phi, ob(offset))
    err1, err10 = (m1 - offset).abs(), (m10 - offset).abs()
    assert (err10 < 0.5 * err1).all() and (err10 < 0.15 * offset.abs() + 0.05).all(), (err1, err10)
    assert (v10 < v1).all()


def test_learned_features_prior_and_noise_train_through_the_predictive():
    """Meta-learning the feature map: residuals r = a_task sin(x) + b_task cos(x) with x hidden inside a random
    projection. After training on support/query NLL the query log-likelihood after 6 support points improves."""
    torch.manual_seed(0)
    cfg = tiny_config(bayes_feat=8, bayes_prior_var_init=1.0, bayes_obs_var_init=1.0)
    bl = BayesLastLayer(cfg)
    bm = BeliefModule(cfg, n_channels=1)
    proj = torch.randn(1, cfg.d_model) / 1.0

    def task(B, g):
        ab = torch.randn(B, 2, generator=g) * 2
        x = torch.rand(B, 12, generator=g) * 6 - 3
        r = ab[:, :1] * torch.sin(2 * x) + ab[:, 1:] * torch.cos(2 * x) + 0.1 * torch.randn(B, 12, generator=g)
        feat = torch.tanh(x.unsqueeze(-1) @ proj)                                 # [B,12,D]
        return feat, r

    def query_ll(B, g):
        feat, r = task(B, g)
        phi = bl.features(feat, None)
        fs = bl.update(bm.init_fast(B), phi[:, :6], ob(r[:, :6], keys=[["c"] * 6] * B))
        return bl.log_prob(fs, phi[:, 6:], ob(r[:, 6:], keys=[["c"] * 6] * B)).sum(-1).mean()

    def evaluate():
        with torch.no_grad():
            return float(sum(query_ll(64, torch.Generator().manual_seed(500 + i)) for i in range(4)) / 4)
    before = evaluate()
    opt = torch.optim.Adam(bl.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(300):
        loss = -query_ll(32, g)
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = evaluate()
    print(f"query log-lik (6 points): before {before:.2f} after {after:.2f}; noise {float(bl.noise().detach()):.3f}")
    assert after > before + 3.0, (before, after)
    assert float(bl.noise().detach()) < 0.5, "the learned noise prior adapts to the task family's residual noise"
