"""The within-episode regression layer: closed-form correctness, causality, and meta-training."""
import math

import pytest
import torch

from superlearner.unified.core.adaptive import EpisodeRegression

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_posterior_mean_matches_ridge_regression():
    torch.manual_seed(0)
    m = EpisodeRegression(n_levers=3, k=2).to(DEV)
    m.decay_logit.data.fill_(50.0)                    # no forgetting
    B, H = 2, 40
    phi = torch.randn(B, H, m.dim, device=DEV)
    y = torch.randn(B, H, device=DEV)
    A, b = m.empty(B, DEV)
    for t in range(H):
        A, b = m.add(A, b, phi[:, t], y[:, t])
    mean, _ = m.posterior(A, b)
    noise = m._noise()
    prior = torch.full((m.dim,), m.log_prec.exp().item(), device=DEV)
    prior[-1] = m.log_prec_bias.exp()
    for i in range(B):
        P = torch.diag(prior) + phi[i].T @ phi[i] / noise
        ref = torch.linalg.solve(P, phi[i].T @ y[i] / noise)
        assert torch.allclose(mean[i], ref, atol=1e-3)


def test_posterior_stays_positive_definite_with_collinear_features():
    m = EpisodeRegression(n_levers=3, k=4, fixed_features=True).to(DEV)
    m.log_prec.data.fill_(-20.0)           # the clamp must keep this usable
    m.log_noise.data.fill_(-20.0)
    phi = torch.randn(2, 1, m.dim, device=DEV).expand(2, 500, m.dim).contiguous()   # one direction repeated
    A, b = m.empty(2, DEV)
    for t in range(0, 500, 50):
        A, b = m.add(A, b, phi[:, t] * 1e3, torch.randn(2, device=DEV))
    mu, var = m.predict(A, b, phi[:, :3])
    assert torch.isfinite(mu).all() and (var > 0).all()


def test_posterior_mean_is_invariant_to_reward_scale():
    m = EpisodeRegression(n_levers=2, k=2).to(DEV)
    phi = torch.randn(1, 30, m.dim, device=DEV)
    y = torch.randn(1, 30, device=DEV)
    A, b = m.empty(1, DEV)
    A2, b2 = m.empty(1, DEV)
    for t in range(30):
        A, b = m.add(A, b, phi[:, t], y[:, t])
        A2, b2 = m.add(A2, b2, phi[:, t], 1000 * y[:, t])
    assert torch.allclose(m.posterior(A2, b2)[0], 1000 * m.posterior(A, b)[0], rtol=1e-3, atol=1e-2)


def test_prediction_at_t_uses_only_earlier_pairs():
    torch.manual_seed(1)
    m = EpisodeRegression(n_levers=2, k=2).to(DEV)
    B, H, k = 3, 20, 12
    x = torch.rand(B, H, 2, device=DEV) * 2 - 1
    r = torch.randn(B, H, device=DEV)
    args = dict(sd=torch.ones(B, H, device=DEV), x0=torch.zeros(B, 2, device=DEV), mask=torch.ones(B, 2, device=DEV),
                valid=torch.ones(B, H, device=DEV), n_points=H, min_pairs=1, return_predictions=True)
    _, a = m.prefix_nll(x, r, **args)
    r2 = r.clone()
    r2[:, k] += 50.0            # only the outcome at k changes; the setting at k is the query and may be used
    _, b = m.prefix_nll(x, r2, **args)
    t = torch.tensor(a["ticks"])
    upto = (t <= k).nonzero().flatten()
    after = (t > k).nonzero().flatten()
    assert torch.equal(a["mu"][:, upto], b["mu"][:, upto])       # the outcome at k never informs predictions up to k
    assert not torch.allclose(a["mu"][:, after], b["mu"][:, after])


@pytest.mark.slow
def test_meta_training_learns_features_that_beat_the_running_mean():
    """Worlds: reward = a * x0 - c * x0^2 with a, c drawn per episode. The regression must learn a useful feature map."""
    torch.manual_seed(2)
    m = EpisodeRegression(n_levers=3, k=3).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)

    def batch(b=32, h=60):
        x = torch.rand(b, h, 3, device=DEV) * 2 - 1
        a = torch.randn(b, 1, device=DEV)
        c = torch.rand(b, 1, device=DEV) + 0.5
        r = a * x[..., 0] - c * x[..., 0] ** 2 + 0.05 * torch.randn(b, h, device=DEV)
        return x, r

    def score():
        x, r = batch()
        _, info = m.prefix_nll(x, r, sd=r.std(1, keepdim=True).expand_as(r), x0=torch.zeros(32, 3, device=DEV),
                               mask=torch.ones(32, 3, device=DEV), valid=torch.ones_like(r))
        return float(info["adapt/r2_vs_mean"])

    before = score()
    for _ in range(400):
        x, r = batch()
        loss, _ = m.prefix_nll(x, r, sd=r.std(1, keepdim=True).expand_as(r), x0=torch.zeros(32, 3, device=DEV),
                               mask=torch.ones(32, 3, device=DEV), valid=torch.ones_like(r))
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = score()
    assert after > 0.8 and after > before + 0.1, (before, after)
