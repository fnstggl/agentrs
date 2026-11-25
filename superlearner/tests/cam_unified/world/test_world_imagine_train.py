"""Imagination training targets: lambda-returns against a reference, uncertainty down-weighting, horizon
shortening at high model error, real/imagined separation and the reliable-horizon rule."""
import pytest
import torch

from superlearner.cam_unified.interfaces import Provenance
from superlearner.cam_unified.world.imagination import WorldImaginedStep, WorldImaginedTrajectory
from superlearner.cam_unified.world.imagine_train import (combine_real_and_imagined, imagination_actor_critic_loss,
                                                          imagined_targets, select_reliable_horizon)


def _traj(H=4, Bb=3, seed=0, err=None):
    g = torch.Generator().manual_seed(seed)
    steps = []
    for k in range(H):
        steps.append(WorldImaginedStep(
            Z=torch.zeros(Bb, 1, 1), action=[None] * Bb, act_tokens=None, outcome_mean=torch.zeros(Bb, 0),
            outcome_var=torch.zeros(Bb, 0), value=torch.randn(Bb, generator=g), reward=torch.randn(Bb, generator=g),
            belief_ess=torch.ones(Bb), model_error=torch.rand(Bb, generator=g) * 0.1 if err is None else err[k],
            dt=torch.rand(Bb, generator=g) * 2, cont=0.5 + 0.5 * torch.rand(Bb, generator=g)))
    return WorldImaginedTrajectory(steps=steps, returns=torch.zeros(Bb), branch_of=torch.zeros(Bb, dtype=torch.long),
                                   first_candidate=torch.zeros(Bb, dtype=torch.long))


def _reference(traj, lam, kappa, v0, L):
    H, Bb = len(traj.steps), traj.steps[0].reward.shape[0]
    V = [v0] + [s.value for s in traj.steps]
    G = torch.zeros(H, Bb)
    for b in range(Bb):
        nxt = None
        for k in reversed(range(int(L[b]))):
            s = traj.steps[k]
            disc = torch.exp(-kappa * s.dt[b]) * s.cont[b]
            if k == int(L[b]) - 1:
                boot = V[k + 1][b]
            else:
                boot = (1 - lam) * V[k + 1][b] + lam * nxt
            G[k, b] = s.reward[b] + disc * boot
            nxt = G[k, b]
    return G


def test_lambda_returns_match_reference():
    traj = _traj()
    v0 = torch.tensor([0.5, -1.0, 2.0])
    t = imagined_targets(traj, lam=0.8, kappa=0.3, value0=v0)
    ref = _reference(traj, 0.8, 0.3, v0, torch.full((3,), 4))
    torch.testing.assert_close(t.returns, ref)
    torch.testing.assert_close(t.advantages, ref - t.values[:4])
    assert t.provenance == Provenance.IMAGINED and t.is_imagined
    assert bool(t.mask.all()) and t.horizon.tolist() == [4, 4, 4]


def test_horizon_shortening_and_uncertainty_weights():
    err = [torch.tensor([0.0, 0.0, 2.0]), torch.tensor([0.0, 3.0, 0.0]), torch.tensor([0.1, 0.0, 0.0]),
           torch.tensor([0.1, 0.0, 0.0])]
    traj = _traj(err=err)
    t = imagined_targets(traj, lam=0.9, kappa=0.1, uncertainty_beta=1.0, error_threshold=1.0)
    # row 0 trusted to the end, row 1 cut after step 1 (its error 3 > 1 at k = 1), row 2 cut after step 0
    assert t.horizon.tolist() == [4, 2, 1]
    assert t.mask[:, 1].tolist() == [True, True, False, False] and t.mask[:, 2].tolist() == [True, False, False, False]
    ref = _reference(traj, 0.9, 0.1, torch.zeros(3), t.horizon)
    torch.testing.assert_close(t.returns[t.mask], ref[t.mask])
    # the last trusted step bootstraps from V there
    s1 = traj.steps[1]
    torch.testing.assert_close(t.returns[1, 1], s1.reward[1] + torch.exp(-0.1 * s1.dt[1]) * s1.cont[1] * s1.value[1])
    # weights: exp(-cumulative error), zero outside the trusted horizon, non-increasing in k
    assert bool((t.weights[~t.mask] == 0).all())
    torch.testing.assert_close(t.weights[:, 0], torch.exp(-torch.cumsum(torch.tensor([0.0, 0.0, 0.1, 0.1]), 0)))
    assert bool((t.weights[1:, 0] <= t.weights[:-1, 0]).all())
    assert float(t.weights[1, 1]) == pytest.approx(float(torch.exp(torch.tensor(-3.0))), rel=1e-5)


def test_actor_critic_loss_is_weighted_and_differentiable():
    traj = _traj(seed=2)
    t = imagined_targets(traj, lam=0.95, kappa=0.05)
    logp = torch.randn(4, 3, requires_grad=True)
    vals = torch.randn(4, 3, requires_grad=True)
    ent = torch.rand(4, 3)
    out = imagination_actor_critic_loss(t, logp, vals, ent)
    out["loss"].backward()
    assert out["provenance"] == "imagined" and torch.isfinite(out["loss"])
    assert logp.grad.abs().sum() > 0 and vals.grad.abs().sum() > 0
    # zero-weight steps contribute no gradient
    t.weights[2:] = 0
    logp2 = logp.detach().clone().requires_grad_(True)
    vals2 = vals.detach().clone().requires_grad_(True)
    imagination_actor_critic_loss(t, logp2, vals2, ent)["loss"].backward()
    assert logp2.grad[2:].abs().sum() == 0 and vals2.grad[2:].abs().sum() == 0
    # step_mask excludes enumerated (non-sampled) first candidates
    logp3 = logp.detach().clone().requires_grad_(True)
    sm = torch.ones(4, 3)
    sm[0] = 0
    imagination_actor_critic_loss(t, logp3, vals.detach(), ent, step_mask=sm)["loss"].backward()
    assert logp3.grad[0].abs().sum() == 0
    t.provenance = Provenance.REAL
    with pytest.raises(ValueError):
        imagination_actor_critic_loss(t, logp, vals)
    real_traj = _traj()
    real_traj.provenance = Provenance.REAL
    with pytest.raises(ValueError):
        imagined_targets(real_traj)


def test_real_and_imagined_are_kept_separate():
    real = {"loss": torch.tensor(2.0), "actor": torch.tensor(1.0), "provenance": "real"}
    imag = {"loss": torch.tensor(4.0), "critic": torch.tensor(3.0), "provenance": "imagined"}
    out = combine_real_and_imagined(real, imag, imagined_weight=0.25)
    assert float(out["loss"]) == pytest.approx(3.0)
    assert float(out["real/actor"]) == 1.0 and float(out["imagined/critic"]) == 3.0 and out["imagined_weight"] == 0.25
    with pytest.raises(ValueError):
        combine_real_and_imagined(None, imag)                  # imagined-only updates are refused
    with pytest.raises(ValueError):
        combine_real_and_imagined(real, {"loss": torch.tensor(1.0)})      # unlabelled imagined losses
    with pytest.raises(ValueError):
        combine_real_and_imagined(dict(real, provenance="imagined"), imag)
    with pytest.raises(ValueError):                            # an unlabelled "real" dict could be imagined
        combine_real_and_imagined({"loss": torch.tensor(2.0)}, imag)
    assert float(combine_real_and_imagined(real, None)["loss"]) == 2.0


def test_reliable_horizon_shrinks_when_model_is_not_better_than_baseline():
    model = {1: 0.1, 2: 0.2, 4: 0.5, 8: 1.5}
    base = {1: 0.6, 2: 0.9, 4: 1.1, 8: 1.0}
    assert select_reliable_horizon(model, base, (1, 2, 4, 8)) == 4
    assert select_reliable_horizon(model, base, (1, 2, 4, 8), margin=0.4) == 2
    assert select_reliable_horizon({1: 2.0}, {1: 1.0}, (1, 2)) == 1
