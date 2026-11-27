"""Imagination: latent rollouts with uncertainty, measured-accuracy horizon control, imagined != real."""
import math

import pytest
import torch

from cam_live.contracts import (Distribution, OutcomeObservation, OutcomeQuery, OutcomeStatus, PredictionRecord,
                                Provenance)
from cam_live.predict import errors as E
from cam_live.predict.imagination import (HorizonController, ImaginedDataError, ImaginedRollout, ImaginedValues,
                                          imagine_candidates, imagined_prediction, is_imagined, multistep_error,
                                          refuse_imagined, rollout)
from cam_live.predict.world import LatentTransition, transition_nll

DM, HOUR = 12, 3600.0


def decision_latent(b, e_a, e_g):
    """Synthetic shared trunk: z depends on the belief, the action and the goal."""
    return e_a + 0.3 * b[:, None] + 0.1 * e_g[:, None]


class Dynamics:
    def __init__(self, gen, noise=0.03):
        self.a = torch.randn(DM, DM, generator=gen) / math.sqrt(DM)
        self.gen, self.noise = gen, noise

    def step(self, b, e_a, e_g, dt, noise=True):
        z = decision_latent(b, e_a[:, None], e_g).squeeze(1)
        nxt = b + (1.0 - torch.exp(-dt / HOUR))[:, None] * torch.tanh(z @ self.a)
        return nxt + (self.noise * torch.randn(b.shape, generator=self.gen) if noise else 0.0)


def _train(gen, steps=600):
    dyn = Dynamics(gen)
    n = 4000
    b = torch.randn(n, DM, generator=gen)
    e_a = torch.randn(n, DM, generator=gen)
    e_g = torch.randn(n, DM, generator=gen)
    dt = torch.tensor([600.0, HOUR, 3 * HOUR])[torch.randint(0, 3, (n,), generator=gen)]
    b_next = dyn.step(b, e_a, e_g, dt)
    tr = LatentTransition(DM, hidden=96, n_members=3)
    opt = torch.optim.Adam(tr.parameters(), lr=3e-3)
    for _ in range(steps):
        idx = torch.randint(0, n, (128,), generator=gen)
        z = decision_latent(b[idx], e_a[idx, None], e_g[idx]).squeeze(1)
        loss, _ = transition_nll(tr(z, dt[idx], b[idx]), b_next[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return dyn, tr


def _real_sequences(dyn, gen, bsz=64, t=6):
    b = torch.randn(bsz, DM, generator=gen)
    e_g = torch.randn(bsz, DM, generator=gen)
    e_a = torch.randn(bsz, t, DM, generator=gen)
    dt = torch.full((bsz, t), HOUR)
    seq = [b]
    for s in range(t):
        seq.append(dyn.step(seq[-1], e_a[:, s], e_g, dt[:, s]))
    return torch.stack(seq, 1), e_a, e_g, dt


def test_rollout_is_flagged_tracks_truth_and_spread_grows(gen):
    dyn, tr = _train(gen)
    b_seq, e_a, e_g, dt = _real_sequences(dyn, gen)
    with torch.no_grad():
        ro = rollout(tr, decision_latent, b_seq[:, 0], e_a, e_g, dt, n_particles=16, generator=gen)
    assert isinstance(ro, ImaginedRollout) and ro.imagined is True and ro.horizon_steps == 6
    assert all(s.imagined is True for s in ro.steps) and is_imagined(ro) and is_imagined(ro.steps[0])
    scale = b_seq.reshape(-1, DM).var(0).mean()
    err1 = float(((ro.steps[0].b_mean - b_seq[:, 1]) ** 2).mean() / scale)
    assert err1 < 0.05
    spread = [float(s.b_std.mean()) for s in ro.steps]
    assert spread[-1] > 1.5 * spread[0]                       # uncertainty compounds with the horizon
    assert torch.allclose(ro.steps[-1].elapsed_s, torch.full((64,), 6 * HOUR, dtype=torch.float64))
    # the horizon limit is enforced by the caller-supplied controller value
    with torch.no_grad():
        short = rollout(tr, decision_latent, b_seq[:, 0], e_a, e_g, dt, max_steps=2, generator=gen)
    assert short.horizon_steps == 2 and short.truncated_at == 2
    # only tensors and numbers come out: imagination never produces text
    for s in ro.steps:
        assert all(isinstance(v, (torch.Tensor, int, bool)) for v in vars(s).values())


def test_horizon_grows_only_while_measured_error_is_small(gen, measure):
    dyn, tr = _train(gen)
    b_seq, e_a, e_g, dt = _real_sequences(dyn, gen, bsz=128, t=6)
    errs = multistep_error(tr, decision_latent, b_seq, e_a, e_g, dt, max_k=5)
    assert bool((errs[1:] >= errs[:-1] * 0.8).all()) and float(errs[-1]) > float(errs[0])
    ctl = HorizonController(max_horizon=5, threshold=float(errs[2]) * 1.01, min_count=3)
    assert ctl.allowed() == 1                                 # starts short, with no evidence
    ctl.update(errs)
    ctl.update(errs)
    assert ctl.allowed() == 1                                 # not yet demonstrated often enough
    ctl.update(errs)
    expect = int((errs <= ctl.threshold).long().cumprod(0).sum())
    measure("imagination_open_loop_nmse_by_steps", [float(e) for e in errs])
    measure("imagination_allowed_horizon", expect)
    assert ctl.allowed() == expect and 3 <= expect <= 5
    # an untrained model is never allowed beyond the minimum
    raw = LatentTransition(DM, hidden=96, n_members=3, seed=3)
    bad = multistep_error(raw, decision_latent, b_seq, e_a, e_g, dt, max_k=5)
    ctl2 = HorizonController(max_horizon=5, threshold=float(errs[2]) * 1.01, min_count=1)
    for _ in range(4):
        ctl2.update(bad)
    assert ctl2.allowed() == 1 and float(bad[0]) > float(errs[0]) * 3
    # the horizon shrinks again when accuracy degrades, and survives a save / load
    for _ in range(30):
        ctl.update(bad)
    assert ctl.allowed() == 1
    ctl3 = HorizonController(max_horizon=5)
    ctl3.load_state(ctl.state())
    assert ctl3.state() == ctl.state()


def test_imagine_candidates_ranks_alternatives_like_the_true_model(gen):
    dyn, tr = _train(gen)
    bsz, n = 48, 5
    b = torch.randn(bsz, DM, generator=gen)
    e_g = torch.randn(bsz, DM, generator=gen)
    e_a = torch.randn(bsz, n, DM, generator=gen)
    dt = torch.full((bsz, n), HOUR)
    v = torch.randn(DM, generator=gen)

    def value_fn(bb):
        return bb @ v

    z = decision_latent(b, e_a, e_g)
    a_mask = torch.ones(bsz, n, dtype=torch.bool)
    a_mask[:, -1] = False
    with torch.no_grad():
        iv = imagine_candidates(tr, z, dt, b, value_fn, a_mask=a_mask, n_particles=16, generator=gen)
    assert isinstance(iv, ImaginedValues) and iv.imagined is True and iv.horizon_steps == 1
    true = torch.stack([value_fn(dyn.step(b, e_a[:, i], e_g, dt[:, i], noise=False)) for i in range(n)], 1)
    agree = (iv.value_mean[:, :-1].argmax(1) == true[:, :-1].argmax(1)).float().mean()
    assert float(agree) > 0.85
    assert bool(torch.isinf(iv.value_mean[:, -1]).all()) and float(iv.value_std[:, -1].abs().max()) == 0.0
    assert bool((iv.value_std[:, :-1] > 0).all()) and iv.b_next_mean.shape == (bsz, n, DM)


def test_imagined_values_are_refused_as_outcomes():
    q = OutcomeQuery("q1", "metric", horizon_s=60.0)
    pred = imagined_prediction(q, Distribution("normal", {"loc": 0.0, "scale": 1.0}), made_at=10.0, horizon_steps=2)
    assert isinstance(pred, PredictionRecord) and pred.imagined_horizon_steps == 2 and is_imagined(pred)
    with pytest.raises(ValueError):
        imagined_prediction(q, pred.dist, 10.0, 0)
    for obj in (pred, {"imagined": True}, {"imagined_horizon_steps": 1},
                type("X", (), {"provenance": Provenance(imagined=True)})()):
        with pytest.raises(ImaginedDataError):
            refuse_imagined(obj)
    real = OutcomeObservation("q1", OutcomeStatus.OBSERVED, 1.0)
    refuse_imagined(real)                                     # a real observation passes
    assert not is_imagined(real) and not is_imagined({"value": 1})


def test_change_detector_and_error_attribution():
    g = torch.Generator().manual_seed(0)
    # no false alarm on 3000 well-calibrated residuals, for several seeds
    for seed in range(5):
        det = E.ChangeDetector()
        res = torch.randn(3000, generator=torch.Generator().manual_seed(seed))
        assert not any(det.update(float(r)) for r in res)
    # a sustained 2-sigma shift is detected quickly, in either direction
    for sign in (1.0, -1.0):
        det = E.ChangeDetector()
        for r in torch.randn(300, generator=g):
            det.update(float(r))
        assert not det.changed
        delay = None
        for i, r in enumerate(torch.randn(200, generator=g)):
            if det.update(float(r) + sign * 2.0):
                delay = i + 1
                break
        assert delay is not None and delay <= 20 and det.score > 1.0
        det.reset()
        assert not det.changed
    # attribution
    quiet, fired = E.ChangeDetector(), E.ChangeDetector()
    for _ in range(40):
        fired.update(3.0)
    c = E.classify_error
    assert c(OutcomeStatus.PENDING, None, 0.0, 1.0, 0.0).label == E.MISSING
    assert c(OutcomeStatus.UNOBSERVABLE, 5.0, 0.0, 1.0, 0.0).label == E.MISSING
    assert c(OutcomeStatus.OBSERVED, 1.5, 0.0, 1.0, 0.0, quiet).label == E.NOISE
    assert c(OutcomeStatus.OBSERVED, 5.0, 0.0, 1.0, 4.0, quiet).label == E.UNCERTAIN
    assert c(OutcomeStatus.OBSERVED, 5.0, 0.0, 1.0, 0.1, quiet).label == E.WRONG
    assert c(OutcomeStatus.OBSERVED, 5.0, 0.0, 1.0, 0.1, fired).label == E.CHANGED
    assert c(OutcomeStatus.OBSERVED, 0.5, 0.0, 1.0, 0.1, fired).label == E.NOISE
    att = c(OutcomeStatus.OBSERVED, 3.0, 0.0, 1.0, 0.0)
    assert att.z_aleatoric == 3.0 and att.z_total == 3.0 and set(E.LABELS) >= {att.label}
