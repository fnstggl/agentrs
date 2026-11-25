"""WorldModel structure: prior/posterior/transition contract, action and time conditioning (A07), no future
leakage into the filter or forecasts, EMA target isolation, episode resets, KL balancing and anti-collapse."""
import copy

import pytest
import torch

from superlearner.cam_unified.interfaces import SequenceBatch, TokenKind, TokenSet
from superlearner.cam_unified.testing import DoubleCore, rand_outcomes, rand_sequence, rand_tokens, tiny_config
from superlearner.cam_unified.world.dynamics import WorldModel, categorical_kl, straight_through, unimix


def _wm(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = tiny_config()
    return cfg, WorldModel(cfg, DoubleCore(cfg), train_horizons=(1, 2, 4), **kw)


def test_prior_contract_shapes_and_straight_through_latent():
    cfg, wm = _wm()
    B = 5
    Z = torch.randn(B, cfg.n_state, cfg.d_model)
    act = rand_tokens(B, 2, cfg.d_model, TokenKind.ACTION, ragged=False)
    bel = rand_tokens(B, cfg.n_particles + 1, cfg.d_model, TokenKind.BELIEF, ragged=False)
    Zh, info = wm.prior(Z, act, bel, torch.ones(B))
    assert Zh.shape == Z.shape
    assert info["logits"].shape == (B, cfg.stoch_groups, cfg.stoch_classes)
    assert info["z"].shape == (B, cfg.stoch_groups * cfg.stoch_classes)
    hard = info["z"].detach().view(B, cfg.stoch_groups, cfg.stoch_classes)
    assert torch.allclose(hard.sum(-1), torch.ones(B, cfg.stoch_groups))          # one-hot per group (forward)
    assert bool(((hard - hard.round()).abs() < 1e-6).all())
    # unimix keeps every class possible
    assert float(info["logits"].exp().min()) >= 0.01 / cfg.stoch_classes - 1e-7
    # mode is deterministic; samples vary
    Zm1, _ = wm.prior(Z, act, bel, 1.0, sample=False)
    Zm2, _ = wm.prior(Z, act, bel, 1.0, sample=False)
    torch.testing.assert_close(Zm1, Zm2)
    # straight-through gradient reaches the prior logits
    lp = torch.randn(3, 4, 6, requires_grad=True)
    z = straight_through(torch.log_softmax(lp, -1))
    (z * torch.randn_like(z)).sum().backward()
    assert lp.grad is not None and lp.grad.abs().sum() > 0
    # dt=None -> predicted duration (days, positive), used and returned
    _, info2 = wm.prior(Z, act, None, None)
    assert info2["dt"].shape == (B,) and bool((info2["dt"] > 0).all())


def test_different_actions_give_different_predictions_from_the_same_state():
    """A07 (structural): same Z and same stochastic latent z, different contemplated actions -> different Z^ and
    outcome distributions; identical inputs -> identical predictions."""
    cfg, wm = _wm(1)
    B = 4
    Z = torch.randn(B, cfg.n_state, cfg.d_model)
    a1 = rand_tokens(B, 2, cfg.d_model, TokenKind.ACTION, ragged=False, seed=1)
    a2 = rand_tokens(B, 2, cfg.d_model, TokenKind.ACTION, ragged=False, seed=2)
    _, info = wm.prior(Z, a1, None, 1.0)
    z = info["z"]
    Z1, _ = wm.prior(Z, a1, None, 1.0, z=z)
    Z1b, _ = wm.prior(Z, a1, None, 1.0, z=z)
    Z2, _ = wm.prior(Z, a2, None, 1.0, z=z)
    torch.testing.assert_close(Z1, Z1b)
    assert (Z1 - Z2).abs().max() > 1e-3
    ob = rand_outcomes(B, 6, cfg.d_model, seed=3)
    with torch.no_grad():
        wm.outcome_head.param.weight.normal_(0, 0.3)
    m1, m2 = wm.outcome_dist(Z1, ob).mean(), wm.outcome_dist(Z2, ob).mean()
    assert (m1 - m2).abs().max() > 1e-3
    # and the prior over z itself is action-conditioned
    l1 = wm.prior_logits(Z, a1, None, 1.0)
    l2 = wm.prior_logits(Z, a2, None, 1.0)
    assert (l1 - l2).abs().max() > 1e-4


def test_elapsed_time_conditions_the_transition():
    cfg, wm = _wm(2)
    B = 3
    Z = torch.randn(B, cfg.n_state, cfg.d_model)
    a = rand_tokens(B, 1, cfg.d_model, TokenKind.ACTION, ragged=False)
    _, info = wm.prior(Z, a, None, 1.0)
    Za, _ = wm.prior(Z, a, None, torch.full((B,), 0.01), z=info["z"])
    Zb, _ = wm.prior(Z, a, None, torch.full((B,), 30.0), z=info["z"])
    assert (Za - Zb).abs().max() > 1e-3
    tok = wm.time_tokens(torch.tensor([0.5, 2.0, 7.0]))
    assert tok.kind.eq(TokenKind.TIME).all() and torch.allclose(tok.time[:, 0], torch.tensor([0.5, 2.0, 7.0]))


def test_filter_and_forecast_never_see_the_future():
    cfg, wm = _wm(3)
    T, B = 6, 3
    batch = rand_sequence(B, T, cfg, seed=4)
    Zp = wm.filter(batch)
    fut = copy.deepcopy(batch)
    for t in range(3, T):                       # perturb observations and outcomes after t = 2
        fut.steps[t].obs.x = fut.steps[t].obs.x + 3.0 * torch.randn_like(fut.steps[t].obs.x)
        fut.steps[t].outcomes.y = fut.steps[t].outcomes.y + 5.0
    fut.steps[2].outcomes.y = fut.steps[2].outcomes.y - 5.0
    Zf = wm.filter(fut)
    torch.testing.assert_close(Zp[:3], Zf[:3])
    assert (Zp[3:] - Zf[3:]).abs().max() > 1e-3
    g = lambda: torch.Generator().manual_seed(0)
    f1 = wm.forecast(batch, 2, (1, 2, 3), n_samples=4, generator=g())
    f2 = wm.forecast(fut, 2, (1, 2, 3), n_samples=4, generator=g())
    for h in (1, 2, 3):
        torch.testing.assert_close(f1[h]["mean"], f2[h]["mean"])
    # but the forecast does depend on the recorded future actions
    with torch.no_grad():
        wm.outcome_head.param.weight.normal_(0, 0.3)
    f1 = wm.forecast(batch, 2, (2,), n_samples=4, generator=g())
    act = copy.deepcopy(batch)
    act.steps[3].act.x = act.steps[3].act.x + 2.0 * torch.randn_like(act.steps[3].act.x)
    f3 = wm.forecast(act, 2, (2,), n_samples=4, generator=g())
    assert (f3[2]["mean"] - f1[2]["mean"]).abs().max() > 1e-4


def test_ema_target_is_isolated_and_tracks_the_online_core():
    cfg, wm = _wm(4)
    batch = rand_sequence(3, 5, cfg, seed=5)
    out = wm.loss(batch)
    out["world"].backward()
    assert all(not p.requires_grad and p.grad is None for p in wm.target_core.parameters())
    assert not any(p is q for p in wm.trainable_parameters() for q in wm.target_core.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in wm.core.parameters())
    # the online filter never reads the target: corrupting the target leaves the filter unchanged
    Z1 = wm.filter(batch).detach()
    with torch.no_grad():
        for p in wm.target_core.parameters():
            p.add_(torch.randn_like(p))
    torch.testing.assert_close(wm.filter(batch).detach(), Z1)
    # EMA update moves the target toward the online core
    dist = lambda: sum(float((a - b).detach().pow(2).sum()) for a, b in zip(wm.target_core.parameters(),
                                                                     wm.core.parameters()))
    d0 = dist()
    wm.update_target(0.5)
    assert dist() < 0.3 * d0
    wm.reset_target()
    assert dist() == 0.0


def test_loss_terms_finite_backprop_and_horizon_errors():
    cfg, wm = _wm(5)
    T, B = 6, 4
    batch = rand_sequence(B, T, cfg, seed=6)
    beliefs = [rand_tokens(B, cfg.n_particles + 1, cfg.d_model, TokenKind.BELIEF, ragged=False, seed=t)
               for t in range(T)]
    out = wm.loss(batch, beliefs=beliefs)
    for k in ("world", "kl", "outcome", "obs", "reward", "duration", "latent", "cont", "open_loop", "reg_var",
              "reg_cov"):
        assert torch.is_tensor(out[k]) and out[k].dim() == 0 and torch.isfinite(out[k]), k
    assert set(out["horizon_errors"]) == {1, 2, 4}
    for h, e in out["horizon_errors"].items():
        for k in ("outcome_nll", "outcome_mse", "reward_mae", "latent_err", "obs_logp"):
            assert k in e and e[k] == e[k], (h, k)
    assert out["Z_post"].shape == (T, B, cfg.n_state, cfg.d_model)
    out["world"].backward()
    grads = {n: p.grad for n, p in wm.named_parameters() if not n.startswith("target_core")}
    for name in ("prior_net", "post_net", "outcome_head", "rew_head", "dur_head", "obs_pred", "noise_proj",
                 "time_enc", "core"):
        assert any(g is not None and g.abs().sum() > 0 for n, g in grads.items() if n.startswith(name)), name
    # horizons longer than the sequence are skipped
    out2 = wm.loss(rand_sequence(B, 3, cfg, seed=7))
    assert set(out2["horizon_errors"]) == {1, 2}


def test_episode_boundary_resets_the_filter_state_and_masks_transitions():
    cfg, wm = _wm(6)
    T, B = 4, 2
    batch = rand_sequence(B, T, cfg, seed=8)
    batch.steps[1].done = torch.tensor([True, False])
    Z = wm.filter(batch)
    # row 0 restarts at t=2 from the initial state with O_2 only (previous action/time masked)
    Z0 = wm.init_state(1)
    o2 = batch.steps[2].obs.select([0])
    Zr, _ = wm.filter_step(Z0, o2, None, None, None, batch.steps[2].goal.select([0]))
    torch.testing.assert_close(Z[2, 0:1], Zr, atol=1e-5, rtol=1e-5)
    # row 1 continues (differs from a reset)
    Zr1, _ = wm.filter_step(Z0, batch.steps[2].obs.select([1]), None, None, None, batch.steps[2].goal.select([1]))
    assert (Z[2, 1:2] - Zr1).abs().max() > 1e-3
    out = wm.loss(batch)
    assert torch.isfinite(out["world"])


def test_kl_balancing_free_nats_and_unimix():
    cfg, wm = _wm(7)
    lp = unimix(torch.randn(4, 3, 5), 0.01)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(4, 3), atol=1e-6)
    assert float(categorical_kl(lp, lp).abs().max()) < 1e-6
    q = unimix(torch.randn(4, 3, 5) * 3, 0.01)
    kl = categorical_kl(lp, q)
    ref = torch.distributions.kl_divergence(torch.distributions.Categorical(logits=lp),
                                            torch.distributions.Categorical(logits=q)).sum(-1)
    torch.testing.assert_close(kl, ref, rtol=1e-5, atol=1e-5)
    # free nats: with an untrained model (near-uniform latents) the balanced KL sits at the floor
    out = wm.loss(rand_sequence(3, 4, cfg, seed=9))
    assert abs(float(out["kl"]) - cfg.kl_free_nats) < 1e-5
    assert float(out["kl_value"]) < cfg.kl_free_nats


def test_variance_covariance_regulariser_penalises_collapse():
    cfg, wm = _wm(8)
    collapsed = torch.randn(1, cfg.n_state, cfg.d_model).expand(32, -1, -1).clone()
    diverse = torch.randn(32, cfg.n_state, cfg.d_model)
    vc, cc = wm._vicreg(collapsed)
    vd, cd = wm._vicreg(diverse)
    assert float(vc) > 0.4 and float(vd) < 0.05
    corr = torch.randn(32, 1, cfg.d_model).expand(-1, cfg.n_state, -1) * torch.linspace(0.5, 2, cfg.d_model)
    corr = corr[..., torch.randperm(cfg.d_model)] + 0.01 * torch.randn(32, cfg.n_state, cfg.d_model)
    shared = torch.randn(32, 1, 1) * torch.ones(1, cfg.n_state, cfg.d_model) * torch.randn(cfg.d_model)
    _, cs = wm._vicreg(shared + 0.01 * torch.randn(32, cfg.n_state, cfg.d_model))
    assert float(cs) > 5 * float(cd)


def test_transition_counter_counts_rows():
    cfg, wm = _wm(9)
    n0 = wm.n_transitions
    Z = torch.randn(7, cfg.n_state, cfg.d_model)
    wm.prior(Z, rand_tokens(7, 1, cfg.d_model, TokenKind.ACTION, ragged=False), None, 1.0)
    assert wm.n_transitions - n0 == 7


def test_imagination_consistent_open_loop_uses_only_sampled_outcomes():
    """With fast_states + belief update/tokens functions, open-loop rows update a private belief from outcomes
    sampled from the model -- the real outcomes after the start (here a sentinel value) are never seen."""
    from superlearner.cam_unified.interfaces import FastState
    from superlearner.cam_unified.testing import zero_fast_state
    cfg, wm = _wm(10)
    T, B = 5, 3
    batch = rand_sequence(B, T, cfg, seed=11)
    for st in batch.steps:
        st.outcomes.y = torch.full_like(st.outcomes.y, 777.0)
    fasts = [zero_fast_state(B, cfg) for _ in range(T)]
    seen, calls = [], []

    def upd(fast, Zp, act, ob):
        seen.append(ob.y[ob.y_mask].detach().clone())
        calls.append(fast.logw.shape[0])
        logw = torch.log_softmax(fast.logw + 0.1 * ob.y.sum(-1, keepdim=True).clamp(-5, 5), -1)
        return FastState(fast.particles, logw, fast.pscale, fast.bayes_mu, fast.bayes_P, fast.bayes_noise,
                         fast.bayes_n, fast.surprise, fast.steps + 1, fast.channels)

    proj = torch.randn(cfg.d_particle, cfg.d_model)

    def toks(fast):
        x = torch.cat([fast.particles @ proj, (fast.logw.exp().unsqueeze(-1) * (fast.particles @ proj)).sum(1,
                                                                                                    keepdim=True)], 1)
        return TokenSet.from_dense(x, TokenKind.BELIEF)

    out = wm.loss(batch, fast_states=fasts, belief_update_fn=upd, belief_tokens_fn=toks)
    assert torch.isfinite(out["world"]) and calls, "belief update never called"
    allv = torch.cat(seen)
    assert allv.numel() > 0 and not bool((allv == 777.0).any())
    assert set(calls) <= {B * wm.open_loop_samples * k for k in range(1, T + 1)}
    out["world"].backward()


def test_dev_scale_config_shapes():
    """The real dev scale (d 384, 32 state tokens, 32x32 categorical latent) runs prior, loss and backward."""
    from superlearner.cam_unified.config import make_config
    cfg = make_config("dev")
    torch.manual_seed(0)
    wm = WorldModel(cfg, DoubleCore(cfg), train_horizons=(1, 2), open_loop_samples=2, open_loop_starts=2)
    batch = rand_sequence(2, 3, cfg, N=6, Q=6, seed=12)
    out = wm.loss(batch)
    out["world"].backward()
    assert torch.isfinite(out["world"])
    Zh, info = wm.prior(torch.randn(2, cfg.n_state, cfg.d_model),
                        rand_tokens(2, 2, cfg.d_model, TokenKind.ACTION, ragged=False), None, None)
    assert Zh.shape == (2, 32, 384) and info["logits"].shape == (2, 32, 32) and info["z"].shape == (2, 1024)
