"""PPO-clip on fresh data, V-trace routing of stale data, gradient signs, clipping, KL tracking, value clipping."""
import pytest
import torch

from superlearner.cam_unified.learn.heads import ValueHead
from superlearner.cam_unified.learn.ppo import ActorCriticObjective, StalePolicyError, approx_kl, ppo_clip_loss
from superlearner.cam_unified.learn.trajectory import Trajectory


def _traj(B, T, actions, logits, reward, version=0, value=None, done_last=True):
    logp = torch.log_softmax(logits.detach(), -1)
    beh = logp[actions]
    beh = beh.view(B, 1).expand(B, T).clone() if actions.dim() == 1 else beh.clone()
    done = torch.zeros(B, T, dtype=torch.bool)
    if done_last:
        done[:, -1] = True
    return Trajectory(reward=reward, dt=torch.ones(B, T), done=done, behavior_logp=beh,
                      model_version=torch.full((B, T), version, dtype=torch.long),
                      value=torch.zeros(B, T + 1) if value is None else value)


def _one_step(obj, traj, theta, actions, current_version):
    prep = obj.prepare(traj, current_version)
    logp_all = torch.log_softmax(theta, -1)
    new_logp = logp_all[actions].view(traj.B, traj.T)
    ent = -(logp_all.exp() * logp_all).sum().expand(traj.B, traj.T)
    out = obj.loss(traj, prep, new_logp, ent, torch.zeros(traj.B, traj.T + 1))
    (g,) = torch.autograd.grad(out["actor"], theta)
    return out, g


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("stale", [False, True])
def test_policy_gradient_sign_follows_advantage(sign, stale):
    """A rewarded action becomes more likely, a penalised one less likely — for PPO (fresh) and V-trace (stale)."""
    theta = torch.zeros(3, requires_grad=True)
    B = 8
    actions = torch.zeros(B, dtype=torch.long)             # every row took action 0
    reward = torch.full((B, 1), sign)
    reward[B // 2:] = 0.0                                  # baseline variation so normalised advantages keep the sign
    actions[B // 2:] = 1
    traj = _traj(B, 1, actions, theta.detach(), reward, version=0 if not stale else 0)
    obj = ActorCriticObjective(kappa=0.05, normalize_advantages=False, entropy_coef=0.0, max_policy_lag=0)
    out, g = _one_step(obj, traj, theta, actions, current_version=3 if stale else 0)
    if stale:
        assert out["n_stale_steps"] == B and out["n_fresh_steps"] == 0
    else:
        assert out["n_fresh_steps"] == B and out["n_stale_steps"] == 0
    new = theta.detach() - 0.5 * g                         # one gradient-descent step on the actor loss
    p_old = torch.softmax(theta.detach(), -1)[0]
    p_new = torch.softmax(new, -1)[0]
    assert (p_new - p_old).item() * sign > 1e-4


def test_ppo_refuses_stale_ratios():
    lp = torch.zeros(2, 3)
    with pytest.raises(StalePolicyError):
        ppo_clip_loss(lp, lp, torch.ones(2, 3), torch.ones(2, 3, dtype=torch.bool), 0.2,
                      lag=torch.tensor([[0, 0, 0], [0, 2, 0]]), max_policy_lag=1)
    loss, _ = ppo_clip_loss(lp, lp, torch.ones(2, 3), torch.ones(2, 3, dtype=torch.bool), 0.2,
                            lag=torch.ones(2, 3, dtype=torch.long), max_policy_lag=1)
    assert torch.isfinite(loss)


def test_routing_by_lag_and_by_measured_divergence():
    B, T = 4, 5
    theta = torch.zeros(3, requires_grad=True)
    actions = torch.zeros(B, T, dtype=torch.long)
    traj = _traj(B, T, actions, theta.detach(), torch.randn(B, T))
    traj.model_version[1] = 0
    traj.model_version[0] = 5          # row 0 produced by the current version, the rest 5 versions ago
    traj.model_version[1:] = 0
    obj = ActorCriticObjective(max_policy_lag=2, max_fresh_kl=0.05)
    prep = obj.prepare(traj, current_version=5)
    assert prep.fresh.tolist() == [True, False, False, False]
    # every row claims version 5 (fresh), but rows 2-3 were actually produced by a very different policy
    traj.model_version[:] = 5
    traj.behavior_logp[2:] = -5.0      # behaviour log-probs inconsistent with the current policy
    prep = obj.prepare(traj, current_version=5)
    assert bool(prep.fresh.all())
    new_logp = torch.log_softmax(theta, -1)[actions]
    out = obj.loss(traj, prep, new_logp, torch.zeros(B, T), torch.zeros(B, T + 1))
    assert out["rerouted_by_kl"] == 2 and prep.fresh.tolist() == [True, True, False, False]
    assert out["n_stale_steps"] == 2 * T and "vtrace_rho_mean" in out


def test_clipping_removes_incentive_beyond_trust_region_and_kl_stop():
    lp_old = torch.zeros(1, 1)
    new = torch.tensor([[0.5]], requires_grad=True)       # ratio e^0.5 = 1.65 > 1.2
    loss, d = ppo_clip_loss(new, lp_old, torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool), 0.2)
    (g,) = torch.autograd.grad(loss, new)
    assert g.abs().item() == 0.0 and d["clip_frac"] == 1.0
    new2 = torch.tensor([[0.05]], requires_grad=True)
    loss2, _ = ppo_clip_loss(new2, lp_old, torch.ones(1, 1), torch.ones(1, 1, dtype=torch.bool), 0.2)
    (g2,) = torch.autograd.grad(loss2, new2)
    assert g2.item() < 0                                   # still pushes the rewarded log-prob up
    m = torch.ones(1, 4, dtype=torch.bool)
    assert approx_kl(torch.zeros(1, 4), torch.zeros(1, 4), m).item() == pytest.approx(0.0)
    assert approx_kl(torch.zeros(1, 4), torch.full((1, 4), -1.0), m).item() > 0.3
    theta = torch.zeros(3, requires_grad=True)
    acts = torch.zeros(2, 3, dtype=torch.long)
    traj = _traj(2, 3, acts, theta.detach(), torch.ones(2, 3))
    obj = ActorCriticObjective(target_kl=0.01, max_fresh_kl=10.0)
    prep = obj.prepare(traj, 0)
    out = obj.loss(traj, prep, traj.behavior_logp - 0.5, torch.zeros(2, 3), torch.zeros(2, 4))
    assert out["stop"] is True


def test_value_clipping_and_twohot_value_path():
    B, T = 2, 3
    theta = torch.zeros(3)
    acts = torch.zeros(B, T, dtype=torch.long)
    traj = _traj(B, T, acts, theta, torch.full((B, T), 10.0))
    obj = ActorCriticObjective(value_clip=0.1, normalize_advantages=False)
    prep = obj.prepare(traj, 0)
    v = torch.full((B, T + 1), 5.0, requires_grad=True)   # far from the old values (0): clipped term dominates
    out = obj.loss(traj, prep, traj.behavior_logp, torch.zeros(B, T), v)
    obj2 = ActorCriticObjective(value_clip=None, normalize_advantages=False)
    out2 = obj2.loss(traj, obj2.prepare(traj, 0), traj.behavior_logp, torch.zeros(B, T), v)
    assert out["value"].item() > out2["value"].item()
    head = ValueHead(16, n_heads=2)
    logits = torch.zeros(B, T + 1, 255, requires_grad=True)
    out3 = obj2.loss(traj, obj2.prepare(traj, 0), traj.behavior_logp, torch.zeros(B, T), value_logits=logits,
                     value_head=head)
    (g,) = torch.autograd.grad(out3["value"], logits)
    assert g.abs().sum() > 0 and torch.isfinite(out3["value"])


def test_advantage_normalisation_does_not_amplify_value_noise_of_a_converged_policy():
    """Deterministic policy + deterministic task: every return is 530 and the advantages are pure value error (+-1).
    Standardising by their std would make them unit-size noise; the relative floor keeps them small."""
    B, T = 16, 4
    g = torch.Generator().manual_seed(0)
    reward = torch.zeros(B, T)
    reward[:, -1] = 530.0
    value = torch.full((B, T + 1), 530.0) + torch.randn(B, T + 1, generator=g)
    value[:, -1] = 0.0
    traj = Trajectory(reward=reward, dt=torch.zeros(B, T), done=torch.zeros(B, T, dtype=torch.bool).index_fill_(
        1, torch.tensor([T - 1]), True), behavior_logp=torch.zeros(B, T),
        model_version=torch.zeros(B, T, dtype=torch.long), value=value)
    obj = ActorCriticObjective(kappa=0.0, gae_lambda=0.95)
    prep = obj.prepare(traj, 0)
    adv = obj._norm_adv(prep.adv, traj.mask, prep)
    assert adv.abs().max().item() < 0.2
    noisy = ActorCriticObjective(kappa=0.0, gae_lambda=0.95, adv_rel_floor=0.0)
    assert noisy._norm_adv(prep.adv, traj.mask, prep).abs().max().item() > 1.0


def _imag(rewards, errors, prov=None):
    from superlearner.cam_unified.interfaces import ImaginedStep, ImaginedTrajectory, Provenance
    steps = [ImaginedStep(Z=None, action=[], act_tokens=None, outcome_mean=None, outcome_var=None, value=None,
                          reward=torch.tensor(r), belief_ess=None, model_error=torch.tensor(e))
             for r, e in zip(rewards, errors)]
    return ImaginedTrajectory(steps, returns=None, branch_of=torch.zeros(2), first_candidate=torch.zeros(2),
                              provenance=prov or Provenance.IMAGINED)


def test_imagined_actor_critic_downweights_unreliable_imagination_and_refuses_real_data():
    from superlearner.cam_unified.interfaces import Provenance
    from superlearner.cam_unified.learn.ppo import imagined_actor_critic
    # two branches, horizon 3: branch 0 is predicted reliably, branch 1's model error explodes after step 0
    imag = _imag([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 5.0], [0.0, 5.0]])
    logp = torch.zeros(2, 3, requires_grad=True)
    value = torch.zeros(2, 4, requires_grad=True)
    d = imagined_actor_critic(imag, logp, torch.zeros(2, 3), value, torch.zeros(2, 3), kappa=0.0, lam=1.0,
                              normalize=False)
    gl, gv = torch.autograd.grad(d["actor"] + d["value"], [logp, value])
    assert gl[0].abs().min() > 0
    assert torch.all(gl[1, 1:].abs() < 0.01 * gl[0, 1:].abs())             # weight exp(-5) or less
    assert gl[1, 0].abs().item() == pytest.approx(gl[0, 0].abs().item())     # step 0 equally reliable
    assert d["effective_horizon"] < 3.0 and d["effective_horizon"] > 2.0     # (3 + ~1) / 2
    assert gv[0, 0].item() < 0                                               # value pulled up towards return 3
    with pytest.raises(ValueError):
        imagined_actor_critic(_imag([[1.0, 1.0]], [[0.0, 0.0]], Provenance.REAL), torch.zeros(2, 1),
                              torch.zeros(2, 1), torch.zeros(2, 2), torch.zeros(2, 1))


def test_from_config_contract_sequence_and_per_family_percentile_scaling():
    from superlearner.cam_unified.config import make_config
    from superlearner.cam_unified.learn.returns import ReturnNormalizer
    from superlearner.cam_unified.testing import rand_sequence, tiny_config
    cfg = make_config("dev")
    obj = ActorCriticObjective.from_config(cfg)
    assert (obj.kappa, obj.lam, obj.clip, obj.entropy_coef, obj.rho_clip) == (cfg.kappa, cfg.gae_lambda, cfg.ppo_clip,
                                                                               cfg.entropy_coef, cfg.vtrace_rho)
    seq = rand_sequence(3, 5, tiny_config(), seed=2)
    for t, s in enumerate(seq.steps):
        s.event_ids = [f"e{b}_{t}" for b in range(3)]
        s.model_version = torch.tensor([0, 4, 3])            # row 1 produced by the current policy, row 2 one behind
    tr = Trajectory.from_sequence(seq, value=torch.zeros(3, 6))
    assert tr.reward.shape == (3, 5) and tr.event_ids[1][2] == "e1_2"
    prep = ActorCriticObjective(max_policy_lag=2).prepare(tr, current_version=4)
    assert prep.fresh.tolist() == [False, True, True]           # row 0 is 4 versions old (> validated lag 2)
    # percentile return scaling is per family: a 100x larger family does not shrink the other family's advantages
    B, T = 4, 3
    rew = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [100.0, 0.0, 200.0], [0.0, 100.0, 0.0]])
    traj = Trajectory(reward=rew, dt=torch.zeros(B, T), done=torch.zeros(B, T, dtype=torch.bool),
                      behavior_logp=torch.zeros(B, T), model_version=torch.zeros(B, T, dtype=torch.long),
                      value=torch.zeros(B, T + 1), family=["small", "small", "large", "large"])
    o = ActorCriticObjective(kappa=0.0, gae_lambda=1.0, return_normalizer=ReturnNormalizer(rate=1.0, limit=0.01))
    p = o.prepare(traj, 0)
    a = o._norm_adv(p.adv, traj.mask, p)
    assert torch.allclose(a[0], a[2], atol=1e-4) and torch.allclose(a[1], a[3], atol=1e-4)
