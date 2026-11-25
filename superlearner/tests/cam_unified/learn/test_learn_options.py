"""Hierarchical options end to end: the option decision's exact log-probability is part of the joint action trained by
the PPO objective; semi-Markov option targets (gamma_Delta = exp(-kappa Delta t)) train an option-conditioned value."""
import math

import pytest
import torch

from superlearner.cam_unified.learn.heads import OptionModule, ValueHead, deliberation_penalty
from superlearner.cam_unified.learn.ppo import ActorCriticObjective
from superlearner.cam_unified.learn.returns import option_targets, semi_markov_discount
from superlearner.cam_unified.learn.trajectory import Trajectory

KAPPA = 0.05


def _rollout(mod, R, T, gen):
    B = R.shape[0]
    prev = torch.full((B,), -1, dtype=torch.long)
    steps = torch.zeros(B, dtype=torch.long)
    rec = {k: [] for k in ("prev", "steps", "option", "new", "term", "vol", "logp")}
    with torch.no_grad():
        for _ in range(T):
            out = mod.step(R, None, prev, steps, generator=gen)
            for k, v in (("prev", prev), ("steps", steps), ("option", out.option), ("new", out.new),
                         ("term", out.terminated), ("vol", out.voluntary), ("logp", out.logp)):
                rec[k].append(v)
            prev, steps = out.option, out.steps
    st = {k: torch.stack(v, 1) for k, v in rec.items()}
    reward = (st["option"] == 1).float() + deliberation_penalty(st["vol"], 0.3)
    return st, reward


def _recompute(mod, R, st):
    B, T = st["option"].shape
    Rr = R.unsqueeze(1).expand(B, T, *R.shape[1:]).reshape(B * T, *R.shape[1:])
    lp, ent = mod.log_prob(Rr, None, st["prev"].reshape(-1), st["steps"].reshape(-1), st["option"].reshape(-1),
                           st["new"].reshape(-1))
    return lp.view(B, T), ent.view(B, T)


def test_options_trained_through_the_joint_action_log_probability_and_smdp_values():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    B, T, D = 128, 8, 32
    mod = OptionModule(D, n_options=4, max_steps=6, n_heads=4)
    R = torch.randn(B, 3, D)
    obj = ActorCriticObjective(kappa=KAPPA, gae_lambda=1.0, entropy_coef=0.0, max_fresh_kl=1.0)
    opt = torch.optim.Adam(mod.parameters(), lr=1e-2)
    dt = torch.full((B, T), 0.5)
    for it in range(60):
        st, reward = _rollout(mod, R, T, gen)
        done = torch.zeros(B, T, dtype=torch.bool)
        done[:, -1] = True
        traj = Trajectory(reward=reward, dt=dt, done=done, behavior_logp=st["logp"],
                          model_version=torch.full((B, T), it, dtype=torch.long), value=torch.zeros(B, T + 1),
                          option=st["option"], option_start=st["new"])
        prep = obj.prepare(traj, it)
        for _ in range(3):
            lp, ent = _recompute(mod, R, st)
            d = obj.loss(traj, prep, lp, ent, torch.zeros(B, T + 1))
            opt.zero_grad()
            d["actor"].backward()
            opt.step()
    st, reward = _rollout(mod, R, T, gen)
    assert (st["option"] == 1).float().mean() > 0.9                       # the paying option is selected
    assert st["vol"].float().sum(1).mean() < 0.5                           # and kept (few voluntary switches)
    assert torch.all(st["term"][:, 6] | st["vol"][:, :6].any(1))           # forced re-selection at max_steps = 6
    assert mod.beta(R, torch.full((B,), 1)).mean() < 0.15
    # ---- semi-Markov option values through the actor-critic objective: an option-conditioned critic
    # Q(R, OPTION token) is regressed on R_option + exp(-kappa Delta t) V(s_next) at every option start.
    # Critic data come from an untrained option policy (every option gets selected).
    gam = semi_markov_discount(dt, KAPPA)
    mod0 = OptionModule(D, n_options=4, max_steps=6, n_heads=4)
    st, reward = _rollout(mod0, R, T, gen)
    done = torch.zeros(B, T, dtype=torch.bool)
    done[:, -1] = True
    traj = Trajectory(reward=reward, dt=dt, done=done, behavior_logp=st["logp"],
                      model_version=torch.zeros(B, T, dtype=torch.long), value=torch.zeros(B, T + 1),
                      option=st["option"], option_start=st["new"])
    prep = obj.prepare(traj, 0)
    starts = st["new"]
    G = prep.option_gamma
    for b in range(B):                                                    # gamma_Delta = exp(-kappa * Delta t)
        idx = starts[b].nonzero().flatten().tolist() + [T]
        for s_, e in zip(idx[:-1], idx[1:]):
            want = 0.0 if e == T else math.exp(-KAPPA * 0.5 * (e - s_))
            assert abs(G[b, s_].item() - want) < 1e-5
    c = torch.ones(B, T)
    c[:, -1] = 0
    tgt_ref, _ = option_targets(reward, torch.zeros(B, T + 1), gam, c, starts)
    assert torch.allclose(prep.option_target, tgt_ref)
    assert len(set(st["option"][starts].tolist())) == 4
    head = ValueHead(D, n_heads=4)
    hopt = torch.optim.Adam(head.parameters(), lr=3e-3)
    Rr = R.unsqueeze(1).expand(B, T, *R.shape[1:]).reshape(B * T, *R.shape[1:])
    tok = _det(mod0.token(st["option"].reshape(-1)))
    lp0, ent0 = _recompute(mod0, R, st)
    for _ in range(300):
        q_logits = head(Rr, extra=[tok]).logits.view(B, T, -1)
        d = obj.loss(traj, prep, lp0.detach(), ent0.detach(), torch.zeros(B, T + 1), option_value_logits=q_logits,
                     option_value_head=head)
        hopt.zero_grad()
        d["value"].backward()
        hopt.step()
    assert d["n_option_targets"] == int(starts.sum())
    with torch.no_grad():
        v1 = head(R, extra=[_det(mod0.token(torch.full((B,), 1)))]).value
        v0 = head(R, extra=[_det(mod0.token(torch.zeros(B, dtype=torch.long)))]).value
        q = head(Rr, extra=[tok]).value.view(B, T)
    assert (v1 - v0).mean() > 2.0
    # the SMDP target of option 1 run from decision 0 until its forced end at max_steps = 6 is sum_{k<6} gamma^k
    full = sum(math.exp(-KAPPA * 0.5 * k) for k in range(6))
    run0 = starts[:, 0] & (starts[:, 1:6].sum(1) == 0) & (st["option"][:, 0] == 1)
    assert bool(run0.any())
    assert torch.allclose(prep.option_target[run0, 0], torch.full((int(run0.sum()),), full), atol=1e-4)
    # the critic sees (R_b, option) only, so it regresses on the mean option target of each (row, option)
    err = []
    for b in range(B):
        for o in st["option"][b][starts[b]].unique().tolist():
            sel = starts[b] & (st["option"][b] == o)
            err.append((q[b][sel] - prep.option_target[b][sel].mean()).abs().mean().item())
    assert sum(err) / len(err) < 0.3
    # option values need option bookkeeping
    plain = Trajectory(reward=reward, dt=dt, done=done, behavior_logp=st["logp"],
                       model_version=torch.zeros(B, T, dtype=torch.long), value=torch.zeros(B, T + 1))
    with pytest.raises(ValueError):
        obj.loss(plain, obj.prepare(plain, 0), lp0, ent0, torch.zeros(B, T + 1), option_value=torch.zeros(B, T))


def test_option_module_from_config_and_sampling_with_a_generator_on_another_device():
    from superlearner.cam_unified.testing import tiny_config
    cfg = tiny_config(n_options=5, option_max_steps=7)
    mod = OptionModule.from_config(cfg)
    assert (mod.n, mod.max_steps, mod.D) == (5, 7, cfg.d_model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mod = mod.to(dev)
    R = torch.randn(4, 3, cfg.d_model, device=dev)
    prev = torch.full((4,), -1, dtype=torch.long, device=dev)
    a = mod.step(R, None, prev, torch.zeros(4, dtype=torch.long, device=dev), generator=torch.Generator().manual_seed(0))
    b = mod.step(R, None, prev, torch.zeros(4, dtype=torch.long, device=dev), generator=torch.Generator().manual_seed(0))
    assert torch.equal(a.option, b.option) and a.option.device.type == torch.device(dev).type


def _det(ts):
    return ts.__class__(ts.x.detach(), ts.mask, ts.kind, ts.time, ts.entity)
