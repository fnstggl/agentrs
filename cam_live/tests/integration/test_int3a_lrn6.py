"""LRN-6 reconciled (INT3A): a refused promotion rolls the learner back but keeps what its gates do not judge.

The promotion gates judge held-out validation losses of the learner's providers and retention on fixed sets of
earlier experience (world loss, decision agreement of the policy). R_psi (changed only by the outer meta-gradient on
verified evidence) and its meta optimizer are judged by neither, so a rollback keeps them; the policy head is judged
only through the decision-agreement retention metric, so it is kept unless that metric is a reason of the refusal.
Before 07be9ba every rollback restored psi, the meta optimizer and the policy head with the rest (the meta-step of
the real-model pathway run was undone before the next live update, LRN.md). Here psi, the meta optimizer state and
the policy head are all changed *after* the last promotion, so a rollback that restored them would be visible.
"""
from __future__ import annotations

import copy

import torch

from cam_live.agent import CamLiveAgent
from cam_live.contracts import SimClock
from int_helpers import Recorder, agent_config, run_lifetime


def _agent(tmp_path):
    return CamLiveAgent(agent_config(use_backbone=False, planner=False, promote_every=10_000, retention_records=8),
                        str(tmp_path / "agent"), clock=SimClock())


def _meta_step(agent, seed):
    """One step of the meta optimizer on R_psi (a stand-in for an applied meta-update: psi and Adam state move)."""
    g = torch.Generator().manual_seed(seed)
    opt = agent.meta.optimizer
    opt.zero_grad()
    for grp in opt.param_groups:
        for p in grp["params"]:
            p.grad = torch.randn(p.shape, generator=g, dtype=p.dtype).to(p.device) * 0.1
    opt.step()
    opt.zero_grad()


def _perturb(agent, which, scale, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in agent.model.named_parameters():
            if p.requires_grad and which in n:
                p.add_(torch.randn(p.shape, generator=g, dtype=p.dtype).to(p.device) * scale)


def _state(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def _same(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def _opt_state(opt):
    sd = copy.deepcopy(opt.state_dict())
    return [{k: v.clone() for k, v in s.items() if torch.is_tensor(v)} for _, s in sorted(sd["state"].items())]


def test_a_refused_promotion_keeps_psi_the_meta_optimizer_and_an_unjudged_policy(tmp_path):
    agent = _agent(tmp_path)
    try:
        rec = Recorder(agent)
        run_lifetime(rec, agent.clock, str(tmp_path), 31, max_steps=30, max_decisions=14)
        run_lifetime(rec, agent.clock, str(tmp_path), 32, max_steps=6, max_decisions=3)
        assert agent.promotions and agent.promotions[0]["promoted"], agent.promotions[:1]
        promoted = agent.ckpts.promoted().checkpoint_id
        good = agent.ckpts.load(promoted)
        # (a) changed after the promotion: psi + meta optimizer (a meta-step), the policy head (slightly: its
        # decision agreement is unchanged), and a damaging change of the world model (the gates refuse)
        _meta_step(agent, 1)
        _perturb(agent, "predict_policy", 1e-6, 2)
        _perturb(agent, "predict_world", 0.5, 3)
        psi, pol, opt = _state(agent.meta.reward), _state(agent.meta.policy), _opt_state(agent.meta.optimizer)
        assert opt, "the meta optimizer has state after a step"
        agent.promote("test: damaged world model")
        last = agent.promotions[-1]
        assert not last["promoted"] and last["rolled_back_to"] == promoted, last
        assert _same(_state(agent.meta.reward), psi)                      # R_psi (whole module state) kept
        got = _opt_state(agent.meta.optimizer)
        assert len(got) == len(opt) and all(_same(a, b) for a, b in zip(got, opt))   # meta optimizer kept
        assert not any("decision_agreement" in r for r in last["reasons"]), last["reasons"]
        assert last["kept"] == ["reward", "policy_head"]
        assert _same(_state(agent.meta.policy), pol)                      # the policy head was not judged: kept
        restored = agent.model.trainable_state()
        world = [n for n in restored if "predict_world" in n]
        assert world and all(torch.equal(restored[n], good.weights[g][n].to(restored[n].dtype))
                             for g in good.weights for n in good.weights[g] if n in world)   # rolled back
        # (b) the policy forgot (its decision agreement on the retention set dropped): it goes back, psi stays
        _meta_step(agent, 4)
        psi2 = _state(agent.meta.reward)
        _perturb(agent, "predict_policy", 5.0, 5)
        _perturb(agent, "predict_world", 0.5, 6)
        agent.promote("test: damaged policy")
        last = agent.promotions[-1]
        assert not last["promoted"]
        if any("decision_agreement" in r for r in last["reasons"]):
            assert last["kept"] == ["reward"]
            ref = {n: good.weights[g][n] for g in good.weights for n in good.weights[g] if "predict_policy" in n}
            now = agent.model.trainable_state()
            assert ref and all(torch.equal(now[n], t.to(now[n].dtype)) for n, t in ref.items())
        else:                                     # the perturbation happened not to change any argmax
            assert last["kept"] == ["reward", "policy_head"]
        assert _same(_state(agent.meta.reward), psi2)
    finally:
        agent.close()
