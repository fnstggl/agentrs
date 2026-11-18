"""CAM milestone tests: real computation, exact probabilities, learning, isolation from hidden state."""
from __future__ import annotations

import copy
import random

import pytest
import torch

from agentrs.cam.envs import ENVS
from agentrs.cam.learner import eval_world_model, update
from agentrs.cam.model import CAM, CAMConfig
from agentrs.cam.runtime import Workers

TINY = dict(d=64, belief=96, layers=1, heads=2, n_options=4)
CFG = dict(gamma=0.99, lam=0.95, clip=0.2, epochs=2, mb_episodes=32, ent_coef=0.01, v_coef=0.5, wm_coef=1.0,
           wm_sample=1024, grad_clip=1.0, target_kl=10.0)


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return CAM(CAMConfig(**TINY))


def test_actions_follow_grammar_and_probabilities_are_exact(model):
    w = Workers(model, 16, {"negotiation": 0.5, "product": 0.3, "bizdev": 0.2}, gen_seed=3)
    eps = w.collect(min_episodes=24, version="v0")
    assert len(eps) >= 24
    for e in eps:
        g = ENVS[e.family]().grammar()
        for s in e.steps:
            prefix = []
            for t in s.act_tokens:  # every generated token was allowed by the grammar at that position
                assert t in g.allowed(prefix)
                prefix.append(t)
            assert s.act_tokens[-1] == "END" and len(s.act_ids) == len(s.allowed) == len(s.act_tokens)
    st = update(model, torch.optim.AdamW(model.parameters(), lr=1e-4), eps, "v0", CFG, random.Random(0))
    assert st["mismatch"] < 1e-4                    # recorded behaviour log-probs == re-scored (same masks, same ids)
    stale = update(model, torch.optim.AdamW(model.parameters(), lr=1e-4), eps, "v1", CFG, random.Random(0))
    assert stale["episodes"] == 0                   # episodes from another version are never used on-policy


def test_verified_rewards_change_all_core_parameters_and_behaviour(model):
    w = Workers(model, 16, {"negotiation": 1.0}, gen_seed=1)
    eps = w.collect(min_episodes=32, version="v0")
    probe = eps[:4]
    from agentrs.cam.learner import policy_terms, unroll

    with torch.no_grad():
        flat, b, g = unroll(model, probe)
        lp0, _, _ = policy_terms(model, probe, flat, b, g)
    snap = {n: p.detach().clone() for n, p in model.named_parameters()}
    update(model, torch.optim.AdamW(model.parameters(), lr=1e-3), eps, "v0", CFG, random.Random(0))
    moved = {n.split(".")[0] for n, p in model.named_parameters() if (p.detach() - snap[n]).abs().max() > 0}
    for comp in ("obs_enc", "belief_cell", "dyn", "value", "opt_pi", "dec_cell", "emb"):
        assert comp in moved, comp
    with torch.no_grad():
        flat, b, g = unroll(model, probe)
        lp1, _, _ = policy_terms(model, probe, flat, b, g)
    assert (lp1 - lp0).abs().max() > 1e-4          # the executed decisions' probabilities depend on CAM's weights
    reset = CAM(CAMConfig(**TINY, vocab=model.vocab))
    with torch.no_grad():
        flat, b, g = unroll(reset, probe)
        lp2, _, _ = policy_terms(reset, probe, flat, b, g)
    assert (lp2 - lp1).abs().max() > 1e-3          # resetting CAM changes behaviour


def test_agent_inputs_do_not_contain_hidden_state():
    for name, E in ENVS.items():
        a, b = E(), E()
        a.reset(5, "train")
        b.reset(5, "train")
        # change hidden parameters only
        for k, v in list(vars(b).items()):
            if k in ("w_c", "r_c", "concern", "cp_type", "patience", "needs", "sizes", "sens", "aff", "eff", "want",
                     "min_share", "anchor", "rival"):
                setattr(b, k, copy.deepcopy(v) if not hasattr(v, "__len__") else v)
        if name == "negotiation":
            b.r_c, b.concern = 0.99, "quality" if a.concern != "quality" else "urgent"
        if name == "product":
            b.needs = b.needs[::-1].copy()
        if name == "bizdev":
            b.want = list(reversed(b.want))
        assert a.obs_tokens() == b.obs_tokens()
        assert (a.features() == b.features()).all()


def test_checkpoint_roundtrip_reproduces_decisions(model, tmp_path):
    w = Workers(model, 8, {"product": 1.0}, gen_seed=2)
    eps = w.collect(min_episodes=8, version="v0")
    torch.save(model.state_dict(), tmp_path / "w.pt")
    m2 = CAM(CAMConfig(**TINY, vocab=model.vocab))
    m2.load_state_dict(torch.load(tmp_path / "w.pt"))
    from agentrs.cam.learner import policy_terms, unroll

    with torch.no_grad():
        a = policy_terms(model, eps, *unroll(model, eps))[0]
        b = policy_terms(m2, eps, *unroll(m2, eps))[0]
    assert torch.allclose(a, b)


def test_world_model_learning_improves_heldout_prediction(model):
    ho = Workers(model, 32, {"negotiation": 0.5, "product": 0.5}, seed0=10_000, gen_seed=5).collect(min_episodes=64, version="h")
    before = eval_world_model(model, ho, random.Random(0))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    cfg = dict(CFG, epochs=2, pg_coef=0.0, v_coef=0.0)  # isolate the predictive objective
    w = Workers(model, 32, {"negotiation": 0.5, "product": 0.5}, gen_seed=4)
    for it in range(12):  # continual loop: fresh experience each round, held-out episodes never trained on
        eps = w.collect(min_episodes=48, version=f"v{it}")
        update(model, opt, eps, f"v{it}", cfg, random.Random(it))
    after = eval_world_model(model, ho, random.Random(0))  # held-out transitions never trained on
    assert after["event_bce"] < before["event_bce"]
    assert after["feat_mse"] < before["feat_mse"]
    assert after["event_brier"] < before["event_brier"]
    assert after["belief_std"] > 1e-3                # no collapse to a constant representation
