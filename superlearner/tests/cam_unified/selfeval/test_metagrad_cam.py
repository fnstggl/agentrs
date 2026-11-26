"""MetaRewardTrainer: the psi meta step over a named fast set phi with the real RewardModel (plan §3 W8)."""
import os

import pytest
import torch
import torch.nn as nn
from torch.func import functional_call

from superlearner.cam_unified.config import make_config
from superlearner.cam_unified.interfaces import TokenKind, TokenSet
from superlearner.cam_unified.selfeval.metagrad import (InnerBatch, MetaRewardTrainer, QueryBatch,
                                                         default_fast_params)
from superlearner.cam_unified.selfeval.reward_model import RewardModel

torch.set_num_threads(4)
D, A, N, T, S = 16, 3, 8, 4, 2


class TinyModel(nn.Module):
    """A policy whose decision head is the fast set; `trunk` stands for the slow core (test double)."""

    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(D, D)
        self.head = nn.Linear(D, A)
        self.weights_epoch = 0

    def logits(self, x):
        return self.head(torch.tanh(self.trunk(x)))

    def bump_weights_epoch(self):
        self.weights_epoch += 1


def logp_fn(call, fast_prefixed, inputs, actions):
    lg = functional_call(call, fast_prefixed, args=(call.m.logits, inputs["x"]), strict=False)
    lp = torch.log_softmax(lg.float(), -1)
    return lp.gather(-1, actions[..., None]).squeeze(-1), -(lp.exp() * lp).sum(-1)


def _setup(seed=0):
    torch.manual_seed(seed)
    cfg = make_config("dev", reward_mode="observation_only", d_model=D, n_heads=4)
    rm = RewardModel(cfg, d_hidden=16)
    with torch.no_grad():                                  # leave the zero-init so rho contributes to the inner step
        rm.intrinsic.mlp.fc2.weight.normal_(0, 0.1)
    model = TinyModel()

    def reward_fn(R, Rn, goal, dt, ig):
        r = rm.reward_psi(R.reshape(N * T, S, D), Rn.reshape(N * T, S, D), goal, dt.reshape(-1),
                          info_gain=ig.reshape(-1))
        return r.reshape(N, T)

    tr = MetaRewardTrainer(model, rm, logp_fn=logp_fn, reward_fn=reward_fn,
                           fast_params={n: p for n, p in model.named_parameters() if n.startswith("head.")},
                           inner_lr=0.5, K=2, outer_lr=1e-2)
    return cfg, rm, model, tr


def _batch(g):
    x = torch.randn(N, T, D, generator=g)
    a = torch.randint(0, A, (N, T), generator=g)
    R = torch.randn(N, T, S, D, generator=g)
    goal = TokenSet.from_dense(torch.randn(N * T, 1, D, generator=g), int(TokenKind.GOAL))
    ri = {"R": R, "Rn": torch.roll(R, -1, 1), "goal": goal, "dt": torch.rand(N, T, generator=g),
          "ig": torch.rand(N, T, generator=g)}
    return x, a, ri


def test_meta_step_changes_psi_bumps_the_version_and_leaves_the_slow_weights():
    cfg, rm, model, tr = _setup()
    assert set(tr.fast) == {"head.weight", "head.bias"} and tr.n_psi == sum(p.numel() for p in rm.meta_parameters())
    g = torch.Generator().manual_seed(1)
    slow = {n: p.detach().clone() for n, p in model.named_parameters()}
    fast = tr.start()
    for k in range(2):
        with tr.act_with(fast):                            # phi_k is what acts while D_k is collected
            assert torch.equal(model.head.weight, fast["head.weight"].detach())
            x, a, ri = _batch(g)
        fast, st = tr.inner_step(fast, InnerBatch({"x": x}, a, torch.ones(N, T, dtype=torch.bool), ri))
        assert fast["head.weight"].requires_grad and st["grad_norm"] > 0
    assert all(torch.equal(p, slow[n]) for n, p in model.named_parameters())     # restored after acting
    assert model.weights_epoch == 4
    x, a, _ = _batch(g)
    q = QueryBatch({"x": x}, a, torch.ones(N, T, dtype=torch.bool), utility=torch.randn(N, generator=g))
    v0 = tr.version
    out = tr.outer_step(fast, q, score_baseline=0.0)
    assert out["psi_change"] > 0 and tr.version == v0 + 1 == out["psi_version"]
    assert out["inner_score_terms"] == 1                   # the second inner batch was on-policy under phi_1(psi)
    assert all(torch.equal(p, slow[n]) for n, p in model.named_parameters())
    tr.commit(fast)
    assert torch.allclose(model.head.weight, fast["head.weight"].detach())
    assert torch.equal(model.trunk.weight, slow["trunk.weight"])                 # core never inner-updated


def test_actor_losses_have_no_gradient_path_into_psi_but_the_meta_loss_does():
    cfg, rm, model, tr = _setup(2)
    g = torch.Generator().manual_seed(3)
    x, a, ri = _batch(g)
    r = rm.reward(ri["R"].reshape(N * T, S, D), ri["Rn"].reshape(N * T, S, D), ri["goal"], ri["dt"].reshape(-1),
                  info_gain=ri["ig"].reshape(-1)).reshape(N, T)
    lp = torch.log_softmax(model.logits(x), -1).gather(-1, a[..., None]).squeeze(-1)
    L_actor = -(lp * (r - r.mean())).sum()
    gs = torch.autograd.grad(L_actor, rm.meta_parameters() + list(model.parameters()), allow_unused=True)
    assert all(g_ is None for g_ in gs[:len(rm.meta_parameters())])
    assert any(g_ is not None for g_ in gs[len(rm.meta_parameters()):])
    fast = tr.start()
    fast, _ = tr.inner_step(fast, InnerBatch({"x": x}, a, torch.ones(N, T, dtype=torch.bool), ri), on_policy=False)
    q = QueryBatch({"x": x}, a, torch.ones(N, T, dtype=torch.bool), utility=torch.randn(N, generator=g))
    gm = tr.mrl.meta_gradient(fast, q)
    assert sum(float(t.abs().sum()) for t in gm) > 0


def test_fast_set_must_not_overlap_psi_and_default_fast_params_skips_the_semantic_path():
    cfg, rm, model, tr = _setup()
    with pytest.raises(ValueError):
        MetaRewardTrainer(model, rm, logp_fn=logp_fn, reward_fn=lambda **k: None,
                          fast_params={"x": rm.log_lambda_ig})

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_generator = nn.ModuleDict({"tool_q": nn.Linear(2, 2), "bridge": nn.Linear(2, 2),
                                                   "sem": nn.Linear(2, 2)})
            self.value_head = nn.Linear(2, 1)
            self.core = nn.Linear(2, 2)
    names = set(default_fast_params(M()))
    assert names == {"action_generator.tool_q.weight", "action_generator.tool_q.bias", "value_head.weight",
                     "value_head.bias"}


@pytest.mark.gpu
def test_real_dom_dev_meta_step_through_the_lifetime_loop(tmp_path):
    """The W8 end-to-end path at smoke size on the real dom_dev stack (Qwen3-0.6B); the full run is
    `python -m superlearner.cam_unified.selfeval.e2e_w8` (evidence JSON)."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from superlearner.cam_unified.selfeval import e2e_w8
    ev = e2e_w8.run(str(tmp_path / "e2e.json"), n_probe_batches=2, B=8, train_steps=5)
    c = ev["checks"]
    for k in ("meta_step_changes_psi", "psi_version_bumped", "reward_recomputed_under_current_psi",
              "d_actor_d_psi_is_zero", "label_canaries_absent", "reward_invariant_to_label_channel"):
        assert c[k], (k, ev)
    assert os.path.exists(tmp_path / "e2e.json")
