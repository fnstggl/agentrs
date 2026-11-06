"""Real gradient-based learning: gradients, parameter updates, loss decrease, checkpoint/resume."""

import random

import torch

from agentrs.models.backbone import LMBackbone
from agentrs.models.critic import Critic, CriticConfig
from agentrs.models.policy import LMPolicy, PolicyConfig
from agentrs.models.world_model import WorldModel, WorldModelConfig
from agentrs.training.bc import BCConfig, bc_loss, make_examples, train_bc
from agentrs.training.common import (
    GradientTrainer,
    OptimConfig,
    load_model,
    load_trainer_state,
    param_change,
    param_snapshot,
    save_checkpoint,
)
from agentrs.training.world_model import make_wm_examples, wm_batch_loss


def test_prefix_cached_scoring_matches_naive(scratch_bb):
    bb = LMBackbone(scratch_bb.model_copy(update={"dtype": "float64"}))
    prompts = ["GOAL: x\nOBSERVATION:\n[e1] button \"A\"\nACTION:", "GOAL: longer goal text here\nACTION:", "z"]
    conts = [[" click(e1)\n", " finish()\n"], [" wait()\n", ' type(e3, "hello")\n', " scroll(down)\n"], []]
    for vr in (False, True):
        a = bb.score(prompts, conts, prefix_cache=False, vocab_restrict=vr)
        bb.zero_grad()
        sum(x.sum() for x in a).backward()
        g1 = {n: p.grad.clone() for n, p in bb.named_parameters() if p.grad is not None}
        b = bb.score(prompts, conts, prefix_cache=True, vocab_restrict=vr)
        bb.zero_grad()
        sum(x.sum() for x in b).backward()
        g2 = {n: p.grad.clone() for n, p in bb.named_parameters() if p.grad is not None}
        for x, y in zip(a, b):
            assert torch.allclose(x, y, atol=1e-5)
        for n in g1:
            assert torch.allclose(g1[n], g2[n], atol=1e-6, rtol=1e-4), n


def test_policy_distribution_is_normalised_over_candidates(scratch_bb):
    pol = LMPolicy(PolicyConfig(backbone=scratch_bb))
    lp = pol.log_probs(["GOAL: g\nACTION:"], [["click(e1)", "click(e2)", "finish()"]])[0]
    assert torch.isclose(lp.exp().sum(), torch.tensor(1.0), atol=1e-5)
    # the log-prob of a candidate equals the backbone sequence score renormalised over the set
    raw = pol.backbone.score(["GOAL: g\nACTION:"], [[" click(e1)\n", " click(e2)\n", " finish()\n"]],
                             vocab_restrict=True)[0]
    assert torch.allclose(lp, raw - torch.logsumexp(raw, 0), atol=1e-5)


def test_bc_gradients_param_change_and_loss_decrease(scratch_bb, demo_data):
    torch.manual_seed(0)
    train, val = demo_data
    pol = LMPolicy(PolicyConfig(backbone=scratch_bb))
    ex = make_examples(train, BCConfig())
    batch = random.Random(0).sample(ex, 16)
    loss, _ = bc_loss(pol, batch)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in pol.parameters() if p.requires_grad]
    assert all(g is not None for g in grads) and sum(float(g.abs().sum()) for g in grads) > 0
    pol.zero_grad()
    snap = param_snapshot(pol)
    cfg = BCConfig(steps=60, batch_size=16, eval_every=60, optim=OptimConfig(lr=3e-3, total_steps=60))
    r = train_bc(pol, train, val, cfg)
    assert param_change(pol, snap) > 0
    first = sum(h["loss"] for h in r["history"][:10]) / 10
    last = sum(h["loss"] for h in r["history"][-10:]) / 10
    assert last < first, (first, last)
    assert r["after"]["val_nll"] < r["before"]["val_nll"]


def test_checkpoint_roundtrip_preserves_predictions(tmp_path, scratch_bb):
    torch.manual_seed(1)
    wm = WorldModel(WorldModelConfig(backbone=scratch_bb, n_heads=3))
    with torch.no_grad():
        for p in wm.heads.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    path = save_checkpoint(tmp_path / "wm", wm)
    wm2 = load_model(path)
    prompts = ["GOAL: a\nOBSERVATION:\nURL: /\nACTION: click(e1)\nEFFECT:"]
    m1, s1 = wm.predict_heads(prompts)
    m2, s2 = wm2.predict_heads(prompts)
    assert torch.allclose(m1, m2) and torch.allclose(s1, s2)
    cr = Critic(CriticConfig(backbone=scratch_bb))
    cr2 = load_model(save_checkpoint(tmp_path / "cr", cr))
    assert torch.allclose(cr.value(["GOAL: a\nACTION:"]), cr2.value(["GOAL: a\nACTION:"]))


def test_resume_restores_optimizer_scheduler_and_rng(tmp_path, scratch_bb, demo_data):
    """Training 2+2 steps with a save/resume in between equals training 4 steps continuously."""
    train, _ = demo_data
    ex = make_examples(train, BCConfig())

    def run(n_steps, resume_from=None, save_at=None):
        torch.manual_seed(0)
        pol = LMPolicy(PolicyConfig(backbone=scratch_bb))
        tr = GradientTrainer(pol, OptimConfig(lr=1e-3, total_steps=4, warmup_steps=1, schedule="cosine"))
        start = 0
        if resume_from:
            from agentrs.training.common import load_weights

            load_weights(resume_from, pol)
            load_trainer_state(resume_from, tr)
            start = tr.global_step
        for step in range(start, n_steps):
            rng = random.Random(step)
            loss, _ = bc_loss(pol, rng.sample(ex, 8))
            tr.step(loss)
            if save_at is not None and step + 1 == save_at:
                save_checkpoint(tmp_path / "ck", pol, tr)
        return pol, tr

    full, trf = run(4)
    run(2, save_at=2)
    resumed, trr = run(4, resume_from=tmp_path / "ck")
    assert trr.global_step == trf.global_step == 4
    assert trr.scheduler.get_last_lr() == trf.scheduler.get_last_lr()
    for (n, p), (_, q) in zip(full.named_parameters(), resumed.named_parameters()):
        assert torch.allclose(p, q, atol=1e-6), n


def test_divergence_guard_skips_nonfinite_updates(scratch_bb):
    pol = LMPolicy(PolicyConfig(backbone=scratch_bb))
    tr = GradientTrainer(pol, OptimConfig(max_bad_steps=2))
    snap = param_snapshot(pol)
    out = tr.step(torch.tensor(float("nan"), requires_grad=True))
    assert out["skipped"] == 1.0 and param_change(pol, snap) == 0.0
    tr.step(torch.tensor(float("inf"), requires_grad=True))
    import pytest

    from agentrs.training.common import DivergenceError

    with pytest.raises(DivergenceError):
        tr.step(torch.tensor(float("nan"), requires_grad=True))


def test_world_model_training_step_changes_params(scratch_bb, demo_data):
    train, _ = demo_data
    wm = WorldModel(WorldModelConfig(backbone=scratch_bb, n_heads=3))
    ex = make_wm_examples(wm, train)[:8]
    loss, st = wm_batch_loss(wm, ex)
    assert torch.isfinite(loss) and st["diff_nll_per_token"] > 0
    tr = GradientTrainer(wm, OptimConfig(lr=1e-3))
    snap = param_snapshot(wm)
    tr.step(loss)
    assert param_change(wm, snap) > 0
    g = wm.backbone.model.gpt_neox.layers[0].attention.query_key_value.weight.grad
    assert g is not None and float(g.abs().sum()) > 0  # gradient reached the pretrained-style backbone
