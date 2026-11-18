"""Adaptive capacity: rank expansion, expert adapters with routing, consolidation, advisor, checkpoints."""
from __future__ import annotations

import pytest
import torch

from agentrs.capacity import (ExpansionAdvisor, ExpertAdapters, distill_into_shared, expand_lora_rank, lora_ranks,
                              match_checkpoint_ranks, param_report, transfer_optimizer_state)

pytestmark = pytest.mark.network  # needs the (cached) pythia-14m weights


def _bb(r=4):
    from agentrs.models.backbone import BackboneConfig, LMBackbone, LoraSettings

    cfg = BackboneConfig(name="EleutherAI/pythia-14m", dtype="float32", device="cpu", max_prompt_tokens=256,
                         lora=LoraSettings(r=r, alpha=2 * r, dropout=0.0, target_modules="all-linear"))
    torch.manual_seed(0)
    return LMBackbone(cfg)


P = ["TASK: reconcile invoice 1042 with payment 88.", "TASK: return the blue backpack from order 7."]


def _ids(bb):
    return [bb.tokenizer(" yes ok done", add_special_tokens=False)["input_ids"]] * 2


def _lp(bb, prompts=P):
    with torch.no_grad():
        v, m = bb.token_logprobs_from_ids(prompts, _ids(bb))
    return (v * m).sum(-1)


def _perturb(bb):
    with torch.no_grad():
        for n, p in bb.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)


def test_rank_expansion_is_function_preserving_and_trainable(tmp_path):
    from agentrs.training.common import GradientTrainer, OptimConfig, load_weights, save_checkpoint

    bb = _bb(4)
    _perturb(bb)  # a "learned" adapter
    rep0 = param_report(bb)
    before = _lp(bb)
    tr = GradientTrainer(bb, OptimConfig(lr=1e-3, total_steps=10))
    loss = -_lp_grad(bb)
    tr.step(loss)                                     # create AdamW moments
    before = _lp(bb)
    old_named = {n: p for n, p in bb.named_parameters() if p.requires_grad}
    changed = expand_lora_rank(bb, 8)
    assert changed and all(v == (4, 8) for v in changed.values())
    after = _lp(bb)
    assert torch.allclose(before, after, atol=1e-6)   # same function after growth
    rep1 = param_report(bb)
    assert rep1["trainable"] == 2 * rep0["trainable"] and rep1["frozen"] == rep0["frozen"]
    tr2 = GradientTrainer(bb, OptimConfig(lr=1e-3, total_steps=10))
    moved = transfer_optimizer_state(tr.optimizer, old_named, tr2.optimizer,
                                     {n: p for n, p in bb.named_parameters() if p.requires_grad})
    assert moved == len(old_named)
    loss = -_lp_grad(bb)
    loss.backward()
    newB = [p.grad[:, 4:] for n, p in bb.named_parameters() if "lora_B" in n and p.grad is not None]
    assert any(float(g.abs().sum()) > 0 for g in newB)  # the new capacity receives gradient
    tr2.optimizer.zero_grad()
    # checkpoint round trip into a model built with the old rank
    save_checkpoint(tmp_path / "ck", bb)
    bb2 = _bb(4)
    grown = match_checkpoint_ranks(bb2, tmp_path / "ck")
    assert grown and set(lora_ranks(bb2).values()) == {8}
    load_weights(tmp_path / "ck", bb2)
    assert torch.allclose(_lp(bb2), after, atol=1e-6)


def _lp_grad(bb):
    v, m = bb.token_logprobs_from_ids(P, _ids(bb))
    return (v * m).sum()


def test_expert_adapters_route_learn_checkpoint_and_consolidate(tmp_path):
    from agentrs.training.common import load_weights, save_checkpoint, trainable_state

    bb = _bb(4)
    _perturb(bb)
    base = _lp(bb)
    ex = ExpertAdapters(bb, bb.model, keys=["finance", "support"], n_experts=2, r=4)
    rep = param_report(bb)
    assert rep["trainable_by_kind"].get("router") == 3 * 2 and ex.expert_params() > 0
    with ex.route(["finance", "support"]):
        assert torch.allclose(_lp(bb), base, atol=1e-6)  # fresh experts (B = 0) do not change behaviour
    params = [p for p in bb.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-3)
    for _ in range(25):  # learn something only for the finance example
        with ex.route(["finance"]):
            v, m = bb.token_logprobs_from_ids(P[:1], _ids(bb)[:1])
            loss = -(v * m).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    g = ex.router(["finance", "support", None]).detach()
    assert not torch.allclose(g[0], g[1])               # routing was learned per domain
    with ex.route(["finance", "support"]):
        routed = _lp(bb)
    assert "capacity_router.logits" in trainable_state(bb)  # router + experts are in checkpoints
    assert any(".expert0." in k for k in trainable_state(bb))
    save_checkpoint(tmp_path / "ck", bb)
    bb2 = _bb(4)
    ex2 = ExpertAdapters(bb2, bb2.model, keys=["finance", "support"], n_experts=2, r=4)
    load_weights(tmp_path / "ck", bb2)
    with ex2.route(["finance", "support"]):
        assert torch.allclose(_lp(bb2), routed, atol=1e-5)
    # consolidation: distil the routed model into the shared adapter, then retire an expert
    ids = _ids(bb)
    kl = distill_into_shared(bb, ex, P * 4, ["finance", "support"] * 4, ids * 4, steps=30, lr=5e-3)
    assert kl[-1] < kl[0]
    ex.retire("expert1")
    assert all(not p.requires_grad for n, p in bb.named_parameters() if ".expert1." in n)


def test_advisor_proposes_only_on_evidence():
    def ev(cycle, rate, fam_rates):
        return {"split": "val", "mode": "direct", "cycle": cycle, "train_families": {"rate": rate},
                "summary": {"by_family": {k: {"rate": r} for k, r in fam_rates.items()}}}

    cyc = [{"success_rate": 0.6 + 0.01 * i, "ppo": {"policy_lora_change_l2": 0.1}} for i in range(15)]
    improving = {"evals": [ev(i, 0.5 + 0.05 * i, {"a/x": 0.5}) for i in range(6)], "cycles": cyc}
    plateau = {"evals": [ev(i, 0.7 + (0.001 * i if i > 1 else 0), {"a/x": 0.5}) for i in range(6)], "cycles": cyc}
    a = ExpansionAdvisor()
    assert not a.assess(improving)["propose"]
    r = a.assess(plateau)
    assert r["propose"] and r["suggestion"]["mechanism"] == "expand_rank"
    interf = {"evals": [ev(0, 0.6, {"crm/a": 1.0, "crm/b": 1.0}), ev(1, 0.6, {"crm/a": 1.0, "crm/b": 1.0}),
                        ev(2, 0.62, {"crm/a": 1.0, "crm/b": 1.0}), ev(3, 0.66, {"crm/a": 1.0, "crm/b": 1.0}),
                        ev(4, 0.7, {"crm/a": 0.5, "crm/b": 0.5})], "cycles": cyc}
    r = a.assess(interf)
    assert r["propose"] and r["suggestion"]["mechanism"] == "experts"
    assert not a.assess({"evals": [], "cycles": []})["propose"]
