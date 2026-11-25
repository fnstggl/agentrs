"""Acting throughput: merged-LoRA decoding with learner-path behaviour log-probabilities (contract v3 decision 12).

`QwenSemantic.acting()` decodes with the LoRA adapter disabled and every adapted linear layer's weight replaced by
W + scale * B A for the duration; `ActionGenerator.sample` always reports the log-probability recomputed on the
learner path (LoRA unmerged, exact rows), so PPO ratios still start at exactly 1."""
from __future__ import annotations

import pytest
import torch

from superlearner.cam_unified.interfaces import EntityRegistry
from superlearner.cam_unified.semantic.actions import ActionGenerator
from superlearner.cam_unified.semantic.grammar import GrammarOptions
from superlearner.cam_unified.testing import sample_observations, sample_tools

from sem_helpers import make_context


@pytest.fixture()
def trained_lora(enc):
    """Give the session encoder's LoRA a non-zero delta (as after training); restore the zero init afterwards."""
    layers = enc._lora_layers()
    saved = [B.detach().clone() for _, _, B, _ in layers]
    g = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        for _, _, B, _ in layers:
            B.copy_((torch.randn(B.shape, generator=g) * 0.02).to(B.device, B.dtype))
    enc.mark_updated()
    yield layers
    with torch.no_grad():
        for (_, _, B, _), s in zip(layers, saved):
            B.copy_(s)
    enc.mark_updated()


def _rows(tok, cfg, B, seed):
    tool_lists = [sample_tools(5) for _ in range(B)]
    obs_lists = [sample_observations(3 + b % 4, seed=seed + b) for b in range(B)]
    R, goal, belief = make_context(cfg, B, seed=seed)
    tools = tok.tools(tool_lists)
    obs = tok.observations(obs_lists, [EntityRegistry(cfg.max_entities, seed=b) for b in range(B)], [0.0] * B)
    return R, goal, belief, tools, obs, tool_lists, obs_lists


def test_merged_weights_are_exact_swapped_only_while_acting_and_rebuilt_after_updates(enc, trained_lora):
    layers = trained_lora
    m, A, Bw, scale = layers[0]
    base = m.get_base_layer()
    W0 = base.weight.detach().clone()
    merged = dict((id(b), w) for b, w in enc.merged_weights())
    ref = (W0.float() + scale * (Bw.float() @ A.float())).to(W0.dtype)
    assert torch.equal(merged[id(base)], ref) and len(merged) == len(layers)
    h0 = enc.hidden(["The invoice total is 1234 cents."])[0].clone()
    enc._cache.clear()
    with enc.acting():
        assert torch.equal(base.weight, ref) and all(l.disable_adapters for l, *_ in layers)
    assert torch.equal(base.weight, W0) and not any(l.disable_adapters for l, *_ in layers)
    h1 = enc.hidden(["The invoice total is 1234 cents."])[0]
    assert torch.equal(h0, h1)                       # the encoder path (frozen base) is untouched by acting
    # an in-place optimiser-style update invalidates the merged cache without an explicit mark_updated()
    builds = enc.stats.get("merged_builds", 0)
    enc.merged_weights()
    assert enc.stats["merged_builds"] == builds      # cached
    with torch.no_grad():
        Bw.add_(0.01)
    new = dict((id(b), w) for b, w in enc.merged_weights())
    assert enc.stats["merged_builds"] == builds + 1
    assert torch.equal(new[id(base)], (W0.float() + scale * (Bw.float() @ A.float())).to(W0.dtype))


def test_acting_logits_match_the_lora_path(enc, trained_lora):
    """Decoder logits on the merged acting path equal the LoRA path up to the bf16 rounding of the merged weight."""
    ids = torch.tensor([enc.tok("Write the JSON arguments for one call of the tool.")["input_ids"]],
                       device=enc.device)
    att = torch.ones_like(ids)
    pos = torch.arange(ids.shape[1], device=enc.device).unsqueeze(0)
    emb = enc.embed_ids(ids)
    with torch.no_grad():
        lo_lora, _ = enc.step(emb, att, pos, enc.new_cache())
        with enc.acting():
            lo_act, _ = enc.step(emb, att, pos, enc.new_cache())
        with enc.peft.disable_adapter():
            lo_base, _ = enc.step(emb, att, pos, enc.new_cache())
    lp = lambda x: torch.log_softmax(x.float(), -1)
    d_act = (lp(lo_act) - lp(lo_lora)).abs()
    d_base = (lp(lo_base) - lp(lo_lora)).abs()
    top = lp(lo_lora).topk(50, -1).indices
    # the trained delta matters (base differs) while the merged path tracks the LoRA path closely on likely tokens
    assert float(d_base.gather(1, top).mean()) > 4 * float(d_act.gather(1, top).mean())
    assert float(d_act.gather(1, top).max()) < 0.25


def test_sampling_with_merged_lora_reports_the_learner_path_logp(enc, tok, cfg, trained_lora):
    """Parity (contract v3 #12): with a non-zero LoRA delta, actions sampled on the merged acting path carry exactly
    the learner-path log-probability (bit-equal to a grad-enabled log_prob in a permuted half batch), and the
    incremental merged value stays within the documented bf16 drift of it."""
    torch.manual_seed(3)
    gen = ActionGenerator(cfg, enc, tok, opts=GrammarOptions(max_string_tokens=40, max_action_tokens=160,
                                                             max_key_tokens=16))
    assert gen.filler.merged_acting
    R, goal, belief, tools, obs, tool_lists, obs_lists = _rows(tok, cfg, 6, seed=21)
    names = ["search_web" if b % 2 else "send_message" for b in range(6)]
    T = tools.N + 2
    mask = torch.zeros(6, T, dtype=torch.bool)
    for b, n in enumerate(names):
        mask[b, [t.name for t in tool_lists[b]].index(n)] = True
    builds = enc.stats.get("merged_builds", 0)
    acts = gen.sample(R, goal, belief, tools, obs, tool_lists, obs_lists, tool_mask=mask,
                      generator=torch.Generator(device=enc.device).manual_seed(5))
    assert enc.stats["merged_builds"] >= builds
    assert sum(a.logp_parts["n_tokens"] for a in acts) > 20
    perm = [4, 1, 5]
    sel = lambda ts: ts.select(torch.tensor(perm, device=enc.device))
    lp, _ = gen.log_prob(R[perm].requires_grad_(True), sel(goal), sel(belief), sel(tools), sel(obs),
                         [tool_lists[i] for i in perm], [obs_lists[i] for i in perm], [acts[i] for i in perm])
    assert [float(v) for v in lp] == [acts[i].logp for i in perm]
    for a in acts:
        assert a.logp_parts["rescored"] and ActionGenerator.importance_weight_usable(a)
        assert abs(a.logp - a.logp_parts["sample_logp"]) < 1.0 + 0.02 * a.logp_parts["n_tokens"]
    # the unmerged acting path gives the same learner-path values for the same recorded actions
    gen.filler.merged_acting = False
    lp2, _ = gen.log_prob(R, goal, belief, tools, obs, tool_lists, obs_lists, acts)
    assert [float(v) for v in lp2] == [a.logp for a in acts]
