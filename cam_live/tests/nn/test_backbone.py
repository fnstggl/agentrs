"""Backbone wrapper on the tiny random LM: LoRA gradient paths, frozen weights, latent bridge, cache, generation
(NA-09, NA-12). The same code path runs on Qwen3-4B in test_gpu_qwen.py."""
import copy

import pytest
import torch

from cam_live.config import CamLiveConfig
from cam_live.contracts import ExperienceItem, ItemKind
from cam_live.nn.api import GROUP_ADAPTERS, GROUP_BACKBONE_FROZEN
from cam_live.nn.model import build_model

TEXTS = ["the page loaded", "click the button", "an error occurred while saving", "42 items"]


def _lora(model):
    return [(m.lora_A, m.lora_B) for m in model.backbone.lora_modules]


def _assert_frozen_untouched(model):
    frozen = model.param_groups()[GROUP_BACKBONE_FROZEN]
    assert frozen and all(not p.requires_grad for p in frozen) and all(p.grad is None for p in frozen)


def test_lora_gets_gradient_from_the_embedding_path(model):
    emb = model.embed_texts(TEXTS, with_grad=True)
    assert emb.requires_grad and emb.shape == (4, model.d_sem) and emb.dtype == torch.float32
    assert not model.embed_texts(TEXTS).requires_grad
    # through a downstream loss: tokens -> core -> belief
    ctx = [[ExperienceItem(kind=ItemKind.OBSERVATION, text=t, ts=float(i)) for i, t in enumerate(TEXTS)]]
    b = model.core_forward(model.tokenize(ctx, with_grad=True)).b
    probe = torch.randn(b.shape, generator=torch.Generator().manual_seed(0))
    model.zero_grad()
    (b * probe).sum().backward()
    for A, B in _lora(model):
        assert B.grad is not None and float(B.grad.abs().sum()) > 0        # B = 0 at init, so only B moves first
    _assert_frozen_untouched(model)
    assert model.grad_norms()[GROUP_ADAPTERS] > 0 and model.grad_norms()["experiential"] > 0
    # after one step B != 0 and A receives gradient as well
    opt = torch.optim.SGD(model.backbone.lora_parameters(), lr=0.5)
    opt.step()
    model.zero_grad()
    b = model.core_forward(model.tokenize(ctx, with_grad=True)).b
    (b * probe).sum().backward()
    for A, B in _lora(model):
        assert float(A.grad.abs().sum()) > 0 and float(B.grad.abs().sum()) > 0
    _assert_frozen_untouched(model)
    # grad_last restricts the adapter graph to the newest items but keeps it alive
    model.zero_grad()
    b = model.core_forward(model.tokenize(ctx, with_grad=True, grad_last=1)).b
    (b * probe).sum().backward()
    assert model.grad_norms()[GROUP_ADAPTERS] > 0


def test_lora_and_bridge_get_gradient_from_generation_logprob(model):
    prompts, comps = ["the page", "click"], ["loaded fine today", "the button now"]
    latent = torch.randn(2, model.d_model, generator=torch.Generator().manual_seed(1), requires_grad=True)
    lp = model.generation_logprob(prompts, comps, latent=latent)
    assert lp.shape == (2,) and lp.dtype == torch.float32 and bool((lp < 0).all())
    model.zero_grad()
    (-lp.sum()).backward()
    for A, B in _lora(model):
        assert float(B.grad.abs().sum()) > 0
    for x in model.backbone.bridge.xattn:
        assert float(x.out.weight.grad.abs().sum()) > 0                    # the zero-initialised bridge output learns
    _assert_frozen_untouched(model)
    assert float(latent.grad.abs().sum()) == 0.0                           # exact no-op at init: no dependence yet
    opt = torch.optim.SGD(model.param_groups()[GROUP_ADAPTERS], lr=0.05)
    opt.step()
    model.zero_grad()
    latent.grad = None
    lp2 = model.generation_logprob(prompts, comps, latent=latent)
    (-lp2.sum()).backward()
    for A, B in _lora(model):
        assert float(A.grad.abs().sum()) > 0 and float(B.grad.abs().sum()) > 0
    for p in model.backbone.bridge.parameters():
        assert p.grad is not None and float(p.grad.abs().sum()) > 0, "every bridge tensor trains after the first step"
    assert float(latent.grad.abs().sum()) > 0                              # differentiable w.r.t. CAM's latent
    _assert_frozen_untouched(model)
    assert float(lp2.sum()) > float(lp.sum())                              # the step raised the likelihood


def test_bridge_is_an_exact_noop_at_init_and_conditions_after_one_step(model):
    bb = model.backbone
    prompts = ["the page loaded", "click the button and wait"]
    g = torch.Generator().manual_seed(2)
    lat = torch.randn(2, model.d_model, generator=g)
    with torch.no_grad():
        base = bb.next_token_logits(prompts)
        with_lat = bb.next_token_logits(prompts, latent=lat)
        assert torch.equal(base, with_lat)                                 # identical logits, bit for bit
        comps = ["with no errors", "for it"]
        assert torch.equal(model.generation_logprob(prompts, comps),
                           model.generation_logprob(prompts, comps, latent=lat))
    assert model.generate(prompts, max_new_tokens=6, seed=3) == model.generate(prompts, latent=lat, max_new_tokens=6,
                                                                               seed=3)
    # one optimisation step on the adapters
    opt = torch.optim.Adam(model.param_groups()[GROUP_ADAPTERS], lr=1e-2)
    loss = -model.generation_logprob(prompts, comps, latent=lat).sum()
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        base2 = bb.next_token_logits(prompts)
        with2 = bb.next_token_logits(prompts, latent=lat)
        other = bb.next_token_logits(prompts, latent=torch.randn(2, model.d_model, generator=g))
        p_base, p_with, p_other = (torch.softmax(x, -1) for x in (base2, with2, other))
    assert float((p_with - p_base).abs().max()) > 1e-4                     # the latent now changes generation
    assert float((p_with - p_other).abs().max()) > 1e-5                    # and different beliefs differ
    assert float((torch.softmax(base, -1) - p_with).abs().max()) > 1e-4


def test_embedding_cache_bypass_and_invalidation(model):
    bb = model.backbone
    e1 = model.embed_texts(TEXTS)
    hits0, miss0 = bb.stats["cache_hits"], bb.stats["cache_misses"]
    e2 = model.embed_texts(TEXTS + TEXTS[:1])
    assert torch.equal(e1, e2[:4]) and torch.equal(e2[0], e2[4])
    assert bb.stats["cache_hits"] == hits0 + 5 and bb.stats["cache_misses"] == miss0
    eg = model.embed_texts(TEXTS, with_grad=True)                          # bypasses the cache
    assert bb.stats["cache_hits"] == hits0 + 5 and eg.requires_grad
    assert torch.allclose(eg.detach(), e1, atol=2e-2, rtol=2e-2)           # same values up to float16 cache storage
    assert float(torch.nn.functional.cosine_similarity(e1[0], e1[1], 0)) < 0.95   # texts are distinguishable
    # an optimizer step on the adapters invalidates the cache automatically
    v0 = bb.adapter_version
    opt = torch.optim.SGD(bb.lora_parameters(), lr=1.0)
    opt.zero_grad()
    eg.pow(2).sum().backward()
    opt.step()
    e3 = model.embed_texts(TEXTS)
    assert bb.adapter_version == v0 + 1 and bb.stats["cache_misses"] == miss0 + 4
    assert not torch.allclose(e3, e1, atol=1e-3)
    assert torch.equal(model.embed_texts(TEXTS), e3)
    assert model.embed_texts([]).shape == (0, model.d_sem)
    # long texts: head and tail are kept, the cut is counted
    t0 = bb.stats["truncated_tokens"]
    long_a = " ".join(f"w{i}" for i in range(400)) + " ending one"
    long_b = " ".join(f"w{i}" for i in range(400)) + " ending two"
    ea = model.embed_texts([long_a, long_b])
    assert bb.stats["truncated_tokens"] > t0 and not torch.equal(ea[0], ea[1])


def test_gradient_checkpointing_gives_the_same_gradients():
    grads = []
    for ckpt in (True, False):
        cfg = CamLiveConfig.debug()
        cfg.backbone.grad_checkpoint = ckpt
        m = build_model(cfg)
        with torch.no_grad():                                  # make the adapters non-trivial first
            for mod in m.backbone.lora_modules:
                mod.lora_B.copy_(torch.randn(mod.lora_B.shape, generator=torch.Generator().manual_seed(5)) * 0.1)
            for x in m.backbone.bridge.xattn:
                x.out.weight.copy_(torch.randn(x.out.weight.shape, generator=torch.Generator().manual_seed(6)) * 0.1)
        lat = torch.randn(2, m.d_model, generator=torch.Generator().manual_seed(7))
        loss = -m.generation_logprob(["a b c", "d e"], ["f g", "h i j"], latent=lat).sum() \
            + m.embed_texts(TEXTS, with_grad=True).pow(2).sum()
        loss.backward()
        grads.append([p.grad.clone() for p in m.param_groups()[GROUP_ADAPTERS]])
        assert not m.backbone.lm.training                      # the frozen LM is back in eval mode
    assert all(torch.allclose(a, b, atol=1e-6, rtol=1e-4) for a, b in zip(*grads))
    assert sum(float(g.abs().sum()) for g in grads[0]) > 0


def test_generate_is_seeded_and_leaves_global_rng_alone(model):
    torch.manual_seed(123)
    expect = torch.rand(3)
    torch.manual_seed(123)
    out = model.generate(["start here", "another prompt that is longer"], max_new_tokens=8, num_return_sequences=3,
                         seed=11)
    assert torch.equal(torch.rand(3), expect)
    assert len(out) == 2 and all(len(o) == 3 and all(isinstance(t, str) for t in o) for o in out)
    assert out == model.generate(["start here", "another prompt that is longer"], max_new_tokens=8,
                                 num_return_sequences=3, seed=11)
    assert out != model.generate(["start here", "another prompt that is longer"], max_new_tokens=8,
                                 num_return_sequences=3, seed=12)
    assert len({*out[0]}) > 1                                  # samples differ within a prompt
    greedy = model.generate(["start here"], max_new_tokens=5, temperature=0.0)
    assert greedy == model.generate(["start here"], max_new_tokens=5, temperature=0.0)
    # batch generation with left padding equals single-prompt generation (greedy)
    both = model.generate(["start here", "another prompt that is longer"], max_new_tokens=5, temperature=0.0)
    assert both[0] == greedy[0]
    # the log-probability of a completion is consistent with the sampler's next-token distribution
    lp = model.generation_logprob(["start here"], ["start"])
    ids = model.backbone._encode("start")
    ref = torch.log_softmax(model.backbone.next_token_logits(["start here"]), -1)[0, ids[0]]
    assert torch.allclose(lp[0], ref, atol=1e-5)
    assert float(model.generation_logprob(["x"], [""])[0]) == 0.0


def test_tiny_backbone_is_labelled_and_deterministic(cfg):
    a, b = build_model(cfg), build_model(copy.deepcopy(cfg))
    assert a.backbone.name == "tiny" and a.backbone.pretrained is False
    assert a.arch_signature()["backbone"]["pretrained"] is False
    assert torch.equal(a.embed_texts(TEXTS), b.embed_texts(TEXTS))
    assert a.group_hashes() == b.group_hashes()
    with pytest.raises(ValueError):
        a.generate(["x", "y"], latent=torch.zeros(1, a.d_model))
