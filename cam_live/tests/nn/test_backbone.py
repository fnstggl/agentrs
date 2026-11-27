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
    assert float(lp2.detach().sum()) > float(lp.detach().sum())                            # the step raised the likelihood


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
    assert torch.equal(eg.detach(), e1)          # the graph path returns exactly the cached numbers (straight-through)
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


def _set_B(model, seed, scale=0.5, via_data=False):
    g = torch.Generator().manual_seed(seed)
    for m in model.backbone.lora_modules:
        new = torch.randn(m.lora_B.shape, generator=g) * scale
        if via_data:
            m.lora_B.data.copy_(new)                 # bumps no version counter
        else:
            with torch.no_grad():
                m.lora_B.copy_(new)


@pytest.mark.parametrize("how", ["data", "inplace", "sgd", "adam_foreach", "adam_fused", "load_state", "calibration"])
def test_caches_are_invalidated_by_any_change_of_what_they_depend_on(how):
    """The embedding cache and the LoRA inference casts follow the adapter *content*: a write through ``.data``
    (no version bump) left both stale before (cached embedding off by 22, no-grad logits off by 5.9)."""
    model = build_model(CamLiveConfig.debug())
    bb = model.backbone
    if how != "calibration":
        _set_B(model, 0, 0.3)                       # non-zero adapters so the inference path caches real casts
    e0 = model.embed_texts(TEXTS).clone()
    with torch.no_grad():
        l0 = bb.next_token_logits(["the page"]).clone()
    v0 = bb.adapter_version
    if how == "data":
        _set_B(model, 1, via_data=True)
    elif how == "inplace":
        _set_B(model, 1)
    elif how in ("sgd", "adam_foreach", "adam_fused"):
        params = bb.lora_parameters()
        opt = {"sgd": lambda: torch.optim.SGD(params, lr=1.0),
               "adam_foreach": lambda: torch.optim.Adam(params, lr=0.1, foreach=True),
               "adam_fused": lambda: torch.optim.Adam(params, lr=0.1, fused=True)}[how]()
        opt.zero_grad()
        model.embed_texts(TEXTS, with_grad=True).pow(2).sum().backward()
        opt.step()
    elif how == "load_state":
        other = build_model(CamLiveConfig.debug())
        _set_B(other, 2)
        sd = {k: v for k, v in other.backbone.state_dict().items() if "lora_" in k}
        bb.load_state_dict(sd, strict=False)        # plain nn.Module API, no mark_adapters_updated()
    else:
        bb.cal_mu.data.add_(0.5)                    # the standardisation statistics of cached embeddings
    e1 = model.embed_texts(TEXTS)
    fresh = model.embed_texts(TEXTS, with_grad=True).detach()
    with torch.no_grad():
        l_ng = bb.next_token_logits(["the page"])
    l_g = bb.next_token_logits(["the page"]).detach()          # grad mode: live float32 adapters, no cast cache
    assert bb.adapter_version > v0
    assert not torch.equal(e1, e0) and torch.equal(e1, fresh)
    assert torch.equal(l_ng, l_g), float((l_ng - l_g).abs().max())
    if how != "calibration":
        assert not torch.allclose(l_ng, l0, atol=1e-4)
    # no change -> no invalidation, cache hits
    hits = bb.stats["cache_hits"]
    v1 = bb.adapter_version
    model.embed_texts(TEXTS)
    assert bb.adapter_version == v1 and bb.stats["cache_hits"] == hits + len(TEXTS)
    # mark_adapters_updated() clears the LoRA inference casts too
    model.mark_adapters_updated()
    assert all(m._cast_key is None for m in bb.lora_modules)


def test_embedding_is_independent_of_the_other_texts_in_the_call(model):
    """Bit-exact: a text's embedding (and so the cache, and b) does not depend on batch composition or order. With
    batch-dependent shapes the bf16 backbone gave ~1 % relative differences and the cache kept whichever came
    first."""
    bb = model.backbone
    t = "the export job finished after twelve minutes"
    long1 = " ".join(f"word{i}" for i in range(150))
    long2 = " ".join(f"other{i}" for i in range(37))
    bb.mark_adapters_updated()
    alone = model.embed_texts([t])[0].clone()
    for others in ([long1, long2], [long2], [long1] * 3 + [long2] * 30, [f"{t} {i}" for i in range(40)]):
        bb.mark_adapters_updated()
        batch = model.embed_texts(others[: len(others) // 2] + [t] + others[len(others) // 2:])
        assert torch.equal(batch[len(others) // 2], alone)
    assert torch.equal(model.embed_texts([t], with_grad=True).detach()[0], alone)
    # through the core: b of a context does not depend on the other contexts in the batch (cold cache)
    ctx = [ExperienceItem(kind=ItemKind.OBSERVATION, text=t, ts=1.0), ExperienceItem(kind=ItemKind.ACTION, text=long2, ts=5.0)]
    bb.mark_adapters_updated()
    b_alone = model.core_forward(model.tokenize([ctx])).b[0].clone()
    bb.mark_adapters_updated()
    other = [ExperienceItem(kind=ItemKind.DOCUMENT, text=long1, ts=2.0)] * 3
    b_batch = model.core_forward(model.tokenize([other, ctx])).b[1]
    assert torch.allclose(b_alone, b_batch, atol=1e-6)         # the core itself is batch-invariant to float32 rounding
    # every forward has the fixed shape of its bucket
    for n in (1, 7, 8, 9, 64, 65, 129, 384):
        Lb = bb.embed_bucket(n)
        assert Lb >= n and Lb - n < max(8, n // 4) and bb.embed_rows(Lb) * Lb <= max(Lb, bb.nn_cfg.embed_batch_tokens)


def test_float32_head_matches_a_float64_reference_with_its_gradient():
    """The output layer used by generation_logprob / the sampler: float32 logits and log-probabilities (no bf16
    rounding of logits), chunked over the vocabulary, gradient w.r.t. the hidden state; the weight is used in its
    stored dtype."""
    from cam_live.nn import backbone as bbm
    g = torch.Generator().manual_seed(0)
    V, H, P = 70001, 48, 9                         # V spans three chunks
    w = (torch.randn(V, H, generator=g) * 0.4).to(torch.bfloat16)
    bias = torch.randn(V, generator=g) * 0.1
    h = (torch.randn(P, H, generator=g) * 3).requires_grad_(True)
    tgt = torch.randint(0, V, (P,), generator=g)
    tgt[0], tgt[1] = 0, V - 1
    ref_logits = h.detach().double() @ w.double().T + bias.double()
    ref = torch.log_softmax(ref_logits, -1).gather(1, tgt[:, None]).squeeze(1)
    lg = bbm._HeadLogits.apply(h, w, bias)
    lp = bbm._TokenLogprob.apply(h, w, bias, tgt)
    assert lg.dtype == torch.float32 and lp.dtype == torch.float32
    assert float((lg.double() - ref_logits).abs().max()) < 2e-4
    assert float((lp.double() - ref).abs().max()) < 2e-4
    # gradients against autograd in float64
    hd = h.detach().double().requires_grad_(True)
    torch.log_softmax(hd @ w.double().T + bias.double(), -1).gather(1, tgt[:, None]).sum().backward()
    lp.sum().backward()
    assert torch.allclose(h.grad.double(), hd.grad, atol=1e-4, rtol=1e-4)
    h.grad = None
    probe = torch.randn(P, V, generator=g)
    (lg * probe).sum().backward()
    assert torch.allclose(h.grad.double(), probe.double() @ w.double(), atol=1e-3, rtol=1e-4)
