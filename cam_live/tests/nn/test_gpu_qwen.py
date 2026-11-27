"""The real configuration: Qwen/Qwen3-4B + LoRA + bridge + the 50M experiential network (NA-01..04, NA-09, NA-12).

Run with ``-m gpu``. The module takes the shared GPU lock once for all its tests (they share one loaded model) and
releases memory afterwards.
"""
import time

import pytest
import torch

from cam_live.config import CamLiveConfig
from cam_live.contracts import ExperienceItem, ItemKind, Quantity
from cam_live.nn.api import GROUP_ADAPTERS, GROUP_BACKBONE_FROZEN, GROUP_EXPERIENTIAL
from cam_live.nn.gpu import GpuUnavailable, free_gpu_memory, gpu_lock
from cam_live.nn.model import build_model

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def qwen():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        with gpu_lock(timeout_s=1800.0, min_free_gb=14.0):
            torch.cuda.reset_peak_memory_stats()
            model = build_model(CamLiveConfig(device="cuda"))
            t0 = time.time()
            yield model
            assert time.time() - t0 < 900, "held the shared GPU lock for more than 15 minutes"
            assert torch.cuda.max_memory_allocated() < 24 * 2 ** 30
            del model
            free_gpu_memory()
    except GpuUnavailable as e:
        pytest.skip(str(e))


def _frozen_untouched(model):
    frozen = model.param_groups()[GROUP_BACKBONE_FROZEN]
    assert all(not p.requires_grad for p in frozen) and all(p.grad is None for p in frozen)


def test_parameter_counts_of_the_real_configuration(qwen):
    c = qwen.count_parameters()
    assert qwen.backbone.name == "Qwen/Qwen3-4B" and qwen.backbone.pretrained
    assert qwen.backbone.revision == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert 3.9e9 < c["frozen"] < 4.1e9, c["frozen"]                       # a 4B-8B pretrained backbone
    assert 10e6 <= c["adapters"] <= 20e6, c["adapters"]                   # 10-20M semantic adapters
    assert c["adapters_lora"] == 11_796_480 and len(qwen.backbone.lora_modules) == 72     # q/v of all 36 layers
    assert 45e6 < c["experiential"] < 55e6, c["experiential"]             # ~50M experiential
    assert 60e6 < c["total_persistent_trainable"] < 70e6
    assert c["scale"] == "50m" and qwen.d_sem == 3 * 2560 and qwen.d_model == 768
    groups = qwen.param_groups()
    ids = [id(p) for ps in groups.values() for p in ps]
    assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in qwen.parameters()}
    assert next(iter(groups[GROUP_BACKBONE_FROZEN])).dtype == torch.bfloat16
    assert all(p.dtype == torch.float32 for p in groups[GROUP_ADAPTERS] + groups[GROUP_EXPERIENTIAL])
    _frozen_untouched(qwen)


def test_semantic_embeddings_are_meaningful_and_cached(qwen):
    texts = ["The page loaded without errors.", "The page loaded with no errors.", "def f(x):\n    return x + 1",
             "HTTP 404 Not Found", "Click the submit button."]
    e = qwen.embed_texts(texts)
    assert e.shape == (5, 7680) and e.dtype == torch.float32 and torch.isfinite(e).all()
    cos = torch.nn.functional.cosine_similarity(e[0], e[1:], -1)
    assert float(cos[0]) > 0.8 and float(cos[1:].max()) < 0.7            # a paraphrase is close, other texts are not
    hits = qwen.backbone.stats["cache_hits"]
    assert torch.equal(qwen.embed_texts(texts), e) and qwen.backbone.stats["cache_hits"] == hits + 5


def test_bridge_is_an_exact_noop_at_init_on_the_real_backbone(qwen):
    prompts = [qwen.backbone.chat_prompt("Name one colour."), "The capital of France is"]
    lat = torch.randn(2, qwen.d_model, device=qwen.device, generator=torch.Generator(qwen.device).manual_seed(0))
    with torch.no_grad():
        base = qwen.backbone.next_token_logits(prompts)
        cond = qwen.backbone.next_token_logits(prompts, latent=lat)
        lp0 = qwen.generation_logprob(prompts, ["Blue", " Paris"])
        lp1 = qwen.generation_logprob(prompts, ["Blue", " Paris"], latent=lat)
    assert torch.equal(base, cond) and torch.equal(lp0, lp1)
    assert float(lp0[1]) > -2.0                                           # the pretrained model knows the answer
    assert qwen.generate(prompts, max_new_tokens=8, temperature=0.0) == \
        qwen.generate(prompts, latent=lat, max_new_tokens=8, temperature=0.0)


def test_generate_returns_text(qwen):
    p = qwen.backbone.chat_prompt("Reply with exactly one word: hello")
    out = qwen.generate([p], max_new_tokens=12, temperature=0.0)
    assert isinstance(out[0][0], str) and "hello" in out[0][0].lower()
    lat = torch.randn(1, qwen.d_model, device=qwen.device)
    samples = qwen.generate([p], latent=lat, max_new_tokens=12, temperature=0.9, num_return_sequences=3, seed=1)
    assert len(samples) == 1 and len(samples[0]) == 3 and all(isinstance(s, str) and s for s in samples[0])
    assert samples == qwen.generate([p], latent=lat, max_new_tokens=12, temperature=0.9, num_return_sequences=3, seed=1)


def test_adapters_get_gradient_through_both_paths_and_the_bridge_conditions(qwen):
    """Runs last: it takes one optimisation step on the adapters."""
    bb = qwen.backbone
    lora = [(m.lora_A, m.lora_B) for m in bb.lora_modules]
    # (1) embedding path through a downstream loss: tokens -> experiential core -> belief
    ctx = [[ExperienceItem(kind=ItemKind.OBSERVATION, text="The form was submitted and a confirmation appeared.",
                           ts=10.0, quantities=[Quantity("visits", 100.0)]),
            ExperienceItem(kind=ItemKind.OUTCOME, text="three sign-ups were recorded", ts=3600.0,
                           quantities=[Quantity("signups", 3.0)])]]
    tb = qwen.tokenize(ctx, with_grad=True)
    assert tb.values[0, :, 0].tolist() == [100.0, 3.0]
    b = qwen.core_forward(tb).b
    qwen.zero_grad()
    b.pow(2).sum().backward()
    n_b = sum(float(B.grad.abs().sum()) > 0 for _, B in lora)
    assert n_b == len(lora), f"{n_b}/{len(lora)} LoRA modules got gradient through the embedding path"
    g_embed = qwen.grad_norms()
    assert g_embed[GROUP_ADAPTERS] > 0 and g_embed[GROUP_EXPERIENTIAL] > 0
    _frozen_untouched(qwen)
    # (2) generation path
    qwen.zero_grad()
    prompts = [bb.chat_prompt("Name one colour.")]
    lat = torch.randn(1, qwen.d_model, device=qwen.device, generator=torch.Generator(qwen.device).manual_seed(1))
    lat.requires_grad_(True)
    with torch.no_grad():
        p_before = torch.softmax(bb.next_token_logits(prompts, latent=lat), -1)
        p_base_before = torch.softmax(bb.next_token_logits(prompts), -1)
    lp = qwen.generation_logprob(prompts, ["Green"], latent=lat)
    (-lp.sum()).backward()
    assert all(float(B.grad.abs().sum()) > 0 for _, B in lora)
    assert all(float(x.out.weight.grad.abs().sum()) > 0 for x in bb.bridge.xattn)
    _frozen_untouched(qwen)
    # (3) one optimisation step: the latent now changes generation probabilities, and gets gradient itself
    opt = torch.optim.Adam(qwen.param_groups()[GROUP_ADAPTERS], lr=1e-3)
    opt.step()
    qwen.zero_grad()
    with torch.no_grad():
        p_after = torch.softmax(bb.next_token_logits(prompts, latent=lat), -1)
        p_base_after = torch.softmax(bb.next_token_logits(prompts), -1)
        other = torch.softmax(bb.next_token_logits(prompts, latent=-lat.detach()), -1)
    assert float((p_after - p_before).abs().max()) > 1e-4
    assert float((p_after - p_base_after).abs().max()) > 1e-5          # conditioned differs from unconditioned
    assert float((p_after - other).abs().max()) > 1e-5                 # and different beliefs differ
    lp2 = qwen.generation_logprob(prompts, ["Green"], latent=lat)
    (-lp2.sum()).backward()
    assert float(lat.grad.abs().sum()) > 0
    assert all(float(A.grad.abs().sum()) > 0 for A, _ in lora)
    assert float(lp2.detach()) > float(lp.detach())
    _frozen_untouched(qwen)
    assert not bb.lm.training
