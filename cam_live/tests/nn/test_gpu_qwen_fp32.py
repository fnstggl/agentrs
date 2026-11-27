"""Decisive gradient and log-probability checks on Qwen/Qwen3-4B loaded in float32 (``-m gpu``).

In bf16 a finite difference is not decisive (rounding of the forward pass), so non-zero gradients were all the
real-backbone test could assert. Here the same code path runs with ``backbone.dtype = "float32"``: central finite
differences along the analytic gradient must equal its norm, for every gradient path the learner uses. The bf16
model's generation log-probabilities are compared with the float32 model's. One lock hold, both models in turn.
"""
import math
import time

import pytest
import torch

from cam_live.config import CamLiveConfig
from cam_live.contracts import ExperienceItem, ItemKind, Quantity
from cam_live.nn.api import GROUP_ADAPTERS
from cam_live.nn.gpu import GpuUnavailable, free_gpu_memory, gpu_lock
from cam_live.nn.model import build_model

pytestmark = pytest.mark.gpu

PROMPTS = ["Name one colour.", "Which tool should be used next? Answer with one word.",
           "Write a short Python function that adds two numbers."]
COMPS = ["Green", "browser", "def add(a, b):\n    return a + b"]
TEXTS = ["The form was submitted and a confirmation appeared.", "def mean(xs):\n    return sum(xs) / len(xs)",
         "HTTP 404 Not Found", "three sign-ups were recorded after the announcement"]
LP_PROMPTS_EXTRA = (["The capital of France is", "2 + 2 ="], [" Paris", " 4"])


def _probe(shape, device, seed):
    return torch.randn(shape, device=device, generator=torch.Generator(device).manual_seed(seed))


def _emb_loss(m):
    e = m.embed_texts(TEXTS, with_grad=True)
    return (e * _probe(e.shape, m.device, 5)).sum() / 100.0


def _gen_loss(m, latent):
    bb = m.backbone
    return -bb.generation_logprob([bb.chat_prompt(p) for p in PROMPTS], COMPS, latent=latent).sum()


def _logprobs(m):
    bb = m.backbone
    prompts = [bb.chat_prompt(p) for p in PROMPTS] + LP_PROMPTS_EXTRA[0]
    comps = COMPS + LP_PROMPTS_EXTRA[1]
    with torch.no_grad():
        lp = bb.generation_logprob(prompts, comps).double().cpu()
    return lp, torch.tensor([len(bb._encode(c)) for c in comps], dtype=torch.float64)


def _fd_ratio(params, grads, loss_fn, dl):
    """(L(p + e g/|g|) - L(p - e g/|g|)) / (2 e |g|) with e = dl / |g|: 1 when the gradient is right."""
    gn = math.sqrt(sum(float(g.double().pow(2).sum()) for g in grads))
    eps = dl / gn
    with torch.no_grad():
        for p, g in zip(params, grads):
            p.add_(g, alpha=eps / gn)
        lp = float(loss_fn().double())
        for p, g in zip(params, grads):
            p.add_(g, alpha=-2 * eps / gn)
        lm = float(loss_fn().double())
        for p, g in zip(params, grads):
            p.add_(g, alpha=eps / gn)
    return gn, (lp - lm) / (2 * eps) / gn


def _grads(m, loss_fn, params):
    m.zero_grad(set_to_none=True)
    loss_fn().backward()
    return [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]


def test_float32_backbone_gradients_and_bf16_logprob_accuracy():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        lock = gpu_lock(timeout_s=1800.0, min_free_gb=22.0)
        t_lock = lock.__enter__()
    except GpuUnavailable as e:
        pytest.skip(str(e))
    try:
        m16 = build_model(CamLiveConfig(device="cuda"))
        lp16, n_tok = _logprobs(m16)
        del m16
        free_gpu_memory()
        cfg = CamLiveConfig(device="cuda")
        cfg.backbone.dtype = "float32"
        m = build_model(cfg)
        bb = m.backbone
        assert next(bb.lm.parameters()).dtype == torch.float32
        lp32, _ = _logprobs(m)
        per_tok = ((lp16 - lp32).abs() / n_tok)
        print("bf16 - float32 generation log-probabilities", (lp16 - lp32).tolist(), "tokens", n_tok.tolist())
        # The output layer is float32 in both models, so this is the bf16 decoder trunk alone. Measured (per
        # completion, bf16 - float32): -0.159, -0.355, +0.123, -0.049, -0.002 nats for 1, 1, 11, 1, 2 tokens, about
        # what it was with the bf16 head (-0.21, -0.31, +0.13, -0.02, +0.01): the head was not the main error here.
        # A sanity bound only; the float32 head itself is pinned in test_gpu_qwen.py against a float64 reference.
        assert float(per_tok.max()) < 0.5 and float((lp16 - lp32).abs().sum()) < 1.5, per_tok.tolist()

        lora_B = [x.lora_B for x in bb.lora_modules]
        out_w = [x.out.weight for x in bb.bridge.xattn]
        lat = _probe((3, m.d_model), m.device, 21)
        ratios = {}
        # at initialisation: embedding loss -> LoRA-B; generation loss -> LoRA-B and -> the zero-initialised bridge
        g = _grads(m, lambda: _emb_loss(m), lora_B)
        ratios["embedding -> LoRA-B"] = _fd_ratio(lora_B, g, lambda: _emb_loss(m), 0.03)
        g = _grads(m, lambda: _gen_loss(m, lat), lora_B + out_w)
        ratios["generation -> LoRA-B"] = _fd_ratio(lora_B, g[:len(lora_B)], lambda: _gen_loss(m, lat), 0.03)
        ratios["generation -> bridge out (zero init)"] = _fd_ratio(out_w, g[len(lora_B):], lambda: _gen_loss(m, lat), 0.03)
        # away from initialisation: the latent, all adapters, and the experiential weights through b
        with torch.no_grad():
            gen = torch.Generator(m.device).manual_seed(11)
            for x in bb.lora_modules:
                x.lora_B.copy_(torch.randn(x.lora_B.shape, device=m.device, generator=gen) * 0.003)
            for x in bb.bridge.xattn:
                x.out.weight.copy_(torch.randn(x.out.weight.shape, device=m.device, generator=gen) * 0.02)
        lat_p = lat.clone().requires_grad_(True)
        ratios["generation -> latent"] = _fd_ratio([lat], _grads(m, lambda: _gen_loss(m, lat_p), [lat_p]),
                                                   lambda: _gen_loss(m, lat), 0.03)
        adapters = m.param_groups()[GROUP_ADAPTERS]
        both = lambda: _gen_loss(m, lat) + _emb_loss(m)
        ratios["generation + embedding -> all adapters"] = _fd_ratio(adapters, _grads(m, both, adapters), both, 0.03)
        ctx = [[ExperienceItem(kind=ItemKind.OBSERVATION, text=TEXTS[0], ts=10.0, quantities=[Quantity("visits", 100.0)]),
                ExperienceItem(kind=ItemKind.OUTCOME, text=TEXTS[3], ts=3600.0, quantities=[Quantity("signups", 3.0)])],
               [ExperienceItem(kind=ItemKind.FAILURE, text=TEXTS[2], ts=5.0)],
               [ExperienceItem(kind=ItemKind.SOFTWARE, text=TEXTS[1], ts=7.0)]]
        exp_params = list(m.exp.parameters())
        through_b = lambda: _gen_loss(m, m.core_forward(m.tokenize(ctx)).b)
        ratios["generation -> bridge -> b -> experiential"] = _fd_ratio(exp_params, _grads(m, through_b, exp_params),
                                                                        through_b, 0.01)
        print("finite difference / |gradient|:", {k: (round(gn, 4), round(r, 4)) for k, (gn, r) in ratios.items()})
        for k, (gn, r) in ratios.items():
            assert gn > 0 and abs(r - 1.0) < 0.03, (k, gn, r)
        assert not bb.lm.training and all(p.grad is None for p in m.param_groups()["backbone_frozen"])
        assert time.time() - t_lock < 900, "held the shared GPU lock for more than 15 minutes"
        del m, bb
    finally:
        free_gpu_memory()
        lock.__exit__(None, None, None)
