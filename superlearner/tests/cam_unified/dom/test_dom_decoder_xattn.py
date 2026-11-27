"""DOM Mechanism A: CAM latent tokens condition the pretrained decoder at several depths (DOM-03, DOM-08, DOM-09).
Uses the cached Qwen3-0.6B (frozen); skipped without CUDA."""
import os

import pytest
import torch

from superlearner.cam_unified.semantic.decoder_xattn import LatentDecoderConditioning

pytestmark = pytest.mark.gpu
HUB = "/workspace/.hf_home/hub"


@pytest.fixture(scope="module")
def lm():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).cuda().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return tok, m


def _cond(m, d_cam=256, alpha_init=0.05):
    # the default gate is now alpha_init = 0 (bit-identical at init, test below); these tests exercise an open gate
    n = m.config.num_hidden_layers
    c = LatentDecoderConditioning(m.config.hidden_size, d_cam, LatentDecoderConditioning.every(n, 4), d_attn=256,
                                  n_heads=4, alpha_init=alpha_init).cuda()
    return c.attach(m)


def test_closed_gates_leave_the_pretrained_decoder_bit_identical(lm):
    tok, m = lm
    c = _cond(m, alpha_init=0.0)
    ids = tok(['{"handle": "bakery-labs", "name": "Bakery Labs"}'], return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        base = m(ids).logits.float()
        with c.condition(torch.randn(1, 8, 256, device="cuda") * 5):
            cond = m(ids).logits.float()
    c.detach_hooks()
    assert torch.equal(base, cond)


def test_decoder_output_depends_on_cam_latent_and_preserves_pretrained_behaviour(lm):
    tok, m = lm
    c = _cond(m)
    ids = tok(["def add(a, b):\n    return"], return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        base = m(ids).logits.float()
        Z1, Z2 = torch.randn(1, 8, 256, device="cuda"), torch.randn(1, 8, 256, device="cuda")
        with c.condition(Z1):
            l1 = m(ids).logits.float()
        with c.condition(Z2):
            l2 = m(ids).logits.float()
    c.detach_hooks()
    assert float((l1 - l2).abs().max()) > 1e-2                     # Z changes the decoder's distribution
    p0, p1 = torch.softmax(base[0, -1], -1), torch.softmax(l1[0, -1], -1)
    assert int(p0.argmax()) == int(p1.argmax())                     # pretrained competence preserved at init
    assert float(0.5 * (p0 - p1).abs().sum()) < 0.5


def test_gradients_flow_through_frozen_decoder_into_cam_latent_and_gates(lm):
    tok, m = lm
    c = _cond(m)
    ids = tok(["{\"tool\": \"bank.transfer\", \"args\": {\"amount\": 5"], return_tensors="pt").input_ids.cuda()
    Z = torch.randn(1, 8, 256, device="cuda", requires_grad=True)
    with c.condition(Z):
        lp = torch.log_softmax(m(ids).logits.float(), -1)
    nll = -lp[0, :-1].gather(-1, ids[0, 1:, None]).mean()
    nll.backward()
    c.detach_hooks()
    assert Z.grad is not None and float(Z.grad.abs().sum()) > 0
    for i in c.layer_ids:
        mod = c.mods[str(i)]
        assert mod.alpha.grad is not None and float(mod.alpha.grad.abs()) > 0
        assert float(mod.kv.weight.grad.abs().sum()) > 0
    assert all(p.grad is None for p in m.parameters())               # the semantic model itself stays frozen


def test_kv_cached_generation_reads_the_latent_at_every_step(lm):
    tok, m = lm
    c = _cond(m)
    ids = tok(["The answer is"], return_tensors="pt").input_ids.cuda()
    torch.manual_seed(0)
    with torch.no_grad():
        outs = []
        for s in range(2):
            Z = torch.randn(1, 8, 256, device="cuda", generator=torch.Generator("cuda").manual_seed(s)) * 3
            with torch.no_grad():
                for mod in c.mods.values():
                    mod.alpha.fill_(1.0)
            calls0 = c.calls
            with c.condition(Z):
                o = m.generate(ids, max_new_tokens=8, do_sample=False, use_cache=True)
            assert c.calls - calls0 >= 8 * len(c.layer_ids)            # every generated token passed the hooks
            outs.append(o[0, ids.shape[1]:].tolist())
    c.detach_hooks()
    assert outs[0] != outs[1]
