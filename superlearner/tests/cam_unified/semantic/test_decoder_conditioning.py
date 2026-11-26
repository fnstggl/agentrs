"""DOM Mechanism A wired into the semantic decoder: QwenSemantic(cfg.dec_xattn_every > 0) attaches
decoder_xattn.LatentDecoderConditioning; `cond=(Z, mask)` on step / decoder_hidden and HoleFiller.sample / score.
Real Qwen3-0.6B (fp32 copy for the numerical-equivalence checks)."""
from __future__ import annotations

import json

import pytest
import torch

from superlearner.cam_unified.interfaces import ToolSchema
from superlearner.cam_unified.semantic.grammar import GrammarOptions, HoleFiller, ToolGrammar, drive_walks

SCHEMA = {"type": "object", "properties": {"to": {"type": "string", "maxLength": 40},
                                           "text": {"type": "string", "maxLength": 200}},
          "required": ["to", "text"], "additionalProperties": False}


@pytest.fixture(scope="module")
def cenc(cuda):
    from superlearner.cam_unified.semantic.encoder import QwenSemantic
    from superlearner.cam_unified.testing import tiny_config
    torch.manual_seed(0)
    cfg = tiny_config(dec_xattn_every=6, dec_alpha_init=0.0)
    return QwenSemantic(cfg, test=True, dtype=torch.float32)


def _set_alpha(enc, a):
    with torch.no_grad():
        for m in enc.conditioning.mods.values():
            m.alpha.fill_(a)


def _inputs(enc, text='{"to": "cust_7", "text": "We can deliver 40 units by Friday."}'):
    ids = enc.tok("<|im_start|>assistant\n" + text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = ids.to(enc.device)
    emb = enc.embed_ids(ids)
    att = torch.ones_like(ids)
    pos = torch.arange(ids.shape[1], device=enc.device).unsqueeze(0)
    return ids, emb, att, pos


def test_conditioning_is_attached_and_closed_gates_are_bit_identical(cenc):
    c = cenc.conditioning
    assert c is not None and c.layer_ids == list(range(5, cenc.n_layers, 6))
    assert all(v == 0.0 for v in c.gates().values())
    _, emb, att, pos = _inputs(cenc)
    Z = torch.randn(1, 12, cenc.cfg.d_model, device=cenc.device) * 3
    with torch.no_grad():
        h0 = cenc.decoder_hidden(emb, att, pos)
        h1 = cenc.decoder_hidden(emb, att, pos, cond=(Z, None))
        s0, _ = cenc.step(emb, att, pos, cenc.new_cache())
        s1, _ = cenc.step(emb, att, pos, cenc.new_cache(), cond=(Z, None))
    assert torch.equal(h0, h1) and torch.equal(s0, s1)


def test_open_gates_condition_the_decoder_and_kv_steps_match_the_full_forward(cenc):
    ids, emb, att, pos = _inputs(cenc)
    Z = torch.randn(1, 12, cenc.cfg.d_model, device=cenc.device)
    m = torch.ones(1, 12, dtype=torch.bool, device=cenc.device)
    _set_alpha(cenc, 0.5)
    try:
        with torch.no_grad():
            full = cenc.lm_logits(cenc.decoder_hidden(emb, att, pos, cond=(Z, m)))[0]
            base = cenc.lm_logits(cenc.decoder_hidden(emb, att, pos))[0]
            assert float((full - base).abs().max()) > 1e-2                  # Z changes the decoder
            k = 5                                                           # prefill k tokens, then one at a time
            cache = cenc.new_cache()
            lg, cache = cenc.step(emb[:, :k], att[:, :k], pos[:, :k], cache, cond=(Z, m))
            diffs = [float((lg[0] - full[k - 1]).abs().max())]
            for t in range(k, ids.shape[1]):
                lg, cache = cenc.step(emb[:, t:t + 1], att[:, :t + 1], pos[:, t:t + 1], cache, cond=(Z, m))
                diffs.append(float((lg[0] - full[t]).abs().max()))
        assert max(diffs) < 1e-3, max(diffs)
        # the encoder path never reads the latent (its hidden-state cache stays valid)
        cenc._cache.clear()
        a = cenc.hidden(["The customer asked for a refund."])[0]
        c = cenc.conditioning
        c.detach_hooks()
        cenc._cache.clear()
        b = cenc.hidden(["The customer asked for a refund."])[0]
        c.attach(cenc.lm)
        assert torch.equal(a, b)
    finally:
        _set_alpha(cenc, 0.0)


def test_gradients_reach_the_latent_and_checkpointing_is_exact(cenc):
    ids, emb, att, pos = _inputs(cenc)
    Z0 = torch.randn(1, 12, cenc.cfg.d_model, device=cenc.device)
    _set_alpha(cenc, 0.3)
    grads = []
    try:
        for ckpt in (True, False):
            cenc.decoder_grad_checkpoint = ckpt
            cenc.conditioning.zero_grad(set_to_none=True)
            Z = Z0.clone().requires_grad_(True)
            lp = torch.log_softmax(cenc.lm_logits(cenc.decoder_hidden(emb, att, pos, cond=(Z, None)))[0], -1)
            nll = -lp[:-1].gather(-1, ids[0, 1:, None]).mean()
            nll.backward()
            mod = cenc.conditioning.mods[str(cenc.conditioning.layer_ids[0])]
            grads.append((Z.grad.clone(), mod.kv.weight.grad.clone(), mod.alpha.grad.clone()))
        assert float(grads[0][0].abs().sum()) > 0 and float(grads[0][1].abs().sum()) > 0
        for a, b in zip(*grads):
            assert torch.allclose(a, b, rtol=1e-4, atol=1e-7)
        assert all(p.grad is None for p in cenc.lm.parameters() if not p.requires_grad)
    finally:
        cenc.decoder_grad_checkpoint = True
        _set_alpha(cenc, 0.0)


def test_closed_gate_still_trains_alpha(cenc):
    _, emb, att, pos = _inputs(cenc)
    Z = torch.randn(1, 12, cenc.cfg.d_model, device=cenc.device, requires_grad=True)
    cenc.conditioning.zero_grad(set_to_none=True)
    h = cenc.decoder_hidden(emb, att, pos, cond=(Z, None))
    cenc.lm_logits(h).logsumexp(-1).mean().backward()
    alphas = [float(m.alpha.grad.abs()) for m in cenc.conditioning.mods.values()]
    assert all(a > 0 for a in alphas)                      # tanh'(0) = 1: the gate learns from step 0
    assert float(Z.grad.abs().sum()) == 0.0                 # nothing else moves while the gate is closed


def test_holefiller_cond_rescoring_is_bit_exact(cenc):
    from superlearner.cam_unified.semantic.constraints import tables_for
    opts = GrammarOptions(max_string_tokens=24, max_action_tokens=64, length_prior=False)
    f = HoleFiller(cenc, tables_for(cenc.tok, cenc.vocab_size, cenc.device, cenc.model_name), opts)
    g = ToolGrammar(ToolSchema("mail.send", "Send a message.", SCHEMA), opts)
    R = 3
    progs = [f.program(drive_walks([g.root], opts, lambda rows, decs: [0 for _ in decs])[0]) for _ in range(R)]
    ctx = [f.context_ids("Write the JSON arguments for mail.send. Schema: " + json.dumps(SCHEMA))[0]] * R
    prefix = torch.zeros(R, 0, cenc.sem_dim, device=cenc.device)
    Z = torch.randn(R, 10, cenc.cfg.d_model, device=cenc.device)
    m = torch.ones(R, 10, dtype=torch.bool, device=cenc.device)
    m[1, 6:] = False
    _set_alpha(cenc, 1.0)
    try:
        lp_inc, _, ns, _ = f.sample(progs, prefix, ctx, 1.0, False, torch.Generator(device=cenc.device).manual_seed(1),
                                    cond=(Z, m))
        s1 = f.score(progs, prefix, ctx, 1.0, exact_rows=True, cond=(Z, m))[0]
        s2 = f.score(progs, prefix, ctx, 1.0, exact_rows=True, cond=(Z, m))[0]
        s3 = f.score(progs[::-1], prefix, ctx, 1.0, exact_rows=True, cond=(Z.flip(0), m.flip(0)))[0]
        s0 = f.score(progs, prefix, ctx, 1.0, exact_rows=True)[0]
    finally:
        _set_alpha(cenc, 0.0)
    assert sum(ns) > 0 and torch.equal(s1, s2) and torch.equal(s1, s3.flip(0))
    assert float((lp_inc - s1).abs().max()) < 1e-2                         # fp32 incremental vs teacher-forced
    assert float((s1 - s0).abs().max()) > 1e-3                              # the latent is part of the logp


def test_cond_requires_attached_conditioning(enc):
    _, emb, att, pos = _inputs(enc)
    assert enc.conditioning is None
    with pytest.raises(ValueError):
        enc.decoder_hidden(emb, att, pos, cond=(torch.zeros(1, 2, enc.cfg.d_model, device=enc.device), None))
