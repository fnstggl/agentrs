"""Persistence of the latent state Z_t through the shared backbone (W3; audit 2026-11-26 h-core).

The pre-DOM core re-added StatePos + KindEmb(STATE) (std 0.5) to Z every decision and re-normalised it: at init Z_t
was ~constant (cos 0.995 with that embedding), depended ~2% on the observation content, and an 8-step delayed recall
was not learnable. The DOM core (cfg.state_gate, cfg.state_emb_std = 0.02) carries Z through a learned gate
u = sigmoid(W [Z_prev; Z~] + b), W = 0, b = -2, with a bounded candidate Z~ (state_candidate_of)."""
import math

import pytest
import torch

from superlearner.cam_unified.config import make_config, make_preset
from superlearner.cam_unified.core.backbone import STATE_GATE_BIAS, UnifiedCore
from superlearner.cam_unified.core.probes import delayed_recall, relative_change, state_constancy
from superlearner.cam_unified.core.semantic_xattn import SemanticContext
from superlearner.cam_unified.interfaces import TokenKind, TokenSet
from superlearner.cam_unified.testing import tiny_config

torch.set_num_threads(8)


def _ctxs(n, B, N, D, seed):
    g = torch.Generator().manual_seed(seed)
    return [TokenSet.from_dense(torch.randn(B, N, D, generator=g), int(TokenKind.OBS_SEM)) for _ in range(n)]


def _dom_dev_core(seed=0, **over):
    torch.manual_seed(seed)
    return UnifiedCore(make_preset("dom_dev", sem_xattn_every=0, **over), autocast=False)


def test_default_config_is_the_pre_dom_core():
    """No gate, no cross-attention, state embeddings at emb_std, Z = out_norm(blocks(...)) at the state positions,
    and the same initialisation stream as an explicit state_emb_std = emb_std."""
    cfg = make_config("dev")
    torch.manual_seed(0)
    core = UnifiedCore(cfg, autocast=False)
    torch.manual_seed(0)
    ref = UnifiedCore(cfg, autocast=False, state_emb_std=0.5)
    assert core.state_gate is None and core.sem_proj is None and core.state_emb_std == core.emb_std == 0.5
    sa, sb = core.state_dict(), ref.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    B = 3
    ctx = _ctxs(1, B, 7, cfg.d_model, 1)[0]
    Zp = torch.randn(B, cfg.n_state, cfg.d_model)
    with torch.no_grad():
        Z, _ = core(Zp, ctx)
        h = torch.cat([core.embed_state(Zp), core.embed_ctx(ctx)], 1)
        manual = core.out_norm(core.run_blocks(h, None)).float()[:, :cfg.n_state]
    assert torch.equal(Z, manual)


def test_dom_initialisation_gate_and_state_embedding_scale():
    core = _dom_dev_core()
    cfg = core.cfg
    assert cfg.state_gate and cfg.state_emb_std == 0.02 and core.state_emb_std == 0.02
    assert torch.count_nonzero(core.state_gate.weight) == 0
    assert torch.all(core.state_gate.bias == STATE_GATE_BIAS) and STATE_GATE_BIAS == -2.0
    assert abs(float(core.state_pos.detach().std()) - 0.02) < 0.004
    for k in (TokenKind.STATE, TokenKind.REASON):
        assert abs(float(core.kind_emb.weight[int(k)].detach().std()) - 0.02) < 0.006, k
    assert abs(float(core.kind_emb.weight[int(TokenKind.OBS_SEM)].detach().std()) - 0.5) < 0.1   # context kinds unchanged
    B = 2
    ctx = _ctxs(1, B, 5, cfg.d_model, 2)[0]
    with torch.no_grad():
        core(core.init_state(B), ctx)
    assert abs(float(core.last_state_gate) - 1 / (1 + math.exp(2.0))) < 1e-6          # u = sigmoid(-2) = 0.119


@pytest.mark.parametrize("mode", ["clamp", "clamp_keep_emb", "rms"])
def test_gated_update_formula(mode):
    core = _dom_dev_core(1)
    core.state_candidate = mode
    cfg = core.cfg
    B, S = 2, cfg.n_state
    with torch.no_grad():
        core.state_gate.weight.normal_(0, 0.02)                           # an input-dependent gate
        ctx = _ctxs(1, B, 6, cfg.d_model, 3)[0]
        Zp = torch.randn(B, S, cfg.d_model) * 0.3
        Z, _ = core(Zp, ctx)
        h = core.run_blocks(torch.cat([core.embed_state(Zp), core.embed_ctx(ctx)], 1), None)[:, :S].float()
        e = core.state_pos + core.kind_emb.weight[int(TokenKind.STATE)]
        if mode == "rms":
            cand = core.out_norm(h).float()
        else:
            x = h - e if mode == "clamp" else h
            cand = x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt().clamp_min(1.0) * core.out_norm.weight
        u = torch.sigmoid(core.state_gate(torch.cat([Zp, cand], -1)))
        manual = (1 - u) * Zp + u * cand
    assert torch.allclose(Z, manual, atol=1e-5), float((Z - manual).abs().max())


def test_state_at_init_is_not_a_constant_and_depends_on_observations():
    """Acceptance (plan 2026-11-26, W3): at initialisation cos(Z_t, const) < 0.9 and observation dependence > 10%
    (first decision from Z_0, and the whole 9-decision history); the pre-DOM core shows the audit's failure."""
    out = {}
    for name, over in (("dom", {}), ("legacy", dict(state_gate=False, state_emb_std=0.5))):
        core = _dom_dev_core(0, **over).eval()
        D, B = core.cfg.d_model, 8
        A, Bc = _ctxs(9, B, 24, D, 1), _ctxs(9, B, 24, D, 2)
        with torch.no_grad():
            Z = core.init_state(B)
            za, zb = core(Z, A[0])[0], core(Z, Bc[0])[0]
            first = relative_change(za, zb)
            cos = []
            Za, Zb = core.init_state(B), core.init_state(B)
            for t in range(9):
                Za, Zb = core(Za, A[t])[0], core(Zb, Bc[t])[0]
                cos.append(float(state_constancy(core, Za).max()))
            out[name] = (first, relative_change(Za, Zb), max(cos))
    first, hist, cos = out["dom"]
    assert cos < 0.9 and first > 0.10 and hist > 0.10, out["dom"]
    lf, lh, lc = out["legacy"]
    assert lc > 0.95 and lf < 0.05 and lh < 0.05, out["legacy"]


def test_gated_state_is_bounded_over_a_long_lifetime():
    core = _dom_dev_core(2).eval()
    D, B = core.cfg.d_model, 2
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        Z = core.init_state(B)
        for t in range(150):
            Z, _ = core(Z, TokenSet.from_dense(5.0 * torch.randn(B, 8, D, generator=g), int(TokenKind.OBS_SEM)))
    rms = Z.pow(2).mean(-1).sqrt()
    assert torch.isfinite(Z).all() and float(rms.max()) <= math.sqrt(2) + 1e-4, float(rms.max())


def test_gate_and_bptt_gradients_and_checkpointing_with_sem():
    """Gradients reach the gate, the earliest state (BPTT through the gate) and Mechanism C; per-block
    activation checkpointing gives identical values and gradients."""
    torch.manual_seed(3)
    cfg = make_preset("dom_dev", n_layers=4, d_model=64, n_heads=4, n_state=8, sem_xattn_dim=32)
    B = 3
    ctxs = _ctxs(3, B, 5, cfg.d_model, 4)
    sem = SemanticContext(torch.randn(B, 7, 32), torch.ones(B, 7, dtype=torch.bool), torch.randn(B, cfg.d_model))
    res = {}
    for ck in (False, True):
        torch.manual_seed(3)
        core = UnifiedCore(make_preset("dom_dev", n_layers=4, d_model=64, n_heads=4, n_state=8, sem_xattn_dim=32,
                                       grad_checkpoint=ck), autocast=False).train()
        Z0 = (core.init_state(B).detach() + 0.1 * torch.randn(B, cfg.n_state, cfg.d_model)).requires_grad_(True)
        Z = Z0
        for c in ctxs:
            Z, _ = core(Z, c, sem=sem)
        (Z * torch.linspace(-1, 1, cfg.d_model)).sum().backward()
        res[ck] = (Z.detach(), Z0.grad.clone(), {n: p.grad.clone() for n, p in core.named_parameters()
                                                  if p.grad is not None})
    assert torch.allclose(res[False][0], res[True][0], atol=1e-6)
    assert torch.allclose(res[False][1], res[True][1], rtol=1e-4, atol=1e-8)
    g = res[False][2]
    assert g.keys() == res[True][2].keys()
    for n in g:
        assert torch.allclose(g[n], res[True][2][n], rtol=1e-4, atol=1e-7), n
    assert g["state_gate.weight"].abs().sum() > 0 and g["state_gate.bias"].abs().sum() > 0
    assert res[False][1].abs().sum() > 0                                   # BPTT reaches Z_0
    assert g["sem_proj.proj.weight"].abs().sum() > 0


def test_tiny_gated_core_learns_delayed_recall_on_cpu():
    """Fast CPU check of the mechanism (tiny 2 x 64 core): delay-6 recall with the gate (the legacy core stays at
    chance on such budgets: see the W3 evidence file docs/evidence/dom/W3/)."""
    torch.manual_seed(0)
    cfg = tiny_config(n_layers=2, d_model=64, n_heads=4, n_state=8, state_gate=True, state_emb_std=0.02)
    core = UnifiedCore(cfg, autocast=False)
    r = delayed_recall(core, 6, iters=500, target=0.95, eval_every=25, batch=32, lr=1e-3, device="cpu")
    assert r["best_acc"] >= 0.9, r["history"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (about 2 min on an A6000, ~1.2 GB)")
def test_delayed_recall_delay16_dom_dev():
    """Acceptance (plan 2026-11-26, W3): delayed recall at delay 16 >= 95% through the real dom_dev backbone
    (8 x 384, gated state, state_emb_std 0.02), BPTT through all 17 steps, held-out accuracy. Activation
    checkpointing keeps it under 2 GB (no exclusive GPU lock needed)."""
    torch.manual_seed(0)
    core = UnifiedCore(make_preset("dom_dev", grad_checkpoint=True)).cuda()
    r = delayed_recall(core, 16, iters=600, target=0.95, eval_every=25, seed=0)
    assert r["best_acc"] >= 0.95, r["history"]
