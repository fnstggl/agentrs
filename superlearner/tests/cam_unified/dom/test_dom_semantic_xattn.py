"""DOM Mechanism C: CAM-controlled gated semantic cross-attention inside the core (DOM-01, DOM-05, DOM-08)."""
import pytest
import torch

from superlearner.cam_unified.config import make_config
from superlearner.cam_unified.core.backbone import UnifiedCore
from superlearner.cam_unified.core.semantic_xattn import GatedUnifiedCore, SemanticContext
from superlearner.cam_unified.interfaces import TokenKind, TokenSet

torch.set_num_threads(8)


def _setup(seed=0):
    torch.manual_seed(seed)
    cfg = make_config("dev", sem_xattn_every=2, sem_xattn_dim=96)
    core = GatedUnifiedCore(cfg, autocast=False)
    B = 3
    core.new_episode(B)
    ctx = TokenSet.from_dense(torch.randn(B, 10, cfg.d_model), int(TokenKind.OBS_NUM))
    sem = SemanticContext(torch.randn(B, 20, 96), torch.ones(B, 20, dtype=torch.bool), torch.randn(B, cfg.d_model))
    return cfg, core, ctx, sem, B


def test_gradients_reach_every_block_projection_attention_and_gate():
    cfg, core, ctx, sem, B = _setup()
    core.set_semantic(sem)
    Z, _ = core(core.init_state(B), ctx)
    (Z * torch.randn_like(Z)).sum().backward()
    missing = [n for n, p in core.named_parameters()
               if p.requires_grad and not n.startswith("slot_codes") and (p.grad is None or p.grad.abs().sum() == 0)]
    assert missing == []
    assert len(core.last_gates) == cfg.n_layers // 2
    assert all(0.0 < g < 0.2 for g in core.last_gates.values())      # starts mostly CAM-driven


def test_output_depends_on_semantic_content_and_gate_controls_it():
    cfg, core, ctx, sem, B = _setup(1)
    Z0 = core.init_state(B)
    core.set_semantic(sem)
    Za, _ = core(Z0, ctx)
    core.set_semantic(SemanticContext(torch.randn_like(sem.h), sem.mask, sem.cond))
    Zb, _ = core(Z0, ctx)
    assert float((Za - Zb).abs().max()) > 1e-3
    with torch.no_grad():
        for m in core.xattn.values():
            m.gate_out.bias.fill_(-30.0)                                  # gates closed: semantic pathway off
    core.set_semantic(sem)
    Zc, _ = core(Z0, ctx)
    core.set_semantic(None)
    Zd, _ = core(Z0, ctx)
    assert float((Zc - Zd).abs().max()) < 1e-5


def test_without_semantic_input_it_is_exactly_the_unified_core():
    cfg, core, ctx, sem, B = _setup(2)
    base = UnifiedCore(cfg, autocast=False)
    base.load_state_dict({k: v for k, v in core.state_dict().items() if k in base.state_dict()}, strict=True)
    base.slot_codes.load_episode_state(core.slot_codes.episode_state())
    core.set_semantic(None)
    Z0 = core.init_state(B)
    a, _ = core(Z0, ctx)
    b, _ = base(Z0, ctx)
    assert torch.allclose(a, b, atol=1e-6)


def test_superlearner_scale_has_about_one_billion_trainable_core_parameters():
    cfg = make_config("superlearner", sem_xattn_every=2, sem_xattn_dim=2560)
    assert (cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_state, cfg.d_ff) == (32, 1536, 24, 128, 4096)
    with torch.device("meta"):
        core = GatedUnifiedCore(cfg, autocast=False)
    n = core.count_parameters()["total"]
    assert 0.9e9 <= n <= 1.2e9, n
