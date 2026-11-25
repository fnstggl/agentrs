"""Backbone F_theta and its building blocks (A01 at dev/tiny scale; main/large in test_core_scales.py)."""
import math

import pytest
import torch
import torch.nn.functional as F

from superlearner.cam_unified.config import make_config
from superlearner.cam_unified.core.backbone import UnifiedCore
from superlearner.cam_unified.core.blocks import (Block, CrossBlock, EntitySlotCodes, MultiHeadAttention, RMSNorm,
                                                  SwiGLU, TimeEncoding, count_parameters, signed_log_time)
from superlearner.cam_unified.interfaces import TokenKind, TokenSet
from superlearner.cam_unified.testing import rand_tokens, tiny_config


def _perm_ts(ts: TokenSet, perm) -> TokenSet:
    refs = None if ts.refs is None else [[r[i] for i in perm.tolist()] for r in ts.refs]
    return TokenSet(ts.x[:, perm], ts.mask[:, perm], ts.kind[:, perm], ts.time[:, perm], ts.entity[:, perm], refs,
                    None if ts.entity2 is None else ts.entity2[:, perm])


def _ctx(B=3, N=20, D=64, seed=0):
    ts = rand_tokens(B, N, D, TokenKind.OBS_SEM, seed=seed)
    g = torch.Generator().manual_seed(seed + 100)
    ts.kind = torch.randint(2, 11, (B, N), generator=g)
    ts.entity2 = torch.where(torch.rand(B, N, generator=g) < 0.3, torch.randint(0, 8, (B, N), generator=g),
                             torch.full((B, N), -1))
    return ts


def test_rmsnorm_matches_reference_and_is_bf16_safe():
    n = RMSNorm(16)
    with torch.no_grad():
        n.weight.copy_(torch.linspace(0.5, 2, 16))
    x = torch.randn(4, 7, 16) * 100
    ref = x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * n.weight
    assert torch.allclose(n(x), ref, atol=1e-5, rtol=1e-5)
    xb = x.to(torch.bfloat16)
    assert n(xb).dtype == torch.bfloat16 and torch.isfinite(n(xb).float()).all()


def test_swiglu_is_gated_silu():
    f = SwiGLU(8, 24)
    x = torch.randn(3, 8)
    a, b = f.w13(x).chunk(2, -1)
    assert torch.allclose(f(x), f.w2(F.silu(a) * b), atol=1e-6)
    assert make_config("main").d_ff == 2048 and make_config("large").d_ff == 2752


def test_attention_matches_manual_softmax_and_ignores_padded_keys():
    torch.manual_seed(0)
    att = MultiHeadAttention(32, 4, qk_norm=False)
    x = torch.randn(2, 9, 32)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[0, 6:] = False
    out = att(x, key_mask=mask)
    # manual reference
    q = att.q(x).view(2, 9, 4, 8).transpose(1, 2)
    k, v = att.kv(x).chunk(2, -1)
    k, v = k.view(2, 9, 4, 8).transpose(1, 2), v.view(2, 9, 4, 8).transpose(1, 2)
    s = q @ k.transpose(-1, -2) / math.sqrt(8)
    s = s.masked_fill(~mask[:, None, None, :], float("-inf"))
    ref = att.o((torch.softmax(s, -1) @ v).transpose(1, 2).reshape(2, 9, 32))
    assert torch.allclose(out, ref, atol=1e-5)
    # changing padded keys' content does not change real outputs
    x2 = x.clone()
    x2[0, 6:] = torch.randn(3, 32) * 50
    out2 = att(x2, key_mask=mask)
    assert torch.allclose(out[0, :6], out2[0, :6], atol=1e-5)


def test_null_kv_gives_finite_output_when_every_key_is_padding():
    cb = CrossBlock(16, 2, 32)
    x = torch.randn(2, 3, 16)
    kv = torch.randn(2, 5, 16)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    out = cb(x, kv, key_mask=mask)
    assert torch.isfinite(out).all()
    # with all keys masked the content of kv is irrelevant
    assert torch.allclose(out, cb(x, kv * 7 + 3, key_mask=mask), atol=1e-6)


def test_time_encoding_resolves_minutes_to_years():
    te = TimeEncoding(32, n_freq=16)
    t = -torch.tensor([[1 / 1440, 1 / 24, 1.0, 7.0, 30.0, 365.0, 3650.0]])
    u = signed_log_time(t)
    assert (u[0, 1:] < u[0, :-1]).all(), "log-scaled time must be monotone"
    f = te.features(t)[0]
    d = torch.cdist(f, f)
    assert (d + torch.eye(7) * 1e3).min() > 0.1, "distinct time scales must have distinct encodings"
    assert te(t).shape == (1, 7, 32)


def test_entity_slot_codes_bind_within_episode_and_change_across_episodes():
    sc = EntitySlotCodes(16, 64)
    slots = torch.tensor([[0, 3, 3, -1], [5, 5, 1, 2]])
    c = sc(slots)
    assert torch.allclose(c[0, 1], c[0, 2]) and c[0, 3].abs().sum() == 0
    g = torch.Generator().manual_seed(0)
    all_slots = torch.arange(64).repeat(2, 1)
    before = sc(all_slots)
    sc.new_episode(2, generator=g)
    c2 = sc(slots)
    assert torch.allclose(c2[1, 0], c2[1, 1]), "same slot, same episode -> same code"
    changed = ((sc(all_slots) - before).abs().amax(-1) > 1e-6).float().mean()
    assert changed > 0.9, "a new episode re-draws slot codes"
    # imagination branches map onto their environment's row
    rows = torch.tensor([1, 1, 0])
    c3 = sc(torch.tensor([[5], [5], [5]]), code_rows=rows)
    assert torch.allclose(c3[0, 0], c2[1, 0]) and torch.allclose(c3[1, 0], c2[1, 0])
    with pytest.raises(ValueError):
        sc(torch.zeros(3, 2, dtype=torch.long))
    mid = sc(all_slots)
    sc.new_episode(2, rows=[0], generator=g)
    after = sc(all_slots)
    assert torch.allclose(after[1], mid[1]), "rows not reset keep their codes"
    assert ((after[0] - mid[0]).abs().amax(-1) > 1e-6).float().mean() > 0.9


def test_slot_index_has_no_meaning_beyond_binding():
    """Relabelling slots with a bijection pi while composing the episode permutation with pi^-1 gives identical
    outputs: the backbone sees only the per-episode code, never the raw slot index."""
    cfg = tiny_config()
    core = UnifiedCore(cfg).eval()
    ctx = _ctx(B=2, N=12, D=cfg.d_model, seed=3)
    core.new_episode(2, generator=torch.Generator().manual_seed(1))
    Z0 = core.init_state(2)
    Z, _ = core(Z0, ctx)
    E = cfg.max_entities
    pi = torch.randperm(E, generator=torch.Generator().manual_seed(5))
    inv = torch.argsort(pi)
    relab = TokenSet(ctx.x, ctx.mask, ctx.kind, ctx.time, torch.where(ctx.entity >= 0, pi[ctx.entity.clamp(min=0)],
                                                                       ctx.entity), ctx.refs,
                     torch.where(ctx.entity2 >= 0, pi[ctx.entity2.clamp(min=0)], ctx.entity2))
    core.slot_codes.perm = torch.gather(core.slot_codes.perm, 1, inv.expand(2, -1))
    Z2, _ = core(Z0, relab)
    assert torch.allclose(Z, Z2, atol=1e-5)


def test_ctx_order_invariance():
    cfg = tiny_config()
    torch.manual_seed(0)
    core = UnifiedCore(cfg).eval()
    ctx = _ctx(B=3, N=20, D=cfg.d_model)
    Z0 = torch.randn(3, cfg.n_state, cfg.d_model)
    Z, co = core(Z0, ctx)
    perm = torch.randperm(20, generator=torch.Generator().manual_seed(7))
    Zp, cop = core(Z0, _perm_ts(ctx, perm))
    assert torch.allclose(Z, Zp, atol=2e-5), (Z - Zp).abs().max()
    assert torch.allclose(co[:, perm], cop, atol=2e-5)


def test_padding_invariance_and_real_dependence_on_context_and_previous_state():
    cfg = tiny_config()
    torch.manual_seed(0)
    core = UnifiedCore(cfg).eval()
    ctx = _ctx(B=2, N=10, D=cfg.d_model)
    ctx.mask[:] = True
    Z0 = torch.randn(2, cfg.n_state, cfg.d_model)
    Z, co = core(Z0, ctx)
    pad = rand_tokens(2, 6, cfg.d_model, seed=9)
    pad.mask[:] = False
    pad.x = pad.x * 100
    big = TokenSet.cat(ctx, pad)
    Zb, cob = core(Z0, big)
    assert torch.allclose(Z, Zb, atol=1e-5) and torch.allclose(co, cob[:, :10], atol=1e-5)
    assert cob[:, 10:].abs().max() == 0
    ctx2 = TokenSet(ctx.x.clone(), ctx.mask, ctx.kind, ctx.time, ctx.entity, ctx.refs, ctx.entity2)
    ctx2.x[:, 0] += 3.0
    assert (core(Z0, ctx2)[0] - Z).abs().max() > 1e-3, "Z must depend on the observations"
    assert (core(Z0 * 0.5 + 1, ctx)[0] - Z).abs().max() > 1e-3, "Z must depend on Z_prev"


def test_context_budget_by_kind_priority_then_recency_with_counter():
    cfg = tiny_config(max_ctx_tokens=8)
    core = UnifiedCore(cfg).eval()
    B, N, D = 1, 14, cfg.d_model
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, N, D, generator=g)
    kind = torch.tensor([[TokenKind.MEMORY] * 6 + [TokenKind.GOAL] * 2 + [TokenKind.OBS_SEM] * 6])
    time = torch.tensor([[-1.0, -2, -3, -4, -5, -6, -50, -60, -0.1, -0.2, -9, -10, -11, -12]])
    ctx = TokenSet(x, torch.ones(B, N, dtype=torch.bool), kind, time, torch.full((B, N), -1), None)
    Z0 = core.init_state(B)
    Z, co = core(Z0, ctx)
    kept = core.last_kept[0]
    # 2 GOAL (highest priority) + 6 OBS_SEM (priority over MEMORY): budget 8 -> all MEMORY dropped
    assert kept.tolist() == [False] * 6 + [True] * 8
    assert core.budget_stats["dropped_tokens"] == 6 and core.budget_stats["dropped_by_kind"]["MEMORY"] == 6
    assert co[0, :6].abs().max() == 0
    # recency within a kind: budget 5 keeps the two goals and the 3 most recent observations
    core.cfg = tiny_config(max_ctx_tokens=5)
    core(Z0, ctx)
    assert core.last_kept[0].nonzero().flatten().tolist() == [6, 7, 8, 9, 10]
    # the result equals running on the kept tokens alone
    core.cfg = tiny_config(max_ctx_tokens=8)
    Z_sel, co_sel = core(Z0, TokenSet(x[:, 6:], ctx.mask[:, 6:], kind[:, 6:], time[:, 6:], ctx.entity[:, 6:], None))
    assert torch.allclose(Z, Z_sel, atol=1e-5) and torch.allclose(co[:, 6:], co_sel, atol=1e-5)


def test_world_model_prior_kinds_and_empty_context():
    cfg = tiny_config()
    core = UnifiedCore(cfg)
    B = 2
    parts = [rand_tokens(B, n, cfg.d_model, k, ragged=False, seed=int(k)) for k, n in
             ((TokenKind.ACTION, 2), (TokenKind.BELIEF, cfg.n_particles + 1), (TokenKind.TIME, 1),
              (TokenKind.NOISE, 1))]
    Z, co = core(core.init_state(B), TokenSet.cat(*parts))
    assert Z.shape == (B, cfg.n_state, cfg.d_model) and co.shape == (B, cfg.n_particles + 5, cfg.d_model)
    Ze, coe = core(core.init_state(B), TokenSet.empty(B, cfg.d_model))
    assert Ze.shape == Z.shape and coe.shape == (B, 0, cfg.d_model) and torch.isfinite(Ze).all()


def test_every_parameter_receives_gradient():
    cfg = tiny_config()
    core = UnifiedCore(cfg)
    ctx = _ctx(B=3, N=16, D=cfg.d_model)
    Z, co = core(core.init_state(3), ctx)
    (Z.pow(2).mean() + co.pow(2).mean() + Z[:, 0].sum() + co[:, :3].sum()).backward()
    missing = [n for n, p in core.named_parameters() if p.grad is None or p.grad.abs().max() == 0]
    assert not missing, missing


def test_activation_checkpointing_gives_identical_gradients():
    cfg = tiny_config()
    torch.manual_seed(0)
    a = UnifiedCore(cfg)
    b = UnifiedCore(tiny_config(grad_checkpoint=True))
    b.load_state_dict(a.state_dict())
    assert b.grad_checkpoint and not a.grad_checkpoint
    ctx = _ctx(B=2, N=12, D=cfg.d_model)
    la = a(a.init_state(2), ctx)[0].pow(2).sum()
    lb = b(b.init_state(2), ctx)[0].pow(2).sum()
    la.backward()
    lb.backward()
    assert torch.allclose(la, lb)
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-6), n


def test_recurrent_rollout_is_stable_over_many_steps():
    """Z_t = F(Z_{t-1}, O_t) for 200 steps stays finite and bounded (output RMSNorm)."""
    cfg = tiny_config()
    core = UnifiedCore(cfg).eval()
    Z = core.init_state(2)
    with torch.no_grad():
        for t in range(200):
            Z, _ = core(Z, rand_tokens(2, 8, cfg.d_model, seed=t))
    assert torch.isfinite(Z).all() and Z.pow(2).mean(-1).sqrt().max() < 10


def test_dev_config_instantiates_runs_and_reports_parameters():
    cfg = make_config("dev")
    core = UnifiedCore(cfg)
    pc = core.count_parameters()
    assert pc["total"] == count_parameters(core) and pc["blocks"] == cfg.n_layers * pc["per_block"]
    # 8 x (4 * 384^2 attention + 3 * 384 * 1024 SwiGLU + norms)
    assert abs(pc["per_block"] - (4 * 384 ** 2 + 3 * 384 * 1024 + 2 * 384 + 2 * 64)) == 0
    ctx = rand_tokens(2, 64, cfg.d_model, seed=1)
    Z, co = core(core.init_state(2), ctx)
    Z.mean().backward()
    assert Z.shape == (2, 32, 384) and torch.isfinite(Z).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_bf16_autocast_matches_fp32_on_gpu():
    cfg = tiny_config(d_model=128, n_heads=4, n_layers=2)
    torch.manual_seed(0)
    core = UnifiedCore(cfg).cuda().eval()
    ctx = _ctx(B=4, N=40, D=cfg.d_model).to("cuda")
    Z0 = core.init_state(4, "cuda")
    core.autocast = False
    Zf, _ = core(Z0, ctx)
    core.autocast = True
    Zb, _ = core(Z0, ctx)
    assert Zb.dtype == torch.float32
    rel = (Zf - Zb).norm() / Zf.norm()
    assert rel < 0.05, rel
