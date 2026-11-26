"""DOM Mechanism C: CAM-controlled gated semantic cross-attention inside the core (DOM-01, DOM-05, DOM-08).

The semantic context is an explicit argument of UnifiedCore.forward / run_blocks and Reasoner.forward (no module
state): these tests cover the audit 2026-11-26 hazards (imagination batches of another size, stale context, EMA /
deep copies, iteration checkpointing with another decision encoded before backward)."""
import copy

import pytest
import torch

from superlearner.cam_unified.config import make_config, make_preset
from superlearner.cam_unified.core.backbone import UnifiedCore
from superlearner.cam_unified.core.reasoning import Reasoner
from superlearner.cam_unified.core.semantic_xattn import GatedUnifiedCore, SemanticContext, SemanticKV
from superlearner.cam_unified.interfaces import TokenKind, TokenSet

torch.set_num_threads(8)


def _setup(seed=0, **over):
    torch.manual_seed(seed)
    cfg = make_config("dev", sem_xattn_every=2, sem_xattn_dim=96, **over)
    core = GatedUnifiedCore(cfg, autocast=False)
    B = 3
    core.new_episode(B)
    ctx = TokenSet.from_dense(torch.randn(B, 10, cfg.d_model), int(TokenKind.OBS_NUM))
    sem = SemanticContext(torch.randn(B, 20, 96), torch.ones(B, 20, dtype=torch.bool), torch.randn(B, cfg.d_model))
    return cfg, core, ctx, sem, B


def _other(sem, seed=7):
    g = torch.Generator().manual_seed(seed)
    return SemanticContext(torch.randn(sem.h.shape, generator=g), sem.mask.clone(),
                           torch.randn(sem.cond.shape, generator=g))


def test_gated_core_is_the_registry_core_class():
    assert GatedUnifiedCore is UnifiedCore
    cfg = make_config("dev")
    assert UnifiedCore(cfg).sem_proj is None and len(UnifiedCore(cfg).xattn) == 0   # off unless configured


def test_gradients_reach_every_block_projection_attention_and_gate():
    cfg, core, ctx, sem, B = _setup()
    Z, _ = core(core.init_state(B), ctx, sem=sem)
    (Z * torch.randn_like(Z)).sum().backward()
    missing = [n for n, p in core.named_parameters()
               if p.requires_grad and not n.startswith("slot_codes") and (p.grad is None or p.grad.abs().sum() == 0)]
    assert missing == []
    assert len(core.last_gates) == cfg.n_layers // 2
    assert all(isinstance(g, torch.Tensor) and not g.requires_grad for g in core.last_gates.values())   # no sync
    assert all(0.0 < g < 0.2 for g in core.gate_stats().values())      # starts mostly CAM-driven


def test_output_depends_on_semantic_content_and_gate_controls_it():
    cfg, core, ctx, sem, B = _setup(1)
    Z0 = core.init_state(B)
    with torch.no_grad():
        Za, _ = core(Z0, ctx, sem=sem)
        Zb, _ = core(Z0, ctx, sem=_other(sem))
        assert float((Za - Zb).abs().max()) > 1e-3
        for m in core.xattn.values():
            m.gate_out.bias.fill_(-30.0)                                  # gates closed: semantic pathway off
        Zc, _ = core(Z0, ctx, sem=sem)
        Zd, _ = core(Z0, ctx)
    assert float((Zc - Zd).abs().max()) < 1e-5


def test_without_semantic_input_it_is_exactly_the_plain_backbone():
    cfg, core, ctx, sem, B = _setup(2)
    plain_cfg = make_config("dev")
    torch.manual_seed(2)
    base = UnifiedCore(plain_cfg, autocast=False)
    base.load_state_dict({k: v for k, v in core.state_dict().items() if k in base.state_dict()}, strict=True)
    base.slot_codes.load_episode_state(core.slot_codes.episode_state())
    Z0 = core.init_state(B)
    with torch.no_grad():
        a, ca = core(Z0, ctx)
        b, cb = base(Z0, ctx)
    assert torch.equal(a, b) and torch.equal(ca, cb)


def test_no_module_state_same_arguments_same_result():
    """The old set_semantic design leaked the previous call's context; now a call is a pure function of its
    arguments, and the API to set module state is gone."""
    cfg, core, ctx, sem, B = _setup(3)
    assert not hasattr(core, "set_semantic") and not hasattr(core, "_sem")
    Z0 = core.init_state(B)
    with torch.no_grad():
        a1, _ = core(Z0, ctx, sem=sem)
        core(Z0, ctx, sem=_other(sem))                                    # another decision in between
        p, _ = core(Z0, ctx)                                              # a world-model style call without sem
        a2, _ = core(Z0, ctx, sem=sem)
        p2, _ = core(Z0, ctx)
    assert torch.equal(a1, a2) and torch.equal(p, p2)


def test_imagination_and_evaluator_batches_select_their_semantic_rows():
    """sem with the environment rows; batches of B*K rows (candidates / branches) map to them by code_rows or an
    explicit sem_rows, and each branch row equals its environment row's result; a mismatch without a mapping is
    refused (it crashed with a shape error before)."""
    cfg, core, ctx, sem, B = _setup(4)
    K = 4
    rows = torch.arange(B).repeat_interleave(K)                            # B*K = 12 rows
    Z0 = core.init_state(B)
    with torch.no_grad():
        Z_env, _ = core(Z0, ctx, sem=sem)
        Zb, _ = core(Z0[rows], ctx.select(rows.tolist()), code_rows=rows, sem=sem)
        Zs, _ = core(Z0[rows], ctx.select(rows.tolist()), code_rows=rows, sem=sem, sem_rows=rows)
        with pytest.raises(ValueError, match="sem_rows"):
            core(Z0[rows], ctx.select(rows.tolist()), sem=sem)
        # a source-aligned sem of the same size as the batch must name its rows explicitly
        perm = torch.tensor([2, 0, 1])
        Zp, _ = core(Z0[perm], ctx.select(perm.tolist()), code_rows=perm, sem=sem, sem_rows=perm)
        # the reasoner and transition blocks take the same mapping
        rs = Reasoner(cfg, core)
        rs.eval()
        R_env, _ = rs(Z_env, None, n_iters=2, sem=sem)
        R_b, _ = rs(Z_env[rows], None, n_iters=2, code_rows=rows, sem=sem)
        T_env, _ = core(Z_env, ctx, sem=sem, blocks=core.transition_blocks(4))
        T_b, _ = core(Z_env[rows], ctx.select(rows.tolist()), code_rows=rows, sem=sem,
                      blocks=core.transition_blocks(4))
    assert Zb.shape == (B * K, cfg.n_state, cfg.d_model)
    assert torch.allclose(Zb, Z_env[rows], atol=1e-5) and torch.allclose(Zs, Zb, atol=1e-6)
    assert torch.allclose(Zp, Z_env[perm], atol=1e-5)
    assert torch.allclose(R_b, R_env[rows], atol=1e-5)
    assert torch.allclose(T_b, T_env[rows], atol=1e-5)


def test_transition_blocks_are_the_last_k_backbone_blocks_with_their_cross_attention():
    cfg, core, ctx, sem, B = _setup(5)
    tb = core.transition_blocks(3)
    assert [id(b) for b in tb] == [id(b) for b in core.blocks[-3:]]
    assert core.block_ids(tb) == [5, 6, 7]
    with pytest.raises(ValueError):
        core.transition_blocks(cfg.n_layers + 1)
    Z0 = core.init_state(B)
    Zt, _ = core(Z0, ctx, sem=sem, blocks=tb)
    assert sorted(core.last_gates) == [5, 7]                              # cross-attention after blocks 5 and 7
    Zt.pow(2).mean().backward()
    assert core.blocks[0].attn.q.weight.grad is None                      # blocks outside the subset unused
    assert core.blocks[-1].attn.q.weight.grad.abs().sum() > 0
    assert core.xattn["7"].attn.kv.weight.grad.abs().sum() > 0


def test_deep_copy_target_core_computes_the_same_function_with_sem():
    """The EMA target (a deep copy) is given the same explicit sem: identical outputs at identical weights (the old
    module-state design differed by 1.2% because the copy had no context)."""
    cfg, core, ctx, sem, B = _setup(6)
    target = copy.deepcopy(core)
    Z0 = core.init_state(B)
    with torch.no_grad():
        a, _ = core(Z0, ctx, sem=sem)
        b, _ = target(Z0, ctx, sem=sem)
    assert torch.equal(a, b)


@pytest.mark.parametrize("core_ckpt", [False, True])
def test_reasoner_iteration_checkpoint_gradient_equals_plain_gradient_with_sem(core_ckpt):
    """Adaptive reasoning with iteration checkpointing and sem: values and gradients (projection, cross-attention,
    gates, reused blocks, halting head) equal the non-checkpointed ones at rtol 1e-4, also when another decision with
    a different semantic context of the same shape is encoded between forward and backward (the old design raised
    CheckpointError or gave a 141% wrong sem_proj gradient)."""
    torch.manual_seed(11)
    cfg = make_config("dev", n_layers=4, d_model=64, n_heads=4, n_state=8, reason_layers=2, sem_xattn_every=2,
                      sem_xattn_dim=48, max_ctx_tokens=64, grad_checkpoint=core_ckpt)
    core = UnifiedCore(cfg, autocast=False).train()
    rs = Reasoner(cfg, core).train()
    B = 5
    ctx = TokenSet.from_dense(torch.randn(B, 6, cfg.d_model), int(TokenKind.BELIEF))
    sem = SemanticContext(torch.randn(B, 9, 48), torch.rand(B, 9) > 0.3, torch.randn(B, cfg.d_model))
    sem2 = SemanticContext(torch.randn(B, 9, 48), sem.mask.clone(), torch.randn(B, cfg.d_model))
    Z = torch.randn(B, cfg.n_state, cfg.d_model)
    params = [p for p in list(core.parameters()) + list(rs.parameters()) if p.requires_grad]
    out = {}
    for ck in (False, True):
        rs.checkpoint_iters = ck
        for p in params:
            p.grad = None
        R, info = rs(Z, ctx, sem=sem)
        # another decision encoded before backward, with another semantic context of the same shape
        rs(Z, ctx, sem=sem2)
        core(core.init_state(B), ctx, sem=sem2)
        (R.pow(2).mean() + info["ponder"].mean()).backward()
        out[ck] = (R.detach(), [None if p.grad is None else p.grad.clone() for p in params], info["n_computed"])
    assert torch.equal(out[False][2], out[True][2])
    assert torch.allclose(out[False][0], out[True][0], atol=1e-6)
    names = [n for n, p in list(core.named_parameters()) + list(rs.named_parameters()) if p.requires_grad]
    for n, a, b in zip(names, out[False][1], out[True][1]):
        assert (a is None) == (b is None), n
        if a is not None:
            assert torch.allclose(a, b, rtol=1e-4, atol=1e-7), (n, float((a - b).abs().max()))
    g = dict(zip(names, out[True][1]))
    assert g["sem_proj.proj.weight"] is not None and g["sem_proj.proj.weight"].abs().sum() > 0
    assert any(n.startswith("xattn.") and v is not None and v.abs().sum() > 0 for n, v in g.items())


def test_reasoner_reads_sem_and_drops_halted_rows_from_it():
    torch.manual_seed(12)
    cfg = make_config("dev", n_layers=4, d_model=64, n_heads=4, n_state=8, reason_layers=2, sem_xattn_every=2,
                      sem_xattn_dim=48, max_ctx_tokens=64)
    core = UnifiedCore(cfg, autocast=False)
    rs = Reasoner(cfg, core).eval()
    B = 6
    sem = SemanticContext(torch.randn(B, 9, 48), torch.ones(B, 9, dtype=torch.bool), None)
    Z = torch.randn(B, cfg.n_state, cfg.d_model)
    easy = torch.arange(B) < 3
    v = torch.randn(cfg.d_model)
    v = v / v.norm()
    with torch.no_grad():
        rs.halt_head.weight.copy_(5.0 * v.unsqueeze(0))
        rs.halt_head.bias.fill_(-3.0)
        mark = lambda j, Rj: Rj + (easy.float().view(B, 1, 1) * 10.0) * v if Rj.shape[0] == B else Rj  # noqa: E731
        with rs.intervention(mark):
            R, info = rs(Z, None, sem=sem)
            gates = dict(rs.last_gates)
            R_other, _ = rs(Z, None, sem=_other(SemanticContext(sem.h, sem.mask, torch.zeros(B, 64)), 3))
    assert torch.all(info["n_computed"][easy] == 1) and torch.all(info["n_computed"][~easy] > 1)
    assert float((R - R_other).abs().max()) > 1e-4                         # the semantic context is read
    assert set(gates) == set(range(1, int(info["n_computed"].max()) + 1))
    assert all(set(g) == {3} for g in gates.values())                      # cross-attention after reused block 3


def test_sem_is_ignored_and_counted_when_mechanism_c_is_off():
    torch.manual_seed(0)
    cfg = make_config("dev")
    core = UnifiedCore(cfg, autocast=False)
    B = 2
    ctx = TokenSet.from_dense(torch.randn(B, 4, cfg.d_model), int(TokenKind.OBS_NUM))
    sem = SemanticContext(torch.randn(B, 5, 16), torch.ones(B, 5, dtype=torch.bool))
    with torch.no_grad():
        a, _ = core(core.init_state(B), ctx, sem=sem)
        b, _ = core(core.init_state(B), ctx)
    assert torch.equal(a, b) and core.sem_stats["calls_sem_ignored"] == 1


def test_semantic_context_rows_and_projected_memory_select():
    sem = SemanticContext(torch.randn(3, 4, 8), torch.ones(3, 4, dtype=torch.bool), torch.randn(3, 6),
                          refs=[["a"] * 4, ["b"] * 4, ["c"] * 4])
    s = sem.select([2, 2, 0])
    assert s.B == 3 and torch.equal(s.h[0], sem.h[2]) and s.refs[2][0] == "a"
    kv = SemanticKV(torch.randn(3, 4, 6), sem.mask, None)
    assert kv.select(torch.tensor([1])).S.shape == (1, 4, 6)


def test_dom_dev_preset_core_has_cross_attention_every_fourth_block():
    cfg = make_preset("dom_dev")
    core = UnifiedCore(cfg, autocast=False)
    assert core.xattn_layers == [3, 7] and sorted(core.xattn) == ["3", "7"]
    assert core.sem_proj.proj.in_features == 1024 and core.state_gate is not None
    # the reasoner's reused blocks (4..7) include cross-attention layer 7; transition_blocks(6) layers 3 and 7
    assert core.block_ids(Reasoner(cfg, core).blocks) == [4, 5, 6, 7]
    assert [i for i in core.block_ids(core.transition_blocks(6)) if str(i) in core.xattn] == [3, 7]


def test_superlearner_scale_has_about_one_billion_trainable_core_parameters():
    cfg = make_config("superlearner", sem_xattn_every=2, sem_xattn_dim=2560)
    assert (cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_state, cfg.d_ff) == (32, 1536, 24, 128, 4096)
    with torch.device("meta"):
        core = GatedUnifiedCore(cfg, autocast=False)
    n = core.count_parameters()["total"]
    assert 0.9e9 <= n <= 1.2e9, n
