"""A01: the main (16 x 768 x 12, 64 state tokens) and large (24 x 1024 x 16, 128 state tokens) configurations are
instantiated and exercised on the GPU — one full workstream-B step (entities -> backbone -> beliefs -> reasoner ->
loss -> backward -> AdamW) at a realistic context — with parameter counts per component and peak memory.
Heavy: runs under the shared GPU lock (fixture `gpu_heavy_lock`)."""
import json

import pytest
import torch

from superlearner.cam_unified.config import make_config
from superlearner.cam_unified.core.bench import benchmark, build_stack, parameter_report, synthetic_context
from superlearner.cam_unified.interfaces import TokenSet


def _full_step(stack, B, N, J):
    """One full workstream-B step; J = None runs the default adaptive reasoning (with its compute penalty)."""
    cfg = stack["cfg"]
    core = stack["core"]
    obs, rest = synthetic_context(cfg, B, N, "cuda")
    ents = stack["entities"](obs)
    fast = stack["beliefs"].init_fast(B, "cuda")
    btok = stack["beliefs"].tokens(fast)
    ctx = TokenSet.cat(TokenSet(obs.x, obs.mask, obs.kind, obs.time, obs.entity, None, obs.entity2), ents, btok, rest)
    Z, ctx_out = core(core.init_state(B, "cuda"), ctx)
    R, info = stack["reasoner"](Z, TokenSet.cat(btok, rest), n_iters=J)
    loss = R.pow(2).mean() + 0.1 * ctx_out.pow(2).mean() + info["ponder"].mean()
    loss.backward()
    return Z, R, ctx, loss, info


def stack_max_iters():
    return make_config("main").max_reason_iters


def test_main_scale_instantiates_runs_forward_backward_and_reports_parameters(gpu_heavy_lock):
    stack = build_stack("main", "cuda")
    cfg = stack["cfg"]
    assert (cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_state) == (16, 768, 12, 64)
    rep = parameter_report(stack)
    core = rep["core"]
    print("main parameters:", json.dumps(rep))
    assert core["blocks"] == 16 * core["per_block"]
    assert core["per_block"] == 4 * 768 ** 2 + 3 * 768 * 2048 + 2 * 768 + 2 * 64
    assert 110e6 < core["total"] < 120e6, core["total"]
    assert 120e6 < rep["workstream_B_total"] < 180e6, rep["workstream_B_total"]
    torch.cuda.reset_peak_memory_stats()
    Z, R, ctx, loss, _ = _full_step(stack, B=8, N=384, J=4)
    assert Z.shape == (8, 64, 768) and R.shape == (8, 64, 768) and 340 <= ctx.N <= 400
    assert torch.isfinite(loss)
    for name in ("core", "entities", "beliefs"):
        grads = [p.grad for p in stack[name].parameters() if p.requires_grad]
        assert any(g is not None and torch.isfinite(g).all() and g.abs().max() > 0 for g in grads), name
    assert all(torch.isfinite(p.grad).all() for p in stack["core"].parameters() if p.grad is not None)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"main full step B=8 ctx={ctx.N}: peak {peak:.2f} GB")
    assert peak < 6.0
    # the default training configuration: adaptive reasoning (bounded per-row unroll, iteration checkpointing)
    for p in stack["core"].parameters():
        p.grad = None
    torch.cuda.reset_peak_memory_stats()
    Z, R, ctx, loss, info = _full_step(stack, B=8, N=384, J=None)
    peak_ad = torch.cuda.max_memory_allocated() / 2 ** 30
    n_it = int(info["n_computed"].max())
    print(f"main full step, adaptive reasoning: {n_it} iterations computed (E[iters] "
          f"{float(info['expected_iters'].mean()):.2f}), peak {peak_ad:.2f} GB")
    assert torch.isfinite(loss) and 1 < n_it < stack["cfg"].max_reason_iters
    assert stack["reasoner"].halt_head.weight.grad is not None
    assert peak_ad < 6.0
    del stack
    torch.cuda.empty_cache()
    r = benchmark("main", B=8, N=384, steps=5, warmup=2)
    print("main benchmark:", json.dumps({k: v for k, v in r.items() if k != "params"}))
    assert r["train"]["peak_mem_GB"] < 6.0 and r["train"]["tokens_per_s"] > 0
    # default adaptive reasoning at the initial parameters (no optimizer step: it would retrain the halting depth on
    # the synthetic loss), and its worst case: every row runs max_reason_iters (memory bounded by iteration
    # checkpointing, compute grows with the depth)
    ra = benchmark("main", B=8, N=384, steps=5, warmup=2, J=None, optimizer=False)
    print("main benchmark, adaptive reasoning:", json.dumps({k: v for k, v in ra.items() if k != "params"}))
    rw = benchmark("main", B=8, N=384, steps=5, warmup=2, J=None, optimizer=False, halt_bias=-20.0)
    print("main benchmark, adaptive worst case:", json.dumps({k: v for k, v in rw.items() if k != "params"}))
    assert rw["train"]["reason_iters_computed"] == stack_max_iters()
    assert ra["train"]["peak_mem_GB"] < 6.0 and rw["train"]["peak_mem_GB"] < 6.0


@pytest.mark.slow
def test_large_scale_instantiates_runs_and_fits_with_checkpointing(gpu_heavy_lock):
    stack = build_stack("large", "cuda", grad_checkpoint=True)
    cfg = stack["cfg"]
    assert (cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_state) == (24, 1024, 16, 128)
    rep = parameter_report(stack)
    print("large parameters:", json.dumps(rep))
    assert 290e6 < rep["core"]["total"] < 320e6
    assert 320e6 < rep["workstream_B_total"] < 450e6
    with torch.no_grad():
        obs, rest = synthetic_context(cfg, 8, 384, "cuda")
        Z, _ = stack["core"](stack["core"].init_state(8, "cuda"), TokenSet.cat(
            TokenSet(obs.x, obs.mask, obs.kind, obs.time, obs.entity, None, obs.entity2), rest))
        assert Z.shape == (8, 128, 1024) and torch.isfinite(Z).all()
    torch.cuda.reset_peak_memory_stats()
    Z, R, ctx, loss, _ = _full_step(stack, B=8, N=384, J=4)
    assert torch.isfinite(loss)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"large full step (checkpointing) B=8 ctx={ctx.N}: peak {peak:.2f} GB")
    assert peak < 8.0
    for m in ("core", "entities", "beliefs", "reasoner"):
        for p in stack[m].parameters():
            p.grad = None
    torch.cuda.reset_peak_memory_stats()
    Z, R, ctx, loss, info = _full_step(stack, B=8, N=384, J=None)
    peak_ad = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"large full step, adaptive reasoning: {int(info['n_computed'].max())} iterations, peak {peak_ad:.2f} GB")
    assert torch.isfinite(loss) and peak_ad < 8.0
