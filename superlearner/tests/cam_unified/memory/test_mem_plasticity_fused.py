"""Plasticity instruments over the REAL UnifiedCore's fused SwiGLU (audit 'archive': find_unit_layers found 0 of the
dev core's 8 layers because w13 is fused; plan 2026-11-26 Memory "Plasticity: fused w13 support, monitor only")."""
from __future__ import annotations

import torch

from superlearner.cam_unified.config import make_config
from superlearner.cam_unified.core.backbone import UnifiedCore
from superlearner.cam_unified.core.blocks import SwiGLU
from superlearner.cam_unified.memory.plasticity import PlasticityMonitor, find_unit_layers, reinit_units


def test_monitor_finds_every_fused_swiglu_layer_of_the_dev_core():
    cfg = make_config("dev")
    core = UnifiedCore(cfg)
    layers = [l for l in find_unit_layers(core) if l.fused]
    n_swiglu = sum(isinstance(m, SwiGLU) for m in core.modules())
    assert n_swiglu >= cfg.n_layers and len(layers) == n_swiglu
    assert all(l.n_units == cfg.d_ff and l.fc_in.out_features == 2 * cfg.d_ff for l in layers)


def test_a_zeroed_fused_unit_is_dormant_and_reinit_resets_both_of_its_rows():
    torch.manual_seed(0)
    ff = SwiGLU(16, 32)
    with torch.no_grad():
        ff.w13.weight[3] = 0.0            # gate row of unit 3 -> silu(0) * b = 0 always
    (layer,) = find_unit_layers(ff)
    assert layer.fused and layer.n_units == 32
    mon = PlasticityMonitor(ff, [layer]).attach()
    ff(torch.randn(256, 16))
    mon.detach()
    d = mon.dormant(0.025)
    assert bool(d["mask"][layer.name][3]) and d["mask"][layer.name].sum() == 1
    before = ff.w13.weight.detach().clone()
    opt = torch.optim.Adam(ff.parameters())
    ff(torch.randn(4, 16)).sum().backward()
    opt.step()
    reinit_units(layer, torch.tensor([3]), opt, torch.Generator().manual_seed(1))
    w = ff.w13.weight.detach()
    assert not torch.equal(w[3], before[3]) and not torch.equal(w[3 + 32], before[3 + 32])
    assert float(ff.w2.weight[:, 3].abs().sum()) == 0.0
    st = opt.state[ff.w13.weight]
    assert float(st["exp_avg"][3].abs().sum()) == 0.0 and float(st["exp_avg"][35].abs().sum()) == 0.0
    assert float(st["exp_avg"][4].abs().sum()) > 0.0                   # other units keep their moments
