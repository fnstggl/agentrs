"""G-nn probe: every scale preset is really instantiated on CPU (weights allocated, not shape-only), counted, run
forward (core -> decision trunk -> feature map) and, up to 150m, backward.

    .venv/bin/python cam_live/docs/verification/repro/nn/probe_presets.py
"""
import sys
import time

import _common  # noqa: F401
from _common import check, finish, info

import torch

from cam_live.config import SCALE_PRESETS, CoreConfig
from cam_live.nn.api import TokenBatch
from cam_live.nn.model import build_experiential
from cam_live.nn.presets import NNConfig, preset_nn, resolve_core

CLAIMED = {"5m": 5044741, "20m": 20087333, "50m": 50953765, "150m": 151022853, "500m": 502708549, "1b": 1000547333}
torch.set_num_threads(8)
for scale, claimed in CLAIMED.items():
    core = CoreConfig().apply_scale(scale)
    rc = resolve_core(core)
    t0 = time.time()
    net = build_experiential(core, NNConfig(**preset_nn(core)), 7680, 3)
    n = sum(p.numel() for p in net.parameters())
    target = SCALE_PRESETS[scale]["target_params"]
    check(f"preset {scale}: instantiated count == claimed {claimed:,} and within 5 % of the target {target:,.0f}",
          n == claimed and abs(n / target - 1) < 0.05, f"{n:,} (d_model {rc.d_model}, layers {rc.n_layers}, d_ff {rc.d_ff}; build {time.time() - t0:.1f} s)")
    B, T, D = 2, 12, rc.d_model
    g = torch.Generator().manual_seed(0)
    tb = TokenBatch(x=torch.randn(B, T, D, generator=g), pad_mask=torch.ones(B, T, dtype=torch.bool),
                    time=(torch.arange(T, dtype=torch.float64) * 60.0).expand(B, T).clone(), kind_ids=torch.zeros(B, T, dtype=torch.long),
                    values=torch.zeros(B, T, 4, dtype=torch.float64), value_mask=torch.zeros(B, T, 4, dtype=torch.bool))
    small = n <= 160e6
    with torch.set_grad_enabled(small):
        out = net.core(tb)
        z = net.trunk(out.b, torch.randn(B, 3, D, generator=g), torch.randn(B, D, generator=g))
        phi = net.feature_map(z, torch.randn(B, 2, D, generator=g))
    ok = bool(torch.isfinite(phi).all()) and phi.shape == (B, 3, 2, rc.n_features) and out.state.slots.shape == (B, rc.n_slots, D)
    if small:
        phi.pow(3).sum().backward()
        gn = sum(float(p.grad.pow(2).sum()) for p in net.parameters() if p.grad is not None) ** 0.5
        ok = ok and gn > 0 and gn == gn
    check(f"preset {scale}: forward{' + backward' if small else ''} finite with the right shapes", ok)
    del net, out, z, phi
info("master config.SCALE_PRESETS d_ff values still differ from the corrected nn/presets.py ones (change request A-1)",
     {s: (SCALE_PRESETS[s]["d_ff"], resolve_core(CoreConfig().apply_scale(s)).d_ff) for s in CLAIMED})
sys.exit(finish())
