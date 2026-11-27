"""Init cosine of NUMERIC tokens under the sign-separated value-channel design (random init, CPU)."""
import math

import torch
import torch.nn as nn

from superlearner.cam_unified.config import make_preset
from superlearner.cam_unified.semantic.layers import RMSNorm

cfg = make_preset("dom_dev")
D = cfg.d_model


class VF(nn.Module):
    def __init__(self, K=24, lo=0.02, hi=200.0):
        super().__init__()
        self.register_buffer("freq", torch.exp(torch.linspace(math.log(lo), math.log(hi), K)))
        self.dim = 2 * (2 * K + 1) + 5

    def forward(self, v):
        v = v.double()
        fin = torch.isfinite(v)
        flags = torch.stack([torch.isnan(v), torch.isposinf(v), torch.isneginf(v)], -1).double()
        v = torch.where(fin, v, torch.zeros_like(v))
        u = torch.log1p(v.abs())
        ang = u.unsqueeze(-1) * self.freq.double()
        m = torch.cat([torch.sin(ang), torch.cos(ang), (u / 10).unsqueeze(-1)], -1)
        s = torch.sign(v).unsqueeze(-1)
        is_int = (torch.abs(v - torch.round(v)) < 1e-9).double().unsqueeze(-1)
        f = torch.cat([m, s * m, (v == 0).double().unsqueeze(-1), is_int], -1) * fin.double().unsqueeze(-1)
        return torch.cat([f, flags], -1).float()


vf = VF()
pairs = [(1.0, -1.0), (100.0, 101.0), (100.0, 200.0), (0.5, 0.25), (1e6, 2e6), (0.0, 1.0), (3.0, 30.0), (-5.0, 5.0),
         (1200, 1250)]


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).detach())


for g in (0.6, 0.7, 1.0):
    worst = 0
    for seed in range(5):
        torch.manual_seed(seed)
        lin = nn.Linear(vf.dim, D, bias=False)
        nn.init.normal_(lin.weight, std=vf.dim ** -0.5)
        proj = nn.Linear(2 * D, D, bias=False)
        nn.init.normal_(proj.weight, std=(2 * D) ** -0.5)
        vn, cn, norm = RMSNorm(D), RMSNorm(D), RMSNorm(D)
        ctx = torch.randn(1, D)
        ctx2 = torch.randn(1, D)
        tok = lambda v, c: norm(proj(torch.cat([vn(lin(vf(torch.tensor([v])))), g * cn(c)], -1)))
        r = {p: round(cos(tok(p[0], ctx), tok(p[1], ctx)), 3) for p in pairs}
        r["same value, other metric"] = round(cos(tok(100.0, ctx), tok(100.0, ctx2)), 3)
        worst = max(worst, r[(1.0, -1.0)])
        if seed == 0:
            print("g", g, r)
    print("   g", g, "max over 5 seeds cos(+1,-1)", worst)
