"""Init cosine of NUMERIC tokens for value pairs under candidate value-channel designs (random init, CPU)."""
import torch
import torch.nn as nn

from superlearner.cam_unified.config import make_preset
from superlearner.cam_unified.semantic.layers import NumericFeatures, RMSNorm

cfg = make_preset("dom_dev")
D = cfg.d_model
nf = NumericFeatures(cfg.n_fourier)
print("n_fourier", cfg.n_fourier, "feat dim", nf.dim)
pairs = [(1.0, -1.0), (100.0, 101.0), (100.0, 200.0), (0.5, 0.25), (1e6, 2e6), (0.0, 1.0), (3.0, 30.0), (-5.0, 5.0)]


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1))


for seed in range(3):
    torch.manual_seed(seed)
    old_mlp = nn.Sequential(nn.Linear(nf.dim, D), nn.SiLU(), nn.Linear(D, D))
    ctx = torch.randn(D) * 0.58 * 2 ** 0.5                     # unit_proj + desc_proj output scale (two ~0.58 terms)
    norm = RMSNorm(D)
    vn, cn = RMSNorm(D), RMSNorm(D)
    proj = nn.Linear(2 * D, D, bias=False)
    nn.init.normal_(proj.weight, std=(2 * D) ** -0.5)
    lin = nn.Linear(nf.dim, D, bias=False)
    nn.init.normal_(lin.weight, std=nf.dim ** -0.5)
    mlp2 = nn.Sequential(nn.Linear(nf.dim, D), nn.SiLU(), nn.Linear(D, D))
    rows = {}
    for a, b in pairs:
        fa, fb = nf(torch.tensor([a])), nf(torch.tensor([b]))
        old = [norm(old_mlp(f) + ctx) for f in (fa, fb)]
        cat_mlp = [norm(proj(torch.cat([vn(old_mlp(f)), cn(ctx[None])], -1))) for f in (fa, fb)]
        cat_lin = [norm(proj(torch.cat([vn(lin(f)), cn(ctx[None])], -1))) for f in (fa, fb)]
        raw = cos(fa, fb)
        rows[(a, b)] = (round(cos(*old), 3), round(cos(*cat_mlp), 3), round(cos(*cat_lin), 3), round(raw, 3))
    print("seed", seed, "(old, cat_mlp, cat_lin, raw features)")
    for k, v in rows.items():
        print("   ", k, v)
