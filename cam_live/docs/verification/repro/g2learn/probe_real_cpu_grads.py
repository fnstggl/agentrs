"""G2-learn probe: the REAL backbone (Qwen/Qwen3-4B, loaded on the CPU in float32 by the same code path) — gradient
paths into the semantic adapters (NA-09) and generation through the bridge (NA-01 / NA-12), checked by central finite
differences along the analytic gradient. Runs without the GPU (used while the shared GPU was held by other tenants).

    .venv/bin/python cam_live/docs/verification/repro/g2learn/probe_real_cpu_grads.py

  1. embedding path: L = <w, raw pooled multi-depth embedding of the texts> (the function whose gradient
     ``embed(with_grad=True)`` attaches): d L / d LoRA-B by autograd vs (L(B + h u) - L(B - h u)) / 2h, u = g / |g|;
     and ``embed(with_grad=True)`` carries exactly that gradient (value = the canonical cached embedding);
  2. generation path: L = generation_logprob(prompt, completion, latent): d L / d (LoRA-B, bridge output projections)
     by autograd vs finite differences, at initialisation (B = 0, bridge output = 0) — non-zero gradients;
  3. greedy generation with and without a latent is identical at initialisation, and returns text.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import check, finish, info  # noqa: E402

import torch  # noqa: E402

from cam_live.config import CamLiveConfig  # noqa: E402
from cam_live.nn.api import GROUP_ADAPTERS  # noqa: E402
from cam_live.nn.model import build_model  # noqa: E402

torch.set_num_threads(16)
t0 = time.time()
TINY = "--tiny" in sys.argv          # dry run of this probe's code on the CPU test backbone
cfg = CamLiveConfig.debug() if TINY else CamLiveConfig(device="cpu")
model = build_model(cfg)
bb = model.backbone
info("load (s), backbone dtype on CPU", (round(time.time() - t0, 1), str(next(bb._lm_module().parameters()).dtype)))
lora_ids = {id(p) for p in bb.lora_parameters()}
B = [m.lora_B for m in bb.lora_modules]
bridge_out = [p for n, p in bb.bridge.named_parameters() if "out" in n and p.dim() == 2]
info("bridge output projections (zero at init)", (len(bridge_out), float(sum(p.abs().sum() for p in bridge_out))))

texts = ["The export job writes its log file to the data directory.", "Three members joined after the post."]
rows = [bb._text_ids(t, cfg.backbone.max_text_tokens) for t in texts]
gen = torch.Generator().manual_seed(0)
W = torch.randn(len(texts), bb.d_sem, generator=gen)


def emb_loss():
    return (W * bb._pooled_dynamic(rows)).sum()


# 1. embedding path
for p in B:
    p.grad = None
with torch.enable_grad():
    L = emb_loss()
    gB = torch.autograd.grad(L, B)
gn = torch.sqrt(sum((g.double() ** 2).sum() for g in gB))
u = [g / gn for g in gB]
fd = {}
for h in (1e-2, 3e-3):
    with torch.no_grad():
        for p, d in zip(B, u):
            p.add_(h * d.to(p.dtype))
        lp = float(emb_loss())
        for p, d in zip(B, u):
            p.sub_(2 * h * d.to(p.dtype))
        lm = float(emb_loss())
        for p, d in zip(B, u):
            p.add_(h * d.to(p.dtype))
    fd[h] = (lp - lm) / (2 * h)
ratio = {h: v / float(gn) for h, v in fd.items()}
info("embedding path: |dL/dB| autograd, finite differences along it (h: value)", (float(gn), fd))
check("NA-09: embedding path -> LoRA-B gradient equals central finite differences to 2 % (float32 backbone, CPU)",
      all(abs(r - 1.0) < 0.02 for r in ratio.values()), f"FD / analytic = {ratio}")
# embed(with_grad=True) carries the gradient of that function and the canonical value
with torch.enable_grad():
    e = bb.embed(texts, with_grad=True)
    L2 = (W * (e * bb.cal_sd)).sum()          # undo the per-dimension standardisation scale: same functional of raw
    g2 = torch.autograd.grad(L2, B)
rel = float(torch.sqrt(sum(((a - b) ** 2).sum() for a, b in zip(g2, gB))) / gn)
with torch.no_grad():
    canon = bb.embed(texts)
check("NA-09: embed(with_grad=True) has the canonical (cached) value and the gradient of the pooled forward",
      torch.equal(e.detach(), canon) and rel < 1e-3, f"value equal {torch.equal(e.detach(), canon)}, gradient rel. diff {rel:.2e}")
check("NA-09: all LoRA modules receive gradient from the embedding path",
      sum(int(bool(g.abs().sum() > 0)) for g in gB) == len(B), f"{sum(int(bool(g.abs().sum() > 0)) for g in gB)} of {len(B)}")

# 2. generation path at initialisation
prompt = bb.chat_prompt("Name one colour.")
comp = " Blue."
lat = torch.randn(1, model.core_cfg.d_model, generator=gen)
params = B + bridge_out


def gen_loss():
    return bb.generation_logprob([prompt], [comp], latent=lat).sum()


with torch.enable_grad():
    G = gen_loss()
    gp = torch.autograd.grad(G, params, allow_unused=True)
gp = [torch.zeros_like(p) if g is None else g for g, p in zip(gp, params)]
nB = float(torch.sqrt(sum((g.double() ** 2).sum() for g in gp[:len(B)])))
nO = float(torch.sqrt(sum((g.double() ** 2).sum() for g in gp[len(B):])))
info("generation path at init: |d logp / d LoRA-B|, |d logp / d bridge output|", (nB, nO))
check("NA-09 / NA-12: generation_logprob gives non-zero gradient to LoRA-B and to the zero-initialised bridge output",
      nB > 0 and nO > 0, f"{nB:.4g}, {nO:.4g}")
gn2 = torch.sqrt(sum((g.double() ** 2).sum() for g in gp))
u2 = [g / gn2 for g in gp]
fd2 = {}
for h in (1e-2, 3e-3):
    with torch.no_grad():
        for p, d in zip(params, u2):
            p.add_(h * d.to(p.dtype))
        lp = float(gen_loss())
        for p, d in zip(params, u2):
            p.sub_(2 * h * d.to(p.dtype))
        lm = float(gen_loss())
        for p, d in zip(params, u2):
            p.add_(h * d.to(p.dtype))
    fd2[h] = (lp - lm) / (2 * h) / float(gn2)
check("NA-09 / NA-12: generation path gradient (LoRA-B + bridge output) equals finite differences to 2 %",
      all(abs(r - 1.0) < 0.02 for r in fd2.values()), f"FD / analytic = {fd2}")

# 3. generation, bridge no-op
with torch.no_grad():
    a = bb.generate([prompt], max_new_tokens=6, temperature=0.0)
    b = bb.generate([prompt], latent=lat * 100.0, max_new_tokens=6, temperature=0.0)
info("greedy continuation", a)
check("NA-01 / NA-12: generation returns text and is identical with and without a latent at initialisation",
      bool(a and a[0] and a[0][0].strip()) and a == b, f"{a} vs {b}")
sys.exit(finish())
