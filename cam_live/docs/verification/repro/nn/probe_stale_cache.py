"""G-nn probe: adapter caches rely on tensor version counters; a write through ``.data`` leaves them stale.

    .venv/bin/python cam_live/docs/verification/repro/nn/probe_stale_cache.py

(No code in the repository writes adapters through ``.data`` today; this documents the hazard behind the statement
"the cache is invalidated automatically when any adapter tensor changes".)
"""
import sys

import _common  # noqa: F401
from _common import check, finish, info

import torch

from cam_live.config import CamLiveConfig
from cam_live.nn.model import build_model

T = ["the page loaded", "click the button"]
# supported route: optimizer step / in-place op under no_grad -> version bump -> automatic invalidation
m = build_model(CamLiveConfig.debug())
e0 = m.embed_texts(T).clone()
with torch.no_grad():
    for mod in m.backbone.lora_modules:
        mod.lora_B.add_(0.3)
check("in-place update under no_grad invalidates the embedding cache automatically", not torch.equal(e0, m.embed_texts(T)))

# a write through .data on a fresh model (adapters still zero, so the LoRA inference path has cached "skip")
m = build_model(CamLiveConfig.debug())
bb = m.backbone
e1 = m.embed_texts(T).clone()
with torch.no_grad():
    bb.next_token_logits(["the page"])
g = torch.Generator().manual_seed(0)
for mod in bb.lora_modules:
    mod.lora_B.data.copy_(torch.randn(mod.lora_B.shape, generator=g) * 0.5)        # no version bump
e2 = m.embed_texts(T).clone()
eg = m.embed_texts(T, with_grad=True).detach()
with torch.no_grad():
    l_ng = bb.next_token_logits(["the page"])
l_g = bb.next_token_logits(["the page"]).detach()
check("a write through .data invalidates the embedding cache", not torch.equal(e1, e2),
      f"cached embedding unchanged; max |fresh - cached| = {float((eg - e2).abs().max()):.2f}")
check("after a write through .data the inference (no_grad) path and the training path use the same adapter weights",
      float((l_ng - l_g).abs().max()) < 1e-4, f"max |logits(no_grad) - logits(grad mode)| = {float((l_ng - l_g).abs().max()):.3f}")
m.mark_adapters_updated()
with torch.no_grad():
    after = float((bb.next_token_logits(["the page"]) - l_g).abs().max())
info("after mark_adapters_updated(): max |embedding - fresh|, max |logits(no_grad) - logits(grad mode)| (the LoRA cast cache is keyed on the "
     "version counter and is not cleared by mark_adapters_updated)", (float((m.embed_texts(T) - eg).abs().max()), after))
sys.exit(finish())
