"""Multilayer latent conditioning of the semantic decoder (DOM amendment Mechanism A; manifest DOM-03, DOM-09).

At decoder layers l in `layers` (QwenSemantic attaches it after every cfg.dec_xattn_every-th layer):

    H'_l = H_l + tanh(alpha_l) * CrossAttention_l( norm(H_l), P_l(Z_t), P_l(Z_t) )

* Z_t are CAM latent / action-intent tokens [B, n, d_cam] (the reasoned state R_t, intent tokens I_t, belief and goal
  tokens) - the decoder reads CAM's numerical state at many depths, not only through a prefix it could ignore.
* P_l is a trainable projection of CAM's tokens into a per-layer key/value space; the cross-attention is low-rank
  (attention width `d_attn`), so 6 conditioned layers of a 2560-wide decoder cost ~50M trainable parameters.
* The injection is a plain gated residual. alpha_l starts at `alpha_init` = 0 (Flamingo-style tanh gate): at
  initialisation the conditioned decoder is the pretrained decoder bit for bit (H + 0 = H), while tanh'(0) = 1 gives
  alpha_l a gradient from the first step (the projections receive gradient once alpha_l moves). Gates are reported
  (`gates()`).
  Repair of the first version (audit 2026-11-26): it injected tanh(alpha) * rms(H) * unit(a) with alpha = 0.05.
  Mid-layer LLM hidden states carry a few huge outlier dimensions, so rms(H) is inflated and spreading 5% of it over
  every dimension flipped 17-32% of the top-1 tokens on JSON text at "initialisation".

The conditioning is attached with forward hooks on the decoder layers (`attach(model)`), so it works unchanged with
KV-cached generation (each new token passes through the hooked layers and reads the same latent memory) and with the
frozen pretrained weights (requires_grad False): gradients of any decoder-output loss (action log-likelihood, policy
gradient through token log-probabilities, grounding objectives) flow through the frozen layers' activations into
alpha_l, the cross-attention, P_l and Z_t - i.e. into CAM's core. transformers >= 5 checkpoints a layer's `forward`,
not `__call__`, so the hooks run outside gradient checkpointing (computed once, nothing recomputed in backward).
`condition(Z, mask)` is a context manager that sets the latent memory for the decoder calls inside it (QwenSemantic
uses it inside each call with an explicit `cond=` argument); batch expansion (several candidates per environment) is
handled by repeat-interleaving the memory when the decoder batch is a multiple of the memory batch.
"""
from __future__ import annotations

import contextlib
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _RMS(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.w.to(x.dtype)


class LatentXAttn(nn.Module):
    def __init__(self, d_lm: int, d_cam: int, d_attn: int = 1024, n_heads: int = 8, alpha_init: float = 0.0):
        super().__init__()
        assert d_attn % n_heads == 0
        self.h, self.dh = n_heads, d_attn // n_heads
        self.norm_h = _RMS(d_lm)
        self.norm_z = _RMS(d_cam)
        self.q = nn.Linear(d_lm, d_attn, bias=False)
        self.kv = nn.Linear(d_cam, 2 * d_attn, bias=False)        # P_l: CAM latent -> this layer's key/value space
        self.o = nn.Linear(d_attn, d_lm, bias=False)
        self.null_kv = nn.Parameter(torch.zeros(2, d_attn))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        nn.init.normal_(self.q.weight, std=d_lm ** -0.5)
        nn.init.normal_(self.kv.weight, std=d_cam ** -0.5)
        nn.init.normal_(self.o.weight, std=d_attn ** -0.5)

    def forward(self, H: Tensor, Z: Tensor, z_mask: Optional[Tensor]) -> Tensor:
        """The residual delta [B, T, d_lm] (in H's dtype) for decoder states H [B, T, d_lm] reading Z [B, n, d_cam]."""
        B, T, _ = H.shape
        dt = self.q.weight.dtype
        q = self.q(self.norm_h(H).to(dt)).view(B, T, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(self.norm_z(Z).to(dt)).chunk(2, -1)
        nk, nv = self.null_kv[0].to(k.dtype), self.null_kv[1].to(v.dtype)
        k = torch.cat([nk.expand(B, 1, -1), k], 1).view(B, -1, self.h, self.dh).transpose(1, 2)
        v = torch.cat([nv.expand(B, 1, -1), v], 1).view(B, -1, self.h, self.dh).transpose(1, 2)
        mask = None
        if z_mask is not None:
            m = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=Z.device), z_mask.bool()], 1)
            mask = m[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=mask)
        a = self.o(a.transpose(1, 2).reshape(B, T, -1))
        return (torch.tanh(self.alpha).to(a.dtype) * a).to(H.dtype)


class LatentDecoderConditioning(nn.Module):
    def __init__(self, d_lm: int, d_cam: int, layers: Sequence[int], d_attn: int = 1024, n_heads: int = 8,
                 alpha_init: float = 0.0):
        super().__init__()
        self.layer_ids = [int(i) for i in layers]
        self.mods = nn.ModuleDict({str(i): LatentXAttn(d_lm, d_cam, d_attn, n_heads, alpha_init)
                                   for i in self.layer_ids})
        self._Z: Optional[Tensor] = None
        self._mask: Optional[Tensor] = None
        self._handles: list = []
        self.calls = 0

    @staticmethod
    def every(n_layers: int, k: int, first: int = 1) -> list:
        """Layer indices first, first+k, ... (k >= 1)."""
        return list(range(first, n_layers, k))

    def attach(self, lm) -> "LatentDecoderConditioning":
        """Register forward hooks on the decoder layers of a HF causal LM (Qwen3: lm.model.layers)."""
        self.detach_hooks()
        layers = _decoder_layers(lm)
        for i in self.layer_ids:
            mod = self.mods[str(i)]
            self._handles.append(layers[i].register_forward_hook(self._hook(mod)))
        return self

    def detach_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _hook(self, mod: LatentXAttn):
        def fn(module, args, output):
            if self._Z is None:
                return output
            H = output[0] if isinstance(output, tuple) else output
            Z, m = self._Z, self._mask
            if H.shape[0] != Z.shape[0]:
                if H.shape[0] % Z.shape[0]:
                    raise ValueError(f"decoder batch {H.shape[0]} is not a multiple of latent batch {Z.shape[0]}")
                r = H.shape[0] // Z.shape[0]
                Z = Z.repeat_interleave(r, 0)
                m = None if m is None else m.repeat_interleave(r, 0)
            self.calls += 1
            out = H + mod(H, Z, m)
            return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out
        return fn

    @contextlib.contextmanager
    def condition(self, Z: Optional[Tensor], mask: Optional[Tensor] = None):
        prev = (self._Z, self._mask)
        self._Z, self._mask = Z, mask
        try:
            yield self
        finally:
            self._Z, self._mask = prev

    def gates(self) -> dict:
        return {i: float(torch.tanh(self.mods[str(i)].alpha.detach())) for i in self.layer_ids}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _decoder_layers(lm) -> nn.ModuleList:
    for path in ("model.layers", "base_model.model.model.layers", "model.model.layers", "transformer.h"):
        obj = lm
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and isinstance(obj, nn.ModuleList):
            return obj
    raise AttributeError("could not find the decoder layers of this language model")
