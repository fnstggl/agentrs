"""LatentBridge: CAM -> semantic decoder conditioning (spec 2.2; DOM Mechanism A).

Two outputs, both functions of the reasoned state R [B, S, D], the goal tokens G and the belief tokens B:

* `forward` - the soft prefix P_t [B, n_prefix, sem_hidden] (float32) that grammar.HoleFiller splices into
  `inputs_embeds`. n_prefix learned queries cross-attend (ContextReader, `n_layers` blocks) to [R; G; B] (each tagged
  with a source embedding and masked):

      P_t = base + embedding_rms * out(norm(q))          out zero-initialised, no renormalisation

  The base is a neutral row (the mean token-embedding direction at the typical token-embedding RMS), not the
  embeddings of a phrase, and the residual starts at exactly zero, so at initialisation the prefix carries no text the
  decoder could copy. The residual is expressed in units of the embedding RMS (a fixed buffer): the decoder's input
  RMSNorm amplifies gradients w.r.t. inputs_embeds by ~1/embedding_rms, and this scale cancels that amplification in
  out's gradient. Repair of the first version (audit 2026-11-26): its base was initialised from the embeddings of
  " Plan, evidence and intent from the controller:" (2/80 probe actions copied the phrase into arguments) and every
  prefix vector was renormalised to the embedding RMS (bridge gradients 10^2-10^3 x the core's).
  The prefix is optional: any 16 prefix vectors perturb the pretrained decoder at initialisation (Qwen3-0.6B, JSON
  arguments, KL to no prefix per token: neutral rows 0.43, the old phrase 1.27, zeros 0.91; W4 probe 2026-11-26), so
  with the latent cross-attention on (cfg.dec_xattn_every > 0, the DOM presets) the bridge has no prefix
  (n_prefix = 0, an empty [B, 0, H] tensor) and the decoder is the pretrained one until its gates open.
* `cond_tokens` - the CAM tokens the decoder's gated cross-attention reads (`cond=(Z, mask)` of QwenSemantic.step /
  decoder_hidden and HoleFiller.sample / score): [R; G; B; I] with a source tag each; I = intent tokens when given.
  This is the primary CAM -> decoder path; the prefix is the small optional one.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import CamConfig
from ..interfaces import TokenSet
from .layers import ContextReader, RMSNorm


def decision_context(R: torch.Tensor, goal: Optional[TokenSet], belief: Optional[TokenSet], src_emb: nn.Embedding):
    """[R; G; B] tokens with source tags: (x [B, N, D], mask [B, N] bool)."""
    B, S, _ = R.shape
    xs = [R + src_emb.weight[0]]
    ms = [torch.ones(B, S, dtype=torch.bool, device=R.device)]
    for k, ts in ((1, goal), (2, belief)):
        if ts is not None and ts.N > 0:
            xs.append(ts.x.to(R.dtype) + src_emb.weight[k])
            ms.append(ts.mask)
    return torch.cat(xs, 1), torch.cat(ms, 1)


class LatentBridge(nn.Module):
    COND_SOURCES = 4                    # R, G, B, I

    def __init__(self, cfg: CamConfig, sem_dim: int, embedding_rms: float, init_rows: Optional[torch.Tensor] = None,
                 n_layers: int = 2, n_prefix: Optional[int] = None):
        super().__init__()
        D = cfg.d_model
        if n_prefix is None:
            n_prefix = 0 if int(getattr(cfg, "dec_xattn_every", 0) or 0) > 0 else int(cfg.n_prefix)
        self.n_prefix = int(n_prefix)
        self.sem_dim = sem_dim
        if self.n_prefix > 0:
            self.src_emb = nn.Embedding(3, D)
            self.queries = nn.Parameter(torch.randn(self.n_prefix, D) * 0.02)
            self.reader = ContextReader(D, cfg.n_heads, n_layers)
            self.norm = RMSNorm(D)
            self.out = nn.Linear(D, sem_dim)
            nn.init.zeros_(self.out.weight)              # zero-initialised residual: P = base at initialisation
            nn.init.zeros_(self.out.bias)
            if init_rows is None:                         # neutral, text-free rows (callers pass init_rows_from)
                init_rows = torch.full((self.n_prefix, sem_dim), float(embedding_rms))
            rows = init_rows.detach().float().clone()
            if rows.shape != (self.n_prefix, sem_dim):
                rows = rows.reshape(-1, sem_dim)[:1].expand(self.n_prefix, sem_dim).clone()
            self.base = nn.Parameter(rows)
        self.register_buffer("unit", torch.tensor(float(embedding_rms)), persistent=True)
        self.cond_src = nn.Embedding(self.COND_SOURCES, D)
        nn.init.normal_(self.cond_src.weight, std=0.02)

    def forward(self, R: torch.Tensor, goal: Optional[TokenSet] = None, belief: Optional[TokenSet] = None):
        """R [B, S, D] (+ goal, belief TokenSets) -> prefix P [B, n_prefix, sem_dim] (float32; [B, 0, H] when the
        bridge has no prefix)."""
        B = R.shape[0]
        if self.n_prefix == 0:
            return R.new_zeros(B, 0, self.sem_dim, dtype=torch.float32)
        ctx, mask = decision_context(R, goal, belief, self.src_emb)
        q = self.reader(self.queries.unsqueeze(0).expand(B, -1, -1).to(ctx.dtype), ctx, mask)
        return self.base.unsqueeze(0) + self.out(self.norm(q)).float() * self.unit

    def cond_tokens(self, R: torch.Tensor, goal: Optional[TokenSet] = None, belief: Optional[TokenSet] = None,
                    intent: Optional[torch.Tensor] = None, intent_mask: Optional[torch.Tensor] = None) -> tuple:
        """CAM tokens for the decoder's latent cross-attention: (Z [B, N, D] float32, mask [B, N] bool) =
        [R; G; B; I] + source tags (I = intent tokens [B, n_intent, D] when given)."""
        B, S, _ = R.shape
        w = self.cond_src.weight
        xs = [R.float() + w[0].float()]
        ms = [torch.ones(B, S, dtype=torch.bool, device=R.device)]
        for k, ts in ((1, goal), (2, belief)):
            if ts is not None and ts.N > 0:
                xs.append(ts.x.float() + w[k].float())
                ms.append(ts.mask.bool())
        if intent is not None and intent.shape[1] > 0:
            xs.append(intent.float() + w[3].float())
            ms.append(intent_mask.bool() if intent_mask is not None else
                      torch.ones(B, intent.shape[1], dtype=torch.bool, device=R.device))
        return torch.cat(xs, 1), torch.cat(ms, 1)

    @staticmethod
    def init_rows_from(encoder, n_prefix: int, text: Optional[str] = None) -> torch.Tensor:
        """Neutral prefix rows [n_prefix, H]: the mean direction of the real token-embedding rows, scaled to the
        typical row RMS (no phrase whose tokens the decoder would read as text). `text` (legacy, discouraged) uses the
        tiled embeddings of that text instead."""
        with torch.no_grad():
            if text:
                ids = encoder.tok(text, add_special_tokens=False)["input_ids"]
                ids = (ids * (n_prefix // max(len(ids), 1) + 1))[:n_prefix]
                return encoder.embed_ids(torch.tensor(ids, device=encoder.device)).float().cpu()
            w = encoder.input_embeddings().weight[: len(encoder.tok)]
            mean = sum(w[i: i + 16384].float().sum(0) for i in range(0, w.shape[0], 16384)) / w.shape[0]
            row = mean / mean.pow(2).mean().sqrt().clamp_min(1e-12) * float(encoder.embedding_rms)
            return row.unsqueeze(0).expand(n_prefix, -1).clone().cpu()
