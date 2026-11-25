"""LatentBridge: P_t = W_theta(R_t, G_t, B_t) -> soft prefix embeddings of the semantic decoder (spec 2.2).

n_prefix learned queries cross-attend (ContextReader, `n_layers` blocks) to the concatenation of the reasoner state R
[B, S, D], the goal tokens G [B, Ng, D] and the belief tokens B [B, Nb, D] (each tagged with a source embedding and
masked). The result is projected to the semantic model's width and added to a learned base initialised from real token
embedding rows; each prefix vector is then RMS-normalised to the input-embedding RMS times a learned scale, so the
prefix stays on the scale of real token embeddings (a random prefix at the wrong scale derails the decoder, Stage-0
measurement). Output [B, n_prefix, sem_hidden] float32; it is spliced into `inputs_embeds` by grammar.HoleFiller.
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
    def __init__(self, cfg: CamConfig, sem_dim: int, embedding_rms: float, init_rows: Optional[torch.Tensor] = None,
                 n_layers: int = 2):
        super().__init__()
        D = cfg.d_model
        self.n_prefix = cfg.n_prefix
        self.sem_dim = sem_dim
        self.queries = nn.Parameter(torch.randn(cfg.n_prefix, D) * 0.02)
        self.src_emb = nn.Embedding(3, D)
        self.reader = ContextReader(D, cfg.n_heads, n_layers)
        self.norm = RMSNorm(D)
        self.out = nn.Linear(D, sem_dim)
        nn.init.normal_(self.out.weight, std=0.02 / D ** 0.5)
        nn.init.zeros_(self.out.bias)
        if init_rows is None:
            init_rows = torch.randn(cfg.n_prefix, sem_dim) * embedding_rms
        self.base = nn.Parameter(init_rows.detach().float().clone())
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.register_buffer("target_rms", torch.tensor(float(embedding_rms)), persistent=True)

    def forward(self, R: torch.Tensor, goal: Optional[TokenSet] = None, belief: Optional[TokenSet] = None):
        """R [B, S, D] (+ goal, belief TokenSets) -> prefix P [B, n_prefix, sem_dim] (float32)."""
        B = R.shape[0]
        ctx, mask = decision_context(R, goal, belief, self.src_emb)
        q = self.reader(self.queries.unsqueeze(0).expand(B, -1, -1).to(ctx.dtype), ctx, mask)
        raw = self.base.unsqueeze(0) + self.out(self.norm(q)).float()
        rms = raw.pow(2).mean(-1, keepdim=True).add(1e-12).sqrt()
        return raw / rms * (self.target_rms * self.log_scale.exp())

    @staticmethod
    def init_rows_from(encoder, n_prefix: int, text: str = " Plan, evidence and intent from the controller:") -> torch.Tensor:
        """Real token-embedding rows (tiled) used to initialise the prefix base."""
        ids = encoder.tok(text, add_special_tokens=False)["input_ids"]
        ids = (ids * (n_prefix // max(len(ids), 1) + 1))[:n_prefix]
        with torch.no_grad():
            return encoder.embed_ids(torch.tensor(ids, device=encoder.device)).float().cpu()
