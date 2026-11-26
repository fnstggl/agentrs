"""IntentHead: CAM's numerical action intent I_t (DOM Mechanism B; plan 2026-11-26 §2 step 2, W5).

    I_t = Intent_theta(R_t, ctx_out_t [, OPTION(o_t), BUDGET(b_t)])        [B, n_intent, D], deterministic

`n_intent` learned queries read, through `n_layers` pre-norm cross-attention + SwiGLU blocks (semantic.layers
.ContextReader), the reasoned state R_t [B,S,D] and the core's processed context ctx_out [B,N,D] (the outputs of
F_theta at the context positions: observations, entities, previous action, beliefs, goal and tool tokens, each
already mixed with the state by the core's self-attention). Every context source carries a learned source tag and a
kind embedding. The intent is the only conditioning the action generator receives in intent mode (pointers, typed
heads, length prior, bridge prefix and decoder cross-attention), so the goal and the beliefs reach the action only
through Z / R / ctx_out / I.

The output is RMS-normalised (the reader's out_norm), so its scale is fixed whatever the state statistics.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ..config import CamConfig
from ..interfaces import TokenKind, TokenSet
from ..semantic.layers import ContextReader

# TokenKind of intent tokens when they are passed on as a TokenSet (value head, option policy, reward model).
# interfaces.TokenKind has no INTENT kind yet (change request to W1: TokenKind.INTENT with a KindEmbedding migration
# hook); REASON ("reasoning scratch tokens") is the closest existing kind and keeps every embedding table's shape.
INTENT_KIND = int(getattr(TokenKind, "INTENT", TokenKind.REASON))


class IntentHead(nn.Module):
    """I_t = IntentHead(cfg)(R [B,S,D], ctx_out [B,N,D], mask [B,N] bool, kind [B,N] long | None,
    extra=[TokenSet ...]) -> I [B, n_intent, D] float32."""

    def __init__(self, cfg: CamConfig, n_layers: int = 2, n_intent: Optional[int] = None):
        super().__init__()
        n = int(n_intent if n_intent is not None else getattr(cfg, "n_intent", 0))
        if n <= 0:
            raise ValueError("IntentHead needs n_intent > 0 (cfg.n_intent)")
        D = cfg.d_model
        self.n_intent = n
        self.d_model = D
        self.queries = nn.Parameter(torch.randn(n, D) * 0.02)
        self.src = nn.Embedding(3, D)                 # 0 = R_t, 1 = ctx_out, 2 = extra tokens (option, budget)
        nn.init.normal_(self.src.weight, std=0.02)
        self.kind = nn.Embedding(int(TokenKind.N_KINDS), D)
        nn.init.normal_(self.kind.weight, std=0.02)
        self.reader = ContextReader(D, cfg.n_heads, n_layers)

    def forward(self, R: Tensor, ctx_out: Optional[Tensor] = None, mask: Optional[Tensor] = None,
                kind: Optional[Tensor] = None, extra: Optional[Sequence[Optional[TokenSet]]] = None) -> Tensor:
        B, S, D = R.shape
        dev = R.device
        xs = [R.float() + self.src.weight[0]]
        ms = [torch.ones(B, S, dtype=torch.bool, device=dev)]
        if ctx_out is not None and ctx_out.shape[1] > 0:
            if ctx_out.shape[0] != B or ctx_out.shape[2] != D:
                raise ValueError(f"ctx_out {tuple(ctx_out.shape)} does not match R {tuple(R.shape)}")
            c = ctx_out.float() + self.src.weight[1]
            if kind is not None:
                c = c + self.kind(kind.clamp(0, int(TokenKind.N_KINDS) - 1))
            xs.append(c)
            ms.append(torch.ones(B, ctx_out.shape[1], dtype=torch.bool, device=dev) if mask is None
                      else mask.to(torch.bool))
        for ts in extra or ():
            if ts is None or ts.N == 0:
                continue
            if ts.B != B:
                raise ValueError(f"extra intent context batch {ts.B} != {B}")
            xs.append(ts.x.float() + self.src.weight[2] + self.kind(ts.kind.clamp(0, int(TokenKind.N_KINDS) - 1)))
            ms.append(ts.mask.to(torch.bool))
        ctx = torch.cat(xs, 1)
        m = torch.cat(ms, 1)
        q = self.queries.unsqueeze(0).expand(B, -1, -1).to(ctx.dtype)
        return self.reader(q, ctx, m).float()


def intent_tokens(I: Tensor, valid: Optional[Tensor] = None) -> TokenSet:
    """I [B, n, D] -> TokenSet of kind INTENT_KIND (all real unless `valid` [B] says otherwise)."""
    B, n, _ = I.shape
    dev = I.device
    mask = torch.ones(B, n, dtype=torch.bool, device=dev)
    if valid is not None:
        mask = mask & valid.view(B, 1).to(torch.bool)
    return TokenSet(I, mask, torch.full((B, n), INTENT_KIND, dtype=torch.long, device=dev),
                    torch.zeros(B, n, device=dev), torch.full((B, n), -1, dtype=torch.long, device=dev),
                    [[f"intent:{j}" for j in range(n)] for _ in range(B)])
