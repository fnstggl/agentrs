"""Observation, goal and action encoders.

Numbers are encoded as numbers: every reading becomes a token built from its value, change, missingness and identity.
Optional pretrained text tokens pass through a trainable projection into the same token space.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import UnifiedConfig
from ..interfaces import ActionCandidateBatch, GoalBatch, ObservationBatch
from .programs import DEFAULT, Language


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


def _encoder(d: int, heads: int, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True,
                                       norm_first=True)
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)


class ObservationEncoder(nn.Module):
    """``X_t [B, N, d]`` through local attention, then a trainable query compression to ``E_t [B, K, d]``."""

    def __init__(self, cfg: UnifiedConfig, n_ids: int, text_dim: int | None = None):
        super().__init__()
        d = cfg.d_model
        self.id_embed = nn.Embedding(n_ids, d)
        self.n_extra = 6
        self.numeric = mlp(3 + self.n_extra, d, d, layers=2)
        self.text_proj = nn.Linear(text_dim, d) if text_dim else None
        self.local = _encoder(d, cfg.obs_heads, cfg.obs_layers)
        self.queries = nn.Parameter(torch.randn(cfg.obs_latent_tokens, d) * 0.02)
        self.compress = nn.MultiheadAttention(d, cfg.obs_heads, batch_first=True)
        self.compress_norm = nn.LayerNorm(d)
        self.pool = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d), nn.SiLU(), nn.Linear(d, 1))

    def forward(self, obs: ObservationBatch) -> tuple[Tensor, Tensor]:
        """Returns ``E [B, K, d]`` and its attention-pooled summary ``[B, d]``. Input must be flat in time."""
        num = torch.stack([obs.values, obs.deltas, obs.present.float()], -1)
        extra = obs.extra if obs.extra is not None else torch.zeros(*obs.values.shape, self.n_extra, device=num.device)
        num = torch.cat([num, extra], -1)
        x = self.numeric(num) + self.id_embed(obs.ids)
        pad = ~obs.present
        if obs.text is not None and self.text_proj is not None:
            x = torch.cat([x, self.text_proj(obs.text)], 1)
            pad = torch.cat([pad, ~obs.text_mask], 1)
        # A row with nothing observed would make attention undefined; keep its first token.
        empty = pad.all(1)
        if empty.any():
            pad = pad.clone()
            pad[empty, 0] = False
        x = self.local(x, src_key_padding_mask=pad)
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        e, _ = self.compress(q, x, x, key_padding_mask=pad, need_weights=False)
        e = self.compress_norm(q + e)
        w = torch.softmax(self.pool(e), 1)
        return e, (w * e).sum(1)


class GoalEncoder(nn.Module):
    """Encodes a machine-readable goal specification; a text embedding can be added to the same space."""

    def __init__(self, cfg: UnifiedConfig, n_features: int):
        super().__init__()
        self.net = mlp(n_features, cfg.goal_dim, cfg.goal_dim, layers=2)

    def forward(self, goal: GoalBatch) -> Tensor:
        return self.net(goal.features)


class ActionEncoder(nn.Module):
    """Program tokens plus compiled structure to ``e_a [..., d]``. Lever tokens share the observation's identity
    embedding, so a program refers to the same variable the observation describes."""

    def __init__(self, cfg: UnifiedConfig, obs_ids: nn.Embedding, lever_id0: int, lang: Language = DEFAULT,
                 extra_features: int = 0):
        super().__init__()
        d = cfg.action_dim
        self.lang = lang
        self.lever_id0 = lever_id0
        self.obs_ids = obs_ids
        self.lever_link = nn.Linear(obs_ids.embedding_dim, d)
        self.token = nn.Embedding(lang.vocab, d, padding_idx=0)
        self.pos = nn.Embedding(lang.max_len, d)
        self.body = _encoder(d, max(1, min(cfg.obs_heads, 4)), 1)
        self.attn = nn.Linear(d, 1)
        self.n_struct = lang.n_features + extra_features
        self.struct = mlp(self.n_struct, d, d, layers=1)
        self.out = mlp(2 * d, d, d, layers=1)
        lever_tokens = torch.arange(lang.lever0, lang.value0)
        self.register_buffer("lever_tokens", lever_tokens, persistent=False)

    def forward(self, tokens: Tensor, features: Tensor) -> Tensor:
        shape = tokens.shape[:-1]
        t = tokens.reshape(-1, tokens.shape[-1])
        f = features.reshape(-1, features.shape[-1])
        x = self.token(t) + self.pos(torch.arange(t.shape[1], device=t.device))
        lever = (t >= self.lang.lever0) & (t < self.lang.value0)
        if lever.any():
            ids = t[lever] - self.lang.lever0 + self.lever_id0
            x = x.clone()
            x[lever] = x[lever] + self.lever_link(self.obs_ids(ids))
        pad = t == 0
        # An all-padding row (a terminal or padded replay step) would make attention undefined.
        empty = pad.all(1)
        if empty.any():
            pad = pad.clone()
            pad[empty, 0] = False
        x = self.body(x, src_key_padding_mask=pad)
        w = self.attn(x).squeeze(-1).masked_fill(pad, float("-inf"))
        pooled = (torch.softmax(w, 1).unsqueeze(-1) * x).sum(1)
        return self.out(torch.cat([pooled, self.struct(f)], -1)).reshape(*shape, -1)

    def encode_candidates(self, c: ActionCandidateBatch) -> Tensor:
        return self.forward(c.tokens, c.features)
