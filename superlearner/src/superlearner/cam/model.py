"""The trainable CAM controller.  [candidate-editable]

A recurrent transformer with pointer outputs. All of its parameters are trained (full-parameter optimisation); the
frozen language model contributes only fixed text embeddings (`EmbStore`) and proposals that appear as items.

    context items --+
    observation ----+--> transformer encoder --> c_t --> fast GRU --> h_fast --+
                                                        slow GRU --> h_slow ---+--> z_t
    z_t x action heads (every available tool, wait, ask-the-LLM, draft and tool-making operations)  -> head logits
    z_t, chosen head, argument name, earlier arguments x argument options                          -> option logits
    z_t -> value ensemble (mean = critic, spread = uncertainty), success probability

Heads and options are *items* embedded from text embeddings, hashed tokens and numeric features, so the output
space is whatever the current situation offers: there are no per-tool or per-strategy output units. Invalid
candidates are masked before the softmax, so probabilities are always over what could actually be chosen.

The fast GRU steps every decision. The slow GRU steps only when `slow_tick` is set (the agent sets it when
simulated time jumps, e.g. after a long wait), giving a second, slower timescale of recurrent memory.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import N_BUCKETS, N_NUM, N_TYPES, OBS_DIM

NEG = -1e9


@dataclass
class ModelConfig:
    d: int = 384
    layers: int = 6
    heads: int = 8
    ff: int = 1536
    bag_dim: int = 128
    emb_dim: int = 512
    hid: int = 384
    n_value: int = 4
    dropout: float = 0.0
    aux_dim: int = 16

    def to_dict(self) -> dict:
        return asdict(self)


class EmbStore:
    """Frozen text embeddings by key. Row 0 is the zero vector (no text)."""

    def __init__(self, dim: int = 512, cap: int = 4096):
        self.dim = dim
        self.mat = np.zeros((cap, dim), np.float32)
        self.key2row: dict[int, int] = {0: 0}
        self.n = 1
        self._tensor = None
        self._tensor_n = 0
        self._dev = None

    def put(self, key: int, vec: np.ndarray) -> None:
        key = int(key)
        if key in self.key2row:
            return
        if self.n >= len(self.mat):
            self.mat = np.concatenate([self.mat, np.zeros_like(self.mat)])
        self.mat[self.n] = vec
        self.key2row[key] = self.n
        self.n += 1

    def has(self, key: int) -> bool:
        return int(key) in self.key2row

    def rows(self, keys: np.ndarray) -> np.ndarray:
        flat = np.asarray(keys, np.int64).reshape(-1)
        uniq, inv = np.unique(flat, return_inverse=True)
        r = np.fromiter((self.key2row.get(int(k), 0) for k in uniq), np.int64, len(uniq))
        return r[inv].reshape(np.shape(keys))

    def tensor(self, device) -> torch.Tensor:
        if self._tensor is None or self._tensor_n != self.n or self._dev != str(device):
            self._tensor = torch.from_numpy(self.mat[: self.n].copy()).to(device)
            self._tensor_n, self._dev = self.n, str(device)
        return self._tensor

    def items_since(self, mark: int) -> list[tuple[int, np.ndarray]]:
        inv = {r: k for k, r in self.key2row.items()}
        return [(inv[r], self.mat[r].copy()) for r in range(max(1, mark), self.n)]


class CAMController(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        d, h = cfg.d, cfg.hid
        self.tok = nn.Embedding(N_BUCKETS, cfg.bag_dim, padding_idx=0)
        self.tok_proj = nn.Linear(cfg.bag_dim, d)
        self.ek_proj = nn.Linear(cfg.emb_dim, d)
        self.num_proj = nn.Linear(N_NUM, d)
        self.typ_emb = nn.Embedding(N_TYPES, d)
        self.item_norm = nn.LayerNorm(d)
        self.obs_proj = nn.Sequential(nn.Linear(OBS_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.aux_proj = nn.Linear(cfg.aux_dim, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, cfg.heads, cfg.ff, cfg.dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(d)
        self.fast = nn.GRUCell(d, h)
        self.slow = nn.GRUCell(h, h)
        self.mix = nn.Sequential(nn.Linear(d + 2 * h, d), nn.GELU(), nn.LayerNorm(d))
        self.head_q = nn.Linear(d, d)
        self.head_score = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.arg_q = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, d))
        self.arg_score = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.n_value))
        self.success = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        nn.init.normal_(self.tok.weight, std=0.1)
        with torch.no_grad():
            self.tok.weight[0].zero_()
            self.head_score[-1].weight.mul_(0.1)
            self.arg_score[-1].weight.mul_(0.1)
        self.table: torch.Tensor | None = None       # frozen text embeddings, set with `set_table`

    # ------------------------------------------------------------------ frozen inputs
    def set_table(self, table: torch.Tensor) -> None:
        self.table = table

    def n_params(self) -> dict:
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"trainable": tr, "total": sum(p.numel() for p in self.parameters()),
                "frozen_text_embedding_rows": 0 if self.table is None else int(self.table.shape[0])}

    # ------------------------------------------------------------------ pieces
    def embed(self, row: torch.Tensor, tok: torch.Tensor, num: torch.Tensor, typ: torch.Tensor) -> torch.Tensor:
        """Items [..] -> embeddings [.., d]. `row` indexes the frozen table; `tok` [.., T]; `num` [.., N]."""
        te = self.tok(tok.long())
        cnt = (tok != 0).sum(-1, keepdim=True).clamp(min=1).to(te.dtype)
        e = self.tok_proj(te.sum(-2) / cnt)
        e = e + self.ek_proj(self.table[row.long()].to(e.dtype)) * (row != 0).unsqueeze(-1).to(e.dtype)
        e = e + self.num_proj(num.to(e.dtype)) + self.typ_emb(typ.long())
        return self.item_norm(e)

    def encode(self, ctx_e: torch.Tensor, ctx_mask: torch.Tensor, obsf: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """Context items [N, Nc, d] + observation features [N, OBS_DIM] + aux [N, aux_dim] -> c [N, d]."""
        n = ctx_e.shape[0]
        state = (self.obs_proj(obsf.to(ctx_e.dtype)) + self.aux_proj(aux.to(ctx_e.dtype))).unsqueeze(1)
        x = torch.cat([self.cls.expand(n, 1, -1).to(ctx_e.dtype), state, ctx_e], 1)
        pad = torch.cat([torch.zeros(n, 2, dtype=torch.bool, device=ctx_e.device), ~ctx_mask], 1)
        y = self.encoder(x, src_key_padding_mask=pad)
        return self.enc_norm(y[:, 0])

    def init_state(self, n: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(n, self.cfg.hid, device=device or self.cls.device)
        return z, z.clone()

    def recur(self, c: torch.Tensor, state, slow_tick: torch.Tensor):
        hf, hs = state
        hf = self.fast(c, hf)
        tick = slow_tick.to(hf.dtype).unsqueeze(-1)
        hs = tick * self.slow(hf, hs) + (1 - tick) * hs
        z = self.mix(torch.cat([c, hf, hs], -1))
        return z, (hf, hs)

    def head_logits(self, z: torch.Tensor, head_e: torch.Tensor, head_mask: torch.Tensor) -> torch.Tensor:
        q = self.head_q(z).unsqueeze(1).expand_as(head_e)
        s = self.head_score(torch.cat([q, head_e, q * head_e], -1)).squeeze(-1)
        return s.masked_fill(~head_mask, NEG)

    def arg_logits(self, z, e_head, par_e, opt_e, opt_mask, prev) -> torch.Tensor:
        """One slot: par_e [N, d], opt_e [N, No, d], prev [N, d] (mean embedding of earlier chosen options)."""
        q = self.arg_q(torch.cat([z, e_head, par_e, prev], -1)).unsqueeze(1).expand_as(opt_e)
        s = self.arg_score(torch.cat([q, opt_e, q * opt_e], -1)).squeeze(-1)
        return s.masked_fill(~opt_mask, NEG)

    def arg_logits_all(self, z, e_head, par_e, opt_e, opt_mask, chosen) -> torch.Tensor:
        """Teacher-forced slots: par_e [N, S, d], opt_e [N, S, No, d], chosen [N, S] -> logits [N, S, No]."""
        n, s, no, d = opt_e.shape
        idx = chosen.clamp(min=0).long().view(n, s, 1, 1).expand(n, s, 1, d)
        ch = opt_e.gather(2, idx).squeeze(2) * (chosen >= 0).unsqueeze(-1).to(opt_e.dtype)      # [N, S, d]
        csum = torch.cumsum(ch, 1) - ch
        cnt = (torch.cumsum((chosen >= 0).to(opt_e.dtype), 1) - (chosen >= 0).to(opt_e.dtype)).clamp(min=1).unsqueeze(-1)
        prev = csum / cnt
        q = self.arg_q(torch.cat([z.unsqueeze(1).expand(n, s, d), e_head.unsqueeze(1).expand(n, s, d), par_e, prev], -1))
        q = q.unsqueeze(2).expand(n, s, no, d)
        sc = self.arg_score(torch.cat([q, opt_e, q * opt_e], -1)).squeeze(-1)
        return sc.masked_fill(~opt_mask, NEG)

    def values(self, z: torch.Tensor) -> torch.Tensor:
        return self.value(z)

    def p_success(self, z: torch.Tensor) -> torch.Tensor:
        return self.success(z).squeeze(-1)


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.masked_fill(~mask, NEG), -1)


def masked_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lp = masked_log_softmax(logits, mask)
    p = lp.exp() * mask.to(lp.dtype)
    return -(p * lp.masked_fill(~mask, 0.0)).sum(-1)
