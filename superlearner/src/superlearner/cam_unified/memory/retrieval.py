"""Learned memory retrieval (implements ``RetrieverP``; spec 10.3-10.5, architecture §8).

Components (all trainable unless noted):

* ``event_encoder``  records -> event tokens [n, D] (default ``HashedEventEncoder``; the integrated system may pass the
  core's own event-token encoder).
* ``SegmentEncoder`` event-token segments [n, L, D] (L <= ``cfg.mem_segment``) -> ``cfg.mem_segment_tokens`` latent
  memory tokens [n, T, D] (Perceiver-style cross-attention from learned queries after self-attention with continuous
  time features) + an L2-normalised retrieval key [n, key_dim]. The same class (``long_encoder``) compresses
  consecutive level-1 segments' memory tokens into longer representations (level 2).
* ``QueryNet``  q_t = Q_theta(Z_t, G_t, B_t): learned query tokens cross-attend to [Z; G; B] -> MLP -> L2 normalise.
* Target snapshot: ``target_*`` are frozen copies of event/segment/long encoders that produced the keys stored in the
  index. ``key_version`` = f"seg-{n}-{digest}" names that snapshot; ``sync_target()`` makes a new version (re-encoding
  then happens through ``Consolidator.reencode_step``). Queries are only compared with keys of a version this module
  produced (``compatible_versions``), never with obsolete vectors of an unknown encoder.

Training signals (selection is not differentiable, spec 10.3):

* ``usefulness_loss`` - multi-positive InfoNCE where positives are candidate segments whose inclusion lowered the
  downstream (world-model / decision) loss, as measured by the trainer, e.g. with ``measure_usefulness``.
* ``compression_loss`` - contrastive reconstruction of each event token from the segment's memory tokens, so the
  latent memory tokens (not language summaries) keep the information of their events.
"""
from __future__ import annotations

import copy
import hashlib
import math
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import TokenKind, TokenSet
from ._nn import CrossBlock, RMSNorm, SelfBlock, TimeFeatures, autocast_for
from .errors import IncompatibleMemoryVersion
from .featurize import HashedEventEncoder


class _Handle:
    """Holds a non-module resource on an nn.Module without deep-copying / pickling it."""

    def __init__(self, obj=None):
        self.obj = obj

    def __deepcopy__(self, memo):
        return _Handle(self.obj)


class SegmentEncoder(nn.Module):
    """Event-token segment -> latent memory tokens + retrieval key.

    forward(x [n,L,D], mask [n,L] bool, dt [n,L] days relative to the segment end) -> (mem [n,T,D], key [n,Kd])."""

    def __init__(self, d: int, key_dim: int, n_tokens: int, n_heads: int, n_layers: int = 2, max_len: int = 256):
        super().__init__()
        self.d, self.n_tokens, self.max_len = d, n_tokens, max_len
        self.inp = nn.Linear(d, d)
        self.time = TimeFeatures(d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([SelfBlock(d, n_heads) for _ in range(n_layers)])
        self.queries = nn.Parameter(torch.randn(n_tokens, d) * 0.02)
        self.pool = CrossBlock(d, n_heads)
        self.post = SelfBlock(d, n_heads)
        self.out_norm = RMSNorm(d)
        self.key_head = nn.Sequential(RMSNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, key_dim))
        # contrastive reconstruction decoder (compression objective)
        self.dec_pos = nn.Embedding(max_len, d)
        self.dec_time = TimeFeatures(d)
        self.dec = CrossBlock(d, n_heads)
        self.dec_out = nn.Linear(d, d)

    def forward(self, x, mask, dt):
        n, L, _ = x.shape
        if L > self.max_len:
            raise ValueError(f"segment length {L} > max_len {self.max_len}")
        mask = mask.clone()
        empty = ~mask.any(1)
        mask[empty, 0] = True                                  # keep SDPA finite for all-padding rows
        pos = torch.arange(L, device=x.device)
        with autocast_for(x):
            h = self.inp(x) + self.time(dt) + self.pos(pos)
            for blk in self.blocks:
                h = blk(h, mask)
            m = self.queries.unsqueeze(0).expand(n, -1, -1)
            m = self.pool(m, h, mask)
            m = self.post(m)
            mem = self.out_norm(m).float()
            key = self.key_head(mem.mean(1)).float()
        return mem, F.normalize(key, dim=-1)

    def reconstruct(self, mem, L: int, dt):
        """Predict the L event tokens from the memory tokens: [n,T,D] -> [n,L,D]."""
        n = mem.shape[0]
        pos = torch.arange(L, device=mem.device)
        with autocast_for(mem):
            qx = self.dec_pos(pos).unsqueeze(0).expand(n, -1, -1) + self.dec_time(dt)
            y = self.dec(qx, mem)
            return self.dec_out(y).float()


class QueryNet(nn.Module):
    """q = Q_theta(Z, G, B): [B,S,D] + TokenSets -> [B, key_dim] (L2-normalised)."""

    def __init__(self, d: int, key_dim: int, n_heads: int, n_query: int = 2):
        super().__init__()
        self.kind = nn.Embedding(int(TokenKind.N_KINDS), d)
        self.q = nn.Parameter(torch.randn(n_query, d) * 0.02)
        self.att = CrossBlock(d, n_heads)
        self.att2 = CrossBlock(d, n_heads)
        self.head = nn.Sequential(RMSNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, key_dim))

    def forward(self, Z, goal: Optional[TokenSet] = None, belief: Optional[TokenSet] = None):
        B, S, _ = Z.shape
        xs = [Z + self.kind.weight[int(TokenKind.STATE)]]
        ms = [torch.ones(B, S, dtype=torch.bool, device=Z.device)]
        for ts in (goal, belief):
            if ts is not None and ts.N > 0:
                xs.append(ts.x.to(Z.dtype) + self.kind(ts.kind))
                ms.append(ts.mask)
        ctx, mask = torch.cat(xs, 1), torch.cat(ms, 1)
        with autocast_for(Z):
            h = self.q.unsqueeze(0).expand(B, -1, -1)
            h = self.att(h, ctx, mask)
            h = self.att2(h, ctx, mask)
            q = self.head(h.mean(1)).float()
        return F.normalize(q, dim=-1)


class Retriever(nn.Module):
    """RetrieverP implementation. See the module docstring.

    Args:
        cfg: CamConfig (d_model, n_heads, mem_key_dim, mem_segment, mem_segment_tokens, mem_top_k, use_memory).
        index: a ``MemoryIndex`` (attach later with ``attach_index``).
        event_encoder: records -> [n, D] module (default ``HashedEventEncoder(cfg.d_model)``).
        fanout: level-1 segments per level-2 representation (bounds the long encoder's input length).
    """

    def __init__(self, cfg, index=None, event_encoder: Optional[nn.Module] = None, n_seg_layers: int = 2,
                 n_query_tokens: int = 2, fanout: int = 8, tau_init: float = 0.07):
        super().__init__()
        D, Kd, T = cfg.d_model, cfg.mem_key_dim, cfg.mem_segment_tokens
        self.cfg = cfg
        self.d, self.key_dim, self.n_mem = D, Kd, T
        self.segment_len = cfg.mem_segment
        self.fanout = fanout
        self.event_encoder = event_encoder if event_encoder is not None else HashedEventEncoder(D)
        self.segment_encoder = SegmentEncoder(D, Kd, T, cfg.n_heads, n_seg_layers, max_len=max(cfg.mem_segment, 8))
        self.long_encoder = SegmentEncoder(D, Kd, T, cfg.n_heads, 1, max_len=max(fanout * T, 8))
        self.query_net = QueryNet(D, Kd, cfg.n_heads, n_query_tokens)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))
        self.target_event_encoder = copy.deepcopy(self.event_encoder)
        self.target_segment_encoder = copy.deepcopy(self.segment_encoder)
        self.target_long_encoder = copy.deepcopy(self.long_encoder)
        for m in self._targets():
            m.requires_grad_(False)
            m.eval()
        self.register_buffer("target_n", torch.zeros((), dtype=torch.long))
        self.register_buffer("target_digest", torch.zeros(8, dtype=torch.uint8))
        self.register_buffer("prev_n", torch.full((), -1, dtype=torch.long))
        self.register_buffer("prev_digest", torch.zeros(8, dtype=torch.uint8))
        self._index = _Handle(index)
        self._refresh_digest()

    # ------------------------------------------------------------------ versions
    def _targets(self):
        return (self.target_event_encoder, self.target_segment_encoder, self.target_long_encoder)

    def _refresh_digest(self) -> None:
        h = hashlib.sha256()
        for m in self._targets():
            for k, v in sorted(m.state_dict().items()):
                h.update(k.encode())
                h.update(v.detach().float().cpu().numpy().tobytes())
        self.target_digest.copy_(torch.tensor(list(h.digest()[:8]), dtype=torch.uint8))

    @staticmethod
    def _ver(n: int, digest: torch.Tensor) -> str:
        return f"seg-{int(n)}-{bytes(digest.tolist()).hex()}"

    @property
    def key_version(self) -> str:
        """Embedding version of the keys/values the target snapshot produces."""
        return self._ver(int(self.target_n), self.target_digest)

    def compatible_versions(self) -> set:
        out = {self.key_version}
        if int(self.prev_n) >= 0:
            out.add(self._ver(int(self.prev_n), self.prev_digest))
        return out

    @torch.no_grad()
    def sync_target(self) -> str:
        """Snapshot the online encoders as the new target version; the previous one stays searchable until the
        index's re-encoding to the new version completes."""
        self.prev_n.copy_(self.target_n)
        self.prev_digest.copy_(self.target_digest)
        for tgt, src in zip(self._targets(), (self.event_encoder, self.segment_encoder, self.long_encoder)):
            tgt.load_state_dict(src.state_dict())
        self.target_n += 1
        self._refresh_digest()
        return self.key_version

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self._targets():
            m.eval()
        return self

    # ------------------------------------------------------------------ index handle
    @property
    def index(self):
        return self._index.obj

    def attach_index(self, index) -> None:
        self._index = _Handle(index)

    def detach_index(self) -> None:
        self._index = _Handle(None)

    # ------------------------------------------------------------------ encoders
    def encode_records(self, records: Sequence, target: bool = True) -> torch.Tensor:
        """records -> event tokens [n, D] (target snapshot by default: what the index stores)."""
        enc = self.target_event_encoder if target else self.event_encoder
        with torch.set_grad_enabled(not target and torch.is_grad_enabled()):
            return enc(list(records)).float()

    def encode_segments(self, x, mask, dt, target: bool = False):
        """Event-token segments [n,L,D] -> (memory tokens [n,T,D], keys [n,Kd]); online (trainable) by default."""
        enc = self.target_segment_encoder if target else self.segment_encoder
        with torch.set_grad_enabled(not target and torch.is_grad_enabled()):
            return enc(x, mask, dt)

    def encode_long(self, child_mem, mask, dt, target: bool = False):
        """Level-2: concatenated child memory tokens [n, F*T, D] -> ([n,T,D], [n,Kd])."""
        enc = self.target_long_encoder if target else self.long_encoder
        with torch.set_grad_enabled(not target and torch.is_grad_enabled()):
            return enc(child_mem, mask, dt)

    # ------------------------------------------------------------------ RetrieverP
    def query(self, Z, goal: Optional[TokenSet] = None, belief: Optional[TokenSet] = None):
        """q_t = Q_theta(Z_t, G_t, B_t) -> [B, key_dim]."""
        return self.query_net(Z, goal, belief)

    def retrieve(self, q, k: Optional[int] = None, filters: Optional[list] = None, now=None,
                 version: Optional[str] = None) -> tuple:
        """Top-k memory segments for each query row -> (M_t TokenSet [B, k*T, D] of kind MEMORY, ids [B][<=k] segment
        ids, scores [B, k] (-inf = empty slot)).

        ``M_t.refs[b][i]`` is the tuple of source event ids of the segment the token belongs to (exact originals via
        ``EventArchive.get``); ``M_t.time`` = segment end time minus ``now`` (days; ``now`` [B] or None = 0).
        Searches the index's current version (it must be one this module produced)."""
        k = int(k or self.cfg.mem_top_k)
        B = q.shape[0]
        dev = q.device
        idx = self.index
        if idx is None or not getattr(self.cfg, "use_memory", True) or k == 0:
            return self._empty(B, dev)
        v = version or idx.current_version
        if v is None or idx.count(v) == 0:
            return self._empty(B, dev)
        if v not in self.compatible_versions():
            raise IncompatibleMemoryVersion(
                f"index version {v} was not produced by this retriever (compatible: {sorted(self.compatible_versions())})")
        res = idx.search(v, q.detach(), k, filters)
        kids = res.kids
        valid = kids >= 0
        flat = kids[valid]
        T, D = self.n_mem, self.d
        x = torch.zeros(B, k, T, D, device=dev)
        tm = torch.zeros(B, k, T, device=dev)
        if len(flat):
            vals = idx.values(v, flat).to(dev)
            metas = idx.segments([s for row in res.seg_ids for s in row if s is not None])
            t_end = torch.tensor([m["t_end"] for m in metas], device=dev, dtype=torch.float32)
            vm = torch.from_numpy(valid).to(dev)
            x[vm] = vals
            nowt = torch.zeros(B, device=dev) if now is None else torch.as_tensor(now, dtype=torch.float32, device=dev)
            tm[vm] = (t_end.unsqueeze(-1) - nowt.unsqueeze(1).expand(B, k)[vm].unsqueeze(-1)).expand(-1, T)
            src = {m["seg_id"]: tuple(m["event_ids"]) if m["event_ids"] else tuple(idx.source_events(m["seg_id"]))
                   for m in metas}
        else:
            src = {}
        mask = torch.from_numpy(valid).to(dev).unsqueeze(-1).expand(B, k, T).reshape(B, k * T)
        refs = [[(src[s] if s is not None else None) for s in row for _ in range(T)] for row in res.seg_ids]
        ts = TokenSet(x.reshape(B, k * T, D), mask, torch.full((B, k * T), int(TokenKind.MEMORY), dtype=torch.long,
                                                                   device=dev),
                      tm.reshape(B, k * T), torch.full((B, k * T), -1, dtype=torch.long, device=dev), refs)
        ids = [[s for s in row if s is not None] for row in res.seg_ids]
        return ts, ids, res.scores

    def _empty(self, B: int, dev):
        ts = TokenSet.empty(B, self.d, device=dev)
        ts.refs = [[] for _ in range(B)]
        return ts, [[] for _ in range(B)], torch.zeros(B, 0, device=dev)

    # ------------------------------------------------------------------ training objectives
    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(0.01, 1.0)

    def usefulness_loss(self, q, cand_keys, utility, margin: float = 0.0, cand_mask=None) -> tuple:
        """Multi-positive InfoNCE from downstream usefulness.

        q [B,Kd]; cand_keys [B,C,Kd] (or [C,Kd] shared); utility [B,C] = loss_without - loss_with (> margin =
        positive); cand_mask [B,C] bool (valid candidates). Rows need >=1 positive and >=1 negative; others are
        ignored. Returns (loss, {"rows", "pos_frac", "acc"}) where acc = top-1 candidate is a positive."""
        if cand_keys.dim() == 2:
            cand_keys = cand_keys.unsqueeze(0).expand(q.shape[0], -1, -1)
        if cand_mask is None:
            cand_mask = torch.ones(utility.shape, dtype=torch.bool, device=q.device)
        s = torch.einsum("bd,bcd->bc", q.float(), cand_keys.float()) / self.tau
        pos = (utility > margin) & cand_mask
        neg = (~pos) & cand_mask
        rows = pos.any(1) & neg.any(1)
        if not rows.any():
            return s.sum() * 0.0, {"rows": 0, "pos_frac": 0.0, "acc": float("nan")}
        ninf = torch.finfo(s.dtype).min
        num = torch.logsumexp(s.masked_fill(~pos, ninf), 1)
        den = torch.logsumexp(s.masked_fill(~cand_mask, ninf), 1)
        loss = (den - num)[rows].mean()
        top = s.masked_fill(~cand_mask, ninf).argmax(1)
        acc = pos.gather(1, top.unsqueeze(1)).squeeze(1)[rows].float().mean()
        return loss, {"rows": int(rows.sum()), "pos_frac": float(pos[rows].float().mean()), "acc": float(acc)}

    def compression_loss(self, x, mask, dt, temperature: float = 0.1) -> torch.Tensor:
        """Contrastive reconstruction: each event token must be identified among all event tokens of the batch
        from its segment's memory tokens (collapse-resistant; trains the online segment encoder)."""
        mem, _ = self.segment_encoder(x, mask, dt)
        pred = self.segment_encoder.reconstruct(mem, x.shape[1], dt)
        p = F.normalize(pred[mask], dim=-1)
        t = F.normalize(x[mask].detach().float(), dim=-1)
        logits = p @ t.T / temperature
        return F.cross_entropy(logits, torch.arange(len(p), device=p.device))

    @staticmethod
    @torch.no_grad()
    def measure_usefulness(loss_fn: Callable, candidates: Sequence[TokenSet], base: Optional[TokenSet] = None):
        """Usefulness of each candidate memory for the downstream loss: u[b, c] = L(base) - L(base + candidate c).

        loss_fn(M: TokenSet | None) -> per-row loss [B] (no reduction); candidates: C TokenSets [B, T, D]."""
        l0 = loss_fn(base)
        us = []
        for c in candidates:
            m = c if base is None else TokenSet.cat(base, c)
            us.append(l0 - loss_fn(m))
        return torch.stack(us, 1)
