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
  produced (``compatible_versions`` = the target version and the version the index served when the target was last
  synced), never with obsolete vectors of an unknown encoder. ``sync_target`` is a no-op when the online encoders equal
  the target (no spurious re-encode) and while a migration is pending (the index has not promoted the target yet), so
  the served version always stays compatible; ``force=True`` abandons the unpromoted version instead.

Width and size: the transformers of the segment / long / query encoders run at ``width`` (default ``d_model`` up to 256,
else ``d_model // 2`` rounded to the heads) with projections from/to ``d_model``; the default ``HashedEventEncoder``
embeds at ``min(d_model, 128)``. At dev scale (d_model 384) the online encoders hold ~7.6M parameters and the frozen
target copies ~6.7M (``param_split()``); the target copies are part of the weights because the stored keys are only
reproducible from them (re-encoding after a restore needs them).

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

SEG_REF = "seg:"          # TokenSet.refs prefix of memory tokens (scalar source id of the segment)


class _Handle:
    """Holds a non-module resource on an nn.Module without deep-copying / pickling it."""

    def __init__(self, obj=None):
        self.obj = obj

    def __deepcopy__(self, memo):
        return _Handle(self.obj)


def default_width(d_model: int, n_heads: int) -> int:
    """Internal width of the memory transformers: d_model up to 256, else ~d_model/2 (a multiple of n_heads)."""
    if d_model <= 256:
        return d_model
    w = (d_model // 2) // n_heads * n_heads
    return max(n_heads * 16, w)


class SegmentEncoder(nn.Module):
    """Event-token segment -> latent memory tokens + retrieval key.

    forward(x [n,L,D], mask [n,L] bool, dt [n,L] days relative to the segment end) -> (mem [n,T,D], key [n,Kd]).
    The transformer runs at ``width`` (default D); memory tokens are projected back to D."""

    def __init__(self, d: int, key_dim: int, n_tokens: int, n_heads: int, n_layers: int = 2, max_len: int = 256,
                 width: Optional[int] = None):
        super().__init__()
        w = int(width or d)
        self.d, self.width, self.n_tokens, self.max_len = d, w, n_tokens, max_len
        self.inp = nn.Linear(d, w)
        self.time = TimeFeatures(w)
        self.pos = nn.Embedding(max_len, w)
        self.blocks = nn.ModuleList([SelfBlock(w, n_heads) for _ in range(n_layers)])
        self.queries = nn.Parameter(torch.randn(n_tokens, w) * 0.02)
        self.pool = CrossBlock(w, n_heads)
        self.post = SelfBlock(w, n_heads)
        self.out_norm = RMSNorm(w)
        self.key_head = nn.Sequential(RMSNorm(w), nn.Linear(w, w), nn.SiLU(), nn.Linear(w, key_dim))
        # contrastive reconstruction decoder (compression objective)
        self.dec_pos = nn.Embedding(max_len, w)
        self.dec_time = TimeFeatures(w)
        self.dec = CrossBlock(w, n_heads)
        self.dec_out = nn.Linear(w, d)
        self.out_proj = nn.Linear(w, d) if w != d else nn.Identity()      # memory tokens back to d_model
        self.dec_in = nn.Linear(d, w) if w != d else nn.Identity()

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
            mw = self.out_norm(m)
            key = self.key_head(mw.mean(1)).float()
            mem = self.out_proj(mw).float()
        return mem, F.normalize(key, dim=-1)

    def reconstruct(self, mem, L: int, dt):
        """Predict the L event tokens from the memory tokens: [n,T,D] -> [n,L,D]."""
        n = mem.shape[0]
        pos = torch.arange(L, device=mem.device)
        with autocast_for(mem):
            qx = self.dec_pos(pos).unsqueeze(0).expand(n, -1, -1) + self.dec_time(dt)
            y = self.dec(qx, self.dec_in(mem))
            return self.dec_out(y).float()


class QueryNet(nn.Module):
    """q = Q_theta(Z, G, B): [B,S,D] + TokenSets -> [B, key_dim] (L2-normalised); attention at ``width``."""

    def __init__(self, d: int, key_dim: int, n_heads: int, n_query: int = 2, width: Optional[int] = None):
        super().__init__()
        w = int(width or d)
        self.kind = nn.Embedding(int(TokenKind.N_KINDS), d)
        self.ctx_proj = nn.Linear(d, w) if w != d else nn.Identity()
        self.q = nn.Parameter(torch.randn(n_query, w) * 0.02)
        self.att = CrossBlock(w, n_heads)
        self.att2 = CrossBlock(w, n_heads)
        self.head = nn.Sequential(RMSNorm(w), nn.Linear(w, w), nn.SiLU(), nn.Linear(w, key_dim))

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
            ctx = self.ctx_proj(ctx)
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
        width: internal width of the segment / long / query transformers (default ``default_width``).
    """

    def __init__(self, cfg, index=None, event_encoder: Optional[nn.Module] = None, n_seg_layers: int = 2,
                 n_query_tokens: int = 2, fanout: int = 8, tau_init: float = 0.07, width: Optional[int] = None):
        super().__init__()
        D, Kd, T = cfg.d_model, cfg.mem_key_dim, cfg.mem_segment_tokens
        W = int(width or default_width(D, cfg.n_heads))
        self.cfg = cfg
        self.d, self.key_dim, self.n_mem, self.width = D, Kd, T, W
        self.segment_len = cfg.mem_segment
        self.fanout = fanout
        self.event_encoder = event_encoder if event_encoder is not None else HashedEventEncoder(D)
        self.segment_encoder = SegmentEncoder(D, Kd, T, cfg.n_heads, n_seg_layers, max_len=max(cfg.mem_segment, 8),
                                              width=W)
        self.long_encoder = SegmentEncoder(D, Kd, T, cfg.n_heads, 1, max_len=max(fanout * T, 8), width=W)
        self.query_net = QueryNet(D, Kd, cfg.n_heads, n_query_tokens, width=W)
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
        self._awaiting = None                  # version a restore scheduled for re-encoding (memory empty until then)
        self.sync_info: dict = {}
        self.retrieve_stats = {"calls": 0, "empty_awaiting_migration": 0}
        self._refresh_digest()

    # ------------------------------------------------------------------ versions
    def _targets(self):
        return (self.target_event_encoder, self.target_segment_encoder, self.target_long_encoder)

    def _onlines(self):
        return (self.event_encoder, self.segment_encoder, self.long_encoder)

    @staticmethod
    def _digest_of(modules) -> torch.Tensor:
        h = hashlib.sha256()
        for m in modules:
            for k, v in sorted(m.state_dict().items()):
                h.update(k.encode())
                h.update(v.detach().float().cpu().numpy().tobytes())
        return torch.tensor(list(h.digest()[:8]), dtype=torch.uint8)

    def _refresh_digest(self) -> None:
        self.target_digest.copy_(self._digest_of(self._targets()))

    def param_split(self) -> dict:
        """Parameter counts: online (trainable) encoders, frozen target copies, query net."""
        cnt = lambda ms: sum(p.numel() for m in ms for p in m.parameters())   # noqa: E731
        return {"online_encoders": cnt(self._onlines()), "target_copies": cnt(self._targets()),
                "query_net": cnt([self.query_net]), "total": sum(p.numel() for p in self.parameters())}

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

    def served_version(self) -> Optional[str]:
        """The attached index's current version when it is one this module can read, else None."""
        idx = self.index
        cur = None if idx is None or getattr(idx, "closed", False) else idx.current_version
        return cur if cur in self.compatible_versions() else None

    @property
    def migration_pending(self) -> bool:
        """True while the index still serves the previous version (re-encoding to ``key_version`` not promoted)."""
        s = self.served_version()
        return s is not None and s != self.key_version

    @torch.no_grad()
    def sync_target(self, force: bool = False) -> str:
        """Snapshot the online encoders as the new target version and return it. The version the index serves stays
        compatible until the re-encoding to the new version is promoted. No-op (returns the unchanged version, see
        ``sync_info``) when the online encoders equal the target, or while a previous migration is pending; with
        ``force`` a pending, unpromoted version is abandoned (the served version stays compatible)."""
        if torch.equal(self._digest_of(self._onlines()), self.target_digest.cpu()):
            self.sync_info = {"synced": False, "reason": "unchanged", "version": self.key_version}
            return self.key_version
        served = self.served_version()
        if served is not None and served != self.key_version:
            if not force:
                self.sync_info = {"synced": False, "reason": "migration-pending", "served": served,
                                  "pending": self.key_version, "version": self.key_version}
                return self.key_version
            # forced: keep the served version in the compatibility slot, drop the unpromoted target
        else:
            self.prev_n.copy_(self.target_n)
            self.prev_digest.copy_(self.target_digest)
        abandoned = self.key_version if served is not None and served != self.key_version else None
        for tgt, src in zip(self._targets(), self._onlines()):
            tgt.load_state_dict(src.state_dict())
        self.target_n += 1
        self._refresh_digest()
        self.sync_info = {"synced": True, "version": self.key_version, "abandoned": abandoned,
                          "compatible": sorted(self.compatible_versions())}
        return self.key_version

    def await_version(self, version: Optional[str]) -> None:
        """Serve empty memory (instead of raising) while the index's current version is incompatible and a
        re-encoding to ``version`` is scheduled (used by checkpoint restore); cleared once it is served."""
        self._awaiting = version

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
        self._awaiting = None

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
        ids, scores [B, k] (-inf = empty slot; ``ids[b][j]`` is the segment of ``scores[b, j]``, empty slots are
        trailing)).

        ``M_t.refs[b][i]`` is the scalar source id ``"seg:<segment id>"`` of the segment the token belongs to;
        ``resolve_ref`` / ``context_event_ids`` expand it to the exact source event ids (originals via
        ``EventArchive.get``). ``M_t.time`` = segment end time minus ``now`` (days; ``now`` [B] or None = 0).
        Searches the index's current version (it must be one this module produced; an explicitly requested
        incompatible ``version`` raises, as does an incompatible current version unless a restore scheduled the
        re-encoding, in which case memory is empty until the new version is served)."""
        k = int(k or self.cfg.mem_top_k)
        B = q.shape[0]
        dev = q.device
        idx = self.index
        self.retrieve_stats["calls"] += 1
        if idx is None or not getattr(self.cfg, "use_memory", True) or k == 0:
            return self._empty(B, dev)
        v = version or idx.current_version
        if v is None or idx.count(v) == 0:
            return self._empty(B, dev)
        if v not in self.compatible_versions():
            if version is None and self._awaiting is not None:
                self.retrieve_stats["empty_awaiting_migration"] += 1
                return self._empty(B, dev)
            raise IncompatibleMemoryVersion(
                f"index version {v} was not produced by this retriever (compatible: {sorted(self.compatible_versions())})")
        if self._awaiting is not None and version is None:
            self._awaiting = None
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
        mask = torch.from_numpy(valid).to(dev).unsqueeze(-1).expand(B, k, T).reshape(B, k * T)
        refs = [[(SEG_REF + s if s is not None else None) for s in row for _ in range(T)] for row in res.seg_ids]
        ts = TokenSet(x.reshape(B, k * T, D), mask, torch.full((B, k * T), int(TokenKind.MEMORY), dtype=torch.long,
                                                                   device=dev),
                      tm.reshape(B, k * T), torch.full((B, k * T), -1, dtype=torch.long, device=dev), refs)
        ids = [[s for s in row if s is not None] for row in res.seg_ids]
        return ts, ids, res.scores

    def _empty(self, B: int, dev):
        ts = TokenSet.empty(B, self.d, device=dev)
        ts.refs = [[] for _ in range(B)]
        return ts, [[] for _ in range(B)], torch.zeros(B, 0, device=dev)

    # ------------------------------------------------------------------ source links of retrieved tokens
    def resolve_ref(self, ref) -> list:
        """A memory token's ref ``"seg:<id>"`` -> the exact source event ids of that segment (any level)."""
        if not isinstance(ref, str) or not ref.startswith(SEG_REF):
            raise ValueError(f"not a memory segment ref: {ref!r}")
        if self.index is None:
            raise RuntimeError("no memory index attached")
        return self.index.source_events(ref[len(SEG_REF):])

    def context_event_ids(self, ts: TokenSet) -> list:
        """[B] lists of unique source event ids behind the MEMORY tokens of ``ts`` (e.g. for
        ``DecisionRecord.context_refs``); non-memory refs (event ids, artifact refs) are passed through."""
        out = []
        for row in (ts.refs or [[] for _ in range(ts.B)]):
            seen, ids = set(), []
            for r in row:
                if r is None:
                    continue
                for e in (self.resolve_ref(r) if isinstance(r, str) and r.startswith(SEG_REF) else [r]):
                    if e not in seen:
                        seen.add(e)
                        ids.append(e)
            out.append(ids)
        return out

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
