"""Generalised entities and relationships (workstream B; implements `EntityEncoderP`).

    e_i  = E_theta(type_i, attributes_i, semantic_i, relationships_i, history_i)
    r_ij = R_theta(e_i, e_j, relation observations, time)

Input: observation tokens (`TokenSet` [B,N,D] from the ObservationTokenizer: OBS_SEM / OBS_NUM / EVENT / RELATION ...)
whose `entity` field holds the episode-local slot (EntityRegistry) of the primary entity and whose `entity2` holds the
second entity of RELATION / MESSAGE / TRANSACTION observations.

* Entities: every slot that occurs in `entity` or `entity2` of a valid token, per row (variable counts, padded).
  attributes / semantic / history: the observation tokens bound to the slot, embedded with their kind and their
  continuous time, are pooled two ways — scatter-mean, and multi-head attention pooling with a query conditioned on
  that mean — plus history statistics (log count, most recent and earliest time). type: the semantic embedding of the
  entity's free-text type when `type_tokens` are given (extensible categories, no per-category head); a learned
  "unknown type" vector otherwise.
* Relations: one token per observed ordered pair (i, j): attention pooling over the pair's relation observations,
  combined with e_i, e_j and the time of the latest relation observation.
* relationships_i: `n_rounds` transformer blocks of attention over [entities; relations] with a learned bias toward
  incident (entity, relation) pairs (graph-biased attention), so entities absorb their relations and relations their
  endpoints.

Slot identity is used for grouping only (equality), never as an embedding: relabelling slots permutes the outputs.
Output: TokenSet of ENTITY tokens (entity = slot, entity2 = -1) followed by RELATION tokens (entity = i, entity2 = j),
time = latest observation time, refs "entity:<slot>" / "relation:<i>-><j>".

Width (W3, DOM): the encoder works at the satellite width W = `cfg.d_satellite` (0 = d_model) with an input projection
d_model -> W after the token norm and an output projection W -> d_model after the output norm, so at the 1B scale it
costs ~W^2 rather than d_model^2 per matrix (the output tokens always have the core's width d_model). With W = d_model
there are no projections (the pre-DOM encoder exactly).

Types: `type_tokens` (e.g. `make_type_tokens(registries, semantic)` = `entity_type_tokens` over the semantic model's
`embed` of the registry's free-text types, cached per semantic version) are consumed through `type_proj`
(type_dim = `cfg.sem_xattn_dim` when it is set, i.e. the semantic model's hidden width; else d_model).
"""
from __future__ import annotations

import contextlib
import logging
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ..config import CamConfig
from ..interfaces import EntityRegistry, TokenKind, TokenSet
from .blocks import (Block, KindEmbedding, RMSNorm, SwiGLU, TimeEncoding, count_parameters,
                     init_transformer_weights, signed_log_time)

_NEG = -1e9
log = logging.getLogger(__name__)


class _SlotPool(nn.Module):
    """Multi-head attention pooling of tokens u [B,N,D] into groups given by assign [B,G,N] (bool), with group-
    specific queries q [B,G,D]. Returns [B,G,D] (zero for empty groups)."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.q = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, u: Tensor, q: Tensor, assign: Tensor) -> Tensor:
        B, N, D = u.shape
        G = q.shape[1]
        k = self.k(u).view(B, N, self.h, self.dh).transpose(1, 2)                # [B,H,N,dh]
        v = self.v(u).view(B, N, self.h, self.dh).transpose(1, 2)
        qq = self.q(q).view(B, G, self.h, self.dh).transpose(1, 2)               # [B,H,G,dh]
        s = (qq @ k.transpose(-1, -2)) / self.dh ** 0.5                          # [B,H,G,N]
        s = s.float().masked_fill(~assign[:, None], _NEG)
        a = torch.softmax(s, -1) * assign[:, None].float()
        out = (a.to(v.dtype) @ v).transpose(1, 2).reshape(B, G, D)
        nonempty = assign.any(-1, keepdim=True)
        return self.o(out) * nonempty.to(out.dtype)


class EntityEncoder(nn.Module):
    """forward(obs_tokens TokenSet [B,N,D], type_tokens TokenSet [B,T,type_dim] | None) -> TokenSet [B,E+P,D].

    Args:
        cfg: CamConfig (d_model, n_heads, d_ff, max_entities, n_fourier).
        n_rounds: graph-biased attention rounds over [entities; relations] (one round already gives every entity its
            relations, which carry the partner's embedding: one-hop relational information).
        ffn_hidden: hidden width of the entity / relation SwiGLUs (default d_ff / 2).
        type_dim: dimension of the type token vectors (e.g. the semantic encoder's sem_dim); default d_model.
        max_relations: per-row working budget of relation tokens (most recent pairs kept; drops are counted).
        strict_slots: raise on entity slots >= cfg.max_entities instead of counting them. Such tokens cannot be
            bound (the slot range is the per-step entity budget); by default they are excluded from grouping and
            counted in `stats["out_of_range_tokens"]` (with a one-time warning) — never dropped silently.
    """

    def __init__(self, cfg: CamConfig, n_rounds: int = 1, type_dim: Optional[int] = None,
                 max_relations: Optional[int] = None, ffn_hidden: Optional[int] = None,
                 autocast: Optional[bool] = None, strict_slots: bool = False, width: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        D_core, H = cfg.d_model, cfg.n_heads
        D = int(width or getattr(cfg, "d_satellite", 0) or D_core)          # satellite width W
        if D % H:
            raise ValueError(f"EntityEncoder width {D} is not divisible by n_heads={H}")
        d_ff = cfg.d_ff if D == D_core else int(round(cfg.ffn_mult * D / 64)) * 64
        hid = ffn_hidden or max(64, (d_ff // 2) // 64 * 64)
        self.width, self.d_core = D, D_core
        if type_dim is None:
            type_dim = int(getattr(cfg, "sem_xattn_dim", 0) or 0) or D_core
        self.type_dim = type_dim
        self.E = cfg.max_entities
        self.max_relations = max_relations or 2 * cfg.max_entities
        self.autocast = autocast
        self.tok_norm = RMSNorm(D_core)
        self.in_proj = nn.Linear(D_core, D, bias=False) if D != D_core else None
        self.kind_emb = KindEmbedding(D)
        self.time_enc = TimeEncoding(D, n_freq=cfg.n_fourier)
        # entity encoder E_theta
        self.mean_proj = nn.Linear(D, D)
        self.pool_query = nn.Parameter(torch.randn(D) * 0.02)
        self.ent_pool = _SlotPool(D, H)
        self.type_proj = nn.Linear(type_dim, D)
        self.unknown_type = nn.Parameter(torch.randn(D) * 0.02)
        self.hist = nn.Linear(4, D)
        self.ent_in = nn.Linear(4 * D, D)
        self.ent_ffn = SwiGLU(D, hid)
        self.ent_norm = RMSNorm(D)
        # relation encoder R_theta
        self.rel_query = nn.Parameter(torch.randn(D) * 0.02)
        self.rel_pool = _SlotPool(D, H)
        self.rel_in = nn.Linear(4 * D, D)
        self.rel_ffn = SwiGLU(D, hid)
        self.rel_norm = RMSNorm(D)
        # relational rounds
        self.rounds = nn.ModuleList([Block(D, H, d_ff) for _ in range(n_rounds)])
        self.incident_bias = nn.Parameter(torch.full((H,), 1.0))
        self.out_norm = RMSNorm(D)
        self.out_proj = nn.Linear(D, D_core, bias=False) if D != D_core else None
        init_transformer_weights(self, max(1, n_rounds))
        nn.init.normal_(self.kind_emb.weight, std=0.5)                      # comparable to RMS-normalised tokens
        nn.init.normal_(self.time_enc.proj.weight, std=0.5 / self.time_enc.proj.in_features ** 0.5)
        self.strict_slots = strict_slots
        self.stats = {"calls": 0, "dropped_relations": 0, "out_of_range_tokens": 0}
        self._warned = False
        self.last_counts: Optional[tuple] = None
        self._type_cache: dict = {}

    def make_type_tokens(self, registries: Sequence[EntityRegistry], semantic, device=None) -> TokenSet:
        """Type tokens for `forward` from the registries' entity types, embedded by the semantic model's `embed`
        (cached by (semantic version, type string); the cache never outlives an encoder version)."""
        ver = getattr(semantic, "version", None)
        cache = self._type_cache.get(ver)
        if cache is None:
            self._type_cache = {ver: {}}
            cache = self._type_cache[ver]
        return entity_type_tokens(registries, semantic.embed, device=device, cache=cache)

    def _autocast_ctx(self, device):
        use = self.autocast if self.autocast is not None else device.type == "cuda"
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use else contextlib.nullcontext()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _masked_time(time: Tensor, assign: Tensor, largest: bool) -> Tensor:
        """Latest / earliest token time per group: time [B,N], assign [B,G,N] -> [B,G] (0 for empty groups)."""
        fill = float("-inf") if largest else float("inf")
        t = time.unsqueeze(1).expand_as(assign).masked_fill(~assign, fill)
        r = t.amax(-1) if largest else t.amin(-1)
        return torch.where(torch.isfinite(r), r, torch.zeros_like(r))

    def _relation_groups(self, ent: Tensor, ent2: Tensor, relmask: Tensor, time: Tensor) -> tuple:
        """Unique ordered pairs per row. Returns (assign [B,P,N] bool, pi [B,P], pj [B,P], pmask [B,P])."""
        B, N = ent.shape
        E = self.E
        big = E * E + 1
        pid = torch.where(relmask, ent.clamp(0, E - 1) * E + ent2.clamp(0, E - 1), torch.full_like(ent, big))
        spid, order = pid.sort(1)
        newg = torch.ones_like(spid, dtype=torch.bool)
        if N > 1:
            newg[:, 1:] = spid[:, 1:] != spid[:, :-1]
        gs = newg.long().cumsum(1) - 1
        grp = torch.empty_like(gs).scatter_(1, order, gs)                      # group id per original token
        n_pairs = (newg & (spid != big)).sum(1)                                 # [B]
        P = int(n_pairs.max()) if B else 0
        if P == 0:
            z = torch.zeros(B, 0, dtype=torch.long, device=ent.device)
            return torch.zeros(B, 0, N, dtype=torch.bool, device=ent.device), z, z, z.bool()
        assign = (grp.unsqueeze(1) == torch.arange(P, device=ent.device)[None, :, None]) & relmask.unsqueeze(1)
        pmask = torch.arange(P, device=ent.device)[None] < n_pairs[:, None]
        # endpoints of each group: any member's (entity, entity2)
        first = torch.where(assign, torch.arange(N, device=ent.device)[None, None], N).amin(-1).clamp(max=N - 1)
        pi = torch.gather(ent, 1, first).masked_fill(~pmask, -1)
        pj = torch.gather(ent2, 1, first).masked_fill(~pmask, -1)
        if P > self.max_relations:                                             # working budget: most recent pairs
            latest = self._masked_time(time, assign, True).masked_fill(~pmask, float("-inf"))
            keep = torch.topk(latest, self.max_relations, dim=1).indices.sort(1).values
            dropped = int((pmask.sum(1) - torch.gather(pmask, 1, keep).sum(1)).sum())
            self.stats["dropped_relations"] += dropped
            assign = torch.gather(assign, 1, keep.unsqueeze(-1).expand(-1, -1, N))
            pi, pj, pmask = torch.gather(pi, 1, keep), torch.gather(pj, 1, keep), torch.gather(pmask, 1, keep)
        return assign, pi, pj, pmask

    # ------------------------------------------------------------------ forward
    def forward(self, obs_tokens: TokenSet, type_tokens: Optional[TokenSet] = None) -> TokenSet:
        obs = obs_tokens
        B, N, _ = obs.x.shape
        D = self.width
        E = self.E
        dev = obs.x.device
        self.stats["calls"] += 1
        ent = obs.entity
        ent2 = obs._e2()
        oor = obs.mask & ((ent >= E) | (ent2 >= E))
        n_oor = int(oor.sum())
        if n_oor:
            if self.strict_slots:
                raise ValueError(f"EntityEncoder: {n_oor} tokens bound to entity slots >= max_entities={E}")
            self.stats["out_of_range_tokens"] += n_oor
            if not self._warned:
                self._warned = True
                log.warning("EntityEncoder: %d tokens bound to entity slots >= max_entities=%d are not grouped "
                            "(registry larger than cfg.max_entities?); counted in stats", n_oor, E)
        v1 = obs.mask & (ent >= 0) & (ent < E)
        v2 = obs.mask & (ent2 >= 0) & (ent2 < E)
        slots = torch.arange(E, device=dev)
        assign_all = (ent.unsqueeze(1) == slots[None, :, None]) & v1.unsqueeze(1)          # [B,E,N] primary
        assign2_all = (ent2.unsqueeze(1) == slots[None, :, None]) & v2.unsqueeze(1)        # [B,E,N] second
        present = assign_all.any(-1) | assign2_all.any(-1)                                  # [B,E]
        n_ent = present.sum(1)
        Eo = int(n_ent.max()) if B else 0
        # compact present slots (ascending slot order) to [B,Eo]
        order = torch.argsort((~present).float() * (E + 1) + slots.float()[None], dim=1)[:, :Eo]
        emask = torch.arange(Eo, device=dev)[None] < n_ent[:, None]
        eslot = order.masked_fill(~emask, -1)
        assign = torch.gather(assign_all, 1, order.unsqueeze(-1).expand(-1, -1, N)) & emask.unsqueeze(-1)
        assign2 = torch.gather(assign2_all, 1, order.unsqueeze(-1).expand(-1, -1, N)) & emask.unsqueeze(-1)

        with self._autocast_ctx(dev):
            u = self.tok_norm(obs.x.float())
            if self.in_proj is not None:
                u = self.in_proj(u)
            u = u + self.kind_emb(obs.kind) + self.time_enc(obs.time)                            # [B,N,W]
            u = u * obs.mask.unsqueeze(-1).to(u.dtype)
            # ---- entities
            af = assign.float()
            cnt = af.sum(-1)                                                                # [B,Eo]
            mean = (af @ u.float()) / cnt.clamp_min(1).unsqueeze(-1)
            q = self.pool_query + self.mean_proj(mean)
            pooled = self.ent_pool(u, q, assign)
            if type_tokens is not None and type_tokens.N > 0:
                if type_tokens.x.shape[-1] != self.type_proj.in_features:
                    raise ValueError(f"EntityEncoder: type tokens of width {type_tokens.x.shape[-1]}, expected "
                                     f"type_dim={self.type_proj.in_features}")
                tv = type_tokens.mask & (type_tokens.entity >= 0)
                tassign = (type_tokens.entity.unsqueeze(1) == eslot.unsqueeze(-1)) & tv.unsqueeze(1) \
                    & emask.unsqueeze(-1)                                                   # [B,Eo,T]
                tf = tassign.float()
                tvec = (tf @ type_tokens.x.float()) / tf.sum(-1, keepdim=True).clamp_min(1)
                has_t = tassign.any(-1, keepdim=True)
                typ = torch.where(has_t, self.type_proj(tvec), self.unknown_type.expand(B, Eo, D))
            else:
                typ = self.unknown_type.expand(B, Eo, D)
            all_assign = assign | assign2
            t_last = self._masked_time(obs.time, all_assign, True)
            t_first = self._masked_time(obs.time, all_assign, False)
            hist = torch.stack([torch.log1p(cnt), torch.log1p(assign2.float().sum(-1)),
                                signed_log_time(t_last) / 10, signed_log_time(t_first) / 10], -1)
            h_e = self.ent_in(torch.cat([pooled, mean.to(pooled.dtype), typ.to(pooled.dtype),
                                         self.hist(hist).to(pooled.dtype)], -1))
            h_e = h_e + self.ent_ffn(self.ent_norm(h_e))
            # ---- relations
            relmask = v1 & v2
            rassign, pi, pj, pmask = self._relation_groups(ent, ent2, relmask, obs.time)
            P = rassign.shape[1]
            if P:
                # slot -> compact entity position
                pos = torch.full((B, E + 1), Eo, dtype=torch.long, device=dev)
                pos.scatter_(1, eslot.clamp(min=0).masked_fill(~emask, E), torch.arange(Eo, device=dev).expand(B, -1)
                             .masked_fill(~emask, Eo))
                h_e_pad = torch.cat([h_e, torch.zeros(B, 1, D, device=dev, dtype=h_e.dtype)], 1)
                ii = torch.gather(pos, 1, pi.clamp(min=0).masked_fill(~pmask, E))
                jj = torch.gather(pos, 1, pj.clamp(min=0).masked_fill(~pmask, E))
                e_i = torch.gather(h_e_pad, 1, ii.unsqueeze(-1).expand(-1, -1, D))
                e_j = torch.gather(h_e_pad, 1, jj.unsqueeze(-1).expand(-1, -1, D))
                r_pool = self.rel_pool(u, self.rel_query + (e_i + e_j).to(u.dtype), rassign)
                rt = self._masked_time(obs.time, rassign, True)
                h_r = self.rel_in(torch.cat([e_i, e_j, r_pool.to(e_i.dtype), self.time_enc(rt).to(e_i.dtype)], -1))
                h_r = h_r + self.rel_ffn(self.rel_norm(h_r))
                # graph bias: entity a <-> relation p incident when slot(a) is an endpoint of p
                inc = (eslot.unsqueeze(-1) == pi.unsqueeze(1)) | (eslot.unsqueeze(-1) == pj.unsqueeze(1))
                inc = inc & emask.unsqueeze(-1) & pmask.unsqueeze(1)                       # [B,Eo,P]
            else:
                h_r = torch.zeros(B, 0, D, device=dev, dtype=h_e.dtype)
                rt = torch.zeros(B, 0, device=dev)
                inc = torch.zeros(B, Eo, 0, dtype=torch.bool, device=dev)
            # ---- relational rounds over [entities; relations]
            h = torch.cat([h_e.float() + self.kind_emb.weight[int(TokenKind.ENTITY)],
                           h_r.float() + self.kind_emb.weight[int(TokenKind.RELATION)]], 1)
            L = Eo + P
            mask = torch.cat([emask, pmask], 1)
            if L and len(self.rounds):
                adj = torch.zeros(B, L, L, dtype=torch.bool, device=dev)
                adj[:, :Eo, Eo:] = inc
                adj[:, Eo:, :Eo] = inc.transpose(1, 2)
                bias = adj.unsqueeze(1).float() * self.incident_bias.view(1, -1, 1, 1)     # [B,H,L,L]
                for blk in self.rounds:
                    h = blk(h, mask, bias)
            h = self.out_norm(h)
            if self.out_proj is not None:
                h = self.out_proj(h)
            h = h.float() * mask.unsqueeze(-1).float()

        kind = torch.cat([torch.full((B, Eo), int(TokenKind.ENTITY), dtype=torch.long, device=dev),
                          torch.full((B, P), int(TokenKind.RELATION), dtype=torch.long, device=dev)], 1)
        kind = kind.masked_fill(~mask, int(TokenKind.PAD))
        time = torch.cat([t_last, rt], 1) * mask.float()
        entity = torch.cat([eslot, pi], 1).masked_fill(~mask, -1)
        entity2 = torch.cat([torch.full((B, Eo), -1, dtype=torch.long, device=dev), pj], 1).masked_fill(~mask, -1)
        refs = []
        for b in range(B):
            es, ps_i, ps_j = eslot[b].tolist(), pi[b].tolist(), pj[b].tolist()
            refs.append([f"entity:{s}" if s >= 0 else None for s in es]
                        + [f"relation:{i}->{j}" if i >= 0 else None for i, j in zip(ps_i, ps_j)])
        self.last_counts = (n_ent.detach(), pmask.sum(1).detach())
        return TokenSet(h, mask, kind, time, entity, refs, entity2)

    def count_parameters(self) -> int:
        return count_parameters(self)


def entity_type_tokens(registries: Sequence[EntityRegistry], embed_fn: Callable[[list], Tensor],
                       device=None, cache: Optional[dict] = None) -> TokenSet:
    """Type tokens for EntityEncoder from EntityRegistry type strings: one token per bound entity with a type, x =
    embed_fn(type strings) (e.g. SemanticEncoder.embed -> [n, sem_dim]), entity = its slot. Embeddings are cached by
    type string in `cache` (pass a dict to reuse across calls)."""
    cache = {} if cache is None else cache
    rows = [[(reg.slot[eid], t) for eid, t in reg.etype.items() if eid in reg.slot] for reg in registries]
    new = sorted({t for r in rows for _, t in r if t not in cache})
    if new:
        emb = embed_fn(new)
        for t, e in zip(new, emb):
            cache[t] = e.detach()
    B = len(rows)
    T = max([len(r) for r in rows] + [0])
    dim = next(iter(cache.values())).shape[-1] if cache else 1
    x = torch.zeros(B, T, dim, device=device)
    ent = torch.full((B, T), -1, dtype=torch.long, device=device)
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b, r in enumerate(rows):
        for n, (s, t) in enumerate(r):
            x[b, n] = cache[t].to(device=device, dtype=x.dtype)
            ent[b, n] = s
            mask[b, n] = True
    return TokenSet(x, mask, torch.full((B, T), int(TokenKind.ENTITY), dtype=torch.long, device=device),
                    torch.zeros(B, T, device=device), ent, [[t for _, t in r] + [None] * (T - len(r)) for r in rows])
