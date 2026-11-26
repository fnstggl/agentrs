"""Memory retrieval runtime of CamUnifiedModel (moved out of model.py by the seam commit, plan 2026-11-26 §3;
owner W9).

    retrieve(model, Z_prev, G, Bt, filters=None, now=None) -> (MEMORY TokenSet, ids [B][k], scores)
    memory_tokens(model, ids, now=None) -> TokenSet | None          differentiable MEMORY tokens of stored ids
    stored_memory_tokens(model, ids, now=None) -> TokenSet | None   the index's stored values (no gradient)

`CamUnifiedModel.retrieve` / `.memory_tokens` / `.stored_memory_tokens` are one-line delegations to these functions.

Import rule: model.py imports this module at import time, so this module never imports `..model` at module level;
model-level helpers are imported inside the functions that use them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from ..interfaces import TokenKind, TokenSet

if TYPE_CHECKING:                                    # pragma: no cover
    from ..model import CamUnifiedModel


def retrieve(model: "CamUnifiedModel", Z_prev: torch.Tensor, G: TokenSet, Bt: TokenSet,
             filters: Optional[list] = None, now: Optional[list] = None) -> tuple:
    """M_t = Retrieve(Q(Z_{t-1}, G_t, B_t)) -> (TokenSet MEMORY [B,<=k,D], ids [B][k], scores)."""
    from ..model import ShapeError, as_kind, check_tokens
    B = Z_prev.shape[0]
    model._count("retriever.retrieve")
    q = model.retriever.query(Z_prev, G, Bt)
    if q.dim() != 2 or q.shape[0] != B:
        raise ShapeError(f"retriever.query: {tuple(q.shape)} != [B={B}, key_dim]")
    if model._retrieve_now:
        M, ids, scores = model.retriever.retrieve(q, model.cfg.mem_top_k, filters, now=now)
    else:
        M, ids, scores = model.retriever.retrieve(q, model.cfg.mem_top_k, filters)
    check_tokens(M, B, model.cfg.d_model, "retriever.retrieve")
    return as_kind(M, TokenKind.MEMORY), ids, scores


def memory_tokens(model: "CamUnifiedModel", ids: list, now: Optional[list] = None) -> Optional[TokenSet]:
    """Differentiable MEMORY tokens of stored retrieved ids (None when the retriever cannot rebuild them)."""
    from ..model import as_kind, check_tokens
    if model.retriever is None:
        return None
    if hasattr(model.retriever, "memory_tokens"):
        M = model.retriever.memory_tokens(ids, now)
        check_tokens(M, len(ids), model.cfg.d_model, "retriever.memory_tokens")
        return as_kind(M, TokenKind.MEMORY)
    return model.stored_memory_tokens(ids, now)


@torch.no_grad()
def stored_memory_tokens(model: "CamUnifiedModel", ids: list, now: Optional[list] = None) -> Optional[TokenSet]:
    """MEMORY tokens of stored retrieved segment ids read back from the retriever's memory index (E's
    MemoryIndex: the same stored values `retrieve` returned when acting, as long as the served embedding version
    is unchanged), laid out as `Retriever.retrieve` lays them out: [B, k*T, D], refs "seg:<id>", time = segment
    end - now. No gradient (the values are the index's, as when acting). None without a readable index."""
    r = model.retriever
    idx = getattr(r, "index", None)
    version = r.served_version() if callable(getattr(r, "served_version", None)) else None
    if idx is None or version is None or not any(ids):
        return None
    B, D = len(ids), model.cfg.d_model
    T = int(getattr(r, "n_mem", model.cfg.mem_segment_tokens))
    k = max(len(i) for i in ids)
    dev = model.device
    x = torch.zeros(B, k, T, D, device=dev)
    tm = torch.zeros(B, k, T, device=dev)
    valid = torch.zeros(B, k, dtype=torch.bool, device=dev)
    flat = [str(s) for row in ids for s in row]
    kids = idx.kids_of(version, flat)
    metas = {m["seg_id"]: m for m in idx.segments([s for s, kd in zip(flat, kids) if kd >= 0])}
    have = [kd for kd in kids if kd >= 0]
    vals = idx.values(version, have).to(dev) if have else None
    p = q = 0
    for b, row in enumerate(ids):
        nb = float(now[b]) if now is not None else 0.0
        for j, s in enumerate(row):
            if kids[p] >= 0 and str(s) in metas:
                x[b, j] = vals[q].to(x.dtype)
                tm[b, j] = float(metas[str(s)]["t_end"]) - nb
                valid[b, j] = True
                q += 1
            elif kids[p] >= 0:
                q += 1
            p += 1
    model._count("memory.stored_tokens", int(valid.sum()))
    refs = [[(f"seg:{row[j]}" if j < len(row) else None) for j in range(k) for _ in range(T)] for row in ids]
    return TokenSet(x.reshape(B, k * T, D), valid.unsqueeze(-1).expand(B, k, T).reshape(B, k * T),
                    torch.full((B, k * T), int(TokenKind.MEMORY), dtype=torch.long, device=dev),
                    tm.reshape(B, k * T), torch.full((B, k * T), -1, dtype=torch.long, device=dev), refs)
