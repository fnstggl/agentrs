"""Replay-sequence filtering and the world objective of CamUnifiedModel (moved out of model.py by the seam commit,
plan 2026-11-26 §3; owner W6).

    filter_sequence(model, seq, aux=None) -> FilterResult     acting-consistent filter over a tokenised SequenceBatch
    world_loss(model, seq, filt) -> dict                       WorldModelP.loss with that filter's context

`CamUnifiedModel.filter_sequence` / `.world_loss` are one-line delegations to these functions. Every model call goes
through the model's own methods (`model.memory_tokens`, `model.belief_view`, ...), so instance-level overrides keep
applying.

Import rule: model.py imports this module at import time, so this module never imports `..model` at module level;
model-level helpers and record types are imported inside the functions that use them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from ..interfaces import TokenSet

if TYPE_CHECKING:                                    # pragma: no cover
    from ..model import CamUnifiedModel, FilterResult, ReplayAux


def filter_sequence(model: "CamUnifiedModel", seq, aux: Optional["ReplayAux"] = None) -> "FilterResult":
    """Run F_theta over a tokenised SequenceBatch with the acting context: Z_t = F(Z_{t-1}, [O_t; A_{t-1};
    TIME(dt_{t-1}); B_t; option_{t-1}; M_t; G_t]), where B_t is re-filtered along the sequence by the belief module
    from the stored outcomes (B_t has seen o_{<t}), M_t is rebuilt from the stored retrieved ids (when the
    retriever can) and option_{t-1} comes from the stored option decisions. Rows whose previous step ended an
    episode restart from the initial state and a fresh belief."""
    from ..model import FilterResult, fast_where, with_mask
    cfg = model.cfg
    steps = seq.steps
    B = steps[0].obs.B
    dev = steps[0].obs.x.device
    Z0 = model.core.init_state(B, device=dev).float()
    Z = Z0
    fast = model.beliefs.init_fast(B, device=dev) if cfg.use_fast_adaptation else None
    prev, prev_done = None, torch.zeros(B, dtype=torch.bool, device=dev)
    res = FilterResult([], [], [])
    with model._autocast():
        for t, st in enumerate(steps):
            if bool(prev_done.any()):
                Z = torch.where(prev_done.view(B, 1, 1), Z0.to(Z.dtype), Z)
                if fast is not None:
                    fast = fast_where(prev_done, model.beliefs.init_fast(B, device=dev), fast)
            alive = ~prev_done
            Bt = model.beliefs.tokens(fast) if fast is not None else model.empty_tokens(B)
            opt_tok = None
            if model.option_policy is not None and aux is not None:
                o = torch.tensor(aux.option_prev[t], dtype=torch.long, device=dev)
                s = torch.tensor(aux.option_steps_prev[t], dtype=torch.long, device=dev)
                opt_tok = model._option_token(torch.where(alive, o, torch.full_like(o, -1)), s)
            M = None
            if cfg.use_memory and aux is not None and any(aux.retrieved[t]):
                M = model.memory_tokens(aux.retrieved[t], aux.now[t])
                res.memory_rebuilt = res.memory_rebuilt or M is not None
            A_prev = None if prev is None else with_mask(prev.act, alive)
            dtp = (torch.full((B,), -1.0, device=dev) if prev is None else
                   torch.where(alive, prev.dt.float().to(dev), torch.full((B,), -1.0, device=dev)))
            TIME = model._time_tokens(dtp)
            ctx = model.context(st.obs, A_prev, TIME, Bt, opt_tok, M, st.goal)
            Z, _ = model.core(Z, ctx)
            Z = Z.float()
            res.Zs.append(Z)
            res.beliefs.append(TokenSet.cat(Bt, opt_tok))
            res.memory.append(M if M is not None else model.empty_tokens(B))
            if fast is not None and bool(st.outcomes.y_mask.any()):
                fast, _ = model.beliefs.update(fast, Z, st.act, model.belief_view(st.outcomes), model.outcome_head)
                res.n_belief_updates += 1
            prev, prev_done = st, st.done.to(torch.bool).to(dev)
    return res


def world_loss(model: "CamUnifiedModel", seq, filt: Optional["FilterResult"]) -> dict:
    """WorldModelP.loss with the acting-consistent belief / memory context when the world model accepts it."""
    kw = {}
    if filt is not None:
        if "beliefs" in model._wm_loss_ctx:
            kw["beliefs"] = filt.beliefs
        if "memory" in model._wm_loss_ctx and model.cfg.use_memory:
            kw["memory"] = filt.memory
    return model.world_model.loss(seq, **kw)
