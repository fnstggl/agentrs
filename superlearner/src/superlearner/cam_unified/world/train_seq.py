"""Replay-sequence filtering and the world objective of CamUnifiedModel (moved out of model.py by the seam commit,
plan 2026-11-26 §3; owner W6).

    filter_sequence(model, seq, aux=None) -> FilterResult                acting-consistent filter over a replayed
    filter_window(model, seq, aux=None, burn_in=None) -> FilterResult    SequenceBatch (explicit burn-in)
    world_loss(model, seq, filt) -> dict                                  WorldModelP.loss on that filter's states

`CamUnifiedModel.filter_sequence` / `.world_loss` are one-line delegations to these functions. Every model call goes
through the model's own methods (`model.memory_tokens`, `model.belief_view`, ...), so instance-level overrides keep
applying.

DOM path (plan 2026-11-26 §2 / §3 W6; on when the config enables the corresponding switches):

* **burn-in** (cfg.burn_in, cfg.bptt_steps): the first `burn_in` decisions of a replay window are filtered without
  gradient; the losses are computed on the graded decisions after them (`FilterResult.burn_in`, see `burn_in_for`);
* **prior on R** (cfg.wm_transition_blocks > 0 with reasoning): the reasoned latent R_t = Reason(Z_t, [B_t; option;
  M_t; G_t]) of every step is computed exactly as `encode_step` does (`FilterResult.Rs`) and is the world-model
  prior's input, so the reasoner receives world-model gradient;
* **single filter** (cfg.wm_ema_target False or the block transition): `world_loss` hands the agent filter's states
  Z_t (with gradients) to `WorldModel.loss` (`Z_post=`), which runs no filter of its own; the latent target is
  sg(Z_post) (no EMA target encoder);
* **world-model likelihood for the belief update** (`belief_runtime.dom_beliefs`): the particle weights use
  feat = outcome_head.features(Z_hat_t) with Z_hat_t = WM.prior(R_t, A_t, B_t, dt_t) (mode latent), the same
  likelihood decide() reports, instead of the belief module's own action-conditioning block.
The world model's prior context is the pure belief tokens B_t (`FilterResult.belief_tokens`, as decide() passes
`enc.belief`); the legacy path keeps [B_t; option] (`FilterResult.beliefs`).

Import rule: model.py imports this module at import time, so this module never imports `..model` at module level;
model-level helpers and record types are imported inside the functions that use them.
"""
from __future__ import annotations

import contextlib
import inspect
from typing import TYPE_CHECKING, Callable, Optional

import torch

from ..interfaces import SequenceBatch, TokenSet

if TYPE_CHECKING:                                    # pragma: no cover
    from ..model import CamUnifiedModel, FilterResult, ReplayAux


# ============================================================================ switches

def prior_on_R(model) -> bool:
    """The world-model prior consumes the reasoned latent R_t (plan §2: Z_hat = WM.prior(R_t, A, B, dt) over the last
    cfg.wm_transition_blocks core blocks)."""
    cfg = model.cfg
    return int(getattr(cfg, "wm_transition_blocks", 0) or 0) > 0 and bool(cfg.use_reasoning) \
        and getattr(model, "reasoner", None) is not None


def single_filter(model) -> bool:
    """`world_loss` trains on the agent filter's own states (no second filter inside the world model)."""
    cfg = model.cfg
    if not ("Z_post" in _params(model.world_model.loss)):
        return False
    return (not bool(getattr(cfg, "wm_ema_target", True))) or int(getattr(cfg, "wm_transition_blocks", 0) or 0) > 0


def burn_in_for(cfg, T: int, burn_in: Optional[int] = None) -> int:
    """Burn-in decisions of a replay window of length T: `burn_in` when given, otherwise cfg.burn_in limited so that
    at least max(2, cfg.bptt_steps) graded decisions remain (a window of burn_in + bptt_steps decisions gets exactly
    cfg.burn_in; shorter windows, e.g. at an episode start, get less or none)."""
    if burn_in is not None:
        return max(0, min(int(burn_in), T - 1))
    b = int(getattr(cfg, "burn_in", 0) or 0)
    if b <= 0:
        return 0
    keep = max(2, int(getattr(cfg, "bptt_steps", 0) or 0))
    return max(0, min(b, T - keep))


def _params(fn) -> set:
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()


def reason_fn_of(model) -> Optional[Callable]:
    """R = Reason(Z_hat, [B; G]) for imagined states (open-loop steps >= 2 and forecasts), or None."""
    if not prior_on_R(model):
        return None

    def reason(Z, belief: Optional[TokenSet], goal: Optional[TokenSet]):
        parts = [t for t in (belief, goal) if t is not None and t.N > 0]
        ctx = TokenSet.cat(*parts) if parts else None
        R, _ = model.reasoner(Z, ctx, model.reason_iters)
        return R.float()
    return reason


# ============================================================================ filter

def filter_sequence(model: "CamUnifiedModel", seq, aux: Optional["ReplayAux"] = None,
                    burn_in: Optional[int] = None, *, sem=None, sem_rows=None) -> "FilterResult":
    """`filter_window` (burn-in None = the configured one, `burn_in_for`; the signature
    `CamUnifiedModel.filter_sequence` delegates to)."""
    return filter_window(model, seq, aux, burn_in, sem=sem, sem_rows=sem_rows)


def _sem_kw(fn, sem, sem_rows) -> dict:
    """sem / sem_rows keywords for a core or reasoner call that accepts them (W3); {} otherwise or without sem."""
    if sem is None:
        return {}
    ps = _params(fn)
    if "sem" not in ps:
        return {}
    kw = {"sem": sem}
    if sem_rows is not None and "sem_rows" in ps:
        kw["sem_rows"] = sem_rows
    return kw


def window_sem(seq, aux, sem=None):
    """The semantic context of a replay window: ``sem`` when given, else ``aux.sem`` / ``seq.sem`` (W1/W4 replay
    carries it there when the semantic stream is on), else None. One context for every step, or a list per step."""
    if sem is not None:
        return sem
    for o in (aux, seq):
        v = getattr(o, "sem", None) if o is not None else None
        if v is not None:
            return v
    return None


def filter_window(model: "CamUnifiedModel", seq, aux: Optional["ReplayAux"] = None,
                  burn_in: Optional[int] = None, *, sem=None, sem_rows=None) -> "FilterResult":
    """Run F_theta over a tokenised SequenceBatch with the acting context: Z_t = F(Z_{t-1}, [O_t; A_{t-1};
    TIME(dt_{t-1}); B_t; option_{t-1}; M_t; G_t]), where B_t is re-filtered along the sequence by the belief module
    from the stored outcomes (B_t has seen o_{<t}), M_t is rebuilt from the stored retrieved ids (when the
    retriever can) and option_{t-1} comes from the stored option decisions. Rows whose previous step ended an
    episode restart from the initial state and a fresh belief.

    Extra fields of the result: ``burn_in`` (decisions filtered without gradient, `burn_in_for`), ``Rs`` ([T] R_t,
    or None entries when the prior runs on Z), ``belief_tokens`` ([T] pure B_t) and ``prior_feat_updates`` (belief
    updates that used the world-model likelihood), ``sem`` (the window's semantic context, `window_sem`).

    ``sem``: the semantic context S_t (Mechanism C) of the window's rows (one for every step, or a list of T), with
    ``sem_rows`` [B] when it is aligned to source rows; passed to the core, the reasoner and the world-model prior
    exactly as decide() passes it (cores / reasoners without a semantic input ignore it)."""
    from ..core import belief_runtime as _belief
    from ..model import FilterResult, fast_where, with_mask
    cfg = model.cfg
    steps = seq.steps
    T = len(steps)
    B = steps[0].obs.B
    dev = steps[0].obs.x.device
    burn = burn_in_for(cfg, T, burn_in)
    use_R = prior_on_R(model)
    wm_feat = _belief.dom_beliefs(model)
    Z0 = model.core.init_state(B, device=dev).float()
    Z = Z0
    fast = model.beliefs.init_fast(B, device=dev) if cfg.use_fast_adaptation else None
    prev, prev_done = None, torch.zeros(B, dtype=torch.bool, device=dev)
    res = FilterResult([], [], [])
    res.burn_in = burn
    res.Rs = []
    res.belief_tokens = []
    res.ctxs, res.resets = [], []           # per-step core context and episode resets (EMA target pass)
    res.prior_feat_updates = 0
    sem_all = window_sem(seq, aux, sem)
    res.sem, res.sem_rows = sem_all, sem_rows
    with model._autocast():
        for t, st in enumerate(steps):
            grad = contextlib.nullcontext() if t >= burn else torch.no_grad()
            with grad:
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
                res.ctxs.append(ctx)
                res.resets.append(prev_done)
                sem_t = sem_all[t] if isinstance(sem_all, (list, tuple)) else sem_all
                Z, _ = model.core(Z, ctx, **_sem_kw(model.core.forward, sem_t, sem_rows))
                Z = Z.float()
                R = None
                if use_R:
                    R, _ = model.reasoner(Z, TokenSet.cat(Bt, opt_tok, M, st.goal), model.reason_iters,
                                          **_sem_kw(model.reasoner.forward, sem_t, sem_rows))
                    R = R.float()
                res.Zs.append(Z)
                res.Rs.append(R)
                res.beliefs.append(TokenSet.cat(Bt, opt_tok))
                res.belief_tokens.append(Bt)
                res.memory.append(M if M is not None else model.empty_tokens(B))
                if fast is not None and bool(st.outcomes.y_mask.any()):
                    ob = model.belief_view(st.outcomes)
                    feat = None
                    if wm_feat:
                        pkw = {} if sem_t is None else {"sem": sem_t, "sem_rows": sem_rows}
                        if pkw and "sem" not in _params(model.world_model.prior):
                            pkw = {}
                        Z_hat, _ = model.world_model.prior(R if R is not None else Z, st.act, Bt, st.dt,
                                                           sample=False, **pkw)
                        feat = model.outcome_head.features(Z_hat, ob)
                        res.prior_feat_updates += 1
                    if feat is None:            # legacy call shape (any BeliefModuleP)
                        fast, _ = model.beliefs.update(fast, Z, st.act, ob, model.outcome_head)
                    else:
                        fast, _ = model.beliefs.update(fast, Z, st.act, ob, model.outcome_head, feat=feat)
                    res.n_belief_updates += 1
                prev, prev_done = st, st.done.to(torch.bool).to(dev)
    return res


# ============================================================================ world objective

@torch.no_grad()
def ema_targets(model, filt) -> Optional[torch.Tensor]:
    """Latent targets [T,B,S,D] of the EMA target encoder on the agent filter's own contexts (`filt.ctxs`, the same
    observation / action / time / belief / option / memory / goal tokens, detached) when the world model keeps an EMA
    target (cfg.wm_ema_target); None otherwise. One no-grad core pass per step."""
    tc = getattr(model.world_model, "target_core", None)
    ctxs = getattr(filt, "ctxs", None)
    if tc is None or not ctxs:
        return None
    from ..model import detach_tokens
    B = filt.Zs[0].shape[0]
    dev = filt.Zs[0].device
    Z0 = tc.init_state(B, device=dev).float()
    Z = Z0
    out = []
    with model._autocast():
        for ctx, reset in zip(ctxs, filt.resets):
            if bool(reset.any()):
                Z = torch.where(reset.view(B, 1, 1), Z0, Z)
            Z, _ = tc(Z, detach_tokens(ctx))
            Z = Z.float()
            out.append(Z)
    return torch.stack(out)


def world_loss(model: "CamUnifiedModel", seq, filt: Optional["FilterResult"]) -> dict:
    """WorldModelP.loss with the acting-consistent belief / memory context when the world model accepts it.

    Single-filter path (`single_filter`): the graded decisions (after `filt.burn_in`) are scored with the agent
    filter's states Z_t (``Z_post``, with gradients), the prior inputs R_t (``R_post``, `prior_on_R`) and the belief
    tokens decide() conditions the prior on; open-loop steps >= 2 re-run the reasoner on the imagined state
    (`reason_fn_of`). Returns the world model's loss dict plus "burn_in" and "graded_steps"."""
    kw = {}
    if filt is not None and single_filter(model):
        burn = int(getattr(filt, "burn_in", 0) or 0)
        steps = seq.steps[burn:]
        Zs = filt.Zs[burn:]
        bel = getattr(filt, "belief_tokens", None) or filt.beliefs
        kw["Z_post"] = torch.stack([z.float() for z in Zs])
        tgt = ema_targets(model, filt)
        if tgt is not None:
            kw["Z_target"] = tgt[burn:]
        Rs = getattr(filt, "Rs", None)
        if prior_on_R(model) and Rs and all(r is not None for r in Rs[burn:]):
            kw["R_post"] = torch.stack([r.float() for r in Rs[burn:]])
            kw["reason_fn"] = reason_fn_of(model)
        if "beliefs" in model._wm_loss_ctx:
            kw["beliefs"] = list(bel[burn:])
        sem = getattr(filt, "sem", None)
        if sem is not None and not isinstance(sem, (list, tuple)) and "sem" in _params(model.world_model.loss):
            kw["sem"] = sem                      # one context for the window (rows mapped per transition)
            if getattr(filt, "sem_rows", None) is not None:
                kw["sem_rows"] = filt.sem_rows
        out = model.world_model.loss(SequenceBatch(steps), **kw)
        out["burn_in"], out["graded_steps"] = burn, len(steps)
        return out
    if filt is not None:
        if "beliefs" in model._wm_loss_ctx:
            kw["beliefs"] = filt.beliefs
        if "memory" in model._wm_loss_ctx and model.cfg.use_memory:
            kw["memory"] = filt.memory
    return model.world_model.loss(seq, **kw)
