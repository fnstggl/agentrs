"""Stochastic action-conditioned world model (spec 6.1-6.4; implements ``WorldModelP``).

Notation (per environment row b; S state tokens of width D):

* filter (posterior state, the agent's own state update; acting = training):
  ``Z_t = F_theta(Z_{t-1}, [O_t; A_{t-1}; TIME(dt_{t-1}); B_t; M_t; G_t])``  -- ``filter_step``.
* prior / imagination transition (no observations):
  ``z_t ~ Cat(prior_logits(Z_t, A_t, B_t, dt_t))`` (``stoch_groups`` x ``stoch_classes``, straight-through, unimix),
  ``Z^_{t+1} = F_theta(Z_t, [A_t; B_t; TIME(dt_t); NOISE(z_t)])``  -- ``prior``. Same backbone weights as the filter.
* posterior latent: ``q(z_t | Z_t, A_t, B_t, dt_t, Z_{t+1})`` where ``Z_{t+1}`` is the filter state after O_{t+1}.
  KL balancing ``kl_balance * max(free, KL(sg q || p)) + (1 - kl_balance) * max(free, KL(q || sg p))``.
* latent target: ``Z_target_{t+1} = stopgrad(EMA-core filter over O_{<=t+1})``; it is only ever a regression target,
  never an input to the online filter, the prior, the policy or anything an actor sees. The latent error is a squared
  error standardised by the target's across-sample spread (a 1 - R^2-like measure), so predicting the shared mean
  state scores ~1 and only predicting the sample-specific content scores well.
* heads on ``Z^_{t+1}``: typed outcomes Y_t of action A_t (``OutcomeHead``), next-observation tokens O_{t+1}
  (InfoNCE), verified reward r_t (two-hot symlog, also reads the goal), continuation 1 - done_t; duration dt_t
  from (Z_t, A_t)
  (log-normal on seconds) so imagination can predict how long an action takes.
* anti-collapse: VICReg-style variance hinge and covariance penalty on the (layer-normalised) filter states.
* every head reads latent states through ``StateNorm`` (running per-position standardisation), so the
  sample-specific part of Z is visible at unit scale even when a shared component dominates the residual stream.
* open-loop multistep: from filtered Z_t roll h steps with the recorded actions and durations and sampled prior
  latents, never seeing O_{>t}; each horizon in ``cfg.train_horizons`` is scored against Z_target_{t+h}, Y_{t+h-1},
  r_{t+h-1} and O_{t+h} under ``open_loop_samples`` sampled futures per start. The objective is the mixture
  likelihood of one coherent future, log mean_s p(all scored targets of the trajectory | future_s), optimised
  through its exact gradient form: every term is weighted by the joint posterior responsibility of each sampled
  future (``open_loop_credit="trajectory"``; "row" / "per_term" are looser variants for ablations). A correctly
  stochastic model is not penalised, and no term can be explained by a different future than the others.
  Per-horizon errors and the joint mixture NLL are returned.

DOM switches (plan 2026-11-26 §2 / §3 W6; config fields, defaults keep everything above):

* ``cfg.wm_transition_blocks = k``: Z^_{t+1} = Prior(R_t, A_t, B_t, dt_t, z_t) runs only the last k shared core
  blocks (by reference; `core.transition_blocks(k)` when the core provides it) between zero-initialised input /
  output adapters, from the reasoned latent R_t (declared deviation from the full-core transition of spec 6.1);
* ``cfg.wm_ema_target = False``: no EMA target encoder; the latent target is sg(Z_post) of the agent's own filter,
  which ``loss(..., Z_post=, R_post=)`` receives from `world/train_seq.py` (one filter, the acting one);
* ``cfg.kl_free_bits_group = f``: per-group free bits on the batch-mean KL (`kl_terms`) instead of free nats;
* ``cfg.wm_consistency = w``: L_cons = ||(Z^_{t+1} - sg(Z_post_{t+1})) / sd||^2 on the teacher-forced transition
  and on open-loop h=1, plus the diagnostics MSE / Var(Z_post);
* mode-latent open loop (on with the free bits): one extra future per start uses the prior mode at every step and
  is scored outside the mixture with weight 1/S, so deterministic delayed effects must sit on the deterministic path;
* ``cfg.reward_mode = "observation_only"``: the verified return increment is never a world-model target.

Batch alignment (``SequenceBatch``): step t holds O_t, the executed action A_t, its observed consequences Y_t, the
elapsed time dt_t until step t+1, reward r_t and done_t (the episode ended after A_t). A row whose obs and action
masks are all False at step t is padding.
"""
from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CamConfig
from ..interfaces import (Likelihood, OutcomeBatch, SequenceBatch, TokenKind, TokenSet)
from ._util import (MLP, SECONDS_PER_DAY, AttnPool, StateNorm, TimeEncoder, TwoHot, cat_tokens, fast_cat, fast_rows,
                    gather_tokens, logmeanexp, masked_mean, outcome_pad, safe_key_padding, stack_tokens)
from .outcomes import OutcomeHead

CONTINUOUS = (int(Likelihood.GAUSSIAN), int(Likelihood.STUDENT_T), int(Likelihood.COUNT))


def _params_of(fn) -> set:
    import inspect
    try:
        return set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return set()


def _core_kwargs(wm, core, sem, sem_rows, code_rows) -> dict:
    """Keyword arguments of one core call: code_rows / sem / sem_rows when given and accepted by the core's forward
    (the seam core takes code_rows only; W3's core also sem and sem_rows). A dropped ``sem`` is counted."""
    kw = {}
    if code_rows is None and sem is None:
        return kw
    ps = _params_of(core.forward)
    if code_rows is not None:
        kw["code_rows"] = code_rows
    if sem is not None:
        if "sem" in ps:
            kw["sem"] = sem
            if sem_rows is not None and "sem_rows" in ps:
                kw["sem_rows"] = sem_rows
        else:
            wm.sem_ignored += 1
    return kw


def _sem_at(sem, t: int):
    """The semantic context of step t: ``sem`` itself (one for every step) or its t-th entry (a list per step)."""
    if sem is None:
        return None
    if isinstance(sem, (list, tuple)):
        return sem[t] if t < len(sem) else None
    return sem


def _as_dt(dt, B: int, device) -> torch.Tensor:
    if dt is None:
        return None
    t = torch.as_tensor(dt, dtype=torch.float32, device=device)
    if t.dim() == 0:
        t = t.expand(B)
    return t.reshape(B)


def unimix(logits: torch.Tensor, mix: float) -> torch.Tensor:
    """Mix ``mix`` of the uniform distribution into softmax(logits); returns log-probabilities."""
    if mix <= 0:
        return F.log_softmax(logits, -1)
    p = F.softmax(logits.float(), -1)
    return torch.log((1 - mix) * p + mix / p.shape[-1])


def categorical_kl(p_logp: torch.Tensor, q_logp: torch.Tensor) -> torch.Tensor:
    """KL(p || q) for log-probabilities [..., G, C] summed over groups -> [...]."""
    return (p_logp.exp() * (p_logp - q_logp)).sum(-1).sum(-1)


def categorical_kl_groups(p_logp: torch.Tensor, q_logp: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per stochastic group for log-probabilities [..., G, C] -> [..., G]."""
    return (p_logp.exp() * (p_logp - q_logp)).sum(-1)


def _mse_over_var(Zh: torch.Tensor, Zt: torch.Tensor, ok: torch.Tensor, Z_all: torch.Tensor,
                  valid: torch.Tensor) -> torch.Tensor:
    """MSE(Zh, Zt) over the rows ``ok`` divided by the mean across-sample variance of the valid states ``Z_all``
    [T,B,S,D] (the plan's consistency gap measure; 1 ~ predicting the mean state, 0 = exact). Detached scalar."""
    with torch.no_grad():
        if not bool(ok.any()):
            return Zh.new_zeros(()).float()
        X = Z_all[valid].float()
        var = X.var(0).mean() if X.shape[0] >= 2 else torch.ones((), device=Zh.device)
        mse = ((Zh[ok].float() - Zt[ok].float()) ** 2).mean()
        return mse / var.clamp_min(1e-12)


def outcome_beta_weights(d, beta: float) -> torch.Tensor:
    """Detached beta-NLL weights of a MixedOutcomeDist (batch shape): scale^(2 beta) for the Gaussian (std) and
    Student-t (scale) entries, 1 for every other family (Seitzer et al. 2022)."""
    with torch.no_grad():
        w = torch.ones(d.batch_shape, device=d.raw.device)
        if beta <= 0:
            return w
        if d._has(Likelihood.GAUSSIAN):
            w = torch.where(d.lik == int(Likelihood.GAUSSIAN), d.g_std.float().clamp(min=1e-6) ** (2 * beta), w)
        if d._has(Likelihood.STUDENT_T):
            w = torch.where(d.lik == int(Likelihood.STUDENT_T), d.t_scale.float().clamp(min=1e-6) ** (2 * beta), w)
        return w


def straight_through(logp: torch.Tensor, sample: bool = True, generator: Optional[torch.Generator] = None):
    """One-hot sample (or mode) over the last axis with straight-through gradients. logp [..., C] -> [..., C]."""
    p = logp.exp()
    if sample:
        idx = torch.multinomial(p.reshape(-1, p.shape[-1]).detach(), 1, generator=generator).reshape(p.shape[:-1])
    else:
        idx = p.argmax(-1)
    hard = F.one_hot(idx, p.shape[-1]).to(p.dtype)
    return hard + p - p.detach()


class LatentHead(nn.Module):
    """Categorical latent logits from several token sources: G learned group queries cross-attend over the
    concatenated sources (each tagged with a learned source embedding) -> [B, G, C] logits."""

    def __init__(self, d_model: int, n_heads: int, groups: int, classes: int, n_sources: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(groups, d_model) * 0.02)
        self.src = nn.Embedding(n_sources, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.att = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = MLP(d_model, 2 * d_model, d_model)
        self.out = nn.Linear(d_model, classes)
        with torch.no_grad():
            self.out.weight.mul_(0.1)
            self.out.bias.zero_()

    def forward(self, sources: Sequence[tuple]) -> torch.Tensor:
        xs, ms = [], []
        for i, (x, m) in enumerate(sources):
            if x is None or x.shape[1] == 0:
                continue
            xs.append(x.float() + self.src.weight[i])
            ms.append(torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if m is None else m.bool())
        x, m = torch.cat(xs, 1), torch.cat(ms, 1)
        kv = self.norm(x)
        h, _ = self.att(self.q.unsqueeze(0).expand(x.shape[0], -1, -1), kv, kv, key_padding_mask=safe_key_padding(m),
                        need_weights=False)
        h = h + self.ff(h)
        return self.out(h)


class ObsPredictor(nn.Module):
    """Predicts the content embedding of a future observation token from Z^ and the token's content-free
    metadata (kind, relative time): q_j = Attn(meta_j -> Z^); scores <norm(W q_j), norm(V x_i)> / tau."""

    def __init__(self, d_model: int, n_heads: int, n_fourier: int, d_embed: int = 128):
        super().__init__()
        self.kind = nn.Embedding(int(TokenKind.N_KINDS), d_model)
        self.time = TimeEncoder(d_model, n_fourier)
        self.norm = nn.LayerNorm(d_model)
        self.att = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = MLP(d_model, 2 * d_model, d_model)
        self.pred = nn.Linear(d_model, d_embed)
        self.key = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_embed))

    def predict(self, Zh: torch.Tensor, kind: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Zh [R,S,D]; kind/time [R,M] -> normalised predictions [R,M,E]."""
        q = self.kind(kind.clamp(0, int(TokenKind.N_KINDS) - 1)) + self.time(time)
        kv = self.norm(Zh.float())
        h, _ = self.att(q, kv, kv, need_weights=False)
        h = h + self.ff(h)
        return F.normalize(self.pred(h), dim=-1)

    def keys(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.key(x.float()), dim=-1)


class WorldModel(nn.Module):
    """p_theta(Z', O', Y', r, dt, done | Z, A, B, dt) on the shared backbone ``core`` (any ``UnifiedCoreP``)."""

    DEFAULT_WEIGHTS = {"latent": 1.0, "kl": 1.0, "outcome": 1.0, "obs": 1.0, "reward": 1.0, "duration": 0.25,
                       "cont": 0.25, "var": 1.0, "cov": 0.04, "open_loop": 1.0}

    def __init__(self, cfg: CamConfig, core: nn.Module, outcome_head: Optional[OutcomeHead] = None, *,
                 unimix: float = 0.01, n_noise_tokens: int = 1, reward_bins: int = 255,
                 loss_weights: Optional[dict] = None, ema_decay: Optional[float] = None,
                 open_loop_samples: int = 4, open_loop_starts: Optional[int] = 8, latent_temperature: float = 0.25,
                 obs_temperature: float = 0.1, obs_max_groups: int = 64, obs_max_tokens: int = 32,
                 var_target: float = 0.5, detach_obs_targets: bool = True, use_autocast: bool = True,
                 train_horizons: Optional[Sequence[int]] = None, open_loop_credit: str = "trajectory",
                 latent_floor_rms: float = 0.1, mode_latent: Optional[bool] = None,
                 latent_grad: Optional[str] = None, latent_rep_weight: Optional[float] = None,
                 outcome_nll_beta: Optional[float] = None, action_input: Optional[bool] = None,
                 trans_act_gain: Optional[float] = None):
        super().__init__()
        if open_loop_credit not in ("trajectory", "row", "per_term"):
            raise ValueError("open_loop_credit must be 'trajectory', 'row' or 'per_term'")
        self.open_loop_credit = open_loop_credit
        self._warned_horizons: set = set()
        D, G, C = cfg.d_model, cfg.stoch_groups, cfg.stoch_classes
        self.cfg = cfg
        self.core = core
        self.outcome_head = outcome_head if outcome_head is not None else OutcomeHead(cfg)
        self.G, self.C = G, C
        self.unimix = unimix
        self.weights = dict(self.DEFAULT_WEIGHTS, **(loss_weights or {}))
        self.ema_decay = cfg.ema_target if ema_decay is None else ema_decay
        self.open_loop_samples = open_loop_samples
        self.open_loop_starts = open_loop_starts
        self.latent_temperature = latent_temperature
        self.latent_floor_rms = latent_floor_rms
        self.obs_temperature = obs_temperature
        self.obs_max_groups, self.obs_max_tokens = obs_max_groups, obs_max_tokens
        self.var_target = var_target
        self.detach_obs_targets = detach_obs_targets
        self.use_autocast = use_autocast
        self.train_horizons = tuple(sorted(train_horizons if train_horizons is not None else cfg.train_horizons))
        self.n_noise = n_noise_tokens

        self.state_norm = StateNorm(cfg.n_state, D)
        self.outcome_head.state_norm = self.state_norm
        self.time_enc = TimeEncoder(D, cfg.n_fourier)
        self.noise_proj = nn.Linear(G * C, n_noise_tokens * D)
        self.prior_net = LatentHead(D, cfg.n_heads, G, C, n_sources=4)      # Z, A, B, TIME
        self.post_net = LatentHead(D, cfg.n_heads, G, C, n_sources=5)       # + Z_{t+1} filter state
        self.dur_pool = AttnPool(D, cfg.n_heads)
        self.dur_src = nn.Embedding(2, D)
        self.dur_head = MLP(D, 2 * D, 2)
        self.rew_pool = AttnPool(D, cfg.n_heads)
        self.rew_src = nn.Embedding(2, D)          # sources: Z^, GOAL tokens
        self.twohot = TwoHot(reward_bins)
        self.rew_head = MLP(D, 2 * D, reward_bins, zero_out=True)
        self.cont_pool = AttnPool(D, cfg.n_heads)
        self.cont_head = MLP(D, 2 * D, 1)
        self.latent_pred = MLP(D, 2 * D, D)
        self.obs_pred = ObsPredictor(D, cfg.n_heads, cfg.n_fourier)
        # ---- DOM switches (plan 2026-11-26 §2/§3 W6); every default keeps the pre-seam model
        # transition through the last k shared core blocks + zero-initialised adapters (0 = a full core pass)
        self.k_trans = int(getattr(cfg, "wm_transition_blocks", 0) or 0)
        if self.k_trans > 0:
            self.trans_in = nn.Linear(D, D)
            nn.init.zeros_(self.trans_in.weight)
            nn.init.zeros_(self.trans_in.bias)
            self.trans_out = MLP(D, 2 * D, D, zero_out=True)
            # direct action input of the block transition (the RSSM "GRU input" role): the pooled ACTION tokens,
            # standardised with running per-channel statistics, are linearly projected and added to every state token
            # before the reused blocks. Without it the action reaches the state stream only as 4 of ~55 attended
            # context tokens: on the real dom_dev stack the executed action moved Z^ by ~1-2 % of its across-row
            # spread and the action-dependent InteractionEnv payout stayed at climatology for > 400 updates.
            # Frozen-encoder action tokens share a dominant component (pairwise cosine 0.95-0.96 between the four
            # lever settings), so the running standardisation (StateNorm over one position; updated after use, in
            # training only) makes their differences unit-scale; a per-row LayerNorm keeps them ~5 % apart. Measured
            # on fixed dom_dev filter states (all outcome keys, beta-NLL, Adam 1e-4, 8-row batches): payout MSE/Var
            # at 200 updates 0.82 (LayerNorm input), 0.15 (running, zero init), 0.03 (running, unit gain); on the full
            # dom_dev world objective (InteractionEnv, Adam 1e-4) t1 after 200 updates 0.06 with a zero-initialised
            # and 0.39 with a unit-gain projection (docs/evidence/dom/W6/world_model_outcome_learning.json).
            # ``trans_act_gain`` = init std x sqrt(D) (default 1: unit gain; 0 = zero init, which keeps the exact
            # last-k-blocks initialisation). cfg.wm_action_input (default True with the block transition),
            # cfg.wm_trans_act_gain.
            on = bool(getattr(cfg, "wm_action_input", True) if action_input is None else action_input)
            g = float(getattr(cfg, "wm_trans_act_gain", 1.0) if trans_act_gain is None else trans_act_gain)
            self.action_input_on, self.trans_act_gain = on, g
            self.act_norm_mode = str(getattr(cfg, "wm_trans_act_norm", "running"))
            if on:
                self.trans_act_norm = nn.LayerNorm(D)
                self.trans_act_stats = StateNorm(1, D)
                self.trans_act = nn.Linear(D, D)
                if g > 0:
                    nn.init.normal_(self.trans_act.weight, std=g * D ** -0.5)
                else:
                    nn.init.zeros_(self.trans_act.weight)
                nn.init.zeros_(self.trans_act.bias)
        else:
            self.action_input_on, self.trans_act_gain = False, 0.0
        # per-group free bits (Kingma et al. 2016: batch-mean KL of each stochastic group floored at the value)
        self.free_bits_group = float(getattr(cfg, "kl_free_bits_group", 0.0) or 0.0)
        # L_cons = ||(Z_hat_{t+1} - sg(Z_post_{t+1})) / sd||^2 (teacher-forced and open-loop h=1)
        self.consistency_w = float(getattr(cfg, "wm_consistency", 0.0) or 0.0)
        # observation-only: the verified return increment is never a world-model target
        self.reward_targets = str(getattr(cfg, "reward_mode", "legacy")) != "observation_only"
        # mode-latent open-loop future (one extra rollout with the prior mode, scored outside the mixture with weight
        # 1/S): deterministic effects must sit on the deterministic path (audit E3); on with the DOM free bits
        self.mode_latent = self.free_bits_group > 0 if mode_latent is None else bool(mode_latent)
        # EMA target encoder (legacy): a frozen copy of the backbone (state_dict-saved, never optimised).
        # cfg.wm_ema_target=False: no copy; the latent target is the detached posterior state of the agent's filter.
        self.use_ema = bool(getattr(cfg, "wm_ema_target", True))
        # gradient routing of the self-predictive terms (latent, consistency) whose target is the model's own state.
        # "full": through the transition into the shared blocks and the filter (legacy, with the EMA target);
        # "transition": only into the transition-specific parameters (output adapter, latent read-out). Without an
        # EMA target the online posterior is the target, and letting these terms shape the shared encoder has a
        # trivial fixed point (a context-insensitive filter predicts itself; measured on DelayWorld with the real
        # core: predictions collapse to climatology, consistency 0.001). The encoder is then shaped by the
        # predictive heads (outcomes, observations, continuation) only, as in the stop-gradient / predictor
        # asymmetry of non-EMA self-prediction.
        # ``latent_rep_weight`` beta (with "transition"; cfg.wm_latent_rep_weight, default 0): a beta-scaled copy of
        # the full gradient is added (representation term, DreamerV3's rep/dyn split with beta_rep = 0.1): the
        # shared encoder is shaped towards predictable states with a small weight while the transition-specific
        # parameters receive the full weight. Values are unchanged (gradient routing only).
        self.latent_grad = ("full" if self.use_ema else "transition") if latent_grad is None else latent_grad
        if self.latent_grad not in ("full", "transition"):
            raise ValueError("latent_grad must be 'full' or 'transition'")
        self.latent_rep_weight = float(getattr(cfg, "wm_latent_rep_weight", 0.0) or 0.0) \
            if latent_rep_weight is None else float(latent_rep_weight)
        # beta-NLL of the continuous outcome terms (Seitzer et al. 2022, "On the pitfalls of heteroscedastic
        # uncertainty estimation"): each Gaussian / Student-t log-likelihood of the outcome loss is weighted by
        # sg(scale^(2 beta)), so the predictive mean receives a gradient (mu - y) / sigma^(2 - 2 beta) instead of
        # (mu - y) / sigma^2. With plain NLL (beta 0) a learned scale absorbs the residual of a state-/action-
        # dependent outcome and the mean stops learning: measured on the real dom_dev stack (InteractionEnv payout
        # through the DOM transition + outcome head, all outcome keys, Adam 1e-4, 8-row batches): plain NLL stays at
        # the climatology (MSE/Var 1.02 after 800 steps) while beta = 0.5 reaches MSE/Var 0.014 (docs/evidence/dom/
        # W6). Only the training loss is reweighted: log-likelihoods, mixture responsibilities, belief updates and
        # every reported NLL are unchanged. cfg.wm_outcome_beta; default 0.5 with the DOM switches, 0 (plain NLL)
        # otherwise.
        ob_default = 0.5 if (self.k_trans > 0 or self.free_bits_group > 0) else 0.0
        self.outcome_nll_beta = float(getattr(cfg, "wm_outcome_beta", ob_default)
                                      if outcome_nll_beta is None else outcome_nll_beta)
        if not 0.0 <= self.outcome_nll_beta <= 1.0:
            raise ValueError("outcome_nll_beta must lie in [0, 1]")
        # latent read-out residual at the state's RMS (DOM switches only; legacy keeps the unit-scale residual)
        self.latent_rms_scale = self.k_trans > 0 or not self.use_ema
        if self.use_ema:
            self.target_core = copy.deepcopy(core)
            for p in self.target_core.parameters():
                p.requires_grad_(False)
        else:
            self.target_core = None
        self.n_transitions = 0              # world-model transition evaluations (rows), for compute accounting
        self.sem_ignored = 0                # semantic contexts dropped because the core has no semantic input

    # ------------------------------------------------------------------ parameters / target
    def trainable_parameters(self):
        """All optimisable parameters (excludes the EMA target copy)."""
        tid = set() if self.target_core is None else {id(p) for p in self.target_core.parameters()}
        return [p for p in self.parameters() if id(p) not in tid and p.requires_grad]

    @torch.no_grad()
    def update_target(self, decay: Optional[float] = None) -> None:
        """target <- decay * target + (1 - decay) * online (call after each optimiser step). No-op without an EMA
        target (cfg.wm_ema_target=False)."""
        if self.target_core is None:
            return
        d = self.ema_decay if decay is None else decay
        for pt, p in zip(self.target_core.parameters(), self.core.parameters()):
            pt.lerp_(p.detach().to(pt.dtype), 1.0 - d)
        for bt, b in zip(self.target_core.buffers(), self.core.buffers()):
            bt.copy_(b)

    @torch.no_grad()
    def reset_target(self) -> None:
        """Copy the online backbone into the target (e.g. after loading new backbone weights). No-op without one."""
        if self.target_core is None:
            return
        self.target_core.load_state_dict(self.core.state_dict())

    def transition_block_list(self):
        """The shared core blocks the DOM transition runs (`core.transition_blocks(k)` when the core provides it,
        W3's signature; otherwise the last k entries of `core.blocks`). Empty when cfg.wm_transition_blocks == 0."""
        if self.k_trans <= 0:
            return []
        core = self.core
        if callable(getattr(core, "transition_blocks", None)):
            return list(core.transition_blocks(self.k_trans))
        blocks = core.blocks
        return list(blocks[max(0, len(blocks) - self.k_trans):])

    # ------------------------------------------------------------------ token builders
    def init_state(self, B: int, device=None):
        return self.core.init_state(B, device=device)

    def time_tokens(self, dt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> TokenSet:
        """dt [B] days -> one TIME token [B,1,D] (token time = +dt, the predicted state's time)."""
        x = self.time_enc(dt).unsqueeze(1)
        return TokenSet.from_dense(x, TokenKind.TIME, time=dt.unsqueeze(1).float(),
                                   mask=None if mask is None else mask.view(-1, 1))

    def noise_tokens(self, z: torch.Tensor, dt: torch.Tensor) -> TokenSet:
        """z [B, G*C] -> n_noise NOISE tokens [B, n_noise, D]."""
        x = self.noise_proj(z.float()).view(z.shape[0], self.n_noise, -1)
        return TokenSet.from_dense(x, TokenKind.NOISE, time=dt.view(-1, 1).expand(-1, self.n_noise).float())

    def _core(self, Z, ctx: TokenSet, core=None, sem=None, sem_rows=None, code_rows=None):
        """One core pass. ``sem`` (a SemanticContext / SemanticKV, Mechanism C), ``sem_rows`` [B] (batch row ->
        semantic row) and ``code_rows`` [B] (batch row -> entity slot-code episode row) are passed to the core when
        given and the core accepts them (W3's `UnifiedCore.forward(Z_prev, ctx, code_rows=None, sem=None, *,
        sem_rows=None)`); a core without a semantic input ignores ``sem`` (counted in ``sem_ignored``)."""
        core = self.core if core is None else core
        kw = _core_kwargs(self, core, sem, sem_rows, code_rows)
        on = self.use_autocast and Z.is_cuda
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=on):
            Zn, ctx_out = core(Z, ctx, **kw)
        return Zn.float(), ctx_out

    # ------------------------------------------------------------------ filter (posterior state)
    def filter_step(self, Z_prev, obs: Optional[TokenSet], act_prev: Optional[TokenSet] = None, dt_prev=None,
                    belief: Optional[TokenSet] = None, goal: Optional[TokenSet] = None,
                    memory: Optional[TokenSet] = None, core=None, reset: Optional[torch.Tensor] = None,
                    sem=None, sem_rows: Optional[torch.Tensor] = None, code_rows: Optional[torch.Tensor] = None):
        """One agent state update Z_t = F(Z_{t-1}, [O_t; A_{t-1}; TIME(dt_{t-1}); B_t; M_t; G_t]).
        ``reset`` [B] bool: rows starting a new episode (Z_prev replaced by the initial state; A_{t-1} and the
        elapsed time masked out). ``sem`` / ``sem_rows`` / ``code_rows``: see `_core`. Returns (Z [B,S,D], ctx_out)."""
        B = Z_prev.shape[0]
        dev = Z_prev.device
        if reset is not None and bool(reset.any()):
            Z0 = (self.core if core is None else core).init_state(B, device=dev).to(Z_prev.dtype)
            Z_prev = torch.where(reset.view(B, 1, 1), Z0, Z_prev)
        parts = [obs]
        if act_prev is not None:
            if reset is not None:
                act_prev = TokenSet(act_prev.x, act_prev.mask & ~reset.view(B, 1), act_prev.kind, act_prev.time,
                                    act_prev.entity, act_prev.refs, act_prev.entity2)
            parts.append(act_prev)
        if dt_prev is not None:
            parts.append(self.time_tokens(_as_dt(dt_prev, B, dev), None if reset is None else ~reset))
        parts += [belief, memory, goal]
        return self._core(Z_prev, cat_tokens(*parts), core, sem, sem_rows, code_rows)

    def filter(self, batch: SequenceBatch, beliefs: Optional[list] = None, memory: Optional[list] = None,
               core=None, Z0=None, sem=None, sem_rows: Optional[torch.Tensor] = None,
               code_rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the filter over a sequence. Returns Z_post [T,B,S,D] (Z_post[t] has seen O_{<=t} only). ``sem``: one
        semantic context for every step, or a list of T (one per step; None entries allowed)."""
        steps = batch.steps
        B = steps[0].obs.B
        dev = steps[0].obs.x.device
        Z = self.init_state(B, device=dev).float() if Z0 is None else Z0
        out = []
        for t, st in enumerate(steps):
            prev = steps[t - 1] if t > 0 else None
            reset = None if prev is None else prev.done.bool().to(dev)
            Z, _ = self.filter_step(Z, st.obs, None if prev is None else prev.act,
                                    None if prev is None else prev.dt,
                                    None if beliefs is None else beliefs[t], st.goal,
                                    None if memory is None else memory[t], core=core, reset=reset,
                                    sem=_sem_at(sem, t), sem_rows=sem_rows, code_rows=code_rows)
            out.append(Z)
        return torch.stack(out)

    # ------------------------------------------------------------------ prior / posterior / transition
    def prior_logits(self, Z, act: TokenSet, belief: Optional[TokenSet], dt) -> torch.Tensor:
        """Unimixed log-probabilities of the prior stochastic latent [B,G,C]."""
        B = Z.shape[0]
        tt = self.time_enc(_as_dt(dt, B, Z.device)).unsqueeze(1)
        src = [(self.state_norm(Z), None), (act.x, act.mask), (None if belief is None else belief.x,
                                                               None if belief is None else belief.mask), (tt, None)]
        return unimix(self.prior_net(src), self.unimix)

    def posterior_logits(self, Z, act: TokenSet, belief: Optional[TokenSet], dt, Z_next) -> torch.Tensor:
        B = Z.shape[0]
        tt = self.time_enc(_as_dt(dt, B, Z.device)).unsqueeze(1)
        src = [(self.state_norm(Z), None), (act.x, act.mask), (None if belief is None else belief.x,
                                                               None if belief is None else belief.mask), (tt, None),
               (self.state_norm(Z_next), None)]
        return unimix(self.post_net(src), self.unimix)

    def sample_latent(self, logp: torch.Tensor, sample: bool = True, generator=None) -> torch.Tensor:
        """[B,G,C] log-probs -> straight-through one-hot z [B, G*C]."""
        return straight_through(logp, sample, generator).reshape(logp.shape[0], -1)

    def transition(self, Z, act: TokenSet, belief: Optional[TokenSet], dt, z, view: bool = False, sem=None,
                   sem_rows: Optional[torch.Tensor] = None, code_rows: Optional[torch.Tensor] = None):
        """Z^_{t+1} = F(Z_t, [A_t; B_t; TIME(dt_t); NOISE(z_t)]) (no observations). With cfg.wm_transition_blocks = k
        the transition runs only the last k shared core blocks (`_block_transition`); otherwise a full core pass.
        ``view=True`` returns (Z^, Z^_target_view): the same values, with gradients routed for the self-predictive
        terms (`latent_grad`). ``sem`` / ``sem_rows`` / ``code_rows`` (rows of Z -> semantic rows / entity slot-code
        episode rows, e.g. B*K candidate rows -> their environment row): see `_core`."""
        B = Z.shape[0]
        dt = _as_dt(dt, B, Z.device)
        self.n_transitions += B
        ctx = cat_tokens(act, belief, self.time_tokens(dt), self.noise_tokens(z, dt))
        if self.k_trans > 0:
            Zn, Zv = self._block_transition(Z, ctx, sem, sem_rows, code_rows, act=act)
        else:
            Zn, _ = self._core(Z, ctx, None, sem, sem_rows, code_rows)
            Zv = Zn if self.latent_grad == "full" else self._rep_mix(Zn.detach(), Zn)
        return (Zn, Zv) if view else Zn

    def _rep_mix(self, Zr: torch.Tensor, Zn: torch.Tensor) -> torch.Tensor:
        """Routed view Zr (transition-specific gradient) plus ``latent_rep_weight`` x the full-gradient path of Zn;
        the values stay Zr's (== Zn's)."""
        b = self.latent_rep_weight
        if b <= 0 or not Zn.requires_grad:
            return Zr
        return Zr + b * (Zn - Zn.detach())

    def action_input(self, act: Optional[TokenSet], B: int, device) -> Optional[torch.Tensor]:
        """[B,1,D] direct action input of the block transition (zero for rows without action tokens), or None when
        the transition has none (`action_input` off)."""
        if not self.action_input_on or act is None or act.N == 0:
            return None
        m = act.mask.float().unsqueeze(-1)
        pooled = (act.x.float() * m).sum(1) / m.sum(1).clamp(min=1.0)
        has = (m.sum(1) > 0).float()
        if self.act_norm_mode == "running":
            x = self.trans_act_stats(pooled.unsqueeze(1)).squeeze(1)
            if self.training and torch.is_grad_enabled():
                ok = has.squeeze(-1) > 0
                if int(ok.sum()) >= 2:
                    self.trans_act_stats.update(pooled.detach()[ok].unsqueeze(1))
        else:
            x = self.trans_act_norm(pooled)
        return (self.trans_act(x) * has).unsqueeze(1)

    def _block_transition(self, R, ctx: TokenSet, sem=None, sem_rows: Optional[torch.Tensor] = None,
                          code_rows: Optional[torch.Tensor] = None, act: Optional[TokenSet] = None) -> torch.Tensor:
        """DOM transition (plan §2, declared deviation from spec 6.1's full-core transition): the input state
        (the reasoned latent R_t when acting; an imagined Z^ in open loop) is embedded like a core state
        (`core.embed_state` + a zero-initialised input adapter), the context [A; B; TIME; NOISE] like core context
        (`core.embed_ctx`, budget-selected), and the joint sequence runs through the last k core blocks shared by
        reference (`transition_block_list`), then a zero-initialised residual output adapter and the core's output
        norm, so Z^ lies on the same RMS-normalised manifold as the posterior state. At initialisation (adapters
        zero) this is exactly the last k core blocks applied to [R + action input; ctx] (`action_input`). With ``sem`` and a core that has Mechanism C
        (`project_semantic`, `run_blocks(..., sem=)`; W3) the semantic cross-attention after each reused block applies
        as in the full pass (row alignment by ``sem_rows`` / ``code_rows``); ``code_rows`` also selects the entity
        slot codes of the context rows (`embed_ctx`)."""
        core = self.core
        S = R.shape[1]
        blocks = self.transition_block_list()
        on = self.use_autocast and R.is_cuda
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=on):
            sel, _, _ = core.select_budget(ctx)
            hs = core.embed_state(R) + self.trans_in(R.float())
            a_in = self.action_input(act, R.shape[0], R.device)
            if a_in is not None:
                hs = hs + a_in
            hc = core.embed_ctx(sel) if code_rows is None else core.embed_ctx(sel, code_rows)
            h = torch.cat([hs, hc.to(hs.dtype)], 1)
            key_mask = torch.cat([torch.ones(R.shape[0], S, dtype=torch.bool, device=R.device), sel.mask.bool()], 1)
            if bool(key_mask.all()):
                key_mask = None
            rb_kw = {}
            if sem is not None:
                proj = getattr(core, "project_semantic", None)
                if callable(proj) and "sem" in _params_of(core.run_blocks):
                    kv = proj(sem, R.shape[0], code_rows, sem_rows)
                    if kv is not None:
                        rb_kw["sem"] = kv
                else:
                    self.sem_ignored += 1
            h = core.run_blocks(h, key_mask, blocks, **rb_kw)
            hS0 = h[:, :S].float()
            Zn = core.out_norm(hS0 + self.trans_out(hS0)).float()
            if self.latent_grad == "full":
                Zv = Zn
            else:
                # same values; gradient only into the output adapter (blocks, input and output norm detached)
                hd = hS0.detach()
                w = core.out_norm.weight.detach().float()
                Zv = F.rms_norm(hd + self.trans_out(hd), (hd.shape[-1],), w, getattr(core.out_norm, "eps", 1e-6))
                Zv = self._rep_mix(Zv.float(), Zn)
        return Zn, Zv.float()

    def duration_params(self, Z, act: TokenSet):
        """(mu, log_sigma) [B] of log1p(seconds) for the action's duration, from (Z_t, A_t) only."""
        x = torch.cat([self.state_norm(Z).float() + self.dur_src.weight[0], act.x.float() + self.dur_src.weight[1]], 1)
        m = torch.cat([torch.ones(Z.shape[:2], dtype=torch.bool, device=Z.device), act.mask.bool()], 1)
        mu, ls = self.dur_head(self.dur_pool(x, m)[:, 0]).unbind(-1)
        return mu + 7.0, ls.clamp(-5.0, 3.0)        # offset: ~1100 s at initialisation

    def predict_duration(self, Z, act: TokenSet) -> torch.Tensor:
        """Median predicted duration [B] in days."""
        mu, _ = self.duration_params(Z, act)
        return torch.expm1(mu.clamp(max=25.0)) / SECONDS_PER_DAY

    def prior(self, Z, act: TokenSet, belief: Optional[TokenSet], dt, sample: bool = True,
              generator: Optional[torch.Generator] = None, z: Optional[torch.Tensor] = None, sem=None,
              sem_rows: Optional[torch.Tensor] = None, code_rows: Optional[torch.Tensor] = None):
        """WorldModelP.prior: Z [B,S,D] (R_t on the DOM path), act ACTION tokens, belief BELIEF tokens (or None),
        dt [B] days (None -> the predicted duration). ``sem`` / ``sem_rows`` / ``code_rows``: the semantic context
        and the row maps of a batch whose rows are not the episode rows (e.g. B*K decision candidates -> their
        environment row b; `transition`). With cfg.wm_transition_blocks = k the transition runs the core's
        `transition_blocks(k)`. Returns (Z_hat [B,S,D], {"logits": [B,G,C], "z": [B,G*C], "dt": [B]})."""
        B = Z.shape[0]
        dt = self.predict_duration(Z, act).detach() if dt is None else _as_dt(dt, B, Z.device)
        logp = self.prior_logits(Z, act, belief, dt)
        if z is None:
            z = self.sample_latent(logp, sample, generator)
        Zh = self.transition(Z, act, belief, dt, z, sem=sem, sem_rows=sem_rows, code_rows=code_rows)
        return Zh, {"logits": logp, "z": z, "dt": dt}

    # ------------------------------------------------------------------ prediction heads on Z^
    def _pool_ctx(self, pool, src, Zh, *ctx: Optional[TokenSet]):
        """Attention-pool [StateNorm(Zh); ctx_1; ctx_2; ...] (each source tagged by a learned embedding) -> [B,D]."""
        xs = [self.state_norm(Zh).float() + src.weight[0]]
        ms = [torch.ones(Zh.shape[:2], dtype=torch.bool, device=Zh.device)]
        for i, c in enumerate(ctx, 1):
            if c is not None and c.N > 0:
                xs.append(c.x.float() + src.weight[i])
                ms.append(c.mask.bool())
        return pool(torch.cat(xs, 1), torch.cat(ms, 1))[:, 0]

    def reward_logits(self, Zh, goal: Optional[TokenSet] = None) -> torch.Tensor:
        """Two-hot reward logits [B, bins] from the predicted state and the goal (belief information reaches the
        reward through Z^, which the transition computed with the BELIEF tokens in context)."""
        return self.rew_head(self._pool_ctx(self.rew_pool, self.rew_src, Zh, goal))

    def reward(self, Zh, goal: Optional[TokenSet] = None) -> torch.Tensor:
        """Predicted verified return increment of the transition into Zh [B]."""
        return self.twohot.mean(self.reward_logits(Zh, goal))

    def continue_logit(self, Zh) -> torch.Tensor:
        return self.cont_head(self.cont_pool(self.state_norm(Zh).float())[:, 0]).squeeze(-1)

    def continue_prob(self, Zh) -> torch.Tensor:
        return torch.sigmoid(self.continue_logit(Zh))

    def outcome_dist(self, Zh, outcomes: OutcomeBatch, cond=None):
        """Typed predictive distribution of the outcomes of the action that led to Zh ([B,Q] or [B,K,Q])."""
        return self.outcome_head.dist(self.outcome_head.features(Zh, outcomes), outcomes, cond)

    def outcome_predictive(self, Z_prev, act: TokenSet, belief: Optional[TokenSet], dt, outcomes: OutcomeBatch,
                           cond=None, sample: bool = False, generator: Optional[torch.Generator] = None):
        """p(o_t | Z_{t-1}, a_{t-1}, B, dt [, m^k]): roll the prior one step from the previous state with the
        executed action (mode of the stochastic latent unless ``sample``) and return the typed outcome distribution,
        particle-conditioned when ``cond`` [B,K,Dc] is given ([B,K,Q]). This is the action-conditioned likelihood the
        belief module's importance weights need (``BeliefModuleP.update``)."""
        Zh, _ = self.prior(Z_prev, act, belief, dt, sample=sample, generator=generator)
        return self.outcome_dist(Zh, outcomes, cond)

    # ------------------------------------------------------------------ loss pieces
    def _latent_dist(self, Zh, Zt, scale) -> torch.Tensor:
        """Per-row standardised squared error between pred(Z^) and the target, mean over tokens and channels -> [R].

        ``scale`` [S,D] is the (detached) across-sample standard deviation of the target, so the objective measures
        how much of the sample-specific variation of Z_target is predicted (1 ~ predicting the mean state, 0 =
        exact); a shared constant component of the state cannot make it small."""
        # DOM path: the read-out is a residual at the state's own scale (per-token RMS, detached): the W3 default keeps
        # Z unit-RMS (scale ~1), a core without unit-RMS states (core.state_unit=False) no longer gets an O(1)
        # residual on O(0.01) states (W3 change request). The legacy path keeps the unit-scale residual.
        Zf = Zh.float()
        r = self.latent_pred(self.state_norm(Zh))
        if self.latent_rms_scale:
            r = Zf.pow(2).mean(-1, keepdim=True).sqrt().detach().clamp_min(1e-6) * r
        p = Zf + r
        return (((p - Zt.detach().float()) / scale) ** 2).mean((-1, -2))

    def _target_scale(self, Z_tgt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Across-sample std of valid target states [S,D] with a floor at 10% of the mean std and, in addition, at
        ``latent_floor_rms`` x the root-mean-square magnitude of the target states.

        The second floor matters for backbones whose output is RMS-normalised (B's UnifiedCore): at initialisation
        their states share almost everything (measured: across-sample std ~1% of the state RMS), so standardising by
        the across-sample spread alone made the latent term ~10^3 times larger than every other world-model term
        (latent 717-1215 vs outcome ~1.1 at init in the integrated stack), and under the trainer's global gradient
        clipping it suppressed every other objective. With the floor the error is measured in units of at least 10%
        of the state magnitude while sample-specific variation is small, and the 1 - R^2 semantics are unchanged once
        the spread exceeds the floor (as for cores whose states carry large sample variation)."""
        X = Z_tgt[valid].float()
        if X.shape[0] < 2:
            return torch.ones(Z_tgt.shape[-2:], device=Z_tgt.device)
        var = X.var(0)
        floor = 0.01 * var.mean()
        if self.latent_floor_rms > 0:
            floor = torch.maximum(floor, (self.latent_floor_rms ** 2) * X.pow(2).mean())
        return torch.sqrt(var + floor + 1e-8)

    def _obs_logp(self, Zh, nxt: TokenSet, n_samples: int, gen: Optional[torch.Generator]):
        """Multi-positive InfoNCE log-probabilities of the actual next-observation tokens.

        Zh [G*n_samples, S, D] (rows grouped by target: row = g*n_samples + s); nxt TokenSet [G, N, D] (one copy
        per group); returns (logp [G, n_samples, M] under each sampled future, valid [G, M]) for up to
        ``obs_max_tokens`` sampled real tokens per group. Candidates are the sampled tokens of all groups; positives
        are tokens of the same group with the same kind and relative time (they are indistinguishable from
        content-free metadata)."""
        Gn, N, _ = nxt.x.shape
        dev = nxt.x.device
        M = min(self.obs_max_tokens, N)
        # sample up to M real tokens per group (random priority among real tokens; reproducible from ``gen``)
        pri = _rand((Gn, N), gen, dev)
        pri = torch.where(nxt.mask, pri, torch.full_like(pri, -1.0))
        sel = pri.topk(M, dim=1).indices                                          # [G,M]
        valid = nxt.mask.gather(1, sel)
        kind = nxt.kind.gather(1, sel)
        time = nxt.time.float().gather(1, sel)
        x = nxt.x.gather(1, sel.unsqueeze(-1).expand(-1, -1, nxt.x.shape[-1]))
        if self.detach_obs_targets:
            x = x.detach()
        keys = self.obs_pred.keys(x).reshape(Gn * M, -1)                          # [G*M, E]
        pred = self.obs_pred.predict(self.state_norm(Zh), kind.repeat_interleave(n_samples, 0),
                                     time.repeat_interleave(n_samples, 0))         # [G*s, M, E]
        logits = pred.reshape(Gn * n_samples * M, -1) @ keys.T / self.obs_temperature    # [G*s*M, G*M]
        cand_valid = valid.reshape(-1)
        logits = logits.masked_fill(~cand_valid.unsqueeze(0), -1e4)
        grp = torch.arange(Gn, device=dev)
        q_grp = grp.repeat_interleave(n_samples * M)
        c_grp = grp.repeat_interleave(M)
        q_kind = kind.repeat_interleave(n_samples, 0).reshape(-1)
        q_time = time.repeat_interleave(n_samples, 0).reshape(-1)
        pos = (q_grp.unsqueeze(1) == c_grp.unsqueeze(0)) & (q_kind.unsqueeze(1) == kind.reshape(1, -1)) \
            & ((q_time.unsqueeze(1) - time.reshape(1, -1)).abs() < 1e-6) & cand_valid.unsqueeze(0)
        lp = torch.logsumexp(logits.masked_fill(~pos, -1e4), 1) - torch.logsumexp(logits, 1)
        return lp.view(Gn, n_samples, M), valid

    def _vicreg(self, Z: torch.Tensor):
        """Z [N,S,D] valid filter states -> (variance hinge, covariance penalty)."""
        if Z.shape[0] < 2:
            z0 = Z.sum() * 0
            return z0, z0
        X = F.layer_norm(Z.float(), Z.shape[-1:])
        std = torch.sqrt(X.var(0) + 1e-4)
        var = F.relu(self.var_target - std).mean()
        Xm = X.mean(1)
        Xm = Xm - Xm.mean(0, keepdim=True)
        cov = (Xm.T @ Xm) / (Xm.shape[0] - 1)
        off = cov - torch.diag(torch.diag(cov))
        return var, (off ** 2).sum() / Xm.shape[1]

    # ------------------------------------------------------------------ WorldModelP.loss
    def loss(self, batch: SequenceBatch, beliefs: Optional[list] = None, memory: Optional[list] = None,
             horizons: Optional[Sequence[int]] = None, generator: Optional[torch.Generator] = None,
             fast_states: Optional[list] = None, belief_update_fn: Optional[Callable] = None,
             belief_tokens_fn: Optional[Callable] = None, Z_post=None, R_post=None,
             reason_fn: Optional[Callable] = None, Z_target=None, sem=None,
             code_rows: Optional[torch.Tensor] = None, sem_rows: Optional[torch.Tensor] = None) -> dict:
        """Full world-model objective on real recorded sequences.

        ``sem``: the semantic context of the batch's B rows (one for the whole window; a list of T per-step contexts
        for the filter when this model runs its own); every transition over flattened rows (teacher-forced (t, b),
        open-loop (start, b, sample)) passes ``sem_rows`` = the batch row of each row. ``code_rows`` [B]: batch row
        -> entity slot-code episode row, mapped the same way; ``sem_rows`` [B]: batch row -> row of a source-aligned
        ``sem`` (composed with the flattened row maps).

        beliefs: optional list of T BELIEF TokenSets (B_t computed from information up to t); memory: optional list
        of T MEMORY TokenSets for the filter.
        Imagination-consistent open loop (optional): with ``fast_states`` (list of T real FastStates B_t),
        ``belief_update_fn(fast, Z_prev, act, OutcomeBatch) -> fast'`` and ``belief_tokens_fn(fast) -> TokenSet``,
        every open-loop row starts from its real B_t and updates a private copy with outcomes *sampled from the
        model's own predictions* (the real outcomes after the start are never used), exactly as ``Imagination`` does;
        otherwise the belief tokens stay at B_t for the whole rollout.
        Belief tokens of the real steps (given as ``beliefs`` or computed from ``fast_states`` with
        ``belief_tokens_fn``) keep their gradients, so the world loss also trains the belief-token encoder; only the
        imagined open-loop belief updates run without gradients.
        Open-loop horizons longer than the sequence cannot be scored: they are skipped with a warning (once per
        horizon set and length) and listed under "horizons_skipped"; "horizons_trained" lists the scored ones.
        Steps without a goal get an empty GOAL token set (the goal of the other steps is kept).
        ``generator`` makes the whole loss reproducible (latent and outcome samples, open-loop start subset,
        observation-token subsampling); the global RNG is used only without it.
        Returns a dict with the total ("world" == "loss"), each term, the balanced KL and its raw value,
        per-horizon open-loop errors ("horizon_errors": {h: {...}}), the joint open-loop mixture NLL per trajectory
        ("open_loop_joint_nll") and the detached filter states "Z_post" [T,B,S,D] for downstream users.

        Single filter (DOM, plan §3 W6): ``Z_post`` ([T,B,S,D] tensor or list of T [B,S,D], with gradients) are the
        agent's own acting-consistent filter states (``train_seq.filter_sequence``); the world model then runs no
        filter of its own, and without an EMA target (cfg.wm_ema_target=False) the latent target is sg(Z_post).
        ``R_post`` (same shape) are the prior's inputs at the real steps (the reasoned latents R_t; default Z_post); ``Z_target`` (same shape): the EMA target encoder's states on the same
        contexts (`train_seq.ema_targets`; default sg(Z_post));
        ``reason_fn(Z_hat, belief_tokens, goal_tokens) -> R`` maps imagined states to prior inputs at open-loop
        steps >= 2 (default: the imagined state itself). Extra DOM terms: per-group free bits
        (cfg.kl_free_bits_group), the consistency loss (cfg.wm_consistency; "consistency", and the diagnostics
        "cons_mse_var" = MSE(Z^_{t+1}, Z_post_{t+1}) / Var(Z_post) of the teacher-forced transition and
        "cons_mse_var_mode1" of the mode-latent one-step prior), and the mode-latent open-loop future
        ("open_loop_mode"). In observation_only mode the reward head has no target ("reward" = 0)."""
        if fast_states is not None and beliefs is None:
            if belief_tokens_fn is None:
                raise ValueError("fast_states need belief_tokens_fn")
            beliefs = [belief_tokens_fn(f) for f in fast_states]
        bel_dyn = None
        if fast_states is not None and belief_update_fn is not None and belief_tokens_fn is not None:
            bel_dyn = (fast_states, belief_update_fn, belief_tokens_fn)
        steps = batch.steps
        T, B = len(steps), steps[0].obs.B
        dev = steps[0].obs.x.device
        W = self.weights
        requested = tuple(sorted(set(horizons or self.train_horizons)))
        horizons = tuple(h for h in requested if h <= T)
        skipped = tuple(h for h in requested if h > T)
        if skipped and (skipped, T) not in self._warned_horizons:
            self._warned_horizons.add((skipped, T))
            warnings.warn(f"WorldModel.loss: open-loop horizons {skipped} exceed the sequence length T={T} and are "
                          f"not trained (sequences need T >= {max(skipped)})", RuntimeWarning, stacklevel=2)
        cfg = self.cfg

        obs = stack_tokens([s.obs for s in steps])                                  # [T,B,N,D]
        act = stack_tokens([s.act for s in steps])
        Qm = max(s.outcomes.q.N for s in steps)
        outs = [outcome_pad(s.outcomes, Qm) for s in steps]
        oq = stack_tokens([o.q for o in outs])
        o_lik = torch.stack([o.lik for o in outs]).long()
        o_nc = torch.stack([o.n_classes for o in outs]).long()
        o_y = torch.stack([o.y.float() for o in outs])
        o_c = torch.stack([o.censored.bool() for o in outs])
        o_m = torch.stack([o.y_mask.bool() for o in outs])
        dt = torch.stack([_as_dt(s.dt, B, dev) for s in steps])                      # [T,B]
        rew = torch.stack([torch.as_tensor(s.reward, device=dev).float().reshape(B) for s in steps])
        done = torch.stack([torch.as_tensor(s.done, device=dev).bool().reshape(B) for s in steps])
        valid = obs.mask.any(-1) | act.mask.any(-1)                                  # [T,B]
        o_m = o_m & valid.unsqueeze(-1)
        bel = stack_tokens(beliefs) if beliefs is not None else None
        goals = [s.goal for s in steps]
        goal = None
        if any(g is not None for g in goals):
            Dm = obs.x.shape[-1]
            goal = stack_tokens([g if g is not None else _empty_tokens(B, Dm, dev) for g in goals])

        def ob_rows(t_idx, b_idx):
            q = gather_tokens(oq, t_idx, b_idx)
            return OutcomeBatch(q, o_lik[t_idx, b_idx], o_nc[t_idx, b_idx], o_y[t_idx, b_idx], o_c[t_idx, b_idx],
                                o_m[t_idx, b_idx], torch.full_like(o_lik[t_idx, b_idx], -1))

        rows = _rows

        # ---- filter (online, with gradients) and latent target (no gradients): the EMA target encoder (legacy) or
        # the detached posterior itself; with given agent-filter states the world model runs no filter of its own
        if Z_post is None:
            Z_post = self.filter(batch, beliefs, memory, sem=sem, sem_rows=sem_rows, code_rows=code_rows)
            if self.target_core is not None:
                with torch.no_grad():
                    Z_tgt = self.filter(batch, beliefs, memory, core=self.target_core, sem=sem, sem_rows=sem_rows,
                                        code_rows=code_rows)
            else:
                Z_tgt = Z_post.detach()
        else:
            Z_post = (torch.stack(list(Z_post)) if isinstance(Z_post, (list, tuple)) else Z_post).float()
            # the EMA target encoder's states on the same contexts when given (train_seq.ema_targets), else sg(Z_post)
            Z_tgt = Z_post.detach() if Z_target is None else \
                (torch.stack(list(Z_target)) if isinstance(Z_target, (list, tuple)) else Z_target).float().detach()
        R_in = Z_post if R_post is None else \
            (torch.stack(list(R_post)) if isinstance(R_post, (list, tuple)) else R_post).float()
        S, D = Z_post.shape[2], Z_post.shape[3]
        sem_w = None if isinstance(sem, (list, tuple)) else sem       # per-step lists: filter only
        ctx = _SeqCtx(T=T, ob_rows=ob_rows, obs=obs, Z_tgt=Z_tgt, rew=rew, done=done, valid=valid, gen=generator,
                      tgt_scale=self._target_scale(Z_tgt, valid), sem=sem_w, code_rows=code_rows, sem_rows=sem_rows)

        tt = torch.arange(T, device=dev).repeat_interleave(B)
        bb = torch.arange(B, device=dev).repeat(T)
        Zs = R_in.reshape(T * B, S, D)                                               # prior inputs per real step
        act_r, bel_r, goal_r = rows(act, tt, bb), rows(bel, tt, bb), rows(goal, tt, bb)
        dt_r = dt.reshape(-1)
        valid_r = valid.reshape(-1)
        done_r = done.reshape(-1)
        out = {}

        # ---- duration of A_t from (Z_t, A_t)
        mu_d, ls_d = self.duration_params(Zs, act_r)
        ysec = torch.log1p(dt_r.clamp(min=0) * SECONDS_PER_DAY)
        nll_d = 0.5 * ((ysec - mu_d) / ls_d.exp()) ** 2 + ls_d + 0.5 * math.log(2 * math.pi)
        out["duration"] = masked_mean(nll_d, valid_r)

        # ---- teacher-forced one-step with posterior latents (t = 0 .. T-2)
        terms = {}
        if T >= 2:
            n1 = (T - 1) * B
            t1, b1 = tt[:n1], bb[:n1]
            ok1 = valid_r[:n1] & valid[1:].reshape(-1) & ~done_r[:n1]               # transition stays in episode
            prior_lp = self.prior_logits(Zs[:n1], rows(act, t1, b1), rows(bel, t1, b1), dt_r[:n1])
            post_lp = self.posterior_logits(Zs[:n1], rows(act, t1, b1), rows(bel, t1, b1), dt_r[:n1],
                                            Z_post[1:].reshape(n1, S, D))
            out.update(self.kl_terms(post_lp, prior_lp, ok1))
            # transitions without a next observation (episode end / sequence padding) have no posterior: use the prior
            lat_lp = torch.where(ok1.view(-1, 1, 1), post_lp, prior_lp)
            z_post = self.sample_latent(lat_lp, True, generator)
            Zh1, Zv1 = self.transition(Zs[:n1], rows(act, t1, b1), rows(bel, t1, b1), dt_r[:n1], z_post, view=True,
                                       **ctx.row_kw(b1))
            rt1 = self._row_terms(Zh1, 1, t1, b1, ctx, valid_r[:n1], rows(goal, t1, b1), self.obs_max_groups, Zv=Zv1)
            terms = self._reduce(rt1, self._responsibilities(rt1))
            with torch.no_grad():
                out["cons_mse_var"] = _mse_over_var(Zh1, Z_tgt[1:].reshape(n1, S, D), rt1["nxt_ok"], Z_tgt, valid)
        else:
            out["kl"] = out["kl_value"] = Z_post.sum() * 0

        # ---- open-loop multistep with prior samples and recorded actions only
        ol, herr, joint_nll, ol_extra = self._open_loop_loss(R_in, B, act, bel, goal, dt, horizons, ctx, bel_dyn,
                                                             reason_fn)
        out.update(ol_extra)

        # ---- anti-collapse regulariser on valid filter states
        var, cov = self._vicreg(Z_post[valid])
        out["reg_var"], out["reg_cov"] = var, cov

        z0 = Z_post.sum() * 0
        for k in ("latent", "outcome", "obs", "reward", "cont", "consistency"):
            out[k] = terms.get(k, z0)
        out["open_loop"] = ol
        total = (W["latent"] * out["latent"] + W["kl"] * out["kl"] + W["outcome"] * out["outcome"]
                 + W["obs"] * out["obs"] + W["reward"] * out["reward"] + W["cont"] * out["cont"]
                 + W["duration"] * out["duration"] + W["var"] * var + W["cov"] * cov + W["open_loop"] * ol
                 + self.consistency_w * out["consistency"])
        out["world"] = out["loss"] = total
        out["horizon_errors"] = herr
        out["horizons_trained"], out["horizons_skipped"] = horizons, skipped
        out["open_loop_joint_nll"] = joint_nll
        out["Z_post"] = Z_post.detach()
        # running state statistics are updated only after every head has read them: a batch is never normalised
        # with its own statistics (the first batch sees the identity, later ones the previous batches' statistics)
        if self.training:
            self.state_norm.update(Z_post[valid].detach())
        return out

    def kl_terms(self, post_lp: torch.Tensor, prior_lp: torch.Tensor, ok: torch.Tensor) -> dict:
        """Balanced KL of the posterior vs prior latent [N,G,C] log-probs over the rows ``ok`` [N].

        Per-group free bits (cfg.kl_free_bits_group = f > 0; Kingma et al. 2016): the batch-mean KL of every group is
        floored at f, so a group's KL gradient vanishes only while that group's own average KL is below f;
        "kl_groups_active" counts the groups above the floor. Otherwise the legacy per-row free nats on the total.
        Returns {"kl", "kl_value" (raw dynamics KL per row, detached)[, "kl_groups_active"]}."""
        cfg = self.cfg
        if self.free_bits_group > 0:
            dyn_g = categorical_kl_groups(post_lp.detach(), prior_lp)                 # [N,G]
            rep_g = categorical_kl_groups(post_lp, prior_lp.detach())
            okf = ok.float().unsqueeze(-1)
            nrm = okf.sum().clamp(min=1.0)
            dyn_m, rep_m = (dyn_g * okf).sum(0) / nrm, (rep_g * okf).sum(0) / nrm      # [G]
            fb = self.free_bits_group
            kl = cfg.kl_balance * dyn_m.clamp(min=fb).sum() + (1 - cfg.kl_balance) * rep_m.clamp(min=fb).sum()
            return {"kl": kl, "kl_value": dyn_m.detach().sum(), "kl_groups_active": int((dyn_m.detach() > fb).sum())}
        dyn = categorical_kl(post_lp.detach(), prior_lp)
        rep = categorical_kl(post_lp, prior_lp.detach())
        fn = cfg.kl_free_nats
        kl = cfg.kl_balance * dyn.clamp(min=fn) + (1 - cfg.kl_balance) * rep.clamp(min=fn)
        return {"kl": masked_mean(kl, ok), "kl_value": masked_mean(dyn.detach(), ok)}

    def _row_terms(self, Zh, n_samples, t_idx, b_idx, ctx: "_SeqCtx", alive, goal_rows: Optional[TokenSet],
                   max_groups: int, Zv: Optional[torch.Tensor] = None) -> dict:
        """Per-row, per-sample terms of predicted states Zh [R*n_samples,S,D] for transitions t_idx[r] -> t_idx[r]+1
        (each target row r repeated n_samples times consecutively; sample j of row r is one sampled future).
        Every likelihood is kept per sample ("out_lp" [R,s,Q], "r_ll"/"c_ll"/"dist" [R,s], "obs_lp" [n,s,M]) so the
        caller can credit all targets of a sampled future jointly (``_responsibilities``). ``alive`` [R]: the
        transition lies inside a valid episode segment. Reduce with ``_reduce``."""
        R = t_idx.shape[0]
        s = n_samples
        dev = Zh.device
        T, obs, Z_tgt, rew, done, valid, gen = ctx.T, ctx.obs, ctx.Z_tgt, ctx.rew, ctx.done, ctx.valid, ctx.gen
        rep = lambda x: x.repeat_interleave(s, 0)
        ob = ctx.ob_rows(t_idx, b_idx)
        om = ob.y_mask & alive.unsqueeze(-1)
        obr = OutcomeBatch(rows_rep(ob.q, s), rep(ob.lik), rep(ob.n_classes), rep(ob.y), rep(ob.censored), rep(om),
                           rep(ob.channel))
        d = self.outcome_dist(Zh, obr)
        rt = {"R": R, "s": s, "alive": alive, "om": om}
        rt["out_lp"] = d.log_prob(obr.y, obr.censored, obr.y_mask).view(R, s, -1)       # 0 where unobserved
        if self.outcome_nll_beta > 0:
            rt["out_w"] = outcome_beta_weights(d, self.outcome_nll_beta).view(R, s, -1)
        with torch.no_grad():
            mean = d.mean().view(R, s, -1).mean(1)
            rt["cm"] = om & torch.isin(ob.lik, torch.tensor(CONTINUOUS, device=dev))
            rt["sq"] = (mean - ob.y) ** 2
        if self.reward_targets:
            g_r = None if goal_rows is None else rows_rep(goal_rows, s)
            r_logits = self.reward_logits(Zh, g_r)
            r_t = rew[t_idx, b_idx]
            rt["r_ll"] = self.twohot.log_lik(r_logits, rep(r_t)).view(R, s)
            with torch.no_grad():
                rt["r_abs"] = (self.twohot.mean(r_logits).view(R, s).mean(1) - r_t).abs()
        else:
            # observation_only: the verified return increment is no target (nothing is read from batch.reward)
            rt["r_ll"] = torch.zeros(R, s, device=dev)
            rt["r_abs"] = torch.zeros(R, device=dev)
        c_logit = self.continue_logit(Zh).view(R, s)
        c_tgt = (~done[t_idx, b_idx]).float()
        rt["c_ll"] = -F.binary_cross_entropy_with_logits(c_logit, c_tgt.unsqueeze(1).expand(-1, s), reduction="none")
        # latent and observation targets at t+1 (only inside the episode and the sequence)
        tn = (t_idx + 1).clamp(max=T - 1)
        nxt_ok = alive & ((t_idx + 1) <= (T - 1)) & ~done[t_idx, b_idx] & valid[tn, b_idx]
        rt["nxt_ok"] = nxt_ok
        Zt_rows = rep(Z_tgt[tn, b_idx])
        Zl = Zh if Zv is None else Zv            # self-predictive terms: gradient routing of `latent_grad`
        rt["dist"] = self._latent_dist(Zl, Zt_rows, ctx.tgt_scale).view(R, s)   # ~1 - R^2 per sample
        # consistency: the predicted state itself (no read-out MLP) against the detached posterior state
        rt["cons"] = (((Zl.float() - Zt_rows.detach().float()) / ctx.tgt_scale) ** 2).mean((-1, -2)).view(R, s)
        idx = torch.nonzero(nxt_ok).flatten()
        if idx.numel() > max_groups:
            idx = idx[_randperm(idx.numel(), gen, dev)[:max_groups]]
        rt["obs_rows"] = idx
        if idx.numel() > 0:
            Zsel = Zh.view(R, s, *Zh.shape[1:])[idx].reshape(idx.numel() * s, *Zh.shape[1:])
            rt["obs_lp"], rt["obs_v"] = self._obs_logp(Zsel, gather_tokens(obs, tn[idx], b_idx[idx]), s, gen)
        return rt

    def _joint_loglik(self, rt: dict) -> torch.Tensor:
        """Log-likelihood of all targets of each row under each sampled future [R, s] (detached): the observed
        outcomes, reward, continuation, the latent pseudo-likelihood -W_latent * d / latent_temperature and the
        observation InfoNCE log-probability (mean over the row's scored tokens, weight W_obs)."""
        with torch.no_grad():
            a = rt["alive"].float().unsqueeze(1)
            ll = (rt["out_lp"] * rt["om"].unsqueeze(1)).sum(-1) + a * (rt["r_ll"] + rt["c_ll"])
            ll = ll - rt["nxt_ok"].float().unsqueeze(1) * (self.weights["latent"] / self.latent_temperature) \
                * rt["dist"]
            idx = rt["obs_rows"]
            if idx.numel() > 0:
                v = rt["obs_v"].float().unsqueeze(1)
                obs_m = (rt["obs_lp"] * v).sum(-1) / v.sum(-1).clamp(min=1.0)                    # [n, s]
                ll = ll.index_add(0, idx, self.weights["obs"] * obs_m)
        return ll

    def _responsibilities(self, rt: dict, traj: Optional[torch.Tensor] = None, n_traj: int = 0) -> dict:
        """Posterior weights over the s sampled futures (detached; each sums to 1 over the sample axis), per term.

        Training a stochastic model with several sampled futures maximises the mixture likelihood
        log mean_s p(all targets | future_s). Its gradient is sum_s pi_s grad log p(targets | future_s) with the
        posterior responsibilities pi_s = softmax_s(joint log-likelihood); ``_reduce`` forms exactly these
        responsibility-weighted log-likelihoods (with per-term normalisation), so every target is credited to the
        same sampled futures.
        ``open_loop_credit``: "trajectory" -- the joint log-likelihood of every scored target of a sampled future
        over all horizons (``traj`` [R] maps rows to their trajectory = (start, environment); rows of one
        trajectory share their weights); "row" -- joint over the terms of each row (one horizon); "per_term" --
        a separate mixture per term, outcome query and observation token (a looser objective: different terms may
        be explained by different futures; kept for ablations). With one sample every weight is 1."""
        s = rt["s"]
        R = rt["R"]
        dev = rt["alive"].device
        idx = rt["obs_rows"]
        mode_last = bool(rt.get("mode_last", False))      # the last sample is the mode-latent future (not mixed)
        if s == 1:
            one = torch.ones(R, 1, device=dev)
            rt["joint_ll"] = self._joint_loglik(rt)
            return {"out": one.unsqueeze(-1), "rew": one, "cont": one, "lat": one,
                    "obs": torch.ones(idx.numel(), 1, 1, device=dev)}

        def ml(x):                                         # exclude the mode sample from a softmax over samples
            if not mode_last:
                return x
            x = x.clone()
            x[:, -1] = float("-inf")
            return x
        with torch.no_grad():
            jl = self._joint_loglik(rt)
            rt["joint_ll"] = jl[:, :-1] if mode_last else jl
            mode = self.open_loop_credit
            if mode == "per_term":
                return {"out": torch.softmax(ml(rt["out_lp"].detach()), 1),
                        "rew": torch.softmax(ml(rt["r_ll"].detach()), 1),
                        "cont": torch.softmax(ml(rt["c_ll"].detach()), 1),
                        "lat": torch.softmax(ml(-rt["dist"].detach() / self.latent_temperature), 1),
                        "obs": torch.softmax(ml(rt["obs_lp"].detach()), 1) if idx.numel() > 0 else None}
            ll = jl
            if mode == "trajectory" and traj is not None:
                ll = torch.zeros(n_traj, s, device=dev).index_add_(0, traj, jl)[traj]
            pi = torch.softmax(ml(ll), 1)
        return {"out": pi.unsqueeze(-1), "rew": pi, "cont": pi, "lat": pi,
                "obs": pi[idx].unsqueeze(-1) if idx.numel() > 0 else None}

    @staticmethod
    def _reduce(rt: dict, resp: dict, sel: Optional[torch.Tensor] = None) -> dict:
        """Scalar losses (responsibility-weighted negative log-likelihoods, each normalised over its own observed
        entries) and diagnostics over the rows ``sel`` [R] (all rows when None)."""
        if sel is None:
            sel = torch.ones(rt["R"], dtype=torch.bool, device=rt["alive"].device)
        a = rt["alive"] & sel
        om = rt["om"] & sel.unsqueeze(-1)
        nx = rt["nxt_ok"] & sel
        olp = rt["out_lp"] if rt.get("out_w") is None else rt["out_lp"] * rt["out_w"]     # beta-NLL (loss only)
        out = {"outcome": -masked_mean((resp["out"] * olp).sum(1), om),
               "reward": -masked_mean((resp["rew"] * rt["r_ll"]).sum(1), a),
               "cont": -masked_mean((resp["cont"] * rt["c_ll"]).sum(1), a),
               "latent": masked_mean((resp["lat"] * rt["dist"]).sum(1), nx)}
        cons = rt.get("cons")
        out["consistency"] = masked_mean((resp["lat"] * cons).sum(1), nx) if cons is not None else out["latent"] * 0
        mix = (lambda x: x[:, :-1]) if rt.get("mode_last", False) else (lambda x: x)   # mixture samples only
        with torch.no_grad():
            out["outcome_nll"] = -masked_mean(logmeanexp(mix(rt["out_lp"]), 1), om)  # marginal mixture NLL per query
            cm = rt["cm"] & sel.unsqueeze(-1)
            out["outcome_mse"] = masked_mean(rt["sq"], cm)
            out["outcome_n"] = cm.sum()
            out["reward_mae"] = masked_mean(rt["r_abs"], a)
            out["latent_err"] = masked_mean(mix(rt["dist"]).mean(1), nx)             # standardised MSE (~1 - R^2)
            out["cons_err"] = masked_mean(mix(cons).mean(1), nx) if cons is not None else out["latent"].detach() * 0
            out["n_rows"] = a.sum()
            out["joint_nll"] = -masked_mean(logmeanexp(rt["joint_ll"], 1), a)        # per row (one horizon)
        idx = rt["obs_rows"]
        if idx.numel() > 0:
            ov = rt["obs_v"] & sel[idx].unsqueeze(-1)
            out["obs"] = -masked_mean((resp["obs"] * rt["obs_lp"]).sum(1), ov)
            out["obs_logp"] = masked_mean(logmeanexp(mix(rt["obs_lp"].detach()), 1), ov)
        else:
            out["obs"] = out["latent"] * 0
            out["obs_logp"] = out["obs"].detach()
        return out

    def _open_loop_loss(self, Z_post, B, act, bel, goal, dt, horizons, ctx: "_SeqCtx", bel_dyn=None,
                        reason_fn: Optional[Callable] = None):
        """Roll every start open-loop with prior samples and the recorded actions; the heads of all scored horizons
        are evaluated in one batched call and credited to sampled futures jointly (``open_loop_credit``). With
        ``bel_dyn`` = (fast_states, update_fn, tokens_fn) each row's belief is updated from its own sampled
        outcomes. ``Z_post`` [T,B,S,D] are the prior inputs at the real start steps (R_t on the DOM path); at later
        open-loop steps the input is the imagined state, through ``reason_fn(Z_hat, belief, goal)`` when given.
        With ``mode_latent`` one extra future per start uses the prior mode (straight-through) at every step; it is
        excluded from the mixture and its terms are added with weight 1/S. Returns (mean over horizons of the horizon
        loss, {h: errors}, joint mixture NLL per trajectory, extra diagnostics)."""
        dev = Z_post.device
        T, valid, done, gen = ctx.T, ctx.valid, ctx.done, ctx.gen
        rows = _rows
        S_ol = self.open_loop_samples
        S_tot = S_ol + (1 if self.mode_latent else 0)
        z0 = Z_post.sum() * 0
        if not horizons:
            return z0, {}, z0.detach(), {}
        starts = torch.arange(T, device=dev)
        if self.open_loop_starts is not None and T > self.open_loop_starts:
            perm = _randperm(T, gen, dev)[:self.open_loop_starts]
            starts = perm.sort().values
        n_s = starts.numel()
        Hmax = max(horizons)
        # rows ordered (start, b, sample); starts ascending so the active rows at step k form a prefix
        st_r = starts.repeat_interleave(B)                                           # [n_s*B]
        b_r = torch.arange(B, device=dev).repeat(n_s)
        Z = Z_post[st_r, b_r].repeat_interleave(S_tot, 0)
        bel_rows = rows(bel, st_r, b_r)
        bel_rows = None if bel_rows is None else rows_rep(bel_rows, S_tot)
        g_rows = rows(goal, st_r, b_r)
        g_rows = None if g_rows is None else rows_rep(g_rows, S_tot)
        mode_row = (torch.arange(n_s * B * S_tot, device=dev) % S_tot) == (S_tot - 1) if self.mode_latent else None
        alive = valid[st_r, b_r].clone()                                             # [n_s*B]
        fast = None
        if bel_dyn is not None:
            fs, upd_fn, tok_fn = bel_dyn
            rep_idx = torch.arange(B, device=dev).repeat_interleave(S_tot)
            fast = fast_cat([fast_rows(fs[int(t0)], rep_idx) for t0 in starts.tolist()])   # rows (start, b, sample)
        coll = []                                                                    # per scored horizon
        for k in range(1, Hmax + 1):
            n_a = int((starts <= T - k).sum())
            if n_a == 0:
                break
            na_rows = n_a * B
            nr = na_rows * S_tot
            Z = Z[:nr]
            st_r, b_r, alive = st_r[:na_rows], b_r[:na_rows], alive[:na_rows]
            if fast is not None and fast.logw.shape[0] > nr:
                fast = fast_rows(fast, torch.arange(nr, device=dev))
            if bel_rows is not None:
                bel_rows = _slice_tokens(bel_rows, nr)
            if g_rows is not None:
                g_rows = _slice_tokens(g_rows, nr)
            t_idx = st_r + (k - 1)
            alive = alive & valid[t_idx, b_r]
            a_k = rows_rep(rows(act, t_idx, b_r), S_tot)
            dt_k = dt[t_idx, b_r].repeat_interleave(S_tot)
            Z_in = Z if (k == 1 or reason_fn is None) else reason_fn(Z, bel_rows, g_rows)
            lp = self.prior_logits(Z_in, a_k, bel_rows, dt_k)
            zk = self.sample_latent(lp, True, gen)
            if mode_row is not None:
                zk = torch.where(mode_row[:nr].unsqueeze(-1), self.sample_latent(lp, False), zk)
            Z_prev = Z_in
            Z, Zv_k = self.transition(Z_in, a_k, bel_rows, dt_k, zk, view=True,
                                      **ctx.row_kw(b_r.repeat_interleave(S_tot)))
            if fast is not None and k < Hmax:
                # imagined belief update from the model's own sampled outcomes (no real outcome after the start)
                with torch.no_grad():
                    ob = ctx.ob_rows(t_idx, b_r)
                    r = lambda x: x.repeat_interleave(S_tot, 0)
                    q = OutcomeBatch(rows_rep(ob.q, S_tot), r(ob.lik), r(ob.n_classes), r(ob.y), r(ob.censored),
                                     r(ob.y_mask), r(ob.channel))
                    ys = self.outcome_dist(Z, q).sample(gen)
                    om = q.y_mask.bool()
                    imagined = OutcomeBatch(q.q, q.lik, q.n_classes, torch.where(om, ys, torch.zeros_like(ys)),
                                            torch.zeros_like(om), om, q.channel)
                    out_f = upd_fn(fast, Z_prev.detach(), a_k, imagined)
                    fast = out_f[0] if isinstance(out_f, tuple) else out_f
                    bel_rows = tok_fn(fast)
            if k in horizons:
                coll.append((k, Z, t_idx, b_r, alive.clone(), st_r, Zv_k))
            alive = alive & ~done[t_idx, b_r]
        if not coll:
            return z0, {}, z0.detach(), {}
        Zall = torch.cat([c[1] for c in coll])
        Zv_all = torch.cat([c[6] for c in coll])
        t_all = torch.cat([c[2] for c in coll])
        b_all = torch.cat([c[3] for c in coll])
        al_all = torch.cat([c[4] for c in coll])
        h_all = torch.cat([torch.full_like(c[2], c[0]) for c in coll])
        # trajectory of each row: rows of every horizon block are a prefix of the (start, b) list
        traj = torch.cat([torch.arange(c[2].numel(), device=dev) for c in coll])
        n_traj = coll[0][2].numel()
        # goal of the start step (known at the start; no future information)
        g_all = rows(goal, torch.cat([c[5] for c in coll]), b_all)
        rt = self._row_terms(Zall, S_tot, t_all, b_all, ctx, al_all, g_all, self.obs_max_groups * len(coll),
                             Zv=Zv_all)
        rt["mode_last"] = self.mode_latent
        resp = self._responsibilities(rt, traj, n_traj)
        resp_mode = None
        if self.mode_latent:
            oh = torch.zeros(rt["R"], S_tot, device=dev)
            oh[:, -1] = 1.0
            idx = rt["obs_rows"]
            resp_mode = {"out": oh.unsqueeze(-1), "rew": oh, "cont": oh, "lat": oh,
                         "obs": oh[idx].unsqueeze(-1) if idx.numel() > 0 else None}
        losses, herr, extra = [], {}, {}
        W = self.weights
        wc = self.consistency_w
        mode_terms = []
        for k, *_ in coll:
            r = self._reduce(rt, resp, h_all == k)
            lk = r["outcome"] + r["reward"] + r["cont"] + W["latent"] * r["latent"] + W["obs"] * r["obs"]
            if k == 1 and wc > 0:
                lk = lk + wc * r["consistency"]
            herr[k] = {"outcome_nll": float(r["outcome_nll"]), "outcome_mse": float(r["outcome_mse"]),
                       "outcome_n": int(r["outcome_n"]), "reward_mae": float(r["reward_mae"]),
                       "cont": float(r["cont"].detach()), "latent_err": float(r["latent_err"]),
                       "cons_err": float(r["cons_err"]), "obs_logp": float(r["obs_logp"]),
                       "joint_nll": float(r["joint_nll"]), "n_rows": int(r["n_rows"])}
            if resp_mode is not None:
                rm = self._reduce(rt, resp_mode, h_all == k)
                lm = rm["outcome"] + rm["reward"] + rm["cont"] + W["latent"] * rm["latent"] + W["obs"] * rm["obs"]
                if k == 1 and wc > 0:
                    lm = lm + wc * rm["consistency"]
                lk = lk + lm / S_ol
                mode_terms.append(lm.detach())
                herr[k]["mode_outcome_nll"] = float(rm["outcome"].detach())
                herr[k]["mode_cons_err"] = float(rm["consistency"].detach())
            losses.append(lk)
        with torch.no_grad():
            tot = torch.zeros(n_traj, rt["joint_ll"].shape[1], device=dev).index_add_(0, traj, rt["joint_ll"])
            joint_nll = -masked_mean(logmeanexp(tot, 1), al_all[:n_traj])
            if mode_terms:
                extra["open_loop_mode"] = torch.stack(mode_terms).mean()
            if coll[0][0] == 1:
                # one-step prior from the real start state vs the next posterior (pure prior prediction, the
                # state decide() scores): mode-latent row when present, else the first sampled future
                k1, Z1, t1, b1, al1, _, _ = coll[0]
                pick = S_tot - 1 if self.mode_latent else 0
                Zp = Z1.view(-1, S_tot, *Z1.shape[1:])[:, pick]
                tn = (t1 + 1).clamp(max=T - 1)
                ok = al1 & ((t1 + 1) <= (T - 1)) & ~done[t1, b1] & valid[tn, b1]
                extra["cons_mse_var_mode1"] = _mse_over_var(Zp, ctx.Z_tgt[tn, b1], ok, ctx.Z_tgt, valid)
        return torch.stack(losses).mean(), herr, joint_nll, extra

    # ------------------------------------------------------------------ open-loop inference (forecasting)
    @torch.no_grad()
    def forecast(self, batch: SequenceBatch, start: int, horizons: Sequence[int], beliefs: Optional[list] = None,
                 n_samples: int = 8, generator: Optional[torch.Generator] = None, sample: bool = True,
                 Z0: Optional[torch.Tensor] = None, reason_fn: Optional[Callable] = None, sem=None,
                 code_rows: Optional[torch.Tensor] = None) -> dict:
        """Open-loop forecast of the outcomes Y_{start+h-1} for each h, from O_{<=start} and the recorded actions
        A_{start..start+h-1} with their recorded durations. Returns {h: {"mean": [B,Q], "var": [B,Q]}} (mixture
        over ``n_samples`` stochastic futures). Never reads observations or outcomes after ``start``.
        ``Z0`` [B,S,D]: the prior input at ``start`` from the agent's own filter (DOM: R_start of
        ``train_seq.filter_sequence``; it must have seen only O_{<=start}); without it the world model's own filter is
        run. ``reason_fn(Z_hat, belief, goal)`` maps imagined states to prior inputs at steps >= 2 (as in training).
        ``sem`` / ``code_rows``: the semantic context / slot-code rows of the batch's B rows (see `loss`)."""
        steps = batch.steps
        B = steps[0].obs.B
        dev = steps[0].obs.x.device
        if Z0 is None:
            Z = self.filter(SequenceBatch(steps[:start + 1]), None if beliefs is None else beliefs[:start + 1],
                            sem=sem, code_rows=code_rows)[-1]
        else:
            Z = Z0.float()
        Z = Z.repeat_interleave(n_samples, 0)
        rkw = _SeqCtx(T=0, ob_rows=None, obs=None, Z_tgt=None, rew=None, done=None, valid=None, gen=None,
                      tgt_scale=None, sem=None if isinstance(sem, (list, tuple)) else sem,
                      code_rows=code_rows).row_kw(torch.arange(B, device=dev).repeat_interleave(n_samples))
        b = None if beliefs is None else rows_rep(beliefs[start], n_samples)
        g = None if steps[start].goal is None else rows_rep(steps[start].goal, n_samples)
        res = {}
        for k in range(1, max(horizons) + 1):
            if start + k - 1 >= len(steps):
                break
            st = steps[start + k - 1]
            a = rows_rep(st.act, n_samples)
            dtk = _as_dt(st.dt, B, dev).repeat_interleave(n_samples)
            if k > 1 and reason_fn is not None:
                Z = reason_fn(Z, b, g)
            Z, _ = self.prior(Z, a, b, dtk, sample=sample, generator=generator, **rkw)
            if k in horizons:
                ob = st.outcomes
                q = OutcomeBatch(rows_rep(ob.q, n_samples), ob.lik.repeat_interleave(n_samples, 0),
                                 ob.n_classes.repeat_interleave(n_samples, 0), ob.y.repeat_interleave(n_samples, 0),
                                 ob.censored.repeat_interleave(n_samples, 0), ob.y_mask.repeat_interleave(n_samples, 0),
                                 ob.channel.repeat_interleave(n_samples, 0))
                d = self.outcome_dist(Z, q)
                m = d.mean().view(B, n_samples, -1)
                v = d.variance().view(B, n_samples, -1)
                mean = m.mean(1)
                res[k] = {"mean": mean, "var": v.mean(1) + ((m - mean.unsqueeze(1)) ** 2).mean(1),
                          "reward": self.reward(Z, None if st.goal is None else rows_rep(st.goal, n_samples)
                                                ).view(B, n_samples).mean(1)}
        return res


def _rand(shape, gen: Optional[torch.Generator], device) -> torch.Tensor:
    """U[0,1) noise drawn from ``gen`` (on the generator's own device) and moved to ``device``; global RNG without
    a generator."""
    if gen is None:
        return torch.rand(shape, device=device)
    return torch.rand(shape, generator=gen, device=gen.device).to(device)


def _randperm(n: int, gen: Optional[torch.Generator], device) -> torch.Tensor:
    """Random permutation of range(n) from ``gen`` (global RNG without a generator)."""
    if gen is None:
        return torch.randperm(n, device=device)
    return torch.randperm(n, generator=gen, device=gen.device).to(device)


def _empty_tokens(B: int, D: int, device) -> TokenSet:
    """A TokenSet with zero tokens per row (stands in for a missing goal)."""
    return TokenSet(torch.zeros(B, 0, D, device=device), torch.zeros(B, 0, dtype=torch.bool, device=device),
                    torch.zeros(B, 0, dtype=torch.long, device=device), torch.zeros(B, 0, device=device),
                    torch.full((B, 0), -1, dtype=torch.long, device=device))


def rows_rep(ts: TokenSet, n: int) -> TokenSet:
    """Repeat each row n times consecutively (no refs)."""
    if n == 1:
        return TokenSet(ts.x, ts.mask, ts.kind, ts.time, ts.entity, None, ts.entity2)
    r = lambda x: x.repeat_interleave(n, 0)
    return TokenSet(r(ts.x), r(ts.mask), r(ts.kind), r(ts.time), r(ts.entity), None,
                    None if ts.entity2 is None else r(ts.entity2))


def _slice_tokens(ts: TokenSet, n: int) -> TokenSet:
    return TokenSet(ts.x[:n], ts.mask[:n], ts.kind[:n], ts.time[:n], ts.entity[:n], None,
                    None if ts.entity2 is None else ts.entity2[:n])


def _rows(ts: Optional[TokenSet], t_idx, b_idx) -> Optional[TokenSet]:
    return None if ts is None else gather_tokens(ts, t_idx, b_idx)


@dataclass
class _SeqCtx:
    """Stacked targets of one SequenceBatch shared by the teacher-forced and open-loop terms."""
    T: int
    ob_rows: Callable
    obs: TokenSet
    Z_tgt: torch.Tensor
    rew: torch.Tensor
    done: torch.Tensor
    valid: torch.Tensor
    gen: Optional[torch.Generator]
    tgt_scale: torch.Tensor
    sem: object = None                       # semantic context of the batch's B rows (or None)
    code_rows: Optional[torch.Tensor] = None  # [B] batch row -> entity slot-code episode row (or None)
    sem_rows: Optional[torch.Tensor] = None   # [B] batch row -> semantic row of a source-aligned sem (or None)

    def row_kw(self, b_idx: torch.Tensor) -> dict:
        """sem / sem_rows / code_rows keywords of a transition over flattened rows whose batch rows are b_idx."""
        kw = {}
        if self.sem is not None:
            kw["sem"] = self.sem
            kw["sem_rows"] = b_idx if self.sem_rows is None else self.sem_rows.to(b_idx.device)[b_idx]
        if self.code_rows is not None:
            kw["code_rows"] = self.code_rows.to(b_idx.device)[b_idx]
        return kw
