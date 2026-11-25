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
                 train_horizons: Optional[Sequence[int]] = None, open_loop_credit: str = "trajectory"):
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
        # EMA target encoder: a frozen copy of the backbone (state_dict-saved, never optimised)
        self.target_core = copy.deepcopy(core)
        for p in self.target_core.parameters():
            p.requires_grad_(False)
        self.n_transitions = 0              # world-model transition evaluations (rows), for compute accounting

    # ------------------------------------------------------------------ parameters / target
    def trainable_parameters(self):
        """All optimisable parameters (excludes the EMA target copy)."""
        tid = {id(p) for p in self.target_core.parameters()}
        return [p for p in self.parameters() if id(p) not in tid and p.requires_grad]

    @torch.no_grad()
    def update_target(self, decay: Optional[float] = None) -> None:
        """target <- decay * target + (1 - decay) * online (call after each optimiser step)."""
        d = self.ema_decay if decay is None else decay
        for pt, p in zip(self.target_core.parameters(), self.core.parameters()):
            pt.lerp_(p.detach().to(pt.dtype), 1.0 - d)
        for bt, b in zip(self.target_core.buffers(), self.core.buffers()):
            bt.copy_(b)

    @torch.no_grad()
    def reset_target(self) -> None:
        """Copy the online backbone into the target (e.g. after loading new backbone weights)."""
        self.target_core.load_state_dict(self.core.state_dict())

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

    def _core(self, Z, ctx: TokenSet, core=None):
        core = self.core if core is None else core
        on = self.use_autocast and Z.is_cuda
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=on):
            Zn, ctx_out = core(Z, ctx)
        return Zn.float(), ctx_out

    # ------------------------------------------------------------------ filter (posterior state)
    def filter_step(self, Z_prev, obs: Optional[TokenSet], act_prev: Optional[TokenSet] = None, dt_prev=None,
                    belief: Optional[TokenSet] = None, goal: Optional[TokenSet] = None,
                    memory: Optional[TokenSet] = None, core=None, reset: Optional[torch.Tensor] = None):
        """One agent state update Z_t = F(Z_{t-1}, [O_t; A_{t-1}; TIME(dt_{t-1}); B_t; M_t; G_t]).
        ``reset`` [B] bool: rows starting a new episode (Z_prev replaced by the initial state; A_{t-1} and the
        elapsed time masked out). Returns (Z [B,S,D], ctx_out)."""
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
        return self._core(Z_prev, cat_tokens(*parts), core)

    def filter(self, batch: SequenceBatch, beliefs: Optional[list] = None, memory: Optional[list] = None,
               core=None, Z0=None) -> torch.Tensor:
        """Run the filter over a sequence. Returns Z_post [T,B,S,D] (Z_post[t] has seen O_{<=t} only)."""
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
                                    None if memory is None else memory[t], core=core, reset=reset)
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

    def transition(self, Z, act: TokenSet, belief: Optional[TokenSet], dt, z) -> torch.Tensor:
        """Z^_{t+1} = F(Z_t, [A_t; B_t; TIME(dt_t); NOISE(z_t)]) (no observations)."""
        B = Z.shape[0]
        dt = _as_dt(dt, B, Z.device)
        self.n_transitions += B
        Zn, _ = self._core(Z, cat_tokens(act, belief, self.time_tokens(dt), self.noise_tokens(z, dt)))
        return Zn

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
              generator: Optional[torch.Generator] = None, z: Optional[torch.Tensor] = None):
        """WorldModelP.prior: Z [B,S,D], act ACTION tokens, belief BELIEF tokens (or None), dt [B] days (None ->
        the predicted duration). Returns (Z_hat [B,S,D], {"logits": [B,G,C], "z": [B,G*C], "dt": [B]})."""
        B = Z.shape[0]
        dt = self.predict_duration(Z, act).detach() if dt is None else _as_dt(dt, B, Z.device)
        logp = self.prior_logits(Z, act, belief, dt)
        if z is None:
            z = self.sample_latent(logp, sample, generator)
        Zh = self.transition(Z, act, belief, dt, z)
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
        p = Zh.float() + self.latent_pred(self.state_norm(Zh))
        return (((p - Zt.detach().float()) / scale) ** 2).mean((-1, -2))

    @staticmethod
    def _target_scale(Z_tgt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Across-sample std of valid target states [S,D] with a floor at 10% of the mean std."""
        X = Z_tgt[valid].float()
        if X.shape[0] < 2:
            return torch.ones(Z_tgt.shape[-2:], device=Z_tgt.device)
        var = X.var(0)
        return torch.sqrt(var + 0.01 * var.mean() + 1e-8)

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
             belief_tokens_fn: Optional[Callable] = None) -> dict:
        """Full world-model objective on real recorded sequences.

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
        ("open_loop_joint_nll") and the detached filter states "Z_post" [T,B,S,D] for downstream users."""
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

        # ---- filter (online, with gradients) and EMA target (no gradients)
        Z_post = self.filter(batch, beliefs, memory)                                 # [T,B,S,D]
        with torch.no_grad():
            Z_tgt = self.filter(batch, beliefs, memory, core=self.target_core)
        S, D = Z_post.shape[2], Z_post.shape[3]
        ctx = _SeqCtx(T=T, ob_rows=ob_rows, obs=obs, Z_tgt=Z_tgt, rew=rew, done=done, valid=valid, gen=generator,
                      tgt_scale=self._target_scale(Z_tgt, valid))

        tt = torch.arange(T, device=dev).repeat_interleave(B)
        bb = torch.arange(B, device=dev).repeat(T)
        Zs = Z_post.reshape(T * B, S, D)
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
            dyn = categorical_kl(post_lp.detach(), prior_lp)
            rep = categorical_kl(post_lp, prior_lp.detach())
            fn = cfg.kl_free_nats
            kl = cfg.kl_balance * dyn.clamp(min=fn) + (1 - cfg.kl_balance) * rep.clamp(min=fn)
            out["kl"] = masked_mean(kl, ok1)
            out["kl_value"] = masked_mean(dyn.detach(), ok1)
            # transitions without a next observation (episode end / sequence padding) have no posterior: use the prior
            lat_lp = torch.where(ok1.view(-1, 1, 1), post_lp, prior_lp)
            z_post = self.sample_latent(lat_lp, True, generator)
            Zh1 = self.transition(Zs[:n1], rows(act, t1, b1), rows(bel, t1, b1), dt_r[:n1], z_post)
            rt1 = self._row_terms(Zh1, 1, t1, b1, ctx, valid_r[:n1], rows(goal, t1, b1), self.obs_max_groups)
            terms = self._reduce(rt1, self._responsibilities(rt1))
        else:
            out["kl"] = out["kl_value"] = Z_post.sum() * 0

        # ---- open-loop multistep with prior samples and recorded actions only
        ol, herr, joint_nll = self._open_loop_loss(Z_post, B, act, bel, goal, dt, horizons, ctx, bel_dyn)

        # ---- anti-collapse regulariser on valid filter states
        var, cov = self._vicreg(Z_post[valid])
        out["reg_var"], out["reg_cov"] = var, cov

        z0 = Z_post.sum() * 0
        for k in ("latent", "outcome", "obs", "reward", "cont"):
            out[k] = terms.get(k, z0)
        out["open_loop"] = ol
        total = (W["latent"] * out["latent"] + W["kl"] * out["kl"] + W["outcome"] * out["outcome"]
                 + W["obs"] * out["obs"] + W["reward"] * out["reward"] + W["cont"] * out["cont"]
                 + W["duration"] * out["duration"] + W["var"] * var + W["cov"] * cov + W["open_loop"] * ol)
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

    def _row_terms(self, Zh, n_samples, t_idx, b_idx, ctx: "_SeqCtx", alive, goal_rows: Optional[TokenSet],
                   max_groups: int) -> dict:
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
        with torch.no_grad():
            mean = d.mean().view(R, s, -1).mean(1)
            rt["cm"] = om & torch.isin(ob.lik, torch.tensor(CONTINUOUS, device=dev))
            rt["sq"] = (mean - ob.y) ** 2
        g_r = None if goal_rows is None else rows_rep(goal_rows, s)
        r_logits = self.reward_logits(Zh, g_r)
        r_t = rew[t_idx, b_idx]
        rt["r_ll"] = self.twohot.log_lik(r_logits, rep(r_t)).view(R, s)
        with torch.no_grad():
            rt["r_abs"] = (self.twohot.mean(r_logits).view(R, s).mean(1) - r_t).abs()
        c_logit = self.continue_logit(Zh).view(R, s)
        c_tgt = (~done[t_idx, b_idx]).float()
        rt["c_ll"] = -F.binary_cross_entropy_with_logits(c_logit, c_tgt.unsqueeze(1).expand(-1, s), reduction="none")
        # latent and observation targets at t+1 (only inside the episode and the sequence)
        tn = (t_idx + 1).clamp(max=T - 1)
        nxt_ok = alive & ((t_idx + 1) <= (T - 1)) & ~done[t_idx, b_idx] & valid[tn, b_idx]
        rt["nxt_ok"] = nxt_ok
        rt["dist"] = self._latent_dist(Zh, rep(Z_tgt[tn, b_idx]), ctx.tgt_scale).view(R, s)   # ~1 - R^2 per sample
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
        if s == 1:
            one = torch.ones(R, 1, device=dev)
            rt["joint_ll"] = self._joint_loglik(rt)
            return {"out": one.unsqueeze(-1), "rew": one, "cont": one, "lat": one,
                    "obs": torch.ones(idx.numel(), 1, 1, device=dev)}
        with torch.no_grad():
            jl = self._joint_loglik(rt)
            rt["joint_ll"] = jl
            mode = self.open_loop_credit
            if mode == "per_term":
                return {"out": torch.softmax(rt["out_lp"].detach(), 1), "rew": torch.softmax(rt["r_ll"].detach(), 1),
                        "cont": torch.softmax(rt["c_ll"].detach(), 1),
                        "lat": torch.softmax(-rt["dist"].detach() / self.latent_temperature, 1),
                        "obs": torch.softmax(rt["obs_lp"].detach(), 1) if idx.numel() > 0 else None}
            ll = jl
            if mode == "trajectory" and traj is not None:
                ll = torch.zeros(n_traj, s, device=dev).index_add_(0, traj, jl)[traj]
            pi = torch.softmax(ll, 1)
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
        out = {"outcome": -masked_mean((resp["out"] * rt["out_lp"]).sum(1), om),
               "reward": -masked_mean((resp["rew"] * rt["r_ll"]).sum(1), a),
               "cont": -masked_mean((resp["cont"] * rt["c_ll"]).sum(1), a),
               "latent": masked_mean((resp["lat"] * rt["dist"]).sum(1), nx)}
        with torch.no_grad():
            out["outcome_nll"] = -masked_mean(logmeanexp(rt["out_lp"], 1), om)      # marginal mixture NLL per query
            cm = rt["cm"] & sel.unsqueeze(-1)
            out["outcome_mse"] = masked_mean(rt["sq"], cm)
            out["outcome_n"] = cm.sum()
            out["reward_mae"] = masked_mean(rt["r_abs"], a)
            out["latent_err"] = masked_mean(rt["dist"].mean(1), nx)                  # standardised MSE (~1 - R^2)
            out["n_rows"] = a.sum()
            out["joint_nll"] = -masked_mean(logmeanexp(rt["joint_ll"], 1), a)        # per row (one horizon)
        idx = rt["obs_rows"]
        if idx.numel() > 0:
            ov = rt["obs_v"] & sel[idx].unsqueeze(-1)
            out["obs"] = -masked_mean((resp["obs"] * rt["obs_lp"]).sum(1), ov)
            out["obs_logp"] = masked_mean(logmeanexp(rt["obs_lp"].detach(), 1), ov)
        else:
            out["obs"] = out["latent"] * 0
            out["obs_logp"] = out["obs"].detach()
        return out

    def _open_loop_loss(self, Z_post, B, act, bel, goal, dt, horizons, ctx: "_SeqCtx", bel_dyn=None):
        """Roll every start open-loop with prior samples and the recorded actions; the heads of all scored horizons
        are evaluated in one batched call and credited to sampled futures jointly (``open_loop_credit``). With
        ``bel_dyn`` = (fast_states, update_fn, tokens_fn) each row's belief is updated from its own sampled
        outcomes. Returns (mean over horizons of the horizon loss, {h: errors}, joint mixture NLL per trajectory)."""
        dev = Z_post.device
        T, valid, done, gen = ctx.T, ctx.valid, ctx.done, ctx.gen
        rows = _rows
        S_ol = self.open_loop_samples
        z0 = Z_post.sum() * 0
        if not horizons:
            return z0, {}, z0.detach()
        starts = torch.arange(T, device=dev)
        if self.open_loop_starts is not None and T > self.open_loop_starts:
            perm = _randperm(T, gen, dev)[:self.open_loop_starts]
            starts = perm.sort().values
        n_s = starts.numel()
        Hmax = max(horizons)
        # rows ordered (start, b, sample); starts ascending so the active rows at step k form a prefix
        st_r = starts.repeat_interleave(B)                                           # [n_s*B]
        b_r = torch.arange(B, device=dev).repeat(n_s)
        Z = Z_post[st_r, b_r].repeat_interleave(S_ol, 0)
        bel_rows = rows(bel, st_r, b_r)
        bel_rows = None if bel_rows is None else rows_rep(bel_rows, S_ol)
        alive = valid[st_r, b_r].clone()                                             # [n_s*B]
        fast = None
        if bel_dyn is not None:
            fs, upd_fn, tok_fn = bel_dyn
            rep_idx = torch.arange(B, device=dev).repeat_interleave(S_ol)
            fast = fast_cat([fast_rows(fs[int(t0)], rep_idx) for t0 in starts.tolist()])   # rows (start, b, sample)
        coll = []                                                                    # per scored horizon
        for k in range(1, Hmax + 1):
            n_a = int((starts <= T - k).sum())
            if n_a == 0:
                break
            na_rows = n_a * B
            Z = Z[:na_rows * S_ol]
            st_r, b_r, alive = st_r[:na_rows], b_r[:na_rows], alive[:na_rows]
            if fast is not None and fast.logw.shape[0] > na_rows * S_ol:
                fast = fast_rows(fast, torch.arange(na_rows * S_ol, device=dev))
            if bel_rows is not None:
                bel_rows = _slice_tokens(bel_rows, na_rows * S_ol)
            t_idx = st_r + (k - 1)
            alive = alive & valid[t_idx, b_r]
            a_k = rows_rep(rows(act, t_idx, b_r), S_ol)
            dt_k = dt[t_idx, b_r].repeat_interleave(S_ol)
            lp = self.prior_logits(Z, a_k, bel_rows, dt_k)
            zk = self.sample_latent(lp, True, gen)
            Z_prev = Z
            Z = self.transition(Z, a_k, bel_rows, dt_k, zk)
            if fast is not None and k < Hmax:
                # imagined belief update from the model's own sampled outcomes (no real outcome after the start)
                with torch.no_grad():
                    ob = ctx.ob_rows(t_idx, b_r)
                    r = lambda x: x.repeat_interleave(S_ol, 0)
                    q = OutcomeBatch(rows_rep(ob.q, S_ol), r(ob.lik), r(ob.n_classes), r(ob.y), r(ob.censored),
                                     r(ob.y_mask), r(ob.channel))
                    ys = self.outcome_dist(Z, q).sample(gen)
                    om = q.y_mask.bool()
                    imagined = OutcomeBatch(q.q, q.lik, q.n_classes, torch.where(om, ys, torch.zeros_like(ys)),
                                            torch.zeros_like(om), om, q.channel)
                    out_f = upd_fn(fast, Z_prev.detach(), a_k, imagined)
                    fast = out_f[0] if isinstance(out_f, tuple) else out_f
                    bel_rows = tok_fn(fast)
            if k in horizons:
                coll.append((k, Z, t_idx, b_r, alive.clone(), st_r))
            alive = alive & ~done[t_idx, b_r]
        if not coll:
            return z0, {}, z0.detach()
        Zall = torch.cat([c[1] for c in coll])
        t_all = torch.cat([c[2] for c in coll])
        b_all = torch.cat([c[3] for c in coll])
        al_all = torch.cat([c[4] for c in coll])
        h_all = torch.cat([torch.full_like(c[2], c[0]) for c in coll])
        # trajectory of each row: rows of every horizon block are a prefix of the (start, b) list
        traj = torch.cat([torch.arange(c[2].numel(), device=dev) for c in coll])
        n_traj = coll[0][2].numel()
        # goal of the start step (known at the start; no future information)
        g_all = rows(goal, torch.cat([c[5] for c in coll]), b_all)
        rt = self._row_terms(Zall, S_ol, t_all, b_all, ctx, al_all, g_all, self.obs_max_groups * len(coll))
        resp = self._responsibilities(rt, traj, n_traj)
        losses, herr = [], {}
        W = self.weights
        for k, *_ in coll:
            r = self._reduce(rt, resp, h_all == k)
            losses.append(r["outcome"] + r["reward"] + r["cont"] + W["latent"] * r["latent"] + W["obs"] * r["obs"])
            herr[k] = {"outcome_nll": float(r["outcome_nll"]), "outcome_mse": float(r["outcome_mse"]),
                       "outcome_n": int(r["outcome_n"]), "reward_mae": float(r["reward_mae"]),
                       "cont": float(r["cont"].detach()), "latent_err": float(r["latent_err"]),
                       "obs_logp": float(r["obs_logp"]), "joint_nll": float(r["joint_nll"]),
                       "n_rows": int(r["n_rows"])}
        with torch.no_grad():
            tot = torch.zeros(n_traj, S_ol, device=dev).index_add_(0, traj, rt["joint_ll"])
            joint_nll = -masked_mean(logmeanexp(tot, 1), al_all[:n_traj])
        return torch.stack(losses).mean(), herr, joint_nll

    # ------------------------------------------------------------------ open-loop inference (forecasting)
    @torch.no_grad()
    def forecast(self, batch: SequenceBatch, start: int, horizons: Sequence[int], beliefs: Optional[list] = None,
                 n_samples: int = 8, generator: Optional[torch.Generator] = None, sample: bool = True) -> dict:
        """Open-loop forecast of the outcomes Y_{start+h-1} for each h, from O_{<=start} and the recorded actions
        A_{start..start+h-1} with their recorded durations. Returns {h: {"mean": [B,Q], "var": [B,Q]}} (mixture
        over ``n_samples`` stochastic futures). Never reads observations or outcomes after ``start``."""
        steps = batch.steps
        B = steps[0].obs.B
        dev = steps[0].obs.x.device
        Z = self.filter(SequenceBatch(steps[:start + 1]), None if beliefs is None else beliefs[:start + 1])[-1]
        Z = Z.repeat_interleave(n_samples, 0)
        b = None if beliefs is None else rows_rep(beliefs[start], n_samples)
        res = {}
        for k in range(1, max(horizons) + 1):
            if start + k - 1 >= len(steps):
                break
            st = steps[start + k - 1]
            a = rows_rep(st.act, n_samples)
            dtk = _as_dt(st.dt, B, dev).repeat_interleave(n_samples)
            Z, _ = self.prior(Z, a, b, dtk, sample=sample, generator=generator)
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
