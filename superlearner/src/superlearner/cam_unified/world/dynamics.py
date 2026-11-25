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
  (InfoNCE), verified reward r_t (two-hot symlog), continuation 1 - done_t; duration dt_t from (Z_t, A_t)
  (log-normal on seconds) so imagination can predict how long an action takes.
* anti-collapse: VICReg-style variance hinge and covariance penalty on the (layer-normalised) filter states.
* every head reads latent states through ``StateNorm`` (running per-position standardisation), so the
  sample-specific part of Z is visible at unit scale even when a shared component dominates the residual stream.
* open-loop multistep: from filtered Z_t roll h steps with the recorded actions and durations and sampled prior
  latents, never seeing O_{>t}; each horizon in ``cfg.train_horizons`` is scored against Z_target_{t+h}, Y_{t+h-1},
  r_{t+h-1} and O_{t+h} with a multi-sample mixture likelihood (``open_loop_samples`` futures per start), which does
  not penalise a correctly stochastic model. Per-horizon errors are returned.

Batch alignment (``SequenceBatch``): step t holds O_t, the executed action A_t, its observed consequences Y_t, the
elapsed time dt_t until step t+1, reward r_t and done_t (the episode ended after A_t). A row whose obs and action
masks are all False at step t is padding.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CamConfig
from ..interfaces import (Likelihood, OutcomeBatch, SequenceBatch, TokenKind, TokenSet)
from ._util import (MLP, SECONDS_PER_DAY, AttnPool, StateNorm, TimeEncoder, TwoHot, cat_tokens, gather_tokens,
                    logmeanexp, masked_mean, outcome_pad, safe_key_padding, stack_tokens)
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
                 train_horizons: Optional[Sequence[int]] = None):
        super().__init__()
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
        self.rew_src = nn.Embedding(2, D)
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
    def _pool_goal(self, pool, src, Zh, goal: Optional[TokenSet]):
        x, m = self.state_norm(Zh).float() + src.weight[0], torch.ones(Zh.shape[:2], dtype=torch.bool, device=Zh.device)
        if goal is not None and goal.N > 0:
            x = torch.cat([x, goal.x.float() + src.weight[1]], 1)
            m = torch.cat([m, goal.mask.bool()], 1)
        return pool(x, m)[:, 0]

    def reward_logits(self, Zh, goal: Optional[TokenSet] = None) -> torch.Tensor:
        return self.rew_head(self._pool_goal(self.rew_pool, self.rew_src, Zh, goal))

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
        per group); returns (logp [G, M] mixture over samples, valid [G, M]) for up to ``obs_max_tokens`` sampled
        real tokens per group. Candidates are the sampled tokens of all groups; positives are tokens of the same
        group with the same kind and relative time (they are indistinguishable from content-free metadata)."""
        Gn, N, _ = nxt.x.shape
        dev = nxt.x.device
        M = min(self.obs_max_tokens, N)
        # sample up to M real tokens per group (random priority among real tokens)
        pri = torch.rand(Gn, N, generator=gen, device=dev) if gen is not None and gen.device == dev else \
            torch.rand(Gn, N, device=dev)
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
        lp = lp.view(Gn, n_samples, M)
        return logmeanexp(lp, 1), valid

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
             horizons: Optional[Sequence[int]] = None, generator: Optional[torch.Generator] = None) -> dict:
        """Full world-model objective on real recorded sequences.

        beliefs: optional list of T BELIEF TokenSets (B_t computed from information up to t); memory: optional list
        of T MEMORY TokenSets for the filter. Returns a dict with the total ("world" == "loss"), each term, the
        balanced KL and its raw value, per-horizon open-loop errors ("horizon_errors": {h: {...}}) and the detached
        filter states "Z_post" [T,B,S,D] for downstream users."""
        steps = batch.steps
        T, B = len(steps), steps[0].obs.B
        dev = steps[0].obs.x.device
        W = self.weights
        horizons = tuple(sorted(h for h in (horizons or self.train_horizons) if h <= T))
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
        goal = stack_tokens([s.goal for s in steps]) if all(s.goal is not None for s in steps) else None

        def ob_rows(t_idx, b_idx):
            q = gather_tokens(oq, t_idx, b_idx)
            return OutcomeBatch(q, o_lik[t_idx, b_idx], o_nc[t_idx, b_idx], o_y[t_idx, b_idx], o_c[t_idx, b_idx],
                                o_m[t_idx, b_idx], torch.full_like(o_lik[t_idx, b_idx], -1))

        rows = _rows

        # ---- filter (online, with gradients) and EMA target (no gradients)
        Z_post = self.filter(batch, beliefs, memory)                                 # [T,B,S,D]
        with torch.no_grad():
            Z_tgt = self.filter(batch, beliefs, memory, core=self.target_core)
            if self.training:
                self.state_norm.update(Z_post[valid].detach())
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
            z_post = self.sample_latent(post_lp, True, generator)
            Zh1 = self.transition(Zs[:n1], rows(act, t1, b1), rows(bel, t1, b1), dt_r[:n1], z_post)
            terms = self._reduce(self._row_terms(Zh1, 1, t1, b1, ctx, valid_r[:n1], rows(goal, t1, b1),
                                                 self.obs_max_groups))
        else:
            out["kl"] = out["kl_value"] = Z_post.sum() * 0

        # ---- open-loop multistep with prior samples and recorded actions only
        ol, herr = self._open_loop_loss(Z_post, B, act, bel, goal, dt, horizons, ctx)

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
        out["Z_post"] = Z_post.detach()
        return out

    def _row_terms(self, Zh, n_samples, t_idx, b_idx, ctx: "_SeqCtx", alive, goal_rows: Optional[TokenSet],
                   max_groups: int) -> dict:
        """Per-row terms of predicted states Zh [R*n_samples,S,D] for transitions t_idx[r] -> t_idx[r]+1 (each target
        row r repeated n_samples times consecutively; likelihoods are mixtures over the samples). ``alive`` [R]: the
        transition lies inside a valid episode segment. Reduce with ``_reduce``."""
        R = t_idx.shape[0]
        s = n_samples
        dev = Zh.device
        T, obs, Z_tgt, rew, done, valid, gen = ctx.T, ctx.obs, ctx.Z_tgt, ctx.rew, ctx.done, ctx.valid, ctx.gen
        rep = lambda x: x.repeat_interleave(s, 0)
        ob = ctx.ob_rows(t_idx, b_idx)
        obr = OutcomeBatch(rows_rep(ob.q, s), rep(ob.lik), rep(ob.n_classes), rep(ob.y), rep(ob.censored),
                           rep(ob.y_mask), rep(ob.channel))
        d = self.outcome_dist(Zh, obr)
        rt = {"R": R, "alive": alive}
        rt["out_lp"] = logmeanexp(d.log_prob(obr.y, obr.censored).view(R, s, -1), 1)      # [R,Q]
        rt["om"] = ob.y_mask & alive.unsqueeze(-1)
        with torch.no_grad():
            mean = d.mean().view(R, s, -1).mean(1)
            rt["cm"] = rt["om"] & torch.isin(ob.lik, torch.tensor(CONTINUOUS, device=dev))
            rt["sq"] = (mean - ob.y) ** 2
        g_r = None if goal_rows is None else rows_rep(goal_rows, s)
        r_logits = self.reward_logits(Zh, g_r)
        r_t = rew[t_idx, b_idx]
        rt["r_ll"] = logmeanexp(self.twohot.log_lik(r_logits, rep(r_t)).view(R, s), 1)
        with torch.no_grad():
            rt["r_abs"] = (self.twohot.mean(r_logits).view(R, s).mean(1) - r_t).abs()
        c_logit = self.continue_logit(Zh).view(R, s)
        c_tgt = (~done[t_idx, b_idx]).float()
        rt["c_ll"] = logmeanexp(-F.binary_cross_entropy_with_logits(c_logit, c_tgt.unsqueeze(1).expand(-1, s),
                                                                    reduction="none"), 1)
        # latent and observation targets at t+1 (only inside the episode and the sequence)
        tn = (t_idx + 1).clamp(max=T - 1)
        nxt_ok = alive & ((t_idx + 1) <= (T - 1)) & ~done[t_idx, b_idx] & valid[tn, b_idx]
        rt["nxt_ok"] = nxt_ok
        dist = self._latent_dist(Zh, rep(Z_tgt[tn, b_idx]), ctx.tgt_scale).view(R, s)
        tau = self.latent_temperature
        rt["lat"] = -tau * logmeanexp(-dist / tau, 1)
        rt["lat_err"] = dist.detach().mean(1)                                       # standardised MSE (~1 - R^2)
        idx = torch.nonzero(nxt_ok).flatten()
        if idx.numel() > max_groups:
            idx = idx[torch.randperm(idx.numel(), device=dev)[:max_groups]]
        rt["obs_rows"] = idx
        if idx.numel() > 0:
            Zsel = Zh.view(R, s, *Zh.shape[1:])[idx].reshape(idx.numel() * s, *Zh.shape[1:])
            rt["obs_lp"], rt["obs_v"] = self._obs_logp(Zsel, gather_tokens(obs, tn[idx], b_idx[idx]), s, gen)
        return rt

    @staticmethod
    def _reduce(rt: dict, sel: Optional[torch.Tensor] = None) -> dict:
        """Scalar losses / diagnostics over the rows ``sel`` [R] (all rows when None)."""
        if sel is None:
            sel = torch.ones(rt["R"], dtype=torch.bool, device=rt["alive"].device)
        a = rt["alive"] & sel
        om = rt["om"] & sel.unsqueeze(-1)
        nx = rt["nxt_ok"] & sel
        out = {"outcome": -masked_mean(rt["out_lp"], om), "reward": -masked_mean(rt["r_ll"], a),
               "cont": -masked_mean(rt["c_ll"], a), "latent": masked_mean(rt["lat"], nx)}
        cm = rt["cm"] & sel.unsqueeze(-1)
        out["outcome_mse"] = masked_mean(rt["sq"], cm)
        out["outcome_n"] = cm.sum()
        out["reward_mae"] = masked_mean(rt["r_abs"], a)
        out["latent_err"] = masked_mean(rt["lat_err"], nx)
        out["n_rows"] = a.sum()
        if rt["obs_rows"].numel() > 0:
            ov = rt["obs_v"] & sel[rt["obs_rows"]].unsqueeze(-1)
            out["obs"] = -masked_mean(rt["obs_lp"], ov)
            out["obs_logp"] = masked_mean(rt["obs_lp"].detach(), ov)
        else:
            out["obs"] = out["latent"] * 0
            out["obs_logp"] = out["obs"].detach()
        return out

    def _open_loop_loss(self, Z_post, B, act, bel, goal, dt, horizons, ctx: "_SeqCtx"):
        """Roll every start open-loop with prior samples and the recorded actions; the heads of all scored horizons
        are evaluated in one batched call. Returns (mean over horizons of the horizon loss, {h: errors})."""
        dev = Z_post.device
        T, valid, done, gen = ctx.T, ctx.valid, ctx.done, ctx.gen
        rows = _rows
        S_ol = self.open_loop_samples
        z0 = Z_post.sum() * 0
        if not horizons:
            return z0, {}
        starts = torch.arange(T, device=dev)
        if self.open_loop_starts is not None and T > self.open_loop_starts:
            perm = torch.randperm(T, device=dev)[:self.open_loop_starts]
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
        coll = []                                                                    # per scored horizon
        for k in range(1, Hmax + 1):
            n_a = int((starts <= T - k).sum())
            if n_a == 0:
                break
            na_rows = n_a * B
            Z = Z[:na_rows * S_ol]
            st_r, b_r, alive = st_r[:na_rows], b_r[:na_rows], alive[:na_rows]
            if bel_rows is not None:
                bel_rows = _slice_tokens(bel_rows, na_rows * S_ol)
            t_idx = st_r + (k - 1)
            alive = alive & valid[t_idx, b_r]
            a_k = rows_rep(rows(act, t_idx, b_r), S_ol)
            dt_k = dt[t_idx, b_r].repeat_interleave(S_ol)
            lp = self.prior_logits(Z, a_k, bel_rows, dt_k)
            zk = self.sample_latent(lp, True, gen)
            Z = self.transition(Z, a_k, bel_rows, dt_k, zk)
            if k in horizons:
                coll.append((k, Z, t_idx, b_r, alive.clone(), st_r))
            alive = alive & ~done[t_idx, b_r]
        if not coll:
            return z0, {}
        Zall = torch.cat([c[1] for c in coll])
        t_all = torch.cat([c[2] for c in coll])
        b_all = torch.cat([c[3] for c in coll])
        al_all = torch.cat([c[4] for c in coll])
        h_all = torch.cat([torch.full_like(c[2], c[0]) for c in coll])
        # goal of the start step (known at the start; no future information)
        g_all = rows(goal, torch.cat([c[5] for c in coll]), b_all)
        rt = self._row_terms(Zall, S_ol, t_all, b_all, ctx, al_all, g_all, self.obs_max_groups * len(coll))
        losses, herr = [], {}
        W = self.weights
        for k, *_ in coll:
            r = self._reduce(rt, h_all == k)
            losses.append(r["outcome"] + r["reward"] + r["cont"] + W["latent"] * r["latent"] + W["obs"] * r["obs"])
            herr[k] = {"outcome_nll": float(r["outcome"].detach()), "outcome_mse": float(r["outcome_mse"]),
                       "outcome_n": int(r["outcome_n"]), "reward_mae": float(r["reward_mae"]),
                       "cont": float(r["cont"].detach()), "latent_err": float(r["latent_err"]),
                       "obs_logp": float(r["obs_logp"]), "n_rows": int(r["n_rows"])}
        return torch.stack(losses).mean(), herr

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
