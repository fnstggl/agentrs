"""Bayesian last layer over learned features: local fast adaptation in Kalman form (workstream B; implements
`BayesLastLayerP`). Retained from CAM 7.0 and generalised to arbitrary outcome channels.

Per outcome channel c (an outcome key bound to a slot in `FastState.channels`) and features
phi = Phi_theta(outcome features) [B,Q,F] (F = cfg.bayes_feat, the last feature a constant 1), the residual of the
nonlinear head's prediction is modelled linearly:

    r = y - head_mean = phi^T w_c + eps,   eps ~ N(0, s2_c),   w_c ~ N(mu_c, Sigma_c)

Recursion per observed (channel, value), float64:

    S = s2 + phi^T Sigma phi                  (innovation variance)
    K = Sigma phi / S                         (gain)
    mu    <- mu + K (r - phi^T mu)
    Sigma <- (I - K phi^T) Sigma (I - K phi^T)^T + s2 K K^T      (Joseph form: symmetric PSD by construction)

`noise_mode="fixed"`: s2 is the learned noise (known-noise Kalman filter).
`noise_mode="nig"`: the noise is inferred jointly (normal-inverse-gamma, exact recursion): with Sigma = s2_hat V,
a <- a + 1/2, b <- b + e^2 / (2 (1 + phi^T V phi)), s2_hat = b / a, and Sigma is rescaled to the new s2_hat; the
predictive is Student-t with 2a degrees of freedom and scale^2 = s2_hat + phi^T Sigma phi. FastState stores
bayes_mu = mu, bayes_P = Sigma, bayes_noise = s2_hat, bayes_n = number of updates (a = a0 + n/2).

The prior variance (per feature) and the noise are learned parameters; a channel slot is (re)initialised from them when
a key is first bound. The fast parameters are conditioned on the shared latent representation through phi (computed
from the outcome head's Z-conditioned features) and optionally on the mechanism belief (FiLM from `cond`, e.g. the
belief summary token). This supplements the nonlinear dynamics model; it is not the world model.

Gated fusion (DOM path, plan 2026-11-26 §1 Beliefs "REDESIGN, interim"; `fusion=True`, default for the `dom*`
presets): the Kalman/NIG predictive is no longer substituted for the particle mixture's variance. The runtime
(`core/belief_runtime.py`) forms a two-expert mixture per bound continuous query,

    p(y) = g * p_kal(y) + (1 - g) * p_mix(y),     g = FusionGate(evidence / fit statistics) in (0, 1),

where p_mix is the particle mixture and p_kal the Student-t residual predictive around a base mean that does not
depend on the particles (the particle-free head mean), so the particles keep their own learning signal and the two
residual decompositions (update time / predict time) agree. The gate is learned through the support/query
objective. RBPF extension point: `predictive` returns [B,Q] statistics for one shared posterior per channel;
per-particle Kalman means (bayes_mu [B,K,C,F]) would give [B,K,Q] locations that the runtime mixes over particles
with the particle weights (see `belief_runtime.FusedPredictive`).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ..config import CamConfig
from ..interfaces import FastState, Likelihood, OutcomeBatch
from .blocks import RMSNorm, count_parameters

F64 = torch.float64


class FusionGate(nn.Module):
    """Weight g [B,Q] of the Kalman/NIG expert in the gated fusion, from per-query statistics f [B,Q,N_FEAT]
    (`belief_runtime.fusion_features`): log1p(n_c), log Kalman scale^2, log mixture variance, their difference
    (Kalman minus mixture), the particle filter's recent surprise, log(noise estimate / prior noise), and the Kalman
    expert's realised per-observation log evidence (closed-form NIG marginal likelihood of the channel's past
    residuals) plus the mixture's predictive entropy (> 0: the Kalman expert fitted the past better than the mixture
    expects to fit the next outcome). Initialisation: trust the narrower expert, more so with evidence:
    logit g = -1 + 0.5 log1p(n) - (log scale^2_kal - log var_mix); every weight is learned (L_adapt)."""

    N_FEAT = 7

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(self.N_FEAT, 1)
        with torch.no_grad():
            self.lin.weight.zero_()
            self.lin.weight[0, 0] = 0.5
            self.lin.weight[0, 3] = -1.0
            self.lin.bias.fill_(-1.0)

    def forward(self, f: Tensor) -> Tensor:
        return torch.sigmoid(self.lin(f.float()).squeeze(-1))


class BayesLastLayer(nn.Module):
    """Args:
        cfg: CamConfig (bayes_feat, bayes_prior_var_init, bayes_obs_var_init, d_model).
        d_in: outcome-feature dimension (default d_model); d_cond: optional belief-conditioning dimension.
        noise_mode: "nig" (noise inferred, Student-t predictive) or "fixed" (learned known noise, Gaussian).
        a0: inverse-gamma prior shape for the noise ("nig"); b0 = a0 * noise so the prior plug-in noise is `noise`.
        kinds: likelihood types handled (continuous outcomes by default).
        fusion: gated fusion with the particle mixture (a `FusionGate` parameter set; see the module docstring).
            None = on for the DOM presets (cfg.preset "dom*"), off otherwise (legacy variance replacement).
    """

    def __init__(self, cfg: CamConfig, d_in: Optional[int] = None, d_cond: Optional[int] = None,
                 noise_mode: str = "nig", a0: float = 2.0, hidden: Optional[int] = None,
                 kinds: Sequence[int] = (Likelihood.GAUSSIAN, Likelihood.STUDENT_T), fusion: Optional[bool] = None):
        super().__init__()
        self.fusion = str(getattr(cfg, "preset", "") or "").startswith("dom") if fusion is None else bool(fusion)
        self.gate = FusionGate() if self.fusion else None
        assert noise_mode in ("nig", "fixed"), noise_mode
        self.cfg = cfg
        self.F = cfg.bayes_feat
        d_in = d_in or cfg.d_model
        hidden = hidden or max(64, 2 * self.F)
        self.noise_mode, self.a0 = noise_mode, float(a0)
        self.kinds = tuple(int(k) for k in kinds)
        self.norm = RMSNorm(d_in)
        self.l1 = nn.Linear(d_in, hidden)
        self.l2 = nn.Linear(hidden, self.F - 1)
        self.cond = nn.Linear(d_cond, 2 * hidden) if d_cond else None
        if self.cond is not None:
            nn.init.zeros_(self.cond.weight)
            nn.init.zeros_(self.cond.bias)
        self.log_prior_var = nn.Parameter(torch.full((self.F,), math.log(cfg.bayes_prior_var_init)))
        self.log_noise = nn.Parameter(torch.tensor(math.log(cfg.bayes_obs_var_init)))
        self.stats = {"updates": 0, "channel_overflow": 0}

    # ------------------------------------------------------------------ features
    def features(self, out_feat: Tensor, outcomes: OutcomeBatch, cond: Optional[Tensor] = None) -> Tensor:
        """phi [B,Q,F] = [tanh(Phi_theta(out_feat)); 1]. out_feat [B,Q,D]; cond [B,Dc] optional."""
        h = self.l1(self.norm(out_feat.float()))
        if self.cond is not None and cond is not None:
            s, b = self.cond(cond.float()).chunk(2, -1)
            h = h * (1 + s.unsqueeze(1)) + b.unsqueeze(1)
        phi = torch.tanh(self.l2(torch.nn.functional.silu(h)))
        return torch.cat([phi, torch.ones_like(phi[..., :1])], -1)

    def prior_var(self) -> Tensor:
        return self.log_prior_var.exp()

    def noise(self) -> Tensor:
        return self.log_noise.exp()

    # ------------------------------------------------------------------ channels
    def _suitable(self, outcomes: OutcomeBatch) -> Tensor:
        ok = torch.zeros_like(outcomes.lik, dtype=torch.bool)
        for k in self.kinds:
            ok |= outcomes.lik == k
        return ok

    def channels_for(self, fast: FastState, outcomes: OutcomeBatch) -> Tensor:
        """Bound channel slot per query [B,Q] (-1 = unbound), without allocating. Keys (FastState.channels) take
        precedence; without keys the batch's `channel` field is used for channels already initialised."""
        B, Q = outcomes.lik.shape
        C = fast.bayes_mu.shape[1]
        ok = self._suitable(outcomes)
        ch = torch.full((B, Q), -1, dtype=torch.long)
        if outcomes.keys is not None:
            for b in range(B):
                d = fast.channels[b]
                for q in range(Q):
                    ch[b, q] = d.get(outcomes.keys[b][q], -1)
            ch = ch.to(outcomes.lik.device)
        else:
            given = outcomes.channel.clone()
            init = torch.tensor([[f"#{int(c)}" in fast.channels[b] for c in given[b].clamp(0, C - 1).tolist()]
                                 for b in range(B)], dtype=torch.bool, device=given.device).view(B, Q)
            ch = torch.where((given >= 0) & (given < C) & init, given, torch.full_like(given, -1))
        return torch.where(ok, ch, torch.full_like(ch, -1))

    def assign_channels(self, fast: FastState, outcomes: OutcomeBatch) -> tuple:
        """Bind keys of observed queries to slots (allocating new ones; unobserved queries only use existing bindings).
        Returns (ch [B,Q], channels [B] dict, fresh [B,C] bool)."""
        B, Q = outcomes.lik.shape
        C = fast.bayes_mu.shape[1]
        ok = self._suitable(outcomes).cpu()
        observed = outcomes.y_mask.cpu()
        channels = [dict(c) for c in fast.channels]
        fresh = torch.zeros(B, C, dtype=torch.bool)
        ch = torch.full((B, Q), -1, dtype=torch.long)
        given = outcomes.channel.cpu()
        for b in range(B):
            d = channels[b]
            used = set(d.values())
            for q in range(Q):
                if not bool(ok[b, q]):
                    continue
                if outcomes.keys is not None:
                    key = outcomes.keys[b][q]
                    if key not in d:
                        if not bool(observed[b, q]):
                            continue
                        free = next((c for c in range(C) if c not in used), None)
                        if free is None:
                            self.stats["channel_overflow"] += 1
                            continue
                        d[key] = free
                        used.add(free)
                        fresh[b, free] = True
                    ch[b, q] = d[key]
                else:
                    c = int(given[b, q])
                    if 0 <= c < C:
                        if f"#{c}" not in d:
                            if not bool(observed[b, q]):
                                continue
                            d[f"#{c}"] = c
                            fresh[b, c] = True
                        ch[b, q] = c
        dev = outcomes.lik.device
        return ch.to(dev), channels, fresh.to(dev)

    # ------------------------------------------------------------------ prediction
    def _gather(self, fast: FastState, ch: Tensor) -> tuple:
        B, Q = ch.shape
        cc = ch.clamp(min=0)
        ar = torch.arange(B, device=ch.device).unsqueeze(1).expand(B, Q)
        return fast.bayes_mu[ar, cc], fast.bayes_P[ar, cc], fast.bayes_noise[ar, cc], fast.bayes_n[ar, cc]

    def predictive(self, fast: FastState, phi: Tensor, outcomes: OutcomeBatch) -> dict:
        """{"loc", "scale2", "dof", "bound"} per query [B,Q] (float64 loc/scale2) of the residual r = y - head_mean.
        dof = inf for noise_mode "fixed" (Gaussian)."""
        ch = self.channels_for(fast, outcomes)
        mu, P, s2, n = self._gather(fast, ch)
        f = phi.to(F64)
        loc = (f * mu).sum(-1)
        quad = torch.einsum("bqf,bqfg,bqg->bq", f, P, f)
        scale2 = s2 + quad
        if self.noise_mode == "nig":
            dof = 2 * (self.a0 + 0.5 * n.to(F64))
        else:
            dof = torch.full_like(scale2, float("inf"))
        bound = ch >= 0
        return {"loc": loc, "scale2": scale2, "dof": dof, "bound": bound}

    def channel_stats(self, fast: FastState, outcomes: OutcomeBatch) -> dict:
        """Per-query channel statistics without allocating: {"n": [B,Q] observations on the bound channel,
        "noise": [B,Q] float64 noise estimate s2_hat, "bound": [B,Q] bool} (n = 0, noise = 1 where unbound)."""
        ch = self.channels_for(fast, outcomes)
        _, _, s2, n = self._gather(fast, ch)
        b = ch >= 0
        return {"n": torch.where(b, n.float(), torch.zeros_like(n.float())),
                "noise": torch.where(b, s2, torch.ones_like(s2)), "bound": b}

    def log_evidence(self, fast: FastState, outcomes: OutcomeBatch) -> Tensor:
        """Per-observation log marginal likelihood [B,Q] of the residuals each query's channel has absorbed, in closed
        form from the NIG posterior (`noise_mode="nig"`): log p(r_1..n) = lgamma(a_n) - lgamma(a_0) + a_0 log b_0 -
        a_n log b_n + 0.5 (log|V_n| - log|V_0|) - n/2 log(2 pi), with V = Sigma / s2, b = s2 * a, a_n = a_0 + n/2 and
        the slot prior (diag(prior_var), noise) at the current learned values; divided by n. 0 where unbound, n = 0
        or noise_mode "fixed"."""
        ch = self.channels_for(fast, outcomes)
        _, P, s2, n = self._gather(fast, ch)
        b = (ch >= 0) & (n > 0)
        if self.noise_mode != "nig" or not bool(b.any()):
            return torch.zeros(ch.shape, dtype=torch.float32, device=ch.device)
        nn_ = n.to(F64)
        a0 = torch.full_like(nn_, self.a0)
        an = a0 + 0.5 * nn_
        s20 = self.noise().detach().to(F64)
        pv = self.prior_var().detach().to(F64)
        Fd = P.shape[-1]
        ld0 = torch.log(pv).sum() - Fd * torch.log(s20)
        Pn = P.to(F64)
        L, info = torch.linalg.cholesky_ex(Pn + 1e-12 * torch.eye(Fd, dtype=F64, device=Pn.device))
        ldn = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1).clamp_min(1e-300)).sum(-1) \
            - Fd * torch.log(s2.to(F64).clamp_min(1e-300))
        bn = s2.to(F64) * an
        ev = (torch.lgamma(an) - torch.lgamma(a0) + a0 * torch.log(a0 * s20) - an * torch.log(bn.clamp_min(1e-300))
              + 0.5 * (ldn - ld0) - 0.5 * nn_ * math.log(2 * math.pi))
        ev = ev / nn_.clamp_min(1.0)
        return torch.where(b & (info == 0), ev, torch.zeros_like(ev)).float()

    def predict(self, fast: FastState, phi: Tensor, outcomes: OutcomeBatch) -> tuple:
        """(mean [B,Q], var [B,Q]) of the residual correction r = y - head_mean: mean = phi^T mu_c, var = predictive
        variance (s2 + phi^T Sigma phi; Student-t variance nu/(nu-2) scale^2 in "nig" mode). Unbound queries (no
        channel or non-continuous likelihood): mean 0, var 0 — use `combine` to merge with the head."""
        p = self.predictive(fast, phi, outcomes)
        var = p["scale2"]
        if self.noise_mode == "nig":
            nu = p["dof"]
            var = var * torch.where(nu > 2.05, nu / (nu - 2), torch.full_like(nu, 41.0))
        b = p["bound"]
        mean = torch.where(b, p["loc"], torch.zeros_like(p["loc"]))
        var = torch.where(b, var, torch.zeros_like(var))
        return mean.to(phi.dtype), var.to(phi.dtype)

    @staticmethod
    def combine(head_mean: Tensor, head_var: Tensor, corr_mean: Tensor, corr_var: Tensor, bound: Tensor) -> tuple:
        """Merge with the nonlinear head: bound queries -> (head_mean + corr_mean, corr_var); others -> head."""
        return (torch.where(bound, head_mean + corr_mean, head_mean), torch.where(bound, corr_var, head_var))

    def log_prob(self, fast: FastState, phi: Tensor, outcomes: OutcomeBatch,
                 head_mean: Optional[Tensor] = None) -> Tensor:
        """Predictive log density of the observed residuals [B,Q] (0 where unbound or unobserved). Differentiable in
        phi, head_mean and (through the fast state) the learned prior/noise — the fast-adaptation training signal."""
        p = self.predictive(fast, phi, outcomes)
        r = outcomes.y.to(F64) - (0 if head_mean is None else head_mean.to(F64))
        z2 = (r - p["loc"]) ** 2 / p["scale2"]
        if self.noise_mode == "nig":
            nu = p["dof"]
            lp = (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * math.pi * p["scale2"])
                  - (nu + 1) / 2 * torch.log1p(z2 / nu))
        else:
            lp = -0.5 * (math.log(2 * math.pi) + torch.log(p["scale2"]) + z2)
        use = p["bound"] & outcomes.y_mask
        return torch.where(use, lp, torch.zeros_like(lp)).to(phi.dtype)

    # ------------------------------------------------------------------ update
    def update(self, fast: FastState, phi: Tensor, outcomes: OutcomeBatch,
               head_mean: Optional[Tensor] = None) -> FastState:
        """Kalman / NIG update with every observed (y_mask) query bound to a channel, in query order (several queries
        on the same channel are applied sequentially, exactly). Returns a new FastState (inputs are not modified)."""
        B, Q, Fd = phi.shape
        dev = phi.device
        self.stats["updates"] += 1
        ch, channels, fresh = self.assign_channels(fast, outcomes)
        mu, P, s2, n = fast.bayes_mu, fast.bayes_P, fast.bayes_noise, fast.bayes_n
        if bool(fresh.any()):
            P0 = torch.diag_embed(self.prior_var().to(F64)).expand(B, P.shape[1], Fd, Fd)
            s20 = self.noise().to(F64).expand(B, s2.shape[1])
            mu = torch.where(fresh.unsqueeze(-1), torch.zeros_like(mu), mu)
            P = torch.where(fresh[..., None, None], P0, P)
            s2 = torch.where(fresh, s20, s2)
            n = torch.where(fresh, torch.zeros_like(n), n)
        r_all = outcomes.y.to(F64) - (0 if head_mean is None else head_mean.to(F64))
        ar = torch.arange(B, device=dev)
        eye = torch.eye(Fd, dtype=F64, device=dev)
        obs = outcomes.y_mask
        for q in range(Q):
            act = (ch[:, q] >= 0) & obs[:, q]
            if not bool(act.any()):
                continue
            cc = ch[:, q].clamp(min=0)
            f = phi[:, q].to(F64)                                         # [B,F]
            m_c, P_c, s2_c, n_c = mu[ar, cc], P[ar, cc], s2[ar, cc], n[ar, cc]
            Pf = torch.einsum("bij,bj->bi", P_c, f)
            S = s2_c + (f * Pf).sum(-1)                                   # innovation variance
            K = Pf / S.unsqueeze(-1)                                      # gain
            e = torch.where(act, r_all[:, q], torch.zeros_like(r_all[:, q])) - (f * m_c).sum(-1)
            m_new = m_c + K * e.unsqueeze(-1)
            A = eye - K.unsqueeze(-1) * f.unsqueeze(-2)                   # I - K phi^T
            P_new = A @ P_c @ A.transpose(-1, -2) + s2_c[:, None, None] * K.unsqueeze(-1) * K.unsqueeze(-2)
            s2_new = s2_c
            if self.noise_mode == "nig":
                a = self.a0 + 0.5 * n_c.to(F64)
                b = s2_c * a
                b_new = b + 0.5 * e * e * s2_c / S                       # e^2 / (2 (1 + phi^T V phi))
                s2_new = b_new / (a + 0.5)
                P_new = P_new * (s2_new / s2_c)[:, None, None]
            P_new = 0.5 * (P_new + P_new.transpose(-1, -2))
            a1, a2 = act.unsqueeze(-1), act[:, None, None]
            mu = mu.index_put((ar, cc), torch.where(a1, m_new, m_c))
            P = P.index_put((ar, cc), torch.where(a2, P_new, P_c))
            s2 = s2.index_put((ar, cc), torch.where(act, s2_new, s2_c))
            n = n.index_put((ar, cc), torch.where(act, n_c + 1, n_c))
        return FastState(fast.particles, fast.logw, fast.pscale, mu, P, s2, n, fast.surprise, fast.steps, channels)

    def count_parameters(self) -> int:
        return count_parameters(self)
