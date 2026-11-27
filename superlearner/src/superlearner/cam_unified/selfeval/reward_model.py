"""Observation-only actor reward model R_psi (redesign plan 2026-11-26 §2; owner W8).

    r_t = [Gamma_t * Phi(R_{t+1}, G) - Phi(R_t, G)] + lambda_ig * IG_t - lambda_c * c_t + rho_psi(b_t, e_I, b_{t+1}, e_G, s)

* Phi = `ProspectEvaluator` P(success by horizon | R, G). Trained ONLY on verified labels joined from the trainer-side
  LabelStore (`label_join.join` -> `losses(label_batch)`); inside r it is stop-gradient. Potential shaping by CAM's
  own learned evaluator (Ng et al. 1999 shaping with the SMDP discount Gamma_t = exp(-kappa * dt)).
* IG_t: actor-visible information gain of the belief update (kalman_ig + KL(w_t || w_t-1) from `belief_change`, or
  `signals.belief_kl` as a fallback); c_t: observed resource spend (actor-visible cents), log-normalised.
* rho_psi = `selfeval.intrinsic.IntrinsicReward` (zero-initialised output layer, bounded) over pooled latent states
  before / after, the pooled intent (or action) embedding, the pooled goal tokens and the actor-visible learning
  signals (`ExperienceSignals` names only: unknown keys are refused, so no label / reward / utility key can reach it).
  Running output normalisation is OFF (the bounded tanh output needs none), so the reward the inner update used is
  exactly the reward the meta-gradient differentiates.
* psi = rho_psi's parameters + log lambda_ig + log lambda_c (`meta_parameters()`), owned by the meta optimiser only
  (`metagrad.MetaRewardTrainer`); `version` is bumped on every psi update (EventRecord.extra["versions"]).
  Phi and V_L (`supervised_parameters()`) are trained by `losses()` on joined verified labels only.

At initialisation Phi is constant (zero-initialised heads: Phi = 0.5) and rho = 0, so r = (Gamma_t - 1)/2 +
lambda_ig IG - lambda_c c: well defined and free of any label.

`reward()` is ALWAYS detached (no actor loss reaches psi or Phi; tested: d L_actor / d psi == 0). `reward_psi()` is
the same function with the graph to psi kept (Phi still detached): the inner objective of the meta step.

reward_mode "legacy" (pre-seam configs): no actor reward model exists (the verified increment is the actor reward),
so `RewardModel(cfg)` with cfg.reward_mode == "legacy" or cfg None is the inert null model of the seam (no parameters,
reward = potential = 0, no losses); pass ``enabled=True`` to build the full model anyway.
"""
from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..config import CamConfig
from ..interfaces import TokenSet
from .evaluator import ProspectEvaluator, ProspectTargets
from .intrinsic import N_SIGNALS, ExperienceSignals, IntrinsicReward, LifetimeValue

SIGNAL_NAMES = tuple(ExperienceSignals.names())


def _rows(R, dt=None) -> tuple:
    """(batch size, device) from R [B,...] (tensor) or dt [B]."""
    for t in (R, dt):
        if torch.is_tensor(t) and t.dim() >= 1:
            return int(t.shape[0]), t.device
    if R is not None and hasattr(R, "__len__"):
        return len(R), torch.device("cpu")
    raise ValueError("RewardModel: cannot infer the batch size (pass R [B,...] or dt [B] as tensors)")


def pool_tokens(ts, B: int, D: int, device) -> Optional[Tensor]:
    """Masked mean of a TokenSet [B,N,D] (or a dense [B,N,D] / [B,D] tensor) -> [B,D] float32; None -> None."""
    if ts is None:
        return None
    if isinstance(ts, TokenSet):
        if ts.N == 0:
            return torch.zeros(B, D, device=device)
        m = ts.mask.float()[..., None]
        return ((ts.x.float() * m).sum(1) / m.sum(1).clamp_min(1.0)).to(device)
    t = ts.float()
    return (t.mean(1) if t.dim() == 3 else t).to(device)


def detach_ts(ts: Optional[TokenSet]) -> Optional[TokenSet]:
    if ts is None:
        return None
    return TokenSet(ts.x.detach().float(), ts.mask, ts.kind, ts.time.detach(), ts.entity, ts.refs, ts.entity2)


def _vec(x, B: int, device) -> Optional[Tensor]:
    if x is None:
        return None
    t = torch.as_tensor(x, dtype=torch.float32, device=device)
    if t.dim() == 0:
        t = t.expand(B)
    return t.float().reshape(B)


class RewardModel(nn.Module):
    signal_names = SIGNAL_NAMES      # the only `signals` keys reward() accepts (callers filter by it)
    """RewardModelP. See the module docstring. d_model = width of R (cfg.d_model)."""

    def __init__(self, cfg: Optional[CamConfig] = None, *, enabled: Optional[bool] = None,
                 d_model: Optional[int] = None, d_hidden: int = 256, kappa: Optional[float] = None,
                 lambda_ig: Optional[float] = None, lambda_c: Optional[float] = None, cost_scale: float = 100.0,
                 rho_scale: float = 1.0, use_action: bool = True, n_heads: int = 4):
        super().__init__()
        self.cfg = cfg
        on = enabled if enabled is not None else (cfg is not None and getattr(cfg, "reward_mode", "legacy") != "legacy")
        self.stub = not on
        self.register_buffer("version_buf", torch.zeros((), dtype=torch.long))
        if self.stub:
            return
        D = int(d_model or cfg.d_model)
        self.d_model = D
        self.kappa = float(kappa if kappa is not None else (cfg.kappa if cfg is not None else 0.05))
        lig = float(lambda_ig if lambda_ig is not None else (cfg.lambda_ig if cfg is not None else 0.05))
        lc = float(lambda_c if lambda_c is not None else (cfg.lambda_c if cfg is not None else 0.01))
        self.cost_scale = float(cost_scale)
        # Phi (theta_pred) and V_L: supervised on joined verified labels only
        self.evaluator = ProspectEvaluator(D, n_heads=n_heads)
        self.lifetime_value = LifetimeValue(D, d_hidden)
        # psi: rho_psi + the two weights (positive through log-parameters)
        self.intrinsic = IntrinsicReward(D, d_hidden=d_hidden, scale=rho_scale, n_signals=N_SIGNALS,
                                         use_action=use_action)
        self.log_lambda_ig = nn.Parameter(torch.tensor(math.log(max(lig, 1e-8))))
        self.log_lambda_c = nn.Parameter(torch.tensor(math.log(max(lc, 1e-8))))
        self._target: Optional[ProspectEvaluator] = None

    # ------------------------------------------------------------------------------------------------ versions
    @property
    def version(self) -> int:
        return int(self.version_buf)

    def bump_version(self) -> None:
        self.version_buf += 1
        if not self.stub:
            self.intrinsic.bump_version()

    def meta_parameters(self) -> list:
        """psi: the parameters only the meta (outer) optimiser updates."""
        if self.stub:
            return []
        return list(self.intrinsic.parameters()) + [self.log_lambda_ig, self.log_lambda_c]

    def supervised_parameters(self) -> list:
        """Phi (ProspectEvaluator) and V_L: trained by `losses()` on verified labels."""
        if self.stub:
            return []
        return list(self.evaluator.parameters()) + list(self.lifetime_value.parameters())

    @property
    def lambda_ig(self) -> Tensor:
        return self.log_lambda_ig.exp()

    @property
    def lambda_c(self) -> Tensor:
        return self.log_lambda_c.exp()

    # ------------------------------------------------------------------------------------------------ Phi
    def refresh_potential_target(self) -> None:
        """Freeze a copy of Phi as the potential the actor reward uses until the next refresh (so the actor does not
        chase the evaluator's moving errors between recalibrations); `clear_potential_target()` goes back to live."""
        self._target = self.evaluator.frozen_target()

    def clear_potential_target(self) -> None:
        self._target = None

    def _phi(self, R: Tensor, G, horizon_days=None, policy_bucket=None, grad: bool = False) -> Tensor:
        B = R.shape[0]
        dev = R.device
        ev = self._target if (self._target is not None and not grad) else self.evaluator
        hz = _vec(horizon_days, B, dev) if horizon_days is not None else torch.zeros(B, device=dev)
        pb = torch.as_tensor(policy_bucket, device=dev).long().expand(B) if policy_bucket is not None else \
            torch.zeros(B, dtype=torch.long, device=dev)
        goal = G if isinstance(G, TokenSet) else None
        with torch.set_grad_enabled(grad and torch.is_grad_enabled()):
            return ev(R.float(), goal, hz, pb).p_success

    def potential(self, R, G=None, horizon_days=None, policy_bucket=None) -> Tensor:
        """Phi(R, G) = P(verified success by the horizon | R, G) [B] in [0, 1], detached."""
        B, dev = _rows(R)
        if self.stub:
            return torch.zeros(B, device=dev)
        return self._phi(R.detach(), G, horizon_days, policy_bucket).detach().float()

    # ------------------------------------------------------------------------------------------------ reward
    def _signal_matrix(self, signals, B: int, dev, filled: dict) -> tuple:
        """signals: dict name -> [B] (ExperienceSignals names only) or an ExperienceSignals; `filled`: values the
        model derives itself (cost, duration, prospects, info gain). -> (values [B, n], mask [B, n])."""
        vals = {}
        if isinstance(signals, ExperienceSignals):
            signals = {n: getattr(signals, n) for n in SIGNAL_NAMES}
        for k, v in (signals or {}).items():
            if k not in SIGNAL_NAMES:
                raise ValueError(f"RewardModel: signal {k!r} is not an actor-visible ExperienceSignals field "
                                 f"(allowed: {SIGNAL_NAMES})")
            if v is not None:
                vals[k] = _vec(v, B, dev)
        for k, v in filled.items():
            if v is not None and k not in vals:
                vals[k] = v
        S = torch.zeros(B, N_SIGNALS, device=dev)
        M = torch.zeros(B, N_SIGNALS, device=dev)
        for i, n in enumerate(SIGNAL_NAMES):
            if n in vals:
                S[:, i] = torch.nan_to_num(vals[n].float())
                M[:, i] = torch.isfinite(vals[n].float()).float()
        return S, M

    def reward_terms(self, R, R_next=None, goal=None, dt=None, *, info_gain=None, cost=None, belief=None,
                     belief_next=None, intent=None, signals=None, horizon_days=None, policy_bucket=None,
                     done=None) -> dict:
        """Every term of r_t [B]. The psi terms (ig, cost, rho) keep their graph to psi; Phi is detached; the
        inputs R / R_next / goal / intent are detached (actor-visible features, never trained through r)."""
        B, dev = _rows(R, dt)
        D = self.d_model
        R = R.detach().float()
        Rn = R_next.detach().float() if R_next is not None else None
        G = detach_ts(goal) if isinstance(goal, TokenSet) else None
        dtv = _vec(dt, B, dev) if dt is not None else torch.zeros(B, device=dev)
        gamma = torch.exp(-self.kappa * dtv.clamp_min(0.0))
        with torch.no_grad():
            phi = self._phi(R, G, horizon_days, policy_bucket)
            phi_next = self._phi(Rn, G, horizon_days, policy_bucket) if Rn is not None else None
        if phi_next is None:
            shaping = torch.zeros(B, device=dev)
        else:
            if done is not None:
                phi_next = phi_next * (1.0 - _vec(done, B, dev))     # absorbing end: Phi(terminal) = 0
            shaping = gamma * phi_next - phi
        ig = _vec(info_gain, B, dev)
        if ig is None:
            ig = torch.zeros(B, device=dev)
        c = _vec(cost, B, dev)
        c_hat = torch.log1p(c.clamp_min(0.0) / self.cost_scale) if c is not None else torch.zeros(B, device=dev)
        filled = {"cost": c_hat if c is not None else None, "duration": dtv if dt is not None else None,
                  "prospect_before": phi, "prospect_after": phi_next,
                  "info_gain": ig if info_gain is not None else None}
        S, M = self._signal_matrix(signals, B, dev, filled)
        b_t = R.mean(1)
        b_n = Rn.mean(1) if Rn is not None else b_t
        bel = pool_tokens(belief, B, D, dev)
        if bel is not None:
            b_t = b_t + bel.detach()
            bn = pool_tokens(belief_next, B, D, dev)
            b_n = b_n + (bn.detach() if bn is not None else bel.detach())
        e_i = pool_tokens(intent, B, D, dev)
        e_i = e_i.detach() if e_i is not None else torch.zeros(B, D, device=dev)
        e_g = pool_tokens(G, B, D, dev)
        e_g = e_g.detach() if e_g is not None else None
        rho = self.intrinsic(b_t, e_i, b_n, e_g, S, M)
        ig_term = self.lambda_ig * ig
        cost_term = self.lambda_c * c_hat
        total = shaping + ig_term - cost_term + rho
        return {"total": total, "shaping": shaping, "info_gain": ig_term, "cost": cost_term, "rho": rho,
                "phi": phi, "phi_next": phi_next, "gamma": gamma}

    def reward_psi(self, R, R_next=None, goal=None, dt=None, **kw) -> Tensor:
        """r_t [B] with the graph to psi (inner objective of the meta step); Phi and the inputs are detached."""
        B, dev = _rows(R, dt)
        if self.stub:
            return torch.zeros(B, device=dev)
        return self.reward_terms(R, R_next, goal, dt, **kw)["total"].float()

    def reward(self, R, R_next=None, goal=None, dt=None, *, info_gain=None, cost=None, belief=None,
               belief_next=None, intent=None, signals=None, **kw) -> Tensor:
        """The actor reward r_t [B] (float32), always detached."""
        B, dev = _rows(R, dt)
        if self.stub:
            return torch.zeros(B, device=dev)
        with torch.no_grad():
            return self.reward_terms(R, R_next, goal, dt, info_gain=info_gain, cost=cost, belief=belief,
                                     belief_next=belief_next, intent=intent, signals=signals,
                                     **kw)["total"].detach().float()

    # ------------------------------------------------------------------------------------------------ losses
    def losses(self, label_batch) -> dict:
        """Supervised objectives of Phi and V_L on a joined verified label batch (`label_join.LabelBatch` with
        features attached: R [N,S,D], goal TokenSet | None). Empty / feature-less batches give {}. R is used as given:
        the learner passes gradient-carrying R for the update's own decisions (so L_pred reaches theta_core, plan §2
        core share >= 0.1; W7 change request) and detached R for cached rows."""
        if self.stub or label_batch is None or not hasattr(label_batch, "R") or len(label_batch) == 0:
            return {}
        if label_batch.R is None:
            raise ValueError("RewardModel.losses: the label batch has no features (LabelBatch.with_features)")
        R = label_batch.R.float()
        dev = R.device
        out = self.evaluator(R, label_batch.goal, label_batch.horizon_days.to(dev),
                             torch.zeros(R.shape[0], dtype=torch.long, device=dev))
        L = self.evaluator.losses(out, label_batch.prospect_targets(dev))
        v = self.lifetime_value(R.mean(1), label_batch.lifetime_ctx(dev) if hasattr(label_batch, "lifetime_ctx")
                                else None)
        L_v = self.lifetime_value.loss(v, label_batch.remaining_utility.to(dev), label_batch.weight.to(dev))
        return {"prospect": L["total"], "prospect/success": L["success"], "prospect/time": L["time"],
                "lifetime_value": L_v, "n_resolved": L["n_resolved"]}

    def predict(self, R: Tensor, goal=None, horizon_days=None) -> Tensor:
        """Live (not target) Phi [B], detached: the evaluator's calibration probe."""
        B, dev = R.shape[0], R.device
        hz = _vec(horizon_days, B, dev) if horizon_days is not None else torch.zeros(B, device=dev)
        with torch.no_grad():
            return self.evaluator(R.detach().float(), goal if isinstance(goal, TokenSet) else None, hz,
                                  torch.zeros(B, dtype=torch.long, device=dev)).p_success.float()

    def state_summary(self) -> dict:
        if self.stub:
            return {"stub": True, "version": self.version}
        return {"stub": False, "version": self.version, "lambda_ig": float(self.lambda_ig.detach()),
                "lambda_c": float(self.lambda_c.detach()), "kappa": self.kappa,
                "n_psi": sum(p.numel() for p in self.meta_parameters()),
                "n_supervised": sum(p.numel() for p in self.supervised_parameters())}
