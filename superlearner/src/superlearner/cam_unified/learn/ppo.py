"""Goal-conditioned actor-critic over decisions: PPO-clip for fresh data, V-trace for stale/asynchronous data.

Spec 12.3: "Use PPO for fresh on-policy trajectories or an explicitly validated off-policy method such as V-trace for
asynchronous trajectories. Do not apply ordinary PPO ratios to arbitrarily stale replay without validating the
estimator. Record complete behaviour-policy provenance."

`ActorCriticObjective` routes every row of a `Trajectory`:

* **fresh** — policy lag ``current_version - model_version <= max_policy_lag`` on every valid step **and** the
  measured divergence between the behaviour log-probabilities and the current policy on the first pass
  (``mean(log mu - log pi)`` estimator of KL(mu || pi), Schulman's k3) is at most ``max_fresh_kl``. These rows get GAE
  from the behaviour-time values and the PPO-clip surrogate (optional value clipping).
* **stale** — everything else: V-trace targets with the current values, truncated importance weights, and the
  V-trace policy gradient ``-rho_t A^vtrace_t log pi(a_t)``. Stale rows never see a PPO ratio.

`ppo_clip_loss` on its own refuses stale rows (`StalePolicyError`). Gradients of the new log-probabilities come from
the action generator's ``log_prob`` (workstream A), the option module and anything else in the joint action; the
caller supplies them through ``new_logp [B,T]`` (plus ``entropy [B,T]`` and ``new_value [B,T+1]``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor

from .returns import ReturnNormalizer, gae, semi_markov_discount, vtrace
from .trajectory import Trajectory, masked_mean


class StalePolicyError(RuntimeError):
    """Raised when PPO ratios would be applied to data beyond the validated policy lag."""


def approx_kl(behavior_logp: Tensor, new_logp: Tensor, mask: Tensor) -> Tensor:
    """k3 estimator of KL(mu || pi) from samples of mu: E[(r - 1) - log r], r = pi/mu. Non-negative, low variance."""
    log_r = (new_logp - behavior_logp).float()
    k3 = torch.expm1(log_r.clamp(max=20.0)) - log_r
    return masked_mean(k3, mask)


def ppo_clip_loss(new_logp: Tensor, behavior_logp: Tensor, advantages: Tensor, mask: Tensor, clip: float,
                  lag: Optional[Tensor] = None, max_policy_lag: int = 0) -> tuple[Tensor, dict]:
    """PPO-clip surrogate ``-E[min(r A, clip(r, 1-eps, 1+eps) A)]`` over valid steps. Refuses stale data."""
    m = mask.bool()
    if lag is not None and bool(((lag > max_policy_lag) & m).any()):
        raise StalePolicyError(f"PPO ratios refused: policy lag up to {int(lag[m].max())} > {max_policy_lag}; "
                               "route these rows to V-trace")
    log_r = (new_logp - behavior_logp.detach()).float()
    ratio = log_r.clamp(-20.0, 20.0).exp()
    adv = advantages.detach().float()
    s1 = ratio * adv
    s2 = ratio.clamp(1 - clip, 1 + clip) * adv
    loss = -masked_mean(torch.min(s1, s2), m)
    with torch.no_grad():
        clip_frac = masked_mean(((ratio - 1).abs() > clip).float(), m)
    return loss, {"clip_frac": float(clip_frac), "ratio_mean": float(masked_mean(ratio.detach(), m))}


@dataclass
class Prepared:
    """Per-batch quantities computed once before the update epochs."""
    gammas: Tensor                 # [B,T] semi-Markov discounts
    lag: Tensor                    # [B,T] long
    fresh: Tensor                  # [B] bool (lag-gated; refined by the KL check on the first pass)
    adv: Tensor                    # [B,T] GAE advantages (fresh rows; normalised if requested)
    ret: Tensor                    # [B,T] lambda-returns (fresh rows)
    old_value: Tensor              # [B,T+1] behaviour-time values
    kl_checked: bool = False
    info: dict = field(default_factory=dict)


class ActorCriticObjective:
    """Actor-critic losses over a `Trajectory` with PPO (fresh) / V-trace (stale) routing.

    Use: ``prep = obj.prepare(traj, current_version)`` once, then for each epoch recompute ``new_logp [B,T]``,
    ``entropy [B,T]`` and ``new_value [B,T+1]`` (or two-hot ``value_logits [B,T,n_bins]`` with ``value_head``) under
    the current parameters and call ``obj.loss(...)``. Returns a dict with ``actor`` (policy loss minus entropy bonus)
    and ``value`` (critic loss) tensors plus float diagnostics; ``stop`` is True when the KL exceeds
    ``1.5 * target_kl`` (early stopping of the epochs)."""

    def __init__(self, kappa: float = 0.05, gae_lambda: float = 0.95, clip: float = 0.2, entropy_coef: float = 3e-3,
                 max_policy_lag: int = 0, max_fresh_kl: float = 0.05, target_kl: Optional[float] = 0.03,
                 value_clip: Optional[float] = None, rho_clip: float = 1.0, c_clip: float = 1.0,
                 vtrace_lambda: float = 1.0, normalize_advantages: bool = True,
                 return_normalizer: Optional[ReturnNormalizer] = None, adv_rel_floor: float = 0.05):
        self.kappa, self.lam, self.clip, self.entropy_coef = kappa, gae_lambda, clip, entropy_coef
        self.max_policy_lag, self.max_fresh_kl, self.target_kl = max_policy_lag, max_fresh_kl, target_kl
        self.value_clip, self.rho_clip, self.c_clip, self.vtrace_lambda = value_clip, rho_clip, c_clip, vtrace_lambda
        self.normalize_advantages = normalize_advantages
        self.return_normalizer = return_normalizer
        self.adv_rel_floor = adv_rel_floor

    @staticmethod
    def from_config(cfg, **over) -> "ActorCriticObjective":
        kw = dict(kappa=cfg.kappa, gae_lambda=cfg.gae_lambda, clip=cfg.ppo_clip, entropy_coef=cfg.entropy_coef,
                  rho_clip=cfg.vtrace_rho, c_clip=cfg.vtrace_c)
        kw.update(over)
        return ActorCriticObjective(**kw)

    def prepare(self, traj: Trajectory, current_version: int) -> Prepared:
        m = traj.mask.bool()
        gam = semi_markov_discount(torch.where(m, traj.dt, torch.zeros_like(traj.dt)), self.kappa)
        lag = int(current_version) - traj.model_version.long()
        if bool(((lag < 0) & m).any()):
            raise ValueError("model_version newer than the learner's current version")
        fresh = ~((lag > self.max_policy_lag) & m).any(1)
        if traj.value is None:
            raise ValueError("Trajectory.value (behaviour-time values) is required for GAE / value clipping")
        old_v = traj.value.detach().float()
        adv, ret = gae(traj.reward.float(), old_v, gam, traj.continues(), self.lam, m)
        info = {"n_fresh_rows": int(fresh.sum()), "n_stale_rows": int((~fresh).sum()),
                "max_lag": int(lag[m].max()) if bool(m.any()) else 0}
        scale = torch.ones(traj.B, device=traj.device)
        if self.return_normalizer is not None:
            fams = list(traj.family or ["default"] * traj.B)
            for f in sorted(set(fams)):
                rows = torch.tensor([x == f for x in fams], device=traj.device)
                self.return_normalizer.update(ret, f, m & rows.unsqueeze(1))
            scale = torch.tensor([self.return_normalizer.scale(f) for f in fams], device=traj.device)
            info["return_scale"] = float(scale.mean())
        info["_adv_scale"] = scale
        return Prepared(gam, lag, fresh, adv, ret, old_v, False, info)

    def _norm_adv(self, adv: Tensor, m: Tensor, prep: Prepared) -> Tensor:
        """Percentile return scaling per family (DreamerV3: A / max(1, S_family)) when a `ReturnNormalizer` is
        given, else per-batch standardisation."""
        if self.return_normalizer is not None:
            return adv / prep.info["_adv_scale"].unsqueeze(1)
        if self.normalize_advantages and int(m.sum()) > 1:
            mu = masked_mean(adv, m)
            sd = masked_mean((adv - mu) ** 2, m).sqrt()
            # floor relative to the return scale: when the policy has become deterministic every return is equal
            # and the advantages are only value-function error; dividing by their tiny std would turn that noise
            # into full-size (Adam-normalised) policy steps and destroy a converged policy
            floor = self.adv_rel_floor * masked_mean(prep.ret.abs(), m) + 1e-6
            return (adv - mu) / torch.maximum(sd, floor)
        return adv

    def loss(self, traj: Trajectory, prep: Prepared, new_logp: Tensor, entropy: Tensor,
             new_value: Optional[Tensor] = None, value_logits: Optional[Tensor] = None,
             value_head=None) -> dict:
        m = traj.mask.bool()
        B, T = m.shape
        if new_logp.shape != (B, T) or entropy.shape != (B, T):
            raise ValueError("new_logp and entropy must be [B, T]")
        if value_logits is not None:
            if value_head is None:
                raise ValueError("value_logits need the value_head that defines the bins / loss")
            new_value = value_head.value_from_logits(value_logits)
        if new_value is None or new_value.shape != (B, T + 1):
            raise ValueError("new_value must be [B, T+1] (or value_logits [B, T+1, n_bins])")
        beh = traj.behavior_logp.float()
        # ---- validate freshness on the first pass: the claimed-fresh policy must actually match the behaviour policy
        if not prep.kl_checked:
            with torch.no_grad():
                log_r = new_logp.detach().float() - beh
                k3 = (torch.expm1(log_r.clamp(max=20.0)) - log_r) * m
                row_kl = k3.sum(1) / m.sum(1).clamp_min(1)
                too_far = row_kl > self.max_fresh_kl
                rerouted = prep.fresh & too_far
                prep.fresh = prep.fresh & ~too_far
                prep.info["rerouted_by_kl"] = int(rerouted.sum())
                prep.info["fresh_kl_max"] = float(row_kl.max()) if B else 0.0
            prep.kl_checked = True
        fresh_m = m & prep.fresh.unsqueeze(1)
        stale_m = m & ~prep.fresh.unsqueeze(1)
        out: dict = {}
        pol = torch.zeros((), device=m.device)
        val = torch.zeros((), device=m.device)
        n_fresh, n_stale = fresh_m.float().sum(), stale_m.float().sum()
        n_all = (n_fresh + n_stale).clamp_min(1.0)
        # ---- PPO on fresh rows
        if bool(fresh_m.any()):
            adv = self._norm_adv(prep.adv, fresh_m, prep)
            lp, d = ppo_clip_loss(new_logp, beh, adv, fresh_m, self.clip, prep.lag, self.max_policy_lag)
            pol = pol + lp * (n_fresh / n_all)
            out.update(d)
            tgt = prep.ret
            if value_logits is not None:
                vl = value_head.loss(value_logits[:, :T], tgt)
            else:
                v = new_value[:, :T].float()
                vl = (v - tgt) ** 2
                if self.value_clip is not None:
                    v_old = prep.old_value[:, :T]
                    vc = v_old + (v - v_old).clamp(-self.value_clip, self.value_clip)
                    vl = torch.max(vl, (vc - tgt) ** 2)
                vl = 0.5 * vl
            val = val + masked_mean(vl, fresh_m) * (n_fresh / n_all)
            with torch.no_grad():
                out["kl"] = float(approx_kl(beh, new_logp.detach().float(), fresh_m))
        else:
            out["kl"] = 0.0
        # ---- V-trace on stale rows (current values as the baseline; truncated importance weights)
        if bool(stale_m.any()):
            log_ratio = (new_logp.detach().float() - beh)
            vs, pg_adv, rho = vtrace(traj.reward.float(), new_value.detach().float(), prep.gammas, traj.continues(),
                                     log_ratio, self.vtrace_lambda, self.rho_clip, self.c_clip, mask=stale_m)
            pg = -masked_mean(pg_adv.detach() * new_logp.float(), stale_m)
            pol = pol + pg * (n_stale / n_all)
            if value_logits is not None:
                vl = value_head.loss(value_logits[:, :T], vs[:, :T])
            else:
                vl = 0.5 * (new_value[:, :T].float() - vs[:, :T]) ** 2
            val = val + masked_mean(vl, stale_m) * (n_stale / n_all)
            out["vtrace_rho_mean"] = float(masked_mean(rho, stale_m))
        ent = masked_mean(entropy.float(), m)
        out["actor"] = pol - self.entropy_coef * ent
        out["value"] = val
        out["entropy"] = float(ent.detach())
        out["policy_loss"] = float(pol.detach())
        out["value_loss"] = float(val.detach())
        out["n_fresh_steps"] = int(n_fresh)
        out["n_stale_steps"] = int(n_stale)
        out["stop"] = bool(self.target_kl is not None and out["kl"] > 1.5 * self.target_kl)
        out.update({k: v for k, v in prep.info.items() if not k.startswith('_')})
        return out


def explained_variance(pred: Tensor, target: Tensor, mask: Tensor) -> float:
    p, y = pred[mask.bool()].float(), target[mask.bool()].float()
    vy = y.var() if y.numel() > 1 else torch.zeros(())
    return float(1.0 - (y - p).var() / vy) if float(vy) > 1e-8 else 0.0


def run_epochs(objective, traj: Trajectory, current_version: int, recompute: Callable[[Trajectory], dict],
               step_fn: Callable[[dict], object], epochs: int = 4) -> list:
    """The PPO update loop over one batch, for an `ActorCriticObjective` or a `meta.MetaRLObjective`.

    ``recompute(traj)`` re-evaluates the recorded decisions under the current parameters and returns
    {"logp": [B,T], "entropy": [B,T], "value": [B,T+1]} or {"value_logits": [B,T+1,n_bins], "value_head": head};
    ``logp`` is the joint log-probability of the whole recorded action (the action generator's ``log_prob`` plus the
    option decision's `OptionModule.log_prob`). ``step_fn({"actor": ..., "value": ...})`` takes the optimiser step
    (e.g. `CombinedTrainer.step`) and returns its info. Stops early when the KL exceeds 1.5 * target_kl.
    Returns the per-epoch loss dicts (with the step info under "update")."""
    p = objective.prepare(traj, current_version)
    used, prep = p if isinstance(p, tuple) else (traj, p)
    hist = []
    for _ in range(epochs):
        r = recompute(traj)
        d = objective.loss(used, prep, r["logp"], r["entropy"], r.get("value"), r.get("value_logits"),
                           r.get("value_head"))
        d["update"] = step_fn({"actor": d["actor"], "value": d["value"]})
        hist.append(d)
        if d["stop"]:
            break
    return hist


def imagined_actor_critic(imag, logp: Tensor, entropy: Tensor, value: Tensor, dt: Tensor, kappa: float = 0.05,
                          lam: float = 0.95, entropy_coef: float = 3e-3, error_scale: float = 1.0,
                          normalize: bool = True) -> dict:
    """Actor-critic on imagined rollouts (spec 7.3), kept apart from real experience.

    imag: `interfaces.ImaginedTrajectory` (provenance must be IMAGINED; its steps carry predicted ``reward`` [Bb] and
    ``model_error`` [Bb]); logp / entropy [Bb,H]: the policy's log-probability of the contemplated actions,
    recomputed under the current parameters; value [Bb,H+1]: the critic on the imagined states (last column =
    bootstrap at the horizon, e.g. ``imag.returns`` is *not* used so the critic stays the learner's own);
    dt [Bb,H] imagined elapsed days. Each step is weighted by ``w_t = prod_{u<=t} exp(-error_scale * model_error_u)``:
    once predictions become unreliable the imagined future contributes less, i.e. the effective rollout horizon
    shrinks with model error instead of treating simulated outcomes as facts. Targets are lambda-returns of the
    predicted rewards with SMDP discounts. Returns {"actor", "value", "weight_mean", "effective_horizon"}."""
    from ..interfaces import Provenance
    if getattr(imag, "provenance", None) != Provenance.IMAGINED:
        raise ValueError("imagined_actor_critic only accepts IMAGINED trajectories (real data uses PPO / V-trace)")
    H = len(imag.steps)
    if logp.shape[1] != H or value.shape[1] != H + 1:
        raise ValueError("logp [Bb,H], value [Bb,H+1] must match the imagined horizon")
    r = torch.stack([s.reward.float() for s in imag.steps], 1)
    err = torch.stack([s.model_error.float().clamp_min(0) for s in imag.steps], 1)
    w = torch.exp(-error_scale * torch.cumsum(err, 1)).detach()
    gam = semi_markov_discount(dt, kappa)
    ones = torch.ones_like(r)
    from .returns import lambda_returns
    ret = lambda_returns(r, value.detach().float(), gam, ones, lam)
    adv = ret - value[:, :H].detach().float()
    if normalize and adv.numel() > 1:
        mu = (w * adv).sum() / w.sum().clamp_min(1e-8)
        sd = ((w * (adv - mu) ** 2).sum() / w.sum().clamp_min(1e-8)).sqrt()
        adv = (adv - mu) / torch.maximum(sd, 0.05 * ret.abs().mean() + 1e-6)
    wsum = w.sum().clamp_min(1e-8)
    actor = -(w * adv * logp.float()).sum() / wsum - entropy_coef * (w * entropy.float()).sum() / wsum
    val = 0.5 * (w * (value[:, :H].float() - ret) ** 2).sum() / wsum
    return {"actor": actor, "value": val, "weight_mean": float(w.mean()),
            "effective_horizon": float(w.sum(1).mean())}
