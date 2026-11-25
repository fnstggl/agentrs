"""Training targets from imagination (spec 7.3), for workstream D's trainer.

Imagined trajectories (``Provenance.IMAGINED``) yield actor-critic targets that are kept explicitly separate from
real experience:

* lambda-returns with semi-Markov discounts gamma_k = exp(-kappa dt_k) and predicted continuation c_k:
  G_k = r_k + gamma_k c_k [(1 - lam) V_{k+1} + lam G_{k+1}], bootstrapped with V at the (effective) horizon;
* uncertainty-aware weights w_k = exp(-beta * sum_{j<=k} e_j) from the per-step model-error estimates e_j;
* horizon shortening: a row's rollout is cut at the first step whose model error exceeds ``error_threshold`` (the
  target bootstraps from the value there instead of trusting later imagined steps); ``select_reliable_horizon``
  shortens the rollout horizon globally when validated open-loop errors on real data are not better than a baseline;
* ``combine_real_and_imagined`` refuses imagined-only updates and keeps both loss families separately logged, so
  imagination can never replace correction from actual experience.

Indexing: step k applies action a_k in imagined state s_k (s_0 = the real state) and reaches s_{k+1};
``traj.steps[k].value`` is V(s_{k+1}); V(s_0) is passed as ``value0`` when known.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from ..interfaces import ImaginedTrajectory, Provenance


@dataclass
class ImaginedTargets:
    returns: torch.Tensor        # [H,Bb] lambda-returns G_k (targets for V(s_k))
    values: torch.Tensor         # [H+1,Bb] V(s_0..s_H) used for bootstrapping
    advantages: torch.Tensor     # [H,Bb] G_k - V(s_k)
    weights: torch.Tensor        # [H,Bb] uncertainty weight x validity mask
    mask: torch.Tensor           # [H,Bb] steps inside the effective horizon
    horizon: torch.Tensor        # [Bb] effective horizon (number of trusted steps)
    discounts: torch.Tensor      # [H,Bb] gamma_k * c_k
    provenance: Provenance = Provenance.IMAGINED
    is_imagined: bool = True


def imagined_targets(traj: ImaginedTrajectory, lam: float = 0.95, kappa: float = 0.05,
                     value0: Optional[torch.Tensor] = None, uncertainty_beta: float = 1.0,
                     error_threshold: Optional[float] = None, min_horizon: int = 1,
                     use_continue: bool = True) -> ImaginedTargets:
    """lambda-return targets with uncertainty weighting and per-row horizon shortening (see module doc)."""
    if traj.provenance != Provenance.IMAGINED:
        raise ValueError("imagined_targets only accepts imagined trajectories")
    steps = traj.steps
    H = len(steps)
    r = torch.stack([s.reward.float() for s in steps])                               # [H,Bb]
    Bb = r.shape[1]
    dev = r.device
    dt = torch.stack([s.dt.float() if getattr(s, "dt", None) is not None else torch.zeros(Bb, device=dev)
                      for s in steps])
    cont = torch.stack([s.cont.float() if (use_continue and getattr(s, "cont", None) is not None)
                        else torch.ones(Bb, device=dev) for s in steps])
    err = torch.stack([s.model_error.float() for s in steps])
    v = torch.stack([s.value.float() for s in steps])                                 # V(s_1..s_H)
    v0 = torch.zeros(Bb, device=dev) if value0 is None else value0.float().reshape(Bb).to(dev)
    values = torch.cat([v0.unsqueeze(0), v], 0)                                       # [H+1,Bb]
    disc = torch.exp(-kappa * dt) * cont
    # effective horizon: cut at the first unreliable step (that step itself is kept, later ones are not)
    L = torch.full((Bb,), H, dtype=torch.long, device=dev)
    if error_threshold is not None:
        bad = err > error_threshold
        first_bad = torch.where(bad.any(0), bad.float().argmax(0), torch.full_like(L, H))
        L = (first_bad + 1).clamp(min=min_horizon, max=H)
    k_idx = torch.arange(H, device=dev).unsqueeze(1)
    mask = k_idx < L.unsqueeze(0)                                                     # [H,Bb]
    G = torch.zeros(H, Bb, device=dev)
    nxt = values[H]
    for k in reversed(range(H)):
        last = (L - 1) == k
        boot = torch.where(last, values[k + 1], (1 - lam) * values[k + 1] + lam * nxt)
        g = r[k] + disc[k] * boot
        G[k] = torch.where(mask[k], g, values[k])
        nxt = G[k]
    w = torch.exp(-uncertainty_beta * torch.cumsum(err.clamp(min=0) * mask, 0)) * mask
    return ImaginedTargets(G, values, G - values[:H], w, mask, L, disc)


def imagination_actor_critic_loss(targets: ImaginedTargets, logp: torch.Tensor, values_pred: torch.Tensor,
                                  entropy: Optional[torch.Tensor] = None, entropy_coef: float = 3e-3,
                                  value_coef: float = 0.5, normalize_advantages: bool = True,
                                  step_mask: Optional[torch.Tensor] = None) -> dict:
    """REINFORCE-with-baseline actor loss and weighted value regression on imagined targets.

    logp [H,Bb]: log pi(a_k | s_k) of the imagined actions (recomputed with gradients by the caller);
    values_pred [H,Bb]: V_theta(s_k) with gradients; ``step_mask`` [H,Bb] optionally excludes steps whose actions
    were not sampled from the policy (e.g. enumerated first candidates). Weighted by ``targets.weights``."""
    if not targets.is_imagined or targets.provenance != Provenance.IMAGINED:
        raise ValueError("expected imagined targets")
    w = targets.weights.detach()
    if step_mask is not None:
        w = w * step_mask.float()
    wsum = w.sum().clamp(min=1e-8)
    adv = targets.advantages.detach()
    if normalize_advantages:
        mu = (w * adv).sum() / wsum
        sd = torch.sqrt((w * (adv - mu) ** 2).sum() / wsum + 1e-8)
        adv = (adv - mu) / sd
    actor = -(w * adv * logp).sum() / wsum
    critic = (w * (values_pred - targets.returns.detach()) ** 2).sum() / wsum
    ent = (w * entropy).sum() / wsum if entropy is not None else torch.zeros((), device=w.device)
    loss = actor + value_coef * critic - entropy_coef * ent
    return {"loss": loss, "actor": actor, "critic": critic, "entropy": ent, "weight_sum": wsum.detach(),
            "provenance": Provenance.IMAGINED.value}


def combine_real_and_imagined(real: Optional[dict], imagined: Optional[dict], imagined_weight: float = 0.5) -> dict:
    """Total = real["loss"] + imagined_weight * imagined["loss"], both kept under separate keys.

    Imagined-only updates are refused: imagined consequences are never treated as facts without continual
    correction from actual environment experience."""
    if imagined is not None and imagined.get("provenance") != Provenance.IMAGINED.value:
        raise ValueError("imagined losses must carry provenance 'imagined'")
    if real is None:
        if imagined is not None:
            raise ValueError("refusing an imagined-only update: real experience is required in every update")
        raise ValueError("nothing to combine")
    if real.get("provenance", Provenance.REAL.value) != Provenance.REAL.value:
        raise ValueError("real losses must come from real experience")
    out = {"loss": real["loss"], "real/loss": real["loss"].detach()}
    for k, v in real.items():
        if k not in ("loss", "provenance") and torch.is_tensor(v):
            out[f"real/{k}"] = v.detach()
    if imagined is not None:
        out["loss"] = out["loss"] + imagined_weight * imagined["loss"]
        out["imagined/loss"] = imagined["loss"].detach()
        for k, v in imagined.items():
            if k not in ("loss", "provenance") and torch.is_tensor(v):
                out[f"imagined/{k}"] = v.detach()
        out["imagined_weight"] = imagined_weight
    return out


def select_reliable_horizon(model_error: dict, baseline_error: dict, horizons: Sequence[int],
                            margin: float = 1.0, min_horizon: int = 1) -> int:
    """Largest horizon h (in ``horizons``) such that the world model's validated open-loop error is below
    ``margin`` x the baseline error at every horizon <= h. ``model_error`` / ``baseline_error``: {h: error} from
    real held-out data (e.g. ``evaluate_multistep``). Falls back to ``min_horizon``."""
    best = min_horizon
    for h in sorted(horizons):
        if h not in model_error or h not in baseline_error:
            break
        if not (model_error[h] < margin * baseline_error[h]) or not math.isfinite(model_error[h]):
            break
        best = h
    return best
