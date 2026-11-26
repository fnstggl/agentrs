"""cam_live.predict.imagination — short latent rollouts with uncertainty (spec 4.3).

Imagination works on tensors only: beliefs, action embeddings, elapsed times. It never decodes or produces text.
Everything it returns is an ``Imagined*`` object carrying ``imagined = True``; :func:`refuse_imagined` is called by
every API of this package that accepts outcomes (prediction registry, target builders), so an imagined value cannot
be recorded as something that happened or be used as a training target for outcome heads.

The usable horizon is decided by :class:`HorizonController`: it starts at one step and extends to k steps only
while the *measured* open-loop k-step error on real sequences (:func:`multistep_error`) stays under a threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from ..contracts import Distribution, OutcomeQuery, PredictionRecord, new_id
from .world import LatentTransition


class ImaginedDataError(TypeError):
    """Raised when a model-generated (imagined) value is offered where only real observations are accepted."""


def is_imagined(obj: Any) -> bool:
    if getattr(obj, "imagined", False) is True:
        return True
    if int(getattr(obj, "imagined_horizon_steps", 0) or 0) > 0:
        return True
    prov = getattr(obj, "provenance", None)
    if prov is not None and getattr(prov, "imagined", False) is True:
        return True
    if isinstance(obj, dict):
        return bool(obj.get("imagined")) or int(obj.get("imagined_horizon_steps", 0) or 0) > 0
    return False


def refuse_imagined(obj: Any, what: str = "outcome") -> None:
    if is_imagined(obj):
        raise ImaginedDataError(f"imagined values cannot be used as a real {what}")


@dataclass(frozen=True)
class ImaginedStep:
    step: int                        # 1-based number of imagined transitions from the real belief
    b_mean: Tensor                   # [B, D] mean over particles
    b_std: Tensor                    # [B, D] spread over particles (epistemic + aleatoric, compounded)
    epistemic_std: Tensor            # [B, D] ensemble disagreement of this step's one-step model
    elapsed_s: Tensor                # [B] imagined time since the real belief
    imagined: bool = True


@dataclass(frozen=True)
class ImaginedRollout:
    steps: Tuple[ImaginedStep, ...]
    particles: Tensor                # [P, B, D] final imagined beliefs
    truncated_at: int                # number of steps actually imagined (<= requested, limited by the controller)
    imagined: bool = True

    @property
    def horizon_steps(self) -> int:
        return len(self.steps)


@dataclass(frozen=True)
class ImaginedValues:
    """Model-estimated consequences of alternative first actions. Not outcomes: ``imagined`` is always True."""
    value_mean: Tensor               # [B, N]
    value_std: Tensor                # [B, N] spread over particles
    b_next_mean: Tensor              # [B, N, D]
    b_next_std: Tensor               # [B, N] mean over dimensions of the per-dimension spread
    horizon_steps: int
    imagined: bool = True


def rollout(transition: LatentTransition, decision_latent: Callable[[Tensor, Tensor, Tensor], Tensor],
            b0: Tensor, e_a_seq: Tensor, e_g: Tensor, dt_s: Tensor, *, max_steps: Optional[int] = None,
            n_particles: int = 8, generator: Optional[torch.Generator] = None) -> ImaginedRollout:
    """Imagine a sequence of actions from the real belief ``b0`` [B, D].

    ``e_a_seq`` [B, S, D] embeddings of the action taken at each step; ``dt_s`` [B, S] its duration;
    ``decision_latent(b [R, D], e_a [R, 1, D], e_g [R, D]) -> z [R, 1, D]`` is the model's shared trunk.
    Uncertainty is propagated with particles: each particle follows a randomly drawn ensemble member at every
    step and adds that member's aleatoric noise, so the spread compounds with the horizon.
    """
    bsz, steps_req, d = e_a_seq.shape
    steps_n = steps_req if max_steps is None else max(0, min(steps_req, int(max_steps)))
    p = max(1, n_particles)
    parts = b0.unsqueeze(0).expand(p, bsz, d).reshape(p * bsz, d)
    eg = e_g.unsqueeze(0).expand(p, bsz, d).reshape(p * bsz, d)
    elapsed = torch.zeros(bsz, dtype=torch.float64, device=b0.device)
    out: List[ImaginedStep] = []
    for s in range(steps_n):
        ea = e_a_seq[:, s].unsqueeze(0).expand(p, bsz, d).reshape(p * bsz, 1, d)
        dt = dt_s[:, s].unsqueeze(0).expand(p, bsz).reshape(p * bsz)
        z = decision_latent(parts, ea, eg).squeeze(1)
        pred = transition(z, dt, parts)
        parts = pred.sample(generator)
        elapsed = elapsed + dt_s[:, s].to(torch.float64)
        view = parts.reshape(p, bsz, d)
        out.append(ImaginedStep(step=s + 1, b_mean=view.mean(0),
                                b_std=view.std(0) if p > 1 else torch.zeros_like(view[0]),
                                epistemic_std=pred.epistemic_var.sqrt().reshape(p, bsz, d).mean(0),
                                elapsed_s=elapsed.clone()))
    return ImaginedRollout(tuple(out), parts.reshape(p, bsz, d), steps_n)


def imagine_candidates(transition: LatentTransition, z: Tensor, dt_s: Tensor, b: Tensor,
                       value_fn: Callable[[Tensor], Tensor], *, a_mask: Optional[Tensor] = None,
                       n_particles: int = 8, generator: Optional[torch.Generator] = None) -> ImaginedValues:
    """Compare alternative feasible first actions by one imagined step each.

    ``z`` [B, N, D] decision latents of the candidates, ``dt_s`` [B, N] their durations, ``b`` [B, D] the current
    belief, ``value_fn(b' [R, D]) -> [R]`` any scalar read-out of a belief (for example the evaluator's success
    probability). Returns means and particle spreads per candidate; masked candidates get ``-inf`` value.
    """
    bsz, n, d = z.shape
    p = max(1, n_particles)
    pred = transition(z.reshape(bsz * n, d), dt_s.reshape(bsz * n), b[:, None].expand(bsz, n, d).reshape(bsz * n, d))
    samples = torch.stack([pred.sample(generator) for _ in range(p)])              # [P, B*N, D]
    vals = value_fn(samples.reshape(p * bsz * n, d)).reshape(p, bsz, n)
    v_mean = vals.mean(0)
    v_std = vals.std(0) if p > 1 else torch.zeros_like(v_mean)
    if a_mask is not None:
        v_mean = torch.where(a_mask, v_mean, torch.full_like(v_mean, float("-inf")))
        v_std = torch.where(a_mask, v_std, torch.zeros_like(v_std))
    return ImaginedValues(v_mean, v_std, samples.mean(0).reshape(bsz, n, d),
                          pred.total_std.mean(-1).reshape(bsz, n), horizon_steps=1)


@torch.no_grad()
def multistep_error(transition: LatentTransition, decision_latent: Callable[[Tensor, Tensor, Tensor], Tensor],
                    b_seq: Tensor, e_a_seq: Tensor, e_g: Tensor, dt_s: Tensor, max_k: int) -> Tensor:
    """[max_k] normalised open-loop error of k-step mean predictions on *real* sequences.

    ``b_seq`` [B, T+1, D] real beliefs, ``e_a_seq`` [B, T, D] executed actions, ``dt_s`` [B, T]. Entry k-1 is
    mean ||b_hat_{t+k} - b_{t+k}||^2 / mean ||b_{t+k} - mean b||^2 (1.0 = no better than predicting the mean)."""
    bsz, t1, d = b_seq.shape
    t = t1 - 1
    scale = b_seq.reshape(-1, d).var(0).mean().clamp_min(1e-12)
    errs = torch.zeros(max_k, dtype=torch.float64)
    counts = torch.zeros(max_k, dtype=torch.float64)
    for start in range(t):
        cur = b_seq[:, start]
        for k in range(1, max_k + 1):
            if start + k > t:
                break
            z = decision_latent(cur, e_a_seq[:, start + k - 1:start + k], e_g).squeeze(1)
            cur = transition(z, dt_s[:, start + k - 1], cur).mean
            errs[k - 1] += float(((cur - b_seq[:, start + k]) ** 2).mean() / scale)
            counts[k - 1] += 1
    return torch.where(counts > 0, errs / counts.clamp_min(1), torch.full_like(errs, float("nan")))


class HorizonController:
    """Decides how many imagined steps may be used. Starts at ``min_horizon`` (one step); the horizon k is
    allowed only when every j <= k has at least ``min_count`` measurements and a smoothed measured error at or
    below ``threshold``. It shrinks again as soon as the measured error rises."""

    def __init__(self, max_horizon: int = 8, threshold: float = 0.5, min_count: int = 3, momentum: float = 0.8,
                 min_horizon: int = 1):
        self.max_horizon, self.threshold, self.min_count = max_horizon, threshold, min_count
        self.momentum, self.min_horizon = momentum, min_horizon
        self.err = [float("nan")] * max_horizon
        self.count = [0] * max_horizon

    def update(self, errors: Tensor) -> int:
        for k, e in enumerate(errors.tolist()[:self.max_horizon]):
            if e != e:
                continue
            self.err[k] = e if self.count[k] == 0 else self.momentum * self.err[k] + (1 - self.momentum) * e
            self.count[k] += 1
        return self.allowed()

    def allowed(self) -> int:
        k = 0
        while k < self.max_horizon and self.count[k] >= self.min_count and self.err[k] <= self.threshold:
            k += 1
        return max(self.min_horizon, k)

    def state(self) -> Dict[str, Any]:
        return {"allowed": self.allowed(), "err": list(self.err), "count": list(self.count),
                "threshold": self.threshold}

    def load_state(self, s: Dict[str, Any]) -> None:
        self.err, self.count = list(s["err"]), list(s["count"])
        self.threshold = float(s.get("threshold", self.threshold))


def imagined_prediction(query: OutcomeQuery, dist: Distribution, made_at: float, horizon_steps: int, *,
                        decision_id: str = "", action_id: str = "", model_version: str = "") -> PredictionRecord:
    """A prediction that came out of an imagined rollout. ``imagined_horizon_steps >= 1`` marks it permanently;
    it may be registered and later scored against reality, but it is never itself an outcome."""
    if horizon_steps < 1:
        raise ValueError("an imagined prediction has at least one imagined step")
    return PredictionRecord(prediction_id=new_id("pred"), query=query, dist=dist, made_at=made_at,
                            decision_id=decision_id, action_id=action_id, model_version=model_version,
                            imagined_horizon_steps=int(horizon_steps))
