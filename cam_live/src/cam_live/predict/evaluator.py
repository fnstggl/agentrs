"""cam_live.predict.evaluator — learned goal-conditioned self-evaluation  V(b, g, horizon, policy_ctx)  (spec Part 6).

``GoalEvaluator`` reads CAM's shared latent, not text, and predicts for an objective:

    success      logit of P(objective verified as achieved within the horizon)        BCE-with-logits
    magnitude    size of the eventual achievement: negative binomial (count) or Student-t (real)
    time         time to achievement: discrete hazard on K bins of the remaining horizon, right-censored
    resources    remaining expenditure (seconds, money, compute): normal on log1p(amount / scale)

``forward`` evaluates a belief (V); ``forward_candidates`` evaluates each candidate's decision latent (Q), which is
what separates the *quality of a decision* from the *outcome that happened to follow it* (:func:`action_quality`,
:func:`outcome_decomposition`).

Supervision comes only from verified terminal outcomes and is applied to every earlier prefix of the trajectory
(:func:`prefix_targets`); a pending objective produces masks that are all False. :func:`delta_v` exposes the change
in predicted prospects as a *signal* for the intrinsic-reward generator. It is not a reward and it is not the
causal contribution of the action.

Derived from the structure of ``superlearner/cam_unified/selfeval/evaluator.py`` (agentrs commit 3968583);
re-written for the CAM Live tensor contract (no token sets, horizon in seconds, resource head, Q read-out).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..contracts import GoalStatus, VerifiedOutcome
from . import dists as D
from .world import time_features


class UnverifiedOutcomeError(ValueError):
    """Raised when an objective outcome whose verifier signature was not checked is offered as supervision."""


@dataclass
class EvaluatorConfig:
    d_model: int
    d_ctx: int = 0                   # width of the policy-context features (0 = none)
    hidden: int = 192
    n_time_bins: int = 12
    n_resources: int = 3             # by convention: seconds, money, compute seconds (contracts.Cost)
    d_time: int = 16
    out_std: float = 0.05            # small random output init (zero-initialised outputs blocked learning before)


class EvalOutput:
    """Evaluator read-outs; every tensor has the leading shape of the input belief / candidate set."""

    def __init__(self, raw: Tensor, cfg: EvaluatorConfig, horizon_s: Tensor, scale: Tensor):
        k, r = cfg.n_time_bins, cfg.n_resources
        self.cfg, self.raw, self.horizon_s, self.scale = cfg, raw, horizon_s, scale
        self.success_logit = raw[..., 0]
        self.mag_log_rate = raw[..., 1].clamp(-20.0, 20.0)
        self.mag_log_conc = (raw[..., 2] + 2.0).clamp(-6.0, 9.0)
        self.mag_loc = raw[..., 3]
        self.mag_scale = F.softplus(raw[..., 4] + 0.5413) + 1e-3
        self.mag_df = 2.0 + F.softplus(raw[..., 5] + 3.0)
        self.time_hazard_logits = raw[..., 6:6 + k] - 2.0
        self.res_loc = raw[..., 6 + k:6 + k + r]
        self.res_scale = F.softplus(raw[..., 6 + k + r:6 + k + 2 * r] + 0.5413) + 1e-3

    @property
    def p_success(self) -> Tensor:
        return torch.sigmoid(self.success_logit)

    def time_edges(self) -> Tensor:
        return D.uniform_edges(self.horizon_s.to(self.raw.dtype).clamp_min(1e-6), self.cfg.n_time_bins)

    def time_dist(self) -> D.DiscreteHazard:
        return D.DiscreteHazard(self.time_hazard_logits, self.time_edges())

    def p_achieved_in_time(self) -> Tensor:
        """P(achievement within the horizon) according to the hazard head (a second, censoring-aware estimate)."""
        return 1.0 - D.hazard_survival(self.time_hazard_logits)[..., -1]

    def expected_time(self) -> Tensor:
        return D.hazard_expected_time(self.time_hazard_logits, self.time_edges())

    def magnitude_count(self) -> D.NegBinomial:
        return D.NegBinomial(torch.exp(self.mag_log_rate), torch.exp(self.mag_log_conc), self.scale)

    def magnitude_real(self) -> D.StudentT:
        return D.StudentT(self.mag_loc * self.scale, self.mag_scale * self.scale, self.mag_df)

    def resources_expected(self, res_scale: Optional[Tensor] = None) -> Tensor:
        """Median remaining expenditure per resource (the head is normal on log1p(amount / scale))."""
        s = 1.0 if res_scale is None else res_scale
        return torch.expm1(self.res_loc).clamp_min(0.0) * s


@dataclass
class EvalTargets:
    """Targets for a batch of prefixes (leading shape [...]). A False mask means "no target": the entry
    contributes exactly zero loss and zero gradient."""
    success: Tensor                  # [...] 1 achieved / 0 definitively failed
    success_mask: Tensor             # [...] bool
    magnitude: Tensor                # [...]
    magnitude_mask: Tensor           # [...] bool
    magnitude_is_count: Tensor       # [...] bool
    time: Tensor                     # [...] seconds from the prefix to achievement (event) or to censoring
    time_event: Tensor               # [...] bool
    time_mask: Tensor                # [...] bool
    resources: Tensor                # [..., R] remaining expenditure after the prefix
    resources_mask: Tensor           # [..., R] bool
    weight: Optional[Tensor] = None  # [...] row weights (prefixes of one trajectory share one unit of weight)

    _FIELDS = ("success", "success_mask", "magnitude", "magnitude_mask", "magnitude_is_count", "time",
               "time_event", "time_mask", "resources", "resources_mask", "weight")

    @classmethod
    def _combine(cls, items: Sequence["EvalTargets"], fn) -> "EvalTargets":
        def one(name):
            vals = [torch.ones_like(i.success) if getattr(i, name) is None else getattr(i, name) for i in items]
            return fn(vals, 0)
        return cls(*(one(n) for n in cls._FIELDS))

    @classmethod
    def cat(cls, items: Sequence["EvalTargets"]) -> "EvalTargets":
        """Concatenate along the leading axis (rows of different trajectories)."""
        return cls._combine(items, torch.cat)

    @classmethod
    def stack(cls, items: Sequence["EvalTargets"]) -> "EvalTargets":
        """Stack equally long trajectories into [B, T, ...] (matches ``GoalEvaluator.forward`` on b [B, T, D])."""
        return cls._combine(items, torch.stack)

    def index(self, idx: Tensor) -> "EvalTargets":
        """Select along the leading axis."""
        return EvalTargets(*(None if getattr(self, n) is None else getattr(self, n)[idx] for n in self._FIELDS))


class GoalEvaluator(nn.Module):
    def __init__(self, cfg: EvaluatorConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.n_out = 6 + cfg.n_time_bins + 2 * cfg.n_resources
        self.in_b = nn.Linear(cfg.d_model, h)                 # belief  -> V
        self.in_z = nn.Linear(cfg.d_model, h)                 # decision latent of a candidate -> Q
        self.in_g = nn.Linear(cfg.d_model, h)
        self.in_xg = nn.Linear(cfg.d_model, h, bias=False)    # multiplicative belief-goal interaction
        self.in_gg = nn.Linear(cfg.d_model, h, bias=False)
        self.in_t = nn.Linear(cfg.d_time, h)
        self.in_c = nn.Linear(cfg.d_ctx, h) if cfg.d_ctx > 0 else None
        self.norm = nn.LayerNorm(h)
        self.mid = nn.Linear(h, h)
        self.out = nn.Linear(h, self.n_out)
        nn.init.normal_(self.out.weight, std=cfg.out_std / math.sqrt(h))
        nn.init.zeros_(self.out.bias)

    def _run(self, x_in: Tensor, x_raw: Tensor, e_g: Tensor, horizon_s: Tensor, policy_ctx: Optional[Tensor],
             scale: Optional[Tensor]) -> EvalOutput:
        lead = x_in.shape[:-1]
        extra = len(lead) - (e_g.dim() - 1)                   # candidate / prefix axes between batch and features
        g = e_g.reshape(e_g.shape[:-1] + (1,) * extra + e_g.shape[-1:])
        hz = horizon_s if horizon_s.dim() == len(lead) else horizon_s.reshape(horizon_s.shape + (1,) * (
            len(lead) - horizon_s.dim()))
        hz = hz.expand(lead)
        x = (x_in + self.in_g(g) + self.in_xg(x_raw) * self.in_gg(g) / math.sqrt(self.cfg.hidden)
             + self.in_t(time_features(hz, self.cfg.d_time).to(x_in.dtype)))
        if self.in_c is not None:
            if policy_ctx is None:
                raise ValueError("this GoalEvaluator was built with d_ctx > 0 and needs policy_ctx")
            c = policy_ctx.reshape(policy_ctx.shape[:-1] + (1,) * (len(lead) - (policy_ctx.dim() - 1))
                                   + policy_ctx.shape[-1:])
            x = x + self.in_c(c)
        raw = self.out(F.silu(self.mid(F.silu(self.norm(x)))))
        sc = torch.ones(lead, device=raw.device, dtype=raw.dtype) if scale is None else (
            scale.reshape(scale.shape + (1,) * (len(lead) - scale.dim())).expand(lead).to(raw.dtype))
        return EvalOutput(raw, self.cfg, hz, sc)

    def forward(self, b: Tensor, e_g: Tensor, horizon_s: Tensor, policy_ctx: Optional[Tensor] = None,
                scale: Optional[Tensor] = None) -> EvalOutput:
        """V. b [B, D] (or [B, T, D] for T prefixes), e_g [B, D], horizon_s [B] (or [B, T]) seconds remaining,
        policy_ctx [B, d_ctx] (or [B, T, d_ctx]), scale [B] natural size of the magnitude (e.g. its target)."""
        return self._run(self.in_b(b), b, e_g, horizon_s, policy_ctx, scale)

    def forward_candidates(self, z: Tensor, e_g: Tensor, horizon_s: Tensor, policy_ctx: Optional[Tensor] = None,
                           scale: Optional[Tensor] = None) -> EvalOutput:
        """Q. z [B, N, D] decision latents (belief + candidate action + goal); other arguments as ``forward``."""
        return self._run(self.in_z(z), z, e_g, horizon_s, policy_ctx, scale)


def evaluator_loss(out: EvalOutput, tgt: EvalTargets, weights: Optional[Dict[str, float]] = None,
                   res_scale: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, float]]:
    """Masked sum of the four likelihood terms. Success uses numerically stable BCE-with-logits."""
    wts = {"success": 1.0, "magnitude": 0.25, "time": 0.25, "resources": 0.25}
    wts.update(weights or {})
    dt = out.raw.dtype
    row_w = torch.ones_like(out.success_logit) if tgt.weight is None else tgt.weight.to(dt)

    def wmean(v: Tensor, mask: Tensor, w: Tensor) -> Tensor:
        mw = torch.where(mask, w, torch.zeros_like(w))
        return (torch.where(mask, v, torch.zeros_like(v)) * mw).sum() / mw.sum().clamp_min(1e-12)

    metrics: Dict[str, float] = {}
    # success
    y = D.safe_targets(tgt.success.to(dt), tgt.success_mask)
    bce = F.binary_cross_entropy_with_logits(out.success_logit, y, reduction="none")
    l_s = wmean(bce, tgt.success_mask, row_w)
    # magnitude
    mag = D.safe_targets(tgt.magnitude.to(dt), tgt.magnitude_mask)
    is_count = tgt.magnitude_is_count
    y_cnt = torch.where(is_count, mag, torch.zeros_like(mag)).clamp_min(0.0)
    nll_cnt = -D.neg_binomial_log_prob_from_log(y_cnt, out.mag_log_rate + torch.log(out.scale.clamp_min(1e-12)),
                                                out.mag_log_conc)
    y_real = torch.where(is_count, torch.zeros_like(mag), mag) / out.scale
    nll_real = -D.student_t_log_prob(y_real, out.mag_loc, out.mag_scale, out.mag_df)
    l_m = wmean(torch.where(is_count, nll_cnt, nll_real), tgt.magnitude_mask, row_w)
    # time to achievement, censoring-aware
    t = D.safe_targets(tgt.time.to(dt), tgt.time_mask).clamp_min(0.0)
    nll_t = -D.hazard_log_prob(t, tgt.time_event, out.time_hazard_logits, out.time_edges())
    l_t = wmean(nll_t, tgt.time_mask, row_w)
    # resources: normal on log1p(amount / scale)
    rs = torch.ones(out.cfg.n_resources, device=out.raw.device, dtype=dt) if res_scale is None else res_scale.to(dt)
    r = torch.log1p(D.safe_targets(tgt.resources.to(dt), tgt.resources_mask).clamp_min(0.0) / rs)
    nll_r = -D.normal_log_prob(r, out.res_loc, out.res_scale)
    l_r = wmean(nll_r, tgt.resources_mask, row_w.unsqueeze(-1).expand_as(nll_r))
    loss = wts["success"] * l_s + wts["magnitude"] * l_m + wts["time"] * l_t + wts["resources"] * l_r
    for name, l, m in (("success", l_s, tgt.success_mask), ("magnitude", l_m, tgt.magnitude_mask),
                       ("time", l_t, tgt.time_mask), ("resources", l_r, tgt.resources_mask)):
        metrics[f"eval_{name}_nll"] = float(l.detach())
        metrics[f"eval_{name}_n"] = float(m.sum())
    metrics["eval_loss"] = float(loss.detach())
    return loss, metrics


def prefix_targets(prefix_ts: Sequence[float], *, status: str, evaluated_at: float, deadline: float,
                   signature_checked: bool, value: float = 0.0, achieved_at: Optional[float] = None,
                   magnitude_is_count: bool = True, resources_final: Optional[Sequence[float]] = None,
                   resources_at_prefix: Optional[Sequence[Sequence[float]]] = None, n_resources: int = 3
                   ) -> Tuple[Tensor, EvalTargets]:
    """Apply one eventual *verified* objective outcome to every earlier prefix of its trajectory.

    ``prefix_ts`` are the times at which the evaluator's inputs (beliefs) were formed. Returns
    ``(horizon_s [T], targets)`` where ``horizon_s = deadline - t`` is the time each prefix had left.

    * ``signature_checked`` must be True: the caller has verified the verifier's signature. Unsigned outcomes are
      refused, because only independently verified outcomes may supervise the evaluator.
    * A pending objective yields masks that are all False (pending stays pending; nothing is invented).
    * A prefix formed at or after ``evaluated_at`` is masked: the evaluator is only trained to predict what was
      still in its future.
    * Achieved: success 1, time = achieved_at - t as an event. Failed: success 0, and the time-to-achievement is
      right-censored at the prefix's remaining horizon.
    * Each valid prefix gets weight 1 / (number of valid prefixes): one trajectory is one unit of evidence.
    """
    if not signature_checked:
        raise UnverifiedOutcomeError("objective outcomes must be signature-checked before they supervise anything")
    if status not in (GoalStatus.PENDING, GoalStatus.ACHIEVED, GoalStatus.FAILED):
        raise ValueError(f"unknown objective status {status!r}")
    ts = torch.as_tensor(list(prefix_ts), dtype=torch.float64)
    n = len(ts)
    horizon = (deadline - ts).clamp_min(0.0)
    resolved = status in (GoalStatus.ACHIEVED, GoalStatus.FAILED)
    valid = (ts < evaluated_at) & resolved
    achieved = status == GoalStatus.ACHIEVED
    t_ach = evaluated_at if achieved_at is None else achieved_at
    time = (torch.full_like(ts, t_ach) - ts).clamp_min(0.0) if achieved else horizon.clone()
    res = torch.zeros(n, n_resources)
    res_mask = torch.zeros(n, n_resources, dtype=torch.bool)
    if resources_final is not None and resources_at_prefix is not None:
        fin = torch.as_tensor(list(resources_final), dtype=torch.float64)
        at = torch.as_tensor([list(r) for r in resources_at_prefix], dtype=torch.float64).reshape(n, -1)
        k = min(n_resources, fin.numel(), at.shape[1])
        res[:, :k] = (fin[:k] - at[:, :k]).clamp_min(0.0).float()
        res_mask[:, :k] = valid[:, None]
    weight = torch.where(valid, torch.full((n,), 1.0 / max(1, int(valid.sum()))), torch.zeros(n))
    tgt = EvalTargets(
        success=torch.full((n,), 1.0 if achieved else 0.0), success_mask=valid.clone(),
        magnitude=torch.full((n,), float(value)), magnitude_mask=valid.clone(),
        magnitude_is_count=torch.full((n,), bool(magnitude_is_count)),
        time=time.float(), time_event=torch.full((n,), bool(achieved)), time_mask=valid.clone(),
        resources=res, resources_mask=res_mask, weight=weight)
    return horizon, tgt


def prefix_targets_from_outcome(outcome: VerifiedOutcome, prefix_ts: Sequence[float], *, deadline: float,
                                signature_checked: bool, achieved_at: Optional[float] = None,
                                magnitude_is_count: bool = True, **kw) -> Tuple[Tensor, EvalTargets]:
    if not outcome.signature:
        raise UnverifiedOutcomeError("the outcome carries no verifier signature")
    return prefix_targets(prefix_ts, status=outcome.status, evaluated_at=outcome.evaluated_at, deadline=deadline,
                          signature_checked=signature_checked, value=outcome.value, achieved_at=achieved_at,
                          magnitude_is_count=magnitude_is_count, **kw)


def delta_v(evaluator: GoalEvaluator, b_prev: Tensor, b_next: Tensor, e_g: Tensor, horizon_prev: Tensor,
            horizon_next: Tensor, policy_ctx: Optional[Tensor] = None, detach: bool = True) -> Dict[str, Tensor]:
    """Changed predicted prospects  dV = V(b_next, g) - V(b_prev, g).

    A *signal* (an input of the intrinsic-reward generator), explicitly not a reward: it is a difference of the
    evaluator's own predictions, it can be wrong, and it is not the causal contribution of the action taken in
    between. Returned detached by default so nothing downstream can train the evaluator through it."""
    prev = evaluator(b_prev, e_g, horizon_prev, policy_ctx)
    nxt = evaluator(b_next, e_g, horizon_next, policy_ctx)
    out = {"p_prev": prev.p_success, "p_next": nxt.p_success, "dv_prob": nxt.p_success - prev.p_success,
           "dv_logit": nxt.success_logit - prev.success_logit}
    return {k: v.detach() for k, v in out.items()} if detach else out


def action_quality(q: Tensor, probs: Tensor, chosen: Tensor, a_mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
    """Quality of a decision under the model, separated from what happened afterwards.

    ``q`` [B, N] model estimate for each candidate (e.g. ``forward_candidates(...).p_success``), ``probs`` [B, N]
    the policy's probabilities for the same candidates, ``chosen`` [B].

        baseline = sum_i pi_i q_i           value of following the policy over the candidate set
        quality  = q[chosen] - baseline     advantage of the chosen action (a good decision has quality > 0)
        regret   = max_i q_i - q[chosen]    how much better the best feasible candidate looked
    """
    if a_mask is None:
        a_mask = torch.ones_like(q, dtype=torch.bool)
    qm = torch.where(a_mask, q, torch.zeros_like(q))
    p = torch.where(a_mask, probs, torch.zeros_like(probs))
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
    baseline = (p * qm).sum(-1)
    q_chosen = q.gather(-1, chosen.view(-1, 1)).squeeze(-1)
    best = torch.where(a_mask, q, torch.full_like(q, float("-inf"))).max(-1).values
    return {"q_chosen": q_chosen, "baseline": baseline, "quality": q_chosen - baseline, "regret": best - q_chosen}


def outcome_decomposition(y: Tensor, q_chosen: Tensor, baseline: Tensor) -> Dict[str, Tensor]:
    """Split a realised outcome ``y`` into the part explained by the decision and the part that was not:

        y - baseline = (q_chosen - baseline) + (y - q_chosen) = quality + luck

    A lucky success has quality < 0 and luck > 0; a good decision that failed has quality > 0 and luck < 0."""
    return {"quality": q_chosen - baseline, "luck": y - q_chosen, "surprise": y - baseline}
