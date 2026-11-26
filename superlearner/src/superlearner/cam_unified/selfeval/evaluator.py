"""Learned goal-conditioned self-evaluation (meta-reward spec Part 4; manifest SE-01..SE-06).

`ProspectEvaluator` predicts, from CAM's reasoned latent state R [B,S,D], the goal tokens G and a policy context, the
distribution of the eventual *independently verified* outcome of the current objective:

  success      p = sigmoid(logit) = P(verified success by the deadline | continuation policy, budget)      (4.1)
  time         discrete-time hazard over `n_time_bins` log-spaced bins of remaining days, censoring-aware     (4.2)
  magnitude    quantiles of the achieved magnitude (symlog scale, e.g. verified revenue in cents)             (4.2)
  resources    quantiles of resources consumed until resolution (symlog scale, e.g. spend in cents)           (4.2)
  loss         P(material loss)                                                                               (4.2)

Targets come only from the training-label channel (released verified outcomes after they occurred). A target with
`status == PENDING` contributes nothing to the success loss (pending is never a failure, SE-03) and contributes a
right-censored likelihood to the time head (survived at least the elapsed time). The predictor is trained on many
prefixes of each completed trajectory: every recorded decision of a resolved objective is one example, the final
label supervising all earlier predictions.

`policy_context` conditions predictions on the continuation policy (SE-05): a learned embedding of the policy-version
bucket plus optional continuous features (e.g. the learner's recent verified success rate). A frozen copy
(`frozen_target()`) is the target evaluator the actor optimises against between recalibrations, so the actor cannot
chase the evaluator's moving errors.

`calibration_report` computes NLL, Brier, ECE, reliability bins and the logistic calibration slope/intercept from
(prediction, outcome) pairs recorded BEFORE the outcomes were known (SE-04).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..interfaces import TokenSet
from ..learn.heads import AttentionPool, _MLP, symexp, symlog

PENDING, SUCCESS, FAILURE, UNCERTAIN = 0, 1, 2, 3     # outcome label status codes (training-label channel)


@dataclass
class ProspectTargets:
    status: Tensor                      # [B] long: PENDING / SUCCESS / FAILURE / UNCERTAIN
    time_days: Tensor                   # [B] float: days from this prediction to resolution (or elapsed, if censored)
    magnitude: Optional[Tensor] = None  # [B] float achieved magnitude (nan = unknown)
    resources: Optional[Tensor] = None  # [B] float resources consumed (nan = unknown)
    loss: Optional[Tensor] = None       # [B] float in {0,1} (nan = unknown)
    weight: Optional[Tensor] = None     # [B] float example weights (e.g. 1/number of prefixes of the trajectory)


@dataclass
class ProspectOutput:
    success_logit: Tensor               # [B]
    hazard_logits: Tensor               # [B, n_time_bins]
    magnitude_q: Tensor                 # [B, n_quantiles] symlog scale, sorted
    resources_q: Tensor                 # [B, n_quantiles]
    loss_logit: Tensor                  # [B]
    feature: Tensor                     # [B, D] pooled evaluation feature (shared with intrinsic reward / credit)

    @property
    def p_success(self) -> Tensor:
        return torch.sigmoid(self.success_logit)

    def survival(self) -> Tensor:
        """S[b, j] = P(not resolved by the end of bin j)."""
        h = torch.sigmoid(self.hazard_logits)
        return torch.cumprod(1 - h, -1)

    def expected_days(self, edges: Tensor) -> Tensor:
        S = self.survival()
        widths = edges[1:] - edges[:-1]
        S_prev = torch.cat([torch.ones_like(S[:, :1]), S[:, :-1]], -1)
        return (S_prev * widths).sum(-1)

    def magnitude_median(self) -> Tensor:
        return symexp(self.magnitude_q[:, self.magnitude_q.shape[1] // 2])


class ProspectEvaluator(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, n_time_bins: int = 16, max_days: float = 365.0,
                 n_quantiles: int = 9, n_policy_buckets: int = 64, n_policy_feats: int = 4, hidden: Optional[int] = None):
        super().__init__()
        self.pool = AttentionPool(d_model, n_heads, 1)
        self.n_time_bins, self.n_q = n_time_bins, n_quantiles
        edges = torch.cat([torch.zeros(1), torch.logspace(math.log10(0.05), math.log10(max_days), n_time_bins)])
        self.register_buffer("edges", edges, persistent=False)
        self.register_buffer("taus", (torch.arange(n_quantiles) + 0.5) / n_quantiles, persistent=False)
        self.horizon = nn.Linear(2, d_model)
        self.policy_emb = nn.Embedding(n_policy_buckets, d_model)
        nn.init.normal_(self.policy_emb.weight, std=0.02)
        self.policy_feat = nn.Linear(n_policy_feats, d_model)
        self.n_policy_feats = n_policy_feats
        h = hidden or d_model
        self.trunk = _MLP(d_model, h, d_model)
        self.success = nn.Linear(d_model, 1)
        self.hazard = nn.Linear(d_model, n_time_bins)
        self.mag = nn.Linear(d_model, n_quantiles)
        self.res = nn.Linear(d_model, n_quantiles)
        self.lossp = nn.Linear(d_model, 1)
        for m in (self.success, self.hazard, self.mag, self.res, self.lossp):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.constant_(self.hazard.bias, -2.0)

    def forward(self, R: Tensor, goal: Optional[TokenSet], horizon_days: Tensor, policy_bucket: Tensor,
                policy_feats: Optional[Tensor] = None, extra: Optional[list] = None) -> ProspectOutput:
        B = R.shape[0]
        hp = self.pool(R, [goal] + list(extra or []))[:, 0]
        hz = torch.stack([torch.log1p(horizon_days.float().clamp_min(0)), (horizon_days > 0).float()], -1)
        pf = policy_feats if policy_feats is not None else R.new_zeros(B, self.n_policy_feats)
        x = hp + self.horizon(hz.to(hp.dtype)) + self.policy_emb(policy_bucket.long().clamp(0, self.policy_emb.num_embeddings - 1)) \
            + self.policy_feat(pf.to(hp.dtype))
        f = x + self.trunk(x)
        f32 = f.float()
        mag = torch.cumsum(F.softplus(self.mag(f32)), -1) + self.mag(f32)[:, :1] - F.softplus(self.mag(f32)[:, :1])
        res = torch.cumsum(F.softplus(self.res(f32)), -1) + self.res(f32)[:, :1] - F.softplus(self.res(f32)[:, :1])
        return ProspectOutput(self.success(f32).squeeze(-1), self.hazard(f32), mag, res, self.lossp(f32).squeeze(-1), f)

    # ------------------------------------------------------------------------------------------------ losses

    def time_bin(self, days: Tensor) -> Tensor:
        return torch.bucketize(days.float().clamp(0, float(self.edges[-1]) - 1e-6), self.edges[1:], right=False) \
            .clamp_max(self.n_time_bins - 1)

    def losses(self, out: ProspectOutput, tgt: ProspectTargets) -> dict:
        st = tgt.status.long()
        w = tgt.weight.float() if tgt.weight is not None else torch.ones_like(out.success_logit)
        resolved = (st == SUCCESS) | (st == FAILURE)
        y = (st == SUCCESS).float()
        bce = F.binary_cross_entropy_with_logits(out.success_logit, y, reduction="none")
        l_succ = (bce * w * resolved.float()).sum() / (w * resolved.float()).sum().clamp_min(1e-8)

        # discrete hazard: success resolved in bin j -> sum_{i<j} log(1-h_i) + log h_j ; censored (pending / failure
        # at deadline) -> survived through its bin: sum_{i<=j} log(1-h_i)
        j = self.time_bin(tgt.time_days)
        logh = F.logsigmoid(out.hazard_logits)
        log1mh = F.logsigmoid(-out.hazard_logits)
        ar = torch.arange(self.n_time_bins, device=j.device)[None, :]
        before = (ar < j[:, None]).float()
        upto = (ar <= j[:, None]).float()
        event = (st == SUCCESS).float()
        ll = event * ((log1mh * before).sum(-1) + logh.gather(1, j[:, None]).squeeze(1)) + \
            (1 - event) * (log1mh * upto).sum(-1)
        valid_t = (st != UNCERTAIN).float()
        l_time = -(ll * w * valid_t).sum() / (w * valid_t).sum().clamp_min(1e-8)

        def pinball(q: Tensor, y: Optional[Tensor]) -> Tensor:
            if y is None:
                return q.sum() * 0
            ok = torch.isfinite(y)
            if not ok.any():
                return q.sum() * 0
            ys = symlog(torch.nan_to_num(y.float()))[:, None]
            d = ys - q
            l = torch.maximum(self.taus * d, (self.taus - 1) * d).mean(-1)
            return (l * w * ok.float()).sum() / (w * ok.float()).sum().clamp_min(1e-8)

        l_mag = pinball(out.magnitude_q, tgt.magnitude)
        l_res = pinball(out.resources_q, tgt.resources)
        if tgt.loss is not None and torch.isfinite(tgt.loss).any():
            okl = torch.isfinite(tgt.loss)
            bl = F.binary_cross_entropy_with_logits(out.loss_logit, torch.nan_to_num(tgt.loss.float()), reduction="none")
            l_loss = (bl * w * okl.float()).sum() / (w * okl.float()).sum().clamp_min(1e-8)
        else:
            l_loss = out.loss_logit.sum() * 0
        total = l_succ + l_time + l_mag + l_res + l_loss
        return {"total": total, "success": l_succ, "time": l_time, "magnitude": l_mag, "resources": l_res,
                "loss": l_loss, "n_resolved": int(resolved.sum())}

    def frozen_target(self) -> "ProspectEvaluator":
        tgt = copy.deepcopy(self)
        for p in tgt.parameters():
            p.requires_grad_(False)
        return tgt.eval()


def progress_signal(p_before: Tensor, p_after: Tensor) -> Tensor:
    """Change in predicted prospects (SE-06): a learned *interpretation*, never verified achievement, and not by
    itself the causal contribution of the action (credit estimators separate that)."""
    return p_after - p_before


# ============================================================================================ calibration

def calibration_report(p: Tensor, y: Tensor, n_bins: int = 10, groups: Optional[dict] = None) -> dict:
    """NLL, Brier, ECE, reliability bins and logistic calibration slope/intercept of predictions p for binary
    outcomes y (both recorded BEFORE y was known; y in {0,1}). `groups`: name -> bool mask for per-group reports."""
    p = p.detach().double().clamp(1e-6, 1 - 1e-6).flatten()
    y = y.detach().double().flatten()
    n = p.numel()
    if n == 0:
        return {"n": 0}
    nll = float(-(y * p.log() + (1 - y) * (1 - p).log()).mean())
    brier = float(((p - y) ** 2).mean())
    prev = float(y.mean())
    brier_prev = float(((prev - y) ** 2).mean())
    nll_prev = float(-(y * math.log(max(prev, 1e-6)) + (1 - y) * math.log(max(1 - prev, 1e-6))).mean())
    edges = torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)
    idx = torch.bucketize(p, edges[1:-1])
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        k = int(m.sum())
        if k == 0:
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += k / n * abs(conf - acc)
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": k, "mean_p": conf, "freq": acc})
    # logistic recalibration y ~ sigmoid(a + b * logit(p)) by Newton's method
    x = torch.log(p / (1 - p))
    X = torch.stack([torch.ones_like(x), x], 1)
    beta = torch.zeros(2, dtype=torch.float64)
    for _ in range(50):
        mu = torch.sigmoid(X @ beta)
        W = (mu * (1 - mu)).clamp_min(1e-9)
        H = X.T @ (X * W[:, None]) + 1e-6 * torch.eye(2, dtype=torch.float64)
        g = X.T @ (y - mu)
        step = torch.linalg.solve(H, g)
        beta = beta + step
        if float(step.abs().max()) < 1e-10:
            break
    rep = {"n": n, "nll": nll, "brier": brier, "ece": ece, "prevalence": prev, "brier_prevalence": brier_prev,
           "nll_prevalence": nll_prev, "brier_skill": 1 - brier / max(brier_prev, 1e-12),
           "calib_intercept": float(beta[0]), "calib_slope": float(beta[1]), "reliability": bins}
    if groups:
        rep["groups"] = {k: calibration_report(p[m], y[m], n_bins) for k, m in groups.items() if int(m.sum()) > 0}
    return rep
