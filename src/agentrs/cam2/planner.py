"""Model-based planning with the learned world model.

For the top-K candidates of the current decision, each world-model ensemble member imagines an H-step rollout:
step 0 takes the candidate; later steps take the policy's expected action embedding over the same candidate set
(page elements and tools persist across steps in the environments used here; this is an approximation that the
evaluation measures rather than assumes). The imagined return is
    R = sum_j gamma^j prod_{i<j}(1 - p_done_i) r_j + gamma^H prod(1 - p_done) V(f_H),
Q_hat = mean over members, u = std over members (epistemic uncertainty). The planner turns this into a logit bias
    bias_k = kappa * (Q_hat_k - mean_K Q_hat) - beta * u_k     (top-K only, 0 elsewhere)
and falls back to the plain policy (bias 0) when the mean uncertainty exceeds ``u_max`` (calibrated on real data by
the trainer) or when the world model's held-out reliability is below ``min_reliability``. The bias is stored with the
transition, so the learner's importance ratios stay exact. Imagined rewards are never logged as outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from agentrs.cam2.model import CAM


@dataclass
class PlanConfig:
    enabled: bool = False
    top_k: int = 8
    horizon: int = 3
    gamma: float = 0.95
    kappa: float = 4.0
    beta: float = 1.0
    u_max: float = 1.0e9          # set by calibration
    min_reliability: float = 0.2
    reliability: float = 0.0      # measured by the trainer on held-out real transitions (0 until measured)


@torch.no_grad()
def imagine_returns(cam: CAM, f: torch.Tensor, cand: torch.Tensor, cmask: torch.Tensor, logits: torch.Tensor,
                    first: torch.Tensor, cfg: PlanConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """f: (B,D); cand: (B,Nc,d); first: (B,K) candidate indices to evaluate. Returns (Q_hat, u) of shape (B,K)."""
    B, K = first.shape
    E = len(cam.wm)
    D = f.shape[-1]
    fk = f.unsqueeze(1).expand(B, K, D).reshape(B * K, D)
    a0 = cand.gather(1, first.unsqueeze(-1).expand(B, K, cand.shape[-1])).reshape(B * K, -1)
    candk = cand.unsqueeze(1).expand(B, K, *cand.shape[1:]).reshape(B * K, *cand.shape[1:])
    maskk = cmask.unsqueeze(1).expand(B, K, cmask.shape[1]).reshape(B * K, -1)
    rets = []
    for e, member in enumerate(cam.wm):
        ft, a, R, alive, disc = fk, a0, torch.zeros(B * K, device=f.device), torch.ones(B * K, device=f.device), 1.0
        for h in range(cfg.horizon):
            ae = cam.action_embedding(a, None, None)
            out = member(torch.cat([ft, ae], -1))
            nf = out[:, :D]
            r = out[:, -2]
            pd = torch.sigmoid(out[:, -1])
            R = R + disc * alive * r
            alive = alive * (1 - pd)
            disc *= cfg.gamma
            ft = nf  # world-model targets are already the (normalised) core features f
            pi = torch.softmax(cam.policy_logits(ft, candk, maskk), -1)
            a = torch.einsum("bn,bnd->bd", pi, candk)
        R = R + disc * alive * cam.v(ft)
        rets.append(R.view(B, K))
    rets = torch.stack(rets, 0)
    return rets.mean(0), rets.std(0) if E > 1 else torch.zeros_like(rets[0])


@torch.no_grad()
def plan_bias(cam: CAM, f, cand, cmask, logits, cfg: PlanConfig) -> tuple[torch.Tensor, dict]:
    B, Nc = logits.shape
    bias = torch.zeros(B, Nc, device=f.device)
    if cfg.reliability < cfg.min_reliability:
        return bias, {"planned": 0, "fallback": B, "reason": "world model unreliable"}
    K = min(cfg.top_k, Nc)
    first = logits.topk(K, dim=1).indices
    q, u = imagine_returns(cam, f, cand, cmask, logits, first, cfg)
    valid = cmask.gather(1, first)
    qm = (q * valid).sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp(min=1)
    b = cfg.kappa * (q - qm) - cfg.beta * u
    ok = (u.mean(1) <= cfg.u_max).float().unsqueeze(1)
    b = b * valid.float() * ok
    bias.scatter_(1, first, b)
    return bias, {"planned": int(ok.sum()), "fallback": int(B - ok.sum()), "q_mean": float(q.mean()),
                  "u_mean": float(u.mean())}
