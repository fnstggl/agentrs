"""CAM-scale meta-gradient of the intrinsic reward through differentiable inner policy updates (meta-reward spec
5.2-5.5, 11.3-11.4; manifest IR-03, IR-04, IR-05).

    inner:  phi_{k+1} = phi_k - alpha * grad_phi L_inner(phi_k; psi, D_k)        (create_graph: phi_K depends on psi)
            L_inner = -E[ log pi_phi(a_t | x_t) * (G^int_t - b_t) ] - c_H * H[pi_phi]
            G^int_t = sum_{j>=t} gamma_j r_psi(transition_j)      (discount from elapsed time: gamma = exp(-kappa dt))
    outer:  J_outer(psi) = E_{tau ~ pi_{phi_K}}[ U_verified(tau) ]   estimated with the score-function surrogate
            L_outer = -E[ log pi_{phi_K}(a | x) * (U_verified - V_L) ]  on FRESH query experience collected by phi_K,
            differentiated through phi_K(psi) into psi.

phi are the *fast policy parameters* (by default: every parameter of the policy module the caller passes; with a 1B
core, the caller passes the decision heads plus low-rank adapters of the last core blocks, and the rest of the core is
consolidated first-order). The environment is never differentiated: query trajectories are sampled with phi_K
detached and their verified utilities enter only as advantages of the score function.

Two-phase API, because fresh query experience must be collected *between* the inner update and the outer loss:

    fast = mrl.inner_update(inner_batches)          # dict of fast params, graph-connected to psi
    ...collect query experience acting with mrl.detached(fast)...
    stats = mrl.outer_step(fast, query_batch)       # meta-gradient step on psi (+ lifetime value regression)

`check_meta_gradient` compares the estimator with an exact or finite-difference reference on small problems.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.func import functional_call


@dataclass
class InnerBatch:
    """One inner-update minibatch of real experience (trajectory windows, time-major within each row).

    policy_inputs: kwargs for the policy module's forward -> logits [N, T, A] (or [N*T, A] when flat=True)
    actions:       [N, T] long, the executed choice (index into the candidate set the behaviour policy scored)
    mask:          [N, T] bool valid steps
    action_mask:   [N, T, A] bool feasible candidates (optional)
    reward_inputs: kwargs for the reward module's forward -> r [N, T] (psi-dependent)
    gamma:         [N, T] per-step discount to the next step (exp(-kappa * dt)), default 1
    """
    policy_inputs: dict
    actions: Tensor
    mask: Tensor
    reward_inputs: dict
    action_mask: Optional[Tensor] = None
    gamma: Optional[Tensor] = None


@dataclass
class QueryBatch:
    """Fresh experience collected by the post-update policy, with independently verified utilities."""
    policy_inputs: dict
    actions: Tensor          # [N, T]
    mask: Tensor             # [N, T]
    utility: Tensor          # [N] verified utility of each query trajectory (or [N, T] verified return-to-go)
    action_mask: Optional[Tensor] = None
    baseline: Optional[Tensor] = None   # [N] lifetime-value / baseline prediction (detached by the learner)


def discounted_returns(r: Tensor, mask: Tensor, gamma: Optional[Tensor] = None) -> Tensor:
    """G_t = r_t + gamma_t G_{t+1} within each row (graph-preserving)."""
    N, T = r.shape
    g = gamma if gamma is not None else torch.ones_like(r)
    out = []
    G = torch.zeros(N, dtype=r.dtype, device=r.device)
    m = mask.to(r.dtype)
    for t in reversed(range(T)):
        G = (r[:, t] + g[:, t] * G) * m[:, t]
        out.append(G)
    return torch.stack(out[::-1], 1)


class MetaRewardLearner:
    def __init__(self, policy: nn.Module, reward: nn.Module, *, inner_lr: float = 0.1, K: int = 2,
                 fast_param_filter: Optional[Callable[[str], bool]] = None, outer_lr: float = 1e-3,
                 entropy_coef: float = 0.0, meta_clip: float = 1.0, reward_optimizer: Optional[torch.optim.Optimizer] = None,
                 lifetime_value: Optional[nn.Module] = None, lv_lr: float = 1e-3, normalize_inner_adv: bool = True):
        self.policy, self.reward = policy, reward
        self.inner_lr, self.K = float(inner_lr), int(K)
        flt = fast_param_filter or (lambda n: True)
        self.fast_names = [n for n, p in policy.named_parameters() if p.requires_grad and flt(n)]
        self.entropy_coef = entropy_coef
        self.meta_clip = meta_clip
        self.opt = reward_optimizer or torch.optim.Adam(reward.parameters(), lr=outer_lr)
        self.lv = lifetime_value
        self.lv_opt = torch.optim.Adam(lifetime_value.parameters(), lr=lv_lr) if lifetime_value is not None else None
        self.normalize_inner_adv = normalize_inner_adv
        self.history: list = []

    # ------------------------------------------------------------------------------------------ policy helpers
    def _params(self) -> dict:
        return {n: p for n, p in self.policy.named_parameters() if n in self.fast_names}

    @staticmethod
    def detached(fast: dict) -> dict:
        return {n: p.detach() for n, p in fast.items()}

    def _logp_ent(self, params: dict, inputs: dict, actions: Tensor, action_mask: Optional[Tensor]) -> tuple:
        logits = functional_call(self.policy, params, args=(), kwargs=inputs, strict=False)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        logp_all = torch.log_softmax(logits.float(), -1)
        lp = logp_all.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
        p = logp_all.exp()
        ent = -(p * logp_all.masked_fill(~torch.isfinite(logp_all), 0)).sum(-1)
        return lp, ent

    # ------------------------------------------------------------------------------------------ inner
    def inner_update(self, batches: Sequence[InnerBatch], create_graph: bool = True) -> dict:
        """K differentiable policy-gradient steps on intrinsic returns (cycling through `batches`). For on-policy
        inner data (fresh experience after every inner step, as in Zheng et al.) call `inner_step` K times,
        collecting each batch with `detached(fast)`."""
        fast = self._params()
        stats = []
        for k in range(self.K):
            fast, st = self.inner_step(fast, batches[k % len(batches)], create_graph)
            stats.append(st)
        self.last_inner = stats
        return fast

    def start(self) -> dict:
        return self._params()

    def inner_step(self, fast: dict, b: InnerBatch, create_graph: bool = True) -> tuple:
        """One differentiable inner policy-gradient step from `fast` on batch `b`; returns (fast', stats)."""
        if True:
            r = self.reward(**b.reward_inputs)                           # [N, T], depends on psi
            G = discounted_returns(r, b.mask, b.gamma)
            m = b.mask.float()
            base = (G * m).sum(0, keepdim=True) / m.sum(0, keepdim=True).clamp_min(1.0)     # per-step mean baseline
            adv = (G - base.detach()) * m
            if self.normalize_inner_adv:
                sd = (adv.pow(2).sum() / m.sum().clamp_min(1.0)).sqrt().detach().clamp_min(1e-6)
                adv = adv / sd
            lp, ent = self._logp_ent({**dict(self.policy.named_parameters()), **fast}, b.policy_inputs, b.actions,
                                     b.action_mask)
            L = -((lp * adv).sum() + self.entropy_coef * (ent * m).sum()) / m.sum().clamp_min(1.0)
            grads = torch.autograd.grad(L, list(fast.values()), create_graph=create_graph, allow_unused=True)
            fast = {n: (p - self.inner_lr * g) if g is not None else p for (n, p), g in zip(fast.items(), grads)}
            st = {"inner_loss": float(L.detach()), "r_mean": float((r * m).sum() / m.sum().clamp_min(1)),
                  "r_std": float(r.detach()[b.mask.bool()].std()) if int(b.mask.sum()) > 1 else 0.0}
        return fast, st

    # ------------------------------------------------------------------------------------------ outer
    def outer_loss(self, fast: dict, q: QueryBatch) -> Tensor:
        lp, _ = self._logp_ent({**dict(self.policy.named_parameters()), **fast}, q.policy_inputs, q.actions,
                               q.action_mask)
        m = q.mask.float()
        U = q.utility.float()
        if U.dim() == 1:
            base = q.baseline.float() if q.baseline is not None else U.mean()
            adv = (U - base).detach()[:, None].expand_as(lp)
        else:
            adv = (U - (q.baseline.float() if q.baseline is not None else U.mean(0, keepdim=True))).detach()
        return -(lp * adv * m).sum() / m.sum().clamp_min(1.0)

    def meta_gradient(self, fast: dict, q: QueryBatch, retain_graph: bool = False) -> list:
        L = self.outer_loss(fast, q)
        params = [p for p in self.reward.parameters() if p.requires_grad]
        g = torch.autograd.grad(L, params, allow_unused=True, retain_graph=retain_graph)
        return [torch.zeros_like(p) if gi is None else gi for p, gi in zip(params, g)]

    def outer_step(self, fast: dict, q: QueryBatch, lv_inputs: Optional[dict] = None,
                   lv_targets: Optional[Tensor] = None) -> dict:
        grads = self.meta_gradient(fast, q)
        norm = math.sqrt(sum(float(g.pow(2).sum()) for g in grads))
        scale = min(1.0, self.meta_clip / max(norm, 1e-12)) if self.meta_clip else 1.0
        self.opt.zero_grad(set_to_none=True)
        for p, g in zip([p for p in self.reward.parameters() if p.requires_grad], grads):
            p.grad = g * scale
        self.opt.step()
        if hasattr(self.reward, "bump_version"):
            self.reward.bump_version()
        out = {"meta_grad_norm": norm, "clip_scale": scale, "query_utility": float(q.utility.float().mean())}
        if self.lv is not None and lv_inputs is not None and lv_targets is not None:
            pred = self.lv(**lv_inputs)
            L = self.lv.loss(pred, lv_targets)
            self.lv_opt.zero_grad(set_to_none=True)
            L.backward()
            self.lv_opt.step()
            out["lifetime_value_loss"] = float(L.detach())
        self.history.append(out)
        return out

    def commit(self, fast: dict) -> None:
        """Make the inner-updated fast parameters the learner's persistent parameters (the policy keeps learning
        within its lifetime, as in Zheng et al.)."""
        with torch.no_grad():
            for n, p in self.policy.named_parameters():
                if n in fast:
                    p.copy_(fast[n].detach())
