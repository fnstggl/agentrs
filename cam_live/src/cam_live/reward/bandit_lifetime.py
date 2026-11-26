"""cam_live.reward.bandit_lifetime — a small neural bandit-lifetime problem with an exact outer objective.

DIAGNOSTIC ONLY (algorithm-correctness tests and cost measurements of ``reward.meta``; never evidence of live
learning). A contextual bandit with ``n_ctx`` contexts given as random feature vectors and ``n_arms`` arms:

    policy   pi_theta(a | x_c) = softmax(MLP_theta(x_c))                         theta: the adaptable subset
    reward   r_psi(c, a) = bound * tanh(MLP_psi([x_c, onehot(a), signals(c, a)]) / bound)
    verified utility U[c, a]: known only to the "verifier" side (used by the outer objective, never an input of
    the reward network, which sees the observable ``signals`` instead)

A lifetime is K inner updates of the policy on support rows rewarded by r_psi; the exact outer objective is
J(theta_K) = sum_c p(c) sum_a pi_{theta_K}(a | x_c) U[c, a], so meta-gradients can be checked by finite differences.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from .integrity import bound_reward
from .meta import Params, QueryData, functional_logp, functional_reward


class _MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, depth: int = 1):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, d_hidden), nn.Tanh()]
            d = d_hidden
        self.net = nn.Sequential(*layers, nn.Linear(d, d_out))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _Reward(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, bound: float):
        super().__init__()
        self.mlp, self.bound = _MLP(d_in, d_hidden, 1), bound

    def forward(self, x: Tensor) -> Tensor:
        return bound_reward(self.mlp(x).squeeze(-1), self.bound)


class BanditLifetime:
    def __init__(self, n_ctx: int = 6, n_arms: int = 4, d_x: int = 8, d_signal: int = 3, hidden: int = 16,
                 reward_hidden: int = 16, bound: float = 1.0, seed: int = 0, dtype: torch.dtype = torch.float64,
                 device: str = "cpu", depth: int = 1):
        g = torch.Generator().manual_seed(seed)
        rn = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64).to(dtype=dtype, device=device)
        self.n_ctx, self.n_arms, self.dtype, self.device = n_ctx, n_arms, dtype, device
        self.x = rn(n_ctx, d_x)
        self.utility = rn(n_ctx, n_arms)                                   # verifier side only
        # observable signals: one noisy view of the utility and distractors (what the reward network may read)
        self.signals = torch.cat([self.utility[..., None] + 0.5 * rn(n_ctx, n_arms, 1),
                                  rn(n_ctx, n_arms, d_signal - 1)], -1)
        self.p_ctx = torch.full((n_ctx,), 1.0 / n_ctx, dtype=dtype, device=device)
        torch.manual_seed(seed)
        self.policy = _MLP(d_x, hidden, n_arms, depth).to(dtype=dtype, device=device)
        self.reward = _Reward(d_x + n_arms + d_signal, reward_hidden, bound).to(dtype=dtype, device=device)
        self.logp_fn = functional_logp(self.policy)
        self.reward_fn = functional_reward(self.reward)

    # ------------------------------------------------------------------------------------------- parameters
    def theta(self, lifetimes: Optional[int] = None) -> Params:
        th = {n: p.detach().clone() for n, p in self.policy.named_parameters()}
        return th if lifetimes is None else {n: v.expand(lifetimes, *v.shape).clone() for n, v in th.items()}

    def psi(self) -> Params:
        return dict(self.reward.named_parameters())

    # ------------------------------------------------------------------------------------------- batches
    def _reward_inputs(self, c: Tensor, a: Tensor) -> Tensor:
        onehot = torch.nn.functional.one_hot(a, self.n_arms).to(self.dtype)
        return torch.cat([self.x[c], onehot, self.signals[c, a]], -1)

    def batch(self, c: Tensor, a: Tensor) -> Dict[str, Any]:
        """Rows (context index, arm). c, a: [n] for one lifetime or [R, n] for R parallel lifetimes."""
        b = {"x": self.x[c], "a": a, "reward_inputs": (self._reward_inputs(c.reshape(-1), a.reshape(-1)),),
             "c": c}
        if c.dim() == 2:
            b["lifetime"] = torch.arange(c.shape[0], device=c.device).repeat_interleave(c.shape[1])
        return b

    def probs(self, theta: Params) -> Tensor:
        """pi_theta(. | x_c) for every context: [n_ctx, A] (or [R, n_ctx, A] for batched theta)."""
        ref = dict(self.policy.named_parameters())
        name = next(iter(theta))
        if theta[name].dim() == ref[name].dim() + 1:
            logits = torch.vmap(lambda th: torch.func.functional_call(self.policy, th, (self.x,)))(theta)
        else:
            logits = torch.func.functional_call(self.policy, theta, (self.x,))
        return torch.softmax(logits, -1)

    def sample(self, theta: Params, n: int, generator: torch.Generator) -> Tuple[Dict[str, Any], Tensor]:
        """n on-policy rows per lifetime from pi_theta (detached). Returns (batch, verified utility of each row)."""
        p = self.probs({k: v.detach() for k, v in theta.items()})
        batched = p.dim() == 3
        R = p.shape[0] if batched else 1
        c = torch.multinomial(self.p_ctx.cpu().expand(R, -1), n, replacement=True, generator=generator).to(p.device)
        pc = (p if batched else p[None])[torch.arange(R, device=p.device)[:, None], c]          # [R, n, A]
        a = torch.multinomial(pc.reshape(R * n, -1).cpu(), 1, generator=generator).view(R, n).to(p.device)
        if not batched:
            c, a = c[0], a[0]
        return self.batch(c, a), self.utility[c, a].reshape(-1)

    # ------------------------------------------------------------------------------------------- exact outer
    def exact_utility(self, theta: Params) -> Tensor:
        """J(theta) = sum_c p(c) sum_a pi_theta(a|x_c) U[c,a] (scalar, or [R] for batched theta)."""
        return (self.probs(theta) * self.utility).sum(-1) @ self.p_ctx

    def all_actions_query(self, theta_K: Params) -> QueryData:
        """Every (context, arm) row weighted by p(c) * pi_{theta_K}(a|x_c) (detached): the surrogate on this
        query has exactly the gradient of :meth:`exact_utility` (no sampling noise). One lifetime only."""
        c = torch.arange(self.n_ctx, device=self.x.device).repeat_interleave(self.n_arms)
        a = torch.arange(self.n_arms, device=self.x.device).repeat(self.n_ctx)
        p = self.probs({k: v.detach() for k, v in theta_K.items()})
        return QueryData(batch=self.batch(c, a), utility=self.utility[c, a], weights=(self.p_ctx[c] * p[c, a]),
                         baseline=0.0)
