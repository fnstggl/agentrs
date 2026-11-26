"""CAM-scale meta-gradient of the intrinsic reward through differentiable inner policy updates (meta-reward spec
5.2-5.5, 11.3-11.4; manifest IR-03, IR-04, IR-05; redesign plan 2026-11-26 §2 "L_meta" / "Inner").

    inner:  phi_{k+1} = phi_k - alpha * grad_phi L_inner(phi_k; psi, D_k)        (create_graph: phi_K depends on psi)
            L_inner = -E[ log pi_phi(a_t | x_t) * (G^int_t - b_t) ] - c_H * H[pi_phi]
            G^int_t = sum_{j>=t} gamma_j r_psi(transition_j)      (discount from elapsed time: gamma = exp(-kappa dt))
    outer:  J_outer(psi) = E_{tau ~ pi_{phi_K}}[ U_verified(tau) ]   estimated with the score-function surrogate
            L_outer = -E[ log pi_{phi_K}(a | x) * (U_verified - V_L) ]  on FRESH query experience collected by phi_K,
                      - sum_{k>=1} log pi_{phi_k}(D_k) * (J_query - c)          (inner-data score term)
            differentiated through phi_k(psi) into psi.

The inner-data score term: batches D_k, k >= 1, are collected by phi_k(psi), so their distribution depends on psi;
leaving the term out biases the meta-gradient for K >= 2 (research note §10.3: rel. error 0.015, max|z| 10 at K = 4).
c is a constant independent of the current inner and query data (`score_baseline`, else the running mean of earlier
query utilities, else 0). Baselines inside the inner and outer surrogates are leave-one-out means (a self-inclusive
batch mean scales the gradient by (1 - 1/N)). The inner advantage normalisation keeps its graph (a detached
data-dependent scale would make the meta-gradient that of a different update rule). Units: the query surrogate is a
per-step mean, so the score term is scaled by N_query / sum(mask) = 1 / (mean steps per query trajectory) — the
earlier 1 / sum(mask) shrank it by a further factor N_query (tested against the exact tabular meta-gradient with
2000-repetition z-scores, tests/cam_unified/credit/test_meta_unbiased_regression.py).

phi are the *fast policy parameters*: with the CAM model, `decision.fast.fast_params(model)` (W5; the decision heads,
<= 50M) or the local stand-in `default_fast_params` (action-generator heads minus the semantic bridge + value head).
The environment is never differentiated: query trajectories are sampled with phi_K detached and their verified
utilities enter only as advantages of the score function.

Two-phase API, because fresh query experience must be collected *between* the inner update and the outer loss:

    fast = mrl.inner_update(inner_batches)          # dict of fast params, graph-connected to psi
    ...collect query experience acting with mrl.detached(fast)...
    stats = mrl.outer_step(fast, query_batch)       # meta-gradient step on psi (+ lifetime value regression)

`MetaRewardTrainer` binds this to a CAM model + `RewardModel` (psi = RewardModel.meta_parameters()): `act_with(fast)`
loads phi_k into the acting model (bumping its weights epoch) while experience for the next inner step / the query is
collected, `inner_step`, `outer_step` (bumps the psi version), `commit` (consolidates phi_K into the slow weights).
"""
from __future__ import annotations

import contextlib
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

    policy_inputs: kwargs for the policy module's forward -> logits [N, T, A] (or for `logp_fn`)
    actions:       [N, T] long, the executed choice (index into the candidate set the behaviour policy scored), or
                   any object `logp_fn` understands (e.g. a list of CAM Actions)
    mask:          [N, T] bool valid steps
    action_mask:   [N, T, A] bool feasible candidates (optional)
    reward_inputs: kwargs for the reward module's forward -> r [N, T] (psi-dependent)
    gamma:         [N, T] per-step discount to the next step (exp(-kappa * dt)), default 1
    """
    policy_inputs: dict
    actions: object
    mask: Tensor
    reward_inputs: dict
    action_mask: Optional[Tensor] = None
    gamma: Optional[Tensor] = None


@dataclass
class QueryBatch:
    """Fresh experience collected by the post-update policy, with independently verified utilities."""
    policy_inputs: dict
    actions: object          # [N, T] (or what logp_fn understands)
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


def loo_mean(x: Tensor, m: Tensor) -> Tensor:
    """Leave-one-out mean over rows (dim 0) per column, restricted to valid entries m; 0 where no other entry."""
    mf = m.to(x.dtype)
    s = (x * mf).sum(0, keepdim=True)
    n = mf.sum(0, keepdim=True)
    return torch.where(n - mf > 0, (s - x * mf) / (n - mf).clamp_min(1.0), torch.zeros_like(x))


class MetaRewardLearner:
    def __init__(self, policy: nn.Module, reward: nn.Module, *, inner_lr: float = 0.1, K: int = 2,
                 fast_param_filter: Optional[Callable[[str], bool]] = None, outer_lr: float = 1e-3,
                 entropy_coef: float = 0.0, meta_clip: float = 1.0, reward_optimizer: Optional[torch.optim.Optimizer] = None,
                 lifetime_value: Optional[nn.Module] = None, lv_lr: float = 1e-3, normalize_inner_adv: bool = True,
                 logp_fn: Optional[Callable] = None, meta_params: Optional[list] = None,
                 reward_fn: Optional[Callable] = None):
        """logp_fn(fast: dict, inputs: dict, actions, action_mask) -> (logp [N,T], entropy [N,T]) replaces the
        default `log_softmax(functional_call(policy, params, **inputs))`; reward_fn(**reward_inputs) -> r [N,T]
        replaces `reward(**reward_inputs)`; meta_params: psi (default reward.meta_parameters() when it exists, else
        every reward parameter)."""
        self.policy, self.reward = policy, reward
        self.inner_lr, self.K = float(inner_lr), int(K)
        flt = fast_param_filter or (lambda n: True)
        self.fast_names = [n for n, p in policy.named_parameters() if p.requires_grad and flt(n)]
        self.entropy_coef = entropy_coef
        self.meta_clip = meta_clip
        if meta_params is None:
            meta_params = list(reward.meta_parameters()) if hasattr(reward, "meta_parameters") else \
                list(reward.parameters())
        self.meta_params = [p for p in meta_params if p.requires_grad]
        self.opt = reward_optimizer or torch.optim.Adam(self.meta_params, lr=outer_lr)
        self.lv = lifetime_value
        self.lv_opt = torch.optim.Adam(lifetime_value.parameters(), lr=lv_lr) if lifetime_value is not None else None
        self.normalize_inner_adv = normalize_inner_adv
        self.logp_fn = logp_fn
        self.reward_fn = reward_fn
        self.history: list = []
        self._scores: list = []          # log-probabilities of on-policy inner batches k >= 1 under phi_k(psi)
        self._n_steps = 0
        self.u_mean: Optional[float] = None   # running mean of earlier query utilities: the inner-score constant c
        self.u_momentum = 0.1
        self.last_inner: list = []

    # ------------------------------------------------------------------------------------------ policy helpers
    def _params(self) -> dict:
        return {n: p for n, p in self.policy.named_parameters() if n in self.fast_names}

    @staticmethod
    def detached(fast: dict) -> dict:
        return {n: p.detach() for n, p in fast.items()}

    def _logp_ent(self, fast: dict, inputs: dict, actions, action_mask: Optional[Tensor]) -> tuple:
        if self.logp_fn is not None:
            return self.logp_fn(fast, inputs, actions, action_mask)
        params = {**dict(self.policy.named_parameters()), **fast}
        logits = functional_call(self.policy, params, args=(), kwargs=inputs, strict=False)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        logp_all = torch.log_softmax(logits.float(), -1)
        lp = logp_all.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
        p = logp_all.exp()
        ent = -(p * logp_all.masked_fill(~torch.isfinite(logp_all), 0)).sum(-1)
        return lp, ent

    def _reward(self, inputs: dict) -> Tensor:
        return self.reward_fn(**inputs) if self.reward_fn is not None else self.reward(**inputs)

    # ------------------------------------------------------------------------------------------ inner
    def inner_update(self, batches: Sequence[InnerBatch], create_graph: bool = True) -> dict:
        """K differentiable policy-gradient steps on intrinsic returns (cycling through `batches`; off-policy reuse,
        so no inner-data score term). For on-policy inner data (fresh experience after every inner step, as in
        Zheng et al.) call `start()` and then `inner_step` K times, collecting each batch with `detached(fast)`."""
        fast = self.start()
        stats = []
        for k in range(self.K):
            fast, st = self.inner_step(fast, batches[k % len(batches)], create_graph, on_policy=False)
            stats.append(st)
        self.last_inner = stats
        return fast

    def start(self) -> dict:
        self._scores, self._n_steps = [], 0
        self.last_inner = []
        return self._params()

    def inner_step(self, fast: dict, b: InnerBatch, create_graph: bool = True, on_policy: bool = True) -> tuple:
        """One differentiable inner policy-gradient step from `fast` on batch `b`; returns (fast', stats).

        on_policy: `b` was collected by acting with `detached(fast)`. For every such batch after the first, the
        batch's log-probability under phi_k(psi) is kept: its score enters the outer loss (the inner data
        distribution depends on psi; omitting it biases the meta-gradient for K >= 2, research note §10.3)."""
        r = self._reward(b.reward_inputs)                            # [N, T], depends on psi
        G = discounted_returns(r, b.mask, b.gamma)
        m = b.mask.float()
        adv = (G - loo_mean(G, b.mask).detach()) * m                # leave-one-out per-step baseline
        if self.normalize_inner_adv:
            sd = (adv.pow(2).sum() / m.sum().clamp_min(1.0)).sqrt().clamp_min(1e-6)   # graph kept (see docstring)
            adv = adv / sd
        lp, ent = self._logp_ent(fast, b.policy_inputs, b.actions, b.action_mask)
        if on_policy and self._n_steps >= 1:
            self._scores.append((lp * m).sum())
        self._n_steps += 1
        L = -((lp * adv).sum() + self.entropy_coef * (ent * m).sum()) / m.sum().clamp_min(1.0)
        names = list(fast)
        grads = torch.autograd.grad(L, [fast[n] for n in names], create_graph=create_graph, allow_unused=True)
        fast = {n: (fast[n] - self.inner_lr * g) if g is not None else fast[n] for n, g in zip(names, grads)}
        st = {"inner_loss": float(L.detach()), "r_mean": float((r.detach() * m).sum() / m.sum().clamp_min(1)),
              "r_std": float(r.detach()[b.mask.bool()].std()) if int(b.mask.sum()) > 1 else 0.0,
              "grad_norm": math.sqrt(sum(float(g.detach().pow(2).sum()) for g in grads if g is not None))}
        self.last_inner.append(st)
        return fast, st

    # ------------------------------------------------------------------------------------------ outer
    def outer_loss(self, fast: dict, q: QueryBatch, score_baseline: Optional[float] = None) -> Tensor:
        lp, _ = self._logp_ent(fast, q.policy_inputs, q.actions, q.action_mask)
        m = q.mask.float()
        U = q.utility.float().to(lp.device)
        if U.dim() == 1:
            if q.baseline is not None:
                base = q.baseline.float().to(lp.device)
            else:
                base = loo_mean(U[:, None], torch.ones_like(U[:, None], dtype=torch.bool))[:, 0]
            adv = (U - base).detach()[:, None].expand_as(lp)
        else:
            base = q.baseline.float().to(lp.device) if q.baseline is not None else loo_mean(U, q.mask)
            adv = (U - base).detach()
        L = -(lp * adv * m).sum() / m.sum().clamp_min(1.0)
        if self._scores:
            n_q = float(m.shape[0])
            u = float((U if U.dim() == 1 else U[:, 0]).float().mean())
            c = score_baseline if score_baseline is not None else (self.u_mean if self.u_mean is not None else 0.0)
            # score of the psi-dependent inner data, in the units of the per-step-mean query surrogate
            L = L - torch.stack(self._scores).sum() * (u - c) * n_q / m.sum().clamp_min(1.0)
        return L

    def meta_gradient(self, fast: dict, q: QueryBatch, retain_graph: bool = False,
                      score_baseline: Optional[float] = None) -> list:
        L = self.outer_loss(fast, q, score_baseline)
        g = torch.autograd.grad(L, self.meta_params, allow_unused=True, retain_graph=retain_graph)
        return [torch.zeros_like(p) if gi is None else gi for p, gi in zip(self.meta_params, g)]

    def outer_step(self, fast: dict, q: QueryBatch, lv_inputs: Optional[dict] = None,
                   lv_targets: Optional[Tensor] = None, score_baseline: Optional[float] = None) -> dict:
        grads = self.meta_gradient(fast, q, score_baseline=score_baseline)
        norm = math.sqrt(sum(float(g.pow(2).sum()) for g in grads))
        scale = min(1.0, self.meta_clip / max(norm, 1e-12)) if self.meta_clip else 1.0
        self.opt.zero_grad(set_to_none=True)
        for p, g in zip(self.meta_params, grads):
            p.grad = (g * scale).to(p.dtype)
        self.opt.step()
        if hasattr(self.reward, "bump_version"):
            self.reward.bump_version()
        U = q.utility.float()
        u = float((U if U.dim() == 1 else U[:, 0]).mean())
        self.u_mean = u if self.u_mean is None else (1 - self.u_momentum) * self.u_mean + self.u_momentum * u
        n_scores = len(self._scores)
        self._scores = []
        out = {"meta_grad_norm": norm, "clip_scale": scale, "query_utility": u, "inner_score_terms": n_scores}
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


# ================================================================================================ CAM binding

SEMANTIC_PREFIXES = ("semantic.", "action_generator.sem.", "action_generator.bridge.", "action_generator.lora.",
                     "tokenizer.")


def default_fast_params(model: nn.Module) -> dict:
    """phi for the meta step: `decision.fast.fast_params(model)` when W5's decision package exists; otherwise the
    local stand-in over the seam-0 decision path: the action generator's heads (tool / evidence pointers, typed
    argument heads, length head), excluding the semantic model, the latent bridge (decoder conditioning, theta_dec_cond)
    and LoRA, plus the value head. Names are the model's own parameter names."""
    try:                                                    # W5's API (plan §3: decision.fast.fast_params -> dict)
        from ..decision.fast import fast_params as _fp      # type: ignore
        return dict(_fp(model))
    except ImportError:
        pass
    out = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not (n.startswith("action_generator.") or n.startswith("value_head.")):
            continue
        if any(n.startswith(s) for s in SEMANTIC_PREFIXES):
            continue
        out[n] = p
    return out


class _Call(nn.Module):
    """functional_call target: forward(fn, *a, **k) = fn(*a, **k) with this module's parameters swapped in, so any
    method of the wrapped model (not only forward) runs with substituted fast parameters."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.m = model

    def forward(self, fn, *a, **k):
        return fn(*a, **k)


class MetaRewardTrainer:
    """psi meta-step over the CAM fast set phi (plan §2 L_meta / Inner; W8 MetaRewardTrainer).

    model: the CamUnifiedModel (or any nn.Module); reward_model: selfeval.reward_model.RewardModel (psi = its
    meta_parameters()); fast_params: dict name -> Parameter of `model` (default `default_fast_params(model)`), or a
    callable(model) -> dict; logp_fn(model_call, fast_prefixed, inputs, actions) -> (logp [N,T], ent [N,T]) evaluates
    the policy with the fast parameters substituted (see `selfeval.fast_policy.cam_logp_fn` for the CAM decision
    path); reward_fn(**reward_inputs) -> r [N,T] graph-connected to psi (default: reward_model.reward_psi on flat
    inputs, see `selfeval.fast_policy.cam_reward_fn`)."""

    def __init__(self, model: nn.Module, reward_model: nn.Module, *, logp_fn: Callable, reward_fn: Callable,
                 fast_params=None, inner_lr: float = 0.1, K: int = 2, outer_lr: float = 1e-3,
                 entropy_coef: float = 0.0, meta_clip: float = 1.0, normalize_inner_adv: bool = True):
        self.model, self.reward_model = model, reward_model
        fp = fast_params(model) if callable(fast_params) else (fast_params if fast_params is not None
                                                                 else default_fast_params(model))
        self.fast = dict(fp)
        if not self.fast:
            raise ValueError("MetaRewardTrainer: empty fast parameter set")
        ids = {id(p) for p in self.fast.values()}
        psi_ids = {id(p) for p in reward_model.meta_parameters()}
        if ids & psi_ids:
            raise ValueError("fast parameters and psi overlap")
        self._call = _Call(model)
        names = set(self.fast)
        self._logp_user = logp_fn

        def lp(fast, inputs, actions, action_mask):
            return logp_fn(self._call, {"m." + n: t for n, t in fast.items()}, inputs, actions)

        self.mrl = MetaRewardLearner(model, reward_model, inner_lr=inner_lr, K=K,
                                     fast_param_filter=lambda n: n in names, outer_lr=outer_lr,
                                     entropy_coef=entropy_coef, meta_clip=meta_clip,
                                     normalize_inner_adv=normalize_inner_adv, logp_fn=lp, reward_fn=reward_fn,
                                     meta_params=list(reward_model.meta_parameters()))
        if set(self.mrl.fast_names) != names:
            raise ValueError("fast parameter names must be parameters of the model")
        self.n_fast = sum(p.numel() for p in self.fast.values())
        self.n_psi = sum(p.numel() for p in self.mrl.meta_params)
        self.log: list = []

    @property
    def version(self) -> int:
        return int(getattr(self.reward_model, "version", 0))

    def start(self) -> dict:
        return self.mrl.start()

    def inner_step(self, fast: dict, batch: InnerBatch, on_policy: bool = True) -> tuple:
        return self.mrl.inner_step(fast, batch, create_graph=True, on_policy=on_policy)

    def inner_update(self, batches: Sequence[InnerBatch]) -> dict:
        return self.mrl.inner_update(batches)

    @contextlib.contextmanager
    def act_with(self, fast: dict):
        """Act with phi = detached(fast): the values are copied into the model's parameters (weights epoch bumped)
        and the slow values restored on exit."""
        saved = {n: p.detach().clone() for n, p in self.fast.items()}
        with torch.no_grad():
            for n, p in self.fast.items():
                p.copy_(fast[n].detach())
        if hasattr(self.model, "bump_weights_epoch"):
            self.model.bump_weights_epoch()
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in self.fast.items():
                    p.copy_(saved[n])
            if hasattr(self.model, "bump_weights_epoch"):
                self.model.bump_weights_epoch()

    def outer_step(self, fast: dict, query: QueryBatch, score_baseline: Optional[float] = None) -> dict:
        before = [p.detach().clone() for p in self.mrl.meta_params]
        v0 = self.version
        st = self.mrl.outer_step(fast, query, score_baseline=score_baseline)
        st["psi_change"] = math.sqrt(sum(float((p.detach() - b).pow(2).sum())
                                         for p, b in zip(self.mrl.meta_params, before)))
        st["psi_version_before"], st["psi_version"] = v0, self.version
        st["n_fast"], st["n_psi"] = self.n_fast, self.n_psi
        self.log.append(st)
        return st

    def commit(self, fast: dict) -> None:
        """Consolidate phi_K into the slow weights (plan §2: act with phi_K, then consolidate)."""
        self.mrl.commit(fast)
        if hasattr(self.model, "bump_weights_epoch"):
            self.model.bump_weights_epoch()

    def state_dict(self) -> dict:
        return {"opt": self.mrl.opt.state_dict(), "u_mean": self.mrl.u_mean, "history": list(self.mrl.history)}

    def load_state_dict(self, d: dict) -> None:
        self.mrl.opt.load_state_dict(d["opt"])
        self.mrl.u_mean = d.get("u_mean")
        self.mrl.history = list(d.get("history", []))
