"""cam_live.reward.meta — the differentiable meta-gradient engine used at CAM scale (spec 7.3, 7.4, 7.7).

    inner   theta_{k+1} = theta_k - alpha * grad_theta L_inner(theta_k; psi, D_support_k)        k = 0 .. K-1
    outer   J(psi) = E[ U_verified(D_query; theta_K(psi)) ], estimated by the policy-gradient surrogate
            sum_i w_i * log pi_{theta_K}(a_i | x_i) * (U_i - baseline_i)                          (maximised)
            + [ sum_k sum_j log pi_{theta_k}(a_j | x_j) ] * (J_hat - c)       inner-data score term, when the
                                                             support actions were sampled from pi_{theta_k}

``theta`` and ``psi`` are plain ``{name: tensor}`` dictionaries (the adaptable subset of the learner and the
reward group); the model is reached only through the two callables the caller supplies, typically built with
``torch.func.functional_call`` over cached latents (:func:`functional_logp`, :func:`functional_reward`), so the
unroll never touches the backbone. The environment is never differentiated: verified utilities are constants.

Outer modes
    on_policy    query actions were sampled from pi_{theta_K} after the inner update
    importance   query actions were logged under another policy with recorded propensities: clipped weights
                 pi_{theta_K}/mu and an explicit support check. Missing or invalid propensities, propensities
                 below the floor, or too small an effective sample size return ``supported=False`` and *no*
                 gradient — never a silently wrong estimate.

Unroll modes (``mode=``)
    "unroll"       full second-order graph through the K steps (exact for the window)
    "checkpoint"   the same gradient by a reverse sweep that rebuilds one inner step at a time (adjoint method):
                   memory of one step instead of K, one extra inner forward/backward per step
    "first_order"  drops the inner Hessian chain (d theta_{k+1} / d theta_k := I) and keeps only the mixed
                   derivative d^2 L / d theta d psi of each step; biased, for comparison only

The window is truncated: theta_0 is a constant (gradients never flow into earlier windows). K in {1, 2, 4, 8} is
the configured range; any K >= 1 works. Everything is dtype-agnostic (float64 for the finite-difference tests).

Structure follows superlearner/cam_unified/selfeval/metagrad.py at commit 3968583 (two-phase inner/outer, score
term), rewritten around parameter dictionaries; leave-one-out baselines, importance mode with refusal, adjoint
checkpointing and parallel-lifetime bookkeeping are new.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.func import functional_call

from .integrity import scale_normalise

Params = Dict[str, Tensor]
InnerLoss = Callable[[Params, Params, Any], Tensor]      # (theta, psi, support batch) -> scalar loss
LogProbFn = Callable[[Params, Any], Tensor]              # (theta, batch) -> [n] log pi_theta(logged action)

ESTIMATOR_ON_POLICY = "on_policy_query"
ESTIMATOR_IMPORTANCE = "importance_weighted"
MODES = ("unroll", "checkpoint", "first_order")
SUPPORTED_K = (1, 2, 4, 8)

__all__ = ["Params", "QueryData", "InnerTrace", "OuterEstimate", "MetaGradResult", "MetaGradient",
           "policy_gradient_inner_loss", "functional_logp", "functional_reward", "discounted_returns",
           "finite_difference", "flat", "ESTIMATOR_ON_POLICY", "ESTIMATOR_IMPORTANCE", "MODES", "SUPPORTED_K"]


# ============================================================================================================ data
@dataclass
class QueryData:
    """Query decisions with their independently verified utilities.

    batch            whatever ``logp_fn`` needs to score the logged query actions
    utility          [n] verified utility credited to each query decision (signed verifier outcomes only)
    baseline         [n] or scalar; None = leave-one-out mean over the *other independent groups*
    behaviour_logp   [n] logged log-propensity of each query action (required in importance mode)
    weights          [n] row weights; None = uniform within each lifetime
    groups           [n] long: independent-outcome group of each row (rows sharing an outcome share a group)
    lifetime         [n] long: index of the independent lifetime a row belongs to (None = one lifetime)
    verified         [n] bool: the utility of the row comes from a signature-checked verifier outcome
    """
    batch: Any
    utility: Tensor
    baseline: Optional[Union[Tensor, float]] = None
    behaviour_logp: Optional[Tensor] = None
    weights: Optional[Tensor] = None
    groups: Optional[Tensor] = None
    lifetime: Optional[Tensor] = None
    verified: Optional[Tensor] = None


@dataclass
class InnerTrace:
    """Result of an inner update. ``thetas[k]`` are graph-connected to psi only when ``has_graph``."""
    thetas: List[Params]
    psi: Params
    batches: List[Any]
    on_policy: List[bool]
    inner_lr: float
    mode: str
    has_graph: bool
    score: Optional[Tensor] = None           # [n_lifetimes], graph-connected (unroll / first_order)
    inner_losses: List[float] = field(default_factory=list)

    @property
    def theta_K(self) -> Params:
        return self.thetas[-1]

    @property
    def K(self) -> int:
        return len(self.thetas) - 1

    def detached(self) -> Params:
        """theta_K as constants: what the agent acts with while collecting query experience."""
        return {n: v.detach() for n, v in self.thetas[-1].items()}


@dataclass
class OuterEstimate:
    supported: bool
    reason: str
    estimator: str
    surrogate: Optional[Tensor] = None       # scalar to maximise, graph into theta_K
    objective: Optional[Tensor] = None       # [n_lifetimes] estimated verified utility J_hat (constant)
    n: int = 0
    n_groups: int = 0
    ess: float = 0.0
    max_weight: float = 1.0
    clipped_fraction: float = 0.0


@dataclass
class MetaGradResult:
    supported: bool
    reason: str
    estimator: str
    grads: Params = field(default_factory=dict)      # dJ/dpsi: the ascent direction, one entry per psi tensor
    grad_norm: float = 0.0
    objective: float = 0.0
    n: int = 0
    n_groups: int = 0
    ess: float = 0.0
    max_weight: float = 1.0
    clipped_fraction: float = 0.0
    K: int = 0
    inner_lr: float = 0.0
    mode: str = ""
    inner_losses: List[float] = field(default_factory=list)
    score_term_used: bool = False

    def record_fields(self) -> Dict[str, Any]:
        """Fields of ``contracts.MetaUpdateRecord`` this result determines."""
        return {"n_query": self.n, "n_independent_outcomes": self.n_groups, "inner_steps": self.K,
                "inner_lr": self.inner_lr, "outer_objective": self.objective, "meta_grad_norm": self.grad_norm,
                "estimator": self.estimator, "max_importance_weight": self.max_weight,
                "skipped_reason": "" if self.supported else self.reason,
                "diagnostics": {"ess": self.ess, "clipped_fraction": self.clipped_fraction,
                                "score_term_used": float(self.score_term_used)}}


# ========================================================================================================= helpers
def flat(params: Mapping[str, Tensor]) -> Tensor:
    return torch.cat([v.reshape(-1) for v in params.values()]) if params else torch.zeros(0)


def _lifetime_of(batch: Any) -> Optional[Tensor]:
    if isinstance(batch, Mapping):
        return batch.get("lifetime")
    return getattr(batch, "lifetime", None)


def _segment_sum(values: Tensor, index: Tensor, n: int) -> Tensor:
    return torch.zeros(n, dtype=values.dtype, device=values.device).index_add(0, index, values)


def discounted_returns(r: Tensor, mask: Optional[Tensor] = None, gamma: Optional[Tensor] = None) -> Tensor:
    """G_t = r_t + gamma_t * G_{t+1} along the last dimension (graph-preserving). ``gamma`` [.., T] is the
    discount from step t to t+1 (for elapsed time dt: exp(-kappa * dt)); masked steps contribute nothing."""
    T = r.shape[-1]
    m = torch.ones_like(r) if mask is None else mask.to(r.dtype)
    g = torch.ones_like(r) if gamma is None else gamma.to(r.dtype)
    out, G = [], torch.zeros_like(r[..., 0])
    for t in reversed(range(T)):
        G = (r[..., t] + g[..., t] * G) * m[..., t]
        out.append(G)
    return torch.stack(out[::-1], -1)


def functional_logp(policy: nn.Module, inputs: str = "x", actions: str = "a", mask: str = "a_mask"
                    ) -> LogProbFn:
    """``logp_fn`` for a module mapping ``batch[inputs]`` to logits [n, A]: log pi_theta(batch[actions]).
    ``theta`` may hold any subset of the module's parameters (the rest keep their current values). If its tensors
    carry one extra leading dimension R the call is vmapped over R independent lifetimes and ``batch[inputs]`` /
    ``batch[actions]`` must be [R, n, ...] / [R, n]; the result is flattened to [R * n]."""
    ref = dict(policy.named_parameters())

    def logp(theta: Params, batch: Mapping[str, Any]) -> Tensor:
        x, a = batch[inputs], batch[actions]
        name = next(iter(theta))
        if theta[name].dim() == ref[name].dim() + 1:
            logits = torch.vmap(lambda th, xi: functional_call(policy, th, (xi,)))(theta, x)
        else:
            logits = functional_call(policy, theta, (x,))
        am = batch.get(mask) if isinstance(batch, Mapping) else None
        if am is not None:
            logits = logits.masked_fill(~am.bool(), float("-inf"))
        lp = torch.log_softmax(logits, -1).gather(-1, a.long().unsqueeze(-1)).squeeze(-1)
        return lp.reshape(-1)
    return logp


def functional_reward(reward: nn.Module, inputs: str = "reward_inputs") -> Callable[[Params, Any], Tensor]:
    """``reward_fn`` for a reward module: ``functional_call(reward, psi, kwargs=batch[inputs])`` flattened to [n].
    ``batch[inputs]`` is a kwargs mapping (or a tuple of positional arguments)."""
    def r(psi: Params, batch: Mapping[str, Any]) -> Tensor:
        args = batch[inputs]
        out = functional_call(reward, psi, (), dict(args)) if isinstance(args, Mapping) \
            else functional_call(reward, psi, tuple(args))
        return out.reshape(-1)
    return r


def policy_gradient_inner_loss(logp_fn: LogProbFn, reward_fn: Callable[[Params, Any], Tensor], *,
                               value_fn: Optional[Callable[[Params, Any], Tensor]] = None,
                               value_coef: float = 0.5,
                               entropy_fn: Optional[Callable[[Params, Any], Tensor]] = None,
                               entropy_coef: float = 0.0, normalise: bool = True,
                               loo_baseline: bool = True) -> InnerLoss:
    """L_inner(theta; psi, D) = -mean_i log pi_theta(a_i|x_i) * A_psi,i + value loss on intrinsic returns
    - entropy bonus.

    ``reward_fn(psi, batch)`` returns the intrinsic return of each support row [n] (a function of psi).
    normalise     standardise the intrinsic returns within the batch (per lifetime), so that multiplying or
                  shifting the reward output cannot change the update (reward-magnitude inflation is a no-op)
    value_fn      V_theta(x) [n]: baseline (detached in the advantage) and regression target = intrinsic return
    loo_baseline  without a value function and without normalisation: leave-one-out mean baseline (in the graph)
    """
    def loss(theta: Params, psi: Params, batch: Any, theta_sg: Optional[Params] = None) -> Tensor:
        lp = logp_fn(theta, batch)
        g = reward_fn(psi, batch)
        life = _lifetime_of(batch)
        if normalise:
            g = scale_normalise(g, group=life)
        n_life = 1 if life is None else int(life.max()) + 1
        per_life = (lambda v: v.mean()) if life is None else \
            (lambda v: (_segment_sum(v, life, n_life) / _segment_sum(torch.ones_like(v), life, n_life)).sum())
        if value_fn is not None:
            v = value_fn(theta, batch)
            # the baseline is fixed for the inner derivative but is still a function of theta_k(psi)
            adv = g - (v.detach() if theta_sg is None else value_fn(theta_sg, batch))
            out = per_life(-lp * adv) + value_coef * per_life(0.5 * (v - g) ** 2)
        else:
            adv = g
            if loo_baseline and not normalise:
                if life is None:
                    n = g.shape[0]
                    adv = g - (g.sum() - g) / max(n - 1, 1)
                else:
                    cnt = _segment_sum(torch.ones_like(g), life, n_life)[life]
                    adv = g - (_segment_sum(g, life, n_life)[life] - g) / (cnt - 1).clamp_min(1.0)
            out = per_life(-lp * adv)
        if entropy_fn is not None and entropy_coef:
            out = out - entropy_coef * per_life(entropy_fn(theta, batch))
        return out
    loss.stopgrad_aware = value_fn is not None
    return loss


def finite_difference(objective: Callable[[Params], float], psi: Params, *, eps: float = 1e-5,
                      coords: Optional[Sequence[int]] = None) -> Tensor:
    """Central differences of ``objective(psi)`` over the flattened psi (``coords``: flat indices to probe)."""
    names, shapes = list(psi), [v.shape for v in psi.values()]
    base = flat(psi).detach().clone()

    def unflat(v: Tensor) -> Params:
        out, i = {}, 0
        for n, s in zip(names, shapes):
            k = int(math.prod(s))
            out[n] = v[i:i + k].view(s)
            i += k
        return out
    g = torch.zeros_like(base)
    for i in (range(base.numel()) if coords is None else coords):
        old = float(base[i])
        base[i] = old + eps
        jp = objective(unflat(base.clone()))
        base[i] = old - eps
        jm = objective(unflat(base.clone()))
        base[i] = old
        g[i] = (jp - jm) / (2 * eps)
    return g


# ========================================================================================================== engine
class MetaGradient:
    """Truncated meta-gradient of an intrinsic reward through K functional inner updates.

    Two-phase use (query experience is collected between the phases):

        trace = mg.inner_update(theta, psi, support_batches, inner_lr, K, create_graph=True, on_policy=...)
        ... act with trace.detached(), wait for verified outcomes, build QueryData ...
        res = mg.meta_gradient(trace, query, estimator="on_policy")     # res.grads = dJ/dpsi, or unsupported
        mg.apply(reward_optimizer, reward_module_or_params, res, clip=cfg.optim.clip_reward)
    """

    def __init__(self, inner_loss: InnerLoss, logp_fn: LogProbFn, *, inner_lr: float = 1e-2, K: int = 2,
                 mode: str = "unroll", weight_clip: float = 10.0, min_ess: float = 4.0,
                 min_ess_fraction: float = 0.05, min_propensity: float = 1e-6, min_independent: int = 1,
                 self_normalise: bool = False, ema: float = 0.1,
                 grad_transform: Optional[Callable[[Sequence[str], Sequence[Optional[Tensor]]],
                                                   Sequence[Optional[Tensor]]]] = None):
        """``grad_transform(names, grads)`` is applied to the inner gradient inside the graph (e.g. a per-group
        norm clip); it is part of the function that is differentiated."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.inner_loss, self.logp_fn = inner_loss, logp_fn
        self.inner_lr, self.K, self.mode = float(inner_lr), int(K), mode
        self.weight_clip, self.min_ess, self.min_ess_fraction = float(weight_clip), float(min_ess), min_ess_fraction
        self.min_propensity, self.min_independent = float(min_propensity), int(min_independent)
        self.self_normalise = self_normalise
        self.ema = ema
        self.grad_transform = grad_transform
        self.stopgrad_aware = bool(getattr(inner_loss, "stopgrad_aware", False))
        self._alias: Optional[Params] = None
        self.objective_ema: Optional[float] = None       # running mean of J_hat: default score-term baseline

    # ---------------------------------------------------------------------------------------------- inner
    def inner_step(self, theta: Params, psi: Params, batch: Any, inner_lr: Optional[float] = None, *,
                   create_graph: bool = True, first_order: bool = False,
                   theta_sg: Optional[Params] = None) -> Tuple[Params, Tensor]:
        """theta' = theta - alpha * grad_theta L_inner(theta; psi, batch). Returns (theta', loss).

        ``theta_sg``: tensors with the values of ``theta`` that are *not* the variables of the inner gradient
        (see :func:`policy_gradient_inner_loss`). An inner loss marked ``stopgrad_aware`` receives them as
        ``theta_sg=`` and uses them wherever it holds a quantity fixed for the inner derivative (the value
        baseline): the update rule still depends on them, so the unroll differentiates through them. A plain
        ``.detach()`` there would silently drop that path for K >= 2."""
        lr = self.inner_lr if inner_lr is None else float(inner_lr)
        at = {n: v.detach().requires_grad_(True) for n, v in theta.items()} if first_order else theta
        with torch.enable_grad():
            if self.stopgrad_aware:
                sg = theta_sg if (theta_sg is not None and not first_order) else {n: v.detach() for n, v in at.items()}
                L = self.inner_loss(at, psi, batch, theta_sg=sg)
            else:
                L = self.inner_loss(at, psi, batch)
            grads = torch.autograd.grad(L, list(at.values()), create_graph=create_graph, allow_unused=True)
            if self.grad_transform is not None:
                grads = self.grad_transform(list(at), grads)
        new = {n: (v if g is None else v - lr * g) for (n, v), g in zip(theta.items(), grads)}
        # theta' once more as separate graph nodes with the same parents (the next step's ``theta_sg``)
        self._alias = {n: (v if g is None else v - lr * g) for (n, v), g in zip(theta.items(), grads)} \
            if (self.stopgrad_aware and create_graph and not first_order) else None
        return new, L

    def inner_update(self, theta: Params, psi_params: Params,
                     support_batches: Union[Sequence[Any], Callable[[int, Params], Any]],
                     inner_lr: Optional[float] = None, K: Optional[int] = None, create_graph: bool = True,
                     on_policy: Union[bool, Sequence[bool]] = False, mode: Optional[str] = None,
                     n_lifetimes: int = 1) -> InnerTrace:
        """K inner steps from ``theta`` (treated as a constant: truncated window), step k on
        ``support_batches[k % len]`` — or on ``support_batches(k, detached theta_k)`` when a callable is given
        (on-policy collection between the steps). ``on_policy[k]`` says batch k was sampled from pi_{theta_k}; its
        log-probability then enters the outer objective as the inner-data score term. ``create_graph=False``
        (or mode "checkpoint") stores values only; "checkpoint" still yields the exact gradient later."""
        mode = self.mode if mode is None else mode
        if mode not in MODES:
            raise ValueError(mode)
        K = self.K if K is None else int(K)
        if K < 1:
            raise ValueError("K must be >= 1")
        lr = self.inner_lr if inner_lr is None else float(inner_lr)
        flags = [bool(on_policy)] * K if isinstance(on_policy, bool) else [bool(f) for f in on_policy]
        if len(flags) != K:
            raise ValueError("on_policy needs one flag per inner step")
        graph = create_graph and mode != "checkpoint"
        th = {n: v.detach().requires_grad_(True) for n, v in theta.items()}
        thetas, losses, used = [th], [], []
        score = None
        alias: Optional[Params] = None                         # same values as th, a different graph node
        for k in range(K):
            batch = support_batches(k, {n: v.detach() for n, v in th.items()}) if callable(support_batches) \
                else support_batches[k % len(support_batches)]
            used.append(batch)
            if graph and flags[k] and k >= 1:                  # theta_0 is a constant: its score has no psi path
                lp = self.logp_fn(th, batch)
                life = _lifetime_of(batch)
                s = lp.sum()[None] if life is None else _segment_sum(lp, life, n_lifetimes)
                score = s if score is None else score + s
            th, L = self.inner_step(th, psi_params, batch, lr, create_graph=graph,
                                    first_order=(mode == "first_order"), theta_sg=alias)
            alias = self._alias
            if not graph:
                th = {n: v.detach().requires_grad_(True) for n, v in th.items()}
                alias = None
            thetas.append(th)
            losses.append(float(L.detach()))
        return InnerTrace(thetas=thetas, psi=psi_params, batches=used, on_policy=flags, inner_lr=lr, mode=mode,
                          has_graph=graph, score=score, inner_losses=losses)

    # ---------------------------------------------------------------------------------------------- outer
    def outer_surrogate(self, theta_K: Params, query: QueryData, estimator: str = "on_policy") -> OuterEstimate:
        """Policy-gradient surrogate of the verified query utility under pi_{theta_K} (to be maximised)."""
        name = {"on_policy": ESTIMATOR_ON_POLICY, "importance": ESTIMATOR_IMPORTANCE}.get(estimator)
        if name is None:
            raise ValueError("estimator must be 'on_policy' or 'importance'")
        lp = self.logp_fn(theta_K, query.batch)
        n = int(lp.shape[0])
        U = query.utility.detach().to(lp.dtype).reshape(-1)
        if n == 0 or U.shape[0] != n:
            return OuterEstimate(False, "empty_or_misaligned_query", name, n=n)
        if not bool(torch.isfinite(U).all()):
            return OuterEstimate(False, "non_finite_utility", name, n=n)
        if query.verified is not None and not bool(query.verified.bool().all()):
            return OuterEstimate(False, "unverified_outcome_in_query", name, n=n)
        life = torch.zeros(n, dtype=torch.long, device=lp.device) if query.lifetime is None else query.lifetime
        n_life = int(life.max()) + 1
        grp_raw = torch.arange(n, device=lp.device) if query.groups is None else query.groups
        _, grp = torch.unique(torch.stack([life, grp_raw]), dim=1, return_inverse=True)
        n_grp = int(grp.max()) + 1
        if n_grp < self.min_independent * n_life:
            return OuterEstimate(False, f"insufficient_independent_outcomes:{n_grp}<{self.min_independent}",
                                 name, n=n, n_groups=n_grp)
        ones = torch.ones(n, dtype=lp.dtype, device=lp.device)
        w = ones / _segment_sum(ones, life, n_life)[life] if query.weights is None \
            else query.weights.detach().to(lp.dtype).reshape(-1)
        ess, max_w, clipped = float(n), 1.0, 0.0
        if estimator == "importance":
            if query.behaviour_logp is None:
                return OuterEstimate(False, "missing_propensities", name, n=n, n_groups=n_grp)
            blp = query.behaviour_logp.detach().to(lp.dtype).reshape(-1)
            if blp.shape[0] != n or not bool(torch.isfinite(blp).all()) or bool((blp > 1e-9).any()):
                return OuterEstimate(False, "invalid_propensities", name, n=n, n_groups=n_grp)
            if bool((blp < math.log(self.min_propensity)).any()):
                return OuterEstimate(False, "propensity_below_floor", name, n=n, n_groups=n_grp)
            rho = torch.exp(lp.detach() - blp)
            ess_l = _segment_sum(rho, life, n_life) ** 2 / _segment_sum(rho ** 2, life, n_life).clamp_min(1e-300)
            cnt = _segment_sum(ones, life, n_life)
            ess, max_w = float(ess_l.min()), float(rho.max())
            if bool((ess_l < torch.maximum(torch.full_like(cnt, self.min_ess), self.min_ess_fraction * cnt)).any()):
                return OuterEstimate(False, f"effective_sample_size_too_small:{ess:.2f}", name, n=n,
                                     n_groups=n_grp, ess=ess, max_weight=max_w)
            iw = rho.clamp(max=self.weight_clip)
            clipped = float((rho > self.weight_clip).to(lp.dtype).mean())
            if self.self_normalise:
                iw = iw / _segment_sum(w * iw, life, n_life)[life].clamp_min(1e-300)
            w = w * iw
        if query.baseline is None:                       # leave-one-out over the other independent groups
            w0 = ones / _segment_sum(ones, life, n_life)[life]
            s_l, c_l = _segment_sum(w0 * U, life, n_life)[life], _segment_sum(w0, life, n_life)[life]
            s_g, c_g = _segment_sum(w0 * U, grp, n_grp)[grp], _segment_sum(w0, grp, n_grp)[grp]
            rest = c_l - c_g
            base = torch.where(rest > 1e-12, (s_l - s_g) / rest.clamp_min(1e-12), torch.zeros_like(U))
        else:
            base = torch.as_tensor(query.baseline, dtype=lp.dtype, device=lp.device).detach().expand(n)
        surrogate = (w * lp * (U - base)).sum()
        return OuterEstimate(True, "", name, surrogate=surrogate, objective=_segment_sum(w * U, life, n_life),
                             n=n, n_groups=n_grp, ess=ess, max_weight=max_w, clipped_fraction=clipped)

    def _score_baseline(self, J: Tensor, score_baseline) -> Tensor:
        if score_baseline is not None:
            return torch.as_tensor(score_baseline, dtype=J.dtype, device=J.device).expand_as(J)
        if J.shape[0] > 1:                               # leave-one-out over parallel lifetimes
            return (J.sum() - J) / (J.shape[0] - 1)
        return torch.full_like(J, self.objective_ema if self.objective_ema is not None else 0.0)

    def meta_gradient(self, trace: InnerTrace, query: QueryData, estimator: str = "on_policy", *,
                      score_baseline: Optional[Union[float, Tensor]] = None) -> MetaGradResult:
        """dJ/dpsi for the window in ``trace`` on verified ``query`` data, or an unsupported result with no
        gradient. ``score_baseline`` c of the inner-data score term: any constant independent of the support
        actions is unbiased (default: leave-one-out over parallel lifetimes, else the running mean of J_hat)."""
        psi_names, psi_list = list(trace.psi), list(trace.psi.values())
        base = dict(K=trace.K, inner_lr=trace.inner_lr, mode=trace.mode, inner_losses=trace.inner_losses)
        use_adjoint = not trace.has_graph
        if use_adjoint and trace.mode != "checkpoint":
            return MetaGradResult(False, "inner_update_has_no_graph", estimator, **base)
        theta_K = {n: v.detach().requires_grad_(True) for n, v in trace.theta_K.items()} if use_adjoint \
            else trace.theta_K
        est = self.outer_surrogate(theta_K, query, estimator)
        common = dict(n=est.n, n_groups=est.n_groups, ess=est.ess, max_weight=est.max_weight,
                      clipped_fraction=est.clipped_fraction, **base)
        if not est.supported:
            return MetaGradResult(False, est.reason, est.estimator, **common)
        J = est.objective
        coef = (J - self._score_baseline(J, score_baseline)).detach()        # [n_lifetimes]
        any_score = any(f and k >= 1 for k, f in enumerate(trace.on_policy))
        if not use_adjoint:
            total = est.surrogate
            if trace.score is not None:
                total = total + (trace.score * coef).sum()
            if not total.requires_grad:
                grads = [None] * len(psi_list)
            else:
                grads = torch.autograd.grad(total, psi_list, allow_unused=True)
        else:
            grads = self._adjoint(trace, theta_K, est.surrogate, coef, psi_list)
        out = {n: (torch.zeros_like(p) if g is None else g.detach()) for n, p, g in zip(psi_names, psi_list, grads)}
        norm = math.sqrt(sum(float((g ** 2).sum()) for g in out.values()))
        Jm = float(J.mean())
        self.objective_ema = Jm if self.objective_ema is None else (1 - self.ema) * self.objective_ema + self.ema * Jm
        return MetaGradResult(True, "", est.estimator, grads=out, grad_norm=norm, objective=Jm,
                              score_term_used=any_score, **common)

    def _adjoint(self, trace: InnerTrace, theta_K: Params, surrogate: Tensor, coef: Tensor,
                 psi_list: List[Tensor]) -> List[Optional[Tensor]]:
        """Reverse sweep over the stored theta_k values: one inner step is rebuilt, differentiated and freed at a
        time. Identical to differentiating the full unroll."""
        names = list(theta_K)
        zero_if_none = lambda gs, like: [torch.zeros_like(v) if g is None else g for g, v in zip(gs, like)]
        lam = zero_if_none(torch.autograd.grad(surrogate, list(theta_K.values()), allow_unused=True),
                           theta_K.values())
        gpsi: List[Optional[Tensor]] = [None] * len(psi_list)
        n_life = coef.shape[0]
        for k in reversed(range(trace.K)):
            th = {n: trace.thetas[k][n].detach().requires_grad_(True) for n in names}
            ths = list(th.values())
            with torch.enable_grad():
                if self.stopgrad_aware:
                    th2 = {n: trace.thetas[k][n].detach().requires_grad_(True) for n in names}
                    ths2 = list(th2.values())
                    L = self.inner_loss(th, trace.psi, trace.batches[k], theta_sg=th2)
                else:
                    ths2 = []
                    L = self.inner_loss(th, trace.psi, trace.batches[k])
                g = torch.autograd.grad(L, ths, create_graph=True, allow_unused=True)
                if self.grad_transform is not None:
                    g = self.grad_transform(names, g)
                dot = sum((gi * li).sum() for gi, li in zip(g, lam) if gi is not None)
                if not isinstance(dot, Tensor) or not dot.requires_grad:
                    vjp = [None] * (len(ths) + len(psi_list))
                else:
                    full = torch.autograd.grad(dot, ths + ths2 + psi_list, allow_unused=True)
                    v1, v2, vp = full[:len(ths)], full[len(ths):len(ths) + len(ths2)], full[len(ths) + len(ths2):]
                    if ths2:                                   # both are theta_k: add the two partial derivatives
                        v1 = [a if c is None else (c if a is None else a + c) for a, c in zip(v1, v2)]
                    vjp = list(v1) + list(vp)
                lam = [l if v is None else l - trace.inner_lr * v for l, v in zip(lam, vjp[:len(ths)])]
                for i, v in enumerate(vjp[len(ths):]):
                    if v is not None:
                        gpsi[i] = -trace.inner_lr * v if gpsi[i] is None else gpsi[i] - trace.inner_lr * v
                if trace.on_policy[k] and k >= 1:
                    lp = self.logp_fn(th, trace.batches[k])
                    life = _lifetime_of(trace.batches[k])
                    s = lp.sum()[None] if life is None else _segment_sum(lp, life, n_life)
                    sg = torch.autograd.grad((s * coef).sum(), ths, allow_unused=True)
                    lam = [l if v is None else l + v for l, v in zip(lam, sg)]
        return gpsi

    # ---------------------------------------------------------------------------------------------- apply
    @staticmethod
    def apply(optimizer: torch.optim.Optimizer, reward_params: Union[nn.Module, Mapping[str, Tensor]],
              result: MetaGradResult, clip: Optional[float] = None) -> Dict[str, float]:
        """One ascent step of the reward optimizer on ``result.grads`` (clipped to norm ``clip``). Nothing is
        applied — the parameters and the optimizer state stay untouched — when the result is unsupported or its
        gradient is zero or not finite (an Adam step on a zero gradient would still move psi through its moments)."""
        if not result.supported:
            return {"applied": 0.0, "grad_norm": 0.0, "clip_scale": 0.0, "reason": "unsupported"}
        if not math.isfinite(result.grad_norm) or result.grad_norm <= 0.0:
            return {"applied": 0.0, "grad_norm": float(result.grad_norm), "clip_scale": 0.0,
                    "reason": "zero_meta_gradient" if result.grad_norm == 0.0 else "non_finite_meta_gradient"}
        named = dict(reward_params.named_parameters()) if isinstance(reward_params, nn.Module) else dict(reward_params)
        if set(named) != set(result.grads):
            raise ValueError("meta-gradient names do not match the reward parameters")
        owned = {id(p) for g in optimizer.param_groups for p in g["params"]}
        if owned != {id(p) for p in named.values()}:
            raise ValueError("the reward optimizer must hold exactly the reward parameters")
        scale = 1.0 if not clip or result.grad_norm <= clip else clip / result.grad_norm
        optimizer.zero_grad(set_to_none=True)
        for n, p in named.items():
            p.grad = (-scale * result.grads[n]).to(p.dtype)          # optimizers minimise; we ascend J
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return {"applied": 1.0, "grad_norm": result.grad_norm, "clip_scale": scale}
