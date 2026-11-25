"""Belief-aware multistep imagination (spec 7.1).

From the actual state S_t = (Z_t, B_t, G_t) of B real environments, ``Imagination.rollout`` expands every
environment into ``n_branches`` contemplated first actions x ``n_samples`` stochastic futures (Bb = B*n*s rows).
Each row owns an independent clone of the FastState. Per imagined step:

1. BELIEF tokens from the row's cloned FastState (``belief_tokens_fn``);
2. z ~ prior, Z^_{k+1} = WorldModel.prior(Z^_k, A_k, B_k, dt_k) (dt from the action's declared duration when given,
   else the model's predicted duration);
3. outcome queries for the contemplated action (``outcome_queries_fn``), the particle-conditioned predictive
   distribution of the outcome head, a sampled imagined outcome (particle k ~ w, then y ~ p(y | Z^, m^k), or the
   marginal head);
4. the cloned belief is updated with the imagined outcome (``belief_update_fn`` -- workstream B's
   ``BeliefModuleP.update``), so later decisions see the revised imagined belief;
5. predicted reward (world-model head or ``reward_fn``), continuation probability and semi-Markov discount
   exp(-kappa dt);
6. the reasoner re-runs on the imagined state (``reason_fn``), the value head scores it (``value_fn``), and the policy
   chooses the next action from the *imagined* belief (``policy_fn``);
7. an epistemic / model-error estimate: weighted particle disagreement of the predicted outcome means plus the
   predictive outcome variance (log1p-compressed, mean over observed queries); without queries, the normalised
   entropy of the prior latent.

The return is sum_k disc_k r_k + disc_h V(R_h) (disc includes continuation); with ``uncertainty_coef`` > 0 the
discounted model error is subtracted. Imagined steps are ``Provenance.IMAGINED``; this module has no access to any
archive or environment, and it never mutates the caller's Z or FastState (every row works on copies).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import torch

from ..interfaces import (Action, FastState, ImaginedStep, ImaginedTrajectory, OutcomeBatch, Provenance, TokenKind,
                          TokenSet)
from ._util import cat_tokens, masked_mean


# ------------------------------------------------------------------------------------------------ data types

@dataclass
class Candidates:
    """Contemplated first actions: ``actions`` [B][n] Action and their ACTION tokens [B*n, Na, D] (row b*n+i)."""
    actions: list
    tokens: TokenSet

    @property
    def n(self) -> int:
        return len(self.actions[0])

    @property
    def B(self) -> int:
        return len(self.actions)


@dataclass
class PolicyContext:
    """Everything the policy may condition on for the next imagined decision (per imagined row)."""
    R: Any                       # [Bb,S,D] reasoned state
    Z: Any                       # [Bb,S,D] imagined latent state
    goal: Optional[TokenSet]
    belief: Optional[TokenSet]   # BELIEF tokens of the revised imagined belief
    fast: Optional[FastState]    # the row's imagined FastState (a clone)
    step: int                    # index of the decision being chosen (1 = the decision after the first action)
    prev_actions: list           # [Bb] Action
    branch_of: Any               # [Bb]
    first_candidate: Any         # [Bb]


@dataclass
class ImaginationHooks:
    """Callbacks into the other components. Only ``policy_fn`` is required for multi-step rollouts.

    policy_fn(ctx: PolicyContext) -> (list[Action] [Bb], TokenSet ACTION tokens [Bb, Na, D])
    outcome_queries_fn(Z_prev, actions, act_tokens, step) -> OutcomeBatch [Bb,Q] (y unused; y_mask = which
        outcomes the action would make observable, default all real queries)
    belief_update_fn(fast, Z_prev, act_tokens, OutcomeBatch) -> FastState | (FastState, info)
    belief_tokens_fn(fast) -> TokenSet [Bb, K+1, D]
    reason_fn(Z, ctx TokenSet) -> R | (R, info)
    value_fn(R, goal, belief_tokens) -> [Bb]
    reward_fn(Z_hat, imagined OutcomeBatch, outcome dist, goal) -> [Bb]  (default: world-model reward head)
    duration_fn(actions) -> [Bb] days or None (default: Action.expected_duration when every action declares it)
    """
    policy_fn: Optional[Callable] = None
    outcome_queries_fn: Optional[Callable] = None
    belief_update_fn: Optional[Callable] = None
    belief_tokens_fn: Optional[Callable] = None
    reason_fn: Optional[Callable] = None
    value_fn: Optional[Callable] = None
    reward_fn: Optional[Callable] = None
    duration_fn: Optional[Callable] = None


@dataclass
class WorldImaginedStep(ImaginedStep):
    """ImaginedStep plus the quantities planners and imagination training need."""
    dt: Any = None                   # [Bb] imagined elapsed days
    cont: Any = None                 # [Bb] continuation probability
    discount: Any = None             # [Bb] cumulative discount applied to this step's reward
    epistemic: Any = None            # [Bb] particle disagreement part of model_error (pre-compression mean)
    aleatoric: Any = None            # [Bb] outcome variance part
    outcomes: Optional[OutcomeBatch] = None   # imagined (sampled) outcomes, y = sample
    R: Any = None                    # [Bb,S,D] reasoned imagined state after the step
    logits: Any = None               # [Bb,G,C] prior latent log-probs
    provenance: Provenance = Provenance.IMAGINED


@dataclass
class WorldImaginedTrajectory(ImaginedTrajectory):
    returns_raw: Any = None          # [Bb] without the uncertainty penalty
    penalty: Any = None              # [Bb] discounted cumulative model error
    bootstrap: Any = None            # [Bb] V at the horizon
    discount_h: Any = None           # [Bb] cumulative discount at the horizon
    n_model_evals: int = 0           # world-model transition evaluations used


@dataclass
class ImagState:
    """Batched imagined state of Bb rows (all tensors are private copies)."""
    Z: Any
    fast: Optional[FastState]
    goal: Optional[TokenSet]
    belief: Optional[TokenSet]
    actions: list
    act_tokens: TokenSet
    disc: Any
    ret: Any
    penalty: Any
    elapsed: Any
    branch_of: Any
    first_candidate: Any
    value: Any = None
    R: Any = None


def fast_select(fast: Optional[FastState], idx) -> Optional[FastState]:
    """Independent copy of rows ``idx`` of a FastState (tensors cloned, channel dicts copied)."""
    if fast is None:
        return None
    s = fast.select(idx)
    return FastState(*(t.clone() for t in (s.particles, s.logw, s.pscale, s.bayes_mu, s.bayes_P, s.bayes_noise,
                                           s.bayes_n, s.surprise, s.steps)), [dict(c) for c in s.channels])


def tokens_select(ts: Optional[TokenSet], idx) -> Optional[TokenSet]:
    return None if ts is None else ts.select(idx)


def _first(x):
    return x[0] if isinstance(x, tuple) else x


def belief_ess(fast: Optional[FastState], n: int, device) -> torch.Tensor:
    if fast is None:
        return torch.ones(n, device=device)
    w = torch.softmax(fast.logw.float(), -1)
    return 1.0 / (w * w).sum(-1)


class Imagination:
    """Batched belief-aware rollouts on a ``WorldModel``. Hooks connect the belief module, reasoner, policy and value
    head of the other workstreams; no archive or environment handle exists here."""

    def __init__(self, world_model, horizon: Optional[int] = None, n_branches: Optional[int] = None,
                 n_samples: Optional[int] = None, kappa: Optional[float] = None, uncertainty_coef: float = 0.0,
                 outcome_sampling: str = "mixture", use_continue: bool = True):
        cfg = world_model.cfg
        self.wm = world_model
        self.horizon = horizon or max(cfg.imag_horizons)
        self.n_branches = n_branches or cfg.imag_branches
        self.n_samples = n_samples or cfg.imag_samples
        self.kappa = cfg.kappa if kappa is None else kappa
        self.uncertainty_coef = uncertainty_coef
        if outcome_sampling not in ("mixture", "marginal"):
            raise ValueError("outcome_sampling must be 'mixture' or 'marginal'")
        self.outcome_sampling = outcome_sampling
        self.use_continue = use_continue

    # ------------------------------------------------------------------ setup
    def init_state(self, Z, fast: Optional[FastState], goal: Optional[TokenSet], cand: Candidates, n_samples: int,
                   hooks: ImaginationHooks) -> ImagState:
        """Expand B real environments into B * cand.n * n_samples imagined rows (copies)."""
        B, n = cand.B, cand.n
        dev = Z.device
        rows = torch.arange(B * n, device=dev).repeat_interleave(n_samples)           # candidate row per imagined row
        branch_of = rows // n
        first = rows % n
        Zr = Z[branch_of].clone()
        fr = fast_select(fast, branch_of)
        gr = tokens_select(goal, branch_of)
        act = cand.tokens.select(rows)
        actions = [cand.actions[int(b)][int(i)] for b, i in zip(branch_of.tolist(), first.tolist())]
        Bb = rows.numel()
        z = torch.zeros(Bb, device=dev)
        st = ImagState(Zr, fr, gr, None, actions, act, torch.ones(Bb, device=dev), z.clone(), z.clone(), z.clone(),
                       branch_of, first)
        st.belief = hooks.belief_tokens_fn(fr) if (hooks.belief_tokens_fn is not None and fr is not None) else None
        return st

    # ------------------------------------------------------------------ one imagined step
    def _durations(self, st: ImagState, hooks: ImaginationHooks):
        if hooks.duration_fn is not None:
            d = hooks.duration_fn(st.actions)
            if d is not None:
                return torch.as_tensor(d, dtype=torch.float32, device=st.Z.device)
        if all(isinstance(a, Action) and a.expected_duration is not None for a in st.actions):
            return torch.tensor([float(a.expected_duration) for a in st.actions], device=st.Z.device)
        return None

    def step(self, st: ImagState, hooks: ImaginationHooks, k: int,
             generator: Optional[torch.Generator] = None) -> tuple:
        """Apply st.actions to every row; returns (new ImagState without next actions, WorldImaginedStep)."""
        wm = self.wm
        Bb = st.Z.shape[0]
        dev = st.Z.device
        dt = self._durations(st, hooks)
        Z_next, info = wm.prior(st.Z, st.act_tokens, st.belief, dt, sample=True, generator=generator)
        dt = info["dt"]
        fast = st.fast
        imagined, d0 = None, None
        o_mean = o_var = torch.zeros(Bb, 0, device=dev)
        epi = ale = torch.zeros(Bb, device=dev)
        p_z = info["logits"].exp()
        lat_h = -(p_z * info["logits"]).sum(-1).mean(-1) / math.log(p_z.shape[-1])   # normalised latent entropy
        model_error = lat_h
        if hooks.outcome_queries_fn is not None:
            q = hooks.outcome_queries_fn(st.Z, st.actions, st.act_tokens, k)
            head = wm.outcome_head
            feat = head.features(Z_next, q)
            d0 = head.dist(feat, q)
            om = q.y_mask.bool() & q.q.mask.bool() if q.y_mask is not None else q.q.mask.bool()
            if fast is not None and fast.particles is not None and fast.particles.dim() == 3:
                dk = head.dist(feat, q, cond=fast.particles)                          # [Bb,K,Q]
                w = torch.softmax(fast.logw.float(), -1)                              # [Bb,K]
                mk, vk = dk.mean(), dk.variance()
                mbar = (w.unsqueeze(-1) * mk).sum(1)
                epi_q = (w.unsqueeze(-1) * (mk - mbar.unsqueeze(1)) ** 2).sum(1)
                ale_q = (w.unsqueeze(-1) * vk).sum(1)
                if self.outcome_sampling == "mixture":
                    kk = torch.multinomial(w, 1, generator=generator).squeeze(1)       # particle per row
                    ys = dk.sample(generator)[torch.arange(Bb, device=dev), kk]
                else:
                    ys = d0.sample(generator)
            else:
                mbar, ale_q = d0.mean(), d0.variance()
                epi_q = torch.zeros_like(mbar)
                ys = d0.sample(generator)
            o_mean, o_var = mbar, epi_q + ale_q
            epi = masked_mean(epi_q, om, dims=-1)
            ale = masked_mean(ale_q, om, dims=-1)
            has = om.any(-1)
            me = masked_mean(torch.log1p(epi_q + ale_q), om, dims=-1)
            model_error = torch.where(has, me, lat_h)
            imagined = OutcomeBatch(q.q, q.lik, q.n_classes, torch.where(om, ys, torch.zeros_like(ys)),
                                    torch.zeros_like(om), om, q.channel, q.keys)
            if hooks.belief_update_fn is not None and fast is not None:
                fast = _first(hooks.belief_update_fn(fast, st.Z, st.act_tokens, imagined))
        goal = st.goal
        reward = hooks.reward_fn(Z_next, imagined, d0, goal) if hooks.reward_fn is not None \
            else wm.reward(Z_next, goal)
        reward = reward.float().reshape(Bb)
        cont = wm.continue_prob(Z_next) if self.use_continue else torch.ones(Bb, device=dev)
        gamma = torch.exp(-self.kappa * dt.float())
        disc_k = st.disc
        ret = st.ret + disc_k * reward
        penalty = st.penalty + disc_k * model_error
        disc = disc_k * gamma * cont
        belief = hooks.belief_tokens_fn(fast) if (hooks.belief_tokens_fn is not None and fast is not None) else None
        if hooks.reason_fn is not None:
            ctx = cat_tokens(belief, goal) if (belief is not None or goal is not None) else \
                TokenSet.from_dense(Z_next[:, :1] * 0, TokenKind.PAD, mask=torch.zeros(Bb, 1, dtype=torch.bool,
                                                                                         device=dev))
            R = _first(hooks.reason_fn(Z_next, ctx))
        else:
            R = Z_next
        value = hooks.value_fn(R, goal, belief).float().reshape(Bb) if hooks.value_fn is not None \
            else torch.zeros(Bb, device=dev)
        rec = WorldImaginedStep(Z=Z_next, action=list(st.actions), act_tokens=st.act_tokens, outcome_mean=o_mean,
                                outcome_var=o_var, value=value, reward=reward, belief_ess=belief_ess(fast, Bb, dev),
                                model_error=model_error, dt=dt, cont=cont, discount=disc_k, epistemic=epi,
                                aleatoric=ale, outcomes=imagined, R=R, logits=info["logits"])
        new = ImagState(Z_next, fast, goal, belief, [], st.act_tokens, disc, ret, penalty, st.elapsed + dt,
                        st.branch_of, st.first_candidate, value, R)
        new.actions = list(st.actions)          # previous actions until the policy chooses
        return new, rec

    def choose(self, st: ImagState, hooks: ImaginationHooks, k: int) -> ImagState:
        """Let the policy choose the next action of every row from its imagined (revised) belief."""
        if hooks.policy_fn is None:
            raise ValueError("multi-step imagination needs hooks.policy_fn")
        ctx = PolicyContext(R=st.R, Z=st.Z, goal=st.goal, belief=st.belief, fast=st.fast, step=k,
                            prev_actions=st.actions, branch_of=st.branch_of, first_candidate=st.first_candidate)
        actions, toks = hooks.policy_fn(ctx)
        st.actions, st.act_tokens = list(actions), toks
        return st

    # ------------------------------------------------------------------ full rollout
    def rollout(self, Z, fast: Optional[FastState], goal: Optional[TokenSet], candidates: Candidates,
                hooks: ImaginationHooks, horizon: Optional[int] = None, n_samples: Optional[int] = None,
                generator: Optional[torch.Generator] = None, grad: bool = False) -> WorldImaginedTrajectory:
        """Z [B,S,D] real latent state, fast the real FastState [B], goal GOAL tokens [B] (or None), candidates
        the contemplated first actions. Returns the imagined trajectory of all B*n*n_samples rows."""
        h = horizon or self.horizon
        s = n_samples or self.n_samples
        n0 = self.wm.n_transitions
        with torch.set_grad_enabled(grad):
            st = self.init_state(Z, fast, goal, candidates, s, hooks)
            steps: List[WorldImaginedStep] = []
            for k in range(h):
                st, rec = self.step(st, hooks, k, generator)
                steps.append(rec)
                if k + 1 < h:
                    st = self.choose(st, hooks, k + 1)
            boot = st.value
            raw = st.ret + st.disc * boot
            ret = raw - self.uncertainty_coef * st.penalty
        return WorldImaginedTrajectory(steps=steps, returns=ret, branch_of=st.branch_of,
                                       first_candidate=st.first_candidate, provenance=Provenance.IMAGINED,
                                       returns_raw=raw, penalty=st.penalty, bootstrap=boot, discount_h=st.disc,
                                       n_model_evals=self.wm.n_transitions - n0)


def candidate_scores(traj: ImaginedTrajectory, B: int, n: int) -> torch.Tensor:
    """Mean imagined return per (environment, first candidate) -> [B, n]."""
    dev = traj.returns.device
    idx = traj.branch_of * n + traj.first_candidate
    tot = torch.zeros(B * n, device=dev).index_add_(0, idx, traj.returns.float())
    cnt = torch.zeros(B * n, device=dev).index_add_(0, idx, torch.ones_like(traj.returns.float()))
    return (tot / cnt.clamp(min=1)).view(B, n)


def trace_summaries(traj: ImaginedTrajectory, b: int, n: int, keys: Optional[list] = None) -> list:
    """DecisionTrace.imagined entries for environment b: one dict per first candidate with the mean imagined
    return, horizon, model error and the first step's predicted outcomes (by key when given)."""
    out = []
    sel_b = traj.branch_of == b
    for i in range(n):
        sel = sel_b & (traj.first_candidate == i)
        if not bool(sel.any()):
            continue
        s0 = traj.steps[0]
        om = s0.outcome_mean[sel].mean(0) if s0.outcome_mean.shape[-1] else torch.zeros(0)
        names = keys if keys is not None else [f"q{j}" for j in range(om.numel())]
        out.append({
            "candidate": i, "horizon": len(traj.steps), "value": float(traj.returns[sel].mean()),
            "value_std": float(traj.returns[sel].std()) if int(sel.sum()) > 1 else 0.0,
            "model_error": float(torch.stack([st.model_error[sel].mean() for st in traj.steps]).mean()),
            "outcomes": {str(k): float(v) for k, v in zip(names, om.tolist())},
            "rewards": [float(st.reward[sel].mean()) for st in traj.steps],
            "n_samples": int(sel.sum()), "provenance": Provenance.IMAGINED.value})
    return out
