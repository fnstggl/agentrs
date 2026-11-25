"""Learning to learn: support/query fast-adaptation loss (spec 12.2), J_meta over support sizes and multi-trial
episodes (12.4, 8.4), and the multi-trial episode harness over any `EnvAdapter`.

**Support/query loss** ``L_adapt``. For every episode of every row and each support size k in
``(0,1,2,4,8,16,32,64)``: infer the belief from the episode's first k steps only,
``B_k = Update(...Update(Init, s_0)..., s_{k-1})``, then score the later steps of the same episode without updating on
them, ``L_k = -mean_j log p(o_j | B_k, a_j)``, ``j >= k``. The support count restarts at each episode start (a fresh
belief per episode), so every episode of a row contributes; support sizes no episode is long enough for are
reported (``skipped_k``), never silently dropped. Support and query are disjoint by construction (and by event id
when the steps carry ids); query outcomes never enter the belief that scores them. Gradients flow through the
(differentiable) adaptation operation. The belief operations are callables, so workstream B's `BeliefModule` plugs in
through `belief_callables_from_modules` (queries conditioned on their actions through the world model's prior).

**J_meta** ``= E_tau[ sum_k w_k J_tau(U(D_{1:k})) - lambda C_experiment ]`` (spec 12.4). Two weightings, combinable:

* support sizes: ``J_tau(U(D_{1:k}))`` is the verified return of the ``window`` decisions that follow the first k
  decisions of the task, so ``J_meta = sum_t W(n_t) r_t`` with ``W(n) = sum_{k: k <= n < k + window} w_k`` and n_t
  the number of earlier decisions of the same task — a per-decision reward weight;
* trials: a meta-RL episode is ``n_trials`` trials of the same task (same hidden mechanism: the adapter is reset with
  the same seeds) during which the agent's own recurrent / fast state is *not* reset; trial k's return gets
  ``trial_weights[k]``.

Boundaries between support and competence (and between trials) are not terminal, so the return of an early
(exploratory) decision contains the weighted rewards of later competence: early experiments receive policy-gradient
credit for later competence through the actual sequence of real decisions (no gradient through the environment).
The estimator is the actor-critic of `ppo.ActorCriticObjective` on the weighted, cost-penalised, optionally
LP-augmented rewards; its loss is reported under the ``meta`` key so `CombinedTrainer` applies ``loss_weights
["meta"]`` and the stage gating (stage 5, ``explore_meta``).
"""
from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence

import torch
from torch import Tensor

from ..interfaces import SequenceBatch
from .explore import _check_disjoint_ids, mixture_log_prob
from .heads import BudgetState
from .ppo import ActorCriticObjective, Prepared
from .trajectory import Trajectory, compact_rows, sequence_mask

SUPPORT_SIZES = (0, 1, 2, 4, 8, 16, 32, 64)


# ================================================================================================ belief row selection

def tree_where(mask: Tensor, new: Any, old: Any) -> Any:
    """Row-wise selection between two beliefs of the same structure: rows with ``mask [B]`` True from ``new``, the
    others from ``old``. Recurses into tuples / lists / dicts / dataclasses; tensors whose leading dimension is B are
    per-row, anything else is taken from ``new``. Objects with a ``where_rows(mask, other)`` method decide
    themselves. Differentiable (torch.where)."""
    if hasattr(new, "where_rows"):
        return new.where_rows(mask, old)
    if torch.is_tensor(new):
        B = mask.shape[0]
        if new.dim() >= 1 and new.shape[0] == B and torch.is_tensor(old) and old.shape == new.shape:
            m = mask.to(new.device).bool().view(B, *([1] * (new.dim() - 1)))
            return torch.where(m, new, old)
        return new
    if isinstance(new, tuple) and hasattr(new, "_fields"):
        return type(new)(*(tree_where(mask, a, b) for a, b in zip(new, old)))
    if isinstance(new, (tuple, list)):
        return type(new)(tree_where(mask, a, b) for a, b in zip(new, old))
    if isinstance(new, dict):
        return {k: tree_where(mask, v, old[k]) if k in old else v for k, v in new.items()}
    if dataclasses.is_dataclass(new) and not isinstance(new, type):
        return dataclasses.replace(new, **{f.name: tree_where(mask, getattr(new, f.name), getattr(old, f.name))
                                           for f in dataclasses.fields(new) if f.init})
    return new


def _accepts(fn: Callable, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return name in sig.parameters or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())


# ================================================================================================ support / query

class SupportQueryLoss:
    """Fast-adaptation loss over support sizes ``ks`` with per-episode support.

    ``__call__(steps, init_fn, update_fn, log_prob_fn, count_fn=None, mask=None)``:
      steps: a `SequenceBatch` or a list of step records (length T, aligned rows);
      init_fn(B, device[, t0]) -> belief (``t0`` [B] long, the step index where the new belief starts, is passed when
        the callable accepts it); update_fn(belief, step, t) -> belief (differentiable);
      log_prob_fn(belief, step, t) -> [B] summed log-likelihood of step t's observed outcomes;
      count_fn(step) -> [B] number of observed outcomes (default: ``step.outcomes.y_mask.sum(-1)`` or 1);
      mask [B,T] valid steps (default `trajectory.sequence_mask`, the contract's done/obs.mask padding rule).
    Episodes are delimited by ``done``. For support size k every episode with more than k valid steps contributes:
    its first k steps update a belief initialised at the episode start (rows are merged with `tree_where`), its later
    steps (up to ``max_query``) are scored and never absorbed.
    Returns {"loss": scalar, "per_k": {k: nll per outcome}, "n_query": {k: outcomes scored}, "per_query":
    {k: [B, T] log-probs (0 outside the query)}, "per_k_tensor", "n_episodes": {k: episodes used},
    "coverage": {k: fraction of valid steps scored}, "skipped_k": support sizes no episode could serve}.
    ``on_skip="raise"`` raises instead of reporting skipped support sizes."""

    def __init__(self, ks: Sequence[int] = SUPPORT_SIZES, weights: Optional[Sequence[float]] = None,
                 max_query: Optional[int] = None, on_skip: str = "report"):
        self.ks = tuple(int(k) for k in ks)
        if any(k < 0 for k in self.ks) or len(set(self.ks)) != len(self.ks):
            raise ValueError("support sizes must be distinct non-negative integers")
        self.weights = None if weights is None else tuple(float(w) for w in weights)
        if self.weights is not None and len(self.weights) != len(self.ks):
            raise ValueError("one weight per support size")
        if on_skip not in ("report", "raise"):
            raise ValueError("on_skip must be 'report' or 'raise'")
        self.max_query, self.on_skip = max_query, on_skip

    def __call__(self, steps, init_fn: Callable, update_fn: Callable, log_prob_fn: Callable,
                 count_fn: Optional[Callable] = None, mask: Optional[Tensor] = None) -> dict:
        steps = list(steps.steps) if isinstance(steps, SequenceBatch) else list(steps)
        T = len(steps)
        if T < 2:
            raise ValueError("need at least two steps (one support, one query)")
        _check_disjoint_ids(steps)
        count_fn = count_fn or _default_count
        B, dev = _infer_B(steps)
        valid = (sequence_mask(steps) if mask is None else mask.bool()).to(dev)
        done = torch.stack([_done_or_false(_done_of(s), B, dev) for s in steps], 1) & valid
        ep = torch.zeros(B, T, dtype=torch.long, device=dev)
        if T > 1:
            ep[:, 1:] = torch.cumsum(done[:, :-1].long(), 1)
        start = torch.ones(B, T, dtype=torch.bool, device=dev)
        start[:, 1:] = done[:, :-1]
        pos = torch.zeros(B, T, dtype=torch.long, device=dev)
        for t in range(1, T):
            pos[:, t] = torch.where(start[:, t], torch.zeros_like(pos[:, t]), pos[:, t - 1] + 1)
        ep_len = torch.zeros(B, T + 1, dtype=torch.long, device=dev).scatter_add_(1, ep, valid.long()).gather(1, ep)
        with_t0 = _accepts(init_fn, "t0")

        def init(t):
            if with_t0:
                return init_fn(B, dev, t0=torch.full((B,), int(t), dtype=torch.long, device=dev))
            return init_fn(B, dev)

        n_valid = max(1, int(valid.sum()))
        per_k, n_q, per_query, n_eps, cover, losses, wts, skipped = {}, {}, {}, {}, {}, [], [], []
        for k in sorted(self.ks):
            usable = valid & (ep_len > k)
            sup = usable & (pos < k)
            qry = usable & (pos >= k)
            if self.max_query is not None:
                qry = qry & (pos < k + self.max_query)
            busy = (sup | qry).any(0)
            if not bool(qry.any()):
                skipped.append(k)
                continue
            t_hi = int(busy.nonzero().max()) + 1
            belief = init(0)
            tot = torch.zeros((), device=dev)
            cnt = torch.zeros((), device=dev)
            pq = torch.zeros(B, T, device=dev)
            for t in range(t_hi):
                if t > 0:
                    rs = start[:, t] & (sup[:, t] | qry[:, t])
                    if bool(rs.any()):                                 # a new episode: a fresh belief for its rows
                        belief = tree_where(rs, init(t), belief)
                q = qry[:, t]
                if bool(q.any()):                                      # query: scored, never used to update
                    lp = log_prob_fn(belief, steps[t], t).float()
                    c = count_fn(steps[t]).float().to(dev)
                    tot = tot + torch.where(q, lp, torch.zeros_like(lp)).sum()
                    cnt = cnt + torch.where(q, c, torch.zeros_like(c)).sum()
                    pq[:, t] = torch.where(q, lp, torch.zeros_like(lp))
                sp = sup[:, t]
                if bool(sp.any()):                                     # support: absorbed
                    new = update_fn(belief, steps[t], t)
                    belief = new if bool(sp.all()) else tree_where(sp, new, belief)
            if float(cnt) == 0:
                skipped.append(k)
                continue
            nll = -tot / cnt
            per_k[k], n_q[k], per_query[k] = nll, int(cnt), pq
            n_eps[k] = int((usable & start).sum())
            cover[k] = float(qry.sum()) / n_valid
            losses.append(nll)
            wts.append(1.0 if self.weights is None else self.weights[self.ks.index(k)])
        if skipped and self.on_skip == "raise":
            raise ValueError(f"support sizes {skipped} have no episode long enough to leave a query")
        if not losses:
            raise ValueError("no query outcomes to score")
        w = torch.tensor(wts, device=losses[0].device)
        loss = (torch.stack(losses) * w).sum() / w.sum()
        return {"loss": loss, "per_k": {k: float(v.detach()) for k, v in per_k.items()}, "n_query": n_q,
                "per_query": per_query, "per_k_tensor": per_k, "n_episodes": n_eps, "coverage": cover,
                "skipped_k": skipped}


def _infer_B(steps) -> tuple:
    s = steps[0]
    for name in ("done", "reward", "dt"):
        v = getattr(s, name, None)
        if torch.is_tensor(v):
            return v.shape[0], v.device
    raise ValueError("cannot infer the batch size: steps need a `done`, `reward` or `dt` tensor")


def _done_of(s) -> Optional[Tensor]:
    d = getattr(s, "done", None)
    return torch.as_tensor(d).bool() if d is not None else None


def _done_or_false(d, B, dev) -> Tensor:
    return torch.zeros(B, dtype=torch.bool, device=dev) if d is None else d.to(dev).bool()


def _episode_index(done: list, B: int, dev) -> Tensor:
    """ep[b, t] = number of episode ends strictly before step t."""
    T = len(done)
    d = torch.stack([_done_or_false(x, B, dev) for x in done], 1).long()
    ep = torch.zeros(B, T, dtype=torch.long, device=dev)
    if T > 1:
        ep[:, 1:] = torch.cumsum(d[:, :-1], 1)
    return ep


def _default_count(step) -> Tensor:
    oc = getattr(step, "outcomes", None)
    if oc is not None and getattr(oc, "y_mask", None) is not None:
        return oc.y_mask.float().sum(-1)
    r = getattr(step, "reward", None)
    return torch.ones(r.shape[0]) if torch.is_tensor(r) else torch.ones(1)


@dataclass
class SupportBelief:
    """A contract `FastState` together with where its support ends: ``end [B]`` long is the index of the first step
    the belief has *not* absorbed (the support is the steps before it in the current episode). ``cache`` memoises the
    belief's open-loop query states (a pure function of the belief, so results never depend on call order)."""
    fast: Any
    end: Tensor
    cache: dict = field(default_factory=dict, repr=False, compare=False)

    def where_rows(self, mask: Tensor, other: "SupportBelief") -> "SupportBelief":
        return SupportBelief(tree_where(mask, self.fast, other.fast), torch.where(mask.to(self.end.device),
                                                                                    self.end, other.end))


def support_belief(fast, end) -> SupportBelief:
    """Wrap an externally produced belief (e.g. the agent filter's B_t^-, ``end = t``) for the module callables."""
    B = fast.logw.shape[0]
    e = torch.as_tensor(end, dtype=torch.long, device=fast.logw.device)
    return SupportBelief(fast, e.expand(B).clone() if e.dim() == 0 else e)


def belief_callables_from_modules(belief_module, outcome_head, Z_seq: Sequence[Tensor], steps=None,
                                  world_model=None, action_condition: Optional[Callable] = None,
                                  query_Z: str = "support_end", allow_action_blind: bool = False,
                                  sample: bool = False):
    """Wrap workstream B/C modules (contract `BeliefModuleP`, `OutcomeHeadP`, `WorldModelP`) into the three callables
    of `SupportQueryLoss` / `explore.hindsight_learning_progress`. Beliefs are `SupportBelief`s.

    ``Z_seq[t]`` [B,S,D] is the filtered latent state when decision t was taken (it has absorbed outcomes < t).
    update(b, step, t) = belief_module.update(b.fast, Z_seq[t], step.act, step.outcomes, outcome_head)[0], end = t+1.
    log_prob(b, step, j) = log sum_k w_k prod_q p(y_q | Zq_j, particle k) with
    ``outcome_head.dist(outcome_head.features(Zq_j, outcomes), outcomes, cond=b.fast.particles)`` (batch [B,K,Q]).

    The query state Zq_j never contains query outcomes (``query_Z="support_end"``): it starts from the latent state
    at the end of each row's support, ``Z_seq[b.end]``, and is conditioned on the query actions (spec 12.2,
    ``-log p(Q | B_S, query_actions)``):

    * ``world_model`` (with ``steps``): open loop through the actions from the support end to j,
      ``Z <- world_model.prior(Z, steps[i].act, belief_module.tokens(b.fast), steps[i].dt)`` for i = end..j; outcome j
      is scored from the predicted state after its own action (the world model's ``outcome_dist`` convention);
    * ``action_condition(Z, act) -> Z_a``: the support-end state conditioned on the query's own action only (e.g. B's
      ``BeliefModule.condition_on_action``);
    * ``allow_action_blind=True``: no action conditioning (only for outcomes that do not depend on the actions).
    ``query_Z="own"`` scores step j from ``Z_seq[j]`` (one-step-ahead filtering where Z_seq[j] provably precedes
    o_j), optionally action-conditioned."""
    if query_Z not in ("support_end", "own"):
        raise ValueError("query_Z must be 'support_end' or 'own'")
    if world_model is not None and steps is None:
        raise ValueError("the world-model rollout needs the steps (query actions and elapsed days)")
    if query_Z == "support_end" and world_model is None and action_condition is None and not allow_action_blind:
        raise ValueError("query outcomes must be conditioned on the query actions (spec 12.2): pass world_model "
                         "(open loop through WorldModelP.prior) or action_condition; allow_action_blind=True only "
                         "when outcomes do not depend on the actions")
    steps = None if steps is None else (list(steps.steps) if isinstance(steps, SequenceBatch) else list(steps))
    n_Z = len(Z_seq)
    tag = object()                     # cache entries of these callables (a belief may be scored by several)

    def init(B, device, t0=None):
        fast = belief_module.init_fast(B, device)
        end = torch.zeros(B, dtype=torch.long, device=fast.logw.device) if t0 is None else \
            torch.as_tensor(t0, dtype=torch.long, device=fast.logw.device).expand(B).clone()
        return SupportBelief(fast, end)

    def update(b: SupportBelief, step, t):
        fast = belief_module.update(b.fast, Z_seq[t], step.act, step.outcomes, outcome_head)[0]
        return SupportBelief(fast, torch.full_like(b.end, t + 1))

    def _support_end_state(b: SupportBelief) -> Tensor:
        if (tag, "Z0") not in b.cache:
            idx = b.end.clamp(0, n_Z - 1).tolist()
            b.cache[(tag, "Z0")] = torch.stack([Z_seq[i][r] for r, i in enumerate(idx)], 0)
        return b.cache[(tag, "Z0")]

    def _rollout(b: SupportBelief, j: int) -> Tensor:
        """Predicted state after the action of step j, open loop from each row's support end."""
        if (tag, "roll", j) in b.cache:
            return b.cache[(tag, "roll", j)]
        lo = int(b.end.min())
        done_i = [i for i in range(lo, j) if (tag, "roll", i) in b.cache]
        i0 = (max(done_i) + 1) if done_i else lo
        Z = b.cache[(tag, "roll", i0 - 1)] if done_i else _support_end_state(b)
        if (tag, "btok") not in b.cache:
            b.cache[(tag, "btok")] = belief_module.tokens(b.fast)
        for i in range(i0, j + 1):
            st = steps[i]
            Zn = world_model.prior(Z, st.act, b.cache[(tag, "btok")], st.dt, sample=sample)[0]
            adv = (b.end <= i).to(Z.device).view(-1, *([1] * (Z.dim() - 1)))
            Z = torch.where(adv, Zn, Z)
            b.cache[(tag, "roll", i)] = Z
        return Z

    def log_prob(b: SupportBelief, step, j):
        if query_Z == "own":
            Z = Z_seq[max(0, min(j, n_Z - 1))]
            if action_condition is not None:
                Z = action_condition(Z, step.act)
        elif world_model is not None:
            Z = _rollout(b, j)
        else:
            Z = _support_end_state(b)
            if action_condition is not None:
                Z = action_condition(Z, step.act)
        fast = b.fast
        feat = outcome_head.features(Z, step.outcomes)
        dist = outcome_head.dist(feat, step.outcomes, cond=fast.particles)
        m = step.outcomes.y_mask.bool()
        y = torch.where(m, step.outcomes.y, torch.zeros_like(step.outcomes.y))
        y = y.unsqueeze(1).expand(-1, fast.particles.shape[1], -1)
        lp = dist.log_prob(y)
        lp = torch.where(m.unsqueeze(1), lp, torch.zeros_like(lp))
        return mixture_log_prob(lp, fast.logw, m)

    return init, update, log_prob


# ================================================================================================ J_meta

class MetaRLObjective:
    """J_meta estimator over support sizes and multi-trial episodes.

    Rewards entering the actor-critic: ``r'_t = W_t r_t - cost_coef * cost_t + lp_coef * intrinsic_t`` with the
    per-decision weight ``W_t = trial_weights[trial(t)] * W_support(n_t)`` (module docstring; either factor is 1 when
    not configured), ``cost_t`` the observable experiment cost and ``intrinsic_t`` the noise-corrected learning
    progress (`explore.exploration_reward` with coef 1), included only while `intrinsic_enabled()`: the ablation switch
    ``use_intrinsic`` (``cfg.use_learned_exploration``) and the trainer's stage flag ``lp_reward`` (stage 5 on).
    Boundaries are not terminal, so credit flows from later competence to earlier decisions.
    ``j_meta(traj)`` reports the per-episode objective value; ``competence_by_support(traj)`` the learning curve."""

    policy_key = "meta"

    def __init__(self, actor_critic: ActorCriticObjective, trial_weights: Optional[Sequence[float]] = None,
                 cost_coef: float = 0.0, use_intrinsic: bool = True, lp_coef: float = 1.0,
                 intrinsic_gate: Optional[Callable[[], bool]] = None,
                 support_weights: Optional[dict] = None, competence_window: Optional[int] = None):
        self.ac = actor_critic
        self.trial_weights = None if trial_weights is None else torch.tensor([float(w) for w in trial_weights])
        self.cost_coef = float(cost_coef)
        self.use_intrinsic = bool(use_intrinsic)
        self.lp_coef = float(lp_coef)
        self.intrinsic_gate = intrinsic_gate
        if support_weights is not None:
            if any(int(k) < 0 for k in support_weights) or not support_weights:
                raise ValueError("support sizes must be non-negative")
            if competence_window is not None and competence_window < 1:
                raise ValueError("competence_window must be >= 1")
        self.support_weights = None if support_weights is None else {int(k): float(w)
                                                                     for k, w in support_weights.items()}
        self.window = competence_window

    @staticmethod
    def from_config(cfg, actor_critic: Optional[ActorCriticObjective] = None, trainer=None,
                    **over) -> "MetaRLObjective":
        """``lp_coef = cfg.lp_coef``; LP on iff ``cfg.use_learned_exploration`` and (with a trainer) the active stage
        sets ``lp_reward``; the actor-critic from ``cfg`` unless given."""
        kw = dict(use_intrinsic=bool(cfg.use_learned_exploration), lp_coef=float(cfg.lp_coef),
                  intrinsic_gate=(lambda: trainer.flag("lp_reward")) if trainer is not None else None)
        kw.update(over)
        return MetaRLObjective(actor_critic or ActorCriticObjective.from_config(cfg), **kw)

    def intrinsic_enabled(self) -> bool:
        return self.use_intrinsic and (self.intrinsic_gate is None or bool(self.intrinsic_gate()))

    @staticmethod
    def task_position(traj: Trajectory) -> Tensor:
        """n_t [B,T]: number of earlier valid decisions of the same task (restarts after a terminal ``done``)."""
        m = traj.mask.bool()
        B, T = m.shape
        pos = torch.zeros(B, T, dtype=torch.long, device=m.device)
        run = torch.zeros(B, dtype=torch.long, device=m.device)
        for t in range(T):
            pos[:, t] = run
            run = torch.where(traj.done[:, t].bool(), torch.zeros_like(run), run + m[:, t].long())
        return pos

    def support_weight(self, n: Tensor) -> Tensor:
        """W(n) = sum_{k: k <= n < k + window} w_k for decision counts ``n`` (any shape)."""
        out = torch.zeros(n.shape, dtype=torch.float32, device=n.device)
        for k, w in self.support_weights.items():
            inside = n >= k if self.window is None else (n >= k) & (n < k + self.window)
            out = out + w * inside.float()
        return out

    def step_weights(self, traj: Trajectory) -> Tensor:
        w = torch.ones_like(traj.reward, dtype=torch.float32)
        if self.trial_weights is not None:
            if traj.trial is None:
                raise ValueError("trial weights need Trajectory.trial")
            tw = self.trial_weights.to(w.device)
            w = w * tw[traj.trial.long().clamp(0, len(tw) - 1)]
        if self.support_weights is not None:
            w = w * self.support_weight(self.task_position(traj))
        return w

    def shaped_reward(self, traj: Trajectory) -> Tensor:
        r = traj.reward.float() * self.step_weights(traj)
        if self.cost_coef and traj.cost is not None:
            r = r - self.cost_coef * traj.cost.float()
        if self.intrinsic_enabled() and traj.intrinsic is not None:
            r = r + self.lp_coef * traj.intrinsic.float()
        return torch.where(traj.mask, r, torch.zeros_like(r))

    def j_meta(self, traj: Trajectory) -> Tensor:
        """[B] value of the objective per episode (sum of shaped rewards)."""
        return self.shaped_reward(traj).sum(1)

    def competence_by_support(self, traj: Trajectory, ks: Optional[Sequence[int]] = None,
                              window: Optional[int] = None) -> dict:
        """{k: mean over tasks with more than k decisions of the verified return of their decisions
        k .. k+window-1} (``J_tau(U(D_{1:k}))``, the competence after k support decisions; ks default to the
        configured support sizes, else `SUPPORT_SIZES`) and {"n": {k: tasks}}."""
        if ks is None:
            ks = sorted(self.support_weights) if self.support_weights is not None else list(SUPPORT_SIZES)
        window = self.window if window is None else window
        n = self.task_position(traj)
        r = torch.where(traj.mask, traj.reward.float(), torch.zeros_like(traj.reward.float()))
        out, cnt = {}, {}
        for k in ks:
            inside = traj.mask & (n >= k)
            if window is not None:
                inside = inside & (n < k + window)
            tasks = int((traj.mask & (n == k)).sum())        # tasks with more than k decisions
            cnt[k] = tasks
            if tasks:
                out[k] = float(torch.where(inside, r, torch.zeros_like(r)).sum() / tasks)
        out["n"] = cnt
        return out

    def per_trial_return(self, traj: Trajectory) -> Tensor:
        """[B, n_trials] raw extrinsic return per trial (competence curve within the meta-episode)."""
        r = torch.where(traj.mask, traj.reward.float(), torch.zeros_like(traj.reward.float()))
        if traj.trial is None:
            return r.sum(1, keepdim=True)
        n = int(traj.trial[traj.mask].max()) + 1 if bool(traj.mask.any()) else 1
        out = torch.zeros(traj.B, n, device=traj.device)
        out.scatter_add_(1, traj.trial.long().clamp_min(0), r)
        return out

    def prepare(self, traj: Trajectory, current_version: int) -> tuple[Trajectory, Prepared]:
        shaped = traj.with_reward(self.shaped_reward(traj))
        prep = self.ac.prepare(shaped, current_version)
        prep.info["lp_in_reward"] = int(self.intrinsic_enabled() and traj.intrinsic is not None)
        return shaped, prep

    def loss(self, shaped: Trajectory, prep: Prepared, new_logp, entropy, new_value=None, value_logits=None,
             value_head=None, **kw) -> dict:
        d = self.ac.loss(shaped, prep, new_logp, entropy, new_value, value_logits, value_head, **kw)
        d["meta"] = d.pop("actor")
        d["j_meta"] = float(shaped.reward.sum(1).mean())     # shaped rewards are 0 at padding
        return d


# ================================================================================================ harness

class EpisodePolicy(Protocol):
    """What `TrialRunner` needs from a policy (the real agent or a diagnostic network)."""
    version: int

    def initial_state(self, B: int) -> Any: ...

    def act(self, state: Any, obs: list, tools: list, budget: BudgetState, goals: list, active: Tensor,
            greedy: bool = False) -> tuple:
        """-> (actions [B] Action|None, logp [B], value [B], new_state, record dict of [B, ...] tensors).
        Must leave the state of inactive rows unchanged (so a compacted replay reproduces the recursion)."""
        ...


@dataclass
class RunResult:
    traj: Trajectory
    per_trial_return: Tensor            # [B, n_trials] verified return per trial
    actions: list                       # [B][L] tool names of the compacted decisions
    observations: list = field(default_factory=list)   # [B][L] observations delivered after each decision
    final_state: Any = None
    info: dict = field(default_factory=dict)


class TrialRunner:
    """Run multi-trial episodes of an `EnvAdapter` with an `EpisodePolicy`.

    Per decision: the harness builds the BUDGET state (decisions / days / funds left from the public goal), asks the
    policy, executes, advances the SMDP clock by the action's duration plus ``think_days``, delivers the direct results
    and released world events as the next observations, and reads the verified return increment from the
    `GoalEvaluator` (``metrics[return_key]`` difference). Rows are compacted (each row's active decisions are
    left-aligned; trial boundaries are not terminal; the last decision of the last trial is).

    Nothing a row's environment produced is lost between decisions: a row's observations accumulate until the row
    next acts (world events released while it waits for the other rows included), and the observations produced by
    the last decision of a trial are delivered at the first decision of the next trial, before that trial's reset
    observations — what an earlier trial revealed reaches the policy in later trials. The batch size is taken from
    the adapter's ``reset`` (``len(env_ids)``); inactive rows get a ``None`` action and their results are ignored."""

    def __init__(self, adapter, evaluator, return_key: str = "return", think_days: float = 0.0,
                 max_decisions_per_trial: int = 10_000):
        self.adapter, self.evaluator = adapter, evaluator
        self.return_key, self.think_days = return_key, float(think_days)
        self.max_decisions = int(max_decisions_per_trial)

    def _budget(self, goals, used: list, spent: list, now: list, t0: list, device) -> BudgetState:
        B = len(goals)
        dl, dt_, fl, ft, nl, nt = ([float("nan")] * B for _ in range(6))
        for b, g in enumerate(goals):
            meta = g.meta or {}
            if "decision_budget" in meta:
                nt[b] = float(meta["decision_budget"])
                nl[b] = nt[b] - used[b]
            if g.deadline_days is not None:
                dt_[b] = float(g.deadline_days)
                dl[b] = dt_[b] - (now[b] - t0[b])
            if "funds_budget" in meta:
                ft[b] = float(meta["funds_budget"])
                fl[b] = ft[b] - spent[b]
        f = lambda v: torch.tensor(v, dtype=torch.float32, device=device)   # noqa: E731
        return BudgetState(f(dl), f(fl), f(nl), f(dt_), f(ft), f(nt))

    def run(self, policy, seeds: Sequence[int], n_trials: int = 1, greedy: bool = False, device=None,
            state: Any = None, family: Optional[str] = None) -> RunResult:
        seeds = list(seeds)
        B = len(seeds)
        rows: list = []
        names: list = []
        seen: list = []
        trial_ret = torch.zeros(B, n_trials)
        carry: list = [[] for _ in range(B)]           # observations not yet shown to the policy
        for k in range(n_trials):
            ob = self.adapter.reset(seeds)
            if len(ob.env_ids) != B:
                raise ValueError(f"one seed per adapter environment: {B} seeds, {len(ob.env_ids)} environments")
            if k == 0:
                state = policy.initial_state(B) if state is None else state
            goals = self.adapter.goals()
            pending = [carry[b] + list(ob.observations[b]) for b in range(B)]
            t0 = list(ob.t)
            now = list(ob.t)
            used = [0] * B
            spent = [0.0] * B
            prev_ret = [self.evaluator.evaluate_goal(b).metrics[self.return_key] for b in range(B)]
            done = list(ob.done)
            n = 0
            while not all(done):
                if n >= self.max_decisions:
                    raise RuntimeError("trial exceeded max_decisions_per_trial; adapter never reported done")
                n += 1
                act_l = [not d for d in done]
                active = torch.tensor(act_l, device=device)
                tools = self.adapter.available_tools()
                budget = self._budget(goals, used, spent, now, t0, device)
                shown = [pending[b] if act_l[b] else [] for b in range(B)]
                pending = [[] if act_l[b] else pending[b] for b in range(B)]
                actions, logp, value, state, rec = policy.act(state, shown, tools, budget, goals, active, greedy)
                actions = [a if act_l[b] else None for b, a in enumerate(actions)]
                results = self.adapter.execute(actions)
                results = [results[b] if act_l[b] else None for b in range(B)]
                dur = [0.0 if r is None else float(r.duration) + self.think_days for r in results]
                ev = self.adapter.advance_time(dur)
                rew, cost, after = [], [], []
                for b in range(B):
                    cur = self.evaluator.evaluate_goal(b).metrics[self.return_key]
                    rew.append(float(cur - prev_ret[b]) if act_l[b] else 0.0)
                    prev_ret[b] = cur
                    c = 0.0 if results[b] is None else float(results[b].cost)
                    cost.append(c)
                    if act_l[b]:
                        used[b] += 1
                        spent[b] += c
                    new_obs = ([] if results[b] is None else list(results[b].observations)) + list(ev.observations[b])
                    pending[b] = pending[b] + new_obs
                    after.append(new_obs if act_l[b] else [])
                now = list(ev.t)
                new_done = list(ev.done)
                last_trial = k == n_trials - 1
                rows.append({"active": active.clone(), "reward": torch.tensor(rew, device=device),
                             "dt": torch.tensor(dur, device=device),
                             "done": torch.tensor([bool(new_done[b]) and last_trial and act_l[b]
                                                   for b in range(B)], device=device),
                             "logp": logp.detach().float().to(device), "value": value.detach().float().to(device),
                             "cost": torch.tensor(cost, device=device),
                             "trial": torch.full((B,), k, dtype=torch.long, device=device),
                             "rec": {kk: v.detach() for kk, v in (rec or {}).items()}})
                names.append([a.tool if a is not None else "" for a in actions])
                seen.append(after)
                for b in range(B):
                    if act_l[b]:
                        trial_ret[b, k] += rew[b]
                done = new_done
            carry = pending
        if not rows:
            raise RuntimeError("the adapter reported every episode done at reset: no decisions")
        act = torch.stack([r["active"] for r in rows], 1)
        series = {kk: torch.stack([r[kk] for r in rows], 1) for kk in
                  ("reward", "dt", "done", "logp", "value", "cost", "trial")}
        rec_keys = list(rows[0]["rec"].keys())
        recs = {kk: torch.stack([r["rec"][kk] for r in rows], 1) for kk in rec_keys}
        keys = list(series) + rec_keys
        mask, comp = compact_rows(act, *[series[k_] for k_ in series], *[recs[k_] for k_ in rec_keys])
        c = dict(zip(keys, comp))
        L = mask.shape[1]
        value = torch.zeros(B, L + 1, device=mask.device)
        value[:, :L] = c["value"]                    # bootstrap after the terminal last decision is 0
        ver = int(getattr(policy, "version", 0))
        traj = Trajectory(reward=c["reward"].float(), dt=c["dt"].float(), done=c["done"].bool(),
                          behavior_logp=c["logp"].float(), model_version=torch.full((B, L), ver, dtype=torch.long,
                                                                                    device=mask.device),
                          mask=mask, value=value, cost=c["cost"].float(), trial=c["trial"].long(),
                          family=[family or getattr(self.adapter, "family", "default")] * B,
                          extras={kk: c[kk] for kk in rec_keys})
        # tool names, compacted the same way
        order = [[t for t in range(len(names)) if bool(act[b, t])] for b in range(B)]
        tool_names = [[names[t][b] for t in order[b]] for b in range(B)]
        obs_after = [[seen[t][b] for t in order[b]] for b in range(B)]
        return RunResult(traj, trial_ret, tool_names, obs_after, state,
                         {"n_decisions": int(mask.sum()), "undelivered": carry})
