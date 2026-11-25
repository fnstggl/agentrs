"""Learning to learn: support/query fast-adaptation loss (spec 12.2), J_meta over multi-trial episodes (12.4, 8.4),
and the multi-trial episode harness over any `EnvAdapter`.

**Support/query loss** ``L_adapt``. For a sequence of decisions and each support size k in ``(0,1,2,4,8,16,32,64)``:
infer the belief from the first k steps only, ``B_k = Update(...Update(Init, s_0)..., s_{k-1})``, then score the
later steps of the same episode without updating on them, ``L_k = -mean_j log p(o_j | B_k)``, ``j >= k``. Support and
query are disjoint by construction (and by event id when the steps carry ids); query outcomes never enter the belief
that scores them. Gradients flow through the (differentiable) adaptation operation. The belief operations are
callables so that workstream B's `BeliefModule` / `BayesLastLayer` plug in through `belief_callables_from_modules`.

**J_meta** ``= E_tau[ sum_k w_k J_tau(trial k) - lambda C_experiment ]``. A meta-RL episode is ``n_trials`` trials
of the same task (same hidden mechanism: the adapter is reset with the same seeds) during which the agent's own
recurrent / fast state is *not* reset. Trial boundaries are not terminal, so the return of an early (exploratory)
decision contains the weighted rewards of later trials: early experiments receive policy-gradient credit for later
competence through the actual sequence of real decisions (no gradient through the environment). The estimator is the
actor-critic of `ppo.ActorCriticObjective` on trial-weighted, cost-penalised, optionally LP-augmented rewards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence

import torch
from torch import Tensor

from ..interfaces import SequenceBatch
from .explore import _check_disjoint_ids, mixture_log_prob
from .heads import BudgetState
from .ppo import ActorCriticObjective, Prepared
from .trajectory import Trajectory, compact_rows

SUPPORT_SIZES = (0, 1, 2, 4, 8, 16, 32, 64)


# ================================================================================================ support / query

class SupportQueryLoss:
    """Fast-adaptation loss over support sizes ``ks``.

    ``__call__(steps, init_fn, update_fn, log_prob_fn, count_fn=None)``:
      steps: a `SequenceBatch` or a list of step records (length T);
      init_fn(B, device) -> belief; update_fn(belief, step, t) -> belief (differentiable);
      log_prob_fn(belief, step, t) -> [B] summed log-likelihood of step t's observed outcomes;
      count_fn(step) -> [B] number of observed outcomes (default: ``step.outcomes.y_mask.sum(-1)`` or 1).
    Returns {"loss": scalar, "per_k": {k: nll per outcome}, "n_query": {k: outcomes scored}, "per_query":
    {k: [B, T] log-probs (0 outside the query)}}. Rows whose support crosses an episode boundary are excluded for that
    k; queries never cross the support's episode.
    """

    def __init__(self, ks: Sequence[int] = SUPPORT_SIZES, weights: Optional[Sequence[float]] = None,
                 max_query: Optional[int] = None):
        self.ks = tuple(int(k) for k in ks)
        if any(k < 0 for k in self.ks) or len(set(self.ks)) != len(self.ks):
            raise ValueError("support sizes must be distinct non-negative integers")
        self.weights = None if weights is None else tuple(float(w) for w in weights)
        if self.weights is not None and len(self.weights) != len(self.ks):
            raise ValueError("one weight per support size")
        self.max_query = max_query

    def __call__(self, steps, init_fn: Callable, update_fn: Callable, log_prob_fn: Callable,
                 count_fn: Optional[Callable] = None) -> dict:
        steps = list(steps.steps) if isinstance(steps, SequenceBatch) else list(steps)
        T = len(steps)
        if T < 2:
            raise ValueError("need at least two steps (one support, one query)")
        _check_disjoint_ids(steps)
        count_fn = count_fn or _default_count
        done = [_done_of(s) for s in steps]
        B, dev = _infer_B(steps)
        ks = sorted(k for k in self.ks if k < T)
        if not ks:
            raise ValueError(f"no support size < sequence length {T}")
        belief = init_fn(B, dev)
        ep = _episode_index(done, B, dev)                              # [B,T]
        per_k, n_q, per_query, losses, wts = {}, {}, {}, [], []
        t_upd = 0
        for k in ks:
            while t_upd < k:                                           # support: steps[:k] only
                belief = update_fn(belief, steps[t_upd], t_upd)
                t_upd += 1
            ep_ref = ep[:, k - 1] if k > 0 else ep[:, 0]
            row_ok = ep_ref == ep[:, 0]                                # the support stays inside one episode
            hi = T if self.max_query is None else min(T, k + self.max_query)
            tot = torch.zeros((), device=dev)
            cnt = torch.zeros((), device=dev)
            pq = torch.zeros(B, T, device=dev)
            for j in range(k, hi):
                valid = row_ok & (ep[:, j] == ep_ref)
                if not bool(valid.any()):
                    continue
                lp = log_prob_fn(belief, steps[j], j).float()           # query: scored, never used to update
                c = count_fn(steps[j]).float().to(dev)
                v = valid.float()
                tot = tot + (lp * v).sum()
                cnt = cnt + (c * v).sum()
                pq[:, j] = torch.where(valid, lp, torch.zeros_like(lp))
            if float(cnt) == 0:
                continue
            nll = -tot / cnt
            per_k[k], n_q[k], per_query[k] = nll, int(cnt), pq
            losses.append(nll)
            wts.append(1.0 if self.weights is None else self.weights[self.ks.index(k)])
        if not losses:
            raise ValueError("no query outcomes to score")
        w = torch.tensor(wts, device=losses[0].device)
        loss = (torch.stack(losses) * w).sum() / w.sum()
        return {"loss": loss, "per_k": {k: float(v.detach()) for k, v in per_k.items()}, "n_query": n_q,
                "per_query": per_query, "per_k_tensor": per_k}


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


def belief_callables_from_modules(belief_module, outcome_head, Z_seq: Sequence[Tensor], query_Z: str = "support_end"):
    """Wrap workstream B/C modules (contract `BeliefModuleP`, `OutcomeHeadP`) into the three callables.

    update(fast, step, t) = belief_module.update(fast, Z_seq[t], step.act, step.outcomes, outcome_head)[0];
    log_prob(fast, step, t) = log sum_k w_k prod_q p(y_q | Z, particle k) using
    ``outcome_head.dist(outcome_head.features(Z, outcomes), outcomes, cond=fast.particles)`` (batch [B,K,Q]).

    Strict separation through the latent state: with ``query_Z="support_end"`` every query is scored from the latent
    state at the end of the support (``Z_seq[k]``, recorded via the most recent update), never from a state that has
    already absorbed later (query) observations. Pass ``query_Z="own"`` only for one-step-ahead scoring where
    ``Z_seq[j]`` provably precedes the outcome of step j (it is computed before o_j by the filter)."""
    state = {"last_t": -1}

    def init(B, device):
        state["last_t"] = -1
        return belief_module.init_fast(B, device)

    def update(fast, step, t):
        state["last_t"] = t
        return belief_module.update(fast, Z_seq[t], step.act, step.outcomes, outcome_head)[0]

    def log_prob(fast, step, t):
        zt = t if query_Z == "own" else state["last_t"] + 1
        zt = max(0, min(zt, len(Z_seq) - 1))
        Z = Z_seq[zt]
        feat = outcome_head.features(Z, step.outcomes)
        dist = outcome_head.dist(feat, step.outcomes, cond=fast.particles)
        y = step.outcomes.y
        y = y.unsqueeze(1).expand(-1, fast.particles.shape[1], -1)
        lp = dist.log_prob(y)
        lp = torch.where(step.outcomes.y_mask.unsqueeze(1), lp, torch.zeros_like(lp))
        return mixture_log_prob(lp, fast.logw, step.outcomes.y_mask)

    return init, update, log_prob


# ================================================================================================ J_meta

class MetaRLObjective:
    """J_meta estimator over multi-trial episodes.

    Rewards entering the actor-critic: ``r'_t = w_{trial(t)} r_t - cost_coef * cost_t + intrinsic_t`` where
    ``trial_weights[k]`` weights trial k's competence (later trials weighted more emphasise *becoming* competent),
    ``cost_t`` is the observable experiment cost and ``intrinsic_t`` the (already ``lp_coef``-scaled) learning-progress
    reward, included only if ``use_intrinsic``. Trial boundaries are not terminal, so credit flows from later trials to
    earlier decisions. ``j_meta(traj)`` reports the undiscounted per-episode objective value."""

    def __init__(self, actor_critic: ActorCriticObjective, trial_weights: Optional[Sequence[float]] = None,
                 cost_coef: float = 0.0, use_intrinsic: bool = True):
        self.ac = actor_critic
        self.trial_weights = None if trial_weights is None else torch.tensor([float(w) for w in trial_weights])
        self.cost_coef = float(cost_coef)
        self.use_intrinsic = use_intrinsic

    def shaped_reward(self, traj: Trajectory) -> Tensor:
        r = traj.reward.float()
        if self.trial_weights is not None:
            if traj.trial is None:
                raise ValueError("trial weights need Trajectory.trial")
            tw = self.trial_weights.to(r.device)
            idx = traj.trial.long().clamp(0, len(tw) - 1)
            r = r * tw[idx]
        if self.cost_coef and traj.cost is not None:
            r = r - self.cost_coef * traj.cost.float()
        if self.use_intrinsic and traj.intrinsic is not None:
            r = r + traj.intrinsic.float()
        return r * traj.mask.float()

    def j_meta(self, traj: Trajectory) -> Tensor:
        """[B] value of the objective per episode (sum of shaped rewards)."""
        return self.shaped_reward(traj).sum(1)

    def per_trial_return(self, traj: Trajectory) -> Tensor:
        """[B, n_trials] raw extrinsic return per trial (competence curve within the meta-episode)."""
        if traj.trial is None:
            return traj.reward.sum(1, keepdim=True)
        n = int(traj.trial[traj.mask].max()) + 1 if bool(traj.mask.any()) else 1
        out = torch.zeros(traj.B, n, device=traj.device)
        out.scatter_add_(1, traj.trial.long().clamp_min(0), traj.reward.float() * traj.mask.float())
        return out

    def prepare(self, traj: Trajectory, current_version: int) -> tuple[Trajectory, Prepared]:
        shaped = traj.with_reward(self.shaped_reward(traj))
        return shaped, self.ac.prepare(shaped, current_version)

    def loss(self, shaped: Trajectory, prep: Prepared, new_logp, entropy, new_value=None, value_logits=None,
             value_head=None) -> dict:
        d = self.ac.loss(shaped, prep, new_logp, entropy, new_value, value_logits, value_head)
        d["meta"] = d["actor"]
        d["j_meta"] = float((shaped.reward * shaped.mask.float()).sum(1).mean())   # shaped rewards already applied
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
    left-aligned; trial boundaries are not terminal; the last decision of the last trial is)."""

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
        B = len(seeds)
        if B != self.adapter.B:
            raise ValueError("one seed per adapter environment")
        state = policy.initial_state(B) if state is None else state
        rows: list = []
        names: list = []
        seen: list = []
        trial_ret = torch.zeros(B, n_trials)
        for k in range(n_trials):
            ob = self.adapter.reset(seeds)
            goals = self.adapter.goals()
            obs = [list(o) for o in ob.observations]
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
                active = torch.tensor([not d for d in done], device=device)
                tools = self.adapter.available_tools()
                budget = self._budget(goals, used, spent, now, t0, device)
                actions, logp, value, state, rec = policy.act(state, obs, tools, budget, goals, active, greedy)
                actions = [a if bool(active[b]) else None for b, a in enumerate(actions)]
                results = self.adapter.execute(actions)
                dur = [0.0 if r is None else float(r.duration) + self.think_days for r in results]
                ev = self.adapter.advance_time(dur)
                rew, cost = [], []
                for b in range(B):
                    cur = self.evaluator.evaluate_goal(b).metrics[self.return_key]
                    rew.append(float(cur - prev_ret[b]) if bool(active[b]) else 0.0)
                    prev_ret[b] = cur
                    c = 0.0 if results[b] is None else float(results[b].cost)
                    cost.append(c)
                    if bool(active[b]):
                        used[b] += 1
                        spent[b] += c
                obs = [([] if results[b] is None else list(results[b].observations)) + list(ev.observations[b])
                       for b in range(B)]
                now = list(ev.t)
                new_done = list(ev.done)
                last_trial = k == n_trials - 1
                rows.append({"active": active.clone(), "reward": torch.tensor(rew, device=device),
                             "dt": torch.tensor(dur, device=device),
                             "done": torch.tensor([bool(new_done[b]) and last_trial and bool(active[b])
                                                   for b in range(B)], device=device),
                             "logp": logp.detach().float().to(device), "value": value.detach().float().to(device),
                             "cost": torch.tensor(cost, device=device),
                             "trial": torch.full((B,), k, dtype=torch.long, device=device),
                             "rec": {kk: v.detach() for kk, v in (rec or {}).items()}})
                names.append([a.tool if a is not None else "" for a in actions])
                seen.append(obs)
                for b in range(B):
                    if bool(active[b]):
                        trial_ret[b, k] += rew[b]
                done = new_done
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
        return RunResult(traj, trial_ret, tool_names, obs_after, state, {"n_decisions": int(mask.sum())})
