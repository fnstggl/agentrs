"""Latent planning over CAM-generated candidate actions at a matched compute budget (spec 7.2).

All planners share ``Imagination.step`` (belief-aware: every imagined row updates its own FastState clone with the
imagined outcomes) and count world-model transition evaluations (``WorldModel.n_transitions``). A budget is the
number of transitions allowed per real environment; each planner sizes itself to stay within it. A budget below a
planner's minimum cost (``min_budget``: policy rollouts n*h, beam s*(n + (h-1)*W*n), CEM iterations*s*h) is refused
with ``ValueError`` instead of being silently exceeded, and every plan checks after the fact that it used at most
budget x B transitions (``RuntimeError`` otherwise). ``compare_planners`` refuses a budget any compared planner
cannot meet, so compared planners always run within the same budget.

* ``PolicyRolloutPlanner`` -- closed loop: every candidate first action is followed by the policy, which chooses
  each later action from the row's *imagined, revised* belief; candidate value = mean imagined return over
  stochastic futures (n_samples = budget // (n * horizon)).
* ``BeamPlanner`` -- open loop: beam search over sequences of candidate actions; a node holds ``n_samples``
  stochastic futures and is scored by its mean discounted return plus bootstrap value; the best ``beam_width``
  nodes per environment are expanded with every candidate at each depth.
* ``SamplingPlanner`` -- open loop, cross-entropy method over candidate-index sequences: per-depth categorical
  distributions are refitted to the elite plans for ``iterations`` rounds.

Each returns a ``PlanResult`` whose chosen action is a copy of the CAM candidate with ``ActionSource.CAM_PLANNER``
and whose ``imagined`` entries are ready for ``DecisionTrace.imagined``. The choice is always the candidate with the
best imagined return under that planner's own estimate.

Behaviour probability of planner actions: the executed action is a deterministic argmax over imagined returns of
sampled candidates, so its behaviour probability mu is not the decoder's probability. A planner action therefore has
``logp = None`` (unknown mu: off-policy learners must not form importance ratios from it); the decoder's own
log-probabilities are kept for diagnostics under ``logp_parts["decoder"]`` / ``logp_parts["decoder_<part>"]`` and
``proposal["behavior"] == "planner_argmax"``.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from ..interfaces import ActionSource, FastState, Provenance, TokenSet
from .imagination import (Candidates, ImaginationHooks, ImagState, Imagination, candidate_scores, fast_select,
                          tokens_select, trace_summaries)


@dataclass
class PlanResult:
    choice: Any                  # [B] long, chosen candidate index per environment
    actions: list                # [B] Action (copies of the chosen candidates, source CAM_PLANNER)
    scores: Any                  # [B, n] imagined value per first candidate (-inf: never evaluated / pruned)
    imagined: list               # [B] list of DecisionTrace.imagined dicts
    n_model_evals: int           # world-model transitions used (all environments)
    method: str
    info: dict = field(default_factory=dict)


def _select_state(st: ImagState, idx: torch.Tensor) -> ImagState:
    """Copy rows ``idx`` of an imagined state (used to branch beam nodes)."""
    il = idx.tolist()
    return ImagState(st.Z[idx].clone(), fast_select(st.fast, idx), tokens_select(st.goal, idx),
                     tokens_select(st.belief, idx), [st.actions[i] for i in il], st.act_tokens.select(idx),
                     st.disc[idx].clone(), st.ret[idx].clone(), st.penalty[idx].clone(), st.elapsed[idx].clone(),
                     st.branch_of[idx].clone(), st.first_candidate[idx].clone(),
                     None if st.value is None else st.value[idx].clone(), None if st.R is None else st.R[idx].clone())


def _set_actions(st: ImagState, cand: Candidates, which: torch.Tensor) -> None:
    """Set every row's next action to candidate ``which[row]`` of its own environment."""
    n = cand.n
    rows = st.branch_of * n + which
    st.act_tokens = cand.tokens.select(rows)
    st.actions = [cand.actions[int(b)][int(i)] for b, i in zip(st.branch_of.tolist(), which.tolist())]


class _Planner:
    method = "base"

    def __init__(self, imagination: Imagination, budget: int, horizon: Optional[int] = None,
                 score_temperature: float = 1.0):
        self.im = imagination
        self.budget = int(budget)
        self.horizon = horizon or imagination.horizon
        self.score_temperature = score_temperature

    def _value(self, st: ImagState) -> torch.Tensor:
        """Per-row imagined value: discounted return so far + discounted bootstrap - uncertainty penalty."""
        v = st.value if st.value is not None else torch.zeros_like(st.ret)
        return st.ret + st.disc * v - self.im.uncertainty_coef * st.penalty

    def min_budget(self, n: int, horizon: Optional[int] = None) -> int:
        """Smallest per-environment transition budget this planner can run within for n candidates."""
        raise NotImplementedError

    def _check_budget(self, n: int) -> None:
        need = self.min_budget(n, self.horizon)
        if self.budget < need:
            raise ValueError(f"{self.method} planner needs a budget of at least {need} transitions per environment "
                             f"for {n} candidates at horizon {self.horizon}; got {self.budget}")

    def _check_used(self, n_evals: int, B: int) -> None:
        if n_evals > self.budget * B:
            raise RuntimeError(f"{self.method} planner used {n_evals} transitions, above its budget "
                               f"{self.budget} x {B} environments")

    def _finish(self, cand: Candidates, scores: torch.Tensor, imagined: list, n_evals: int, info: dict) -> PlanResult:
        self._check_used(n_evals, cand.B)
        choice = scores.argmax(1)
        acts = []
        for b in range(cand.B):
            i = int(choice[b])
            a = copy.deepcopy(cand.actions[b][i])
            a.source = ActionSource.CAM_PLANNER
            sc = scores[b]
            finite = torch.isfinite(sc)
            probs = torch.zeros_like(sc)
            probs[finite] = torch.softmax(sc[finite] / self.score_temperature, 0)
            a.proposal = {"n": cand.n, "probs": [float(p) for p in probs], "chosen": i,
                          "scores": [float(x) if math.isfinite(float(x)) else None for x in sc],
                          "method": self.method, "deterministic": True, "behavior": "planner_argmax",
                          "budget": self.budget}
            # behaviour probability of an argmax over sampled candidates is unknown: no logp (see module doc)
            parts = {f"decoder_{k}": v for k, v in (a.logp_parts or {}).items()}
            parts["decoder"] = a.logp
            a.logp_parts = parts
            a.logp = None
            acts.append(a)
        return PlanResult(choice, acts, scores, imagined, n_evals, self.method, info)


class PolicyRolloutPlanner(_Planner):
    """Closed-loop policy-generated rollouts from every candidate first action."""
    method = "policy_rollout"

    def min_budget(self, n: int, horizon: Optional[int] = None) -> int:
        return n * (horizon or self.horizon)

    def plan(self, Z, fast: Optional[FastState], goal: Optional[TokenSet], cand: Candidates,
             hooks: ImaginationHooks, generator: Optional[torch.Generator] = None) -> PlanResult:
        h, n = self.horizon, cand.n
        self._check_budget(n)
        s = self.budget // (n * h)
        traj = self.im.rollout(Z, fast, goal, cand, hooks, horizon=h, n_samples=s, generator=generator)
        scores = candidate_scores(traj, cand.B, n)
        imagined = [[dict(e, method=self.method) for e in trace_summaries(traj, b, n)] for b in range(cand.B)]
        return self._finish(cand, scores, imagined, traj.n_model_evals, {"n_samples": s, "horizon": h,
                                                                           "trajectory": traj})


class BeamPlanner(_Planner):
    """Open-loop beam search over candidate sequences with stochastic futures per node."""
    method = "beam"

    def __init__(self, imagination: Imagination, budget: int, horizon: Optional[int] = None, beam_width: int = 2,
                 n_samples: Optional[int] = None, **kw):
        super().__init__(imagination, budget, horizon, **kw)
        self.beam_width = beam_width
        self.n_samples = n_samples

    def _per_sample(self, n: int, h: int) -> int:
        W = min(self.beam_width, n)
        return n + (h - 1) * W * n

    def min_budget(self, n: int, horizon: Optional[int] = None) -> int:
        return (self.n_samples or 1) * self._per_sample(n, horizon or self.horizon)

    def plan(self, Z, fast, goal, cand: Candidates, hooks: ImaginationHooks, generator=None) -> PlanResult:
        im, h, n, B = self.im, self.horizon, cand.n, cand.B
        self._check_budget(n)
        W = min(self.beam_width, n)
        per_sample = self._per_sample(n, h)
        s = self.n_samples or self.budget // per_sample
        dev = Z.device
        n0 = im.wm.n_transitions
        with torch.no_grad():
            st = im.init_state(Z, fast, goal, cand, s, hooks)            # rows (b, i, sample)
            st, rec0 = im.step(st, hooks, 0, generator)
            node_root = torch.arange(B * n, device=dev) % n              # node -> root candidate
            node_env = torch.arange(B * n, device=dev) // n
            n_nodes = B * n
            first_scores = self._value(st).view(B, n, s).mean(-1)
            imagined0 = [[dict(e, method=self.method) for e in _first_step_summaries(rec0, st, b, n, s, first_scores)]
                         for b in range(B)]
            for d in range(1, h):
                node_val = self._value(st).view(n_nodes, s).mean(-1)
                # keep the best W nodes per environment
                vals = node_val.view(B, -1)
                k = min(W, vals.shape[1])
                top = vals.topk(k, dim=1).indices                         # [B, k] local node index
                keep = (top + torch.arange(B, device=dev).unsqueeze(1) * vals.shape[1]).reshape(-1)   # global node ids
                # children: every kept node x every candidate, each with the node's s futures
                child_node = keep.repeat_interleave(n)                    # [B*k*n]
                child_act = torch.arange(n, device=dev).repeat(B * k)
                row_idx = (child_node.unsqueeze(1) * s + torch.arange(s, device=dev)).reshape(-1)
                st = _select_state(st, row_idx)
                _set_actions(st, cand, child_act.repeat_interleave(s))
                st, _ = im.step(st, hooks, d, generator)
                node_root = node_root[child_node]
                n_nodes = child_node.numel()
            final = self._value(st).view(n_nodes, s).mean(-1).view(B, -1)
            roots = node_root.view(B, -1)
        scores = torch.full((B, n), float("-inf"), device=dev)
        scores.scatter_reduce_(1, roots, final, reduce="amax")
        imagined = imagined0
        for b in range(B):
            for e in imagined[b]:
                e["value"] = float(scores[b, e["candidate"]]) if torch.isfinite(scores[b, e["candidate"]]) else None
                e["horizon"] = h
                e["pruned"] = not bool(torch.isfinite(scores[b, e["candidate"]]))
        return self._finish(cand, scores, imagined, im.wm.n_transitions - n0,
                            {"n_samples": s, "beam_width": W, "horizon": h, "first_step_scores": first_scores})


class SamplingPlanner(_Planner):
    """Cross-entropy method over open-loop candidate-index sequences."""
    method = "cem"

    def __init__(self, imagination: Imagination, budget: int, horizon: Optional[int] = None, iterations: int = 3,
                 n_samples: int = 2, elite_frac: float = 0.25, smoothing: float = 0.7, **kw):
        super().__init__(imagination, budget, horizon, **kw)
        self.iterations, self.n_samples = iterations, n_samples
        self.elite_frac, self.smoothing = elite_frac, smoothing

    def min_budget(self, n: int, horizon: Optional[int] = None) -> int:
        return self.iterations * self.n_samples * (horizon or self.horizon)

    def plan(self, Z, fast, goal, cand: Candidates, hooks: ImaginationHooks, generator=None) -> PlanResult:
        im, h, n, B = self.im, self.horizon, cand.n, cand.B
        self._check_budget(n)
        s = self.n_samples
        N = self.budget // (self.iterations * s * h)                  # plans per iteration per environment
        dev = Z.device
        n0 = im.wm.n_transitions
        P = torch.full((B, h, n), 1.0 / n, device=dev)
        best = torch.full((B, n), float("-inf"), device=dev)          # best single plan per first candidate (report)
        best_plan = torch.zeros(B, n, h, dtype=torch.long, device=dev)
        score = torch.full((B, n), float("-inf"), device=dev)         # mean value of the latest plans per candidate
        n_elite = max(1, int(round(self.elite_frac * N)))
        first_out = {}
        with torch.no_grad():
            for it in range(self.iterations):
                plans = torch.multinomial(P.reshape(B * h, n), N, replacement=True,
                                          generator=generator).view(B, h, N).permute(0, 2, 1)   # [B,N,h]
                # imagined rows (b, plan, sample): start from the plan's first candidate
                first = plans[:, :, 0]                                                       # [B,N]
                sub = Candidates([[cand.actions[b][int(i)] for i in first[b].tolist()] for b in range(B)],
                                 cand.tokens.select((torch.arange(B, device=dev).unsqueeze(1) * n + first).reshape(-1)))
                st = im.init_state(Z, fast, goal, sub, s, hooks)
                # restore the true candidate index of each row
                st.first_candidate = first.reshape(-1).repeat_interleave(s)
                for d in range(h):
                    if d > 0:
                        which = plans[:, :, d].reshape(-1).repeat_interleave(s)
                        _set_actions(st, cand, which)
                    st, rec = im.step(st, hooks, d, generator)
                    if d == 0 and it == 0:
                        first_out = {"rec": rec, "first": st.first_candidate.clone(), "branch": st.branch_of.clone()}
                val = self._value(st).view(B, N, s).mean(-1)                                 # [B,N]
                # candidate score = mean over this iteration's plans that start with it (the max over noisy plan
                # values would be optimistically biased); candidates not sampled keep their latest score
                oh = torch.nn.functional.one_hot(first, n).float()                          # [B,N,n]
                cnt = oh.sum(1)
                it_mean = (oh * val.unsqueeze(-1)).sum(1) / cnt.clamp(min=1)
                score = torch.where(cnt > 0, it_mean, score)
                # bookkeeping of the best plan per first candidate
                for b in range(B):
                    for j in range(N):
                        i = int(first[b, j])
                        if val[b, j] > best[b, i]:
                            best[b, i] = val[b, j]
                            best_plan[b, i] = plans[b, j]
                elite = val.topk(n_elite, dim=1).indices                                     # [B,E]
                ep = plans.gather(1, elite.unsqueeze(-1).expand(-1, -1, h))                  # [B,E,h]
                freq = torch.zeros(B, h, n, device=dev).scatter_add_(2, ep.permute(0, 2, 1),
                                                                     torch.ones(B, h, n_elite, device=dev))
                freq = freq / n_elite
                P = self.smoothing * freq + (1 - self.smoothing) * P
                P = (P + 1e-3) / (P + 1e-3).sum(-1, keepdim=True)
        imagined = []
        rec = first_out.get("rec")
        for b in range(B):
            ent = []
            for i in range(n):
                e = {"candidate": i, "horizon": h, "method": self.method,
                     "value": float(score[b, i]) if torch.isfinite(score[b, i]) else None,
                     "best_plan_value": float(best[b, i]) if torch.isfinite(best[b, i]) else None,
                     "plan": [int(x) for x in best_plan[b, i].tolist()] if torch.isfinite(best[b, i]) else None,
                     "first_prob": float(P[b, 0, i]), "provenance": Provenance.IMAGINED.value}
                if rec is not None and rec.outcome_mean.shape[-1]:
                    sel = (first_out["branch"] == b) & (first_out["first"] == i)
                    if bool(sel.any()):
                        e["outcomes"] = {f"q{j}": float(v) for j, v in enumerate(rec.outcome_mean[sel].mean(0))}
                ent.append(e)
            imagined.append(ent)
        return self._finish(cand, score, imagined, im.wm.n_transitions - n0,
                            {"plans_per_iter": N, "n_samples": s, "iterations": self.iterations, "final_P": P})


def _first_step_summaries(rec, st: ImagState, b: int, n: int, s: int, first_scores) -> list:
    out = []
    for i in range(n):
        sel = (st.branch_of == b) & (st.first_candidate == i)
        e = {"candidate": i, "first_step_value": float(first_scores[b, i]),
             "rewards": [float(rec.reward[sel].mean())], "model_error": float(rec.model_error[sel].mean()),
             "n_samples": s, "provenance": Provenance.IMAGINED.value}
        if rec.outcome_mean.shape[-1]:
            e["outcomes"] = {f"q{j}": float(v) for j, v in enumerate(rec.outcome_mean[sel].mean(0))}
        out.append(e)
    return out


PLANNERS = {"policy_rollout": PolicyRolloutPlanner, "beam": BeamPlanner, "cem": SamplingPlanner}


def compare_planners(imagination: Imagination, budget: int, Z, fast, goal, cand: Candidates,
                     hooks: ImaginationHooks, horizon: Optional[int] = None, seed: int = 0, kinds=None) -> dict:
    """Run every planner at the same transition budget on the same real state; {kind: PlanResult}.

    Refuses (ValueError) a budget that any of the compared planners cannot meet, so no planner gets more compute."""
    planners = {kind: PLANNERS[kind](imagination, budget, horizon) for kind in (kinds or PLANNERS)}
    need = {kind: p.min_budget(cand.n) for kind, p in planners.items()}
    short = {k: v for k, v in need.items() if v > budget}
    if short:
        raise ValueError(f"budget {budget} per environment is below the minimum cost of {short}; compared planners "
                         f"must all run within the same budget")
    out = {}
    for kind, p in planners.items():
        g = torch.Generator(device=Z.device).manual_seed(seed)
        out[kind] = p.plan(Z, fast, goal, cand, hooks, generator=g)
    return out
