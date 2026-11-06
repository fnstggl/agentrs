"""Short-horizon model-predictive planning with a learned world model.

At each real decision::

    1. the policy proposes the top-K candidate actions (by pi(a|s))
    2. for each, the world model predicts P(reward), P(terminal), event
       probabilities with ensemble disagreement, and samples B next-state diffs
    3. imagined next observations are built by applying the diffs; optionally
       the search branches further (policy proposals on imagined states)
    4. leaves are scored by the critic V(s', g)
    5. score(a) = Q(a) - lambda_u * U(a) - cost(a) + beta * log pi(a|s)
       with Q(a) = p_r + gamma * (1 - p_term) * E_b[V or recursive Q]
    6. the chosen action is executed in the REAL environment; the agent then
       observes the real outcome and replans from it (receding horizon).

Imagined states are never fed back as real observations. The planner records
both log pi(a|s) and the log-probability of its own selection rule
(``behavior_logprob``) so planner-collected data is handled as off-policy data.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
from pydantic import ConfigDict, BaseModel

from agentrs.actions.candidates import candidates_from_observation
from agentrs.actions.schema import Action, ActionType
from agentrs.agent.belief import BeliefState, HistoryItem
from agentrs.agent.runtime import Decision
from agentrs.data.diff import StateDiff
from agentrs.data.schemas import WMPredictionRecord
from agentrs.environments.base import Observation
from agentrs.models.critic import Critic
from agentrs.models.policy import LMPolicy
from agentrs.models.world_model import WorldModel


class PlannerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    num_candidates: int = 4
    horizon: int = 1  # imagined depth beyond the first action
    num_branches: int = 1  # sampled next states per (state, action)
    branch_candidates: int = 2  # policy proposals expanded at imagined states
    gamma: float = 0.95
    uncertainty_penalty: float = 1.0
    diff_uncertainty_weight: float = 0.0
    action_cost: float = 0.0
    policy_prior_weight: float = 0.1
    selection: str = "argmax"  # argmax | softmax
    selection_temperature: float = 0.05
    epsilon: float = 0.0
    max_wm_calls: int = 24  # compute budget per real decision
    generate_next_states: bool = True
    greedy_diffs: bool = True


@dataclass
class _Node:
    action: Action
    prior_logp: float
    p_reward: float = 0.0
    p_term: float = 0.0
    unc: float = 0.0
    q: float = 0.0
    score: float = 0.0
    diffs: list[str] = field(default_factory=list)
    event_probs: list[float] = field(default_factory=list)
    epistemic: float = 0.0
    diff_nll: float = 0.0


def _outcome_from_diff(d: StateDiff) -> str:
    if d.error:
        return "error: " + d.error[:60]
    if d.url:
        return f"navigated to {d.url}"
    return "ok"


class MPCPlanner:
    def __init__(self, policy: LMPolicy, world_model: WorldModel, critic: Critic, cfg: PlannerConfig,
                 seed: int = 0, name: str = "mpc"):
        self.policy, self.wm, self.critic, self.cfg = policy, world_model, critic, cfg
        self.rng = random.Random(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.name = name
        self.wm_calls = 0
        self.stats: dict[str, float] = {"decisions": 0, "wm_calls": 0, "changed_choice": 0}

    def begin_episode(self, belief):
        pass

    # ---------------------------------------------------------------- helpers
    def _history(self, belief: BeliefState, extra: list[HistoryItem]) -> list[tuple[str, str]]:
        return [(h.action, h.outcome) for h in belief.history + extra]

    @torch.no_grad()
    def _values(self, belief: BeliefState, states: list[tuple[Observation, list[HistoryItem]]]) -> list[float]:
        if not states:
            return []
        prompts = [belief.prompt(observation=o, history=belief.history + h) for o, h in states]
        return self.critic.value(prompts).tolist()

    @torch.no_grad()
    def _expand(self, belief: BeliefState, obs: Observation, hist: list[HistoryItem], actions: list[Action],
                priors: list[float], depth: int) -> list[_Node]:
        cfg = self.cfg
        hs = self._history(belief, hist)
        mean, std = self.wm.predict_batch_heads(belief.task.goal, obs, [a.to_str() for a in actions], hs)
        self.wm_calls += len(actions)
        nodes = []
        for i, (a, lp) in enumerate(zip(actions, priors)):
            m, s = mean[i].tolist(), std[i].tolist()
            n = _Node(action=a, prior_logp=lp, p_reward=m[-2], p_term=m[-1], event_probs=m[:-2])
            n.epistemic = float(sum(s) / len(s))
            n.unc = s[-2] + s[-1] + n.epistemic
            nodes.append(n)
        # imagined next states
        leaf_states: list[tuple[Observation, list[HistoryItem]]] = []
        owners: list[int] = []
        expand = [i for i, n in enumerate(nodes)
                  if n.action.type != ActionType.FINISH and cfg.generate_next_states]
        budget = max(0, (cfg.max_wm_calls - self.wm_calls) // max(1, cfg.num_branches))
        expand = expand[:budget]
        gen_prompts, gen_owner = [], []
        for i in expand:
            pr = self.wm.prompt(belief.task.goal, obs, nodes[i].action.to_str(), hs)
            gen_prompts += [pr] * cfg.num_branches
            gen_owner += [i] * cfg.num_branches
        samples = self.wm.sample_diffs_batch(gen_prompts, greedy=cfg.greedy_diffs and cfg.num_branches == 1,
                                             generator=self.gen)
        self.wm_calls += len(gen_prompts)
        per_node: dict[int, list[float]] = {}
        for (d_text, lp_tok), i in zip(samples, gen_owner):
            n = nodes[i]
            d = StateDiff.from_text(d_text)
            n.diffs.append(d_text)
            per_node.setdefault(i, []).append(lp_tok)
            nxt = d.apply(obs, id_prefix=f"n{depth}x")
            leaf_states.append((nxt, hist + [HistoryItem(action=n.action.to_str(), outcome=_outcome_from_diff(d))]))
            owners.append(i)
        for i, lps_ in per_node.items():
            nodes[i].diff_nll = float(-sum(lps_) / len(lps_))
            nodes[i].unc += cfg.diff_uncertainty_weight * nodes[i].diff_nll
        cont = [[] for _ in nodes]
        if depth < cfg.horizon and leaf_states and self.wm_calls < cfg.max_wm_calls:
            for (nxt, h2), i in zip(leaf_states, owners):
                cands = belief.augment_candidates(candidates_from_observation(nxt, belief.task.goal), nxt)
                acts, lps = self._propose(belief, nxt, h2, cands, cfg.branch_candidates)
                sub = self._expand(belief, nxt, h2, acts, lps, depth + 1)
                cont[i].append(max(x.q - cfg.uncertainty_penalty * x.unc for x in sub) if sub else 0.0)
        else:
            vals = self._values(belief, leaf_states)
            for v, i in zip(vals, owners):
                cont[i].append(v)
        for i, n in enumerate(nodes):
            if n.action.type == ActionType.FINISH:
                future = 0.0
            elif cont[i]:
                future = sum(cont[i]) / len(cont[i])
            else:  # no imagined state (budget exhausted): fall back to value of the current state
                future = self._values(belief, [(obs, hist)])[0]
            n.q = n.p_reward + cfg.gamma * (1 - n.p_term) * future
        return nodes

    @torch.no_grad()
    def _propose(self, belief: BeliefState, obs: Observation, hist: list[HistoryItem], cands: list[Action], k: int):
        cands = self.policy.truncate_candidates(cands)
        prompt = belief.prompt(observation=obs, history=belief.history + hist)
        lp = self.policy.log_probs([prompt], [[c.to_str() for c in cands]])[0]
        top = torch.topk(lp, min(k, len(cands)))
        return [cands[int(i)] for i in top.indices], [float(v) for v in top.values]

    # ---------------------------------------------------------------- decide
    @torch.no_grad()
    def decide(self, belief: BeliefState, candidates: list[Action]) -> Decision:
        cfg = self.cfg
        cands, lp = self.policy.distribution(belief, candidates)
        cand_strs = [c.to_str() for c in cands]
        if not cfg.enabled:
            i = int(torch.argmax(lp))
            return Decision(action=cands[i], candidates=cand_strs, chosen_index=i, policy_logprob=float(lp[i]),
                            behavior_logprob=0.0, selection="policy")
        self.wm_calls = 0
        k = min(cfg.num_candidates, len(cands))
        top = torch.topk(lp, k)
        acts = [cands[int(i)] for i in top.indices]
        priors = [float(v) for v in top.values]
        nodes = self._expand(belief, belief.observation, [], acts, priors, depth=1)
        for n in nodes:
            n.score = n.q - cfg.uncertainty_penalty * n.unc - cfg.action_cost + cfg.policy_prior_weight * n.prior_logp
        scores = torch.tensor([n.score for n in nodes])
        if cfg.selection == "softmax":
            probs = torch.softmax(scores / cfg.selection_temperature, 0)
        else:
            probs = torch.zeros(len(nodes))
            probs[int(torch.argmax(scores))] = 1.0
        if cfg.epsilon > 0:  # mixture with uniform over the full candidate set
            full = torch.full((len(cands),), cfg.epsilon / len(cands))
            for j, n in enumerate(nodes):
                full[cand_strs.index(n.action.to_str())] += (1 - cfg.epsilon) * probs[j]
            ci = int(torch.multinomial(full, 1, generator=self.gen))
            beh = math.log(float(full[ci]))
        else:
            j = int(torch.multinomial(probs, 1, generator=self.gen)) if cfg.selection == "softmax" else int(
                torch.argmax(probs))
            ci = cand_strs.index(nodes[j].action.to_str())
            beh = math.log(max(float(probs[j]), 1e-12))
        chosen = cands[ci]
        node = next((n for n in nodes if n.action.to_str() == chosen.to_str()), None)
        self.stats["decisions"] += 1
        self.stats["wm_calls"] += self.wm_calls
        self.stats["changed_choice"] += int(chosen.to_str() != acts[0].to_str())
        wm_rec = None
        if node is not None:
            wm_rec = WMPredictionRecord(event_probs=node.event_probs, reward_prob=node.p_reward,
                                        terminal_prob=node.p_term, diff_text=node.diffs[0] if node.diffs else None,
                                        epistemic=node.epistemic)
        return Decision(action=chosen, candidates=cand_strs, chosen_index=ci, policy_logprob=float(lp[ci]),
                        behavior_logprob=beh, selection="planner", wm_prediction=wm_rec,
                        info={"scores": {n.action.to_str(): round(n.score, 4) for n in nodes},
                              "q": {n.action.to_str(): round(n.q, 4) for n in nodes},
                              "wm_calls": self.wm_calls})
