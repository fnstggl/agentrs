"""Consequence-based planning over generated content.

At each decision:
 1. the policy chooses among operations (or finishing);
 2. for a content operation, ``CandidateContentGenerator`` drafts K content variants;
 3. every COMPLETE candidate action (operation + exact content) goes through the
    content-conditioned world model: P(objective met), P(reward), outcome uncertainty;
 4. score = P(objective) - lambda * epistemic std (unreliable predictions are penalised);
 5. optional revision rounds draft more variants while the best score is below ``target_score``;
 6. the best authorised candidate is executed in the real environment — or the planner
    DECLINES (``finish()``) when the best score is below ``min_score_to_act``, uncertainty is
    above ``max_uncertainty``, or the permission policy forbids the operation;
 7. the real outcome is recorded by the runtime (predicted values are stored next to it and are
    never used as success).

Planner-chosen content is not a sample from the policy: it is marked ``selection='planner'`` and
the RL trainer does not treat it as on-policy data (it can be used for success-filtered
distillation). Compute is bounded by ``num_candidates * (1 + revision_rounds)`` world-model calls.
"""

from __future__ import annotations

from typing import Optional

import torch
from pydantic import BaseModel, ConfigDict

from agentrs.actions.schema import Action, ActionType
from agentrs.agent.belief import BeliefState
from agentrs.agent.runtime import Decision
from agentrs.content.policy import CandidateContentGenerator, ContentGeneratingPolicy, generation_prompt
from agentrs.content.world_model import ContentConditionedWorldModel
from agentrs.data.schemas import WMPredictionRecord
from agentrs.security.permissions import PermissionPolicy, check_action


class ContentPlannerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_candidates: int = 4
    temperature: float = 1.0
    revision_rounds: int = 1
    target_score: float = 0.8
    uncertainty_penalty: float = 1.0
    min_score_to_act: float = 0.0  # decline below this predicted score
    max_uncertainty: float = 1.0  # decline above this epistemic std
    predict_responses: bool = True


class ContentOutcomePlanner:
    def __init__(self, cp: ContentGeneratingPolicy, wm: ContentConditionedWorldModel, cfg: ContentPlannerConfig,
                 permissions: Optional[PermissionPolicy] = None, seed: int = 0, name: str = "content-planner"):
        self.cp, self.wm, self.cfg = cp, wm, cfg
        self.perm = permissions
        self.gen = torch.Generator().manual_seed(seed)
        self.drafter = CandidateContentGenerator(cp, cfg.num_candidates, cfg.temperature)
        self.name = name
        self.stats = {"decisions": 0, "candidates_scored": 0, "declined": 0, "revisions": 0}
        self.last_candidates: list[dict] = []

    def begin_episode(self, belief):
        pass

    def _hist(self, belief: BeliefState):
        return [(h.action, h.outcome) for h in belief.history]

    @torch.no_grad()
    def decide(self, belief: BeliefState, candidates: list[Action]) -> Decision:
        cfg = self.cfg
        prompt = belief.prompt()
        lp = self.cp.op_log_probs([prompt], [candidates])[0]
        cand_strs = [c.to_str() for c in candidates]
        sub = [i for i, c in enumerate(candidates) if c.type == ActionType.SUBMIT]
        fin = next((i for i, c in enumerate(candidates) if c.type == ActionType.FINISH), None)
        self.stats["decisions"] += 1
        if not sub:
            i = fin if fin is not None else 0
            return Decision(action=candidates[i], candidates=cand_strs, chosen_index=i, policy_logprob=float(lp[i]),
                            behavior_logprob=0.0, selection="planner")
        i = max(sub, key=lambda k: float(lp[k]))  # most likely content operation
        op = candidates[i]
        if self.perm is not None:
            reason = check_action(self.perm, Action.submit(op.name or "", "x"))
            if reason is not None:
                return self._decline(candidates, cand_strs, lp, fin, f"not authorised: {reason}")
        gp = generation_prompt(prompt, op.name or "")
        scored: list[dict] = []
        for rnd in range(1 + max(0, cfg.revision_rounds)):
            drafts = self.drafter.drafts(gp, generator=self.gen)
            new = [d for d in drafts if d["content"] not in {s["content"] for s in scored}]
            if new:
                acts = [Action.submit(op.name or "", d["content"]).to_str() for d in new]
                preds = self.wm.predict_content(belief.task.goal, belief.observation, acts, self._hist(belief))
                self.stats["candidates_scored"] += len(new)
                for d, a, p in zip(new, acts, preds):
                    scored.append({**d, "action": a, "pred": p,
                                   "score": p.p_objective - cfg.uncertainty_penalty * p.epistemic})
            best = max(scored, key=lambda s: s["score"]) if scored else None
            if best is None or best["score"] >= cfg.target_score:
                break
            if rnd < cfg.revision_rounds:
                self.stats["revisions"] += 1
        self.last_candidates = [{"content": s["content"], "score": round(s["score"], 4),
                                 "p_objective": round(s["pred"].p_objective, 4),
                                 "epistemic": round(s["pred"].epistemic, 4)} for s in scored]
        best = max(scored, key=lambda s: s["score"])
        if best["score"] < cfg.min_score_to_act or best["pred"].epistemic > cfg.max_uncertainty:
            return self._decline(candidates, cand_strs, lp, fin,
                                 f"predicted score {best['score']:.2f} / uncertainty {best['pred'].epistemic:.3f} "
                                 f"outside thresholds")
        sa = self.cp.semantic_for(belief, candidates, i, float(lp[i]), gp, best, belief.task.env_id,
                                  belief.task.family, source="planner")
        rec = WMPredictionRecord(reward_prob=best["pred"].p_reward, terminal_prob=best["pred"].p_terminal,
                                 event_probs=best["pred"].content_probs, epistemic=best["pred"].epistemic)
        if cfg.predict_responses:
            pr = self.wm.predict_content(belief.task.goal, belief.observation, [best["action"]], self._hist(belief),
                                         sample_responses=True)[0]
            rec.diff_text = pr.diffs[0][0] if pr.diffs else None
        return Decision(action=sa.to_action(), candidates=cand_strs, chosen_index=i, policy_logprob=None,
                        behavior_logprob=None, selection="planner", wm_prediction=rec, semantic=sa,
                        info={"candidates": self.last_candidates})

    def _decline(self, candidates, cand_strs, lp, fin, why: str) -> Decision:
        self.stats["declined"] += 1
        a = candidates[fin] if fin is not None else Action.finish()
        return Decision(action=a, candidates=cand_strs, chosen_index=fin, policy_logprob=None, behavior_logprob=None,
                        selection="planner_declined", info={"declined": why})
