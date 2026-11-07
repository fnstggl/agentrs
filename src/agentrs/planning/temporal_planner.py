"""Consequence-based planning over time horizons.

For one decision the planner samples K complete actions from the policy (operation + content), asks
the temporal world model what each would cause **at the horizon the goal cares about** (e.g. a
useful answer within one day), and executes the draft with the best

    score = P(objective by h) - lambda * epistemic_std(objective at h) [- mu * proxy]

or declines when even the best draft is below ``min_score_to_act``. Objectives can be any modelled
event, or a count (e.g. ``views``, normalised by ``count_scale``) — which is how the short-term
proxy trap is studied: optimising views at 1 h is a different, measurable choice from optimising
answers at 1 day. The forecast for the chosen draft is attached to the decision (and, through the
runtime, stored in the ledger before execution).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from pydantic import BaseModel, ConfigDict

from agentrs.actions.schema import Action, ActionType
from agentrs.agent.belief import BeliefState
from agentrs.agent.runtime import Decision
from agentrs.content.policy import ContentGeneratingPolicy, generation_prompt
from agentrs.data.schemas import WMPredictionRecord
from agentrs.temporal.temporal_world_model import TemporalQuery, TemporalWorldModel


class TemporalPlannerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_candidates: int = 6
    objective: str = "answered"
    objective_kind: str = "event"  # event | count
    horizon_s: float = 86400.0
    count_scale: float = 50.0
    uncertainty_penalty: float = 0.5
    min_score_to_act: float = -1.0
    seed: int = 0


class TemporalContentPlanner:
    name = "temporal-planner"

    def __init__(self, cp: ContentGeneratingPolicy, wm: TemporalWorldModel, cfg: TemporalPlannerConfig,
                 context_fn: Callable[[str], str]):
        self.cp, self.wm, self.cfg = cp, wm, cfg
        self.context_fn = context_fn  # operation -> context text of our own action (same format as observed)
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.last: list[dict] = []
        self.stats = {"decisions": 0, "declined": 0, "candidates": 0}

    def begin_episode(self, belief):
        pass

    @torch.no_grad()
    def decide(self, belief: BeliefState, candidates: list[Action]) -> Decision:
        prompt = belief.prompt()
        subs = [c for c in candidates if c.type == ActionType.SUBMIT]
        lp = self.cp.op_log_probs([prompt], [candidates])[0]
        K = self.cfg.num_candidates
        ops = torch.multinomial(lp.exp(), K, replacement=True, generator=self.gen).tolist()
        drafts = []
        for i in ops:
            a = candidates[i]
            if a.type != ActionType.SUBMIT:
                continue
            gp = generation_prompt(prompt, a.name or "")
            s = self.cp.sample_content([gp], temperature=1.0, generator=self.gen)[0]
            drafts.append((i, gp, s))
        if not drafts and subs:
            i = candidates.index(subs[0])
            gp = generation_prompt(prompt, subs[0].name or "")
            drafts.append((i, gp, self.cp.sample_content([gp], temperature=1.0, generator=self.gen)[0]))
        uniq, seen = [], set()
        for d in drafts:
            key = (d[0], d[2]["content"])
            if key not in seen:
                seen.add(key)
                uniq.append(d)
        qs, acts = [], []
        for i, gp, s in uniq:
            act = Action.submit(candidates[i].name or "", s["content"])
            acts.append(act)
            qs.append(TemporalQuery(context=self.context_fn(candidates[i].name or ""), action=act.to_str(),
                                    source_type="self", actor_type="agent", goal=belief.task.goal))
        preds = self.wm.predict_batch(qs, [self.cfg.horizon_s]) if qs else []
        scored = []
        for (i, gp, s), act, p in zip(uniq, acts, preds):
            hp = p.horizons[0]
            if self.cfg.objective_kind == "count":
                val = hp.count_mean[self.cfg.objective] / self.cfg.count_scale
                unc = hp.count_std[self.cfg.objective] / self.cfg.count_scale
            else:
                val, unc = hp.event_prob[self.cfg.objective], hp.event_std[self.cfg.objective]
            scored.append((val - self.cfg.uncertainty_penalty * unc, val, unc, i, gp, s, act, p))
        self.stats["decisions"] += 1
        self.stats["candidates"] += len(scored)
        self.last = [{"content": x[6].text, "op": x[6].name, "score": x[0], "value": x[1], "unc": x[2]} for x in scored]
        cand_strs = [c.to_str() for c in candidates]
        can_decline = any(c.type == ActionType.FINISH for c in candidates)
        if can_decline and (not scored or max(scored, key=lambda x: x[0])[0] < self.cfg.min_score_to_act):
            self.stats["declined"] += 1
            fin = next(c for c in candidates if c.type == ActionType.FINISH)
            return Decision(action=fin, candidates=cand_strs, chosen_index=candidates.index(fin),
                            selection="planner_declined", info={"candidates": self.last})
        best = max(scored, key=lambda x: x[0])
        score, val, unc, i, gp, s, act, p = best
        sa = self.cp.semantic_for(belief, candidates, i, float(lp[i]), gp, s, belief.task.env_id, belief.task.family,
                                  source="planner")
        rec = WMPredictionRecord(reward_prob=val, epistemic=unc, checkpoint=p.checkpoint)
        return Decision(action=sa.to_action(), candidates=cand_strs, chosen_index=i, selection="planner",
                        wm_prediction=rec, semantic=sa, info={"candidates": self.last, "forecast": p.to_dict()})
