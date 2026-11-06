"""Inverse dynamics model  P(action | observation_before, observation_after).

Used to label recordings that contain observations but no action log. The
model scores generic candidate actions (``actions.candidates``) given the
before-observation and the observed state difference. Labels are uncertain:
the model returns a confidence (max posterior) and abstains below a threshold.
An observed change does not uniquely identify its cause (e.g. ``wait`` vs a
no-op click both produce no change); abstention handles such ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import ConfigDict, BaseModel, Field

from agentrs.actions.candidates import candidates_from_observation
from agentrs.actions.schema import Action
from agentrs.agent.belief import action_text_for_lm
from agentrs.data.diff import compute_diff
from agentrs.environments.base import Observation
from agentrs.models.backbone import BackboneConfig, LMBackbone


class IDMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    abstain_threshold: float = 0.6
    max_candidates: int = 40
    vocab_restrict: bool = True


@dataclass
class IDMLabel:
    action: Optional[Action]
    confidence: float
    candidates: list[str]
    probs: list[float]

    @property
    def abstained(self) -> bool:
        return self.action is None


class InverseDynamicsModel(nn.Module):
    component = "idm"

    def __init__(self, cfg: IDMConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = LMBackbone(cfg.backbone)

    @staticmethod
    def prompt(goal: str, before: Observation, after: Observation) -> str:
        d = compute_diff(before, after)
        return f"GOAL: {goal}\nBEFORE:\n{before.to_text(30)}\nCHANGE:\n{d.to_text()}\nACTION:"

    def candidates(self, goal: str, before: Observation, after: Observation) -> list[str]:
        return [a.to_str() for a in candidates_from_observation(before, goal, after=after)][: self.cfg.max_candidates]

    def log_probs(self, prompts: Sequence[str], cands: Sequence[Sequence[str]]) -> list[torch.Tensor]:
        conts = [[action_text_for_lm(c) for c in cs] for cs in cands]
        scores = self.backbone.score(prompts, conts, prefix_cache=True, vocab_restrict=self.cfg.vocab_restrict)
        return [F.log_softmax(s, -1) for s in scores]

    @torch.no_grad()
    def label(self, goal: str, before: Observation, after: Observation) -> IDMLabel:
        cs = self.candidates(goal, before, after)
        lp = self.log_probs([self.prompt(goal, before, after)], [cs])[0]
        p = lp.exp()
        i = int(torch.argmax(p))
        conf = float(p[i])
        act = Action.parse(cs[i]) if conf >= self.cfg.abstain_threshold else None
        return IDMLabel(action=act, confidence=conf, candidates=cs, probs=p.tolist())
