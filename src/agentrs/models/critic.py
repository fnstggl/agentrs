"""Value function V_psi(belief, goal) = expected discounted verified task reward.

The critic *predicts* outcomes; it is never used as the verifier. Rewards in
MiniWeb are binary verified success at termination, so V is in [0, 1] and is
parameterised with a sigmoid and trained with (soft-label) binary cross-entropy
against Monte-Carlo or TD(lambda) targets computed from real trajectories.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import ConfigDict, BaseModel, Field

from agentrs.models.backbone import BackboneConfig, LMBackbone


class CriticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    head_hidden: int = 128
    gamma: float = 0.95
    td_lambda: float = 1.0  # 1.0 = Monte-Carlo targets


def value_prompt(belief_prompt: str) -> str:
    """Critic input: the policy's belief prompt with the action cue replaced by a value cue."""
    if belief_prompt.endswith("ACTION:"):
        belief_prompt = belief_prompt[: -len("ACTION:")]
    return belief_prompt + "VALUE:"


class Critic(nn.Module):
    component = "critic"

    def __init__(self, cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = LMBackbone(cfg.backbone)
        d = self.backbone.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.head_hidden), nn.GELU(),
                                  nn.Linear(cfg.head_hidden, 1))

    def logits(self, belief_prompts: Sequence[str]) -> torch.Tensor:
        h = self.backbone.pooled([value_prompt(p) for p in belief_prompts])
        return self.head(h).squeeze(-1)

    def value(self, belief_prompts: Sequence[str]) -> torch.Tensor:
        return torch.sigmoid(self.logits(belief_prompts))

    def loss(self, belief_prompts: Sequence[str], targets: torch.Tensor) -> torch.Tensor:
        z = self.logits(belief_prompts)
        return F.binary_cross_entropy_with_logits(z, targets.to(z.device).clamp(0, 1))
