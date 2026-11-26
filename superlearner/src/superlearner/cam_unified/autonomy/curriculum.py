"""Learned autonomous curriculum C_chi and learning-gain estimation (meta-reward spec 8.3, 9.3, 9.4; manifest AU-03,
KV-02).

    g_self ~ C_chi(b_t, lifetime_state, uncertainty, available_resources)

`CurriculumPolicy` scores candidate self-goals (from `GoalCompiler.candidates`, i.e. instantiated over what the agent
currently observes) with a learned network over CAM's pooled latent state, a lifetime-state vector and each
candidate's descriptor (+ optional goal-token embedding from the shared goal encoder). It returns a categorical
distribution with exact log-probabilities, so it is trained by policy gradient.

Its training signal is the *measured learning gain* of the experience that pursuing the goal produced (KV-02):

    learning_gain(D_g) = J_query(U(theta, D_g)) - J_query(theta_control)

measured on independent training-query tasks that were NOT disclosed during the goal-free period, with a matched
control learner (same compute, no D_g, or D_random). Completing a self-goal earns nothing by itself (trivial goals
would otherwise win) and nothing rewards difficulty per se (impossible goals would). Because measuring the gain is
expensive, `LearningGainPredictor` regresses measured gains from (latent, goal descriptor, outcome of the attempt)
and the curriculum is trained on predicted gains between measurements, the predictor being corrected whenever a
real measurement arrives (and it reports its calibration on held-out measurements).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

N_GOAL_FEATS = 5
N_LIFE_FEATS = 6


@dataclass
class CurriculumChoice:
    index: Tensor        # [B] chosen candidate
    logp: Tensor         # [B] log-probability of the choice (behaviour probability)
    probs: Tensor        # [B, C]
    entropy: Tensor      # [B]


class CurriculumPolicy(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = 256, n_goal_feats: int = N_GOAL_FEATS,
                 n_life_feats: int = N_LIFE_FEATS, d_goal_emb: int = 0):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(d_model + n_life_feats, d_hidden), nn.SiLU(), nn.Linear(d_hidden, d_hidden))
        self.goal = nn.Sequential(nn.Linear(n_goal_feats + d_goal_emb, d_hidden), nn.SiLU(),
                                  nn.Linear(d_hidden, d_hidden))
        self.score = nn.Sequential(nn.Linear(2 * d_hidden, d_hidden), nn.SiLU(), nn.Linear(d_hidden, 1))
        self.d_goal_emb = d_goal_emb

    def logits(self, b: Tensor, life: Tensor, goal_feats: Tensor, mask: Tensor,
               goal_emb: Optional[Tensor] = None) -> Tensor:
        """b [B, D], life [B, Fl], goal_feats [B, C, Fg], mask [B, C] -> [B, C]."""
        s = self.state(torch.cat([b.float(), life.float()], -1))
        gx = goal_feats.float() if goal_emb is None else torch.cat([goal_feats.float(), goal_emb.float()], -1)
        g = self.goal(gx)
        x = torch.cat([s[:, None, :].expand_as(g), g], -1)
        return self.score(x).squeeze(-1).masked_fill(~mask.bool(), float("-inf"))

    def choose(self, b, life, goal_feats, mask, goal_emb=None, greedy: bool = False,
               generator: Optional[torch.Generator] = None) -> CurriculumChoice:
        lg = self.logits(b, life, goal_feats, mask, goal_emb)
        p = torch.softmax(lg, -1)
        idx = p.argmax(-1) if greedy else torch.multinomial(p.detach().cpu(), 1, generator=generator).squeeze(-1).to(p.device)
        logp = torch.log_softmax(lg, -1).gather(-1, idx[:, None]).squeeze(-1)
        ent = -(p * torch.log_softmax(lg, -1).masked_fill(~mask.bool(), 0)).sum(-1)
        return CurriculumChoice(idx, logp, p, ent)

    def loss(self, b, life, goal_feats, mask, chosen: Tensor, gain: Tensor, baseline: Optional[Tensor] = None,
             entropy_coef: float = 1e-2, goal_emb=None) -> Tensor:
        """REINFORCE on (predicted or measured) learning gain of the chosen goal."""
        lg = self.logits(b, life, goal_feats, mask, goal_emb)
        logp = torch.log_softmax(lg, -1)
        lp = logp.gather(-1, chosen.long()[:, None]).squeeze(-1)
        adv = gain.float() - (baseline.float() if baseline is not None else gain.float().mean())
        p = logp.exp()
        ent = -(p * logp.masked_fill(~mask.bool(), 0)).sum(-1)
        return -(lp * adv.detach()).mean() - entropy_coef * ent.mean()


class LearningGainPredictor(nn.Module):
    """Predicts the measured learning gain of the experience produced by pursuing a goal."""

    def __init__(self, d_model: int, d_hidden: int = 256, n_goal_feats: int = N_GOAL_FEATS, n_outcome: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model + n_goal_feats + n_outcome, d_hidden), nn.SiLU(),
                                 nn.Linear(d_hidden, d_hidden), nn.SiLU(), nn.Linear(d_hidden, 2))

    def forward(self, b: Tensor, goal_feats: Tensor, outcome: Tensor) -> tuple:
        o = self.net(torch.cat([b.float(), goal_feats.float(), outcome.float()], -1))
        return o[..., 0], F.softplus(o[..., 1]) + 1e-3          # mean, std

    def loss(self, b, goal_feats, outcome, measured: Tensor) -> Tensor:
        mu, sd = self(b, goal_feats, outcome)
        return (0.5 * ((measured.float() - mu) / sd) ** 2 + torch.log(sd)).mean()


def measured_learning_gain(j_after: Tensor, j_control: Tensor) -> Tensor:
    """learning_gain(D) = J_query(U(theta, D)) - J_query(theta_control): both evaluated on the same independent query
    tasks with matched compute; per-task differences [n] (the mean is the estimate, its SE the uncertainty)."""
    return j_after.float() - j_control.float()
