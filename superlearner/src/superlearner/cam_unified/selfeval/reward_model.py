"""Observation-only actor reward model R_psi (redesign plan 2026-11-26 §2; owner W8). SEAM STUB.

The full model (W8) computes

    r_t = [Gamma_t * Phi(R_{t+1}, G) - Phi(R_t, G)] + lambda_ig * IG_t - lambda_c * c_t + rho_psi(b_t, e_I, b_{t+1}, e_G, s)

with Phi = ProspectEvaluator P(success by horizon | R, G) trained on verified LabelStore labels only, Gamma_t =
exp(-kappa dt), IG_t the actor-visible information gain, c_t the observed resource spend and rho_psi the meta-learned
intrinsic term. This stub implements `interfaces.RewardModelP` with the reward identically zero, Phi = 0 and no
losses, so the canonical loop, the learner and the reward-integrity tests can be wired against the final protocol
before W8 merges. It has no parameters; `version` stays 0.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import CamConfig


def _rows(R, dt=None) -> tuple:
    """(batch size, device) from R [B,...] (tensor) or dt [B]."""
    for t in (R, dt):
        if torch.is_tensor(t) and t.dim() >= 1:
            return int(t.shape[0]), t.device
    if R is not None and hasattr(R, "__len__"):
        return len(R), torch.device("cpu")
    raise ValueError("RewardModel: cannot infer the batch size (pass R [B,...] or dt [B] as tensors)")


class RewardModel(nn.Module):
    """Seam stub of RewardModelP: reward() and potential() return zeros [B] (float32, detached, on R's device);
    losses() returns {} (nothing to train)."""

    stub = True

    def __init__(self, cfg: Optional[CamConfig] = None):
        super().__init__()
        self.cfg = cfg
        self.version = 0

    def reward(self, R, R_next=None, goal=None, dt=None, *, info_gain=None, cost=None, belief=None,
               belief_next=None, intent=None, signals=None) -> torch.Tensor:
        B, dev = _rows(R, dt)
        return torch.zeros(B, device=dev)

    def potential(self, R, G=None) -> torch.Tensor:
        B, dev = _rows(R)
        return torch.zeros(B, device=dev)

    def losses(self, label_batch) -> dict:
        return {}
