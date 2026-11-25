"""Helpers shared by the workstream-A tests."""
from __future__ import annotations

import torch


def make_context(cfg, B: int, seed: int = 0, device="cuda", requires_grad: bool = False):
    """Random R [B,S,D] plus goal/belief TokenSets of contract shape."""
    from superlearner.cam_unified.interfaces import TokenKind
    from superlearner.cam_unified.testing import rand_tokens
    g = torch.Generator().manual_seed(seed)
    R = torch.randn(B, cfg.n_state, cfg.d_model, generator=g).to(device)
    if requires_grad:
        R.requires_grad_(True)
    goal = rand_tokens(B, 3, cfg.d_model, TokenKind.GOAL, device=device, ragged=False, seed=seed + 1)
    belief = rand_tokens(B, cfg.n_particles + 1, cfg.d_model, TokenKind.BELIEF, device=device, ragged=False,
                         seed=seed + 2)
    return R, goal, belief
