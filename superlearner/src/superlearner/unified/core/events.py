"""Event-token memory: causal attention over the episode's recent decisions.

An event is what the agent knew at one decision: the encoded observation (including its within-episode regression
evidence on the lever tokens), the program it had just executed, and how long that program ran. The belief at a
decision is the output of a causal transformer over the last ``window`` events, so evidence from any earlier step in
the window is one attention hop away instead of being carried through many recurrent updates.

This replaces the RSSM as CAM 7.0's memory because the retention probe showed the RSSM transition losing a cue within
four steps even under direct supervision, while causal attention kept it at every delay tested (ledger, 2026-11-24).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import UnifiedConfig
from ..encoders.observation import _encoder, mlp


class EventCore(nn.Module):
    def __init__(self, cfg: UnifiedConfig):
        super().__init__()
        d = cfg.d_model
        self.window = cfg.event_window
        self.event_in = mlp(d + cfg.action_dim + 2, d, d, layers=1)
        self.body = _encoder(d, cfg.event_heads, cfg.event_layers)
        # The belief reads the attention summary of the window and, directly, the current event. Without the direct
        # path the current step's evidence reached the belief only through the residual stream mixed with averaged
        # history, and learning to read it failed in 1-2 of 3 seeds (ledger, reward isolation, 2026-11-24).
        self.project = mlp(2 * d + cfg.goal_dim, d, d, layers=2)

    def event(self, pooled: Tensor, prev_action: Tensor, prev_dt: Tensor) -> Tensor:
        """``[..., d]`` event tokens from the observation summary, the previous program and its duration."""
        time_in = torch.stack([torch.log1p(prev_dt), (prev_dt > 0).float()], -1)
        return self.event_in(torch.cat([pooled, prev_action, time_in], -1))

    def forward(self, events: Tensor, valid: Tensor, goal: Tensor) -> Tensor:
        """``events [B, T, d]`` in time order, ``valid [B, T]``. Returns the belief ``u [B, T, d]`` where position t
        attends to positions t - window + 1 ... t. There are no absolute position embeddings: order comes from the
        clock and durations inside each event, so a sliding window while acting computes exactly what training does."""
        B, T, _ = events.shape
        x = events
        i = torch.arange(T, device=events.device)
        age = i.unsqueeze(1) - i.unsqueeze(0)                        # query minus key
        causal = (age < 0) | (age >= self.window)                    # True = not allowed
        pad = ~valid.bool()
        empty = pad.all(1)
        if empty.any():
            pad = pad.clone()
            pad[empty, 0] = False
        z = self.body(x, mask=causal, src_key_padding_mask=pad)
        return self.project(torch.cat([z, events, goal.unsqueeze(1).expand(-1, T, -1)], -1))
