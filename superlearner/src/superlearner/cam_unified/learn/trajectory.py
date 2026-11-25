"""Decision trajectories for the learner: the scalar per-decision quantities of a batch of rows.

`Trajectory` is the learner-side view of a `SequenceBatch` (contract `interfaces.StepBatch` per decision) plus what the
learning objectives of workstream D need beyond it: validity mask for right-padded rows, observable experiment cost,
trial index (multi-trial meta-RL episodes), option bookkeeping, intrinsic learning-progress reward and the
behaviour-time value estimates used by GAE / value clipping. All tensors are batch-major ``[B, T]`` (``value`` is
``[B, T + 1]``, its last valid column the bootstrap value).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

import torch
from torch import Tensor

from ..interfaces import SequenceBatch


@dataclass
class Trajectory:
    reward: Tensor                       # [B,T] verified extrinsic return increment of decision t
    dt: Tensor                           # [B,T] simulated days from decision t to decision t+1 (SMDP clock)
    done: Tensor                         # [B,T] bool: transition t ended the episode (true terminal; no bootstrap)
    behavior_logp: Tensor                # [B,T] log mu(a_t | s_t) recorded when acting (whole action, incl. option)
    model_version: Tensor                # [B,T] long: policy version that produced a_t
    mask: Optional[Tensor] = None        # [B,T] bool: valid (non-padding) decision
    value: Optional[Tensor] = None       # [B,T+1] behaviour-time value estimates; value[b, L_b] = bootstrap
    cost: Optional[Tensor] = None        # [B,T] observable resource cost of decision t (experiment cost C)
    trial: Optional[Tensor] = None       # [B,T] long: trial index inside a multi-trial meta-RL episode
    intrinsic: Optional[Tensor] = None   # [B,T] learning-progress reward (explore.py), separate from extrinsic
    option: Optional[Tensor] = None      # [B,T] long: active option id (-1 none)
    option_start: Optional[Tensor] = None   # [B,T] bool: a new option was selected at t
    event_ids: Optional[list] = None     # [B][T] event ids of the decisions (provenance / support-query separation)
    family: Optional[list] = None        # [B] task family (per-family return normalisation; never a model input)
    extras: dict = field(default_factory=dict)   # policy-specific per-step records, tensors with leading [B, T]

    def __post_init__(self):
        if self.mask is None:
            self.mask = torch.ones_like(self.reward, dtype=torch.bool)
        B, T = self.reward.shape
        for name in ("dt", "done", "behavior_logp", "model_version", "mask"):
            v = getattr(self, name)
            if tuple(v.shape) != (B, T):
                raise ValueError(f"Trajectory.{name} must be [B, T] = {(B, T)}, got {tuple(v.shape)}")
        if self.value is not None and tuple(self.value.shape) != (B, T + 1):
            raise ValueError(f"Trajectory.value must be [B, T+1], got {tuple(self.value.shape)}")
        if (self.dt[self.mask.bool()] < 0).any():
            raise ValueError("dt must be non-negative")

    @property
    def B(self) -> int:
        return self.reward.shape[0]

    @property
    def T(self) -> int:
        return self.reward.shape[1]

    @property
    def device(self):
        return self.reward.device

    def continues(self) -> Tensor:
        return (~self.done.bool()).to(self.reward.dtype)

    def lengths(self) -> Tensor:
        return self.mask.long().sum(1)

    def with_reward(self, reward: Tensor) -> "Trajectory":
        return replace(self, reward=reward)

    def to(self, device) -> "Trajectory":
        def mv(x):
            return x.to(device) if torch.is_tensor(x) else x
        return replace(self, **{f: mv(getattr(self, f)) for f in
                                ("reward", "dt", "done", "behavior_logp", "model_version", "mask", "value", "cost",
                                 "trial", "intrinsic", "option", "option_start")},
                       extras={k: mv(v) for k, v in self.extras.items()})

    def select_rows(self, idx) -> "Trajectory":
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        il = [int(i) for i in idx]

        def sel(x):
            return x[idx] if torch.is_tensor(x) else x
        return replace(self, **{f: sel(getattr(self, f)) for f in
                                ("reward", "dt", "done", "behavior_logp", "model_version", "mask", "value", "cost",
                                 "trial", "intrinsic", "option", "option_start")},
                       event_ids=None if self.event_ids is None else [self.event_ids[i] for i in il],
                       family=None if self.family is None else [self.family[i] for i in il],
                       extras={k: (v[idx] if torch.is_tensor(v) else v) for k, v in self.extras.items()})

    @staticmethod
    def from_sequence(seq: SequenceBatch, value: Optional[Tensor] = None, mask: Optional[Tensor] = None,
                      cost: Optional[Tensor] = None) -> "Trajectory":
        """Stack a contract `SequenceBatch` (list of `StepBatch`) into a batch-major trajectory."""
        if not seq.steps:
            raise ValueError("empty SequenceBatch")

        def st(name, dtype=None):
            xs = [torch.as_tensor(getattr(s, name)) for s in seq.steps]
            out = torch.stack(xs, 1)
            return out.to(dtype) if dtype is not None else out
        ev = None
        if all(s.event_ids is not None for s in seq.steps):
            B = len(seq.steps[0].event_ids)
            ev = [[s.event_ids[b] for s in seq.steps] for b in range(B)]
        return Trajectory(reward=st("reward", torch.float32), dt=st("dt", torch.float32), done=st("done", torch.bool),
                          behavior_logp=st("behavior_logp", torch.float32), model_version=st("model_version", torch.long),
                          mask=mask, value=value, cost=cost, event_ids=ev)


def compact_rows(active: Tensor, *series: Tensor) -> tuple[Tensor, list]:
    """Left-align the active steps of every row: ``active [B, T]`` bool; each series ``[B, T, ...]``.
    Returns ``(mask [B, L], compacted series [B, L, ...])`` with ``L = max_b sum_t active[b, t]``; padding is 0."""
    B, T = active.shape
    n = active.long().sum(1)
    L = int(n.max()) if B else 0
    # stable order: active steps first, in time order
    key = (~active).long() * (T + 1) + torch.arange(T, device=active.device).unsqueeze(0)
    order = key.argsort(1)[:, :L]
    mask = torch.arange(L, device=active.device).unsqueeze(0) < n.unsqueeze(1)
    out = []
    for s in series:
        idx = order.view(B, L, *([1] * (s.dim() - 2))).expand(B, L, *s.shape[2:])
        g = torch.gather(s, 1, idx)
        z = mask.view(B, L, *([1] * (s.dim() - 2)))
        out.append(torch.where(z, g, torch.zeros_like(g)))
    return mask, out


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp_min(1.0)


def stack_steps(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Tensor]:
    """Stack a list (over time) of dicts of ``[B, ...]`` tensors into ``[B, T, ...]`` tensors for the given keys."""
    return {k: torch.stack([r[k] for r in rows], 1) for k in keys}
