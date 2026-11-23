"""Typed structures shared by CAM 6.1. Shapes: B batch, T time, N observation tokens, K candidates, L program tokens."""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import torch
from torch import Tensor


def _map(obj, fn):
    return type(obj)(**{f.name: (fn(v) if isinstance(v, Tensor) else v)
                        for f in fields(obj) for v in [getattr(obj, f.name)]})


@dataclass
class ObservationBatch:
    """Structured numeric observation tokens. Nothing here may come from hidden simulator state."""
    values: Tensor    # [..., N] transformed reading
    deltas: Tensor    # [..., N] change since the previous decision
    present: Tensor   # [..., N] bool, False = missing/masked
    ids: Tensor       # [..., N] long, variable identity within the episode
    text: Tensor | None = None       # [..., M, D] optional pretrained token embeddings
    text_mask: Tensor | None = None  # [..., M] bool
    extra: Tensor | None = None      # [..., N, E] public per-variable intervention evidence

    def map(self, fn) -> "ObservationBatch":
        return _map(self, fn)


@dataclass
class GoalBatch:
    features: Tensor  # [B, G] public, machine-readable goal description

    def map(self, fn) -> "GoalBatch":
        return _map(self, fn)


@dataclass
class ActionCandidateBatch:
    tokens: Tensor    # [B, K, L] long program tokens
    features: Tensor  # [B, K, F] execution-relevant structure compiled from the program
    mask: Tensor      # [B, K] bool, False = invalid or padding
    durations: Tensor # [B, K] ticks the program runs for


@dataclass
class BeliefState:
    """The agent's internal state after step ``t``."""
    h: Tensor                        # [B, D] deterministic recurrent state
    z: Tensor                        # [B, V, C] stochastic one-hot latent
    memory: Tensor                   # [B, S, Dm] slots after this step's write
    read: Tensor                     # [B, Dm] slot read used at this step
    u: Tensor                        # [B, d] shared belief feature
    prior_logits: Tensor             # [B, V, C]
    post_logits: Tensor | None = None  # [B, V, C], None in imagination

    def map(self, fn) -> "BeliefState":
        return _map(self, fn)

    def detach(self) -> "BeliefState":
        return self.map(lambda x: x.detach())


@dataclass
class WorldModelOutput:
    reward_logits: Tensor
    continue_logits: Tensor
    obs_pred: Tensor
    reward_ensemble: Tensor | None = None


@dataclass
class PolicyOutput:
    logits: Tensor    # [B, K] masked log-probabilities
    action: Tensor    # [B] chosen candidate
    log_prob: Tensor  # [B]


@dataclass
class ReplaySequence:
    """A batch of decision sequences. ``reward[t]`` is earned by the action chosen at ``t``."""
    values: Tensor
    deltas: Tensor
    present: Tensor
    ids: Tensor
    extra: Tensor
    goal: Tensor
    cand_tokens: Tensor
    cand_features: Tensor
    cand_mask: Tensor
    cand_dur: Tensor
    cand_bonus: Tensor   # [B, T, K] fixed planning term that was part of the behaviour policy
    rscale: Tensor       # [B, T] public reward variability of this episode so far, in scaled reward units
    action: Tensor
    behavior_log_prob: Tensor
    reward: Tensor
    dt: Tensor
    cont: Tensor      # 0 where action t reached a true terminal state
    first: Tensor     # 1 at the first decision of an episode
    mask: Tensor      # 1 for real steps, 0 for padding
    family: list = field(default_factory=list)

    def to(self, device) -> "ReplaySequence":
        return _map(self, lambda x: x.to(device, non_blocking=True))


@dataclass
class TrainMetrics:
    values: dict = field(default_factory=dict)


@dataclass
class CheckpointState:
    model: dict
    optimizers: dict
    normalizer: dict
    counters: dict
    config: dict


def masked_log_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    """Normalise over valid entries only. Invalid entries get a large negative log-probability."""
    if not mask.any(-1).all():
        raise ValueError("every row needs at least one valid candidate")
    masked = logits.masked_fill(~mask, float("-inf"))
    out = torch.log_softmax(masked, -1)
    return out.masked_fill(~mask, -1e9)
