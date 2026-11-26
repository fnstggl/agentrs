"""Decision trajectories for the learner: the scalar per-decision quantities of a batch of rows.

`Trajectory` is the learner-side view of a `SequenceBatch` (contract `interfaces.StepBatch` per decision) plus what the
learning objectives of workstream D need beyond it: validity mask for right-padded rows, observable experiment cost,
trial index (multi-trial meta-RL episodes), option bookkeeping, intrinsic learning-progress reward and the
behaviour-time value estimates used by GAE / value clipping. All tensors are batch-major ``[B, T]`` (``value`` is
``[B, T + 1]``, its last valid column the bootstrap value).

Padding. Entries at padded (``mask == False``) positions are replaced by 0 on construction (``value``: every column
that is neither a valid step nor the bootstrap of one), so NaN / inf written into padding by a producer (e.g. an
``EventRecord.behavior_logp`` of None stored as NaN) can never leak into valid steps through a masked product.
`sequence_mask` derives the validity mask of a contract `SequenceBatch` from its padding convention ("padded rows
masked via done/obs.mask").
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence

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
    # W7 (plan 2026-11-26 §2), additive: the observation-only actor reward r_psi of decision t (RewardModelP.reward,
    # detached) kept apart from `reward`; `reward_source` names what `reward` holds ("verified_increment" = the
    # pre-seam evaluator increment, "r_psi", "sparse_verified", "legacy_shaped"), `reward_version` the psi version
    # that produced it (EventRecord.extra["versions"]["reward_model"])
    reward_int: Optional[Tensor] = None  # [B,T]
    reward_source: str = "verified_increment"
    reward_version: Optional[int] = None

    _PER_STEP = ("reward", "dt", "behavior_logp", "cost", "intrinsic", "reward_int")

    def __post_init__(self):
        if self.mask is None:
            self.mask = torch.ones_like(self.reward, dtype=torch.bool)
        self.mask = self.mask.bool()
        B, T = self.reward.shape
        for name in ("dt", "done", "behavior_logp", "model_version", "mask"):
            v = getattr(self, name)
            if tuple(v.shape) != (B, T):
                raise ValueError(f"Trajectory.{name} must be [B, T] = {(B, T)}, got {tuple(v.shape)}")
        for name in ("cost", "trial", "intrinsic", "option", "option_start", "reward_int"):
            v = getattr(self, name)
            if v is not None and tuple(v.shape) != (B, T):
                raise ValueError(f"Trajectory.{name} must be [B, T] = {(B, T)}, got {tuple(v.shape)}")
        if self.value is not None and tuple(self.value.shape) != (B, T + 1):
            raise ValueError(f"Trajectory.value must be [B, T+1], got {tuple(self.value.shape)}")
        m = self.mask
        # padded entries -> 0 (torch.where, not a product: 0 * nan = nan)
        for name in self._PER_STEP:
            v = getattr(self, name)
            if v is not None:
                setattr(self, name, torch.where(m, v, torch.zeros_like(v)))
        self.done = self.done.bool() & m
        if self.value is not None:
            vm = self.value_mask()
            self.value = torch.where(vm, self.value, torch.zeros_like(self.value))
        if (self.dt < 0).any():
            raise ValueError("dt must be non-negative")

    def value_mask(self) -> Tensor:
        """[B, T+1] bool: columns of ``value`` that are used — every valid step and the bootstrap after it."""
        B, T = self.mask.shape
        f = torch.zeros(B, 1, dtype=torch.bool, device=self.mask.device)
        return torch.cat([self.mask, f], 1) | torch.cat([f, self.mask], 1)

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
                                 "trial", "intrinsic", "option", "option_start", "reward_int")},
                       extras={k: mv(v) for k, v in self.extras.items()})

    def select_rows(self, idx) -> "Trajectory":
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        il = [int(i) for i in idx]

        def sel(x):
            return x[idx] if torch.is_tensor(x) else x
        return replace(self, **{f: sel(getattr(self, f)) for f in
                                ("reward", "dt", "done", "behavior_logp", "model_version", "mask", "value", "cost",
                                 "trial", "intrinsic", "option", "option_start", "reward_int")},
                       event_ids=None if self.event_ids is None else [self.event_ids[i] for i in il],
                       family=None if self.family is None else [self.family[i] for i in il],
                       extras={k: (v[idx] if torch.is_tensor(v) else v) for k, v in self.extras.items()})

    @staticmethod
    def from_sequence(seq: SequenceBatch, value: Optional[Tensor] = None, mask: Optional[Tensor] = None,
                      cost: Optional[Tensor] = None) -> "Trajectory":
        """Stack a contract `SequenceBatch` (list of `StepBatch`) into a batch-major trajectory.
        ``mask`` [B,T] (valid decisions) defaults to `sequence_mask` (the contract's done/obs.mask padding rule)."""
        if not seq.steps:
            raise ValueError("empty SequenceBatch")
        if mask is None:
            mask = sequence_mask(seq.steps)

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


def sequence_mask(steps: Sequence[Any]) -> Tensor:
    """Validity mask [B, T] of aligned contract steps (`StepBatch`) under the padding convention of
    `interfaces.SequenceBatch` ("padded rows masked via done/obs.mask").

    A row is right-padded after it ended: every step after the row's last ``done`` is padding iff none of those
    steps carries an observation token (``obs.mask`` all False). If any later step has observations, a new episode
    started in that row and its steps are valid (a later step without observations, e.g. after a pure ``wait``, is
    a real decision). Steps whose ``obs`` is None are treated as carrying observations (cannot be padding).
    A truncated row padded *without* a done cannot be recognised by the convention: pass an explicit mask."""
    if not steps:
        raise ValueError("no steps")
    T = len(steps)
    ref = None
    for s in steps:
        for name in ("done", "reward", "dt"):
            v = getattr(s, name, None)
            if torch.is_tensor(v):
                ref = v
                break
        if ref is not None:
            break
    if ref is None:
        raise ValueError("cannot infer the batch size: steps need a `done`, `reward` or `dt` tensor")
    B, dev = ref.shape[0], ref.device
    done = torch.stack([torch.zeros(B, dtype=torch.bool, device=dev) if getattr(s, "done", None) is None
                        else torch.as_tensor(s.done).bool().to(dev) for s in steps], 1)          # [B,T]
    has_obs = []
    for s in steps:
        ob = getattr(s, "obs", None)
        if ob is None or getattr(ob, "mask", None) is None:
            has_obs.append(torch.ones(B, dtype=torch.bool, device=dev))
        else:
            mk = ob.mask.bool().to(dev)
            has_obs.append(mk.reshape(B, -1).any(1) if mk.numel() else torch.zeros(B, dtype=torch.bool, device=dev))
    has_obs = torch.stack(has_obs, 1)                                                          # [B,T]
    t = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
    last_done = torch.where(done, t, torch.full_like(t, -1)).max(1).values                    # [B]
    after = t > last_done.unsqueeze(1)
    revived = (after & has_obs).any(1)                                                         # a new episode began
    pad = after & (last_done >= 0).unsqueeze(1) & ~revived.unsqueeze(1)
    return ~pad


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """Mean of ``x`` over ``mask``; entries outside the mask never matter (even NaN / inf)."""
    m = mask.bool()
    return torch.where(m, x, torch.zeros_like(x)).sum() / m.to(x.dtype).sum().clamp_min(1.0)


def stack_steps(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Tensor]:
    """Stack a list (over time) of dicts of ``[B, ...]`` tensors into ``[B, T, ...]`` tensors for the given keys."""
    return {k: torch.stack([r[k] for r in rows], 1) for k in keys}
