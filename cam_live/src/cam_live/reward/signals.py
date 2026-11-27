"""cam_live.reward.signals — the named signal vector the reward generator consumes (spec 7.2, Part 8).

A fixed, versioned schema: names, order and normalisation hints. Every signal is something CAM computes from its
own model and observations at the time it rewards an action. The verified return that supervises the *outer*
objective is not among them: ``verified_progress_delta`` is only a verifier measurement that had already been
observed when the reward was computed (masked otherwise), never a query-window outcome.

No relative value of these quantities is prescribed here; the transforms only make magnitudes comparable
(log-compression of unbounded quantities). What deserves reinforcement is left to R_psi.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..contracts import ActionResult, Cost, FailureClass, ResultStatus

SIGNAL_SCHEMA_VERSION = "1.0"

# transform hints: "identity" (already bounded), "log1p" (non-negative, unbounded), "symlog" (signed, unbounded)
IDENTITY, LOG1P, SYMLOG, ONEHOT = "identity", "log1p", "symlog", "onehot"


@dataclass(frozen=True)
class SignalSpec:
    name: str
    transform: str
    description: str


SCALAR_SPECS: Tuple[SignalSpec, ...] = (
    SignalSpec("success_before", IDENTITY, "evaluator's success probability for the objective before the action"),
    SignalSpec("success_after", IDENTITY, "evaluator's success probability after the result was observed"),
    SignalSpec("delta_v", IDENTITY, "success_after - success_before"),
    SignalSpec("action_quality", SYMLOG, "policy/evaluator estimate of the chosen action's value relative to the candidates"),
    SignalSpec("info_gain", LOG1P, "belief change the action's own outcomes are expected to cause (KL of the fast "
                                   "posterior, nats; estimated before acting)"),
    SignalSpec("voi", SYMLOG, "decision-relevant value-of-information estimate made before acting"),
    SignalSpec("cost_money", LOG1P, "money spent by the action"),
    SignalSpec("cost_seconds", LOG1P, "wall-clock seconds the action took"),
    SignalSpec("cost_compute_s", LOG1P, "inference compute seconds spent on the action"),
    SignalSpec("elapsed_s", LOG1P, "seconds since the previous decision"),
    SignalSpec("prediction_error_std", LOG1P, "absolute prediction error in predictive standard deviations"),
    SignalSpec("epistemic_before", LOG1P, "epistemic (reducible) predictive std before the observation"),
    SignalSpec("epistemic_after", LOG1P, "epistemic predictive std after the observation"),
    SignalSpec("verified_progress_delta", SYMLOG, "change of a verifier measurement already observed at reward time"),
    SignalSpec("context_novelty", IDENTITY, "novelty of the decision context in [0, 1]"),
)
SCALAR_NAMES: Tuple[str, ...] = tuple(s.name for s in SCALAR_SPECS)
STATUS_NAMES: Tuple[str, ...] = tuple(f"status.{s}" for s in (ResultStatus.OK, ResultStatus.FAILED,
                                                              ResultStatus.DENIED, ResultStatus.NEEDS_APPROVAL,
                                                              ResultStatus.TIMEOUT, ResultStatus.CANCELLED))
FAILURE_NAMES: Tuple[str, ...] = tuple(f"failure.{f}" for f in FailureClass.ALL)
SIGNAL_NAMES: Tuple[str, ...] = SCALAR_NAMES + STATUS_NAMES + FAILURE_NAMES
N_SIGNALS = len(SIGNAL_NAMES)
SIGNAL_TRANSFORMS: Tuple[str, ...] = tuple(s.transform for s in SCALAR_SPECS) + (ONEHOT,) * (len(STATUS_NAMES)
                                                                                       + len(FAILURE_NAMES))


def schema() -> Dict[str, Any]:
    return {"version": SIGNAL_SCHEMA_VERSION, "names": list(SIGNAL_NAMES), "transforms": list(SIGNAL_TRANSFORMS)}


def schema_hash() -> str:
    return hashlib.sha256(json.dumps(schema(), sort_keys=True).encode()).hexdigest()


def _transform(x: Tensor, kind: str) -> Tensor:
    if kind == LOG1P:
        return torch.log1p(x.clamp_min(0.0))
    if kind == SYMLOG:
        return torch.sign(x) * torch.log1p(x.abs())
    return x


@dataclass
class RewardSignals:
    """One batch of signals. Scalar fields are [B] tensors or None (= not available: value 0, mask 0).
    ``status`` / ``failure_class`` are per-row ``contracts.ResultStatus`` / ``FailureClass`` strings."""
    success_before: Optional[Tensor] = None
    success_after: Optional[Tensor] = None
    delta_v: Optional[Tensor] = None
    action_quality: Optional[Tensor] = None
    info_gain: Optional[Tensor] = None
    voi: Optional[Tensor] = None
    cost_money: Optional[Tensor] = None
    cost_seconds: Optional[Tensor] = None
    cost_compute_s: Optional[Tensor] = None
    elapsed_s: Optional[Tensor] = None
    prediction_error_std: Optional[Tensor] = None
    epistemic_before: Optional[Tensor] = None
    epistemic_after: Optional[Tensor] = None
    verified_progress_delta: Optional[Tensor] = None
    context_novelty: Optional[Tensor] = None
    status: Optional[Sequence[str]] = None
    failure_class: Optional[Sequence[str]] = None

    schema_version = SIGNAL_SCHEMA_VERSION

    # ------------------------------------------------------------------------------------------ construction
    @classmethod
    def names(cls) -> Tuple[str, ...]:
        return SIGNAL_NAMES

    @classmethod
    def from_step(cls, *, cost: Optional[Cost] = None, result: Optional[ActionResult] = None,
                  **scalars: Optional[float]) -> "RewardSignals":
        """One decision (batch of 1) from plain floats and contract records. ``cost`` / ``result`` fill the cost
        terms, execution status and failure class; unknown scalars stay missing. ``delta_v`` is derived when both
        success estimates are given."""
        unknown = set(scalars) - set(SCALAR_NAMES)
        if unknown:
            raise KeyError(f"unknown signals: {sorted(unknown)}")
        vals = dict(scalars)
        c = cost if cost is not None else (result.cost if result is not None else None)
        if c is not None:
            vals.setdefault("cost_money", c.money)
            vals.setdefault("cost_seconds", c.seconds)
            vals.setdefault("cost_compute_s", c.compute_s)
        if vals.get("delta_v") is None and vals.get("success_before") is not None \
                and vals.get("success_after") is not None:
            vals["delta_v"] = vals["success_after"] - vals["success_before"]
        kw: Dict[str, Any] = {k: torch.tensor([float(v)], dtype=torch.float64) for k, v in vals.items()
                              if v is not None}
        if result is not None:
            kw["status"], kw["failure_class"] = [result.status], [result.failure_class]
        return cls(**kw)

    @classmethod
    def stack(cls, items: Sequence["RewardSignals"]) -> "RewardSignals":
        """Concatenate batches; a scalar missing in some items is NaN there (and masked by ``to_tensor``)."""
        sizes = [it.batch_size for it in items]
        kw: Dict[str, Any] = {}
        for name in SCALAR_NAMES:
            cols = [getattr(it, name) for it in items]
            if all(c is None for c in cols):
                continue
            kw[name] = torch.cat([torch.full((n,), float("nan"), dtype=torch.float64) if c is None
                                  else c.to(torch.float64).reshape(-1) for c, n in zip(cols, sizes)])
        for name in ("status", "failure_class"):
            cols = [getattr(it, name) for it in items]
            if any(c is not None for c in cols):
                kw[name] = [x for c, n in zip(cols, sizes) for x in (list(c) if c is not None else [""] * n)]
        return cls(**kw)

    @property
    def batch_size(self) -> int:
        for name in SCALAR_NAMES:
            v = getattr(self, name)
            if v is not None:
                return int(v.reshape(-1).shape[0])
        for name in ("status", "failure_class"):
            v = getattr(self, name)
            if v is not None:
                return len(v)
        raise ValueError("empty RewardSignals")

    # ------------------------------------------------------------------------------------------ tensors
    def _columns(self, dtype, device) -> Tuple[Tensor, Tensor]:
        B = self.batch_size
        vals = torch.zeros(B, N_SIGNALS, dtype=dtype, device=device)
        mask = torch.zeros(B, N_SIGNALS, dtype=dtype, device=device)
        for j, spec in enumerate(SCALAR_SPECS):
            v = getattr(self, spec.name)
            if v is None:
                continue
            v = v.to(dtype=dtype, device=device).reshape(-1)
            ok = torch.isfinite(v)
            vals[:, j] = torch.where(ok, _transform(torch.where(ok, v, torch.zeros_like(v)), spec.transform),
                                     torch.zeros_like(v))
            mask[:, j] = ok.to(dtype)
        for offset, names, col, prefix in ((len(SCALAR_NAMES), STATUS_NAMES, self.status, "status."),
                                           (len(SCALAR_NAMES) + len(STATUS_NAMES), FAILURE_NAMES,
                                            self.failure_class, "failure.")):
            if col is None:
                continue
            index = {n: i for i, n in enumerate(names)}
            for b, s in enumerate(col):
                i = index.get(prefix + s)
                if i is not None:                         # unknown / empty label: row stays masked
                    vals[b, offset + i] = 1.0
                    mask[b, offset:offset + len(names)] = 1.0
        return vals, mask

    def to_tensor(self, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> Tensor:
        """[B, N_SIGNALS] in schema order with the fixed transforms applied; missing entries are 0."""
        return self._columns(dtype, device)[0]

    def mask(self, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> Tensor:
        """[B, N_SIGNALS]: 1 where the signal was available and finite."""
        return self._columns(dtype, device)[1]

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"schema_version": SIGNAL_SCHEMA_VERSION}
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None:
                out[f.name] = v.tolist() if isinstance(v, Tensor) else list(v)
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RewardSignals":
        if d.get("schema_version", SIGNAL_SCHEMA_VERSION) != SIGNAL_SCHEMA_VERSION:
            raise ValueError(f"signal schema {d.get('schema_version')} != {SIGNAL_SCHEMA_VERSION}")
        kw: Dict[str, Any] = {}
        for k, v in d.items():
            if k in SCALAR_NAMES:
                kw[k] = torch.tensor(v, dtype=torch.float64)
            elif k in ("status", "failure_class"):
                kw[k] = list(v)
        return cls(**kw)
