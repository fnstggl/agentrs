"""Baselines for temporal prediction (Experiment A).

* ``DiscreteHorizonModel`` — same LM trunk, but one independent Bernoulli head per (event, trained
  horizon) and no representation of time: it can only answer at the horizons it was trained on;
  other durations are linearly interpolated in log time between neighbouring trained horizons.
* ``cap_example`` — what a single-step (next-observation) model learns from: labels truncated at a
  short horizon (everything later is unknown to it). Queried at later horizons it can only assume
  persistence (``persistence_predict_fn``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from agentrs.data.temporal_transition import EventLabel, TemporalExample
from agentrs.models.backbone import BackboneConfig, LMBackbone
from agentrs.models.world_model import EnsembleHeads, bootstrap_mask
from agentrs.temporal.temporal_evaluation import event_truth
from agentrs.temporal.temporal_world_model import TemporalQuery, TemporalWorldModel


class DiscreteHorizonModel(nn.Module):
    component = "discrete_horizon_model"

    def __init__(self, backbone: BackboneConfig, events: Sequence[str], horizons: Sequence[float], n_heads: int = 5):
        super().__init__()
        self.events, self.horizons = list(events), sorted(horizons)
        self.backbone = LMBackbone(backbone)
        self.heads = EnsembleHeads(self.backbone.hidden_size, 128, n_heads, len(self.events) * len(self.horizons))
        self.n_heads = n_heads
        self._fmt = TemporalWorldModel.query_prompt

    def prompt(self, q: TemporalQuery) -> str:
        lines = [f"SOURCE: {q.source_type} | ACTOR: {q.actor_type}", f"GOAL: {q.goal if q.goal else 'unknown'}",
                 "CONTEXT:", q.context.strip(), f"ACTION: {q.action}", "FORECAST:"]
        return "\n".join(lines)

    def logits(self, exs: Sequence[TemporalExample]) -> torch.Tensor:
        h = self.backbone.pooled([self.prompt(TemporalQuery.from_example(e)) for e in exs])
        return self.heads(h).reshape(self.n_heads, len(exs), len(self.events), len(self.horizons))

    def loss(self, exs: Sequence[TemporalExample]):
        lg = self.logits(exs)
        y = torch.zeros(lg.shape[1:])
        m = torch.zeros(lg.shape[1:])
        for i, e in enumerate(exs):
            for j, n in enumerate(self.events):
                for k, h in enumerate(self.horizons):
                    t = event_truth(e, n, h)
                    if t is not None:
                        y[i, j, k], m[i, j, k] = t, 1.0
        bm = bootstrap_mask([e.example_id for e in exs], self.n_heads, 0.8, lg.device)[:, :, None, None]
        bce = F.binary_cross_entropy_with_logits(lg, y.expand_as(lg), reduction="none") * m * bm
        return bce.sum() / (m * bm).sum().clamp(min=1), {}

    @torch.no_grad()
    def predict_fn(self):
        lh = np.log1p(np.asarray(self.horizons))

        def f(examples, horizons):
            out = {n: np.zeros((len(examples), len(horizons))) for n in self.events}
            for s in range(0, len(examples), 64):
                p = torch.sigmoid(self.logits(examples[s:s + 64])).mean(0).numpy()
                for j, n in enumerate(self.events):
                    for i in range(p.shape[0]):
                        out[n][s + i] = np.interp(np.log1p(horizons), lh, p[i, j])
            return {"event": out, "count": {}}

        return f


def cap_example(ex: TemporalExample, max_s: float) -> TemporalExample:
    """Labels as seen by a learner that only observes the first ``max_s`` seconds after an action."""
    e = ex.model_copy(deep=True)
    ev = {}
    for n, lab in e.events.items():
        if lab.kind == "none":
            ev[n] = lab
        elif lab.kind in ("exact", "interval") and lab.b_s <= max_s:
            ev[n] = lab
        elif lab.kind == "interval" and lab.a_s < max_s:
            ev[n] = EventLabel(kind="right_censored", a_s=lab.a_s, b_s=lab.a_s, status="pending")
        else:
            b = min(lab.b_s if lab.kind == "right_censored" else max_s, max_s)
            ev[n] = EventLabel(kind="right_censored", a_s=b, b_s=b, status="pending")
    e.events = ev
    e.counts = {k: [i for i in v if i.b_s <= max_s] for k, v in e.counts.items()}
    e.states = [s for s in e.states if s.h_s <= max_s]
    e.continuous = {k: [o for o in v if o.h_s <= max_s] for k, v in e.continuous.items()}
    return e


def persistence_predict_fn(base_fn, max_s: float):
    """Answer every horizon beyond ``max_s`` with the prediction at ``max_s`` (no notion of later time)."""

    def f(examples, horizons):
        hs = [min(h, max_s) for h in horizons]
        return base_fn(examples, hs)

    return f


def log_interp(x: float, xs, ys) -> float:
    return float(np.interp(math.log1p(x), np.log1p(xs), ys))
