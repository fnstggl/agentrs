"""Training the temporal world model from matured, real outcomes.

* ``TemporalTrainingDataBuilder`` turns ledger rows into censoring-aware ``TemporalExample``s
  (initial forecasts and, optionally, updated forecasts at later information cutoffs), for chosen
  splits / source types / strata.
* ``TemporalWorldModelTrainer`` performs real gradient updates (``GradientTrainer``: AdamW, clipping,
  divergence guards) on minibatches, evaluates on held-out examples and checkpoints weights +
  optimizer state.
* ``progressive_update`` is the learner's incremental step: it reads actions whose labels changed
  since this learner last consumed them (a 1-minute outcome can train long before the 7-day outcome
  matures), mixes them with historical replay, trains, and records the ingestion (training clock).
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field

from agentrs.data.temporal_transition import TemporalExample
from agentrs.temporal.delayed_outcome_ledger import DelayedOutcomeLedger
from agentrs.temporal.outcome_labels import OutcomeMapping, build_examples
from agentrs.temporal.temporal_evaluation import evaluate_temporal, model_predict_fn
from agentrs.temporal.temporal_world_model import TemporalSpec, TemporalWorldModel
from agentrs.training.common import (
    GradientTrainer,
    OptimConfig,
    load_trainer_state,
    param_change,
    param_snapshot,
    save_checkpoint,
)


class TemporalTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int = 200
    batch_size: int = 16
    optim: OptimConfig = Field(default_factory=lambda: OptimConfig(lr=2e-4, total_steps=1000))
    eval_every: int = 0
    replay_ratio: float = 0.5  # progressive updates: fraction of each batch drawn from history
    progressive_min_new: int = 16
    progressive_steps: int = 20
    seed: int = 0


class TemporalTrainingDataBuilder:
    def __init__(self, ledger: DelayedOutcomeLedger, spec: TemporalSpec, mapping: OutcomeMapping | None = None):
        self.ledger = ledger
        self.spec = spec
        self.mapping = mapping or OutcomeMapping()

    def build(self, as_of: dict[str, float] | None = None, splits: Sequence[str] | None = None,
              source_types: Sequence[str] | None = None, groups: Sequence[str] | None = None,
              exclude_groups: Sequence[str] = (), cutoffs="observations", max_cutoffs: int = 2,
              action_ids: Sequence[str] | None = None) -> list[TemporalExample]:
        out = []
        rows = self.ledger.actions() if action_ids is None else [self.ledger.action(a) for a in action_ids]
        for a in rows:
            if a is None or not a["training_eligible"]:
                continue
            if splits and a["split"] not in splits:
                continue
            if source_types and a["source_type"] not in source_types:
                continue
            g = a["grp"] or ""
            if groups and not any(g.startswith(x) for x in groups):
                continue
            if any(g.startswith(x) for x in exclude_groups):
                continue
            t = (as_of or {}).get(a["time_base"])
            out += build_examples(self.ledger, a["action_id"], self.spec, self.mapping, as_of=t, cutoffs=cutoffs,
                                  max_cutoffs=max_cutoffs)
        return [e for e in out if e.n_labels() > 0]


class TemporalWorldModelTrainer:
    def __init__(self, model: TemporalWorldModel, cfg: TemporalTrainConfig, name: str = "temporal-wm"):
        self.model = model
        self.cfg = cfg
        self.name = name
        self.opt = GradientTrainer(model, cfg.optim)
        self.rng = random.Random(cfg.seed)
        self.replay: list[TemporalExample] = []
        self.history: list[dict] = []
        self.examples_seen = 0
        self.version = 0

    def _batches(self, ex: Sequence[TemporalExample], steps: int):
        idx = list(range(len(ex)))
        k = 0
        for _ in range(steps):
            if k + self.cfg.batch_size > len(idx) or k == 0:
                self.rng.shuffle(idx)
                k = 0
            yield [ex[i] for i in idx[k:k + self.cfg.batch_size]]
            k += self.cfg.batch_size

    def train(self, examples: Sequence[TemporalExample], steps: int | None = None,
              eval_examples: Sequence[TemporalExample] | None = None, eval_horizons: Sequence[float] = (3600, 86400),
              eval_events: Sequence[str] | None = None) -> dict:
        steps = steps if steps is not None else self.cfg.steps
        if not examples or steps <= 0:
            return {"steps": 0}
        self.model.train()
        snap = param_snapshot(self.model)
        t0 = time.perf_counter()
        losses = []
        for i, b in enumerate(self._batches(list(examples), steps)):
            loss, st = self.model.temporal_loss(b)
            info = self.opt.step(loss)
            losses.append(float(loss.detach()))
            self.examples_seen += len(b)
            rec = {"step": self.opt.global_step, "loss": losses[-1], **st, **info}
            if self.cfg.eval_every and eval_examples and (i + 1) % self.cfg.eval_every == 0:
                self.model.eval()
                rec["eval"] = self.evaluate(eval_examples, eval_horizons, eval_events)
                self.model.train()
            self.history.append(rec)
        self.model.eval()
        self.version += 1
        self.model.checkpoint_id = f"{self.name}-v{self.version}-s{self.opt.global_step}"
        return {"steps": steps, "first_loss": losses[0], "last_loss": float(sum(losses[-10:]) / min(10, len(losses))),
                "param_change": param_change(self.model, snap), "seconds": time.perf_counter() - t0,
                "examples_per_s": steps * self.cfg.batch_size / max(1e-9, time.perf_counter() - t0),
                "checkpoint": self.model.checkpoint_id}

    @torch.no_grad()
    def evaluate(self, examples: Sequence[TemporalExample], horizons: Sequence[float] = (3600, 86400),
                 events: Sequence[str] | None = None) -> dict:
        self.model.eval()
        spec = self.model.cfg.spec
        return evaluate_temporal(model_predict_fn(self.model), examples, horizons, events or spec.events, spec.counts)

    def progressive_update(self, builder: TemporalTrainingDataBuilder, as_of: dict[str, float],
                           splits: Sequence[str] = ("train",), force: bool = False, **build_kw) -> dict:
        changed = builder.ledger.changed_since_ingest(self.name)
        new = builder.build(as_of=as_of, splits=splits, action_ids=changed, **build_kw)
        if len(new) < self.cfg.progressive_min_new and not force:
            return {"trained": False, "new_examples": len(new)}
        keep = {e.example_id for e in new}
        self.replay = [e for e in self.replay if e.example_id not in keep] + new
        n_new = max(1, int(round(self.cfg.batch_size * (1 - self.cfg.replay_ratio))))
        old = [e for e in self.replay if e.example_id not in keep]
        steps = self.cfg.progressive_steps
        losses = []
        snap = param_snapshot(self.model)
        self.model.train()
        for _ in range(steps):
            b = self.rng.sample(new, min(n_new, len(new)))
            if old:
                b += self.rng.sample(old, min(self.cfg.batch_size - len(b), len(old)))
            loss, _ = self.model.temporal_loss(b)
            self.opt.step(loss)
            losses.append(float(loss.detach()))
        self.model.eval()
        builder.ledger.mark_ingested({e.action_id for e in new}, self.name, self.opt.global_step)
        self.version += 1
        self.model.checkpoint_id = f"{self.name}-v{self.version}-s{self.opt.global_step}"
        return {"trained": True, "new_examples": len(new), "replay": len(old), "steps": steps,
                "loss": sum(losses) / len(losses), "param_change": param_change(self.model, snap),
                "checkpoint": self.model.checkpoint_id}

    def save(self, path: str | Path, extra: dict | None = None) -> Path:
        return save_checkpoint(path, self.model, self.opt, extra={"checkpoint_id": self.model.checkpoint_id,
                                                                   "examples_seen": self.examples_seen, **(extra or {})})

    def load_optimizer(self, path: str | Path) -> None:
        load_trainer_state(path, self.opt, restore_rng=False)
