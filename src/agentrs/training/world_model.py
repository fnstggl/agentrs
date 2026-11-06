"""World-model training on real transitions (demonstrations and own experience).

The same model learns from teacher/human demonstrations and from the agent's
own interaction: every real transition (obs, action, next_obs, reward, done)
is a supervised dynamics example, regardless of whether the action was good.
Imagined transitions are never used as world-model targets.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from pydantic import ConfigDict, BaseModel, Field

from agentrs.data.schemas import Transition
from agentrs.models.world_model import WorldModel, targets_from_transition
from agentrs.monitoring.tracker import Tracker
from agentrs.training.common import GradientTrainer, OptimConfig, param_change, param_snapshot, save_checkpoint


class WMTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optim: OptimConfig = Field(default_factory=lambda: OptimConfig(lr=5e-4, total_steps=600, warmup_steps=30,
                                                                   schedule="cosine"))
    batch_size: int = 16
    steps: int = 600
    eval_every: int = 150
    seed: int = 0


@dataclass
class WMExample:
    prompt: str
    diff_text: str
    targets: list[float]
    key: str
    action_type: str
    transition: Optional[Transition] = None


def outcome_text(t: Transition) -> str:
    if t.error:
        return "error: " + t.error[:80]
    if t.next_observation.url != t.observation.url:
        return f"navigated to {t.next_observation.url}"
    return "ok"


def episode_histories(ts: Sequence[Transition]) -> dict[tuple[str, int], list[tuple[str, str]]]:
    by_ep: dict[str, list[Transition]] = {}
    for t in ts:
        by_ep.setdefault(t.episode_id, []).append(t)
    out = {}
    for eid, items in by_ep.items():
        items.sort(key=lambda t: t.step)
        hist: list[tuple[str, str]] = []
        for t in items:
            out[(eid, t.step)] = list(hist)
            hist.append((t.action, outcome_text(t)))
    return out


def make_wm_examples(wm: WorldModel, ts: Sequence[Transition]) -> list[WMExample]:
    real = [t for t in ts if t.is_real]
    hists = episode_histories(real)
    out = []
    for t in real:
        h = hists.get((t.episode_id, t.step), [])
        out.append(WMExample(
            prompt=wm.prompt(t.goal, t.observation, t.action, h), diff_text=t.diff_text or "(no change)",
            targets=targets_from_transition(t.events, t.reward, t.terminated), key=f"{t.episode_id}:{t.step}",
            action_type=t.action_type, transition=t))
    return out


def wm_batch_loss(wm: WorldModel, batch: Sequence[WMExample]) -> tuple[torch.Tensor, dict]:
    tg = torch.tensor([b.targets for b in batch], dtype=torch.float32)
    return wm.loss([b.prompt for b in batch], [b.diff_text for b in batch], tg, [b.key for b in batch])


@torch.no_grad()
def wm_validation_loss(wm: WorldModel, examples: Sequence[WMExample], batch_size: int = 32) -> dict:
    if not examples:
        return {}
    wm.eval()
    agg: dict[str, float] = {}
    n = 0
    for i in range(0, len(examples), batch_size):
        b = examples[i : i + batch_size]
        loss, st = wm_batch_loss(wm, b)
        for k, v in {"loss": float(loss), **st}.items():
            agg[k] = agg.get(k, 0.0) + v * len(b)
        n += len(b)
    wm.train()
    return {f"val_{k}": v / n for k, v in agg.items()}


def train_world_model(wm: WorldModel, train: Sequence[Transition], val: Sequence[Transition], cfg: WMTrainConfig,
                      tracker: Optional[Tracker] = None, out_dir: Optional[str] = None,
                      trainer: Optional[GradientTrainer] = None, start_step: int = 0,
                      log_prefix: str = "wm/") -> dict:
    rng = random.Random(cfg.seed)
    tr = make_wm_examples(wm, train)
    va = make_wm_examples(wm, val)
    if not tr:
        raise ValueError("no real transitions to train the world model on")
    trainer = trainer or GradientTrainer(wm, cfg.optim)
    snap = param_snapshot(wm)
    before = wm_validation_loss(wm, va)
    wm.train()
    hist = []
    for step in range(start_step, cfg.steps):
        batch = rng.sample(tr, min(cfg.batch_size, len(tr)))
        loss, st = wm_batch_loss(wm, batch)
        upd = trainer.step(loss)
        rec = {"loss": float(loss.detach()), **st, **upd}
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            rec.update(wm_validation_loss(wm, va))
            rec["param_change_l2"] = param_change(wm, snap)
            if out_dir:
                save_checkpoint(out_dir, wm, trainer, extra={"stage": "world_model", "step": step + 1})
        if tracker:
            tracker.log(rec, step, prefix=log_prefix)
        hist.append(rec)
    after = wm_validation_loss(wm, va)
    return {"n_train": len(tr), "n_val": len(va), "before": before, "after": after,
            "param_change_l2": param_change(wm, snap), "history": hist, "trainer": trainer}
