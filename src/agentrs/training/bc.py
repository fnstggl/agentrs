"""Behavior cloning from demonstrations.

Objective: weighted negative log-likelihood of the demonstrated action under
the candidate-restricted policy, ``-w * log pi(a_demo | s)``.

Demonstration quality handling (demonstrators are not assumed optimal):
* steps from verified-successful episodes: weight ``w_success`` (1.0)
* steps from episodes with unknown outcome: ``w_unknown`` (0.3)
* steps from failed episodes: ``w_failure`` (0.0 by default -> excluded from BC,
  but still used for world-model training)
* steps whose action was invalid (environment error): excluded
* inferred (IDM-labelled) actions: multiplied by the label confidence
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from pydantic import ConfigDict, BaseModel, Field

from agentrs.data.schemas import Source, Transition
from agentrs.models.policy import LMPolicy
from agentrs.monitoring.tracker import Tracker
from agentrs.training.common import GradientTrainer, OptimConfig, param_change, param_snapshot, save_checkpoint


class BCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optim: OptimConfig = Field(default_factory=lambda: OptimConfig(lr=3e-4, total_steps=400, warmup_steps=20,
                                                                   schedule="cosine"))
    batch_size: int = 16
    steps: int = 400
    w_success: float = 1.0
    w_unknown: float = 0.3
    w_failure: float = 0.0
    eval_every: int = 100
    seed: int = 0


@dataclass
class BCExample:
    prompt: str
    candidates: list[str]
    target: int
    weight: float


def demo_weight(t: Transition, cfg: BCConfig) -> float:
    if not t.action_valid or t.belief_prompt is None:
        return 0.0
    if t.episode_success is True:
        w = cfg.w_success
    elif t.episode_success is False:
        w = cfg.w_failure
    else:
        w = cfg.w_unknown
    if t.provenance.source == Source.INFERRED:
        w *= float(t.action_confidence or 0.0)
    if t.provenance.source == Source.IMAGINED:
        return 0.0
    return w


def make_examples(ts: Sequence[Transition], cfg: BCConfig, max_candidates: int = 40) -> list[BCExample]:
    out = []
    for t in ts:
        w = demo_weight(t, cfg)
        if w <= 0:
            continue
        cands = list(t.candidates[:max_candidates]) if t.candidates else [t.action]
        if t.action not in cands:
            cands = cands[: max_candidates - 1] + [t.action]
        out.append(BCExample(prompt=t.belief_prompt or "", candidates=cands, target=cands.index(t.action), weight=w))
    return out


def bc_loss(policy: LMPolicy, batch: Sequence[BCExample]) -> tuple[torch.Tensor, dict]:
    lps = policy.log_probs([b.prompt for b in batch], [b.candidates for b in batch])
    nll = torch.stack([-lp[b.target] for lp, b in zip(lps, batch)])
    w = torch.tensor([b.weight for b in batch], dtype=nll.dtype, device=nll.device)
    loss = (w * nll).sum() / w.sum().clamp(min=1e-8)
    acc = sum(int(torch.argmax(lp)) == b.target for lp, b in zip(lps, batch)) / len(batch)
    return loss, {"nll": float(nll.detach().mean()), "acc": acc}


@torch.no_grad()
def evaluate_imitation(policy: LMPolicy, examples: Sequence[BCExample], batch_size: int = 32) -> dict:
    if not examples:
        return {"val_nll": float("nan"), "val_acc": float("nan")}
    policy.eval()
    tot_nll, tot_acc, n = 0.0, 0.0, 0
    for i in range(0, len(examples), batch_size):
        b = examples[i : i + batch_size]
        _, st = bc_loss(policy, b)
        tot_nll += st["nll"] * len(b)
        tot_acc += st["acc"] * len(b)
        n += len(b)
    policy.train()
    return {"val_nll": tot_nll / n, "val_acc": tot_acc / n}


def train_bc(policy: LMPolicy, train: Sequence[Transition], val: Sequence[Transition], cfg: BCConfig,
             tracker: Optional[Tracker] = None, out_dir: Optional[str] = None,
             trainer: Optional[GradientTrainer] = None, start_step: int = 0) -> dict:
    rng = random.Random(cfg.seed)
    tr_ex = make_examples(train, cfg, policy.cfg.max_candidates)
    va_ex = make_examples([t for t in val if t.episode_success], cfg, policy.cfg.max_candidates)
    if not tr_ex:
        raise ValueError("no usable demonstration steps (check outcomes / validity)")
    trainer = trainer or GradientTrainer(policy, cfg.optim)
    snap = param_snapshot(policy)
    policy.train()
    hist = []
    before = evaluate_imitation(policy, va_ex)
    for step in range(start_step, cfg.steps):
        batch = rng.sample(tr_ex, min(cfg.batch_size, len(tr_ex)))
        loss, st = bc_loss(policy, batch)
        upd = trainer.step(loss)
        rec = {"loss": float(loss.detach()), "train_acc": st["acc"], **upd}
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            rec.update(evaluate_imitation(policy, va_ex))
            rec["param_change_l2"] = param_change(policy, snap)
        if tracker:
            tracker.log(rec, step, prefix="bc/")
        hist.append(rec)
        if out_dir and ((step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps):
            save_checkpoint(out_dir, policy, trainer, extra={"stage": "bc", "step": step + 1})
    after = evaluate_imitation(policy, va_ex)
    return {"n_train_examples": len(tr_ex), "n_val_examples": len(va_ex), "before": before, "after": after,
            "param_change_l2": param_change(policy, snap), "history": hist, "trainer": trainer}
