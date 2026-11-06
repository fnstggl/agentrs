"""Inverse-dynamics training and labelling of action-free recordings.

Training: real transitions (obs_before, obs_after, action) -> cross-entropy over generic
candidates derived from obs_before (+ values visible in obs_after).
Labelling: a recording that has observations but no actions gets uncertain labels
(``Source.INFERRED``, ``action_confidence``); low-confidence steps are abstained on and
dropped. BC down-weights inferred labels by their confidence.
"""

from __future__ import annotations

import random
import uuid
from typing import Optional, Sequence

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from agentrs.data.diff import compute_diff
from agentrs.data.schemas import Provenance, Source, Transition
from agentrs.environments.base import Observation
from agentrs.models.idm import InverseDynamicsModel
from agentrs.training.common import GradientTrainer, OptimConfig


class IDMTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    optim: OptimConfig = Field(default_factory=lambda: OptimConfig(lr=1e-3, total_steps=300))
    steps: int = 300
    batch_size: int = 16
    seed: int = 0


def idm_examples(idm: InverseDynamicsModel, ts: Sequence[Transition]):
    out = []
    for t in ts:
        if not t.is_real or not t.action_valid:
            continue
        cands = idm.candidates(t.goal, t.observation, t.next_observation)
        if t.action not in cands:
            continue
        out.append((idm.prompt(t.goal, t.observation, t.next_observation), cands, cands.index(t.action)))
    return out


def train_idm(idm: InverseDynamicsModel, train: Sequence[Transition], cfg: IDMTrainConfig) -> dict:
    rng = random.Random(cfg.seed)
    ex = idm_examples(idm, train)
    tr = GradientTrainer(idm, cfg.optim)
    hist = []
    for _ in range(cfg.steps):
        b = rng.sample(ex, min(cfg.batch_size, len(ex)))
        lps = idm.log_probs([x[0] for x in b], [x[1] for x in b])
        loss = -torch.stack([lp[x[2]] for lp, x in zip(lps, b)]).mean()
        tr.step(loss)
        hist.append(float(loss.detach()))
    return {"n_examples": len(ex), "loss_first10": float(np.mean(hist[:10])), "loss_last10": float(np.mean(hist[-10:]))}


@torch.no_grad()
def evaluate_idm(idm: InverseDynamicsModel, ts: Sequence[Transition]) -> dict:
    """Accuracy, abstention rate and selective accuracy (accuracy on non-abstained steps).
    Steps where several actions produce identical observable changes are inherently ambiguous."""
    n = correct = abstain = sel_correct = 0
    for t in ts:
        if not t.action_valid:
            continue
        lab = idm.label(t.goal, t.observation, t.next_observation)
        n += 1
        top = lab.candidates[int(np.argmax(lab.probs))]
        correct += int(top == t.action)
        if lab.abstained:
            abstain += 1
        else:
            sel_correct += int(lab.action.to_str() == t.action)
    kept = n - abstain
    return {"n": n, "accuracy": correct / max(1, n), "abstention_rate": abstain / max(1, n),
            "selective_accuracy": sel_correct / max(1, kept)}


@torch.no_grad()
def label_recording(idm: InverseDynamicsModel, goal: str, observations: Sequence[Observation], task_id: str = "",
                    family: str = "", split: str = "train", source_id: str = "video", env_id: str = "miniweb",
                    outcome: Optional[bool] = None) -> list[Transition]:
    """Turn an action-free observation sequence into inferred-action transitions."""
    ep_id = "inferred-" + uuid.uuid4().hex[:12]
    out = []
    for i in range(len(observations) - 1):
        before, after = observations[i], observations[i + 1]
        lab = idm.label(goal, before, after)
        if lab.abstained:
            continue
        d = compute_diff(before, after)
        out.append(Transition(
            episode_id=ep_id, step=i, task_id=task_id, env_id=env_id, family=family, split=split, goal=goal,
            observation=before, action=lab.action.to_str(), action_type=lab.action.type.value,
            candidates=lab.candidates, chosen_index=lab.candidates.index(lab.action.to_str()),
            next_observation=after, diff_text=d.to_text(), events=d.events(after), episode_success=outcome,
            action_confidence=lab.confidence, selection="inferred",
            provenance=Provenance(source=Source.INFERRED, source_id=source_id)))
    return out
