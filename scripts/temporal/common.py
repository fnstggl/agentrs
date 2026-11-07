"""Shared helpers for the temporal / observational-learning experiments (CPU, Pythia-14M)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from agentrs.models.backbone import BackboneConfig
from agentrs.sensors.socialweblab_source import lab_temporal_spec
from agentrs.temporal.temporal_world_model import TemporalWorldModel, TemporalWorldModelConfig

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(os.environ.get("AGENTRS_RUNS", ROOT / "runs" / "temporal"))
RESULTS = ROOT / "results" / "temporal"
EVENTS = ["first_response", "answered", "positive_feedback", "resolved", "removed"]
HORIZONS = [300.0, 3600.0, 6 * 3600.0, 86400.0, 3 * 86400.0]
HELD_HORIZONS = [1800.0, 3 * 3600.0, 12 * 3600.0, 2 * 86400.0]  # between the observed profile horizons


def threads(n: int | None = None) -> None:
    torch.set_num_threads(int(n or os.environ.get("AGENTRS_THREADS", "4")))


def make_wm(seed: int = 0, model: str = "EleutherAI/pythia-14m", n_heads: int = 5, **kw) -> TemporalWorldModel:
    torch.manual_seed(seed)
    cfg = TemporalWorldModelConfig(backbone=BackboneConfig(name=model, max_prompt_tokens=320, seed=seed),
                                   n_heads=n_heads, spec=lab_temporal_spec(), **kw)
    return TemporalWorldModel(cfg)


def save_json(name: str, obj) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    return p


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def s(self) -> float:
        return round(time.perf_counter() - self.t0, 1)


def strip_content(ex):
    """Ablation: the action without its generated content (operation only)."""
    e = ex.model_copy(deep=True)
    op = e.complete_action.split("(", 1)[0]
    e.complete_action = f'{op}(…)'
    return e


def strip_context(ex, keep=("ENV", "TOPIC")):
    """Ablation: drop audience (followers) and timing covariates from the context."""
    e = ex.model_copy(deep=True)
    e.initial_state = "\n".join(line for line in e.initial_state.splitlines() if line.split(":")[0] in keep)
    return e
