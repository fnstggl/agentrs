"""Task sources and the three-way evaluation protocol.

INNER TRAINING  — tasks a CAM candidate learns on (MiniWoB TRAIN families, train seed range; AppWorld ``train``).
OUTER VALIDATION — tasks the outer learner may use to compare/select candidates (MiniWoB VAL families + val seeds of
                   train families; AppWorld ``dev``). Task families are disjoint from training where feasible.
LOCKED TEST     — MiniWoB TEST families (test seed range), AppWorld ``test_normal``. Only reachable through
                   ``locked_test_tasks`` with an explicit acknowledgement string; every access is appended to an
                   audit log. Outer-learner code never imports it (enforced by ``tests/cam2/test_protocol.py``).

``protocol_hash()`` fingerprints every split definition; the outer learner records it at start and refuses to run
if it changes (it cannot modify splits, success criteria or evaluators).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

from agentrs.cam2.envs.base import TaskRef
from agentrs.cam2.envs.miniwob import SEED_RANGES, TEST_FAMILIES, TRAIN_FAMILIES, VAL_FAMILIES

APPWORLD_ROOT = os.environ.get("APPWORLD_ROOT", "/workspace/ext/appworld_root")
APPWORLD_SPLITS = {"train": "train", "val": "dev", "test": "test_normal"}
ACK = "I am running the final locked-test evaluation and will not use it for selection"
ACCESS_LOG = Path(os.environ.get("CAM2_LOCKED_LOG", "runs/cam2/LOCKED_TEST_ACCESS.log"))


def appworld_ids(split: str) -> list:
    p = Path(APPWORLD_ROOT, "data", "datasets", f"{APPWORLD_SPLITS[split]}.txt")
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def protocol_hash() -> str:
    d = {"miniwob": {"train": TRAIN_FAMILIES, "val": VAL_FAMILIES, "test": TEST_FAMILIES, "seeds": SEED_RANGES},
         "appworld": APPWORLD_SPLITS}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


class TrainSampler:
    """Endless inner-training task stream."""

    def __init__(self, env: str, families: Optional[list] = None, seed: int = 0, weights: Optional[dict] = None):
        self.env = env
        self.rng = random.Random(seed)
        if env == "miniwob":
            self.families = list(families or TRAIN_FAMILIES)
            for f in self.families:
                if f in VAL_FAMILIES or f in TEST_FAMILIES:
                    raise ValueError(f"{f} is a held-out family")
        elif env == "appworld":
            self.ids = appworld_ids("train")
        elif env == "fixture":  # unit-test fixture only
            self.families = list(families or ["press:3", "type"])
        elif env == "crucible":  # Crucible goal instances "<goal>:<world family>:<level>" (envs/crucible_env.py)
            from agentrs.cam2.envs.crucible_env import DEFAULT_TRAIN_FAMILIES

            self.families = list(families or DEFAULT_TRAIN_FAMILIES)
        self.weights = weights

    def __call__(self) -> TaskRef:
        if self.env == "miniwob":
            fam = self.rng.choice(self.families)
            lo, hi = SEED_RANGES["train"]
            return TaskRef("miniwob", fam, self.rng.randrange(lo, hi))
        if self.env == "appworld":
            tid = self.rng.choice(self.ids)
            return TaskRef("appworld", tid.rsplit("_", 1)[0], 0, tid)
        if self.env == "fixture":
            return TaskRef("fixture", self.rng.choice(self.families), self.rng.randrange(0, 10_000))
        if self.env == "crucible":
            from agentrs.cam2.envs.crucible_env import SEED_RANGES as CR

            return TaskRef("crucible", self.rng.choice(self.families), self.rng.randrange(*CR["train"]))
        raise ValueError(self.env)


def val_tasks(env: str, per_family: int = 8, families: Optional[list] = None, include_train_families: bool = True,
              limit: Optional[int] = None) -> list:
    """Outer-validation tasks (deterministic). MiniWoB: held-out VAL families plus the val seed range of TRAIN
    families (instance-level generalisation)."""
    if env == "miniwob":
        lo = SEED_RANGES["val"][0]
        fams = list(families) if families is not None else (VAL_FAMILIES + (TRAIN_FAMILIES if include_train_families else []))
        out = [TaskRef("miniwob", f, lo + i) for f in fams for i in range(per_family)]
    elif env == "appworld":
        out = [TaskRef("appworld", t.rsplit("_", 1)[0], 0, t) for t in appworld_ids("val")]
    elif env == "fixture":
        out = [TaskRef("fixture", f, 50_000 + i) for f in (families or ["press:3", "type"]) for i in range(per_family)]
    elif env == "crucible":
        from agentrs.cam2.envs.crucible_env import DEFAULT_VAL_FAMILIES, SEED_RANGES as CR

        out = [TaskRef("crucible", f, CR["val"][0] + i) for f in (families or DEFAULT_VAL_FAMILIES) for i in range(per_family)]
    else:
        raise ValueError(env)
    return out[:limit] if limit else out


def train_eval_tasks(env: str, per_family: int = 4, families: Optional[list] = None) -> list:
    """Fixed tasks drawn from the TRAIN families and train seed range (to measure retention of trained skills)."""
    if env == "miniwob":
        return [TaskRef("miniwob", f, 7_000 + i) for f in (families or TRAIN_FAMILIES) for i in range(per_family)]
    return [TaskRef("appworld", t.rsplit("_", 1)[0], 0, t) for t in appworld_ids("train")[: per_family * 8]]


def locked_test_tasks(env: str, ack: str, per_family: int = 16, who: str = "") -> list:
    if ack != ACK:
        raise PermissionError("the locked test split requires the explicit acknowledgement string")
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ACCESS_LOG.open("a") as fh:
        fh.write(json.dumps({"time": time.time(), "env": env, "who": who, "protocol": protocol_hash()}) + "\n")
    if env == "miniwob":
        lo = SEED_RANGES["test"][0]
        return [TaskRef("miniwob", f, lo + i) for f in TEST_FAMILIES for i in range(per_family)]
    if env == "appworld":
        return [TaskRef("appworld", t.rsplit("_", 1)[0], 0, t) for t in appworld_ids("test")]
    raise ValueError(env)
