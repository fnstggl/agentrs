"""Common interface of CAM environments.

Every environment is an executable simulator with hidden parameters that the agent never sees:
the agent receives only ``goal_tokens`` / ``obs_tokens`` / ``features`` (observable), acts with a
token sequence that must follow the environment's action grammar (tool + complete arguments), and
the environment — not any model — computes the consequences, events, verified progress and the final
verified success. ``split``: ``train``/``val`` draw hidden parameters from the training distribution
(val uses disjoint seeds); ``test`` draws them from held-out distributions (unseen counterparty
policies, unseen preference / market structures).
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

NBINS = 21  # numeric tokens N0..N20


def nb(x: float, lo: float = 0.0, hi: float = 1.0) -> str:
    """Bin a number into one of the shared numeric tokens."""
    t = 0.0 if hi == lo else (x - lo) / (hi - lo)
    return f"N{int(round(min(1.0, max(0.0, t)) * (NBINS - 1)))}"


def logb(x: float, hi: float) -> str:
    return nb(np.log1p(max(0.0, x)) / np.log1p(hi))


@dataclass
class StepResult:
    elapsed_h: float                       # real (simulated) time the action took until its response
    events: dict = field(default_factory=dict)
    done: bool = False
    success: Optional[bool] = None         # set by the environment's verifier when done
    valid: bool = True
    info: dict = field(default_factory=dict)


class Grammar:
    """Slot grammar: an action is TYPE followed by one value per slot of that type, then END.
    ``allowed(prefix)`` returns the tokens that may come next (used as the decoding mask, both when
    sampling and when re-scoring for training, so the recorded probabilities are exact)."""

    def __init__(self, spec: dict[str, list[list[str]]], name: str = ""):
        self.spec = spec
        self.name = name

    def allowed(self, prefix: list[str]) -> list[str]:
        if not prefix:
            return list(self.spec)
        slots = self.spec[prefix[0]]
        k = len(prefix) - 1
        return slots[k] if k < len(slots) else ["END"]

    def complete(self, prefix: list[str]) -> bool:
        return bool(prefix) and prefix[-1] == "END"

    def tokens(self) -> set[str]:
        out = set(self.spec) | {"END"}
        for slots in self.spec.values():
            for s in slots:
                out |= set(s)
        return out


class CamEnv:
    family = "base"
    keywords: list[str] = []
    events: list[str] = []
    n_features = 0
    max_steps = 20

    def __init__(self):
        self.rng = random.Random(0)
        self.np = np.random.default_rng(0)
        self.split = "train"
        self.t = 0
        self.done = False
        self.success: Optional[bool] = None

    # ---- to implement
    def reset(self, seed: int, split: str = "train") -> None:
        raise NotImplementedError

    def goal_tokens(self) -> list[str]:
        raise NotImplementedError

    def obs_tokens(self) -> list[str]:
        raise NotImplementedError

    def features(self) -> np.ndarray:
        raise NotImplementedError

    def grammar(self) -> Grammar:
        raise NotImplementedError

    def step(self, action: list[str]) -> StepResult:
        raise NotImplementedError

    def progress(self) -> float:
        """Verified, environment-measured progress toward the goal in [0, 1] (not a model output)."""
        raise NotImplementedError

    def hidden(self) -> dict:
        """Hidden parameters (for analysis only; never given to the agent)."""
        return {}

    def _seed(self, seed: int, split: str) -> None:
        self.rng = random.Random(f"{self.family}:{split}:{seed}")
        self.np = np.random.default_rng(zlib.crc32(f"{self.family}:{split}:{seed}".encode()))
        self.split = split
        self.t = 0
        self.done = False
        self.success = None
