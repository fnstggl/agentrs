"""cam_live.predict.errors — telling apart the reasons a prediction missed (spec 4.2).

A miss is attributed, where the evidence allows, to one of:

    missing_observation   the outcome was never measured (pending / unobservable) — not an error at all
    within_noise          the outcome lies inside the model's own aleatoric spread: irreducible randomness
    model_uncertain       outside the aleatoric spread but inside the total (aleatoric + epistemic) spread:
                          the model already knew that it did not know — a reducible prediction error
    changed_conditions    a sustained shift of standardised residuals detected by :class:`ChangeDetector`
    model_wrong           a confident miss with no detected change: an incorrect assumption of the model

The classifier only labels; it creates no training signal and no reward.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from ..contracts import OutcomeStatus

MISSING = "missing_observation"
NOISE = "within_noise"
UNCERTAIN = "model_uncertain"
CHANGED = "changed_conditions"
WRONG = "model_wrong"
LABELS = (MISSING, NOISE, UNCERTAIN, CHANGED, WRONG)


class ChangeDetector:
    """Two-sided Page-Hinkley / CUSUM test on standardised residuals ``(y - mean) / total_std``.

    Under a correct model the residuals have mean 0 and variance ~1; ``drift`` is the slack per observation and
    ``threshold`` the decision level in residual standard deviations. ``update`` returns True while a sustained
    shift is detected; ``reset`` is called by the owner once it has adapted (for example after raising the fast
    layer's process noise)."""

    def __init__(self, drift: float = 0.5, threshold: float = 8.0, clip: float = 6.0):
        self.drift, self.threshold, self.clip = drift, threshold, clip
        self.pos = self.neg = 0.0
        self.n = 0
        self.detected_at: Optional[int] = None

    def update(self, residual: float) -> bool:
        if residual != residual:
            return self.changed
        r = max(-self.clip, min(self.clip, float(residual)))
        self.n += 1
        self.pos = max(0.0, self.pos + r - self.drift)
        self.neg = max(0.0, self.neg - r - self.drift)
        if self.detected_at is None and max(self.pos, self.neg) > self.threshold:
            self.detected_at = self.n
        return self.changed

    @property
    def changed(self) -> bool:
        return self.detected_at is not None

    @property
    def score(self) -> float:
        return max(self.pos, self.neg) / self.threshold

    def reset(self) -> None:
        self.pos = self.neg = 0.0
        self.detected_at = None

    def state(self) -> Dict[str, float]:
        return {"pos": self.pos, "neg": self.neg, "n": self.n,
                "detected_at": -1 if self.detected_at is None else self.detected_at}


@dataclass
class ErrorAttribution:
    label: str
    z_aleatoric: float               # |error| in units of the aleatoric standard deviation
    z_total: float                   # |error| in units of the total standard deviation
    change_score: float


def classify_error(status: str, value: Optional[float], mean: float, aleatoric_std: float, epistemic_std: float,
                   change: Optional[ChangeDetector] = None, z_noise: float = 2.0) -> ErrorAttribution:
    """Attribute one resolved prediction. ``z_noise`` is the half-width (in standard deviations) of what counts as
    consistent with a spread."""
    score = change.score if change is not None else 0.0
    if status not in (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED) or value is None:
        return ErrorAttribution(MISSING, float("nan"), float("nan"), score)
    err = abs(float(value) - float(mean))
    total = math.sqrt(aleatoric_std ** 2 + epistemic_std ** 2)
    za = err / max(aleatoric_std, 1e-12)
    zt = err / max(total, 1e-12)
    if za <= z_noise:
        label = NOISE
    elif change is not None and change.changed:
        label = CHANGED
    elif zt <= z_noise:
        label = UNCERTAIN
    else:
        label = WRONG
    return ErrorAttribution(label, za, zt, score)
