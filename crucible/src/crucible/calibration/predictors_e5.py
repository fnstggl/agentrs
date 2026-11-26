"""Predictor kind of the E5 advertising-response model (registered into `predictors.KINDS` on import; the card names
this module in its ``module`` field so that `registry.load` imports it first). NumPy only.

`AdExposure` maps the log-odds of a displayed individual's organic (no-ad) visit probability onto its visit
probability when it is displayed an advertiser's campaign (plan E5, docs/v5/calibration/plans/E5_ad_exposure_plan.json):

* mode 0 (``B0_no_effect``): q = p0;
* mode 1 (``B1_constant_difference``): q = clip(p0 + max(delta, 0));
* mode 2 (``M1``/``M2``): q = sigmoid(l0 + max(a + b (l0 - c), 0)) (M1 has b = 0);
* mode 3 (``M3``): q = sigmoid(l0 + max(D_k, 0)), D_k of the knot nearest to l0.

Every shift is floored at 0 (no negative persuasion; plan E5 ``negative_effects``). Row-wise: a row's output depends
only on its own input.
"""
from __future__ import annotations

import numpy as np

from . import predictors as P

MODES = ("none", "difference", "linear_logit", "step_logit")


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


class AdExposure:
    kind = "ad_exposure"

    def __init__(self, mode: str, delta: float = 0.0, a: float = 0.0, b: float = 0.0, c: float = 0.0,
                 knots=(), shifts=()):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.delta, self.a, self.b, self.c = float(delta), float(a), float(b), float(c)
        self.knots = np.asarray(knots, dtype=np.float64).reshape(-1)
        self.shifts = np.asarray(shifts, dtype=np.float64).reshape(-1)
        if self.mode == "step_logit" and (self.knots.size == 0 or self.knots.size != self.shifts.size):
            raise ValueError("step_logit needs one shift per knot")

    def params(self) -> dict:
        return {"mode": np.array([MODES.index(self.mode)]), "coef": np.array([self.delta, self.a, self.b, self.c]),
                "knots": self.knots, "shifts": self.shifts}

    @classmethod
    def from_params(cls, Pm, card=None):
        d, a, b, c = (float(x) for x in Pm["coef"])
        return cls(MODES[int(Pm["mode"][0])], d, a, b, c, Pm["knots"], Pm["shifts"])

    def shift(self, l0) -> np.ndarray:
        """The log-odds shift a display adds at organic log-odds l0 (>= 0; modes 2 and 3 only)."""
        l0 = np.asarray(l0, dtype=np.float64).reshape(-1)
        if self.mode == "linear_logit":
            return np.maximum(self.a + self.b * (l0 - self.c), 0.0)
        if self.mode == "step_logit":
            j = np.abs(l0[:, None] - self.knots[None, :]).argmin(axis=1)
            return np.maximum(self.shifts[j], 0.0)
        p0 = P._sigmoid(l0)
        q = self.predict(l0[:, None])
        return logit(q) - logit(p0)

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        l0 = X[:, 0] if X.ndim == 2 else X.reshape(-1)
        p0 = P._sigmoid(l0)
        if self.mode == "none":
            return p0
        if self.mode == "difference":
            return np.clip(p0 + max(self.delta, 0.0), 1e-6, 1 - 1e-6)
        return P._sigmoid(l0 + self.shift(l0))


P.register_kind(AdExposure.kind, AdExposure)
