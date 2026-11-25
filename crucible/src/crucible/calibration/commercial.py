"""Randomized commercial-intervention response model (workstream E2), loaded through the registry.

`load("commercial_response@1")` returns a `CommercialResponseModel` fitted on the Criteo Uplift v2.1 randomized
incrementality tests (see its model card and report). Two uses:

* `predict(X)` on Criteo's anonymised features f0..f11 gives the four per-arm probabilities
  [p_visit_control, p_visit_treated, p_conversion_control, p_conversion_treated] (evaluation and auditing);
* the simulator cannot reproduce those anonymised features, so it uses the **response-type table**: the held-out
  validated joint distribution of (baseline propensity, treatment effect). A simulated individual draws a type once
  (`sample_types`), and when it is targeted by an advertising intervention it responds with that type's treated
  probabilities, otherwise with its control probabilities (`respond`). `transport` re-bases the table to a world's
  own base rates while keeping every type's odds ratio (an explicit, documented assumption).

This module never reads a dataset.
"""
from __future__ import annotations

import numpy as np

from . import portable
from .registry_e2 import FittedModel

OUTCOMES = ("visit", "conversion")
ARMS = ("control", "treated")
OUTPUTS = ["p_visit_control", "p_visit_treated", "p_conversion_control", "p_conversion_treated"]
TYPE_FIELDS = ("weight", "p0_visit", "p1_visit", "p0_conversion", "p1_conversion")


def assign_cells(p0: np.ndarray, tau: np.ndarray, p0_edges: np.ndarray, tau_edges: np.ndarray) -> np.ndarray:
    """Type index = p0 bin * n_tau + tau bin, using the inner edges of the stored bin boundaries."""
    n_p0 = p0_edges.size - 1
    n_tau = tau_edges.shape[1] - 1
    b = np.searchsorted(p0_edges[1:-1], p0, side="right")
    cell = np.empty(p0.size, dtype=np.int64)
    for i in range(n_p0):
        m = b == i
        cell[m] = i * n_tau + np.searchsorted(tau_edges[i, 1:-1], tau[m], side="right")
    return cell


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class CommercialResponseModel(FittedModel):
    """Per-arm visit / conversion probabilities (T-learner) plus the simulator-facing response-type table."""

    @property
    def kept(self) -> str:
        return self.card["selection"]["kept"]

    def candidate_predict(self, candidate: str, X: np.ndarray) -> np.ndarray:
        X = self._check(X)
        cols = []
        for outcome in OUTCOMES:
            for arm in ARMS:
                cols.append(portable.predict(self.submodel(f"{candidate}/{outcome}_{arm}"), X))
        return np.column_stack(cols)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """[n, 4]: p_visit_control, p_visit_treated, p_conversion_control, p_conversion_treated."""
        return self.candidate_predict(self.kept, X)

    def effect(self, X: np.ndarray) -> dict[str, np.ndarray]:
        P = self.predict(X)
        return {"visit": P[:, 1] - P[:, 0], "conversion": P[:, 3] - P[:, 2]}

    # ------------------------------------------------------------------------------------------- response types

    def response_types(self, estimate: str = "model") -> dict[str, np.ndarray]:
        """The type table: weight and per-arm probabilities of each of the K types (copies).

        estimate="model": the kept model's mean predictions per type (the prespecified table);
        estimate="honest": per-type arm rates observed on the validation split (cells defined by the model, rates
        estimated on rows the model was not fitted on; noisier, but free of the regression to the mean that makes
        the model's extreme types too extreme). Empty validation cells fall back to the model values.
        """
        tab = {k: np.array(self.arrays[f"types/{k}"], dtype=np.float64) for k in TYPE_FIELDS}
        if estimate == "model":
            return tab
        if estimate != "honest":
            raise ValueError("estimate must be 'model' or 'honest'")
        for outcome in ("visit", "conversion"):
            for arm, key in (("control", "p0"), ("treated", "p1")):
                n = self.arrays[f"types/validation_{outcome}_{arm}_n"]
                r = self.arrays[f"types/validation_{outcome}_{arm}_rate"]
                tab[f"{key}_{outcome}"] = np.where(n > 0, r, tab[f"{key}_{outcome}"])
        return tab

    @property
    def types_usable(self) -> bool:
        """Whether the table passed its prespecified held-out acceptance rule."""
        return bool(self.card["response_types"]["usable"])

    def assign_types(self, X: np.ndarray) -> np.ndarray:
        """Type of each Criteo row (evaluation only: simulated individuals draw types with `sample_types`)."""
        P = self.predict(X)
        return assign_cells(P[:, 0], P[:, 1] - P[:, 0], self.arrays["types/p0_edges"], self.arrays["types/tau_edges"])

    def sample_types(self, rng: np.random.Generator, n: int) -> np.ndarray:
        w = self.arrays["types/weight"].astype(np.float64)
        return rng.choice(w.size, size=int(n), p=w / w.sum())

    def respond(self, rng: np.random.Generator, types: np.ndarray, treated: np.ndarray,
                table: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """Draw visit and conversion for individuals of the given types and arms.

        Conversion only happens after a visit (true for every row of the data): visit ~ Bernoulli(p_visit) and
        conversion | visit ~ Bernoulli(min(1, p_conversion / p_visit)), which keeps the marginal conversion rate.
        """
        tab = table or self.response_types()
        types = np.asarray(types, dtype=np.int64); treated = np.asarray(treated).astype(bool)
        pv = np.where(treated, tab["p1_visit"][types], tab["p0_visit"][types])
        pc = np.where(treated, tab["p1_conversion"][types], tab["p0_conversion"][types])
        visit = rng.random(types.size) < pv
        conv = visit & (rng.random(types.size) < np.minimum(1.0, pc / np.maximum(pv, 1e-12)))
        return {"visit": visit, "conversion": conv}

    def transport(self, control_visit_rate: float | None = None,
                  control_conversion_rate: float | None = None) -> dict[str, np.ndarray]:
        """Re-base the table to a world's own control-arm rates, keeping each type's odds ratio and the ranking.

        Assumption (recorded in the card's limits): the treatment acts on the log-odds scale and the shift of the
        baseline is common to all types. Without arguments the table is returned unchanged.
        """
        tab = self.response_types()
        w = tab["weight"] / tab["weight"].sum()
        for outcome, target in (("visit", control_visit_rate), ("conversion", control_conversion_rate)):
            if target is None:
                continue
            if not 0.0 < target < 1.0:
                raise ValueError("target rates must be in (0, 1)")
            z0 = _logit(tab[f"p0_{outcome}"]); z1 = _logit(tab[f"p1_{outcome}"])
            lo, hi = -30.0, 30.0
            for _ in range(200):
                mid = (lo + hi) / 2
                if float(np.dot(w, _sigmoid(z0 + mid))) < target:
                    lo = mid
                else:
                    hi = mid
            delta = (lo + hi) / 2
            tab[f"p0_{outcome}"] = _sigmoid(z0 + delta)
            tab[f"p1_{outcome}"] = _sigmoid(z1 + delta)
        return tab
