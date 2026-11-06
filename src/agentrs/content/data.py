"""Content experience: complete content + context + predicted effects + real effects.

``ContentExperienceDataset`` views the experience store's transitions that carry a content
action. Each row keeps the complete content, the goal/context, what the world model predicted
before execution (probabilities and uncertainty), the real observed outcome, the verified
reward, the prediction error and recorded confounders.

Causal caution: ``observational_association`` reports correlations in logged data and labels
them as non-causal; ``randomized_variant_test`` runs a controlled, randomised comparison of
content variants in an authorised sandbox environment and is the only estimate labelled causal.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from agentrs.data.schemas import Transition


@dataclass
class ContentRow:
    transition: Transition
    content: str
    operation: str
    goal: str
    context: str
    source: str
    predicted_objective: Optional[float]
    predicted_epistemic: Optional[float]
    objective_met: bool
    verified_reward: float
    engagement: Optional[float]
    confounders: dict

    @property
    def prediction_error(self) -> Optional[float]:
        if self.predicted_objective is None:
            return None
        return abs(self.predicted_objective - float(self.objective_met))


class ContentExperienceDataset:
    def __init__(self, transitions: Sequence[Transition]):
        self.rows: list[ContentRow] = []
        for t in transitions:
            sa = t.semantic_action
            if sa is None or t.content_outcome is None:
                continue
            o = t.content_outcome
            pred = t.wm_prediction
            self.rows.append(ContentRow(
                transition=t, content=sa.content, operation=sa.operation, goal=t.goal, context=sa.context,
                source=t.provenance.source.value,
                predicted_objective=(pred.event_probs[1] if pred is not None and len(pred.event_probs) > 1 else None),
                predicted_epistemic=pred.epistemic if pred is not None else None,
                objective_met=bool(o.get("objective_met")), verified_reward=float(t.reward),
                engagement=o.get("engagement"), confounders=dict(o.get("confounders") or {})))

    def __len__(self) -> int:
        return len(self.rows)

    def by_source(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.rows:
            out[r.source] = out.get(r.source, 0) + 1
        return out

    def mean_prediction_error(self) -> Optional[float]:
        errs = [r.prediction_error for r in self.rows if r.prediction_error is not None]
        return float(np.mean(errs)) if errs else None


def observational_association(rows: Sequence[ContentRow], feature: Callable[[ContentRow], bool],
                              outcome: Callable[[ContentRow], float]) -> dict:
    """Difference in mean outcome between rows with and without a content feature in LOGGED data.
    This is a correlation: timing, topic or audience may confound it."""
    a = [outcome(r) for r in rows if feature(r)]
    b = [outcome(r) for r in rows if not feature(r)]
    if not a or not b:
        return {"n_with": len(a), "n_without": len(b), "difference": None, "causal": False}
    return {"n_with": len(a), "n_without": len(b), "difference": float(np.mean(a) - np.mean(b)),
            "causal": False, "note": "observational association; not a causal effect"}


def randomized_variant_test(env, tasks, variants: dict[str, Callable[[object], str]], operation_for,
                            outcome_key: str = "objective_met", seed: int = 0, authorized: bool = False) -> dict:
    """Randomised controlled comparison of content variants. Each task is assigned a variant at
    random and executed once; the estimate is causal for the population of tasks used.
    Requires ``authorized=True`` and is meant for sandbox environments (never real communities)."""
    if not authorized:
        raise PermissionError("randomised content experiments require explicit authorisation")
    from agentrs.actions.schema import Action

    rng = random.Random(seed)
    names = sorted(variants)
    res: dict[str, list[float]] = {n: [] for n in names}
    for t in tasks:
        n = rng.choice(names)
        env.reset(t)
        r = env.step(Action.submit(operation_for(t), variants[n](t)))
        res[n].append(float(bool((r.info.get("content_outcome") or {}).get(outcome_key))))
    out = {"causal": True, "assignment": "randomised", "n": {k: len(v) for k, v in res.items()},
           "mean": {k: (float(np.mean(v)) if v else None) for k, v in res.items()}}
    if len(names) == 2 and all(res.values()):
        a, b = res[names[0]], res[names[1]]
        out["difference"] = float(np.mean(b) - np.mean(a))
        out["difference_of"] = f"{names[1]} - {names[0]}"
        boots = []
        r2 = np.random.default_rng(seed)
        for _ in range(1000):
            boots.append(r2.choice(b, len(b)).mean() - r2.choice(a, len(a)).mean())
        out["ci95"] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    return out
