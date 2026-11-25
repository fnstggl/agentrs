"""Early stopping that keeps the best validation iteration for scikit-learn histogram gradient boosting.

scikit-learn's HistGradientBoosting* stops after `n_iter_no_change` iterations without improvement but keeps every
tree fitted up to the stop, including those after the best validation iteration. For rare outcomes with tiny
Hessians (conversions in Criteo's treated arm, next-day stars of quiet repositories) a Newton step can overshoot,
the validation loss jumps, and the kept ensemble is the diverged one. `truncate_to_best` drops the trees after the
best validation score, which is what "early stopping on validation loss" means in the E2 plans (amendment 1).
"""
from __future__ import annotations

import numpy as np


def best_iteration(est) -> int:
    """Number of trees at the best validation score (validation_score_[0] is the score before any tree)."""
    scores = np.asarray(est.validation_score_, dtype=np.float64)
    if scores.size == 0:
        return int(est.n_iter_)
    return int(np.argmax(scores))       # scores are 'higher is better' (negative loss)


def truncate_to_best(est) -> dict:
    """Truncate a fitted estimator in place to its best validation iteration; returns what was done."""
    n_before = int(est.n_iter_)
    best = best_iteration(est)
    scores = np.asarray(est.validation_score_, dtype=np.float64)
    if best < n_before:
        est._predictors = est._predictors[:best]      # n_iter_ is len(_predictors) in scikit-learn >= 1.x
        if int(est.n_iter_) != best:
            raise RuntimeError("could not truncate the ensemble")
    return {"iterations_fitted": n_before, "best_iteration": best, "iterations_kept": int(est.n_iter_),
            "validation_loss_at_best": float(-scores[best]) if scores.size else None,
            "validation_loss_at_stop": float(-scores[-1]) if scores.size else None}
