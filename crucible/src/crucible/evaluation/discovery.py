"""Evaluation interface for architecture-discovery systems (outer learners) and any candidate learner.

A `Candidate` is a learner factory plus declared metadata (parameter count, model ids). The evaluator - not the
candidate - builds worlds, runs lifetimes and measures; the candidate cannot see goal verifiers, evaluator seeds or
the locked test partition (this module only uses `train_goals`/`validation_goals`; the locked test requires
`harness.locked_test_goals` with its acknowledgement).

Per candidate and seed the protocol is:
  1. training lifetime(s) on train-family goals (the learner may update itself online; its state persists across
     goals within the seed - continual learning),
  2. transfer: the same learner instance on validation-family goals (structurally different worlds),
  3. retention: back to a train-family goal after the transfer phase.
Measures: initial competence (progress after the first simulated day), within-lifetime learning (progress AUC),
final competence (success, final progress), transfer (val success/progress), retention (train progress after vs
before transfer), and resources (environment actions, wall time, policy-reported usage: tokens, GPU seconds,
FLOPs when measurable; declared parameter count). Comparisons are paired by (goal, seed) and an improvement is only
reported when >= `min_seeds` independent seeds were run and the bootstrap CI excludes 0.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from ..goals.contracts import GoalContract
from .harness import bootstrap_ci, run_goal

DAY = 86400


@dataclass
class Candidate:
    name: str
    make_policy: Callable[[int], object]          # seed -> Policy (a fresh learner)
    params: int | None = None                     # declared trainable parameters
    meta: dict = field(default_factory=dict)


@dataclass
class Budget:
    max_days_per_goal: float = 14.0
    dt: int = 3600
    max_wall_s: float = 3600.0


def _phase(policy, goals, budget, t_end, tag):
    rows = []
    for g in goals:
        if time.time() > t_end:
            rows.append({"goal": g.id, "phase": tag, "skipped": "wall budget"})
            continue
        r, run = run_goal(g, policy, dt=budget.dt, max_days=budget.max_days_per_goal)
        tr = run.trajs[next(iter(run.trajs))] if run.trajs else None
        d = asdict(r)
        d["phase"] = tag
        rows.append(d)
    return rows


def evaluate_candidate(c: Candidate, train: list[GoalContract], val: list[GoalContract], seeds=(0, 1),
                       budget: Budget | None = None) -> dict:
    budget = budget or Budget()
    out = {"candidate": c.name, "params": c.params, "meta": c.meta, "seeds": list(seeds), "rows": []}
    t_end = time.time() + budget.max_wall_s
    for s in seeds:
        pol = c.make_policy(int(s))
        out["rows"] += [dict(r, seed=s) for r in _phase(pol, train, budget, t_end, "train")]
        out["rows"] += [dict(r, seed=s) for r in _phase(pol, val, budget, t_end, "transfer")]
        out["rows"] += [dict(r, seed=s) for r in _phase(pol, train[:1], budget, t_end, "retention")]
    out["summary"] = summarize_candidate(out["rows"])
    return out


def summarize_candidate(rows: list[dict]) -> dict:
    def col(phase, key):
        return [r[key] for r in rows if r.get("phase") == phase and key in r]
    train_first = [r for r in rows if r.get("phase") == "train"]
    first_prog = {}
    for r in train_first:
        first_prog.setdefault((r["seed"], r["goal_id"]), r["progress"])
    ret_before = {r["seed"]: r["progress"] for r in train_first[:1]} if train_first else {}
    return {
        "train_success": bootstrap_ci(col("train", "success")),
        "train_progress_auc": bootstrap_ci(col("train", "progress_auc")),
        "transfer_success": bootstrap_ci(col("transfer", "success")),
        "transfer_progress": bootstrap_ci(col("transfer", "progress")),
        "retention_progress": bootstrap_ci(col("retention", "progress")),
        "actions_mean": float(np.mean(col("train", "actions") + col("transfer", "actions"))) if rows else 0.0,
        "wall_s_total": float(sum(r.get("wall_s", 0.0) for r in rows)),
        "policy_usage": [r.get("policy_usage") for r in rows if r.get("policy_usage")][:3],
    }


def compare(a: dict, b: dict, phase: str = "transfer", key: str = "progress", min_seeds: int = 2) -> dict:
    """Paired comparison of two candidates on identical (goal, seed) instances."""
    ia = {(r["goal_id"], r["seed"]): r for r in a["rows"] if r.get("phase") == phase and key in r}
    ib = {(r["goal_id"], r["seed"]): r for r in b["rows"] if r.get("phase") == phase and key in r}
    common = sorted(set(ia) & set(ib))
    d = [float(ia[k][key]) - float(ib[k][key]) for k in common]
    seeds = len({k[1] for k in common})
    mean, lo, hi = bootstrap_ci(d)
    verdict = "insufficient seeds" if seeds < min_seeds else ("a better" if lo > 0 else ("b better" if hi < 0 else "no resolved difference"))
    return {"phase": phase, "metric": key, "pairs": len(common), "seeds": seeds, "mean_diff": mean, "ci95": [lo, hi],
            "verdict": verdict}
