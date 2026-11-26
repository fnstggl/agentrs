"""Hindsight relabeling of partial successes (meta-reward spec Part 10; manifest AU-07).

A failed attempt at a hard objective often verifiably achieved something else (a deployed service, a first paying
customer, a revenue level below the target). `relabel_episode` checks candidate goals — built with `GoalCompiler`
from the observation at the episode start, i.e. *before* the outcome was known, so the candidate set cannot be
tailored to what happened — against the operator's recorded world observations over the episode, and returns a
`LabelRecord(kind="hindsight_relabel")` for every candidate that was verifiably achieved, with the time it was
first satisfied. The original objective's label is untouched (its failure stays a failure); relabeled labels carry
the original goal id in `meta["original_goal"]`.

Caveat (recorded in every label): hindsight relabeling is not unbiased in stochastic environments (the relabeled
goal was conditioned on the realised trajectory), so relabeled experience trains goal-conditioned evaluators and
skills, not the original objective's value; the learner keeps these labels in a separate stream and reports the
benefit against a no-relabel control (Exp D condition).
"""
from __future__ import annotations

from typing import Sequence

from ..envs.labels import LabelRecord
from .goals import GoalCompiler, SelfGoal, numeric_view


def relabel_episode(env_id: str, episode: int, snapshots: Sequence[tuple], original_goal: str,
                    candidates: Sequence[SelfGoal] | None = None, compiler: GoalCompiler | None = None,
                    max_labels: int = 16) -> list:
    """snapshots: [(t_days, world_observations)] recorded operator-side, in time order (first = episode start)."""
    if not snapshots:
        return []
    t0, obs0 = snapshots[0]
    gc = compiler or GoalCompiler()
    cands = list(candidates) if candidates is not None else gc.candidates(obs0, t0)
    start = numeric_view(obs0)
    out = []
    for g in cands:
        if not g.verifiable or not g.predicates:
            continue
        if all(p.holds(start.get(p.path)) for p in g.predicates):
            continue                                   # already true at the start: trivial, never relabeled
        hit = None
        for t, obs in snapshots[1:]:
            if t - t0 > g.horizon_days + 1e-9:
                break
            nv = numeric_view(obs)
            if all(p.holds(nv.get(p.path)) for p in g.predicates):
                hit = t
                break
        if hit is not None:
            out.append(LabelRecord(env_id=env_id, episode=int(episode), status="success", utility=None, value=None,
                                   t_resolved_days=float(hit - t0), evaluator="operator:hindsight",
                                   goal_id=g.key(), kind="hindsight_relabel",
                                   meta={"original_goal": original_goal, "text": g.text,
                                         "caveat": "conditioned on the realised trajectory (biased)"}))
            if len(out) >= max_labels:
                break
    return out
