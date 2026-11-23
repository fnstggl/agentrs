"""Does success on a task come from information obtained by experiment?

`informed_vs_deprived(tasks)` runs a matched pair of operators on each task:

* both run the *same* experiments on the facility (one lever at a time: low, middle and high for `arm_ticks`
  ticks each, everything else at its default), so both pay the same exploration cost and see the same number of
  ticks;
* both then commit, for the rest of the episode, to the settings their decision rule picks: for each lever, the
  arm with the best mean net in the experiment;
* the **informed** operator applies the rule to what its experiments actually measured; the **deprived** operator
  applies the same rule to the same measurements with the association between arms and outcomes destroyed (the
  outcomes of each lever's three arms are permuted with a seeded shuffle that never leaves them in place).

The only difference between the two is whether the information the experiments revealed reaches the decision.
If the mechanism were irrelevant to success (or the experiments uninformative), the two would do equally well.
Scores are reported for the commit phase and for the whole episode, paired by task.

The operators use nothing but the public surface: lever ranges and their own net per tick.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .stats import mean_ci, paired_diff, proportion_ci
from .tasks import Task


def _derangement(rng: np.random.Generator, n: int) -> np.ndarray:
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def probe_and_commit(task: Task, deprived: bool = False, arm_ticks: int = 3, seed: int = 0) -> dict:
    """One episode of the experiment-then-commit operator on `task` (seat 0). Returns the episode record."""
    sim = task.sim()
    levers = sim.engine.levers(0)
    defaults = {lv.name: lv.default for lv in levers}
    h = task.horizon
    rng = np.random.default_rng([seed, task.world_seed])
    budget = min(h // 2, len(levers) * 3 * arm_ticks)            # never spend more than half the episode probing
    ticks_per_arm = max(1, budget // max(1, len(levers) * 3))
    nets: list[float] = []
    measured: dict[str, list[float]] = {}
    arms: dict[str, list[float]] = {}
    for lv in levers:
        arms[lv.name] = [lv.lo, lv.clip((lv.lo + lv.hi) / 2.0), lv.hi]
        out = []
        for v in arms[lv.name]:
            got = []
            for _ in range(ticks_per_arm):
                r = sim.step({**defaults, lv.name: v})
                got.append(r.revenue[0] - r.cost[0])
            nets.extend(got)
            out.append(float(np.mean(got)))
        measured[lv.name] = out
    probe_ticks = len(nets)
    choice = {}
    for lv in levers:
        m = np.asarray(measured[lv.name])
        if deprived:
            m = m[_derangement(rng, 3)]                           # same numbers, association with the arms destroyed
        choice[lv.name] = arms[lv.name][int(np.argmax(m))]
    commit = []
    while sim.t < h:
        r = sim.step(choice)
        commit.append(r.revenue[0] - r.cost[0])
    nets.extend(commit)
    total = float(np.sum(nets))
    return {"task": task.id, "family": task.family, "engine": task.engine, "deprived": deprived,
            "probe_ticks": probe_ticks, "commit_ticks": len(commit), "net_cents": total,
            "score": task.score(total / h), "success": task.success(total),
            "commit_score": task.score(float(np.mean(commit))) if commit else None,
            "probe_score": task.score(float(np.mean(nets[:probe_ticks]))) if probe_ticks else None,
            "funds_end_cents": sim.funds[0], "choice": choice}


def informed_vs_deprived(tasks: Sequence[Task], arm_ticks: int = 3, seed: int = 0) -> dict:
    rows = []
    for t in tasks:
        a = probe_and_commit(t, False, arm_ticks, seed)
        b = probe_and_commit(t, True, arm_ticks, seed)
        rows.append({"task": t.id, "family": t.family, "engine": t.engine, "informed": a, "deprived": b})
    inf_c = [r["informed"]["commit_score"] for r in rows]
    dep_c = [r["deprived"]["commit_score"] for r in rows]
    inf_e = [r["informed"]["score"] for r in rows]
    dep_e = [r["deprived"]["score"] for r in rows]
    out = {"n_tasks": len(rows), "arm_ticks": arm_ticks,
           "commit_phase": {"informed": mean_ci(inf_c), "deprived": mean_ci(dep_c), "paired": paired_diff(inf_c, dep_c)},
           "whole_episode": {"informed": mean_ci(inf_e), "deprived": mean_ci(dep_e), "paired": paired_diff(inf_e, dep_e)},
           "success": {"informed": proportion_ci(sum(bool(r["informed"]["success"]) for r in rows), len(rows)),
                       "deprived": proportion_ci(sum(bool(r["deprived"]["success"]) for r in rows), len(rows))},
           "probe_cost_equal": all(r["informed"]["probe_score"] == r["deprived"]["probe_score"] for r in rows),
           "by_engine": {}, "rows": rows}
    for eng in sorted({r["engine"] for r in rows}):
        sub = [r for r in rows if r["engine"] == eng]
        out["by_engine"][eng] = {"n": len(sub), "paired_commit": paired_diff(
            [r["informed"]["commit_score"] for r in sub], [r["deprived"]["commit_score"] for r in sub])}
    return out
