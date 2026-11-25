"""Stage-1 tasks on the real Crucible OpenWorld (spec Part 21 Stage 1; architecture decisions 2, 10).

* ``open_economy`` - an ``ow_tiny`` world with Crucible's open-economy objective (workstream G's
  `protocols.ow1.open_economy_task`: 10,000 cents of independent revenue and 100 requests within a 3-day window on 2
  consecutive daily checks, 8 days, 3,000,000 cents endowment). Verified reward: `crucible.goals.contracts.
  RewardContract` (potential shaping on the verified metric + success bonus) inside the operator process; sparse.
* ``rig`` - the ``ow_works`` hidden-mechanism production facility inside the full economy world (G's
  `envs.mechworld.RigTask`, `rig.list / take / spec / set / read`): verified dense reward = change of the actor's
  realised facility net income from the facility's own records, in units of the task's public money scale.

Seeds: world seed = ``DEV_SEED0 + index`` = 100,000 + i, i.e. Crucible's *training* seed range (<= TRAIN_SEED_MAX)
at CAM Unified's development offset (experiment register); every task passes workstream G's `SplitGuard` (refuses
held-out families such as ow_heldout and any validation / sealed seed) before a world is created.
"""
from __future__ import annotations

from typing import Optional

DEV_SEED0 = 100_000
TASKS = ("open_economy", "rig")


def make_task(name: str, index: int = 0, *, days: Optional[float] = None, seed: Optional[int] = None):
    """The task object (operator side; picklable for the isolated operator process) and its (family, seed)."""
    s = int(DEV_SEED0 + index if seed is None else seed)
    if name == "open_economy":
        from ..envs.protocols.ow1 import open_economy_task
        t = open_economy_task(seed=s, days=int(days or 8))
        fam = t.world["family"]
    elif name == "rig":
        from ..envs.mechworld import RigTask
        horizon = int(round((days or 4) * 86400))
        t = RigTask(world={"family": "ow_works", "seed": s, "overrides": {}}, endowment=2_000_000,
                    horizon_s=horizon, goal_net_cents=1)
        fam = "ow_works"
    else:
        raise ValueError(f"unknown Stage-1 task {name!r}; one of {TASKS}")
    guard_training_task(fam, s)
    return t, fam, s


def guard_training_task(family: str, seed: int) -> None:
    """Workstream G's held-out guard: raises HeldOutFamilyError for a held-out family or a non-training seed."""
    from ..envs.worldgen import SplitGuard
    SplitGuard().check(family, int(seed), what="Stage-1 training world")


def open_task_world(task, *, store_url: str, seed: int = 0, isolated: bool = True):
    """(adapter, operator handle, evaluator) for one task. ``isolated`` (default, learner runs): worlds, operator and
    evaluator live in a spawned process and only records cross the pipe (G's `open_world_isolated`, the
    hidden-state firewall); otherwise in-process (`open_world` + `OpenWorldEvaluator`; API discipline only)."""
    if isolated:
        from ..envs.openworld import open_world_isolated
        return open_world_isolated([task], evaluator={}, seed=seed, store_url=store_url)
    from ..envs.openworld import open_world
    from ..envs.openworld_eval import OpenWorldEvaluator
    ad, op = open_world([task], seed=seed, store_url=store_url)
    return ad, op, OpenWorldEvaluator(op)
