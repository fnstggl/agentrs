"""Independent evaluation harness.

* Partitions: world families and goal families are tagged train / val / test (worldgen + goals). The locked test
  partition is reachable only through `locked_test_goals(ack=...)`; every access is appended to an access log.
  Nothing under `crucible/goals` or `crucible/agents` imports this module's locked accessors (checked by tests).
* `evaluate(policy_factory, goals, ...)` runs one agent lifetime per goal instance in a fresh world built from the
  goal's world family and seed, with a RewardContract objective, and returns a vector of interpretable measures
  per run: success, time to success, final progress, progress AUC (learning-curve area within the lifetime),
  reward, actions, rejected actions, sandbox compute, money spent, integrity findings, wall time, policy usage
  (e.g. LLM tokens) - plus bootstrap CIs over runs.
* Paired evaluation: candidates are run on identical goal instances (same world seeds) and compared per instance.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from ..agents.base import Policy, Runner
from ..goals.contracts import GOAL_FAMILIES, GoalContract, RewardContract, goal_suite, instantiate

DAY = 86400
LOCKED_ACK = "I am running the final locked evaluation and will not use it for tuning"
ACCESS_LOG = os.environ.get("CRUCIBLE_LOCKED_LOG", os.path.expanduser("~/.crucible/locked_test_access.jsonl"))


@dataclass
class RunResult:
    goal_id: str
    goal_family: str
    world_family: str
    split: str
    level: int
    policy: str
    success: bool
    success_day: float | None
    progress: float
    progress_auc: float
    reward: float
    actions: int
    rejected: int
    sandbox_cpu_ms: float
    spent_cents: int
    integrity: dict
    wall_s: float
    sim_days: float
    policy_usage: dict = field(default_factory=dict)
    world_digest: str = ""


def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_goal(goal: GoalContract, policy: Policy, dt: int = 3600, max_days: float | None = None, handle: str = "agent",
             stop_on_success: bool = True, record: bool = False, others: list | None = None) -> tuple[RunResult, Runner]:
    from .audit import integrity_report
    t0 = time.perf_counter()
    w, actor = instantiate(goal, handle)
    agents = [(actor, policy)] + list(others or [])
    for a, _ in agents[1:]:
        if not w.state.has("principals", a):
            w.system_call("admin.join", {"handle": a.split(":", 1)[1], "endowment": goal.endowment})
    obj = RewardContract(goal)
    run = Runner(w, agents, dt=dt, objectives={actor: obj}, record=record)
    horizon = goal.horizon_s if max_days is None else min(goal.horizon_s, int(max_days * DAY))
    start_cash = _cash(w, actor)
    prog, rew = [], 0.0
    while w.now < horizon:
        run.round()
        rew += 0.0
        st = obj.status(w, actor)
        prog.append(st["progress"])
        if stop_on_success and st["success"]:
            break
    st = obj.status(w, actor)
    traj = run.trajs[actor]
    reward = float(sum(s.reward for s in traj.steps)) if record else float(obj.phi[actor] + (1.0 if st["success"] else 0.0))
    actions = sum(1 for s in traj.steps if s.action) if record else int(w.state.get("principals", actor)["actions"])
    rejected = sum(1 for e in w.log if e.type == "cmd.rejected" and e.payload.get("actor") == actor)
    cpu = sum(o["out"].get("cpu_ms", 0.0) for e in w.log if e.type == "ext" for o in e.payload["outcomes"]
              if o and o.get("kind") == "code.run" and e.actor == actor)
    res = RunResult(goal.id, goal.family, goal.world["family"], goal.split, int(goal.params.get("level", 0)),
                    policy.name, bool(st["success"]),
                    None if st["success_t"] is None else st["success_t"] / DAY, float(st["progress"]),
                    float(np.mean(prog)) if prog else 0.0, reward, actions, rejected, float(cpu),
                    int(max(0, start_cash - _cash(w, actor))), integrity_report(w, actor),
                    time.perf_counter() - t0, w.now / DAY, policy.usage(), w.digest())
    return res, run


def _cash(w, actor):
    from ..goals.metrics import m_cash
    return int(m_cash(w, actor) + sum(m_cash(w, m["org"]) for m in w.state.find("members", "person", actor)
                                       if m["role"] == "owner"))


def evaluate(policy_factory: Callable[[], Policy], goals: list[GoalContract], **kw) -> dict:
    runs = []
    for g in goals:
        r, _ = run_goal(g, policy_factory(), **kw)
        runs.append(r)
    return summarize(runs)


def summarize(runs: list[RunResult]) -> dict:
    succ = [float(r.success) for r in runs]
    prog = [r.progress for r in runs]
    auc = [r.progress_auc for r in runs]
    return {"n": len(runs), "success": bootstrap_ci(succ), "progress": bootstrap_ci(prog), "progress_auc": bootstrap_ci(auc),
            "actions_mean": float(np.mean([r.actions for r in runs])) if runs else 0.0,
            "integrity_flags": sum(1 for r in runs if r.integrity.get("flags")),
            "runs": [asdict(r) for r in runs]}


def paired(a: list[RunResult], b: list[RunResult], key: str = "progress") -> dict:
    by = {r.goal_id: r for r in b}
    d = [getattr(r, key) - getattr(by[r.goal_id], key) for r in a if r.goal_id in by]
    return {"n": len(d), "mean_diff": bootstrap_ci(d)}


# ---------------------------------------------------------------------------- partitions
def train_goals(**kw) -> list[GoalContract]:
    return goal_suite("train", **kw)


def validation_goals(**kw) -> list[GoalContract]:
    return goal_suite("val", **kw)


def locked_test_goals(ack: str, purpose: str, **kw) -> list[GoalContract]:
    """Held-out world generators (structurally novel families, held-out need formats). Logged on every access."""
    if ack != LOCKED_ACK:
        raise PermissionError("locked test access requires the exact acknowledgement string")
    os.makedirs(os.path.dirname(ACCESS_LOG), exist_ok=True)
    with open(ACCESS_LOG, "a") as f:
        f.write(json.dumps({"t": time.time(), "purpose": purpose, "kw": {k: str(v) for k, v in kw.items()}}) + "\n")
    return goal_suite("test", **kw)
