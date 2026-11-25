"""OW-6 Self-directed lifetime learning (spec 16): a persistent world with no fixed external task.

The agent may pursue any project and register its own goals (``goals.propose``: private predicates over its own
observable metrics, checked by the kernel at their due time). Operator side, `envs.goals.SelfDirectedObjective`
screens every self-goal with `AntiTrivialFilter` (already true, immaterial gain, reachable by an idle fork, repeats,
non-own metrics are rejected) and records the kernel's met/missed verdicts; intrinsic reward for accepted, verified
self-goals is off by default (``self_goal_bonus``) and, when enabled, reported separately. External competence is
probed every 10 days on held-out objectives the agent is never given (Crucible procedural goal families revenue /
service / solvency, bound to the agent's own organisations): the question is whether self-directed experience
raises those probes. CAM's internal learning-progress objective (D01) is the agent-side driver; this protocol
supplies the independent measurement.
"""
from __future__ import annotations

from .. import goals as G
from .base import COMMON_UNSUPPORTED, TaskProtocol, idle_control, inductor_control

DAY = 86400


def probe_goals(world_family: str, seed: int) -> list:
    from crucible.goals.contracts import generate_goal
    out = []
    for fam, lvl in (("revenue", 1), ("service", 0), ("solvency", 0)):
        try:
            out.append(generate_goal(fam, world_family, int(seed), lvl, split="test"))
        except Exception:  # noqa: BLE001 - a family without a feasible draw for this world: skip it
            continue
    return out


def make_task(world_family: str, seed: int, horizon_days: int = 90, endowment: int = 3_000_000,
              self_goal_bonus: float = 0.0):
    t = G.SelfDirectedTask(world={"family": world_family, "seed": int(seed), "overrides": {}},
                           horizon_s=horizon_days * DAY, endowment=endowment,
                           probe_goals=probe_goals(world_family, seed), probe_every_s=10 * DAY,
                           self_goal_bonus=self_goal_bonus)
    return t


PROTOCOL = TaskProtocol(
    name="ow6", title="Self-directed lifetime learning with anti-trivial self-goals and held-out probes",
    spec_ref="16 OW-6", make_task=make_task,
    splits={"train": [("ow_market", 3)], "val": [("open_tight_compute", 3)], "test": [("ow_heldout", 4)]},
    horizon_days=90, thresholds={"probe_every_days": 10},
    controls=[inductor_control(), idle_control()],
    unsupported=COMMON_UNSUPPORTED + [
        "Self-goals may only use the kernel's agent-observable metrics (cash, independent revenue, service requests "
        "and availability, repository commits, independent members, organisation status)."],
    arms=["CAM Unified self-directed", "CAM with external tasks only", "frozen semantic agent"])
