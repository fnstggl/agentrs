"""OW-4 Long-horizon adaptation (spec 16): a persistent world with unexpected changes over 90-180 simulated days.

Changes: an evaluator-injected shock at a hidden day (drawn per seed in [30, 60]; never in any agent record) -
a 48-hour outage of the compute provider hosting the most running cores (``provider_outage``), the bankruptcy of
the largest seller (``bankrupt_largest``) or of a whole sector (``bankrupt_sector``) - on top of each family's own
scheduled shocks (preference drift, demand surges, compute shortages, entrants, new needs) and the endogenous
dynamics of firm minds and founders. Measured by `envs.goals.RecoveryObjective`: the verified daily independent
revenue of the agent's organisation; baseline = mean of the 7 days before the shock; success = back at >= 80% of
the baseline on 3 consecutive daily checks after the shock (requires a business of >= $1/day before it). Reports
drop, trough, days to recover and the post-shock area ratio.
"""
from __future__ import annotations

import numpy as np

from .. import goals as G
from .base import COMMON_UNSUPPORTED, TaskProtocol, builder_control, idle_control, inductor_control

DAY = 86400
SHOCKS = ("provider_outage", "bankrupt_largest", "bankrupt_sector")


def make_task(world_family: str, seed: int, horizon_days: int = 120, shock: str | None = None,
              endowment: int = 1_000_000):
    rng = np.random.default_rng([int(seed), 0x0E4])
    kind = shock or SHOCKS[int(rng.integers(0, len(SHOCKS)))]
    spec = {"kind": kind}
    if kind == "provider_outage":
        spec["hours"] = 48
    t = G.RecoveryTask(world={"family": world_family, "seed": int(seed), "overrides": {}, "extra_modules": ["disturb"]},
                       shock=spec, shock_day=float(rng.uniform(30, 60)), horizon_s=horizon_days * DAY,
                       endowment=endowment, min_baseline=100.0)
    return t


PROTOCOL = TaskProtocol(
    name="ow4", title="Long-horizon adaptation: recover verified revenue after a hidden shock", spec_ref="16 OW-4",
    make_task=make_task,
    splits={"train": [("ow_market", 1), ("bazaar", 2)], "val": [("open_tight_compute", 2)],
            "test": [("ow_heldout", 3)]},
    horizon_days=120, thresholds={"recover_frac": 0.8, "sustain_days": 3, "baseline_days": 7},
    controls=[builder_control(), inductor_control(), idle_control()],
    unsupported=COMMON_UNSUPPORTED + [
        "Randomised shock schedules exist only through the evaluator's injection here; Crucible's family shocks are "
        "fixed per family.",
        "'Detects change / revises understanding' is measured behaviourally (recovery); internal belief revision is "
        "reported from CAM's own traces (workstream B/F), not by the environment."],
    arms=["CAM Unified", "CAM without fast adaptation", "CAM without continual updates", "frozen semantic agent"])
