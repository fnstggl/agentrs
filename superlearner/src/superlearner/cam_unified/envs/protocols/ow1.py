"""OW-1 Business creation (spec 16): $10,000 per month of independently verified recurring revenue within 90 simulated
days, with positive operating economics and genuine independent customers.

Goal (verified by Crucible's `GoalVerifier` on authoritative state, sustained on 7 consecutive daily checks):
``independent_revenue($org, 30 d, netting over the whole run) >= target`` - related parties, self-purchases,
circular payments (max-flow netting over the run, not just the window), loan principal and payments pulled under a
delegation never count; households (simulated consumers) are independent customers - and
``cam_operating_profit($org, 30 d) > 0`` (`envs.goals`: costs counted group-wide). "Recurring" is the 30-day window
held for 7 consecutive daily checks (Crucible has no subscription notion for need-format services).

Variation: starting capital is drawn per seed from {5k, 10k, 30k dollars}; customer populations, technological
opportunities and competing firms vary with the world family and seed (generated need categories, firm minds,
founders, consumer segment mixes). Threshold ladder in `THRESHOLDS` (including the $100k annualised objective).

Also `OPEN_ECONOMY`: Crucible's smallest open-economy objective on ``ow_tiny`` (dev), which the non-privileged
ExampleInductor reaches on day 4 - the first calibration of the full path.
"""
from __future__ import annotations

import numpy as np

from .. import goals as _goals          # noqa: F401 - registers cam_operating_profit
from .base import COMMON_UNSUPPORTED, TaskProtocol, builder_control, idle_control, inductor_control, random_control

DAY = 86400
THRESHOLDS = {"monthly_10k": 1_000_000, "annualized_100k": 833_334, "monthly_3k": 300_000, "monthly_30k": 3_000_000}


def make_task(world_family: str, seed: int, target_cents: int = THRESHOLDS["monthly_10k"], horizon_days: int = 90,
              sustain: int = 7, endowment: int | None = None, require_profit: bool = True, overrides=None):
    from crucible.goals.contracts import GoalContract
    rng = np.random.default_rng([int(seed), 0x0E1])
    endow = int(endowment if endowment is not None else rng.choice([500_000, 1_000_000, 3_000_000]))
    w30 = 30 * DAY
    leaves = [{"metric": "independent_revenue",
               "args": {"principal": "$org", "window_s": w30, "netting_window_s": horizon_days * DAY},
               "op": ">=", "value": int(target_cents)}]
    if require_profit:
        leaves.append({"metric": "cam_operating_profit", "args": {"principal": "$org", "window_s": w30}, "op": ">",
                       "value": 0})
    text = (f"Create and operate a business. Success is verified on an organisation you own: over the preceding 30 "
            f"days it has earned at least {int(target_cents)} cents from customers unrelated to you"
            + (" while its operating costs stayed below that revenue" if require_profit else "")
            + f", on {sustain} consecutive daily checks, within {horizon_days} days. You start with {endow} cents.")
    g = GoalContract("ow1_business", text, {"family": world_family, "seed": int(seed),
                                            "overrides": dict(overrides or {})}, "$self", {"all": leaves},
                     {"metric": "independent_revenue", "args": {"principal": "$org", "window_s": w30},
                      "target": int(target_cents)}, horizon_days * DAY, sustain, endowment=endow,
                     tags=("ow1", "outcome_only"), params={"target_cents": int(target_cents)})
    g.protocol = "ow1"
    return g


PROTOCOL = TaskProtocol(
    name="ow1", title="Business creation: $10k/month independent recurring revenue in 90 days", spec_ref="16 OW-1",
    make_task=make_task,
    splits={"train": [("ow_market", 1), ("ow_market", 2), ("bazaar", 1), ("ow_frontier", 1)],
            "val": [("open_tight_compute", 1)], "test": [("ow_heldout", 1), ("ow_heldout", 2)]},
    horizon_days=90, thresholds=THRESHOLDS,
    controls=[builder_control(), inductor_control(), idle_control(), random_control()],
    unsupported=COMMON_UNSUPPORTED + [
        "No subscription / repeat-purchase notion for need-format services: 'recurring' = a 30-day window held for 7 "
        "daily checks.",
        "In ow_heldout no non-privileged control exists for non-exemplified categories (only the privileged Builder "
        "witnesses feasibility there)."],
    arms=["CAM Unified (full)", "frozen semantic agent with tools (baselines.FrozenSemanticAgent)",
          "+ persistent memory", "+ matched planning compute", "CAM Unified ablations (Part 19 #8-#14)"])


def open_economy_task(seed: int = 1, days: int = 8):
    from crucible.research.llm_eval import open_economy_goal
    g = open_economy_goal(world_family="ow_tiny", seed=int(seed), overrides={"n_consumers": 800}, days=days,
                          window_days=3, revenue_cents=10_000, requests=100, sustain_checks=2, endowment=3_000_000)
    g.protocol = "open_economy"
    return g


OPEN_ECONOMY = TaskProtocol(
    name="open_economy", title="Open economy smoke objective (ow_tiny, 8 days, 10,000 cents / 3 days + 100 requests)",
    spec_ref="15, 16", make_task=lambda fam, seed: open_economy_task(seed),
    splits={"dev": [("ow_tiny", 1), ("ow_tiny", 2)]}, horizon_days=8,
    thresholds={"revenue_cents_3d": 10_000, "requests_3d": 100},
    controls=[inductor_control(), builder_control(), idle_control(), random_control()],
    unsupported=COMMON_UNSUPPORTED)
