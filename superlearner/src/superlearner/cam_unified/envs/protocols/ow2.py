"""OW-2 Novel software creation (spec 16): discover an unmet technical need, build an original executable solution,
deploy it, produce independently measured utility and genuine adoption by other participants.

Environment: OpenWorld economies with *generated* need categories (record wire formats, allocation, transforms ...)
whose rules are never given: only public worked examples, format documents and market signals. Verified on
**hidden work instances** - the real requests of independent customers - by Crucible's usefulness module:
``use_independent_requests`` (successful requests to unrelated customers), ``use_verified_quality`` (mean verified
quality of those results) and ``use_retained_customers`` (customers who stayed through the window), all for the
service `$host` of an organisation the agent owns, sustained on 3 consecutive daily checks within 30 days.
The architecture is never supplied; a garbage service (well formed, worthless) must fail.
"""
from __future__ import annotations

from .base import COMMON_UNSUPPORTED, TaskProtocol, builder_control, garbage_control, idle_control, inductor_control

DAY = 86400
THRESHOLDS = {"requests_7d": 300, "quality_7d": 0.9, "retained_7d": 5}


def make_task(world_family: str, seed: int, horizon_days: int = 30, sustain: int = 3, endowment: int = 3_000_000,
              requests: int = THRESHOLDS["requests_7d"], quality: float = THRESHOLDS["quality_7d"],
              retained: int = THRESHOLDS["retained_7d"]):
    from crucible.goals.contracts import GoalContract
    w7 = 7 * DAY
    pred = {"all": [{"metric": "use_independent_requests", "args": {"host": "$host", "window_s": w7}, "op": ">=",
                     "value": int(requests)},
                    {"metric": "use_verified_quality", "args": {"host": "$host", "window_s": w7}, "op": ">=",
                     "value": float(quality)},
                    {"metric": "use_retained_customers", "args": {"host": "$host", "window_s": w7}, "op": ">=",
                     "value": int(retained)}]}
    text = (f"Find a problem other participants need solved, build and deploy software that solves it, and get it "
            f"adopted. Success is verified on a service an organisation you own operates: in the preceding 7 days it "
            f"served at least {int(requests)} successful requests to independent customers, the verified quality of "
            f"those results averaged at least {quality}, and at least {int(retained)} customers stayed with it - on "
            f"{sustain} consecutive daily checks within {horizon_days} days. You start with {endowment} cents.")
    g = GoalContract("ow2_software", text, {"family": world_family, "seed": int(seed), "overrides": {}}, "$self", pred,
                     {"metric": "use_customer_value", "args": {"host": "$host", "window_s": w7},
                      "target": float(requests) * float(quality)}, horizon_days * DAY, sustain, endowment=endowment,
                     tags=("ow2", "outcome_only"))
    g.protocol = "ow2"
    return g


PROTOCOL = TaskProtocol(
    name="ow2", title="Novel software creation with verified utility and independent adoption", spec_ref="16 OW-2",
    make_task=make_task,
    splits={"train": [("ow_frontier", 1), ("ow_market", 1), ("ow_tiny", 3)], "val": [("ow_composite", 1)],
            "test": [("ow_heldout", 1)]},
    horizon_days=30, thresholds=THRESHOLDS,
    controls=[inductor_control(), builder_control(), garbage_control(), idle_control()],
    unsupported=COMMON_UNSUPPORTED + [
        "Adoption is by simulated consumers of need-format categories and by firm minds (capability adoption through "
        "noisy switchback trials); a category nobody demands is never adopted unless the world declares it latent.",
        "Non-privileged feasibility is witnessed only for exemplified categories (ExampleInductor)."],
    arms=["CAM Unified", "frozen semantic agent with tools", "+ persistent memory", "+ matched planning compute"])
