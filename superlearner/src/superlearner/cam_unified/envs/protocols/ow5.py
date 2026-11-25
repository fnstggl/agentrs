"""OW-5 Cross-domain transfer (spec 16): learn through digital-business experience, then acquire unfamiliar
industrial and technical capabilities.

Source domain (training streams): digital OpenWorld economies (ow_market, ow_frontier, bazaar, open_digital) and the
procedural mechanism catalogue's training families. Target domains, declared held out for this experiment (the
split guard refuses them in source training even where Crucible tags a family 'train', e.g. open_industry):

* industrial OpenWorlds (multi-tier supply chains, energy grids, freight, perishable stock): open_industry,
  open_scarce_energy, open_perishable - goal: positive operating profit and >= $3,000 independent revenue over 30
  days, held 3 daily checks, within 60 days;
* technical: the held-out engineering lab (OW-3 task on engineering_lab_heldout);
* hidden-mechanism queueing systems with hidden gauges (``cw1_tech_*``, cross-implementation) through
  `MechWorldAdapter`.

Measured: acquisition curves (verified return vs. target experience) and zero-shot results for the arms below at
matched target budgets; checkpoint semantics (weights only vs weights + optimiser vs + replay) come from
workstream E's `CheckpointOpsP` so the CAM 7.0 negative-transfer confound is controlled.
"""
from __future__ import annotations

from .. import goals as _goals          # noqa: F401 - registers cam_operating_profit
from . import ow3
from .base import COMMON_UNSUPPORTED, Control, TaskProtocol, builder_control, idle_control, random_control

DAY = 86400
SOURCE_FAMILIES = ("ow_market", "ow_frontier", "bazaar", "open_digital")
MECH_TARGETS = ("cw1_tech_pipeline_hidequeue", "cw1_tech_cascade_hideerr", "cw1_tech_lostwork_hideshed")
THRESHOLDS = {"industrial_revenue_30d": 300_000}


def make_task(world_family: str, seed: int, horizon_days: int = 60, sustain: int = 3, endowment: int = 3_000_000,
              target: int = THRESHOLDS["industrial_revenue_30d"]):
    if world_family.startswith("engineering_lab"):
        return ow3.make_task(world_family, seed)
    from crucible.goals.contracts import GoalContract
    w30 = 30 * DAY
    pred = {"all": [{"metric": "independent_revenue", "args": {"principal": "$org", "window_s": w30}, "op": ">=",
                     "value": int(target)},
                    {"metric": "cam_operating_profit", "args": {"principal": "$org", "window_s": w30}, "op": ">",
                     "value": 0}]}
    text = (f"Build a profitable business in this economy. Success is verified on an organisation you own: over the "
            f"preceding 30 days it earned at least {int(target)} cents from independent customers with operating "
            f"costs below that revenue, on {sustain} consecutive daily checks within {horizon_days} days. You start "
            f"with {endowment} cents.")
    g = GoalContract("ow5_industrial", text, {"family": world_family, "seed": int(seed), "overrides": {}}, "$self",
                     pred, {"metric": "independent_revenue", "args": {"principal": "$org", "window_s": w30},
                            "target": int(target)}, horizon_days * DAY, sustain, endowment=endowment,
                     tags=("ow5", "outcome_only"))
    g.protocol = "ow5"
    return g


PROTOCOL = TaskProtocol(
    name="ow5", title="Cross-domain transfer: digital business -> industrial / technical", spec_ref="16 OW-5",
    make_task=make_task,
    splits={"train": [(f, 1) for f in SOURCE_FAMILIES],
            "val": [("open_scarce_energy", 1)],
            "test": [("open_industry", 1), ("open_perishable", 1), ("engineering_lab_heldout", 11)]},
    horizon_days=60, thresholds=THRESHOLDS,
    controls=[builder_control(), idle_control(), random_control()],
    unsupported=COMMON_UNSUPPORTED + [
        "No scripted industrial entrepreneur exists in Crucible: industrial-target feasibility is NOT witnessed by a "
        "competent control (the need-format Builder is expected to fail there: industrial worlds have no need-format "
        "demand). Thresholds for industrial targets are therefore provisional until a control exists.",
        "Crucible's goal suite excludes industrial families (is_economic=False); the goal here is hand-built."],
    notes=f"Mechanism targets via MechWorldAdapter: {', '.join(MECH_TARGETS)} (cross-implementation, hidden gauges).",
    arms=["pretrained CAM (full lifetime)", "weights-only transferred CAM", "fresh CAM", "frozen semantic agent",
          "CAM 7.0 / model_based on mechanism targets"])
