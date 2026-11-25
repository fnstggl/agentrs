"""OW task protocols: definitions are complete and valid, splits feed the split guard, public goal texts leak
nothing hidden; feasibility calibrations in forks (open economy, OW-7 dev; OW-1 Builder 90 days is slow)."""
from __future__ import annotations

import json
import os

import pytest

import crucible
crucible.load_all_modules()

from superlearner.cam_unified.envs import protocols as P              # noqa: E402
from superlearner.cam_unified.envs.openworld import public_goal       # noqa: E402
from superlearner.cam_unified.envs.worldgen import HeldOutFamilyError, SplitGuard   # noqa: E402

RECORDS = os.path.join(os.path.dirname(P.__file__), "records")


def test_every_protocol_is_complete():
    allp = P.all_protocols()
    assert set(allp) == {"ow1", "ow2", "ow3", "ow4", "ow5", "ow6", "ow7", "open_economy"}
    for name, p in allp.items():
        d = p.describe()
        assert d["controls"] and d["unsupported"] and d["horizon_days"] > 0, name
        assert any("social-media" in u for u in d["unsupported"]), name          # the recorded environment gap
        roles = {c["role"] for c in d["controls"]}
        assert "shortcut" in roles, f"{name}: needs a shortcut control that must fail"
        for split, rows in p.splits.items():
            for fam, seed in rows:
                t = p.make_task(fam, seed)
                assert t.horizon_s > 0 and isinstance(t.world, dict) and t.world["family"] == fam
                if hasattr(t, "validate"):
                    assert t.validate() == [], (name, fam)
                g = public_goal(t, t.horizon_s)
                txt = json.dumps(g.to_dict())
                assert fam not in txt and f'"seed": {seed}' not in txt, (name, fam)
                assert g.text and g.deadline_days == t.horizon_s / 86400


def test_protocol_splits_are_guarded():
    for name, p in P.all_protocols().items():
        g = SplitGuard(extra=p.split_declarations())
        for fam, seed in p.splits.get("train", []):
            g.check_task(p.make_task(fam, seed))
        for split in ("val", "test"):
            for fam, seed in p.splits.get(split, []):
                with pytest.raises(HeldOutFamilyError):
                    g.check_task(p.make_task(fam, seed))
    # OW-5 declares Crucible's 'train' industrial family a held-out target for the transfer experiment
    g5 = SplitGuard(extra=P.load("ow5").split_declarations())
    with pytest.raises(HeldOutFamilyError):
        g5.check("open_industry")


def test_ow1_goal_requires_profit_and_netting():
    p = P.load("ow1")
    t = p.make_task("ow_market", 1)
    leaves = t.success["all"]
    assert leaves[0]["metric"] == "independent_revenue" and leaves[0]["value"] == 1_000_000
    assert leaves[0]["args"]["netting_window_s"] == 90 * 86400      # circular payments netted over the whole run
    assert leaves[1]["metric"] == "cam_operating_profit" and leaves[1]["op"] == ">"
    assert t.sustain_checks == 7 and t.horizon_s == 90 * 86400
    endows = {p.make_task("ow_market", s).endowment for s in range(12)}
    assert len(endows) >= 2                                           # starting capital varies


def test_open_economy_calibration_in_forks():
    rec = P.load("open_economy").calibrate("dev", n_tasks=1)
    t = rec["tasks"][0]
    by = {c["control"]: c for c in t["controls"]}
    assert t["feasible"] and t["fairly_feasible"] and t["shortcuts_succeeded"] == []
    assert by["example_inductor"]["achieved"] and by["example_inductor"]["success_day"] <= 5
    assert by["builder"]["achieved"]
    assert not by["idle"]["achieved"] and not by["random"]["achieved"]
    assert by["idle"]["metrics"]["independent_revenue"] == 0.0


def test_ow7_cooperation_calibration_dev():
    rec = P.load("ow7").calibrate("dev", n_tasks=1)
    t = rec["tasks"][0]
    by = {c["control"]: c for c in t["controls"]}
    assert by["relay_integrator"]["achieved"] and t["fairly_feasible"]
    m = by["relay_integrator"]["metrics"]
    assert m["cam_partner_payments"] >= 500 and m["independent_revenue"] >= 2_000
    assert not by["idle"]["achieved"] and by["idle"]["metrics"]["cam_joint_utility"] == 0.0


@pytest.mark.slow
def test_ow1_builder_calibration_90_days():
    """OW-1 feasibility: the privileged Builder reaches $10k / 30 days of independent revenue with positive operating
    profit, held 7 daily checks, inside 90 simulated days on ow_market seed 1; shortcuts fail. Recorded."""
    rec = P.load("ow1").calibrate("train", n_tasks=1, controls=["builder", "example_inductor", "idle"],
                                  record=os.path.join(RECORDS, "ow1.jsonl"))
    t = rec["tasks"][0]
    by = {c["control"]: c for c in t["controls"]}
    assert by["builder"]["achieved"] and by["builder"]["success_day"] < 90
    assert not by["idle"]["achieved"]
    assert t["feasible"]
