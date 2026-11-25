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
        g = SplitGuard(extra=p.split_declarations(), extra_pairs=p.pair_declarations())
        for fam, seed in p.splits.get("train", []):
            t = p.make_task(fam, seed)
            assert getattr(t, "split", "train") == "train"
            g.check_task(t)
        for split in ("val", "test"):
            for fam, seed in p.splits.get(split, []):
                t = p.make_task(fam, seed)
                assert t.split == split                       # the declared split travels with the task
                with pytest.raises(HeldOutFamilyError):
                    g.check_task(t)
                with pytest.raises(HeldOutFamilyError):         # even a guard that knows nothing of the protocol
                    SplitGuard(allow_unknown=True).check_task(t)
    # OW-5 declares Crucible's 'train' industrial family a held-out target for the transfer experiment
    g5 = SplitGuard(extra=P.load("ow5").split_declarations())
    with pytest.raises(HeldOutFamilyError):
        g5.check("open_industry")


def test_tasks_carry_their_seat_and_participants():
    """OW-3 needs the engineer seat and OW-7 the partner: a task made by the protocol is feasible through plain
    `open_world` and through `TaskProtocol.open`, without knowing the protocol's internals."""
    from superlearner.cam_unified.envs.openworld import open_world
    p3 = P.load("ow3")
    t3 = p3.make_task(*p3.splits["train"][0])
    assert t3.actor_handle == "eng"
    ad, op, ev = p3.open("train", 0)
    try:
        ad.reset()
        assert op.principal(0) == "p:eng"
        w = op.world(0)
        assert [m["title"] for m in w.state.find("members", "person", "p:eng")] == ["network engineer"]   # the seat
    finally:
        ad.close()
    # the learner-run path: the same task in another process, seen only through records
    ad, rop, rev = p3.open("train", 0, isolated=True)
    try:
        b = ad.reset()
        me = [o for o in b.observations[0] if o.meta.get("self")]
        assert me[0].entity == "p:eng" and rop.pid != __import__("os").getpid()
        assert rev.evaluate_goal(0).achieved is False
    finally:
        ad.close()
    p7 = P.load("ow7")
    t7 = p7.make_task("ow_tiny", 1)
    assert callable(t7.participant_factory)
    ad, op = open_world([t7], seed=0)
    try:
        ad.reset()
        assert op.world(0).principal("p:partner-01") is not None and op.principal(0) != "p:partner-01"
        assert [x[0] for x in op.slots[0].participants] == ["p:partner-01", "p:bystander-02"]
    finally:
        ad.close()
    # the operator's explicit lists still win
    ad, op = open_world([t7], seed=0, participants=[("other-09", __import__("crucible.agents.scripted",
                                                                          fromlist=["IdlePolicy"]).IdlePolicy(), 1)])
    try:
        ad.reset()
        assert [x[0] for x in op.slots[0].participants] == ["p:other-09"]
    finally:
        ad.close()


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
    """The composed (partner-dependent) service succeeds; an own business plus token payments to the partner - the
    earlier definition's loophole - and idling fail."""
    rec = P.load("ow7").calibrate("dev", n_tasks=1)
    t = rec["tasks"][0]
    by = {c["control"]: c for c in t["controls"]}
    assert by["relay_integrator"]["achieved"] and t["fairly_feasible"] and t["feasible"]
    m = by["relay_integrator"]["metrics"]
    assert m["cam_partner_payments"] >= 500 and m["independent_revenue"] >= 2_000
    assert m["cam_joint_utility"] >= 30_000 and m["cam_partner_dependence"] >= 0.5
    s = by["pay_partner_shortcut"]
    assert not s["achieved"] and t["shortcuts_succeeded"] == []
    # it did what the loophole needed (own independent revenue >> the joint target, payments >= the floor) ...
    assert s["metrics"]["independent_revenue"] >= 30_000 and s["metrics"]["cam_partner_payments"] >= 500
    # ... and produced no joint value
    assert s["metrics"]["cam_joint_utility"] == 0.0
    assert not by["idle"]["achieved"] and by["idle"]["metrics"]["cam_joint_utility"] == 0.0


@pytest.mark.slow
def test_ow1_builder_calibration_90_days(tmp_path):
    """OW-1 feasibility: the privileged Builder reaches $10k / 30 days of independent revenue with positive operating
    profit, held 7 daily checks, inside 90 simulated days on ow_market seed 1; shortcuts fail. The test writes its
    record to a temporary file; committed records come only from the calibrate CLI."""
    out = str(tmp_path / "ow1.jsonl")
    rec = P.load("ow1").calibrate("train", n_tasks=1, controls=["builder", "example_inductor", "idle"], record=out)
    assert json.loads(open(out).read().splitlines()[-1])["protocol"] == "ow1"
    t = rec["tasks"][0]
    by = {c["control"]: c for c in t["controls"]}
    assert by["builder"]["achieved"] and by["builder"]["success_day"] < 90
    assert not by["idle"]["achieved"]
    assert t["feasible"]
