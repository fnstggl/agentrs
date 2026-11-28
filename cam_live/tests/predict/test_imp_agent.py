"""IMP on the real runtime path (debug scale): the integrated agent imagines every candidate's continuation in each
decision, rule ``thompson_policy`` records the distribution with and without imagination, verified multi-step
windows reach the imagination calibration, and a checkpoint restores it exactly; a planner state written before the
imagination calibration existed (the live planning run's checkpoints) still loads.

The agent-side hook is change request docs/change_requests/IMP.md (agent.py is ST's); until it is applied the
agent test is skipped with that reason."""
import inspect
import math
import shutil

import pytest
import torch

from cam_live.agent import CamLiveAgent
from cam_live.contracts import SimClock
from cam_live.eval.conditions import base_config, eval_config
from cam_live.eval.runner import run_lifetime
from cam_live.predict.planner import LatentPlanner
from cam_live.predict.policy import RULE_THOMPSON_POLICY, imagination_calibration

HOOKED = "predictor=self.predictor" in inspect.getsource(CamLiveAgent.evaluate)


def test_old_planner_state_loads_and_reports_a_fresh_calibration(lab_rig):
    rig = lab_rig(n_lifetimes=0)
    planner = LatentPlanner(rig.model, heads=rig.heads)
    # the planner state of the live planning run's checkpoint ckpt_00000036 (written by the code before IMP)
    old = {"allowed": 1, "count": [4, 4, 4, 4], "err": [2.2959, 1.2556, 2.3044, 1.5575], "threshold": 0.5}
    planner.load_state(old)
    assert planner.controller.err == old["err"] and planner.controller.count == old["count"]
    assert planner.controller.value_count == [0, 0, 0, 0] and planner.controller.allowed() == 1
    assert planner.restore_info["imagination_calibration"].startswith("fresh")
    new = planner.state()
    assert {"allowed", "err", "count", "threshold"} <= set(new)          # older loaders read it too
    other = LatentPlanner(rig.model, heads=rig.heads)
    other.load_state(new)
    assert other.calibration is planner.calibration is imagination_calibration(rig.heads.transition)


@pytest.mark.skipif(not HOOKED, reason="change request docs/change_requests/IMP.md (agent.py hook) not applied")
def test_agent_decides_with_imagination_and_learns_its_weight_from_verified_windows(tmp_path):
    torch.set_num_threads(2)
    cfg = eval_config(base_config(real=False, seed=5), "E", seed=5)
    clock = SimClock()
    agent = CamLiveAgent(cfg, str(tmp_path / "agent"), clock=clock)
    try:
        cal = agent.planner.calibration
        assert cal is agent.decider.imagination is imagination_calibration(agent.heads.transition)
        lts = [run_lifetime(agent, "T1", 9_100 + i, str(tmp_path / f"lt{i}"), clock, max_decisions=8, k=16)
               for i in range(2)]
        n_im = 0
        for lt in lts:
            for row in lt["decision_rows"]:
                rec = agent.store.by_decision(row["decision_id"])
                ex = rec.decision.candidates[rec.decision.chosen_index].extras
                assert ex["imagined_horizon_steps"] >= 1 and "im_cont" in ex and ex["im_info_value"] >= 0.0
                if row["exploration"] != RULE_THOMPSON_POLICY:
                    continue
                im = rec.decision.diagnostics["selection"].get("imagination")
                if im is None:                  # the first decisions before any goal-relevant template exists
                    continue
                n_im += 1
                w, wo = im["with_probabilities"], im["without_probabilities"]
                assert abs(sum(w) - 1) < 1e-9 and abs(sum(wo) - 1) < 1e-9 and 0.0 <= im["share_tv"] <= 1.0
                assert im["enabled"] is True and math.isfinite(im["weight"]["mean"])
                assert im["horizon"] == rec.decision.diagnostics["planner_horizon"]
        assert n_im >= 4
        # verified windows of executed decisions reached the calibration (the close-out delivers the outcomes; the
        # next lifetime's measurement reads them)
        assert len(cal) >= 3 and set(k for c in cal.by_h.values() for k in c.keys) <= {
            r.exp_id for lt in lts for r in agent.store.trajectory(lt["objective_id"])}
        meas = agent.last.get("planner") or {}
        assert len(meas.get("value_errors", [])) == agent.planner.controller.max_horizon
        ckpt = agent.save(str(tmp_path / "ckpt"))
        summ = cal.summary()
    finally:
        agent.close()
    shutil.copytree(str(tmp_path / "agent"), str(tmp_path / "agent_b"))
    fresh = CamLiveAgent(cfg, str(tmp_path / "agent_b"), clock=clock)
    try:
        fresh.load(str(tmp_path / "ckpt"))
        assert fresh.checkpoint_id == ckpt and fresh.planner.calibration.summary() == summ
        assert fresh.planner.restore_info["imagination_calibration"] == "restored"
    finally:
        fresh.close()
