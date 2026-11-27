"""The integrated agent with rule ``thompson_policy`` (change request S-5 / LRN-1): through the real runtime on a
LabWorld objective, every decision records the rule's inputs, the learner feeds the selection calibration with the
verifier-signed outcomes of executed decisions, and a checkpoint restores it exactly."""
import math
import shutil

from cam_live.agent import CamLiveAgent
from cam_live.contracts import SimClock
from cam_live.eval.conditions import base_config, eval_config
from cam_live.eval.runner import run_lifetime
from cam_live.predict.policy import RULE_THOMPSON_POLICY, selection_calibration


def test_agent_selects_with_thompson_policy_and_learns_its_calibration_from_verified_outcomes(tmp_path):
    cfg = eval_config(base_config(real=False, seed=3), "E", seed=3)
    cfg.sections["agent"]["selection_rule"] = RULE_THOMPSON_POLICY
    clock = SimClock()
    agent = CamLiveAgent(cfg, str(tmp_path / "agent"), clock=clock)
    try:
        assert agent.decider.calibration is agent.meta.selection is selection_calibration(agent.heads.policy)
        lt = run_lifetime(agent, "F1", 10_300, str(tmp_path / "lt0"), clock, max_decisions=8, k=16)
        assert lt["decisions"] >= 6
        rules = [r["exploration"] for r in lt["decision_rows"]]
        assert set(rules) <= {RULE_THOMPSON_POLICY, "randomized"} and RULE_THOMPSON_POLICY in rules
        for row in lt["decision_rows"]:
            p = row["probabilities"]
            assert abs(sum(p) - 1.0) < 1e-9 and min(p) >= 0.0
            if row["exploration"] != RULE_THOMPSON_POLICY:
                continue
            rec = agent.store.by_decision(row["decision_id"])
            sel = rec.decision.diagnostics["selection"]
            assert sel["rule"] == RULE_THOMPSON_POLICY and len(sel["value_only_probabilities"]) == len(p)
            assert 0.0 <= sel["policy_share_tv"] <= 1.0 and math.isfinite(sel["calibration"]["beta_q"])
        # the verifier's signed outcomes of the executed decisions (delivered at the close-out) reached the calibration
        cal = agent.meta.selection
        n = len(cal)
        assert n >= 3 and cal.scale is not None and cal.scale > 0
        assert set(cal.keys) <= {r.exp_id for r in agent.store.trajectory(lt["objective_id"])}
        ckpt = agent.save(str(tmp_path / "ckpt"))
        post = cal.posterior()
    finally:
        agent.close()
    shutil.copytree(str(tmp_path / "agent"), str(tmp_path / "agent_b"))      # the run directory (ledger, store)
    fresh = CamLiveAgent(cfg, str(tmp_path / "agent_b"), clock=clock)
    try:
        fresh.load(str(tmp_path / "ckpt"))
        assert fresh.checkpoint_id == ckpt
        assert len(fresh.meta.selection) == n and fresh.meta.selection.posterior() == post
        assert fresh.decider.calibration is fresh.meta.selection
    finally:
        fresh.close()
