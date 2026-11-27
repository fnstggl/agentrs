"""cam_live.eval: the 16.5 checks and the 16.6 attack battery on the integrated agent (CPU, tiny backbone)."""
from __future__ import annotations

import pytest
import torch

from cam_live.eval.experiments import run_condition
from cam_live.eval.integrity import attack_battery
from cam_live.eval.meta_checks import pathway_ablation, run_checks

torch.set_num_threads(2)


def test_meta_checks_on_a_saved_run(tmp_path):
    res = run_condition("E_mix0", "F1", 7, n_lifetimes=2, max_decisions=10, save_dir=str(tmp_path / "run"),
                        world_seed_base=10_700)
    chk = run_checks(str(tmp_path / "run"), res)
    assert "skipped" not in chk, chk
    m0 = chk["1_reward_params_affect_inner_update"]["mix0_differentiated"]
    assert m0["noise_floor_same_psi"] == 0.0                     # the inner update is deterministic
    assert m0["update_norm"] > 0.0
    assert chk["2_inner_update_changes_policy"]["tv_mean"] > 0.0
    six = chk["6_beyond_magnitude"]
    # a pure magnitude change of the reward cannot change the update (returns are standardised); psi_pre vs
    # psi_post is reported as measured (zero when no meta-update was applied)
    assert six["delta_theta_output_scaled_x3_vs_post"] <= 1e-5 * max(1e-12, m0["update_norm"])
    applied = chk["3_4_meta_updates"]["applied_nonzero_changed_version"]
    assert chk["group_changed_pre_to_post"]["reward"] == (applied > 0)
    assert chk["3_4_meta_updates"]["applied_with_fewer_independent_outcomes"] == 0
    assert chk["shadow_checkpoints"].get("n_decisions", 0) > 0


def test_meta_pathway_removed_keeps_psi_bit_identical(tmp_path):
    run_condition("D", "F1", 8, n_lifetimes=2, max_decisions=8, save_dir=str(tmp_path / "d"), world_seed_base=10_800)
    ch = pathway_ablation(str(tmp_path / "d"))
    assert ch["reward"] is False and ch["experiential"] is True


@pytest.mark.slow
def test_attack_battery_detects_every_delivered_attack(tmp_path):
    r = attack_battery(1, str(tmp_path / "att"), max_lifetimes=2, decisions=10)
    det = r["detected"]
    delivered = {k: v for k, v in det.items() if v is not None}
    assert "forged_objective_outcome" in delivered and "duplicate_valid_verdict_x5" in delivered
    assert all(delivered.values()), det
