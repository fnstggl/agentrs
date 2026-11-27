"""Slow evidence tests: the experiments of ``cam_live.credit.diagnose`` on the real model at debug scale, asserted.
Run with ``-m slow`` (CPU, several minutes); the GPU check on the real configuration needs ``-m gpu``."""
import json
import math

import pytest
import torch

from cam_live.credit import diagnose
from cam_live.credit.providers import COCOA, KEY_CONTRIB, KEY_RTG, RUDDER
from cam_live.nn.api import GROUP_ADAPTERS, GROUP_EXPERIENTIAL, GROUP_HEADS, GROUP_REWARD


@pytest.mark.slow
def test_stored_experience_changes_experiential_weights_and_improves_held_out_prediction(tmp_path):
    torch.manual_seed(0)
    r = diagnose.exp_learning(str(tmp_path))
    print("LEARNING", json.dumps({k: r[k] for k in ("heldout_prediction_nll", "total_weight_delta", "counters")}))
    # triggered by matured independent outcomes; every frequent update changed the experiential group and the heads
    assert r["trigger"]["plan"].startswith("matured_outcomes") and r["first_update"]["trigger"] == r["trigger"]["plan"]
    assert r["first_update"]["hash_changed"] == {GROUP_EXPERIENTIAL: True, GROUP_HEADS: True, GROUP_REWARD: False,
                                                 GROUP_ADAPTERS: False}
    assert r["frequent"]["groups_changed_every_update"] and r["frequent"]["updates_flagged_no_behaviour_change"] == 0
    assert r["total_weight_delta"][GROUP_EXPERIENTIAL] > 0.1 and r["total_weight_delta"][GROUP_HEADS] > 0.1
    assert r["total_weight_delta"][GROUP_REWARD] == 0.0
    assert r["first_update"]["prediction_change"]["world.pred_mean"] > 0
    # ... and the model predicts held-out outcomes (objectives never trained on) better than before
    nll = r["heldout_prediction_nll"]
    assert nll["n"] >= 100 and nll["after"] < nll["before"] and nll["change"] < -nll["se"]
    assert r["heldout_provider_loss"]["after"]["world"] < r["heldout_provider_loss"]["before"]["world"]
    assert r["heldout_summary_after"]["calibration_error"] < r["heldout_summary_before"]["calibration_error"]
    # slow path: LoRA and bridge both get a gradient and move, from experience-grounded losses, and only they move
    s = r["slow_adapters"]
    assert s["applied"] >= 10 and s["only_adapters_changed"] and s["losses"] == ["bridge_grounding", "world"]
    assert min(s["grad_norm_lora"]) > 0 and min(s["grad_norm_bridge"]) > 0
    assert min(s["delta_norm_lora"]) > 0 and min(s["delta_norm_bridge"]) > 0 and s["total_weight_delta"] > 0
    assert all(g["world"] > 0 and g["bridge_grounding"] > 0 for g in s["grad_norm_by_loss"])
    assert r["fast"]["parameters_unchanged"] and r["fast"]["n"] > 0


@pytest.mark.slow
def test_delayed_unlock_is_credited_retrospectively_and_estimators_are_compared_with_randomised_differences(tmp_path):
    torch.manual_seed(0)
    r = diagnose.exp_credit(str(tmp_path))
    print("CREDIT", json.dumps({"experiments": r["experiments"], "selection": r["selection"],
                                "first_decision_mean_credit": r["first_decision_mean_credit"]}))
    t = r["single_lifetime_trace"]
    assert t["first_decision"] == "lab.unlock"
    assert t["learned_credit_keys_before_any_measurement"] == [] and t["unlock_credit_before_measurement"] == {}
    assert t["unlock_credit_after_measurement_objective_open"] == {}            # not final: nothing yet
    after = t["unlock_credit_after_objective_closed"]
    assert after[KEY_RTG] == pytest.approx(t["return_G_discounted"], rel=1e-9) and RUDDER in after and COCOA in after
    assert 0 < t["return_G_discounted"] < t["return_undiscounted"] and t["conservation_error"] < 1e-9
    a = r["assignment"]
    assert a["max_conservation_error"] < 1e-9 and a["matured_objectives"] == r["experience"]["objectives"]
    assert r["links_to_a_later_decision"] == 0 and r["context_leaks"] == 0
    # the unlock decision has no linked outcome: the simple attribution says nothing about it; the learned
    # estimators credit it more than the alternative it was randomised against
    m = r["first_decision_mean_credit"]
    assert m["unlock"]["simple"] is None and m["wait"]["simple"] is None
    assert m["unlock"][KEY_CONTRIB] > m["wait"][KEY_CONTRIB] and m["unlock"][COCOA] > m["wait"][COCOA]
    assert m["unlock"][KEY_RTG] > m["wait"][KEY_RTG]
    assert r["heldout_loss"]["after"][RUDDER] < r["heldout_loss"]["before"][RUDDER]
    assert r["heldout_loss"]["after"][COCOA] < r["heldout_loss"]["before"][COCOA]
    # against the randomised differences (worlds with and without the unlock)
    ex = r["experiments"]
    assert len(ex) == r["config"]["n_experiments"] and all(e["randomized"] for e in ex)
    assert sum(e["measured_difference"] > 0 for e in ex) >= len(ex) - 1
    sc = r["selection"]["scores"]
    assert sc[RUDDER]["n"] == sc[COCOA]["n"] == len(ex) and sc["simple"]["n"] == 0
    assert sc[RUDDER]["sign_agreement"] == 1.0 and sc[COCOA]["sign_agreement"] == 1.0
    assert sc[RUDDER]["rmse"] < 0.75 * r["rmse_of_claiming_no_effect"]       # far better than "no effect" ...
    assert sc[COCOA]["rmse"] < 0.75 * r["rmse_of_claiming_no_effect"]
    assert t["unlock_credit_after_estimators_trained"][KEY_CONTRIB] != after[KEY_CONTRIB]     # revised, and versioned
    assert len({h["version"] for h in t["unlock_credit_history"]}) >= 2
    # whichever estimator becomes the default got there through the gates, and the reasons are recorded
    sel = r["selection"]
    assert sel["default_before"] == "simple" and sel["default_after"] in ("simple", RUDDER, COCOA) and sel["reasons"]
    if sel["default_after"] != "simple":
        assert sc[sel["default_after"]]["passes"]


@pytest.mark.slow
def test_forgetting_is_measured_and_the_retention_mechanism_reduces_it(tmp_path):
    torch.manual_seed(0)
    r = diagnose.exp_retention(str(tmp_path))
    print("RETENTION", json.dumps(r["summary"]))
    s = r["summary"]
    naive, kept = r["arms"]["naive_sequential"], r["arms"]["retention"]
    assert set(naive["families_in_last_batches"]) == {"lab/B"}                   # the naive arm replays only the new family
    assert "lab/A" in kept["families_in_last_batches"] and "retention" in kept["losses_last_update"]
    assert kept["max_experiential_share"] <= 0.9 + 1e-6
    # learning B without protection damages A (measured on held-out A objectives) ...
    assert s["nll_increase_on_A_naive"] > 2 * s["se_naive"] > 0
    # ... and replay + functional regularisation on the anchors reduces that damage
    assert s["nll_increase_on_A_retention"] < 0.5 * s["nll_increase_on_A_naive"]
    assert math.isfinite(s["nll_change_on_B_retention"]) and s["nll_change_on_B_naive"] < 0
    assert kept["retention_metrics"]["lab/A/neg_nll"] > naive["retention_metrics"]["lab/A/neg_nll"]
    assert 0 <= s["decision_agreement_A_naive"] <= 1 and 0 <= s["decision_agreement_A_retention"] <= 1


@pytest.mark.slow
def test_kill_minus_nine_mid_update_resumes_from_the_last_atomic_checkpoint_identically(tmp_path):
    r = diagnose.exp_restart(str(tmp_path))
    print("RESTART", json.dumps({k: {m: v.get(m) for m in ("all_identical", "updates_compared", "resumed_from",
                                                           "victim_killed_by_sigkill")}
                                 for k, v in r.items() if k.startswith("kill")}))
    for mode in ("kill_during_update", "kill_during_checkpoint"):
        x = r[mode]
        assert x["reference_returncode"] == 0 and x["resume_returncode"] == 0, x
        assert x["victim_killed_by_sigkill"] and x["victim_returncode"] == -9
        assert x["checkpoints_identical"] and x["resumed_from"] == x["victim_checkpoint"]
        assert x["recovery"]["weights_clean"] and x["recovery"]["arch_signature_differences"] == []
        assert x["recovery"]["provider_state"]["restored"] == ["retention"]
        assert x["all_identical"] and x["final_hashes_identical"] and x["first_difference"] is None
        assert x["updates_compared"] >= 4 and "slow_adapters" in x["timescales"] and "frequent" in x["timescales"]
        assert x["recovery"]["debris_left"] == []
    assert r["kill_during_checkpoint"]["debris_after_kill"] != []               # the half-written checkpoint was there
    assert r["kill_during_checkpoint"]["recovery"]["recover"]["removed"] != []


@pytest.mark.slow
def test_fresh_process_weights_only_reload_reproduces_predictions_without_experience(tmp_path):
    r = diagnose.exp_export(str(tmp_path))
    print("EXPORT", json.dumps(r))
    assert "error" not in r, r
    assert r["all_bit_exact"] and set(r["bit_exact"]) >= {"b", "z", "world_mean", "action_probs", "rudder_prefix_prediction"}
    assert max(r["max_abs_difference"].values()) == 0.0
    assert r["export_files"] == ["config.json", "weights.safetensors"]
    assert r["experience_available_to_reader"] is False and r["reader_cwd_files"] == []
    assert r["reader_load_clean"] and r["reader_fast_state_updates"] == 0 and r["writer_fast_state_updates"] > 0
    assert all(r["group_hashes_identical"].values()) and r["writer_updates"]["frequent"] > 0


@pytest.mark.gpu
def test_real_configuration_one_frequent_and_one_slow_adapter_update(tmp_path, gpu_heavy):
    """Qwen3-4B + the 50m experiential core, under the shared GPU lock."""
    r = diagnose.exp_real(str(tmp_path))
    print("REAL", json.dumps({k: r[k] for k in ("frequent", "slow_adapters", "parameters")}, default=str))
    assert r["backbone_pretrained"] and r["scale"] == "50m"
    p = r["parameters"]
    assert p["backbone_frozen"] > 3.9e9 and 45e6 < p[GROUP_EXPERIENTIAL] < 60e6 and 10e6 < p[GROUP_ADAPTERS] < 20e6
    f, s = r["frequent"], r["slow_adapters"]
    assert set(f["param_groups"]) == {GROUP_EXPERIENTIAL, GROUP_HEADS}
    assert f["grad_norms"][GROUP_EXPERIENTIAL] > 0 and f["weight_delta"][GROUP_EXPERIENTIAL] > 0
    assert f["grad_norms"][GROUP_HEADS] > 0 and f["weight_delta"][GROUP_HEADS] > 0
    assert f["hash_changed"][GROUP_ADAPTERS] is False and f["prediction_change"]["max_abs"] > 0
    assert math.isfinite(f["losses"]["world"])
    n = s["notes"]
    assert n["adapter_grad_norm"]["lora"] > 0 and n["adapter_grad_norm"]["bridge"] > 0
    if not (n.get("guard") or {}).get("reverted"):
        assert s["param_groups"] == [GROUP_ADAPTERS] and s["weight_delta"][GROUP_ADAPTERS] > 0
        assert n["adapter_delta_norm"]["lora"] > 0 and n["adapter_delta_norm"]["bridge"] > 0
    assert s["hash_changed"][GROUP_EXPERIENTIAL] is False
    assert max(f["peak_gpu_gb"], s["peak_gpu_gb"]) < 24.0
