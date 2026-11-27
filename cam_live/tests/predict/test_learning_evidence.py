"""Learning evidence at debug scale in LabWorld (slow; CPU). The full-size numbers come from
``python -m cam_live.predict.diagnose`` (docs/evidence/B/diagnostics.json); these tests run the same code at a
reduced size and assert the invariants and the directions that hold robustly there. Hypotheses that the full run
did not confirm are reported in docs/status/B.json, not asserted here."""
import math

import pytest
import torch

from cam_live.contracts import OutcomeStatus
from cam_live.predict import diagnose as G

pytestmark = pytest.mark.slow


def test_world_model_learns_action_conditioned_counts_at_two_horizons(measure):
    torch.set_num_threads(1)
    res = G.world_vs_marginal(mechanism_seed=7, n_train=24, n_test=10, n_decisions=16, steps=900)
    before, after = res["before_training"], res["after_training"]
    for h in ("6h", "2d"):
        assert after[h]["n_heldout"] >= 40
        assert after[h]["model_nll"] < before[h]["model_nll"] - 0.2          # learned from matured outcomes
        assert after[h]["oracle_nll"] < after[h]["marginal_nll"]             # room for an action-conditional model
        assert after[h]["model_mean_vs_truth_spearman"] > 0.5                # predictions order the actions correctly
        measure(f"slow_world_gain_vs_marginal_{h}", after[h]["gain_vs_marginal"]["mean"])
    assert after["2d"]["mean_count"] > after["6h"]["mean_count"]             # the horizon matters in the data
    assert res["beats_marginal_mean_both_horizons"], after                   # beats the action-independent baseline


@pytest.fixture(scope="module")
def family():
    torch.set_num_threads(1)
    fam = G.train_family(n_train=16, n_decisions=14, steps=150, n_eval_lifetimes=24)
    hp = G.heldout_pass(fam, range(20_000, 20_006))
    return fam, hp


def test_family_training_uses_all_three_providers(family):
    fam, _ = family
    fin = fam["info"]["final"]
    assert {"world_nll_count", "fast_online_nll_count", "evaluator_loss", "transition_nmse"} <= set(fin)
    assert all(math.isfinite(v) for v in fin.values())
    assert 0.0 < fam["info"]["train_success_rate"] < 1.0                     # both achieved and failed objectives
    h0, h1 = fam["twin"].group_hashes(), fam["model"].group_hashes()
    assert h0["experiential"] != h1["experiential"] and h0["heads"] != h1["heads"]
    assert h0["semantic_adapters"] == h1["semantic_adapters"] and h0["backbone_frozen"] == h1["backbone_frozen"]


def test_fast_adaptation_is_the_only_difference_and_needs_no_gradient(family, measure):
    fam, hp = family
    rows = hp["adaptation_rows"]
    assert len(rows) > 30
    k0 = [r for r in rows if r["k"] == 0]
    later = [r for r in rows if r["k"] > 0]
    assert k0 and later
    # before the first observation of a new world the adapted and the non-adapted predictor are the same model
    assert all(abs(r["nll_adapt"] - r["nll_static"]) < 1e-9 for r in k0)
    # afterwards they differ: the belief state changed although no weight did
    assert any(abs(r["nll_adapt"] - r["nll_static"]) > 1e-3 for r in later)
    rep = G.fast_adaptation(rows)
    assert rep["by_k"]["k0"]["nll_gain"]["mean"] == 0.0
    measure("slow_fast_adaptation_nll_gain_after_first_observation", rep["after_first_observation"]["nll_gain"]["mean"])


def test_success_forecasts_are_prospective_and_scored_once(family, measure):
    fam, hp = family
    rep = G.calibration(hp["success"])
    for who in ("trained", "untrained"):
        rows = hp["success"][who]
        assert rep[who]["n"] == len(rows) == 6 * 14 and rep[who]["n_lifetimes"] == 6
        assert all(0.0 < p < 1.0 and y in (0.0, 1.0) for p, y, _ in rows)
    assert abs(rep["untrained"]["brier"]["mean"] - 0.25) < 0.02              # an untrained evaluator says ~0.5
    assert rep["trained"]["mean_p"] != rep["untrained"]["mean_p"]
    measure("slow_brier_untrained", rep["untrained"]["brier"]["mean"])
    measure("slow_brier_trained", rep["trained"]["brier"]["mean"])


def test_action_quality_is_separated_from_luck(family, measure):
    _, hp = family
    rows = hp["quality_rows"]
    assert len(rows) > 15
    for r in rows:                                                           # quality + luck = surprise, exactly
        assert abs(r["quality"] + r["luck"] - r["surprise"]) < 1e-5
    rep = G.quality_vs_outcome(rows)
    assert abs(rep["spearman_raw_outcome_vs_true_luck"]) > 0.2               # raw outcomes carry the luck
    measure("slow_quality_vs_true_luck", rep["spearman_quality_vs_true_luck"])
    measure("slow_outcome_vs_true_luck", rep["spearman_raw_outcome_vs_true_luck"])


def test_value_of_information_fast_path_ignores_the_noisy_gauge(family, measure):
    fam, _ = family
    rep = G.voi_probe_vs_gauge(fam, range(30_000, 30_003), samples=6)
    fast = rep["fast"]
    assert abs(fast["voi_noisy_gauge"]["mean"]) < 1e-12                      # unpredictable but decision-irrelevant
    assert abs(fast["voi_probe"]["mean"]) < 1e-12                            # (and so is the probe: see status)
    assert fast["voi_act"]["mean"] > 0.0                                     # acting on an uncertain channel informs
    assert rep["slow"]["voi_probe"]["mean"] >= 0.0 and rep["slow"]["voi_noisy_gauge"]["mean"] >= 0.0
    measure("slow_voi_probe_minus_gauge_slow_path", rep["slow"]["probe_minus_gauge"]["mean"])


def test_closed_loop_agent_registers_before_acting_and_learns_within_the_lifetime(family):
    fam, _ = family
    for mode in ("greedy", "voi"):
        r = G.run_agent(fam, 40_000, mode, max_decisions=12)
        assert r["decisions"] >= 1 and r["interventions"] <= r["decisions"]
        assert r["adopters"] >= 0 and r["elapsed_s"] > 0
        if r["fast_updates"]:
            assert r["surprise"] > 0 and r["belief_change_kl"] > 0
