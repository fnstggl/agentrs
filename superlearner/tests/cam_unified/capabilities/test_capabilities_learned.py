"""Learned capabilities through the integrated agent (workstream Y; slow: full training protocols on CPU).

Each test re-runs a protocol of `cam_unified.capabilities` (the same function and settings that produced the
committed numbers in results/, seed 0) and asserts the capability claim. Claims the measurements did not support
are marked xfail with the measured diagnosis (they are requirements, not weakened tests: an XPASS is reported).
Run with: pytest -m slow superlearner/tests/cam_unified/capabilities/test_capabilities_learned.py
"""
from __future__ import annotations

import pytest

from superlearner.cam_unified.capabilities.beliefs import BeliefSettings, run_beliefs
from superlearner.cam_unified.capabilities.causal import CausalSettings, run_causal
from superlearner.cam_unified.capabilities.exploration import ExplorationSettings, run_exploration
from superlearner.cam_unified.capabilities.imagination import ImaginationSettings, run_imagination
from superlearner.cam_unified.capabilities.memory import MemorySettings, run_memory
from superlearner.cam_unified.capabilities.reasoning import ReasoningSettings, run_reasoning

DEV = "cpu"


@pytest.fixture(scope="module")
def beliefs_res(semantic_ok, tmp_path_factory):
    return run_beliefs(BeliefSettings(seed=0), root=str(tmp_path_factory.mktemp("bel")), device=DEV)


@pytest.mark.slow
def test_a05_trained_agent_adapts_its_predictive_distribution(beliefs_res):
    s = beliefs_res["summary"]
    for m in ("low", "high"):
        assert s[m]["abs_err_last"] < 0.2 and s[m]["abs_err_k0"] > 0.8, s[m]
        assert abs(s[m]["belief_effect_full_last"]) > 0.8, s[m]          # caused by the fast state
        assert s[m]["pred_var_last"] < s[m]["max_pred_var"]
    assert s["familiar_expansions_total"] == 0 and s["extreme"]["expansions_total"] > 0
    assert s["extreme"]["var_ratio_vs_familiar"] > 3.0


@pytest.mark.slow
@pytest.mark.xfail(reason="measured (results/beliefs_s0.json): particle-conditioned reward predictions differ by "
                          "~0.006 across particles against a mechanism gap of 2.0, ESS stays 7.4-7.7 of 8, the particle "
                          "mixture moves the prediction by -0.02 while the Kalman layer moves it by +-0.96. The FiLM "
                          "particle conditioning of C's OutcomeHead starts at exactly zero (every particle predicts the "
                          "same at init: spread 0.000). With film_init_std=0.2 (beliefs_film02_s0.json) the spread "
                          "starts at 0.06 and shrinks to 0.026; mass on the active mechanism 0.74 / 0.20. For a "
                          "Bernoulli mechanism (no Kalman channel; beliefs_bern_s0.json) P(ok) stays 0.48 for p=0.2 "
                          "and p=0.8 after 7 observations: no fast adaptation at all", strict=False)
def test_a06_particle_weights_concentrate_on_the_active_mechanism(beliefs_res):
    s = beliefs_res["summary"]
    assert s["particle_pred_spread_last"] > 0.25 * s["mechanism_gap"]
    for m in ("low", "high"):
        assert s[m]["mass_on_active_last"] > 0.9 and s[m]["ess_last"] < 0.75 * s[m]["ess_k0"], s[m]


@pytest.mark.slow
@pytest.mark.xfail(reason="measured (results/reasoning_adaptive_s0.json): PPO through the full agent did not learn the "
                          "chain task in 200 updates (accuracy 0.27, chance 0.25) and forcing J in {1..16} changed 0% "
                          "of the trained agent's greedy decisions", strict=False)
def test_a03_more_reasoning_iterations_are_needed_and_used(semantic_ok):
    r = run_reasoning(ReasoningSettings(seed=0), device=DEV)
    assert r["trained"]["per_hop"]["2"]["acc"] > 0.6
    assert r["forced_iters"]["1"]["per_hop"]["2"]["acc"] < r["trained"]["per_hop"]["2"]["acc"] - 0.2


@pytest.mark.slow
@pytest.mark.xfail(reason="measured (results/exploration_lp_s0.json, exploration_ext_s0.json, "
                          "exploration_lp_film02_s0.json): every condition collapses to play_safe within 80 rounds "
                          "(P(test first) < 1e-4, identical decisions with and without LP); the hindsight learning "
                          "progress credited to the informative test is +0.0002 nats: the belief cannot carry the "
                          "revealed state (degenerate particles; Kalman channels are per outcome key, so the test "
                          "signal never informs the reward prediction; LP holds Z_hat fixed)", strict=False)
def test_a09_learning_progress_agent_learns_to_test_when_informative(semantic_ok, tmp_path):
    lp = run_exploration(ExplorationSettings(seed=0, condition="lp", update_every=2, checkpoints=(80, 160, 240),
                                             train_rounds=240), root=str(tmp_path), device=DEV)
    ext = run_exploration(ExplorationSettings(seed=0, condition="extrinsic", update_every=2,
                                              checkpoints=(80, 160, 240), train_rounds=240), root=str(tmp_path),
                          device=DEV)
    a, b = lp["evals"]["240"], ext["evals"]["240"]
    assert a["informative"]["p_test_first"] > 0.5 > b["informative"]["p_test_first"] + 0.2
    assert a["uninformative"]["p_test_first"] < 0.2


@pytest.mark.slow
@pytest.mark.xfail(reason="measured (results/imagination_s0.json, imagination_es_s0.json): the policy collapses to "
                          "play_safe by round 80, after which the planner's candidates (sampled from that policy) never "
                          "contain the test (coverage 0.0); with exploring starts the world model still predicts the "
                          "same launch reward after a positive and a negative test signal (-0.60 vs -0.61, truth +10 / "
                          "-10) and the policy launches with p ~ 3e-6 after a positive signal", strict=False)
def test_c05_planner_chooses_the_costly_test_and_ablations_do_not(semantic_ok, tmp_path):
    r = run_imagination(ImaginationSettings(seed=0, exploring_starts=True, update_every=2,
                                            checkpoints=(80, 160, 240), train_rounds=240), root=str(tmp_path),
                        device=DEV)
    e = r["evals"]["240"]
    assert e["planner"]["frac"]["run_test"] > 0.5
    assert e["planner"]["frac"]["run_test"] > e["policy"]["frac"]["run_test"] + 0.3
    assert e["planner_frozen_beliefs"]["frac"]["run_test"] < e["planner"]["frac"]["run_test"] - 0.3


@pytest.mark.slow
def test_c03_causal_model_tests_through_the_integrated_world_model(semantic_ok, tmp_path):
    r = run_causal(CausalSettings(seed=0), root=str(tmp_path), device=DEV)
    d, i, s = r["worlds"]["delay"], r["worlds"]["interaction"], r["worlds"]["switch"]
    assert d["t1_actions_differ"]["mean_abs_pairwise_diff"] > 0.3
    assert d["t7_imagination_isolated"]["env_and_archive_unchanged"]
    assert d["t7_imagination_isolated"]["fast_state_unchanged"]
    assert s["t6_intermediate_event"]["later_pred_difference"] > 0.2
    assert s["t6_intermediate_event"]["real_fast_state_untouched"]


@pytest.mark.slow
def test_e03_memory_pipeline_retrieves_the_cue_through_the_complete_architecture(semantic_ok, tmp_path):
    r = run_memory(MemorySettings(seed=0, condition="memory_episode"), root=str(tmp_path), device=DEV)
    assert r["trained"]["retrieval_hit_rate"] == 1.0
