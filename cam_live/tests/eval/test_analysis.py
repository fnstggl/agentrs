"""cam_live.eval.analysis on synthetic run records with known answers (fast)."""
from __future__ import annotations

import random

import pytest

from cam_live.eval import analysis


def _run(seed: int, cond: str, progress, rng: random.Random, n_dec: int = 6):
    lts, rows, recs = [], [], []
    for i, p in enumerate(progress):
        oid = f"obj_{seed}_{i}"
        lts.append({"index": i, "objective_id": oid, "quality": p / 2, "progress": p, "achieved": p >= 1.0,
                    "decisions": n_dec, "training_s": 1.0, "inference_s": 2.0, "sim_seconds": 86400.0,
                    "money": 10.0, "conservation": {"closed": True, "conservation_error": 0.0}})
        lr = []
        for t in range(n_dec):
            truth = [rng.uniform(0, 5) for _ in range(4)]
            probs = [0.1, 0.2, 0.3, 0.4]
            k = rng.choices(range(4), probs)[0]
            cands = [{"operation": "lab.act", "arguments": {"channel": j, "effort": 1}, "truth": truth[j],
                      "p_success": 0.5, "value": 0.0, "voi": 0.0, "goal_mean": truth[j] + rng.gauss(0, 0.1),
                      "goal_std": 1.0} for j in range(4)]
            did = f"d{seed}_{i}_{t}"
            lr.append({"decision_id": did, "chosen_index": k, "probabilities": probs, "exploration": "policy",
                       "experiment_id": "", "t": 0.0, "candidates": cands})
            recs.append({"exp_id": did, "objective_id": oid, "seq": i * 100 + t, "operation": "lab.act",
                         "arguments": cands[k]["arguments"], "intrinsic_reward": 0.1, "bootstrap_reward": 0.1,
                         "reward_version": "", "credit": {}, "n_outcomes": 1, "outcome_values": [truth[k]],
                         "lifetime": i})
        rows.append(lr)
    return {"executed": True, "condition": cond, "seed": seed, "family": "F1", "real_backbone": False,
            "backbone": "tiny", "scale": "debug", "lifetimes": lts, "decision_rows": rows, "wall_seconds": 10.0,
            "extract": {"predictions": [], "records": recs, "experiments": [], "credit_events": []}}


def test_baselines_paired_difference_and_nq():
    rng = random.Random(0)
    by = {"E": [_run(s, "E", [0.5, 0.8, 0.9, 1.0], rng) for s in range(6)],
          "D": [_run(s, "D", [0.2, 0.2, 0.3, 0.3], rng) for s in range(6)]}
    b = analysis.baselines(by)
    assert b["conditions"]["E"]["N_q"]["reached"]["k"] == 6 and b["conditions"]["D"]["N_q"]["reached"]["k"] == 0
    d = b["pairwise_vs_E"]["D"]["aulc_ref_minus_c"]
    assert d["mean"] == pytest.approx(0.8 - 0.25) and d["p"] == pytest.approx(2 / 64)
    assert b["conditions"]["E"]["N_q"]["restricted_mean_n"]["mean"] == pytest.approx(18.0)   # 3rd lifetime x 6


def test_shadow_behaviour_policy_is_consistent_and_greedy_on_good_predictions_wins():
    rng = random.Random(1)
    rs = [_run(s, "E", [0.5] * 6, rng, n_dec=30) for s in range(6)]
    sh = analysis.shadow(rs)["targets"]
    beh = sh["behaviour"]
    assert beh["ess_fraction"]["mean"] == pytest.approx(1.0)
    greedy = sh["greedy_predicted_goal_mean"]
    # predictions are truth + small noise: greedy's true value beats the behaviour's, and SNIPS sees it too
    assert greedy["ground_truth_same_contexts"]["mean"] > beh["ground_truth_same_contexts"]["mean"] + 0.5
    assert greedy["snips_estimate_observed"]["mean"] > beh["snips_estimate_observed"]["mean"]
    assert abs(greedy["snips_estimate_observed"]["mean"] - greedy["ground_truth_same_contexts"]["mean"]) < 0.6


def test_action_quality_detects_informative_predictions():
    rng = random.Random(2)
    aq = analysis.action_quality([_run(s, "E", [0.5] * 3, rng) for s in range(4)])
    assert aq["spearman_predicted_vs_true_value_over_candidates"]["mean"] > 0.9
    assert aq["true_value_ratio_of_top_predicted_to_best"]["mean"] > 0.95
