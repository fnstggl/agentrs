"""The multi-lifetime LabWorld comparison (slow) and the 50m meta-update cost on the GPU (gpu)."""
import json
import os

import pytest
import torch

from cam_live.reward import diagnose as dg

EVIDENCE = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "evidence", "C")


@pytest.mark.slow
def test_three_learners_with_equal_experience_in_labworld():
    ec = dg.ExperimentConfig(n_lifetimes=24)
    out = dg.labworld_experiment([0, 1], ec, kinds=("bootstrap", "learned", "mixed", "verified_pg"), workers=8,
                                 block=8)
    runs = {(r["kind"], r["seed"]): r for r in out["runs"]}
    assert len(runs) == 8
    for (kind, seed), r in runs.items():
        assert len(r["achieved"]) == ec.n_lifetimes and r["ledger"]["chain_ok"]
        assert r["theta_change_norm"] > 0                         # every learner actually changed its policy
        if kind == "verified_pg":
            assert r["policy_updates"] == 0 and r["ledger"]["entries"] == 0
        else:
            assert r["policy_updates"] >= ec.n_lifetimes and r["ledger"]["entries"] == ec.n_lifetimes
        if kind == "bootstrap":
            assert r["n_meta_applied"] == 0 and set(r["mix"]) == {1.0}
        if kind == "learned":
            assert r["n_meta_applied"] >= 4 and set(r["mix"]) == {0.0}
            assert {m["estimator"] for m in r["meta_updates"]} <= {"on_policy_query", "importance_weighted",
                                                                   "mixed_on_policy_and_importance"}
            assert all(m["n_independent"] >= ec.min_independent for m in r["meta_updates"])
        if kind == "mixed":
            assert r["n_meta_applied"] >= 2 and r["mix"][0] == 1.0
    assert set(out["curves"]) == {"bootstrap", "learned", "mixed", "verified_pg"}
    for name in ("learned_minus_bootstrap", "learned_minus_verified_pg", "mixed_minus_bootstrap"):
        d = out["paired_differences"][name]
        assert d["final_quarter_verdict"] in ("helped", "hurt", "no detectable difference")
        assert d["final_quarter"]["lo"] <= d["final_quarter"]["mean"] <= d["final_quarter"]["hi"]
    assert set(out["reward_pays_for"]["learned"]) == set(dg.CATEGORIES)
    json.dumps(out)                                               # the report is plain JSON


@pytest.mark.gpu
def test_meta_update_cost_at_the_50m_preset_on_the_gpu():
    """Loads Qwen3-4B: ``meta_cost_50m`` takes the shared GPU lock itself (never wrap it in another lock)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from cam_live.nn.gpu import GpuUnavailable
    try:
        out = dg.meta_cost_50m(Ks=(2, 4))
    except GpuUnavailable as e:
        pytest.skip(str(e))
    assert out["preset"] == "50m" and out["backbone_pretrained"] and out["n_theta"] > 1_000_000
    for K in ("2", "4"):
        for mode in ("unroll", "checkpoint"):
            r = out["K"][K][mode]
            assert r["supported"] and r["grad_norm"] > 0 and r["seconds_mean"] < 60 and r["extra_peak_gpu_mb"] < 8192
        assert out["end_to_end_meta_update"][K]["applied"]
    assert out["gpu_peak_total_gb"] < 24 and out["lock_held_seconds"] < 15 * 60
    os.makedirs(EVIDENCE, exist_ok=True)
    out["command"] = "pytest cam_live/tests/reward/test_reward_experiment.py -m gpu"
    with open(os.path.join(EVIDENCE, "meta_cost_50m.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
