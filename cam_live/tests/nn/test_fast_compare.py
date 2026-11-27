"""The fast-adaptation comparison experiment runs end to end and reports every method (NA-13).
The full run is ``python -m cam_live.nn.fast_compare`` -> docs/evidence/A/fast_adaptation.json."""
import json
import math
import os

import pytest
import torch

from cam_live.config import PROJECT_ROOT
from cam_live.contracts import OutcomeKind
from cam_live.nn import fast_compare as fc


def test_task_families_and_scale_protocol():
    g = torch.Generator().manual_seed(0)
    assert not set(fc.TRAIN_FAMILIES) & set(fc.HELDOUT_FAMILIES)
    for fams in (fc.TRAIN_FAMILIES, fc.HELDOUT_FAMILIES):
        d = fc.sample_tasks(fams, 400, 32, OutcomeKind.REAL, g)
        assert d["x"].shape == (400, 32, 2) and d["y"].shape == (400, 32) and torch.isfinite(d["y"]).all()
        # full-data standardisation: over many tasks the targets are zero-mean / unit-variance, whatever the raw units
        assert abs(float(d["y"].mean())) < 0.05 and abs(float(d["y"].std()) - 1.0) < 0.1
        c = fc.sample_tasks(fams, 200, 32, OutcomeKind.COUNT, g)
        assert bool((c["y"] >= 0).all()) and bool((c["y"] == c["y"].round()).all())
        assert float(c["exposure"].min()) >= 0.5 and float(c["exposure"].max()) <= 4.0


def test_comparison_runs_all_methods_on_both_targets():
    res = fc.run(steps=6, seeds=(0,), batch=8, T=12, n_eval=24, log=lambda s: None)
    assert set(res["targets"]) == {"gaussian", "count"}
    for t in res["targets"].values():
        seed0 = t["per_seed"][0]
        assert set(seed0["methods"]) == {"learned_prior_bayes", "blr_raw", "gru", "running_stats"}
        assert seed0["stateful_vs_functional_max_abs_nll_diff"] < 1e-6     # the agent's stateful path is the same maths
        for split in ("in_distribution", "held_out_families"):
            s = t["summary"][split]
            assert s["winner"] == s["ranking_by_nll"][0] and len(s["ranking_by_nll"]) == 4
            assert set(s["learned_prior_minus_baseline"]) == {"blr_raw", "gru", "running_stats"}
            for m in seed0["methods"].values():
                assert math.isfinite(m[split]["mean_nll_k_ge_1"]) and "0" in m[split]["per_support_size"]
    json.dumps(res)                                                       # serialisable as written


def test_recorded_result_is_complete_and_states_its_verdict():
    path = os.path.join(PROJECT_ROOT, "docs", "evidence", "A", "fast_adaptation.json")
    if not os.path.exists(path):
        pytest.skip("run python -m cam_live.nn.fast_compare first")
    res = json.load(open(path))
    assert res["protocol"]["heldout_families"] == list(fc.HELDOUT_FAMILIES) and res["protocol"]["meta_train_steps"] >= 1000
    assert "support-estimated" in res["protocol"]["gaussian_scale"]
    for name in ("gaussian", "count"):
        s = res["targets"][name]["summary"]["held_out_families"]
        assert s["learned_prior_wins"] == (s["winner"] == "learned_prior_bayes")
        assert ("WINS" if s["learned_prior_wins"] else "LOSES") in res["verdict"][name]
