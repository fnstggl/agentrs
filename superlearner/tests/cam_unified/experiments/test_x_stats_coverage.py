"""W9 statistics repair (plan 2026-11-26 §3 W9; audit 'h' Statistics finding): studentised random-effects intervals
over training seeds x evaluation worlds, and paired comparisons whose seeds are independent. The audit's own
simulation design (seed sd 1, world sd 1, cell noise 0.3; true mean 0) is re-run here: the old percentile bootstrap
covered 0.77-0.79 at S=3 and the old unequal-seed paired comparison 0.49 (nominal 0.95)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from superlearner.cam_unified.experiments import stats as ST


def test_student_t_quantiles_and_cdf_match_reference_values():
    for (q, df), want in {(0.975, 1): 12.7062047, (0.975, 2): 4.30265273, (0.975, 5): 2.57058184,
                          (0.975, 10): 2.22813885, (0.95, 3): 2.35336343, (0.995, 30): 2.74999565}.items():
        assert abs(ST.t_ppf(q, df) - want) < 1e-6, (q, df)
    assert abs(ST.t_ppf(0.975, 1e12) - 1.959963985) < 1e-6
    assert abs(ST.t_cdf(-1.0, 1) - 0.25) < 1e-12 and ST.t_cdf(0.0, 3.7) == 0.5
    assert abs(ST.t_cdf(ST.t_ppf(0.9, 2.5), 2.5) - 0.9) < 1e-9 and ST.t_ppf(0.025, 4) == -ST.t_ppf(0.975, 4)


@pytest.mark.parametrize("kw", [dict(S=3, W=16), dict(S=3, W=48, world_sd=0.5), dict(S=3, W=16, seed_sd=0.2),
                                dict(S=5, W=16)])
def test_two_level_t_interval_covers_the_population_mean_at_three_seeds(kw):
    r = ST.coverage_simulation(reps=1500, rng_seed=1, **kw)
    assert r["coverage"] >= 0.93, r


def test_the_old_bootstrap_under_covers_and_stays_available_only_on_request():
    r = ST.coverage_simulation(S=3, W=16, reps=300, method="bootstrap", rng_seed=2)
    assert r["coverage"] < 0.88                    # the audit's finding (0.79), reproduced
    assert ST.two_level(np.ones((3, 4)) + np.arange(4))["method"] == "t"


def test_paired_comparison_with_independent_and_unequal_seeds_covers():
    r = ST.coverage_simulation(S=3, W=16, S_b=1, reps=1500, rng_seed=3)
    assert r["coverage"] >= 0.93, r
    r = ST.coverage_simulation(S=3, W=16, S_b=3, reps=1500, rng_seed=4)
    assert r["coverage"] >= 0.93, r
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (3, 8))
    B = rng.normal(0, 1, (1, 8))
    out = ST.paired(A, B)
    assert out["pairing"] == "per world (independent seeds)" and out["assumptions"]
    zero = ST.paired(A, B, single_seed="zero")
    assert zero["hi"] - zero["lo"] < out["hi"] - out["lo"]          # pooling adds B's seed term
    forced = ST.paired(A, rng.normal(0, 1, (3, 8)), independent_seeds=True)
    assert forced["pairing"].startswith("per world")


def test_interval_width_scales_with_seed_variance_and_one_seed_is_flagged():
    rng = np.random.default_rng(5)
    w = rng.normal(0, 0.3, (1, 16))
    flat = ST.two_level(np.ones((3, 16)) + w)
    spread = ST.two_level(np.ones((3, 16)) + w + np.array([[-2.0], [0.0], [2.0]]))
    assert spread["hi"] - spread["lo"] > 3 * (flat["hi"] - flat["lo"])
    one = ST.two_level(rng.normal(0, 1, (1, 16)))
    assert one["seed_variance_estimable"] is False and one["lo"] < one["mean"] < one["hi"]
    single = ST.two_level(np.array([[2.0]]))
    assert single["mean"] == 2.0 and single["lo"] is None                   # no variance component: no interval
    nan = ST.two_level(np.full((2, 3), np.nan))
    assert nan["mean"] is None


def test_verdict_rule_follows_the_interval_and_direction():
    assert ST.verdict({"lo": 0.1, "hi": 0.5}) == "DEMONSTRATED"
    assert ST.verdict({"lo": -0.1, "hi": 0.5}) == "INCONCLUSIVE"
    assert ST.verdict({"lo": -0.5, "hi": -0.1}) == "FAILED"
    assert ST.verdict({"lo": -0.5, "hi": -0.1}, direction="<") == "DEMONSTRATED"
    assert ST.verdict({"lo": 0.2, "hi": 0.4}, margin=0.25) == "INCONCLUSIVE"
    assert ST.verdict({"lo": None, "hi": None}) == "INCONCLUSIVE"
    with pytest.raises(ValueError):
        ST.verdict({"lo": 0, "hi": 1}, direction="=")


def test_missing_cells_are_skipped_not_imputed():
    M = np.array([[1.0, 2.0, np.nan, 4.0], [1.5, np.nan, 3.0, 4.5], [0.5, 2.5, 3.5, np.nan]])
    out = ST.two_level(M)
    assert out["n_cells"] == 9 and math.isclose(out["mean"], float(np.nanmean(M)))
    assert out["lo"] < out["mean"] < out["hi"]
