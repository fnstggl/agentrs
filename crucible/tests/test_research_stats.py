"""Statistics used by every learner comparison: checked against cases with known answers."""
import numpy as np
import pytest

from crucible.research import stats as S


def test_mean_ci_covers_the_true_mean_at_roughly_the_stated_rate():
    rng = np.random.default_rng(0)
    hits = 0
    for k in range(300):
        x = rng.normal(2.0, 1.0, 30)
        c = S.mean_ci(x, n_boot=600, seed=k)
        hits += c["lo"] <= 2.0 <= c["hi"]
    assert 0.88 <= hits / 300 <= 0.99       # nominal 0.95; percentile bootstrap undercovers slightly at n=30


def test_mean_ci_flags_small_samples_and_handles_empty_and_nan():
    assert S.mean_ci([1.0, 2.0, 3.0])["low_n"] is True
    assert S.mean_ci([])["mean"] is None
    c = S.mean_ci([1.0, float("nan"), 3.0, 2.0, 2.0, 2.0])
    assert c["n"] == 5 and c["mean"] == pytest.approx(2.0)


def test_wilson_interval_known_values_and_edges():
    c = S.proportion_ci(8, 10)
    assert c["p"] == 0.8 and c["lo"] == pytest.approx(0.4902, abs=2e-3) and c["hi"] == pytest.approx(0.9433, abs=2e-3)
    z = S.proportion_ci(0, 20)
    assert z["lo"] == 0.0 and 0.1 < z["hi"] < 0.2
    assert S.proportion_ci(0, 0)["p"] is None


def test_paired_diff_detects_a_real_difference_and_not_a_null_one():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 5, 40)                       # large between-world variance
    a, b = base + 0.5 + rng.normal(0, 0.2, 40), base + rng.normal(0, 0.2, 40)
    d = S.paired_diff(a, b)
    assert d["lo"] > 0 and d["p_value"] < 0.001 and d["frac_better"] > 0.9
    # the same data unpaired cannot see it: this is why experiments pair on worlds
    u = S.unpaired_diff(a, b)
    assert u["lo"] < 0 < u["hi"]
    null = S.paired_diff(base + rng.normal(0, 0.2, 40), base + rng.normal(0, 0.2, 40))
    assert null["lo"] < 0 < null["hi"] and null["p_value"] > 0.05


def test_paired_permutation_is_exact_for_small_samples():
    d = S.paired_diff([1, 2, 3, 4], [0, 0, 0, 0])
    # all four differences positive: only the 2 sign patterns (all +, all -) are as extreme, out of 16
    assert d["p_value"] == pytest.approx(2 / 16)
    assert d["low_n"] is True


def test_null_p_values_are_roughly_uniform():
    rng = np.random.default_rng(2)
    ps = [S.paired_diff(rng.normal(0, 1, 12), rng.normal(0, 1, 12), n_boot=50, seed=k)["p_value"]
          for k in range(400)]
    assert 0.02 <= np.mean(np.array(ps) < 0.05) <= 0.09


def test_prob_improvement_and_iqm_and_holm():
    assert S.prob_improvement([3, 4, 5], [0, 1, 2]) == 1.0
    assert S.prob_improvement([1, 1], [1, 1]) == 0.5
    assert S.iqm([0, 1, 2, 3, 4, 5, 6, 100]) == pytest.approx(3.5)
    adj = S.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_auc_is_the_mean_height_and_cuts_and_extends_correctly():
    assert S.auc([0, 10], [0, 1]) == pytest.approx(0.5)
    assert S.auc([0, 10], [0, 1], x_max=5) == pytest.approx(0.25)          # cut by interpolation
    assert S.auc([0, 5], [0, 1], x_max=10) == pytest.approx(0.75)          # last value held
    assert S.auc([0, 10], [0, 1], normalise=False) == pytest.approx(5.0)
    assert S.auc([], []) is None


def test_steps_to_threshold_interpolates_and_requires_sustain():
    x, y = [0, 10, 20, 30, 40], [0.0, 0.2, 0.6, 0.3, 0.7]
    assert S.steps_to_threshold(x, y, 0.4) == pytest.approx(15.0)
    assert S.steps_to_threshold(x, y, 0.5, sustain=2) is None               # the dip at 30 breaks the first run
    assert S.steps_to_threshold(x, [0.0, 0.2, 0.6, 0.6, 0.7], 0.5, sustain=2) == pytest.approx(17.5)
    assert S.steps_to_threshold(x, y, 0.9) is None
    assert S.at_budget(x, y, 25) == pytest.approx(0.45)
    assert S.at_budget(x, y, -1) is None


def test_curve_band_does_not_extrapolate_short_runs():
    band = S.curve_band([([0, 10, 20], [0, 1, 2]), ([0, 10], [0, 3])], grid=[0, 10, 20])
    assert band["n"] == [2, 2, 1]
    assert band["mean"][1] == pytest.approx(2.0) and band["mean"][2] == pytest.approx(2.0)
