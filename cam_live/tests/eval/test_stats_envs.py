"""cam_live.eval: statistics, evaluation environments and the ground-truth tap (fast, CPU)."""
from __future__ import annotations

import math
import random

import pytest

from cam_live.contracts import ActionSpec, AuthorizationDecision, AuthVerdict, GoalStatus, OutcomeStatus, SimClock
from cam_live.eval import stats
from cam_live.eval.envs import CODECS, CodecVerifier, CodecWorld, TruthTap, lab_efficiency, make_world, run_codec
from cam_live.eval.runner import run_anchor

ALLOW = AuthorizationDecision(AuthVerdict.ALLOW, "", ["test"], grant="t")


def test_mean_ci_covers_and_degenerates():
    rng = random.Random(1)
    xs = [rng.gauss(2.0, 1.0) for _ in range(200)]
    r = stats.mean_ci(xs)
    assert r["lo"] < 2.0 < r["hi"] and r["hi"] - r["lo"] < 0.4
    assert stats.mean_ci([3.0])["lo"] is None and stats.mean_ci([])["n"] == 0


def test_paired_sign_flip_exact_p():
    r = stats.diff_ci([1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5], paired=True)
    assert r["mean"] == pytest.approx(1.0) and r["p"] == pytest.approx(2 / 64)
    r0 = stats.diff_ci([1, -1, 1, -1], [0, 0, 0, 0], paired=True)
    assert r0["p"] == pytest.approx(1.0)


def test_unpaired_permutation_detects_shift_and_not_noise():
    rng = random.Random(3)
    a = [rng.gauss(1.0, 1.0) for _ in range(40)]
    b = [rng.gauss(0.0, 1.0) for _ in range(40)]
    assert stats.diff_ci(a, b, n_perm=2000)["p"] < 0.01
    c = [rng.gauss(0.0, 1.0) for _ in range(40)]
    assert stats.diff_ci(c, b, n_perm=2000)["p"] > 0.01


def test_n_to_threshold_and_censoring():
    r = stats.n_to_threshold([0.1, 0.6, 0.4, 0.8, 0.9], 0.6, experience=[10, 20, 30, 40, 50], window=2)
    assert r == {"reached": True, "n": 40.0, "index": 3}          # mean(0.4, 0.8) = 0.6 >= q
    r2 = stats.n_to_threshold([0.1, 0.6, 0.4, 0.8, 0.9], 0.61, experience=[10, 20, 30, 40, 50], window=2)
    assert r2 == {"reached": True, "n": 50.0, "index": 4}
    c = stats.n_to_threshold([0.1, 0.2], 0.9, experience=[16, 32])
    assert not c["reached"] and c["n"] == 32.0
    s = stats.summarise_nq([r, c])
    assert s["reached"]["k"] == 1 and s["never_reached"] == 1


def test_speedup_ratio_and_interval():
    init = [{"reached": True, "n": 40.0}] * 6
    exp_ = [{"reached": True, "n": 20.0}] * 6
    s = stats.speedup(init, exp_)
    assert s["S_q"] == pytest.approx(2.0) and s["lo"] == pytest.approx(2.0) and s["reached_experienced"] == 6
    both = stats.speedup([{"reached": False, "n": 64.0}], [{"reached": False, "n": 64.0}])
    assert both["S_q"] == pytest.approx(1.0) and both["units_both_censored"] == 1


def test_wilson_and_calibration_metrics():
    w = stats.wilson(5, 10)
    assert w["lo"] < 0.5 < w["hi"]
    assert stats.brier([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert stats.ece([0.9] * 10, [1] * 9 + [0]) == pytest.approx(0.0, abs=1e-12)
    assert stats.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert stats.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_codec_world_is_real_software_and_verifier_reruns():
    clock = SimClock()
    w = CodecWorld(3, clock)
    v = CodecVerifier(w)
    for op, (arg, levels) in CODECS.items():
        lv = levels[-1]
        assert w.ratio_of(ActionSpec(action_id="x", operation=op, arguments={arg: lv})) == pytest.approx(
            len(w.data) / run_codec(op, lv, w.data))
    t = w.task("obj")
    best = max(((op, lv) for op, (_, levels) in CODECS.items() for lv in levels), key=lambda x: w._ratio(*x))
    worst = min(((op, lv) for op, (_, levels) in CODECS.items() for lv in levels), key=lambda x: w._ratio(*x))
    a = ActionSpec(action_id="a1", operation=worst[0], arguments={CODECS[worst[0]][0]: worst[1]})
    r = w.execute(a, ALLOW)
    assert r.status == "ok" and v.verify(t).status == GoalStatus.PENDING
    b = ActionSpec(action_id="a2", operation=best[0], arguments={CODECS[best[0]][0]: best[1]})
    w.execute(b, ALLOW)
    vo = v.verify(t)
    assert vo.status == GoalStatus.ACHIEVED and v.check_signature(vo)
    vo.value = 99.0
    assert not v.check_signature(vo)
    q = r.expected_outcomes[0]
    clock.advance(3600)
    m = v.measure([q])[0]
    assert m.status == OutcomeStatus.OBSERVED and v.check_measurement(m)
    m.value += 1.0
    assert not v.check_measurement(m)


def test_truth_tap_records_hidden_rate_and_efficiency():
    world, verifier, task = make_world("F1", 10_001, SimClock())
    tap = TruthTap(world)
    a = ActionSpec(action_id="t1", operation="lab.act", arguments={"channel": 1, "effort": 2})
    rate = world._rate(1)
    tap.execute(a, ALLOW)
    row = tap.rows[-1]
    assert row["expected_adopters"] == pytest.approx(2 * rate)
    assert 0.0 < lab_efficiency(tap.rows) <= 1.0 + 1e-12
    tap.execute(ActionSpec(action_id="t2", operation="lab.noise", arguments={}), ALLOW)
    assert "expected_adopters" not in tap.rows[-1]


def test_anchors_bracket_the_competence_scale():
    orc = [run_anchor("oracle", "F1", s, max_decisions=16) for s in range(10_000, 10_008)]
    uni = [run_anchor("uniform", "F1", s, max_decisions=16) for s in range(10_000, 10_008)]
    assert all(r["achieved"] for r in orc)
    assert sum(r["progress"] for r in uni) / 8 < 0.7 < sum(r["progress"] for r in orc) / 8
    t2 = [run_anchor("oracle", "T2", s, max_decisions=12)["first_achieving_decision"] for s in range(70_000, 70_004)]
    assert t2 == [1, 1, 1, 1]
