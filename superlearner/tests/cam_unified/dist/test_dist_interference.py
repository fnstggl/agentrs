"""Retention and cross-family interference measurement on the real pipeline (manifest F04, spec 14.5), and the
dashboard's retention/transfer section rendered from the evaluation rows such a run writes (manifest A20)."""
from __future__ import annotations

import json
import os

from superlearner.cam_unified.dist.interference import run_condition
from superlearner.cam_unified.observe.dashboard import build_dashboard


def test_sequential_families_measure_retention_and_feed_the_dashboard(tmp_path):
    out = str(tmp_path / "interf")
    r = run_condition(out, "sequential", seed=0, n=12, eval_every=4, keep_run=True)
    assert r["trained_by_family"] == {0: 12 * 256, 1: 12 * 256}      # exact schedule: 12 updates per family
    c0, c1 = dict(r["curves"]["mech0"]), dict(r["curves"]["mech1"])
    assert sorted(c0) == [0, 4, 8, 12, 16, 20, 24]
    assert c0[12] > c0[0] + 0.25                                      # phase 1 learned family 0
    assert all(0.0 <= v <= 1.0 for v in list(c0.values()) + list(c1.values()))
    # (how much of family 0 survives phase 2, and how well family 1 is then learned, are the experiment's results:
    # see dist/interference.py measure(); on the toy, phase 2 forgets part of family 0 and learns family 1 slowly)
    rows = [json.loads(l) for l in open(os.path.join(r["run_dir"], "eval", "heldout.jsonl"))]
    assert {x["family"] for x in rows} == {"mech0", "mech1"} and len(rows) == 14
    assert all(x["metric"] == "heldout_payout" and x["condition"] == "sequential" for x in rows)
    src = open(build_dashboard(r["run_dir"])).read()
    assert "No retention or transfer evaluations" not in src
    assert "heldout_payout" in src and "mech0" in src and "mech1" in src


def test_mixed_schedule_alternates_families_into_one_learner(tmp_path):
    r = run_condition(str(tmp_path / "interf"), "mixed", seed=0, n=25, eval_every=25)
    assert r["trained_by_family"] == {0: 25 * 256, 1: 25 * 256}
    c0, c1 = dict(r["curves"]["mech0"]), dict(r["curves"]["mech1"])
    assert c0[50] > 0.8 and c1[50] > 0.8                              # one set of weights learned both families
    assert c0[0] < 0.35 and c1[0] < 0.35
