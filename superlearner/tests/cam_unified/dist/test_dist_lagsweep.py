"""Validated policy lag (manifest F03, spec 14.4 "discard samples outside validated training conditions"): the
injected-lag pipeline really trains at the requested lag, V-trace beats uncorrected training on stale data, and the
router's default max lag is the one the stored sweep validated."""
from __future__ import annotations

import json
import os

import pytest

from superlearner.cam_unified.dist.lagsweep import lag_sweep, run_one, summarize
from superlearner.cam_unified.dist.learner import CentralLearner
from superlearner.cam_unified.dist.offpolicy import VALIDATION_FILE, LagRouter, validated_max_lag


def test_injected_lag_is_exact_and_correction_matters(tmp_path):
    out = str(tmp_path / "sweep")
    vt = run_one(out, lag=8, seed=0, correction="vtrace", updates=30, eval_every=10)
    # the first 8 updates see lags 0..7 (fewer versions exist), every later one exactly 8
    assert vt["trained_lag_hist"] == {**{str(k): 256 for k in range(8)}, "8": 256 * 22}
    assert vt["exact_on_policy"] == 256                  # only the lag-0 batch is exactly on-policy
    none = run_one(out, lag=8, seed=0, correction="none", updates=30, eval_every=10)
    print("lag 8, 30 updates: vtrace curve", vt["curve"], "uncorrected curve", none["curve"])
    assert vt["curve"][0] == none["curve"][0]            # same start
    assert vt["final"] > 0.6 and vt["final"] > none["final"] + 0.05


def test_router_default_is_the_stored_validated_lag(tmp_path):
    doc = json.load(open(VALIDATION_FILE))
    s = summarize(doc["runs"])
    assert s["validated_max_lag"] == doc["validated_max_lag"]           # the file's verdict follows from its runs
    lags = sorted({r["lag"] for r in doc["runs"]})
    seeds = {r["seed"] for r in doc["runs"]}
    assert 0 in lags and doc["validated_max_lag"] in lags and len(seeds) >= 3
    assert {r["correction"] for r in doc["runs"]} == {"vtrace", "none"}
    v, prov = validated_max_lag()
    assert v == doc["validated_max_lag"] and prov["source"] == "lag_validation.json"
    R = LagRouter()
    assert R.max_lag == v and R.totals()["max_lag_source"]["source"] == "lag_validation.json"
    L = CentralLearner(str(tmp_path / "run"), _Tiny(), sink=None)
    assert L.router.max_lag == v
    L.close()


class _Tiny:
    def state_dict(self):
        import torch
        return {"w": torch.zeros(1)}

    def load_state_dict(self, sd):
        pass

    def update(self, batch):
        return {}


@pytest.mark.slow
def test_full_lag_sweep_reproduces_the_stored_validation(tmp_path):
    doc = json.load(open(VALIDATION_FILE))
    lags = tuple(sorted({r["lag"] for r in doc["runs"]}))
    res = lag_sweep(str(tmp_path / "sweep"), lags=lags, seeds=(0, 1, 2), corrections=("vtrace",))
    s = summarize(res)
    print(json.dumps(s["means"], indent=1))
    assert s["validated_max_lag"] == doc["validated_max_lag"]
