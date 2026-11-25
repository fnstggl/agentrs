"""Workstream X: task streams, counterfactual forks, condition wrappers on real Crucible mechanism worlds (CPU)."""
from __future__ import annotations

import json
import random

import numpy as np
import pytest

from superlearner.cam_unified.experiments import mechrun as M
from superlearner.cam_unified.experiments import predict as P
from superlearner.cam_unified.interfaces import ObsKind


def test_task_stream_refuses_heldout_families_and_buckets_horizons():
    from superlearner.cam_unified.envs.worldgen import HeldOutFamilyError
    M.ensure_registered()
    with pytest.raises(HeldOutFamilyError):
        M.TaskStream(["scm_delay_gate"], start=10)             # a validation family never trains
    with pytest.raises(HeldOutFamilyError):
        M.TaskStream(["scm_additive"], start=2_000_000).batch(1)   # validation-range seed
    st = M.TaskStream(M.domain_families("scm"), start=100_000, seed=3)
    for _ in range(4):
        b = st.batch(5)
        assert len({t.horizon for t in b}) == 1                 # one horizon per batch: no idle environments
        assert all(t.split == "train" for t in b)
    seeds = [t.seed for t in st.batch(3)]
    assert seeds == sorted(seeds) and len(set(seeds)) == 3
    assert set(M.domain_families("abm")).isdisjoint(M.domain_families("scm"))


def test_fork_rewards_equal_the_realised_tick_and_leave_the_world_unchanged(dev_tasks):
    w = M.make_world(dev_tasks[:1], seed=0)
    ob = w.adapter.reset()
    tools = w.adapter.available_tools()[0]
    e = w.operator.envs[0]
    lv = e.lever_spec[0]
    v = lv.lo + 0.7 * (lv.hi - lv.lo)
    raw = list(e.raw)
    raw[0] = float(lv.clip(v))
    f1 = M.fork_rewards(w, 0, [raw, list(e.raw)])
    f2 = M.fork_rewards(w, 0, [raw, list(e.raw)])
    assert f1 == f2                                            # forks do not advance the real world
    a = P._set_action(e.lever_shown[0], raw[0], lv.integer, tools)
    res = w.adapter.execute([a])
    w.adapter.advance_time([res[0].duration + M.THINK_S / M.DAY])
    got = w.evaluator.take_rewards()[0]
    assert len(w.rewards[0]) == 1                               # one decision = exactly one tick
    assert got == pytest.approx(f1[0], abs=1e-6)               # the fork predicted the realised outcome exactly


def test_delayed_partial_hides_gauges_and_delays_money(dev_tasks):
    w = M.make_world(dev_tasks[:1], seed=1)
    dp = M.DelayedPartial(w, delay=3, hide_frac=0.5, seed=5)
    ob = dp.reset()
    hidden = dp.hidden[0]
    assert len(hidden) == len(w.operator.envs[0].gauge_shown) // 2 and hidden
    tools = w.adapter.available_tools()[0]
    rng = random.Random(0)
    delivered, rewards = [], []
    for t in range(12):
        tl = [x for x in tools if x.name.startswith("set_")]
        tool = rng.choice(tl)
        p = tool.parameters["properties"]["value"]
        val = rng.uniform(p["minimum"], p["maximum"])
        a = P.Action(tool.name, {"value": int(round(val)) if p["type"] == "integer" else val})
        res = w.adapter.execute([a])
        adv = dp.advance([res[0].duration + M.THINK_S / M.DAY])
        rewards.append(dp.take_rewards(adv.done)[0])
        delivered.append((w.operator.envs[0].ticks, adv.observations[0]))
    for ticks, obs in delivered:
        for o in obs:
            assert not (o.entity or "").startswith(tuple(f"gauge:{h}" for h in hidden))
            if o.kind == ObsKind.EVENT and '"facility.report"' in o.content:
                p = json.loads(o.content)["payload"]
                assert set(p["readings"]).isdisjoint(hidden)
                assert "net_cents" not in p                      # money left the report
            if o.kind == ObsKind.NUMERIC and (o.meta or {}).get("field") == "net_cents":
                assert ticks - 1 - o.meta["tick"] == 3          # tick t's money arrives 3 ticks later
    true = w.rewards[0]
    assert rewards[:3] == [0.0, 0.0, 0.0]
    assert rewards[3:] == pytest.approx(true[:9], abs=1e-6)     # the verified reward is delayed, not changed


def test_predecessor_notes_report_real_experiments(dev_tasks):
    w = M.make_world(dev_tasks[:1], seed=2)
    w.adapter.reset()
    notes = M.PredecessorNotes().notes(w, 0, 3, random.Random(9))
    assert len(notes) == 3 and all(n.kind == ObsKind.TEXT and n.source == "predecessor" for n in notes)
    e = w.operator.envs[0]
    r = random.Random(9)
    for n in notes:
        j = r.randrange(len(e.lever_spec))
        lv = e.lever_spec[j]
        v = r.uniform(lv.lo, lv.hi)
        v = float(int(round(v))) if lv.integer else round(v, 3)
        base, after = M.PredecessorNotes()._experiment(e.task, w.operator.spec, j, v)
        r.randrange(3)
        assert e.lever_shown[j] in n.content and str(int(round(after))) in n.content
        assert e.lever_engine[j] not in n.content or e.lever_engine[j] == e.lever_shown[j]


def test_window_scores_follow_task_normalisation(dev_tasks):
    t = dev_tasks[0]
    nets = [t.default_net] * 4 + [t.reference_net] * 4
    ws = M.window_scores(t, nets)
    assert ws["0"] == pytest.approx(0.0) and ws["1"] == pytest.approx(0.0) and ws["2"] == pytest.approx(0.0)
    assert ws["4"] == pytest.approx(1.0)
