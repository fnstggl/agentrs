"""W9 (W1 change request): experiments/predict.collect_support runs its design policy through the canonical loop
(mechrun.run_batch -> runtime.run_lifetime where W1's loop is present) and reproduces the pre-migration support
exactly (legacy_support_ref.py keeps the old loop as the reference): actions, rewards, times, observations,
settings, gauges, queries, notes. The package contains no support loop of its own any more."""
from __future__ import annotations

import ast
import os

import numpy as np
import pytest

from superlearner.cam_unified.experiments import predict as P

from legacy_support_ref import legacy_collect_support


def _obs_key(o):
    return (getattr(o.kind, "value", o.kind), o.content, o.entity, round(float(o.t or 0.0), 9))


def _act_key(a):
    return None if a is None else (a.tool, tuple(sorted((k, repr(v)) for k, v in a.args.items())))


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.condition == y.condition and x.lever_names == y.lever_names and x.delay == y.delay
        for f in ("lo", "hi", "x0", "X", "G", "gmask", "Y"):
            assert np.array_equal(getattr(x, f), getattr(y, f)), f
        assert x.notes == y.notes and x.integer == y.integer
        assert len(x.steps) == len(y.steps)
        for s, t in zip(x.steps, y.steps):
            assert _act_key(s["action"]) == _act_key(t["action"])
            assert s["reward"] == t["reward"] and s["t"] == t["t"]
            assert [_obs_key(o) for o in s["obs"]] == [_obs_key(o) for o in t["obs"]]
            assert [_obs_key(o) for o in s["next_obs"]] == [_obs_key(o) for o in t["next_obs"]]
        assert [_obs_key(o) for o in x.obs_final] == [_obs_key(o) for o in y.obs_final]
        assert sorted(x.queries) == sorted(y.queries)
        for k in x.queries:
            assert np.array_equal(x.queries[k]["Y"], y.queries[k]["Y"])
            assert np.array_equal(x.queries[k]["X"], y.queries[k]["X"])
            assert [_act_key(a) for a in x.queries[k]["actions"]] == [_act_key(a) for a in y.queries[k]["actions"]]


@pytest.mark.parametrize("kw", [dict(condition="A", ks=(0, 2, 6)), dict(condition="C", notes_m=2, ks=(0, 6)),
                                dict(condition="D", delay=2, ks=(3, 6))])
def test_canonical_loop_support_equals_the_pre_migration_loop(dev_tasks, kw):
    new = P.collect_support(dev_tasks, k_max=6, n_queries=3, design_seed=11, **kw)
    old = legacy_collect_support(dev_tasks, k_max=6, n_queries=3, design_seed=11, **kw)
    _same(new, old)
    assert all(len(e.steps) == 6 for e in new) and all(sorted(e.queries) == sorted(kw["ks"]) for e in new)
    if kw["condition"] == "D":       # the support record holds the delayed rewards the agent view would receive
        assert any(s["reward"] != y for e in new for s, y in zip(e.steps, e.Y))


def test_design_policy_holds_no_operator_side_object(dev_tasks):
    pol = P.DesignPolicy([t.id for t in dev_tasks], 0)
    with pytest.raises(RuntimeError):
        pol.act(None, [[], []])
    P.collect_support(dev_tasks[:1], k_max=2, n_queries=2, design_seed=0, ks=(2,))
    from superlearner.cam_unified.experiments import mechrun as M
    w = M.make_world(dev_tasks[:1], seed=0)
    pol = P.DesignPolicy([dev_tasks[0].id], 0)
    rec = P.SupportRecorder(w, pol, dev_tasks[:1], (), 2, 0, 2)
    M.run_batch(pol, w, max_rounds=2, on_round=rec)
    banned = (type(w), type(w.operator), type(w.evaluator), type(w.adapter))
    assert not [k for k, v in vars(pol).items() if isinstance(v, banned)]
    assert pol.decisions == 2 and rec.rounds == 2


def test_predict_has_no_act_execute_advance_loop():
    src = open(P.__file__).read()
    for fn in ast.walk(ast.parse(src)):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for loop in ast.walk(fn):
            if isinstance(loop, (ast.For, ast.While)):
                names = {n.attr for n in ast.walk(loop) if isinstance(n, ast.Attribute)}
                assert not ("execute" in names and names & {"advance_time", "advance"}), fn.name
