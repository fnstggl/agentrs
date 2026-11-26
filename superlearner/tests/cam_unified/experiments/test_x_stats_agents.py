"""Workstream X: two-level bootstrap, the NIG knowledge-gradient / Thompson experimenters on real mechanism worlds,
and the import hygiene of the experiments package (CPU)."""
from __future__ import annotations

import ast
import os

import numpy as np
import pytest

from superlearner.cam_unified.experiments import stats as ST


def test_two_level_bootstrap_covers_the_mean_and_widens_with_seed_variation():
    rng = np.random.default_rng(0)
    hits = 0
    for rep in range(60):
        seed_eff = rng.normal(0, 0.5, size=(3, 1))
        world_eff = rng.normal(0, 0.5, size=(1, 16))
        M = 1.0 + seed_eff + world_eff + rng.normal(0, 0.3, size=(3, 16))
        ci = ST.two_level(M, n_boot=600, seed=rep)
        truth = 1.0 + seed_eff.mean() + world_eff.mean()
        hits += ci["lo"] <= truth <= ci["hi"]
    assert hits >= 48                                         # nominal 95%; a small sample of reps
    flat = ST.two_level(np.ones((3, 16)) + rng.normal(0, 0.3, (1, 16)), n_boot=600)
    spread = ST.two_level(np.ones((3, 16)) + rng.normal(0, 0.3, (1, 16)) + np.array([[-2], [0], [2]]), n_boot=600)
    assert (spread["hi"] - spread["lo"]) > 3 * (flat["hi"] - flat["lo"])   # seed variation enters the interval
    d = ST.paired(np.full((3, 8), 2.0) + rng.normal(0, 0.1, (3, 8)), np.full((1, 8), 1.0))
    assert d["lo"] > 0.8 and d["p_gt0"] == 1.0
    assert ST.two_level(np.full((2, 3), np.nan))["mean"] is None


def test_kg_and_thompson_experimenters_learn_within_an_episode_on_real_worlds(dev_tasks):
    from superlearner.cam_unified.experiments import mechrun as M
    from superlearner.cam_unified.experiments.nig_agent import NIGToolAgent
    scores = {}
    for rule in ("kg", "thompson", "random"):
        w = M.make_world(dev_tasks, seed=3)
        ag = NIGToolAgent(rule, seed=0)
        res = M.run_batch(ag, w, learn=False)
        assert all(r.decisions == len(r.nets) for r in res)            # one decision, one tick
        scores[rule] = float(np.mean([r.score for r in res]))
        late = [r.windows.get("64") for r in res]
        if rule != "random":
            assert all(x is not None and x > r.windows["0"] for x, r in zip(late, res))   # improves with evidence
        assert ag.compute["decisions"] == sum(r.decisions for r in res)
    assert scores["kg"] > scores["random"] and scores["thompson"] > scores["random"]


def test_experiments_package_never_imports_test_doubles():
    import superlearner.cam_unified.experiments as pkg
    d = os.path.dirname(pkg.__file__)
    for f in os.listdir(d):
        if not f.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(d, f)).read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "")] + [a.name for a in node.names]
            for n in names:
                assert "testing" not in n.split(".") and "integration_doubles" not in n, (f, n)


def test_flat_predictive_scores_the_expected_regret_of_a_random_choice(dev_tasks):
    from superlearner.cam_unified.experiments import predict as P
    ep = P.collect_support(dev_tasks[:1], k_max=4, n_queries=6, design_seed=0, ks=(4,))[0]
    y = ep.queries[4]["Y"]
    flat = P.score_queries(ep, 4, np.zeros(6), np.ones(6))
    assert flat["regret_raw"] == pytest.approx(float(y.max() - y.mean()))     # no information: random choice
    oracle = P.score_queries(ep, 4, y.copy(), np.ones(6))
    assert oracle["regret_raw"] == 0.0


def test_obs_needed_requires_sustained_identification():
    from superlearner.cam_unified.experiments.exp2 import obs_needed
    rows = [{"task": "t", "k": k, "regret": r} for k, r in ((0, 0.05), (1, 0.3), (2, 0.08), (4, 0.02), (8, 0.0))]
    assert obs_needed(rows, "t") == 2                       # k=0 was luck: regret rose again at k=1
    assert obs_needed([{"task": "t", "k": 0, "regret": 0.5}], "t") is None


def test_trajectory_recorder_captures_an_acting_learners_own_support(dev_tasks):
    from superlearner.cam_unified.experiments import mechrun as M
    from superlearner.cam_unified.experiments import predict as P
    from superlearner.cam_unified.experiments.baselines import NIGPredictor
    from superlearner.cam_unified.experiments.nig_agent import NIGToolAgent
    w = M.make_world(dev_tasks, seed=1)
    rec = P.TrajectoryRecorder(w, ks=(0, 4, 16), n_queries=5, seed=2)
    res = M.run_batch(NIGToolAgent("kg", seed=1), w, learn=False, on_round=rec)
    eps = rec.episodes()
    for ep, r in zip(eps, res):
        assert len(ep.steps) == r.decisions == len(ep.Y) and set(ep.queries) == {0, 4, 16}
        assert np.allclose(ep.Y, r.rewards)                   # one tick per decision: reward = the tick's reward
        assert ep.steps[3]["action"].source.value == "baseline"
        m, v, d = NIGPredictor().predict(ep, 16)
        assert np.isfinite(P.score_queries(ep, 16, m, v, d)["logscore"])
