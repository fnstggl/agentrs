"""Procedural mechanism families: holdout invariants, every family builds and responds, validation of samples,
the split guard, surfaces."""
from __future__ import annotations

import itertools
import time

import pytest

import crucible
crucible.load_all_modules()
from crucible.research import tasks as T                       # noqa: E402

from superlearner.cam_unified.envs import worldgen as W        # noqa: E402


def test_catalogue_holdouts_are_disjoint():
    rep = W.holdout_report()
    assert rep["problems"] == []
    cat = W.catalogue()
    assert rep["n"]["train"] >= 30 and rep["n"]["val"] >= 8 and rep["n"]["test"] >= 12
    names = [f.name for f in cat]
    assert len(names) == len(set(names)) and all(n.startswith(W.VERSION + "_") for n in names)
    train = [set(f.motifs) for f in cat if f.split == "train" and f.engine == "scm"]
    for p in W.VAL_PAIRS + W.TEST_PAIRS:                           # held-out combinations never co-occur in training
        assert not any(set(p) <= s for s in train), p
        assert all(any(m in s for s in train) for m in p)          # ... but each motif is trained
    for f in cat:                                                  # every compositional triple's pairs are trained
        if f.engine == "scm" and f.relation == "compositional" and len(f.motifs) == 3:
            for pr in itertools.combinations(f.motifs, 2):
                assert any(set(pr) <= s for s in train)
            assert not any(set(f.motifs) <= s for s in train)
    assert not any("invu" in s for s in train)                     # structural holdout motif
    # the training set covers most of the spec's diversity dimensions; what only held-out families exercise is named
    assert {"delays", "feedback_loops", "changing_mechanisms", "partial_observability", "hidden_state",
            "nonlinearities", "network_effects", "resource_constraints", "competing_agents"} <= set(rep["train_dimensions"])
    assert rep["heldout_only_dimensions"] == ["technical_failures"]
    # relations of Crucible's own held-out scm families to this training set are computed, not assumed
    rel = rep["crucible_relations"]
    assert rel["scm_invu"]["relation_to_catalogue_training"] == "structural"
    assert rel["scm_feedback"]["relation_to_catalogue_training"] == "compositional"
    assert rel["scm_stock_gate"]["relation_to_catalogue_training"] == "compositional"
    # Crucible's scm_regime (a test family) is within this training distribution (the regime motif is trained in
    # combinations): it is listed as an in-distribution evaluation, never as transfer
    assert rel["scm_regime"]["relation_to_catalogue_training"] == "within"
    assert "scm_regime" in rep["in_distribution_evaluations"]
    assert all(rel[n]["crucible_split"] not in W.TRAINABLE_SPLITS for n in rep["in_distribution_evaluations"])
    assert any(g.startswith("technical_failures") for g in rep["known_gaps"])


def test_every_family_builds_and_responds():
    t0 = time.time()
    out = W.validate_catalogue(seeds=(0,), ticks=24)
    assert set(out) == {f.name for f in W.catalogue()}
    for fam, rows in out.items():
        for r in rows:
            assert r["levers"] >= 2 and r["gauges"] >= 2, fam
            assert r["responds"], f"{fam}: random settings do not change the outcome"
    # structure varies within a family across seeds (partial observability draws), and across families
    import numpy as np
    fd = next(f for f in W.catalogue() if f.name == "cw1_scm_delay_stock")
    draws = [W._builder(fd)(np.random.default_rng(s), {}) for s in range(12)]
    assert len({d["gauge_lag_p"] for d in draws}) >= 2 and len({d["gauge_noise"][1] for d in draws}) >= 2
    assert all(d["required"] == ["delay", "stock"] for d in draws)
    assert time.time() - t0 < 120


def test_sampled_families_pass_crucible_validity():
    """Crucible's validity checks (purity, bounded, consequences, discoverability, difficulty ...) and a reference
    policy by search accept draws of train, compositional and structural families."""
    W.register_families()
    for fam in ("cw1_scm_delay_stock", "cw1_scm_confounder_regime_threshold", "cw1_scm_invu_delay",
                "cw1_abm_congest_wom"):
        t = T.make_task(fam, 424_242, validate=True)     # a training-range seed: no val set is spent
        assert t.validated and t.reference_net is not None and t.reference_net > t.default_net, fam
        assert t.goal_net_cents is not None


def test_validity_sample_covers_every_split_and_engine():
    """One family of every (split, engine) cell of the catalogue passes Crucible's validity checks."""
    rep = W.validity_sample(per_cell=1)
    cells = {(f.split, f.engine) for f in W.catalogue()}
    assert set(rep["cells"]) == {f"{sp}/{en}" for sp, en in cells}
    bad = [r for r in rep["rows"] if not r["valid"]]
    assert bad == [], bad
    assert all(r["reference_net"] > r["default_net"] for r in rep["rows"])


@pytest.mark.slow
def test_validity_of_the_whole_catalogue(tmp_path):
    """Every family of the catalogue through Crucible's validity checks (recorded by the calibrate CLI; slow)."""
    rep = W.validity_sample(per_cell=None)
    assert len(rep["rows"]) == len(W.catalogue())
    bad = [r["family"] for r in rep["rows"] if not r["valid"]]
    assert bad == [], bad


def test_split_guard_refuses_heldout_families_and_seeds():
    g = W.SplitGuard()
    for fam in W.families("train")[:5] + ["scm_additive", "abm_utility", "ow_market", "bazaar", "ow_tiny"]:
        g.check(fam)
    for fam in W.families("val")[:2] + W.families("test")[:2] + ["scm_invu", "scm_feedback", "tech_pipeline",
                                                                  "ow_heldout", "ow_composite", "open_tight_compute",
                                                                  "mech_scm_regime", "lab_improvement"]:
        with pytest.raises(W.HeldOutFamilyError):
            g.check(fam)
    with pytest.raises(W.HeldOutFamilyError):
        g.check("never_declared_family")
    with pytest.raises(W.HeldOutFamilyError):                       # validation-range seed of a train family
        g.check_task(T.make_task("scm_additive", T.VAL_SEED0 + 3, validate=False))
    g.check_task(T.make_task("scm_additive", 3, validate=False))
    from crucible.goals.contracts import generate_goal
    g.check_task(generate_goal("revenue", "bazaar", 1, 0))
    with pytest.raises(W.HeldOutFamilyError):
        g.check_task(generate_goal("revenue", "open_tight_compute", 1, 0))
    # a guarded stream stops at the first held-out item
    stream = [T.make_task("scm_additive", 1, validate=False), T.make_task("scm_invu", 2, validate=False)]
    it = g.guard(stream)
    assert next(it).family == "scm_additive"
    with pytest.raises(W.HeldOutFamilyError):
        next(it)
    with pytest.raises(W.HeldOutFamilyError):
        next(W.training_stream(["cw1_scm_delay", "cw1_scm_invu"]))
    s = W.training_stream(["cw1_scm_delay", "cw1_abm_util_wom"], start=10)
    got = [next(s) for _ in range(4)]
    assert [t.family for t in got] == ["cw1_scm_delay", "cw1_abm_util_wom"] * 2 and got[2].seed == 11


def test_split_guard_declarations_only_tighten_and_cover_openworld_seeds():
    # a declaration may hold a trainable family out, never relabel a held-out one as trainable
    g = W.SplitGuard(extra={"ow_market": "test"})
    with pytest.raises(W.HeldOutFamilyError):
        g.check("ow_market")
    for fam in ("ow_heldout", "scm_invu", W.families("val")[0]):
        with pytest.raises(ValueError):
            W.SplitGuard(extra={fam: "train"})
    with pytest.raises(ValueError):
        W.SplitGuard(extra_pairs={("ow_heldout", 1): "train"})
    # OpenWorld tasks: the seed is checked (validation-range convention) and protocol-declared pairs are honoured
    from crucible.goals.contracts import GoalContract
    from crucible.research import llm_eval as L

    def ow(fam, seed):
        return L.open_economy_goal(world_family=fam, seed=seed, overrides={}, days=3)
    g = W.SplitGuard(extra_pairs={("ow_market", 7): "val"})
    g.check_task(ow("ow_market", 6))
    with pytest.raises(W.HeldOutFamilyError):
        g.check_task(ow("ow_market", 7))
    with pytest.raises(W.HeldOutFamilyError):
        g.check_task(ow("ow_market", T.VAL_SEED0 + 1))
    # a task labelled with a held-out split (e.g. from a Crucible val/test goal suite) never trains
    t = ow("ow_market", 6)
    t.split = "test"
    with pytest.raises(W.HeldOutFamilyError):
        g.check_task(t)
    assert isinstance(t, GoalContract)


def test_surfaces_change_appearance_only():
    a, b = W.invariance_pair(3)
    orig = ["ab12", "cd34", "ef56", "gh78"]
    assert a.names(orig) != b.names(orig) and a.variant != b.variant
    assert sorted(a.order(6)) == list(range(6))
    assert a.phrase("goal", horizon=96, goal=500) != b.phrase("goal", horizon=96, goal=500)
    plain = W.Surface(1, rename=False, shuffle=False, scale_units=False, reword=False)
    assert plain.names(orig) == orig and plain.order(4) == [0, 1, 2, 3] and plain.unit() == ("reading", 1.0)
    # the default surface shows numbers as the simulator states them; only declared invariance pairs rescale
    assert all(W.Surface(k).unit() == ("reading", 1.0) for k in range(20))
    assert {W.invariance_pair(k)[0].unit() for k in range(20)} != {("reading", 1.0)}
