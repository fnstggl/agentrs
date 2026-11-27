"""The five I-3 domains (9.11-9.15) on `domains-framework/3` (regression tests of the migration, DEF-I3-1).

The framework MEASURES and never accepts declarations: objectives come from the independent goal generator
(`GoalFamily` over registered `crucible.goals.metrics` metrics), structure hooks read built state only, mechanic
evidence is a list of recorded event sequence numbers (a number is an unverified self-report), autonomous
participants' decisions are attributed to their own principals (`Event.actor`), and no unregistered failure draw
remains in the domains' files or the kernel module files of their worlds."""
from __future__ import annotations

import pytest

import crucible.domains as D
from crucible.domains import base

DOMAINS = ("information", "organization", "marketplace", "crisis", "commercialization")


@pytest.fixture(scope="module", autouse=True)
def _loaded():
    D.load_all()


@pytest.mark.parametrize("name", DOMAINS)
def test_goal_generator_structure_from_built_state_and_no_dice(name):
    d = base.REGISTRY[name]
    assert base.validate(d) == []
    assert d.goal is not None and d.objective is None
    g = base.goal_report(d)
    assert g["ok"] and g["generator"] == "crucible.goals" and g["distinct_texts"] == len(d.difficulties), g
    from crucible.goals import metrics as GM
    w = base.world(name, d.configs[0].name, 0)
    try:
        obj = d.make_objective(w, d.principal_of(w), 1)
        leaves = GM.predicate_leaves(obj["success"])
        assert leaves and all(lf["metric"] in GM.METRICS for lf in leaves)
        # evaluator-only: no agreement between participants can read the goal's metrics
        assert all(lf["metric"] in GM.EVALUATOR_ONLY_METRICS for lf in leaves)
        assert base.reads_declarations(d.structure) == []
        fp = d.structure(base.BuiltWorld(w))
        assert fp and all(v is not None for v in fp.values()), fp
        with pytest.raises(base.DeclarationAccess):
            base.BuiltWorld(w).modules["pop"]
        srcs = base.domain_sources(d) + base.world_sources(w)
    finally:
        w.close()
    scan = base.dice_scan(srcs)
    assert scan["mechanisms"] == 0, scan["hits"]


@pytest.mark.parametrize("name", ("organization", "crisis"))
def test_evidence_is_recorded_events_and_actors_decide_as_themselves(name):
    """A short null run: every mechanic's evidence function returns event sequence numbers the framework verifies
    (no unverified, rejected or crashing evidence), and every declared actor - the organization's providers and the
    crisis operators included, whose events were the system's before - makes attributed decisions."""
    d = base.REGISTRY[name]
    r = base.run(name, d.configs[0].name, 0, policy="null", days=3)
    assert r["mechanics"]["unverified"] == {} and r["mechanics"]["rejected"] == {} and r["mechanics"]["errors"] == {}
    assert r["actors"]["unverified"] == {} and r["actors"]["errors"] == {}
    assert r["actors"]["active"] == r["actors"]["declared"], r["actors"]
    assert r["proc_failed"] == 0 and r["invariant_violations"] == 0
