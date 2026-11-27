"""Unit and short integration tests of the software-ecosystem domain (9.2): the capability catalogue, the
developers' generated tests and applications, adoption by execution, preferences and the degenerate strategies."""
import ast

import numpy as np

import crucible

crucible.load_all_modules()
from crucible.domains import _t2kit as K  # noqa: E402
from crucible.domains import base  # noqa: E402
from crucible.domains import software as D  # noqa: E402


def test_catalogue_tests_and_apps_are_valid_python():
    rng = np.random.default_rng(0)
    for need in D.NEEDS:
        cases = {need: D.draw_cases(need, rng, 3)}
        ast.parse(D.tests_source([need], cases))
        ast.parse(D.app_source({need: {"kind": "diy"}}))
        ast.parse(D.app_source({need: {"kind": "pkg", "import": "deps.x.lib", "fn": D.NEEDS[need]["hint"]}}))
        f = D._reference_fn(need)
        assert all(f(*a) is not None for a in cases[need])
    rep = base.structure_report(D.DOMAIN, (0, 1))
    assert rep["train_structures"] == 3 and rep["compositional_eval"] == 1 and rep["structural_eval"] == 1


def test_developers_adopt_by_execution_with_provenance():
    """The population's developers alone (null seat), 3 days: hand-coded and package-based applications, adoption
    only through executions measured by ablation, every decision organic."""
    r = base.run("software", "libraries", 0, policy="null", days=3, keep_env=True)
    w = r["env"].world
    assert r["proc_failed"] == 0 and r["invariant_violations"] == 0
    devs = list(w.state.scan("devpop_d"))
    assert len(devs) >= 10
    kinds = {v.get("kind") for _, d in devs for v in d["via"].values()}
    assert "diy" in kinds
    prov = K.decision_provenance(w, ("devpop.",))
    assert prov["batches"] > 0 and prov["forced"] == 0
    assert r["mechanics"]["counts"]["Executable code"] > 0 and r["mechanics"]["counts"]["Repositories"] > 0
    adopted = [a for _, a in w.state.scan("pkg_adoptions")]
    assert all(D.population_owner(w, a["owner"]) for a in adopted)
    assert len({json.dumps(d["w"], sort_keys=True) for _, d in devs}) == len(devs)   # heterogeneous preferences
    r["env"].close()


def test_stub_package_and_self_adoption_fail():
    for pol in ("stub_package", "self_adoption"):
        r = base.run("software", "libraries", 0, policy=pol, days=2)
        assert r["success"] is False and r["outcome"]["details"]["independent_adopters"] == 0, (pol, r["outcome"])


def test_reference_reads_requests_from_forum_listings():
    """forum.list returns post summaries (a snippet of the body): the maintainer must parse those, the format the
    population developers post in (regression: requests were never read, so every capability shipped at once and
    the package was never extended)."""
    nd = "slugify"
    body = (f"I need a function like {D.NEEDS[nd]['hint']}(...): {D.NEEDS[nd]['doc']}. I wrote it myself for now. "
            f"Keywords: {D.NEEDS[nd]['keywords']}.")
    got = D._parse_requests([{"post": "p1", "title": "Looking for a package", "snippet": body[:200]},
                             {"post": "p2", "title": "Looking for a package", "snippet": body[:200]}])
    assert got == {"slugify": {"doc": D.NEEDS[nd]["doc"], "count": 2}}
    assert D.FEATURE_RE.match("Please add median()").group(1) == "median"


import json  # noqa: E402


def test_reference_test_cases_match_the_catalogue_functions():
    """The reference maintainer writes its package's tests from textbook cases (no exec in the reference: it is
    scanned like any policy); each case must hold for the catalogue implementation it ships."""
    for need, (args, expected) in D.TEXTBOOK_CASES.items():
        env: dict = {}
        exec(D.NEEDS[need]["src"], env)           # test side: the catalogue source the reference commits
        assert env[D.NEEDS[need]["hint"]](*args) == expected, need
    assert set(D.TEXTBOOK_CASES) == set(D.NEEDS)
    assert base.reads_world(D.reference_script) == [] and base.reads_world(D.DOMAIN.reference) == []


def test_goal_generator_and_evidence_are_framework_3():
    """Objectives come from the domain's goal generator (registered evaluator-only metric); every evidence function
    returns recorded event sequence numbers."""
    from crucible.goals import metrics as GM
    w = base.world("software", "libraries", 0)
    try:
        objs = {k: D.DOMAIN.make_objective(w, "p:maintainer", k) for k in D.DOMAIN.difficulties}
        assert len({o["text"] for o in objs.values()}) == len(objs)
        assert all(o["generator"] == "crucible.goals" and o["success"]["metric"] == "software_independent_adopters"
                   for o in objs.values())
        assert "software_independent_adopters" in GM.EVALUATOR_ONLY_METRICS
        for m in D.MECHANICS:
            if m.count is not None:
                assert isinstance(m.count(w, 0), list), m.name
    finally:
        w.close()
