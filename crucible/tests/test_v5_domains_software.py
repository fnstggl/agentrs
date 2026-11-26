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


import json  # noqa: E402
