"""SP-6.5 - software ecosystem: hosting, packages, dependency graphs, issues, PRs, review, tests, releases, docs,
deployment, adoption by execution, maintenance, forks, compatibility, failures (workstream G).

World `crucible.innovation.scenarios.software` (seed 3): five scripted reference developers act only through the
pkg/code/issues/cloud tools (crucible.innovation.reference.software_history). Every feature is counted from recorded
state; adoption counts only (package, dependent) pairs where the dependent's own successful sandboxed execution
depended on the package - measured by the parent, by ablating the package and observing that the outcome changes
(D-20) - never stars, declared dependencies or the in-sandbox tracer's self-reported counts. Advisories are
verified only when a later release makes the reproduction pass. The audit's forgeries (tracer counter overwrite,
shadow copy of a dependency, an `assert False` advisory against a competitor, lineage claimed in a description)
are run in the same world and must leave no trace in adoption, advisories or lineage.
"""
import pytest

import crucible

crucible.load_all_modules()
from crucible.acceptance.v5 import evidence  # noqa: E402
from crucible.innovation import measure as M  # noqa: E402
from crucible.innovation import scenarios as SC  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1200)]


@pytest.fixture(scope="module")
def software():
    r = SC.software(seed=3)
    yield r
    SC.close(r)


def test_SP_6_5_software_ecosystem(software):
    w = software["world"]
    f = M.software_features(w)
    ad = M.adoptions_by_execution(w)
    stars = sum(p["stars"] for _, p in w.state.scan("pkg_packages"))
    declared_not_executed = sum(1 for _, d in w.state.scan("pkg_deps") if d["status"] == "active" and not any(
        a["repo"] == d["repo"] and a["package"] == d["package"] for a in ad))
    h = software["history"]
    m = {"software_features": f["count"], "adoptions_by_execution": len(ad),
         "stars": stars, "dependencies_declared_but_not_counted": declared_not_executed,
         "transitive_dependency": f["transitive_dependencies"],
         "advisory_verified_by_fixing_release": bool(w.state.tables["pkg_advisories"][h["advisory"]["advisory"]]["verified"]),
         "advisory_public_before_fix": bool(h["advisory"]["verified"]),
         "advisory_fixed_by_reviewed_pr": bool(h["merge"]["merged"] and h["geokit_1_0_1"]["fixes_advisories"]),
         "breaking_release_flagged": bool(h["geokit_1_1_0"]["compat"]["breaking"]
                                          and not h["geokit_1_1_0"]["compat"]["semver_ok"]),
         "incompatible_upgrade_refused": not h["ben_upgrade_breaking"]["upgraded"],
         "invariants_clean": not w.check_invariants(),
         **{f"feature[{k}]": v for k, v in sorted(f["evidence"].items())}}
    forg = _forgeries(w)
    m.update(forg)
    evidence.record("SP-6.5", m, notes="Scripted reference developers (feasibility). Features present: "
                                       f"{f['present']}. Adoptions: {[(a['package'], a['repo'], a['executions']) for a in ad]}")
    assert m["software_features"] >= 15 and m["adoptions_by_execution"] >= 1
    assert m["invariants_clean"] and forg["forgeries_recorded"] == 0, forg


def _forgeries(w) -> dict:
    """The audit's forgery attempts, by fresh personas, against the history's packages."""
    from crucible.innovation import reference as R

    def ex(who, tool, **a):
        return w.execute(f"p:{who}", {"tool": tool, "args": a})
    for h in ("yan", "xia", "zed", "mal"):
        R.join(w, h, 1_000_000)
    before = {k: dict(v) for k, v in w.state.scan("pkg_adoptions")}
    n_adv = len(w.state.tables["pkg_advisories"])
    out = {"forgery_attempts": 4}
    ex("yan", "code.create_repo", name="forge")
    ex("yan", "code.commit", repo="yan/forge", files={"app.py": "def run(args, caller):\n    import _pkgtrace\n"
                                                    "    _pkgtrace._C['vecmath'] = 1000\n    return 1\n"})
    ex("yan", "pkg.depend", repo="yan/forge", package="vecmath")
    ex("yan", "pkg.exec", repo="yan/forge", target="app:run", input=None)
    ex("xia", "code.create_repo", name="shadow")
    ex("xia", "code.commit", repo="xia/shadow", files={"deps/vecmath/fake.py": "def f():\n    return 2\n",
                                                      "app.py": "from deps.vecmath.fake import f\n\n\n"
                                                                "def run(args, caller):\n    return f()\n"})
    ex("xia", "pkg.depend", repo="xia/shadow", package="vecmath")
    ex("xia", "pkg.exec", repo="xia/shadow", target="app:run", input=None)
    ex("zed", "code.create_repo", name="grief", visibility="public", license="open")
    ex("zed", "code.commit", repo="zed/grief", files={"tests/test_x.py": "def test_x():\n    assert False\n"})
    adv = ex("zed", "pkg.advisory", package="vecmath", version="1.0.0", kind="security", severity="critical",
             description="critical", repro_repo="zed/grief")
    ex("mal", "code.create_repo", name="fakefork", visibility="public", license="open", description="fork of eve/vecmath")
    ex("mal", "code.commit", repo="mal/fakefork", files={"lib.py": R.VEC_LIB})
    reg = ex("mal", "pkg.register", name="vec-fake", repo="mal/fakefork", description="vectors")
    after = {k: dict(v) for k, v in w.state.scan("pkg_adoptions")}
    new_adopt = [k for k in after if k not in before and k.split("|", 1)[1] in ("yan/forge", "xia/shadow")]
    out["forged_adoptions"] = len(new_adopt)
    out["forged_advisories"] = int(adv.ok) + (len(w.state.tables["pkg_advisories"]) - n_adv - int(adv.ok))
    out["forged_lineage"] = int(reg.ok and bool(reg.result.get("forked_from")))
    out["forgeries_recorded"] = out["forged_adoptions"] + out["forged_advisories"] + out["forged_lineage"]
    return out
