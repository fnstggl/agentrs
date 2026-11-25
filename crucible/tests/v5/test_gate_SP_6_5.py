"""SP-6.5 - software ecosystem: hosting, packages, dependency graphs, issues, PRs, review, tests, releases, docs,
deployment, adoption by execution, maintenance, forks, compatibility, failures (workstream G).

World `crucible.innovation.scenarios.software` (seed 3): five scripted reference developers act only through the
pkg/code/issues/cloud tools (crucible.innovation.reference.software_history). Every feature is counted from recorded
state; adoption counts only (package, dependent) pairs where the package's code ran inside the dependent's own
successful sandboxed execution (call tracing), never stars or declared dependencies alone.
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
         "advisory_verified_by_reproduction": bool(h["advisory"]["verified"]),
         "advisory_fixed_by_reviewed_pr": bool(h["merge"]["merged"] and h["geokit_1_0_1"]["fixes_advisories"]),
         "breaking_release_flagged": bool(h["geokit_1_1_0"]["compat"]["breaking"]
                                          and not h["geokit_1_1_0"]["compat"]["semver_ok"]),
         "incompatible_upgrade_refused": not h["ben_upgrade_breaking"]["upgraded"],
         "invariants_clean": not w.check_invariants(),
         **{f"feature[{k}]": v for k, v in sorted(f["evidence"].items())}}
    evidence.record("SP-6.5", m, notes="Scripted reference developers (feasibility). Features present: "
                                       f"{f['present']}. Adoptions: {[(a['package'], a['repo'], a['executions']) for a in ad]}")
    assert m["software_features"] >= 15 and m["adoptions_by_execution"] >= 1
    assert m["invariants_clean"]
