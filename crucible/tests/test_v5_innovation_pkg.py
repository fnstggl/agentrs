"""Software ecosystem (module pkg): adoption only by execution, review independence, verified advisories,
compatibility, and exact replay of a full scripted package history."""
import json

import pytest

import crucible

crucible.load_all_modules()
from crucible.core.kernel import World  # noqa: E402
from crucible.innovation import measure as M  # noqa: E402
from crucible.innovation import reference as R  # noqa: E402
from crucible.innovation.worlds import innovation_spec  # noqa: E402


def ex(w, who, tool, **args):
    return w.execute(f"p:{who}", {"tool": tool, "args": args})


def must(w, who, tool, **args):
    r = ex(w, who, tool, **args)
    assert r.ok, (tool, r.error)
    return r.result


@pytest.fixture()
def world():
    w = World.create(innovation_spec(9, with_adopters=False))
    for h in ("eve", "ben", "cat", "dan"):
        R.join(w, h, 1_000_000)
    must(w, "eve", "code.create_repo", name="vecmath", visibility="public", license="open")
    must(w, "eve", "code.commit", repo="eve/vecmath", files={"lib.py": R.VEC_LIB, "tests/test_lib.py": R.VEC_TESTS,
                                                             "README.md": "# vecmath\n"})
    must(w, "eve", "pkg.register", name="vecmath", repo="eve/vecmath", description="vector maths")
    must(w, "eve", "pkg.release", package="vecmath", version="1.0.0")
    yield w
    w.close()


APP = "from deps.vecmath.lib import norm\n\n\ndef run(args, caller):\n    return norm(args)\n"


def test_dependency_and_stars_are_not_adoption_only_execution_is(world):
    w = world
    must(w, "ben", "code.create_repo", name="app")
    must(w, "ben", "code.commit", repo="ben/app", files={"app.py": APP, "other.py": "def run(a, c):\n    return 1\n"})
    must(w, "ben", "pkg.depend", repo="ben/app", package="vecmath", constraint="^1.0.0")
    must(w, "cat", "pkg.star", package="vecmath")
    assert M.adoptions_by_execution(w) == [] and w.state.get("pkg_packages", "vecmath")["stars"] == 1
    r = must(w, "ben", "pkg.exec", repo="ben/app", target="other:run", input=None)   # never touches vecmath
    assert r["ok"] and r["calls"] == {} and M.adoptions_by_execution(w) == []
    r = must(w, "ben", "pkg.exec", repo="ben/app", target="app:run", input=[3, 4])
    assert r["ok"] and r["result"] == 5.0 and r["calls"]["vecmath"] >= 1
    ad = M.adoptions_by_execution(w)
    assert len(ad) == 1 and ad[0]["repo"] == "ben/app" and ad[0]["independent"]
    lock = json.loads(w.query("p:ben", "code.read", {"repo": "ben/app", "path": "crucible.json"})["content"])
    assert lock["packages"]["vecmath"]["version"] == "1.0.0" and lock["deps"][0].startswith("eve/vecmath@c")
    # the package owner running it in its own repo is not independent adoption
    must(w, "eve", "code.create_repo", name="selfuse")
    must(w, "eve", "code.commit", repo="eve/selfuse", files={"app.py": APP})
    must(w, "eve", "pkg.depend", repo="eve/selfuse", package="vecmath")
    must(w, "eve", "pkg.exec", repo="eve/selfuse", target="app:run", input=[6, 8])
    assert len(M.adoptions_by_execution(w)) == 1
    assert w.check_invariants() == []


def test_private_dependents_stay_private(world):
    w = world
    must(w, "ben", "code.create_repo", name="secret")
    must(w, "ben", "code.commit", repo="ben/secret", files={"app.py": APP})
    must(w, "ben", "pkg.depend", repo="ben/secret", package="vecmath")
    ev = [e for e in w.log if e.type == "pkg.dependency"][-1]
    assert ev.vis == ("p:ben",)
    w.advance_time(w.now + 7 * 3600)                     # a needs-graph update
    g = w.query("p:cat", "needs.graph", {})
    assert not any("ben/secret" in e["src"] for e in g["edges"])
    assert any("ben/secret" in e["src"] for e in w.query("p:ben", "needs.graph", {})["edges"])
    must(w, "ben", "code.create_repo", name="secret2")
    must(w, "ben", "code.commit", repo="ben/secret2", files={"app.py": APP + "\n"})
    pr = must(w, "ben", "pkg.pr_open", repo="ben/secret", source_repo="ben/secret2", title="t")["pr"]
    assert not ex(w, "cat", "pkg.pr_diff", pr=pr).ok


def test_reviews_must_be_read_independent_and_approving_before_merge(world):
    w = world
    must(w, "dan", "pkg.fork", package="vecmath", name="vec-fork")
    must(w, "dan", "code.commit", repo="dan/vec-fork", files={"lib.py": R.VEC_LIB + "\n\ndef dot(a, b):\n    return sum(x * y for x, y in zip(a, b))\n"})
    pr = must(w, "dan", "pkg.pr_open", repo="eve/vecmath", source_repo="dan/vec-fork", title="add dot")["pr"]
    assert not ex(w, "dan", "pkg.pr_review", pr=pr, verdict="approve").ok                # the author
    assert not ex(w, "ben", "pkg.pr_review", pr=pr, verdict="approve").ok                # has not read it
    # a reviewer from the author's own organisation is recorded but does not count
    must(w, "dan", "id.create_org", handle="danco", name="Dan Co")
    inv = must(w, "dan", "id.invite", org="o:danco", person="p:cat", role="member")
    must(w, "cat", "id.respond_invite", invite=inv["invite"], accept=True)
    must(w, "cat", "pkg.pr_diff", pr=pr)
    rv = must(w, "cat", "pkg.pr_review", pr=pr, verdict="approve")
    assert not w.state.get("pkg_reviews", rv["review"])["independent"]
    assert not ex(w, "eve", "pkg.pr_merge", pr=pr).ok
    must(w, "ben", "pkg.pr_diff", pr=pr)
    must(w, "ben", "pkg.pr_review", pr=pr, verdict="approve", comment="ok")
    assert not ex(w, "ben", "pkg.pr_merge", pr=pr).ok                                     # not the maintainer
    m = must(w, "eve", "pkg.pr_merge", pr=pr)
    assert m["merged"] and "dot" in w.query("p:eve", "code.read", {"repo": "eve/vecmath", "path": "lib.py"})["content"]
    rel = must(w, "eve", "pkg.release", package="vecmath", version="1.1.0")
    assert rel["compat"]["added"] == ["dot"] and not rel["compat"]["breaking"] and rel["compat"]["semver_ok"]


def test_advisories_are_verified_only_by_a_failing_reproduction(world):
    w = world
    must(w, "cat", "code.create_repo", name="audit", visibility="public", license="open")
    must(w, "cat", "code.commit", repo="cat/audit", files={
        "tests/test_ok.py": "from deps.vecmath.lib import norm\n\n\ndef test_norm():\n    assert norm([3, 4]) == 5\n"})
    r = must(w, "cat", "pkg.advisory", package="vecmath", version="1.0.0", kind="reliability", severity="low",
             description="claims norm is wrong", repro_repo="cat/audit")
    assert not r["verified"] and r["affected"] == []
    ev = [e for e in w.log if e.type == "pkg.advisory"][-1]
    assert ev.vis != "public"                                                              # unverified: not published
    must(w, "cat", "code.commit", repo="cat/audit", files={
        "tests/test_ok.py": "from deps.vecmath.lib import norm\n\n\ndef test_norm_of_empty_is_defined():\n"
                            "    assert norm([]) == 0 and norm(['x']) is None\n"})
    r = must(w, "cat", "pkg.advisory", package="vecmath", version="1.0.0", kind="reliability", severity="low",
             description="norm raises on non-numbers", repro_repo="cat/audit")
    assert r["verified"] and r["affected"] == ["1.0.0"]


def test_full_software_history_has_all_mechanisms_and_replays_exactly():
    w = World.create(innovation_spec(3, with_adopters=False))
    try:
        out = R.software_history(w)
        f = M.software_features(w)
        assert f["count"] == 15 and f["transitive_dependencies"], f
        assert out["geokit_1_1_0"]["compat"]["breaking"] and not out["geokit_1_1_0"]["compat"]["semver_ok"]
        assert not out["ben_upgrade_breaking"]["upgraded"] and out["ben_upgrade"]["upgraded"]
        assert out["advisory"]["verified"] and out["geokit_1_0_1"]["fixes_advisories"] == [out["advisory"]["advisory"]]
        assert {a["package"] for a in M.adoptions_by_execution(w)} == {"geokit", "vecmath"}   # vecmath transitively
        need = [n for _, n in w.state.scan("needs_open") if n["target"] == "pkg:geokit"]
        assert need and need[0]["status"] == "resolved" and need[0]["holders_ever"] == ["p:ben"]
        assert w.check_invariants() == []
        evs = [e.to_dict() for e in w.log]
        assert World.replay(w.spec, evs).digest() == w.digest()
        g = World.regenerate(w.spec, evs)
        assert g.last_hash == w.last_hash
        g.close()
    finally:
        w.close()
