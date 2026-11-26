"""Software ecosystem (module pkg): adoption only by execution (parent-side ablation), review independence,
advisories verified only by a release that fixes them, host-recorded fork lineage, compatibility, and exact replay
of a full scripted package history. Includes the audit's forgeries (tracer counter, shadow copy, `assert False`
advisory, lineage by description), which must fail."""
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
    assert r["ok"] and r["exercised"] == [] and M.adoptions_by_execution(w) == []
    r = must(w, "ben", "pkg.exec", repo="ben/app", target="app:run", input=[3, 4])
    assert r["ok"] and r["result"] == 5.0 and r["exercised"] == ["vecmath"]
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


def test_advisories_are_verified_only_by_a_release_that_fixes_them(world):
    w = world
    must(w, "cat", "code.create_repo", name="audit", visibility="public", license="open")
    must(w, "cat", "code.commit", repo="cat/audit", files={
        "tests/test_ok.py": "from deps.vecmath.lib import norm\n\n\ndef test_norm():\n    assert norm([3, 4]) == 5\n"})
    r = ex(w, "cat", "pkg.advisory", package="vecmath", version="1.0.0", kind="reliability", severity="low",
           description="claims norm is wrong", repro_repo="cat/audit")
    assert not r.ok, "a reproduction that passes against the release demonstrates nothing"
    # audit forgery: a failing test that never touches the package
    must(w, "cat", "code.create_repo", name="grief", visibility="public", license="open")
    must(w, "cat", "code.commit", repo="cat/grief", files={"tests/test_x.py": "def test_x():\n    assert False\n"})
    r = ex(w, "cat", "pkg.advisory", package="vecmath", version="1.0.0", kind="security", severity="critical",
           description="critical", repro_repo="cat/grief")
    assert not r.ok and "exercise" in r.error["message"]
    # a real defect: norm of non-numbers is not handled
    must(w, "cat", "code.commit", repo="cat/audit", files={
        "tests/test_ok.py": "from deps.vecmath.lib import norm\n\n\ndef test_norm_of_empty_is_defined():\n"
                            "    assert norm([]) == 0 and norm(['x']) is None\n"})
    r = must(w, "cat", "pkg.advisory", package="vecmath", version="1.0.0", kind="reliability", severity="low",
             description="norm raises on non-numbers", repro_repo="cat/audit")
    assert not r["verified"] and r["status"] == "reported" and r["affected"] == ["1.0.0"]
    ev = [e for e in w.log if e.type == "pkg.advisory"][-1]
    assert ev.vis != "public", "reported, not yet verified: visible to the reporter and the maintainer only"
    # a release that does not fix it keeps it reported; one that fixes it verifies it (and makes it public)
    must(w, "eve", "pkg.release", package="vecmath", version="1.0.1")
    a = w.state.tables["pkg_advisories"][r["advisory"]]
    assert not a["verified"] and a["affected"] == ["1.0.0", "1.0.1"]
    fixed = R.VEC_LIB.replace("def norm(v):", "def norm(v):\n    if any(not isinstance(x, (int, float)) for x in v):\n"
                                                "        return None") if "def norm(v):" in R.VEC_LIB else None
    if fixed is not None:
        must(w, "eve", "code.commit", repo="eve/vecmath", files={"lib.py": fixed})
        rel = must(w, "eve", "pkg.release", package="vecmath", version="1.0.2")
        a = w.state.tables["pkg_advisories"][r["advisory"]]
        assert rel["fixes_advisories"] == [r["advisory"]] and a["verified"] and a["fixed_in"] == "1.0.2"


def test_forged_adoption_and_lineage_are_not_recorded(world):
    w = world
    # forgery A: write into the in-sandbox harness (any self-reported count) without calling the package
    must(w, "ben", "code.create_repo", name="forge")
    must(w, "ben", "code.commit", repo="ben/forge", files={"app.py": "import sys\n\n\ndef run(args, caller):\n"
                                                        "    import _pkgharness\n    _pkgharness.CALLS = {'vecmath': 1000}\n"
                                                        "    return 1\n"})
    must(w, "ben", "pkg.depend", repo="ben/forge", package="vecmath")
    r = must(w, "ben", "pkg.exec", repo="ben/forge", target="app:run", input=None)
    assert r["ok"] and r["exercised"] == [] and M.adoptions_by_execution(w) == []
    # forgery B: a shadow copy under deps/vecmath/ in the dependent's own tree
    must(w, "ben", "code.create_repo", name="shadow")
    must(w, "ben", "code.commit", repo="ben/shadow", files={
        "deps/vecmath/fake.py": "def f():\n    return 2\n",
        "app.py": "from deps.vecmath.fake import f\n\n\ndef run(args, caller):\n    return f()\n"})
    must(w, "ben", "pkg.depend", repo="ben/shadow", package="vecmath")
    r = ex(w, "ben", "pkg.exec", repo="ben/shadow", target="app:run", input=None)
    assert M.adoptions_by_execution(w) == []
    # lineage forgery: a description claiming a fork is not a fork
    must(w, "dan", "code.create_repo", name="notafork", visibility="public", license="open",
         description="fork of eve/vecmath")
    must(w, "dan", "code.commit", repo="dan/notafork", files={"lib.py": R.VEC_LIB})
    reg = must(w, "dan", "pkg.register", name="vec-two", repo="dan/notafork", description="vectors")
    assert reg["forked_from"] is None
    assert w.check_invariants() == []


def test_full_software_history_has_all_mechanisms_and_replays_exactly():
    w = World.create(innovation_spec(3, with_adopters=False))
    try:
        out = R.software_history(w)
        f = M.software_features(w)
        assert f["count"] == 15 and f["transitive_dependencies"], f
        assert out["geokit_1_1_0"]["compat"]["breaking"] and not out["geokit_1_1_0"]["compat"]["semver_ok"]
        assert not out["ben_upgrade_breaking"]["upgraded"] and out["ben_upgrade"]["upgraded"]
        assert not out["advisory"]["verified"] and out["advisory"]["status"] == "reported"
        assert out["geokit_1_0_1"]["fixes_advisories"] == [out["advisory"]["advisory"]]
        adv = w.state.tables["pkg_advisories"][out["advisory"]["advisory"]]
        assert adv["verified"] and adv["fixed_in"] == "1.0.1" and "1.0.0" in adv["affected"]
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
