"""Autonomous developers of the one population (module devs, wave 3b): they build from public requests, are
adopted by independent individuals for their own tasks, publish and reuse packages (adoption by execution across the
attested package boundary), maintain from their own observables, never read evaluator-side or hidden state, and the
world replays exactly. Measured in the ECON creation world with developers (`innovation.inventworld.creation_spec`)
for speed; the gates measure the same module in the integrated civilisation."""
import ast
import inspect

import pytest

import crucible

crucible.load_all_modules()
from crucible.core.kernel import World  # noqa: E402
from crucible.innovation import devs as D  # noqa: E402
from crucible.innovation import inventworld as IW  # noqa: E402
from crucible.innovation import measure as M  # noqa: E402
from crucible.substrate import columns, ids  # noqa: E402

DAY = 86400


def _run(seed, days, n=400, **devs):
    w = World.create(IW.creation_spec(seed, n=n, devs=devs or None))
    for d in range(1, days + 1):
        w.advance_time(d * DAY)
    return w


@pytest.fixture(scope="module")
def world():
    w = _run(3, 12)
    yield w
    w.close()


def test_developers_build_from_public_requests_and_independent_individuals_adopt(world):
    w = world
    st = w.state
    devs = set(int(i) for i in columns.col(w, "devs", "dev").nonzero()[0])
    projs = [r for _, r in st.scan("dev_projects")]
    assert len(projs) >= 3 and all(r["dev"] in devs for r in projs)
    prods = [p for p in M.agent_products(w) if p["kind"] == "service" and p["creator_kind"] == "population"]
    assert len(prods) >= 3
    used = [p for p in prods if len(M.users(w, p)) >= 3]
    assert len(used) >= 2, [(p["id"], len(M.users(w, p))) for p in prods]
    # every user counted is an individual of the population, independent of the developer, who ran it on its own
    # input (work_exp uses), and some of them pay per call from their own accounts
    p = max(used, key=lambda x: len(M.users(w, x)))
    assert all(ids.index(u) is not None and u != p["creator"] for u in M.users(w, p))
    assert M.independent_transactions(w, p)["n"] >= 1
    # each developer's tasks come from public requests: a request for every task built before it was built
    for r in projs:
        asked = [e.t for e in w.log if e.type == "forum.post" and e.payload["title"] == D.ASK + r["task"]]
        assert asked and min(asked) < r["t"]
    assert w.check_invariants() == []


def test_developers_publish_and_reuse_packages_adopted_by_execution(world):
    w = world
    st = w.state
    pk = [p for p in M.agent_products(w) if p["kind"] == "package" and p["creator_kind"] == "population"]
    assert pk and all(p["published"] for p in pk)
    ad = M.adoptions_by_execution(w)
    dev_owners = {f"p{int(i)}" for i in columns.col(w, "devs", "dev").nonzero()[0]}
    auto = [a for a in ad if a["owner"] in dev_owners and st.tables["pkg_packages"][a["package"]]["owner"] in dev_owners]
    assert auto, "a developer's service runs another developer's package (CI across the attested boundary)"
    reusing = [r for _, r in st.scan("dev_projects") if r["uses_package"]]
    assert reusing and all(st.has("pkg_deps", f"{r['repo']}|{r['uses_package']}") for r in reusing)


def test_developers_act_only_on_public_or_own_observables():
    """Static: the developer module never imports evaluator-side code, never scores outputs or reads a
    generator's hidden truth, and reads world state only through tools (ctx.call) - except its own table and the
    reference implementations of the developer-labour model (FR-DEV-1)."""
    src = inspect.getsource(D)
    tree = ast.parse(src)
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any(m and ("measure" in m or "relations" in m or "goals" in m or "integrity" in m) for m in mods)
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not {"score", "generate", "is_related", "label"} & calls
    def state_read(c):          # st.<read>(...) or <x>.state.<read>(...)
        if not (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and c.func.attr in ("find_ids", "find", "scan", "get", "has")):
            return False
        v = c.func.value
        return (isinstance(v, ast.Name) and v.id == "st") or (isinstance(v, ast.Attribute) and v.attr == "state")
    reads = [c for c in ast.walk(tree) if state_read(c)]
    tables = {c.args[0].value for c in reads if c.args and isinstance(c.args[0], ast.Constant)}
    # its own projects and accounts (and, in check_invariants only, the hosts of live projects)
    assert tables <= {"dev_projects", "accounts", "hosts"}, tables
    tab = {s.value for n in ast.walk(tree) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute)
           and n.value.attr == "tables" for s in [n.slice] if isinstance(s, ast.Constant)}
    assert tab <= {"dev_projects", "accounts"}, tab


def test_garbage_programs_are_tried_and_dropped_replay_is_exact():
    w = _run(5, 9, n=300, program="garbage")
    try:
        st = w.state
        prods = [p for p in M.agent_products(w) if p["kind"] == "service"]
        assert prods
        kept = sum(1 for _, r in st.scan("work_exp") if r["prov"].startswith("cdecl_") and r["active"])
        tried = sum(1 for _, r in st.scan("work_exp") if r["prov"].startswith("cdecl_") and r["uses"] > 0)
        assert tried >= 5 and kept <= 0.2 * tried, (kept, tried)
        evs = [e.to_dict() for e in w.log]
        assert World.replay(w.spec, evs).digest() == w.digest()
        assert w.check_invariants() == []
    finally:
        w.close()
