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
    # its own projects, notes and accounts (and, in check_invariants only, the hosts of live projects)
    assert tables <= {"dev_projects", "dev_worked", "accounts", "hosts"}, tables
    tab = {s.value for n in ast.walk(tree) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute)
           and n.value.attr == "tables" for s in [n.slice] if isinstance(s, ast.Constant)}
    assert tab <= {"dev_projects", "accounts"}, tab


def test_developers_learn_undocumented_formats_from_worked_examples(world):
    """FR-DEV-1 (verifier repair pass: "each program is a designer reference tier; the best tier implements the
    generated category's hidden rule"): a developer's program for a task known only through worked examples is
    written from what it induced from the worked examples requesters posted (its own notes): it covers only the
    feeds its notes show, is none of the reference solvers, and was measured on its notes in the sandbox."""
    from crucible.usefulness import category as C
    w = world
    st = w.state
    posts = [e for e in w.log if e.type == "forum.post" and "What I make of my example by hand: " in e.payload["body"]]
    assert posts and all(e.payload["title"].startswith(D.ASK) for e in posts)
    induced = [(pid, r) for pid, r in st.scan("dev_projects") if r.get("kind") in ("induced", "mixed")
               and r["status"] == "live" and r.get("prog")]
    assert len(induced) >= 2, [(r["task"][:30], r.get("kind")) for _, r in st.scan("dev_projects")]
    for pid, r in induced:
        prog = r["prog"] if r["kind"] == "induced" else r["prog"]["a"]
        fmt = r["need"] if r["kind"] == "induced" else C.category_of(r["need"]).parts[0].name
        notes = st.get("dev_worked", f"{r['dev']}|{fmt}")
        seen = {row["src"] for pr in notes["pairs"] for row in pr["input"]["rows"]}
        assert set(prog["rule"]) <= seen, (pid, sorted(prog["rule"]), sorted(seen))
        svc = w.modules["code"].files(st, r["repo"])["service.py"]
        assert all(src not in svc and repr(src) not in svc for _, src in C.category_of(fmt).tiers)
    # versions were measured on the developers' own examples in the sandbox (bench repositories, pkg.exec)
    benches = [x for _, x in st.scan("pkg_execs") if x["repo"].endswith("bench")]
    assert benches and all(ids.index(x["owner"]) is not None for x in benches)


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



def test_a_developer_reply_says_what_the_service_does_and_ranks_above_off_topic_replies(world):
    """Regression (civilisation run at 7bc3ddc: a workflow service nobody ever tried): the search document of a
    reply carries the request's title ("Re: Looking for a service: <task>"), so a reply that only named a host ranked
    below every short off-topic reply other people wrote under the same title and the individuals searching the
    task never saw it. A developer's reply says what the service does in the words of the request."""
    w = world.fork("t-replies")
    try:
        st = w.state
        mine = [e for e in world.log if e.type == "forum.reply" and "I built a service for this" in e.payload["body"]]
        assert mine
        for e in mine:
            task = st.get("forum_posts", e.payload["post"])["title"][len(D.ASK):]
            assert f"It does what you asked for: {task}." in e.payload["body"] and e.payload["hosts"]
        e = mine[-1]
        post = st.get("forum_posts", e.payload["post"])
        task, host = post["title"][len(D.ASK):], e.payload["hosts"][0]
        dev = set(int(i) for i in columns.col(w, "devs", "dev").nonzero()[0])
        others = [ids.pid(j) for j in range(200) if j not in dev and ids.pid(j) != post["author"]][:10]
        for k, q in enumerate(others):
            w.system_call("forum.reply", {"post": e.payload["post"], "body": f"Nice weather here today {k}."},
                          as_principal=q)
        # what an individual searching for the task types: the words of its own request. Several developers may
        # serve the task (their replies are as relevant); none of the off-topic replies ranks above this one's
        sp, org, _ = w.modules["search"].search(w, task, None, 40)
        rank = next((k for k, r in enumerate(org) if host in (r["hosts"] or [])), None)
        off = [k for k, r in enumerate(org) if "Nice weather" in (r.get("snippet") or "")]
        assert rank is not None and rank < 8 and all(rank < k for k in off), \
            (rank, off, [(r["hosts"], r["snippet"][:40]) for r in org[:10]])
    finally:
        w.close()


def test_a_workflow_budget_grows_with_its_own_profitable_use():
    """The workflow's service pays its components under a 30-day spending grant (a fixed cap would make every call
    fail with 402 once a few hundred calls were made, and the users drop it): the developer reads its own statement
    and grants more room only when the components' charges approach the cap and the workflow's own sales cover
    them."""
    now = 40 * DAY

    def lines(n_calls, days, comp_price=20, price=52, refunds=0):
        out = []
        for k in range(n_calls):
            t = now - int(days * DAY * (k + 0.5) / n_calls)
            out.append({"t": t, "amount": price, "kind": "sale", "ref": "svc:wf", "counterparties": ["a_x"]})
            for h in ("c1", "c2"):
                out.append({"t": t, "amount": -comp_price, "kind": "sale", "ref": f"svc:{h}", "counterparties": ["a_y"]})
        for k in range(refunds):
            out.append({"t": now - DAY, "amount": -price, "kind": "refund", "ref": "svc:wf", "counterparties": ["a_x"]})
        return sorted(out, key=lambda x: -x["t"])
    room = D.Developers.budget_room
    assert room(lines(20, 10), "wf", ["c1", "c2"], 16000, now) == 0                 # 800c of 16000: plenty of room
    # a busy workflow: 200 calls in 2 days, the statement shows its newest 500 lines (about 1.7 days):
    # ~120,000c of component charges projected over 30 days against a 16,000c cap
    busy = lines(200, 2)[:500]
    add = room(busy, "wf", ["c1", "c2"], 16000, now)
    assert 16000 + add >= 1.5 * 100000
    assert room(lines(160, 2), "wf", ["c1", "c2"], 16000, now) == 0                 # all of it shown: 6,400c
    assert room(lines(200, 2, comp_price=60)[:500], "wf", ["c1", "c2"], 16000, now) == 0   # loses money per call
    assert room(busy, "wf", [], 16000, now) == 0 and room([], "wf", ["c1"], 16000, now) == 0
    stale = [dict(x, t=x["t"] - 40 * DAY) for x in busy]
    assert room(stale, "wf", ["c1", "c2"], 16000, now) == 0                         # nothing in the last 30 days

@pytest.fixture(scope="module")
def arm5():
    from crucible.innovation import civexp as CX
    r = CX.run_arm("creation", "factual", 5, days=24, n=300)
    yield r
    r["world"].close()


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_a_developer_switches_a_weak_two_stage_service_to_a_workflow_of_independent_services(arm5):
    """A developer whose own program for the composed task converts poorly finds, through the registry's public
    counts, services of two other providers that people keep for the two stages and rebuilds its service as a
    workflow on the same host and declaration; the registry verifies it from the service's own calls."""
    w = arm5["world"]
    if True:
        st = w.state
        wfs = [x for _, x in st.scan("cap_workflows")]
        # a workflow is verified by the registry once its service's own calls to the components are observed; one
        # switched near the end of the run may still be declared (not called yet)
        assert all(x["status"] in ("verified", "declared") for x in wfs)
        verified = [x for x in wfs if x["status"] == "verified"]
        assert verified
        for x in verified:
            proj = next(p for _, p in st.scan("dev_projects") if p["host"] == x["host"])
            assert proj["approach"] == -1 and proj["composes"] == [c["declaration"] for c in x["components"]]
            assert proj["decl"] == x["declaration"] and x["t"] > proj["t"]          # built first, switched later
            assert all(M.independent(w, x["composer"], c["provider"]) for c in x["components"])
            assert len({c["provider"] for c in x["components"]}) == 2
        assert not w.check_invariants()


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_developers_collaborate_through_issues_pull_requests_review_and_forks(arm5):
    """Software collaboration among the autonomous developers (SP-6.5 in a world with nobody scripted): a dependent
    whose service does poorly on a package proposes its better approach as a pull request from a fork (and links
    it from an issue), the maintainer reads and reviews it (an independent review), merges it when its tests pass
    and the approach is better, and releases; dependents upgrade (compatibility recorded); issues are closed only
    after a later release; support tickets are resolved only after a change shipped."""
    w = arm5["world"]
    st = w.state
    prs = [p for _, p in st.scan("pkg_prs")]
    assert prs, "no pull request"
    pkgs = {p["repo"]: (n, p) for n, p in st.scan("pkg_packages")}
    for pr in prs:
        assert ids.index(pr["author"]) is not None                  # a population developer
        assert pkgs[pr["repo"]][1]["owner"] != pr["author"]         # proposed to someone else's package
    revs = [r for _, r in st.scan("pkg_reviews")]
    assert revs and all(r["independent"] and M.independent(w, r["reviewer"], st.tables["pkg_prs"][r["pr"]]["author"])
                        for r in revs)
    merged = [k for k, p in st.scan("pkg_prs") if p["status"] == "merged"]
    assert merged
    for k in merged:
        pr = st.tables["pkg_prs"][k]
        name = pkgs[pr["repo"]][0]
        assert any(r["t"] >= pr["merged_t"] for r in w.modules["pkg"].releases(st, name))   # released after merging
    assert int(st.meta_get("pkg:forks", 0)) >= 1
    # issues on package trackers: filed by dependents, closed only after a release that followed them
    projects = {p["project"]: n for n, p in st.scan("pkg_packages")}
    issues = [i for i in st.tables.get("iss_issues", {}).values() if i["project"] in projects]
    assert issues and all(ids.index(i["creator"]) is not None for i in issues)
    for i in issues:
        if i["status"] == "closed":
            rels = w.modules["pkg"].releases(st, projects[i["project"]])
            assert any(r["t"] > i["t"] for r in rels)
    # tickets: resolved only after a change to the service was shipped after the ticket was opened
    changes = {}
    for e in w.log:
        if e.type == "devs.project_update" and e.payload.get("changed_t") is not None:
            host = st.tables["dev_projects"][e.payload["id"]]["host"]
            changes.setdefault(host, []).append(int(e.payload["changed_t"]))
    tickets = [t for t in st.tables.get("sup_tickets", {}).values()
               if any(p["host"] == t["host"] for _, p in st.scan("dev_projects"))]
    for t in tickets:
        if t["status"] == "resolved":
            assert any(t["opened"] < c <= t["resolved"] for c in changes.get(t["host"], [])), t
    f = M.software_features(w)
    assert f["evidence"]["pull_requests"] >= 1 and f["evidence"]["independent_review"] >= 1
    assert f["evidence"]["issue_tracker"] >= 1 and f["evidence"]["forks"] >= 1 and f["evidence"]["maintenance"] >= 1
    assert not w.check_invariants()
