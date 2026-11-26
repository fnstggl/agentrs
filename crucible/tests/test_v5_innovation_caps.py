"""Capability registry v5 (module caps): typed declarations, discovery, trials through the real cloud/sandbox path,
independent verification, adoption only after own execution, and compatibility with the 4.0 registry."""
import pytest

import crucible

crucible.load_all_modules()
from crucible.core.kernel import World  # noqa: E402
from crucible.innovation.reference import ServiceCreator, drive, join  # noqa: E402
from crucible.innovation.scenarios import BROKEN_ROUTE, SUITE_ROUTE  # noqa: E402
from crucible.innovation.worlds import FORMAT_SCHEMAS, innovation_spec  # noqa: E402
from crucible.economy.needs import NEEDS  # noqa: E402
from crucible.worldgen.admission import admit  # noqa: E402

HOUR, DAY = 3600, 86400


@pytest.fixture()
def world():
    w = World.create(innovation_spec(4, with_adopters=False))
    ana = ServiceCreator("ana", "route.shortest", "anaroute", tier=1, price=3)
    ana.act(w)
    assert ana.declaration, ana.log
    for h in ("bob", "tess"):
        join(w, h, 500_000)
    yield w, ana
    w.close()


def ex(w, who, tool, **args):
    return w.execute(f"p:{who}", {"tool": tool, "args": args})


def sample(seed=0):
    import numpy as np
    return NEEDS["route.shortest"].generate(np.random.default_rng(seed), 0.3)


def test_declare_requires_the_operator_a_live_service_and_valid_schemas(world):
    w, ana = world
    cap = {"name": "route.shortest", "doc": "x", **FORMAT_SCHEMAS["route.shortest"]}
    r = ex(w, "bob", "caps.declare", service_host="anaroute", capability=cap)
    assert not r.ok and r.error["code"] == "forbidden"
    r = ex(w, "ana", "caps.declare", service_host="nosuch", capability=cap)
    assert not r.ok and r.error["code"] == "not_found"
    bad = dict(cap, inputs={"type": "object", "oneOf": []})
    r = ex(w, "ana", "caps.declare", service_host="anaroute", capability=bad)
    assert not r.ok and "unsupported keyword" in r.error["message"]
    r = ex(w, "ana", "caps.declare", service_host="anaroute", capability=dict(cap, endpoint="nope"))
    assert not r.ok and "not served" in r.error["message"]
    # re-declaring supersedes; the old declaration leaves the index
    r = ex(w, "ana", "caps.declare", service_host="anaroute", capability=cap)
    assert r.ok and r.result["supersedes"] == [ana.declaration]
    assert w.state.get("cap_decls", ana.declaration)["status"] == "superseded"
    assert not w.state.has("search_docs", f"capdecl:{ana.declaration}")


def test_try_executes_the_real_service_charges_the_caller_and_checks_outputs(world):
    w, ana = world
    inp, hidden = sample()
    acct = w.modules["bank"].primary_account(w.state, "p:bob")
    bal0 = w.state.get("accounts", acct)["balance"]
    cu0 = sum(r["cu"] for _, r in w.state.scan("svc_days"))
    r = ex(w, "bob", "caps.try", service_host="anaroute", capability="route.shortest", input=inp)
    assert r.ok and r.result["status"] == 200 and r.result["ok"]
    assert NEEDS["route.shortest"].score(inp, hidden, r.result["output"]) == 1.0   # dijkstra tier
    assert w.state.get("accounts", acct)["balance"] == bal0 - 3                     # paid the endpoint price
    assert sum(r["cu"] for _, r in w.state.scan("svc_days")) > cu0                  # used the service's compute
    t = w.state.get("cap_trials", r.result["trial"])
    assert t["by"] == "p:bob" and t["ok"] and "independent" not in t     # independence is evaluator-side only
    seq = w.seq
    r = ex(w, "bob", "caps.try", service_host="anaroute", capability="route.shortest", input={"n": 3})
    assert not r.ok and "inputs schema" in r.error["message"]
    assert not any(e.type == "cloud.usage" for e in w.log if e.seq > seq)         # nothing was executed
    # a service whose responses violate its declared outputs
    mal = ServiceCreator("mal", "route.shortest", "malroute", price=0,
                         files={"crucible.json": '{"entry": "service.py", "endpoints": {"solve": {"handler": "solve"}}}',
                                "service.py": "def solve(args, caller):\n    return {'route': []}\n"})
    mal.act(w)
    r = ex(w, "bob", "caps.try", service_host="malroute", capability="route.shortest", input=inp)
    assert r.ok and r.result["status"] == 200 and not r.result["ok"] and r.result["schema_errors"]


def test_adoption_requires_own_successful_execution_and_counts_real_use(world):
    w, ana = world
    r = ex(w, "bob", "caps.adopt_service", declaration=ana.declaration)
    assert not r.ok and r.error["code"] == "forbidden"
    inp, _ = sample(1)
    assert ex(w, "bob", "caps.try", service_host="anaroute", capability=ana.declaration, input=inp).result["ok"]
    a = ex(w, "bob", "caps.adopt_service", declaration=ana.declaration)
    assert a.ok
    assert not ex(w, "bob", "caps.adopt_service", declaration=ana.declaration).ok        # no double adoption
    for k in range(3):
        assert ex(w, "bob", "anaroute.solve", **sample(10 + k)[0]).result["status"] == 200
    rec = w.state.get("cap_adoptions", a.result["adoption"])
    assert rec["uses"] == 3 and rec["uses_ok"] == 3 and "independent" not in rec
    assert w.state.get("cap_decls", ana.declaration)["adopters"] == 1
    # the provider adopting its own service is recorded (public raw count) but the evaluator never counts it
    from crucible.innovation import measure as M
    assert ex(w, "ana", "caps.try", service_host="anaroute", capability=ana.declaration, input=inp).result["ok"]
    own = ex(w, "ana", "caps.adopt_service", declaration=ana.declaration)
    assert own.ok and w.state.get("cap_decls", ana.declaration)["adopters"] == 2
    prod = next(p for p in M.agent_products(w) if p["host"] == "anaroute")
    assert M.adopters(w, prod) == ["p:bob"]
    # nothing a participant or the provider can read carries an independence verdict
    for e in w.log:
        if e.type in ("caps.tried", "caps.adopted", "caps.verification") and e.vis != "internal":
            assert "independent" not in e.payload
    assert ex(w, "bob", "caps.drop_service", declaration=ana.declaration, reason="done").ok
    assert w.state.get("cap_decls", ana.declaration)["adopters"] == 1
    assert w.check_invariants() == []


def test_verification_runs_an_independent_suite_on_fresh_seeds(world):
    w, ana = world
    mal = ServiceCreator("mal", "route.shortest", "malroute", price=0,
                         files={**__import__("crucible.innovation.reference", fromlist=["x"]).service_files(
                             "route.shortest", 1, 0), "service.py": BROKEN_ROUTE})
    mal.act(w)
    assert ex(w, "tess", "code.create_repo", name="suite", visibility="public", license="open").ok
    assert ex(w, "tess", "code.commit", repo="tess/suite", files={"suite.py": SUITE_ROUTE}).ok
    good = ex(w, "tess", "caps.verify", declaration=ana.declaration, suite_repo="tess/suite", cases=6)
    bad = ex(w, "tess", "caps.verify", declaration=mal.declaration, suite_repo="tess/suite", cases=6)
    assert good.ok and bad.ok, (good.error, bad.error)
    assert not good.result["provider_supplied"] and "independent" not in good.result
    assert good.result["mean_score"] == 1.0 and bad.result["mean_score"] < 0.5
    again = ex(w, "tess", "caps.verify", declaration=ana.declaration, suite_repo="tess/suite", cases=6)
    gen = [e for e in w.log if e.type == "ext" for o in e.payload["outcomes"] if o and o["kind"] == "caps.suite"]
    assert again.ok and len(gen) >= 2
    # provider-supplied tests are labelled as such and never independent
    assert ex(w, "ana", "code.create_repo", name="mytests", visibility="public", license="open").ok
    assert ex(w, "ana", "code.commit", repo="ana/mytests", files={"suite.py": SUITE_ROUTE}).ok
    cap = {"name": "route.shortest", "doc": "d", **FORMAT_SCHEMAS["route.shortest"]}
    d2 = ex(w, "ana", "caps.declare", service_host="anaroute", capability=cap, tests_ref="ana/mytests")
    assert d2.ok
    own = ex(w, "bob", "caps.verify", declaration=d2.result["declaration"], cases=3)
    assert own.ok and own.result["provider_supplied"]
    pub = w.query("p:bob", "caps.declaration", {"declaration": d2.result["declaration"]})["public_verifications"]
    assert pub and pub[0]["provider_supplied"]


def test_search_finds_declarations_records_discovery_and_keeps_the_4_0_form(world):
    w, ana = world
    r = ex(w, "bob", "caps.search", query="shortest path route network")
    assert r.ok and r.result and r.result[0]["declaration"] == ana.declaration and r.result[0]["kind"] == "declaration"
    assert w.state.has("cap_seen", f"{ana.declaration}|p:bob") and w.knows("p:bob", "host", "anaroute")
    assert ex(w, "bob", "caps.search", query="anaroute").result[0]["host"] == "anaroute"     # exact host match
    seq = w.seq
    old = ex(w, "bob", "caps.search", q="shortest path")
    assert old.ok and old.result == []                     # 4.0 form: published (installable) capabilities only
    assert not any(e.type == "caps.seen" for e in w.log if e.seq > seq)
    assert not ex(w, "bob", "caps.search").ok


def test_registry_is_opt_in_so_4_0_worlds_keep_their_tools_and_state():
    from crucible.innovation.caps import V5_TABLES
    from crucible.services.capabilities import Capabilities
    spec = innovation_spec(4, with_adopters=False)
    for m in spec.modules:
        if m["name"] == "caps":
            m["config"] = {}
    w = World.create(spec)
    try:
        assert not set(V5_TABLES) & set(w.state.tables)
        # the tool surface of host caps is exactly the 4.0 one (names, schemas, docs, versions)
        want = {}
        for attr in dir(Capabilities):
            e = getattr(getattr(Capabilities, attr), "__crucible_endpoint__", None)
            if e:
                want[f"caps.{e['name']}"] = (e["params"], e["doc"], str(e.get("version") or Capabilities.version))
        got = {n: (ep.params, ep.doc, ep.version) for n, ep in w.endpoints.items() if ep.host == "caps"}
        assert got == want
        join(w, "bob", 1000)
        r = ex(w, "bob", "caps.declare", service_host="xyz",
               capability={"name": "a.b", "inputs": {"type": "object"}, "outputs": {"type": "object"}, "doc": "d"})
        assert not r.ok and r.error["code"] == "unknown_tool"
        assert not ex(w, "bob", "caps.search", query="anything").ok          # 4.0 schema: q only
        assert ex(w, "bob", "caps.search", q="anything").result == []
    finally:
        w.close()


def test_world_with_adopters_replays_regenerates_and_forks_exactly():
    w = World.create(innovation_spec(6, participants=8, adopt={"start_hours": [1, 6]}))
    try:
        ana = ServiceCreator("ana", "route.shortest", "anaroute", tier=1, price=3, at=2 * HOUR)
        drive(w, [ana], int(1.5 * DAY))
        assert w.state.count("cap_adoptions") >= 4 and w.check_invariants() == []
        evs = [e.to_dict() for e in w.log]
        assert World.replay(w.spec, evs).digest() == w.digest()
        g = World.regenerate(w.spec, evs)
        assert g.last_hash == w.last_hash
        g.close()
        f1, f2 = w.fork("f1"), w.fork("f2")
        drive(f1, [], w.now + DAY)
        drive(f2, [], w.now + DAY)
        assert f1.digest() == f2.digest() and f1.last_hash == f2.last_hash
        f1.close()
        f2.close()
    finally:
        w.close()


@pytest.mark.parametrize("mods,cfg", [(["caps"], {"caps": {"registry": True}}),
                                      (["caps", "needs"], {"caps": {"registry": True}}),
                                      (["caps", "pkg"], {"caps": {"registry": True}}),
                                      (["caps", "infra", "adopt"], {"caps": {"registry": True}, "adopt": {"n": 4}})])
def test_modules_pass_admission(mods, cfg):
    rep = admit(mods, base=["identity", "bank", "admin", "code", "search", "forum", "issues", "cloud", "wf"],
                steps=400, configs=cfg)
    assert rep.passed, (rep.checks, rep.errors[:5])
