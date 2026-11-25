"""Needs graph (module needs): relations read off world state, generic derivation rules, holders, visibility,
addressing by new services and resolution."""
import ast
import os

import pytest

import crucible

crucible.load_all_modules()
from crucible.core.kernel import World  # noqa: E402
from crucible.innovation import measure as M  # noqa: E402
from crucible.innovation import scenarios as SC  # noqa: E402
from crucible.innovation.needs import RELATIONS, special_case_triggers  # noqa: E402
from crucible.innovation.reference import ServiceCreator, drive, join  # noqa: E402
from crucible.innovation.worlds import innovation_spec  # noqa: E402

HOUR, DAY = 3600, 86400


def test_structural_graph_is_read_off_world_state():
    w = World.create(innovation_spec(8, participants=6, adopt={"start_hours": [1, 4]}))
    try:
        a = ServiceCreator("ana", "route.shortest", "anaroute", at=HOUR)
        b = ServiceCreator("bo", "route.shortest", "boroute", at=HOUR)
        drive(w, [a, b], DAY)
        nodes, edges = w.modules["needs"].structural_graph(w)
        rels = {(e["src"], e["rel"], e["dst"]) for e in edges.values()}
        inst = w.state.get("deployments", w.state.get("hosts", "anaroute")["deployment"])["instance"]
        assert ("svc:anaroute", "consumes", f"compute:{inst}") in rels
        assert (f"capacity:{inst}", "constrains", f"compute:{inst}") in rels
        assert ("svc:anaroute", "produces", "cap:route.shortest") in rels
        assert ("svc:anaroute", "substitutes", "svc:boroute") in rels
        assert ("cap:route.shortest", "complements", "cap:route.render") in rels     # schemas: path -> path
        users = [s for s, r, d in rels if r == "requires" and d == "cap:route.shortest"]
        assert len(users) == 6                                                      # every participant's task
        assert {nodes[n]["kind"] for n in nodes} <= {"activity", "resource", "capability", "constraint", "need"}
        assert {e["rel"] for e in edges.values()} <= set(RELATIONS)
        # the stored graph equals the derived one after an update
        stored = set(w.state.ids("needs_edges"))
        assert stored == set(edges)
    finally:
        w.close()


@pytest.fixture(scope="module")
def cap():
    r = SC.capacity(counterfactual=False)
    yield r
    SC.close(r)


def test_growing_use_creates_a_supply_need_that_a_new_service_addresses(cap):
    w = cap["world"]
    sup = [n for _, n in w.state.scan("needs_open") if n["kind"] == "supply"]
    assert sup, "no supply need emerged"
    n = sup[0]
    assert [e["rel"] for e in n["path"]] == ["consumes", "constrains", "produces"] and n["rule"] == "consumes"
    assert n["evidence"]["unmet_24h"] > 0
    users = [h for h in n["holders_ever"] if h.startswith("p:u")]
    assert len(users) >= 5                                            # the participants relying on the service
    adr = M.needs_addressed_by_new_services(w)
    assert adr and adr[0]["host"] == "boroute" and adr[0]["holders_adopting"] >= 5
    assert n["status"] == "resolved" and n["resolved_t"] > n["addressed_t"] > n["t"]
    assert special_case_triggers(w)["without_valid_path"] == 0
    assert w.check_invariants() == []


def test_need_events_are_visible_to_holders_only(cap):
    w = cap["world"]
    derived = {}
    for ev in w.log:
        if ev.type == "needs.derived" and ev.payload["holders"]:
            assert ev.vis == tuple(sorted(ev.payload["holders"])) and "evidence" not in ev.payload
            derived[ev.payload["id"]] = ev.payload["holders"]
        if ev.type == "needs.evidence":                   # operators' metrics never travel in visible events
            assert ev.vis == "internal"
    for nid, n in w.state.scan("needs_open"):
        for h in n["holders"]:                    # current holders see it (a user who switched away no longer holds it)
            assert nid in {x["need"] for x in w.query(h, "needs.mine", {"status": "any"})}
    n = next(n for _, n in w.state.scan("needs_open") if n["kind"] == "supply")
    for op, own in n["own"].items():                                       # each operator sees only its own metrics
        if op in n["holders"]:
            mine_op = [x for x in w.query(op, "needs.mine", {"status": "any"}) if x["need"] == n["id"]]
            assert mine_op and mine_op[0]["evidence"] == dict(own) and own["activity"].startswith("svc:")
            assert w.state.get("needs_nodes", own["activity"])["owner"] == op
    holder = next(h for _, x in w.state.scan("needs_open") for h, r in x["holders"].items() if r == "user")
    join(w, "stranger", 10)
    assert w.query("p:stranger", "needs.mine", {"status": "any"}) == []
    assert not w.execute("p:stranger", {"tool": "needs.publish", "args": {"need": n["id"]}}).ok
    g = w.query("p:stranger", "needs.graph", {})
    assert g["edges"] and not any(e["src"].startswith("user:") for e in g["edges"])
    mine = w.query(holder, "needs.mine", {"status": "any"})
    assert all(x["evidence"] == {} for x in mine if "operator" not in x["roles"])    # no operator metrics leak
    req = w.query("p:stranger", "needs.requests", {})
    assert req and all(set(r) == {"need", "kind", "target", "requesters", "first", "texts", "status"} for r in req)


def test_derivation_rules_name_no_particular_entity():
    """The rule code is keyed by relation type only: no string literal in needs.py names a host, capability,
    package, principal or format of any world."""
    import crucible.innovation.needs as N
    src = open(N.__file__).read()
    tree = ast.parse(src)
    consts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    from crucible.economy.needs import NEEDS
    from crucible.innovation.formats import FORMATS
    named = set(NEEDS) | set(FORMATS) | {"anaroute", "boroute", "geokit", "vecmath", "route.render"}
    assert not consts & named
    assert not any(c.startswith(("p:", "o:", "svc:")) and len(c) > 4 for c in consts)


def test_every_relation_type_has_a_rule_or_an_addressing_role():
    import crucible.innovation.needs as N
    src = open(N.__file__).read()
    for rel in RELATIONS:
        assert src.count(f'"{rel}"') >= 2, rel
    assert os.path.basename(N.__file__) == "needs.py"
