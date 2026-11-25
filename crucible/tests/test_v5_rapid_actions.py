"""The structured action interface and the Observation shape of Tier 1."""
from __future__ import annotations

import json

import pytest

from crucible.rapid import actions as A
from crucible.rapid import families as F
from crucible.rapid import observation as O


def space():
    return A.ActionSpace({
        "levers": A.levers_schema([{"name": "zu47", "min": 0.0, "max": 10.0, "default": 1.0}]),
        "discrete": A.discrete_schema({"pick": ["ka", "mo"], "other": ["xi"]}),
        "graph_edit": A.graph_edit_schema(["add_edge", "remove_edge"], ["n1", "n2", "n3"]),
        "query": A.query_schema({"probe": "a probe"}),
        "experiment": A.experiment_schema([{"name": "zu47", "min": 0.0, "max": 6.0}], ["be12"], 3, 4, 1.0),
        "message": A.message_schema(["journal", "a1"]),
        "program": A.program_schema(10),
        "allocate": A.allocate_schema(["s1", "s2"], 5.0),
        "coordinate": A.coordinate_schema(["a1"], {"cell": A.S.s_enum(["c0_0", "c1_1"])}),
        "sequence": A.sequence_schema(["levers", "discrete", "query"]),
    }, {"allocate_total": 5.0, "decisions": {"pick": ["ka", "mo"], "other": ["xi"]},
        "sequence_kinds": ["levers", "discrete", "query"], "parties": ["a1"]})


GOOD = [
    {"kind": "levers", "values": {"zu47": 3.5}},
    {"kind": "discrete", "decision": "pick", "choice": "mo"},
    {"kind": "graph_edit", "op": "add_edge", "src": "n1", "dst": "n2"},
    {"kind": "query", "target": "probe", "args": {"cell": "c0_0"}},
    {"kind": "experiment", "design": {"set": {"zu47": 2.0}, "measure": ["be12"], "ticks": 2, "replicates": 3}},
    {"kind": "message", "to": "a1", "text": "hi", "content": {"x": 1}},
    {"kind": "program", "source": "def act(observation, caller):\n    return None\n", "ticks": 3},
    {"kind": "allocate", "amounts": {"s1": 2.0, "s2": 3.0}},
    {"kind": "coordinate", "with": ["a1"], "plan": {"cell": "c1_1"}},
    {"kind": "sequence", "steps": [{"kind": "levers", "values": {"zu47": 1.0}},
                                   {"kind": "discrete", "decision": "other", "choice": "xi"}]},
]
BAD = [
    ({"kind": "levers", "values": {"zu47": 11.0}}, "above maximum"),
    ({"kind": "levers", "values": {"nope": 1.0}}, "unexpected"),
    ({"kind": "discrete", "decision": "other", "choice": "ka"}, "not an option"),
    ({"kind": "graph_edit", "op": "add_edge", "src": "n1"}, "dst"),
    ({"kind": "graph_edit", "op": "add_edge", "src": "n1", "dst": "n1"}, "differ"),
    ({"kind": "experiment", "design": {"set": {"zu47": 9.0}, "measure": ["be12"]}}, "maximum"),
    ({"kind": "program", "source": "def act(:\n", "ticks": 3}, "syntax"),
    ({"kind": "program", "source": "x = 1\n", "ticks": 3}, "act"),
    ({"kind": "allocate", "amounts": {"s1": 4.0, "s2": 3.0}}, "exceeds"),
    ({"kind": "coordinate", "with": ["a9"], "plan": {"cell": "c1_1"}}, "a9"),
    ({"kind": "sequence", "steps": [{"kind": "sequence", "steps": []}]}, "kind must be"),
    ({"kind": "sequence", "steps": [{"kind": "program", "source": "def act(o, c): pass", "ticks": 1}]}, "kind must"),
    ({"kind": "teleport"}, "not available"),
    ("levers", "JSON object"),
]


def test_every_kind_validates_good_actions():
    sp = space()
    assert set(sp.kinds) == set(A.KINDS)
    for a in GOOD:
        assert sp.validate(a) == [], (a, sp.validate(a))
    assert [x["kind"] for x in sp.expand(GOOD[-1])] == ["levers", "discrete"]


@pytest.mark.parametrize("action,why", BAD)
def test_malformed_actions_are_refused_with_a_reason(action, why):
    errs = space().validate(action)
    assert errs and any(why in e for e in errs), errs


def test_unknown_kinds_cannot_be_declared():
    with pytest.raises(ValueError):
        A.ActionSpace({"teleport": {"type": "object"}})


def test_tools_are_mcp_shaped_and_json():
    tools = space().tools()
    assert {t["name"] for t in tools} == set(A.KINDS)
    for t in tools:
        assert t["inputSchema"]["type"] == "object" and t["description"]
    json.dumps(tools)


def test_spec_coverage_maps_the_ten_part_8_11_types():
    assert len(A.SPEC_ACTION_TYPES) == 10
    cov = A.spec_coverage([space()])
    assert cov["tool discovery"] is False                       # this space has no 'tools' query target
    sp2 = A.ActionSpace({"query": A.query_schema({"tools": "the catalogue"})})
    assert A.spec_coverage([space(), sp2])["tool discovery"] is True


def test_observations_have_every_section_11_field_and_no_evaluator_keys():
    for fam in ("hc_confounded", "coop_split_info", "srch_field", "des_bottleneck", "posm_fault_diagnosis"):
        env = F.make(fam, 0)
        obs = env.reset()
        assert O.check(obs) == [] and tuple(obs) == O.FIELDS
        assert obs["tools"] and obs["objective"]
        r = env.step(None)
        assert O.check(r["observation"]) == []
    bad = O.make(t=0, agent="a0", numbers={"x": float("nan"), "y": 1.0})
    assert bad["missing"] == ["x"] and bad["numbers"] == {"y": 1.0}
    leak = O.make(t=0, agent="a0", analytics={"oracle_value": 3.0})
    assert any("oracle" in e for e in O.check(leak))


def test_history_is_given_as_arrays_at_reset():
    env = F.make("hc_confounded", 0)
    obs = env.reset()
    hist = {k: v for k, v in obs["arrays"].items() if k.startswith("history.")}
    assert len(hist) == 4 and all(len(v) == 40 for v in hist.values())


def test_invalid_actions_cost_a_tick_and_change_nothing_else():
    env = F.make("hc_confounded", 0)
    before = env._state["lev"].copy()
    r = env.step({"kind": "levers", "values": {"nope": 1.0}})
    assert r["result"]["ok"] is False and env.t == 1 and (env._state["lev"] == before).all()
    assert env.outcome()["invalid_actions"]["a0"] == 1
