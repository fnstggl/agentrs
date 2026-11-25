"""SP-6.4 - emerging needs from reusable activity-resource-capability-constraint-need relations, not special-case
triggers (workstream G).

Three worlds driven by the same needs-graph machinery (crucible.innovation.needs), each producing a different
emergence:

* capacity: growing use of a service creates a reliability (supply) bottleneck - consumes + constrains + produces;
  failing producers create an alternative need - requires; a substitute addresses it - substitutes + produces;
  adoption of route.shortest creates demand for a complementary standard interface - complements;
* batch: a technology (route.batch over route.shortest) increases compute demand on another service until its
  constraint binds;
* software: a verified advisory constrains a package version that a dependent requires.

`special_case_need_triggers` counts (a) recorded needs whose rule or path is not made of graph edges of a relation
type, (b) string literals in the rule code that name any entity, format or principal, and (c) differences in the
needs derived when every agent and host of the capacity world is renamed (the rules must not care what things are
called).
"""
import ast

import pytest

import crucible

crucible.load_all_modules()
from crucible.acceptance.v5 import evidence  # noqa: E402
from crucible.innovation import measure as M  # noqa: E402
from crucible.innovation import scenarios as SC  # noqa: E402
from crucible.innovation.needs import RELATIONS, special_case_triggers  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.timeout(2400)]


@pytest.fixture(scope="module")
def worlds():
    r = {"capacity": SC.capacity(counterfactual=False), "batch": SC.batch(counterfactual=False),
         "software": SC.software(),
         "capacity_renamed": SC.capacity(counterfactual=False, names={"ana": "zed", "anaroute": "zedpaths",
                                                                      "bo": "kim", "boroute": "kimpaths"})}
    yield r
    for x in r.values():
        SC.close(x)


def _shape(w, names: dict):
    inv = {v: k for k, v in names.items()}

    def norm(x):
        for a, b in sorted(inv.items(), key=lambda kv: -len(kv[0])):
            x = x.replace(a, b)
        return x
    return [(n["t"], n["kind"], norm(n["target"]), n["rule"], tuple((norm(e["src"]), e["rel"], norm(e["dst"]))
                                                                  for e in n["path"]), n["status"], n["resolved_t"],
             tuple(sorted(norm(h) for h in n["holders_ever"])))
            for _, n in w.state.scan("needs_open")]


def test_SP_6_4_needs_emerge_from_reusable_relations(worlds):
    used: dict[str, int] = {}
    rules: dict[str, int] = {}
    invalid = 0
    emergences = []
    for name in ("capacity", "batch", "software"):
        w = worlds[name]["world"]
        chk = special_case_triggers(w)
        invalid += chk["without_valid_path"]
        for k, v in chk["relations_used"].items():
            used[k] = used.get(k, 0) + v
        for k, v in chk["rules"].items():
            rules[k] = rules.get(k, 0) + v
        for n in M.emerged_needs(w):
            emergences.append(f"{name}: {n['kind']} {n['target']} via {'+'.join(e['rel'] for e in n['path'])}")
    # (b) the rule code names no entity
    import crucible.innovation.needs as N
    from crucible.economy.needs import NEEDS
    from crucible.innovation.formats import FORMATS
    consts = {n.value for n in ast.walk(ast.parse(open(N.__file__).read()))
              if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    entity_names = set(NEEDS) | set(FORMATS) | {"route.render"}
    for name in ("capacity", "batch", "software"):
        st = worlds[name]["world"].state
        entity_names |= set(st.ids("hosts")) | set(st.ids("pkg_packages")) | set(st.ids("principals"))
    literal_hits = sorted(consts & entity_names)
    # (c) renaming every agent and host changes nothing about which needs arise, when, how and for whom
    a = _shape(worlds["capacity"]["world"], {})
    b = _shape(worlds["capacity_renamed"]["world"], worlds["capacity_renamed"]["names"])
    rename_diff = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
    m = {"reusable_need_relations": sum(1 for r in RELATIONS if used.get(r, 0) > 0),
         "special_case_need_triggers": invalid + len(literal_hits) + rename_diff,
         "needs_without_valid_relation_path": invalid, "rule_literals_naming_entities": len(literal_hits),
         "rename_invariance_differences": rename_diff, "emerged_needs": len(emergences),
         "distinct_emergences": len({e.split(" via ")[0].split(": ", 1)[1] + e.split(" via ")[1] for e in emergences}),
         **{f"relation_uses[{r}]": used.get(r, 0) for r in RELATIONS}}
    evidence.record("SP-6.4", m, notes="Relation types counted when they occur in the derivation path of a recorded "
                                       f"need or in the path by which a need was addressed. Rules fired: {rules}. "
                                       f"Emergences: {emergences}. Literal hits: {literal_hits}.")
    assert m["reusable_need_relations"] >= 5
    assert m["special_case_need_triggers"] == 0
    kinds = " ".join(emergences)
    assert "supply cap:route.shortest" in kinds and "pkg:geokit" in kinds and "complements" in kinds
