"""SP-6.4 - emerging needs from reusable activity-resource-capability-constraint-need relations, not special-case
triggers (workstream F+G, D-18).

The instrument is the evaluator-side relation graph (`crucible.innovation.relations`): relation providers read
world state (the population's work activities, cloud deployments, the capability registry, packages) or relations
a world declares as data; generic rules keyed only by relation type and observable names label needs. Nothing here
is visible in-world.

Measured:
* in a world driven by the population (`economy.work`), needs arise from changed conditions: hand work on an
  undocumented task is a bottleneck (`requires` + time share); after a provider automates the first step of a
  two-step activity, the second step becomes the bottleneck (adoption creates a need); a service whose use grows
  past its rented capacity creates a supply need (`consumes` + `constrains` + `produces`);
* generality: relations supplied as data with no code change - a skill an activity requires (education), a
  transport link whose capacity binds (`loads` + `constrains`), storage a new content form fills (`consumes` +
  `constrains`), organisational coordination load (`requires`), a content community's moderation tool, and a
  verified-advisory constraint - are labelled by the same rules;
* `special_case_need_triggers`: needs whose rule is not one of the generic rules or whose path is not made of
  graph edges, plus string literals in the rule code that name an entity, a format or a principal, plus
  differences in the labels when every node of the data scenario is renamed.
"""
import ast
import inspect

import pytest

import crucible

crucible.load_all_modules()
from crucible.acceptance.v5 import evidence  # noqa: E402
from crucible.core.kernel import World, WorldSpec  # noqa: E402
from crucible.innovation import relations as R  # noqa: E402

DAY = 86400


def _data_world(names=None, seed=7):
    nm = (lambda x: x) if names is None else (lambda x: names.get(x, x))
    rels = [
        # education: an activity requires a skill people mostly lack (time spent working around it)
        {"src": nm("activity/analytics"), "src_kind": "activity", "rel": "requires", "dst": nm("skill/python"),
         "dst_kind": "skill", "obs": {"time_share": 0.6, "holders": [nm(f"p{i}") for i in range(5)]}},
        {"src": nm("course/py101"), "src_kind": "product", "rel": "teaches", "dst": nm("skill/python"),
         "dst_kind": "skill"},
        # transport: a market served over a link whose capacity binds
        {"src": nm("service/freight"), "src_kind": "service", "rel": "loads", "dst": nm("transport/route7"),
         "dst_kind": "resource"},
        {"src": nm("capacity/route7"), "src_kind": "constraint", "rel": "constrains", "dst": nm("transport/route7"),
         "dst_kind": "resource"},
        {"src": nm("service/freight"), "src_kind": "service", "rel": "produces", "dst": nm("access/coastal-market"),
         "dst_kind": "capability"},
        # content: a new content form fills the storage a hosting service consumes (its capacity binds)
        {"src": nm("service/clipshost"), "src_kind": "service", "rel": "consumes", "dst": nm("resource/storage"),
         "dst_kind": "resource"},
        {"src": nm("capacity/storage"), "src_kind": "constraint", "rel": "constrains", "dst": nm("resource/storage"),
         "dst_kind": "resource"},
        {"src": nm("service/clipshost"), "src_kind": "service", "rel": "produces", "dst": nm("capability/clip-hosting"),
         "dst_kind": "capability"},
        # coordination: a growing organisation's coordination load
        {"src": nm("activity/acme-ops"), "src_kind": "activity", "rel": "requires", "dst": nm("capability/coordination"),
         "dst_kind": "capability", "obs": {"time_share": 0.45, "holders": [nm("p:acme-a"), nm("p:acme-b")]}},
        # content communities: a new content form whose community lacks a moderation tool (failures of a stopgap)
        {"src": nm("activity/clips-community"), "src_kind": "activity", "rel": "requires",
         "dst": nm("capability/moderation"), "dst_kind": "capability",
         "obs": {"failure_share": 0.4, "holders": [nm("p9"), nm("p10")], "relied_holders": [nm("p9"), nm("p10")]}},
        # software: an advisory constrains a package that two repositories require
        {"src": nm("repo/a/app"), "src_kind": "repository", "rel": "requires", "dst": nm("package/vec"),
         "dst_kind": "package", "obs": {"holders": [nm("p:a")]}},
        {"src": nm("repo/b/app"), "src_kind": "repository", "rel": "requires", "dst": nm("package/vec"),
         "dst_kind": "package", "obs": {"holders": [nm("p:b")]}},
        {"src": nm("advisory/1"), "src_kind": "constraint", "rel": "constrains", "dst": nm("package/vec"),
         "dst_kind": "package", "attrs": {"affects": [nm("repo/a/app"), nm("repo/b/app")]}},
    ]
    nodes = {nm("transport/route7"): {"utilisation": 0.99, "refused_share": 0.2},
             nm("resource/storage"): {"utilisation": 0.97},
             nm("service/clipshost"): {"holders": [nm("p30"), nm("p31")]},
             nm("service/freight"): {"holders": [nm("p20"), nm("p21"), nm("p22")]}}
    spec = WorldSpec("relations-data", seed, [{"name": "identity", "config": {}}],
                     {"relations": rels, "relation_nodes": nodes})
    return World.create(spec)


def _shape(needs, names=None):
    inv = {v: k for k, v in (names or {}).items()}
    return sorted((n["kind"], inv.get(n["target"], n["target"]), n["rule"], len(n["holders"])) for n in needs)


def _literals_naming_entities() -> list[str]:
    """String literals in the rule code (label) that look like entity names (contain '/', ':' or '.')."""
    tree = ast.parse(inspect.getsource(R.label))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value
            if len(v) > 2 and any(c in v for c in "/.:") and " " not in v:      # entity-like names, not prose
                bad.append(v)
    return bad


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_SP_6_4_emerging_needs_from_reusable_relations():
    from crucible.innovation import experiments as X
    # (1) a population world: needs from changed conditions
    res = X.run_arm(11, "factual", n=700, days=20)
    w = res["world"]
    t0 = res["history"][0]
    ever = {}
    for h in res["history"]:
        for nid, n in h["needs"].items():
            ever.setdefault(nid, dict(n, first_t=h["t"]))
    emerged = {k: v for k, v in ever.items() if v["first_t"] > t0["t"] + DAY}
    pop_rules = sorted({v["rule"] for v in ever.values()})
    pop_rels = sorted({p["rel"] for v in ever.values() for p in v["path"]})
    # (2) relations as data, and the same with every node renamed
    dw = _data_world()
    data_needs = R.label(dw)
    names = {"activity/analytics": "activity/x1", "skill/python": "skill/x2", "transport/route7": "transport/x3",
             "service/freight": "service/x4", "access/coastal-market": "access/x5", "capability/coordination": "c/x6",
             "activity/acme-ops": "activity/x7", "capability/moderation": "c/x8", "package/vec": "package/x9",
             "repo/a/app": "repo/x10", "repo/b/app": "repo/x11", "advisory/1": "advisory/x12",
             "service/clipshost": "service/x13", "resource/storage": "resource/x14", "capacity/storage": "c/x15",
             "capability/clip-hosting": "c/x16"}
    rw = _data_world(names)
    renamed = R.label(rw)
    rename_diff = int(_shape(data_needs) != _shape(renamed, names))
    data_rels = sorted({p["rel"] for n in data_needs for p in n["path"]})
    data_targets = sorted(n["target"] for n in data_needs)
    used_rels = sorted(set(pop_rels) | set(data_rels))
    triggers = R.special_case_triggers(dw, data_needs) + R.special_case_triggers(rw, renamed)
    last = R.label(w)
    triggers += R.special_case_triggers(w, last)
    literals = _literals_naming_entities()
    special = triggers + len(literals) + rename_diff
    m = {"reusable_need_relations": len(used_rels), "special_case_need_triggers": int(special),
         "relation_types_used": ",".join(used_rels), "population_world_needs_ever": len(ever),
         "population_world_needs_emerged_after_genesis": len(emerged),
         "population_rules": ",".join(pop_rules), "data_scenario_needs": len(data_needs),
         "data_scenario_targets": ";".join(data_targets), "rename_differences": rename_diff,
         "entity_literals_in_rules": len(literals)}
    evidence.record("SP-6.4", m, notes="emerged needs (population world): " + "; ".join(
        f"{k} ({v['rule']}, first day {v['first_t'] // DAY}, {v['holders']} holders)" for k, v in sorted(emerged.items())))
    assert len(used_rels) >= 5 and special == 0, m
    assert emerged, "needs arise from changed conditions in the population world"
    assert {"skill/python", "access/coastal-market", "capability/coordination", "capability/moderation",
            "package/vec", "capability/clip-hosting"} <= set(data_targets), data_targets
