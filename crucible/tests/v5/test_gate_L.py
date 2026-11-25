"""Gate L - endogenous opportunities (workstream G).

Four independently generated worlds (different seeds and structures, crucible.innovation.scenarios):

* capacity (seed 2): an agent-created route.shortest service is adopted until its instance's capacity binds; the
  needs graph derives a supply need (consumes + constrains + produces); participants publish it; an entrepreneur
  builds a substitute; participants switch; the need resolves.
* batch (seed 5): a composite technology (route.batch calling route.shortest) is adopted, multiplying compute demand
  on the base service until its constraint binds; a supply need for the base capability emerges.
* social (seed 7): adopters recommend the service on the forum; participants who only read the forum discover it
  there, test it, adopt it and pay for it.
* software (seed 3): a package is executed by an independent dependent; a verified advisory constrains the locked
  version; the dependent's need for an alternative emerges and is addressed by a fixed release.

Each chain is traced from recorded events; counterfactual removal of its first stage is a fork of one snapshot
taken just before (or just after) that stage, in which the stage is removed (the service withdrawn, the technology
never built, the posts deleted). Creators are scripted reference agents (feasibility, labelled); participants are
the rule-based adopters of crucible.innovation.adopt.
"""
import pytest

import crucible

crucible.load_all_modules()
from crucible.acceptance.v5 import evidence  # noqa: E402
from crucible.innovation import measure as M  # noqa: E402
from crucible.innovation import scenarios as SC  # noqa: E402
from crucible.innovation.measure import Stage  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.timeout(2400)]
PROVENANCE = ("creators are scripted reference agents acting through tools; participants are rule-based need-driven "
              "adopters (crucible.innovation.adopt); needs come only from the needs graph's relation rules")


@pytest.fixture(scope="module")
def worlds():
    r = {"capacity": SC.capacity(), "batch": SC.batch(), "social": SC.social(), "software": SC.software()}
    yield r
    for x in r.values():
        SC.close(x)


def _ev(w, etype, pred=lambda e: True):
    return [e for e in w.log if e.type == etype and pred(e)]


def chain_capacity(w):
    d = _ev(w, "caps.declared", lambda e: e.payload["host"] == "anaroute")
    ad = _ev(w, "caps.adopted", lambda e: e.payload["host"] == "anaroute" and e.payload["independent"])
    sup = _ev(w, "needs.derived", lambda e: e.payload["kind"] == "supply")
    new = [x for x in M.needs_addressed_by_new_services(w) if x["kind"] == "supply"]
    nd = _ev(w, "caps.declared", lambda e: new and e.payload["host"] == new[0]["host"])
    return [Stage("agent creates and declares a service", "registry/cloud", d[0].t if d else None,
                  {"declaration": d[0].payload["id"] if d else None}),
            Stage("independent participants test and adopt it", "participants", ad[4].t if len(ad) >= 5 else None,
                  {"independent_adoptions": len(ad)}),
            Stage("growing use binds the compute constraint: supply need", "needs/compute", sup[0].t if sup else None,
                  {"path": [e["rel"] for e in sup[0].payload["path"]] if sup else []}),
            Stage("a new service addresses the need and holders switch", "registry/participants",
                  nd[0].t if nd else None, {"addressed": new[:1]})]


def chain_batch(w):
    d = _ev(w, "caps.declared", lambda e: e.payload["host"] == "batchroute")
    ad = _ev(w, "caps.adopted", lambda e: e.payload["host"] == "batchroute" and e.payload["independent"])
    dep = _ev(w, "needs.graph", lambda e: any(v["src"] == "svc:batchroute" and v["rel"] == "requires"
                                              for v in e.payload["edges"].values()))
    sup = _ev(w, "needs.derived", lambda e: e.payload["kind"] == "supply" and e.payload["target"] == "cap:route.shortest")
    return [Stage("agent creates a composite technology on another service", "registry/cloud", d[0].t if d else None),
            Stage("participants adopt the technology", "participants", ad[0].t if ad else None,
                  {"independent_adoptions": len(ad)}),
            Stage("its use is observed to require the base capability", "needs graph", dep[0].t if dep else None),
            Stage("compute demand on the base service binds its constraint: supply need", "needs/compute",
                  sup[0].t if sup else None, {"path": [e["rel"] for e in sup[0].payload["path"]] if sup else []})]


def chain_social(w):
    forum = {p for p, r in w.state.scan("adopt_people") if r["channel"] == "forum"}
    posts = _ev(w, "forum.post", lambda e: "anasort" in e.payload["hosts"])
    seen = _ev(w, "caps.seen", lambda e: e.payload["by"] in forum)
    ad = _ev(w, "caps.adopted", lambda e: e.payload["host"] == "anasort" and e.payload["adopter"] in forum)
    sales = [t for _, t in w.state.scan("txns") if t["kind"] == "sale" and t["ref"] == "svc:anasort" and
             w.state.tables["accounts"][[a for a, d in t["postings"] if d < 0][0]]["owner"] in forum]
    return [Stage("an adopter recommends the service on the forum", "social (forum)", posts[0].t if posts else None,
                  {"posts": len(posts)}),
            Stage("forum readers discover it", "social -> registry", seen[0].t if seen else None,
                  {"forum_discoverers": len({e.payload["by"] for e in seen})}),
            Stage("forum readers test and adopt it", "participants", ad[0].t if ad else None,
                  {"forum_adopters": len(ad)}),
            Stage("they pay for its use", "economy (bank)", min(t["t"] for t in sales) if sales else None,
                  {"paid_calls": len(sales)})]


def chain_software(w):
    rel = _ev(w, "pkg.released", lambda e: e.payload["id"] == "geokit@1.0.0")
    ex = _ev(w, "pkg.executed", lambda e: e.payload["ok"] and "geokit" in e.payload["calls"]
             and e.payload["independent"].get("geokit"))
    adv = _ev(w, "pkg.advisory", lambda e: e.payload["verified"])
    nd = _ev(w, "needs.derived", lambda e: e.payload["target"] == "pkg:geokit")
    fix = _ev(w, "needs.addressed", lambda e: nd and e.payload["id"] == nd[0].payload["id"])
    return [Stage("agent releases a package", "software", rel[0].t if rel else None),
            Stage("an independent dependent executes it", "software/sandbox", ex[0].t if ex else None),
            Stage("a reproduction verifies a security defect", "software/security", adv[0].t if adv else None),
            Stage("the dependent's need for an alternative emerges", "needs", nd[0].t if nd else None,
                  {"path": [e["rel"] for e in nd[0].payload["path"]] if nd else []}),
            Stage("a fixed release addresses it", "software", fix[0].t if fix else None)]


def _chains(worlds):
    return [M.chain_record("capacity", chain_capacity(worlds["capacity"]["world"]), "scripted creators + adopters"),
            M.chain_record("technology_compute", chain_batch(worlds["batch"]["world"]), "scripted creators + adopters"),
            M.chain_record("social_discovery", chain_social(worlds["social"]["world"]), "scripted creator + adopters"),
            M.chain_record("software_security", chain_software(worlds["software"]["world"]),
                           "scripted developers (feasibility)")]


def test_L1_agent_created_capabilities_alter_existing_conditions(worlds):
    effects = []
    cf = worlds["capacity"]["counterfactual"]["arms"]
    effects.append(("needs: supply needs derived", cf["factual"]["supply_needs"], cf["removed"]["supply_needs"]))
    effects.append(("participants switched to a new service", cf["factual"]["adoptions_new_service"],
                    cf["removed"]["adoptions_new_service"]))
    bt = worlds["batch"]["counterfactual"]["arms"]
    effects.append(("compute units consumed by the pre-existing base service", bt["factual"]["base_cu_since_fork"],
                    bt["removed"]["base_cu_since_fork"]))
    effects.append(("refused (overloaded) requests at the base service", bt["factual"]["base_overload"],
                    bt["removed"]["base_overload"]))
    so = worlds["social"]["counterfactual"]["arms"]
    effects.append(("paid calls by forum readers", so["factual"]["forum_channel_paid_calls"],
                    so["removed"]["forum_channel_paid_calls"]))
    measured = [e for e in effects if e[1] != e[2]]
    m = {"capability_effects_measured": len(measured),
         **{f"effect[{n}]": f"{a} with vs {b} without" for n, a, b in effects}}
    evidence.record("L.1", m, notes=f"{PROVENANCE}. Each effect compares forks of one snapshot with and without the "
                                    "agent-created capability (or its social channel), same horizon.")
    assert m["capability_effects_measured"] >= 1


def test_L2_new_participant_needs_arise(worlds):
    rows = []
    for name in ("capacity", "batch", "software"):
        for n in M.emerged_needs(worlds[name]["world"]):
            rows.append((name, n["kind"], n["target"], n["rule"], len(n["holders_ever"])))
    m = {"emerged_needs": len(rows), "emerged_need_kinds": ",".join(sorted({r[1] for r in rows})),
         "needs_with_participant_holders": sum(1 for r in rows if r[4] > 0)}
    evidence.record("L.2", m, notes=f"{PROVENANCE}. Needs first derived after the needs graph's first update (so not "
                                    f"present when the world began): {rows}")
    assert m["emerged_needs"] >= 1


def test_L3_new_services_address_these_needs(worlds):
    rows = []
    for name in ("capacity", "batch", "software"):
        rows += [{"world": name, **x} for x in M.needs_addressed_by_new_services(worlds[name]["world"])]
    m = {"needs_addressed_by_new_services": len(rows),
         "holders_adopting_new": sum(r["holders_adopting"] for r in rows),
         "resolved_after_addressing": sum(1 for r in rows if r["status"] == "resolved")}
    evidence.record("L.3", m, notes=f"{PROVENANCE}. A need counts when something created after it arose offers its "
                                    "target (a new declared service or a fixed release) and holders adopted it. "
                                    f"Rows: {rows}")
    assert m["needs_addressed_by_new_services"] >= 1


def test_L4_three_independent_multi_stage_chains(worlds):
    chains = _chains(worlds)
    done = [c for c in chains if c["complete"]]
    m = {"independent_chains": len(done), "min_chain_stages": min((c["n_stages"] for c in done), default=0),
         "chains_attempted": len(chains),
         "chains_with_agent_created_artifact": sum(1 for c in done if c["chain"] in ("capacity", "technology_compute",
                                                                                     "software_security")),
         "chains_with_social_activity": sum(1 for c in done if any("social" in s for s in c["subsystems"])),
         "chains_with_technical_activity": sum(1 for c in done if any(s.startswith(("software", "needs/compute"))
                                                                      for s in c["subsystems"]))}
    evidence.record("L.4", m, notes=f"{PROVENANCE}. Each chain comes from its own world and seed. Chains: "
                                    + "; ".join(f"{c['chain']}: " + " -> ".join(f"{s['name']} (t={s['t']})"
                                                                               for s in c["stages"]) for c in chains))
    assert m["independent_chains"] >= 3 and m["min_chain_stages"] >= 3


def test_L5_counterfactual_removal_verifies_the_dependency(worlds):
    rows = []
    cap = worlds["capacity"]["counterfactual"]["arms"]
    rows.append(("capacity: withdraw the service right after creation", cap["factual"]["supply_needs"] > 0 and
                 cap["factual"]["needs_addressed_by_new"] > 0 and cap["removed"]["supply_needs"] == 0 and
                 cap["removed"]["new_service_built"] == 0, cap))
    bt = worlds["batch"]["counterfactual"]["arms"]
    rows.append(("technology: never build the composite", bt["factual"]["supply_needs_on_base"] > 0 and
                 bt["removed"]["supply_needs_on_base"] == 0 and
                 bt["factual"]["base_cu_since_fork"] > 1.2 * bt["removed"]["base_cu_since_fork"], bt))
    so = worlds["social"]["counterfactual"]["arms"]
    rows.append(("social: delete the posts about the service", so["factual"]["forum_channel_adopters"] > 0 and
                 so["removed"]["forum_channel_adopters"] <= 0.25 * so["factual"]["forum_channel_adopters"] and
                 so["removed"]["search_channel_adopters"] >= 0.75 * so["factual"]["search_channel_adopters"], so))
    ok = [r for r in rows if r[1]]
    m = {"counterfactual_removals_confirming": len(ok), "counterfactual_removals_run": len(rows),
         **{f"confirmed[{r[0]}]": bool(r[1]) for r in rows},
         "forks_invariant_clean": all(not v["invariants"] for r in rows for v in r[2].values())}
    evidence.record("L.5", m, notes=f"{PROVENANCE}. Forks of one snapshot, exact under the deterministic cost model; "
                                    "removal arms: " + str({r[0]: {k: {kk: vv for kk, vv in v.items() if kk != 'invariants'}
                                                                    for k, v in r[2].items()} for r in rows}))
    assert m["counterfactual_removals_confirming"] >= 3


def test_Ln1_no_special_case_success_bonuses(worlds):
    import ast
    import os
    import crucible.innovation as pkg
    rows = {}
    for name in ("capacity", "batch", "social"):
        w = worlds[name]["world"]
        creators = sorted({a.actor for a in worlds[name]["agents"] if hasattr(a, "actor")})
        rows[name] = M.special_case_bonuses(w, creators)
    # static: money from system accounts appears in workstream-G code only in genesis endowments
    static_hits = []
    root = os.path.dirname(pkg.__file__)
    for f in sorted(os.listdir(root)):
        if not f.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(root, f)).read())
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and fn.name != "genesis":
                for n in ast.walk(fn):
                    if isinstance(n, ast.Constant) and n.value in ("acct_genesis", "acct_external", "acct_fees"):
                        if f != "measure.py":         # the evaluator lists them to exclude them
                            static_hits.append(f"{f}:{fn.name}")
    m = {"special_case_bonuses": sum(r["bonuses"] for r in rows.values()) + len(static_hits),
         "runtime_bonus_txns": sum(r["bonuses"] for r in rows.values()), "static_bonus_paths": len(static_hits)}
    evidence.record("L.n1", m, notes="Runtime: ledger credits to creators (and related parties) from system accounts "
                                     "after genesis, other than their admission endowment. Static: references to "
                                     f"system accounts outside genesis in crucible/innovation. {rows} {static_hits}")
    assert m["special_case_bonuses"] == 0
