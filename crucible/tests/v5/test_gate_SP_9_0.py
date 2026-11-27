"""Gate SP-9.0 (MASTER_SPEC Part 9 introduction): every Tier 2 domain has autonomous participants and open-ended
actions, its configurations differ in world structure (not only in objective text), and it composes into Tier 3.

Measured by `crucible.domains.base.check` on every registered Part 9 domain (test fixtures excluded); a Part 9
section counts when one of its domains satisfies the criterion, a section without a domain fails. While fewer than
fifteen sections have a domain the test FAILS (it never skips).

* autonomous participants: in null-baseline runs (the seats do nothing for the evaluation horizon) every declared
  autonomous participant kind (`Actor`) produces recorded DECISIONS - events whose actor is a principal other than
  the seats, or population column batches with decision provenance (D-15); kernel bookkeeping, system events and
  constant counts never count - and the one population records decisions of an activity other than its own
  bookkeeping;
* open-ended actions: the seat's tool list (the universal tool interface through `crucible.experience.make(2, ...)`)
  has >= 10 tools over >= 4 MASTER_SPEC 11.2 categories, an authoring category and a free-form content tool;
* configurations differing only in objective text: pairs of configurations whose BUILT-world structure
  (`base.fingerprint`: engaged modules, magnitudes reduced to presence/absence, the domain hook on built state only)
  is equal, configurations whose structure changes with the seed, and configurations of DIFFERENT domains with the
  same framework-measured structure - must be 0;
* composable into Tier 3: the domain's first training configuration merged into the `balanced` civilisation
  (`base.tier3_spec`) builds, runs six hours with 0 failed processes and 0 invariant errors, its seat exists there
  and the domain's evaluator runs on it.

Positive controls on the framework fixture (tests/v5/_domain_fixture.py, never counted): it passes all four; a
planted numbers-only configuration, a planted inert actor, actors declared on kernel bookkeeping and system events
(verifier probe p7) and a second domain over the same world (probe p1 d) are detected."""
import os
import sys
from dataclasses import replace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _domain_fixture as F  # noqa: E402
from crucible.acceptance.v5 import evidence  # noqa: E402
from crucible.domains import base  # noqa: E402

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_9_0.py"
CTL_DAYS = 7.0          # the fixture's controls run 7 of its 21 horizon days (its reference succeeds within 3)


@pytest.mark.slow
@pytest.mark.timeout(8 * 3600)
def test_SP_9_0_autonomous_open_ended_structural_composable():
    reps = base.gate_reports()
    m = base.gate_metrics(reps)
    per = {n: base.verdicts(r) for n, r in reps.items()}
    F.register()
    try:
        ctl = base.check(F.DOMAIN, days=CTL_DAYS)
        cv = base.verdicts(ctl)
        inert = replace(F.DOMAIN, name="fixture_inert", actors=F.DOMAIN.actors + (
            base.Actor("ghosts", ("ghost.acted",), "never acts"),
            base.Actor("system", ("goods.wom",), "recorded by the system, decided by nobody")))
        base.register(inert)
        null = base.run("fixture_inert", "ring_standard", 0, policy="null", days=3, keep_env=True)
        env = null.pop("env")
        try:          # not registrable (validate refuses them), measured directly: bookkeeping and constant evidence
            unreg = base.actors_active(env.world, replace(inert, actors=(
                base.Actor("clock", ("artifact.stored", "proc"), "kernel bookkeeping"),
                base.Actor("constant", (), "a constant count", count=lambda w, s: 3))), null["since_seq"])
        finally:
            env.close()
        F.SHAPES["ring_standard6"] = {"chain": "standard", "regions": 6, "links": "ring"}
        planted = replace(F.DOMAIN, name="fixture_numbers", configs=F.DOMAIN.configs + (
            base.DomainConfig("ring_standard6", "train", dict(F.SHAPES["ring_standard6"]), "only more regions"),))
        base.register(planted)
        same = base.structure_report(planted, seeds=(0,))["same_structure_pairs"]
        dup = replace(F.DOMAIN, name="fixture_dup", goal=replace(F.GOAL, text=lambda p: "Another prompt."))
        base.register(dup)
        cross = base.cross_domain_same_structure({
            F.NAME: {"structure": base.structure_report(F.DOMAIN, seeds=(0,))},
            "fixture_dup": {"structure": base.structure_report(dup, seeds=(0,))}})
    finally:
        for n in (F.NAME, "fixture_inert", "fixture_numbers", "fixture_dup"):
            base.unregister(n)
        F.SHAPES.pop("ring_standard6", None)
    metrics = {k: m[k] for k in ("domains_with_autonomous_participants", "domains_with_open_ended_actions",
                                 "configurations_differing_only_in_objective_text", "domains_composable_into_tier3",
                                 "domains_registered", "sections_with_domain", "sections_missing",
                                 "cross_domain_same_structure", "domain_load_errors")}
    for k in ("autonomous", "open_ended", "same_structure_pairs", "tier3"):
        metrics[f"per_domain_{k}"] = ";".join(f"{n}={v[k]}" for n, v in sorted(per.items())) or "none"
    metrics["population_activity_kinds"] = ";".join(
        f"{n}={','.join(sorted(r['autonomy']['population_activity_kinds']))}" for n, r in sorted(reps.items())) \
        or "none"
    metrics.update({"control_fixture_autonomous": cv["autonomous"], "control_fixture_open_ended": cv["open_ended"],
                    "control_fixture_tier3": cv["tier3"], "control_fixture_same_structure_pairs":
                    cv["same_structure_pairs"], "control_inert_actor_active": null["actors"]["counts"]["ghosts"],
                    "control_bookkeeping_actor_active": unreg["counts"]["clock"],
                    "control_constant_actor_active": unreg["counts"]["constant"],
                    "control_system_event_actor_active": null["actors"]["counts"]["system"],
                    "control_numbers_only_pairs": len(same), "control_cross_domain_pairs": len(cross)})
    evidence.record("SP-9.0", metrics, command=CMD, labels={"scripted": True})
    assert cv["autonomous"] and cv["open_ended"] and cv["tier3"] and cv["same_structure_pairs"] == 0
    assert null["actors"]["active"] == 2 and null["actors"]["counts"]["ghosts"] == 0
    assert unreg["counts"]["clock"] == 0 and unreg["counts"]["constant"] == 0 and null["actors"]["counts"]["system"] == 0
    assert "ring_standard=ring_standard6" in same and len(cross) == len(F.DOMAIN.configs)
    assert m["domains_with_autonomous_participants"] == 15, metrics
    assert m["domains_with_open_ended_actions"] == 15, metrics
    assert m["domains_composable_into_tier3"] == 15, metrics
    assert m["sections_with_domain"] == 15 and m["configurations_differing_only_in_objective_text"] == 0, metrics
