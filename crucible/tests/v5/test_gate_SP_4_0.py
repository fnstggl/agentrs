"""SP-4.0 (MASTER_SPEC Part 4, introduction): all platforms share the civilization's participant population and world
state: one population (D-15) acts on every platform, with no second copy of any per-person fact.

Measured on a generated civilisation (`crucible.civ.generate("balanced", 3, population="development")`, every
subsystem composed on the one population) run for three simulated days, from its recorded log (evaluator side,
`crucible.population.analysis`):

populations_in_v5_world == 1: kernel modules of the world that keep a population of their own (the one population
`pop`, and any stand-in such as crowd, patrons, 4.0 consumers or npcs);
platforms_used_by_population >= 15 and platforms_without_population_activity == 0: the fifteen platforms Part 4
lists, each counted when the population's own recorded decisions or actions on it exist (impression and consumption
decision batches per audience platform, relation-table decision batches - follows, shares, community memberships,
newsletter subscriptions - and events whose actor is a population member: forum posts, messages, memberships);
duplicate_person_facts == 0: the kernel's shared-fact and data-level duplicate scan (`substrate.primitives.problems`,
D-21: every per-person fact lives in exactly one owner table)."""
import pytest

import crucible
from crucible.acceptance.v5 import evidence
from crucible.acceptance.v5 import manifest as MF
from crucible.acceptance.v5.checker import criterion_met
from crucible.population import analysis as A
from crucible.substrate import primitives

CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --requirement SP-4.0"
DAYS = 3
pytestmark = [pytest.mark.slow, pytest.mark.timeout(3 * 3600)]


@pytest.fixture(scope="module")
def civ_world():
    import torch
    torch.set_num_threads(1)
    crucible.load_all_modules()
    from crucible import civ
    from crucible.calibration import runtime
    runtime.reset_usage()
    w = civ.generate("balanced", 3, population="development")
    w.advance_time(DAYS * 86400)
    yield w, dict(runtime.usage())
    w.close()


def test_SP_4_0_one_population_on_every_platform(civ_world):
    w, _ = civ_world
    pa = A.platform_activity(w)
    pops = A.populations(w)
    dup = primitives.problems(w)
    m = {"populations_in_v5_world": int(pops["n"]),
         "platforms_used_by_population": len(pa["used"]),
         "platforms_without_population_activity": len(pa["unused"]),
         "duplicate_person_facts": len(dup),
         "unused_platforms": ",".join(pa["unused"]),
         **{f"rows_{k}": int(v) for k, v in pa["platforms"].items()}}
    evidence.record("SP-4.0", m, command=CMD, notes=(
        f"Population modules: {pops['population_modules']}; activity per audience platform "
        f"{pa['by_aud_platform']}; duplicate-fact problems: {dup[:10]}"))
    assert sum(pa["platforms"].values()) > 0
    crit = next(e for e in MF.load()["requirements"] if e["id"] == "SP-4.0")["criteria"]
    fails = [msg for c in crit for ok, msg in [criterion_met(c, m)] if not ok]
    assert not fails, (fails, m)


def test_population_classes_act_and_learn_in_the_civilisation(civ_world):
    """The mechanisms of Gates B and G inside the same civilisation (development population, 3 days): autonomous
    sessions with no population decision inside an actor's command, every population-table batch with decision
    provenance, fitted runtime models behind counted decisions, the class-2 shared policy and the class-3 groups
    learning from their own members' experience (weights change, the frozen control does not)."""
    from crucible.calibration import inworld
    from crucible.population import check_provenance
    w, usage = civ_world
    sess = A.sessions_by_day(w)
    li = A.learner_initiated(w)
    pm = inworld.provenance_metrics(w, inworld.audit(w, usage=usage))
    ups = [e.payload for e in w.log if e.type == "pop.policy_update"]
    by = {}
    for u in ups:
        by.setdefault(u["group"], set()).add(u["weights_hash"])
    assert sum(sess.values()) > 1000, sess
    assert li["population_decision_events"] > 0 and li["learner_initiated"] == 0, li
    assert pm["decision_events_without_model_provenance"] == 0, pm["_without_by_kind"]
    assert pm["fitted_models_used_in_world"] >= 3, pm["_used"]
    assert len(by.get("adaptive:0", ())) > 1, sorted(by)
    learned = [g for g, h in by.items() if g.startswith("strategic:") and len(h) > 1]
    assert len(learned) >= 2 and len(by.get("strategic:A0", {"x"})) == 1, {g: len(h) for g, h in by.items()}
    cp = check_provenance(w)
    assert cp["missing_provenance_batches"] == 0, cp
