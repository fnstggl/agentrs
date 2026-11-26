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


def test_SP_4_0_one_population_on_every_platform():
    import torch
    torch.set_num_threads(1)
    crucible.load_all_modules()
    from crucible import civ
    w = civ.generate("balanced", 3, population="development")
    try:
        w.advance_time(DAYS * 86400)
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
    finally:
        w.close()
    assert sum(pa["platforms"].values()) > 0
    crit = next(e for e in MF.load()["requirements"] if e["id"] == "SP-4.0")["criteria"]
    fails = [msg for c in crit for ok, msg in [criterion_met(c, m)] if not ok]
    assert not fails, (fails, m)
