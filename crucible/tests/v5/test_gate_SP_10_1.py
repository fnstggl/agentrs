"""SP-10.1 (MASTER_SPEC Part 10.1): civilisation profiles (the ten listed) that differ structurally. Workstream J.

    civilization_profiles                  profiles of crucible.civ.PROFILES that build, cover the Part 10.1 list and
                                           whose built-world fingerprint differs from every other profile's
    profiles_differing_only_in_parameters  pairs of profiles whose built worlds have identical discrete fingerprints
                                           (module set, coupling set, supply-chain graph, freight topology, energy
                                           and compute regimes, institutions, social placement, boards, science
                                           bindings: crucible.civ.integration.fingerprint)
"""
import pytest

from crucible.acceptance.v5 import evidence

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_10_1.py"


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_SP_10_1_structurally_different_profiles():
    import crucible
    crucible.load_all_modules()
    from crucible.civ import gates, profiles
    r = gates.structure()
    same = {n for pair in r["pairs_identical"] for n in pair}
    listed = {p.part_10_1 for p in profiles.PROFILES.values() if p.part_10_1}
    distinct_ok = [n for n in r["profiles"] if n not in same]
    covered = sorted(listed & set(profiles.PART_10_1))
    m = {"civilization_profiles": len(distinct_ok) if len(covered) == len(profiles.PART_10_1) else len(covered),
         "profiles_differing_only_in_parameters": len(r["pairs_identical"]),
         "part_10_1_covered": len(covered),
         "min_structural_distance": min(r["distances"].values()) if r["distances"] else 0.0}
    evidence.record("SP-10.1", m, command=CMD, notes=f"profiles {r['part_10_1']}; identical pairs "
                    f"{r['pairs_identical']}; distances {r['distances']}")
    assert m["civilization_profiles"] >= 10 and m["profiles_differing_only_in_parameters"] == 0, m
