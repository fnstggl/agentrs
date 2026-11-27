"""Gate SP-9.15 - Tier 2 domain Research commercialization (MASTER_SPEC Part 9.15, workstream I-3).

Every required mechanic (research, experimentation, technical development, product creation, financing,
deployment, independent adoption, commercial sustainability) must be exercised by recorded events in an integrated
run of a generated configuration, with the shared population as the autonomous participants (practitioners who
adopt or reject the product on their own bills; labs, investors and apparatus operators are autonomous too),
observation_only by default, reachable through `crucible.experience.make(2, ...)`; and the domain must genuinely
connect scientific discovery and economic outcomes, confirmed by counterfactual removal of the discovery.

Mechanics: measured by the framework (`base.section_metrics`) on reference runs of every configuration (seed 15;
a mechanic counts only when exercised in every configuration's run, tests/v5/_dom3_gate.py); every mechanic is
measured on the founder's own venture, and the null baseline (idle seat) exercises none of them and fails.
Counterfactual removal: the same world (configuration and seed) driven by the same founder with the discovery
removed - it builds its advisor from common practice (the population's habitual setting, the most generous
knowledge available without the phenomenon's data, handed to it by the test) instead of from its measurements -
on two seeds. The link is confirmed when, on every seed, the founder with the discovery succeeds while the founder
without it fails, and the discovery at least doubles the retained independent adopters and raises the last 30
days' revenue."""
from __future__ import annotations

import pytest

from crucible.acceptance.v5 import checker
from crucible.acceptance.v5.evidence import record
from crucible.domains import base
from crucible.domains import commercialization as R

from _dom3_gate import assert_participant_needed, section

pytestmark = [pytest.mark.slow, pytest.mark.timeout(4 * 3600)]


def _arm(seed: int, discovery: bool) -> dict:
    env = base.make_env("commercialization", "linear_optimum", seed)
    a = env.agents()[0]
    obj = R.DOMAIN.make_objective(env.world, a, 1)
    common = env.world.modules["rcomm"].c["own_setting"][0]
    pol = R.ScienceFounder(a, obj, discovery=discovery, common_practice=common)
    base.drive(env, {a: pol}, days=R.DOMAIN.horizon_days, dt=R.DOMAIN.dt)
    ev = env.verify()["evaluation"][a]
    env.close()
    return ev


def test_SP_9_15_research_commercialization_mechanics_and_discovery_link():
    m, runs, notes = section("commercialization", 15)
    link, extra = [], []
    for seed in (2, 3):
        ea, eb = _arm(seed, True), _arm(seed, False)
        da, db = ea["details"], eb["details"]
        ok = bool(ea["success"]) and not eb["success"] and \
            da["adopters_retained"] >= 2 * max(1, db["adopters_retained"]) and \
            da["revenue_30d_cents"] > db["revenue_30d_cents"]
        link.append(ok)
        extra.append(f"seed {seed}: with discovery {da}; without {db}")
    m["discovery_to_economic_outcome_confirmed"] = bool(link) and all(link)
    m["counterfactual_seeds"] = len(link)
    record("SP-9.15", m, labels={"scripted": True}, notes=(" | ".join(extra) + " || " + notes)[:8000])
    assert len(R.MECHANICS) == 8 and tuple(R.DOMAIN.required_mechanics) == R.MECHANICS
    assert m["clean_runs"] == m["configurations_run"] == 5, notes[:3000]
    assert_participant_needed(m, notes)
    assert not checker.unmet("SP-9.15", m), (m, extra)
