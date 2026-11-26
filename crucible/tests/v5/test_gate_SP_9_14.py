"""Gate SP-9.14 - Tier 2 domain Crisis recovery (MASTER_SPEC Part 9.14, workstream I-3).

Every required mechanic (cascading disruptions; scarce resources; prioritization; coordination; recovery; resilience; independent service-restoration metrics) must be exercised by recorded events in an integrated run of a generated
configuration, with the shared population as the autonomous participants, observation_only by default and the
world reached through `crucible.experience.make(2, ...)`.

Measured by the framework (`base.section_metrics`) on reference runs of EVERY configuration of the domain (three
training, one compositional, one structural evaluation configuration; seed 14), the domain's scripted tool-only
reference (labelled scripted) driven through the learner interface; a mechanic counts only when it is exercised in
every configuration's run (tests/v5/_dom3_gate.py). Every run must be clean."""
from __future__ import annotations

import pytest

from crucible.acceptance.v5 import checker
from crucible.acceptance.v5.evidence import record
from crucible.domains import crisis as C

from _dom3_gate import section

pytestmark = [pytest.mark.slow, pytest.mark.timeout(4 * 3600)]


def test_SP_9_14_crisis_mechanics_exercised():
    m, runs, notes = section("crisis", 14)
    record("SP-9.14", m, labels={"scripted": True}, notes=notes[:8000])
    assert len(C.MECHANICS) == 7 and tuple(C.DOMAIN.required_mechanics) == C.MECHANICS
    assert m["clean_runs"] == m["configurations_run"] == 5, notes[:3000]
    assert not checker.unmet("SP-9.14", m), (m, notes[:3000])
