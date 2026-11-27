"""Gate SP-9.11 - Tier 2 domain Information ecosystem (MASTER_SPEC Part 9.11, workstream I-3).

Every required mechanic (search; documents; knowledge artifacts; queries; source provenance; information quality; updating facts; independent user satisfaction; verified retrieval performance) must be exercised by recorded events in an integrated run of a generated
configuration, with the shared population as the autonomous participants, observation_only by default and the
world reached through `crucible.experience.make(2, ...)`.

Measured by the framework (`base.section_metrics`) on reference runs of EVERY configuration of the domain (three
training, one compositional, one structural evaluation configuration; seed 11), the domain's scripted tool-only
reference (labelled scripted) driven through the learner interface; a mechanic counts only when it is exercised in
every configuration's run (tests/v5/_dom3_gate.py). Every run must be clean. The participant is needed: the
domain's PARTICIPANT_MECHANICS (measured on the participant's own activity) are exercised by none of the null
baseline's runs (idle seat, first configuration), and the null baseline fails."""
from __future__ import annotations

import pytest

from crucible.acceptance.v5 import checker
from crucible.acceptance.v5.evidence import record
from crucible.domains import information as I

from _dom3_gate import assert_participant_needed, section

pytestmark = [pytest.mark.slow, pytest.mark.timeout(4 * 3600)]


def test_SP_9_11_information_mechanics_exercised():
    m, runs, notes = section("information", 11)
    record("SP-9.11", m, labels={"scripted": True}, notes=notes[:8000])
    assert len(I.MECHANICS) == 9 and tuple(I.DOMAIN.required_mechanics) == I.MECHANICS
    assert m["clean_runs"] == m["configurations_run"] == 5, notes[:3000]
    assert_participant_needed(m, notes)
    assert not checker.unmet("SP-9.11", m), (m, notes[:3000])
