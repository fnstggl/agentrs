"""SP-9.6 (MASTER_SPEC Part 9.6): Tier 2 domain Technology invention. Workstream I-2 (dom2).

Every required mechanic (Executable inventions, Technical constraints, Independent evaluation, Adoption,
Compatibility, Economic consequences, Downstream resource changes) is exercised by recorded events in integrated
reference runs reached through crucible.experience.make(2, ...) in the default observation_only mode
(`base.section_metrics`). Intervention controls on the same instance: a program that violates its interface fails
every trial on the world's constraint checks and is adopted nowhere (Technical constraints are enforced); a
technology that changes nothing yields no evaluator-measured gain even where a plant kept it after a lucky trial.
"""
import pytest

from crucible.acceptance.v5 import evidence
from crucible.domains import base
from crucible.domains import technology_invention as T

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_9_6.py"
CONFIG, SEED = "forming_standard", 1


@pytest.mark.slow
@pytest.mark.timeout(4 * 3600)
def test_SP_9_6_technology_invention():
    m, runs = base.section_metrics(T.NAME)
    ref = base.cached_run(T.NAME, CONFIG, SEED, "reference")
    broken = base.cached_run(T.NAME, CONFIG, SEED, "broken_program")
    freeze = base.cached_run(T.NAME, CONFIG, SEED, "freeze")
    m["broken_program_adopters"] = int(broken["outcome"]["details"].get("adopters", 0))
    m["broken_program_constraint_rejections"] = int(broken["mechanics"]["counts"].get("Technical constraints", 0))
    m["freeze_gain"] = float(freeze["outcome"]["details"].get("gain", 0.0))
    m["reference_gain"] = float(ref["outcome"]["details"].get("gain", 0.0))
    m["reference_adopters"] = int(ref["outcome"]["details"].get("adopters", 0))
    evidence.record("SP-9.6", m, command=CMD, labels={"scripted": True},
                    notes=f"reference {ref['outcome']['details']}; broken {broken['outcome']['details']}; "
                          f"freeze {freeze['outcome']['details']}")
    assert m["required_mechanics_listed"] == 7 and m["required_mechanics_exercised"] == 7, m
    assert m["required_mechanics_missing"] == 0, m
    assert m["observation_only_default"] and m["reachable_through_experience"], m
    assert m["broken_program_adopters"] == 0 < m["broken_program_constraint_rejections"], m
    assert m["freeze_gain"] == 0.0 < m["reference_gain"], m
