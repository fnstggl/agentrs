"""SP-9.9 (MASTER_SPEC Part 9.9): Tier 2 domain Labor marketplace. Workstream I-2 (dom2).

Every required mechanic (Worker skills, Preferences, Compensation, Recruitment, Delegation, Project dependencies,
Performance, Retention, Independent agency) is exercised by recorded events in integrated reference runs reached
through crucible.experience.make(2, ...) in the default observation_only mode (`base.section_metrics`). The workers
are population individuals in the labour market (their own job_search sessions). Intervention controls on the same
instance: a team that is recruited but never delegated to finishes no task (Delegation is causal); offering a third of
the market salary hires nobody (Compensation and the workers' own reservation wages are causal).
"""
import pytest

from crucible.acceptance.v5 import evidence
from crucible.domains import base
from crucible.domains import labor_marketplace as L

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_9_9.py"
CONFIG, SEED = "pipeline_open", 0


@pytest.mark.slow
@pytest.mark.timeout(4 * 3600)
def test_SP_9_9_labor_marketplace():
    m, runs = base.section_metrics(L.NAME)
    ref = base.cached_run(L.NAME, CONFIG, SEED, "reference")
    post_only = base.cached_run(L.NAME, CONFIG, SEED, "post_only")
    low = base.cached_run(L.NAME, CONFIG, SEED, "lowball")
    m["reference_tasks_done"] = int(ref["outcome"]["details"]["tasks_done"])
    m["no_delegation_tasks_done"] = int(post_only["outcome"]["details"]["tasks_done"])
    m["lowball_team"] = int(low["outcome"]["details"]["team"])
    m["reference_team"] = int(ref["outcome"]["details"]["team"])
    m["population_job_search_decisions"] = int(ref["population"]["activity_kinds"].get("firms.jobsearch", 0))
    evidence.record("SP-9.9", m, command=CMD, labels={"scripted": True},
                    notes=f"reference {ref['outcome']['details']}; no delegation {post_only['outcome']['details']}; "
                          f"lowball {low['outcome']['details']}")
    assert m["required_mechanics_listed"] == 9 and m["required_mechanics_exercised"] == 9, m
    assert m["required_mechanics_missing"] == 0, m
    assert m["observation_only_default"] and m["reachable_through_experience"], m
    assert m["no_delegation_tasks_done"] == 0 < m["reference_tasks_done"], m
    assert m["lowball_team"] < m["reference_team"], m
    assert m["population_job_search_decisions"] > 0, m
