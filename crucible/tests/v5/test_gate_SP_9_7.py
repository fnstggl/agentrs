"""SP-9.7 (MASTER_SPEC Part 9.7): Tier 2 domain Engineering infrastructure. Workstream I-2 (dom2).

Every required mechanic (Compute, Networking, Workloads, Capacity, Energy, Failures, Technical optimization, Delayed
effects, Deployment of new controllers and software) is exercised by recorded events in integrated reference runs
reached through crucible.experience.make(2, ...) in the default observation_only mode (`base.section_metrics`).
Intervention controls on the same instance: without the repair crew, failed machines pile up (Failures and Delayed
effects are causal, the one-call staffing cut is not a free saving); firing the crew and handing every failed machine
back to the lessor (the shortcut that passed 9.7 before broken returns were charged and counted as failure debt) is
not a success either; a controller with no headroom loses requests (Capacity binds); the population's own adoption is
the tenants' traffic (Workloads: users move demand). The reference's saving comes from the fleet, not from staff: it
pays the same wages as the incumbent operation. With the users' and the tenants' decisions switched off (the
framework's autonomy ablation) the reference no longer succeeds: reliability is judged from the customers' own
recorded decisions.
"""
import pytest

from crucible.acceptance.v5 import evidence
from crucible.domains import base
from crucible.domains import engineering_infrastructure as E

CMD = "PYTHONPATH=src python -m pytest -q tests/v5/test_gate_SP_9_7.py"
CONFIG, SEED = "single_site_tou", 0


@pytest.mark.slow
@pytest.mark.timeout(4 * 3600)
def test_SP_9_7_engineering_infrastructure():
    m, runs = base.section_metrics(E.NAME)
    ref = base.cached_run(E.NAME, CONFIG, SEED, "reference")
    fire = base.cached_run(E.NAME, CONFIG, SEED, "fire_repair_crew")
    ret = base.cached_run(E.NAME, CONFIG, SEED, "fire_and_return_broken")
    starve = base.cached_run(E.NAME, CONFIG, SEED, "starve_capacity")
    # autonomy: the same reference with the decisions of the users and the tenants switched off (the processes that
    # made them in the null run) - the verdict rests on the customers' own recorded decisions, so it must fail
    null = base.cached_run(E.NAME, CONFIG, SEED, "null")
    procs = sorted(set(null["actors"].get("processes") or []) | set(null["population"].get("processes") or []))
    frozen = base.run(E.NAME, CONFIG, SEED, policy="reference", freeze=procs)
    d_ref, d_fire, d_starve = ref["outcome"]["details"], fire["outcome"]["details"], starve["outcome"]["details"]
    d_ret = ret["outcome"]["details"]
    m["fire_and_return_success"] = bool(ret["success"])
    m["fire_and_return_cost_reduction"] = float(d_ret["cost_reduction"])
    m["fire_and_return_failure_debt"] = int(d_ret["repair_backlog"])
    m["reference_wage_cents"] = int(d_ref["cost"]["wage"])
    m["baseline_wage_cents"] = int(d_ref["baseline_cost"]["wage"])
    m["ablated_crew_repair_backlog"] = int(d_fire["repair_backlog"])
    m["reference_repair_backlog"] = int(d_ref["repair_backlog"])
    m["no_headroom_attainment"] = float(d_starve["attainment"])
    m["reference_attainment"] = float(d_ref["attainment"])
    m["reference_cost_reduction"] = float(d_ref["cost_reduction"])
    m["autonomy_ablation_processes"] = ",".join(procs)
    m["autonomy_ablation_success"] = bool(frozen["success"])
    m["autonomy_ablation_services_reviewed"] = int(frozen["outcome"]["details"]["services_reviewed_by_tenants"])
    evidence.record("SP-9.7", m, command=CMD, labels={"scripted": True},
                    notes=f"reference {d_ref['cost_reduction']} cost reduction at attainment {d_ref['attainment']}; "
                          f"no crew backlog {d_fire['repair_backlog']} vs {d_ref['repair_backlog']}; no headroom "
                          f"attainment {d_starve['attainment']}")
    assert m["required_mechanics_listed"] == 9 and m["required_mechanics_exercised"] == 9, m
    assert m["required_mechanics_missing"] == 0, m
    assert m["observation_only_default"] and m["reachable_through_experience"], m
    assert m["ablated_crew_repair_backlog"] > m["reference_repair_backlog"] + 2, m
    assert not m["fire_and_return_success"] and m["fire_and_return_failure_debt"] > m["reference_repair_backlog"], m
    assert ref["success"] and m["reference_wage_cents"] == m["baseline_wage_cents"], m
    assert m["no_headroom_attainment"] < m["reference_attainment"] - 0.01, m
    assert "fleet.tenants" in procs and "pop.tick" in procs and not m["autonomy_ablation_success"], m
