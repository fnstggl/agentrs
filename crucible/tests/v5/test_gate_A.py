"""Gate A (MASTER_SPEC Part 17): baseline preservation, measured on the real tree against Crucible 4.0 (c07098b).

A.1 runs the 4.0 test files of the recorded baseline - byte-identical to c07098b, restored in a fresh clone of HEAD, as
the explicitly selected 4.0 historical baseline (DECISION_LOG D-82) - and requires every one of its 1,076 node ids to
pass by exact id (run it with the GPU visible through the slot scheduler, so the two GPU tests do not skip, and with
CRUCIBLE_GATE_A_WORKERS=<cpus> to split the files). A.2 compares every 4.0 world, mechanism and useful-work family
definition by generated content on two seeds with an extracted c07098b tree. A.3 builds and advances every 4.0 world
family on two seeds. A.4 checks the sealed store against its committed metadata baseline, its access log, static
references in v5 code and runtime opens under an audit hook (the sealed task contents are never read). A.5
regenerates the 4.0 acceptance records A-P and the deterministic demos in a fresh clone and compares them with the
committed records, and compares the 4.0 world trajectories in 4.0's deterministic mode with c07098b's.

Machinery: crucible.acceptance.v5.baseline. Every test records its measurement before asserting on it.
"""
from __future__ import annotations

import pytest

from crucible.acceptance.v5 import baseline as B
from crucible.acceptance.v5.checker import unmet
from crucible.acceptance.v5.evidence import record

CMD = "PYTHONPATH=src python -m crucible.acceptance.v5 run --gate A --agent <name>"


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_A_1_regression_suite_passes_by_baseline_node_id():
    res = B.regression()
    m = res["metrics"]
    record("A.1", m, command=CMD, notes=f"D-82: the c07098b bytes of every 4.0 test file, restored in a fresh clone, "
                                        f"run as the 4.0 historical baseline; not run: {res['not_run'][:10]}; not passed: "
                                        f"{res['not_passed'][:10]}; tree copies carrying v5 additions (restored for the run): "
                                        f"{res['changed_test_files'][:12]}; still differing after the restore: "
                                        f"{res['differ_after_restore'][:5]}; exit codes {res['rcs']}")
    bad = unmet("A.1", m)
    assert not bad, (bad, res["not_passed"][:10], res["not_run"][:10])


@pytest.mark.timeout(1800)
def test_A_2_original_benchmark_definitions_intact():
    res = B.definitions()
    m = res["metrics"]
    record("A.2", m, command=CMD, notes=f"changed: {res['changed'][:20]}; missing: {res['missing'][:20]}; "
                                        f"v5 additions to the 4.0 world registry: {res['v5_added_world_families'][:20]}")
    bad = unmet("A.2", m)
    assert not bad, (bad, res["changed"][:10], res["missing"][:10])


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_A_3_every_v4_world_configuration_executes():
    res = B.execution()
    m = res["metrics"]
    record("A.3", m, command=CMD, notes=f"failed: {res['failed'][:20]}; absent: {res['absent'][:20]}")
    bad = unmet("A.3", m)
    assert not bad, (bad, res["failed"][:10], res["absent"][:10])


@pytest.mark.timeout(1800)
def test_A_4_sealed_datasets_remain_isolated():
    res = B.sealed()
    m = res["metrics"]
    record("A.4", m, command=CMD, notes=f"changed: {res['changed']}; static references: {res['static']}; runtime "
                                        f"opens: {res['runtime_hits']}")
    bad = unmet("A.4", m)
    assert not bad, (bad, res["changed"], res["static"][:5], res["runtime_hits"][:5], res["probe_stderr"][-500:])


@pytest.mark.slow
@pytest.mark.timeout(6 * 3600)
def test_A_5_prior_results_reproducible():
    prior = B.prior_results()
    traj = B.trajectories()
    m = {**prior["metrics"], **traj["metrics"]}
    bad_records = [r for r in prior["results"] if not r.get("attempted") or r.get("rc") or r.get("n_mismatches")]
    record("A.5", m, command=CMD, notes=f"records not reproduced: {[(r['record'], r.get('rc'), r.get('mismatches', [])[:3]) for r in bad_records][:10]}; "
                                        f"trajectory mismatches: {traj['mismatches'][:10]}; nondeterministic in 4.0 itself: "
                                        f"{traj['nondeterministic'][:10]}; documented as not reproducible: "
                                        f"{sorted(prior['not_reproducible'])}")
    bad = unmet("A.5", m)
    assert not bad, (bad, bad_records[:5], traj["mismatches"][:5])
