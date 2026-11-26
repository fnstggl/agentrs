"""Gate P (MASTER_SPEC Part 17): long-horizon persistence. Workstream J.

`crucible.civ.longrun.suite` runs the documented Gate P civilisation (profile `balanced`, population profile
`development`: 1,000 behavioural, 32 adaptive, 8 strategic and 2 semantic individuals; crucible.civ.longrun
.GATE_P_PROFILE) for 30, 90, 180 and 365 simulated days with two independent seeds per horizon (eight separate runs,
distinct seeds), each on its own SQLite store, checking every simulated day (invariants, process failures, activity),
auditing the event accounting every ten days and at the end, snapshotting every ten days, replaying every
snapshot segment from the recorded events, and resuming in a NEW process at two points of every run.

    P.1   runs_30/90/180/365_days >= 2, failed_runs == 0
    P.2   unexplained_invariant_violations == 0     (crucible.civ.longrun.EXPLAINED lists explained ones: none)
    P.3   snapshot_resume_mismatches == 0          (cross-process, exact (D-47): a fresh interpreter loads the
                                                    world from a copy of the store and re-executes the live run's
                                                    recorded next day through the handlers; the free resume that
                                                    recomputes inference is reported beside it)
    P.4   replay_mismatches == 0                   (every segment replayed from the store's events)
    P.5   event_accounting_mismatches == 0         (crucible.civ.accounting.Accountant)
    P.6   min_late_over_early_economic/social_activity >= 0.3  (last 30 days over the first 30, worst run)
    P.n1  independent_seeds_per_horizon >= 2

The suite costs about 1,330 simulated days; run it with CPUs:
`python -m crucible.scale.slots --cpu 4 -- env CRUCIBLE_CIV_WORKERS=4 python -m pytest -q tests/v5/test_gate_P.py`.
"""
import pytest

from crucible.acceptance.v5 import evidence

CMD = "CRUCIBLE_CIV_WORKERS=4 PYTHONPATH=src python -m pytest -q tests/v5/test_gate_P.py"


@pytest.fixture(scope="module")
def suite():
    import crucible
    crucible.load_all_modules()
    from crucible.civ import gates
    return gates.long_horizon()


def _notes(suite):
    rows = []
    for r in suite["reports"]:
        s = r.get("summary") or {}
        rows.append(f"{r['days']}d seed {r['seed']}: failed={r.get('failed')} days={s.get('completed_days')} "
                    f"wall/day={s.get('wall_s_per_day_mean')} peak_rss={s.get('peak_rss_mb')}MB "
                    f"store_max={s.get('store_mb_max')}MB pruned={r.get('pruned_blob_mb')}MB "
                    f"eco={s.get('late_over_early_economic')} soc={s.get('late_over_early_social')} "
                    f"error={(r.get('error') or '')[:200]}")
    return "; ".join(rows)


def _rec(req, suite, keys):
    m = {k: suite["metrics"][k] for k in keys}
    evidence.record(req, m, command=CMD, notes=_notes(suite))
    return m


@pytest.mark.slow
@pytest.mark.timeout(30 * 3600)
def test_P_1_long_horizon_runs(suite):
    m = _rec("P.1", suite, ["runs_30_days", "runs_90_days", "runs_180_days", "runs_365_days", "failed_runs"])
    assert all(m[f"runs_{h}_days"] >= 2 for h in (30, 90, 180, 365)) and m["failed_runs"] == 0, m


@pytest.mark.slow
@pytest.mark.timeout(30 * 3600)
def test_P_2_no_unexplained_invariant_violations(suite):
    m = _rec("P.2", suite, ["unexplained_invariant_violations"])
    assert m["unexplained_invariant_violations"] == 0


@pytest.mark.slow
@pytest.mark.timeout(30 * 3600)
def test_P_3_snapshot_and_resume(suite):
    m = _rec("P.3", suite, ["snapshot_resume_mismatches", "snapshot_resume_checks", "snapshot_resume_exact_checks",
                            "free_resume_equal", "free_resume_divergence_origins"])
    assert m["snapshot_resume_checks"] > 0 and m["snapshot_resume_mismatches"] == 0, m


@pytest.mark.slow
@pytest.mark.timeout(30 * 3600)
def test_P_4_recorded_replay(suite):
    m = _rec("P.4", suite, ["replay_mismatches"])
    assert m["replay_mismatches"] == 0


@pytest.mark.slow
@pytest.mark.timeout(30 * 3600)
def test_P_5_consistent_event_accounting(suite):
    m = _rec("P.5", suite, ["event_accounting_mismatches"])
    assert m["event_accounting_mismatches"] == 0


@pytest.mark.slow
@pytest.mark.timeout(30 * 3600)
def test_P_6_continuing_activity(suite):
    m = _rec("P.6", suite, ["min_late_over_early_economic_activity", "min_late_over_early_social_activity"])
    assert m["min_late_over_early_economic_activity"] >= 0.3 and m["min_late_over_early_social_activity"] >= 0.3, m


@pytest.mark.slow
@pytest.mark.timeout(30 * 3600)
def test_P_n1_independent_seeds(suite):
    m = _rec("P.n1", suite, ["independent_seeds_per_horizon"])
    assert m["independent_seeds_per_horizon"] >= 2
