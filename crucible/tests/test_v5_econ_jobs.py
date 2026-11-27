"""ECON wave 3b: the population's labour-market activity (`economy.jobs`, activity `jobs`) shares the labour market
with other modules that enter individuals as simulated workers (workstream I-2's `firms` `job_search` activity and
initial staff). One decision path per worker: `jobs` decides only for the individuals it registered itself, and its
column table does not collide with the `jobs` table another module may declare."""
import numpy as np

import crucible

crucible.load_all_modules()
from crucible.commerce.worlds import startup_spec  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.economy import jobs  # noqa: E402
from crucible.population import features  # noqa: E402
from crucible.substrate import columns, ids  # noqa: E402

DAY = 86400


def test_jobs_leaves_simulated_workers_to_their_own_worker_model():
    w = World.create(startup_spec(21, n=300, formats=["schedule.jobs"], extra=("inst", "labor")))
    assert jobs.TABLE in columns.tables(w) and "jobs" not in columns.tables(w)
    out_of_work = np.flatnonzero(~features(w, np.arange(300))["employed"].astype(bool))
    sim_idx = out_of_work[:12]
    assert sim_idx.size == 12

    def enter(ctx):                      # what another module does when it enters individuals as simulated workers
        labor = ctx.world.modules["labor"]
        for i in sim_idx.tolist():
            p = ids.ensure_principal(ctx, int(i), account=True)
            labor.add_worker(ctx, p, {"writing": 0.6, "research": 0.5}, {"writing": 0.5, "research": 0.5}, 150_000,
                             sim=True, hours_week=30, declared=["writing", "research"])
    w._run_system("test.enter_sim_workers", enter)
    for d in range(1, 11):
        w.advance_time(d * DAY)
    st = w.state
    # the activity ran for the rest of the population (registrations and searches of its own) ...
    regs = [p for p, r in st.scan("labor_workers") if p[1:].isdigit() and not r["sim"]]
    assert len(regs) >= 10
    searched = np.flatnonzero(columns.col(w, jobs.TABLE, "last_search") >= 0)
    assert searched.size >= 10
    # ... and never for the simulated workers: no search, registration value or shift of its own for them, and the
    # daily wake-up of labour-market workers does not include them
    assert not set(sim_idx.tolist()) & set(searched.tolist())
    assert (columns.col(w, jobs.TABLE, "vot")[sim_idx] == 0).all()
    assert (columns.col(w, jobs.TABLE, "last_shift_day")[sim_idx] == -1).all()
    for e in w.log:
        if e.type == "cols.batch" and str(e.payload.get("kind", "")).startswith("work.job_"):
            assert e.payload["table"] == jobs.TABLE
    assert all(st.tables["labor_workers"][ids.pid(int(i))]["sim"] for i in sim_idx)
    assert w.check_invariants() == []
    assert not [e for e in w.log if e.type == "proc.failed"]
