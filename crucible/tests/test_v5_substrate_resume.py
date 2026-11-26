"""Cross-process snapshot / resume / replay of a world with the adaptive and strategic population (audit K1, D-21).

The adaptive and strategic participants rebuild their neural weights in `Module.on_load` from checkpoint blobs.
Those blobs live only in the world's store once the process that wrote them is gone. The test runs three separate
processes (tests/v5_resume_child.py):

1. reference: one uninterrupted run, no store;
2. part: a SQLite-stored run that snapshots at day 1, runs on to day 1.5 and dies with os._exit (a crash: nothing
   is closed, the process's artifact pool is lost);
3. resume: a NEW process loads the world from the store (latest snapshot + replay of the events after it) and runs on
   to day 2.5; and replay: a NEW process replays the whole stored log from genesis using the store's blobs only.

The resumed run must continue with exactly the reference's event hashes and state digest, and replay must reproduce
the state without executing a single neural inference.
"""
import json
import os
import subprocess
import sys

import pytest

import crucible
from crucible.population.worlds import population_world

crucible.load_all_modules()
HERE = os.path.dirname(os.path.abspath(__file__))
CHILD = os.path.join(HERE, "v5_resume_child.py")


def child(*args, timeout=900) -> dict:
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONPATH=os.path.join(os.path.dirname(HERE), "src"))
    p = subprocess.run([sys.executable, CHILD, *map(str, args)], capture_output=True, text=True, timeout=timeout,
                       env=env)
    assert p.returncode == 0, p.stderr[-3000:]
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    assert lines, (p.stdout[-2000:], p.stderr[-2000:])
    return json.loads(lines[-1])


def cross_process_resume(tmp_path, spec, snap_day=1.0, crash_day=1.5, end_day=2.5) -> dict:
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec.to_dict()))
    db = tmp_path / "world.sqlite"
    ref = child("ref", sp, end_day)
    part = child("part", sp, db, snap_day, crash_day)
    res = child("resume", sp, db, end_day)
    rep = child("replay", sp, db)
    return {"ref": ref, "part": part, "resume": res, "replay": rep}


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_a_fresh_process_resumes_a_snapshot_of_a_world_with_neural_participants(tmp_path):
    spec = population_world("development", 3, n=300, sites=8)
    assert {"pop_adaptive", "pop_strategic"} <= {m["name"] for m in spec.modules}
    r = cross_process_resume(tmp_path, spec)
    ref, part, res, rep = r["ref"], r["part"], r["resume"], r["replay"]
    assert "pop_adaptive" in res["modules"] and ref["inference_runs"] > 0
    # the resumed process started from the snapshot and replayed the events written after it
    assert part["snapshot_seq"] < res["loaded_seq"] == part["seq"] and res["loaded_hash"] == part["hash"]
    # ...and continued with exactly the uninterrupted run's events and state
    assert (res["seq"], res["hash"], res["digest"]) == (ref["seq"], ref["hash"], ref["digest"])
    assert res["invariants"] == [] and ref["invariants"] == []
    # inference ran only for the decisions taken after the resume point, never for recorded ones
    assert 0 < res["inference_runs"] < ref["inference_runs"]
    # a fresh process replays the stored log from genesis exactly, with zero neural inference
    assert (rep["seq"], rep["hash"], rep["digest"]) == (part["seq"], part["hash"], part["digest"])
    assert rep["inference_runs"] == 0
