"""crucible.civ.longrun: checkpoints and resume after an interruption (a process killed without warning).

A 3-day civilisation run is started in a child process and killed (SIGKILL) as soon as its day-2 checkpoint is on
disk; the same run started again in the same directory resumes at that checkpoint in a new process. It must end in
the state (event hash, state digest, sequence number) of an uninterrupted run of the same spec and seed - or, where
the two runs part, the first divergence must come from a recorded non-reproducible outcome (batched neural
inference recomputed in another process; crucible.civ.determinism.first_divergence), never from the interruption or
the world's logic - and the interrupted run's whole record must regenerate exactly from genesis in a fresh process
(nothing the cut and the resume left is unexplained by the log), with every Gate P check of the report clean.
"""
import json
import os
import signal
import subprocess
import sys
import time

import pytest

import crucible

crucible.load_all_modules()

from crucible.civ import determinism, gates, longrun  # noqa: E402
from crucible.storage.sqlite import SQLiteStore  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]
PROFILE, SEED, DAYS = "balanced", 5, 3


def _child(workdir, report):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # the same snapshot schedule as the reference run (a snapshot flushes the store; the trajectory is compared under
    # an identical schedule, so the comparison isolates the interruption)
    return subprocess.Popen([sys.executable, "-m", "crucible.civ.longrun", "run", PROFILE, str(SEED), str(DAYS),
                             "development", str(workdir), str(report), "--snapshot-every=1", "--resume-days="],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_interrupted_run_resumes_at_its_checkpoint_and_matches_an_uninterrupted_run(tmp_path):
    ref = longrun.run_horizon(PROFILE, SEED, DAYS, workdir=str(tmp_path / "ref"), snapshot_every=1,
                              resume_days=[], keep_store=True)
    assert not ref["failed"], ref["error"]
    wd = tmp_path / "cut"
    wd.mkdir()
    pr = _child(wd, wd / "report.json")
    prog = wd / "progress.json"
    t0 = time.time()
    while time.time() - t0 < 1800:
        if pr.poll() is not None:
            break
        try:
            day = json.loads(prog.read_text())["checkpoint"]["day"]
        except (OSError, ValueError, KeyError):
            day = -1
        if day >= 2:
            os.kill(pr.pid, signal.SIGKILL)
            pr.wait()
            break
        time.sleep(0.5)
    assert pr.returncode == -signal.SIGKILL, "the child finished before it could be interrupted"
    assert not (wd / "report.json").exists()
    rep = longrun.run_horizon(PROFILE, SEED, DAYS, workdir=str(wd), snapshot_every=1, resume_days=[],
                              keep_store=True)
    assert not rep["failed"], rep["error"]
    assert len(rep["interruptions"]) == 1 and rep["interruptions"][0]["load_ok"], rep["interruptions"]
    assert rep["interruptions"][0]["resumed_at_day"] == 2
    wid = f"civ-{PROFILE}-{SEED}-{DAYS}d"
    name = f"{wid}.sqlite"
    mine = SQLiteStore(str(wd / name)).read_events(wid, from_seq=0, to_seq=rep["final"]["seq"])
    theirs = SQLiteStore(str(tmp_path / "ref" / name)).read_events(wid, from_seq=0, to_seq=ref["final"]["seq"])
    div = determinism.first_divergence(mine, theirs)
    # the interruption left no trace in the trajectory: equal, or parted only by a recorded inference outcome
    assert rep["final"] == ref["final"] or div["origin"] == "recorded_outcome", div
    # and the interrupted run's record explains its state exactly (fresh process, from genesis)
    rg = gates._regenerate_child(str(wd / name), wid, rep["final"]["seq"])
    assert rg["ok"], rg.get("error")
    assert (rg["hash"], rg["digest"], rg["seq"]) == (rep["final"]["hash"], rep["final"]["digest"],
                                                     rep["final"]["seq"]), (rg, rep["final"])
    s = rep["summary"]
    assert s["completed_days"] == DAYS and not s["failed"]
    assert s["unexplained_invariant_violations"] == 0 and s["accounting_mismatches"] == 0, rep["accounting"]
    assert s["replay_mismatches"] == 0 and s["replay_segments"] >= DAYS    # every day closed a verified segment
    assert [r["day"] for r in rep["daily"]] == list(range(1, DAYS + 1))
    if div["first_seq"] is None or div["first_seq"] > ref["daily"][0]["seq"]:
        assert rep["daily"][0]["events"] == ref["daily"][0]["events"]


def test_suite_skips_complete_runs_and_reports_missing_ones(tmp_path, monkeypatch):
    monkeypatch.setenv(longrun.RUNS_ENV, str(tmp_path))
    d = longrun.job_dir("balanced", "development", 30, 101)
    os.makedirs(d)
    with open(os.path.join(d, "report.json"), "w") as fh:
        json.dump({"profile": "balanced", "seed": 101, "days": 30, "failed": False, "summary": {"failed": False}}, fh)
    reps = longrun.suite((30,), 2, only_existing=True)
    by_seed = {r["seed"]: r for r in reps}
    assert longrun.complete(by_seed[101]) and not longrun.complete(by_seed[102])
    assert "not run to completion" in by_seed[102]["error"]
