"""Actor-throughput benchmark on the toy environment for 1/4/8 actors (manifest F04/X06). Slow; serialised through
the shared heavy-job lock because 8 actors exceed the per-agent process budget."""
from __future__ import annotations

import fcntl
import json
import os

import pytest

from superlearner.cam_unified.dist.launcher import benchmark_actors

LOCK = "/workspace/agentrs/runs/cam_unified/gpu_heavy.lock"


@pytest.mark.slow
def test_actor_throughput_1_4_8(tmp_path):
    out = os.environ.get("CAM_BENCH_OUT", str(tmp_path / "bench"))
    os.makedirs(out, exist_ok=True)
    fd = os.open(LOCK, os.O_RDWR | os.O_CREAT, 0o666) if os.path.isdir(os.path.dirname(LOCK)) else None
    try:
        if fd is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        secs = float(os.environ.get("CAM_BENCH_SECONDS", "15"))
        gen = benchmark_actors(out, (1, 4, 8), seconds=secs, with_learner=False, max_spool_bytes=8 << 30)
        res = benchmark_actors(out, (1, 4, 8), seconds=secs)
    finally:
        if fd is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    rows = []
    for r in res:
        rows.append({k: r[k] for k in ("n_actors", "decisions_per_s_steady", "ingested_records_per_s_steady",
                                       "trained_decisions_per_s_steady", "decisions", "learner_version",
                                       "actor_blocked_s", "restarts")} | {"compute": r["compute"],
                                                                         "lag_hist": r["lag"].get("lag_hist")})
    print(json.dumps(rows, indent=1))
    print("actors-only generation:", [(g["n_actors"], round(g["decisions_per_s_steady"] or 0)) for g in gen])
    print("end-to-end:", [(r["n_actors"], round(r["decisions_per_s_steady"] or 0),
                           round(r["trained_decisions_per_s_steady"] or 0)) for r in res])
    # experience generation scales with actors (independent processes, separate worlds)
    assert gen[1]["decisions_per_s_steady"] > 2.5 * gen[0]["decisions_per_s_steady"]
    assert gen[2]["decisions_per_s_steady"] > gen[1]["decisions_per_s_steady"]
    for g in gen:
        assert g["restarts"] == {} and g["decisions"] > 0
        for d in os.listdir(out):                    # free the actors-only spools (hundreds of MB)
            if d.startswith("actors_only"):
                import shutil
                shutil.rmtree(os.path.join(out, d, "spool"), ignore_errors=True)
    for r in res:
        assert r["restarts"] == {} and r["decisions"] > 0 and r["learner_version"] > 0
        assert r["queue"]["gaps"] == {} and r["queue"]["totals"]["quarantined"] == 0
        assert r["compute"]["decisions"] == r["decisions"]
    # end to end the single learner's ingestion bounds the sustained rate; it must not collapse as actors are added
    assert min(r["decisions_per_s_steady"] for r in res) > 0.5 * max(r["decisions_per_s_steady"] for r in res)
