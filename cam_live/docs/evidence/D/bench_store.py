"""Throughput of SqliteExperienceStore on this machine. Run:
    cd /workspace/camlive && .venv/bin/python cam_live/docs/evidence/D/bench_store.py
"""
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tests", "credit"))
os.environ.setdefault("OMP_NUM_THREADS", "4")

from conftest import build_record  # noqa: E402

from cam_live.contracts import OutcomeObservation, OutcomeStatus  # noqa: E402
from cam_live.credit.replay import SqliteExperienceStore  # noqa: E402

out = {}
with tempfile.TemporaryDirectory() as d:
    st = SqliteExperienceStore(os.path.join(d, "durable.sqlite"))
    t = time.perf_counter()
    for i in range(300):
        st.append(build_record(i, n_cand=8))
    out["durable_appends_per_s"] = round(300 / (time.perf_counter() - t), 1)
    st.close()
    st = SqliteExperienceStore(os.path.join(d, "bulk.sqlite"))
    n = 20000
    t = time.perf_counter()
    for start in range(0, n, 1000):
        with st.batch():
            for i in range(start, start + 1000):
                st.append(build_record(i, n_cand=8, family=f"fam{i % 7}"))
    out["batched_appends_per_s"] = round(n / (time.perf_counter() - t), 1)
    t = time.perf_counter()
    for i in range(0, 2000):
        r = st.add_outcome(OutcomeObservation(f"q{i}", OutcomeStatus.OBSERVED, float(i % 3), observed_at=1e6))
        assert r.accepted
    out["durable_add_outcome_per_s"] = round(2000 / (time.perf_counter() - t), 1)
    t = time.perf_counter()
    ids = st.sample_ids(32, loss="world", seed=0)
    out["first_sample_ids_ms_20k_records"] = round(1e3 * (time.perf_counter() - t), 2)
    t = time.perf_counter()
    for s in range(200):
        st.sample_ids(32, loss="world", seed=s)
    out["cached_sample_ids_ms"] = round(1e3 * (time.perf_counter() - t) / 200, 3)
    t = time.perf_counter()
    for s in range(20):
        st.sample(32, loss="world", seed=s)
    out["sample_32_records_ms"] = round(1e3 * (time.perf_counter() - t) / 20, 2)
    out["db_bytes_per_record"] = round(os.path.getsize(st.path) / n)
    st.close()
print(json.dumps(out))
