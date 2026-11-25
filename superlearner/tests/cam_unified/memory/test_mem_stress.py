"""Streaming stress tests (slow; run with ``-m slow``): one million lightweight events through the real archive,
a million-key index, and consolidation throughput. They print a JSON line with the measured numbers."""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import psutil
import pytest
import torch

from memhelpers import light_events
from superlearner.cam_unified.config import make_config
from superlearner.cam_unified.memory.archive import EventArchive
from superlearner.cam_unified.memory.consolidate import Consolidator
from superlearner.cam_unified.memory.index import MemoryIndex
from superlearner.cam_unified.memory.replay import ReplaySampler
from superlearner.cam_unified.memory.retrieval import Retriever


@pytest.mark.slow
def test_one_million_lightweight_records_streaming(tmp_path):
    proc = psutil.Process()
    cfg = make_config("dev")                                 # archive_chunk_rows = 65536, archive_max_bytes = 200 GiB
    root = str(tmp_path / "archive")
    a = EventArchive(root, cfg)
    N, B = 1_000_000, 2000
    rss0 = proc.memory_info().rss
    peak, t_gen, t_app = rss0, 0.0, 0.0
    for i in range(0, N, B):
        t = time.time()
        evs = light_events(B, start=i, episode_len=500)
        t_gen += time.time() - t
        t = time.time()
        a.append(evs)
        t_app += time.time() - t
        if i % 50_000 == 0:
            a.flush()
            peak = max(peak, proc.memory_info().rss)
    a.flush()
    assert a.count() == N                                    # no silent cessation of ingestion
    info = a.info()
    m = {"events": N, "ingest_s": round(t_app, 1), "ingest_events_per_s": int(N / t_app),
         "record_build_s": round(t_gen, 1), "chunks": info["chunks"], "wal_rows": info["wal_rows"],
         "bytes_total": info["used_bytes"], "bytes_per_event": round(info["used_bytes"] / N, 1),
         "rss_growth_mb": round((peak - rss0) / 2**20, 1)}
    t = time.time()
    n_meta = sum(len(b) for b in a.scan_meta(batch_size=65536, columns=["gseq"]))
    m["scan_meta_s"] = round(time.time() - t, 1)
    t = time.time()
    n, rss_scan, last = 0, 0, None
    for b in a.scan(batch_size=4096):
        n += len(b)
        last = b[-1]
        if n % 200_000 < 4096:
            rss_scan = max(rss_scan, proc.memory_info().rss)
    m["scan_records_s"] = round(time.time() - t, 1)
    m["scan_rss_growth_mb"] = round((rss_scan - rss0) / 2**20, 1)
    assert n_meta == N and n == N and last.event_id == f"env0-0-{N - 1:09d}"
    rng = random.Random(0)
    ids = [f"env0-0-{rng.randrange(N):09d}" for _ in range(300)]
    t = time.time()
    for e in ids:
        a.get([e])
    m["random_get_ms"] = round((time.time() - t) / len(ids) * 1e3, 2)
    t = time.time()
    got = a.get(ids)
    m["batched_get_300_ms"] = round((time.time() - t) * 1e3, 1)
    assert [g.event_id for g in got] == ids
    rs = ReplaySampler(a, seed=0)
    t = time.time()
    rs.sync()
    m["replay_sync_s"] = round(time.time() - t, 1)
    t = time.time()
    sl = rs.sample(64, 32)
    m["replay_sample_64x32_s"] = round(time.time() - t, 2)
    assert len(sl) == 64 and all(len(s) == 32 for s in sl)
    assert min(s.anchor_gseq for s in sl if s.stratum == "uniform") < N // 2       # the whole lifetime is reachable
    rs.close()
    a.close()
    t = time.time()
    b = EventArchive(root, cfg)
    m["reopen_s"] = round(time.time() - t, 2)
    assert b.count() == N
    b.append(light_events(10, start=N))
    assert b.count() == N + 10
    b.close()
    print("\nSTRESS_1M " + json.dumps(m))
    assert m["rss_growth_mb"] < 1500 and m["scan_rss_growth_mb"] < 1500


@pytest.mark.slow
def test_million_key_index_search_latency(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("GPU search latency")
    kd, n = 256, 1_000_000
    idx = MemoryIndex(str(tmp_path / "i"), kd, 1, 8, device="cuda", fsync=False)
    rng = np.random.default_rng(0)
    for i in range(0, n, 100_000):
        k = rng.standard_normal((100_000, kd)).astype(np.float32)
        k /= np.linalg.norm(k, axis=1, keepdims=True)
        ids = [f"k{j}" for j in range(i, i + 100_000)]
        idx.add_segments([{"seg_id": s, "level": 1, "t_start": float(j), "t_end": float(j), "env_id": f"e{j % 4}"}
                          for j, s in zip(range(i, i + 100_000), ids)])
        idx.add("v", ids, k, np.zeros((100_000, 1, 8), np.float32))
    q = torch.from_numpy(rng.standard_normal((16, kd)).astype(np.float32))
    idx.search("v", q, 16)                                   # warm the device cache
    t = time.time()
    for _ in range(5):
        idx.search("v", q, 16)
    lat = (time.time() - t) / 5
    t = time.time()
    idx.search("v", q, 16, {"env_id": "e1"})                # builds the columnar metadata cache once
    lat_cold = time.time() - t
    t = time.time()
    for _ in range(5):
        res = idx.search("v", q, 16, [{"env_id": "e1", "t_max": 500_000.0}] * 8 + [{"t_min": 990_000.0}] * 8)
    lat_f = (time.time() - t) / 5
    assert all(int(s[1:]) % 4 == 1 and int(s[1:]) <= 500_000 for s in res.seg_ids[0])
    assert all(int(s[1:]) >= 990_000 for s in res.seg_ids[9])
    m = {"keys": n, "key_dim": kd, "search_16q_ms": round(lat * 1e3, 1),
         "filtered_search_16q_ms": round(lat_f * 1e3, 1), "first_filtered_search_ms": round(lat_cold * 1e3, 1),
         "index_bytes": idx.storage_bytes()}
    print("\nINDEX_1M " + json.dumps(m))
    assert lat < 1.0 and lat_f < 1.0
    idx.close()


@pytest.mark.slow
def test_consolidation_throughput_dev_scale(tmp_path):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = make_config("dev")                                  # d_model 384, key 256, segment 32 events -> 4 tokens
    a = EventArchive(str(tmp_path / "a"), cfg)
    for i in range(0, 100_000, 5000):
        a.append(light_events(5000, start=i, episode_len=400))
    idx = MemoryIndex(str(tmp_path / "i"), cfg.mem_key_dim, cfg.mem_segment_tokens, cfg.d_model, device=dev)
    ret = Retriever(cfg, idx).to(dev)
    con = Consolidator(a, idx, ret, batch_segments=256)
    t = time.time()
    out = con.step(max_events=200_000)
    dt = time.time() - t
    m = {"events": out["events"], "segments1": out["segments1"], "segments2": out["segments2"],
         "seconds": round(dt, 1), "events_per_s": int(out["events"] / dt), "encode_s": round(con.stats["encode_s"], 1),
         "index_bytes": idx.storage_bytes(), "retriever_params": ret.param_split()}
    print("\nCONSOLIDATE_100K " + json.dumps(m))
    assert out["events"] == 100_000 and out["segments1"] >= 100_000 // cfg.mem_segment
    a.close()
