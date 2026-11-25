"""ComputeMeter: CPU/GPU seconds, counters, peaks, cost, atomic snapshots and merging (manifest X06)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest
import torch

from superlearner.cam_unified.observe.accounting import (ComputeMeter, atomic_write_json, merge_snapshots,
                                                          read_snapshots)

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))


def test_counters_cpu_rates_and_cost():
    m = ComputeMeter("a0", role="actor")
    m.add(env_steps=10, decisions=10, sim_days=0.5, tokens_prompt=100)
    m.add(decisions=5, custom_metric=2)
    x = 0
    t0 = time.process_time()
    while time.process_time() - t0 < 0.2:
        x += 1
    s = m.snapshot()
    assert s["decisions"] == 15 and s["env_steps"] == 10 and s["custom_metric"] == 2 and s["tokens_prompt"] == 100
    assert s["cpu_s"] >= 0.18 and s["wall_s"] >= s["cpu_s"] * 0.5
    assert s["rates"]["decisions_per_s"] == pytest.approx(15 / s["wall_s"], rel=0.05)
    assert s["peak_rss_mb"] > 10 and s["cost_usd"] is None          # no price -> no invented cost
    priced = ComputeMeter("p", cpu_usd_per_core_hour=3.6)
    t0 = time.process_time()
    while time.process_time() - t0 < 0.1:
        x += 1
    s2 = priced.snapshot()
    assert s2["cost_usd"] == pytest.approx(s2["cpu_s"] / 3600 * 3.6, rel=1e-6)


def test_gpu_section_never_initialises_cuda_in_cpu_processes():
    code = ("import sys; sys.path.insert(0, %r); import torch\n"
            "from superlearner.cam_unified.observe.accounting import ComputeMeter\n"
            "m = ComputeMeter('x')\n"
            "with m.gpu():\n    torch.ones(4).sum()\n"
            "s = m.snapshot(); print(torch.cuda.is_initialized(), s['gpu_s'], s['peak_gpu_mb'])" % SRC)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120).stdout.split()
    assert out == ["False", "0.0", "0.0"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_seconds_measure_device_work():
    torch.cuda.init()
    m = ComputeMeter("g")
    a = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with m.gpu():
        for _ in range(20):
            a = a @ a
            a = a / a.norm()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    s = m.snapshot()
    assert 0 < s["gpu_s"] <= wall * 1.5 and s["gpu_sections"] == 1 and s["peak_gpu_mb"] > 15


def test_snapshots_are_atomic_and_merge_without_double_counting(tmp_path):
    d = str(tmp_path / "compute")
    a, b = ComputeMeter("a0", role="actor"), ComputeMeter("learner", role="learner")
    a.add(decisions=100, env_steps=100, records=200)
    b.add(updates=3, samples_trained=96)
    a.write(os.path.join(d, "a0.1.json"))
    b.write(os.path.join(d, "learner.2.json"))
    with open(os.path.join(d, "partial.json.tmp.9"), "w") as fh:
        fh.write('{"name": ')                     # a writer's temp file is never read as a snapshot
    snaps = read_snapshots(d)
    assert sorted(s["name"] for s in snaps) == ["a0", "learner"]
    tot = merge_snapshots(snaps)
    assert tot["decisions"] == 100 and tot["updates"] == 3 and tot["n_processes"] == 2
    assert tot["cpu_s"] == pytest.approx(sum(s["cpu_s"] for s in snaps))
    assert tot["peak_rss_mb_sum"] >= tot["peak_rss_mb_max"] > 0
    assert set(tot["by_role"]) == {"actor", "learner"}
    sup = ComputeMeter("launcher", include_children=False)
    subprocess.run([sys.executable, "-c", "x=0\nfor i in range(3_000_000): x+=i"], check=True)
    assert sup.snapshot()["cpu_s"] < 0.2          # reaped children are not counted by the supervisor
    atomic_write_json(os.path.join(d, "x.json"), {"k": 1})
    assert json.load(open(os.path.join(d, "x.json"))) == {"k": 1}
