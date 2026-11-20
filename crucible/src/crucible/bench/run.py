"""Crucible benchmarks. Every number written by this script is measured on the machine it runs on.

    python -m crucible.bench.run --out docs/benchmarks/bench.json [--quick]

Measures: kernel command throughput (state transitions/s), replay speed, snapshot/fork/restore latency and size
vs world size, simulated time per wall second vs consumer population, NPC population scaling, concurrent world
replicas (processes), sandbox program start/persistent-call latency, cost per agent step with scripted agents,
and resource usage (RSS, threads). Storage throughput is measured by tests/test_storage.py; browser and LLM
overheads by tests/test_web.py and the RL integration scripts (see docs/SCALING.md).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import resource
import threading
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

DAY = 86400


def _setup():
    import crucible
    crucible.load_all_modules()


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def bench_kernel(n_actors=200, n_cmds=20_000):
    from crucible.core.kernel import World, WorldSpec
    _setup()
    spec = WorldSpec("bench", 1, [{"name": "identity"}, {"name": "bank"}, {"name": "admin"}])
    w = World.create(spec)
    for i in range(n_actors):
        w.system_call("admin.join", {"handle": f"a{i}", "endowment": 10_000_000})
    accts = {f"p:a{i}": w.modules["bank"].primary_account(w.state, f"p:a{i}") for i in range(n_actors)}
    import numpy as np
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, n_actors, (n_cmds, 2))
    t = time.perf_counter()
    ok = 0
    for a, b in pairs:
        r = w.execute(f"p:a{a}", {"tool": "bank.transfer", "args": {"from_account": accts[f"p:a{a}"], "to": f"a{b}" if a != b else f"a{(b + 1) % n_actors}", "amount": 100}})
        ok += r.ok
    dt = time.perf_counter() - t
    t2 = time.perf_counter()
    reads = 0
    for a, _ in pairs[:5000]:
        w.execute(f"p:a{a}", {"tool": "bank.accounts", "args": {}})
        reads += 1
    dt2 = time.perf_counter() - t2
    evs = [e.to_dict() for e in w.log]
    t3 = time.perf_counter()
    w2 = World.replay(spec, evs)
    dt3 = time.perf_counter() - t3
    assert w2.digest() == w.digest()
    return {"transfers_per_s": round(ok / dt, 1), "read_cmds_per_s": round(reads / dt2, 1),
            "events": len(evs), "replay_events_per_s": round(len(evs) / dt3, 1), "invariants_ok": w.check_invariants() == []}


def bench_world(family="bazaar", consumers=6000, days=2, seed=1):
    from crucible.core.kernel import World
    from crucible.worldgen import generate
    _setup()
    spec = generate({"family": family, "overrides": {"n_consumers": consumers}}, seed)
    t0 = time.perf_counter()
    w = World.create(spec)
    create_s = time.perf_counter() - t0
    w.advance_time(w.now + DAY // 4)
    t = time.perf_counter()
    w.advance_time(w.now + days * DAY)
    sim_s = time.perf_counter() - t
    t = time.perf_counter()
    blob = w.snapshot_blob()
    snap_s = time.perf_counter() - t
    t = time.perf_counter()
    f = w.fork("bench-fork")
    fork_s = time.perf_counter() - t
    t = time.perf_counter()
    d = w.digest()
    digest_s = time.perf_counter() - t
    return {"family": family, "consumers": consumers, "create_s": round(create_s, 3),
            "sim_days_per_wall_s": round(days / sim_s, 3), "wall_s_per_sim_day": round(sim_s / days, 3),
            "events_per_sim_day": round((w.seq) / (days + 0.25), 1), "snapshot_ms": round(snap_s * 1000, 1),
            "snapshot_mb": round(len(blob) / 2**20, 2), "fork_ms": round(fork_s * 1000, 1),
            "digest_ms": round(digest_s * 1000, 1), "fork_equal": f.digest() == d, "rss_mb": round(rss_mb(), 1),
            "principals": w.state.count("principals")}


def _replica(args):
    family, consumers, days, seed = args
    r = bench_world(family, consumers, days, seed)
    return r["wall_s_per_sim_day"]


def bench_replicas(n_list=(1, 2, 4, 8), consumers=3000, days=1):
    out = []
    for n in n_list:
        t = time.perf_counter()
        ctx = mp.get_context("spawn")
        with ctx.Pool(n) as pool:
            per = pool.map(_replica, [("bazaar", consumers, days, s) for s in range(n)])
        wall = time.perf_counter() - t
        out.append({"replicas": n, "wall_s": round(wall, 2), "aggregate_sim_days_per_wall_s": round(n * days / wall, 3),
                    "per_replica_wall_s_per_sim_day": round(sum(per) / n, 3)})
    return out


def bench_sandbox(n=200):
    from crucible.sandbox.runner import Worker, run_program
    files = {"svc.py": "def f(a, c):\n    return sorted(a)\n"}
    t = time.perf_counter()
    for _ in range(10):
        run_program(files, "svc:f", [3, 1, 2])
    one_shot = (time.perf_counter() - t) / 10
    w = Worker(files).start()
    t = time.perf_counter()
    for i in range(n):
        w.run({"kind": "call", "target": "svc:f", "args": [i, 2, 1]})
    per = (time.perf_counter() - t) / n
    w.close()
    return {"one_shot_program_ms": round(one_shot * 1000, 2), "persistent_call_ms": round(per * 1000, 3)}


def bench_agents(days=3):
    from crucible.agents.base import Runner
    from crucible.agents.scripted import Builder, Composer, RandomPolicy
    from crucible.core.kernel import World
    from crucible.worldgen import generate
    _setup()
    w = World.create(generate({"family": "bazaar"}, 2))
    pols = [("p:b1", Builder()), ("p:c1", Composer(need="route.shortest")), ("p:r1", RandomPolicy())]
    for a, _ in pols:
        w.system_call("admin.join", {"handle": a[2:], "endowment": 3_000_000})
    run = Runner(w, pols, dt=3600)
    t = time.perf_counter()
    run.run(24 * days)
    wall = time.perf_counter() - t
    steps = sum(len(tr.steps) for tr in run.trajs.values())
    return {"agents": len(pols), "agent_steps": steps, "sim_days": days, "wall_s": round(wall, 2),
            "ms_per_agent_step_incl_world": round(wall / steps * 1000, 2)}


def bench_npcs(workers=(50, 500, 2000), days=1):
    from crucible.core.kernel import World
    from crucible.worldgen import generate
    _setup()
    out = []
    for n in workers:
        spec = generate({"family": "tiny", "overrides": {"workers": n, "n_consumers": 2000}}, 3)
        t = time.perf_counter()
        w = World.create(spec)
        c = time.perf_counter() - t
        t = time.perf_counter()
        w.advance_time(days * DAY)
        out.append({"npc_workers": n, "create_s": round(c, 2), "wall_s_per_sim_day": round((time.perf_counter() - t) / days, 3)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/benchmarks/bench.json")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    res = {"machine": {"cpu_count": os.cpu_count(), "python": platform.python_version(), "platform": platform.platform()},
           "t": time.strftime("%Y-%m-%d %H:%M:%S")}
    res["kernel"] = bench_kernel(n_cmds=5000 if a.quick else 20_000)
    print("kernel", res["kernel"], flush=True)
    res["sandbox"] = bench_sandbox()
    print("sandbox", res["sandbox"], flush=True)
    sizes = [1000, 6000] if a.quick else [1000, 6000, 25_000, 100_000]
    res["population"] = [bench_world("bazaar", n, 1 if n > 20000 else 2) for n in sizes]
    for r in res["population"]:
        print("population", r, flush=True)
    res["npcs"] = bench_npcs((50, 500) if a.quick else (50, 500, 2000))
    print("npcs", res["npcs"], flush=True)
    res["agents"] = bench_agents(1 if a.quick else 3)
    print("agents", res["agents"], flush=True)
    res["replicas"] = bench_replicas((1, 2) if a.quick else (1, 2, 4, 8, 16))
    print("replicas", res["replicas"], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
