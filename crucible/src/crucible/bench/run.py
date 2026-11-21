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


def bench_open(days=4, seed=1):
    """Open-world mechanisms: capability registration, triggers, flows, negotiation, production, minds, and the
    cost of a simulated day / snapshot / fork / replay of open worlds."""
    import json as _json

    from crucible.core.kernel import World
    from crucible.worldgen import generate
    _setup()
    out = {}
    w = World.create(generate({"family": "open_digital", "overrides": {"n_consumers": 300,
                                                                       "params": {"max_actions_per_hour": 0}}}, seed))
    w.system_call("admin.join", {"handle": "bb", "endowment": 50_000_000})
    ex = lambda t, **k: w.execute("p:bb", {"tool": t, "args": k})  # noqa: E731
    me = w.modules["bank"].primary_account(w.state, "p:bb")
    ex("id.create_org", handle="bbco", name="B", kind="company")
    ex("biz.register", org="o:bbco", account=me)
    acct = ex("bank.open_account", owner="o:bbco").result["account"]
    ex("bank.transfer", from_account=me, to=acct, amount=20_000_000)
    repo = ex("code.create_repo", name="svc", owner="o:bbco").result["repo"]
    ex("code.commit", repo=repo, files={"crucible.json": _json.dumps({"entry": "service.py", "endpoints": {
        "tick": {"handler": "tick"}}}), "service.py": "def tick(args, caller):\n    return {'ok': 1}\n"})
    inst = ex("cloud.rent", provider="o:cloud1", cores=4, account=acct, owner="o:bbco").result["instance"]
    ex("cloud.deploy", repo=repo, host="benchsvc", instance=inst)
    t = time.perf_counter()
    n = 50
    ok_n = 0
    for i in range(n):
        ok_n += ex("caps.publish", host="benchsvc", title=f"t{i}", description="benchmark capability " * 5).ok
    assert ok_n == n, "capability publish failed"
    out["capability_publish_ms"] = round((time.perf_counter() - t) / n * 1000, 3)
    for i in range(20):
        ex("wf.trigger", tool="world.clock", every_s=600)
    t = time.perf_counter()
    w.advance_time(w.now + DAY)
    dt = time.perf_counter() - t
    fires = sum(tr["runs"] for _, tr in w.state.scan("wf_triggers"))
    out["trigger_fires_per_day"] = fires
    out["simulated_day_with_triggers_s"] = round(dt, 3)
    steps = [{"id": f"s{i}", "tool": "world.clock", "args": {}} for i in range(5)]
    fid = ex("wf.define", name="bench", steps=steps).result["flow"]
    t = time.perf_counter()
    ok_n = sum(ex("wf.run", flow=fid).ok for _ in range(300))
    assert ok_n == 300, "flow runs failed"
    out["flow_5_steps_per_s"] = round(300 / (time.perf_counter() - t), 1)
    people = sorted(k for k, m in w.state.scan("minds") if m["kind"] == "person")
    t = time.perf_counter()
    k = 0
    for p in people:
        for _ in range(10):
            r = ex("deal.propose", to=p, as_org="o:bbco", title="x", accounts={"0": acct}, terms=[
                {"type": "payment", "from": "$me", "to": "$you", "amount": 100, "every_s": None, "count": None,
                 "start_in_s": 0}])
            k += r.ok
    assert k == 10 * len(people), f"only {k} deal proposals succeeded"
    out["deal_proposals_per_s"] = round(k / (time.perf_counter() - t), 1)
    t = time.perf_counter()
    w.advance_time(w.now + 2 * DAY)
    out["deal_replies_and_day_s"] = round(time.perf_counter() - t, 3)
    # industrial and mixed worlds
    for fam, extra in (("open_industry", {}), ("open_mixed", {"n_consumers": 2000})):
        wi = World.create(generate({"family": fam, "overrides": extra}, seed))
        t = time.perf_counter()
        wi.advance_time(days * DAY)
        el = time.perf_counter() - t
        n_m = wi.state.count("minds")
        rec = {"sim_day_s": round(el / days, 3), "events_per_day": round(wi.seq / days), "minds": n_m,
               "runs": wi.state.count("runs"), "retail_units": sum(r.get("sold_total", 0) for _, r in wi.state.scan("retail"))}
        t = time.perf_counter()
        blob = wi.snapshot_blob()
        rec["snapshot_ms"] = round((time.perf_counter() - t) * 1000, 1)
        rec["snapshot_kb"] = round(len(blob) / 1024, 1)
        t = time.perf_counter()
        wf_ = wi.fork("bench-fork")
        rec["fork_ms"] = round((time.perf_counter() - t) * 1000, 1)
        t = time.perf_counter()
        wr = World.replay(wi.spec, [e.to_dict() for e in wi.log])
        rec["replay_events_per_s"] = round(wi.seq / (time.perf_counter() - t))
        rec["replay_identical"] = wr.digest() == wi.digest()
        del wf_
        out[fam] = rec
    out["minds_scaling"] = bench_minds()
    out["rss_mb"] = round(rss_mb(), 1)
    return out


def bench_minds(sizes=(50, 500, 2000), days=2, seed=1):
    """Cost of Tier-2 minds: simulated people (workers with minds, daily routines, negotiation) at scale. The
    consumer population is kept small to isolate the cost of minds."""
    from crucible.core.kernel import World
    from crucible.worldgen import generate
    _setup()
    out = []
    for n in sizes:
        t = time.perf_counter()
        w = World.create(generate({"family": "open_digital", "overrides": {"n_consumers": 200, "workers": n}}, seed))
        tg = time.perf_counter() - t
        w.advance_time(DAY)
        t = time.perf_counter()
        w.advance_time((1 + days) * DAY)
        el = (time.perf_counter() - t) / days
        out.append({"minds": w.state.count("minds"), "genesis_s": round(tg, 2), "sim_day_s": round(el, 3),
                    "snapshot_kb": round(len(w.snapshot_blob()) / 1024, 1)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/benchmarks/bench.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", choices=["open"], help="run only one section and merge it into --out")
    a = ap.parse_args(argv)
    res = {"machine": {"cpu_count": os.cpu_count(), "python": platform.python_version(), "platform": platform.platform()},
           "t": time.strftime("%Y-%m-%d %H:%M:%S")}
    if a.only == "open":            # add/refresh only the open-world section of an existing results file
        prev = json.load(open(a.out)) if os.path.exists(a.out) else {}
        prev["open"] = bench_open(2 if a.quick else 4)
        print("open", prev["open"], flush=True)
        with open(a.out, "w") as f:
            json.dump(prev, f, indent=1)
        print("wrote", a.out)
        return
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
    res["open"] = bench_open(2 if a.quick else 4)
    print("open", res["open"], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
