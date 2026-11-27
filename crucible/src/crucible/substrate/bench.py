"""Substrate performance measurements (workstream A; reported in docs/v5/components/A.md).

    PYTHONPATH=src python -m crucible.substrate.bench [--rows 1000000] [--table-rows 10000000] [--store sqlite]

Measures, single-threaded, on the real kernel:

* one `cols.batch` of `rows` rows into a column table of `table_rows` rows: blob encoding (npz), SHA-256, store write,
  the whole recorded transaction (handler + event + applier with row journaling), replay of that event from the
  store (blob read, hash check, decode, apply) and regeneration;
* the per-event cost of the kernel's subscriber hook: emitting small events with no subscriber registered (4.0 path),
  with a subscriber registered for another type, and with a no-op subscriber on the emitted type.

Store growth over a long run (`store_growth`; D-78, wave 3b):

    PYTHONPATH=src python -m crucible.substrate.bench growth --world kernel|population --days 365 [--json out.json]

A world in a SQLite store advances day by day; every `every_days` it takes a snapshot and runs
`retention.compact(prune=True)`, and records per snapshot: its bytes and the retained snapshot bytes, live blob bytes,
blob count, event payload bytes, `artifacts` rows, compaction seconds and wall seconds per simulated day. At the end a
fresh store object replays the log from genesis and resumes the latest snapshot; both must equal the live digest.
`kernel`: the substrate bench world with `batches_per_day` cohort updates (`cols.batch`, one system array blob each)
per simulated day - the snapshot must not grow with the number of system blobs written. `population`: the population
world of A.json's 365-day measurement (development profile, seed 11, 400 individuals, 6 sites, adaptive policies with
daily checkpoints, no strategic class); it runs the shared space's text path (a GPU slot).
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import numpy as np

from ..core import schema as S
from ..core.kernel import Module, World, WorldSpec, applier, endpoint, register_module
from . import artifacts, bus, columns


@register_module
class SubstrateBench(Module):
    """Benchmark driver: one column table and two recorded entry points (a batch process, an event burst)."""
    name = "substrate_bench"
    host = "sbench"
    depends = ("cols",)

    def install(self, world):
        columns.declare(world, "bench", int(self.config.get("n", 1000)),
                        {"x": ("float32", (), 0.0), "y": ("float32", (), 0.0), "z": ("int32", (), 0)})
        if self.config.get("subscribe") == "other":
            bus.subscribe(world, "sbench.other", lambda ctx, ev: None, owner=self.name)
        elif self.config.get("subscribe") == "same":
            bus.subscribe(world, "sbench.tick", lambda ctx, ev: None, owner=self.name)

    @endpoint("batch", S.obj({"rows": S.s_int(minimum=1), "seed": S.s_int(minimum=0)}, ["rows", "seed"]),
              doc="one cohort update of `rows` random rows (operator benchmark)")
    def ep_batch(self, ctx, rows: int, seed: int):
        rng = ctx.rng("bench", seed)
        n = columns.rows(ctx.world, "bench")
        idx = np.sort(rng.choice(n, size=rows, replace=False)) if rows < n else np.arange(n)
        columns.batch(ctx, "bench", "bench.update", idx,
                      {"x": rng.random(rows, dtype=np.float32), "y": rng.random(rows, dtype=np.float32),
                       "z": rng.integers(0, 1000, rows).astype(np.int32)}, mode="add")

    @endpoint("burst", S.obj({"k": S.s_int(minimum=1, maximum=1_000_000)}, ["k"]), doc="emit k small events")
    def ep_burst(self, ctx, k):
        for i in range(k):
            ctx.emit("sbench.tick", {"i": i}, vis="internal")
        return {"k": k}

    @applier("sbench.tick")
    def ap_tick(self, ev):
        return None

    @applier("sbench.other")
    def ap_other(self, ev):
        return None


def _world(n, store=None, subscribe=None, wid=None):
    import crucible
    crucible.load_all_modules()
    cfg = {"n": n}
    if subscribe:
        cfg["subscribe"] = subscribe
    spec = WorldSpec("substrate-bench", 7, [{"name": "identity", "config": {}}, {"name": "bank", "config": {}},
                                            {"name": "admin", "config": {}}, {"name": "artifacts", "config": {}},
                                            {"name": "cols", "config": {}},
                                            {"name": "substrate_bench", "config": cfg}])
    return World.create(spec, world_id=wid, store=store)


def batch_throughput(rows: int = 1_000_000, table_rows: int = 10_000_000, store: str = "sqlite") -> dict:
    tmp = tempfile.mkdtemp(prefix="crucible-sbench-")
    st = None
    if store == "sqlite":
        from ..storage.sqlite import SQLiteStore
        st = SQLiteStore(os.path.join(tmp, "bench.db"))
    elif store == "memory":
        from ..storage.memory import MemoryStore
        st = MemoryStore()
    t0 = time.perf_counter()
    w = _world(table_rows, store=st, wid="sbench")
    t_create = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(table_rows, size=rows, replace=False)) if rows < table_rows else np.arange(table_rows)
    arrays = {"idx": idx.astype(np.int32), "v.x": rng.random(rows, dtype=np.float32),
              "v.y": rng.random(rows, dtype=np.float32), "v.z": rng.integers(0, 1000, rows).astype(np.int32)}
    t0 = time.perf_counter()
    data = artifacts.npz_bytes(arrays)
    t_npz = time.perf_counter() - t0
    t0 = time.perf_counter()
    artifacts.sha256_hex(data)
    t_sha = time.perf_counter() - t0
    t0 = time.perf_counter()
    artifacts.npz_arrays(data)
    t_decode = time.perf_counter() - t0
    # the recorded transaction: one process run = handler (draw, blob, store write, event) + applier
    w.system_call("admin.join", {"handle": "bench", "endowment": 0})
    with_events = w.seq

    t0 = time.perf_counter()
    w.system_call("sbench.batch", {"rows": rows, "seed": 1})
    t_tx = time.perf_counter() - t0
    w.flush()
    ev = [e for e in w.log if e.type == "cols.batch"][-1]
    blob_bytes = len(artifacts.get(w, ev.payload["ref"]))
    # isolated apply cost (a second identical batch on top, measured inside a transaction)
    evs = [e.to_dict() for e in w.log]
    d0 = w.digest()
    t0 = time.perf_counter()
    w.digest()
    t_digest = time.perf_counter() - t0
    # replay from genesis, reading the blob from the store (fresh world object, empty caches)
    artifacts._POOL.clear()
    t0 = time.perf_counter()
    r = World.replay(w.spec, evs, world_id="sbench-replay", store=st)
    t_replay = time.perf_counter() - t0
    replay_ok = r.digest() == d0
    t0 = time.perf_counter()
    g = World.regenerate(w.spec, evs, world_id="sbench-regen")
    t_regen = time.perf_counter() - t0
    regen_ok = g.last_hash == w.last_hash
    # snapshot and fork with the populated columns
    t0 = time.perf_counter()
    blob = w.snapshot_blob()
    t_snap = time.perf_counter() - t0
    t0 = time.perf_counter()
    f = World.from_blob(blob, world_id="sbench-fork")
    t_fork = time.perf_counter() - t0
    fork_ok = f.digest() == d0
    out = {"rows": rows, "table_rows": table_rows, "store": store, "columns": 3, "blob_bytes": int(blob_bytes),
           "world_create_s": round(t_create, 4), "npz_encode_s": round(t_npz, 4), "sha256_s": round(t_sha, 4),
           "npz_decode_s": round(t_decode, 4), "recorded_transaction_s": round(t_tx, 4),
           "rows_per_s_transaction": round(rows / t_tx), "replay_from_genesis_s": round(t_replay, 4),
           "replay_exact": replay_ok, "regenerate_s": round(t_regen, 4), "regeneration_exact": regen_ok,
           "state_digest_s": round(t_digest, 4), "snapshot_s": round(t_snap, 4), "snapshot_bytes": len(blob),
           "fork_from_blob_s": round(t_fork, 4), "fork_exact": fork_ok, "events_in_log": len(evs),
           "events_before_batch": with_events}
    w.close()
    if st is not None:
        st.close()
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)      # the benchmark store is scratch data
    return out


def event_overhead(k: int = 20000, reps: int = 5) -> dict:
    res = {}
    for mode in (None, "other", "same"):
        w = _world(10, subscribe=mode, wid=f"sb-{mode}")
        w.system_call("admin.join", {"handle": "bench", "endowment": 0})
        best = None
        for _ in range(reps):
            t0 = time.perf_counter()
            r = w.execute("p:bench", {"tool": "sbench.burst", "args": {"k": k}})
            dt = time.perf_counter() - t0
            assert r.ok, r.error
            best = dt if best is None else min(best, dt)
        res[str(mode or "none")] = round(best / k * 1e6, 3)
    return {"events_per_command": k, "us_per_event": res,
            "subscriber_hook_overhead_us": round(res["other"] - res["none"], 3),
            "noop_subscriber_cost_us": round(res["same"] - res["none"], 3)}


def _payload_bytes(db: str, wid: str) -> int:
    import sqlite3
    c = sqlite3.connect(db)
    try:
        cols = [r[1] for r in c.execute("PRAGMA table_info(events)")]
        col = "payload" if "payload" in cols else cols[-1]
        return int(c.execute(f"SELECT COALESCE(SUM(LENGTH({col})), 0) FROM events WHERE world_id=?", (wid,))
                   .fetchone()[0])
    finally:
        c.close()


def store_growth(world: str = "kernel", days: int = 365, *, every_days: int = 7, db: str | None = None,
                 n: int = 20_000, rows: int = 2_000, batches_per_day: int = 4, out: str | None = None) -> dict:
    """Store growth of a `days`-long run with a snapshot and `retention.compact(prune=True)` every `every_days`
    (see the module docstring). Returns {"rows": [per snapshot], "end": {...}}; `out` is rewritten after every row."""
    from ..storage.sqlite import SQLiteStore
    from . import retention
    tmp = None
    if db is None:
        tmp = tempfile.mkdtemp(prefix="crucible-growth-")
        db = os.path.join(tmp, "growth.db")
    for f in (db, db + "-wal", db + "-shm"):
        if os.path.exists(f):
            os.remove(f)
    st = SQLiteStore(db)
    wid = f"growth-{world}"
    if world == "kernel":
        w = _world(n, store=st, wid=wid)
    elif world == "population":
        import crucible
        crucible.load_all_modules()
        from ..population.worlds import population_world
        w = World.create(population_world("development", 11, n=400, sites=6, strategic=False, adaptive=True),
                         world_id=wid, store=st)
    else:
        raise ValueError(f"unknown growth world {world!r} (kernel or population)")
    res = {"schema": "crucible-v5-store-growth/1", "world": world, "days": int(days), "every_days": int(every_days),
           "rows": [], "done": False}
    if world == "kernel":
        res["kernel"] = {"n": int(n), "rows_per_batch": int(rows), "batches_per_day": int(batches_per_day)}

    def dump():
        if out:
            with open(out, "w") as fh:
                json.dump(res, fh, indent=1)
    t_run = last = time.time()
    for day in range(1, int(days) + 1):
        if world == "kernel":
            for b in range(int(batches_per_day)):
                w.system_call("sbench.batch", {"rows": int(rows), "seed": day * 1000 + b})
        w.advance_time(day * 86400)
        if day % int(every_days) and day != int(days):
            continue
        sim_s = time.time() - last
        w.snapshot()
        t = time.time()
        comp = retention.compact(w, prune=True)
        comp_s = time.time() - t
        s = retention.stats(w)
        latest = max(st.list_snapshots(wid), key=lambda x: x["seq"])
        res["rows"].append({"day": day, "events": int(w.seq), "wall_s_per_day": round(sim_s / every_days, 3),
                            "compact_s": round(comp_s, 3), "packed": comp["pack"]["packed"],
                            "dropped": comp["collect"]["dropped"], "snapshots": s["snapshots"],
                            "snapshot_bytes_latest": int(latest.get("bytes") or 0),
                            "snapshot_bytes_retained": s["snapshot_bytes"], "live_blob_bytes": s["live_blob_bytes"],
                            "blobs": s["blobs"], "blobs_packed": s["blobs_packed"],
                            "event_payload_bytes": _payload_bytes(db, wid),
                            "artifact_rows": int(w.state.count("artifacts")), "sqlite_bytes": os.path.getsize(db)})
        dump()
        last = time.time()
    w.flush()
    live = w.digest()
    end = {"events": int(w.seq), "run_s": round(time.time() - t_run, 1)}
    artifacts._CHUNKS.clear()
    t = time.time()
    st2 = SQLiteStore(db)
    rep = World.replay(w.spec, st2.read_events(wid), world_id=f"{wid}-replay", store=st2)
    end["replay_from_genesis_exact"] = rep.digest() == live
    end["replay_s"] = round(time.time() - t, 1)
    del rep
    t = time.time()
    end["resume_exact"] = World.load(SQLiteStore(db), wid).digest() == live
    end["resume_s"] = round(time.time() - t, 1)
    w.close()
    res.update(end=end, done=True)
    dump()
    if tmp is not None:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return res


def main(argv=None) -> int:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["growth"]:
        ap = argparse.ArgumentParser(prog="python -m crucible.substrate.bench growth")
        ap.add_argument("--world", choices=("kernel", "population"), default="kernel")
        ap.add_argument("--days", type=int, default=365)
        ap.add_argument("--every-days", type=int, default=7)
        ap.add_argument("--db", default=None, help="SQLite file (default: a temporary one, deleted at the end)")
        ap.add_argument("--json", default=None, help="write the result here (rewritten after every snapshot)")
        a = ap.parse_args(argv[1:])
        out = store_growth(a.world, a.days, every_days=a.every_days, db=a.db, out=a.json)
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
        return 0
    ap = argparse.ArgumentParser(prog="python -m crucible.substrate.bench")
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--table-rows", type=int, default=10_000_000)
    ap.add_argument("--store", choices=("sqlite", "memory", "none"), default="sqlite")
    ap.add_argument("--events", type=int, default=20000)
    ap.add_argument("--json", default=None, help="write the result here")
    a = ap.parse_args(argv)
    out = {"schema": "crucible-v5-substrate-bench/1", "batch": batch_throughput(a.rows, a.table_rows, a.store),
           "events": event_overhead(a.events)}
    s = json.dumps(out, indent=1)
    print(s)
    if a.json:
        with open(a.json, "w") as fh:
            fh.write(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
