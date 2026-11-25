"""Substrate performance measurements (workstream A; reported in docs/v5/components/A.md).

    PYTHONPATH=src python -m crucible.substrate.bench [--rows 1000000] [--table-rows 10000000] [--store sqlite]

Measures, single-threaded, on the real kernel:

* one `cols.batch` of `rows` rows into a column table of `table_rows` rows: blob encoding (npz), SHA-256, store write,
  the whole recorded transaction (handler + event + applier with row journaling), replay of that event from the
  store (blob read, hash check, decode, apply) and regeneration;
* the per-event cost of the kernel's subscriber hook: emitting small events with no subscriber registered (4.0 path),
  with a subscriber registered for another type, and with a no-op subscriber on the emitted type.
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
    blob_bytes = w.state.tables["artifacts"][ev.payload["ref"]]["size"]
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


def main(argv=None) -> int:
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
