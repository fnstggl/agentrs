"""Gate V workloads that are not a profile benchmark (crucible.scale.bench5): event-driven computation (V.5),
content-representation caching across worlds (V.6) and the 10M-identity capacity run (V.2).

**Event-driven computation (13.2).** The population activates an individual only when its own rhythm or a recorded
trigger (`population.wake`: a message, a notification, a share) makes it due, and lets class-2 participants that
become due between decision rounds wait for one batched decision (event aggregation). `event_driven()` runs one
world and, on every population tick, evaluates the participants' decision policies twice with the population's own
decision code and models: for the individuals that tick activated (what event-driven computation computes), and for
every eligible individual (what a dense per-tick simulation would compute: the class-1 session decision, the
activity logits and the class-2 shared network). Both evaluations are observers (their own random streams, no
event, no state change), so the world's trajectory - its event hash chain - is exactly the one of the same world
run without them (checked: equal outcomes). The speedup is the dense evaluation's CPU over the event-driven
evaluation's CPU for the same ticks; activations, triggered wake-ups and cost per activation are reported with it.

**Caching (13.5).** `cache_workload()` runs three worlds in separate processes on one fresh representation cache:
two replicas of one world (same spec and seed: a counterfactual arm shares its content up to the intervention) and
one world of another seed. Hit rate = served representations / requested representations (world records, memory
and disk cache hits over all lookups); encoder runs per distinct artifact; representations reused across worlds.

**Capacity (V.2).** `capacity()` runs the internet profile of the one population (10M identities, columnar path) with
the social internet's posts platform, as Gate I does (social.cascade), until 10M qualifying distinct consumption
events are recorded, and recounts them from the recorded decision blobs; resource use is measured by the governor.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time

import numpy as np

DAY = 86400


# ====================================================================== event-driven computation
class DenseEvaluator:
    """World listener: on each population tick, the decision policies for the activated and for all eligible rows."""

    def __init__(self, world):
        self.w = world
        self.ticks = 0
        self.cpu = {"event": 0.0, "dense": 0.0}
        self.rows = {"event": 0, "dense": 0}
        self.rng = np.random.default_rng(12345)          # its own stream: never the world's
        self.net = None
        self.dims = None
        try:
            mod = world.modules.get("pop_adaptive")
            if mod is not None:
                from ..population import nets
                self.net = nets.AdaptiveNet(mod.obs_dim, mod.cand_dim, mod.M, int(mod.cfg["hidden"]))
                nets.init_params(self.net, np.random.default_rng(1))
                self.net.eval()
                self.dims = (mod.obs_dim, mod.cand_dim, mod.M, mod.K + 1)
        except Exception:                                 # noqa: BLE001 - no adaptive class in this world
            self.net = None

    def __call__(self, ev) -> None:
        if ev.type != "proc" or ev.payload.get("proc") != "pop.tick":
            return
        from ..substrate import columns
        w = self.w
        now = int(ev.payload.get("at", w.now))
        cls = columns.col(w, "pop", "class")
        last = columns.col(w, "pop", "last_session_at")
        eligible = np.flatnonzero(cls <= 2)
        activated = eligible[last[eligible] >= now - 1]
        self.ticks += 1
        for arm, idx in (("event", activated), ("dense", eligible)):
            if idx.size == 0:
                continue
            t = time.process_time()
            self._policies(idx, now)
            self.cpu[arm] += time.process_time() - t
            self.rows[arm] += int(idx.size)

    def _policies(self, idx: np.ndarray, now: int) -> None:
        import torch
        from ..population import adaptive as AD
        from ..substrate import columns
        w = self.w
        c = lambda n: columns.col(w, "pop", n)[idx]        # noqa: E731
        ds = w.runtime.get("pop.models") or {}
        bud = c("attention_budget_min").astype(np.float64)
        left = c("attention_left_min").astype(np.float64)
        habit = c("habit").astype(np.float64)
        minute = (now % DAY) // 60
        cs, cl = c("commit_start_min").astype(np.int64), c("commit_len_min").astype(np.int64)
        hour = minute / 60.0
        F = {"attention_ratio": left / np.maximum(bud, 1e-3), "satisfaction": c("satisfaction").astype(np.float64),
             "habit_max": habit.max(axis=1) if habit.size else np.zeros(idx.size),
             "experience_log": np.log1p(c("experience").astype(np.float64)),
             "in_commitment": ((cl > 0) & (((minute - cs) % 1440) < cl)).astype(float),
             "woken": (c("wake_at") <= now).astype(float),
             "hour_sin": np.full(idx.size, math.sin(2 * math.pi * hour / 24)),
             "hour_cos": np.full(idx.size, math.cos(2 * math.pi * hour / 24))}
        if "session_start" in ds:
            ds["session_start"].sample(F, self.rng)
        logits = habit + self.rng.gumbel(size=habit.shape) * 0.6
        np.argmax(logits, axis=1)
        if self.net is not None:
            sel = idx[columns.col(w, "pop", "class")[idx] == 2]
            if sel.size:
                obs = AD.obs_features(w, sel)
                mem = columns.col(w, "pop", "memory")[sel].astype(np.float32)
                cand = np.zeros((sel.size, self.dims[3], self.dims[1]), np.float32)
                with torch.no_grad():
                    self.net(torch.from_numpy(obs), torch.from_numpy(mem), torch.from_numpy(cand))


def event_driven(kind: str = "population", profile: str = "standard", seed: int = 0, hours: float = 24.0,
                 warm_hours: float = 8.0) -> dict:
    """Run a world with and without the dense observer from the same snapshot; see the module docstring."""
    import torch
    torch.set_num_threads(1)
    from ..core.kernel import World
    from . import bench5
    from ..substrate import columns
    w = bench5.make_world(kind, profile, seed)
    w.advance_time(w.now + int(warm_hours * 3600))         # into the day: sessions happen
    blob = w.snapshot_blob()
    end = w.now + int(hours * 3600)
    base = w                                  # stays open: its lease keeps the snapshot's blobs in the process pool
    w = World.from_blob(blob, world_id="event-driven-arm", fork=True)   # both arms: forks of the same snapshot
    counts = bench5.EventCounter()
    w.listeners.append(counts)
    t, c = time.time(), time.process_time()
    w.advance_time(end)
    plain = {"wall_s": time.time() - t, "cpu_s": time.process_time() - c, "hash": w.last_hash, "seq": w.seq}
    n = int(columns.rows(w, "pop"))
    w2 = World.from_blob(blob, world_id="event-driven-arm", fork=True)
    dense = DenseEvaluator(w2)
    w2.listeners.append(dense)
    t, c = time.time(), time.process_time()
    w2.advance_time(end)
    dense_run = {"wall_s": time.time() - t, "cpu_s": time.process_time() - c, "hash": w2.last_hash, "seq": w2.seq}
    w2.close()
    w.close()
    base.close()
    wakes = counts.batches.get("pop.wake", 0)
    ticks = max(1, dense.ticks)
    out = {"world": kind, "profile": profile, "seed": seed, "hours": hours, "individuals": n,
           "equal_outcomes": plain["hash"] == dense_run["hash"] and plain["seq"] == dense_run["seq"],
           "ticks": dense.ticks, "activations": int(counts.activations), "sessions": int(counts.sessions),
           "policy_rows_event": dense.rows["event"], "policy_rows_dense": dense.rows["dense"],
           "policy_cpu_event_s": round(dense.cpu["event"], 4), "policy_cpu_dense_s": round(dense.cpu["dense"], 4),
           "event_driven_speedup": round(dense.cpu["dense"] / max(dense.cpu["event"], 1e-9), 3),
           "activation_ratio": round(dense.rows["dense"] / max(1, dense.rows["event"]), 3),
           "world_cpu_s": round(plain["cpu_s"], 3), "world_with_dense_cpu_s": round(dense_run["cpu_s"], 3),
           "world_cpu_ratio": round(dense_run["cpu_s"] / max(plain["cpu_s"], 1e-9), 3),
           "world_cpu_per_activation_ms": round(1000 * plain["cpu_s"] / max(1, counts.activations), 4),
           "wake_batches": int(wakes), "mean_rows_per_tick_event": round(dense.rows["event"] / ticks, 2),
           "mean_rows_per_tick_dense": round(dense.rows["dense"] / ticks, 2)}
    return out


# ====================================================================== caching across worlds
CACHE_CHILD = r'''
import json, sys, time
import crucible
crucible.load_all_modules()
from crucible.scale import bench5
from crucible.semantic import space
kind, profile, seed, hours, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
s0 = space.stats()
w = bench5.make_world(kind, profile, seed)
w.advance_time(w.now + int(hours * 3600))
refs = sorted(k for k in w.state.tables.get("space_items", {}))
st = space.stats()
json.dump({"stats": st, "items": len(refs), "item_keys": refs[:200000], "seq": w.seq}, open(out, "w"))
'''


def cache_workload(cache_dir: str, work: str, kind: str = "population", profile: str = "development",
                   hours: float = 12.0, seeds=(0, 0, 1)) -> dict:
    """Three worlds (two replicas of one seed and another seed), one process each, sharing one fresh cache."""
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    env = dict(os.environ)
    env["CRUCIBLE_CACHE"] = cache_dir
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    runs = []
    for i, s in enumerate(seeds):
        out = os.path.join(work, f"cache-{i}.json")
        p = subprocess.run([sys.executable, "-c", CACHE_CHILD, kind, profile, str(s), str(hours), out], env=env,
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"cache workload world {i} failed: {p.stderr[-2000:]}")
        runs.append(json.load(open(out)))
    hits = req = computed = 0
    per = []
    for r in runs:
        st = r["stats"]
        ch = st.get("cache") or {}
        h = int(st.get("world_hits", 0)) + int(ch.get("memory_hits", 0)) + int(ch.get("disk_hits", 0))
        m = int(ch.get("misses", 0))
        comp = int(sum((st.get("computed") or {}).values()))
        hits += h
        req += h + m
        computed += comp
        per.append({"world_hits": int(st.get("world_hits", 0)), "memory_hits": int(ch.get("memory_hits", 0)),
                    "disk_hits": int(ch.get("disk_hits", 0)), "misses": m, "computed": comp, "items": r["items"]})
    keys = [set(r["item_keys"]) for r in runs]
    distinct = set().union(*keys)
    shared = len(keys[0] & keys[1]) + len((keys[0] | keys[1]) & keys[2])
    return {"worlds": len(runs), "requests": req, "hits": hits, "hit_rate": round(hits / max(1, req), 5),
            "computed": computed, "distinct_items": len(distinct),
            "encoder_runs_per_artifact": round(computed / max(1, len(distinct)), 4),
            "cross_world_reused_items": int(shared), "per_world": per,
            "disk_hits_in_later_worlds": int(sum(p["disk_hits"] for p in per[1:]))}


# ====================================================================== 10M capacity
def capacity(n: int = 10_000_000, seed: int = 7, max_hours: int = 30, target_events: int = 10_000_000) -> dict:
    """The internet profile with the social internet's posts platform (social.cascade, as Gate I) until
    `target_events` qualifying distinct consumption events; recounted from the recorded decision blobs."""
    from ..core.kernel import World
    from ..social import audience as AUD
    from ..social import cascade
    from . import bench5
    t0, c0 = time.time(), bench5.cpu_s()
    w = World.create(cascade.pop_internet_spec(n, seed, positive_control=True))
    item = cascade.positive_control_post(w)
    create = {"wall_s": round(time.time() - t0, 1), "cpu_s": round(bench5.cpu_s() - c0, 1),
              "rss_mb": round(bench5.vm_hwm_mb(), 1)}
    progress = []
    while w.now < max_hours * 3600:
        progress += cascade.run(w, w.now + 3600, step=3600)
        if progress[-1]["distinct_views"] >= target_events:
            break
    run_wall = time.time() - t0 - create["wall_s"]
    m = cascade.measure(w, item)
    out = {"identities": int(m["distinct_identities"]), "identities_with_decisions": int(m["identities_with_decisions"]),
           "qualifying_distinct_consumption_events": int(m["qualifying_distinct_consumption_events"]),
           "recount_mismatches": int(m["total_recount_mismatches"]), "simulated_hours": w.now / 3600,
           "create": create, "run_wall_s": round(run_wall, 1), "wall_s": round(time.time() - t0, 1),
           "cpu_s": round(bench5.cpu_s() - c0, 1), "peak_ram_mb": round(bench5.vm_hwm_mb(), 1), "events": int(w.seq),
           "item_unique_viewers": int(AUD.item_col(w, "viewers")[item]), "estimated": False,
           "progress": [(r["t"] // 3600, r["distinct_views"], r["wall_s"]) for r in progress]}
    w.close()
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("event_driven", "cache", "capacity"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--world", default="population")
    ap.add_argument("--profile", default="standard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--n", type=int, default=10_000_000)
    ap.add_argument("--work", default=None)
    ap.add_argument("--max-hours", type=int, default=30)
    ap.add_argument("--target", type=int, default=10_000_000)
    a = ap.parse_args(argv)
    if a.what == "event_driven":
        rep = event_driven(a.world, a.profile, a.seed, a.hours)
    elif a.what == "cache":
        rep = cache_workload(os.path.join(a.work, "cache"), a.work, a.world, a.profile, a.hours)
    else:
        rep = capacity(a.n, a.seed, max_hours=a.max_hours, target_events=a.target)
    with open(a.out + ".tmp", "w") as fh:
        json.dump(rep, fh, default=str)
    os.replace(a.out + ".tmp", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
