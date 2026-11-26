"""Crucible 5.0 performance benchmark (MASTER_SPEC 13.8, Gate V, SP-13.8): the twelve quantities per population profile,
measured on the real world, and an equivalent-workload comparison with Crucible 4.0.

One run = one world of one profile and seed, advanced day by day for `days` simulated days, with a SQLite store whose
blobs are compacted by workstream A's retention after every day (crucible.substrate.retention.compact):

| quantity (Part 13.8) | how it is measured |
|---|---|
| environment transitions per second | recorded kernel events per wall second (and world transactions per second) |
| participant decisions per second | individual decisions recorded in the run: session decisions of `pop.session` batches plus the rows of every batch carrying decision provenance (`population.record_decisions`) |
| simulated days per wall-clock hour | also per CPU-hour of the world process and its children (tenant load does not move it) |
| CPU utilisation | CPU seconds of the process and its children per wall second |
| peak RAM | the process's high-water mark (VmHWM) and its children's peak |
| peak VRAM | torch's peak reserved memory when the process initialised CUDA (0.0 measured otherwise) |
| model parameter counts | counted on the loaded objects: the shared class-2 policies, class-3 policy groups, the shared-space encoders and language model, the semantic service when loaded |
| inference token counts | tokens the shared space's language model scored, items each encoder computed, recorded inference calls by kind, generated tokens of the semantic service |
| event storage growth | bytes per simulated day and per individual-day: raw = event rows as the store holds them (read with one SQL aggregate per day, outside the measured CPU) + array-blob bytes written (before compaction); resident = growth of the database pages in use after compaction and a WAL checkpoint (no VACUUM: it needs a second copy of the store on disk) |
| snapshot size | bytes and seconds of `World.snapshot_blob()` at the end; a fork from it must have the same state digest |
| world-generation cost | wall and CPU seconds, events and RSS of `World.create`, plus the first simulated day's CPU beyond a median day (deployments and warm-up the world's first processes do; the T5 check uses the sum) |
| approximation error | from crucible.scale.approx (the estimated large-population mode against individual simulation) |

Class activity is measured per participant class (rows; individuals whose session count grew; decisions recorded by
the class's policies), because a profile whose class-4 rows never act does not have four active classes.

    PYTHONPATH=src python -m crucible.scale.bench5 run --world civ --profile development --seed 0 --days 30 \\
        --store-dir /tmp/x --out run.json
    PYTHONPATH=src python -m crucible.scale.bench5 v4 --consumers 1000 --days 30 --out v4.json   # 4.0 equivalent

`suite()` launches the runs as governed sub-jobs (crucible.scale.slots, declared grants in every dimension).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter

DAY = 86400
MB = 1 << 20
CIV_PROFILE = "balanced"
PROFILES = ("development", "standard", "large")
WORLD_KINDS = ("civ", "population")
QUANTITIES = ("transitions_per_s", "decisions_per_s", "sim_days_per_wall_hour", "cpu_utilisation", "peak_ram_mb",
              "peak_vram_mb", "model_parameter_counts", "inference_token_counts", "event_storage_growth",
              "snapshot_size", "world_generation_cost", "approximation_error")
#: grants a run declares (per profile): RAM MB, disk MB, GPU MB. The shared space's language model and encoders run
#: on the GPU when the job is granted one (crucible.semantic.space.lm "selective GPU"); on CPU they dominate a run.
#: RAM grants are about 1.6-2x the measured peak (2026-11-26: development 2.7 GB over 30 days, standard 3.1 GB and
#: large 3.9 GB over their first days) so that a run reserves no more of the shared RAM budget than it can use.
#: Disk grants (store growth, the run's cold machine cache included): measured 2026-11-26 on 30-day civ runs, standard
#: 758 MB and large about 120 MB per simulated day (3.6 GB at 30 days), so large is granted 5 GB; with ~10 GB free on
#: the machine, the governor's free-disk admission lets at most two large runs hold their grants at once.
#: The internet profile (the 10M capacity run, V.2) measured 16.4 GB summed PSS and 1.4 GB of GPU memory on
#: 2026-11-26 (596 s); its grant is 1.5x / 1.8x that, below T4's 32 GB ceiling, which is a target, not a grant.
GRANTS = {"development": (4500, 1500, 3500), "standard": (6000, 1500, 3500), "large": (8000, 5000, 3500),
          "internet": (25000, 5120, 2500)}
GPU_S = 4 * 86400                     # GPU-time grant of one run: seconds of held CUDA context (the run's wall time)


# ====================================================================== process measurements
def cpu_s() -> float:
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def vm_hwm_mb() -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def peak_vram_mb() -> tuple[float, bool]:
    """(peak reserved MB, whether this process used CUDA). Never imports torch just to ask."""
    t = sys.modules.get("torch")
    if t is None:
        return 0.0, False
    try:
        if t.cuda.is_available() and t.cuda.is_initialized():
            return float(t.cuda.max_memory_reserved()) / MB, True
    except Exception:              # noqa: BLE001 - a CUDA query that fails means no CUDA use here
        pass
    return 0.0, False


class EventCounter:
    """A world listener counting what a run records (observers never change a trajectory)."""

    def __init__(self, count_bytes: bool = True):
        self.count_bytes = bool(count_bytes)   # canonical bytes per event (costs as much as the kernel's hashing:
        self.types = Counter()                 # benchmark runs measure event bytes from the store instead)
        self.batches = Counter()
        self.decisions = 0
        self.decisions_by = Counter()
        self.sessions = 0
        self.activations = 0
        self.public = 0
        self.public_kinds = Counter()
        self.infer = Counter()
        self.actors = set()
        self.event_bytes = 0
        self.semantic_calls = 0               # recorded semantic-service batches (infer.semantic.* outcomes)
        self.semantic_rows = 0
        self.model_refs: list = []            # (class, output ref) of class-2/3 recorded calls, sized after each day
        self.batch_sizes = {"class2": Counter(), "class3": Counter()}
        from ..core.canonical import canonical_bytes
        self._cb = canonical_bytes

    def __call__(self, ev) -> None:
        self.types[ev.type] += 1
        if self.count_bytes:
            self.event_bytes += len(self._cb(ev.body()))
        if ev.vis == "public":
            self.public += 1
            self.public_kinds[ev.type] += 1
        if ev.actor and not ev.actor.startswith("sys"):
            self.actors.add(ev.actor)
        if ev.type == "cols.batch":
            p = ev.payload
            k = p.get("kind", "")
            self.batches[k] += 1
            ex = p.get("extra") or {}
            if k == "pop.session":
                self.sessions += int(ex.get("sessions", 0))
                self.activations += int(ex.get("activations", 0))
                self.decisions += int(ex.get("sessions", 0))
                self.decisions_by["session"] += int(ex.get("sessions", 0))
            elif isinstance(ex.get("decision"), dict):
                n = int(p.get("n", 0))
                self.decisions += n
                self.decisions_by[str(ex["decision"].get("model_id"))] += n
        elif ev.type == "ext":
            for o in ev.payload.get("outcomes") or ():
                if isinstance(o, dict) and str(o.get("kind", "")).startswith("infer."):
                    self.infer[o["kind"]] += 1
                    if o["kind"].startswith("infer.semantic."):      # the shared semantic service: one reply per row
                        out = (o.get("out") or {}).get("output")
                        self.semantic_calls += 1
                        self.semantic_rows += len(out) if isinstance(out, list) else 1
                    parts = o["kind"].split(".")
                    cls = BATCH_KINDS.get(".".join(parts[1:3])) if len(parts) >= 4 else None
                    rec = o.get("out")
                    ref = rec.get("output_ref") if isinstance(rec, dict) else None
                    if cls and isinstance(ref, str):
                        self.model_refs.append((cls, ref))

    def size_batches(self, world) -> None:
        """Rows of every class-2/3 recorded call since the last call, read from its recorded output (evaluator
        side, outside the measured CPU): the exact in-world batch sizes behind V.4 and SP-13.1."""
        from ..substrate import artifacts
        refs, self.model_refs = self.model_refs, []
        for cls, ref in refs:
            try:
                arrs = artifacts.npz_arrays(artifacts.get(world, ref))
            except Exception:                  # noqa: BLE001 - an unreadable record is not counted
                continue
            first = arrs[sorted(arrs)[0]] if arrs else None
            if first is not None and getattr(first, "ndim", 0) >= 1:
                self.batch_sizes[cls][int(first.shape[0])] += 1


#: recorded-call kinds (`infer.pop.<model>.<group>`) whose rows are class-2 / class-3 decisions (class 2: the
#: resource choice and, where the population records it, the activity choice of the adaptive policies)
BATCH_KINDS = {"pop.adaptive": "class2", "pop.adaptive_act": "class2", "pop.strategic": "class3"}


# ====================================================================== the world
def make_world(kind: str, profile: str, seed: int, store=None):
    import crucible
    crucible.load_all_modules()
    from ..core.kernel import World
    if kind == "civ":
        from .. import civ
        return civ.generate(CIV_PROFILE, seed, population=profile, store=store)
    if kind == "population":
        from ..population.worlds import population_world
        return World.create(population_world(profile, seed), store=store)
    raise ValueError(f"world kind must be one of {WORLD_KINDS}")


def model_parameters(world) -> dict:
    """Parameter counts of the models a world has loaded (counted on the objects)."""
    out = {}
    rt = world.runtime
    for key, label in (("pop_adaptive.nets", "class2_adaptive"), ("pop_strategic.nets", "class3_strategic")):
        for gid, obj in sorted((rt.get(key) or {}).items(), key=lambda kv: str(kv[0])):
            try:
                n = int(obj.n_params())
            except Exception:              # noqa: BLE001 - an object without a count is not reported
                try:
                    n = int(sum(p.numel() for p in obj.net.parameters()))
                except Exception:          # noqa: BLE001
                    continue
            out[f"{label}:{gid}"] = n
    try:
        from ..semantic.space import compute as SC
        for name in ("TEXT", "LM", "IMAGE", "AUDIO", "CODE", "VIDEO"):
            enc = getattr(SC, name, None)
            m = None
            for attr in ("_model", "model", "_net", "net"):
                m = getattr(enc, attr, None) if enc is not None else None
                if m is not None and hasattr(m, "parameters"):
                    break
                m = None
            if m is not None:
                out[f"space:{name.lower()}"] = int(sum(p.numel() for p in m.parameters()))
    except Exception:                      # noqa: BLE001 - the space is optional in a population-only world
        pass
    try:
        from ..semantic import service as SS
        for svc in list(getattr(SS, "_SERVICES", {}).values()):
            info = svc.info()
            out[f"semantic:{info.get('model_id')}"] = int(info.get("params") or 0)
    except Exception:                      # noqa: BLE001
        pass
    return out


def inference_tokens(ec: EventCounter) -> dict:
    out = {"recorded_inference_calls": int(sum(ec.infer.values())), "by_kind": dict(ec.infer)}
    try:
        from ..semantic import space
        st = space.stats()
        out["space_lm_tokens"] = int(st.get("lm", {}).get("tokens", 0))
        out["space_lm_forward_passes"] = int(st.get("lm", {}).get("forward_passes", 0))
        out["space_items_computed"] = {k: int(v) for k, v in (st.get("computed") or {}).items()}
        out["space_world_hits"] = int(st.get("world_hits", 0))
        out["space_cache"] = dict(st.get("cache") or {})
    except Exception:                      # noqa: BLE001
        pass
    try:
        from ..semantic import service as SS
        gen = 0
        for svc in list(getattr(SS, "_SERVICES", {}).values()):
            gen += int((svc.info().get("stats") or {}).get("generated_tokens", 0))
        out["semantic_generated_tokens"] = gen
    except Exception:                      # noqa: BLE001
        pass
    out["total_tokens"] = int(out.get("space_lm_tokens", 0) + out.get("semantic_generated_tokens", 0))
    return out


def class_activity(world, n_sessions0, ec: EventCounter) -> dict:
    import numpy as np
    from ..substrate import columns
    cls = columns.col(world, "pop", "class")
    grew = columns.col(world, "pop", "n_sessions").astype(np.int64) - n_sessions0
    out = {}
    for c in (1, 2, 3, 4):
        m = cls == c
        out[str(c)] = {"rows": int(m.sum()), "active": int(((grew > 0) & m).sum()), "sessions": int(grew[m].sum())}
    pol = Counter()
    for mid, n in ec.decisions_by.items():
        if "adaptive" in mid:
            pol["2"] += n
        elif "strategic" in mid:
            pol["3"] += n
        elif "semantic" in mid:
            pol["4"] += n
    # class 3 acts through its own decision process (pop_strategic.decide), not through sessions: one
    # pop_strategic.exp event per member decision
    pol["3"] += int(ec.types.get("pop_strategic.exp", 0))
    for c, n in pol.items():
        out[c]["policy_decisions"] = int(n)
    for c in out:
        out[c]["acting"] = out[c]["active"] > 0 or out[c].get("policy_decisions", 0) > 0
    return out


def batching(w, ec: EventCounter) -> dict:
    """Batched decisions per learning class (SP-13.1, V.4): recorded model calls and the decisions they made.
    The shared batching service (crucible.substrate.inference_batch) accounts per class directly; callers that
    still record their own batches are counted from their records: class 2 from the adaptive policies' counters,
    class 3 from its recorded `pop.strategic.*` calls and one `pop_strategic.exp` event per member decision,
    class 4 from the shared semantic service's recorded batches (`infer.semantic.choose|generate` outcomes, rows =
    replies; in the civilisation world only the class-4 runtime calls that service)."""
    from ..substrate import inference_batch as IB
    svc = IB.report(IB.stats(w))
    by = svc.get("by_class", {})
    ad = dict(w.runtime.get("pop_adaptive.stats") or {})
    strat_calls = sum(n for k, n in ec.infer.items() if k.startswith("infer.pop.strategic"))
    strat_dec = int(ec.types.get("pop_strategic.exp", 0)) if strat_calls else 0
    legacy = {"class2": (int(ad.get("calls", 0)), int(ad.get("decisions", 0)), int(ad.get("max_batch", 0))),
              "class3": (int(strat_calls), strat_dec, 0),
              "class4": (int(ec.semantic_calls), int(ec.semantic_rows), 0)}
    out = {}
    for c, (calls, dec, mx) in legacy.items():
        s = by.get(c[-1], {})
        calls += int(s.get("calls", 0))
        dec += int(s.get("rows", 0))
        out[c] = {"calls": calls, "decisions": dec, "max_batch": max(mx, int(s.get("max_rows", 0))),
                  "mean_batch": round(dec / max(1, calls), 3), "via_service_calls": int(s.get("calls", 0)),
                  "merged_calls": int(s.get("merged_calls", 0))}
        hist = getattr(ec, "batch_sizes", {}).get(c)
        if hist:                                # exact rows per recorded call (from the recorded outputs)
            out[c]["sizes"] = {str(k): int(v) for k, v in sorted(hist.items())}
    out["service"] = svc
    return out


def stored_event_bytes(store, world_id: str, seq_from: int, seq_to: int) -> int:
    """Bytes of the events seq_from < seq <= seq_to as the store holds them (every column of the row; payloads above
    the store's threshold are zlib-compressed there), read with one SQL aggregate."""
    c = store._conn()
    r = c.execute("SELECT COALESCE(SUM(LENGTH(type) + LENGTH(COALESCE(actor, '')) + LENGTH(COALESCE(cmd, '')) + "
                  "LENGTH(payload) + LENGTH(COALESCE(vis, '')) + LENGTH(COALESCE(hash, '')) + "
                  "LENGTH(COALESCE(via, '')) + 16), 0) FROM events WHERE world_id = ? AND seq > ? AND seq <= ?",
                  (world_id, int(seq_from), int(seq_to))).fetchone()
    return int(r[0] or 0)


def _settle(store) -> None:
    """Fold the write-ahead log into the database file (checkpoint, WAL truncated). No VACUUM: VACUUM writes a second
    copy of the database (into the WAL in WAL mode), so measuring a 30-day standard store of 758 MB needed 1.85 GB of
    disk and the run was stopped by its disk grant at the very end (2026-11-26); `_live_bytes` gives the resident
    size without it."""
    store._conn().execute("PRAGMA wal_checkpoint(TRUNCATE)")


def _live_bytes(store) -> int:
    """Bytes of the database pages in use after a checkpoint: (page_count - freelist_count) x page_size, i.e. the
    file without the pages that retention freed (what VACUUM would return to the file system; VACUUM can be smaller
    still by repacking partly filled pages, so this resident size is an upper bound)."""
    c = store._conn()
    pages = int(c.execute("PRAGMA page_count").fetchone()[0])
    free = int(c.execute("PRAGMA freelist_count").fetchone()[0])
    size = int(c.execute("PRAGMA page_size").fetchone()[0])
    return max(0, pages - free) * size


def _store_bytes(path: str) -> int:
    return sum(os.path.getsize(p) for p in (path, path + "-wal", path + "-shm") if os.path.exists(p))


def run(kind: str, profile: str, seed: int, days: float, store_dir: str, *, compact_every: float = 1.0,
        snapshot: bool = True, keep_store: bool = False, partial_out: str | None = None) -> dict:
    """One benchmark run (see the module docstring). Returns the report. With `partial_out`, the measurements so far
    are written there after every simulated day (a run stopped by its time grant still reports what it measured)."""
    from ..storage.sqlite import SQLiteStore
    from ..substrate import columns, retention
    os.makedirs(store_dir, exist_ok=True)
    db = os.path.join(store_dir, f"{kind}-{profile}-{seed}.sqlite")
    for p in (db, db + "-wal", db + "-shm"):
        if os.path.exists(p):
            os.unlink(p)
    store = SQLiteStore(db)
    rep = {"world": kind, "profile": profile, "seed": int(seed), "days": float(days), "pid": os.getpid(),
           "started": time.time()}
    t0, c0 = time.time(), cpu_s()
    w = make_world(kind, profile, seed, store=store)
    w.flush()
    gen = {"wall_s": round(time.time() - t0, 3), "cpu_s": round(cpu_s() - c0, 3), "events": int(w.seq),
           "rss_mb": round(vm_hwm_mb(), 1), "modules": len(w.modules)}
    rep["world_generation_cost"] = gen
    n_ind = int(columns.rows(w, "pop"))
    rep["individuals"] = n_ind
    n_sessions0 = columns.col(w, "pop", "n_sessions").astype("int64").copy()
    ec = EventCounter(count_bytes=False)
    w.listeners.append(ec)
    _settle(store)
    bytes0 = _live_bytes(store)
    unpacked0 = retention.stats(w)["blob_bytes_unpacked"]
    seq0, tx0 = w.seq, w.stats["commands"] + w.stats["procs"]
    series = []
    raw_blob = 0
    compact_cpu = 0.0
    t_run, c_run = time.time(), cpu_s()
    end = w.now + int(days * DAY)
    step = int(min(DAY, days * DAY))
    next_compact = w.now + int(compact_every * DAY)
    while w.now < end:
        until = min(end, w.now + step)
        td, cd, sq, ss = time.time(), cpu_s(), w.seq, ec.sessions
        w.advance_time(until)
        w.flush()
        row = {"t": int(w.now), "wall_s": round(time.time() - td, 3), "cpu_s": round(cpu_s() - cd, 3),
               "events": int(w.seq - sq), "sessions": int(ec.sessions - ss)}
        tm = cpu_s()
        row["event_bytes"] = stored_event_bytes(store, w.world_id, sq, w.seq)
        ec.event_bytes += row["event_bytes"]
        ec.size_batches(w)
        compact_cpu += cpu_s() - tm                # measurement, not simulation
        if w.now >= next_compact or w.now >= end:
            tc, cc = time.time(), cpu_s()
            unpacked = retention.stats(w)["blob_bytes_unpacked"]
            row["blob_bytes"] = int(max(0, unpacked - unpacked0))
            raw_blob += row["blob_bytes"]
            retention.compact(w)
            unpacked0 = retention.stats(w)["blob_bytes_unpacked"]
            compact_cpu += cpu_s() - cc
            row["compact_s"] = round(time.time() - tc, 3)
            next_compact = w.now + int(compact_every * DAY)
        row["store_bytes"] = _store_bytes(db)           # the files on disk (database, WAL, shared memory)
        series.append(row)
        if partial_out:
            done = (w.now - (end - int(days * DAY))) / DAY
            part = {**rep, "partial": True, "sim_days": round(done, 3), "series": series,
                    "cpu_s": round(cpu_s() - c_run - compact_cpu, 2), "wall_s": round(time.time() - t_run, 2),
                    "sim_days_per_cpu_hour": round(done / max((cpu_s() - c_run - compact_cpu) / 3600, 1e-9), 3),
                    "sim_days_per_wall_hour": round(done / max((time.time() - t_run) / 3600, 1e-9), 3),
                    "peak_ram_mb": round(vm_hwm_mb(), 1), "sessions": int(ec.sessions), "decisions": int(ec.decisions)}
            with open(partial_out + ".tmp", "w") as fh:
                json.dump(part, fh, default=str)
            os.replace(partial_out + ".tmp", partial_out)
    wall = time.time() - t_run
    cpu = cpu_s() - c_run - compact_cpu
    if len(series) >= 3:
        # what the first simulated day costs beyond an ordinary day: deployments and warm-up that World.create leaves
        # to the world's first processes, model loading and lazy imports (counted as generation cost: stricter)
        rest = sorted(r["cpu_s"] for r in series[1:])
        gen["first_day_excess_cpu_s"] = round(max(0.0, series[0]["cpu_s"] - rest[len(rest) // 2]), 3)
    gen["inclusive_cpu_s"] = round(gen["cpu_s"] + gen.get("first_day_excess_cpu_s", 0.0), 3)
    w.listeners.remove(ec)
    sim_days = (w.now - (end - int(days * DAY))) / DAY
    tc, cc = time.time(), cpu_s()
    _settle(store)                             # the resident size: WAL folded in, pages freed by retention excluded
    compact_cpu += cpu_s() - cc
    st = retention.stats(w)
    resident = _live_bytes(store) - bytes0
    raw_written = raw_blob + ec.event_bytes
    ind_days = max(1.0, n_ind * sim_days)
    rep.update({
        "wall_s": round(wall, 2), "cpu_s": round(cpu, 2), "compact_cpu_s": round(compact_cpu, 2),
        "sim_days": round(sim_days, 3),
        "transitions_per_s": round((w.seq - seq0) / max(wall, 1e-9), 2),
        "transactions_per_s": round((w.stats["commands"] + w.stats["procs"] - tx0) / max(wall, 1e-9), 2),
        "events": int(w.seq - seq0), "decisions": int(ec.decisions),
        "decisions_per_s": round(ec.decisions / max(wall, 1e-9), 2),
        "decisions_by_model": dict(ec.decisions_by), "sessions": int(ec.sessions), "activations": int(ec.activations),
        "sim_days_per_wall_hour": round(sim_days / max(wall / 3600, 1e-9), 3),
        "sim_days_per_cpu_hour": round(sim_days / max(cpu / 3600, 1e-9), 3),
        "cpu_utilisation": round(cpu / max(wall, 1e-9), 3), "peak_ram_mb": round(vm_hwm_mb(), 1),
        "children_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1),
        "event_storage_growth": {"raw_bytes": int(raw_written), "raw_event_bytes": int(ec.event_bytes),
                                 "raw_blob_bytes": int(raw_blob), "resident_bytes": int(resident),
                                 "raw_bytes_per_day": round(raw_written / max(sim_days, 1e-9)),
                                 "resident_bytes_per_day": round(resident / max(sim_days, 1e-9)),
                                 "raw_bytes_per_individual_day": round(raw_written / ind_days, 1),
                                 "bytes_per_individual_day": round(resident / ind_days, 1), "retention": st},
        "class_activity": class_activity(w, n_sessions0, ec),
        "events_by_type": dict(ec.types.most_common(40)), "batches_by_kind": dict(ec.batches.most_common(40)),
        "public_events": int(ec.public), "public_event_kinds": len(ec.public_kinds), "actors": len(ec.actors),
        "series": series,
    })
    vram, used_cuda = peak_vram_mb()
    rep["peak_vram_mb"], rep["cuda_used"] = round(vram, 1), used_cuda
    rep["model_parameter_counts"] = model_parameters(w)
    rep["inference_token_counts"] = inference_tokens(ec)
    rep["batching"] = batching(w, ec)
    rep["principals"] = int(len(w.state.tables.get("principals", {})))
    if snapshot:
        ts = time.time()
        blob = w.snapshot_blob()
        snap_s = time.time() - ts
        digest = w.digest()
        tf = time.time()
        from ..core.kernel import World
        fk = World.from_blob(blob, world_id=f"{w.world_id}.bench-fork", fork=True)
        fork_s = time.time() - tf
        rep["snapshot_size"] = {"bytes": len(blob), "seconds": round(snap_s, 3), "fork_seconds": round(fork_s, 3),
                                "fork_exact": fk.digest() == digest, "bytes_per_individual": round(len(blob) / max(1, n_ind), 1)}
        fk.close()
        del fk, blob
    rep["experience"] = experience(rep)
    w.close()
    store.close()
    if not keep_store:
        for p in (db, db + "-wal", db + "-shm"):
            if os.path.exists(p):
                os.unlink(p)
    rep["ended"] = time.time()
    return rep


def experience(rep: dict) -> dict:
    """Learner experience per unit of compute (Part 13 preamble, SP-13.8b): events a learner can observe (public),
    the diversity of what is observable (distinct public event kinds, distinct acting principals) and the
    individual decisions behind them, per CPU-second of the world."""
    cpu = max(float(rep.get("cpu_s") or 0), 1e-9)
    kinds = max(1, int(rep.get("public_event_kinds") or 0))
    return {"public_events_per_cpu_s": round(rep.get("public_events", 0) / cpu, 3),
            "decisions_per_cpu_s": round(rep.get("decisions", 0) / cpu, 3),
            "distinct_public_kinds": kinds, "acting_principals": int(rep.get("actors") or 0),
            "diverse_experience_per_cpu_s": round(rep.get("public_events", 0) / cpu * min(1.0, kinds / 20.0), 3)}


# ====================================================================== Crucible 4.0 equivalent workload
V4_PROBE = r'''
import json, os, resource, sys, time
t0 = time.time(); c0 = sum(os.times()[:4])
import crucible
crucible.load_all_modules()
from crucible.core.kernel import World
from crucible.storage.sqlite import SQLiteStore
from crucible.worldgen import generator as G
n, days, seed, db = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
store = SQLiteStore(db)
cpu = lambda: sum(os.times()[:4])
tg, cg = time.time(), cpu()
spec = G.generate({"family": "bazaar", "overrides": {"n_consumers": n}}, seed)
spec.params["cost_model"] = "deterministic"
w = World.create(spec, store=store)
w.flush()
gen = {"wall_s": time.time() - tg, "cpu_s": cpu() - cg, "events": w.seq}
size = lambda: sum(os.path.getsize(p) for p in (db, db + "-wal") if os.path.exists(p))
b0, s0, pub = size(), w.seq, [0]
w.listeners.append(lambda ev: pub.__setitem__(0, pub[0] + (ev.vis == "public")))
tr, cr = time.time(), cpu()
end = w.now + int(days * 86400)
while w.now < end:
    w.advance_time(min(end, w.now + 86400)); w.flush()
wall, c = time.time() - tr, cpu() - cr
ts = time.time(); blob = w.snapshot_blob(); snap = time.time() - ts
out = {"consumers": n, "days": days, "seed": seed, "generation": gen, "wall_s": wall, "cpu_s": c,
       "events": w.seq - s0, "public_events": pub[0], "store_bytes": size() - b0, "snapshot_bytes": len(blob),
       "snapshot_s": snap, "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
       "sim_days_per_cpu_hour": days / max(c / 3600, 1e-9), "sim_days_per_wall_hour": days / max(wall / 3600, 1e-9),
       "transitions_per_s": (w.seq - s0) / max(wall, 1e-9)}
print("@@JSON@@" + json.dumps(out))
'''


def v4_equivalent(consumers: int, days: float, seed: int, work: str, src_root: str | None = None) -> dict:
    """The 4.0 workload equivalent to a v5 profile: the `bazaar` family (4.0's marketplace with aggregate
    consumers) with as many consumers as the profile has individuals, the same simulated days, run from the 4.0
    source tree (c07098b, extracted by git archive) with a store."""
    from ..acceptance.v5 import baseline
    root = src_root or baseline.v4_tree(dest=os.path.join(work, "v4tree"))
    os.makedirs(work, exist_ok=True)
    db = os.path.join(work, f"v4-bazaar-{consumers}-{seed}.sqlite")
    for p in (db, db + "-wal", db + "-shm"):
        if os.path.exists(p):
            os.unlink(p)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONPATH=os.path.join(root, "src"), PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-P", "-c", V4_PROBE, str(consumers), str(days), str(seed), db], cwd=work,
                       env=env, capture_output=True, text=True)
    for f in (db, db + "-wal", db + "-shm"):
        if os.path.exists(f):
            os.unlink(f)
    if "@@JSON@@" not in p.stdout:
        raise RuntimeError(f"4.0 probe failed: {p.stderr[-2000:]}")
    out = json.loads(p.stdout.rsplit("@@JSON@@", 1)[1])
    out["source"] = {"tree": root, "commit": baseline.V4_COMMIT, "family": "bazaar"}
    return out


def comparison_rows(v5: list[dict], v4: list[dict]) -> list[dict]:
    """Equivalent-workload rows: per profile, v5 world vs 4.0 bazaar with the same number of consumers and days."""
    rows = []
    for r5 in v5:
        n = r5["class_activity"]["1"]["rows"]
        m = [r for r in v4 if int(r["consumers"]) == n and float(r["days"]) == float(r5["days"])]
        if not m:
            continue
        r4 = m[0]
        ind_days = max(1.0, n * r5["sim_days"])
        rows.append({
            "profile": r5["profile"], "world": r5["world"], "consumers": n, "days": r5["days"],
            "v5_sim_days_per_cpu_hour": r5["sim_days_per_cpu_hour"], "v4_sim_days_per_cpu_hour": round(r4["sim_days_per_cpu_hour"], 3),
            "v5_cpu_s_per_individual_day": round(r5["cpu_s"] / ind_days, 6),
            "v4_cpu_s_per_consumer_day": round(r4["cpu_s"] / max(1.0, n * r4["days"]), 6),
            "v5_peak_ram_mb": r5["peak_ram_mb"], "v4_peak_ram_mb": round(r4["peak_ram_mb"], 1),
            "v5_store_bytes_per_individual_day": r5["event_storage_growth"]["bytes_per_individual_day"],
            "v4_store_bytes_per_consumer_day": round(r4["store_bytes"] / max(1.0, n * r4["days"]), 1),
            "v5_generation_cpu_s": r5["world_generation_cost"]["cpu_s"], "v4_generation_cpu_s": round(r4["generation"]["cpu_s"], 3),
            "v5_decisions_per_cpu_s": r5["experience"]["decisions_per_cpu_s"],
            "v5_public_events_per_cpu_s": r5["experience"]["public_events_per_cpu_s"],
            "v4_public_events_per_cpu_s": round(r4["public_events"] / max(r4["cpu_s"], 1e-9), 3),
            "v5_transitions_per_s": r5["transitions_per_s"], "v4_transitions_per_s": round(r4["transitions_per_s"], 2),
            "fidelity_note": "4.0 bazaar consumers are aggregate demand parameters; v5 individuals have persistent "
                             "identity, attention, money, habits and learning, so equal consumer counts are equal "
                             "population size, not equal fidelity"})
    return rows


# ====================================================================== suite of governed runs
def _run_cmd(kind, profile, seed, days, store_dir, out) -> list[str]:
    return [sys.executable, "-m", "crucible.scale.bench5", "run", "--world", kind, "--profile", profile, "--seed",
            str(seed), "--days", str(days), "--store-dir", store_dir, "--out", out]


def time_grant_s(profile: str, days: float, headroom: float = 3.0) -> float | None:
    """Wall-time grant of a benchmark run: `headroom` x the time the profile's throughput target allows for `days`
    (a run that cannot finish in that time has missed its target anyway; it is stopped, and reports what it
    measured), at least one hour."""
    try:
        tg = {t["profile"]: t["value"] for t in targets()["targets"] if t["quantity"] == "sim_days_per_cpu_hour"}
    except (OSError, ValueError, KeyError):
        return None
    if profile not in tg:
        return None
    return max(3600.0, headroom * days / float(tg[profile]) * 3600.0)


def governed(cmd: list[str], profile: str, label: str, store_dir: str, *, cpu: int = 1, gpu_mb: int | None = None,
             timeout_s: float | None = None, cold_cache: bool = False) -> dict:
    """Run `cmd` as a governed job with declared grants in every dimension; returns the usage record. With
    `cold_cache` the job gets a fresh machine cache of its own (`$CRUCIBLE_CACHE` inside `store_dir`): nothing an
    earlier run computed on this machine (representations, metering anchors, weights hashes) is reused, so a
    benchmark run pays the full cost of new content."""
    from . import slots
    ram, disk, gpu = GRANTS[profile]
    gpu_mb = gpu if gpu_mb is None else gpu_mb
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ)
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("OMP_NUM_THREADS", "1")
    if cold_cache:
        env["CRUCIBLE_CACHE"] = os.path.join(store_dir, "machine-cache")
        os.makedirs(env["CRUCIBLE_CACHE"], exist_ok=True)
    return slots.run_job(cmd, cpu=cpu, gpu_mb=gpu_mb, label=label, timeout_s=timeout_s, ram_mb=ram, disk_mb=disk,
                         disk_dirs=[store_dir],
                         inference=2, worlds=4, gpu_s=(GPU_S if gpu_mb else None), env=env, sample_s=2.0, poll_s=2.0)


def cache_dir() -> str | None:
    """Where runs of a clean committed tree are kept, keyed by commit: separate gate test files of one commit reuse
    one measured run instead of repeating hours of simulation. None (no reuse) for a tree with uncommitted changes."""
    from ..acceptance.v5 import evidence
    try:
        st = evidence.tree_state()
    except Exception:                      # noqa: BLE001 - not a git checkout: no reuse
        return None
    if not st["clean"]:
        return None
    root = os.environ.get("CRUCIBLE_BENCH5_CACHE") or os.path.join("/tmp", "crucible-bench5-cache")
    d = os.path.join(root, st["commit"])
    os.makedirs(d, exist_ok=True)
    return d


def suite(kinds=("civ",), profiles=PROFILES, seeds=(0, 1, 2), days: float = 30, work: str | None = None,
          parallel: int = 3) -> dict:
    """Every (world kind, profile, seed) as a governed sub-job, `parallel` at a time; returns reports and the
    usage records of the jobs. A run already measured for this commit (cache_dir) is reused, with its own usage
    record."""
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    work = work or tempfile.mkdtemp(prefix="bench5-")
    cache = cache_dir()
    jobs = [(k, p, s) for k in kinds for p in profiles for s in seeds]
    out = {"runs": [], "failed": [], "usage": [], "work": work, "cache": cache, "reused": 0}

    def one(j):
        k, p, s = j
        name = f"run-{k}-{p}-{s}-{days:g}d"
        if cache and os.path.exists(os.path.join(cache, name + ".json")):
            try:
                got = json.load(open(os.path.join(cache, name + ".json")))
                got["run"]["reused_from_cache"] = True
                return j, got["run"], got["usage"]
            except (OSError, ValueError, KeyError):
                pass
        sd = os.path.join(work, f"store-{k}-{p}-{s}")
        f = os.path.join(work, name + ".json")
        rec = governed(_run_cmd(k, p, s, days, sd, f), p, f"M: bench5 {k} {p} seed {s} {days:g}d", sd,
                       timeout_s=time_grant_s(p, days), cold_cache=True)
        shutil.rmtree(sd, ignore_errors=True)
        try:
            r = json.load(open(f))
        except (OSError, ValueError):
            r = None
        if r is None:
            try:
                rec["partial_report"] = json.load(open(f + ".partial"))
            except (OSError, ValueError):
                pass
        if r is not None and rec.get("rc") == 0 and cache:
            with open(os.path.join(cache, name + ".json.tmp"), "w") as fh:
                json.dump({"run": r, "usage": rec}, fh, default=str)
            os.replace(os.path.join(cache, name + ".json.tmp"), os.path.join(cache, name + ".json"))
        return j, r, rec
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        for (k, p, s), r, rec in ex.map(one, jobs):
            out["usage"].append(rec)
            out["reused"] += int(bool(r and r.get("reused_from_cache")))
            if r is None or rec.get("rc") != 0:
                pr = rec.get("partial_report") or {}
                out["failed"].append({"world": k, "profile": p, "seed": s, "rc": rec.get("rc"),
                                      "partial_sim_days": pr.get("sim_days"),
                                      "partial_sim_days_per_cpu_hour": pr.get("sim_days_per_cpu_hour"),
                                      "killed_by": rec.get("killed_by")})
            else:
                r["usage"] = {x: rec.get(x) for x in ("peak_rss_mb", "peak_cpu", "mean_cpu", "cpu_s", "wall_s",
                                                      "peak_gpu_mb", "gpu_s", "disk_peak_mb", "violations",
                                                      "declared", "cpus")}
                out["runs"].append(r)
    return out


def aggregate(runs: list[dict]) -> dict:
    """Per (world, profile): mean and spread over seeds of the headline quantities."""
    import statistics as stt
    groups: dict = {}
    for r in runs:
        groups.setdefault((r["world"], r["profile"]), []).append(r)
    out = {}
    for (k, p), rs in sorted(groups.items()):
        def col(f):
            return [float(f(r)) for r in rs]
        agg = {"seeds": sorted(r["seed"] for r in rs), "days": min(r["sim_days"] for r in rs)}
        for name, f in (("sim_days_per_cpu_hour", lambda r: r["sim_days_per_cpu_hour"]),
                        ("sim_days_per_wall_hour", lambda r: r["sim_days_per_wall_hour"]),
                        ("transitions_per_s", lambda r: r["transitions_per_s"]),
                        ("decisions_per_s", lambda r: r["decisions_per_s"]),
                        ("cpu_utilisation", lambda r: r["cpu_utilisation"]),
                        ("peak_ram_mb", lambda r: r["peak_ram_mb"]),
                        ("bytes_per_individual_day", lambda r: r["event_storage_growth"]["bytes_per_individual_day"]),
                        ("raw_bytes_per_individual_day",
                         lambda r: r["event_storage_growth"]["raw_bytes_per_individual_day"]),
                        ("snapshot_bytes", lambda r: r["snapshot_size"]["bytes"]),
                        ("snapshot_s", lambda r: r["snapshot_size"]["seconds"]),
                        ("generation_cpu_s", lambda r: r["world_generation_cost"].get(
                            "inclusive_cpu_s", r["world_generation_cost"]["cpu_s"]))):
            v = col(f)
            agg[name] = {"mean": round(stt.mean(v), 4), "min": round(min(v), 4), "max": round(max(v), 4)}
        agg["classes_acting"] = min(sum(1 for c in r["class_activity"].values() if c["acting"]) for r in rs)
        agg["fork_exact"] = all(r["snapshot_size"]["fork_exact"] for r in rs)
        out[f"{k}/{p}"] = agg
    return out


def targets(path: str | None = None) -> dict:
    from ..acceptance.v5.manifest import docs_dir
    return json.load(open(path or os.path.join(docs_dir(), "performance", "targets.json")))


def check_targets(agg: dict, tg: dict, world: str = "civ") -> list[dict]:
    """Each throughput / Gate V target against the measured aggregate of the target world."""
    res = []
    for t in tg["targets"]:
        q = t["quantity"]
        for prof in str(t["profile"]).split(","):
            a = agg.get(f"{world}/{prof}")
            val = None
            if a is None:
                if q in ("sim_days_per_cpu_hour", "world_generation_cpu_s", "bytes_per_individual_day", "snapshot"):
                    res.append({"id": t["id"], "profile": prof, "measured": None, "met": False,
                                "why": f"no {world} run of profile {prof}"})
                continue
            if q == "sim_days_per_cpu_hour":
                val = a["sim_days_per_cpu_hour"]["min"]
                met = val >= t["value"]
            elif q == "world_generation_cpu_s":
                val = a["generation_cpu_s"]["max"]
                met = val <= t["value"]
            elif q == "bytes_per_individual_day":
                val = a["bytes_per_individual_day"]["max"]
                met = val <= t["value"]
            elif q == "snapshot":
                val = {"bytes": a["snapshot_bytes"]["max"], "seconds": a["snapshot_s"]["max"]}
                met = val["bytes"] <= t["value"]["bytes"] and val["seconds"] <= t["value"]["seconds"]
            else:
                continue
            res.append({"id": t["id"], "group": t["group"], "profile": prof, "quantity": q, "target": t["value"],
                        "measured": val, "met": bool(met)})
    return res


# ====================================================================== CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--world", choices=WORLD_KINDS, default="civ")
    r.add_argument("--profile", choices=PROFILES + ("internet",), default="development")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--days", type=float, default=1.0)
    r.add_argument("--store-dir", required=True)
    r.add_argument("--compact-every", type=float, default=1.0)
    r.add_argument("--out", required=True)
    v = sub.add_parser("v4")
    v.add_argument("--consumers", type=int, default=1000)
    v.add_argument("--days", type=float, default=1.0)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--work", required=True)
    v.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        rep = run(a.world, a.profile, a.seed, a.days, a.store_dir, compact_every=a.compact_every,
                  partial_out=a.out + ".partial")
    else:
        rep = v4_equivalent(a.consumers, a.days, a.seed, a.work)
    with open(a.out + ".tmp", "w") as fh:
        json.dump(rep, fh, default=str)
    os.replace(a.out + ".tmp", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
