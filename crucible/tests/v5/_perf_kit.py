"""Shared measurements of the performance gate tests (Gate V, SP-13.x, SP-0.10, SP-3.3b), computed once per process
and, for a clean committed tree, once per commit (crucible.scale.bench5.cache_dir): the benchmark suite of governed
runs, the 4.0 equivalent workloads, the backend matrix, the estimated-mode validation, the event-driven and caching
workloads and the 10M capacity run. Every expensive job runs through the resource governor with declared grants.

Sizes follow the gate requirements (3 profiles x 3 seeds x 30 simulated days); the environment may make a run
larger, and the tests assert the required minimums on what was measured, so it can never make one smaller."""
from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
import tempfile
import time

from crucible.scale import approx, backends, bench5, gatev, slots

DAYS = max(30.0, float(os.environ.get("CRUCIBLE_GATE_V_DAYS", 30)))
SEEDS = tuple(range(max(3, int(os.environ.get("CRUCIBLE_GATE_V_SEEDS", 3)))))
KINDS = ("civ", "population")          # civ: the target world (D-17); population: the second row
WORK = os.environ.get("CRUCIBLE_GATE_V_WORK") or tempfile.mkdtemp(prefix="gate-v-")
T_START = time.time()


def parallel() -> int:
    return max(1, min(len(os.sched_getaffinity(0)), int(os.environ.get("CRUCIBLE_GATE_V_PARALLEL", 8))))


@functools.lru_cache(maxsize=1)
def suite() -> dict:
    out = bench5.suite(kinds=KINDS, profiles=bench5.PROFILES, seeds=SEEDS, days=DAYS,
                       work=os.path.join(WORK, "suite"), parallel=parallel())
    out["aggregate"] = bench5.aggregate(out["runs"])
    return out


def _governed_json(cmd: list[str], out: str, label: str, profile: str = "standard", cpu: int = 1,
                   gpu_mb: int | None = None) -> tuple[dict | None, dict]:
    store = os.path.join(WORK, "stores", os.path.basename(out))
    os.makedirs(store, exist_ok=True)
    rec = bench5.governed(cmd, profile, label, store, cpu=cpu, gpu_mb=gpu_mb)
    try:
        return json.load(open(out)), rec
    except (OSError, ValueError):
        return None, rec


def _cached(name: str, compute):
    d = bench5.cache_dir()
    p = os.path.join(d, name + ".json") if d else None
    if p and os.path.exists(p):
        try:
            got = json.load(open(p))
            got["reused_from_cache"] = True
            return got
        except (OSError, ValueError):
            pass
    got = compute()
    if p and got and got.get("ok"):
        with open(p + ".tmp", "w") as fh:
            json.dump(got, fh, default=str)
        os.replace(p + ".tmp", p)
    return got


@functools.lru_cache(maxsize=1)
def v4_rows() -> dict:
    def compute():
        rows, recs = [], []
        for prof, n in (("development", 1000), ("standard", 10000), ("large", 100000)):
            out = os.path.join(WORK, f"v4-{n}.json")
            r, rec = _governed_json([sys.executable, "-m", "crucible.scale.bench5", "v4", "--consumers", str(n),
                                     "--days", str(DAYS), "--seed", "0", "--work", os.path.join(WORK, "v4"),
                                     "--out", out], out, f"M: 4.0 bazaar {n} consumers {DAYS:g}d", prof, gpu_mb=0)
            recs.append(rec)
            if r is not None:
                rows.append(r)
        return {"ok": len(rows) == 3, "rows": rows, "usage": recs}
    return _cached(f"v4-bazaar-{DAYS:g}d", compute)


@functools.lru_cache(maxsize=1)
def backend_matrix() -> dict:
    def compute():
        out = os.path.join(WORK, "backends.json")
        code = ("import json, sys\nfrom crucible.scale import backends\n"
                "b = backends.bench_adaptive(batches=(1, 8, 32, 64, 256, 1024), reps=5)\n"
                "json.dump({'adaptive': b, 'choice': backends.choose(b), 'strategic': backends.bench_strategic(),"
                " 'encoder': backends.bench_encoder(), 'service': backends.bench_service()}, open(sys.argv[1], 'w'),"
                " default=str)\n")
        r, rec = _governed_json([sys.executable, "-c", code, out], out, "M: backend matrix", "standard", gpu_mb=3500)
        return {"ok": r is not None, "result": r, "usage": rec}
    return _cached("backends", compute)


@functools.lru_cache(maxsize=1)
def approximation() -> dict:
    def compute():
        out = os.path.join(WORK, "approx.json")
        code = ("import json, sys\nfrom crucible.scale import approx\n"
                "json.dump(approx.validate(n_fit=400, n_target=1600, days=4), open(sys.argv[1], 'w'), default=str)\n")
        r, rec = _governed_json([sys.executable, "-c", code, out], out, "M: estimated mode validation", "standard")
        return {"ok": r is not None, "result": r, "usage": rec}
    return _cached("approx", compute)


@functools.lru_cache(maxsize=1)
def event_driven() -> dict:
    def compute():
        out = os.path.join(WORK, "event_driven.json")
        r, rec = _governed_json([sys.executable, "-m", "crucible.scale.gatev", "event_driven", "--world", "civ",
                                 "--profile", "standard", "--hours", "24", "--out", out], out,
                                "M: event-driven vs dense", "standard")
        return {"ok": r is not None, "result": r, "usage": rec}
    return _cached("event_driven", compute)


@functools.lru_cache(maxsize=1)
def cache_workload() -> dict:
    # 96 simulated hours per world: content is consumed over days after it is published (a 30-day development civ
    # world looks each representation up about 36 times), so a 12-hour world measures little more than first
    # lookups (measured 2026-11-26: 0.67 over three 12-hour worlds, each item requested about twice)
    def compute():
        out = os.path.join(WORK, "cache.json")
        work = os.path.join(WORK, "cachework")
        r, rec = _governed_json([sys.executable, "-m", "crucible.scale.gatev", "cache", "--world", "civ",
                                 "--profile", "development", "--hours", "96", "--work", work, "--out", out], out,
                                "M: representation cache over three worlds", "standard")
        return {"ok": r is not None, "result": r, "usage": rec}
    return _cached("cache", compute)


@functools.lru_cache(maxsize=1)
def capacity() -> dict:
    def compute():
        out = os.path.join(WORK, "capacity.json")
        r, rec = _governed_json([sys.executable, "-m", "crucible.scale.gatev", "capacity", "--n", "10000000",
                                 "--out", out], out, "M: 10M capacity", "internet")
        return {"ok": r is not None, "result": r, "usage": rec}
    return _cached("capacity-10M", compute)


def targets_adoption(root: str) -> dict:
    """Whether every committed version of the targets file is adopted by a DECISION_LOG entry citing its SHA-256,
    and when the first adopting commit was made."""
    import hashlib
    rel = "docs/v5/performance/targets.json"
    log = subprocess.run(["git", "log", "--format=%H %ct", "--", rel], cwd=root, capture_output=True,
                         text=True).stdout.split()
    commits = list(zip(log[0::2], [int(x) for x in log[1::2]]))
    decisions = open(os.path.join(root, "docs/v5/DECISION_LOG.md")).read()
    versions = []
    for c, t in commits:
        data = subprocess.run(["git", "show", f"{c}:crucible/{rel}"], cwd=root, capture_output=True).stdout
        if not data:
            data = subprocess.run(["git", "show", f"{c}:{rel}"], cwd=root, capture_output=True).stdout
        h = hashlib.sha256(data).hexdigest()
        versions.append({"commit": c, "time": t, "sha256": h, "adopted": h in decisions})
    cur = hashlib.sha256(open(os.path.join(root, rel), "rb").read()).hexdigest()
    first = min((v["time"] for v in versions), default=None)
    return {"versions": versions, "current_sha256": cur, "current_adopted": cur in decisions,
            "all_adopted": bool(versions) and all(v["adopted"] for v in versions), "first_commit_time": first,
            "current_committed": any(v["sha256"] == cur for v in versions)}


def all_usage() -> list[dict]:
    """Usage records of every governed job the performance gates ran in this process."""
    recs = []
    if suite.cache_info().currsize:
        recs += suite()["usage"]
    for f in (v4_rows, backend_matrix, approximation, event_driven, cache_workload, capacity):
        if f.cache_info().currsize:
            r = f()
            u = r.get("usage")
            recs += u if isinstance(u, list) else ([u] if u else [])
    return recs


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
__all__ = ["suite", "v4_rows", "backend_matrix", "approximation", "event_driven", "cache_workload", "capacity",
           "targets_adoption", "all_usage", "DAYS", "SEEDS", "ROOT", "T_START", "slots", "approx", "backends", "gatev"]
