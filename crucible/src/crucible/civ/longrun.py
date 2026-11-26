"""Long-horizon civilisation runs (MASTER_SPEC Part 10.4, Gate P).

`run_horizon(profile, seed, days, population=...)` runs one civilisation for `days` simulated days on a SQLite
store and checks, as it goes:

* every simulated day: the world's invariants (every module's `check_invariants`), classified as explained (listed
  in `EXPLAINED` with the reason) or unexplained; process failures (`proc.failed`); the cheap accounting checks;
  economic and social activity of the day (crucible.civ.accounting.activity); wall time, peak and current RSS, and
  the store's size;
* every `audit_every` days and at the end: the full event accounting (`Accountant.check`, including recomputing
  every balance from its postings);
* every `snapshot_every` days: a snapshot in the store;
* at `resume_days`: cross-process resume - the store is copied (SQLite backup), a NEW Python process loads the world
  from the copy (`World.load`: latest snapshot plus the events after it), advances it one simulated day and reports
  its hash and state digest; this process advances its own world the same day and compares;
* at the end: recorded replay - every segment between consecutive snapshots (and genesis to the first) is replayed
  from the store's events on top of the earlier snapshot and must give the later snapshot's state digest, and the
  last segment the live world's digest and hash.

The report has everything Gate P measures; `suite(...)` runs several horizons and seeds, each in its own process,
`workers` at a time (launch the caller through `crucible.scale.slots --cpu <workers>`), and removes each run's store
after it has been checked unless `keep_stores`.
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

DAY = 86400
# invariant messages that are understood and documented (substring -> reason). Empty: every violation counts.
EXPLAINED: dict[str, str] = {}
# the documented population profile of Gate P runs (MASTER_SPEC Part 3.3 "development": 1,000 behavioural, 32
# adaptive, 8 strategic and 2 semantic individuals)
GATE_P_PROFILE = {"profile": "balanced", "population": "development"}


def _rss_mb() -> float:
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        return float("nan")


def _store_bytes(path: str) -> int:
    return sum(os.path.getsize(p) for p in (path, path + "-wal", path + "-shm") if os.path.exists(p))


def _classify(errs: list[str]) -> tuple[list[str], list[dict]]:
    unexplained, explained = [], []
    for e in errs:
        why = next((r for k, r in EXPLAINED.items() if k in e), None)
        if why is None:
            unexplained.append(e)
        else:
            explained.append({"violation": e, "reason": why})
    return unexplained, explained


def _backup(src_store, dst_path: str) -> None:
    src = src_store._conn()
    dst = sqlite3.connect(dst_path)
    with dst:
        src.backup(dst)
    dst.close()


def _resume_child(copy_path: str, world_id: str, until: int, timeout_s: int = 3600) -> dict:
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    pr = subprocess.run([sys.executable, "-m", "crucible.civ.longrun", "resume", copy_path, world_id, str(until)],
                        env=env, capture_output=True, timeout=timeout_s)
    if pr.returncode != 0:
        return {"ok": False, "error": pr.stderr.decode(errors="replace")[-1500:]}
    return {"ok": True, **json.loads(pr.stdout.decode().strip().splitlines()[-1])}


def run_horizon(profile: str, seed: int, days: int, *, population: str = "development", workdir: str | None = None,
                snapshot_every: int = 10, audit_every: int = 10, resume_days: list[int] | None = None,
                replay: bool = True, overrides: dict | None = None, log=None, keep_store: bool = False) -> dict:
    import torch
    torch.set_num_threads(1)
    from ..core.kernel import World
    from ..storage.sqlite import SQLiteStore
    from . import generator
    from .accounting import Accountant, activity
    workdir = workdir or tempfile.mkdtemp(prefix="civlong-")
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, f"civ-{profile}-{seed}-{days}d.sqlite")
    for p in (path, path + "-wal", path + "-shm"):
        if os.path.exists(p):
            os.remove(p)
    store = SQLiteStore(path)
    wid = f"civ-{profile}-{seed}-{days}d"
    t_start = time.time()
    rep: dict = {"profile": profile, "seed": int(seed), "days": int(days), "population": population,
                 "world_id": wid, "failed": False, "error": None, "daily": [], "unexplained_violations": [],
                 "explained_violations": [], "proc_failed": 0, "accounting": [], "resume": [], "replay": None}
    try:
        w = generator.generate(profile, seed, population, overrides=overrides, store=store, world_id=wid)
        rep["spec_hash"] = w.spec.hash()
        rep["modules"] = len(w.modules)
        acc = Accountant(w)
        acc.feed(w.events_range(0))
        snaps = [(0, None)]            # (seq, snapshot id); genesis segment starts at seq 0
        sid = w.snapshot()
        snaps.append((w.seq, sid))
        resume_days = sorted(set(resume_days if resume_days is not None else
                                 [d for d in (min(days, 2), days // 2) if 1 <= d < days]))
        pending_resume = None
        prev = w.seq
        for d in range(days):
            t1 = time.time()
            w.advance_time((d + 1) * DAY)
            evs = w.events_range(prev)
            prev = w.seq
            acc.feed(evs)
            inv = w.check_invariants()
            un, ex = _classify(inv)
            fails = [e for e in evs if e.type == "proc.failed"]
            rep["proc_failed"] += len(fails)
            un += [f"proc.failed {e.payload['proc']}: {e.payload['error'][:160]}" for e in fails]
            if un:
                rep["unexplained_violations"].extend({"day": d + 1, "violation": v} for v in un[:20])
            rep["explained_violations"].extend(dict(x, day=d + 1) for x in ex[:20])
            a = activity(evs)
            row = {"day": d + 1, "wall_s": round(time.time() - t1, 2), "events": len(evs), "seq": w.seq,
                   "invariant_violations": len(un), "explained": len(ex), **a,
                   "rss_mb": round(_rss_mb(), 1), "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF)
                                                                    .ru_maxrss / 1024, 1),
                   "store_mb": round(_store_bytes(path) / 2**20, 2)}
            if (d + 1) % audit_every == 0 or d + 1 == days:
                chk = acc.check()
                bad = {k: v for k, v in chk.items() if v}
                rep["accounting"].append({"day": d + 1, "mismatches": sum(len(v) for v in chk.values()),
                                          "detail": bad})
                row["accounting_mismatches"] = rep["accounting"][-1]["mismatches"]
            if (d + 1) % snapshot_every == 0 or d + 1 in resume_days or d + 1 == days:
                snaps.append((w.seq, w.snapshot()))
            if pending_resume is not None and pending_resume["until"] == w.now:
                child = pending_resume["child"]
                ok = child.get("ok") and child["hash"] == w.last_hash and child["digest"] == w.digest()
                rep["resume"].append({"from_day": pending_resume["day"], "to_day": d + 1, "ok": bool(ok),
                                      "child": {k: child.get(k) for k in ("ok", "hash", "digest", "seq", "error",
                                                                          "wall_s")},
                                      "live": {"hash": w.last_hash, "digest": w.digest(), "seq": w.seq}})
                pending_resume = None
            if d + 1 in resume_days and d + 1 < days:
                w.flush()
                cp = os.path.join(workdir, f"resume-copy-{d + 1}.sqlite")
                _backup(store, cp)
                child = _resume_child(cp, wid, (d + 2) * DAY)
                for p in (cp, cp + "-wal", cp + "-shm"):
                    if os.path.exists(p):
                        os.remove(p)
                pending_resume = {"day": d + 1, "until": (d + 2) * DAY, "child": child}
            rep["daily"].append(row)
            if log:
                log(json.dumps(row))
        w.flush()
        if replay:
            rep["replay"] = _replay_segments(w, store, wid, snaps)
        rep["final"] = {"seq": w.seq, "hash": w.last_hash, "digest": w.digest()}
        w.close()
    except Exception as e:  # noqa: BLE001 - a run that crashes is a failed run, with the reason
        import traceback
        rep["failed"] = True
        rep["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}"
    rep["wall_s"] = round(time.time() - t_start, 1)
    rep["store_mb_final"] = round(_store_bytes(path) / 2**20, 2) if os.path.exists(path) else None
    rep["summary"] = summarize(rep)
    store.close()
    if not keep_store and not rep["failed"]:
        for p in (path, path + "-wal", path + "-shm"):
            if os.path.exists(p):
                os.remove(p)
    return rep


def _replay_segments(w, store, wid, snaps) -> dict:
    """Replay every segment between snapshots from the recorded events; compare state digests."""
    from ..core.kernel import World
    t = time.time()
    mism = []
    n_ev = 0
    for (s0, sid0), (s1, sid1) in zip(snaps, snaps[1:]):
        evs = store.read_events(wid, from_seq=s0, to_seq=s1)
        n_ev += len(evs)
        base = store.get_snapshot(sid0)[1] if sid0 else None
        rw = World.replay(w.spec, evs, world_id=f"{wid}-replay", base_blob=base, store=store)
        want = World.from_blob(store.get_snapshot(sid1)[1], world_id=f"{wid}-check").digest()
        if rw.digest() != want or rw.seq != s1:
            mism.append({"segment": [s0, s1], "replayed": rw.digest(), "recorded": want})
    # the last snapshot is at the end of the run: compare with the live world too
    if snaps[-1][0] == w.seq:
        last = World.from_blob(store.get_snapshot(snaps[-1][1])[1], world_id=f"{wid}-last")
        if last.digest() != w.digest() or last.last_hash != w.last_hash:
            mism.append({"segment": "end", "replayed": last.digest(), "recorded": w.digest()})
    return {"segments": len(snaps) - 1, "events": n_ev, "mismatches": len(mism), "detail": mism[:5],
            "wall_s": round(time.time() - t, 1)}


def _ratio(series: list[float], window: int) -> float:
    if not series:
        return 0.0
    w = max(1, min(window, len(series) // 3 or 1))
    early = sum(series[:w]) / w
    late = sum(series[-w:]) / w
    return float(late / early) if early > 0 else (0.0 if late == 0 else float("inf"))


def summarize(rep: dict) -> dict:
    daily = rep.get("daily") or []
    eco = [r["economic"] for r in daily[1:]] or [r["economic"] for r in daily]
    soc = [r["social"] for r in daily[1:]] or [r["social"] for r in daily]
    months = [{"month": k // 30 + 1, "economic": sum(r["economic"] for r in daily[k:k + 30]),
               "social": sum(r["social"] for r in daily[k:k + 30])} for k in range(0, len(daily), 30)]
    walls = [r["wall_s"] for r in daily]
    return {"completed_days": len(daily), "failed": bool(rep.get("failed")) or len(daily) < rep.get("days", 0),
            "unexplained_invariant_violations": len(rep.get("unexplained_violations") or []),
            "accounting_mismatches": sum(a["mismatches"] for a in rep.get("accounting") or []),
            "resume_checks": len(rep.get("resume") or []),
            "resume_mismatches": sum(0 if r["ok"] else 1 for r in rep.get("resume") or []),
            "replay_mismatches": (rep["replay"] or {}).get("mismatches") if rep.get("replay") else None,
            "late_over_early_economic": _ratio(eco, 30), "late_over_early_social": _ratio(soc, 30),
            "months": months, "wall_s_per_day_mean": round(sum(walls) / len(walls), 2) if walls else None,
            "wall_s_per_day_last10": round(sum(walls[-10:]) / len(walls[-10:]), 2) if walls else None,
            "peak_rss_mb": max((r["peak_rss_mb"] for r in daily), default=None),
            "store_mb_max": max((r["store_mb"] for r in daily), default=None),
            "events": daily[-1]["seq"] if daily else 0}


def suite(horizons=(30, 90, 180, 365), seeds_per_horizon: int = 2, *, profile: str | None = None,
          population: str | None = None, workers: int = 1, workdir: str | None = None, base_seed: int = 101,
          keep_stores: bool = False, log=None, timeout_s: int = 24 * 3600) -> list[dict]:
    """Every (horizon, seed) run in its own process; independent seeds (distinct per horizon)."""
    profile = profile or GATE_P_PROFILE["profile"]
    population = population or GATE_P_PROFILE["population"]
    workdir = workdir or tempfile.mkdtemp(prefix="civsuite-")
    jobs = [(h, base_seed + 10 * i + k) for i, h in enumerate(horizons) for k in range(seeds_per_horizon)]
    jobs.sort(key=lambda x: -x[0])               # longest first
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    running: list = []
    out = []
    while jobs or running:
        while jobs and len(running) < max(1, workers):
            h, s = jobs.pop(0)
            wd = os.path.join(workdir, f"h{h}-s{s}")
            os.makedirs(wd, exist_ok=True)
            rp = os.path.join(wd, "report.json")
            args = [sys.executable, "-m", "crucible.civ.longrun", "run", profile, str(s), str(h), population, wd, rp]
            if keep_stores:
                args.append("--keep")
            running.append((subprocess.Popen(args, env=env, stdout=open(os.path.join(wd, "daily.jsonl"), "w"),
                                             stderr=subprocess.STDOUT), rp, h, s, time.time()))
        time.sleep(2)
        still = []
        for pr, rp, h, s, t0 in running:
            if pr.poll() is None and time.time() - t0 < timeout_s:
                still.append((pr, rp, h, s, t0))
                continue
            if pr.poll() is None:
                pr.kill()
            try:
                with open(rp) as fh:
                    r = json.load(fh)
            except (OSError, ValueError):
                r = {"profile": profile, "seed": s, "days": h, "failed": True, "error": f"no report (exit {pr.poll()})",
                     "summary": {"failed": True}}
            out.append(r)
            if log:
                log(f"horizon {h} seed {s}: {json.dumps(r.get('summary'))[:600]}")
        running = still
    return out


def _main(argv) -> int:
    if argv and argv[0] == "resume":
        import torch
        torch.set_num_threads(1)
        from ..core.kernel import World
        from ..storage.sqlite import SQLiteStore
        path, wid, until = argv[1], argv[2], int(argv[3])
        t = time.time()
        w = World.load(SQLiteStore(path), wid)
        w.advance_time(until)
        print(json.dumps({"hash": w.last_hash, "digest": w.digest(), "seq": w.seq, "wall_s": round(time.time() - t, 1)}))
        return 0
    if argv and argv[0] == "run":
        profile, seed, days, population, wd, rp = argv[1], int(argv[2]), int(argv[3]), argv[4], argv[5], argv[6]
        rep = run_horizon(profile, seed, days, population=population, workdir=wd, keep_store="--keep" in argv,
                          log=lambda m: print(m, flush=True))
        with open(rp, "w") as fh:
            json.dump(rep, fh)
        return 0 if not rep["failed"] else 1
    print("usage: python -m crucible.civ.longrun run <profile> <seed> <days> <population> <workdir> <report.json> "
          "[--keep] | resume <store.sqlite> <world_id> <until_t>")
    return 2


if __name__ == "__main__":
    import crucible
    crucible.load_all_modules()
    sys.exit(_main(sys.argv[1:]))
