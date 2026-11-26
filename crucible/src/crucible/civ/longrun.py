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
* recorded replay - as soon as a snapshot closes a segment, the segment is replayed from the store's events on top
  of the previous snapshot (genesis for the first) and must give the new snapshot's state digest; the last
  snapshot is taken at the end of the run and must equal the live world (digest and hash). With
  `prune_verified_blobs`, the events of segments that were verified and are older than the last two, the bytes of
  their `cols.batch` blobs (and of the array blobs their `extra` cites) and their opening snapshots are then removed
  from the store (a retention policy for long runs on a small disk; the report states how many bytes were pruned;
  full replay from genesis is then no longer possible, segment-wise verification was).

Interruption (usage limits stop processes without warning): every snapshot is a checkpoint - after its segment was
replayed, `<workdir>/progress.json` records the report so far, the accountant's counters and the snapshot list. A
run started again in the same directory on the same spec and code resumes there: the store is cut back to the
checkpoint (what was recorded after it is regenerated, and compared with the dropped events when they had been
flushed), the world is rebuilt by `World.load` in the new process and checked against the checkpoint's hash and
state digest. A run loses at most `snapshot_every` simulated days of work.

The report has everything Gate P measures; `suite(...)` runs several horizons and seeds, each in its own process,
`workers` at a time (launch the caller through `crucible.scale.slots --cpu <workers>`), in persistent per-run
directories (`job_dir`: $CRUCIBLE_CIV_RUNS or /workspace/data/crucible/runs-c5/civ, keyed by the code digest) so
that a suite started again skips completed runs and resumes interrupted ones; it removes each run's store after it
has been checked unless `keep_stores`.
"""
from __future__ import annotations

import json
import os
import resource
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


def _resume_child(copy_path: str, world_id: str, until: int, timeout_s: int = 3600, events_out: str = "-") -> dict:
    """Free resume: a NEW process loads the world from a copy of the store and advances it to `until`, recomputing
    every inference; its continuation's events (hashes; payloads of `ext` events) go to `events_out`."""
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    pr = subprocess.run([sys.executable, "-m", "crucible.civ.longrun", "resume", copy_path, world_id, str(until),
                         events_out], env=env, capture_output=True, timeout=timeout_s)
    if pr.returncode != 0:
        return {"ok": False, "error": pr.stderr.decode(errors="replace")[-1500:]}
    return {"ok": True, **json.loads(pr.stdout.decode().strip().splitlines()[-1])}


def _regen_resume_child(copy_path: str, world_id: str, at_seq: int, live_path: str, to_seq: int,
                        timeout_s: int = 3600) -> dict:
    """Exact resume (contract 1.7): a NEW process cuts the copy back to `at_seq`, loads the world from it
    (`World.load`: snapshot plus events) and re-executes the live run's recorded continuation (at_seq, to_seq]
    through the handlers with its recorded external outcomes (crucible.civ.determinism.regenerate_continuation)."""
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    pr = subprocess.run([sys.executable, "-m", "crucible.civ.longrun", "resume-regen", copy_path, world_id,
                         str(at_seq), live_path, str(to_seq)], env=env, capture_output=True, timeout=timeout_s)
    if pr.returncode != 0:
        return {"ok": False, "error": pr.stderr.decode(errors="replace")[-1500:]}
    return {"ok": True, **json.loads(pr.stdout.decode().strip().splitlines()[-1])}


def _rm_store(path: str) -> None:
    for p in (path, path + "-wal", path + "-shm"):
        if os.path.exists(p):
            os.remove(p)


def _read_json(path: str):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_json(path: str, obj) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _truncate_after(store, wid: str, seq: int) -> dict:
    """Drop the events and snapshots of `wid` after `seq` from the run's own store (an interrupted run resumes at
    its last checkpoint; what it recorded after that is regenerated). Returns what was dropped, so the regenerated
    events can be compared with the dropped ones (determinism across the interruption)."""
    con = store._conn()
    r = con.execute("SELECT seq, hash FROM events WHERE world_id=? AND seq>? ORDER BY seq DESC LIMIT 1",
                    (wid, int(seq))).fetchone()
    n = con.execute("SELECT COUNT(*) FROM events WHERE world_id=? AND seq>?", (wid, int(seq))).fetchone()[0]
    con.execute("DELETE FROM events WHERE world_id=? AND seq>?", (wid, int(seq)))
    ns = con.execute("SELECT COUNT(*) FROM snapshots WHERE world_id=? AND seq>?", (wid, int(seq))).fetchone()[0]
    con.execute("DELETE FROM snapshots WHERE world_id=? AND seq>?", (wid, int(seq)))
    con.commit()
    return {"events": int(n), "snapshots": int(ns), "last_seq": int(r[0]) if r else None,
            "last_hash": r[1] if r else None}


def code_key() -> str:
    """Digest of the source of the whole `crucible` package (every .py file) - a long run resumes only on the code
    it started with (a resumed trajectory on other code would mix two worlds)."""
    import hashlib
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha256()
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for f in sorted(files):
            if f.endswith(".py"):
                fp = os.path.join(d, f)
                with open(fp, "rb") as fh:
                    h.update(os.path.relpath(fp, root).encode() + b"\0" + fh.read())
    return h.hexdigest()[:16]


def run_horizon(profile: str, seed: int, days: int, *, population: str = "development", workdir: str | None = None,
                snapshot_every: int = 10, audit_every: int = 10, resume_days: list[int] | None = None,
                replay: bool = True, overrides: dict | None = None, log=None, keep_store: bool = False,
                prune_verified_blobs: bool = False, resume: bool = True, strict: bool = False) -> dict:
    """One long run (see the module docstring). Checkpoints: every snapshot (`snapshot_every` days, the resume
    days and the end) writes `<workdir>/progress.json` after its segment was replayed; a run started again in the
    same `workdir` with the same spec and code (`resume=True`) continues from the last checkpoint instead of from
    genesis - the store is cut back to the checkpoint, `World.load` rebuilds the world in this new process (a
    cross-process resume, verified against the checkpoint's hash and state digest), and every report field
    continues where it stopped. `report["interruptions"]` lists each resume."""
    import torch
    torch.set_num_threads(1)
    from ..core.kernel import World
    from ..storage.sqlite import SQLiteStore
    from . import generator
    from .accounting import Accountant, activity
    workdir = workdir or tempfile.mkdtemp(prefix="civlong-")
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, f"civ-{profile}-{seed}-{days}d.sqlite")
    progress_path = os.path.join(workdir, "progress.json")
    wid = f"civ-{profile}-{seed}-{days}d"
    t_start = time.time()
    sp = generator.spec(profile, seed, population, overrides=overrides)
    key = code_key()
    prog = _read_json(progress_path) if resume else None
    stale = None
    if prog is not None and (prog.get("spec_hash") != sp.hash() or prog.get("code") != key
                             or not os.path.exists(path)):
        stale = {"spec": prog.get("spec_hash") == sp.hash(), "code": prog.get("code") == key,
                 "store": os.path.exists(path)}
        prog = None
    resume_days = sorted(set(resume_days if resume_days is not None else
                             [d for d in (min(days, 2), days // 2) if 1 <= d < days]))
    store = None
    rep: dict = {"profile": profile, "seed": int(seed), "days": int(days), "population": population,
                 "world_id": wid, "failed": False, "error": None, "daily": [], "unexplained_violations": [],
                 "explained_violations": [], "proc_failed": 0, "accounting": [], "resume": [],
                 "replay": {"segments": 0, "events": 0, "mismatches": 0, "detail": [], "wall_s": 0.0},
                 "pruned_blob_mb": 0.0, "pruned_snapshots": 0, "interruptions": [], "code": key,
                 "spec_hash": sp.hash(), "restarted_stale": stale, "wall_s_before_resume": 0.0}
    try:
        if prog is None:
            for p in (path, path + "-wal", path + "-shm", progress_path):
                if os.path.exists(p):
                    os.remove(p)
            store = SQLiteStore(path)
            w = generator.generate(profile, seed, population, overrides=overrides, store=store, world_id=wid,
                                   strict=strict)
            rep["modules"] = len(w.modules)
            acc = Accountant(w)
            acc.feed(w.events_range(0))
            snaps = [(0, None), (w.seq, w.snapshot())]   # (seq, snapshot id); genesis segment starts at seq 0
            d0, pending_resume, tail = 0, None, None
            _checkpoint(progress_path, rep, w, acc, snaps, 0, None, None)
        else:
            rep = prog["rep"]
            store = SQLiteStore(path)
            ck = prog["checkpoint"]
            t1 = time.time()
            cut = _truncate_after(store, wid, ck["seq"])
            w = World.load(store, wid, strict=strict)
            ok = w.seq == ck["seq"] and w.last_hash == ck["hash"] and w.digest() == ck["digest"]
            rep["interruptions"].append({"resumed_at_day": ck["day"], "load_ok": bool(ok), "dropped": cut,
                                         "load_s": round(time.time() - t1, 1), "at": time.time()})
            if not ok:
                raise RuntimeError(f"resume from checkpoint day {ck['day']} gave seq {w.seq} hash {w.last_hash} "
                                   f"digest {w.digest()}, checkpoint {ck}")
            acc = Accountant(w).restore(prog["accountant"])
            snaps = [tuple(x) for x in prog["snaps"]]
            d0, pending_resume = int(ck["day"]), prog.get("pending_resume")
            tail = cut if cut.get("last_seq") else None
        prev = w.seq
        for d in range(d0, days):
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
            if pending_resume is not None and pending_resume["until"] == w.now:
                rep["resume"].append(_resume_compare(w, store, path, wid, pending_resume, d + 1))
                pending_resume = None
            rep["daily"].append(row)
            if (d + 1) % snapshot_every == 0 or d + 1 in resume_days or d + 1 == days:
                snaps.append((w.seq, w.snapshot()))
                if tail is not None and w.seq >= tail["last_seq"]:
                    got = store.read_events(wid, from_seq=tail["last_seq"] - 1, to_seq=tail["last_seq"])
                    rep["interruptions"][-1]["regenerated_equal"] = bool(got and got[0]["hash"] == tail["last_hash"])
                    tail = None
                if replay:
                    _replay_segment(w, store, wid, snaps[-2], snaps[-1], rep["replay"])
                    if prune_verified_blobs and len(snaps) >= 4:
                        rep["pruned_blob_mb"] += _prune(w, store, wid, snaps[-4], snaps[-3])
                        if snaps[-4][1]:
                            store.delete_snapshot(snaps[-4][1])     # verified, and no longer a resume point
                            rep["pruned_snapshots"] += 1
                if d + 1 in resume_days and d + 1 < days:
                    w.flush()
                    cp = os.path.join(workdir, f"resume-copy-{d + 1}.sqlite")
                    _backup(store, cp)
                    evo = cp + ".events.json"
                    child = _resume_child(cp, wid, (d + 2) * DAY, events_out=evo)
                    # the copy is kept until the live world has recorded the same day: the exact check
                    # re-executes the live record on it (`_resume_compare`)
                    pending_resume = {"day": d + 1, "until": (d + 2) * DAY, "child": child, "copy": cp,
                                      "events": evo, "at_seq": w.seq}
                _checkpoint(progress_path, rep, w, acc, snaps, d + 1, pending_resume, t_start)
            if log:
                log(json.dumps(row))
        w.flush()
        if replay:
            # the last snapshot against the live world, loaded in a child (no second state in the live process)
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            pr = subprocess.run([sys.executable, "-m", "crucible.civ.longrun", "snapshot", store.path, wid,
                                 snaps[-1][1]], env=env, capture_output=True, timeout=3600)
            last = json.loads(pr.stdout.decode().strip().splitlines()[-1]) if pr.returncode == 0 else \
                {"error": pr.stderr.decode(errors="replace")[-800:]}
            if snaps[-1][0] != w.seq or last.get("digest") != w.digest() or last.get("hash") != w.last_hash:
                rep["replay"]["mismatches"] += 1
                rep["replay"]["detail"].append({"segment": "end", "snapshot": last, "live": w.digest()})
        rep["final"] = {"seq": w.seq, "hash": w.last_hash, "digest": w.digest()}
        rep["entities"] = entity_census(w)
        w.close()
    except Exception as e:  # noqa: BLE001 - a run that crashes is a failed run, with the reason
        import traceback
        rep["failed"] = True
        rep["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-3000:]}"
    rep["wall_s"] = round(rep.get("wall_s_before_resume", 0.0) + time.time() - t_start, 1)
    rep["store_mb_final"] = round(_store_bytes(path) / 2**20, 2) if os.path.exists(path) else None
    rep["summary"] = summarize(rep)
    if store is not None:
        store.close()
    if not keep_store and not rep["failed"]:
        for p in (path, path + "-wal", path + "-shm"):
            if os.path.exists(p):
                os.remove(p)
    return rep


def _checkpoint(path, rep, w, acc, snaps, day, pending_resume, t_start) -> None:
    rep = dict(rep)
    if t_start is not None:
        rep["wall_s_before_resume"] = round(rep.get("wall_s_before_resume", 0.0) + time.time() - t_start, 1)
    _write_json(path, {"spec_hash": rep["spec_hash"], "code": rep["code"], "rep": rep, "accountant": acc.state(),
                       "snaps": [list(x) for x in snaps], "pending_resume": pending_resume,
                       "checkpoint": {"day": int(day), "seq": w.seq, "hash": w.last_hash, "digest": w.digest(),
                                      "sid": snaps[-1][1]}})


def entity_census(w) -> dict:
    """How many of each persistent entity kind the world holds (participants, organisations, relationships,
    products, technologies, institutions; Part 10.4)."""
    st = w.state
    from ..substrate import columns

    def n(t):
        return st.count(t) if t in st.tables else 0
    out = {"principals": n("principals"), "orgs": n("orgs"), "members": n("members"),
           "forum_follows": n("forum_follows"), "products": n("products"), "subscriptions": n("subscriptions"),
           "designs": n("designs"), "sci_papers": n("sci_papers"), "constitutions": n("constitutions"),
           "deployments": n("deployments"), "repos": n("repos"), "communities": n("communities"),
           "cap_declarations": n("cap_decls") or n("capabilities")}
    if "pop" in columns.tables(w):
        out["population_rows"] = int(columns.rows(w, "pop"))
    return out


def _replay_segment(w, store, wid, a, b, acc: dict) -> None:
    """Replay the recorded events of segment a -> b on top of snapshot a and compare with snapshot b - in a CHILD
    process: building a second world in the live world's process (World.replay / from_blob run every module's
    install and on_load) shares process-global state with it and was seen to change the live trajectory (an observer
    effect of the verification itself; test_v5_civ_longrun)."""
    t = time.time()
    (s0, sid0), (s1, sid1) = a, b
    w.flush()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    pr = subprocess.run([sys.executable, "-m", "crucible.civ.longrun", "verify", store.path, wid, str(s0),
                         sid0 or "-", str(s1), sid1], env=env, capture_output=True, timeout=6 * 3600)
    acc["segments"] += 1
    if pr.returncode != 0:
        acc["mismatches"] += 1
        acc["detail"] = (acc["detail"] + [{"segment": [s0, s1], "error": pr.stderr.decode(errors="replace")[-800:]}])[:5]
    else:
        r = json.loads(pr.stdout.decode().strip().splitlines()[-1])
        acc["events"] += int(r["events"])
        if r["replayed"] != r["recorded"] or r["seq"] != s1:
            acc["mismatches"] += 1
            acc["detail"] = (acc["detail"] + [{"segment": [s0, s1], **r}])[:5]
    acc["wall_s"] = round(acc["wall_s"] + time.time() - t, 1)


def _verify_child(path, wid, s0, sid0, s1, sid1) -> dict:
    from ..core.kernel import World
    from ..storage.sqlite import SQLiteStore
    store = SQLiteStore(path)
    evs = store.read_events(wid, from_seq=int(s0), to_seq=int(s1))
    base = store.get_snapshot(sid0)[1] if sid0 != "-" else None
    rw = World.replay(World_spec(store, wid), evs, world_id=f"{wid}-replay", base_blob=base, store=store)
    got = {"replayed": rw.digest(), "seq": rw.seq, "events": len(evs)}
    # one world at a time in this process (the peak is one state, not two: long runs verify near the job's RAM grant)
    rw.close()
    del rw, evs, base
    import gc
    gc.collect()
    # the recorded snapshot is loaded with the store attached (its modules' on_load read blobs, e.g. the adaptive
    # population's weights); nothing is written to the store (no event is applied or flushed)
    chk = World.from_blob(store.get_snapshot(sid1)[1], world_id=f"{wid}-check", store=store)
    got["recorded"] = chk.digest()
    got["recorded_hash"] = chk.last_hash
    chk.close()
    return got


def _snapshot_child(path, wid, sid) -> dict:
    """The digest, hash and seq of one recorded snapshot, loaded with the store attached in its own process."""
    from ..core.kernel import World
    from ..storage.sqlite import SQLiteStore
    store = SQLiteStore(path)
    chk = World.from_blob(store.get_snapshot(sid)[1], world_id=f"{wid}-last", store=store)
    return {"digest": chk.digest(), "hash": chk.last_hash, "seq": chk.seq}


def World_spec(store, wid):
    from ..core.kernel import WorldSpec
    return WorldSpec.from_dict(store.world_info(wid)["spec"])


def _prune(w, store, wid, a, b) -> float:
    """Remove the bytes of `cols.batch` blobs of a verified segment from the store (see the module docstring)."""
    from ..substrate import artifacts
    (s0, _), (s1, _) = a, b
    c = artifacts._cache(w)
    con = store._conn()
    freed = 0
    import re
    ref_re = re.compile(r"sha256:[0-9a-f]{64}")
    for e in store.read_events(wid, from_seq=s0, to_seq=s1):
        if e["type"] != "cols.batch":
            continue
        # the batch's own array blob and the array blobs its `extra` cites (e.g. the population's visit logs)
        refs = {e["payload"]["ref"]} | set(ref_re.findall(json.dumps(e["payload"].get("extra") or {})))
        for ref in sorted(refs):
            h = artifacts.parse_ref(ref)
            key = artifacts.blob_key(h)
            r = con.execute("SELECT length(data) FROM blobs WHERE key=?", (key,)).fetchone()
            if r is None:
                continue
            con.execute("DELETE FROM blobs WHERE key=?", (key,))
            c.persisted.discard(h)        # identical bytes recorded later are written again
            freed += int(r[0])
    # the segment's events too: it was replayed against its closing snapshot, and resume starts from a later one
    r = con.execute("SELECT COALESCE(SUM(LENGTH(payload)), 0), COUNT(*) FROM events WHERE world_id=? AND seq>? "
                    "AND seq<=?", (wid, int(s0), int(s1))).fetchone()
    con.execute("DELETE FROM events WHERE world_id=? AND seq>? AND seq<=?", (wid, int(s0), int(s1)))
    freed += int(r[0])
    return round(freed / 2**20, 2)


def _ratio(series: list[float], window: int) -> float:
    if not series:
        return 0.0
    w = max(1, min(window, len(series) // 3 or 1))
    early = sum(series[:w]) / w
    late = sum(series[-w:]) / w
    return float(late / early) if early > 0 else (0.0 if late == 0 else float("inf"))


def _resume_compare(w, store, path: str, wid: str, pend: dict, day: int) -> dict:
    """One cross-process resume check, when the live world `w` reached the day the resumed child ran to.

    `ok` (counted by Gate P) is the exact check: a new process loads the world from the copy of the store taken at
    the resume day and re-executes the live run's recorded continuation through the handlers with its recorded
    external outcomes; it must reach the live hash, state digest and sequence number. `free_ok` reports whether the
    free resume (the child that recomputed every inference) reached the same; when it did not, `free_divergence`
    says where and whether the first divergence came from a recorded non-reproducible outcome (inference numerics)
    or from the world's logic (crucible.civ.determinism.first_divergence)."""
    from .determinism import first_divergence
    child = pend["child"]
    live = {"hash": w.last_hash, "digest": w.digest(), "seq": w.seq}
    free_ok = bool(child.get("ok") and child["hash"] == live["hash"] and child["digest"] == live["digest"])
    out = {"from_day": pend["day"], "to_day": day, "free_ok": free_ok,
           "child": {k: child.get(k) for k in ("ok", "hash", "digest", "seq", "error", "wall_s")}, "live": live}
    evo = pend.get("events")
    if not free_ok and evo and os.path.exists(evo) and pend.get("at_seq") is not None:
        try:
            with open(evo) as fh:
                theirs = json.load(fh)
            mine = [{"seq": e.seq, "type": e.type, "hash": e.hash, "payload": e.payload if e.type == "ext" else None}
                    for e in w.events_range(int(pend["at_seq"]))]
            out["free_divergence"] = first_divergence(mine, theirs)
        except (OSError, ValueError) as e:
            out["free_divergence"] = {"error": f"{type(e).__name__}: {e}"}
    cp = pend.get("copy")
    if cp and os.path.exists(cp) and pend.get("at_seq") is not None:
        w.flush()
        rg = _regen_resume_child(cp, wid, int(pend["at_seq"]), path, w.seq)
        out["regen"] = {k: rg.get(k) for k in ("ok", "hash", "digest", "seq", "error", "wall_s", "events")}
        out["ok"] = bool(rg.get("ok") and rg["hash"] == live["hash"] and rg["digest"] == live["digest"]
                         and rg["seq"] == live["seq"])
    else:
        # the copy did not survive an interruption of this process: no exact check; the free one decides
        out["regen"] = None
        out["ok"] = free_ok
    if cp:
        _rm_store(cp)
    if evo and os.path.exists(evo):
        os.remove(evo)
    return out


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
            "resume_exact_checks": sum(1 for r in rep.get("resume") or [] if r.get("regen")),
            "free_resume_equal": sum(1 for r in rep.get("resume") or [] if r.get("free_ok", r.get("ok"))),
            "free_resume_divergence_origins": sorted({(r.get("free_divergence") or {}).get("origin") or "?"
                                                      for r in rep.get("resume") or [] if not r.get("free_ok", True)}),
            "replay_mismatches": (rep["replay"] or {}).get("mismatches") if rep.get("replay") else None,
            "replay_segments": (rep.get("replay") or {}).get("segments", 0),
            "late_over_early_economic": _ratio(eco, 30), "late_over_early_social": _ratio(soc, 30),
            "months": months, "wall_s_per_day_mean": round(sum(walls) / len(walls), 2) if walls else None,
            "wall_s_per_day_last10": round(sum(walls[-10:]) / len(walls[-10:]), 2) if walls else None,
            "peak_rss_mb": max((r["peak_rss_mb"] for r in daily), default=None),
            "store_mb_max": max((r["store_mb"] for r in daily), default=None),
            "events": daily[-1]["seq"] if daily else 0}


RUNS_ENV = "CRUCIBLE_CIV_RUNS"
DEFAULT_RUNS = "/workspace/data/crucible/runs-c5/civ"


def runs_root() -> str:
    """Where long runs keep their stores, checkpoints and reports: `$CRUCIBLE_CIV_RUNS`, else
    /workspace/data/crucible/runs-c5/civ when /workspace/data/crucible exists (outside the repository), else a
    temporary directory."""
    r = os.environ.get(RUNS_ENV)
    if r:
        return r
    if os.path.isdir(os.path.dirname(os.path.dirname(DEFAULT_RUNS))):
        return DEFAULT_RUNS
    return os.path.join(tempfile.gettempdir(), "crucible-civ-runs")


def job_dir(profile: str, population: str, h: int, s: int, root: str | None = None) -> str:
    """A run's directory: keyed by the code digest, so a run only ever resumes on the code it started with."""
    return os.path.join(root or runs_root(), code_key(), f"{profile}-{population}-h{h}-s{s}")


def jobs(horizons=(30, 90, 180, 365), seeds_per_horizon: int = 2, base_seed: int = 101) -> list[tuple[int, int]]:
    """(horizon, seed) of Gate P: distinct seeds per horizon (independent replications)."""
    return [(h, base_seed + 10 * i + k) for i, h in enumerate(horizons) for k in range(seeds_per_horizon)]


def complete(rep: dict | None) -> bool:
    return bool(rep) and not rep.get("failed") and not (rep.get("summary") or {}).get("failed")


def suite(horizons=(30, 90, 180, 365), seeds_per_horizon: int = 2, *, profile: str | None = None,
          population: str | None = None, workers: int = 1, workdir: str | None = None, base_seed: int = 101,
          keep_stores: bool = False, prune: bool = True, log=None, timeout_s: int = 24 * 3600,
          only_existing: bool = False, max_long: int | None = None, long_from: int = 365) -> list[dict]:
    """Every (horizon, seed) run in its own process, `workers` at a time; independent seeds (distinct per horizon).
    `max_long` caps how many runs of `long_from` days or more run at once (a run's memory grows with its simulated
    days and its verification child holds one more state: at day 365 one run and its child need about twice the
    memory of a 180-day run), the others fill the remaining workers.

    Runs live in `job_dir(...)` (persistent, outside the repository): a run whose report is complete is not run
    again; an interrupted one (no report, or a process stopped by a time or usage limit) resumes from its last
    checkpoint. `only_existing=True` runs nothing and returns the reports found (missing runs are reported as not
    run, i.e. failed)."""
    profile = profile or GATE_P_PROFILE["profile"]
    population = population or GATE_P_PROFILE["population"]
    todo = jobs(horizons, seeds_per_horizon, base_seed)
    todo.sort(key=lambda x: -x[0])               # longest first
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    running: list = []
    out = []
    pending = []
    for h, s in todo:
        wd = job_dir(profile, population, h, s, workdir)
        rep = _read_json(os.path.join(wd, "report.json"))
        if complete(rep):
            out.append(rep)
        elif only_existing:
            prog = _read_json(os.path.join(wd, "progress.json"))
            day = (prog or {}).get("checkpoint", {}).get("day", 0)
            out.append(rep or {"profile": profile, "seed": s, "days": h, "failed": True,
                               "error": f"not run to completion (checkpoint at day {day})",
                               "summary": {"failed": True, "completed_days": day}})
        else:
            pending.append((h, s, wd))
    while pending or running:
        while pending and len(running) < max(1, workers):
            n_long = sum(1 for x in running if x[2] >= long_from)
            pick = next((k for k, (h_, _, _) in enumerate(pending)
                         if max_long is None or h_ < long_from or n_long < max_long), None)
            if pick is None:
                break
            h, s, wd = pending.pop(pick)
            os.makedirs(wd, exist_ok=True)
            rp = os.path.join(wd, "report.json")
            args = [sys.executable, "-m", "crucible.civ.longrun", "run", profile, str(s), str(h), population, wd, rp]
            if keep_stores:
                args.append("--keep")
            if prune:
                args.append("--prune")
            running.append((subprocess.Popen(args, env=env, stdout=open(os.path.join(wd, "daily.jsonl"), "a"),
                                             stderr=subprocess.STDOUT), rp, h, s, time.time()))
        time.sleep(2)
        still = []
        for pr, rp, h, s, t0 in running:
            if pr.poll() is None and time.time() - t0 < timeout_s:
                still.append((pr, rp, h, s, t0))
                continue
            if pr.poll() is None:
                pr.kill()
                pr.wait()
            r = _read_json(rp) if pr.returncode == 0 or os.path.exists(rp) else None
            if r is None:
                r = {"profile": profile, "seed": s, "days": h, "failed": True,
                     "error": f"no report (exit {pr.returncode})", "summary": {"failed": True}}
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
        evo = argv[4] if len(argv) > 4 and argv[4] != "-" else None
        t = time.time()
        w = World.load(SQLiteStore(path), wid)
        s0 = w.seq
        w.advance_time(until)
        if evo:
            _write_json(evo, [{"seq": e.seq, "type": e.type, "hash": e.hash,
                               "payload": e.payload if e.type == "ext" else None} for e in w.events_range(s0)])
        print(json.dumps({"hash": w.last_hash, "digest": w.digest(), "seq": w.seq,
                          "wall_s": round(time.time() - t, 1)}))
        return 0
    if argv and argv[0] == "resume-regen":
        import torch
        torch.set_num_threads(1)
        from ..core.kernel import World
        from ..storage.sqlite import SQLiteStore
        from .determinism import regenerate_continuation
        cp, wid, at_seq, live_path, to_seq = argv[1], argv[2], int(argv[3]), argv[4], int(argv[5])
        t = time.time()
        st = SQLiteStore(cp)
        _truncate_after(st, wid, at_seq)          # the free resume recorded its own continuation in this copy
        w = World.load(st, wid)
        if w.seq != at_seq:
            raise RuntimeError(f"the copy loads at seq {w.seq}, not at {at_seq}")
        live = SQLiteStore(live_path)
        evs = live.read_events(wid, from_seq=at_seq, to_seq=to_seq)
        # recorded outcomes kept as blobs (array outputs of inference) were written to the live store after the copy
        w.runtime["blob_sources"] = [*w.runtime.get("blob_sources", ()), live]
        regenerate_continuation(w, evs)
        print(json.dumps({"hash": w.last_hash, "digest": w.digest(), "seq": w.seq, "events": len(evs),
                          "wall_s": round(time.time() - t, 1)}))
        return 0
    if argv and argv[0] == "snapshot":
        import torch
        torch.set_num_threads(1)
        print(json.dumps(_snapshot_child(*argv[1:4])))
        return 0
    if argv and argv[0] == "verify":
        import torch
        torch.set_num_threads(1)
        print(json.dumps(_verify_child(*argv[1:7])))
        return 0
    if argv and argv[0] == "run":
        profile, seed, days, population, wd, rp = argv[1], int(argv[2]), int(argv[3]), argv[4], argv[5], argv[6]
        opt = {}
        for a in argv[7:]:
            if a.startswith("--snapshot-every="):
                opt["snapshot_every"] = int(a.split("=", 1)[1])
            elif a.startswith("--resume-days="):
                v = a.split("=", 1)[1]
                opt["resume_days"] = [int(x) for x in v.split(",") if x]
        rep = run_horizon(profile, seed, days, population=population, workdir=wd, keep_store="--keep" in argv,
                          prune_verified_blobs="--prune" in argv, log=lambda m: print(m, flush=True), **opt)
        _write_json(rp, rep)
        return 0 if not rep["failed"] else 1
    print("usage: python -m crucible.civ.longrun run <profile> <seed> <days> <population> <workdir> <report.json> "
          "[--keep] [--prune] [--snapshot-every=N] [--resume-days=a,b] | resume <store.sqlite> <world_id> <until_t> "
          "[<events_out.json>] | resume-regen <copy.sqlite> <world_id> <at_seq> <live.sqlite> <to_seq> "
          "| verify <store.sqlite> <world_id> <seq0> <snapshot0|-> <seq1> <snapshot1>")
    return 2


if __name__ == "__main__":
    import crucible
    crucible.load_all_modules()
    sys.exit(_main(sys.argv[1:]))
