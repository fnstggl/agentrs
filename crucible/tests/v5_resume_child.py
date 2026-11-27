"""Child process for the cross-process snapshot/resume tests (tests/test_v5_substrate_resume.py, gate SP-2.3).

    python tests/v5_resume_child.py ref    SPEC.json DAYS [DB]        uninterrupted run to DAYS (in memory; with DB its
                                                                       events are also written there, for diagnosis)
    python tests/v5_resume_child.py part   SPEC.json DB SNAP_DAY DAYS SQLite-stored run: snapshot at SNAP_DAY, run to
                                                                       DAYS, flush, then die with os._exit (no close:
                                                                       the process's artifact pool is lost)
    python tests/v5_resume_child.py resume SPEC.json DB DAYS [--flush] a NEW process: World.load from the store
                                                                       (snapshot + replay of the events after it), run
                                                                       on to DAYS
    python tests/v5_resume_child.py replay SPEC.json DB               a NEW process: replay the whole stored log from
                                                                       genesis with the store's blobs only
    python tests/v5_resume_child.py regen  SPEC.json DB               a NEW process: regenerate the stored log (handlers
                                                                       re-run, recorded outcomes injected)

Every mode prints one JSON line: seq, last hash, digest, invariant violations, how many recorded neural inferences
were (re-)executed in this process (`inference_runs`, counted at the recorded-inference boundary) and how many torch
backward passes ran (`backward_passes`: training re-executed, e.g. when on_load rebuilds weights from an older
checkpoint plus logged updates), and the deterministic cost model's memo outcomes in this process (`meter`: anchors
this process set, remembered, and measurements above the tolerance that were billed fresh - the one way measurement
noise can enter a trajectory, D-20/D-80).
"""
import json
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import crucible  # noqa: E402

crucible.load_all_modules()
from crucible.core.kernel import World, WorldSpec  # noqa: E402
from crucible.storage.sqlite import SQLiteStore  # noqa: E402
from crucible.substrate import inference  # noqa: E402

RUNS = {"n": 0, "backward": 0}
try:
    import torch
    torch.set_num_threads(1)
    _bw = torch.Tensor.backward

    def _count_backward(self, *a, **kw):     # training steps re-executed in this process (skeptic S2)
        RUNS["backward"] += 1
        return _bw(self, *a, **kw)
    torch.Tensor.backward = _count_backward
except ImportError:   # worlds without neural participants do not need it
    pass

_orig = inference.recorded


def _counting(ctx, kind, model_id, model_hash, inputs, fn):
    def counted():
        RUNS["n"] += 1
        return fn()
    return _orig(ctx, kind, model_id, model_hash, inputs, counted)


inference.recorded = _counting
for _m in list(sys.modules.values()):          # modules that imported the function by name
    if getattr(_m, "recorded", None) is _orig and _m is not inference:
        _m.recorded = _counting


_EXCEEDED: list = []
_CLOSE: dict = {}


def _watch_exceeded():
    """Record every job the deterministic cost model billed fresh above its anchor's tolerance (diagnosis)."""
    from crucible.sandbox import runner as R
    orig = R.Worker._finish

    def _finish(self, ok, result, error, stdout, **kw):
        out = orig(self, ok, result, error, stdout, **kw)
        m = out.get("meter") or {}
        if m.get("memo") in ("remembered", "exceeded") and m.get("billed_ns"):
            # how close the job came to the tolerance (cpu / (MEMO_TOL x billed + MEMO_SLACK_NS)): first jobs of a
            # worker the platform cold-started vs. the rest
            from crucible.sandbox import metering as M
            first = "warmup_exempt" if kw.get("warmup") else ("first_job" if getattr(self, "jobs", 0) <= 1
                                                              else "later_job")
            anchor = int(m["billed_ns"]) - max(0, int(kw.get("load") or 0))     # remembered: anchor + restart load
            _CLOSE.setdefault(f"{getattr(self, 'cold', None)}:{first}", []).append(
                round(max(0, int(kw.get("cpu") or 0)) / (M.MEMO_TOL * max(anchor, 0) + M.MEMO_SLACK_NS), 4))
        if m.get("memo") == "exceeded":
            _EXCEEDED.append({"label": getattr(self, "label", None), "cold": getattr(self, "cold", None),
                              "worker_jobs": getattr(self, "jobs", None), "cpu_ns": m.get("cpu_ns"),
                              "billed_ns": m.get("billed_ns"), "kind": (kw.get("job") or {}).get("kind"),
                              "target": str((kw.get("job") or {}).get("target"))[:60]})
        return out
    R.Worker._finish = _finish


_watch_exceeded()


def _meter() -> dict:
    from crucible.sandbox import metering as M
    return {"first": M.MEMO.misses, "remembered": M.MEMO.hits, "exceeded": M.MEMO.exceeded,
            "disk_errors": M.MEMO.disk_errors, "memo_path": M.default_memo_path(), "exceeded_jobs": _EXCEEDED[:20],
            "tolerance_use": {k: {"n": len(v), "max": max(v), "over_half": sum(1 for x in v if x > 0.5)}
                              for k, v in _CLOSE.items() if v}}


def report(w, **extra):
    out = {"seq": w.seq, "hash": w.last_hash, "digest": w.digest(), "invariants": w.check_invariants()[:5],
           "inference_runs": RUNS["n"], "backward_passes": RUNS["backward"], "modules": sorted(w.modules),
           "meter": _meter(), **extra}
    print(json.dumps(out), flush=True)


def main(argv):
    mode, spec_path = argv[1], argv[2]
    spec = WorldSpec.from_dict(json.load(open(spec_path)))
    day = 86400
    if mode == "ref":
        st = SQLiteStore(argv[4]) if len(argv) > 4 else None
        w = World.create(spec, world_id="resume", store=st)
        w.advance_time(int(float(argv[3]) * day))
        w.flush()
        report(w)
    elif mode == "part":
        db, snap_day, days = argv[3], float(argv[4]), float(argv[5])
        st = SQLiteStore(db)
        w = World.create(spec, world_id="resume", store=st)
        w.advance_time(int(snap_day * day))
        sid = w.snapshot()
        snap_seq = w.seq
        w.advance_time(int(days * day))
        w.flush()
        report(w, snapshot=sid, snapshot_seq=snap_seq)
        os._exit(0)        # a crash: no close(), nothing of this process survives but the store
    elif mode == "resume":
        db, days = argv[3], float(argv[4])
        w = World.load(SQLiteStore(db), "resume")
        loaded = {"loaded_seq": w.seq, "loaded_hash": w.last_hash}
        w.advance_time(int(days * day))
        if "--flush" in argv:          # diagnosis: keep the resumed run's events in the store (the test does not)
            w.flush()
        report(w, **loaded)
    elif mode == "replay":
        st = SQLiteStore(argv[3])
        w = World.replay(spec, st.read_events("resume"), world_id="resume-replay", store=st)
        report(w)
    elif mode == "regen":
        # re-execute every recorded command and process through the handlers, recorded outcomes injected
        st = SQLiteStore(argv[3])
        evs = st.read_events("resume")
        recs = sum(1 for e in evs if e["type"] in ("ext", "cmd.rejected", "proc.failed"))
        w = World.regenerate(spec, evs, world_id="resume-regen", store=st)
        report(w, recorded_ext_events=recs)
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main(sys.argv)
