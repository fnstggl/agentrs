"""Child process for the cross-process snapshot/resume tests (tests/test_v5_substrate_resume.py, gate SP-2.3).

    python tests/v5_resume_child.py ref    SPEC.json DAYS             uninterrupted in-memory run to DAYS
    python tests/v5_resume_child.py part   SPEC.json DB SNAP_DAY DAYS SQLite-stored run: snapshot at SNAP_DAY, run to
                                                                       DAYS, flush, then die with os._exit (no close:
                                                                       the process's artifact pool is lost)
    python tests/v5_resume_child.py resume SPEC.json DB DAYS          a NEW process: World.load from the store
                                                                       (snapshot + replay of the events after it), run
                                                                       on to DAYS
    python tests/v5_resume_child.py replay SPEC.json DB               a NEW process: replay the whole stored log from
                                                                       genesis with the store's blobs only
    python tests/v5_resume_child.py regen  SPEC.json DB               a NEW process: regenerate the stored log (handlers
                                                                       re-run, recorded outcomes injected)

Every mode prints one JSON line: seq, last hash, digest, invariant violations, how many recorded neural inferences
were (re-)executed in this process (`inference_runs`, counted at the recorded-inference boundary) and how many torch
backward passes ran (`backward_passes`: training re-executed, e.g. when on_load rebuilds weights from an older
checkpoint plus logged updates).
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


def report(w, **extra):
    out = {"seq": w.seq, "hash": w.last_hash, "digest": w.digest(), "invariants": w.check_invariants()[:5],
           "inference_runs": RUNS["n"], "backward_passes": RUNS["backward"], "modules": sorted(w.modules), **extra}
    print(json.dumps(out), flush=True)


def main(argv):
    mode, spec_path = argv[1], argv[2]
    spec = WorldSpec.from_dict(json.load(open(spec_path)))
    day = 86400
    if mode == "ref":
        w = World.create(spec, world_id="resume")
        w.advance_time(int(float(argv[3]) * day))
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
