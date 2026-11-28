"""Shared setup for the G2-decision probes. Run from /workspace/camlive:

    G2D_SRC=<copy of cam_live/src> .venv/bin/python cam_live/docs/verification/repro/g2decision/<probe>.py [--json P]

G2D_SRC selects the code under test (the freeze snapshot: `git archive f54d8cc | tar -x -C <dir>`, then
G2D_SRC=<dir>/cam_live/src); default: this worktree's cam_live/src. A check prints PASS when the claimed property
HOLDS and FAIL when it is refuted. Exit status 1 when any check failed. Probes only read and call the code.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("G2D_SRC") or os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

RESULTS = []
print(f"[INFO] cam_live source: {SRC}", flush=True)


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    return bool(ok)


def info(name, value):
    RESULTS.append({"info": name, "value": value})
    print(f"[INFO] {name}: {value}", flush=True)


def finish(argv=None):
    argv = sys.argv if argv is None else argv
    if "--json" in argv:
        with open(argv[argv.index("--json") + 1], "w") as f:
            json.dump(RESULTS, f, indent=1, default=str)
    n_fail = sum(1 for r in RESULTS if r.get("ok") is False)
    print(f"\n{sum(1 for r in RESULTS if 'check' in r)} checks, {n_fail} failed")
    return 1 if n_fail else 0
