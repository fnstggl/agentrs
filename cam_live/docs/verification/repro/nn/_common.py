"""Shared setup for the G-nn verification probes. Run every probe from /workspace/camlive:

    .venv/bin/python cam_live/docs/verification/repro/nn/<probe>.py

Probes only read the code under test; they write results to stdout (and optionally --json PATH).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

RESULTS = []


def check(name, ok, detail=""):
    """Record one check. ok=True: the claim held; ok=False: refuted / defect reproduced."""
    RESULTS.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    return bool(ok)


def info(name, value):
    RESULTS.append({"info": name, "value": value})
    print(f"[INFO] {name}: {value}", flush=True)


def finish(argv=None):
    argv = sys.argv if argv is None else argv
    if "--json" in argv:
        path = argv[argv.index("--json") + 1]
        with open(path, "w") as f:
            json.dump(RESULTS, f, indent=1, default=str)
    n_fail = sum(1 for r in RESULTS if r.get("ok") is False)
    print(f"\n{sum(1 for r in RESULTS if 'check' in r)} checks, {n_fail} failed")
    return 1 if n_fail else 0
