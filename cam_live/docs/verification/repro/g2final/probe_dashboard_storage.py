"""G2-final probe: the dashboard's storage level (OP-03, INT3B round 3 2247328) through the served API.

    G2F_SRC=<archive>/cam_live/src .venv/bin/python cam_live/docs/verification/repro/g2final/probe_dashboard_storage.py

LabWorld runs with the runtime's storage thresholds set so that this machine's free space falls in each band; the
dashboard app (FastAPI TestClient, viewing token) serves /api/health. Then state.json — which the runtime writes and,
in a separated run, the learner's uid owns — carries malformed thresholds: the operator's health view must still
answer (no 500) and must not claim "ok" for values it cannot read.
  1  min 0, warn = free + 1e6 GB          -> level "warning", the thresholds are the run's
  2  min = free + 1e6 GB                  -> level "low"
  3  min 0, warn 0                        -> level "ok"
  4  a run that recorded no thresholds    -> the configuration's defaults, marked as such
  5  malformed thresholds in state.json ("x", null, NaN, a list): /api/health answers 200 and the level is not "ok"
"""
import json
import os
import secrets
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2fcommon import Report, scratch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from cam_live.contracts import RunFiles  # noqa: E402
from cam_live.runtime.dashboard import create_app  # noqa: E402
from cam_live.runtime.logs import atomic_write_json, read_json  # noqa: E402
from cam_live.runtime.loop import RuntimeConfig  # noqa: E402
from rt_helpers import make_lab  # noqa: E402

R = Report("probe_dashboard_storage")
TOK, OP = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
H = {"Authorization": f"Bearer {TOK}"}


def health(run):
    c = TestClient(create_app(run, TOK, operator_token=OP), raise_server_exceptions=False)
    r = c.get("/api/health", headers=H)
    try:
        return r.status_code, r.json()
    except Exception:  # noqa: BLE001
        return r.status_code, {"text": r.text[:200]}


def run_with(mn, wn):
    d = scratch("dstore")
    lab = make_lab(os.path.join(d, "run"), rt_cfg=RuntimeConfig(fsync=False, min_free_gb=mn, warn_free_gb=wn))
    lab.rt.run(max_iterations=3)
    lab.rt.close()
    return lab.run_dir


free = shutil.disk_usage(scratch("free")).free / 2 ** 30
for label, (mn, wn, want) in {"1 warning band": (0.0, free + 1e6, "warning"), "2 low band": (free + 1e6, free + 2e6, "low"),
                              "3 ok band": (0.0, 0.0, "ok")}.items():
    run = run_with(mn, wn)
    code, h = health(run)
    s = (h or {}).get("storage") or {}
    R.check(f"{label}: /api/health level {want!r} with the run's thresholds", code == 200 and s.get("level") == want
            and s.get("min_free_gb") == mn and s.get("warn_free_gb") == wn, {"http": code, "storage": s})
run = run_with(0.0, 0.0)
st = read_json(os.path.join(run, RunFiles.STATE))
st.pop("storage_guard", None)
atomic_write_json(os.path.join(run, RunFiles.STATE), st)
code, h = health(run)
s = (h or {}).get("storage") or {}
R.check("4 a run without recorded thresholds: the configuration's defaults, marked as defaults",
        code == 200 and "default" in str(s.get("source", "")), {"http": code, "storage": s})
for label, val in {"string": {"min_free_gb": "x", "warn_free_gb": "y"}, "null": {"min_free_gb": None},
                   "NaN": {"min_free_gb": float("nan"), "warn_free_gb": float("nan")}, "list": [1, 2]}.items():
    st["storage_guard"] = val
    with open(os.path.join(run, RunFiles.STATE), "w") as f:
        f.write(json.dumps(st))                       # json.dumps writes NaN as a bare NaN token (Python reads it)
    code, h = health(run)
    s = (h or {}).get("storage") or {}
    R.check(f"5 malformed thresholds in state.json ({label}): /api/health answers and does not report 'ok'",
            code == 200 and s.get("level") != "ok", {"http": code, "storage": s, "body": h if code != 200 else ""})
sys.exit(R.done())
