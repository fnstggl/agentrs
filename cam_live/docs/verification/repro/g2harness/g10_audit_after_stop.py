"""G2-harness probe g10 (root; real launcher): HX's intermittent launcher-test failure (``verify_audit`` not ok once,
tests/harness/auth/test_ea_privileged.py::test_launcher_starts_supervises_and_stops_services_under_their_own_uids,
line 398). Repeats the test's sequence -- launch the authorization service, use it, emergency stop, stop the
launcher, relaunch on the same run directory, verify the audit chain -- N times and prints the verifier's errors
when the chain does not verify. A false tamper alarm after an ordinary stop / relaunch would make the audit log's
alarm meaningless (IH-16).

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/g2harness/g10_audit_after_stop.py [N]
"""
import os
import shutil
import sys
import tempfile
import time

from _g2 import C, Report  # noqa: I001

R = Report("g10_audit_after_stop")
if os.geteuid() != 0:
    R.note("not root: skipped")
    R.done()

from cam_live.harness.auth.client import connect  # noqa: E402
from cam_live.harness.auth.launcher import launch  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
FAST = {"separated": True, "poll_s": 0.05, "restart_backoff_s": 0.05, "stop_grace_s": 2.0}   # as test_ea_privileged.py
bad = []
for i in range(N):
    run = tempfile.mkdtemp(prefix="g2hau", dir="/tmp")
    os.chmod(run, 0o755)
    run = os.path.join(run, "run")
    try:
        h = launch(run, ["auth"], dict(FAST))
        try:
            c = connect(run)
            c.put_secret("site_token", "tok_" + "Q" * 20, ["127.0.0.1"], [], "operator")
            t = C.TaskSpec(objective_id="obj-g10", objective="probe", mode="staging",
                           limits=C.Limits(duration_s=3600, max_decisions=100, max_external_actions=10))
            for _ in range(3):
                a = C.ActionSpec(action_id=C.new_id("a"), operation="wait", modality=C.Modality.INTERNAL,
                                 arguments={"seconds": 1})
                c.authorize(a, t, {})
            c.emergency_stop("drill", "operator")
            time.sleep(0.2)
        finally:
            h.stop()
        h2 = launch(run, ["auth"], dict(FAST))
        try:
            v = connect(run).verify_audit()
            if not v.get("ok"):
                bad.append({"round": i, "errors": v.get("errors"), "entries": v.get("entries")})
        finally:
            h2.stop()
    finally:
        shutil.rmtree(os.path.dirname(run), ignore_errors=True)
R.check(not bad, f"after an emergency stop, a launcher stop and a relaunch on the same run directory the audit chain "
                 f"verifies ({N} rounds)", f"failed rounds: {bad[:3]}")
R.done()
