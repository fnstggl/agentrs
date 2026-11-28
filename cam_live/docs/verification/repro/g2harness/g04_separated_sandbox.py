"""G2-harness probe g04 (root; real launcher, real service users): in the separated deployment that `cam-live run`
starts (authorization service + environment service under their own uids, exactly the exec arguments of
runtime/cli.py cmd_supervise), does generated code run in its own sandbox uid, separate from the browser uid?
Spec 12A.14 "Separate the learner, browser execution process, generated-code sandbox, credential service and
independent evaluator"; requirement CU-37.

The run directory is a short directory under /tmp (service uids must reach it; unix-socket paths are limited to
~107 bytes); it is removed at exit.

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/g2harness/g04_separated_sandbox.py
"""
import json
import os
import shutil
import sys
import tempfile
import time

from _g2 import C, Report  # noqa: I001

R = Report("g04_separated_sandbox")
if os.geteuid() != 0:
    R.note("not root: skipped")
    R.done()

from cam_live.harness import SOCK_AUTH, SOCK_EXEC, uid_of  # noqa: E402
from cam_live.harness.auth.client import connect  # noqa: E402
from cam_live.harness.auth.launcher import LauncherConfig, launch  # noqa: E402
from cam_live.harness.service import RemoteEnvironment  # noqa: E402

D = tempfile.mkdtemp(prefix="g2hsep", dir="/tmp")
os.chmod(D, 0o755)
sock = os.path.join(D, "sock")
lc = LauncherConfig(extra_args={"exec": ["--auth-socket", os.path.join(sock, SOCK_AUTH), "--provenance", "staging",
                                         "--run-dir", os.path.join(D, "exec")]})
handle = launch(D, ["auth", "exec"], lc)
try:
    auth = connect(D)
    task = C.TaskSpec(objective_id="obj-g2h-sep", objective="G2-harness probe (no strategy)", mode="staging",
                      interfaces=["internal", "code", "structured_browser"],
                      limits=C.Limits(duration_s=3600, money=1, max_decisions=100, max_external_actions=100))
    auth.register_task(task, "operator")
    env = RemoteEnvironment(os.path.join(sock, SOCK_EXEC), connect_timeout=60.0)
    caps = env.get_capabilities()
    pid = env.ping()["pid"]
    exec_uid = os.stat(f"/proc/{pid}").st_uid
    code_cap = caps.get("executors", {}).get("code", {})
    R.note(f"environment service uid {exec_uid} (browser uid {uid_of('browser')}, sandbox uid {uid_of('sandbox')}); "
           f"code executor capability: {json.dumps(code_cap)[:600]}")
    a = C.ActionSpec(action_id=C.new_id("g2h"), operation="sandbox.python", modality=C.Modality.CODE,
                     arguments={"code": "import os; print('UID', os.getuid(), os.geteuid())"}, timeout_s=60.0)
    dec = auth.authorize(a, task, {})
    res = env.execute(a, dec)
    text = " ".join(o.text for o in res.observations)[:400]
    R.note(f"sandbox.python: authorization {dec.verdict}; result {res.status} {res.failure_class} "
           f"{str(res.failure_detail)[:300]!r}; output {text!r}")
    ran_uid = None
    for tok in text.split("UID", 1)[1:]:
        try:
            ran_uid = int(tok.split()[0])
        except (ValueError, IndexError):
            pass
    R.check(res.status == "ok" and ran_uid == uid_of("sandbox"),
            "in the deployment cam-live run starts, generated code runs under the dedicated sandbox uid (camsbx), "
            "separate from the browser uid", f"status {res.status}, code ran as uid {ran_uid}")
finally:
    try:
        handle.stop()
    except Exception as e:  # noqa: BLE001
        print("stop:", e)
    time.sleep(0.5)
    shutil.rmtree(D, ignore_errors=True)
R.done()
