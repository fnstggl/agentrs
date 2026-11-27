"""G-env probe 7: the separated deployment (what `cam-live run` starts): authorization service + environment service
under their own uids, driven over AuthRPC / ExecRPC. Needs root and the service users.

    cd /workspace/camlive && .venv/bin/python cam_live/docs/verification/repro/env/probe_separated.py

The run directory must be reachable by the service uids and unix-socket paths are limited to ~107 bytes, so this
probe uses a short directory under /tmp (removed at the end) instead of the session scratch directory.
"""
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "verification", "repro", "env"))
import shutil
import sys
import tempfile
import time

from g_common import C, CAM, action, brief, report, staging_task, summary

if os.geteuid() != 0:
    report("SEP", "NOTE", "not root: skipped")
    sys.exit(summary("probe_separated"))

from cam_live.harness import SOCK_AUTH, SOCK_EXEC, uid_of  # noqa: E402
from cam_live.harness.auth.client import connect  # noqa: E402
from cam_live.harness.auth.launcher import LauncherConfig, launch  # noqa: E402
from cam_live.harness.service import RemoteEnvironment  # noqa: E402
from cam_live.harness.staging import StagingSites  # noqa: E402

R = tempfile.mkdtemp(prefix="genv", dir="/tmp")
os.chmod(R, 0o755)
sites = StagingSites().start()
s = sites.urls["structured"]
sock = os.path.join(R, "sock")
# exactly the arguments runtime/cli.py cmd_supervise gives the exec service
lc = LauncherConfig(extra_args={"exec": ["--auth-socket", os.path.join(sock, SOCK_AUTH), "--provenance", "staging",
                                         "--run-dir", os.path.join(R, "exec")]})
handle = launch(R, ["auth", "exec"], lc)
try:
    auth = connect(R)
    task = staging_task("g_env_separated")
    auth.register_task(task, "operator")
    auth.register_local_service(s.split("//", 1)[1].split("/", 1)[0], "operator", "E-fix variant: the staging site this probe started")  # the ONLY change
    env = RemoteEnvironment(os.path.join(sock, SOCK_EXEC), connect_timeout=60.0)
    caps = env.get_capabilities()
    import subprocess
    pid = env.ping()["pid"]
    ex_uid = os.stat(f"/proc/{pid}").st_uid
    cmdline = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")
    report("SEP-UIDS", "HOLDS" if ex_uid == uid_of("browser") else "DEFECT",
           "the environment service runs under the browser uid, not root and not the learner's",
           exec_uid=ex_uid, browser_uid=uid_of("browser"), learner_uid=uid_of("learner"),
           modalities=caps.get("modalities"), tools=caps.get("tools"), secret_scrubbing=caps["features"]["secret_scrubbing"],
           has_secrets_option="--secrets" in cmdline, has_executor_option="--executor" in cmdline)
    report("SEP-COMPOSITION", "DEFECT" if (not caps["features"]["secret_scrubbing"] or sorted(caps.get("modalities", [])) == ["internal", "structured_browser"]) else "HOLDS",
           "what the launched environment service contains: started the way `cam-live run` starts it, it registers only "
           "the internal and structured_browser executors (no code, visual, api, skill) and is given no secret broker "
           "handle, so observations and the execution log are NOT scrubbed and no executor can obtain a secret",
           modalities=caps.get("modalities"), secret_scrubbing=caps["features"]["secret_scrubbing"],
           discovered_tools=sorted(t.name for t in env.discover_tools()))
    a = action("browser.navigate", C.Modality.STRUCTURED_BROWSER, url=s + "/items")
    dec = auth.authorize(a, task, {})
    res = env.execute(a, dec)
    bcap = env.get_capabilities()["executors"].get("structured_browser", {})
    report("SEP-BROWSER", "HOLDS" if res.status == "ok" else "DEFECT",
           "a browser.navigate to the staging site through the separated services (authorize over AuthRPC, execute "
           "over ExecRPC)", authorization=dec.verdict, result=brief(res)[:400],
           uid_separation=bcap.get("uid_separation"), chromium_version=bcap.get("version"))
    w = action("wait", C.Modality.INTERNAL, seconds=30)
    rw = env.execute(w, auth.authorize(w, task, {}))
    logp = os.path.join(R, "exec", "events.jsonl")
    st = os.stat(logp) if os.path.exists(logp) else None
    report("SEP-INTERNAL", "HOLDS" if rw.status == "ok" else "DEFECT",
           "an internal action through the separated services; the execution log belongs to the executor uid",
           result=brief(rw)[:60], log_owner=st.st_uid if st else None, log_mode=oct(st.st_mode & 0o777) if st else None)
finally:
    try:
        handle.stop()
    except Exception as e:  # noqa: BLE001
        print("stop:", e)
    sites.stop()
    time.sleep(0.5)
    shutil.rmtree(R, ignore_errors=True)
sys.exit(summary("probe_separated"))
