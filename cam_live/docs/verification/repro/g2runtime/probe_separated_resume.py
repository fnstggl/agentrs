"""G2-runtime probe: `cam-live resume` of a separated run after the whole run was killed (root; CM-09, LX-16).

    .venv/bin/python cam_live/docs/verification/repro/g2runtime/probe_separated_resume.py      (as root)

F's status: "a real supervisor restart of a paused separated run was not exercised" (unit-tested with a stubbed spawn).
Here: a separated staging run is started through `cam-live run` (utility_staging task, G-rt's stand-in learner from
repro/rt/g_factory.py, no browser — as G-rt p11 does), the operator pauses it, then every process of the run
(supervisor, executor, authorization service, verifier, learner) is SIGKILLed, then `cam-live resume` is called.
Run directory next to this file (.sep_<pid>, removed at the end): the service uids cannot reach the session scratchpad.
"""
import json
import os
import pwd
import shutil
import signal
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.join(os.path.dirname(HERE), "rt")
sys.path.insert(0, HERE)
sys.path.insert(0, RT)
from g2common import PROJECT, SRC, Report  # noqa: E402

from cam_live.contracts import RunFiles  # noqa: E402
from cam_live.harness import SERVICE_USERS, users_exist  # noqa: E402
from cam_live.runtime import tasks as T  # noqa: E402
from cam_live.runtime.control import pid_alive, read_pid  # noqa: E402

R = Report("probe_separated_resume")
if os.geteuid() != 0 or not users_exist():
    R.note("skipped", "needs root and the service users")
    sys.exit(R.done())
base = os.path.join(HERE, f".sep_{os.getpid()}")
inp, run_dir = os.path.join(base, "in"), os.path.join(base, "run")
os.makedirs(inp)
os.chmod(base, 0o755)
os.chmod(inp, 0o755)
ENV = dict(os.environ, CAM_LIVE_HOME=os.path.join(base, "home"), CAM_LIVE_GPU_LOCK=os.path.join(base, "gpu.lock"),
           PYTHONPATH=os.pathsep.join([SRC, RT]))


def cli(*args, timeout=240):
    p = subprocess.run([sys.executable, "-m", "cam_live.runtime.cli", *args], capture_output=True, text=True, env=ENV,
                       timeout=timeout, cwd="/workspace/camlive")
    return p.returncode, p.stdout, p.stderr


def mine():
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit() and int(d) != os.getpid():
            try:
                if base.encode() in open(f"/proc/{d}/cmdline", "rb").read():
                    out.append(int(d))
            except OSError:
                pass
    return out


def st():
    try:
        return json.load(open(os.path.join(run_dir, RunFiles.STATE)))
    except (OSError, ValueError):
        return {}


def wait(pred, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.2)
    return False


def uid_of_pid(pid):
    return int([ln for ln in open(f"/proc/{pid}/status") if ln.startswith("Uid:")][0].split()[1])


def events(t=None):
    try:
        return [json.loads(x) for x in open(os.path.join(run_dir, RunFiles.EVENTS)) if x.strip()
                and (t is None or f'"{t}"' in x)]
    except OSError:
        return []


try:
    raw = T.load_task_raw(os.path.join(PROJECT, "tasks", "utility_staging.yaml"))
    json.dump({"utility.distinct_successful_sessions": 0}, open(os.path.join(inp, "metrics.json"), "w"))
    raw["success"]["evidence_source"] = "json:" + os.path.join(inp, "metrics.json")
    raw["resources"]["staging_site"] = "http://127.0.0.1:9"
    raw["limits"]["max_decisions"] = 500
    task_path, cfg_path = os.path.join(inp, "task.yaml"), os.path.join(inp, "cfg.yaml")
    yaml.safe_dump(raw, open(task_path, "w"))
    yaml.safe_dump({"seed": 3, "home": os.path.join(base, "home"), "backbone": {"name": "tiny", "revision": ""},
                    "runtime": {"control_poll_s": 0.2, "min_idle_sleep_s": 0.0},
                    "g_standin": {"script": [["internal.note", {"text": "g2 stand-in note"}], "wait"], "wait_seconds": 5.0},
                    "launcher": {"pythonpath": [RT], "extra_args": {"exec": ["--no-browser"]}}}, open(cfg_path, "w"))
    common = ("--config", cfg_path, "--run-dir", run_dir)
    code, out, err = cli("run", "--task", task_path, *common, "--agent-factory", "g_factory:build_agent", "--no-doctor",
                         "--start-timeout", "120", "--json")
    info = json.loads(out) if code == 0 else {}
    R.check("0 separated staging run started through `cam-live run`", code == 0 and info.get("separated"), err[-400:])
    if code != 0:
        raise SystemExit
    sup1, pid1 = info["launcher_pid"], info["runtime_pid"] or read_pid(run_dir)
    wait(lambda: (st().get("counters") or {}).get("decisions", 0) >= 3)
    code, out, err = cli("pause", *common, "--json")
    d_paused = (st().get("counters") or {}).get("decisions", 0)
    R.check("1 operator pause acknowledged", code == 0 and st().get("status") == "paused", err[-200:])
    procs = mine()
    for p in procs:
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass
    wait(lambda: not mine(), 10)
    R.check("2 every process of the run is gone after SIGKILL", not mine() and not pid_alive(sup1) and not pid_alive(pid1),
            {"killed": len(procs), "left": mine()})
    n_resumed = len(events("run_resumed"))
    code, out, err = cli("resume", *common, "--json", "--timeout", "120")
    res = json.loads(out) if out.strip().startswith("{") else {}
    sup2 = res.get("supervisor_pid")
    recorded = None
    try:
        recorded = int(open(os.path.join(run_dir, "supervisor.pid")).read().strip())
    except (OSError, ValueError):
        pass
    R.check("3 `cam-live resume` restarts the run: exit 0, a new supervisor whose pid is recorded and alive",
            code == 0 and sup2 and recorded == sup2 and pid_alive(sup2) and sup2 != sup1,
            {"exit": code, "supervisor_pid": sup2, "recorded": recorded, "err": err[-600:]})
    ok = wait(lambda: len(events("run_resumed")) > n_resumed and pid_alive(read_pid(run_dir)), 60)
    pid2 = read_pid(run_dir)
    R.check("4 the learner process is back, under the learner uid, and restored the persisted state",
            ok and pid2 != pid1 and uid_of_pid(pid2) == pwd.getpwnam(SERVICE_USERS["learner"]).pw_uid,
            {"run_resumed": events("run_resumed")[-1]["data"] if events("run_resumed") else None})
    R.check("5 the pending signed resume is applied: the run continues and decides again",
            wait(lambda: st().get("status") == "running" and (st().get("counters") or {}).get("decisions", 0) > d_paused, 60),
            {"status": st().get("status"), "decisions": (st().get("counters") or {}).get("decisions"), "before": d_paused})
    ar = [e for e in events("action_result")]
    ids = [e["data"]["action_id"] for e in ar]
    R.check("6 no action has two action_result records across the restart", len(ids) == len(set(ids)),
            {"results": len(ids), "distinct": len(set(ids))})
    code, out, err = cli("stop", *common, "--json")
    R.check("7 operator stop after the restart", code == 0 and wait(lambda: st().get("status") == "stopped", 30), err[-200:])
except SystemExit:
    pass
finally:
    for sig, wait_s in ((signal.SIGTERM, 8.0), (signal.SIGKILL, 2.0)):
        for p in mine():
            try:
                os.kill(p, sig)
            except OSError:
                pass
        end = time.time() + wait_s
        while time.time() < end and mine():
            time.sleep(0.2)
    shutil.rmtree(base, ignore_errors=True)
sys.exit(R.done())
