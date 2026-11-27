"""The separated-uid mode end to end through ``cam-live run`` (needs root and the service users; run with -m privileged).

Every service runs under its own uid: the authorization service, the computer environment (no browser here), the
independent verifier, and the learner process running LiveRuntime with a small real test agent.
"""
import json
import os
import pwd
import shutil
import signal
import stat
import subprocess
import sys
import time

import pytest
import yaml

from cam_live.contracts import RunFiles
from cam_live.harness import SERVICE_USERS, users_exist
from cam_live.runtime import tasks as T
from cam_live.runtime.control import pid_alive, read_pid
from cam_live.runtime.logs import RunReader, read_json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
pytestmark = pytest.mark.privileged


def _cli(*args, timeout=180):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(HERE, "..", "..", "src"), HERE])
    p = subprocess.run([sys.executable, "-m", "cam_live.runtime.cli", *args], capture_output=True, text=True, env=env,
                       timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def _uid_of_pid(pid):
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("Uid:"):
                return int(line.split()[1])


def _wait(pred, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def test_separated_run_through_the_cli():
    if os.geteuid() != 0 or not users_exist():
        pytest.skip("needs root and the service users")
    pytest.importorskip("cam_live.harness.auth.launcher")
    base = os.path.join(REPO, "runs", f"f_privileged_{os.getpid()}")
    inp, run_dir = os.path.join(base, "in"), os.path.join(base, "run")
    os.makedirs(inp)
    for d in (os.path.join(REPO, "runs"), base, inp):
        os.chmod(d, 0o755)
    try:
        raw = T.load_task_raw(os.path.join(HERE, "..", "..", "tasks", "utility_staging.yaml"))
        metrics = os.path.join(inp, "metrics.json")
        json.dump({"utility.distinct_successful_sessions": 0}, open(metrics, "w"))
        raw["success"]["evidence_source"] = "json:" + metrics
        raw["resources"]["staging_site"] = "http://127.0.0.1:9"
        raw["limits"]["max_decisions"] = 8
        task_path, cfg_path = os.path.join(inp, "task.yaml"), os.path.join(inp, "cfg.yaml")
        yaml.safe_dump(raw, open(task_path, "w"))
        yaml.safe_dump({"seed": 3, "home": os.path.join(base, "home"), "backbone": {"name": "tiny", "revision": ""},
                        "runtime": {"control_poll_s": 0.2, "min_idle_sleep_s": 0.0},
                        "lab_agent": {"only_ops": ["internal.inspect", "internal.note", "wait"], "wait_seconds": 60.0},
                        "launcher": {"pythonpath": [HERE], "extra_args": {"exec": ["--no-browser"]}}}, open(cfg_path, "w"))
        common = ("--config", cfg_path, "--run-dir", run_dir)
        code, out, err = _cli("run", "--task", task_path, *common, "--agent-factory", "rt_lab_agent:build_agent",
                              "--no-doctor", "--start-timeout", "120", "--json")
        assert code == 0, err + open(os.path.join(run_dir, "logs", "supervisor.log")).read()[-2000:]
        info = json.loads(out)
        assert info["separated"] is True and info["started"]
        uid = {role: pwd.getpwnam(user).pw_uid for role, user in SERVICE_USERS.items()}
        pid = info["runtime_pid"] or read_pid(run_dir)
        assert _uid_of_pid(pid) == uid["learner"] != 0                    # the learner process is not root
        assert _wait(lambda: (read_json(os.path.join(run_dir, RunFiles.STATE), {}) or {}).get("status") == "exhausted")
        # ownership: the learner owns its records; the verifier alone owns its key and log
        assert os.stat(os.path.join(run_dir, RunFiles.STATE)).st_uid == uid["learner"]
        key = os.stat(os.path.join(run_dir, "verifier", "verifier.key"))
        vlog = os.stat(os.path.join(run_dir, RunFiles.VERIFIER))
        assert key.st_uid == vlog.st_uid == uid["verifier"]
        assert stat.S_IMODE(key.st_mode) == 0o600 and stat.S_IMODE(vlog.st_mode) & 0o077 == 0
        assert stat.S_IMODE(os.stat(os.path.join(run_dir, "verifier")).st_mode) == 0o700
        r = RunReader(run_dir)
        st = r.state()
        assert st["counters"]["decisions"] == 8 and st["committed"]["ckpt"]["id"]
        results = [e["data"] for e in r.records(RunFiles.EVENTS, types=["action_result"])]
        assert any(d["kind"] == "execute" and d["status"] == "ok" for d in results)      # executed by the executor uid
        auths = [e["data"]["verdict"] for e in r.records(RunFiles.EVENTS, types=["authorization"])]
        assert auths and set(auths) <= {"allow", "deny", "needs_approval"} and "allow" in auths
        assert r.records(RunFiles.EVENTS, types=["verified_outcome"]) and st["refused_verified"] == 0
        assert not r.records(RunFiles.EVENTS, types=["error"])
        pre = read_json(os.path.join(run_dir, "preflight.json"))
        assert pre["live"]["blocked"] is True                              # a staging run is never live-ready
        # stop through the CLI: the runtime checkpoints and every service ends
        code, out, err = _cli("stop", *common, "--reason", "test end", "--json")
        assert code == 0 and json.loads(out)["result"]["status"] == "stopped", err
        sup = int(open(os.path.join(run_dir, "supervisor.pid")).read())
        assert _wait(lambda: not pid_alive(pid) and not pid_alive(sup), 60)
    finally:
        for name in ("supervisor.pid", "runtime.pid"):
            try:
                p = int(open(os.path.join(run_dir, name)).read())
                if pid_alive(p):
                    os.kill(p, signal.SIGTERM)
            except (OSError, ValueError):
                pass
        time.sleep(1.0)
        shutil.rmtree(base, ignore_errors=True)
