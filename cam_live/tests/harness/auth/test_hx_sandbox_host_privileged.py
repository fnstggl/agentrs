"""CU-37 / CU-01 (G2-harness D2): in the separated deployment ``cam-live run`` starts (authorization service and
environment service under their own uids, the exec arguments of ``runtime/cli.py cmd_supervise``), generated code
runs -- in the sandbox host under the sandbox uid, not under the browser uid, and the environment service is still
not root. Run with ``-m privileged`` as root (real launcher, real service users).

Before the sandbox host existed, ``sandbox.python`` was unavailable there ("not root")."""
import json
import os
import shutil
import tempfile
import time

import pytest

from cam_live import contracts as C
from cam_live.harness import SOCK_AUTH, SOCK_EXEC, uid_of
from cam_live.harness.auth.client import connect
from cam_live.harness.auth.launcher import LauncherConfig, launch
from cam_live.harness.service import RemoteEnvironment
from ea_helpers import jout, run_as

pytestmark = pytest.mark.privileged

PING = r"""
import json, socket, struct, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    s.connect(sys.argv[1])
except OSError as e:
    print(json.dumps({"connect": type(e).__name__})); sys.exit(0)
data = json.dumps({"op": "ping"}).encode()
s.sendall(struct.pack(">I", len(data)) + data)
head = s.recv(4)
n = struct.unpack(">I", head)[0] if len(head) == 4 else 0
print(json.dumps({"answer": json.loads(s.recv(n).decode()) if n else None}))
"""


@pytest.fixture
def deployment(service_users):
    D = tempfile.mkdtemp(prefix="hxsbx", dir="/tmp")          # short: unix socket paths are limited
    os.chmod(D, 0o755)
    sock = os.path.join(D, "sock")
    lc = LauncherConfig(extra_args={"exec": ["--auth-socket", os.path.join(sock, SOCK_AUTH), "--provenance", "staging",
                                             "--run-dir", os.path.join(D, "exec")]})
    h = launch(D, ["auth", "exec"], lc)
    try:
        yield D, h
    finally:
        h.stop()
        time.sleep(0.3)
        shutil.rmtree(D, ignore_errors=True)


def run(env, auth, task, op, **args):
    a = C.ActionSpec(action_id=C.new_id("hx"), operation=op, modality=C.Modality.CODE, arguments=args, timeout_s=60.0)
    dec = auth.authorize(a, task, {})
    assert dec.verdict == C.AuthVerdict.ALLOW, dec.reasons
    res = env.execute(a, dec)
    return a, res, " ".join(o.text for o in res.observations)


def test_generated_code_runs_as_the_sandbox_uid_in_the_separated_deployment(deployment):
    D, h = deployment
    st = h.status()["services"]
    assert st["sandbox"]["state"] == "running" and st["sandbox"]["uid"] == 0, st["sandbox"]
    assert st["exec"]["uid"] == uid_of("browser") != 0
    auth = connect(D)
    task = C.TaskSpec(objective_id="obj-hx-sbx", objective="sandbox in the separated deployment", mode="staging",
                      interfaces=["internal", "code"],
                      limits=C.Limits(duration_s=3600, money=1, max_decisions=100, max_external_actions=100))
    auth.register_task(task, "operator")
    env = RemoteEnvironment(os.path.join(D, "sock", SOCK_EXEC), connect_timeout=60.0)
    exec_uid = os.stat(f"/proc/{env.ping()['pid']}").st_uid
    assert exec_uid == uid_of("browser")
    caps = env.get_capabilities()["executors"]["code"]
    assert caps["available"] is True and caps["isolation"]["uid"] == uid_of("sandbox"), caps
    assert caps["isolation"]["seccomp"] and caps["isolation"]["no_new_privs"] and caps["isolation"]["landlock_abi"] >= 1
    # a program runs, under the sandbox uid
    a, res, text = run(env, auth, task, "sandbox.python", code="import os; print('UID', os.getuid(), os.geteuid())")
    assert res.status == C.ResultStatus.OK, res.failure_detail
    assert f"UID {uid_of('sandbox')} {uid_of('sandbox')}" in text, text
    # its isolation holds: no socket, no reading outside the workspace, the workspace persists between actions
    _, res2, text2 = run(env, auth, task, "sandbox.python", code=(
        "import socket\ntry:\n    socket.socket(); print('SOCKET-OPENED')\nexcept OSError as e:\n    print('NOSOCK', e.errno)\n"
        "try:\n    open('/etc/shadow').read(1); print('SHADOW-READ')\nexcept OSError as e:\n    print('NOSHADOW')\n"
        "open('kept.txt', 'w').write('x')"))
    assert res2.status == C.ResultStatus.OK and "NOSOCK" in text2 and "NOSHADOW" in text2, text2
    assert "SOCKET-OPENED" not in text2 and "SHADOW-READ" not in text2
    _, res3, text3 = run(env, auth, task, "sandbox.list_files")
    assert res3.status == C.ResultStatus.OK and "kept.txt" in text3, text3
    # every primitive the sandbox host executes was checked by the authorization service against the running grant
    tail = json.dumps(auth.audit_tail(200), default=str)
    assert "sandbox.python" in tail and a.action_id not in ("",)
    # only the environment service's uid may use the sandbox host: the learner and the sandbox uid are refused
    path = os.path.join(D, "sock", "sandbox.sock")
    for role in ("learner", "sandbox"):
        out = jout(run_as({"learner": "camlearn", "sandbox": "camsbx"}[role], PING, path))
        refused = out.get("connect") or "may not use" in json.dumps(out.get("answer"))
        assert refused, (role, out)
    # a denied program is not run: the host asks the environment, which asks the authorization service
    stopped = C.ActionSpec(action_id=C.new_id("hx"), operation="sandbox.python", modality=C.Modality.CODE,
                           arguments={"code": "print('RAN')"}, timeout_s=30.0)
    res4 = env.execute(stopped, C.AuthorizationDecision(C.AuthVerdict.ALLOW, grant="forged"))
    assert res4.status != C.ResultStatus.OK and "RAN" not in " ".join(o.text for o in res4.observations)


PLANT = r"""
import os, sys
with open(MARK, "a") as f:
    f.write(f"{os.geteuid()} {' '.join(sys.argv[1:3])}\n")
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
del sys.modules["argparse"]
import importlib as _il
_real = _il.import_module("argparse")
sys.modules["argparse"] = _real
globals().update(_real.__dict__)
"""


def _pid_of(run_dir, module):
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            cmd = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if module.encode() in cmd and run_dir.encode() in b" ".join(cmd) and b"-c" not in cmd[:3]:
            return int(p)
    return None


def test_root_services_never_import_what_a_service_uid_can_write(deployment):
    """G2-harness R2-C1: a module the learner uid plants in the run directory is not imported by the root sandbox
    host (or any service) when it restarts; root services run isolated (-I), with cwd /, HOME /root, a system PATH,
    and nothing on their import path is writable by a service uid. Fails on 0a30cd8 (the host imported it as root)."""
    import stat
    import pwd
    import grp
    import subprocess
    import sys
    from cam_live.harness import SERVICE_USERS
    D, h = deployment
    mark_dir = tempfile.mkdtemp(prefix="hxmark", dir="/tmp")
    os.chmod(mark_dir, 0o1777)
    mark = os.path.join(mark_dir, "imported_by")
    try:
        assert _pid_of(D, "cam_live.harness.sandbox_host"), "sandbox host not found"
        # the learner plants argparse.py in the run directory (its group may create files there) ...
        planted = os.path.join(D, "argparse.py")
        cp = run_as("camlearn", "import sys\nopen(sys.argv[1], 'w').write(sys.argv[2])\n", planted,
                    f"MARK = {mark!r}\n" + PLANT)
        assert cp.returncode == 0, cp.stderr
        # ... and the host and the authorization service crash and are restarted by the supervisor
        for module, name in (("cam_live.harness.sandbox_host", "sandbox"), ("cam_live.harness.auth.service", "auth")):
            old = _pid_of(D, module)
            os.kill(old, 9)
            deadline = time.time() + 60
            while time.time() < deadline and (_pid_of(D, module) in (None, old)
                                              or h.status()["services"][name]["state"] != "running"):
                time.sleep(0.3)
            assert _pid_of(D, module) not in (None, old), module
        time.sleep(2.0)
        marks = open(mark).read().split("\n") if os.path.exists(mark) else []
        assert not [m for m in marks if m.strip()], marks          # nothing imported it, as root or otherwise
        # the root host's process context
        pid = _pid_of(D, "cam_live.harness.sandbox_host")
        assert os.stat(f"/proc/{pid}").st_uid == 0
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        env = dict(x.split("=", 1) for x in open(f"/proc/{pid}/environ").read().split("\0") if "=" in x)
        from cam_live.harness.auth.launcher import ROOT_PATH, _BOOT_PATH, _src_root
        assert os.readlink(f"/proc/{pid}/cwd") == "/" and b"-I" in cmd[:3], cmd[:4]
        assert env.get("HOME") in ("/root", "/") and env.get("PATH") == ROOT_PATH
        # nothing on the import path of a launched root service is writable by a service uid
        probe = os.path.join(mark_dir, "path_probe.py")
        with open(probe, "w") as f:
            f.write("import sys, json; print(json.dumps(sys.path))\n")
        out = subprocess.run([sys.executable, "-I", _BOOT_PATH, _src_root(), probe], cwd="/",
                             env={"PATH": ROOT_PATH, "PYTHONPATH": D}, capture_output=True, text=True, timeout=60)
        paths = json.loads(out.stdout.strip().splitlines()[-1])
        assert D not in paths and "" not in paths and mark_dir not in paths, paths
        uids = {pwd.getpwnam(u).pw_uid for u in SERVICE_USERS.values()}
        gids = {pwd.getpwnam(u).pw_gid for u in SERVICE_USERS.values()} | {
            g.gr_gid for g in grp.getgrall() if set(g.gr_mem) & set(SERVICE_USERS.values())}
        for p in paths:
            cur = p
            while cur and cur != "/" and os.path.exists(cur):
                st = os.stat(cur)
                writable = st.st_uid in uids or (st.st_mode & stat.S_IWGRP and st.st_gid in gids) or \
                    (st.st_mode & stat.S_IWOTH and not st.st_mode & stat.S_ISVTX)
                assert not writable, (p, cur, oct(st.st_mode), st.st_uid, st.st_gid)
                cur = os.path.dirname(cur)
    finally:
        shutil.rmtree(mark_dir, ignore_errors=True)
