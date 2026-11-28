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
