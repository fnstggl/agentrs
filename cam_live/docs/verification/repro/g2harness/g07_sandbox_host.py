"""G2-harness probe g07 (root; real launcher, real service users): the root sandbox host of round 2
(cam_live.harness.sandbox_host, started by the launcher next to the environment service).

Launched exactly like g04 (``launch(R, ["auth", "exec"])``, the exec arguments of runtime/cli.py; the launcher adds
the sandbox host). Checks:
  H1  who may use the host socket: the learner (camlearn) and the sandbox uid (camsbx) are refused;
  H2  the browser uid talking to the host directly (no environment service, every check answered "allow" by
      itself): what runs, as which uid, and whether the host's own egress broker still refuses loopback /
      link-local destinations and the sandbox still cannot read the vault, the browser profile or /root;
  H3  the host's process context: cwd, sys.path[0], HOME -- and whether a file the LEARNER can write is imported
      by the ROOT host when it is (re)started. The run directory is the cwd of every service and is created
      group-writable for the learner's group (auth/launcher.py prepare_run_dir: mode 1771, group = learner gid);
      ``python -m`` puts the cwd first on sys.path. The planted module only appends "<euid> <argv>" to a marker
      file and then hands over to the real stdlib module, so the host keeps working.
The run directory is a short directory under /tmp (service uids must reach it); it is removed at exit.

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/g2harness/g07_sandbox_host.py
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time

from _g2 import C, SRC, Report  # noqa: I001

R = Report("g07_sandbox_host")
if os.geteuid() != 0:
    R.note("not root: skipped")
    R.done()

import grp  # noqa: E402
import pwd  # noqa: E402

from cam_live.harness import SOCK_AUTH, uid_of  # noqa: E402
from cam_live.harness.auth.launcher import LauncherConfig, launch  # noqa: E402


def run_as(user, code, *args, timeout=120):
    pw = pwd.getpwnam(user)
    return subprocess.run([sys.executable, "-c", code, *map(str, args)], user=pw.pw_uid, group=pw.pw_gid,
                          extra_groups=[grp.getgrnam("camlive").gr_gid], cwd="/", timeout=timeout,
                          env={"PATH": "/usr/bin:/bin", "PYTHONPATH": SRC, "HOME": "/nonexistent"},
                          capture_output=True, text=True)


def host_pid(run_dir, module="cam_live.harness.sandbox_host"):
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


CLIENT = textwrap.dedent(r"""
    import json, socket, sys
    from cam_live import contracts as C
    from cam_live.harness import rpc
    path, code, network = sys.argv[1], sys.argv[2], sys.argv[3]
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(120); s.connect(path)
    if code == "PING":
        rpc._send(s, {"op": "ping"}); print(json.dumps(rpc._recv(s))); sys.exit(0)
    a = C.ActionSpec(action_id=C.new_id("g2h"), operation="sandbox.python", modality=C.Modality.CODE,
                     arguments={"code": code, "network": network}, timeout_s=60.0)
    rpc._send(s, {"op": "execute", "action": rpc.to_wire(a), "deadline": 0.0, "authorization": None})
    checks, out = [], {}
    while True:
        m = rpc._recv(s)
        if m is None:
            break
        if "check" in m:                      # answer every check of the host with ALLOW ourselves
            op = rpc.from_wire(m["check"]); checks.append([op.op, (op.arguments or {}).get("url", "")])
            rpc._send(s, {"id": m["id"], "decision": rpc.to_wire(C.AuthorizationDecision(C.AuthVerdict.ALLOW))})
        elif "result" in m:
            r = rpc.from_wire(m["result"])
            out = {"status": r.status, "detail": str(r.failure_detail)[:200],
                   "text": " ".join(o.text for o in r.observations)[:600]}
            break
        elif m.get("error"):
            out = {"error": m["error"]}; break
    out["checks"] = checks
    print(json.dumps(out))
""")

D = tempfile.mkdtemp(prefix="g2hsh", dir="/tmp")
os.chmod(D, 0o755)
MARK_DIR = tempfile.mkdtemp(prefix="g2hmark", dir="/tmp")
os.chmod(MARK_DIR, 0o1777)
MARK = os.path.join(MARK_DIR, "imported_by")
sock = os.path.join(D, "sock")
lc = LauncherConfig(extra_args={"exec": ["--auth-socket", os.path.join(sock, SOCK_AUTH), "--provenance", "staging",
                                         "--run-dir", os.path.join(D, "exec")]})
handle = launch(D, ["auth", "exec"], lc)
planted = os.path.join(D, "argparse.py")
try:
    deadline = time.time() + 60
    while time.time() < deadline and not os.path.exists(os.path.join(sock, "sandbox.sock")):
        time.sleep(0.5)
    hsock = os.path.join(sock, "sandbox.sock")
    pid = host_pid(D)
    st = os.stat(hsock)
    R.note(f"sandbox host pid {pid} euid {os.stat(f'/proc/{pid}').st_uid if pid else None}; socket mode "
           f"{oct(st.st_mode & 0o777)} group {grp.getgrgid(st.st_gid).gr_name}; group members "
           f"{grp.getgrgid(st.st_gid).gr_mem}")

    # H1 -------------------------------------------------------------------------------------------------------
    for user in ("camlearn", "camsbx", "camverify"):
        cp = run_as(user, CLIENT, hsock, "PING", "none")
        out = (cp.stdout.strip().splitlines() or [""])[-1]
        R.check("may not use the sandbox host" in out or cp.returncode != 0,
                f"uid {user} is refused by the sandbox host", (out or cp.stderr[-200:])[:200])

    # H2 -------------------------------------------------------------------------------------------------------
    code = ("import os\nprint('UID', os.getuid(), os.geteuid())\n"
            "for p in ('/root', %r, %r, %r):\n"
            "    try:\n        os.listdir(p); print('READ', p)\n"
            "    except Exception as e:\n        print('NOREAD', p, type(e).__name__)\n"
            % (os.path.join(D, "vault"), os.path.join(D, "browser_profile"), os.path.join(D, "auth")))
    cp = run_as("cambrowser", CLIENT, hsock, code, "none")
    res = json.loads((cp.stdout.strip().splitlines() or ["{}"])[-1] or "{}") if cp.returncode == 0 else {"err": cp.stderr[-300:]}
    R.note(f"browser uid -> host directly, self-answered checks: {json.dumps(res)[:700]}")
    text = res.get("text", "")
    R.check(f"UID {uid_of('sandbox')}" in text and "READ /" not in text.replace("NOREAD", ""),
            "code sent by the browser uid without the environment still runs only as camsbx and cannot read /root, "
            "the vault, the auth directory or the browser profile", text[:300])
    R.note("the host verifies no grant: the browser uid can run programs as camsbx without the authorization "
           f"service (checks answered by the client itself: {res.get('checks', [])[:3]})")
    net = ("import socket\nfor h, p in (('127.0.0.1', 11111), ('169.254.169.254', 80), ('localhost', 8080)):\n"
           "    try:\n        socket.create_connection((h, p), timeout=3).close(); print('CONNECTED', h, p)\n"
           "    except Exception as e:\n        print('REFUSED', h, p, type(e).__name__, str(e)[:60])\n")
    cp = run_as("cambrowser", CLIENT, hsock, net, "scoped")
    res = json.loads((cp.stdout.strip().splitlines() or ["{}"])[-1] or "{}") if cp.returncode == 0 else {"err": cp.stderr[-300:]}
    R.check("CONNECTED" not in res.get("text", "") and res.get("text"),
            "with every check answered 'allow' by the client, the host's egress broker still refuses loopback, "
            "link-local and localhost destinations", res.get("text", json.dumps(res))[:300])

    # H3 -------------------------------------------------------------------------------------------------------
    cwd = os.readlink(f"/proc/{pid}/cwd") if pid else ""
    env = dict(x.split("=", 1) for x in open(f"/proc/{pid}/environ").read().split("\0") if "=" in x) if pid else {}
    rst = os.stat(D)
    R.note(f"host cwd {cwd}; HOME {env.get('HOME')}; run dir owner {rst.st_uid} group "
           f"{grp.getgrgid(rst.st_gid).gr_name} mode {oct(rst.st_mode & 0o7777)}")
    cp = run_as("camlearn", "import sys\nopen(sys.argv[1], 'w').write(sys.argv[2])\n", planted, textwrap.dedent(f"""
        import os, sys
        with open({MARK!r}, "a") as f:
            f.write(f"{{os.geteuid()}} {{' '.join(sys.argv[1:3])}}\\n")
        _here = os.path.dirname(os.path.abspath(__file__))
        sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
        del sys.modules["argparse"]
        import importlib as _il
        _real = _il.import_module("argparse")
        sys.modules["argparse"] = _real
        globals().update(_real.__dict__)
    """))
    R.note(f"the learner uid writes {planted}: exit {cp.returncode} {cp.stderr[-120:]!r}")
    if cp.returncode == 0 and pid:
        os.kill(pid, signal.SIGKILL)            # a crash of the host; the launcher's supervisor restarts it
        deadline = time.time() + 60
        while time.time() < deadline and not (os.path.exists(MARK) and open(MARK).read().strip()):
            time.sleep(0.5)
        time.sleep(1.0)
    marks = open(MARK).read().split("\n") if os.path.exists(MARK) else []
    root_runs = [m for m in marks if m.startswith("0 ")]
    R.check(not root_runs, "a module the learner uid writes into the run directory is never imported by the root "
                           "sandbox host", f"marker lines: {[m for m in marks if m]}")
    # H4: the same file and the authorization service (pre-existing at 4d6044a: every service starts with
    # cwd = run directory and imports argparse)
    auth_pid = host_pid(D, "cam_live.harness.auth.service")
    if auth_pid and os.path.exists(planted):
        os.kill(auth_pid, signal.SIGKILL)
        deadline = time.time() + 60
        while time.time() < deadline and not any(m.startswith(f"{uid_of('auth')} ") for m in open(MARK).read().split("\n")):
            time.sleep(0.5)
    time.sleep(3.0)
    R.note(f"authorization service pid {auth_pid} -> after the kill {host_pid(D, 'cam_live.harness.auth.service')} "
           f"(restarted by the launcher); non-root services cannot list the run directory (mode 1771, 'others' "
           f"have search only), so Python's path finder does not see files in their cwd")
    marks = open(MARK).read().split("\n") if os.path.exists(MARK) else []
    auth_runs = [m for m in marks if m.startswith(f"{uid_of('auth')} ")]
    R.check(not auth_runs, "a module the learner uid writes into the run directory is never imported by the "
                           "authorization service (camauth)", f"marker lines: {[m for m in marks if m]}")
    new_pid = host_pid(D)
    if new_pid:
        cp = run_as("cambrowser", CLIENT, hsock, "print('alive')", "none")
        R.note(f"restarted host pid {new_pid}: still serves -> {(cp.stdout.strip().splitlines() or [''])[-1][:120]}")
finally:
    if os.path.exists(planted):
        os.unlink(planted)
    try:
        handle.stop()
    except Exception as e:  # noqa: BLE001
        print("stop:", e)
    time.sleep(0.5)
    shutil.rmtree(D, ignore_errors=True)
    shutil.rmtree(MARK_DIR, ignore_errors=True)
R.done()
