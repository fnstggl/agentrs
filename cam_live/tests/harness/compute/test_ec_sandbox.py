"""Sandboxed code execution: Python, shell, every limit, filesystem scope, network, cleanup, authorization."""
import http.server
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import replace

import pytest

from cam_live import contracts as c
from cam_live import harness
from cam_live.harness import sandbox as sbx


def lim(**kw):
    return replace(sbx.SandboxLimits(), **kw)


def data(result):
    assert result.observations, result
    return result.observations[0].data


# ---------------------------------------------------------------------------------------------------------- basics
def test_embedded_programs_compile():
    compile(sbx._BOOT, "sandbox-boot", "exec")
    compile(sbx._SHIM, "sandbox-shim", "exec")


def test_python_runs_as_unprivileged_user_in_persistent_workspace(sandbox):
    sb = sandbox()
    r = sb.run_python("import os, sys\nopen('state.txt', 'w').write('41')\nprint(os.getuid(), os.getcwd())\n"
                      "import numpy\nprint(int(numpy.arange(4).sum()))")
    assert r.ok, r.stderr
    uid, cwd = r.stdout.split()[:2]
    assert int(uid) != 0 and int(uid) == sb.uid
    assert cwd == sb.workspace
    assert r.stdout.split()[2] == "6"
    assert r.isolation == {"uid": sb.uid, "landlock_abi": sb.capabilities()["isolation"]["landlock_abi"],
                           "seccomp": True, "no_new_privs": True}
    r2 = sb.run_python("print(int(open('state.txt').read()) + 1)")          # a new process sees the same workspace
    assert r2.ok and r2.stdout.strip() == "42"
    again = sbx.Sandbox(sb.workspace, sb.cfg)                                 # and so does a new Sandbox object
    try:
        assert again.read_file("state.txt")["data"] == b"41"
    finally:
        again.close()


def test_shell_runs_pipelines_and_finds_python(sandbox):
    sb = sandbox()
    r = sb.run_shell("set -e\nprintf 'a\\nb\\nc\\n' > lines.txt\nwc -l < lines.txt\npython -c 'print(6 * 7)'\n"
                     "id -u\necho err >&2\nexit 3")
    assert r.exit_code == 3
    out = r.stdout.split()
    assert out[0] == "3" and out[1] == "42" and int(out[2]) == sb.uid
    assert r.stderr.strip() == "err"


def test_stdin_and_arguments(sandbox):
    sb = sandbox()
    argv, _ = sb.python_argv("import sys\nprint(sys.argv[1:], sys.stdin.read().upper())", ["x", "y z"])
    r = sb.run(argv, stdin=b"hello")
    assert r.ok and r.stdout.strip() == "['x', 'y z'] HELLO"


# ---------------------------------------------------------------------------------------------------------- limits
def test_cpu_limit_enforced(sandbox):
    sb = sandbox()
    t0 = time.monotonic()
    r = sb.run_python("while True:\n    pass", limits=lim(cpu_s=1, wall_s=30))
    assert r.signal == signal.SIGXCPU and r.limit_hit == "cpu"
    assert 0.8 <= r.cpu_s <= 3.5 and time.monotonic() - t0 < 10


def test_memory_limit_enforced(sandbox):
    sb = sandbox()
    r = sb.run_python("b = b'x' * (1 << 30)\nprint('allocated', len(b))", limits=lim(mem_mb=256))
    assert r.exit_code not in (0, None) and "allocated" not in r.stdout
    assert "MemoryError" in r.stderr and r.limit_hit == "memory"
    assert r.max_rss_kb < 256 * 1024
    ok = sb.run_python("b = b'x' * (32 << 20)\nprint('allocated', len(b))", limits=lim(mem_mb=256))
    assert ok.ok and "allocated 33554432" in ok.stdout


def test_total_memory_of_the_process_tree_enforced(sandbox):
    sb = sandbox()
    code = ("import subprocess, sys, time\n"
            "kids = [subprocess.Popen([sys.executable, '-c', \"b = b'x' * (90 << 20); import time; time.sleep(60)\"])"
            " for _ in range(4)]\ntime.sleep(60)\nprint('survived')")
    t0 = time.monotonic()
    r = sb.run_python(code, limits=lim(mem_mb=512, mem_total_mb=200, wall_s=40))
    assert r.killed_for == "memory_total" and r.limit_hit == "memory_total"
    assert "survived" not in r.stdout and r.leftover == 0
    assert r.peak_tree_rss_kb > 200 * 1024 and time.monotonic() - t0 < 20


def test_process_limit_enforced(sandbox, procs_matching):
    sb = sandbox()
    marker = f"bomb-{uuid.uuid4().hex}"
    code = (f"import subprocess, sys, time\n"
            f"kids = []\n"
            f"for i in range(400):\n"
            f"    try:\n"
            f"        kids.append(subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)', '{marker}']))\n"
            f"    except OSError as e:\n"
            f"        print('fork failed', e.errno)\n"
            f"        time.sleep(0.2)\n"
            f"print('spawned', len(kids))\ntime.sleep(120)")
    t0 = time.monotonic()
    r = sb.run_python(code, limits=lim(nproc=24, wall_s=40))
    assert r.killed_for == "nproc" and r.limit_hit == "nproc"
    assert r.peak_tasks > 24 and r.leftover == 0
    assert time.monotonic() - t0 < 25
    assert procs_matching(marker) == []


def test_kernel_task_ceiling_stops_a_fork_bomb_the_monitor_does_not_see(sandbox):
    """With the per-job watchdog set out of reach, RLIMIT_NPROC of the sandbox uid is what stops the forks."""
    sb = sandbox(uid_nproc=40)
    code = ("import os, time\nn = 0\ntry:\n    for i in range(2000):\n        if os.fork() == 0:\n"
            "            time.sleep(30)\n            os._exit(0)\n        n += 1\nexcept OSError as e:\n"
            "    print('EAGAIN' if e.errno == 11 else e.errno, n)\n")
    r = sb.run_python(code, limits=lim(nproc=100000, wall_s=30))
    assert r.exit_code == 0, r.stderr
    word, n = r.stdout.split()
    assert word == "EAGAIN" and 0 < int(n) <= 40 and r.leftover == 0


def test_file_size_limit_enforced(sandbox):
    sb = sandbox()
    r = sb.run_python("with open('big.bin', 'wb') as f:\n    for _ in range(64):\n        f.write(b'x' * (1 << 20))\n"
                      "print('wrote all')", limits=lim(fsize_mb=2))
    assert r.exit_code not in (0, None) and "wrote all" not in r.stdout and r.limit_hit == "fsize"
    assert sb.read_file("big.bin", max_bytes=1)["size"] <= 2 << 20
    r2 = sb.run_shell("dd if=/dev/zero of=big2.bin bs=1M count=64 2>/dev/null; echo rc=$?", limits=lim(fsize_mb=2))
    assert "rc=153" in r2.stdout                                    # dd died of SIGXFSZ
    assert sb.read_file("big2.bin", max_bytes=1)["size"] <= 2 << 20


def test_open_file_limit_enforced(sandbox):
    sb = sandbox()
    r = sb.run_python("fs = []\ntry:\n    for i in range(5000):\n        fs.append(open('/dev/null'))\n"
                      "except OSError as e:\n    print(e.errno, len(fs))", limits=lim(nofile=64))
    assert r.ok
    err, n = map(int, r.stdout.split())
    assert err == 24 and n < 64


def test_wall_clock_timeout_kills_the_whole_process_group(sandbox, procs_matching):
    sb = sandbox()
    marker = f"straggler-{uuid.uuid4().hex}"
    code = (f"import subprocess, sys, time\n"
            f"for _ in range(3):\n"
            f"    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)', '{marker}'])\n"
            f"subprocess.Popen(['sh', '-c', 'sleep 300 # {marker}'])\n"
            f"print('started', flush=True)\ntime.sleep(300)")
    t0 = time.monotonic()
    r = sb.run_python(code, limits=lim(wall_s=1.5))
    took = time.monotonic() - t0
    assert r.timed_out and r.killed_for == "timeout" and r.exit_code is None and r.signal == signal.SIGKILL
    assert "started" in r.stdout and 1.4 <= took < 6
    assert r.leftover == 0 and procs_matching(marker) == []


def test_nothing_outlives_a_program_that_exits_normally(sandbox, procs_matching):
    sb = sandbox()
    marker = f"daemon-{uuid.uuid4().hex}"
    r = sb.run_shell(f"(exec -a '{marker}' sleep 300 &)\nnohup python -c 'import time; time.sleep(300)' {marker} "
                     f">/dev/null 2>&1 &\nsleep 0.3\necho done")
    assert r.exit_code == 0 and "done" in r.stdout and r.leftover == 0
    assert r.peak_tasks >= 3                                         # the two background processes did exist
    assert procs_matching(marker) == []


def test_a_job_cannot_leave_its_process_group(sandbox):
    sb = sandbox()
    r = sb.run_python("import os\nfor f in (os.setsid, lambda: os.setpgid(0, 0)):\n    try:\n        f()\n"
                      "        print('escaped')\n    except PermissionError:\n        print('EPERM')\n"
                      "print(os.getpgid(0) == os.getsid(0))")
    assert r.ok and r.stdout.split() == ["EPERM", "EPERM", "True"]


def test_output_is_truncated_and_counted(sandbox):
    sb = sandbox()
    r = sb.run_python("import sys\nsys.stdout.write('HEAD' + 'a' * 300000 + 'TAIL')\nsys.stderr.write('e' * 10)",
                      limits=lim(out_bytes=4096))
    assert r.ok and r.stdout_truncated and not r.stderr_truncated
    assert r.stdout_bytes == 300008 and len(r.stdout) < 4096 + 64
    assert r.stdout.startswith("HEAD") and r.stdout.endswith("TAIL") and "bytes truncated" in r.stdout
    assert r.stderr == "e" * 10


def test_cancellation_kills_the_job(sandbox, procs_matching):
    sb = sandbox()
    marker = f"cancel-{uuid.uuid4().hex}"
    argv, _ = sb.python_argv(f"import subprocess, sys, time\n"
                             f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)', '{marker}'])\n"
                             f"time.sleep(300)")
    cancel = threading.Event()
    threading.Timer(0.5, cancel.set).start()
    t0 = time.monotonic()
    r = sb.run(argv, cancel=cancel, limits=lim(wall_s=60))
    assert r.cancelled and not r.timed_out and time.monotonic() - t0 < 5
    assert r.leftover == 0 and procs_matching(marker) == []


# ---------------------------------------------------------------------------------------------------------- kernel
PROBE = r"""
import ctypes, os, socket, json
libc = ctypes.CDLL(None, use_errno=True)
out = {}
def sys_(name, nr, *args):
    ctypes.set_errno(0)
    rc = libc.syscall(nr, *args)
    out[name] = [rc, ctypes.get_errno()]
sys_("ptrace", 101, 0, 0, 0, 0)                    # PTRACE_TRACEME
sys_("unshare_user", 272, 0x10000000)              # CLONE_NEWUSER
sys_("unshare_net", 272, 0x40000000)
sys_("setns", 308, 0, 0)
sys_("mount", 165, b"none", b"/tmp", b"tmpfs", 0, None)
sys_("bpf", 321, 0, None, 0)
sys_("clone_newns", 56, 0x00020000 | 17, 0, 0, 0, 0)
sys_("keyctl", 250, 0, 0)
sys_("io_uring_setup", 425, 1, None)
sys_("perf_event_open", 298, None, 0, -1, -1, 0)
sys_("process_vm_readv", 310, 1, None, 0, None, 0, 0)
out["no_new_privs"] = libc.prctl(39, 0, 0, 0, 0)   # PR_GET_NO_NEW_PRIVS
out["seccomp_mode"] = libc.prctl(21, 0, 0, 0, 0)   # PR_GET_SECCOMP
socks = {}
for name, fam, typ in (("inet", socket.AF_INET, socket.SOCK_STREAM), ("inet6", socket.AF_INET6, socket.SOCK_STREAM),
                       ("udp", socket.AF_INET, socket.SOCK_DGRAM), ("unix", socket.AF_UNIX, socket.SOCK_STREAM),
                       ("netlink", 16, socket.SOCK_RAW), ("packet", 17, socket.SOCK_RAW)):
    try:
        socket.socket(fam, typ).close()
        socks[name] = "created"
    except OSError as e:
        socks[name] = e.errno
out["sockets"] = socks
a, b = socket.socketpair()                         # a local pair is allowed (asyncio, subprocess plumbing)
a.send(b"x"); out["socketpair"] = b.recv(1).decode()
try:
    socket.socketpair(socket.AF_INET)
    out["socketpair_inet"] = "created"
except OSError as e:
    out["socketpair_inet"] = e.errno
out["uid"] = [os.getuid(), os.geteuid()]
try:
    os.setuid(0); out["setuid0"] = "ok"
except OSError as e:
    out["setuid0"] = e.errno
print(json.dumps(out))
"""


def test_seccomp_blocks_sockets_and_kernel_interfaces(sandbox):
    sb = sandbox()
    r = sb.run_python(PROBE)
    assert r.ok, r.stderr
    out = json.loads(r.stdout)
    for name in ("ptrace", "unshare_user", "unshare_net", "setns", "mount", "bpf", "clone_newns", "keyctl",
                 "io_uring_setup", "perf_event_open", "process_vm_readv"):
        assert out[name] == [-1, 1], (name, out[name])                # EPERM from the filter
    assert out["no_new_privs"] == 1 and out["seccomp_mode"] == 2
    assert out["sockets"] == {k: 1 for k in ("inet", "inet6", "udp", "unix", "netlink", "packet")}
    assert out["socketpair"] == "x" and out["socketpair_inet"] == 1
    assert 0 not in out["uid"] and out["setuid0"] == 1


def test_network_is_closed_by_default_for_programs_and_shell_tools(sandbox):
    sb = sandbox()
    r = sb.run_python("import urllib.request\ntry:\n    urllib.request.urlopen('http://127.0.0.1:11111/', timeout=3)\n"
                      "    print('reached')\nexcept Exception as e:\n    print(type(e).__name__, e)")
    assert r.ok and "reached" not in r.stdout and "Operation not permitted" in r.stdout
    if shutil.which("curl"):
        r2 = sb.run_shell("curl -s -m 3 http://127.0.0.1:11111/ ; echo rc=$?")
        assert "rc=0" not in r2.stdout


# ---------------------------------------------------------------------------------------------------------- filesystem
@pytest.fixture
def planted(sbx_base):
    """World-readable look-alikes of everything generated code must never read: Landlock alone has to hold."""
    root = os.path.join(sbx_base, f"outside-{uuid.uuid4().hex[:8]}")
    files = {
        "vault/discord_bot_token.json": '{"token": "FAKE-SECRET-TOKEN-0001"}',
        "browser_profile/Default/Cookies": "FAKE-SESSION-COOKIE-0002",
        "run/audit.jsonl": '{"entry": "FAKE-AUDIT-0003"}\n',
        "run/verifier.jsonl": '{"signed": "FAKE-VERIFIER-0004"}\n',
        "run/checkpoints/ckpt_000001/model.safetensors": "FAKE-WEIGHTS-0005",
        "run/sock/auth.sock.txt": "FAKE-0006",
    }
    for rel, content in files.items():
        full = os.path.join(root, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
        os.chmod(full, 0o666)
    for d, _, _ in os.walk(root):
        os.chmod(d, 0o777)
    yield root, files
    shutil.rmtree(root, ignore_errors=True)


READ_ATTEMPT = r"""
import json, os, sys
paths = json.loads(sys.argv[1])
out = {}
for p in paths:
    try:
        with open(p) as f:
            out[p] = "READ:" + f.read(64)
    except OSError as e:
        out[p] = type(e).__name__
    try:
        with open(p, "a") as f:
            f.write("tamper")
        out[p] += "+WROTE"
    except OSError:
        pass
dirs = {}
for d in sorted({os.path.dirname(p) for p in paths}):
    try:
        dirs[d] = os.listdir(d)
    except OSError as e:
        dirs[d] = type(e).__name__
    try:
        open(os.path.join(d, "dropped.txt"), "w").write("x")
        dirs[d] = "CREATED"
    except OSError:
        pass
print(json.dumps({"files": out, "dirs": dirs}))
"""


def test_generated_code_cannot_read_or_write_outside_its_workspace(sandbox, planted):
    root, files = planted
    sb = sandbox()
    here = os.path.dirname(os.path.abspath(sbx.__file__))
    targets = [os.path.join(root, rel) for rel in files] + [
        "/etc/environment", "/etc/shadow", "/proc/1/environ", f"/proc/{os.getpid()}/environ",
        f"/proc/{os.getpid()}/mem", os.path.join(here, "sandbox.py"), os.path.join(here, "..", "contracts.py"),
        os.path.expanduser("~/.bashrc"), "/workspace/.hf_home/hub/version.txt"]
    argv, _ = sb.python_argv(READ_ATTEMPT, [json.dumps(targets)])
    r = sb.run(argv)
    assert r.ok, r.stderr
    out = json.loads(r.stdout)
    for p in targets:
        assert out["files"][p] in ("PermissionError", "FileNotFoundError"), (p, out["files"][p])
    for rel in files:                                                # the planted ones exist: must be a denial
        assert out["files"][os.path.join(root, rel)] == "PermissionError"
    assert all(v == "PermissionError" for d, v in out["dirs"].items() if d.startswith(root)), out["dirs"]
    assert "FAKE-" not in r.stdout
    for rel, content in files.items():                               # untouched
        assert open(os.path.join(root, rel)).read() == content
        assert not os.path.exists(os.path.join(os.path.dirname(os.path.join(root, rel)), "dropped.txt"))
    # the same through shell tools, and through a symlink planted inside the workspace
    secret = os.path.join(root, "vault/discord_bot_token.json")
    r2 = sb.run_shell(f"cat '{secret}'; echo rc=$?; ln -s '{secret}' link.json; cat link.json; echo rc=$?; "
                      f"ls '{root}'; echo rc=$?; cp '{secret}' stolen.json; echo rc=$?")
    assert "FAKE-" not in r2.stdout and r2.stdout.count("rc=0") == 0, r2.stdout
    with pytest.raises(OSError):                                     # the harness (root) does not follow it either
        sb.read_file("link.json")
    assert "stolen.json" not in [e["path"] for e in sb.list_files(".")["entries"]]


def test_workspace_files_are_private_to_the_sandbox_user(sandbox):
    sb = sandbox()
    sb.run_python("open('note.txt', 'w').write('x')")
    st = os.stat(os.path.join(sb.workspace, "note.txt"))
    assert stat.S_IMODE(st.st_mode) & 0o077 == 0 and st.st_uid == sb.uid
    assert stat.S_IMODE(os.stat(sb.workspace).st_mode) == 0o700


def test_file_tools_stay_inside_the_workspace(sandbox, planted):
    root, files = planted
    sb = sandbox()
    for bad in ("../x.txt", "/etc/passwd", "a/../../x", "", "a\x00b"):
        with pytest.raises(ValueError):
            sb.write_file(bad, b"x")
        with pytest.raises(ValueError):
            sb.read_file(bad)
    info = sb.write_file("docs/a/b.txt", b"hello")
    assert info["bytes"] == 5 and sb.read_file("docs/a/b.txt")["data"] == b"hello"
    sb.write_file("docs/a/b.txt", b" world", append=True)
    assert sb.read_file("docs/a/b.txt")["data"] == b"hello world"
    assert sb.read_file("docs/a/b.txt", offset=6, max_bytes=3) == {
        "path": "docs/a/b.txt", "size": 11, "offset": 6, "data": b"wor", "truncated": True,
        "mtime": sb.read_file("docs/a/b.txt")["mtime"]}
    listing = sb.list_files("docs", recursive=True)
    assert [(e["path"], e["type"]) for e in listing["entries"]] == [("docs/a", "dir"), ("docs/a/b.txt", "file")]
    # generated code plants links that point out of the workspace; the harness must not write or read through them
    target_dir = os.path.join(root, "run")
    r = sb.run_shell(f"ln -s '{target_dir}' outdir; ln -s '{target_dir}/audit.jsonl' outfile; mkfifo pipe; echo ok")
    assert "ok" in r.stdout
    with pytest.raises(OSError):
        sb.write_file("outdir/planted.txt", b"x")
    assert not os.path.exists(os.path.join(target_dir, "planted.txt"))
    sb.write_file("outfile", b"replaced")                            # replaces the link itself, not its target
    assert open(os.path.join(target_dir, "audit.jsonl")).read() == files["run/audit.jsonl"]
    assert sb.read_file("outfile")["data"] == b"replaced"
    with pytest.raises(OSError):
        sb.read_file("outdir/audit.jsonl")
    with pytest.raises(ValueError):
        sb.read_file("pipe")                                         # not a plain file; and it must not block
    with pytest.raises(ValueError):
        sb.write_file("big.bin", b"x" * (sb.cfg.max_file_bytes + 1))


def test_environment_is_clean(sandbox, monkeypatch):
    monkeypatch.setenv("CAM_FAKE_API_TOKEN", "tok-should-not-leak")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok-should-not-leak-2")
    sb = sandbox()
    argv, _ = sb.python_argv("import os, json\nprint(json.dumps(dict(os.environ)))")
    r = sb.run(argv, env={"MY_SETTING": "1", "LD_PRELOAD": "/tmp/evil.so", "PATH": "/evil", "bad key": "x"})
    env = json.loads(r.stdout)
    assert "tok-should-not-leak" not in r.stdout
    assert env["MY_SETTING"] == "1" and "LD_PRELOAD" not in env and "bad key" not in env
    assert env["HOME"] == sb.workspace and env["PATH"].endswith("/usr/local/bin:/usr/bin:/bin")
    assert set(env) <= {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "PYTHONPATH", "PYTHONPYCACHEPREFIX",
                        "PYTHONUNBUFFERED", "PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CAM_SANDBOX", "CAM_SANDBOX_NETWORK", "USER",
                        "LOGNAME", "SHELL", "TERM", "MY_SETTING", "LC_CTYPE"}


def test_workspace_quota_is_checked(sandbox):
    sb = sandbox(workspace_max_mb=1)
    sb.run_python("open('fill.bin', 'wb').write(b'x' * (3 << 20))")
    assert sb.usage()["bytes"] > 1 << 20
    with pytest.raises(OSError):
        sb.write_file("more.txt", b"x")


# ---------------------------------------------------------------------------------------------------------- executor
def test_executor_contract_and_tools(sandbox, make_ctx, act):
    ex = sbx.SandboxExecutor(sandbox=sandbox())
    assert isinstance(ex, c.ModalityExecutor) and ex.modality == c.Modality.CODE
    tools = {t.name: t for t in ex.tools()}
    assert {"sandbox.python", "sandbox.shell", "sandbox.write_file", "sandbox.read_file", "sandbox.list_files",
            "sandbox.job_start", "sandbox.job_status", "sandbox.job_result", "sandbox.job_cancel"} == set(tools)
    assert all(t.available and t.modality == c.Modality.CODE and t.arg_schema["type"] == "object" for t in tools.values())
    assert ex.validate(act("sandbox.python", code="print(1)")) == []
    assert ex.validate(act("sandbox.python")) == ["missing argument 'code'"]
    assert ex.validate(act("sandbox.python", code=3))[0].startswith("code: expected string")
    assert ex.validate(act("sandbox.shell", command="ls", network="open"))[0].startswith("network: must be one of")
    assert ex.validate(act("sandbox.shell", command="ls", bogus=1)) == ["unknown argument 'bogus'"]
    assert ex.validate(act("sandbox.write_file", path="../x", content="a"))[0].startswith("path:")
    assert ex.validate(act("sandbox.write_file", path="x", content="!!", encoding="base64")) == ["content is not valid base64"]
    assert ex.validate(act("sandbox.format_disk")) != []
    assert ex.validate(act("sandbox.python", code="1", limits={"mem_mb": "lots"})) == ["limits.mem_mb: expected integer"]
    bad = ex.execute(act("sandbox.python"), make_ctx())
    assert bad.status == c.ResultStatus.FAILED and bad.failure_class == c.FailureClass.INVALID_ARGUMENTS


def test_executor_runs_code_and_reports_what_happened(sandbox, make_ctx, act, policy_cls):
    ex = sbx.SandboxExecutor(sandbox=sandbox())
    policy = policy_cls()
    ctx = make_ctx(policy)
    res = ex.execute(act("sandbox.python", code="print('answer', 6 * 7)",
                         verify=c.VerificationCondition(kind="text_present", spec={"text": "answer 42"})), ctx)
    assert res.status == c.ResultStatus.OK and res.failure_class == c.FailureClass.NONE and res.verified is True
    d = data(res)
    assert d["exit_code"] == 0 and d["stdout"] == "answer 42\n" and d["leftover_processes"] == 0
    obs = res.observations[0]
    assert obs.provenance.trust == c.Trust.UNTRUSTED and obs.kind == "tool_result" and obs in ctx.sink
    assert {q.name for q in obs.quantities} >= {"sandbox.exit_code", "sandbox.wall_s", "sandbox.cpu_s"}
    assert [o.op for o in res.ops] == ["sandbox.python"] and res.cost.primitive_ops == 1 and res.cost.seconds > 0
    assert policy.ops("sandbox.python") and policy.ops("sandbox.python")[0][1]["network"] == "none"
    assert not hasattr(res, "reward") and "reward" not in json.dumps(res.to_dict())

    failed = ex.execute(act("sandbox.shell", command="echo partial; exit 7",
                            verify=c.VerificationCondition(kind="text_present", spec={"text": "partial"})), ctx)
    assert failed.status == c.ResultStatus.FAILED and data(failed)["exit_code"] == 7 and failed.verified is False

    wrong = ex.execute(act("sandbox.shell", command="echo something else",
                           verify=c.VerificationCondition(kind="text_present", spec={"text": "expected"})), ctx)
    assert wrong.status == c.ResultStatus.OK and wrong.verified is False       # ran fine, effect not there

    checked = ex.execute(act("sandbox.shell", command="echo 5 > five.txt",
                             verify=c.VerificationCondition(kind="program", spec={"command": "test $(cat five.txt) = 5"})), ctx)
    assert checked.verified is True
    unknown = ex.execute(act("sandbox.shell", command="true"), ctx)
    assert unknown.status == c.ResultStatus.OK and unknown.verified is None

    slow = ex.execute(act("sandbox.python", code="import time; time.sleep(30)", timeout_s=1.0), ctx)
    assert slow.status == c.ResultStatus.TIMEOUT and slow.failure_class == c.FailureClass.RESOURCE_LIMIT
    oom = ex.execute(act("sandbox.python", code="b = b'x' * (1 << 30)", limits={"mem_mb": 128}), ctx)
    assert oom.status == c.ResultStatus.FAILED and oom.failure_class == c.FailureClass.RESOURCE_LIMIT
    assert data(oom)["limit_hit"] == "memory"
    loose = ex.execute(act("sandbox.python", code="import resource\nprint(resource.getrlimit(resource.RLIMIT_AS)[0] >> 20)",
                           limits={"mem_mb": 10 ** 6}), ctx)
    assert data(loose)["stdout"].strip() == str(ex.cfg.limits.mem_mb)            # a request cannot loosen a limit


def test_executor_file_tools(sandbox, make_ctx, act, policy_cls):
    ex = sbx.SandboxExecutor(sandbox=sandbox())
    policy = policy_cls()
    ctx = make_ctx(policy)
    w = ex.execute(act("sandbox.write_file", path="site/index.html", content="<h1>hi</h1>"), ctx)
    assert w.status == c.ResultStatus.OK and data(w)["size"] == 11 and w.artifacts == {"file": "site/index.html"}
    import base64
    ex.execute(act("sandbox.write_file", path="site/logo.bin", content=base64.b64encode(b"\xff\x00\x01").decode(),
                   encoding="base64"), ctx)
    rd = ex.execute(act("sandbox.read_file", path="site/index.html"), ctx)
    assert data(rd)["content"] == "<h1>hi</h1>" and data(rd)["encoding"] == "utf-8"
    assert rd.observations[0].kind == "document" and rd.observations[0].text == "<h1>hi</h1>"
    rb = ex.execute(act("sandbox.read_file", path="site/logo.bin"), ctx)
    assert data(rb)["encoding"] == "base64" and base64.b64decode(data(rb)["content"]) == b"\xff\x00\x01"
    ls = ex.execute(act("sandbox.list_files", path="site"), ctx)
    assert [e["path"] for e in data(ls)["entries"]] == ["site/index.html", "site/logo.bin"]
    seen = ex.execute(act("sandbox.shell", command="cat site/index.html"), ctx)       # code sees the written file
    assert data(seen)["stdout"] == "<h1>hi</h1>"
    missing = ex.execute(act("sandbox.read_file", path="nope.txt"), ctx)
    assert missing.status == c.ResultStatus.FAILED and missing.failure_class == c.FailureClass.INVALID_ARGUMENTS
    assert [x[0] for x in policy.calls] == ["sandbox.write_file", "sandbox.write_file", "sandbox.read_file",
                                            "sandbox.read_file", "sandbox.list_files", "sandbox.shell",
                                            "sandbox.read_file"]


def test_denied_by_policy_runs_nothing(sandbox, make_ctx, act, policy_cls):
    sb = sandbox()
    ex = sbx.SandboxExecutor(sandbox=sb)
    for verdict, status in ((c.AuthVerdict.DENY, c.ResultStatus.DENIED),
                            (c.AuthVerdict.NEEDS_APPROVAL, c.ResultStatus.NEEDS_APPROVAL)):
        ctx = make_ctx(policy_cls(lambda op, action, v=verdict: v))
        for a in (act("sandbox.python", code="open('ran.txt', 'w').write('x')"),
                  act("sandbox.shell", command="touch ran.txt"),
                  act("sandbox.write_file", path="ran.txt", content="x"),
                  act("sandbox.job_start", purpose="p", kind="shell", code="touch ran.txt", deadline_s=5)):
            res = ex.execute(a, ctx)
            assert res.status == status and res.failure_class == c.FailureClass.MISSING_PERMISSION
            assert res.observations == [] and res.ops[0].status == status
    time.sleep(0.2)
    assert [e["path"] for e in sb.list_files(".")["entries"] if not e["path"].startswith(".")] == []
    no_callback = c.ExecContext(clock=c.WallClock(), check=None, obs_sink=lambda o: None)
    res = ex.execute(act("sandbox.shell", command="touch ran.txt"), no_callback)       # fail-closed
    assert res.status == c.ResultStatus.DENIED
    broken = make_ctx(lambda op, action: 1 / 0)
    assert ex.execute(act("sandbox.shell", command="touch ran.txt"), broken).status == c.ResultStatus.DENIED
    assert "ran.txt" not in [e["path"] for e in sb.list_files(".")["entries"]]


def test_executor_honours_cancel_and_deadline(sandbox, make_ctx, act):
    ex = sbx.SandboxExecutor(sandbox=sandbox())
    ctx = make_ctx()
    threading.Timer(0.4, ctx.cancel.set).start()
    t0 = time.monotonic()
    res = ex.execute(act("sandbox.python", code="import time; time.sleep(60)"), ctx)
    assert res.status == c.ResultStatus.CANCELLED and time.monotonic() - t0 < 5
    assert ex.execute(act("sandbox.shell", command="true"), ctx).status == c.ResultStatus.CANCELLED   # still set
    ctx2 = make_ctx(deadline=time.time() + 1.0)
    t0 = time.monotonic()
    res2 = ex.execute(act("sandbox.python", code="import time; time.sleep(60)"), ctx2)
    assert res2.status == c.ResultStatus.TIMEOUT and time.monotonic() - t0 < 6
    ctx3 = make_ctx(deadline=time.time() - 1.0)
    assert ex.execute(act("sandbox.shell", command="true"), ctx3).status == c.ResultStatus.TIMEOUT


def test_secrets_known_to_the_broker_are_scrubbed_from_output(sandbox, make_ctx, act):
    class Broker:
        def scrub(self, text):
            return text.replace("hunter2", "[redacted]")
    ex = sbx.SandboxExecutor(sandbox=sandbox())
    res = ex.execute(act("sandbox.shell", command="echo the password is hunter2"), make_ctx(secrets=Broker()))
    assert "hunter2" not in json.dumps(res.to_dict()) and "[redacted]" in data(res)["stdout"]


# ---------------------------------------------------------------------------------------------------------- scoped network
class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"                    # keep-alive: two requests can share one connection

    def do_GET(self):
        body = f"served {self.path} by {self.server.server_port}".encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = b"echo:" + self.rfile.read(n)
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def two_servers():
    servers = [http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler) for _ in range(2)]
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    yield [s.server_port for s in servers]
    for s in servers:
        s.shutdown()
        s.server_close()


FETCH = r"""
import json, sys, urllib.request, http.client, socket, _socket
allowed, blocked = int(sys.argv[1]), int(sys.argv[2])
out = {}
out["get"] = urllib.request.urlopen(f"http://127.0.0.1:{allowed}/hello", timeout=10).read().decode()
conn = http.client.HTTPConnection("127.0.0.1", allowed, timeout=10)
conn.request("POST", "/p", body=b"payload")
out["post"] = conn.getresponse().read().decode()
conn.request("GET", "/again")                                   # same connection, second request
out["keepalive"] = conn.getresponse().read().decode()
try:
    urllib.request.urlopen(f"http://127.0.0.1:{blocked}/secret", timeout=10)
    out["blocked"] = "reached"
except Exception as e:
    out["blocked"] = repr(e)
try:
    urllib.request.urlopen("http://blocked.invalid/", timeout=10)
    out["name"] = "reached"
except Exception as e:
    out["name"] = repr(e)
try:
    _socket.socket(socket.AF_INET, socket.SOCK_STREAM)          # around the shim: the kernel still says no
    out["raw"] = "created"
except OSError as e:
    out["raw"] = e.errno
print(json.dumps(out))
"""


def test_scoped_network_sends_every_connection_through_the_policy_check(sandbox, make_ctx, act, policy_cls, two_servers):
    allowed, blocked = two_servers
    ex = sbx.SandboxExecutor(sandbox=sandbox(allow_private_destinations=True))

    def rule(op, action):
        if op.op == "net.connect":
            ok = op.arguments["host"] == "127.0.0.1" and op.arguments["port"] == allowed
            return c.AuthVerdict.ALLOW if ok else c.AuthVerdict.DENY
        return c.AuthVerdict.ALLOW
    policy = policy_cls(rule)
    res = ex.execute(act("sandbox.python", code=FETCH, args=[str(allowed), str(blocked)], network="scoped",
                         hosts=[f"127.0.0.1:{allowed}"]), make_ctx(policy))
    assert res.status == c.ResultStatus.OK, data(res)
    out = json.loads(data(res)["stdout"])
    assert out["get"] == f"served /hello by {allowed}"
    assert out["post"] == "echo:payload" and out["keepalive"] == f"served /again by {allowed}"
    assert "reached" not in (out["blocked"], out["name"]) and "not authorized" in out["blocked"]
    assert out["raw"] == 1                                           # EPERM: no socket without the broker
    checks = policy.ops("net.connect")
    assert [(a["host"], a["port"], v) for _, a, v in checks] == [
        ("127.0.0.1", allowed, c.AuthVerdict.ALLOW), ("127.0.0.1", allowed, c.AuthVerdict.ALLOW),
        ("127.0.0.1", blocked, c.AuthVerdict.DENY), ("blocked.invalid", 80, c.AuthVerdict.DENY)]
    net_ops = [o for o in res.ops if o.op == "net.connect"]
    assert [o.status for o in net_ops] == [c.ResultStatus.OK, c.ResultStatus.OK, c.ResultStatus.DENIED,
                                           c.ResultStatus.DENIED]
    # authorization is per connection: the POST and the GET that reused its connection were one check (as stated
    # in capabilities()["network"]["scoped"]["not_enforced"])
    conns = data(res)["connections"]
    assert [x["status"] for x in conns] == ["closed", "closed", "denied", "denied"]
    assert conns[0]["bytes_in"] > 0 and conns[0]["bytes_out"] > 0
    assert conns[3]["host"] == "blocked.invalid" and conns[3]["ip"] == ""      # refused before any name lookup


def test_scoped_network_refuses_private_destinations_unless_configured(sandbox, make_ctx, act, policy_cls, two_servers):
    allowed, _ = two_servers
    ex = sbx.SandboxExecutor(sandbox=sandbox())                      # default: allow_private_destinations=False
    policy = policy_cls()
    code = (f"import urllib.request\ntry:\n    print(urllib.request.urlopen('http://127.0.0.1:{allowed}/x', timeout=5).read())\n"
            f"except Exception as e:\n    print('refused', e)")
    res = ex.execute(act("sandbox.python", code=code, network="scoped", hosts=[f"127.0.0.1:{allowed}"]), make_ctx(policy))
    assert "refused" in data(res)["stdout"] and "served" not in data(res)["stdout"]
    assert policy.ops("net.connect") == [] and data(res)["connections"][0]["status"] == "denied"


def test_scoped_network_reaches_children_started_by_the_job(sandbox, make_ctx, act, two_servers):
    allowed, _ = two_servers
    ex = sbx.SandboxExecutor(sandbox=sandbox(allow_private_destinations=True))
    cmd = (f"python -c \"import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:{allowed}/child').read().decode())\"\n"
           f"curl -s -m 3 http://127.0.0.1:{allowed}/curl; echo curl_rc=$?")
    res = ex.execute(act("sandbox.shell", command=cmd, network="scoped", hosts=[f"127.0.0.1:{allowed}"]), make_ctx())
    out = data(res)["stdout"]
    assert f"served /child by {allowed}" in out
    assert "served /curl" not in out and "curl_rc=0" not in out      # only Python gets the shim: stated in capabilities


# ---------------------------------------------------------------------------------------------------------- honesty
def test_capabilities_state_what_is_enforced(sandbox):
    sb = sandbox()
    caps = sbx.SandboxExecutor(sandbox=sb).capabilities()
    iso = caps["isolation"]
    assert caps["available"] and iso["seccomp"] and iso["landlock_abi"] >= 1 and iso["no_new_privs"]
    assert iso["uid_mode"] in ("dedicated-uid", "shared-unprivileged-uid", "same-uid")
    assert caps["weak"] == (iso["uid_mode"] != "dedicated-uid")
    if iso["uid_mode"] != "dedicated-uid":
        assert caps["weaknesses"]
    assert any("Landlock" in s for s in caps["kernel_enforced"]) and any("seccomp" in s for s in caps["kernel_enforced"])
    assert any("cgroup" in s for s in caps["not_enforced"])
    scoped = caps["network"]["scoped"]
    assert caps["network"]["default"] == "none"
    assert scoped["kernel_enforced"] and scoped["harness_enforced"] and scoped["not_enforced"]
    assert any("TLS" in s for s in scoped["not_enforced"])
    json.dumps(caps)
    weak = sandbox(seccomp=False).capabilities()
    assert weak["weak"] and any("seccomp is off" in w for w in weak["weaknesses"]) and not weak["isolation"]["seccomp"]


def test_unavailable_hosts_are_reported_not_worked_around(monkeypatch, sbx_base, make_ctx, act):
    host = dict(sbx.probe(), root=False, uid_mode="same-uid", uid=os.geteuid(), gid=os.getegid(), user="")
    monkeypatch.setattr(sbx, "probe", lambda: host)
    sb = sbx.Sandbox(os.path.join(sbx_base, "ws", "never"), sbx.SandboxConfig(base_dir=sbx_base))
    assert not sb.available and "allow_same_uid" in sb.unavailable_reason
    ex = sbx.SandboxExecutor(sandbox=sb)
    assert all(not t.available and t.unavailable_reason for t in ex.tools())
    res = ex.execute(act("sandbox.shell", command="true"), make_ctx())
    assert res.status == c.ResultStatus.FAILED and "unavailable" in res.failure_detail
    with pytest.raises(sbx.SandboxUnavailable):
        sb.run_shell("true")
    monkeypatch.setattr(sbx, "probe", lambda: dict(host, root=True, uid_mode="unavailable"))
    assert "refusing to run code as root" in sbx.Sandbox(os.path.join(sbx_base, "ws", "never2"),
                                                           sbx.SandboxConfig(base_dir=sbx_base)).unavailable_reason
    monkeypatch.setattr(sbx, "probe", lambda: dict(sbx_probe_real, landlock_abi=0))
    assert "Landlock" in sbx.Sandbox(os.path.join(sbx_base, "ws", "never3"),
                                     sbx.SandboxConfig(base_dir=sbx_base, allow_same_uid=True)).unavailable_reason


sbx_probe_real = sbx.probe()


# ---------------------------------------------------------------------------------------------------------- privileged
def _ensure_users():
    for _ in range(20):                                              # other test sessions may be creating them too
        try:
            return harness.ensure_service_users()
        except subprocess.CalledProcessError:
            time.sleep(0.5)
    return harness.ensure_service_users()


@pytest.mark.privileged
def test_dedicated_uid_cannot_read_files_of_other_service_uids(sbx_base, sbx_cfg):
    """The uid layer alone (Landlock switched off): vault, browser profile, checkpoints, audit and verifier logs
    owned by their service users are unreadable for generated code."""
    uids = _ensure_users()
    root = os.path.join(sbx_base, f"services-{uuid.uuid4().hex[:8]}")
    os.makedirs(root)
    os.chmod(root, 0o755)
    layout = {"vault/token.json": ("auth", 0o600, 0o700), "browser_profile/Cookies": ("browser", 0o600, 0o700),
              "checkpoints/model.safetensors": ("learner", 0o640, 0o750), "audit.jsonl": ("auth", 0o640, None),
              "verifier.jsonl": ("verifier", 0o640, None)}
    paths = []
    for rel, (role, fmode, dmode) in layout.items():
        full = os.path.join(root, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(f"FAKE-{role}-SECRET")
        os.chown(full, uids[role], uids[role])
        os.chmod(full, fmode)
        if dmode is not None:
            os.chown(os.path.dirname(full), uids[role], uids[role])
            os.chmod(os.path.dirname(full), dmode)
        paths.append(full)
    try:
        for landlock in (False, True):
            sb = sbx.Sandbox(os.path.join(sbx_base, "ws", uuid.uuid4().hex[:10]), sbx_cfg(landlock=landlock))
            try:
                caps = sb.capabilities()
                assert caps["isolation"]["uid_mode"] == "dedicated-uid" and sb.uid == uids["sandbox"]
                assert caps["weak"] == (not landlock)
                argv, _ = sb.python_argv(READ_ATTEMPT, [json.dumps(paths)])
                r = sb.run(argv)
                assert r.ok, r.stderr
                out = json.loads(r.stdout)
                assert all(v == "PermissionError" for v in out["files"].values()), out
                assert "FAKE-" not in r.stdout and r.isolation["uid"] == uids["sandbox"]
                me = sb.run_shell("id -un; id -Gn")
                assert me.stdout.split()[0] == harness.SERVICE_USERS["sandbox"]
                assert harness.SERVICE_GROUP not in me.stdout.split()[1:]      # supplementary groups are dropped
            finally:
                sb.close()
        for full in paths:
            assert open(full).read().startswith("FAKE-")
    finally:
        shutil.rmtree(root, ignore_errors=True)


SAME_UID = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
from cam_live.harness import sandbox as sbx
base, secret = sys.argv[2], sys.argv[3]
refused = sbx.Sandbox(os.path.join(base, "ws", "a"), sbx.SandboxConfig(base_dir=base))
sb = sbx.Sandbox(os.path.join(base, "ws", "b"), sbx.SandboxConfig(base_dir=base, allow_same_uid=True))
r = sb.run_python("import os\nprint(os.getuid())\ntry:\n    print(open(%r).read())\nexcept OSError as e:\n    print(type(e).__name__)" % secret)
caps = sb.capabilities()
print(json.dumps({"refused": [refused.available, refused.unavailable_reason], "available": sb.available,
                  "uid_mode": caps["isolation"]["uid_mode"], "weak": caps["weak"], "weaknesses": caps["weaknesses"],
                  "stdout": r.stdout, "stderr": r.stderr[-500:], "exit": r.exit_code, "me": os.getuid()}))
"""


@pytest.mark.privileged
def test_without_root_the_weaker_mode_is_refused_by_default_and_reported_when_allowed(sbx_base):
    """Run the sandbox from an unprivileged process: no uid drop is possible. It must refuse unless explicitly
    allowed, say so in capabilities(), and Landlock must still keep a readable file outside the workspace closed."""
    import pwd
    nobody = pwd.getpwnam("nobody")
    base = os.path.join(sbx_base, f"unpriv-{uuid.uuid4().hex[:8]}")
    os.makedirs(base)
    os.chown(base, nobody.pw_uid, nobody.pw_gid)
    secret = os.path.join(sbx_base, f"readable-secret-{uuid.uuid4().hex[:8]}.txt")
    with open(secret, "w") as f:
        f.write("FAKE-SECRET-WORLD-READABLE")
    os.chmod(secret, 0o644)
    src = os.path.abspath(os.path.join(os.path.dirname(sbx.__file__), "..", ".."))
    try:
        p = subprocess.run([os.path.realpath(sys.executable), "-c", SAME_UID, src, base, secret], capture_output=True,
                           text=True, user=nobody.pw_uid, group=nobody.pw_gid, extra_groups=[], timeout=60,
                           env={"PATH": "/usr/bin:/bin", "HOME": base}, cwd=base)
        assert p.returncode == 0, p.stderr[-2000:]
        out = json.loads(p.stdout.strip().splitlines()[-1])
        assert out["refused"][0] is False and "allow_same_uid" in out["refused"][1]
        assert out["available"] and out["uid_mode"] == "same-uid" and out["weak"]
        assert any("no uid drop" in w for w in out["weaknesses"])
        assert out["exit"] == 0, out
        assert out["stdout"].split() == [str(nobody.pw_uid), "PermissionError"]
    finally:
        shutil.rmtree(base, ignore_errors=True)
        os.unlink(secret)
