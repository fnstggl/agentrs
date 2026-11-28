"""G2-harness probe g11 (round 3; root; real launcher, real service users): the isolated boot path of the separated
services (``python -I service_boot.py <roots> -m <module>``, cwd ``/``) and the other root writers in the run
directory.

  S1  process context of every launched service: cwd, argv, HOME, PATH, LD_* and PYTHON* variables, NoNewPrivs.
  S2  everything the root sandbox host imports from or loads: each sys.path entry of the exact boot command (and
      the files below it), the interpreter chain and pyvenv.cfg, the LD_LIBRARY_PATH directories -- is any of it
      writable by a service uid (camlearn, cambrowser, camsbx, camauth, camverify)?
  S3  planted import hooks: argparse.py / sitecustomize.py / usercustomize.py / a .pth file / a ``cam_live``
      package in the run directory (learner), in the old HOME ``<run>/exec`` (browser uid) and in the sandbox
      workspace (sandbox uid); the root host and the authorization service are restarted; nothing may run.
  S4  the run directory is still group-writable for the learner (1771): the root supervisor's
      ``runtime.logs.atomic_write_text`` (used for ``preflight.json``, ``learner.go`` and the supervisor pid file
      while the learner runs) opens its temporary file ``.<name>.<pid>.<thread id>.tmp`` without O_EXCL /
      O_NOFOLLOW. Does root follow a symlink the learner planted under that name? (fs.protected_symlinks does not
      cover a sticky directory that is not world-writable.) The victim is a root-owned file in a scratch directory.
Run directories are short directories under /tmp (service uids must reach them); removed at exit.

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/g2harness/g11_boot_path.py
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

from _g2 import SRC, Report  # noqa: I001

R = Report("g11_boot_path")
if os.geteuid() != 0:
    R.note("not root: skipped")
    R.done()

import grp  # noqa: E402
import pwd  # noqa: E402

from cam_live.harness import SOCK_AUTH, uid_of  # noqa: E402
from cam_live.harness.auth import launcher as L  # noqa: E402

USERS = ("camlearn", "cambrowser", "camsbx", "camauth", "camverify")


def run_as(user, code, *args, timeout=300):
    pw = pwd.getpwnam(user)
    return subprocess.run([sys.executable, "-I", "-c", code, *map(str, args)], user=pw.pw_uid, group=pw.pw_gid,
                          extra_groups=[grp.getgrnam("camlive").gr_gid], cwd="/", timeout=timeout,
                          env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"}, capture_output=True, text=True)


def pid_of(run_dir, module):
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                cmd = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
            except OSError:
                continue
            if module.encode() in cmd and run_dir.encode() in b" ".join(cmd) and cmd[1:3] != [b"-I", b"-c"]:
                return int(p)
    return None


D = tempfile.mkdtemp(prefix="g2hbp", dir="/tmp")
os.chmod(D, 0o755)
MARK_DIR = tempfile.mkdtemp(prefix="g2hmk", dir="/tmp")
os.chmod(MARK_DIR, 0o1777)
MARK = os.path.join(MARK_DIR, "m")
VICTIM_DIR = tempfile.mkdtemp(prefix="g2hvc", dir="/tmp")
os.chmod(VICTIM_DIR, 0o755)
VICTIM = os.path.join(VICTIM_DIR, "precious")
with open(VICTIM, "w") as f:
    f.write("precious root-owned content\n")
os.chmod(VICTIM, 0o600)
sock = os.path.join(D, "sock")
lc = L.LauncherConfig(extra_args={"exec": ["--auth-socket", os.path.join(sock, SOCK_AUTH), "--provenance", "staging",
                                           "--run-dir", os.path.join(D, "exec")]})
handle = L.launch(D, ["auth", "exec"], lc)
planted = []
try:
    deadline = time.time() + 60
    while time.time() < deadline and not os.path.exists(os.path.join(sock, "sandbox.sock")):
        time.sleep(0.5)
    # S1 -------------------------------------------------------------------------------------------------------
    host = pid_of(D, "cam_live.harness.sandbox_host")
    pids = {"sandbox host": host, "auth": pid_of(D, "cam_live.harness.auth.service"),
            "exec": pid_of(D, "cam_live.harness.service")}
    host_env, host_argv = {}, []
    for name, pid in pids.items():
        if not pid:
            R.defect(f"{name} is running", "not found")
            continue
        try:
            env = dict(x.split("=", 1) for x in open(f"/proc/{pid}/environ").read().split("\0") if "=" in x)
        except PermissionError:
            env = {}                       # a non-dumpable service: its environment is not readable even by root
        argv = open(f"/proc/{pid}/cmdline").read().split("\0")[:-1]
        status = open(f"/proc/{pid}/status").read()
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except PermissionError:
            cwd = "/"   # non-dumpable (unreadable even for root); separated services get cwd "/" (launcher _spawn)
            R.note(f"{name}: /proc/{pid}/cwd and environ unreadable (non-dumpable service)")
        bad_env = sorted(k for k in env if k.startswith(("LD_PRELOAD", "LD_AUDIT")))   # PYTHON* are ignored under -I (S2)
        R.note(f"{name}: uid {os.stat(f'/proc/{pid}').st_uid} cwd {cwd} HOME {env.get('HOME')} PATH {env.get('PATH')} "
               f"LD_LIBRARY_PATH {env.get('LD_LIBRARY_PATH')} PYTHON* {sorted(k for k in env if k.startswith('PYTHON'))} "
               f"argv {' '.join(argv[:5])} ...")
        R.check(cwd == "/" and "-I" in argv[1:2] and not bad_env and "NoNewPrivs:\t1" in status,
                f"{name} starts isolated (cwd /, python -I, no LD_PRELOAD / LD_AUDIT, no_new_privs)",
                f"cwd {cwd}; argv[1] {argv[1:2]}; vars {bad_env}")
        if name == "sandbox host":
            host_env, host_argv = env, argv
    # S2 -------------------------------------------------------------------------------------------------------
    if host_argv:
        boot, roots = host_argv[2], host_argv[3]
        probe_script = os.path.join(VICTIM_DIR, "show_path.py")
        with open(probe_script, "w") as f:
            f.write("import json, sys, site\nprint(json.dumps({'path': sys.path, 'site': site.getsitepackages(), "
                    "'user_site': site.ENABLE_USER_SITE, 'prefix': sys.prefix, 'exe': sys.executable}))\n")
        cp = subprocess.run([host_argv[0], "-I", boot, roots, probe_script], env=host_env, cwd="/",
                            capture_output=True, text=True, timeout=60)
        info = json.loads(cp.stdout.strip().splitlines()[-1])
        R.note(f"root host import path (exact boot command): {info['path']}; user site {info['user_site']}")
        targets = [p for p in info["path"] if p and os.path.exists(p)]
        exe_chain = [host_argv[0]]
        try:
            exe_chain.append(os.path.realpath(host_argv[0]))
        except OSError:
            pass
        venv_cfg = os.path.join(os.path.dirname(os.path.dirname(host_argv[0])), "pyvenv.cfg")
        libdirs = [x for x in (host_env.get("LD_LIBRARY_PATH") or "").split(":") if x]
        walker = textwrap.dedent(r"""
            import json, os, sys
            roots, singles = json.loads(sys.argv[1]), json.loads(sys.argv[2])
            hits, n = [], 0
            for p in singles:
                for q in (p, os.path.dirname(p)):
                    if os.path.exists(q) and os.access(q, os.W_OK):
                        hits.append(q)
            for r in roots:
                for dp, dns, fns in os.walk(r):
                    n += 1
                    if os.access(dp, os.W_OK):
                        hits.append(dp)
                    for fn in fns:
                        if fn.endswith((".py", ".pth", ".so", ".pyc")) and os.access(os.path.join(dp, fn), os.W_OK):
                            hits.append(os.path.join(dp, fn))
                    if len(hits) > 20 or n > 60000:
                        break
            print(json.dumps({"hits": hits[:20], "dirs": n}))
        """)
        for user in USERS:
            cp = run_as(user, walker, json.dumps(targets + libdirs), json.dumps(exe_chain + [venv_cfg]))
            res = json.loads(cp.stdout.strip().splitlines()[-1]) if cp.returncode == 0 else {"error": cp.stderr[-300:]}
            R.check(cp.returncode == 0 and not res.get("hits"),
                    f"nothing the root host imports or loads is writable by {user} (sys.path entries and every file "
                    f"below them, interpreter chain, pyvenv.cfg, LD_LIBRARY_PATH)",
                    json.dumps(res)[:300])
    # S3 -------------------------------------------------------------------------------------------------------
    hook = textwrap.dedent(f"""
        import os, sys
        with open({MARK!r}, "a") as f:
            f.write(f"{{os.geteuid()}} {{__file__}}\\n")
    """)
    plants = [("camlearn", os.path.join(D, "argparse.py")), ("camlearn", os.path.join(D, "sitecustomize.py")),
              ("camlearn", os.path.join(D, "usercustomize.py")), ("camlearn", os.path.join(D, "g2h.pth")),
              ("camlearn", os.path.join(D, "cam_live", "__init__.py")),
              ("cambrowser", os.path.join(D, "exec", "sitecustomize.py")),
              ("cambrowser", os.path.join(D, "exec", ".local", "lib", "python3.12", "site-packages", "g2h.pth")),
              ("camsbx", os.path.join(D, "sandbox", "argparse.py"))]
    for user, path in plants:
        body = ("import os; open(%r, 'a').write(str(os.geteuid()) + ' pth\\n')\n" % MARK) if path.endswith(".pth") \
            else hook
        cp = run_as(user, "import os, sys\nos.makedirs(os.path.dirname(sys.argv[1]), exist_ok=True)\n"
                          "open(sys.argv[1], 'w').write(sys.argv[2])\n", path, body)
        if cp.returncode == 0:
            planted.append(path)
        R.note(f"{user} plants {path}: {'ok' if cp.returncode == 0 else cp.stderr.strip()[-100:]}")
    for name, module in (("sandbox host", "cam_live.harness.sandbox_host"), ("auth", "cam_live.harness.auth.service"),
                         ("exec", "cam_live.harness.service")):
        old = pid_of(D, module)
        if old:
            os.kill(old, signal.SIGKILL)
        deadline = time.time() + 30
        new = None
        while time.time() < deadline:
            new = pid_of(D, module)
            if new and new != old:
                break
            time.sleep(0.5)
        time.sleep(1.0)
        R.note(f"{name}: pid {old} killed -> restarted as {new}")
    marks = [m for m in (open(MARK).read().split("\n") if os.path.exists(MARK) else []) if m]
    R.check(not marks, "no planted hook ran in a restarted service (root host, authorization, environment)",
            f"marker lines: {marks}")
    # S4 -------------------------------------------------------------------------------------------------------
    st = os.stat(D)
    R.note(f"run directory mode {oct(st.st_mode & 0o7777)} group {grp.getgrgid(st.st_gid).gr_name}; "
           f"fs.protected_symlinks={open('/proc/sys/fs/protected_symlinks').read().strip()}")
    helper = textwrap.dedent(r"""
        import os, sys, threading
        sys.path.insert(0, sys.argv[1])
        from cam_live.runtime.logs import atomic_write_text
        print(os.getpid(), threading.get_ident(), flush=True)
        sys.stdin.readline()                                 # wait until the learner has planted its link
        try:
            atomic_write_text(os.path.join(sys.argv[2], "learner.go"), "go")
            print("written", flush=True)
        except Exception as e:
            print("refused", type(e).__name__, e, flush=True)
    """)
    p = subprocess.Popen([sys.executable, "-I", "-c", helper, SRC, D], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True)
    pid_s, tid_s = p.stdout.readline().split()
    link = os.path.join(D, f".learner.go.{pid_s}.{tid_s}.tmp")
    cp = run_as("camlearn", "import os, sys\nos.symlink(sys.argv[1], sys.argv[2])\n", VICTIM, link)
    R.note(f"camlearn plants {os.path.basename(link)} -> {VICTIM}: {'ok' if cp.returncode == 0 else cp.stderr[-120:]}")
    p.stdin.write("\n")
    p.stdin.flush()
    out = p.stdout.readline().strip()
    p.wait(timeout=30)
    content = open(VICTIM).read()
    R.check(content.startswith("precious"), "the root writer of the run directory (runtime.logs.atomic_write_text) "
                                            "does not follow a symlink the learner planted",
            f"writer: {out}; victim now {content[:40]!r}; learner.go is a symlink: {os.path.islink(os.path.join(D, 'learner.go'))}")
    # S4b: the same without any help -- the learner learns the temporary name from the supervisor's first write
    # (preflight.json, IN_CREATE through inotify on the run directory it may read) and plants the link for the second
    # (learner.go), the order of runtime/cli.py cmd_supervise; both are main-thread writes of one process
    racer = textwrap.dedent(r"""
        import ctypes, os, struct, sys, time
        d, victim = sys.argv[1], sys.argv[2]
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.inotify_init()
        libc.inotify_add_watch(fd, d.encode(), 0x100)          # IN_CREATE
        print("ready", flush=True)
        end = time.time() + 30
        while time.time() < end:
            buf = os.read(fd, 4096)
            i = 0
            while i + 16 <= len(buf):
                wd, mask, cookie, ln = struct.unpack_from("iIII", buf, i)
                name = buf[i + 16:i + 16 + ln].split(b"\0")[0].decode()
                i += 16 + ln
                if name.startswith(".preflight.json.") and name.endswith(".tmp"):
                    rest = name[len(".preflight.json."):-len(".tmp")]
                    os.symlink(victim, os.path.join(d, ".learner.go." + rest + ".tmp"))
                    print("planted", rest, flush=True)
                    sys.exit(0)
        print("timeout", flush=True)
    """)
    wins = 0
    trials = 5
    for k in range(trials):
        with open(VICTIM, "w") as f:
            f.write("precious root-owned content\n")
        for nm in ("learner.go", "preflight.json"):
            q = os.path.join(D, nm)
            if os.path.lexists(q):
                os.unlink(q)
        pw = pwd.getpwnam("camlearn")
        rp = subprocess.Popen([sys.executable, "-I", "-c", racer, D, VICTIM], user=pw.pw_uid, group=pw.pw_gid,
                              extra_groups=[grp.getgrnam("camlive").gr_gid], cwd="/", stdout=subprocess.PIPE, text=True,
                              env={"PATH": "/usr/bin:/bin"})
        rp.stdout.readline()
        writer = textwrap.dedent(r"""
            import os, sys
            sys.path.insert(0, sys.argv[1])
            from cam_live.runtime.logs import atomic_write_json, atomic_write_text
            atomic_write_json(os.path.join(sys.argv[2], "preflight.json"), {"live": {"ready": True}, "checks": list(range(200))})
            atomic_write_text(os.path.join(sys.argv[2], "learner.go"), "go")
        """)
        subprocess.run([sys.executable, "-I", "-c", writer, SRC, D], timeout=60)
        rp.wait(timeout=40)
        if open(VICTIM).read().strip() == "go":
            wins += 1
    R.check(wins == 0, "the learner cannot redirect the supervisor's preflight.json -> learner.go writes into a root "
                       "file by watching the run directory", f"victim overwritten in {wins} of {trials} races")
finally:
    for path in planted:
        try:
            os.unlink(path)
        except OSError:
            pass
    try:
        handle.stop()
    except Exception as e:  # noqa: BLE001
        print("stop:", e)
    time.sleep(0.5)
    for x in (D, MARK_DIR, VICTIM_DIR):
        shutil.rmtree(x, ignore_errors=True)
R.done()
