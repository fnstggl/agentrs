"""Sandbox isolation (EC), run as root so the real uid drop to camsbx happens. Generated programs try to:
read /root, the vault and another service's files, read other processes' /proc/<pid>/environ, open a network
socket, fork-bomb, exhaust memory, escape the workspace via .. and symlinks, write outside the workspace,
persist a child beyond the program, and regain privileges. Everything the program must NOT do is checked by its
observable effect, not by trusting the sandbox's own report.
"""
import os
import time

from _common import C, Report, cleanup, fresh_dir
from cam_live.harness import sandbox as sbx

R = Report("p11_sandbox")
if os.geteuid() != 0:
    R.note("not root: skipping (the uid drop and file-mode separation need root)")
    R.done()

base = fresh_dir("p11base", world=True)
# plant a secret-looking file owned by root 0600 and a fake vault owned by camauth, both outside the workspace
secret = os.path.join(base, "root_secret.txt")
open(secret, "w").write("TOP-SECRET-ROOT-FILE")
os.chmod(secret, 0o600)
import pwd
auth = pwd.getpwnam("camauth")
vault = os.path.join(base, "vault")
os.makedirs(vault, exist_ok=True)
vfile = os.path.join(vault, "bot.secret.json")
open(vfile, "w").write('{"value":"VAULT-CANARY"}')
os.chown(vfile, auth.pw_uid, auth.pw_gid)
os.chmod(vfile, 0o600)
os.chown(vault, auth.pw_uid, auth.pw_gid)
os.chmod(vault, 0o700)

cfg = sbx.SandboxConfig(base_dir=base, limits=sbx.SandboxLimits(wall_s=15.0, mem_mb=256, mem_total_mb=256, nproc=16))
sb = sbx.Sandbox(os.path.join(base, "ws", "probe"), cfg)
if not sb.available:
    R.defect("sandbox is unavailable even as root", sb.unavailable_reason)
    R.done()
R.note(f"isolation mode: {sb._probe['uid_mode']} uid={sb.uid} landlock_abi={sb._probe['landlock_abi']} "
       f"seccomp={bool(sb._probe['seccomp_lib'])}")


def run(code, **kw):
    return sb.run_python(code, **kw)


# 1. the uid really dropped
res = run("import os; print(os.getuid(), os.geteuid())")
uid_line = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
R.check(uid_line == f"{sb.uid} {sb.uid}" and sb.uid not in (0,), "generated code runs as the unprivileged sandbox uid",
        f"getuid/geteuid -> {uid_line!r}")

# 2. read /root, the planted root secret, the vault
for what, path, canary in (("a root-owned 0600 file", secret, "TOP-SECRET-ROOT-FILE"),
                           ("the camauth vault file", vfile, "VAULT-CANARY")):
    res = run(f"\nimport os\ntry:\n print('DATA', open({path!r}).read())\nexcept Exception as e:\n print('ERR', type(e).__name__)")
    R.check(canary not in res.stdout and "DATA " not in res.stdout, f"cannot read {what}",
            (res.stdout.strip().splitlines()[-1] if res.stdout.strip() else res.stderr[-80:])[:80])

res = run("import os; print(sorted(os.listdir('/root'))[:3])")
R.check("Permission" in res.stderr or res.exit_code != 0 or "[]" in res.stdout,
        "cannot list /root", (res.stdout.strip()[-80:] or res.stderr[-80:]))

# 3. read another live process's environ (this very test process holds no secret, but prove the barrier)
res = run(f"print(open('/proc/{os.getpid()}/environ','rb').read()[:20])")
R.check(res.exit_code != 0 or "ERR" in res.stdout or "b''" in res.stdout or not res.stdout.strip(),
        "cannot read another process's /proc/<pid>/environ", (res.stdout.strip()[-60:] or res.stderr[-60:]))

# 4. open a network socket (seccomp denies socket())
res = run("import socket; s=socket.socket(); print('SOCKET-OK')")
R.check("SOCKET-OK" not in res.stdout, "socket() is blocked for a no-network program", res.stderr[-80:])
res = run("import socket\ntry:\n socket.create_connection(('127.0.0.1',11111),2); print('CONNECT-OK')\nexcept Exception as e:\n print('ERR',type(e).__name__)")
R.check("CONNECT-OK" not in res.stdout, "cannot connect to a loopback service", res.stdout.strip()[-80:])

# 5. escape the workspace with .. and with a symlink
res = run(f"open('../../root_secret.txt').read()")
R.check(res.exit_code != 0 and "TOP-SECRET" not in res.stdout, "a program cannot read ../.. out of the workspace")
res = run(f"import os; os.symlink({secret!r}, 'link'); print('LINK', open('link').read())")
R.check("TOP-SECRET" not in res.stdout, "a symlink out of the workspace cannot be read through", res.stderr[-80:])
res = run(f"open('/etc/shadow').read()")
R.check(res.exit_code != 0 and "root:" not in res.stdout, "cannot read /etc/shadow")

# 6. write outside the workspace
res = run(f"open({os.path.join(base,'escaped.txt')!r},'w').write('x'); print('WROTE')")
R.check("WROTE" not in res.stdout and not os.path.exists(os.path.join(base, "escaped.txt")),
        "cannot create a file outside the workspace", res.stderr[-80:])
res = run("open('/tmp/cam_sbx_escape','w').write('x'); print('WROTE')")
R.check("WROTE" not in res.stdout and not os.path.exists("/tmp/cam_sbx_escape"), "cannot write to /tmp")

# 7. fork bomb -> killed by the per-uid task ceiling / monitor, nothing left behind
res = run("import os\nwhile True:\n try: os.fork()\n except Exception: pass", limits=sbx.SandboxLimits(wall_s=8, nproc=8, mem_mb=256, mem_total_mb=256))
R.check(res.leftover == 0, "a fork bomb leaves no surviving process", f"killed_for={res.killed_for} leftover={res.leftover}")

# 8. memory exhaustion -> RLIMIT_AS / monitor, no OOM of the host
res = run("x=bytearray()\nwhile True: x += bytes(10_000_000)", limits=sbx.SandboxLimits(wall_s=8, mem_mb=256, mem_total_mb=256, nproc=8))
R.check(res.exit_code not in (0,) or res.killed_for in ("memory_total", "memory"),
        "unbounded allocation is stopped", f"limit_hit={res.limit_hit!r} killed_for={res.killed_for!r}")

# 9. a background child must not outlive the program (monitor kills the whole group)
marker = os.path.join(sb.workspace, "child_alive")
res = run(f"import os,subprocess,sys,time\n"
          f"subprocess.Popen([sys.executable,'-c',\"import time,os\\nopen({marker!r},'w').write(str(os.getpid()))\\ntime.sleep(60)\"])\n"
          f"time.sleep(0.5); print('PARENT-DONE')")
time.sleep(2.0)
leaked = []
if os.path.exists(marker):
    try:
        pid = int(open(marker).read().strip())
        if os.path.exists(f"/proc/{pid}"):
            leaked.append(pid)
    except (ValueError, OSError):
        pass
R.check(not leaked, "a child process does not outlive the program that spawned it", f"leftover={res.leftover} leaked={leaked}")
for pid in leaked:
    try:
        os.kill(pid, 9)
    except OSError:
        pass

# 10. cannot regain privileges (no_new_privs; setuid denied by seccomp)
res = run("import os\ntry:\n os.setuid(0); print('SETUID-OK', os.getuid())\nexcept Exception as e:\n print('ERR',type(e).__name__)")
R.check("SETUID-OK" not in res.stdout, "setuid(0) fails inside the sandbox", res.stdout.strip()[-60:])

sb.close()
cleanup(base)
R.done()
