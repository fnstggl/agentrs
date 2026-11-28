"""G2-harness probe g16 (round 4; root; launcher layout, real service users): cam_live.harness.safefs under attack by
the learner in the run directory (mode 1771, group = learner gid).

  F1  races on the target: while root repeats ``write_atomic(<run>/target.json)``, the learner keeps swapping the
      entry between a regular file of its own and a symlink to a root-owned victim (rename over its own entry).
  F2  races on a path component: while root repeats ``write_atomic(<run>/sub/f.json)``, the learner keeps swapping
      ``<run>/sub`` between a directory of its own and a symlink to a root-owned victim directory.
  F3  non-regular entries: the learner replaces a file root reads every second (``state.json``, read by the
      supervisor's loop through ``runtime.logs.read_json`` -> ``safefs.read_nofollow``) with a FIFO. Does the root
      read return (refusal) or block? (``read_nofollow`` checks the file type after ``open``; ``open`` of a FIFO
      without O_NONBLOCK waits for a writer.)
Victims are root-owned files in a root-only scratch directory; everything is removed at exit.

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/g2harness/g16_safefs.py
"""
import grp
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
import textwrap

from _g2 import SRC, Report  # noqa: I001

R = Report("g16_safefs")
if os.geteuid() != 0:
    R.note("not root: skipped")
    R.done()

from cam_live.harness.auth import launcher as L  # noqa: E402


def spawn_as(user, code, *args):
    pw = pwd.getpwnam(user)
    return subprocess.Popen([sys.executable, "-I", "-c", code, *map(str, args)], user=pw.pw_uid, group=pw.pw_gid,
                            extra_groups=[grp.getgrnam("camlive").gr_gid], cwd="/", env={"PATH": "/usr/bin:/bin"},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


D = tempfile.mkdtemp(prefix="g2hsf", dir="/tmp")
os.chmod(D, 0o755)
VD = tempfile.mkdtemp(prefix="g2hsv", dir="/tmp")
os.chmod(VD, 0o700)
VICTIM = os.path.join(VD, "precious")
open(VICTIM, "w").write("precious\n")
os.chmod(VICTIM, 0o600)
procs = []
try:
    L.prepare_run_dir(D, True, dict(L.DEFAULT_ROLES), None, None)
    writer = textwrap.dedent(r"""
        import os, sys
        sys.path.insert(0, sys.argv[1])
        from cam_live.harness import safefs
        ok = refused = 0
        for i in range(int(sys.argv[3])):
            try:
                safefs.write_atomic(sys.argv[2], b'{"root": %d}' % i, 0o644, durable=False)
                ok += 1
            except OSError:
                refused += 1
        print(ok, refused)
    """)
    # F1 -------------------------------------------------------------------------------------------------------
    target = os.path.join(D, "target.json")
    swap = textwrap.dedent(r"""
        import os, sys, time
        d, victim = sys.argv[1], sys.argv[2]
        t, a, b = os.path.join(d, "target.json"), os.path.join(d, ".g2h_a"), os.path.join(d, ".g2h_b")
        end = time.time() + 20
        while time.time() < end:
            try:
                open(a, "w").write("learner")
                os.replace(a, t)                  # a regular file of the learner's at the target name
                os.symlink(victim, b)
                os.replace(b, t)                  # ... and now a symlink to the victim
            except OSError:
                pass
    """)
    procs.append(spawn_as("camlearn", swap, D, VICTIM))
    out = subprocess.run([sys.executable, "-I", "-c", writer, SRC, target, "3000"], capture_output=True, text=True,
                         timeout=120)
    procs[-1].kill()
    R.check(open(VICTIM).read() == "precious\n", "write_atomic never writes through a target the learner swaps "
                                                 "between a file and a symlink (3000 writes)",
            f"writes ok/refused: {out.stdout.strip()}; victim {open(VICTIM).read()[:20]!r}")
    # F2 -------------------------------------------------------------------------------------------------------
    sub = os.path.join(D, "sub")
    swap2 = textwrap.dedent(r"""
        import os, sys, time, shutil
        d, vdir = sys.argv[1], sys.argv[2]
        s, a, b = os.path.join(d, "sub"), os.path.join(d, ".g2h_da"), os.path.join(d, ".g2h_db")
        end = time.time() + 20
        while time.time() < end:
            try:
                os.makedirs(a, exist_ok=True)
                os.chmod(a, 0o777)
                if os.path.islink(s) or not os.path.exists(s):
                    if os.path.lexists(s):
                        os.unlink(s)
                    os.rename(a, s)
                else:
                    os.symlink(vdir, b)
                    shutil.rmtree(s, ignore_errors=True)
                    os.rename(b, s)
            except OSError:
                pass
    """)
    procs.append(spawn_as("camlearn", swap2, D, VD))
    out = subprocess.run([sys.executable, "-I", "-c", writer, SRC, os.path.join(sub, "f.json"), "3000"],
                         capture_output=True, text=True, timeout=120)
    procs[-1].kill()
    leaked = [x for x in os.listdir(VD) if x != "precious"]
    R.check(not leaked, "write_atomic never writes through a path component the learner swaps between a directory "
                        "and a symlink (3000 writes)", f"writes ok/refused: {out.stdout.strip()}; files in victim dir: {leaked}")
    # F3 -------------------------------------------------------------------------------------------------------
    st = os.path.join(D, "state.json")
    cp = subprocess.run([sys.executable, "-I", "-c", "import os,sys\nos.mkfifo(sys.argv[1])\n", st],
                        user=pwd.getpwnam("camlearn").pw_uid, group=pwd.getpwnam("camlearn").pw_gid,
                        extra_groups=[grp.getgrnam("camlive").gr_gid], cwd="/", capture_output=True, text=True)
    R.note(f"learner makes {st} a FIFO: {'ok' if cp.returncode == 0 else cp.stderr[-120:]}")
    reader = ("import sys\nsys.path.insert(0, sys.argv[1])\nfrom cam_live.runtime.logs import read_json\n"
              "print(read_json(sys.argv[2], {'default': True}))\n")
    try:
        out = subprocess.run([sys.executable, "-I", "-c", reader, SRC, st], capture_output=True, text=True, timeout=10)
        R.hold("root's read_json of a learner-planted FIFO returns instead of blocking", out.stdout.strip()[:80])
    except subprocess.TimeoutExpired:
        R.defect("root's read_json of a learner-planted FIFO returns instead of blocking",
                 "still blocked after 10 s: the supervisor's loop (runtime/cli.py cmd_supervise reads state.json every "
                 "second) and the operator's status / resume / export-checkpoint would hang")
finally:
    for p in procs:
        try:
            p.kill()
        except OSError:
            pass
    shutil.rmtree(D, ignore_errors=True)
    shutil.rmtree(VD, ignore_errors=True)
R.done()
