"""G2-harness probe g15 (round 4; root; real launcher layout, real service users, real ``cam-live`` CLI): operator
commands that the operator runs AS ROOT on a run directory the learner can write (mode 1771, group = learner gid).
The learner's own files (``state.json``, the runtime's state, is written by the learner process) and new names in the
run directory are under the learner's control.

  E1  ``cam-live export-checkpoint --run-dir R`` with no runtime alive takes the checkpoint id and path from
      ``state.json``: the destination is ``<run>/exports/export_<id>`` and, when the path is not a loadable
      checkpoint, the directory is copied with ``shutil.copytree(..., dirs_exist_ok=True)``. The learner writes
      ``state.json`` with ``id = "x/../../../../../../<victim dir>"`` and a path to a directory of its own holding
      one file. Does root copy the learner's file into the victim directory (a root-only scratch directory standing
      in for, e.g., /root/.ssh)?
  E2  ``cam-live plan-report --run-dir R`` (default output ``<run>/plan_report.md``) writes through ``<out>.tmp``
      opened with ``open(..., "w")``. The learner plants ``<run>/plan_report.md.tmp`` as a symlink to a root-owned
      file; does root write the report into it?
Run directories and victims are short directories under /tmp; removed at exit.

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/g2harness/g15_operator_cli_root.py
"""
import grp
import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
import time

from _g2 import ROOT, Report  # noqa: I001

R = Report("g15_operator_cli_root")
if os.geteuid() != 0:
    R.note("not root: skipped")
    R.done()

from cam_live.harness.auth import launcher as L  # noqa: E402

CAMDIR = os.path.dirname(ROOT)                       # /workspace/camlive


def run_as(user, code, *args):
    pw = pwd.getpwnam(user)
    return subprocess.run([sys.executable, "-I", "-c", code, *map(str, args)], user=pw.pw_uid, group=pw.pw_gid,
                          extra_groups=[grp.getgrnam("camlive").gr_gid], cwd="/", env={"PATH": "/usr/bin:/bin"},
                          capture_output=True, text=True, timeout=60)


def cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "src")
    return subprocess.run([sys.executable, "-m", "cam_live.runtime.cli", *args], cwd=CAMDIR, env=env,
                          capture_output=True, text=True, timeout=300)


D = tempfile.mkdtemp(prefix="g2hcl", dir="/tmp")
os.chmod(D, 0o755)
VDIR = tempfile.mkdtemp(prefix="g2hvd", dir="/tmp")         # root 0700: stands in for a root-only directory
os.chmod(VDIR, 0o700)
VFILE_DIR = tempfile.mkdtemp(prefix="g2hvf", dir="/tmp")
VFILE = os.path.join(VFILE_DIR, "precious")
open(VFILE, "w").write("precious root-owned content\n")
os.chmod(VFILE, 0o600)
try:
    L.prepare_run_dir(D, True, dict(L.DEFAULT_ROLES), None, None)     # the launcher's layout of a separated run
    st = os.stat(D)
    R.note(f"run directory mode {oct(st.st_mode & 0o7777)} group {grp.getgrgid(st.st_gid).gr_name}")
    # E1 -------------------------------------------------------------------------------------------------------
    plant = os.path.join(D, "learner", "fakeckpt")
    ups = "/".join([".."] * 6)
    code = ("import json, os, sys\nd, plant, ident = sys.argv[1], sys.argv[2], sys.argv[3]\n"
            "os.makedirs(plant, exist_ok=True)\nopen(os.path.join(plant, 'authorized_keys'), 'w').write("
            "'ssh-ed25519 AAAAG2HARNESSPROBEKEY learner@cam\\n')\n"
            "json.dump({'status': 'stopped', 'committed': {'ckpt': {'id': ident, 'path': plant}}}, "
            "open(os.path.join(d, 'state.json'), 'w'))\n")
    cp = run_as("camlearn", code, D, plant, f"x/{ups}{VDIR}")
    R.note(f"learner writes state.json and {plant}: {'ok' if cp.returncode == 0 else cp.stderr[-200:]}")
    out = cli("export-checkpoint", "--run-dir", D)
    R.note(f"root: cam-live export-checkpoint --run-dir R -> exit {out.returncode}: {(out.stdout + out.stderr)[-300:]!r}")
    landed = os.path.join(VDIR, "authorized_keys")
    R.check(not os.path.exists(landed), "export-checkpoint run as root does not copy learner-chosen files into a "
                                        "directory outside the run (checkpoint id / path taken from the learner's "
                                        "state.json)",
            f"{landed} exists: {os.path.exists(landed)}" + (f", content {open(landed).read()[:60]!r}, owner uid "
                                                             f"{os.stat(landed).st_uid}" if os.path.exists(landed) else ""))
    # E2 -------------------------------------------------------------------------------------------------------
    for p in (os.path.join(D, "plan_report.md"), os.path.join(D, "plan_report.md.tmp")):
        if os.path.lexists(p):
            os.unlink(p)
    cp = run_as("camlearn", "import os, sys\nos.symlink(sys.argv[1], sys.argv[2])\n", VFILE,
                os.path.join(D, "plan_report.md.tmp"))
    R.note(f"learner plants plan_report.md.tmp -> {VFILE}: {'ok' if cp.returncode == 0 else cp.stderr[-200:]}")
    out = cli("plan-report", "--run-dir", D)
    R.note(f"root: cam-live plan-report --run-dir R -> exit {out.returncode}: {(out.stdout + out.stderr)[-200:]!r}")
    content = open(VFILE).read()
    R.check(content.startswith("precious"), "plan-report run as root does not write through a symlink the learner "
                                            "planted", f"victim now starts {content[:60]!r}")
finally:
    for x in (D, VDIR, VFILE_DIR):
        shutil.rmtree(x, ignore_errors=True)
R.done()
