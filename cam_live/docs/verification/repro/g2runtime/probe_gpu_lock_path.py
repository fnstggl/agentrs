"""G2-runtime probe (round 2): is the default GPU lock the machine's shared lock for every process that must coordinate?

    .venv/bin/python cam_live/docs/verification/repro/g2runtime/probe_gpu_lock_path.py      (as root)

At the final freeze ``config.GPU_LOCK_PATH`` defaults to ``<repo>/runs/gpu_heavy.lock`` (G2I-10); in this worktree that
path is an untracked symbolic link to the other tenant's lock (/workspace/agentrs/runs/cam_unified/gpu_heavy.lock).
flock excludes per file (inode), so two processes coordinate only when they lock the same inode. Nothing here holds
the shared lock: inodes are compared with stat; the learner-uid check aborts its wait at once.
  1  in this tree: GPU_LOCK_PATH is the machine lock (same inode) — runtime lease, nn.gpu.gpu_lock, cam-live, tests
  2  in a `git archive` copy of the same commit (how the validations and verifiers run frozen code): is it still?
  3  as the learner uid of separated mode (camlearn), can the runtime's GpuLease take the default lock?
"""
import os
import pwd
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2common import PROJECT, SRC, Report, scratch  # noqa: E402

R = Report("probe_gpu_lock_path")
MACHINE = "/workspace/agentrs/runs/cam_unified/gpu_heavy.lock"
REPO = os.path.dirname(PROJECT)


def ino(p):
    try:
        return os.stat(p).st_ino
    except OSError:
        return None


env = {k: v for k, v in os.environ.items() if k != "CAM_LIVE_GPU_LOCK"}
code = "from cam_live.config import GPU_LOCK_PATH; print(GPU_LOCK_PATH)"
here = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**env, "PYTHONPATH": SRC},
                      cwd="/").stdout.strip()
R.check("1 in this tree the default lock is the machine's shared lock (same inode)", ino(here) == ino(MACHINE) is not None,
        {"GPU_LOCK_PATH": here, "inode": ino(here), "machine_lock_inode": ino(MACHINE),
         "is_symlink": os.path.islink(here)})

copy = scratch("archive")
head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
ar = subprocess.run(f"git -C {REPO} archive {head} cam_live/src | tar -x -C {copy}", shell=True, capture_output=True)
there = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={**env, "PYTHONPATH": os.path.join(copy, "cam_live", "src")}, cwd="/").stdout.strip()
R.check("2 in a git-archive copy of the same commit the default lock is still the machine's lock",
        ino(there) == ino(MACHINE) is not None,
        {"GPU_LOCK_PATH": there, "exists": os.path.exists(there), "inode": ino(there), "machine_lock_inode": ino(MACHINE),
         "note": "runs/gpu_heavy.lock is untracked, so an archive (or a fresh clone) has no link; the first lease "
                 "creates a private lock file there"})

if os.geteuid() == 0:
    pw = pwd.getpwnam("camlearn")
    child = (
        "import os, sys, signal\n"
        "signal.alarm(20)\n"
        "from cam_live.runtime.gpu_cycle import GpuLease\n"
        "from cam_live.config import GPU_LOCK_PATH\n"
        "try:\n"
        "    w = GpuLease(GPU_LOCK_PATH, hold_s=840).acquire(lambda: True)\n"
        "    print('acquired-or-aborted', w)\n"
        "except Exception as e:\n"
        "    print('error', type(e).__name__, e)\n")

    def demote():
        os.setgroups([g for g in os.getgrouplist("camlearn", pw.pw_gid)])
        os.setgid(pw.pw_gid)
        os.setuid(pw.pw_uid)
    out = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, preexec_fn=demote,
                         env={"PYTHONPATH": SRC, "PATH": os.environ.get("PATH", "")}, cwd="/", timeout=60)
    res = (out.stdout + out.stderr).strip().splitlines()[-1:] or [""]
    R.check("3 the learner uid (camlearn) can take the default lock with the runtime's GpuLease (the learner process "
            "takes it in separated mode when the backbone is on CUDA: loop_service.run_service)",
            res[0].startswith("acquired-or-aborted"), {"camlearn": res[0][:300],
                                                        "lock_mode": oct(os.stat(here).st_mode & 0o777),
                                                        "lock_owner_uid": os.stat(here).st_uid})
else:
    R.note("3 skipped", "needs root")
sys.exit(R.done())
