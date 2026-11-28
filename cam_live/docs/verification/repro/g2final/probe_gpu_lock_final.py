"""G2-final probe: the shared GPU lock after 7265843 / 59807fc (never created, read-only for service uids, fails
closed) — every lock path, in an archive copy and for the service uids. Nothing here takes the machine's lock.

    G2F_SRC=<archive>/cam_live/src .venv/bin/python cam_live/docs/verification/repro/g2final/probe_gpu_lock_final.py
    (as root; part B runs children as camlearn / cambrowser with this tree's code, which equals fdab7d3 in cam_live/src)

A  code under test = G2F_SRC (an archive copy, where <repo>/runs/gpu_heavy.lock does not exist), CAM_LIVE_GPU_LOCK
   unset, one fresh process per check: every lock path refuses and creates no lock file
   (nn.gpu.gpu_lock, runtime.gpu_cycle.GpuLease, runtime.cli.gpu_lock, harness.visual._shared_gpu_lock,
   harness.grounding.bench._gpu_lock); doctor's gpu_lock check fails for live and names CAM_LIVE_GPU_LOCK;
   with CAM_LIVE_GPU_LOCK set the path is the machine's lock (same inode).
B  a scratch lock file root:root 0664 (like the machine's); root holds an exclusive flock; children as camlearn and
   cambrowser try to take it with the lock paths their processes use: they must WAIT (exclusion with root), then get
   it once root releases; the file's owner / mode are left as they were.
"""
import fcntl
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2fcommon import HERE, Report, SRC  # noqa: E402

R = Report("probe_gpu_lock_final")
MACHINE = "/workspace/agentrs/runs/cam_unified/gpu_heavy.lock"
PY = sys.executable
TREE_SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "src"))
env0 = {k: v for k, v in os.environ.items() if k != "CAM_LIVE_GPU_LOCK"}
env0["PYTHONPATH"] = SRC

PATH_CODE = "from cam_live.config import GPU_LOCK_PATH; print(GPU_LOCK_PATH)"
default = subprocess.run([PY, "-c", PATH_CODE], capture_output=True, text=True, env=env0, cwd="/").stdout.strip()
R.check("A0 in the copy the default lock path does not exist (no symlink in an archive)", not os.path.exists(default),
        {"GPU_LOCK_PATH": default})

PATHS = {
    "nn.gpu.gpu_lock": "from cam_live.nn.gpu import gpu_lock\nwith gpu_lock(timeout_s=1, poll_s=0.1):\n    pass",
    "runtime.gpu_cycle.GpuLease": "from cam_live.runtime.gpu_cycle import GpuLease\nfrom cam_live.config import "
                                  "GPU_LOCK_PATH\nGpuLease(GPU_LOCK_PATH, hold_s=60).acquire(lambda: True)",
    "runtime.cli.gpu_lock": "from cam_live.runtime.cli import gpu_lock\nwith gpu_lock(True, timeout_s=1):\n    pass",
    "harness.visual._shared_gpu_lock": "from cam_live.harness.visual import _shared_gpu_lock\n"
                                       "with _shared_gpu_lock(1) as got:\n    print('got', got)",
    "harness.grounding.bench._gpu_lock": "from cam_live.harness.grounding.bench import _gpu_lock\n"
                                         "with _gpu_lock(True, 1) as got:\n    print('got', got)",
}
for name, code in PATHS.items():
    p = subprocess.run([PY, "-c", code], capture_output=True, text=True, env=env0, cwd="/", timeout=120)
    err = (p.stderr.strip().splitlines() or [""])[-1]
    R.check(f"A {name}: refuses without the lock file and creates none", p.returncode != 0 and not os.path.exists(default)
            and "CAM_LIVE_GPU_LOCK" in p.stderr, {"rc": p.returncode, "error": err[:220],
                                                  "lock_file_created": os.path.exists(default)})
runs_dir = os.path.dirname(default)
R.note("A runs/ directory left in the copy by runtime.cli.gpu_lock (it still makes the lock's directory)",
       {"exists": os.path.isdir(runs_dir), "entries": sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []})
doc = ("import json\nfrom cam_live.config import CamLiveConfig\nfrom cam_live.runtime.doctor import harness_checks\n"
       "print(json.dumps([c for c in harness_checks(CamLiveConfig()) if c.get('name') == 'gpu_lock']))")
p = subprocess.run([PY, "-c", doc], capture_output=True, text=True, env=env0, cwd="/", timeout=300)
import json  # noqa: E402
try:
    chk = json.loads(p.stdout.strip().splitlines()[-1])
except Exception:  # noqa: BLE001
    chk = [{"raw": (p.stdout + p.stderr)[-400:]}]
c0 = chk[0] if chk else {}
R.check("A doctor: the gpu_lock check fails at level live and names CAM_LIVE_GPU_LOCK",
        c0.get("ok") is False and "live" in json.dumps(c0) and "CAM_LIVE_GPU_LOCK" in json.dumps(c0), c0)
p = subprocess.run([PY, "-c", PATH_CODE], capture_output=True, text=True,
                   env={**env0, "CAM_LIVE_GPU_LOCK": MACHINE}, cwd="/")
there = p.stdout.strip()
R.check("A with CAM_LIVE_GPU_LOCK set, the copy's lock is the machine's lock (same inode; nothing locked here)",
        os.path.exists(there) and os.stat(there).st_ino == os.stat(MACHINE).st_ino, {"GPU_LOCK_PATH": there})

# ---- B service uids ----------------------------------------------------------------------------------------------
if os.geteuid() != 0:
    R.note("B skipped", "needs root")
    sys.exit(R.done())
d = tempfile.mkdtemp(prefix="g2f_lock_", dir=HERE)          # inside this tree: the service uids can traverse it
os.chmod(d, 0o755)
lock = os.path.join(d, "gpu_heavy.lock")
open(lock, "w").close()
os.chown(lock, 0, 0)
os.chmod(lock, 0o664)
CHILD = {
    "camlearn": ("import os, sys, time\nfrom cam_live.runtime.gpu_cycle import GpuLease\nfrom cam_live.nn.gpu import "
                 "gpu_lock, GpuUnavailable\nL = os.environ['CAM_LIVE_GPU_LOCK']\nprint('ready', flush=True)\n"
                 "t = time.time()\n"
                 "w = GpuLease(L, hold_s=60, poll_s=0.1).acquire(lambda: time.time() - t > 2)\n"
                 "print('lease_while_root_holds', w, flush=True)\n"
                 "try:\n    with gpu_lock(timeout_s=1.5, poll_s=0.1):\n        print('gpu_lock_while_root_holds got')\n"
                 "except GpuUnavailable as e:\n    print('gpu_lock_while_root_holds refused', e, flush=True)\n"
                 "while not os.path.exists(os.path.join(os.environ['G2F_D'], 'released')):\n    time.sleep(0.1)\n"
                 "t = time.time()\nlease = GpuLease(L, hold_s=60, poll_s=0.1)\n"
                 "w = lease.acquire(lambda: time.time() - t > 20)\nprint('lease_after_release', w is not None)\n"
                 "lease.release()\n"),
    "cambrowser": ("import os, time\nfrom cam_live.harness.visual import _shared_gpu_lock\nprint('ready', flush=True)\n"
                   "with _shared_gpu_lock(2) as got:\n    print('visual_while_root_holds', got, flush=True)\n"
                   "while not os.path.exists(os.path.join(os.environ['G2F_D'], 'released')):\n    time.sleep(0.1)\n"
                   "with _shared_gpu_lock(20) as got:\n    print('visual_after_release', got)\n"),
}
holder = open(lock, "r+")
fcntl.flock(holder, fcntl.LOCK_EX)
procs, first = {}, {}
for user, code in CHILD.items():
    pw = pwd.getpwnam(user)

    def demote(pw=pw, user=user):
        os.setgroups(os.getgrouplist(user, pw.pw_gid))
        os.setgid(pw.pw_gid)
        os.setuid(pw.pw_uid)
    procs[user] = subprocess.Popen([PY, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   preexec_fn=demote, cwd="/",
                                   env={"PATH": "/usr/bin:/bin", "PYTHONPATH": TREE_SRC, "CAM_LIVE_GPU_LOCK": lock,
                                        "G2F_D": d, "HOME": d})
for user, p in procs.items():
    first[user] = p.stdout.readline().strip()           # imports done: the child now tries while root holds it
time.sleep(6.0)
fcntl.flock(holder, fcntl.LOCK_UN)
holder.close()
open(os.path.join(d, "released"), "w").close()
outs = {}
for user, p in procs.items():
    try:
        o, e = p.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        p.kill()
        o, e = p.communicate()
    outs[user] = {"rc": p.returncode, "first": first[user], "out": o.strip().splitlines(),
                  "err": e.strip().splitlines()[-2:]}
st = os.stat(lock)
cl, cb = outs["camlearn"]["out"], outs["cambrowser"]["out"]
R.check("B1 camlearn (separated learner): GpuLease and nn.gpu.gpu_lock WAIT while root holds the lock (same lock)",
        "lease_while_root_holds None" in cl and any(x.startswith("gpu_lock_while_root_holds refused") for x in cl),
        outs["camlearn"])
R.check("B2 camlearn takes the lease once root releases", "lease_after_release True" in cl, outs["camlearn"])
R.check("B3 cambrowser (visual executor's model lock): waits while root holds it, gets it after the release",
        "visual_while_root_holds False" in cb and "visual_after_release True" in cb, outs["cambrowser"])
R.check("B4 the lock file is left root:root 0664", (st.st_uid, st.st_gid, st.st_mode & 0o777) == (0, 0, 0o664),
        {"uid": st.st_uid, "gid": st.st_gid, "mode": oct(st.st_mode & 0o777)})
shutil.rmtree(d, ignore_errors=True)
sys.exit(R.done())
