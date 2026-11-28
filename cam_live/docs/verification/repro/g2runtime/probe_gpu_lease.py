"""G2-runtime probe: the GPU lease (runtime.gpu_cycle) — bounded holds, no learner work without the lock.

    .venv/bin/python cam_live/docs/verification/repro/g2runtime/probe_gpu_lease.py

The lock file is a scratch file (never the machine's shared GPU lock). LabWorld run from tests/runtime/rt_helpers.
  1  one long cycle (the learner's learn() takes 2.5 s; hold bound 1.0 s): the hold is not bounded by gpu_hold_s
  2  learner calls made while the lease is NOT held: an operator pause / export applied after a sleep (lock released)
     runs agent.save / agent.export through LiveRuntime._apply_commands before the before_cycle hook re-takes the lock
  3  the lock is released while the runtime sleeps (control)
"""
import fcntl
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2common import Report, scratch  # noqa: E402

from cam_live.diagnostics.world import LabConfig  # noqa: E402
from cam_live.runtime.control import OperatorControl  # noqa: E402
from cam_live.runtime.gpu_cycle import GpuLease, attach  # noqa: E402
from cam_live.runtime.loop import RuntimeConfig  # noqa: E402
from rt_helpers import make_lab  # noqa: E402

R = Report("probe_gpu_lease")
base = scratch("gpu")
LOCK = os.path.join(base, "gpu.lock")


def free():
    with open(LOCK, "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_UN)
            return True
        except BlockingIOError:
            return False


# ---- 1 one long cycle ----------------------------------------------------------------------------------------
L = make_lab(os.path.join(base, "run1"), lab_cfg=LabConfig(seed=3, target=100000), rt_cfg=RuntimeConfig(fsync=False))
lease = GpuLease(LOCK, hold_s=1.0, yield_s=0.2, poll_s=0.05, release_sleep_s=1e9)
lease.acquire()
attach(L.rt, lease)
orig = L.agent.learn
n = {"calls": 0}


def slow_learn(budget):
    n["calls"] += 1
    if n["calls"] == 3:
        time.sleep(2.5)                     # e.g. a slow learner call on a loaded machine
    return orig(budget)


L.agent.learn = slow_learn
L.rt.start()
t0 = time.time()
while time.time() - t0 < 8 and L.rt.status == "running":
    L.rt.step()
lease.release()
R.check("1 no hold of the shared lock is longer than gpu_hold_s (+ a margin of 0.5 s)",
        lease.stats["longest_hold_s"] <= lease.hold_s + 0.5,
        {"hold_s": lease.hold_s, "longest_hold_s": round(lease.stats["longest_hold_s"], 2), "yields": lease.stats["yields"]})
L.rt.close()

# ---- 2 learner calls without the lock --------------------------------------------------------------------------
L = make_lab(os.path.join(base, "run2"), lab_cfg=LabConfig(seed=3, target=100000), rt_cfg=RuntimeConfig(fsync=False))
lease = GpuLease(LOCK, hold_s=840.0, yield_s=0.0, poll_s=0.05, release_sleep_s=5.0)
lease.acquire()
attach(L.rt, lease)
unlocked = []
for name in ("save", "export_weights", "set_objective", "learn", "propose", "evaluate", "load"):
    fn = getattr(L.agent, name, None)
    if not callable(fn):
        continue

    def wrap(*a, _fn=fn, _name=name, **k):
        if not lease.held:
            unlocked.append(_name)
        return _fn(*a, **k)
    setattr(L.agent, name, wrap)
L.rt.start()
for _ in range(4):
    L.rt.step()
L.rt.sleeper.sleep_until(L.clock.now() + 3600.0, lambda: False)        # the runtime sleeps: the lease is given back
R.check("3 the lock is released while the runtime sleeps", not lease.held and free(), {"held": lease.held})
op = OperatorControl(L.run_dir)
op.export_checkpoint(os.path.join(base, "export"))
op.pause("operator pause")
L.rt.step()
R.check("2 no learner call runs while the shared GPU lock is not held (export / pause applied after a sleep)",
        not unlocked, {"learner calls without the lock": unlocked, "status": L.rt.status})
lease.release()
L.rt.close()
sys.exit(R.done())
