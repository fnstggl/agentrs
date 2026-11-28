"""G2-final: run G2-runtime's probe_gpu_lease.py unchanged except that its scratch lock file is created first.

    G2RT_SCRATCH=<scratch> .venv/bin/python cam_live/docs/verification/repro/g2final/run_gpu_lease_prelock.py <probe>

Since 7265843 every lock path refuses a lock file that does not exist (it is never created); probe_gpu_lease.py opens
its scratch lock only later (in free()), so as written its first acquire() fails closed. This runner inserts
``open(LOCK, "a").close()`` right after LOCK is defined and executes the probe as its own __main__ (same sys.path).
"""
import sys

path = sys.argv[1]
src = open(path).read()
anchor = 'LOCK = os.path.join(base, "gpu.lock")\n'
assert anchor in src, "probe changed: anchor not found"
src = src.replace(anchor, anchor + 'open(LOCK, "a").close()                 # G2-final: the lock file exists\n', 1)
sys.argv = [path]
g = {"__name__": "__main__", "__file__": path}
exec(compile(src, path, "exec"), g)
