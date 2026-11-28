"""The separated learner (uid camlearn) can take the shared GPU lock although the lock file is root:root 0664.

Measured 2026-11-28: ``open(lock, "a+")`` as camlearn raised PermissionError (it is not in the file's group), so a
separated live run (`cam-live run`) could not lease the GPU. Both lock paths (``nn.gpu.gpu_lock``, the runtime's
``GpuLease``) now open the file read-only when they may not write it; ``flock`` works on any open file.
"""
import os
import pwd
import shutil
import subprocess
import tempfile

import pytest

pytestmark = pytest.mark.privileged

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
PY = os.path.abspath(os.path.join(SRC, "..", "..", ".venv", "bin", "python"))

CHILD = r"""
import fcntl, os, sys
from cam_live.nn.gpu import gpu_lock
from cam_live.runtime.gpu_cycle import GpuLease
with gpu_lock(timeout_s=5.0, poll_s=0.1):
    print("gpu_lock held")
lease = GpuLease(os.environ["CAM_LIVE_GPU_LOCK"], hold_s=60.0)
waited = lease.acquire()
print("lease held", waited is not None)
lease.release()
"""


@pytest.mark.skipif(os.geteuid() != 0, reason="needs root to run a child as camlearn")
def test_camlearn_takes_a_root_owned_0664_lock_with_both_lock_paths():
    try:
        pwd.getpwnam("camlearn")
    except KeyError:
        pytest.skip("service user camlearn does not exist")
    d = tempfile.mkdtemp(prefix="cam_lock_uid_")
    try:
        os.chmod(d, 0o755)
        lock = os.path.join(d, "gpu_heavy.lock")
        open(lock, "w").close()
        os.chown(lock, 0, 0)
        os.chmod(lock, 0o664)
        env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": SRC, "CAM_LIVE_GPU_LOCK": lock, "HOME": d}
        p = subprocess.run(["su", "-s", "/bin/sh", "camlearn", "-c", f"cd / && {PY} -c '{CHILD}'"],
                           env=env, capture_output=True, text=True, timeout=120)
        assert p.returncode == 0, p.stderr[-2000:]
        assert "gpu_lock held" in p.stdout and "lease held True" in p.stdout, p.stdout + p.stderr[-1000:]
        st = os.stat(lock)
        assert (st.st_uid, st.st_mode & 0o777) == (0, 0o664)          # the lock file itself is left as it was
    finally:
        shutil.rmtree(d, ignore_errors=True)
