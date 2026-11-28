"""G2I-18 / G2R-17: CAM Live never creates the shared GPU lock file (a private lock coordinates with nobody)."""
import os

import pytest

from cam_live.nn.gpu import GpuUnavailable, open_lock_file


def test_a_missing_lock_file_is_refused_not_created(tmp_path):
    p = str(tmp_path / "absent" / "gpu_heavy.lock")
    with pytest.raises(GpuUnavailable, match="CAM_LIVE_GPU_LOCK"):
        open_lock_file(p)
    assert not os.path.exists(p) and not os.path.exists(os.path.dirname(p))


def test_an_existing_lock_file_is_opened(tmp_path):
    p = tmp_path / "gpu_heavy.lock"
    p.touch()
    with open_lock_file(str(p)) as f:
        assert not f.closed
