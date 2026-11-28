"""Fixtures for the AU tests (operational autonomy: standing authorization, scoped credentials, sandboxed package
installation, the capability audit of the 2026-11-28 amendment)."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("", "../auth", "../env"):
    p = os.path.normpath(os.path.join(HERE, sub))
    if p not in sys.path:
        sys.path.insert(0, p)

from cam_live import contracts as C  # noqa: E402
from cam_live.harness.auth import PolicyService  # noqa: E402

_OPT_IN = ("browser", "privileged", "network")


def pytest_collection_modifyitems(config, items):
    """browser / privileged tests run only when named with -m; privileged ones need root."""
    expr = config.getoption("-m") or ""
    for item in items:
        if not str(item.fspath).startswith(HERE):
            continue
        for mark in _OPT_IN:
            if mark in item.keywords and mark not in expr:
                item.add_marker(pytest.mark.skip(reason=f"run with -m {mark}"))
        if "privileged" in item.keywords and os.geteuid() != 0:
            item.add_marker(pytest.mark.skip(reason="needs root"))


@pytest.fixture
def clock():
    return C.SimClock()


@pytest.fixture
def svc(tmp_path, clock):
    d = tmp_path / "run"
    d.mkdir()
    s = PolicyService(str(d), clock=clock, fsync=False)
    yield s
    s.close()


@pytest.fixture
def shared_dir():
    """A directory other uids can reach (pytest's tmp_path lives under a 0700 directory)."""
    import shutil
    import tempfile
    d = tempfile.mkdtemp(prefix="camlive-au-", dir="/tmp")
    os.chmod(d, 0o755)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def service_users():
    from cam_live.harness.auth.launcher import _ensure_users
    return _ensure_users()
