"""Fixtures for the authorization-layer tests (workstream EA)."""
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from cam_live import contracts as C  # noqa: E402
from cam_live.harness.auth import PolicyService  # noqa: E402


def pytest_collection_modifyitems(config, items):
    """privileged tests create the service users and switch uid: only with ``-m privileged`` and as root."""
    expr = config.getoption("-m") or ""
    for item in items:
        if "privileged" in item.keywords:
            if "privileged" not in expr:
                item.add_marker(pytest.mark.skip(reason="run with -m privileged"))
            elif os.geteuid() != 0:
                item.add_marker(pytest.mark.skip(reason="needs root"))


@pytest.fixture
def clock():
    return C.SimClock()


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    return str(d)


@pytest.fixture
def svc(run_dir, clock):
    s = PolicyService(run_dir, clock=clock, fsync=False)
    yield s
    s.close()


@pytest.fixture
def shared_dir():
    """A directory other uids can reach (pytest's tmp_path lives under a 0700 directory)."""
    d = tempfile.mkdtemp(prefix="camlive-ea-", dir="/tmp")
    os.chmod(d, 0o755)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def service_users():
    from cam_live.harness.auth.launcher import _ensure_users
    return _ensure_users()
