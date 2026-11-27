"""Fixtures for the computer-environment tests (workstream EB). Helpers live in ``eb_support``."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from eb_support import StagingSites, SimClock, TaskSpec  # noqa: E402


def pytest_collection_modifyitems(config, items):
    """browser / privileged / network tests run only when selected with -m (default suite: CPU, fast, no Chromium)."""
    expr = config.getoption("-m") or ""
    for item in items:
        if not str(item.fspath).startswith(HERE + os.sep):
            continue                                    # only this directory's tests are gated here
        gate = next((m for m in ("privileged", "network", "browser") if m in item.keywords), None)   # strictest first
        if gate is not None and gate not in expr:
            item.add_marker(pytest.mark.skip(reason=f"run with -m {gate}"))


@pytest.fixture(scope="session")
def sites():
    clock = SimClock()
    with StagingSites(clock=clock) as s:
        s.clock = clock
        yield s


@pytest.fixture
def task():
    return TaskSpec(objective_id="obj_eb_test", objective="exercise the computer environment", mode="staging")
