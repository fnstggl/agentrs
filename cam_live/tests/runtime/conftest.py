"""Local fixtures for the runtime tests (workstream F)."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def pytest_collection_modifyitems(config, items):
    expr = config.getoption("-m") or ""
    for item in items:
        for mark in ("privileged", "browser", "network"):
            if mark in item.keywords and mark not in expr:
                item.add_marker(pytest.mark.skip(reason=f"run with -m {mark}"))


@pytest.fixture
def lab(tmp_path):
    """A LabWorld run: (runtime, agent, world, policy, verifier, clock, run_dir)."""
    from rt_helpers import make_lab
    return make_lab(str(tmp_path / "run"))
