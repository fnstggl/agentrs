"""Workstream E tests. Tests marked ``slow`` run only when selected explicitly (``-m slow``) or with CAM_RUN_SLOW=1,
so the default run of this directory stays a few minutes."""
import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("CAM_RUN_SLOW") == "1" or "slow" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="slow test: run with -m slow or CAM_RUN_SLOW=1")
    for it in items:
        if "slow" in it.keywords:
            it.add_marker(skip)
