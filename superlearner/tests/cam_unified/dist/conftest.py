"""Slow tests (the 1/4/8-actor throughput benchmark) run only when asked for: ``-m slow`` or CAM_RUN_SLOW=1."""
import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("CAM_RUN_SLOW") == "1" or "slow" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="slow benchmark: run with -m slow or CAM_RUN_SLOW=1")
    for it in items:
        if "slow" in it.keywords:
            it.add_marker(skip)
