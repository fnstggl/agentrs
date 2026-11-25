"""Workstream C test configuration: slow learning tests run only with CAM_RUN_SLOW=1 (they were run once and their
numbers reported); everything else runs by default."""
import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("CAM_RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow learning test: set CAM_RUN_SLOW=1 to run")
    for it in items:
        if "slow" in it.keywords and "/world/" in str(it.fspath):
            it.add_marker(skip)
