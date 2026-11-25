"""Slow tests of this directory (the 1/4/8-actor throughput benchmark, the full lag sweep) run only when asked for:
``-m slow`` or CAM_RUN_SLOW=1. The hook only touches items collected from this directory, so slow tests elsewhere in
the same session keep their own policy."""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


def pytest_collection_modifyitems(config, items):
    if os.environ.get("CAM_RUN_SLOW") == "1" or "slow" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="slow benchmark: run with -m slow or CAM_RUN_SLOW=1")
    for it in items:
        path = os.path.abspath(str(getattr(it, "path", None) or it.fspath))
        if "slow" in it.keywords and os.path.commonpath([path, _HERE]) == _HERE:
            it.add_marker(skip)
