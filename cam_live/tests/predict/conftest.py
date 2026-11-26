"""Local fixtures for tests/predict (Workstream B)."""
import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    yield


@pytest.fixture
def gen():
    return torch.Generator().manual_seed(1234)


_MEASURED = {}


@pytest.fixture
def measure():
    """Record a measured number. With CAM_LIVE_B_MEASURE=<path> the collected numbers are written there as JSON
    at the end of the session (used to produce docs/evidence/B/measured.json)."""
    def _rec(name, value):
        _MEASURED[name] = value
    return _rec


def pytest_sessionfinish(session, exitstatus):
    import json
    import os
    path = os.environ.get("CAM_LIVE_B_MEASURE")
    if path and _MEASURED:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(dict(sorted(_MEASURED.items())), f, indent=1, sort_keys=True)
