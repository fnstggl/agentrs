"""Workstream-Y capability tests: the real assembled CAM (Qwen3-0.6B semantic encoder, A-E components) on small
diagnostic environments. Slow tests (full training protocols; their committed numbers are in results/) run only with
-m slow or CAM_RUN_SLOW=1."""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

torch = pytest.importorskip("torch")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full capability protocol (run with -m slow or CAM_RUN_SLOW=1)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("markexpr") or os.environ.get("CAM_RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow capability protocol; run with -m slow")
    for it in items:
        if "slow" in it.keywords and str(it.fspath).startswith(HERE):
            it.add_marker(skip)


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _hf_cached(name: str) -> bool:
    home = os.environ.get("HF_HOME", "/workspace/.hf_home")
    return os.path.isdir(os.path.join(home, "hub", "models--" + name.replace("/", "--")))


@pytest.fixture(scope="session")
def semantic_ok():
    from superlearner.cam_unified.capabilities.stack import capability_config
    cfg = capability_config()
    if not _hf_cached(cfg.sem_test_model):
        pytest.skip(f"{cfg.sem_test_model} not in the local HF cache")
    return True
