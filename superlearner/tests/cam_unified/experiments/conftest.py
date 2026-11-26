"""Fixtures for the controlled-experiment tests (workstream X).

Tests that need the real stack load Qwen3-0.6B once per session (frozen, ~1.2 GB on the GPU) and skip without CUDA.
`@pytest.mark.slow` tests run only with ``-m slow`` or CAM_RUN_SLOW=1.
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test (run with -m slow or CAM_RUN_SLOW=1)")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("CAM_RUN_SLOW") == "1" or "slow" in (config.option.markexpr or ""):
        return
    skip = pytest.mark.skip(reason="slow: run with -m slow or CAM_RUN_SLOW=1")
    for it in items:
        if str(it.fspath).startswith(HERE) and "slow" in it.keywords:
            it.add_marker(skip)


@pytest.fixture(scope="session")
def cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the real semantic encoder")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def sem(cuda):
    from superlearner.cam_unified.experiments import stack as S
    from superlearner.cam_unified.semantic.encoder import QwenSemantic
    cfg = S.experiment_config()
    return QwenSemantic(cfg, model_name=cfg.sem_model, lora=False, device="cuda")


@pytest.fixture(scope="session")
def dev_tasks():
    """Two validated tasks from the development range (offset 100,000) with horizon > 64."""
    from superlearner.cam_unified.experiments import mechrun as M
    return M.eval_task_set(["scm_additive", "cw1_scm_delay_stock"], 1, offset=100_000)
