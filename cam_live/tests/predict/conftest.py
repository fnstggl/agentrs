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
