"""Workstream C test setup: deterministic, CPU, quiet about scalar conversions of graph tensors in assertions."""
import pytest
import torch


@pytest.fixture(autouse=True)
def _seeded():
    torch.manual_seed(0)
    torch.set_num_threads(4)
    yield
