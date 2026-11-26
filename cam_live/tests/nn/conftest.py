"""Fixtures for the neural-architecture tests (Workstream A)."""
import pytest
import torch

from cam_live.config import CamLiveConfig
from cam_live.contracts import ExperienceItem, ItemKind, Quantity
from cam_live.nn.model import build_model


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield


@pytest.fixture
def cfg():
    return CamLiveConfig.debug()


@pytest.fixture
def model(cfg):
    return build_model(cfg)


def make_items(n=5, t0=1000.0, dt=60.0, tag=""):
    kinds = [ItemKind.OBSERVATION, ItemKind.ACTION, ItemKind.OUTCOME, ItemKind.EVENT, ItemKind.TOOL]
    out = []
    for i in range(n):
        qs = [Quantity(f"metric_{i % 3}", float(i * 7 - 3))] if i % 2 == 0 else []
        out.append(ExperienceItem(kind=kinds[i % len(kinds)], text=f"{tag} item number {i} alpha beta",
                                  ts=t0 + dt * i, quantities=qs, ref=f"r{i}"))
    return out


@pytest.fixture
def items():
    return make_items
