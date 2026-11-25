"""Shared builders for workstream I integration tests (tiny configs, CPU, doubles from testing.py and
integration_doubles.py)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import pytest
import torch

from superlearner.cam_unified import integration_doubles as dbl
from superlearner.cam_unified.agent import CamUnifiedAgent
from superlearner.cam_unified.model import build_model
from superlearner.cam_unified.testing import tiny_config


def make_cfg(**over):
    base = dict(imag_branches=3, imag_samples=2, imag_horizons=(2,), mem_top_k=4)
    base.update(over)
    return tiny_config(**base)


class Stack:
    """model + agent + environment + evaluator + archive/recorder/replay/trainer doubles."""

    def __init__(self, cfg=None, B=2, seed=0, device="cpu", components=None, **agent_kw):
        torch.manual_seed(seed)
        self.cfg = cfg or make_cfg()
        comps = components if components is not None else dbl.double_components(self.cfg)
        self.model = build_model(self.cfg, components=comps).to(device)
        self.archive = dbl.DoubleArchive()
        self.recorder = dbl.ListTraceRecorder()
        self.replay = dbl.DoubleReplay(self.archive, seed=seed)
        self.trainer = dbl.DoubleTrainer(self.model, self.cfg)
        kw = dict(archive=self.archive, recorder=self.recorder, replay=self.replay, trainer=self.trainer,
                  index=self.model.retriever, seq_len=4, n_seqs=2, seed=seed)
        kw.update(agent_kw)
        self.agent = CamUnifiedAgent(self.model, **kw)
        self.env = dbl.ToyShopEnv(B=B, seed=seed)
        self.evaluator = dbl.ToyShopEvaluator(self.env)


@pytest.fixture
def stack():
    return Stack()
