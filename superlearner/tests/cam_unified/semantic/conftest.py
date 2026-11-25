"""Shared fixtures for workstream-A tests: one Qwen3-0.6B (cfg.sem_test_model, bf16) per session."""
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
    config.addinivalue_line("markers", "slow: long-running benchmark (run explicitly with -m slow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("markexpr"):
        return
    skip = pytest.mark.skip(reason="slow benchmark; run with -m slow")
    for it in items:
        if "slow" in it.keywords and str(it.fspath).startswith(HERE):
            it.add_marker(skip)


@pytest.fixture(scope="session")
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Qwen tests")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def cfg():
    from superlearner.cam_unified.testing import tiny_config
    return tiny_config()


@pytest.fixture(scope="session")
def enc(cuda, cfg):
    from superlearner.cam_unified.semantic.encoder import QwenSemantic
    return QwenSemantic(cfg, test=True)


@pytest.fixture(scope="session")
def tok(enc, cfg):
    from superlearner.cam_unified.semantic.tokenizer import ObservationTokenizer
    return ObservationTokenizer(cfg, enc)


@pytest.fixture(scope="session")
def opts(cfg):
    from superlearner.cam_unified.semantic.grammar import GrammarOptions
    # short string budgets keep the untrained decoder's rambling cheap; validity does not depend on the budget
    return GrammarOptions(max_string_tokens=40, max_action_tokens=160, max_key_tokens=16)


@pytest.fixture(scope="session")
def gen(enc, tok, cfg, opts):
    from superlearner.cam_unified.semantic.actions import ActionGenerator
    torch.manual_seed(0)
    return ActionGenerator(cfg, enc, tok, opts=opts)


@pytest.fixture(scope="session")
def ow_tiny_tools():
    """The real Crucible ow_tiny catalogue through the actor API (WorldServer.available_tools)."""
    crucible = pytest.importorskip("crucible")
    crucible.load_all_modules()
    from crucible.api import WorldServer
    from crucible.worldgen import generate
    from superlearner.cam_unified.interfaces import ToolSchema
    srv = WorldServer("memory://", lease_ttl_s=None)
    wid = srv.create_world(generate({"family": "ow_tiny"}, 1))
    me = srv.join_world(wid, "nova", endowment=3_000_000)
    specs = srv.available_tools(wid, me)
    tools = [ToolSchema(t["name"], t.get("description", ""), t["inputSchema"],
                        cost_hint=None, duration_hint=(t.get("annotations") or {}).get("costSeconds", 0) / 86400.0)
             for t in specs]
    return tools
