"""Frozen semantic agents (baselines 5-7) through the OpenWorld adapter with a real frozen Qwen3 model
(cfg.sem_test_model = Qwen3-0.6B in tests)."""
from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.gpu]
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import crucible  # noqa: E402
crucible.load_all_modules()
from crucible.research import llm_eval as L                      # noqa: E402

from superlearner.cam_unified.config import make_config          # noqa: E402
from superlearner.cam_unified.interfaces import ActionSource      # noqa: E402
from superlearner.cam_unified.envs.baselines import ExperienceMemory, FrozenSemanticAgent, make_backend   # noqa: E402
from superlearner.cam_unified.envs.harness import run_lifetime    # noqa: E402
from superlearner.cam_unified.envs.openworld import open_world    # noqa: E402
from superlearner.cam_unified.envs.openworld_eval import OpenWorldEvaluator   # noqa: E402


@pytest.fixture(scope="module")
def backend():
    be = make_backend(make_config("dev").sem_test_model, device="cuda", max_context=6144)
    yield be
    del be
    torch.cuda.empty_cache()


def goal():
    return L.open_economy_goal(world_family="ow_tiny", seed=1, overrides={"n_consumers": 800}, days=8)


def _run(agent, decisions):
    ad, op = open_world([goal()], seed=0)
    ev = OpenWorldEvaluator(op)
    res = run_lifetime(agent, ad, ev, max_decisions=decisions, think_s=600)
    ad.close()
    return res


def test_frozen_agent_with_tools_acts_through_adapter(backend):
    agent = FrozenSemanticAgent("tools", backend=backend, max_new_tokens=96)
    seen = []
    res = _run(agent, 3)
    u = res.accounting["agent"]
    assert u["calls"] == 3 and u["prompt_tokens"] > 3 * 1000 and u["gen_tokens"] > 0
    assert res.steps == 3
    pol = agent.policies[0]
    assert pol.interface == "v4" and pol.last_action_line is False
    assert len(pol.transcript) == 3 and all("prompt_tokens" in t for t in pol.transcript)
    assert agent.update() == {"updates": 0}                   # frozen
    assert agent.source is ActionSource.BASELINE


def test_memory_variant_persists_notes_across_episodes(backend, tmp_path):
    mem = ExperienceMemory()
    a1 = FrozenSemanticAgent("memory", backend=backend, max_new_tokens=96, memory=mem)
    _run(a1, 3)
    acted = sum(1 for t in a1.policies[0].transcript if t.get("action"))
    assert len(mem.notes) == acted                    # one note per executed action (rejected ones included)
    mem.add("market.browse {\"limit\": 5} -> ok: listings of record normalisation services", {"marker": True})
    path = str(tmp_path / "mem.json")
    a1.save(path)
    a2 = FrozenSemanticAgent("memory", backend=backend, max_new_tokens=96, record_prompts=True)   # e.g. a new process
    a2.load(path)
    assert len(a2.memory.notes) == acted + 1
    _run(a2, 1)
    pol = a2.policies[0]
    assert pol.retrieved and (acted in pol.retrieved or len(pol.retrieved) == min(6, acted + 1))
    user = pol.transcript[0]["messages"][-1]["content"]
    assert "Notes from your earlier experience" in user and "record normalisation services" in user


def test_matched_planning_variant_counts_candidates(backend):
    agent = FrozenSemanticAgent("planning", backend=backend, max_new_tokens=96, n_candidates=3)
    res = _run(agent, 2)
    u = res.accounting["agent"]
    assert u["candidates"] == 6 and u["calls"] == 6              # N generations per decision
    assert u["score_forwards"] == u["valid_candidates"] or u["score_forwards"] <= u["valid_candidates"]
    for t in agent.policies[0].transcript:
        for c in t["candidates"]:
            assert 0.0 <= c["p_yes"] <= 1.0


def test_frozen_llm_as_independent_participant(backend):
    """W05: a frozen language agent is one of the independently controlled participants of a world (driven by the
    operator through actor operations), next to the world's own minds, firms and consumers."""
    from crucible.research.llm_eval import EvalLLMPolicy
    llm = EvalLLMPolicy(backend=backend, interface="v4", max_new_tokens=64, tool_chars=4000)
    ad, op = open_world([goal()], seed=0, participants=[("llm-03", llm, 100_000)])
    try:
        ad.reset()
        ad.advance_time([3 / 24])                         # three operator-driven participant turns
        pol = op.slots[0].participants[0][1]
        assert len(pol.transcript) == 3 and pol.stats["calls"] == 3
        w = op.world(0)
        assert w.principal("p:llm-03") is not None
    finally:
        ad.close()
