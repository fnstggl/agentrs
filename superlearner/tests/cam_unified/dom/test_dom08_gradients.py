"""DOM-08 (verified experiential gradients throughout the core), dev variant: the real `dom_dev` stack (8 x 384 core,
frozen Qwen3-0.6B) on 4 Crucible rapid training worlds, observation-only, two IntegratedLearner updates through
`learn.integrated_e2e.run`. After one optimiser step, every DOM-08 group the stack builds must receive gradient from
the experiential objectives; groups the stack does not build yet are reported as "absent" (the 1B variant is run by
W1 with the same driver). No test double anywhere.

GPU + cached Qwen3 weights required (`-m gpu`); hold the shared GPU lock when running it by hand:
    flock /workspace/agentrs/runs/cam_unified/gpu_heavy.lock pytest -m gpu tests/cam_unified/dom/test_dom08_gradients.py
"""
import os
import sys

import pytest
import torch

CRUCIBLE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../crucible/src"))
if CRUCIBLE not in sys.path:
    sys.path.insert(0, CRUCIBLE)

pytestmark = [pytest.mark.gpu, pytest.mark.slow,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    from superlearner.cam_unified.learn import integrated_e2e as E
    return E.run(n_updates=3, seed=0, device="cuda", run_dir=str(tmp_path_factory.mktemp("dom08")))


def test_every_built_dom08_group_receives_gradient_after_one_step(report):
    from superlearner.cam_unified.learn.grad_probe import DOM08_REQUIRED
    chk = report["checks"]["dom08_nonzero_present_groups"]
    table = chk["table"]
    assert chk["update"] == 2 and set(table) == set(DOM08_REQUIRED)
    assert [g for g, v in table.items() if v == "zero"] == [], table
    for g in ("core.block0", "core.block_mid", "core.block_last", "sem_proj", "reasoner", "beliefs",
              "world_model", "exploration", "policy", "v_int"):
        assert table[g] == "nonzero", (g, table)


def test_the_actor_reaches_every_core_block_and_psi_gets_nothing(report):
    tab = report["grad_tables"]["2"]["norm"]
    for g in ("core.block0", "core.block_mid", "core.block_last"):
        assert tab["actor"][g] > 0 and tab["world"][g] > 0 and tab["value"][g] > 0, g
    for u in report["per_update"]:
        assert u["actor_psi_grad_norm"] in (None, 0.0)
    assert report["integrity"]["env_reward_nonzero"] == 0 and report["integrity"]["fresh_reward_nonzero"] == 0


def test_survivors_go_to_vtrace_and_the_learner_is_observation_only(report):
    assert report["checks"]["stale_routed_to_vtrace"]["passed"]
    pu = report["per_update"]
    assert pu[0]["policy"]["n_stale_steps"] == 0 and pu[1]["policy"]["n_stale_steps"] > 0
    assert all(u["reward"]["mode"] == "observation_only" and u["reward"]["condition"] == "D0" for u in pu)
