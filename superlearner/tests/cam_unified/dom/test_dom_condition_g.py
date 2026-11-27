"""DOM Condition G (latent-state intervention) on the REAL dom_dev stack (Qwen3-0.6B, 8 x 384 core).

After a short imitation phase on the two-goal task (decision/condition_g.py; the goal text alone decides the right
tool, observations are identical across goals) the decision must run through CAM's latent: swapping (R_t, I_t)
between rows with different goals makes >= 90% of the executed (evaluated, selected, stamped) actions follow the
DONOR's goal, the tool distribution moves by KL > 0.1 nats when the intent is permuted across rows, and with I_t = 0
the tool distribution is the same for both goals. Before training the swap does not matter (KL << 0.1).
"""
import os

import pytest
import torch

from superlearner.cam_unified.config import make_preset

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def trained():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA and the cached Qwen3-0.6B")
    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from superlearner.cam_unified.capabilities.stack import build_stack
    from superlearner.cam_unified.decision import condition_g as cg
    torch.manual_seed(0)
    cfg = make_preset("dom_dev", declared_conditions=("demo-init:two_goal_oracle",))
    m = build_stack(cfg, device="cuda", include_builtins=True, exact_rows=True, seed=0).model
    before = cg.kl_under_permutation(m, B=8, seed=31)
    log = cg.imitate(m, steps=30, B=16, seed=1)
    return m, before, log


def test_imitation_learns_the_goal_to_tool_mapping(trained):
    from superlearner.cam_unified.decision import condition_g as cg
    m, before, log = trained
    assert log["history"][-1]["p_correct"] > 0.9
    g = torch.Generator(device="cuda").manual_seed(5)
    assert cg.accuracy(m, B=32, seed=41, generator=g) >= 0.9
    assert before["kl_tool_mean"] < 0.05                          # untrained: permuting I barely matters


def test_condition_g_swapped_latent_drives_the_executed_action(trained):
    from superlearner.cam_unified.decision import condition_g as cg
    m, _, _ = trained
    g = torch.Generator(device="cuda").manual_seed(6)
    r = cg.condition_g(m, B=16, seed=51, generator=g)
    assert r["no_swap_follow_own_goal"] >= 0.9
    assert r["swap_RI_follow_donor"] >= 0.9
    assert r["swap_I_only_follow_donor"] >= 0.9
    assert r["swap_R_only_after_intent_follow_own"] >= 0.9        # R reaches the action only through I (+ the Q)
    assert r["all_swapped_actions_stamped"]


def test_kl_under_intent_permutation_and_zero_intent(trained):
    from superlearner.cam_unified.decision import condition_g as cg
    m, _, _ = trained
    kl = cg.kl_under_permutation(m, B=16, seed=61)
    assert kl["kl_tool_min"] > 0.1
    z = cg.zero_intent_check(m, B=8, seed=71)
    assert z["max_prob_gap_between_goals_I_zero"] == 0.0
    assert z["max_prob_gap_between_goals_trained_I"] > 0.5        # the trained intent carries the goal
