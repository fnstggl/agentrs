"""The AGENTRS superlearner World adapter for Crucible goals (reward audit).

Relocated from crucible/tests/test_agents_llm.py with the adapter it tests (Crucible DECISION_LOG D-81): Crucible is
independent of CAM (Clarification 1), so the CAM 2.0 / superlearner adapters and their tests live on the AGENTRS side,
which imports Crucible. The test bodies are the Crucible 4.0 ones (c07098b) with the module path renamed and the 4.0
reward contract selected explicitly (Crucible OO.8: observation-only is the default)."""
from __future__ import annotations

import json  # noqa: F401
import os
import sys

import pytest

CRUCIBLE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "crucible", "src"))
if CRUCIBLE_SRC not in sys.path:
    sys.path.insert(0, CRUCIBLE_SRC)

import crucible  # noqa: E402

crucible.load_all_modules()


# ============================================================================================== superlearner World
def test_sl_world_adapter_runs_and_passes_reward_audit():
    from agentrs.sl import worlds as W
    from agentrs.sl.train import reward_audit
    from agentrs.sl.world import RewardContract, RewardSpec, Task
    import agentrs.sl.crucible_world as S
    assert "crucible/service" in W.FAMILIES and "crucible/revenue/deployed_service" in W.FAMILIES
    from crucible.goals.contracts import HELDOUT_GOAL_FAMILIES
    assert not any(f.split("/")[1] in HELDOUT_GOAL_FAMILIES for f in W.FAMILIES if f.startswith("crucible/"))
    w = W.make_world("crucible/service")
    w.initialize_world(Task("crucible/service", 3, "val"))
    sp = w.available_actions("0")
    first = sp.allowed([])
    assert first[0] == "wait" and "bank" in first and len(first) <= 24
    act = ["bank", "transfer"]
    while sp.allowed(act) != ["END"]:
        act.append(sp.allowed(act)[0])
    snap = w.save_world_state()
    w.execute("0", act)
    w.advance_time()
    c1, g1, d1 = w.observe_consequences("0"), w.evaluate_goal("0"), w.world.digest()
    w.restore_world_state(snap)
    w.execute("0", act)
    w.advance_time()
    assert (w.observe_consequences("0"), w.evaluate_goal("0"), w.world.digest()) == (c1, g1, d1)
    w.reset_world()                                  # same task, fresh episode from the task's start snapshot
    assert w.world.now == 0 and w.n_dec == 0
    res = reward_audit(["crucible/service"], RewardContract(RewardSpec()), max_steps=10, n=2)
    a = res["crucible/service"]
    assert a["passed"], a
    assert all(a[p]["ended"] == 1.0 for p in ("random", "stall", "terminate"))
