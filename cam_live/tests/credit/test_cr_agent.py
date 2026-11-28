"""CR, integrated agent (slow): the real ``CamLiveAgent`` driven by the real ``LiveRuntime`` through one long open
LabWorld objective (debug scale, CPU) produces learned RUDDER / COCOA credit for matured segments *before the objective
closes*, revises it as outcomes and the estimators change, and every learning update evaluates the experiments with the
pooled rule's reason logged. No agent.py change is needed for this (the segment split comes from the configuration,
``credit.learner.val_split: segment``, which the agent honours)."""
import pytest

from cam_live.credit.cr_agent_eval import run


@pytest.mark.slow
def test_integrated_agent_gets_learned_credit_inside_one_open_objective(tmp_path):
    import torch
    torch.set_num_threads(2)
    res = run("simple", 0, 20, str(tmp_path / "w"), segment=4, split="segment")
    assert res["summary"]["decisions"] >= 16
    keys = res["credit_keys"]
    assert keys.get("rudder", 0) >= 8 and keys.get("rudder_contribution", 0) >= 8      # >= 2 matured segments
    assert keys.get("cocoa", 0) >= 1
    assert max(res["credit_versions_first64"].values()) >= 2                           # earlier credit revised
    assert res["default_at_end"] == "simple"                                           # nothing validated on 20
    assert res["n_selection_rows"] >= 1
    assert all(r["reason"] for r in res["selection_log"])
