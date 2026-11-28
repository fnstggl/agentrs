"""ST: realised information value of an executed decision (master's item, 2026-11-28 12:10): what the action brought
into the context (its action item, its result's observations) is removed from the later decisions' contexts and the
world model's NLL of their matured outcomes is compared. Generic: no tool name is involved."""
from __future__ import annotations

import math

from cam_live.contracts import ActionResult, Observation
from test_st_strategy import _agent, _choose, _fixed_generator, _tools


def _step(agent, op, text):
    d = _choose(agent, _tools(), op)
    t0 = agent.clock.now()
    obs = [Observation(obs_id=f"obs_iv{agent.n_decisions}", ts=t0 + 0.5, kind="tool_result", text=text)]
    agent.record_result(d, ActionResult(action_id=d.chosen.action_id, status="ok", started=t0, ended=t0 + 1.0,
                                        observations=obs))
    agent.clock.advance(3.0)
    return d


def test_information_value_is_measured_once_from_later_predictions(tmp_path):
    agent = _agent(tmp_path)
    _fixed_generator(agent, [['{"operation": "internal.note", "arguments": {"text": "the page lists 3 groups"}}\n'
                              '{"operation": "web.get", "arguments": {"url": "https://a.example/"}}']])
    try:
        agent.ingest([Observation(obs_id="obs_0", ts=agent.clock.now(), kind="metric", text="start")])
        d0 = _step(agent, "internal.note", "internal.note: saved")
        _step(agent, "web.get", "page text: groups A, B and C")
        assert agent.iv_pending and agent.store.by_decision(d0.decision_id).exp_id in agent.iv_pending
        agent._information_value()
        rec0 = agent.store.by_decision(d0.decision_id)
        assert "reward_signals.information_value" not in rec0.decision.diagnostics     # needs 2 later decisions
        _step(agent, "web.get", "page text: group A has 40 members")
        agent._information_value()
        rec0 = agent.store.by_decision(d0.decision_id)
        iv = rec0.decision.diagnostics.get("reward_signals.information_value")
        assert iv is not None and math.isfinite(iv)
        assert rec0.exp_id not in agent.iv_pending                                        # measured once
        log = [x for x in agent.iv_log if x["exp_id"] == rec0.exp_id]
        assert log and log[0]["operation"] == "internal.note" and len(log[0]["later"]) == 2
        assert agent.status()["information_value"]["measured"] >= 1
        # the reward assembler does not read it yet (bootstrap input: change request ST-3 to RW)
        from cam_live.reward.assemble import stored_signals
        assert stored_signals(rec0.decision.diagnostics)["information_value"] == iv
    finally:
        agent.close()


def test_information_value_is_off_when_disabled(tmp_path):
    agent = _agent(tmp_path, information_value=False)
    _fixed_generator(agent, [['{"operation": "internal.note", "arguments": {"text": "n"}}']])
    try:
        _step(agent, "internal.note", "saved")
        assert agent.iv_pending == []
    finally:
        agent.close()


def test_information_value_reaches_the_bootstrap_reward_when_rw_applies_st3(tmp_path):
    """ST-3 (change request to RW): the annotation is a bootstrap input like learning_progress. Skipped until applied."""
    import pytest
    import torch
    from cam_live.reward.bootstrap import BootstrapReward, BootstrapWeights
    if not hasattr(BootstrapWeights(), "information_value"):
        pytest.skip("ST-3 not applied in reward/")
    from cam_live.reward.assemble import SignalAssembler
    agent = _agent(tmp_path)
    _fixed_generator(agent, [['{"operation": "internal.note", "arguments": {"text": "the page lists 3 groups"}}\n'
                              '{"operation": "web.get", "arguments": {"url": "https://a.example/"}}']])
    try:
        d0 = _step(agent, "internal.note", "internal.note: saved")
        _step(agent, "web.get", "page text: groups A, B and C")
        _step(agent, "web.get", "page text: group A has 40 members")
        agent._information_value()
        rec0 = agent.store.by_decision(d0.decision_id)
        iv = rec0.decision.diagnostics["reward_signals.information_value"]
        asm = SignalAssembler().assemble([rec0], agent.store)
        assert float(asm.information_value[0]) == iv
        out = BootstrapReward()(learning_progress=torch.zeros(1, dtype=torch.float64),
                                information_value=asm.information_value)
        assert abs(float(out.components["information_value"][0]) - math.tanh(iv)) < 1e-9
        out0 = BootstrapReward()(learning_progress=torch.zeros(1, dtype=torch.float64))
        assert float(out0.components["information_value"][0]) == 0.0
    finally:
        agent.close()
