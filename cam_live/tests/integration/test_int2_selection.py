"""Selection and credit mechanisms changed after the LabWorld diagnosis (INT2; change requests S-5, S-6, INT2-1).

* rule ``policy`` adds ``value_weight`` x the standardised predicted goal value (net of cost) to the policy's logits:
  the recorded probabilities are exactly the softmax of the sum (with the exploration floor);
* in an objective whose environment attributes outcomes to decisions, the objective-metric level is no longer asked
  about every candidate (it would mix earlier decisions' consequences into a candidate's value), and the verified
  progress annotation of a decision is its own attributed outcome (0 for decisions without attribution there).
"""
from __future__ import annotations

import math

import torch

from cam_live.agent import CamLiveAgent
from cam_live.contracts import ActionSpec, CandidateEvaluation, SimClock
from int_helpers import Recorder, agent_config, run_lifetime


def _ev(i, logit, q, cost=0.0):
    return CandidateEvaluation(action=ActionSpec(action_id=f"a{i}", operation="x", modality="api"), value=logit,
                               extras={"q_goal": q, "net_cost": cost})


def test_value_prior_is_a_standardised_shift_of_the_logits():
    evs = [_ev(0, 0.1, 1.0), _ev(1, 0.0, 3.0), _ev(2, -0.1, 2.0, cost=1.0)]
    out = CamLiveAgent._with_value_prior(evs, 1.0)
    q = [1.0, 3.0, 1.0]
    mu = sum(q) / 3
    sd = math.sqrt(sum((x - mu) ** 2 for x in q) / 3)
    for e, o, x in zip(evs, out, q):
        assert abs(o.value - (e.value + (x - mu) / sd)) < 1e-12
        assert o.extras["policy_logit_only"] == e.value
    assert CamLiveAgent._with_value_prior([_ev(0, 0.0, 1.0), _ev(1, 0.0, 1.0)], 1.0)[0].value == 0.0   # no contrast


def test_selection_probabilities_include_the_value_prior(tmp_path):
    clock = SimClock()
    agent = CamLiveAgent(agent_config(use_backbone=False, planner=False, value_weight=2.0, selection_rule="policy"), str(tmp_path / "a"),
                         clock=clock)
    try:
        lt = run_lifetime(Recorder(agent), clock, str(tmp_path), 51, max_steps=14, max_decisions=8)
        recs = [r for r in agent.store.all() if r.result is not None and r.decision.exploration == "policy"]
        with_prior = [r for r in recs if (r.decision.diagnostics.get("selection") or {}).get("rule") == "policy+value"]
        assert with_prior, "no decision used the value prior"
        for r in with_prior:
            d = r.decision
            lg = torch.tensor([c.value for c in d.candidates], dtype=torch.float64)
            assert all("value_prior" in c.extras for c in d.candidates)
            sm = torch.softmax(lg, -1).tolist()
            pr = list(d.probabilities)
            i, j = max(range(len(sm)), key=lambda k: sm[k]), min(range(len(sm)), key=lambda k: sm[k])
            assert abs(sum(pr) - 1.0) < 1e-9
            if sm[i] - sm[j] > 1e-9:
                keep = (pr[i] - pr[j]) / (sm[i] - sm[j])                 # 1 - exploration floor
                eps = (1.0 - keep) / len(sm)
                assert all(abs(pr[k] - (keep * sm[k] + eps)) < 1e-6 for k in range(len(sm))), (pr, sm)
        # attribution: after the first attributed intervention the level query is no longer asked
        obj = lt.task.objective_id
        assert obj in agent.attributing
        later = [r for r in agent.store.trajectory(obj) if r.result is not None][-3:]
        goal = lt.task.success["metric"]
        assert all(not any(q.metric == goal for q in r.pending_queries) for r in later)
    finally:
        agent.close()
