"""INT2: a decision's goal-relevant questions are all answerable, so the decision resolves.

Defect found by INT2 (measured with the meta-update test of ``test_agent_runtime.py``, which then failed at random):
the agent asks, before execution, the environment's declared measurements of the operation (learned templates,
instantiated with the candidate's action id). After execution the environment declares the same measurement under
its own query id. The runtime queues no learner copy of a question the environment covers
(``LiveRuntime._add_learner_queries``), but the agent kept its copy among the record's pending queries: that copy
was never answered, so every intervention stayed unresolved for the meta-learner (``MetaLearner._resolved``) and
for the verified-progress annotation, and a return-to-go query window stopped at the first intervention.
"""
from __future__ import annotations

import collections

from cam_live.agent import CamLiveAgent
from cam_live.contracts import OutcomeStatus, SimClock
from cam_live.reward.metalearner import MetaLearner
from int_helpers import Recorder, agent_config, run_lifetime

DONE = (OutcomeStatus.OBSERVED, OutcomeStatus.CENSORED, OutcomeStatus.UNOBSERVABLE)


def test_no_unanswerable_duplicate_question_and_decisions_resolve(tmp_path):
    clock = SimClock()
    agent = CamLiveAgent(agent_config(use_backbone=False, planner=False), str(tmp_path / "a"), clock=clock)
    try:
        lt = run_lifetime(Recorder(agent), clock, str(tmp_path), 31, max_steps=60, max_decisions=14,
                          lab_kw={"outcome_horizon_s": 1800.0, "mean_delay_s": (300.0, 900.0)})
        recs = [r for r in agent.store.trajectory(lt.task.objective_id) if r.result is not None]
        acted = [r for r in recs if r.result.expected_outcomes]
        assert acted, "no decision with measurements the environment declared"
        # templates were learned and asked: some decision predicted the environment's measurement beforehand
        assert any(any(p.query.metric == q.metric for e in r.decision.candidates
                       if e.action.action_id == r.decision.chosen.action_id for p in e.predictions)
                   for r in acted for q in r.result.expected_outcomes), "no template question was asked"
        for r in recs:
            goal = [q for q in r.pending_queries if q.goal_relevant]
            dup = [m for m, n in collections.Counter(q.metric for q in goal).items() if n > 1]
            assert not dup, (r.decision.chosen.operation, dup)
        # the horizons are short: before the lifetime ended, the early interventions' questions were all answered
        early = [r for r in acted if r.decision.ts + 2 * 1800.0 + r.duration_s < clock.now() - 3600.0]
        assert early, "no intervention early enough to be due"
        resolved = [MetaLearner._resolved(r) for r in early]
        assert all(resolved), [(r.decision.chosen.operation,
                                [(q.metric, q.query_id) for q in r.pending_queries
                                 if q.goal_relevant and q.query_id not in
                                 {o.query_id for o in r.outcomes if o.status in DONE}]) for r in early]
    finally:
        agent.close()
