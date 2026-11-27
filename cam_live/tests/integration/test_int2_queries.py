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


def test_backbone_program_takes_the_modality_of_the_program_tool(tmp_path):
    """Defect measured on the real configuration (S-6 sequence, MECH objective): the parser gives a program its first
    step's modality (``api`` for ``[http.get, internal.note]``); the authorization rule T-PROGRAM allows the program
    operation only as ``internal``, so such a program was refused before any step was judged. The agent now gives a
    composition the modality its own tool declares; its steps keep theirs, and the task's interfaces still bound them."""
    from cam_live.contracts import ActionSpec, Cost, Limits, TaskSpec, ToolDescriptor
    from cam_live.harness.auth import PolicyService
    from cam_live.predict.candidates import parse_actions
    tools = [ToolDescriptor(name="http.get", modality="api", description="GET a url", side_effects="read",
                            arg_schema={"type": "object", "properties": {"url": {"type": "string"}},
                                          "required": ["url"]}, est_cost=Cost(seconds=1.0)),
             ToolDescriptor(name="internal.note", modality="internal", description="write a note", side_effects="none",
                            arg_schema={"type": "object", "properties": {"text": {"type": "string"}},
                                          "required": ["text"]}),
             ToolDescriptor(name="program", modality="internal", description="steps", side_effects="none",
                            arg_schema={"type": "object"})]
    text = ('{"steps": [{"operation": "http.get", "arguments": {"url": "http://127.0.0.1:9/measure?x=1"}}, '
            '{"operation": "internal.note", "arguments": {"text": "measured"}}]}')
    progs = [a for a in parse_actions(text, tools, provenance="backbone", id_prefix="t").actions if a.steps]
    assert progs and progs[0].modality == "api"             # what the parser gives it
    task = TaskSpec(objective_id="o", objective="measure", interfaces=["api", "internal"], mode="staging",
                    limits=Limits(duration_s=600, max_decisions=5))
    def judged(action, name):
        p = PolicyService(str(tmp_path / name))                # fresh: a repeat would count as circumvention
        p.register_task(task, "operator")
        d = p.authorize(action, task, {})
        return d.verdict, " ".join(d.reasons) + " " + " ".join(d.policy_ids)
    verdict, why = judged(progs[0], "auth1")
    assert verdict == "deny" and "T-PROGRAM" in why             # refused as `api` before any step is judged
    clock = SimClock()
    agent = CamLiveAgent(agent_config(use_backbone=False, planner=False), str(tmp_path / "a"), clock=clock)
    try:
        agent.set_objective(task)
        (a,) = agent._feasible(progs, tools, clock.now(), {})
        assert a.modality == "internal" and [s.modality for s in a.steps] == ["api", "internal"]
        _verdict, why = judged(a, "auth2")
        assert "T-PROGRAM" not in why, why                     # the steps are judged now (a loopback address that is
                                                               # not a registered local service is still refused)
        # a step whose interface the task does not permit still removes the composition
        t2 = TaskSpec(objective_id="o2", objective="x", interfaces=["internal"], mode="staging",
                      limits=Limits(duration_s=600, max_decisions=5))
        agent.set_objective(t2)
        progs2 = [x for x in parse_actions(text, tools, provenance="backbone", id_prefix="u").actions if x.steps]
        inv: dict = {}
        assert agent._feasible(progs2 + [ActionSpec(action_id="w", operation="internal.note", modality="internal",
                                                    arguments={"text": "n"})], tools, clock.now(), inv) and \
            inv.get("interface_not_permitted") == 1
    finally:
        agent.close()
