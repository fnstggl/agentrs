"""Planning phase of the live objective (operator amendment 2026-11-27; prepared, not run).

* ``tasks/discord_100_planning.yaml`` holds the operator's objective verbatim (confirmed by its hash), no strategy,
  the interfaces structured_browser / code / api / internal, member_list false, publishing with approval, and only
  operator-specific REQUIRED items left unset;
* the learner receives the objective verbatim through ``learner_view``;
* what the backbone writes for ``internal.note`` (the plan, hypotheses) and ``internal.request_operator`` (a
  resource request with its justification) becomes a valid candidate the agent offers — open-ended generation is
  the only source of such text (the schema walker never invents strings).
"""
from __future__ import annotations

import json
import os

from cam_live.agent import CamLiveAgent
from cam_live.contracts import SimClock
from cam_live.runtime.tasks import check_runnable, learner_view, load_task, load_task_raw, strategy_findings
from int_helpers import agent_config

TASK = os.path.join(os.path.dirname(__file__), "..", "..", "tasks", "discord_100_planning.yaml")
AMENDMENT_OBJECTIVE = (
    "Your objective is to create and operate a genuinely useful Discord community that attracts and retains 100 "
    "distinct, voluntary human members. You have access to a blank Discord server, general-purpose internet research, "
    "browser interaction and sandboxed software development. Independently research opportunities, determine the "
    "community's niche and value proposition, and develop an initial strategy for attracting and retaining members. "
    "Identify which additional accounts, publishing channels, hosting services, permissions or resources would "
    "materially improve your ability to achieve the objective. Request these from the operator and explain their "
    "expected usefulness. Do not assume that your initial strategy is correct. Continually revise it through "
    "authorized experiments, observed outcomes and trial and error. Attract participants through genuine value, "
    "transparent communication and permitted publishing channels. Do not use spam, deceptive identities or "
    "artificial engagement. Record your hypotheses, experiments, predictions, outcomes, lessons and neural learning "
    "updates. The operator will provide authorized resources, not strategic direction. You are responsible for "
    "discovering how to accomplish the objective and learning from the experience.")


def test_planning_task_file():
    raw = load_task_raw(TASK)
    assert strategy_findings(raw) == []
    t = load_task(TASK, for_run=False)
    assert t.objective == AMENDMENT_OBJECTIVE and t.mode == "live"
    assert t.interfaces == ["structured_browser", "code", "api", "internal"]
    assert t.data_permissions["member_list"] is False and t.data_permissions["store_member_identifiers"] is False
    assert t.resources["publishing"]["approval_required"] is True
    problems = check_runnable(t)
    # only the operator's items can be missing (the operator filled them in on 2026-11-27: then nothing is)
    assert len(problems) <= 1 and all(p.startswith("unset items:") for p in problems), problems
    assert learner_view(t).objective == AMENDMENT_OBJECTIVE                           # CAM receives it verbatim


def test_backbone_notes_and_operator_requests_become_candidates(tmp_path):
    from cam_live.harness.internal import InternalExecutor
    t = load_task(TASK, for_run=False)
    lv = learner_view(t)
    clock = SimClock(start=1000.0)
    agent = CamLiveAgent(agent_config(use_backbone=True, planner=False), str(tmp_path / "a"), clock=clock, task=lv)
    ix = InternalExecutor(str(tmp_path / "ix"), clock=clock)
    written = [json.dumps({"operation": "internal.request_operator", "arguments": {
                   "kind": "resource", "title": "an example request",
                   "description": "what is requested", "justification": "why it is expected to be useful"}}),
               json.dumps({"operation": "internal.note", "arguments": {"text": "a hypothesis to test",
                                                                       "tags": ["hypothesis"]}})]
    agent.model.generate = lambda prompts, **kw: [["\n".join(written)] for _ in prompts]
    try:
        cands = agent.propose(ix.tools(), 8)
        bb = [c for c in cands if "backbone" in c.provenance.split("+")]
        ops = {c.operation for c in bb}
        assert ops == {"internal.request_operator", "internal.note"}, [c.render() for c in cands]
        assert all(ix.validate(c) == [] for c in bb)
        req = next(c for c in bb if c.operation == "internal.request_operator")
        assert req.arguments["justification"]
    finally:
        agent.close()
