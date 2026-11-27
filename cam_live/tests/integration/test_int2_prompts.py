"""Nothing the agent sends to the backbone carries a niche, strategy or plan (operator amendment 2026-11-27).

Every prompt template the deployed agent can send to the backbone is scanned with the task loader's strategy
scanner (``runtime.tasks.strategy_findings``): the proposer's instruction block and section headers
(``predict.candidates``), the framing the agent adds to its observation digest (``agent.surface_digest`` /
``CamLiveAgent._digest``) and the bridge-grounding scoring prompt (``credit.providers.BridgeGroundingProvider``).
The prompt the agent actually sends is captured and decomposed: once the JSON data lines (objective verbatim, tool
descriptors, observation digest) are removed, only the constant instruction text is left, and the data lines decode
to the operator's objective, the discovered tools and text derived from observations.
"""
from __future__ import annotations

import json

from cam_live.agent import PROPOSAL_INSTRUCTIONS, CamLiveAgent, compact_tool, surface_digest, surface_of
from cam_live.contracts import Limits, SimClock, TaskSpec
from cam_live.predict import candidates as C
from cam_live.runtime.tasks import strategy_findings
from int_helpers import agent_config
from test_int2_grounding import browser_tools, page

CONSTANTS = [PROPOSAL_INSTRUCTIONS.format(k=8), C.INSTRUCTIONS.format(k=8), C.H_OBJECTIVE, C.H_TOOLS, C.H_OBSERVATION,
             C.H_END, C.H_ANSWER]


def _clean(text: str) -> None:
    assert strategy_findings({"text": text}) == [], (text[:200], strategy_findings({"text": text}))


def test_constant_templates_carry_no_strategy():
    for t in CONSTANTS:
        _clean(t)
    # the framing the agent adds around observed data
    empty = {"obs_id": "obs_x", "ts": 0.0, "url": "", "title": "", "tab": "t1", "elements": [], "links": [],
             "tabs": [], "tables": [], "forms": []}
    _clean(surface_digest(empty))
    # the bridge-grounding scoring prompt (credit.providers.BridgeGroundingProvider): constant part only
    _clean("Action: \nObserved:")


def _agent(tmp_path, objective):
    task = TaskSpec(objective_id="obj_p", objective=objective, success={"metric": "m.count", "target": 1},
                    interfaces=["structured_browser", "internal"], limits=Limits(duration_s=600.0, max_decisions=9),
                    mode="staging")
    return CamLiveAgent(agent_config(use_backbone=True, planner=False, proposer_max_new_tokens=8),
                        str(tmp_path / "a"), clock=SimClock(start=1000.0), task=task)


def test_the_prompt_sent_is_constants_plus_objective_tools_and_observations(tmp_path):
    objective = ("Your objective is to create and operate a genuinely useful Discord community that attracts and "
                 "retains 100 distinct, voluntary human members.")
    agent = _agent(tmp_path, objective)
    sent = []
    real = agent.model.generate

    def capture(prompts, **kw):
        sent.extend(prompts)
        return real(prompts, **kw)
    agent.model.generate = capture
    try:
        o = page()
        o.ts = agent.clock.now()
        agent.ingest([o])
        agent.propose(browser_tools(), 8)
        assert len(sent) == 1
        lines = sent[0].splitlines()
        data = {}
        for h in (C.H_OBJECTIVE, C.H_TOOLS, C.H_OBSERVATION):
            i = lines.index(h)
            data[h] = json.loads(lines[i + 1])
            lines[i + 1] = ""
        rest = "\n".join(lines)
        for t in [PROPOSAL_INSTRUCTIONS.format(k=agent.settings.proposer_k)] + CONSTANTS[2:]:
            rest = rest.replace(t, "")
        assert not rest.strip(), rest[:300]                       # nothing but the constant instruction text
        assert data[C.H_OBJECTIVE] == objective                   # the objective verbatim
        names = {t.name: t for t in browser_tools()}
        assert data[C.H_TOOLS] and all(t["name"] in names for t in data[C.H_TOOLS])
        assert all(t == compact_tool(names[t["name"]]) for t in data[C.H_TOOLS])   # only the tools' own descriptors
        for t in data[C.H_TOOLS]:
            _clean(json.dumps(t))
        obs = data[C.H_OBSERVATION]
        surf = surface_of(o)
        assert obs.startswith(surface_digest(surf, max_chars=len(obs) + 10).splitlines()[0])
        # every element line of the digest is derived from the observation (ref, role, name)
        for e in surf["elements"]:
            if f"[{e['ref']}]" in obs:
                assert f"\"{e['name']}\"" in obs
        _clean(obs)
    finally:
        agent.close()


def test_normalisation_is_meaning_preserving_and_logged():
    from cam_live.agent import normalise_proposal
    from cam_live.harness.http_api import HttpApiExecutor
    from cam_live.predict.candidates import parse_actions
    tools = {t.name: t for t in HttpApiExecutor().tools()}
    get = next(n for n in tools if n == "http.get")
    notes = {}
    obj = {"operation": get, "arguments": {"url": "http://127.0.0.1:9/x", "method": "GET"}, "why": "x"}
    out = normalise_proposal(obj, tools, notes)
    assert out == {"operation": get, "arguments": {"url": "http://127.0.0.1:9/x"}}
    assert notes == {f"stripped:{get}.method": 1, "stripped_field:why": 1}
    assert parse_actions(json.dumps(out), list(tools.values())).actions                       # now valid
    assert not parse_actions(json.dumps(obj), list(tools.values())).actions                   # strict parser refuses
    prog = {"operation": "program", "arguments": {"steps": [{"operation": get, "url": "http://127.0.0.1:9/y"}],
                                                  "expected": "a reading"}}
    p2 = normalise_proposal(prog, tools, notes)
    assert p2 == {"steps": [{"operation": get, "arguments": {"url": "http://127.0.0.1:9/y"}}], "expected": "a reading"}
    assert notes["program_form_lifted"] == 1 and notes["moved_into_arguments"] == 1
    parsed = parse_actions(json.dumps(p2), list(tools.values()))
    assert parsed.actions and parsed.actions[0].operation == "program" and parsed.actions[0].steps[0].arguments["url"]


def test_literal_numbers_are_coerced_not_invented():
    from cam_live.agent import normalise_proposal
    from cam_live.diagnostics.world import LabWorld
    tools = {t.name: t for t in LabWorld().discover_tools()}
    notes = {}
    assert normalise_proposal({"operation": "lab.act", "arguments": {"channel": "2", "effort": "3"}}, tools, notes) == \
        {"operation": "lab.act", "arguments": {"channel": 2, "effort": 3}}
    assert notes == {"coerced:lab.act.channel": 1, "coerced:lab.act.effort": 1}
    out = normalise_proposal({"operation": "lab.act", "arguments": {"channel": "channel-2", "effort": 9}}, tools, {})
    assert out["arguments"] == {"channel": "channel-2", "effort": 9}       # not a number / out of range: left as is
