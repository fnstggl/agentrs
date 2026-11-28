"""S-6b (INT3A): what the backbone's proposal prompt contains, every decision.

Measured on the real configuration before this change (``docs/INTEGRATION.md`` 7.5, ``docs/INT3A.md``): the prompt
showed 12 tool slots of which half rotated at random, the success definition was not in it, an action's failure
was visible only if its result carried an observation, and which tools a decision's prompt listed was not recorded
with the decision. Now:

* the prompt lists the task's complete authorized action space each decision — every available tool of a permitted
  modality (internal tools always), the policy's top-ranked operations first; nothing permitted is left out;
* the decision records the listed tools in prompt order, the prompt's sha256 and length, the permitted tools that
  were not listed (none by default) and the permitted tools the environment reports unavailable, with its reason;
* the digest carries the task's success definition verbatim and the latest action results (status, failure class
  and the environment's detail), and the prompt is fitted to the backbone's prompt window by shortening the
  observation digest only (the backbone keeps a too-long prompt's END, which would cut the instructions first).

The old code fails the first three tests (12 of 16 listed, no hash / unavailable record, no success line).
"""
from __future__ import annotations

import hashlib
import json

from cam_live.agent import CamLiveAgent, compact_tool
from cam_live.contracts import (ActionResult, Cost, Level, Limits, Modality, Observation, SimClock, TaskSpec,
                                ToolDescriptor)
from cam_live.predict import candidates as C
from int_helpers import agent_config

SUCCESS = {"verifier": "hidden_tests", "metric": "sw.pass_fraction", "target": 0.9}


def _tools():
    obj = lambda **p: {"type": "object", "properties": p, "required": list(p)[:1]}           # noqa: E731
    out = [ToolDescriptor(f"code.tool{i}", Modality.CODE, f"Code tool number {i}. It does thing {i}.",
                          obj(path={"type": "string"}, content={"type": "string"}), Level.PRIMITIVE, "code.run",
                          "local", Cost(seconds=1.0)) for i in range(9)]
    out += [ToolDescriptor(f"internal.t{i}", Modality.INTERNAL, f"Internal tool {i}.", obj(text={"type": "string"}),
                           Level.PRIMITIVE, "", "none", Cost(seconds=0.1)) for i in range(4)]
    out += [ToolDescriptor("wait", Modality.INTERNAL, "Wait.", {"type": "object", "properties": {
        "seconds": {"type": "number", "minimum": 1, "maximum": 600}}, "required": ["seconds"]})]
    out += [ToolDescriptor(f"api.t{i}", Modality.API, f"API tool {i}.", obj(url={"type": "string"}),
                           Level.PRIMITIVE, "http.read", "external_read") for i in range(3)]
    out += [ToolDescriptor("code.broken", Modality.CODE, "A code tool that is unavailable.", obj(x={"type": "string"}),
                           available=False, unavailable_reason="workspace below a directory the sandbox uid cannot "
                                                               "traverse")]
    return out


def _agent(tmp_path):
    task = TaskSpec(objective_id="obj_s6b", objective="Write a file that does the thing.", success=dict(SUCCESS),
                    interfaces=["code", "internal"], limits=Limits(duration_s=3600.0, max_decisions=20),
                    mode="staging")
    return CamLiveAgent(agent_config(use_backbone=True, planner=False, proposer_max_new_tokens=8),
                        str(tmp_path / "a"), clock=SimClock(start=1000.0), task=task)


def _capture(agent):
    sent = []
    real = agent.model.generate

    def capture(prompts, **kw):
        sent.extend(prompts)
        return real(prompts, **kw)
    agent.model.generate = capture
    return sent


def _data(prompt, header):
    lines = prompt.splitlines()
    return json.loads(lines[lines.index(header) + 1])


def _decide(agent, tools):
    cands = agent.propose(tools, 10)
    evals = agent.evaluate(cands)
    return agent.select(evals)


def test_every_permitted_tool_is_listed_every_decision_and_recorded(tmp_path):
    agent = _agent(tmp_path)
    sent = _capture(agent)
    tools = _tools()
    permitted = [t.name for t in tools if t.available and t.modality in ("code", "internal")]
    assert len(permitted) == 14
    try:
        agent.ingest([Observation(obs_id="obs_1", ts=agent.clock.now(), kind="metric", text="nothing yet")])
        for i in range(3):
            sent.clear()
            d = _decide(agent, tools)
            listed = [t["name"] for t in _data(sent[0], C.H_TOOLS)]
            assert sorted(listed) == sorted(permitted)                    # complete, every decision; no api tool
            names = {t.name: t for t in tools}
            assert all(t == compact_tool(names[t["name"]]) for t in _data(sent[0], C.H_TOOLS))
            g = d.diagnostics["generation"]
            assert g["prompt_tools"] == listed                            # exactly what the prompt listed, in order
            assert g["prompt_sha256"] == hashlib.sha256(sent[0].encode("utf-8")).hexdigest()
            assert g["prompt_chars"] == len(sent[0]) and g["tools_permitted"] == 14 and g["tools_not_listed"] == []
            assert set(g["tools_unavailable"]) == {"code.broken"}         # permitted, but unavailable: recorded
            assert "cannot traverse" in g["tools_unavailable"]["code.broken"]
            assert isinstance(g["prompt_tokens"], int) and g["prompt_tokens"] > 0
            agent.clock.advance(5.0)
        assert agent.status()["counters"]["decisions_with_unavailable_permitted_tools"] == 3
    finally:
        agent.close()


def test_the_digest_carries_the_success_definition_and_the_latest_results(tmp_path):
    agent = _agent(tmp_path)
    sent = _capture(agent)
    tools = _tools()
    try:
        agent.ingest([Observation(obs_id="obs_1", ts=agent.clock.now(), kind="metric", text="nothing yet")])
        d = _decide(agent, tools)
        obs = _data(sent[0], C.H_OBSERVATION)
        assert "[task] success definition: " + json.dumps(SUCCESS, sort_keys=True) in obs
        assert _data(sent[0], C.H_OBJECTIVE) == agent.task.objective              # the objective verbatim
        t0 = agent.clock.now()
        res = ActionResult(action_id=d.chosen.action_id, status="error", failure_class="invalid_arguments",
                           failure_detail="path must be inside the workspace", started=t0, ended=t0 + 1.0)
        agent.record_result(d, res)
        agent.clock.advance(2.0)
        sent.clear()
        _decide(agent, tools)
        obs = _data(sent[0], C.H_OBSERVATION)
        line = f"[result] {d.chosen.operation} -> error (invalid_arguments): path must be inside the workspace"
        assert line in obs                                       # a failure without observations is still visible
        assert obs.index("[task] success definition") < obs.index(line)          # newest last
    finally:
        agent.close()


def test_the_walker_vocabulary_ignores_result_lines(tmp_path):
    agent = _agent(tmp_path)
    try:
        o = Observation(obs_id="obs_v", ts=agent.clock.now(), kind="page", text="p",
                        data={"url": "https://a.example/x", "links": [{"href": "https://a.example/y"}]})
        agent.ingest([o])
        v0 = agent._vocab()
        for i in range(4):
            agent.recent_obs.append({"obs_id": f"result:a{i}", "ts": agent.clock.now(), "kind": "result",
                                     "text": "x -> ok", "quantities": [], "vocab": {}})
        assert agent._vocab() == v0
    finally:
        agent.close()


def test_the_prompt_is_fitted_to_the_backbone_window_by_shortening_observations_only(tmp_path):
    agent = _agent(tmp_path)
    sent = _capture(agent)
    tools = _tools()
    bb = agent.model.backbone
    try:
        long_text = " ".join(f"word{i}" for i in range(600))
        agent.ingest([Observation(obs_id=f"obs_{i}", ts=agent.clock.now() + i, kind="tool_result",
                                  text=long_text[:590]) for i in range(8)])
        full = agent.proposer.prompt([t for t in tools if t.available and t.modality != "api"], agent._digest(),
                                     agent.task.objective, agent.settings.proposer_k)
        n_full = len(bb._encode(full))
        bb.pretrained = True                                     # the fit applies to a pretrained backbone only
        old = bb.nn_cfg.max_prompt_tokens
        bb.nn_cfg.max_prompt_tokens = n_full - 400
        try:
            d = _decide(agent, tools)
        finally:
            bb.nn_cfg.max_prompt_tokens = old
            bb.pretrained = False
        g = d.diagnostics["generation"]
        fit = g["prompt_fit"]
        assert fit["fitted"] and fit["tokens"] <= n_full - 400 and not fit["over_window"]
        assert len(bb._encode(sent[0])) <= n_full - 400                       # nothing is cut by the backbone
        assert len(_data(sent[0], C.H_TOOLS)) == 14                            # every permitted tool still listed
        assert _data(sent[0], C.H_OBJECTIVE) == agent.task.objective
        assert "[task] success definition" in _data(sent[0], C.H_OBSERVATION)
    finally:
        agent.close()


def test_argument_names_are_listed_exactly_and_nested_objects_show_their_properties():
    """Measured on the real configuration (docs/INT3A.md, probe before): with the "name*" marker the backbone wrote
    "path*" / "content*" as argument names, and with nested objects shown as "object" it invented their properties;
    every such action was refused."""
    sch = {"type": "object", "required": ["code"], "additionalProperties": False,
           "properties": {"code": {"type": "string"},
                          "limits": {"type": "object", "additionalProperties": False,
                                     "properties": {"cpu_s": {"type": "integer", "minimum": 1}}},
                          "env": {"type": "object", "additionalProperties": {"type": "string"}}}}
    c = compact_tool(ToolDescriptor("x.run", Modality.CODE, "Run code.", sch))
    assert set(c["args"]) == {"code", "limits", "env"}                     # exact names, no marker
    assert c["args"]["code"] == "string, required"
    assert c["args"]["limits"] == "object {cpu_s: integer 1..}" and c["args"]["env"] == "object of string"


def test_a_marker_in_an_argument_name_and_a_multiline_string_are_repaired_and_logged(tmp_path):
    from cam_live.agent import normalise_proposal
    agent = _agent(tmp_path)
    try:
        tools = _tools()
        tmap = {t.name: t for t in tools}
        notes = {}
        out = normalise_proposal({"operation": "code.tool1", "arguments": {"path*": "a.py", "content*": "x"}}, tmap,
                                 notes)
        assert out == {"operation": "code.tool1", "arguments": {"path": "a.py", "content": "x"}}
        assert notes == {"unmarked:code.tool1.path": 1, "unmarked:code.tool1.content": 1}
        text = ('{"operation": "code.tool2", "arguments": {"path": "s.py", "content": "def solve(s):\n'
                '    return s"}}\n{"operation": "wait", "arguments": {"seconds": 5}}\n'
                '{"operation": "code.tool3", "arguments": {"path": "t.py", "content": "unterminated\n'
                '{"operation": "wait", "arguments": {"seconds": 6}}')
        agent.model.generate = lambda prompts, **kw: [[text]]
        res = agent.proposer.propose([t for t in tools if t.available], "digest", objective="o")
        ops = [(a.operation, dict(a.arguments)) for a in res.actions]
        assert ("code.tool2", {"path": "s.py", "content": "def solve(s):\n    return s"}) in ops   # joined, exact
        assert ("wait", {"seconds": 5}) in ops and ("wait", {"seconds": 6}) in ops
        assert not any(o == "code.tool3" for o, _ in ops)                   # never closed: refused, not invented
        assert res.stats["normalised"].get("multiline_joined") == 1
    finally:
        agent.close()
