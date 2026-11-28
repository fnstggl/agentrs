"""ST hotfix (live planning run, first 35 decisions, measured 2026-11-28 ~09:45 UTC; docs/ST.md §1):

1. 14 of 35 chosen actions were the schema walker's ``http.request`` — modality ``api`` is among the task's
   interfaces, but ``http.*`` is not among ``resources.authorized_tools``: every one was refused
   (P-TOOL-NOT-AUTHORIZED). The agent now lists and offers only tools that list authorizes (names, globs, modality
   names: the authorization service's own reading).
2. 4 were walker navigations / downloads to ``https://*/``: the authorization's destination PATTERN taken as a URL
   (refused: "bad host '*'"). Patterns are no argument vocabulary any more; only concrete task-given and observed
   values are.
3. The backbone's single 256-token answer for 8 alternatives was cut off in 34 of 35 decisions (the last line
   ``not_json`` "Unterminated string"). Now ``proposer_samples`` independent continuations (one batched generate
   call), each asked for ``proposer_k`` alternatives with its own ``proposer_max_new_tokens``; each is parsed on its
   own (a cut-off loses only that sample's last line) and the per-sample outcome is recorded with the decision.

Every test here fails on HEAD 87eab34 (the code the live run started with). Where a test needs a known backbone
output it replaces ``model.generate`` with fixed text: parser and bookkeeping, not what the real backbone writes.
"""
from __future__ import annotations

from cam_live.agent import CamLiveAgent
from cam_live.contracts import Cost, Level, Limits, Modality, Observation, SimClock, TaskSpec, ToolDescriptor
from cam_live.predict import candidates as C
from int_helpers import agent_config

import json

SUCCESS = {"verifier": "membership", "metric": "m.count", "target": 5}


def _tools():
    obj = lambda req, **p: {"type": "object", "properties": p, "required": req,                 # noqa: E731
                            "additionalProperties": False}
    return [
        ToolDescriptor("internal.note", Modality.INTERNAL, "Append a note.", obj(["text"], text={"type": "string"}),
                       Level.PRIMITIVE, "", "local", Cost()),
        ToolDescriptor("web.get", Modality.STRUCTURED_BROWSER, "Read a page.", obj(["url"], url={"type": "string"}),
                       Level.PRIMITIVE, "web.read", "external_read", Cost(seconds=1.0)),
        ToolDescriptor("web.download", Modality.STRUCTURED_BROWSER, "Download.", obj(["url"], url={"type": "string"}),
                       Level.PRIMITIVE, "web.read", "external_read", Cost(seconds=1.0)),
        ToolDescriptor("http.request", Modality.API, "Raw request.", obj(["url"], url={"type": "string"}),
                       Level.PRIMITIVE, "", "external_write", Cost(seconds=1.0)),
        ToolDescriptor("wait", Modality.INTERNAL, "Wait.", obj(["seconds"], seconds={"type": "number", "minimum": 1,
                                                                                      "maximum": 600}),
                       Level.PRIMITIVE, "", "none", Cost()),
    ]


def _agent(tmp_path, resources=None, **settings):
    task = TaskSpec(objective_id="obj_hf", objective="Grow a group to five members.", success=dict(SUCCESS),
                    interfaces=["structured_browser", "api", "internal"],
                    limits=Limits(duration_s=3600.0, max_decisions=50), mode="staging",
                    resources=resources if resources is not None else {
                        "authorized_tools": ["web.*", "internal.*", "wait"],
                        "authorized_destinations": ["https://*", "http://*", "channel:x:*", "account:x:42"],
                        "read": ["https://*"], "home": "https://owned.example/"})
    s = dict(use_backbone=True, planner=False, proposer_max_new_tokens=8)
    s.update(settings)
    return CamLiveAgent(agent_config(**s), str(tmp_path / "a"), clock=SimClock(start=1000.0), task=task)


def _fixed_generator(agent, texts):
    calls = []

    def gen(prompts, **kw):
        calls.append({"prompts": list(prompts), **kw})
        return [list(texts)[:kw.get("num_return_sequences", 1)]]
    agent.model.generate = gen
    return calls


def _data(prompt, header):
    lines = prompt.splitlines()
    return json.loads(lines[lines.index(header) + 1])


def test_unauthorized_tools_are_neither_listed_nor_offered(tmp_path):
    agent = _agent(tmp_path)
    calls = _fixed_generator(agent, ['{"operation": "http.request", "arguments": {"url": "https://x.example/"}}\n'
                                     '{"operation": "web.get", "arguments": {"url": "https://x.example/"}}'])
    try:
        agent.ingest([Observation(obs_id="obs_1", ts=agent.clock.now(), kind="page", text="p",
                                  data={"url": "https://x.example/", "links": [{"href": "https://x.example/b"}]})])
        cands = agent.propose(_tools(), 16)
        listed = [t["name"] for t in _data(calls[-1]["prompts"][0], C.H_TOOLS)]
        assert "http.request" not in listed and {"web.get", "internal.note", "wait"} <= set(listed)
        assert cands and all(c.operation != "http.request" for c in cands)          # neither walker nor backbone
        assert any(c.operation == "web.get" and "backbone" in c.provenance for c in cands)
        assert agent.last["proposal"]["stats"]["n_invalid_by_operation"].get("tool_not_authorized", 0) >= 1
    finally:
        agent.close()


def test_without_an_authorized_tool_list_interfaces_alone_decide(tmp_path):
    agent = _agent(tmp_path, resources={})
    calls = _fixed_generator(agent, [""])
    try:
        agent.propose(_tools(), 16)
        listed = [t["name"] for t in _data(calls[-1]["prompts"][0], C.H_TOOLS)]
        assert "http.request" in listed                                              # api is a permitted interface
    finally:
        agent.close()


def test_authorization_patterns_are_never_argument_values(tmp_path):
    agent = _agent(tmp_path)
    _fixed_generator(agent, [""])
    try:
        v = agent._resource_vocab()
        assert not any("*" in x for vals in v.values() for x in vals), v
        assert v.get("url") == ["https://owned.example/"] and v.get("account_id") == ["42"]
        cands = agent.propose(_tools(), 16)                       # before anything is observed
        assert not any("*" in str(c.arguments.get("url", "")) for c in cands)
        assert any(c.operation == "web.get" and c.arguments.get("url") == "https://owned.example/" for c in cands)
    finally:
        agent.close()


def test_independent_samples_each_with_its_own_budget_parsed_separately(tmp_path):
    agent = _agent(tmp_path, proposer_samples=3, proposer_k=2, proposer_max_new_tokens=64)
    a1 = '{"operation": "web.get", "arguments": {"url": "https://a.example/"}}'
    a2 = '{"operation": "internal.note", "arguments": {"text": "first observations"}}'
    cut = '{"operation": "web.get", "arguments": {"url": "https://b.exa'
    calls = _fixed_generator(agent, [a1 + "\n" + cut, a2, a1])
    try:
        d = agent.select(agent.evaluate(agent.propose(_tools(), 16)))
        c = calls[0]
        assert len(calls) == 1 and len(c["prompts"]) == 1 and c["num_return_sequences"] == 3
        assert c["max_new_tokens"] == 64                                   # per sample, not shared
        assert "Write up to 2 candidate next actions" in c["prompts"][0]
        g = d.diagnostics["generation"]
        assert g["samples"] == 3 and g["n_backbone_valid"] == 2            # the same action from two samples: once
        assert [s["valid"] for s in g["per_sample"]] == [1, 1, 1]
        assert [s["last_line_rejected"] for s in g["per_sample"]] == [1, 0, 0]  # the cut line, recorded as such
        assert g["raw"][0].startswith(a1) and len(g["raw"]) == 3
        assert {x.action.operation for x in d.candidates if "backbone" in x.action.provenance} == {"web.get",
                                                                                                  "internal.note"}
    finally:
        agent.close()


def test_default_budget_per_alternative(tmp_path):
    """The defaults give each alternative several times the budget it had (256 / 8 = 32 tokens on HEAD) and keep
    the number of alternatives asked per decision (8)."""
    agent = CamLiveAgent(agent_config(), str(tmp_path / "d"), clock=SimClock(start=1000.0))
    from cam_live.agent import AgentSettings
    try:
        s = AgentSettings()
        assert s.proposer_samples * s.proposer_k == 8
        assert s.proposer_max_new_tokens / s.proposer_k >= 150
        assert agent.proposer.samples == agent.settings.proposer_samples
    finally:
        agent.close()
