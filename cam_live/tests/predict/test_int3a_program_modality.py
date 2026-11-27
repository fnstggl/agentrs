"""INT2-9 (INT3A): a parsed program has the modality of the environment's program tool, not its first step's.

The harness declares ``program`` as an internal operation (``harness.env``) and the authorization rule T-PROGRAM
admits ``program`` only with modality ``internal``; every step is judged on its own. The parser gave a composition its
first step's modality, so ``program [http.get, ...]`` was refused before any step was judged (measured on the real
configuration, ``docs/evidence/validation/s6/oe_seq_20261127.json``). Fails on the code before the fix.
"""
from __future__ import annotations

import json

from cam_live.contracts import Level, Modality, ToolDescriptor
from cam_live.predict.candidates import PROGRAM_OPERATION, parse_actions


def _tools(program_modality=Modality.INTERNAL):
    return [
        ToolDescriptor("http.get", Modality.API, "Send one HTTP GET request.",
                       {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
                       Level.PRIMITIVE, "http.read", "external_read"),
        ToolDescriptor("sandbox.write_file", Modality.CODE, "Create or replace a file.",
                       {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"]}, Level.PRIMITIVE, "sandbox.write", "local"),
        ToolDescriptor("internal.note", Modality.INTERNAL, "Append a note.",
                       {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}),
        ToolDescriptor(PROGRAM_OPERATION, program_modality, "Run the given steps in order.",
                       {"type": "object", "properties": {"steps": {"type": "array"}}, "required": ["steps"]},
                       Level.PROCEDURE),
    ]


def _program(*steps):
    return json.dumps({"steps": list(steps)})


GET = {"operation": "http.get", "arguments": {"url": "http://127.0.0.1:9/measure?x=1&y=2"}}
WRITE = {"operation": "sandbox.write_file", "arguments": {"path": "model.py", "content": "def predict(x, y): return 0"}}
NOTE = {"operation": "internal.note", "arguments": {"text": "value 3.1"}}


def test_a_program_takes_the_program_tools_modality_not_its_first_steps():
    for first in (GET, WRITE):
        res = parse_actions(_program(first, NOTE), _tools(), provenance="backbone")
        assert res.rejections == [] and len(res.actions) == 1
        prog = res.actions[0]
        assert prog.operation == PROGRAM_OPERATION and prog.level == Level.PROCEDURE
        assert prog.modality == Modality.INTERNAL                     # before the fix: api / code (the first step's)
        assert [s.modality for s in prog.steps] == [ {"http.get": Modality.API,
                                                       "sandbox.write_file": Modality.CODE}[first["operation"]],
                                                     Modality.INTERNAL]


def test_the_declared_modality_of_the_program_tool_is_used():
    res = parse_actions(_program(GET, NOTE), _tools(program_modality="skill"))
    assert res.actions[0].modality == "skill"


def test_without_a_program_tool_a_composition_is_internal():
    tools = [t for t in _tools() if t.name != PROGRAM_OPERATION]
    res = parse_actions(_program(WRITE, GET), tools)
    assert res.actions[0].modality == Modality.INTERNAL
    assert [s.modality for s in res.actions[0].steps] == [Modality.CODE, Modality.API]


def test_steps_are_still_bounded_by_the_tools_given():
    # the parser admits only steps of the tools it is given (the agent gives the task's permitted tools; its
    # feasibility check additionally drops a program with any step outside the task's interfaces)
    tools = [t for t in _tools() if t.name != "http.get"]
    res = parse_actions(_program(NOTE, GET), tools)
    assert not res.actions and res.reasons == ["invalid_step"]
