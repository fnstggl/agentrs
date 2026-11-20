"""B1 (LLM-only) uses exactly CAM's tool schemas and validation; its parser does no hidden repair."""
from agentrs.cam2.baselines import parse_action, render
from agentrs.cam2.envs.miniwob import miniwob_tools
from agentrs.cam2.tools import Element, Observation, validate_call

OBS = Observation(goal="Enter 'cat' and press Submit", elements=[
    Element("12", "textbox", "textbox ''"), Element("13", "button", "button 'Submit'"),
    Element("t2", "StaticText", "StaticText 'hello'", targetable=False)])
TOOLS = miniwob_tools()
TMAP = {t.name: t for t in TOOLS}


def test_render_lists_elements_and_tools():
    s = render(OBS, TOOLS, [])
    assert "[0] textbox ''" in s and "[1] button 'Submit'" in s and "fill(" in s and "press(" in s


def test_valid_answers_become_valid_tool_calls():
    c, err = parse_action('{"tool": "fill", "target": 0, "value": "cat"}', OBS, TOOLS)
    assert err is None and c.element == "12" and c.text == "cat" and validate_call(c, TMAP, OBS) is None
    c, err = parse_action('Sure! {"tool": "press", "target": 1, "value": "Enter"} {"x": 1}', OBS, TOOLS)
    assert err is None and c.enum == "Enter" and validate_call(c, TMAP, OBS) is None


def test_invalid_answers_stay_invalid():
    for txt in ['click the button', '{"tool": "teleport", "target": 0}', '{"tool": "click", "target": 7}',
                '{"tool": "press", "target": 1, "value": "F13"}', '{"tool": "click", "target": 2}']:
        c, err = parse_action(txt, OBS, TOOLS)
        assert err is not None or validate_call(c, TMAP, OBS) is not None, txt
