"""The 4.0 prompt interface of the LLM agent: no host is ever missing from the tool list, failures arrive with
their structured detail, and the run records which interface it used. The 3.0 prompt stays byte-identical."""
import json

import crucible

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.llm import EchoLLM, LLMPolicy  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.worldgen import generate  # noqa: E402


def _world():
    w = World.create(generate({"family": "tiny"}, 3))
    w.system_call("admin.join", {"handle": "ann", "endowment": 500_000})
    return w


def _prompt(echo, i=-1):
    return echo.seen[i][1]["content"]


def test_v4_tool_list_names_every_host_at_any_budget_and_v3_can_hide_them():
    w = _world()
    hosts = {t["name"].split(".")[0] for t in w.available_tools("p:ann") if not t["name"].startswith("admin.")}
    assert len(hosts) > 15
    for budget in (2500, 6000, 12000, 40000):
        echo = EchoLLM(['{"tool": "wait", "args": {}}'])
        pol = LLMPolicy(backend=echo, interface="v4", tool_chars=budget)
        Runner(w.fork(f"f{budget}"), [("p:ann", pol)], dt=600, record=False).run(1)
        tools_part = _prompt(echo).split("Tools (", 1)[1]
        assert len(tools_part) <= budget + 80
        missing = sorted(h for h in hosts if f"{h}" not in tools_part)
        assert not missing, (budget, missing)
        assert "world.find_tools" in tools_part or "find_tools" in tools_part
    # the 3.0 renderer with the 3.0 default budget gives full lines to the alphabetically first hosts only
    echo = EchoLLM(['{"tool": "wait", "args": {}}'])
    Runner(w.fork("v3"), [("p:ann", LLMPolicy(backend=echo, tool_chars=3000))], dt=600, record=False).run(1)
    v3 = _prompt(echo).split("Tools (", 1)[1]
    full_hosts = {ln.split("(")[0].split(".")[0] for ln in v3.splitlines() if "(" in ln and "." in ln.split("(")[0]
                  and not ln.startswith("(+")}
    assert len(full_hosts) < len(hosts) / 2


def test_v4_shows_structured_error_detail_and_v3_prompt_is_unchanged():
    w = _world()
    replies = ['{"tool": "bank.acounts", "args": {}}', '{"tool": "bank.transfer", "args": {"to": 5}}',
               '{"tool": "wait", "args": {}}']
    echo4, echo3 = EchoLLM(list(replies)), EchoLLM(list(replies))
    p4 = LLMPolicy(backend=echo4, interface="v4", tool_chars=6000)
    p3 = LLMPolicy(backend=echo3, tool_chars=6000)
    Runner(w.fork("a"), [("p:ann", p4)], dt=600, record=False).run(3)
    Runner(w.fork("b"), [("p:ann", p3)], dt=600, record=False).run(3)
    after_typo = _prompt(echo4, 1)
    assert "Result of your last action:" in after_typo and "bank.accounts" in after_typo      # the near name
    after_schema = _prompt(echo4, 2)
    assert "args.to" in after_schema or "to" in after_schema.split("Result of your last action:")[1][:400]
    # the 3.0 prompt has neither section nor suggestion: it is what recorded 3.0 transcripts were produced with
    assert "Result of your last action:" not in _prompt(echo3, 1)
    assert "bank.accounts" not in _prompt(echo3, 1).split("Tools (")[0]
    # both worlds recorded the same rejected commands: error detail is returned to the caller, not written to the log
    assert p4.transcript[0]["result"]["ok"] is False and p3.transcript[0]["result"]["ok"] is False


def test_usage_records_the_interface_and_per_turn_truncation():
    class Trunc(EchoLLM):
        max_context = 512

        def generate(self, messages, max_new_tokens=256, temperature=0.0, seed=None):
            g = super().generate(messages, max_new_tokens, temperature, seed)
            g.truncated = self.i % 2 == 0
            return g
    pol = LLMPolicy(backend=Trunc(['{"tool": "world.clock", "args": {}}']), interface="v4")
    Runner(_world(), [("p:ann", pol)], dt=600, record=False).run(4)
    u = pol.usage()
    assert u["interface"]["version"] == "v4" and u["interface"]["renderer"] == "catalogue"
    assert u["max_context"] == 512 and u["max_new_tokens"] == 256
    flags = [bool(r.get("truncated")) for r in pol.transcript]
    assert flags == [False, True, False, True] and u["truncated_prompts"] == 2
    json.dumps(u)
