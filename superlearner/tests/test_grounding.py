"""Grounded execution, the action-program interpreter, agent-made tools, tasks and the trusted session core."""
import json

import pytest

from superlearner.actions.program import Interpreter, ProgramError, parse_program
from superlearner.cam.agent import AgentConfig, CAMAgent, RandomChooser
from superlearner.cam.model import EmbStore
from superlearner.grounding.executor import GroundedExecutor, Outcome
from superlearner.grounding.knowledge import CapabilityIndex, Knowledge, tool_doc
from superlearner.grounding.toolsmith import ToolLibrary, macro_from_trace
from superlearner.llm.base import ScriptedLLM
from superlearner.llm.server import EmbedOnly
from superlearner.train.demos import TeacherChooser
from superlearner.world.core import CoreConfig, SessionCore
from superlearner.world.tasks import (TaskSpec, assert_disjoint, check_split, load_final, lock_final, task, task_list)


@pytest.fixture()
def core():
    return SessionCore(task("found", "tiny", 3), CoreConfig(max_decisions=30))


def grounded(core):
    know, caps = Knowledge(core.actor), CapabilityIndex(EmbedOnly())
    caps.update(core.tools(True))
    return GroundedExecutor(core.execute, caps, know), know, caps


def test_failure_taxonomy_distinguishes_syntax_references_resources_and_loops(core):
    ex, know, caps = grounded(core)
    assert ex.execute("nope.tool", {}).fclass == "unknown_tool"
    o = ex.execute("id.create_org", {"handle": "Bad Handle", "name": "x"})
    assert o.fclass == "syntax" and not o.sent and "handle" in o.feedback and "pattern" in o.feedback
    o = ex.execute("bank.transfer", {"from_account": "ACCT9", "to": "ACCT8", "amount": 5})
    assert o.fclass == "unresolved_reference" and not o.sent           # an alias the actor never saw
    o = ex.execute("bank.transfer", {"from_account": "acct_999999", "to": "acct_999998", "amount": 5})
    assert o.fclass == "nonexistent_resource" and o.sent
    o = ex.execute("bank.transfer", {"from_account": "acct_999999", "to": "acct_999998", "amount": 5})
    assert o.fclass == "nonexistent_resource"
    o = ex.execute("bank.transfer", {"from_account": "acct_999999", "to": "acct_999998", "amount": 5})
    assert o.fclass == "loop_blocked" and not o.sent                   # third identical failing call is not sent
    assert ex.execute("world.whoami", {}).ok
    st = ex.stats()
    assert st["failures"]["loop_blocked"] == 1 and st["calls"] == 7 and st["sent"] == 3


def test_a_successful_state_change_reopens_previously_failing_calls(core):
    ex, know, _ = grounded(core)
    bad = {"owner": "o:nova-labs"}
    assert not ex.execute("bank.open_account", bad).ok
    assert not ex.execute("bank.open_account", bad).ok
    assert ex.execute("id.create_org", {"handle": "nova-labs", "name": "Nova", "kind": "company"}).ok
    o = ex.execute("bank.open_account", bad)             # the retry guard was cleared by the state change
    assert o.sent


def test_aliases_round_trip_and_documents_are_collected(core):
    ex, know, _ = grounded(core)
    o = ex.execute("bank.accounts", {})
    know.observe(action={"tool": "bank.accounts", "args": {}}, result=o.to_result())
    acct = o.result[0]["account"]
    a = know.alias_of(acct)
    assert a.startswith("ACCT") and know.resolve({"x": [a]}) == {"x": [acct]}
    o = ex.execute("search.query", {"q": "sort records format"})
    know.observe(action={"tool": "search.query", "args": {}}, result=o.to_result())
    assert any("sort.records" in d.text for d in know.docs.values())


def test_capability_search_and_whole_descriptions(core):
    _, _, caps = grounded(core)
    assert "crucible.json" in tool_doc(caps.tools["cloud.deploy"], 600) and "handler" in tool_doc(caps.tools["cloud.deploy"], 600)
    assert "cloud.deploy" in caps.consumers("code.commit")
    assert caps.nearest_names("bank.transfr")[0] == "bank.transfer"
    assert len(caps.search("send money to another account", 5)) == 5


def test_program_interpreter_runs_calls_branches_loops_and_references(core):
    ex, know, _ = grounded(core)
    waited = []
    it = Interpreter(lambda t, a: ex.execute(t, a), lambda s: waited.append(s) or s)
    prog = parse_program(json.dumps({"plan": "found a company and look around", "steps": [
        {"call": "id.create_org", "args": {"handle": "acme-co", "name": "Acme", "kind": "company"}, "as": "org"},
        {"call": "bank.accounts", "as": "accts"},
        {"if": {"ok": "$org"}, "then": [{"call": "biz.register", "args": {"org": "$org.org", "account": "$accts.0.account"}, "as": "reg"}],
         "else": [{"stop": "no organisation"}]},
        {"repeat": 2, "do": [{"call": "world.clock"}]},
        {"set": "half", "value": {"expr": ["int", {"expr": ["mul", "$accts.0.balance", 0.5]}]}},
        {"if": {"cmp": ["$half", ">", 10]}, "then": [{"wait": 60}]},
        {"note": "done"}]}))
    tr = it.run(prog)
    assert tr.ok and tr.stopped == "" and tr.n_calls == 5 and waited == [60] and tr.notes == ["done"]
    assert tr.env["half"] > 10 and tr.env["reg"]["ok"]
    assert core.evaluator.v.predicate(core.world)        # the company really exists in authoritative state


def test_program_stops_at_the_first_failure_and_says_why(core):
    ex, _, _ = grounded(core)
    it = Interpreter(lambda t, a: ex.execute(t, a), lambda s: s, max_ops=3)
    tr = it.run(parse_program([{"call": "bank.balance", "args": {"account": "acct_424242"}}, {"call": "world.clock"}]))
    assert not tr.ok and tr.n_calls == 1 and "nonexistent_resource" in tr.stopped
    tr = it.run(parse_program([{"repeat": 9, "do": [{"call": "world.clock"}]}]))
    assert tr.stopped.startswith("budget")                # bounded: more operations than the budget allows
    with pytest.raises(ProgramError):
        parse_program('{"steps": [{"frobnicate": 1}]}')
    with pytest.raises(ProgramError):
        parse_program("no json here")


def test_agent_made_tools_are_generalised_registered_versioned_and_verified_only_by_later_use():
    calls = [("id.create_org", {"handle": "acme-co", "name": "Acme", "kind": "company"}, {"org": "o:acme-co"}),
             ("biz.register", {"org": "o:acme-co", "account": "acct_000017"}, {"company": "o:acme-co"}),
             ("bank.open_account", {"owner": "o:acme-co"}, {"account": "acct_000018"}),
             ("bank.transfer", {"from_account": "acct_000017", "to": "acct_000018", "amount": 1000}, {"ok": True})]
    spec = macro_from_trace(calls, "p:agent", "m_company")
    steps = spec["steps"]
    assert steps[1]["args"]["org"] == "$s0.org" and steps[3]["args"]["to"] == "$s2.account"     # outputs of earlier steps
    assert steps[1]["args"]["account"] == "$account" and steps[3]["args"]["from_account"] == "$account"   # a parameter
    assert spec["params"]["required"] == ["account"] and steps[3]["args"]["amount"] == 1000
    lib = ToolLibrary()
    ok, _ = lib.register(spec, {"episode": "e1", "mechanism": "macro_from_trace"})
    assert ok and lib.tools["m_company"]["deps"] == ["bank.open_account", "bank.transfer", "biz.register", "id.create_org"]
    assert lib.tools["m_company"]["effects"] == ["bank", "biz", "id"] and not lib.tools["m_company"]["verified"]
    lib.record("m_company", True, episode="e1")
    assert not lib.tools["m_company"]["verified"]         # success in the episode that created it proves nothing new
    lib.record("m_company", True, episode="e2")
    assert lib.tools["m_company"]["verified"] and lib.tools["m_company"]["stats"]["ok"] == 2
    assert lib.register(dict(spec, steps=steps[:2]))[0] and lib.tools["m_company"]["version"] == 2
    assert not lib.register({"name": "Bad Name", "steps": steps})[0]
    assert not lib.register({"name": "code_tool", "kind": "code", "code": "def nope(:"})[0]
    assert not lib.register({"name": "code_tool", "kind": "code", "code": "def helper(x):\n    return x\n"})[0]   # no main
    assert lib.register({"name": "code_tool", "kind": "code", "code": "def main(input, caller):\n    return 1\n"})[0]
    assert [h["event"] for h in lib.history].count("rejected") == 3


def test_a_macro_made_in_one_world_runs_in_another(core):
    ag = CAMAgent(None, EmbStore(), EmbedOnly(), AgentConfig(use_llm=False))
    ep = ag.run(core, TeacherChooser("auto", 0), "e1")
    assert ep["summary"]["success"]
    spec = macro_from_trace(ag.wm.calls[-2:], ag.actor, "m_found")       # id.create_org + biz.register
    assert ag.library.register(spec, {"episode": "e1"})[0]
    core2 = SessionCore(task("found", "bazaar", 9), CoreConfig(max_decisions=10))
    ag.begin(core2, "e2")
    o = ag._call("bank.accounts", {})
    out = ag._interp().use(ag.library.tools["m_found"], {"account": o.result[0]["account"], "me": ag.actor},
                           __import__("superlearner.actions.program", fromlist=["ProgramTrace"]).ProgramTrace(), 0)
    # the account is a parameter, the organisation id flows from step 0 to step 1: the tool works in a new world
    assert out.ok, out.feedback
    assert core2.evaluator.v.predicate(core2.world)                  # a registered company exists in world 2
    ag.library.record("m_found", out.ok, out.fclass, "e2")
    assert ag.library.tools["m_found"]["verified"] and ag.library.tools["m_found"]["stats"]["calls"] == 1


def test_code_written_by_the_agent_runs_in_the_world_sandbox(core):
    ag = CAMAgent(None, EmbStore(), EmbedOnly(), AgentConfig(use_llm=False))
    ag.begin(core, "e")
    out = ag._run_code("def main(input, caller):\n    return {'double': input['x'] * 2}\n", {"x": 21})
    assert out.ok and out.result == {"double": 42}
    bad = ag._run_code("def main(input, caller):\n    return 1 / 0\n", {})
    assert not bad.ok and bad.fclass == "runtime" and "ZeroDivisionError" in bad.feedback
    assert [c[0] for c in core.commands][-3:] == ["code.commit", "code.run", "code.commit"] or "code.run" in [c[1] for c in core.commands]


def test_llm_written_values_are_validated_and_drafts_can_be_run_or_skipped(core):
    def reply(messages, variant):
        u = messages[1]["content"]
        if "Write the next program" in u:
            return json.dumps({"plan": "create the company", "steps": [
                {"call": "id.create_org", "args": {"handle": "llm-co", "name": "LLM Co", "kind": "company"}, "as": "o"},
                {"call": "bank.accounts", "as": "a"},
                {"call": "biz.register", "args": {"org": "$o.org", "account": "$a.0.account"}}]})
        return '"hello"'
    ag = CAMAgent(None, EmbStore(), ScriptedLLM(reply), AgentConfig(use_llm=True))
    ag.begin(core, "e")

    class Follow:                                  # the "frozen LLM with grounding" rule
        name = "t"

        def head(self, ag_, items, logits):
            kinds = [m[0] for m in items.meta]
            return kinds.index("draft_run") if "draft_run" in kinds else kinds.index("propose")

        def option(self, *a):
            return 0
    r1 = ag.step(Follow())
    assert r1["kind"] == "propose" and len(ag.wm.draft) == 3 and ag.wm.plan == "create the company"
    r2 = ag.step(Follow())
    assert r2["kind"] == "draft_run" and r2["ok"] and r2["n_calls"] == 3 and r2["done"]
    assert core.summary()["success"]


def test_evaluator_rewards_only_verified_state_and_idle_or_random_agents_earn_nothing():
    for chooser in (None, RandomChooser(0)):
        core = SessionCore(task("found", "tiny", 5), CoreConfig(max_decisions=12))
        if chooser is None:
            total = 0.0
            while not core.done:
                core.wait(3600)
                total += core.end_decision()["reward"]
        else:
            ag = CAMAgent(None, EmbStore(), EmbedOnly(), AgentConfig(use_llm=False))
            total = ag.run(core, chooser, "r")["summary"]["return"]
        assert total == 0.0 and not core.summary()["success"]
    core = SessionCore(task("found", "tiny", 5), CoreConfig(max_decisions=12))
    ep = CAMAgent(None, EmbStore(), EmbedOnly(), AgentConfig(use_llm=False)).run(core, TeacherChooser("auto", 0), "t")
    assert ep["summary"]["success"] and abs(ep["summary"]["return"] - 2.0) < 1e-6      # progress 1 + success bonus 1


def test_splits_are_disjoint_and_the_final_list_is_sealed(tmp_path):
    with pytest.raises(ValueError):
        check_split(TaskSpec("found", "tiny", 5, split="train"))                  # a final-range seed in training
    with pytest.raises(ValueError):
        check_split(TaskSpec("found", "novel_needs", 10_000_001, split="train"))  # a held-out world family
    train = [task("found", "tiny", i) for i in range(5)]
    final = task_list(["found/0/scratch", "service/0/listed"], ["tiny", "novel_needs"], 6, "final", 700_000)
    assert len(final) == 6
    assert_disjoint(train, final)
    with pytest.raises(ValueError):
        assert_disjoint(train + [TaskSpec("found", final[0].world_family, final[0].seed, split="train")], final)
    p = str(tmp_path / "final.json")
    lock_final(p, final)
    assert load_final(p) == final
    d = json.load(open(p))
    d["tasks"][0]["seed"] += 1
    import os
    os.chmod(p, 0o644)
    json.dump(d, open(p, "w"))
    with pytest.raises(ValueError):
        load_final(p)
    with pytest.raises(ValueError):
        lock_final(p, train)                                                    # only final-split tasks can be locked


def test_remote_core_never_reveals_heldout_task_identity_and_refuses_heldout_specs():
    from superlearner.world.core import spawn_env_server
    held = [task("found", "entrenched_incumbents", 77, "select")]
    proc, rc = spawn_env_server(eval_tasks=held, cfg={"max_decisions": 3})
    try:
        with pytest.raises(RuntimeError):
            rc.reset(held[0])                                  # a learner cannot ask for a held-out task by name
        info = rc.reset(eval_index=0)
        assert "seed" not in json.dumps(info) and info["split"] == "select"
        while True:
            rc.wait(3600)
            if rc.end_decision()["truncated"]:
                break
        s = rc.summary()
        assert s["task"] == "eval#0" and not s["success"]
        rc.reset(task("found", "tiny", 1))
        assert rc.summary()["task"].startswith("found:tiny:")
    finally:
        rc.close()
        proc.terminate()
