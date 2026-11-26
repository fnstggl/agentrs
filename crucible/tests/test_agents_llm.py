"""Learning-agent integrations: frozen-LLM policy (plumbing with a test double; greedy determinism on real Qwen3 when
available), structured action templates and the factored neural RL learner (online updates, checkpoint resume).

The CAM 2.0 EnvironmentAdapter and superlearner World adapter tests moved to the AGENTRS side with the adapters
(tests/cam2/test_crucible_adapter.py, tests/test_sl_crucible_world.py; DECISION_LOG D-81: Crucible never imports CAM)."""
from __future__ import annotations

import json

import numpy as np
import pytest

import crucible

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.llm import EchoLLM, LLMPolicy, ReplayLLM, render_tool_list  # noqa: E402
from crucible.core.kernel import World, WorldSpec  # noqa: E402

SPEC = {"family": "test", "seed": 1, "modules": [{"name": "identity"}, {"name": "bank"}, {"name": "admin"}]}
FENCE = "`" * 3


def small_world():
    w = World.create(WorldSpec.from_dict(SPEC))
    a = w.system_call("admin.join", {"handle": "alice", "endowment": 10_000})["principal"]
    w.system_call("admin.join", {"handle": "bob", "endowment": 0})
    return w, a


def _acct(w, owner):
    return w.modules["bank"].primary_account(w.state, owner)


def _gpu_and_model(name: str) -> bool:
    try:
        import torch
        from transformers.utils import cached_file
        cached_file(name, "config.json", local_files_only=True)
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


# ============================================================================================== LLM policy
def test_llm_policy_plumbing_with_test_double():
    w, a = small_world()
    acct = _acct(w, a)
    replies = [f'Sending money.\n{FENCE}json\n{{"tool": "bank.transfer", "args": {{"from_account": "{acct}", '
               f'"to": "bob", "amount": 2500}}}}\n{FENCE}',
               "I am not sure what to do here.",                     # unparseable -> no action, counted
               '{"tool": "wait", "args": {}}',                       # explicit no-op
               '<tool_call>{"name": "bank.balance", "arguments": {"account": "' + acct + '"}}</tool_call>']
    echo = EchoLLM(replies)
    pol = LLMPolicy(backend=echo, goal="Hold at most $50 in your checking account.", history=3)
    run = Runner(w, [(a, pol)], dt=60)
    run.run(4)
    steps = run.trajs[a].steps
    assert [s.action["tool"] if s.action else None for s in steps] == ["bank.transfer", None, None, "bank.balance"]
    assert steps[0].result["ok"] and steps[0].result["result"]["balance"] == 7_500
    assert steps[3].result["result"]["balance"] == 7_500
    u = pol.usage()
    assert u["calls"] == 4 and u["parse_failures"] == 1 and u["waits"] == 1
    assert u["prompt_tokens"] > 0 and u["gen_tokens"] > 0
    assert [r["text"] for r in pol.transcript] == replies          # every generated text is recorded
    assert "parse_error" in pol.transcript[1] and pol.transcript[1]["parse_error"]["code"] == "parse_error"
    # the prompt carries the goal verbatim, the history with results, and the tool signatures; never admin tools
    last = echo.seen[-1][1]["content"]
    assert "Goal: Hold at most $50 in your checking account." in last
    assert "bank.transfer" in last and "-> ok" in last and "unparseable reply" in last
    assert "bank.transfer(amount: int, from_account: str, to: str" in last
    assert "admin.join" not in last and "admin.endow" not in last


def test_llm_prompt_summarises_long_tool_lists_and_announces_new_tools():
    tools = [{"name": f"svc{i // 10}.op{i % 10}", "description": "Does a thing " * 10,
              "inputSchema": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}}
             for i in range(120)]
    txt = render_tool_list(tools, budget_chars=1500, priority=("svc11.op9",))
    assert len(txt) < 2600 and "svc11.op9(x: int)" in txt and "more tools, names only" in txt
    assert all(f"op{k}" in txt for k in range(10))                 # every tool still appears (by name)
    echo = EchoLLM(['{"tool": "wait"}'])
    pol = LLMPolicy(backend=echo, goal="g", tool_chars=1500)
    pol.reset("p:x", np.random.default_rng(0))
    obs = {"time": 0, "principal": "p:x", "events": []}
    pol.act(obs, tools[:100])
    assert "Newly available tools" not in echo.seen[-1][1]["content"]
    pol.act(obs, tools)                                              # 20 tools discovered later
    msg = echo.seen[-1][1]["content"]
    assert "Newly available tools: " + ", ".join(sorted(t["name"] for t in tools[100:])) in msg
    assert "svc10.op0(x: int)" in msg                                # new tools keep their full signature
    assert pol.transcript[-1]["new_tools"] == sorted(t["name"] for t in tools[100:])


def test_llm_transcript_replays_the_trajectory_exactly():
    def script(msgs):
        n = msgs[1]["content"].count("\n- ")
        return '{"tool": "bank.accounts", "args": {}}' if n % 2 == 0 else '{"tool": "world.clock", "args": {}}'

    digests = []
    for backend in ("echo", "replay"):
        w, a = small_world()
        if backend == "echo":
            pol = LLMPolicy(backend=EchoLLM(script), goal="look around")
        else:
            pol = LLMPolicy(backend=ReplayLLM(transcript), goal="look around")
        Runner(w, [(a, pol)], dt=600).run(6)
        digests.append(w.digest())
        transcript = pol.transcript
    assert digests[0] == digests[1]
    with pytest.raises(RuntimeError, match="diverged"):              # a different prompt is detected
        w, a = small_world()
        Runner(w, [(a, LLMPolicy(backend=ReplayLLM(transcript), goal="another goal"))], dt=600).run(1)


@pytest.mark.skipif(not _gpu_and_model("Qwen/Qwen3-0.6B"), reason="needs GPU + local Qwen3-0.6B")
def test_real_qwen_llm_policy_is_deterministic_and_counts_tokens():
    from crucible.agents.llm import HFBackend
    be = HFBackend("Qwen/Qwen3-0.6B", device="cuda")
    texts = []
    for _ in range(2):
        w, a = small_world()
        pol = LLMPolicy(backend=be, goal="Find out how much money you have.", max_new_tokens=64)
        Runner(w, [(a, pol)], dt=60).run(2)
        texts.append([r["text"] for r in pol.transcript])
        u = pol.usage()
        assert u["calls"] == 2 and u["prompt_tokens"] > 200 and u["gen_tokens"] > 0
    assert texts[0] == texts[1]


# ============================================================================================== structured actions
def test_action_templates_fill_slots_from_what_the_actor_has_seen():
    from crucible.integrations import OMIT, ActionTemplates, EntityMemory
    mem = EntityMemory()
    mem.observe(None, {"tool": "market.list", "args": {"title": "x"}}, {"ok": True, "result": {"listing": "lst_000003"}})
    mem.observe({"time": 0, "bank": {"balances": {"acct_000018": 500}}, "events": [
        {"type": "market.listing", "actor": "o:rival", "payload": {"id": "lst_000002", "price": 7}}]})
    tools = {"name": "market.update", "inputSchema": {"type": "object", "required": ["listing"], "properties": {
        "listing": {"type": "string", "maxLength": 40}, "price_cents": {"type": "integer", "minimum": 0},
        "status": {"type": "string", "enum": ["active", "closed"]}}}}
    at = ActionTemplates(mem, "p:me")
    slots = at.slots(tools)
    assert [s.param for s in slots] == ["listing", "price_cents", "status"]
    assert slots[0].options[0].value == "lst_000003" and slots[0].options[0].source == "key"
    assert slots[1].options[0].value == OMIT and 5 in [o.value for o in slots[1].options]
    assert [o.value for o in slots[2].options] == [OMIT, "active", "closed"]
    call = ActionTemplates.build(tools, slots, [0, [o.value for o in slots[1].options].index(5), 0])
    assert call == {"tool": "market.update", "args": {"listing": "lst_000003", "price_cents": 5}}
    commit = {"name": "code.commit", "inputSchema": {"type": "object", "required": ["repo", "files"], "properties": {
        "repo": {"type": "string"}, "files": {"type": "object"}}}}
    assert at.slots(commit) is None            # free-form content cannot be filled from observations


# ============================================================================================== neural RL
def test_rl_lifetime_learns_online_checkpoints_and_resumes(tmp_path):
    import torch
    from crucible.agents.rl import Lifetime, LifetimeConfig
    cfg = LifetimeConfig(mode="fork", policy="a2c", dt=3600, episode_steps=24, total_steps=48, ckpt_every=1,
                         a2c={"n_steps": 12}, out_dir=str(tmp_path / "rl"), seed=1)
    lt = Lifetime(cfg)
    p0 = [p.detach().clone() for p in lt.net.parameters()]
    hist = lt.run()
    assert len(hist) == 2 and lt.learner.updates == 4
    assert any(not torch.equal(a, b) for a, b in zip(p0, lt.net.parameters()))
    assert all(set(h) >= {"ret", "progress", "rev7_cents", "listing_price", "service_live"} for h in hist)
    lt2 = Lifetime.resume(str(tmp_path / "rl" / "ckpt.pt"))
    assert lt2.steps == 48 and all(torch.equal(a, b) for a, b in zip(lt.net.parameters(), lt2.net.parameters()))
    lt2.cfg.total_steps = 72
    lt2.run()
    assert lt2.steps == 72 and len(lt2.history) == 3
    rnd = Lifetime(LifetimeConfig(mode="persistent", policy="random", dt=3600, episode_steps=24, total_steps=24,
                                  out_dir=str(tmp_path / "rnd"), seed=1))
    h = rnd.run()
    assert len(h) == 1 and h[0]["actions"] + h[0]["waits"] == 24
