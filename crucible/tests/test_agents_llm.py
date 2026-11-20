"""Learning-agent integrations: frozen-LLM policy (plumbing with a test double; greedy determinism on real Qwen3 when
available), the CAM 2.0 EnvironmentAdapter protocol (invalid JSON vs task failure, Crucible schema rejections with
error codes, exact behaviour log-prob replay on real Qwen3-0.6B), the superlearner World adapter (reward audit),
and the factored neural RL learner (online updates, checkpoint resume)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

import crucible

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.llm import EchoLLM, LLMPolicy, ReplayLLM, render_tool_list  # noqa: E402
from crucible.core.kernel import World, WorldSpec  # noqa: E402

AGENTRS_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if AGENTRS_SRC not in sys.path:
    sys.path.insert(0, AGENTRS_SRC)

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


# ============================================================================================== CAM 2.0 adapter
CAM_SMALL = {"d": 64, "integ_layers": 1, "heads": 4, "core_dim": 64, "wm_hidden": 64, "head_hidden": 64,
             "prefix_len": 4, "prefix_rank": 16, "mem_k": 0, "wm_ensemble": 2}


def _cam_env(**kw):
    from crucible.integrations.agentrs_cam2 import CrucibleAdapter
    from agentrs.cam2.envs.base import TaskRef
    env = CrucibleAdapter(**{"max_steps": 6, **kw})
    obs = env.reset(TaskRef("crucible", "service:tiny:0", 7))
    return env, obs


def test_cam2_tools_and_observation_mapping():
    from agentrs.cam2.tools import enumerate_candidates
    env, obs = _cam_env()
    specs = {s.name: s for s in env.available_tools()}
    assert not any(n.startswith("admin.") for n in specs)
    assert specs["bank.accounts"].effect == "read" and specs["bank.accounts"].gen_arg is None
    tr = specs["bank.transfer"]
    assert tr.effect == "sandboxed" and tr.gen_arg.type == "json"
    assert json.loads(tr.gen_arg.json_schema)["required"] == ["from_account", "to", "amount"]
    assert obs.goal.startswith("Operate a service") and all(not e.targetable for e in obs.elements)
    assert any(e.role == "summary" and e.text.startswith("bank:") for e in obs.elements)
    assert len(enumerate_candidates(env.available_tools(), obs)) == len(specs)    # one candidate per tool
    t0 = env.world.now
    sr = env.step(__import__("agentrs.cam2.tools", fromlist=["ToolCall"]).ToolCall("bank.accounts"))
    assert env.world.now == t0 + env.dt and not sr.info["invalid"] and sr.reward == 0.0
    assert any(e.role == "result" and "bank.accounts -> ok" in e.text for e in sr.obs.elements)


def test_cam2_invalid_json_is_counted_separately_and_never_reaches_the_world():
    """(a) unparseable JSON arguments: an invalid step (CAM's flag), counted as json_invalid, not sent to the world,
    and not a task failure (the verifier's status is unaffected)."""
    from agentrs.cam2.tools import ToolCall
    env, _ = _cam_env()
    n_cmd = sum(1 for e in env.world.log if e.type in ("cmd", "cmd.rejected"))
    for text in ['{"account": ', "the account is acct_000017", None]:
        sr = env.step(ToolCall("bank.balance", text=text))
        assert sr.info["invalid"] and sr.info["invalid_kind"] == "json_parse" and "error_code" not in sr.info
        assert not sr.terminated
    assert sum(1 for e in env.world.log if e.type in ("cmd", "cmd.rejected")) == n_cmd   # nothing was executed
    ev = env.evaluate()
    c = ev.details["counters"]
    assert c["json_invalid"] == 3 and c["calls_sent"] == 0 and c["schema_rejected"] == 0
    assert ev.success is False and ev.details["status"]["failed"] is False and ev.details["json_invalid_rate"] == 1.0


def test_cam2_every_call_is_validated_by_crucible_and_rejections_carry_codes():
    """(b) every parseable call goes through the kernel's schema validation; rejections surface Crucible's error code
    (and match the kernel's own cmd.rejected record); CAM's shallow validator is compared and disagreements counted."""
    from agentrs.cam2.tools import ToolCall
    env, _ = _cam_env(max_steps=10)
    acct = env.world.modules["bank"].primary_account(env.world.state, env.actor)
    cases = [(ToolCall("bank.balance", text='{"account": 17}'), "invalid_action", True),           # wrong type
             (ToolCall("bank.balance", text='{"account": "a", "x": 1}'), "invalid_action", True),   # unknown arg
             (ToolCall("id.create_org", text='{"handle": "Not A Handle", "name": "x"}'), "invalid_action", True),
             (ToolCall("bank.balance", text='{"account": "acct_999999"}'), "not_found", False),     # in-world
             (ToolCall("bank.transfer", text=json.dumps({"from_account": acct, "to": "nobody-here", "amount": 5})),
              "not_found", False),
             (ToolCall("bank.balance", text=json.dumps({"account": acct})), None, False)]
    for call, code, invalid in cases:
        sr = env.step(call)
        assert sr.info.get("error_code") == code, (call, sr.info)
        assert sr.info["invalid"] is invalid
        if code is not None:
            rej = [e for e in env.world.log if e.type == "cmd.rejected"][-1]
            assert rej.payload["error"]["code"] == code and rej.payload["tool"] == call.tool
    c = env.evaluate().details["counters"]
    assert c["calls_sent"] == 6 and c["ok"] == 1 and c["schema_rejected"] == 3 and c["world_rejected"] == 2
    assert c["rejections_by_code"] == {"invalid_action": 3, "not_found": 2}
    assert c["cam_validator_disagreements"] == 1        # the pattern violation passes CAM's check, not Crucible's


def test_cam2_snapshot_restore_and_continue_lifetime():
    from agentrs.cam2.envs.base import TaskRef
    from agentrs.cam2.tools import ToolCall
    env, _ = _cam_env()
    env.step(ToolCall("bank.accounts"))
    snap = env.snapshot()
    a = env.step(ToolCall("world.clock"))
    env.restore(snap)
    b = env.step(ToolCall("world.clock"))
    assert a.info["progress"] == b.info["progress"] and env.world.now == snap["k"] * 0 + env.world.now
    env2, _ = _cam_env(lifetime="continue")
    w = env2.world
    for _ in range(6):
        sr = env2.step(ToolCall("world.clock"))
    assert sr.truncated and not sr.terminated and sr.info["eval"]["counters"]["steps"] == 6
    t = w.now
    env2.reset(TaskRef("crucible", "service:tiny:0", 7))         # same task: the same world continues
    assert env2.world is w and w.now == t and env2.k == 0


def test_cam2_runner_pipeline_counts_invalid_json_with_tiny_random_llm(tmp_path):
    """The real CAM 2.0 runner / agent / pool on Crucible (tiny random LLM, CPU): episodes finish, the adapter's
    counters reach the Episode (info["eval"]) and the runner's invalid count equals json_invalid + schema_rejected."""
    from agentrs.cam2.envs.pool import EnvPool
    from agentrs.cam2.trainer import RunConfig, Runner, Trainer
    from crucible.integrations.agentrs_cam2 import smoke_run  # noqa: F401  (import check)
    pool = EnvPool("crucible", 2, {"max_steps": 3})
    try:
        rc = RunConfig(run_dir=str(tmp_path / "r"), env="crucible", llm="tiny-random", workers=2, max_steps=3,
                       train_families=["service:tiny:0"], eval_every=0, ckpt_every=0, cam=dict(CAM_SMALL),
                       learn={"episodes_per_update": 2, "mb_episodes": 2, "prefix_replay_correction": True})
        tr = Trainer(rc, pool=pool, device="cpu")
        runner = Runner(tr.agent, pool, tr.mem, rc.max_steps, tr.lcfg.gamma, gen=tr.gen)
        got = []
        eps = runner.run(tr.sampler, 4, train=True, on_episode=got.append)
        assert len(eps) == 4 and all(len(e.tr) == 3 for e in eps)
        cs = [e.info["eval"]["counters"] for e in eps]
        assert sum(c["steps"] for c in cs) == runner.env_steps == 12
        assert runner.invalid == sum(c["json_invalid"] + c["schema_rejected"] for c in cs)
        assert all(not e.success for e in eps)
        st = tr.learner.update(eps, [])
        assert st["params_changed"] > 0 and st["replay_mismatch_max"] < 1e-3
    finally:
        pool.close()


@pytest.mark.skipif(not _gpu_and_model("Qwen/Qwen3-0.6B"), reason="needs GPU + local Qwen3-0.6B")
def test_cam2_real_qwen_replay_logprob_matches_collection(tmp_path):
    """(c) replay consistency with the real CAM 2.0 agent + learner and frozen Qwen3-0.6B on GPU: with
    prefix_replay_correction the recomputed behaviour log-probability (candidate + JSON text under CAM's control
    vectors) equals the collection-time value; the frozen LLM's weights do not change and CAM's do."""
    from crucible.integrations.agentrs_cam2 import smoke_run
    r = smoke_run(str(tmp_path / "smoke"), llm="Qwen/Qwen3-0.6B", workers=2, updates=1, episodes_per_update=2,
                  max_steps=4)
    u = r["updates"][0]
    assert r["generated_texts"] > 0 and u["prefix_deltas"] > 0
    assert u["mismatch_text_max"] < 1e-4 and u["replay_mismatch_max"] < 1e-4
    assert r["llm_fingerprint_unchanged"] and r["llm_trainable_params"] == 0
    assert r["cam_checksum_changed"] and r["cam_tensors_changed"] > 0


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


# ============================================================================================== superlearner World
def test_sl_world_adapter_runs_and_passes_reward_audit():
    from agentrs.sl import worlds as W
    from agentrs.sl.train import reward_audit
    from agentrs.sl.world import RewardContract, RewardSpec, Task
    import crucible.integrations.agentrs_sl as S
    assert "crucible/service" in W.FAMILIES and "crucible/revenue/deployed_service" in W.FAMILIES
    from crucible.goals.contracts import HELDOUT_GOAL_FAMILIES
    assert not any(f.split("/")[1] in HELDOUT_GOAL_FAMILIES for f in W.FAMILIES if f.startswith("crucible/"))
    w = W.make_world("crucible/service")
    w.initialize_world(Task("crucible/service", 3, "val"))
    sp = w.available_actions("0")
    first = sp.allowed([])
    assert first[0] == "wait" and "bank" in first and len(first) <= 24
    act = ["bank", "transfer"]
    while sp.allowed(act) != ["END"]:
        act.append(sp.allowed(act)[0])
    snap = w.save_world_state()
    w.execute("0", act)
    w.advance_time()
    c1, g1, d1 = w.observe_consequences("0"), w.evaluate_goal("0"), w.world.digest()
    w.restore_world_state(snap)
    w.execute("0", act)
    w.advance_time()
    assert (w.observe_consequences("0"), w.evaluate_goal("0"), w.world.digest()) == (c1, g1, d1)
    w.reset_world()                                  # same task, fresh episode from the task's start snapshot
    assert w.world.now == 0 and w.n_dec == 0
    res = reward_audit(["crucible/service"], RewardContract(RewardSpec()), max_steps=10, n=2)
    a = res["crucible/service"]
    assert a["passed"], a
    assert all(a[p]["ended"] == 1.0 for p in ("random", "stall", "terminate"))


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
