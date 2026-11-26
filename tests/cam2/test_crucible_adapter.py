"""CAM 2.0 EnvironmentAdapter for Crucible worlds: invalid JSON vs task failure, Crucible schema rejections with error codes,
snapshots and lifetimes, the CAM 2.0 runner on Crucible, and exact behaviour log-prob replay on real Qwen3-0.6B.

Relocated from crucible/tests/test_agents_llm.py with the adapter it tests (Crucible DECISION_LOG D-81): Crucible is
independent of CAM (Clarification 1), so the CAM 2.0 / superlearner adapters and their tests live on the AGENTRS side,
which imports Crucible. The test bodies are the Crucible 4.0 ones (c07098b) with the module path renamed and the 4.0
reward contract selected explicitly (Crucible OO.8: observation-only is the default)."""
from __future__ import annotations

import json  # noqa: F401
import os
import sys

import pytest

CRUCIBLE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "crucible", "src"))
if CRUCIBLE_SRC not in sys.path:
    sys.path.insert(0, CRUCIBLE_SRC)

import crucible  # noqa: E402

crucible.load_all_modules()


def _gpu_and_model(name: str) -> bool:
    try:
        import torch
        from transformers.utils import cached_file
        cached_file(name, "config.json", local_files_only=True)
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


# ============================================================================================== CAM 2.0 adapter
CAM_SMALL = {"d": 64, "integ_layers": 1, "heads": 4, "core_dim": 64, "wm_hidden": 64, "head_hidden": 64,
             "prefix_len": 4, "prefix_rank": 16, "mem_k": 0, "wm_ensemble": 2}


def _cam_env(**kw):
    from agentrs.cam2.envs.crucible_adapter import CrucibleAdapter
    from agentrs.cam2.envs.base import TaskRef
    env = CrucibleAdapter(**{"max_steps": 6, "reward_mode": "legacy_shaped_reward", **kw})   # the 4.0 adapter, named
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
    from agentrs.cam2.envs.crucible_adapter import smoke_run  # noqa: F401  (import check)
    pool = EnvPool("crucible", 2, {"max_steps": 3, "reward_mode": "legacy_shaped_reward"})   # 4.0 adapter, named
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
    from agentrs.cam2.envs.crucible_adapter import smoke_run
    r = smoke_run(str(tmp_path / "smoke"), llm="Qwen/Qwen3-0.6B", workers=2, updates=1, episodes_per_update=2,
                  max_steps=4)
    u = r["updates"][0]
    assert r["generated_texts"] > 0 and u["prefix_deltas"] > 0
    assert u["mismatch_text_max"] < 1e-4 and u["replay_mismatch_max"] < 1e-4
    assert r["llm_fingerprint_unchanged"] and r["llm_trainable_params"] == 0
    assert r["cam_checksum_changed"] and r["cam_tensors_changed"] > 0
