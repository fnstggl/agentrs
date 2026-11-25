"""Launcher wiring of the shared semantic server: one Qwen3-0.6B process started and supervised by the launcher,
actor processes receive a SemanticClient through their policy factory, and the server's compute is accounted with
the run (spec 22 shared semantic-model serving; manifest X06)."""
from __future__ import annotations

import json
import os

import pytest
import torch

from superlearner.cam_unified.dist.actor import ActorSpec
from superlearner.cam_unified.dist.launcher import Launcher, SemServeSpec
from superlearner.cam_unified.dist.learner import LearnerSpec
from superlearner.cam_unified.dist.toyworld import TinyRecurrentPolicy
from superlearner.cam_unified.observe.trace import load_traces

HF = os.environ.get("HF_HOME", "/workspace/.hf_home")
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    not (torch.cuda.is_available() and os.path.isdir(os.path.join(HF, "hub", "models--Qwen--Qwen3-0.6B"))),
    reason="needs CUDA + Qwen3-0.6B")]


class _GoalEncodingPolicy(TinyRecurrentPolicy):
    """Toy policy that encodes each episode's goal text and first observation through the shared server."""

    def __init__(self, semantic=None, **kw):
        super().__init__(**kw)
        assert semantic is not None, "the launcher must hand the policy a semantic client"
        self.sem = semantic
        self.sem_info = {}

    def reset(self, obs, goals, tools):
        super().reset(obs, goals, tools)
        e = self.sem.embed([g.text for g in goals])
        h, m = self.sem.hidden([f"gauge {o.meta.get('i')} reads {o.value:.3f} {o.unit}" for o in
                                obs.observations[0][:3]], max_tokens=32)
        self.sem_info = {"sem_version": self.sem.version, "goal_dim": int(e.shape[-1]),
                         "goal_norm": float(e.norm(dim=-1).mean()), "obs_tokens": int(m.sum())}

    def act(self, obs_lists, tools, goals, active=None):
        actions, infos = super().act(obs_lists, tools, goals, active=active)
        for i in infos:
            if i:
                i["reasoning"] = {**i.get("reasoning", {}), **self.sem_info}
        return actions, infos


def make_goal_encoding_policy(seed=0, semantic=None, **kw):
    return _GoalEncodingPolicy(semantic=semantic, seed=seed, **kw)


def test_launcher_runs_shared_semantic_server_for_all_actors(tmp_path):
    run = str(tmp_path / "run")
    L = LearnerSpec(run_dir=run, model_kwargs={"seed": 0}, min_decisions=64)
    A = [ActorSpec(actor_id=f"a{i}", run_dir=run, env_kwargs={"B": 2, "episode_len": 8}, seed=30 + i,
                   policy_factory="test_dist_launch_semserve:make_goal_encoding_policy", max_episodes=3,
                   batch_records=64) for i in range(2)]
    sem = SemServeSpec(model_name="Qwen/Qwen3-0.6B", max_wait_ms=30.0, lora_rank=4, lora_targets=("q_proj",))
    s = Launcher(run, L, A, semserve=sem).run(duration_s=180)
    assert s["restarts"] == {} and s["episodes"] == 2 * 3
    ev = [json.loads(l) for l in open(os.path.join(run, "launcher.jsonl"))]
    assert {(e["name"], e["exitcode"]) for e in ev if e["event"] == "exit"} == \
        {("a0", 0), ("a1", 0), ("learner", 0), ("semserve", 0)}
    st = json.load(open(os.path.join(run, "status", "semserve.json")))
    assert st["state"] == "finished" and st["stats"]["requests"] >= 2 * 3 * 2
    assert st["model"] == "Qwen/Qwen3-0.6B"
    traces = load_traces(os.path.join(run, "traces"))
    assert len(traces) == 2 * 3 * 2 * 8
    for t in traces:
        r = t.reasoning
        assert r["sem_version"] == "base:Qwen/Qwen3-0.6B" and r["goal_dim"] == 1024 and r["obs_tokens"] > 3
    comp = json.load(open(os.path.join(run, "compute_summary.json")))
    roles = comp["total"]["by_role"]
    assert set(roles) >= {"actor", "learner", "semserve", "launcher"}
    assert roles["semserve"]["gpu_s"] > 0 and comp["total"]["tokens_prompt"] > 0
    sem_snaps = [p for p in comp["processes"] if p["role"] == "semserve"]
    assert len(sem_snaps) == 1 and 0 < sem_snaps[0]["peak_gpu_mb"] < 6000
    actor_snaps = [p for p in comp["processes"] if p["role"] == "actor"]
    assert all(p["peak_gpu_mb"] == 0 for p in actor_snaps)    # actors hold no model copy on the GPU
