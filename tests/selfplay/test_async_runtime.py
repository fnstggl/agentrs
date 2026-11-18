"""Integration tests for the asynchronous self-play runtime on the real economic environments."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
import time

import pytest

from agentrs.econ.core import REGISTRY, load_all, make_task
from agentrs.selfplay import Collector, InferenceEngine, WorkUnit
from agentrs.selfplay.workers import collect

load_all()


def _goal_tag(messages) -> str:
    m = re.search(r"TASK: (.*)", messages[-1]["content"])
    return hashlib.sha1(m.group(1).encode()).hexdigest()[:8]


class FakePolicy:
    """Deterministic stand-in for the GPU policy: records batch sizes, echoes the episode's own goal
    hash into the generated action (so cross-talk between workers would be visible)."""

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.batches: list[int] = []
        self.threads = set()

    def __call__(self, reqs, greedy):
        self.batches.append(len(reqs))
        self.threads.add(threading.get_ident())
        time.sleep(self.delay)
        out = []
        for msgs, prefill in reqs:
            tag = _goal_tag(msgs)
            text = f'noop", "args": {{"tag": "{tag}"}}}}'
            out.append({"text": text, "ids": [1, 2, 3], "logprobs": [-0.1, -0.2, -0.3], "finished": True,
                        "prompt": f"PROMPT[{tag}]"})
        return out


def _tasks(n, fams=("finance/payment_matching", "support/return_items", "crm/pipeline_update", "research/comparison")):
    return [make_task(fams[i % len(fams)], "train", 800_000 + i) for i in range(n)]


def test_concurrent_requests_share_batches_and_workers_are_isolated():
    pol = FakePolicy(delay=0.05)
    tasks = _tasks(16)
    for t in tasks:
        t.max_steps = 3
    eps, st = collect(pol, "v7", [WorkUnit(t) for t in tasks], n_workers=16, max_batch=32, max_wait_ms=20)
    assert len(eps) == 16 and st["episodes"] == 16
    assert max(pol.batches) > 1, pol.batches               # requests from different workers were batched
    assert st["mean_batch"] > 2
    assert len(pol.threads) == 1                           # one shared engine thread (one model), not one per worker
    for e in eps:
        assert len(e.steps) == 3 and e.policy_version == "v7"
        own = hashlib.sha1(e.goal.encode()).hexdigest()[:8]
        for s in e.steps:                                  # every action/prompt belongs to this episode's own task
            assert f'"tag": "{own}"' in s.action and s.prompt == f"PROMPT[{own}]"
            assert s.token_ids == [1, 2, 3] and s.behavior_logprobs == [-0.1, -0.2, -0.3]
        assert e.verification and "error" not in e.verification   # the hidden verifier ran on the final state
        assert e.success is False


def test_slow_environment_does_not_block_other_workers(monkeypatch):
    """One worker's environment step takes 2 s; the others keep generating and stepping meanwhile."""
    pol = FakePolicy(delay=0.01)
    tasks = _tasks(6)
    for t in tasks:
        t.max_steps = 4
    slow_id = tasks[0].id
    from agentrs.econ import rollout as ro

    orig = ro.make_env

    def make_env(task):
        env = orig(task)
        if task.id == slow_id:
            real = env.step

            def slow_step(a):
                time.sleep(2.0)
                return real(a)
            env.step = slow_step
        return env

    monkeypatch.setattr("agentrs.selfplay.workers.make_env", make_env)
    t0 = time.time()
    eps, st = collect(pol, "v1", [WorkUnit(t) for t in tasks], n_workers=6, max_batch=8, max_wait_ms=5)
    wall = time.time() - t0
    fast = [e for e in eps if e.task_id != slow_id]
    assert len(fast) == 5 and all(len(e.steps) == 4 for e in fast)
    assert all(e.wall_s < 2.0 for e in fast), [e.wall_s for e in fast]   # finished while the slow env was stepping
    assert wall >= 8.0                                                     # the slow episode itself took 4 x 2 s


def test_failures_are_contained_and_reported(monkeypatch):
    pol = FakePolicy(delay=0.0)
    tasks = _tasks(6)
    for t in tasks:
        t.max_steps = 2
    bad_setup, bad_step, hang = tasks[0].id, tasks[1].id, tasks[2].id
    from agentrs.econ import rollout as ro

    orig = ro.make_env

    def make_env(task):
        if task.id == bad_setup:
            raise RuntimeError("database unavailable")
        env = orig(task)
        if task.id == bad_step:
            env.step = lambda a: (_ for _ in ()).throw(ValueError("tool crashed"))
        if task.id == hang:
            env.step = lambda a: time.sleep(5)
        return env

    monkeypatch.setattr("agentrs.selfplay.workers.make_env", make_env)

    async def run():
        eng = InferenceEngine(pol, "v1", max_batch=8, max_wait_ms=5)
        await eng.start()
        col = Collector(eng, 3, step_timeout_s=1.0)
        try:
            return await col.run([WorkUnit(t) for t in tasks])
        finally:
            await eng.stop()
            col.close()

    eps = asyncio.run(run())
    by = {e.task_id: e for e in eps}
    assert len(eps) == 6
    assert "env setup failed" in by[bad_setup].verification["error"]
    assert "step crashed" in by[bad_step].verification["error"]
    assert "step timeout" in by[hang].verification["error"]
    for t in tasks[3:]:
        assert "error" not in by[t.id].verification and len(by[t.id].steps) == 2
    assert all(not e.success and e.reward == 0.0 for e in eps)


def test_policy_version_is_pinned_per_episode():
    pol = FakePolicy(delay=0.02)
    tasks = _tasks(4)
    for t in tasks:
        t.max_steps = 6

    async def run():
        eng = InferenceEngine(pol, "v1", max_batch=8, max_wait_ms=5)
        await eng.start()
        col = Collector(eng, 4)

        async def bump():
            await asyncio.sleep(0.1)
            while True:  # only allowed between batches
                try:
                    eng.set_version("v2")
                    return
                except RuntimeError:
                    await asyncio.sleep(0.001)
        try:
            res = await asyncio.gather(col.run([WorkUnit(t) for t in tasks]), bump())
        finally:
            await eng.stop()
            col.close()
        return res[0]

    eps = asyncio.run(run())
    assert eps and all(e.policy_version == "v1" for e in eps)
    assert all("stale" in e.verification.get("error", "") for e in eps)  # never silently mixed versions


def test_durable_sink_survives_and_counts(tmp_path):
    pol = FakePolicy(delay=0.0)
    tasks = _tasks(5)
    for t in tasks:
        t.max_steps = 2
    sink = tmp_path / "stream.jsonl"
    eps, _ = collect(pol, "v3", [WorkUnit(t, group=f"g{i // 2}") for i, t in enumerate(tasks)], n_workers=3, sink=sink)
    lines = [json.loads(x) for x in sink.read_text().splitlines()]
    assert len(lines) == 5 and {r["episode_id"] for r in lines} == {e.episode_id for e in eps}
    assert all(r["policy_version"] == "v3" and r["steps"][0]["token_ids"] == [1, 2, 3] for r in lines)
    assert {r["group"] for r in lines} == {"g0", "g1", "g2"}


@pytest.mark.network
@pytest.mark.slow
def test_real_qwen_exact_logprobs_and_nonzero_rl_gradient():
    """Real Qwen3-0.6B (+LoRA) on CPU behind the engine: the behaviour log-probs recorded at sampling
    time equal the learner's re-scored log-probs of the exact sampled ids; re-tokenising the decoded
    text is never needed; a verified-reward policy-gradient step changes the LoRA weights."""
    import torch

    from agentrs.econ.learner import GenPolicy, group_advantages, pg_samples, policy_backbone_cfg, ppo_update
    from agentrs.training.common import GradientTrainer, OptimConfig, param_snapshot

    torch.manual_seed(0)
    cfg = policy_backbone_cfg("Qwen/Qwen3-0.6B").model_copy(update={"device": "cpu", "dtype": "float32",
                                                                     "gradient_checkpointing": False})
    pol = GenPolicy(cfg, max_new_tokens=24, gen_batch=8)
    tasks = _tasks(4)
    for t in tasks:
        t.max_steps = 2
    act = lambda reqs, g: pol.act(reqs, greedy=g, temperature=1.0)  # noqa: E731
    units = [WorkUnit(t, group="G") for t in tasks]                  # one GRPO group
    eps, st = collect(act, "v0", units, n_workers=4, max_batch=8, max_wait_ms=50)
    assert st["mean_batch"] > 1
    steps = [s for e in eps for s in e.steps if s.token_ids]
    assert steps
    lp, mask = pol.token_logps([s.prompt for s in steps], [s.token_ids for s in steps])
    for i, s in enumerate(steps):
        n = int(mask[i].sum())
        assert n == len(s.token_ids) == len(s.behavior_logprobs)
        diff = (lp[i, :n].detach() - torch.tensor(s.behavior_logprobs)).abs().max().item()
        assert diff < 2e-3, diff
        assert s.action == s.prefill + pol.tok.decode(s.token_ids, skip_special_tokens=True).split("\n")[0]
    # verified rewards (hidden verifiers) with one forced success to get a non-degenerate group
    eps[0].reward, eps[0].success = 1.0, True
    adv = group_advantages(eps)
    samples = pg_samples(eps, adv)
    assert samples and all(s.version == "v0" for s in samples)
    tr = GradientTrainer(pol.bb, OptimConfig(lr=1e-4, total_steps=10))
    snap = param_snapshot(pol.bb)
    stats = ppo_update(pol, tr, samples, "v0", minibatch=8, micro=4)
    assert stats["updates"] >= 1 and stats["rejected_stale"] == 0 and stats["grad_norm"] > 0
    change = sum(float((p.detach() - snap[n]).pow(2).sum()) for n, p in pol.bb.named_parameters() if n in snap)
    assert change > 0
    stale = ppo_update(pol, tr, samples, "v1")   # samples from v0 are not used by a v1 learner
    assert stale["updates"] == 0 and stale["rejected_stale"] == len(samples)
