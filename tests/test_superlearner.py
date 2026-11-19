"""Superlearner core: interface, exact replay, adaptation modes, production CLI start-to-finish."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from agentrs.sl import worlds as W
from agentrs.sl.learner import PPO
from agentrs.sl.model import SLConfig, SLCore
from agentrs.sl.runner import Runner

SRC = Path(__file__).resolve().parents[1] / "src"
TINY = {"d": 48, "dh": 64, "layers": 1, "heads": 2, "dk": 16, "dv": 24, "mem_slots": 32, "skills": 3, "ds": 8, "da": 32, "ens": 2}


def test_replay_reproduces_collection_logprobs_exactly_in_every_mode():
    torch.manual_seed(0)
    m = SLCore(SLConfig(**TINY))
    for mode in ("full", "episode", "none", "gru"):
        r = Runner(m, {"cam/negotiation": 1.0, "cam/product": 1.0}, W.make_world, W.sample_task, 6, trial_episodes=2, mode=mode, seed=1)
        seg = r.collect(24)  # spans episode and trial boundaries
        lp, ent, val, wml = PPO(m, {})._replay(seg, torch.arange(r.n), mode, "cpu")
        old = torch.stack([s["lp"] for s in seg["steps"]])
        assert (lp.detach() - old).abs().max() < 1e-4, mode
        assert (val.detach() - torch.stack([s["v"] for s in seg["steps"]])).abs().max() < 1e-4


def test_memory_persists_across_episodes_only_in_full_mode():
    m = SLCore(SLConfig(**TINY))
    for mode in ("full", "episode"):
        r = Runner(m, {"cam/negotiation": 1.0}, W.make_world, W.sample_task, 2, trial_episodes=3, mode=mode, seed=2)
        for _ in range(80):
            rec, _ = r._act(True)
            rows = rec["newep"] & ~rec["newtrial"]  # first step of a later episode of the same trial
            if rows.any():
                writes = r.state["ptr"][rows]
                # full: the previous episode's entries are still in memory; episode: only this step's write
                assert bool((writes > 1).all()) if mode == "full" else bool((writes == 1).all()), (mode, writes)
                break
        else:
            raise AssertionError("no episode boundary inside a trial reached")


def test_unified_interface_vending_world_runs_and_verifies():
    if "ext/vending" not in W.FAMILIES:
        return
    from agentrs.sl.world import Task
    w = W.make_world("ext/vending")
    w.initialize_world(Task("ext/vending", 1, "val"))
    o = w.observe("0")
    assert o.numbers["balance"] > 0 and o.entities
    w.execute("0", ["wait"])
    c = w.observe_consequences("0")
    g = w.evaluate_goal("0")
    assert c.dt == 1.0 and 0 <= g.progress <= 1 and g.violated is None


def test_production_cli_start_to_finish_with_transfer_init(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(SRC), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4"}
    common = ["--set", f"model={json.dumps(TINY)}", "n_worlds=4", "seg_len=8", "trial_episodes=2", "max_env_steps=96",
              "eval_at=[0, 48]", "eval_tasks=2", "test_tasks=2", "lifetime_ckpts=[64]", 'learner={"mb_streams": 2}']
    a = tmp_path / "pre"
    p = subprocess.run([sys.executable, "-m", "agentrs.sl.train", *common, f"run_dir={a}", 'families={"cam/negotiation": 1.0}',
                        'heldout_families=["cam/bizdev"]'], env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    st = json.loads((a / "state.json").read_text())
    assert st["phase"] == "done" and not st.get("diagnostic_errors") and (a / "ckpt" / "life_64" / "weights.pt").exists()
    b = tmp_path / "tr"
    p = subprocess.run([sys.executable, "-m", "agentrs.sl.train", *common, f"run_dir={b}", 'families={"cam/bizdev": 1.0}',
                        f"init_from={a / 'ckpt' / 'life_64'}"], env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    st = json.loads((b / "state.json").read_text())
    assert st["phase"] == "done" and st["init_from"]["loaded"] == st["init_from"]["total"]


def test_reward_audit_closes_stalling_and_early_termination_exploits(monkeypatch):
    """Regression for the v1-v5 contract defects: stalling must end within the cap, and quitting at once must not be
    a cheaper failure than running out the clock (the audit's early-termination policy uses world snapshots)."""
    from agentrs.sl.train import reward_audit
    from agentrs.sl.world import RewardContract, RewardSpec
    from agentrs.sl.worlds import cam_worlds

    rc = RewardContract(RewardSpec(native=1.0, success=1.0, shaping=0.5, violation=1.0, gamma=0.99))
    a = reward_audit(["cam/negotiation", "cam/bizdev"], rc, 2000, n=6)
    assert all(v["passed"] for v in a.values()), a
    assert abs(a["cam/negotiation"]["terminate"]["return"] - a["cam/negotiation"]["stall"]["return"]) < 0.05
    # without the failure-budget charge the audit must flag the exploit
    orig = cam_worlds.CamWorld.observe_consequences

    def no_budget(self, agent="0"):
        c = orig(self, agent)
        if c.done and not (self.env.done and self.env.success):
            c.reward = -cam_worlds.DECISION_COST
        return c
    monkeypatch.setattr(cam_worlds.CamWorld, "observe_consequences", no_budget)
    assert not reward_audit(["cam/negotiation"], rc, 2000, n=6)["cam/negotiation"]["passed"]
