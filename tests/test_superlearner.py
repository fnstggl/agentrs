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


def test_progress_shaping_telescopes_under_the_runners_discount():
    """Potential-based shaping is policy-invariant only under the learner's own discount. With shaping as the only
    reward, the runner's discounted return of every completed episode must equal -Phi(s_0) = 0, whatever the policy
    did. sl-exp-v1..v10 discounted shaping by raw elapsed hours instead: product (dt ~ 400 h) and bizdev (dt ~ 38 h)
    episodes then returned strongly negative values for any progress held, so the learner unlearned progress."""
    from agentrs.sl.world import RewardContract, RewardSpec
    torch.manual_seed(0)
    m = SLCore(SLConfig(**TINY))
    rc = RewardContract(RewardSpec(native=0.0, success=0.0, shaping=1.0, time_cost=0.0, violation=0.0))
    for fam in ("cam/product", "cam/bizdev", "cam/negotiation"):
        w = W.make_world(fam)
        w.initialize_world(W.sample_task(fam, __import__("random").Random(0), "train"))
        assert w.evaluate_goal().progress == 0.0
        r = Runner(m, {fam: 1.0}, W.make_world, W.sample_task, 4, trial_episodes=2, seed=3, contract=rc)
        seg = r.collect(160)
        r.close()
        rew = torch.stack([s["r"] for s in seg["steps"]])
        disc = torch.stack([s["disc"] for s in seg["steps"]]).float()
        done = torch.stack([s["done"] for s in seg["steps"]])
        n_ep, worst, progressed = 0, 0.0, 0
        for k in range(rew.shape[1]):
            acc, D, nonzero = 0.0, 1.0, False
            for t in range(rew.shape[0]):
                acc += D * float(rew[t, k])
                nonzero |= abs(float(rew[t, k])) > 1e-9
                D *= float(disc[t, k])
                if done[t, k]:
                    worst = max(worst, abs(acc))
                    n_ep += 1
                    progressed += nonzero
                    acc, D, nonzero = 0.0, 1.0, False
        assert n_ep >= 4, fam
        assert worst < 1e-4, (fam, worst)
        if fam != "cam/product":
            assert progressed > 0, fam  # the check is not vacuous: some episodes made verified progress


def test_reward_audit_catches_the_v10_shaping_discount(monkeypatch):
    from agentrs.sl.train import reward_audit
    from agentrs.sl.world import RewardContract, RewardSpec
    rc = RewardContract(RewardSpec())
    assert reward_audit(["cam/bizdev"], rc, 2000, n=6)["cam/bizdev"]["passed"]
    orig = RewardContract.components

    def v10(self, before, after, cons, disc):  # the v1-v10 contract: shaping discounted by raw elapsed hours
        return orig(self, before, after, cons, self.s.gamma ** max(0.0, cons.dt))
    monkeypatch.setattr(RewardContract, "components", v10)
    a = reward_audit(["cam/bizdev"], rc, 2000, n=6)["cam/bizdev"]
    assert not a["passed"] and a["random"]["shaping_telescope_err"] > 1e-3


def test_goal_curriculum_lowers_training_targets_only_and_adapts_to_training_success():
    """sl-exp-v12 (amendment 7): a curriculum family's training tasks carry the current difficulty and product's
    training target shrinks accordingly; validation / test tasks never do; the update rule raises the difficulty
    when training success is above ``hi``, lowers it below ``lo``, and ignores episodes played at a stale difficulty."""
    import random as _r

    from agentrs.sl import worlds as W
    from agentrs.sl.train import SLTrainer, curriculum_step
    from agentrs.sl.world import Task

    full = W.make_world("cam/product")
    full.initialize_world(Task("cam/product", 7, "train"))
    easy = W.make_world("cam/product")
    easy.initialize_world(Task("cam/product", 7, "train", params={"difficulty": 0.1}))
    assert easy.env.target == max(50.0, full.env.target * 0.1) < full.env.target
    easy.reset_world()  # later episodes of the trial keep the lowered goal
    assert easy.env.target == max(50.0, full.env.target * 0.1)
    v_full, v_easy = W.make_world("cam/product"), W.make_world("cam/product")
    v_full.initialize_world(Task("cam/product", 3, "val"))
    v_easy.initialize_world(Task("cam/product", 3, "val", params={"difficulty": 0.1}))
    assert v_easy.env.target == v_full.env.target  # a difficulty on a validation task is ignored

    class T:  # the trainer's task sampler without building a model
        state = {"curriculum": {"cam/product": {"d": 0.3, "n": 0, "s": 0}}}
    rng = _r.Random(0)
    assert SLTrainer._sample_task(T, "cam/product", rng, "train").params["difficulty"] == 0.3
    assert "difficulty" not in SLTrainer._sample_task(T, "cam/product", rng, "val").params
    assert "difficulty" not in SLTrainer._sample_task(T, "cam/negotiation", rng, "train").params

    spec = {"lo": 0.2, "hi": 0.5, "step": 2.0, "min": 0.05, "window": 4}
    cur = {"d": 0.2, "n": 0, "s": 0}
    assert curriculum_step(cur, spec, [{"difficulty": 0.2, "success": True}] * 3) is None  # window not full
    assert curriculum_step(cur, spec, [{"difficulty": 0.2, "success": True}]) == 0.4      # 4/4 > hi: harder
    assert curriculum_step(cur, spec, [{"difficulty": 0.2, "success": False}] * 8) is None  # stale difficulty ignored
    assert curriculum_step(cur, spec, [{"difficulty": 0.4, "success": False}] * 4) == 0.2   # 0/4 < lo: easier
    for _ in range(5):
        curriculum_step(cur, spec, [{"difficulty": cur["d"], "success": False}] * 4)
    assert cur["d"] == 0.05                                                                   # floor
    cur = {"d": 0.8, "n": 0, "s": 0}
    curriculum_step(cur, spec, [{"difficulty": 0.8, "success": True}] * 4)
    assert cur["d"] == 1.0                                                                    # ceiling: the full goal
