"""End-to-end loops on the offline scratch backbone."""

import json

import torch

from agentrs.agent.runtime import run_episode
from agentrs.config import load_config
from agentrs.data.storage import ExperienceStore
from agentrs.environments.miniweb.tasks import make_task, task_set
from agentrs.evaluation.benchmark import make_env
from agentrs.models.critic import Critic, CriticConfig
from agentrs.models.policy import LMPolicy, PolicyActor, PolicyConfig
from agentrs.models.world_model import WorldModel, WorldModelConfig
from agentrs.planning.mpc import MPCPlanner, PlannerConfig
from agentrs.training.bc import BCConfig, train_bc
from agentrs.training.common import OptimConfig, load_model, param_change, param_snapshot, save_checkpoint
from agentrs.training.rl import RLConfig, RLTrainer
from agentrs.training.world_model import WMTrainConfig, train_world_model


def test_observe_predict_act_verify_train_cycle(scratch_bb, tmp_path):
    torch.manual_seed(0)
    pol = LMPolicy(PolicyConfig(backbone=scratch_bb))
    wm = WorldModel(WorldModelConfig(backbone=scratch_bb, n_heads=2))
    cr = Critic(CriticConfig(backbone=scratch_bb))
    planner = MPCPlanner(pol, wm, cr, PlannerConfig(num_candidates=3, horizon=2, branch_candidates=2,
                                                    max_wm_calls=12), seed=0)
    env = make_env("miniweb")
    ep = run_episode(env, make_task("fill_field", "train", 0), planner)
    assert planner.stats["decisions"] == ep.num_steps and planner.stats["wm_calls"] > 0
    for t in ep.transitions:
        assert t.selection == "planner" and t.wm_prediction is not None  # predictions recorded before acting
        assert t.behavior_logprob is not None and t.policy_logprob is not None
    # real outcomes are stored and used to train the world model
    st = ExperienceStore(tmp_path)
    st.write_episodes([ep])
    snap = param_snapshot(wm)
    train_world_model(wm, st.load_transitions(), [], WMTrainConfig(steps=3, batch_size=4, eval_every=10**9))
    assert param_change(wm, snap) > 0
    # planning can be disabled (policy-only ablation)
    planner.cfg.enabled = False
    ep2 = run_episode(env, make_task("fill_field", "train", 0), planner)
    assert all(t.selection == "policy" for t in ep2.transitions)


def test_demonstrations_to_bc_checkpoint(scratch_bb, demo_data, tmp_path):
    train, val = demo_data
    pol = LMPolicy(PolicyConfig(backbone=scratch_bb))
    prompt = train[0].belief_prompt
    before = pol.log_probs([prompt], [train[0].candidates])[0].detach()
    train_bc(pol, train, val, BCConfig(steps=20, batch_size=8, eval_every=10, optim=OptimConfig(lr=3e-3)),
             out_dir=str(tmp_path / "bc"))
    pol2 = load_model(tmp_path / "bc")
    after = pol2.log_probs([prompt], [train[0].candidates])[0].detach()
    assert not torch.allclose(before, after)
    assert json.loads((tmp_path / "bc" / "model.json").read_text())["extra"]["stage"] == "bc"


def test_reward_to_rl_update_to_checkpoint(scratch_bb, tmp_path):
    pol = LMPolicy(PolicyConfig(backbone=scratch_bb, length_norm=1.0))
    cr = Critic(CriticConfig(backbone=scratch_bb))
    tr = RLTrainer(pol, cr, RLConfig(algo="ppo", epochs=2, policy_optim=OptimConfig(lr=1e-3)))
    env = make_env("miniweb")
    eps = [run_episode(env, t, PolicyActor(pol, "sample", seed=i))
           for i, t in enumerate(task_set("train", ["click_button", "menu"], n_per_family=6))]
    assert any(e.success for e in eps) and not all(e.success for e in eps)
    snap = param_snapshot(pol)
    stats = tr.update([e.transitions for e in eps])
    assert param_change(pol, snap) > 0 and "entropy" in stats and stats["n_steps"] > 0
    p = save_checkpoint(tmp_path / "rl_policy", pol, tr.pt)
    assert (p / "trainer.pt").exists()
    assert torch.allclose(load_model(p).log_probs(["GOAL: x\nACTION:"], [["finish()", "wait()"]])[0],
                          pol.log_probs(["GOAL: x\nACTION:"], [["finish()", "wait()"]])[0])


def test_continual_cycles_with_separate_processes(scratch_bb, tmp_path):
    from agentrs.training.continual import ContinualConfig, run_continual

    cfg = ContinualConfig(root=str(tmp_path / "cont"), cycles=2, episodes_per_cycle=6,
                          policy=PolicyConfig(backbone=scratch_bb, length_norm=1.0),
                          world_model=WorldModelConfig(backbone=scratch_bb, n_heads=2),
                          critic=CriticConfig(backbone=scratch_bb), wm_steps_per_cycle=2, eval_tasks_per_family=1,
                          separate_processes=True)
    cfg.curriculum.stages = [["click_button", "menu"], ["fill_field"]]
    cfg.curriculum.min_episodes = 2
    rep = run_continual(cfg)
    assert len(rep["cycles"]) == 2
    from agentrs.checkpoints.registry import CheckpointRegistry

    reg = CheckpointRegistry(tmp_path / "cont" / "registry.sqlite")
    statuses = {r.status for r in reg.list("policy")}
    assert reg.production("policy") is not None
    assert statuses & {"promoted", "rejected", "retired"}
    for c in rep["cycles"]:
        assert c["actor"]["policy"]  # actor acted with a fixed, versioned checkpoint
        assert "reason" in c["evaluation"]
    # learner state persists: shards are not re-ingested
    state = json.loads((tmp_path / "cont" / "learner_state.json").read_text())
    assert len(state["processed"]) == 2


def test_smoke_experiment_and_report(tmp_path):
    from agentrs.evaluation.report import build_report
    from agentrs.training.pipeline import ExperimentConfig, run_experiment

    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    runs = []
    for v in ("A", "D"):
        cfg = load_config(ExperimentConfig, root / "configs/experiments/smoke.yaml",
                          [f"out_dir={tmp_path}", f"name=smoke_{v}", f"variant={v}", "rl_iterations=1",
                           "eval_sets.val=1", "eval_sets.test=1", "eval_sets.test_family=1"])
        res = run_experiment(cfg)
        runs.append(str(tmp_path / f"smoke_{v}"))
        assert "policy/test_family" in res["final"]
    assert "planner/val" in res["final"]  # D evaluates with and without planning
    md = build_report(runs)
    assert "A policy-only" in md and "D model-based" in md
