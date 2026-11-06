"""Command-line entry point: ``agentrs <command> ...`` (or ``python -m agentrs``).

Run ``agentrs --help`` for the list of commands; every command has ``--help``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

MODEL_PRESETS = {
    "smoke": "configs/models/smoke_scratch.yaml",
    "dev": "configs/models/dev_pythia14m.yaml",
    "dev31m": "configs/models/dev_pythia31m.yaml",
    "research": "configs/models/research_qwen3.yaml",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _model_cfg(preset_or_path: str, component: str) -> dict:
    from agentrs.config import load_yaml

    path = MODEL_PRESETS.get(preset_or_path, preset_or_path)
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = _repo_root() / path
    return load_yaml(p).get(component, {})


def _print(obj) -> None:
    print(json.dumps(obj, indent=1, default=str))


def _threads(a) -> None:
    if getattr(a, "threads", 0):
        import torch

        torch.set_num_threads(a.threads)


def _load_split(store_dir: str, split: str | None = None, partition: str = "real"):
    from agentrs.data.storage import ExperienceStore

    st = ExperienceStore(store_dir)
    ts = st.load_transitions(partition)
    if split:
        ts = [t for t in ts if t.split == split]
    return ts


def _guard_train(ts) -> None:
    from agentrs.data.replay import LeakageError

    bad = {t.split for t in ts if t.split != "train"}
    if bad:
        raise LeakageError(f"training data contains non-train splits {bad}; refusing to train")


# ------------------------------------------------------------------------------- commands
def cmd_setup_check(a) -> int:
    report: dict = {"python": sys.version.split()[0]}
    try:
        import peft
        import torch
        import transformers

        report.update(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
                      cuda=torch.cuda.is_available(), threads=torch.get_num_threads())
    except Exception as e:  # pragma: no cover
        report["torch_error"] = repr(e)
    from agentrs.agent.runtime import TeacherActor, run_episode
    from agentrs.demonstrations.teacher import ScriptedTeacher
    from agentrs.environments.miniweb.tasks import make_task
    from agentrs.evaluation.benchmark import make_env

    ep = run_episode(make_env("miniweb"), make_task("checkout", "train", 0), TeacherActor(ScriptedTeacher()))
    report["miniweb_teacher_episode"] = {"success": ep.success, "steps": ep.num_steps}
    mw = _repo_root() / "third_party" / "miniwob-plusplus" / "miniwob" / "html" / "miniwob"
    report["miniwob_html"] = str(mw) if mw.exists() else "missing (run scripts/setup.sh)"
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401

        report["playwright"] = "importable"
    except Exception as e:
        report["playwright"] = f"unavailable: {e!r}"
    if a.check_model:
        from agentrs.models.backbone import BackboneConfig, LMBackbone

        bb = LMBackbone(BackboneConfig(name=a.check_model))
        report["model"] = {"name": a.check_model, "hidden": bb.hidden_size}
    _print(report)
    return 0 if ep.success else 1


def _build_actor(a, env_id="miniweb"):
    from agentrs.agent.runtime import HumanCLIActor, RandomActor, TeacherActor
    from agentrs.demonstrations.teacher import ScriptedTeacher

    if a.actor == "teacher":
        return TeacherActor(ScriptedTeacher(epsilon=a.epsilon, seed=a.seed)), []
    if a.actor == "random":
        return RandomActor(a.seed), []
    if a.actor == "human":
        return HumanCLIActor(), []
    from agentrs.models.policy import PolicyActor
    from agentrs.training.common import load_model

    if not a.policy:
        raise SystemExit("--policy checkpoint required for actor policy/planner")
    pol = load_model(a.policy)
    if a.actor == "policy":
        return PolicyActor(pol, "greedy" if a.greedy else "sample", seed=a.seed), [pol.backbone]
    from agentrs.planning.mpc import MPCPlanner, PlannerConfig

    if not (a.world_model and a.critic):
        raise SystemExit("--world-model and --critic checkpoints required for the planner")
    wm, cr = load_model(a.world_model), load_model(a.critic)
    pc = PlannerConfig(horizon=a.horizon, num_candidates=a.num_candidates, uncertainty_penalty=a.uncertainty_penalty,
                       num_branches=a.branches, enabled=not a.no_planning)
    return MPCPlanner(pol, wm, cr, pc, seed=a.seed), [pol.backbone, wm.backbone, cr.backbone]


def cmd_run_episode(a) -> int:
    from agentrs.agent.runtime import run_episode
    from agentrs.data.schemas import Provenance, Source
    from agentrs.data.storage import ExperienceStore
    from agentrs.environments.miniweb.tasks import make_task
    from agentrs.evaluation.benchmark import make_env

    _threads(a)
    env = make_env(a.env)
    task = make_task(a.family, a.split, a.index)
    actor, _ = _build_actor(a, a.env)
    src = {"teacher": Source.TEACHER, "human": Source.HUMAN}.get(a.actor, Source.AGENT)
    prov = Provenance(source=src, source_id=getattr(actor, "name", a.actor), consent=(a.actor == "human"))
    ep = run_episode(env, task, actor, provenance=prov)
    print(f"GOAL: {task.goal}")
    for t in ep.transitions:
        extra = ""
        if t.wm_prediction is not None:
            extra = f"  [wm: P(r)={t.wm_prediction.reward_prob:.2f} P(done)={t.wm_prediction.terminal_prob:.2f}]"
        print(f"{t.step:2d}. {t.action:40s} valid={t.action_valid} r={t.reward}{extra}")
        if a.verbose:
            print("    diff: " + t.diff_text.replace("\n", "\n          "))
    print(f"outcome: success={ep.success} ({ep.termination_reason}), verifier evidence: {ep.verification}")
    if a.out:
        ExperienceStore(a.out).write_episodes([ep], tag="episode")
        print(f"saved to {a.out}")
    env.close()
    return 0


def cmd_record_demo(a) -> int:
    from agentrs.data.storage import ExperienceStore
    from agentrs.environments.miniweb.tasks import make_task

    if not a.consent or not a.demonstrator_id:
        raise SystemExit("recording requires --consent and --demonstrator-id (explicit consent is mandatory)")
    task = make_task(a.family, a.split, a.index)
    if a.env == "localweb":
        from agentrs.demonstrations.browser_recorder import record_episode

        ep = record_episode(task, demonstrator_id=a.demonstrator_id, consent=True, privacy=a.privacy,
                            headless=a.headless)
    else:
        from agentrs.agent.runtime import HumanCLIActor, run_episode
        from agentrs.data.schemas import Provenance, Source
        from agentrs.evaluation.benchmark import make_env

        prov = Provenance(source=Source.HUMAN, source_id=a.demonstrator_id, consent=True, privacy=a.privacy,
                          collected_by="agentrs record-demo (terminal)")
        ep = run_episode(make_env("miniweb"), task, HumanCLIActor(), provenance=prov)
    ExperienceStore(a.out).write_episodes([ep], tag="human")
    print(f"recorded {ep.num_steps} steps, verified success={ep.success}; stored under {a.out} "
          f"({'private' if a.privacy == 'private' else 'real'} partition)")
    return 0


def cmd_gen_data(a) -> int:
    from agentrs.agent.runtime import RandomActor, TeacherActor
    from agentrs.data.schemas import Provenance, Source
    from agentrs.data.storage import ExperienceStore
    from agentrs.demonstrations.teacher import ScriptedTeacher
    from agentrs.environments.miniweb.tasks import task_set
    from agentrs.evaluation.benchmark import make_env, run_tasks

    env = make_env("miniweb")
    fams = a.families.split(",") if a.families else None
    tasks = task_set(a.split, fams, n_per_family=a.n_per_family, start=a.start)
    st = ExperienceStore(a.out)
    eps = []
    for i, t in enumerate(tasks):
        if a.actor == "teacher":
            actor = TeacherActor(ScriptedTeacher(epsilon=a.epsilon, seed=a.seed * 100003 + i))
            prov = Provenance(source=Source.TEACHER, source_id=ScriptedTeacher.source_id)
        else:
            actor, prov = RandomActor(a.seed * 100003 + i), Provenance(source=Source.AGENT, source_id="random")
        eps += run_tasks(env, [t], actor, provenance=prov)
    st.write_episodes(eps, tag=f"{a.actor}_{a.split}")
    _print({"episodes": len(eps), "success_rate": sum(bool(e.success) for e in eps) / max(1, len(eps)),
            "store": st.summary()})
    return 0


def _trainer_resume(model, trainer, path):
    from agentrs.training.common import load_trainer_state, load_weights

    load_weights(path, model)
    if (Path(path) / "trainer.pt").exists():
        load_trainer_state(path, trainer)
    return trainer.global_step


def cmd_train_bc(a) -> int:
    from agentrs.models.policy import LMPolicy, PolicyConfig
    from agentrs.monitoring.tracker import Tracker
    from agentrs.training.bc import BCConfig, train_bc
    from agentrs.training.common import GradientTrainer, OptimConfig

    _threads(a)
    train = _load_split(a.data, "train")
    _guard_train(train)
    val = _load_split(a.val_data, None) if a.val_data else []
    pol = LMPolicy(PolicyConfig.model_validate(_model_cfg(a.model, "policy")))
    cfg = BCConfig(steps=a.steps, batch_size=a.batch_size, eval_every=a.eval_every,
                   optim=OptimConfig(lr=a.lr, total_steps=a.steps, warmup_steps=min(20, a.steps // 10),
                                     schedule="cosine"))
    trainer = GradientTrainer(pol, cfg.optim)
    start = _trainer_resume(pol, trainer, a.resume) if a.resume else 0
    r = train_bc(pol, train, val, cfg, tracker=Tracker(Path(a.out).parent / (Path(a.out).name + "_log")),
                 out_dir=a.out, trainer=trainer, start_step=start)
    _print({k: r[k] for k in ("n_train_examples", "n_val_examples", "before", "after", "param_change_l2")})
    return 0


def cmd_train_wm(a) -> int:
    from agentrs.models.world_model import WorldModel, WorldModelConfig
    from agentrs.monitoring.tracker import Tracker
    from agentrs.training.common import GradientTrainer, OptimConfig
    from agentrs.training.world_model import WMTrainConfig, train_world_model

    _threads(a)
    train = _load_split(a.data, "train")
    _guard_train(train)
    val = _load_split(a.val_data, None) if a.val_data else []
    wm = WorldModel(WorldModelConfig.model_validate(_model_cfg(a.model, "world_model")))
    cfg = WMTrainConfig(steps=a.steps, batch_size=a.batch_size, eval_every=a.eval_every,
                        optim=OptimConfig(lr=a.lr, total_steps=a.steps, warmup_steps=min(30, a.steps // 10),
                                          schedule="cosine"))
    trainer = GradientTrainer(wm, cfg.optim)
    start = _trainer_resume(wm, trainer, a.resume) if a.resume else 0
    r = train_world_model(wm, train, val, cfg, tracker=Tracker(Path(a.out).parent / (Path(a.out).name + "_log")),
                          out_dir=a.out, trainer=trainer, start_step=start)
    _print({k: r[k] for k in ("n_train", "n_val", "before", "after", "param_change_l2")})
    return 0


def cmd_eval_wm(a) -> int:
    from agentrs.evaluation.wm_eval import evaluate_world_model
    from agentrs.training.common import load_model

    _threads(a)
    wm = load_model(a.world_model)
    res = evaluate_world_model(wm, _load_split(a.train_data, "train"), _load_split(a.val_data, None),
                               n_generate=a.n_generate, drift_k=a.drift_k, n_drift=a.n_drift)
    for k in ("model", "marginal", "action_type"):
        res[k].pop("per_output", None) if not a.full else None
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    _print({k: res[k] for k in ("model", "marginal", "action_type", "generation", "drift",
                                "uncertainty_flags_error_auroc", "diff_nll_per_token")})
    return 0


def cmd_train_critic(a) -> int:
    from agentrs.models.critic import Critic, CriticConfig
    from agentrs.training.common import GradientTrainer, OptimConfig, load_weights
    from agentrs.training.critic import CriticTrainConfig, train_critic

    _threads(a)
    train = _load_split(a.data, "train")
    _guard_train(train)
    val = _load_split(a.val_data, None) if a.val_data else []
    cr = Critic(CriticConfig.model_validate(_model_cfg(a.model, "critic")))
    cfg = CriticTrainConfig(steps=a.steps, optim=OptimConfig(lr=a.lr, total_steps=a.steps, schedule="cosine"))
    trainer = GradientTrainer(cr, cfg.optim)
    if a.resume:
        _trainer_resume(cr, trainer, a.resume)
    r = train_critic(cr, train, val, cfg, out_dir=a.out, trainer=trainer)
    _ = load_weights
    _print({k: r[k] for k in ("n_train", "before", "after", "param_change_l2")})
    return 0


def cmd_train_rl(a) -> int:
    """Standalone real-environment RL loop starting from a policy (and optional critic) checkpoint."""
    import random as _r

    from agentrs.data.storage import ExperienceStore
    from agentrs.environments.miniweb.tasks import task_set
    from agentrs.evaluation.benchmark import evaluate_actor, make_env, run_tasks
    from agentrs.models.critic import Critic, CriticConfig
    from agentrs.models.policy import PolicyActor
    from agentrs.monitoring.tracker import Tracker
    from agentrs.training.common import load_model, load_trainer_state, save_checkpoint
    from agentrs.training.rl import RLConfig, RLTrainer, episode_transitions

    _threads(a)
    pol = load_model(a.policy)
    cr = load_model(a.critic) if a.critic else Critic(CriticConfig.model_validate(_model_cfg(a.model, "critic")))
    ref = load_model(a.policy) if a.kl_coef > 0 else None
    cfg = RLConfig(algo=a.algo, kl_coef=a.kl_coef, epochs=2 if a.algo == "ppo" else 1)
    tr = RLTrainer(pol, cr, cfg, ref_policy=ref, seed=a.seed)
    start = 0
    out = Path(a.out)
    if a.resume:
        load_trainer_state(Path(a.resume) / "policy", tr.pt)
        load_trainer_state(Path(a.resume) / "critic", tr.ct)
        start = json.loads((Path(a.resume) / "rl_state.json").read_text())["iteration"]
    env = make_env("miniweb")
    fams = a.families.split(",") if a.families else None
    tasks = task_set("train", fams, n_per_family=200, start=1000)
    rng = _r.Random(a.seed + start)
    tracker = Tracker(out / "log")
    store = ExperienceStore(out / "experience")
    for it in range(start, a.iterations):
        eps = run_tasks(env, rng.sample(tasks, a.episodes_per_iter), PolicyActor(pol, "sample", seed=a.seed + it))
        store.write_episodes(eps, tag=f"rl{it}")
        upd = tr.update(episode_transitions(eps))
        upd["collect_success"] = sum(bool(e.success) for e in eps) / len(eps)
        tracker.log(upd, it, prefix="rl/")
        print(f"iter {it}: success={upd['collect_success']:.2f} entropy={upd.get('entropy', 0):.3f}")
        save_checkpoint(out / "policy", pol, tr.pt, extra={"iteration": it + 1})
        save_checkpoint(out / "critic", cr, tr.ct, extra={"iteration": it + 1})
        (out / "rl_state.json").write_text(json.dumps({"iteration": it + 1}))
    m, _ = evaluate_actor(env, task_set("val", fams, n_per_family=3), PolicyActor(pol, "greedy"))
    _print({"val_success": m["success_rate"], "by_family": m["by_family"]})
    return 0


def _eval_sets(sets: str, n: int, families):
    from agentrs.environments.miniweb.tasks import task_set

    out = {}
    for s in sets.split(","):
        fams = None if s == "test_family" else families
        out[s] = task_set(s, fams, n_per_family=n)
    return out


def cmd_evaluate(a) -> int:
    from agentrs.evaluation.benchmark import evaluate_actor, make_env

    _threads(a)
    actor, models = _build_actor(a)
    env = make_env("miniweb")
    fams = a.families.split(",") if a.families else None
    res = {}
    for s, tasks in _eval_sets(a.sets, a.n_per_family, fams).items():
        m, _ = evaluate_actor(env, tasks, actor, compute_models=models)
        res[s] = m
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    _print({s: {k: m[k] for k in ("success_rate", "success_ci95", "invalid_action_rate", "real_steps_per_success",
                                  "by_family")} for s, m in res.items()})
    return 0


def cmd_compare(a) -> int:
    import numpy as np

    from agentrs.evaluation.benchmark import evaluate_actor, make_env
    from agentrs.evaluation.metrics import bootstrap_ci
    from agentrs.models.policy import PolicyActor
    from agentrs.training.common import load_model

    _threads(a)
    env = make_env("miniweb")
    fams = a.families.split(",") if a.families else None
    out = {}
    for s, tasks in _eval_sets(a.sets, a.n_per_family, fams).items():
        _, ea = evaluate_actor(env, tasks, PolicyActor(load_model(a.a), "greedy"))
        _, eb = evaluate_actor(env, tasks, PolicyActor(load_model(a.b), "greedy"))
        d = [float(bool(y.success)) - float(bool(x.success)) for x, y in zip(ea, eb)]
        out[s] = {"a_success": float(np.mean([bool(e.success) for e in ea])),
                  "b_success": float(np.mean([bool(e.success) for e in eb])),
                  "paired_diff_b_minus_a": float(np.mean(d)), "diff_ci95": bootstrap_ci(d), "n": len(d)}
    _print(out)
    return 0


def cmd_continual(a) -> int:
    from agentrs.config import load_config
    from agentrs.training.continual import ContinualConfig, run_continual

    _threads(a)
    cfg = load_config(ContinualConfig, a.config, a.set)
    rep = run_continual(cfg)
    _print([{"cycle": c["cycle"], "actor_success": c["actor"]["success"], "promoted": c["evaluation"]["promoted"],
             "reason": c["evaluation"]["reason"], "stage": c["curriculum"]["stage"],
             "mean_forgetting": c["mean_forgetting"]} for c in rep["cycles"]])
    return 0


def cmd_rollback(a) -> int:
    from agentrs.checkpoints.registry import CheckpointRegistry

    reg = CheckpointRegistry(a.registry)
    print(reg.rollback(a.component, a.reason) or "nothing to roll back to")
    return 0


def cmd_experiment(a) -> int:
    from agentrs.config import load_config
    from agentrs.training.pipeline import ExperimentConfig, run_experiment

    _threads(a)
    cfg = load_config(ExperimentConfig, a.config, a.set)
    res = run_experiment(cfg)
    _print({k: v["success_rate"] for k, v in res["final"].items()})
    return 0


def cmd_baselines(a) -> int:
    from agentrs.config import load_config
    from agentrs.training.pipeline import ExperimentConfig, run_experiment

    _threads(a)
    summary = {}
    for seed in [int(s) for s in a.seeds.split(",")]:
        for v in a.variants.split(","):
            cfg = load_config(ExperimentConfig, a.config, list(a.set) + [f"variant={v}", f"seed={seed}",
                                                                          f"name={a.prefix}_{v}_s{seed}"])
            t0 = time.time()
            res = run_experiment(cfg)
            summary[f"{v}_s{seed}"] = {k: m["success_rate"] for k, m in res["final"].items()}
            summary[f"{v}_s{seed}"]["wall_s"] = round(time.time() - t0)
            _print(summary)
    return 0


def cmd_report(a) -> int:
    from agentrs.evaluation.report import build_report

    md = build_report(a.runs, a.out)
    print(md)
    return 0


def cmd_visualize(a) -> int:
    from agentrs.evaluation.visualize import plot_run, render_trajectory

    if a.run:
        for p in plot_run(a.run, a.out or a.run):
            print("wrote", p)
    if a.store:
        print(render_trajectory(a.store, a.episode, a.world_model))
    return 0


def cmd_data_summary(a) -> int:
    from agentrs.data.storage import ExperienceStore

    _print(ExperienceStore(a.store).summary())
    return 0


# ------------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentrs", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(fn=fn)
        sp.add_argument("--threads", type=int, default=0)
        return sp

    s = add("setup-check", cmd_setup_check, "validate the installation and environment")
    s.add_argument("--check-model", default="", help="optionally load this HF model id")

    def actor_args(sp):
        sp.add_argument("--actor", default="teacher", choices=["teacher", "random", "policy", "planner", "human"])
        sp.add_argument("--policy")
        sp.add_argument("--world-model")
        sp.add_argument("--critic")
        sp.add_argument("--greedy", action="store_true")
        sp.add_argument("--epsilon", type=float, default=0.0)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--horizon", type=int, default=1)
        sp.add_argument("--num-candidates", type=int, default=4)
        sp.add_argument("--branches", type=int, default=1)
        sp.add_argument("--uncertainty-penalty", type=float, default=1.0)
        sp.add_argument("--no-planning", action="store_true", help="planner in policy-only ablation mode")

    s = add("run-episode", cmd_run_episode, "run one agent episode (teacher/random/policy/planner/human)")
    s.add_argument("--env", default="miniweb", choices=["miniweb", "localweb"])
    s.add_argument("--family", default="checkout")
    s.add_argument("--split", default="train")
    s.add_argument("--index", type=int, default=0)
    s.add_argument("--out", help="experience store to append the episode to")
    s.add_argument("-v", "--verbose", action="store_true")
    actor_args(s)

    s = add("record-demo", cmd_record_demo, "record a consenting human demonstration")
    s.add_argument("--env", default="miniweb", choices=["miniweb", "localweb"])
    s.add_argument("--family", default="fill_field")
    s.add_argument("--split", default="train")
    s.add_argument("--index", type=int, default=0)
    s.add_argument("--consent", action="store_true")
    s.add_argument("--demonstrator-id", default="")
    s.add_argument("--privacy", default="private", choices=["private", "internal", "shareable"])
    s.add_argument("--headless", action="store_true")
    s.add_argument("--out", default="runs/demos")

    s = add("gen-data", cmd_gen_data, "generate a trajectory dataset (teacher or random actor)")
    s.add_argument("--out", required=True)
    s.add_argument("--split", default="train")
    s.add_argument("--families", default="")
    s.add_argument("--n-per-family", type=int, default=20)
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--actor", default="teacher", choices=["teacher", "random"])
    s.add_argument("--epsilon", type=float, default=0.15)
    s.add_argument("--seed", type=int, default=0)

    for name, fn, h in (("train-bc", cmd_train_bc, "behaviour-clone the policy on demonstrations"),
                        ("train-wm", cmd_train_wm, "train the world model on real transitions"),
                        ("train-critic", cmd_train_critic, "train the critic on real trajectories")):
        s = add(name, fn, h)
        s.add_argument("--data", required=True, help="experience store with training (split=train) data")
        s.add_argument("--val-data", help="experience store with validation data")
        s.add_argument("--model", default="dev", help="model preset (smoke/dev/dev31m/research) or YAML path")
        s.add_argument("--out", required=True, help="checkpoint directory")
        s.add_argument("--steps", type=int, default=300)
        s.add_argument("--batch-size", type=int, default=16)
        s.add_argument("--eval-every", type=int, default=100)
        s.add_argument("--lr", type=float, default=1e-3 if name != "train-critic" else 5e-4)
        s.add_argument("--resume", help="checkpoint dir to resume (weights, optimizer, scheduler, RNG)")

    s = add("eval-wm", cmd_eval_wm, "evaluate world-model predictions vs baselines")
    s.add_argument("--world-model", required=True)
    s.add_argument("--train-data", required=True)
    s.add_argument("--val-data", required=True)
    s.add_argument("--n-generate", type=int, default=60)
    s.add_argument("--drift-k", type=int, default=3)
    s.add_argument("--n-drift", type=int, default=25)
    s.add_argument("--full", action="store_true")
    s.add_argument("--out")

    s = add("train-rl", cmd_train_rl, "real-environment policy RL (reinforce/ppo/grpo)")
    s.add_argument("--policy", required=True)
    s.add_argument("--critic")
    s.add_argument("--model", default="dev")
    s.add_argument("--algo", default="ppo", choices=["reinforce", "ppo", "grpo"])
    s.add_argument("--iterations", type=int, default=10)
    s.add_argument("--episodes-per-iter", type=int, default=32)
    s.add_argument("--kl-coef", type=float, default=0.0)
    s.add_argument("--families", default="")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.add_argument("--resume", help="previous --out directory to resume")

    s = add("evaluate", cmd_evaluate, "held-out evaluation of a policy or planner")
    s.add_argument("--sets", default="val,test,test_family")
    s.add_argument("--n-per-family", type=int, default=3)
    s.add_argument("--families", default="")
    s.add_argument("--out")
    actor_args(s)
    s.set_defaults(actor="policy", greedy=True)

    s = add("compare", cmd_compare, "paired comparison of two policy checkpoints")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--sets", default="val,test,test_family")
    s.add_argument("--n-per-family", type=int, default=3)
    s.add_argument("--families", default="")

    s = add("experiment", cmd_experiment, "run one experiment config (variant A/B/C/D or ablation)")
    s.add_argument("config")
    s.add_argument("--set", nargs="*", default=[])

    s = add("baselines", cmd_baselines, "run variants A,B,C,D over seeds with a shared config")
    s.add_argument("--config", default="configs/experiments/dev_base.yaml")
    s.add_argument("--variants", default="A,B,C,D")
    s.add_argument("--seeds", default="0")
    s.add_argument("--prefix", default="baseline")
    s.add_argument("--set", nargs="*", default=[])

    s = add("continual", cmd_continual, "run collect/train/evaluate/promote cycles")
    s.add_argument("--config", required=False)
    s.add_argument("--set", nargs="*", default=[])

    s = add("rollback", cmd_rollback, "restore the previously promoted checkpoint")
    s.add_argument("--registry", required=True)
    s.add_argument("--component", default="policy")
    s.add_argument("--reason", default="manual rollback")

    s = add("report", cmd_report, "aggregate experiment results into markdown tables")
    s.add_argument("--runs", nargs="+", required=True, help="experiment run directories")
    s.add_argument("--out")

    s = add("visualize", cmd_visualize, "plot learning curves / render a trajectory with predictions")
    s.add_argument("--run", help="experiment run directory (plots learning curves and WM losses)")
    s.add_argument("--store", help="experience store containing the episode")
    s.add_argument("--episode", help="episode id (default: first)")
    s.add_argument("--world-model", help="checkpoint to show predicted vs actual diffs")
    s.add_argument("--out")

    s = add("data-summary", cmd_data_summary, "summarise an experience store")
    s.add_argument("--store", required=True)
    return p


def main(argv=None) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    a = build_parser().parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
