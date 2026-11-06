"""End-to-end experiment pipeline for the four required configurations.

A  policy_only : pretrained starting policy, same tools, no task-specific updates
B  imitation   : A + behaviour cloning on demonstrations
C  model_free  : B (if demos allowed) + policy-gradient RL from verified rewards
D  model_based : C + world-model learning (demos + real experience), world-model
                 planning (collection and/or evaluation), optional imagined
                 training, search distillation

All variants share the starting checkpoint, task distribution, demonstrations,
reward definition and evaluation sets. C and D use the same number of real
environment episodes. Real interactions, training FLOPs and inference FLOPs are
reported separately.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Optional

import torch
from pydantic import ConfigDict, BaseModel, Field

from agentrs.agent.runtime import TeacherActor
from agentrs.data.replay import ReplayBuffer
from agentrs.data.schemas import EpisodeRecord, Provenance, Source, Transition
from agentrs.data.storage import ExperienceStore
from agentrs.demonstrations.teacher import ScriptedTeacher
from agentrs.environments.miniweb.families import HELDOUT_FAMILIES, TRAIN_FAMILIES
from agentrs.environments.miniweb.tasks import task_set
from agentrs.evaluation.benchmark import evaluate_actor, make_env, run_tasks
from agentrs.models.critic import Critic, CriticConfig
from agentrs.models.policy import LMPolicy, PolicyActor, PolicyConfig
from agentrs.models.world_model import WorldModel, WorldModelConfig
from agentrs.monitoring.tracker import Tracker
from agentrs.planning.mpc import MPCPlanner, PlannerConfig
from agentrs.training.bc import BCConfig, train_bc
from agentrs.training.common import save_checkpoint, seed_everything
from agentrs.training.critic import CriticTrainConfig, train_critic
from agentrs.training.imagination import ExploitationMonitor, ImaginationConfig, imagine_rollouts
from agentrs.training.rl import RLConfig, RLTrainer, episode_transitions
from agentrs.training.world_model import WMTrainConfig, train_world_model


class EvalSets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    val: int = 3  # tasks per family
    test: int = 3
    test_family: int = 5


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "exp"
    seed: int = 0
    out_dir: str = "runs"
    variant: str = "D"  # A | B | C | D
    use_demos: bool = True
    use_world_model: bool = True
    use_planning_collect: bool = False
    use_planning_eval: bool = True
    use_imagination: bool = False
    historical_replay: bool = True
    train_families: list[str] = Field(default_factory=lambda: list(TRAIN_FAMILIES))
    heldout_families: list[str] = Field(default_factory=lambda: list(HELDOUT_FAMILIES))
    n_demo_tasks_per_family: int = 20
    teacher_epsilon: float = 0.15
    n_rl_tasks_per_family: int = 200
    eval_sets: EvalSets = Field(default_factory=EvalSets)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    world_model: WorldModelConfig = Field(default_factory=WorldModelConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    bc: BCConfig = Field(default_factory=BCConfig)
    wm_train: WMTrainConfig = Field(default_factory=WMTrainConfig)
    critic_train: CriticTrainConfig = Field(default_factory=CriticTrainConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    rl_iterations: int = 10
    episodes_per_iter: int = 20
    wm_steps_per_iter: int = 20
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    imagination: ImaginationConfig = Field(default_factory=ImaginationConfig)
    eval_every_iters: int = 5
    exploration_epsilon: float = 0.0
    save_checkpoints: bool = True
    # Deterministic stages shared by variants with the same seed (BC policy, initial world model,
    # initial critic) are computed once and reused; set to null to disable.
    cache_dir: Optional[str] = "runs/cache"

    def resolved(self) -> "ExperimentConfig":
        c = self.model_copy(deep=True)
        if c.variant == "A":
            c.use_demos = False
            c.use_world_model = False
            c.rl_iterations = 0
        elif c.variant == "B":
            c.use_world_model = False
            c.rl_iterations = 0
        elif c.variant == "C":
            c.use_world_model = False
            c.use_imagination = False
            c.use_planning_collect = False
            c.use_planning_eval = False
        if not c.use_world_model:
            c.use_planning_collect = c.use_planning_eval = c.use_imagination = False
        return c


def generate_demonstrations(cfg: ExperimentConfig, store: Optional[ExperienceStore] = None) -> list[EpisodeRecord]:
    env = make_env("miniweb")
    tasks = task_set("train", cfg.train_families, n_per_family=cfg.n_demo_tasks_per_family, start=0)
    prov = Provenance(source=Source.TEACHER, source_id=ScriptedTeacher.source_id, license="internal")
    eps = []
    for i, t in enumerate(tasks):
        from agentrs.agent.runtime import run_episode

        eps.append(run_episode(env, t, TeacherActor(ScriptedTeacher(cfg.teacher_epsilon, seed=cfg.seed * 7919 + i)),
                               provenance=prov))
    if store is not None:
        store.write_episodes(eps, tag="demos")
    return eps


def eval_tasks(cfg: ExperimentConfig) -> dict[str, list]:
    return {
        "val": task_set("val", cfg.train_families, n_per_family=cfg.eval_sets.val),
        "test": task_set("test", cfg.train_families, n_per_family=cfg.eval_sets.test),
        "test_family": task_set("test_family", cfg.heldout_families, n_per_family=cfg.eval_sets.test_family),
    }


class Experiment:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg.resolved()
        c = self.cfg
        seed_everything(c.seed)
        self.dir = Path(c.out_dir) / c.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tracker = Tracker(self.dir, config=c.model_dump())
        self.store = ExperienceStore(self.dir / "experience")
        self.env = make_env("miniweb")
        self.replay = ReplayBuffer(seed=c.seed)
        self.results: dict = {"config": c.model_dump(), "evals": [], "stages": {}}
        self.real_env_steps = 0
        self.real_episodes = 0
        self.flops = {"policy_train": 0.0, "wm_train": 0.0, "critic_train": 0.0, "inference": 0.0}
        self.policy = LMPolicy(c.policy)
        self.wm: Optional[WorldModel] = WorldModel(c.world_model) if c.use_world_model else None
        self.critic: Optional[Critic] = Critic(c.critic) if c.variant in ("C", "D") else None
        self.rl_tasks = task_set("train", c.train_families, n_per_family=c.n_rl_tasks_per_family,
                                 start=c.n_demo_tasks_per_family)
        self._rl_rng = random.Random(c.seed + 1)
        self.evsets = eval_tasks(c)

    # ------------------------------------------------------------------ helpers
    def _models(self):
        return [m.backbone for m in (self.policy, self.wm, self.critic) if m is not None]

    def _flops_now(self) -> float:
        return float(sum(b.flops_counter for b in self._models()))

    def _train_stage(self, key: str, fn):
        f0 = self._flops_now()
        t0 = time.time()
        out = fn()
        # counters record forward FLOPs (2N/token); training adds ~2x for backward
        self.flops[key] += 3 * (self._flops_now() - f0)
        self.results["stages"].setdefault(key, []).append({"wall_s": time.time() - t0})
        return out

    def actor(self, mode: str):
        c = self.cfg
        if mode == "planner" and self.wm is not None and self.critic is not None:
            pc = c.planner.model_copy()
            return MPCPlanner(self.policy, self.wm, self.critic, pc, seed=c.seed)
        return PolicyActor(self.policy, "greedy", seed=c.seed)

    def evaluate(self, tag: str, sets=("val",), modes=("policy",)) -> dict:
        out = {}
        for mode in modes:
            for s in sets:
                f0 = self._flops_now()
                m, eps = evaluate_actor(self.env, self.evsets[s], self.actor(mode), compute_models=self._models())
                self.flops["inference"] += self._flops_now() - f0
                out[f"{mode}/{s}"] = m
                self.tracker.log({"success_rate": m["success_rate"], "invalid_rate": m["invalid_action_rate"],
                                  "steps_per_success": m["real_steps_per_success"]},
                                 step=self.real_env_steps, prefix=f"eval/{tag}/{mode}/{s}/")
        rec = {"tag": tag, "real_env_steps": self.real_env_steps, "real_episodes": self.real_episodes,
               "flops": dict(self.flops), "results": out}
        self.results["evals"].append(rec)
        self._dump()
        return out

    def _dump(self):
        r = dict(self.results)
        r["real_env_steps"] = self.real_env_steps
        r["real_episodes"] = self.real_episodes
        r["flops"] = self.flops
        (self.dir / "results.json").write_text(json.dumps(r, indent=1, default=str))

    def _cache_key(self, stage: str, *parts) -> str:
        c = self.cfg
        blob = json.dumps({"stage": stage, "seed": c.seed, "fams": c.train_families, "demos": c.n_demo_tasks_per_family,
                           "eps": c.teacher_epsilon, "parts": [p.model_dump() if hasattr(p, "model_dump") else p
                                                               for p in parts], "v": 2}, sort_keys=True, default=str)
        return f"{stage}_{hashlib.sha1(blob.encode()).hexdigest()[:12]}"

    @contextlib.contextmanager
    def _cache_lock(self, key: str):
        d = Path(self.cfg.cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{key}.lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # other variants with the same seed wait here
            try:
                yield d / key
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _cached_stage(self, key_stage: str, model, parts: tuple, train_fn, flops_key: str):
        """Run ``train_fn`` (returns a summary dict) or load model weights + summary from the cache."""
        from agentrs.training.common import load_weights

        if not self.cfg.cache_dir:
            return self._train_stage(flops_key, train_fn), False
        key = self._cache_key(key_stage, *parts)
        with self._cache_lock(key) as path:
            if (path / "summary.json").exists():
                load_weights(path, model)
                summ = json.loads((path / "summary.json").read_text())
                self.flops[flops_key] += summ.get("_train_flops", 0.0)  # account the compute it cost
                return summ, True
            f0 = self.flops[flops_key]
            r = self._train_stage(flops_key, train_fn)
            save_checkpoint(path, model, r.get("trainer"), keep_optimizer=True)
            summ = {k: v for k, v in r.items() if k not in ("trainer", "history")}
            summ["_train_flops"] = self.flops[flops_key] - f0
            (path / "summary.json").write_text(json.dumps(summ, default=str))
            return r, False

    def _ckpt(self, name: str, model, trainer=None):
        if self.cfg.save_checkpoints and model is not None:
            save_checkpoint(self.dir / "checkpoints" / name, model, trainer, keep_optimizer=False)

    # ------------------------------------------------------------------ stages
    def run(self) -> dict:
        c = self.cfg
        t_start = time.time()
        demos: list[EpisodeRecord] = []
        if c.use_demos:  # "no observational learning" ablation: no demos for ANY component
            demos = generate_demonstrations(c, self.store)
            for ep in demos:
                self.replay.add_episode(ep)
        demo_ts = [t for ep in demos for t in ep.transitions]
        val_demo = [t for ep in generate_val_demos(c) for t in ep.transitions]
        self.results["n_demo_episodes"] = len(demos)
        self.results["demo_success_rate"] = (sum(bool(e.success) for e in demos) / len(demos)) if demos else None
        self.evaluate("start", sets=("val",))
        if c.use_demos:
            r, hit = self._cached_stage("bc", self.policy, (c.policy, c.bc), lambda: train_bc(
                self.policy, demo_ts, val_demo, c.bc, tracker=self.tracker), "policy_train")
            self.results["bc"] = {k: r[k] for k in ("n_train_examples", "before", "after", "param_change_l2")}
            self.results["bc"]["from_cache"] = hit
            self._ckpt("policy_bc", self.policy)
        if self.wm is not None and not demo_ts:  # world model learns only from own experience
            from agentrs.training.common import GradientTrainer

            self.wm_trainer = GradientTrainer(self.wm, c.wm_train.optim)
        if self.wm is not None and demo_ts:
            from agentrs.training.common import GradientTrainer, load_trainer_state

            r, hit = self._cached_stage("wm", self.wm, (c.world_model, c.wm_train), lambda: train_world_model(
                self.wm, demo_ts, val_demo, c.wm_train, tracker=self.tracker), "wm_train")
            self.results["wm_initial"] = {k: r[k] for k in ("n_train", "before", "after", "param_change_l2")}
            self.results["wm_initial"]["from_cache"] = hit
            if hit:
                self.wm_trainer = GradientTrainer(self.wm, c.wm_train.optim)
                load_trainer_state(Path(c.cache_dir) / self._cache_key("wm", c.world_model, c.wm_train),
                                   self.wm_trainer, restore_rng=False)
            else:
                self.wm_trainer = r["trainer"]
            self._ckpt("world_model_init", self.wm)
        if self.critic is not None and demo_ts:
            r, hit = self._cached_stage("critic", self.critic, (c.critic, c.critic_train), lambda: train_critic(
                self.critic, demo_ts, val_demo, c.critic_train, tracker=self.tracker), "critic_train")
            self.results["critic_initial"] = {k: r[k] for k in ("n_train", "before", "after")}
        if c.rl_iterations > 0:
            self.run_rl()
        modes = ("policy", "planner") if c.use_planning_eval else ("policy",)
        final = self.evaluate("final", sets=("val", "test", "test_family"), modes=modes)
        self.results["final"] = final
        self.results["total_wall_s"] = time.time() - t_start
        self.results["real_env_steps"] = self.real_env_steps
        self.results["real_episodes"] = self.real_episodes
        self.results["flops"] = self.flops
        self._ckpt("policy_final", self.policy)
        self._ckpt("world_model_final", self.wm)
        self._ckpt("critic_final", self.critic)
        self._dump()
        self.tracker.close()
        return self.results

    def run_rl(self) -> None:
        c = self.cfg
        trainer = RLTrainer(self.policy, self.critic, c.rl, seed=c.seed)
        monitor = ExploitationMonitor(c.imagination.exploitation_gap)
        imagination_on = c.use_imagination
        for it in range(c.rl_iterations):
            # ---- collect with a fixed snapshot of the current weights
            if c.rl.algo == "grpo":
                base = self._rl_rng.sample(self.rl_tasks, max(1, c.episodes_per_iter // c.rl.group_size))
                tasks = [t for t in base for _ in range(c.rl.group_size)]
            else:
                tasks = self._rl_rng.sample(self.rl_tasks, c.episodes_per_iter)
            if c.use_planning_collect and self.wm is not None:
                # planner-collected data is off-policy: use reinforce with truncated IS and/or distillation
                pc = c.planner.model_copy(update={"selection": "softmax", "epsilon": max(c.exploration_epsilon, 0.05)})
                actor = MPCPlanner(self.policy, self.wm, self.critic, pc, seed=c.seed * 1000 + it)
            else:
                mode = "epsilon" if c.exploration_epsilon > 0 else "sample"
                actor = PolicyActor(self.policy, mode, epsilon=c.exploration_epsilon, seed=c.seed * 1000 + it)
            f0 = self._flops_now()
            eps = run_tasks(self.env, tasks, actor, checkpoints={"policy": f"iter{it}"})
            self.flops["inference"] += self._flops_now() - f0
            self.store.write_episodes(eps, tag=f"rl{it}")
            for ep in eps:
                self.replay.add_episode(ep)
                self.real_env_steps += ep.num_steps
                self.real_episodes += 1
            succ = sum(bool(e.success) for e in eps) / len(eps)
            stats = {"collect_success": succ, "collect_invalid": sum(e.num_invalid for e in eps) /
                     max(1, sum(e.num_steps for e in eps))}
            # ---- world model on real experience (new + historical replay)
            if self.wm is not None and c.wm_steps_per_iter > 0:
                new_ts = [t for e in eps for t in e.transitions]
                if c.historical_replay:
                    hist_ts = self.replay.sample(len(new_ts), "historical")
                    wm_data = new_ts + hist_ts
                else:
                    wm_data = new_ts
                wcfg = c.wm_train.model_copy(update={"steps": c.wm_steps_per_iter, "eval_every": 10**9,
                                                     "seed": c.seed + it})
                r = self._train_stage("wm_train", lambda: train_world_model(
                    self.wm, wm_data, [], wcfg, trainer=self.wm_trainer, log_prefix="wm_online/"))
                stats["wm_loss"] = r["history"][-1]["loss"] if r["history"] else float("nan")
            # ---- imagination
            imagined = []
            if imagination_on and self.wm is not None:
                starts = self.replay.sample(64, "recent", recent_fraction=0.5,
                                            predicate=lambda t: not t.terminated and t.belief_prompt is not None)
                imagined, ist = imagine_rollouts(self.policy, self.wm, self.critic, starts, self.replay.real,
                                                 c.imagination.model_copy(update={"seed": c.seed * 31 + it}))
                stats.update(ist)
                if not monitor.update(ist.get("imagined_success_rate", 0.0), succ):
                    imagination_on = False
                    stats["imagination_disabled_exploitation"] = 1.0
                for traj in imagined:
                    for t in traj:
                        self.replay.add(t)
            # ---- policy (and critic) update
            # policy gradient uses the fresh batch only (historical replay feeds the world model / critic)
            batch_eps = episode_transitions(eps)
            upd = self._train_stage("policy_train", lambda: trainer.update(batch_eps, imagined))
            stats.update({f"rl_{k}": v for k, v in upd.items()})
            self.tracker.log(stats, step=self.real_env_steps, prefix="rl/")
            if (it + 1) % c.eval_every_iters == 0 and it + 1 < c.rl_iterations:
                self.evaluate(f"iter{it + 1}", sets=("val",))
        self.results["rl"] = {"iterations": c.rl_iterations, "imagination_final_enabled": imagination_on}


def generate_val_demos(cfg: ExperimentConfig) -> list[EpisodeRecord]:
    """Teacher episodes on validation tasks: used ONLY for validation losses, never training."""
    from agentrs.agent.runtime import run_episode

    env = make_env("miniweb")
    tasks = task_set("val", cfg.train_families, n_per_family=2, start=100)
    return [run_episode(env, t, TeacherActor(ScriptedTeacher(cfg.teacher_epsilon, seed=i)),
                        provenance=Provenance(source=Source.TEACHER, source_id="teacher:scripted-v1"))
            for i, t in enumerate(tasks)]


def run_experiment(cfg: ExperimentConfig) -> dict:
    return Experiment(cfg).run()


_ = (torch, Transition)
