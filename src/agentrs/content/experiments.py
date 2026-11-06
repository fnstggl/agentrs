"""Content-learning experiments: configurations A-E on the same data, interactions and evaluation.

A  pretrained language policy without content-specific learning (the shared high-level policy)
B  A + imitation of demonstrated content (successful demos weighted; high-level rehearsal)
C  B + verified content-level RL (group advantages, token-level clipped surrogate, KL to B)
D  C + content-conditioned world model (trained on demos + all real content transitions) and
   consequence-based planning over content drafts at decision time (policy weights = C's)
E  D + controlled imagined content training (world-model-scored on-policy drafts,
   uncertainty-filtered, down-weighted)

C, D and E use the same number of real environment episodes. Engagement is reported, never
optimised. The shared network's high-level MiniWeb skill and general-text perplexity are
measured before and after content learning.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from agentrs.agent.runtime import TeacherActor, run_episode
from agentrs.content.evaluation import (
    ContentEvaluationSuite,
    RewardHackingMonitor,
    action_regression,
    evaluate_content,
    lm_perplexity,
    regression_check,
)
from agentrs.content.planner import ContentOutcomePlanner, ContentPlannerConfig
from agentrs.content.policy import ContentGenConfig, ContentGeneratingPolicy, ContentPolicyActor
from agentrs.content.training import (
    ContentBCConfig,
    ContentImaginationConfig,
    ContentPolicyRLTrainer,
    ContentPredictionTrainer,
    ContentRLConfig,
    ContentWMTrainConfig,
    imagined_content_steps,
    train_content_bc,
)
from agentrs.content.world_model import ContentConditionedWorldModel, ContentWorldModelConfig
from agentrs.data.schemas import Provenance, Source
from agentrs.data.storage import ExperienceStore
from agentrs.demonstrations.content_teacher import ContentTeacher
from agentrs.environments.content.env import content_task_set
from agentrs.evaluation.benchmark import make_env
from agentrs.models.backbone import BackboneConfig
from agentrs.models.policy import LMPolicy, PolicyConfig
from agentrs.monitoring.tracker import Tracker
from agentrs.training.common import OptimConfig, load_model, save_checkpoint, seed_everything


class ContentEvalSets(BaseModel):
    model_config = ConfigDict(extra="forbid")
    val: int = 6
    test: int = 6
    test_family: int = 16


class ContentExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "content"
    out_dir: str = "runs/content"
    seed: int = 0
    variant: str = "E"  # A | B | C | D | E
    init_policy: Optional[str] = "runs/cache/bc_1318e51fdd21"  # shared high-level policy; None = pretrained
    init_world_model: Optional[str] = "runs/cache/wm_c962f7759108"  # MiniWeb world model to extend; None = fresh
    policy: PolicyConfig = Field(default_factory=lambda: PolicyConfig(backbone=BackboneConfig(
        name="EleutherAI/pythia-14m", max_prompt_tokens=512)))
    world_model: ContentWorldModelConfig = Field(default_factory=lambda: ContentWorldModelConfig(
        backbone=BackboneConfig(name="EleutherAI/pythia-14m", max_prompt_tokens=512)))
    gen: ContentGenConfig = Field(default_factory=lambda: ContentGenConfig(max_content_tokens=40))
    n_demo_per_family: int = 40
    bc: ContentBCConfig = Field(default_factory=lambda: ContentBCConfig(
        steps=400, batch_size=16, rehearsal_weight=0.5,
        optim=OptimConfig(lr=5e-4, total_steps=400, warmup_steps=20, schedule="cosine")))
    rl: ContentRLConfig = Field(default_factory=ContentRLConfig)
    rl_iterations: int = 30
    tasks_per_iter: int = 8
    group_size: int = 4
    wm_initial_steps: int = 400
    wm_steps_per_iter: int = 15
    wm_train: ContentWMTrainConfig = Field(default_factory=ContentWMTrainConfig)
    planner: ContentPlannerConfig = Field(default_factory=ContentPlannerConfig)
    imagination: ContentImaginationConfig = Field(default_factory=ContentImaginationConfig)
    eval_sets: ContentEvalSets = Field(default_factory=ContentEvalSets)
    eval_every: int = 10
    miniweb_eval_tasks_per_family: int = 2
    cache_dir: Optional[str] = "runs/cache"
    families: Optional[list[str]] = None  # restrict train/val/test families (None = all train families)


def _load_policy(cfg: ContentExperimentConfig) -> LMPolicy:
    if cfg.init_policy:
        return load_model(cfg.init_policy)
    return LMPolicy(cfg.policy)


def _load_wm(cfg: ContentExperimentConfig) -> ContentConditionedWorldModel:
    wm = ContentConditionedWorldModel(cfg.world_model)
    if cfg.init_world_model:
        from safetensors.torch import load_file

        sd = load_file(str(Path(cfg.init_world_model) / "weights.safetensors"))
        missing, unexpected = wm.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected[:3]
        assert all(k.startswith("content_heads") for k in missing), missing[:3]
    return wm


class ContentExperiment:
    def __init__(self, cfg: ContentExperimentConfig):
        self.cfg = cfg
        seed_everything(cfg.seed)
        self.dir = Path(cfg.out_dir) / cfg.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tracker = Tracker(self.dir, config=cfg.model_dump())
        self.store = ExperienceStore(self.dir / "experience")
        self.env = make_env("content")
        self.results: dict = {"config": cfg.model_dump(), "evals": [], "rl": [], "hacking": []}
        self.real_episodes = 0
        self.flops = {"policy_train": 0.0, "wm_train": 0.0, "inference": 0.0}
        self.rng = random.Random(cfg.seed + 7)
        # high-level (MiniWeb) regression data for the shared network
        from agentrs.training.bc import BCConfig, make_examples
        from agentrs.training.pipeline import ExperimentConfig, generate_demonstrations, generate_val_demos

        mw_cfg = ExperimentConfig(seed=cfg.seed, n_demo_tasks_per_family=40)
        self.mw_val = make_examples([t for e in generate_val_demos(mw_cfg) for t in e.transitions if e.success], BCConfig())
        self.mw_rehearsal = make_examples([t for e in generate_demonstrations(mw_cfg) for t in e.transitions], BCConfig())
        from agentrs.environments.miniweb.tasks import task_set

        self.mw_tasks = task_set("val", n_per_family=cfg.miniweb_eval_tasks_per_family)
        self.mw_env = make_env("miniweb")
        s = cfg.eval_sets
        self.suite = ContentEvaluationSuite(self.env, {
            "val": content_task_set("val", cfg.families, n_per_family=s.val),
            "test": content_task_set("test", cfg.families, n_per_family=s.test),
            "test_family": content_task_set("test_family", n_per_family=s.test_family)},
            self.mw_val, self.mw_env, self.mw_tasks)

    # ------------------------------------------------------------------ helpers
    def _dump(self):
        r = dict(self.results)
        r.update(real_episodes=self.real_episodes, flops=self.flops)
        (self.dir / "results.json").write_text(json.dumps(r, indent=1, default=str))

    def _eval(self, tag: str, cp: ContentGeneratingPolicy, planner=None, sets=("val",), with_regression=False) -> dict:
        out = {"tag": tag, "real_episodes": self.real_episodes}
        f0 = cp.bb.flops_counter
        for name in sets:
            out[f"policy/{name}"], _ = evaluate_content(self.env, self.suite.sets[name], ContentPolicyActor(cp, "greedy"))
            if planner is not None:
                out[f"planner/{name}"], _ = evaluate_content(self.env, self.suite.sets[name], planner)
        self.flops["inference"] += cp.bb.flops_counter - f0
        if with_regression:
            out["lm_perplexity"] = lm_perplexity(cp.bb)
            out["action_regression"] = action_regression(cp.policy, self.mw_val, self.mw_env, self.mw_tasks)
        for k, v in out.items():
            if isinstance(v, dict) and "success_rate" in v:
                self.tracker.log({"success": v["success_rate"], "first_attempt": v["first_attempt_success"],
                                  "quality": v["quality_partial_mean"]}, self.real_episodes, prefix=f"eval/{tag}/{k}/")
        self.results["evals"].append(out)
        self._dump()
        return out

    def _cache(self, key: str) -> Optional[Path]:
        return Path(self.cfg.cache_dir) / key if self.cfg.cache_dir else None

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        c = self.cfg
        t_start = time.time()
        cp = ContentGeneratingPolicy(_load_policy(c), c.gen, checkpoint_id="init")
        start = self._eval("start", cp, sets=("val", "test", "test_family"), with_regression=True)
        self.results["start"] = start
        if c.variant == "A":
            self.results["final"] = start
            return self._finish(t_start)
        # demonstrations by another agent (content + outcomes), shared by B-E
        prov = Provenance(source=Source.TEACHER, source_id=ContentTeacher.source_id)
        demo_tasks = content_task_set("train", c.families, n_per_family=c.n_demo_per_family)
        demos = [run_episode(self.env, t, TeacherActor(ContentTeacher(c.seed * 7919 + i)), prov)
                 for i, t in enumerate(demo_tasks)]
        self.store.write_episodes(demos, tag="content_demos")
        demo_ts = [t for e in demos for t in e.transitions]
        self.results["demos"] = {"episodes": len(demos), "success_rate": sum(e.success for e in demos) / len(demos)}
        # B: imitation (cached per seed)
        key = "content_bc_" + hashlib.sha1(json.dumps([c.seed, c.init_policy, c.n_demo_per_family, c.bc.model_dump(),
                                                       c.gen.model_dump()] + ([c.families] if c.families else []),
                                                      default=str).encode()).hexdigest()[:12]
        cache = self._cache(key)
        f0 = cp.bb.flops_counter
        if cache is not None and (cache / "weights.safetensors").exists():
            from agentrs.training.common import load_weights

            load_weights(cache, cp.policy)
            self.results["bc"] = json.loads((cache / "summary.json").read_text())
        else:
            r = train_content_bc(cp, demo_ts, c.bc, rehearsal=self.mw_rehearsal, tracker=self.tracker)
            self.results["bc"] = {k: r[k] for k in ("n_demos", "first10", "last10")}
            self.results["bc"]["train_flops"] = 3 * (cp.bb.flops_counter - f0)
            if cache is not None:
                save_checkpoint(cache, cp.policy, keep_optimizer=False)
                (cache / "summary.json").write_text(json.dumps(self.results["bc"]))
        self.flops["policy_train"] += self.results["bc"].get("train_flops", 3 * (cp.bb.flops_counter - f0))
        cp.checkpoint_id = "content-bc"
        if c.variant == "B":
            self.results["final"] = self._eval("final", cp, sets=("val", "test", "test_family"), with_regression=True)
            save_checkpoint(self.dir / "checkpoints" / "content_policy_final", cp.policy, keep_optimizer=False)
            return self._finish(t_start)
        self._eval("after_bc", cp, sets=("val",))
        # world model (D, E)
        wm = wmt = None
        if c.variant in ("D", "E"):
            wm = _load_wm(c)
            wmt = ContentPredictionTrainer(wm, c.wm_train)
            f1 = wm.backbone.flops_counter
            r = wmt.train(demo_ts, steps=c.wm_initial_steps)
            self.flops["wm_train"] += 3 * (wm.backbone.flops_counter - f1)
            self.results["wm_initial"] = {k: r[k] for k in ("n", "first10", "last10")}
        # C/D/E: content RL from verified outcomes
        ref = copy.deepcopy(cp)
        ref.policy.eval()
        trainer = ContentPolicyRLTrainer(cp, c.rl, ref=ref, seed=c.seed)
        pool = content_task_set("train", c.families, n_per_family=400, start=c.n_demo_per_family)
        monitor = RewardHackingMonitor()
        all_real = list(demo_ts)
        for it in range(c.rl_iterations):
            base = self.rng.sample(pool, c.tasks_per_iter)
            f0 = cp.bb.flops_counter
            eps = [run_episode(self.env, t, ContentPolicyActor(cp, "sample", seed=c.seed * 10**6 + it * 1000 + k * 10 + g),
                               checkpoints={"policy": cp.checkpoint_id})
                   for k, t in enumerate(base) for g in range(c.group_size)]
            self.flops["inference"] += cp.bb.flops_counter - f0
            self.real_episodes += len(eps)
            self.store.write_episodes(eps, tag=f"content_rl{it}")
            new_ts = [t for e in eps for t in e.transitions]
            all_real += new_ts
            stats = {"collect_success": sum(e.success for e in eps) / len(eps)}
            stats.update({k: v for k, v in monitor.record(it, [e for e in eps if e.family == "content_forum"]).items()
                          if v is not None})
            imagined = []
            if wm is not None:
                f1 = wm.backbone.flops_counter
                hist = self.rng.sample(all_real, min(len(all_real), 48))
                wr = wmt.train(new_ts + hist, steps=c.wm_steps_per_iter)
                stats["wm_loss"] = wr["last10"]
                if c.variant == "E":
                    starts = [t for t in self.rng.sample(new_ts, min(len(new_ts), c.imagination.states_per_update))
                              if t.semantic_action is not None]
                    imagined, ist = imagined_content_steps(cp, wm, starts, c.imagination, seed=c.seed * 977 + it)
                    stats.update(ist)
                self.flops["wm_train"] += 3 * (wm.backbone.flops_counter - f1)
            f0 = cp.bb.flops_counter
            upd = trainer.update([e.transitions for e in eps], imagined)
            self.flops["policy_train"] += 3 * (cp.bb.flops_counter - f0)
            stats.update({f"rl_{k}": v for k, v in upd.items() if isinstance(v, (int, float))})
            self.tracker.log(stats, self.real_episodes, prefix="content_rl/")
            self.results["rl"].append(stats)
            if (it + 1) % c.eval_every == 0 and it + 1 < c.rl_iterations:
                self._eval(f"iter{it + 1}", cp, sets=("val",))
        self.results["hacking"] = monitor.summary()
        planner = None
        if wm is not None:
            planner = ContentOutcomePlanner(cp, wm, c.planner, seed=c.seed)
            held = []
            for t in self.suite.sets["val"] + self.suite.sets["test"]:
                held += run_episode(self.env, t, TeacherActor(ContentTeacher(t.seed)),
                                    Provenance(source=Source.TEACHER)).transitions
            self.results["wm_heldout"] = wmt.evaluate(held)
            fam = []
            for t in self.suite.sets["test_family"]:
                fam += run_episode(self.env, t, ContentPolicyActor(cp, "sample", seed=t.seed)).transitions
            self.results["wm_heldout_family"] = wmt.evaluate(fam)
            save_checkpoint(self.dir / "checkpoints" / "content_wm_final", wm, keep_optimizer=False)
        self.results["final"] = self._eval("final", cp, planner, sets=("val", "test", "test_family"),
                                           with_regression=True)
        self.results["regressions"] = regression_check(start, self.results["final"])
        save_checkpoint(self.dir / "checkpoints" / "content_policy_final", cp.policy, keep_optimizer=False)
        return self._finish(t_start)

    def _finish(self, t_start: float) -> dict:
        self.results["total_wall_s"] = time.time() - t_start
        self._dump()
        self.tracker.close()
        return self.results


def run_content_experiment(cfg: ContentExperimentConfig) -> dict:
    return ContentExperiment(cfg).run()



