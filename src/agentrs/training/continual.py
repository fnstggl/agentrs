"""Continual learning with separated actor / learner / evaluator processes.

* **Actor** (separate process): loads the *promoted* policy checkpoint (a fixed,
  versioned snapshot), collects real episodes on curriculum tasks and writes
  Parquet shards to ``<root>/inbox``. Weights never change mid-episode.
* **Learner** (this process): ingests unseen shards into the replay buffer
  (historical replay), updates the world model, critic and policy. Episodes were
  collected by the production checkpoint, so PPO importance ratios are computed
  against the recorded behaviour log-probs (valid off-policy correction).
  Produces candidate checkpoints registered in the ``CheckpointRegistry``.
* **Evaluator** (separate process): evaluates candidate vs production on held-out
  validation tasks plus a regression suite of already-mastered families and
  promotes only if the ``PromotionGate`` passes. Rollback restores the previous
  production checkpoint.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import Optional

from pydantic import ConfigDict, BaseModel, Field

from agentrs.checkpoints.registry import CheckpointRegistry, PromotionGate
from agentrs.data.replay import ReplayBuffer
from agentrs.data.storage import ExperienceStore
from agentrs.models.critic import CriticConfig
from agentrs.models.policy import PolicyConfig
from agentrs.models.world_model import WorldModelConfig
from agentrs.monitoring.tracker import Tracker
from agentrs.training.curriculum import Curriculum, CurriculumConfig, update_from_episodes
from agentrs.training.rl import RLConfig


class ContinualConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: str = "runs/continual"
    seed: int = 0
    cycles: int = 3
    episodes_per_cycle: int = 40
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    world_model: WorldModelConfig = Field(default_factory=WorldModelConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    use_world_model: bool = True
    rl: RLConfig = Field(default_factory=lambda: RLConfig(algo="ppo", epochs=2))
    wm_steps_per_cycle: int = 30
    critic_steps_per_cycle: int = 20
    eval_tasks_per_family: int = 3
    gate: PromotionGate = Field(default_factory=PromotionGate)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    initial_policy: Optional[str] = None  # checkpoint dir (e.g. a BC policy); None = pretrained backbone
    initial_world_model: Optional[str] = None
    initial_critic: Optional[str] = None
    separate_processes: bool = True
    reset_on_reject: bool = True
    threads_per_process: int = 1


# ------------------------------------------------------------------------------ actor
def actor_main(registry_path: str, inbox: str, families: list[str], seed: int, start_index: int,
               threads: int = 1) -> dict:
    import torch

    from agentrs.environments.miniweb.tasks import make_task
    from agentrs.evaluation.benchmark import make_env, run_tasks
    from agentrs.models.policy import PolicyActor
    from agentrs.training.common import load_model

    torch.set_num_threads(threads)
    reg = CheckpointRegistry(registry_path)
    prod = reg.production("policy")
    assert prod is not None, "no production policy"
    policy = load_model(prod.path)
    policy.eval()
    tasks = [make_task(f, "train", start_index + i) for i, f in enumerate(families)]
    eps = run_tasks(make_env("miniweb"), tasks, PolicyActor(policy, "sample", seed=seed),
                    checkpoints={"policy": prod.id})
    ExperienceStore(inbox).write_episodes(eps, tag=f"actor{seed}")
    return {"episodes": len(eps), "success": sum(bool(e.success) for e in eps) / max(1, len(eps)),
            "steps": sum(e.num_steps for e in eps), "policy": prod.id,
            "by_family": [(e.family, bool(e.success)) for e in eps]}


# ------------------------------------------------------------------------------ evaluator
def evaluator_main(registry_path: str, candidate_id: str, families: list[str], regression_families: list[str],
                   n_per_family: int, gate: dict, threads: int = 1) -> dict:
    import torch

    from agentrs.environments.miniweb.tasks import task_set
    from agentrs.evaluation.benchmark import evaluate_actor, make_env
    from agentrs.models.policy import PolicyActor
    from agentrs.training.common import load_model

    torch.set_num_threads(threads)
    reg = CheckpointRegistry(registry_path)
    g = PromotionGate.model_validate(gate)
    env = make_env("miniweb")
    fams = sorted(set(families) | set(regression_families))
    tasks = task_set("val", fams, n_per_family=n_per_family)
    cand = reg.get(candidate_id)
    m_c, _ = evaluate_actor(env, tasks, PolicyActor(load_model(cand.path), "greedy"))
    prod = reg.production(cand.component)
    m_p = None
    if prod is not None:
        m_p, _ = evaluate_actor(env, tasks, PolicyActor(load_model(prod.path), "greedy"))
    ok, why = g.decide(m_c, m_p)
    reg.update_metrics(candidate_id, {"val": m_c})
    if ok:
        reg.promote(candidate_id, why)
    else:
        reg.reject(candidate_id, why)
    return {"candidate": candidate_id, "promoted": ok, "reason": why, "candidate_metrics": _brief(m_c),
            "production_metrics": _brief(m_p) if m_p else None, "production_before": prod.id if prod else None}


def _brief(m: dict) -> dict:
    return {k: m[k] for k in ("success_rate", "invalid_action_rate", "by_family", "n") if k in m}


def _run(fn, kwargs: dict, separate: bool) -> dict:
    if not separate:
        return fn(**kwargs)
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        return pool.apply(fn, kwds=kwargs)


# ------------------------------------------------------------------------------ learner
class Learner:
    def __init__(self, cfg: ContinualConfig, reg: CheckpointRegistry):
        from agentrs.models.critic import Critic
        from agentrs.models.policy import LMPolicy
        from agentrs.models.world_model import WorldModel
        from agentrs.training.common import GradientTrainer, OptimConfig, load_model
        from agentrs.training.rl import RLTrainer

        self.cfg, self.reg = cfg, reg
        self.root = Path(cfg.root)
        self.inbox = ExperienceStore(self.root / "inbox")
        self.state_path = self.root / "learner_state.json"
        st = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"processed": []}
        self.processed: set[str] = set(st["processed"])
        self.replay = ReplayBuffer(seed=cfg.seed)
        prod = reg.production("policy")
        self.policy = load_model(prod.path) if prod else (load_model(cfg.initial_policy) if cfg.initial_policy
                                                          else LMPolicy(cfg.policy))
        self.critic = load_model(cfg.initial_critic) if cfg.initial_critic else Critic(cfg.critic)
        self.wm = None
        if cfg.use_world_model:
            self.wm = load_model(cfg.initial_world_model) if cfg.initial_world_model else WorldModel(cfg.world_model)
            self.wm_trainer = GradientTrainer(self.wm, OptimConfig(lr=5e-4, total_steps=10**6))
        self.rl = RLTrainer(self.policy, self.critic, cfg.rl, seed=cfg.seed)
        # previously ingested shards are replayed (historical experience survives restarts)
        old = [p for p in self.inbox.shards("real") if p.name in self.processed]
        if old:
            self.replay.extend(self.inbox.load_transitions(shards=old))

    def save_state(self) -> None:
        self.state_path.write_text(json.dumps({"processed": sorted(self.processed)}))

    def ingest(self):
        new = [p for p in self.inbox.shards("real") if p.name not in self.processed]
        ts = self.inbox.load_transitions(shards=new) if new else []
        self.replay.extend(ts)
        for p in new:
            self.processed.add(p.name)
        self.save_state()
        eps: dict[str, list] = {}
        for t in ts:
            eps.setdefault(t.episode_id, []).append(t)
        return [sorted(v, key=lambda t: t.step) for v in eps.values()]

    def train(self, episodes, cycle: int) -> dict:
        from agentrs.training.common import save_checkpoint
        from agentrs.training.world_model import WMTrainConfig, train_world_model

        stats: dict = {}
        if self.wm is not None and self.cfg.wm_steps_per_cycle:
            new_ts = [t for ep in episodes for t in ep]
            data = new_ts + self.replay.sample(len(new_ts), "historical")
            r = train_world_model(self.wm, data, [], WMTrainConfig(steps=self.cfg.wm_steps_per_cycle,
                                                                   eval_every=10**9, seed=cycle),
                                  trainer=self.wm_trainer)
            stats["wm_loss"] = r["history"][-1]["loss"]
        upd = self.rl.update(episodes)
        stats.update({f"rl_{k}": v for k, v in upd.items()})
        ck = self.root / "checkpoints" / f"policy_c{cycle}"
        save_checkpoint(ck, self.policy, self.rl.pt, extra={"cycle": cycle})
        prod = self.reg.production("policy")
        cid = self.reg.register("policy", str(ck), parent=prod.id if prod else None, metrics={"train": stats})
        if self.wm is not None:
            save_checkpoint(self.root / "checkpoints" / f"wm_c{cycle}", self.wm, self.wm_trainer)
        save_checkpoint(self.root / "checkpoints" / f"critic_c{cycle}", self.critic, self.rl.ct)
        stats["candidate"] = cid
        return stats

    def reset_to_production(self) -> None:
        from agentrs.training.common import load_weights

        prod = self.reg.production("policy")
        if prod is not None:
            load_weights(prod.path, self.policy)


def run_continual(cfg: ContinualConfig) -> dict:
    from agentrs.environments.miniweb.families import TRAIN_FAMILIES
    from agentrs.models.policy import LMPolicy
    from agentrs.training.common import load_model, save_checkpoint

    root = Path(cfg.root)
    root.mkdir(parents=True, exist_ok=True)
    reg = CheckpointRegistry(root / "registry.sqlite")
    tracker = Tracker(root / "tracking", config=cfg.model_dump())
    rng = random.Random(cfg.seed)
    cur = Curriculum(cfg.curriculum)
    cstate = root / "curriculum.json"
    if cstate.exists():
        cur.load_state(json.loads(cstate.read_text()))
    if reg.production("policy") is None:  # bootstrap production with the initial policy
        init = load_model(cfg.initial_policy) if cfg.initial_policy else LMPolicy(cfg.policy)
        p = save_checkpoint(root / "checkpoints" / "policy_c0_init", init)
        reg.promote(reg.register("policy", str(p), notes="initial"), "bootstrap")
    learner = Learner(cfg, reg)
    report = {"cycles": []}
    hist_by_family: dict[str, list[float]] = {}
    task_counter = rng.randint(10_000, 20_000)
    for cycle in range(1, cfg.cycles + 1):
        t0 = time.time()
        fams = cur.sample_families(cfg.episodes_per_cycle, rng)
        act = _run(actor_main, dict(registry_path=str(root / "registry.sqlite"), inbox=str(root / "inbox"),
                                    families=fams, seed=cfg.seed * 100 + cycle, start_index=task_counter,
                                    threads=cfg.threads_per_process), cfg.separate_processes)
        task_counter += len(fams)
        for fam, s in act["by_family"]:
            cur.record(fam, s)
        eps = learner.ingest()
        lst = learner.train(eps, cycle)
        mastered = [f for s in cur.cfg.stages[: cur.stage] for f in s]
        ev = _run(evaluator_main, dict(registry_path=str(root / "registry.sqlite"), candidate_id=lst["candidate"],
                                       families=cur.frontier_families(), regression_families=mastered,
                                       n_per_family=cfg.eval_tasks_per_family, gate=cfg.gate.model_dump(),
                                       threads=cfg.threads_per_process), cfg.separate_processes)
        if not ev["promoted"] and cfg.reset_on_reject:
            learner.reset_to_production()
        advanced = cur.maybe_advance()
        cstate.write_text(json.dumps(cur.state()))
        prodm = ev["candidate_metrics"] if ev["promoted"] else (ev["production_metrics"] or ev["candidate_metrics"])
        for fam, v in prodm.get("by_family", {}).items():
            hist_by_family.setdefault(fam, []).append(v)
        forgetting = {f: max(v) - v[-1] for f, v in hist_by_family.items() if len(v) > 1}
        rec = {"cycle": cycle, "actor": {k: act[k] for k in ("episodes", "success", "steps", "policy")},
               "learner": {k: v for k, v in lst.items() if not isinstance(v, dict)}, "evaluation": ev,
               "curriculum": cur.state(), "advanced": advanced,
               "mean_forgetting": (sum(forgetting.values()) / len(forgetting)) if forgetting else 0.0,
               "wall_s": time.time() - t0}
        report["cycles"].append(rec)
        tracker.log({"actor_success": act["success"], "promoted": float(ev["promoted"]),
                     "production_val_success": prodm.get("success_rate", 0.0), "stage": cur.stage,
                     "mean_forgetting": rec["mean_forgetting"]}, step=cycle, prefix="continual/")
        (root / "continual_report.json").write_text(json.dumps(report, indent=1, default=str))
    tracker.close()
    _ = TRAIN_FAMILIES
    return report
