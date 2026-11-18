"""Overnight multi-domain continual-RL run (``python -m agentrs.econ.overnight``).

Phases (persisted in <run>/state.json; a restarted process resumes where it stopped):
  baseline  A: evaluate the untrained pretrained policy on validation and the sealed test set
  bc        B: behaviour cloning on verified expert demonstrations for families where the
               untrained policy succeeds rarely; then evaluate on val + sealed test
  rl        C: repeated cycles — sample a domain-balanced curriculum batch, run GRPO groups with the
               current policy, verify, store, PPO-update the shared policy (+demo replay), train the
               world model and critic, checkpoint; every ``eval_every`` cycles evaluate on validation
               and promote the candidate or roll back to the last promoted policy
  final        evaluate final + best checkpoints on the sealed test set, write the report, back up
Stops at the wall-clock deadline, the spending cap, a STOP file, or low disk — always through
the final phase.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import torch
from pydantic import BaseModel

from agentrs.econ.core import REGISTRY, load_all, make_task
from agentrs.econ.learner import (HORIZON_LABELS, GenPolicy, bc_examples, critic_targets, eval_critic, eval_wm,
                                  eval_wm_horizons, group_advantages, pg_samples, policy_backbone_cfg, ppo_update,
                                  small_backbone_cfg, state_text, summarize, train_bc, train_critic, train_wm,
                                  wm_example, wm_examples)
from agentrs.task_factory import TaskFactory
from agentrs.econ.rollout import Episode, rollout_batch, run_expert
from agentrs.models.backbone import LMBackbone
from agentrs.models.critic import Critic, CriticConfig
from agentrs.training.common import (GradientTrainer, OptimConfig, load_trainer_state, load_weights, param_snapshot,
                                     rng_state, save_checkpoint, set_rng_state)


class RunConfig(BaseModel):
    run_dir: str = "runs/overnight"
    hours: float = 8.0
    final_reserve_h: float = 0.8
    max_spend_usd: float = 6.0
    dph_usd: float = 0.0              # filled from the Vast API when available
    policy_model: str = "Qwen/Qwen3-4B"
    small_model: str = "Qwen/Qwen3-0.6B"
    val_per_family: int = 2
    test_per_family: int = 6
    tasks_per_cycle: int = 16
    group_size: int = 4
    gen_batch: int = 32
    max_new_tokens: int = 256
    bc_threshold: float = 0.5         # BC for training families whose untrained success is below this
    bc_demos_per_family: int = 40
    bc_steps: int = 160
    bc_lr: float = 1e-4
    rl_lr: float = 2e-5
    rl_min_lr: float = 2.5e-6
    ppo_clip: float = 0.2
    ppo_epochs: int = 1
    bc_replay_coef: float = 0.05
    wm_steps_per_cycle: int = 40
    critic_steps_per_cycle: int = 30
    eval_every: int = 2
    promote_tolerance: float = 0.04
    snapshot_every_min: float = 30.0
    keep_snapshots: int = 3
    min_free_disk_gb: float = 8.0
    seed: int = 0
    backup_target: str = ""
    max_cycles: int = 0               # >0: stop RL after this many cycles (dry runs)
    wm_horizons: list = [1, 2, 4, 0]     # delayed-consequence heads: +k actions later, 0 = end of task
    wm_assist_every: int = 6          # matched val comparison: direct policy vs world-model-assisted selection
    wm_assist_k: int = 3              # sampled candidates (+ the greedy action) re-ranked by the world model
    final_wm_assist_test: bool = True # one pre-registered WM-assisted run on the sealed test set at the end
    factory_validate_expert: bool = True


# ---------------------------------------------------------------------------- utilities
def now() -> float:
    return time.time()


def jdump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def git_info() -> dict:
    root = Path(__file__).resolve().parents[3]
    try:
        h = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
        br = subprocess.run(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.strip()
        return {"commit": h, "dirty": dirty, "branch": br}
    except OSError:
        return {}


def hf_revision(model: str) -> Optional[str]:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    ref = home / f"models--{model.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def vast_dph() -> float:
    try:
        r = subprocess.run(["vastai", "show", "instance", os.environ.get("CONTAINER_ID", ""), "--api-key",
                            os.environ.get("CONTAINER_API_KEY", ""), "--raw"], capture_output=True, text=True, timeout=60)
        return float(json.loads(r.stdout)["dph_total"])
    except Exception:  # noqa: BLE001 - optional information
        return 0.0


class Run:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.dir = Path(cfg.run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ckdir = self.dir / "ckpt"
        self.expdir = self.dir / "experience"
        self.expdir.mkdir(exist_ok=True)
        self.log_path = self.dir / "trainer.log"
        self.stop_requested = False
        signal.signal(signal.SIGTERM, self._sigterm)
        load_all()
        self.train_fams = sorted(k for k, f in REGISTRY.items() if not f.heldout)
        self.heldout_fams = sorted(k for k, f in REGISTRY.items() if f.heldout)
        self.domains = sorted({k.split("/")[0] for k in self.train_fams})
        sp = self.dir / "state.json"
        if sp.exists():
            self.state = json.loads(sp.read_text())
            self.state["restarts"] = self.state.get("restarts", 0) + 1
            self.log(f"resuming run (restart #{self.state['restarts']}) in phase {self.state['phase']}")
            self._maybe_new_session(sp)
            # bounded recovery: repeated crashes -> go straight to the final evaluation/report
            if self.state["restarts"] > 6 and self.state["phase"] in ("baseline", "bc", "rl"):
                self.state["stop_reason"] = f"too many restarts ({self.state['restarts']})"
                self.state["phase"] = "final"
                self.log("too many restarts: skipping to the final phase")
            self.save_state()
        else:
            dph = cfg.dph_usd or vast_dph()
            t0 = now()
            self.state = {
                "started_at": t0, "deadline": t0 + cfg.hours * 3600, "phase": "baseline", "cycle": 0, "version": 0,
                "config": cfg.model_dump() | {"dph_usd": dph}, "git": git_info(),
                "models": {"policy": cfg.policy_model, "policy_revision": hf_revision(cfg.policy_model),
                           "small": cfg.small_model, "small_revision": hf_revision(cfg.small_model)},
                "versions": {"torch": torch.__version__, "python": sys.version.split()[0]},
                "train_families": self.train_fams, "heldout_families": self.heldout_fams,
                "train_index": {k: 0 for k in self.train_fams}, "curriculum": {k: {"ema": None, "n": 0} for k in self.train_fams},
                "fingerprints": {}, "counters": {"attempts": 0, "successes": 0, "by_domain": {}, "policy_updates": 0,
                                                 "bc_updates": 0, "wm_updates": 0, "critic_updates": 0, "cycles": 0,
                                                 "rejected_stale_samples": 0, "env_errors": 0},
                "evals": [], "cycles": [], "promotions": [], "rl_lr": cfg.rl_lr, "best": None, "promoted": None,
                "bc_families": None, "errors": [], "restarts": 0, "last_snapshot": t0, "snapshots": [],
            }
            self.save_state()
            self.log(f"new run: deadline in {cfg.hours}h, dph ${dph:.3f}")
        self.state["config"]["dph_usd"] = self.state["config"].get("dph_usd") or cfg.dph_usd
        self._build_models()
        self.factory = TaskFactory(self.state, self.train_fams, out_dir=self.dir / "factory",
                                   validate_expert=cfg.factory_validate_expert, seed=cfg.seed)
        self.wm_data: list = []
        self.critic_data: list = []
        self.demo_examples: list = []
        self._rebuild_buffers()

    # ----------------------------------------------------------------- infra
    def _maybe_new_session(self, sp: Path) -> None:
        """A ``NEW_SESSION`` file (content: hours) starts a new training session of an existing run:
        same weights, optimizer, replay and curriculum, a fresh wall-clock deadline and restart budget.
        Consumed once, so supervisor restarts within the session keep its deadline."""
        f = self.dir / "NEW_SESSION"
        if not f.exists() or self.state.get("phase") == "done":
            return
        try:
            hours = float(f.read_text().strip() or self.cfg.hours)
        except ValueError:
            hours = self.cfg.hours
        dph = self.state["config"].get("dph_usd") or 0.0
        prev_start = self.state.get("session_started_at") or self.state["started_at"]
        last_active = sp.stat().st_mtime
        prev_spend = max(0.0, last_active - prev_start) / 3600 * dph
        self.state["prior_spend_usd"] = self.state.get("prior_spend_usd", 0.0) + prev_spend
        t = now()
        self.state.setdefault("sessions", []).append({"started_at": t, "hours": hours, "phase_at_start": self.state["phase"],
                                                      "version_at_start": self.state["version"], "git": git_info(),
                                                      "previous_session_spend_usd": round(prev_spend, 3)})
        self.state["session_started_at"] = t
        self.state["deadline"] = t + hours * 3600
        self.state["restarts"] = 0
        self.state["git"] = git_info()
        self.state["config"] = self.cfg.model_dump() | {"dph_usd": dph}
        self.save_state()
        f.unlink()
        self.log(f"NEW SESSION: {hours} h from now (phase {self.state['phase']}, version {self.state['version']}); "
                 f"previous sessions ≈ ${self.state['prior_spend_usd']:.2f}")

    def _sigterm(self, *_):
        self.stop_requested = True
        self.log("SIGTERM received: will finish the current unit of work, checkpoint and stop")

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def save_state(self) -> None:
        jdump(self.dir / "state.json", self.state)

    def status(self, stage: str, **extra) -> None:
        s = self.state
        jdump(self.dir / "trainer_status.json", {
            "heartbeat": now(), "pid": os.getpid(), "phase": s["phase"], "stage": stage, "cycle": s["cycle"],
            "policy_version": s["version"], "started_at": s["started_at"], "deadline": s["deadline"],
            "counters": s["counters"], "best": s["best"], "promoted": s["promoted"], "rl_lr": s["rl_lr"],
            "last_eval": s["evals"][-1] if s["evals"] else None, "last_cycle": s["cycles"][-1] if s["cycles"] else None,
            "latest_checkpoint": s.get("latest_checkpoint"), "last_snapshot": s.get("last_snapshot"),
            "errors": s["errors"][-5:], "restarts": s["restarts"], "unique_task_fingerprints": len(s["fingerprints"]),
            "factory": self.factory.coverage() if hasattr(self, "factory") else None,
            "session_started_at": s.get("session_started_at"), "prior_spend_usd": s.get("prior_spend_usd", 0.0),
            **extra})

    def time_left(self) -> float:
        return self.state["deadline"] - now()

    def spent(self) -> float:
        """Spend of the current session (the cap applies per session); earlier sessions are in prior_spend_usd."""
        start = self.state.get("session_started_at") or self.state["started_at"]
        return (now() - start) / 3600 * (self.state["config"].get("dph_usd") or 0.0)

    def should_stop_training(self) -> Optional[str]:
        if self.stop_requested:
            return "sigterm"
        if (self.dir / "STOP").exists():
            return "stop-file"
        if self.time_left() < self.cfg.final_reserve_h * 3600:
            return "deadline"
        dph = self.state["config"].get("dph_usd") or 0.0
        if dph and self.spent() + dph * self.cfg.final_reserve_h > self.cfg.max_spend_usd:
            return "budget"
        if shutil.disk_usage(self.dir).free < self.cfg.min_free_disk_gb * 2**30:
            return "disk"
        if self.cfg.max_cycles and self.state["cycle"] >= self.cfg.max_cycles:
            return "max-cycles"
        return None

    # ----------------------------------------------------------------- models / checkpoints
    def _build_models(self):
        c = self.cfg
        t0 = now()
        self.policy = GenPolicy(policy_backbone_cfg(c.policy_model), max_new_tokens=c.max_new_tokens, gen_batch=c.gen_batch)
        self.critic = Critic(CriticConfig(backbone=small_backbone_cfg(c.small_model)))
        self.wm = LMBackbone(small_backbone_cfg(c.small_model).model_copy(update={"seed": 1}))
        self.pt = GradientTrainer(self.policy.bb, OptimConfig(lr=self.state["rl_lr"], total_steps=10**6, grad_clip=1.0))
        self.ct = GradientTrainer(self.critic, OptimConfig(lr=1e-4, total_steps=10**6))
        self.wt = GradientTrainer(self.wm, OptimConfig(lr=1e-4, total_steps=10**6))
        latest = self.ckdir / "latest"
        after_bc = self.ckdir / "after_bc"
        if (latest / "DONE").exists():
            self.load_ckpt(latest)
            self.log(f"loaded checkpoint {latest} (version {self.state['version']})")
        elif (after_bc / "DONE").exists() and self.state["phase"] in ("bc", "rl"):
            # stopped after BC finished but before the RL phase checkpointed: resume from the BC weights
            self.load_ckpt(after_bc)
            self.log(f"loaded checkpoint {after_bc} (BC already done; not repeating it)")
        elif not (self.ckdir / "initial" / "DONE").exists():
            self.save_ckpt("initial", note="pretrained policy (LoRA at init = base model)")
        self.log(f"models ready in {now() - t0:.0f}s; trainable policy params "
                 f"{sum(p.numel() for p in self.policy.bb.parameters() if p.requires_grad) / 1e6:.1f}M")

    def save_ckpt(self, name: str, note: str = "") -> Path:
        final = self.ckdir / name
        tmp = self.ckdir / (name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        save_checkpoint(tmp / "policy", self.policy.bb, self.pt)
        save_checkpoint(tmp / "critic", self.critic, self.ct)
        save_checkpoint(tmp / "world_model", self.wm, self.wt)
        torch.save({"rng": rng_state(), "gen": self.policy.gen.get_state()}, tmp / "rng.pt")
        jdump(tmp / "meta.json", {"name": name, "note": note, "time": now(), "version": self.state["version"],
                                  "cycle": self.state["cycle"], "phase": self.state["phase"], "models": self.state["models"],
                                  "git": self.state["git"], "rl_lr": self.state["rl_lr"]})
        jdump(tmp / "state.json", self.state)
        (tmp / "DONE").write_text(str(now()))
        old = self.ckdir / (name + ".old")
        if old.exists():
            shutil.rmtree(old)
        if final.exists():
            os.replace(final, old)
        os.replace(tmp, final)
        shutil.rmtree(old, ignore_errors=True)
        if name == "latest":
            self.state["latest_checkpoint"] = {"path": str(final), "time": now(), "version": self.state["version"]}
        return final

    def load_ckpt(self, path: Path, weights_only: bool = False) -> None:
        load_weights(path / "policy", self.policy.bb)
        load_weights(path / "critic", self.critic)
        load_weights(path / "world_model", self.wm)
        if not weights_only:
            load_trainer_state(path / "policy", self.pt, restore_rng=False)
            load_trainer_state(path / "critic", self.ct, restore_rng=False)
            load_trainer_state(path / "world_model", self.wt, restore_rng=False)
            r = torch.load(path / "rng.pt", weights_only=False)
            set_rng_state(r["rng"])
            self.policy.gen.set_state(r["gen"])

    def load_policy_only(self, path: Path) -> None:
        load_weights(path / "policy", self.policy.bb)

    def maybe_snapshot(self) -> None:
        if now() - self.state.get("last_snapshot", 0) < self.cfg.snapshot_every_min * 60:
            return
        name = "snap_" + time.strftime("%H%M")
        self.save_ckpt(name, note="periodic snapshot")
        self.state["snapshots"].append(name)
        self.state["last_snapshot"] = now()
        while len(self.state["snapshots"]) > self.cfg.keep_snapshots:
            shutil.rmtree(self.ckdir / self.state["snapshots"].pop(0), ignore_errors=True)
        self.log(f"snapshot {name}")

    # ----------------------------------------------------------------- experience
    def write_episodes(self, name: str, eps: list[Episode], provenance_note: str) -> None:
        p = self.expdir / f"{name}.jsonl.gz"
        with gzip.open(p, "at") as f:
            for e in eps:
                rec = e.to_record()
                for s, st in zip(rec["steps"], e.steps):
                    s.pop("messages", None)
                f.write(json.dumps(rec, default=str) + "\n")
        reg_p = self.dir / "registry.json"
        reg = json.loads(reg_p.read_text()) if reg_p.exists() else {"schema": "econ-exp-v1", "datasets": {}}
        d = reg["datasets"].setdefault(name, {"path": str(p), "episodes": 0, "provenance": provenance_note})
        d["episodes"] += len(eps)
        d["reward_eligible"] = provenance_note in ("agent_policy",)
        d["imitation_eligible"] = provenance_note in ("expert_demonstration",)
        d["splits"] = sorted(set(d.get("splits", [])) | {e.split for e in eps})
        jdump(reg_p, reg)

    def _rebuild_buffers(self) -> None:
        """After a restart: rebuild WM / critic replay from stored training-split agent experience."""
        for p in sorted(self.expdir.glob("cycle_*.jsonl.gz"))[-30:]:
            try:
                with gzip.open(p, "rt") as f:
                    for line in f:
                        self._add_buffers([_episode_from_record(json.loads(line))])
            except (OSError, EOFError, json.JSONDecodeError) as e:
                self.log(f"warning: could not read {p}: {e}")
        if self.wm_data:
            self.log(f"rebuilt replay: {len(self.wm_data)} WM / {len(self.critic_data)} critic examples")
        if self.state.get("bc_families") and self.state["phase"] in ("bc", "rl", "final") \
                and (self.ckdir / "after_bc" / "DONE").exists():
            demos = self.demos_for(self.state["bc_families"], self.cfg.bc_demos_per_family)  # deterministic
            self.demo_examples = bc_examples(self.policy, demos)
            self.log(f"regenerated {len(self.demo_examples)} demonstration examples for BC replay")

    def _add_buffers(self, eps: list[Episode]) -> None:
        for e in eps:
            if e.split != "train":
                continue  # held-out data never enters training buffers
            self.wm_data += wm_examples(e, tuple(self.cfg.wm_horizons))
            self.critic_data += critic_targets([e])
        self.wm_data = self.wm_data[-60000:]
        self.critic_data = self.critic_data[-30000:]

    # ----------------------------------------------------------------- evaluation
    def eval_tasks(self, split: str, fams: list[str], per_family: int):
        return [make_task(k, split, i) for k in fams for i in range(per_family)]

    def evaluate(self, tag: str, split: str, fams: Optional[list[str]] = None, per_family: Optional[int] = None,
                 version: Optional[str] = None, act=None, mode: str = "direct") -> dict:
        fams = fams or (self.train_fams + self.heldout_fams)
        per = per_family or (self.cfg.test_per_family if split == "test" else self.cfg.val_per_family)
        tasks = self.eval_tasks(split, fams, per)
        t0 = now()
        self.status(f"eval:{tag}:{split}", eval_tasks=len(tasks))
        eps = rollout_batch(tasks, act or (lambda r: self.policy.act(r, greedy=True)),
                            version or f"v{self.state['version']}", provenance="evaluation")
        self.write_episodes(f"eval_{tag}_{split}", eps, "evaluation")
        summ = summarize(eps)
        tr = [e for e in eps if f"{e.domain}/{e.family}" in self.train_fams]
        ho = [e for e in eps if f"{e.domain}/{e.family}" in self.heldout_fams]
        rec = {"tag": tag, "split": split, "mode": mode, "time": now(), "wall_s": now() - t0, "version": self.state["version"],
               "cycle": self.state["cycle"], "summary": summ,
               "train_families": summarize(tr)["overall"] if tr else None,
               "heldout_families": summarize(ho)["overall"] if ho else None}
        if split == "val" and self.wm_data and mode == "direct":
            vdata = [wm_example(e, t) for e in eps for t in range(len(e.steps))]
            rec["world_model"] = eval_wm(self.wm, vdata[:400])
            rec["world_model_horizons"] = eval_wm_horizons(self.wm, eps, horizons=(1, 2, 4, 0))
            rec["critic"] = eval_critic(self.critic, critic_targets(eps)[:600])
        self.state["evals"].append(rec)
        self.save_state()
        o = summ["overall"]
        self.log(f"eval {tag}/{split}: {o['success']}/{o['n']} = {o['rate']:.3f} CI{o['ci95']} | train-fam "
                 f"{rec['train_families']['rate'] if tr else None} heldout-fam {rec['heldout_families']['rate'] if ho else None} "
                 f"({now() - t0:.0f}s)")
        return rec

    def wm_assisted_act(self):
        """MuZero-inspired optional planning, used only in matched evaluations: the policy proposes the
        greedy action plus ``wm_assist_k`` sampled ones; the learned world model scores each by
        log P(valid) + log P(task ends in success | state, action) from its delayed-consequence head, and
        the best is executed. Direct generation stays the default policy; this tests whether the
        learned consequence model improves verified task success."""
        k = self.cfg.wm_assist_k

        def act(reqs, ctx):
            greedy = self.policy.act(reqs, greedy=True)
            samp = self.policy.act([r for r in reqs for _ in range(k)], greedy=False, temperature=1.0)
            out = []
            for i, ((task, ep), g) in enumerate(zip(ctx, greedy)):
                cands, seen = [], set()
                for o in [g] + samp[i * k:(i + 1) * k]:
                    if o["text"] not in seen:
                        seen.add(o["text"])
                        cands.append(o)
                if len(cands) == 1:
                    out.append(g)
                    continue
                t = len(ep.steps)
                prefill = reqs[i][1]
                base = state_text(ep, t)
                pv = [f"{base}\nNEXT ACTION {(prefill + o['text'])[:400]}\nELAPSED_S 0.00\nPREDICT: VALID" for o in cands]
                # expected real time to the end of the task (not known at decision time): steps left x measured s/step
                eta = self.state.get("sec_per_step", 15.0) * max(1, min(4, task.max_steps - t))
                pe = [f"{base}\nNEXT ACTION {(prefill + o['text'])[:400]}\nHORIZON END\nELAPSED_S {eta:.1f}\nPREDICT STATUS:"
                      for o in cands]
                with torch.no_grad():
                    sv = self.wm.score(pv, [[" yes", " no"]] * len(cands))
                    se = self.wm.score(pe, [HORIZON_LABELS] * len(cands))
                vals = [float(torch.log_softmax(a.float(), -1)[0] + torch.log_softmax(b.float(), -1)[1]) for a, b in zip(sv, se)]
                out.append(cands[max(range(len(cands)), key=lambda j: vals[j])])
            return out

        act.wants_ctx = True
        return act

    # ----------------------------------------------------------------- phases
    def phase_baseline(self):
        done = {(e["tag"], e["split"]) for e in self.state["evals"]}
        if ("A_pretrained", "val") not in done:
            self.evaluate("A_pretrained", "val")
        if ("A_pretrained", "test") not in done:
            self.evaluate("A_pretrained", "test")
        self.state["phase"] = "bc"
        self.save_state()

    def _family_rates(self, tag: str) -> dict:
        rates: dict = {}
        for e in self.state["evals"]:
            if e["tag"] == tag and e["split"] == "val":  # selection decisions use validation only
                for k, v in e["summary"]["by_family"].items():
                    rates[k] = v["rate"]
        return rates

    def demos_for(self, fams: list[str], n: int) -> list[Episode]:
        eps = []
        for k in fams:
            for i in range(n):
                # demonstrations use a disjoint index range of the training split
                t = make_task(k, "train", 500_000 + i)
                ep = run_expert(t)
                if ep.success:
                    eps.append(ep)
        return eps

    def phase_bc(self):
        if (self.ckdir / "after_bc" / "DONE").exists() and self.state.get("bc"):
            self.log("BC already completed (ckpt/after_bc); finishing its evaluations, then RL")
            if not self.wm_data:
                demos = self.demos_for(self.state["bc_families"], self.cfg.bc_demos_per_family)
                self._add_buffers(demos)
            return self._after_bc_evals()
        rates = self._family_rates("A_pretrained")
        fams = [k for k in self.train_fams if REGISTRY[k].oracle is not None or k.startswith("browser/")]
        bc_fams = [k for k in fams if rates.get(k, 0.0) < self.cfg.bc_threshold]
        self.state["bc_families"] = bc_fams
        self.log(f"BC on {len(bc_fams)} families with untrained val success < {self.cfg.bc_threshold}: {bc_fams}")
        demos = self.demos_for(bc_fams, self.cfg.bc_demos_per_family)
        self.write_episodes("demos", demos, "expert_demonstration")
        ex = bc_examples(self.policy, demos)
        self.demo_examples = ex
        self.log(f"{len(demos)} verified demonstrations -> {len(ex)} BC examples")
        if ex:
            self.status("bc", bc_examples=len(ex))
            bt = GradientTrainer(self.policy.bb, OptimConfig(lr=self.cfg.bc_lr, total_steps=self.cfg.bc_steps,
                                                             warmup_steps=10, schedule="cosine"))
            snap = param_snapshot(self.policy.bb)
            hist = train_bc(self.policy, bt, ex, self.cfg.bc_steps, batch=8, micro=4, seed=self.cfg.seed, log=self.log)
            change = math.sqrt(sum(float((p.detach() - snap[n]).float().pow(2).sum())
                                   for n, p in self.policy.bb.named_parameters() if n in snap))
            self.state["counters"]["bc_updates"] += len(hist)
            k = min(10, len(hist))
            self.state["bc"] = {"examples": len(ex), "steps": len(hist), "loss_first10": sum(h["loss"] for h in hist[:k]) / k,
                                "loss_last10": sum(h["loss"] for h in hist[-k:]) / k, "param_change_l2": change}
            self.log(f"BC done: {self.state['bc']}")
        # world model / critic warm-start on demonstrations (training split only)
        self._add_buffers(demos)
        if self.wm_data:
            l = train_wm(self.wm, self.wt, self.wm_data, 120, seed=1)
            self.state["counters"]["wm_updates"] += len(l)
            self.log(f"WM warm-start loss {l[0]:.3f} -> {sum(l[-10:]) / 10:.3f}")
        self.state["version"] += 1
        self.save_ckpt("after_bc", note="after behaviour cloning")
        self._after_bc_evals()

    def _eval_rec(self, tag: str, split: str) -> Optional[dict]:
        return next((e for e in reversed(self.state["evals"]) if e["tag"] == tag and e["split"] == split), None)

    def _after_bc_evals(self):
        if not self._eval_rec("B_after_bc", "val"):
            self.evaluate("B_after_bc", "val")
        if not self._eval_rec("B_after_bc", "test"):
            self.evaluate("B_after_bc", "test")
        if self.stop_requested:
            return
        val = self._eval_rec("B_after_bc", "val")["train_families"]["rate"]
        self.state["promoted"] = {"version": self.state["version"], "val": val, "ckpt": "promoted", "cycle": 0}
        self.state["best"] = {"version": self.state["version"], "val": val, "ckpt": "best", "cycle": 0}
        self.save_ckpt("promoted", "promoted after BC")
        self.save_ckpt("best", "best validation so far (after BC)")
        self.state["phase"] = "rl"
        self.save_ckpt("latest")
        self.save_state()

    def pick_tasks(self) -> list:
        """Admitted, validated, curriculum-weighted, domain-balanced fresh training tasks (task factory)."""
        picks = self.factory.sample(self.cfg.tasks_per_cycle, self.state["cycle"], self.state["train_index"],
                                    self.state["curriculum"])
        for t in picks:
            fp = t.fingerprint()
            self.state["fingerprints"][fp] = self.state["fingerprints"].get(fp, 0) + 1
        return picks

    def pick_tasks_legacy(self) -> list:
        rng = random.Random(self.cfg.seed * 7919 + self.state["cycle"])
        cur = self.state["curriculum"]
        picks = []
        doms = []
        while len(doms) < self.cfg.tasks_per_cycle:
            d = list(self.domains)
            rng.shuffle(d)
            doms += d
        for d in doms[:self.cfg.tasks_per_cycle]:
            fams = [k for k in self.train_fams if k.startswith(d + "/")]
            w = []
            for k in fams:
                ema = cur[k]["ema"]
                # learning-progress weighting: most weight where success is neither ~0 nor ~1
                w.append(1.0 if ema is None else max(0.08, 4 * ema * (1 - ema)) + (0.15 if ema < 0.05 else 0.0))
            k = rng.choices(fams, w)[0]
            idx = self.state["train_index"][k]
            self.state["train_index"][k] = idx + 1
            t = make_task(k, "train", idx)
            fp = t.fingerprint()
            self.state["fingerprints"][fp] = self.state["fingerprints"].get(fp, 0) + 1
            picks.append(t)
        return picks

    def rl_cycle(self):
        c = self.cfg
        cyc = self.state["cycle"] + 1
        version = f"v{self.state['version']}"
        t0 = now()
        tasks = self.pick_tasks()
        group_tasks, groups = [], []
        for t in tasks:
            for g in range(c.group_size):
                group_tasks.append(t)
                groups.append(f"c{cyc}:{t.id}")
        self.status("collect", cycle_tasks=len(group_tasks))
        eps = rollout_batch(group_tasks, lambda r: self.policy.act(r, greedy=False, temperature=1.0), version,
                            provenance="agent_policy", groups=groups)
        t_col = now() - t0
        self.state["sec_per_step"] = t_col / max(1, max(len(e.steps) for e in eps))
        torch.cuda.empty_cache()  # generation KV caches -> free before the gradient phase
        self.write_episodes(f"cycle_{cyc:04d}", eps, "agent_policy")
        cnt = self.state["counters"]
        for e in eps:
            cnt["attempts"] += 1
            cnt["successes"] += int(e.success)
            dd = cnt["by_domain"].setdefault(e.domain, {"attempts": 0, "successes": 0})
            dd["attempts"] += 1
            dd["successes"] += int(e.success)
            if "error" in e.verification:
                cnt["env_errors"] += 1
        for k in {f"{e.domain}/{e.family}" for e in eps}:
            es = [e for e in eps if f"{e.domain}/{e.family}" == k]
            r = sum(e.success for e in es) / len(es)
            cur = self.state["curriculum"][k]
            cur["ema"] = r if cur["ema"] is None else 0.7 * cur["ema"] + 0.3 * r
            cur["n"] += len(es)
        # ---- policy update (on-policy samples of this version only)
        adv = group_advantages(eps)
        samples = pg_samples(eps, adv)
        self.status("update", samples=len(samples))
        t1 = now()
        snap = param_snapshot(self.policy.bb)
        st = ppo_update(self.policy, self.pt, samples, version, demo_examples=self.demo_examples,
                        bc_coef=c.bc_replay_coef, clip=c.ppo_clip, epochs=c.ppo_epochs, seed=cyc, log=None)
        st["policy_lora_change_l2"] = math.sqrt(sum(float((p.detach() - snap[n]).float().pow(2).sum())
                                                    for n, p in self.policy.bb.named_parameters() if n in snap))
        del snap
        cnt["policy_updates"] += st["updates"]
        cnt["rejected_stale_samples"] += st["rejected_stale"]
        if st["updates"]:
            self.state["version"] += 1
        t_upd = now() - t1
        torch.cuda.empty_cache()
        # ---- world model + critic on verified training experience
        self._add_buffers(eps)
        t2 = now()
        wl = train_wm(self.wm, self.wt, self.wm_data, c.wm_steps_per_cycle, seed=cyc) if self.wm_data else []
        cl = train_critic(self.critic, self.ct, self.critic_data, c.critic_steps_per_cycle, seed=cyc) if self.critic_data else []
        cnt["wm_updates"] += len(wl)
        cnt["critic_updates"] += len(cl)
        cnt["cycles"] = cyc
        self.state["cycle"] = cyc
        n_groups = len({e.group for e in eps})
        informative = len({e.group for e in eps if adv.get(e.episode_id, 0.0) != 0.0})
        rec = {"cycle": cyc, "time": now(), "version_collected": version, "version_after": self.state["version"],
               "episodes": len(eps), "success_rate": sum(e.success for e in eps) / len(eps),
               "valid_action_rate": 1 - sum(e.n_invalid for e in eps) / max(1, sum(len(e.steps) for e in eps)),
               "by_domain": {d: summarize([e for e in eps if e.domain == d])["overall"]["rate"] for d in sorted({e.domain for e in eps})},
               "groups": n_groups, "informative_groups": informative, "ppo": st,
               "wm_loss": sum(wl[-10:]) / max(1, len(wl[-10:])) if wl else None,
               "critic_loss": sum(cl[-10:]) / max(1, len(cl[-10:])) if cl else None,
               "collect_s": t_col, "update_s": t_upd, "aux_s": now() - t2, "rl_lr": self.state["rl_lr"],
               "factory": self.factory.coverage()}
        self.state["cycles"].append(rec)
        self.log(f"cycle {cyc}: success {rec['success_rate']:.3f} valid {rec['valid_action_rate']:.3f} informative "
                 f"{informative}/{n_groups} groups, {st['samples']} samples, {st['updates']} updates, "
                 f"mismatch {st['mismatch_first']}, kl {st['kl_approx']:.4f}, clipfrac {st['clipfrac']:.3f}, "
                 f"dLoRA {st['policy_lora_change_l2']:.4f}, WM {rec['wm_loss']}, critic {rec['critic_loss']} | "
                 f"collect {t_col:.0f}s update {t_upd:.0f}s aux {now() - t2:.0f}s")
        self.save_ckpt("latest")
        self.save_state()
        self.maybe_snapshot()
        if cyc % c.eval_every == 0:
            self.promotion_gate()
        if c.wm_assist_every and cyc % c.wm_assist_every == 0 and not self.should_stop_training():
            self.evaluate(f"cycle{cyc}_wm", "val", act=self.wm_assisted_act(), mode="wm_assist")

    def promotion_gate(self):
        rec = self.evaluate(f"cycle{self.state['cycle']}", "val")
        val = rec["train_families"]["rate"]
        prom = self.state["promoted"]
        decision = {"cycle": self.state["cycle"], "version": self.state["version"], "val": val, "promoted_val": prom["val"]}
        if val >= prom["val"] - self.cfg.promote_tolerance:
            decision["decision"] = "promote"
            self.state["promoted"] = {"version": self.state["version"], "val": val, "ckpt": "promoted", "cycle": self.state["cycle"]}
            self.save_ckpt("promoted", note=f"promoted at cycle {self.state['cycle']}")
            if val > self.state["best"]["val"]:
                self.state["best"] = {"version": self.state["version"], "val": val, "ckpt": "best", "cycle": self.state["cycle"]}
                self.save_ckpt("best", note=f"best validation at cycle {self.state['cycle']}")
                decision["new_best"] = True
        else:
            decision["decision"] = "reject"
            self.load_policy_only(self.ckdir / "promoted")
            self.state["rl_lr"] = max(self.cfg.rl_min_lr, self.state["rl_lr"] * 0.5)
            for g in self.pt.optimizer.param_groups:
                g["lr"] = self.state["rl_lr"]
            self.pt.scheduler.base_lrs = [self.state["rl_lr"] for _ in self.pt.scheduler.base_lrs]
            self.state["version"] += 1  # reverted weights are a new behaviour version
            decision["new_lr"] = self.state["rl_lr"]
        self.state["promotions"].append(decision)
        self.log(f"promotion gate: {decision}")
        self.save_ckpt("latest")
        self.save_state()

    def phase_rl(self):
        while True:
            why = self.should_stop_training()
            if why == "sigterm":  # restart/stop request: checkpoint and exit, RL resumes on restart
                self.log("SIGTERM: leaving RL loop without changing phase")
                return
            if why:
                self.log(f"stopping RL: {why}")
                self.state["stop_reason"] = why
                break
            try:
                self.rl_cycle()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self.state["errors"].append({"time": now(), "error": "CUDA OOM", "cycle": self.state["cycle"]})
                ooms = sum(1 for e in self.state["errors"] if e["error"] == "CUDA OOM")
                self.policy.gen_batch = max(8, self.policy.gen_batch // 2)
                self.log(f"CUDA OOM #{ooms}: gen_batch -> {self.policy.gen_batch}")
                if ooms >= 4:
                    self.state["stop_reason"] = "repeated OOM"
                    break
                self.load_ckpt(self.ckdir / "latest")
            self.save_state()
        self.state["phase"] = "final"
        self.save_state()

    def phase_final(self):
        done = {(e["tag"], e["split"]) for e in self.state["evals"]}
        self.save_ckpt("final", note="end of training")
        if ("C_final", "test") not in done:
            self.evaluate("C_final", "test")
        best = self.state.get("best") or {}
        if best and best.get("version") != self.state["version"] and ("C_best", "test") not in done \
                and (self.ckdir / "best" / "DONE").exists() and self.time_left() > -1800:
            self.load_policy_only(self.ckdir / "best")
            self.evaluate("C_best", "test")
        if self.cfg.final_wm_assist_test and ("C_final_wm", "test") not in done and self.time_left() > -1800:
            if (self.ckdir / "final" / "DONE").exists():
                self.load_policy_only(self.ckdir / "final")
            self.evaluate("C_final_wm", "test", act=self.wm_assisted_act(), mode="wm_assist")
        self.write_report()
        if self.cfg.backup_target:
            try:
                from agentrs.backup import backup_once, make_target

                r = backup_once(make_target(self.cfg.backup_target), self.dir.parent)
                self.state["final_backup"] = r
            except Exception as e:  # noqa: BLE001
                self.state["errors"].append({"time": now(), "error": f"final backup failed: {e}"})
        self.state["phase"] = "done"
        self.state["finished_at"] = now()
        self.save_state()
        self.status("done")
        self.log("run complete")

    # ----------------------------------------------------------------- report
    def write_report(self):
        from agentrs.econ.report import build_report

        rep = build_report(self.dir, self.state)
        jdump(self.dir / "report.json", rep["json"])
        (self.dir / "report.md").write_text(rep["md"])
        self.log("report written: " + str(self.dir / "report.md"))

    def run(self):
        try:
            while self.state["phase"] != "done":
                ph = self.state["phase"]
                self.status(ph)
                if ph == "baseline":
                    self.phase_baseline()
                elif ph == "bc":
                    self.phase_bc()
                elif ph == "rl":
                    self.phase_rl()
                elif ph == "final":
                    self.phase_final()
                if self.stop_requested and self.state["phase"] not in ("final", "done"):
                    self.log("stopping on SIGTERM after checkpoint (will resume on restart)")
                    self.save_ckpt("latest")
                    self.save_state()
                    return 3
        except Exception as e:
            self.state["errors"].append({"time": now(), "error": f"{type(e).__name__}: {e}",
                                         "trace": traceback.format_exc()[-2000:]})
            self.save_state()
            self.log("FATAL: " + traceback.format_exc())
            raise
        return 0


def _episode_from_record(r: dict) -> Episode:
    from agentrs.econ.rollout import Step

    steps = [Step(**{k: v for k, v in s.items() if k in Step.__dataclass_fields__}) for s in r.pop("steps")]
    ep = Episode(**{k: v for k, v in r.items() if k in Episode.__dataclass_fields__})
    ep.steps = steps
    return ep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agentrs.econ.overnight")
    sp = None
    ap.add_argument("--config", help="YAML with RunConfig fields")
    ap.add_argument("--set", nargs="*", default=[], help="overrides key=value")
    a = ap.parse_args(argv)
    import yaml

    d = yaml.safe_load(Path(a.config).read_text()) if a.config else {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        d[k] = yaml.safe_load(v)
    cfg = RunConfig(**d)
    sp = Path(cfg.run_dir) / "state.json"
    if sp.exists() and json.loads(sp.read_text()).get("restarts", 0) > 10:
        print("more than 10 restarts: refusing to start again (exit 0 so the supervisor stops)", flush=True)
        return 0
    if sp.exists() and json.loads(sp.read_text()).get("phase") == "done":
        print("run already complete", flush=True)
        return 0
    torch.manual_seed(cfg.seed)
    return Run(cfg).run()


if __name__ == "__main__":
    raise SystemExit(main())
