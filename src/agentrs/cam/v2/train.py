"""CAM-v2 continual off-policy training (``python -m agentrs.cam.v2.train --config configs/cam2/<arm>.yaml``).

Iteration: persistent batched workers collect real episodes with the current actor (some starting from
archived promising states) → episodes are persisted to replay (``<run>/replay``) → ``updates_per_iter``
off-policy updates on replayed episodes (world model, twin critics + expectile value, AWR or discrete
SAC actor; hindsight goal views) → optional dream updates from imagined rollouts → Lagrange
multipliers, training-goal curriculum and archive reverse curriculum adapt → checkpoint. Every
``eval_every`` iterations: greedy validation on unchanged full-difficulty goals (best checkpoint), and
world-model validation on the held-out validation episodes (open-loop errors and calibration by
horizon; sets the adaptive dream reliability). At the end: sealed test (held-out distributions and the
never-trained family), counterfactual action-ranking against real simulator outcomes and the
corrupted-model control. Real-interaction and compute budgets are recorded per phase.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from pydantic import BaseModel, model_validator

from agentrs.cam.audit import audit
from agentrs.cam.contract import RewardContract, RewardSpec
from agentrs.cam.v2.collector import Collector, summarize
from agentrs.cam.v2.data import Archive, Replay
from agentrs.cam.v2.dream import Dreamer
from agentrs.cam.v2.learner import OffPolicy, observe
from agentrs.cam.v2.model import CAM2, CAM2Config
from agentrs.cam.v2.wm_eval import corrupted, counterfactual_ranking, open_loop

ENV_VERSION = "cam-envs-contract-v1"


class TrainConfig2(BaseModel):
    run_dir: str = "runs/cam2"
    hours: float = 4.0
    final_reserve_h: float = 0.3
    train_families: dict = {"negotiation": 0.5, "product": 0.5}
    heldout_families: list = ["bizdev"]
    model: dict = {}
    seed: int = 0
    gpu_mem_fraction: float = 0.12
    workers: int = 256
    steps_per_iter: int = 4096
    max_iters: int = 0
    max_env_steps: int = 0
    reward: dict = {"mode": "contract", "gamma": 0.99, "shaping": 0.0, "time_cost": 0.005}
    learner: dict = {"actor": "awr"}
    updates_per_iter: int = 32
    batch_episodes: int = 48
    learn_start_steps: int = 4096
    her_p: float = 0.3
    prot_frac: float = 0.25
    replay_capacity: int = 300_000
    replay_protected: int = 40_000
    lagrangian: dict = {}            # {"kappa": 0.05, "lr": 0.5, "max": 2.0}
    curriculum: dict = {}            # {"product": {"start": 0.1, "lo": 0.15, "hi": 0.5, "up": 1.2, "down": 0.85, "min": 0.02}}
    archive: dict = {}               # {"frac": 0.25, "cells": 400, "investigate": 0.0, "rev_start": 0.5}
    dream: dict = {}                 # {"mode": "fixed"|"adaptive", "horizon": 1, "updates_per_iter": 16, "starts": 256, ...}
    eval_every: int = 10
    eval_episodes: int = 200
    test_episodes: int = 400
    cf_states: int = 30
    snapshot_every_min: float = 30.0
    strict_audit: bool = True
    audit_episodes: int = 30
    init_from: str = ""              # transfer: initialise all CAM-v2 weights from another run's checkpoint dir
    monitor_families: list = []      # evaluated at every gate but never trained (forgetting / transfer curves)

    @model_validator(mode="after")
    def _no_flow_mapping_typos(self):
        def walk(x, path):
            if isinstance(x, dict):
                for k, v in x.items():
                    if isinstance(k, str) and ":" in k:
                        raise ValueError(f"config key {path}{k!r} contains ':' (YAML needs a space after ':')")
                    walk(v, f"{path}{k}.")
        for name in ("model", "learner", "reward", "lagrangian", "curriculum", "archive", "dream", "train_families"):
            walk(getattr(self, name), f"{name}.")
        return self


def git_commit() -> dict:
    root = Path(__file__).resolve().parents[4]
    g = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(g("status", "--porcelain"))}


def jdump(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, p)


class Trainer2:
    def __init__(self, cfg: TrainConfig2):
        self.cfg = cfg
        self.dir = Path(cfg.run_dir)
        self.ck = self.dir / "ckpt"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stop = False
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(cfg.gpu_mem_fraction)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CAM2(CAM2Config(**cfg.model)).to(self.dev)
        self.learner = OffPolicy(self.model, cfg.learner)
        self.dreamer = Dreamer(self.model, {**cfg.dream, "gamma": cfg.reward.get("gamma", 0.99)}) if cfg.dream else None
        self.rc = RewardContract(RewardSpec(**cfg.reward))
        self.replay = Replay(cfg.replay_capacity, cfg.replay_protected, self.dir / "replay")
        self.archive = Archive(cfg.archive.get("cells", 400), cfg.seed, cfg.archive.get("investigate", 0.0)) if cfg.archive else None
        sp = self.dir / "state.json"
        if sp.exists() and (self.ck / "latest" / "DONE").exists():
            self.state = json.loads(sp.read_text())
            self.load(self.ck / "latest")
            self.state["restarts"] = self.state.get("restarts", 0) + 1
            n = self.replay.load()
            self.log(f"resumed from ckpt/latest (iteration {self.state['iter']}); replay reloaded: {n} episodes")
        else:
            t0 = time.time()
            self.state = {"started_at": t0, "deadline": t0 + cfg.hours * 3600, "phase": "train", "iter": 0, "version": 0,
                          "config": cfg.model_dump(), "git": git_commit(), "env_version": ENV_VERSION,
                          "counters": {"episodes": 0, "successes": 0, "transitions": 0, "grad_updates": 0,
                                       "dream_updates": 0, "imagined_transitions": 0, "by_family": {}},
                          "compute_s": {"collect": 0.0, "learn": 0.0, "dream": 0.0, "eval": 0.0},
                          "evals": [], "wm_evals": [], "iters": [], "best": None, "first_success": {},
                          "lagrange": {}, "curriculum": {}, "restarts": 0, "snapshots": [], "last_snapshot": t0,
                          "models": {"cam2": self.model.cfg.model_dump(exclude={"vocab"}), "qwen": None,
                                     "params": self.model.n_params()}, "dream_fallbacks": []}
            if cfg.init_from:
                self.model.load_state_dict(torch.load(Path(cfg.init_from) / "weights.pt", map_location=self.dev))
                self.state["init_from"] = cfg.init_from
                self.log(f"initialised CAM-v2 weights from {cfg.init_from}")
            self.run_audit()
            self.save_state()
            self.log(f"new CAM-v2 run: {self.model.n_params()}")
        cur = self.state.setdefault("curriculum", {})
        for fam, cc in cfg.curriculum.items():
            cur.setdefault(fam, float(cc.get("start", 1.0)))
        lam = self.state.setdefault("lagrange", {})
        if self.archive is not None:
            for f in cfg.train_families:
                self.archive.min_used.setdefault(f, cfg.archive.get("rev_start", 0.5))
        self.collector = Collector(self.model, cfg.workers, cfg.train_families, "train", seed0=self.state["iter"] * 100_000,
                                   gen_seed=cfg.seed + self.state["iter"], reward=self.rc, lam=lam, difficulty=cur,
                                   archive=self.archive, archive_frac=cfg.archive.get("frac", 0.0), env_version=ENV_VERSION)
        self.rng = random.Random(cfg.seed + 17 * self.state["iter"])

    # ------------------------------------------------------------------ infra
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.dir / "train.log", "a") as f:
            f.write(line + "\n")

    def save_state(self) -> None:
        jdump(self.dir / "state.json", self.state)

    def run_audit(self) -> None:
        fams = list(self.cfg.train_families) + list(self.cfg.heldout_families)
        lam_probe = self.cfg.lagrangian.get("probe", 0.5) if self.cfg.lagrangian else 0.0
        a = audit(self.rc, fams, n=self.cfg.audit_episodes, lam=lam_probe)
        self.state["reward_audit"] = {"passed": a["passed"], "violations": a["violations"], "lam_probe": lam_probe,
                                      "J": {f: {k: round(v["J"], 4) for k, v in d["policies"].items()} for f, d in a["families"].items()}}
        self.log(f"reward-contract audit ({self.rc.spec.mode}, lam={lam_probe}): {'PASSED' if a['passed'] else 'FAILED'} "
                 + ("; ".join(a["violations"]) if a["violations"] else ""))
        if not a["passed"] and self.cfg.strict_audit:
            raise RuntimeError("reward contract failed the adversarial audit: " + "; ".join(a["violations"]))

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def save(self, name: str, note: str = "") -> None:
        tmp = self.ck / (name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        torch.save(self.model.state_dict(), tmp / "weights.pt")
        torch.save(self.learner.state_dict(), tmp / "optimizer.pt")
        torch.save({"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "python": random.getstate(), "numpy": np.random.get_state(), "rng": self.rng.getstate() if hasattr(self, "rng") else None,
                    "dream": {"scale": self.dreamer.scale, "u_thr": self.dreamer.u_thr, "reliability": self.dreamer.reliability,
                              "max_h": self.dreamer.max_h} if self.dreamer else None}, tmp / "rng.pt")
        if self.archive is not None:
            with open(tmp / "archive.pkl", "wb") as f:
                pickle.dump({"cells": self.archive.cells, "min_used": self.archive.min_used, "stats": self.archive.stats}, f)
        (tmp / "config.json").write_text(json.dumps(self.model.cfg.model_dump(), indent=1))
        meta = {"name": name, "note": note, "time": time.time(), "version": self.state["version"], "cycle": self.state["iter"],
                "phase": self.state["phase"], "models": {"policy": "CAM-v2 (RSSM world model + off-policy actor/critics; no Qwen)",
                                                          "policy_revision": None, "small": None, "cam_params": self.model.n_params()},
                "git": git_commit(), "architecture": "agentrs.cam.v2.model.CAM2", "learner": self.cfg.learner,
                "dream": self.cfg.dream}
        jdump(tmp / "meta.json", meta)
        jdump(tmp / "state.json", self.state)
        (tmp / "DONE").write_text(str(time.time()))
        final = self.ck / name
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        if name == "latest":
            self.state["latest_checkpoint"] = {"path": str(final), "time": time.time(), "version": self.state["version"]}

    def load(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path / "weights.pt", map_location=self.dev))
        self.learner.load_state_dict(torch.load(path / "optimizer.pt", map_location=self.dev))
        r = torch.load(path / "rng.pt", weights_only=False)
        torch.set_rng_state(r["torch"])
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        if self.dreamer and r.get("dream"):
            for k, v in r["dream"].items():
                setattr(self.dreamer, k, v)
        if self.archive is not None and (path / "archive.pkl").exists():
            with open(path / "archive.pkl", "rb") as f:
                d = pickle.load(f)
            self.archive.cells, self.archive.min_used, self.archive.stats = d["cells"], d["min_used"], d["stats"]

    def time_left(self) -> float:
        return self.state["deadline"] - time.time()

    def status(self, stage: str) -> None:
        s = self.state
        jdump(self.dir / "trainer_status.json", {"heartbeat": time.time(), "pid": os.getpid(), "phase": s["phase"], "stage": stage,
              "iter": s["iter"], "version": s["version"], "counters": s["counters"], "best": s["best"],
              "last_iter": s["iters"][-1] if s["iters"] else None, "last_eval": s["evals"][-1] if s["evals"] else None,
              "gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None})

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, tag: str, split: str, families: list[str], n: int, wm: bool = False) -> dict:
        self._sync()
        t0 = time.time()
        col = Collector(self.model, min(self.cfg.workers, n * len(families)), {f: 1.0 for f in families}, split,
                        seed0=1_000_000 if split == "val" else 0, gen_seed=12345, reward=self.rc, lam=self.state["lagrange"],
                        env_version=ENV_VERSION)
        fixed = [(f, (1_000_000 if split == "val" else 0) + i) for f in families for i in range(n)]
        eps = col.collect(greedy=True, version=f"v{self.state['version']}", fixed=fixed)
        rec = {"tag": tag, "split": split, "iter": self.state["iter"], "version": self.state["version"], "time": time.time(),
               "by_family": summarize(eps), "env_interactions_so_far": self.state["counters"]["transitions"],
               "compute_s_so_far": dict(self.state["compute_s"])}
        if wm:
            cal = open_loop(self.model, eps, self.state["lagrange"])
            rec["world_model"] = cal
            if self.dreamer:
                self.dreamer.set_reliability(cal)
                rec["dream_gate"] = {"reliability": self.dreamer.reliability, "u_thr": self.dreamer.u_thr,
                                     "max_h": self.dreamer.max_h}
        self._sync()
        rec["wall_s"] = round(time.time() - t0, 1)
        self.state["compute_s"]["eval"] += rec["wall_s"]
        self.state["evals"].append(rec)
        self.save_state()
        self.log(f"eval {tag}/{split}: " + ", ".join(
            f"{k} {v['success']}/{v['n']}={v['rate']:.3f} prog {v['final_progress']:.2f} viol {v['violation_rate']:.2f}"
            for k, v in rec["by_family"].items())
                 + (f" | wm ret_err k1 {rec['world_model']['ret_abs_err_k1']:.3f} k4 {rec['world_model']['ret_abs_err_k4']:.3f}"
                    f" u~err {rec['world_model']['u_err_spearman']:.2f}" if wm else ""))
        return rec

    def val_score(self, rec) -> float:
        return float(np.mean([v["rate"] + 0.25 * v["final_progress"] for k, v in rec["by_family"].items()
                              if k in self.cfg.train_families]))

    # ------------------------------------------------------------------ loop
    def train_iter(self) -> None:
        c = self.cfg
        it = self.state["iter"] + 1
        ver = f"v{self.state['version']}"
        cnt = self.state["counters"]
        self.status("collect")
        self._sync()
        t0 = time.time()
        self.collector.stats = {"env_s": 0.0, "net_s": 0.0, "steps": 0, "rounds": 0}
        eps = self.collector.collect(min_steps=c.steps_per_iter, version=ver)
        self._sync()
        t1 = time.time()
        self.state["compute_s"]["collect"] += t1 - t0
        self.replay.add(eps, it)
        new_steps = sum(e.n_learn for e in eps)
        prev_tr = cnt["transitions"]
        cnt["episodes"] += len(eps)
        cnt["successes"] += sum(e.success for e in eps)
        cnt["transitions"] += new_steps
        for f, v in summarize(eps).items():
            d = cnt["by_family"].setdefault(f, {"episodes": 0, "successes": 0})
            d["episodes"] += v["n"]
            d["successes"] += v["success"]
        # first verified successes (real interactions needed to discover a successful strategy)
        fs = self.state["first_success"]
        for e in eps:
            full = e.start == "scratch" and e.difficulty >= 1.0
            for key, ok in ((f"{e.family}:any", True), (f"{e.family}:full_goal_from_scratch", full)):
                if e.success and ok and key not in fs:
                    fs[key] = {"iter": it, "env_interactions": prev_tr + new_steps, "episodes": cnt["episodes"]}
                    self.log(f"first verified success {key} after {prev_tr + new_steps} real interactions")
        # ---- off-policy learning from replay
        self.status("learn")
        lst: dict = {}
        if self.replay.steps() >= c.learn_start_steps:
            for _ in range(c.updates_per_iter):
                views = self.replay.sample(c.batch_episodes, self.rng, c.her_p, c.prot_frac)
                st = self.learner.update(views, self.state["lagrange"], self.rng)
                for k, v in st.items():
                    lst.setdefault(k, []).append(v)
                cnt["grad_updates"] += 1
        self._sync()
        t2 = time.time()
        self.state["compute_s"]["learn"] += t2 - t1
        # ---- imagination
        dst: dict = {}
        if self.dreamer and self.replay.steps() >= c.learn_start_steps and it >= c.dream.get("start_iter", 5):
            for _ in range(c.dream.get("updates_per_iter", 16)):
                d = self.dream_step()
                for k, v in d.items():
                    dst.setdefault(k, []).append(v)
                cnt["dream_updates"] += 1
                cnt["imagined_transitions"] += d.get("imagined_transitions", 0)
        self._sync()
        t3 = time.time()
        self.state["compute_s"]["dream"] += t3 - t2
        # ---- constraint multipliers, curricula
        tr = summarize(eps)
        lam = self.state["lagrange"]
        if c.lagrangian:
            for f, v in tr.items():
                from agentrs.cam.envs import ENVS
                if ENVS[f]().contract().constraints:
                    lam[f] = float(np.clip(lam.get(f, c.lagrangian.get("init", 0.0))
                                           + c.lagrangian.get("lr", 0.5) * (v["violation_rate"] - c.lagrangian.get("kappa", 0.05)),
                                           0.0, c.lagrangian.get("max", 2.0)))
        cur = self.state["curriculum"]
        used = dict(cur)
        for fam, cc in c.curriculum.items():
            sc = [e for e in eps if e.family == fam and e.start == "scratch"]
            if len(sc) >= 20:
                rate = float(np.mean([e.success for e in sc]))
                if rate > cc.get("hi", 0.5):
                    cur[fam] = min(1.0, cur[fam] * cc.get("up", 1.2))
                elif rate < cc.get("lo", 0.15):
                    cur[fam] = max(cc.get("min", 0.02), cur[fam] * cc.get("down", 0.85))
        if self.archive is not None:
            for fam in c.train_families:
                ae = [e for e in eps if e.family == fam and e.start == "archive"]
                if len(ae) >= 10:
                    r = float(np.mean([e.success for e in ae]))
                    mu = self.archive.min_used.get(fam, 0.5)
                    self.archive.min_used[fam] = max(0.0, mu - 0.1) if r > 0.3 else (min(0.9, mu + 0.05) if r < 0.05 else mu)
        self.state["version"] += 1
        self.state["iter"] = it
        rec = {"iter": it, "time": time.time(), "version_collected": ver, "train": tr, "new_real_steps": new_steps,
               "learn": {k: float(np.mean(v)) for k, v in lst.items()}, "dream": {k: float(np.mean(v)) for k, v in dst.items()},
               "lagrange": dict(lam), "curriculum_used": used, "curriculum_next": dict(cur),
               "archive": {"size": self.archive.size(), "min_used": dict(self.archive.min_used), **self.archive.stats}
               if self.archive else None, "replay": self.replay.stats(),
               "collect_s": round(t1 - t0, 1), "learn_s": round(t2 - t1, 1), "dream_s": round(t3 - t2, 1),
               "steps_per_s": self.collector.stats["steps"] / max(1e-9, t1 - t0)}
        self.state["iters"].append(rec)
        self.state["iters"] = self.state["iters"][-300:]
        with open(self.dir / "iters.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        L = rec["learn"]
        self.log(f"iter {it}: " + ", ".join(f"{k} {v['rate']:.3f} (prog {v['final_progress']:.2f}, viol {v['violation_rate']:.2f}, "
                                            f"arch {v['from_archive']})" for k, v in tr.items())
                 + (f" | q {L.get('q_loss', 0):.4f} v {L.get('v_loss', 0):.4f} adv {L.get('adv_mean', 0):+.3f} ent {L.get('entropy', 0):.2f}"
                    f" wm(kl {L.get('wm_kl_dyn', 0):.2f} rew {L.get('wm_rew', 0):.4f} ens {L.get('wm_ens', 0):.3f})" if L else " | (warm-up)")
                 + (f" | dream len {rec['dream'].get('dream_len_mean', 0):.2f} kappa {rec['dream'].get('dream_kappa', 0):.2f}" if dst else "")
                 + f" | lam {json.dumps({k: round(v, 2) for k, v in lam.items()})} diff {json.dumps({k: round(v, 3) for k, v in cur.items()})}"
                 + f" | replay {rec['replay']['steps']} reuse {rec['replay']['replay_ratio']:.1f}"
                 + f" | collect {rec['collect_s']}s learn {rec['learn_s']}s dream {rec['dream_s']}s")
        self.version_tag = ver
        self.save("latest")
        if time.time() - self.state.get("last_snapshot", 0) > c.snapshot_every_min * 60:
            name = "snap_" + time.strftime("%H%M")
            self.save(name, "periodic snapshot")
            self.state["snapshots"].append(name)
            self.state["last_snapshot"] = time.time()
            while len(self.state["snapshots"]) > 3:
                shutil.rmtree(self.ck / self.state["snapshots"].pop(0), ignore_errors=True)
        if it % c.eval_every == 0:
            self.gate()
        self.save_state()

    def dream_step(self) -> dict:
        c = self.cfg
        m = self.model
        views = self.replay.sample(c.dream.get("batch_episodes", 32), self.rng, 0.0, c.prot_frac)
        with torch.no_grad():
            ob = observe(m, views)
        cand = [(i, t) for i, v in enumerate(views) for t in range(len(v.steps)) if v.steps[t].learn]
        self.rng.shuffle(cand)
        cand = cand[:c.dream.get("starts", 256)]
        k = torch.tensor([ob["pos"][x] for x in cand], device=m.device)
        f = ob["feat"][k]
        gi = torch.tensor([i for i, _ in cand], device=m.device)
        steps = [views[i].steps[t] for i, t in cand]
        return self.dreamer.update(self.learner.opt["actor"], self.learner.opt["critic"], f[:, :m.cfg.deter], f[:, m.cfg.deter:],
                                   ob["goal"][gi], [views[i].ep.family for i, _ in cand],
                                   torch.tensor([s.prev_opt for s in steps], device=m.device),
                                   torch.tensor([s.first for s in steps], device=m.device), self.state["lagrange"])

    def gate(self) -> None:
        rec = self.evaluate(f"iter{self.state['iter']}", "val", list(self.cfg.train_families), self.cfg.eval_episodes, wm=True)
        if self.cfg.monitor_families:
            self.evaluate(f"iter{self.state['iter']}-monitor", "val", list(self.cfg.monitor_families), self.cfg.eval_episodes)
        score = self.val_score(rec)
        best = self.state["best"]
        if best is None or score >= best["score"]:
            self.state["best"] = {"version": self.state["version"], "cycle": self.state["iter"], "score": score, "ckpt": "best"}
            self.save("best", f"best validation at iteration {self.state['iter']}")
            self.log(f"new best validation score {score:.4f}")
        elif self.dreamer and score < best["score"] - 0.1:
            # imagined policy improvement failed real validation: fall back towards real-experience learning
            self.dreamer.c["kappa"] = self.dreamer.c.get("kappa", 1.0) * 0.5
            self.state["dream_fallbacks"].append({"iter": self.state["iter"], "score": score, "best": best["score"],
                                                  "kappa": self.dreamer.c["kappa"]})
            self.log(f"validation dropped ({score:.3f} vs best {best['score']:.3f}): imagination weight halved to "
                     f"{self.dreamer.c['kappa']:.3f}")

    def budget_left(self) -> bool:
        c = self.cfg
        if c.max_iters and self.state["iter"] >= c.max_iters:
            return False
        if c.max_env_steps and self.state["counters"]["transitions"] >= c.max_env_steps:
            return False
        return True

    def run(self) -> int:
        c = self.cfg
        if self.state["phase"] == "train":
            if not any(e["tag"] == "untrained" for e in self.state["evals"]):
                self.evaluate("untrained", "val", list(c.train_families), c.eval_episodes)
                self.save("initial", "untrained CAM-v2")
            while not self.stop and self.time_left() > c.final_reserve_h * 3600 and not (self.dir / "STOP").exists() \
                    and self.budget_left():
                self.train_iter()
            if self.stop:
                self.log("SIGTERM: checkpointed, exiting (resumes on restart)")
                self.save("latest")
                self.save_state()
                return 3
            self.state["phase"] = "final"
            self.save_state()
        if self.state["phase"] == "final":
            self.save("final", "end of training")
            done = {(e["tag"], e["split"]) for e in self.state["evals"]}
            fams = list(dict.fromkeys(list(c.train_families) + list(c.heldout_families) + list(c.monitor_families)))
            if ("final", "test") not in done:
                self.evaluate("final", "test", fams, c.test_episodes)
            if (self.ck / "best" / "DONE").exists() and ("best", "test") not in done:
                self.model.load_state_dict(torch.load(self.ck / "best" / "weights.pt", map_location=self.dev))
                self.evaluate("best", "test", fams, c.test_episodes)
            if "counterfactual" not in self.state:
                self._sync()
                t0 = time.time()
                cf = {f: counterfactual_ranking(self.model, f, c.cf_states, 4, self.state["lagrange"].get(f, 0.0), rc=self.rc)
                      for f in fams}
                val = [e for e in self.state["evals"] if e["split"] == "val" and e.get("world_model")]
                col = Collector(self.model, 64, {f: 1.0 for f in c.train_families}, "val", seed0=3_000_000, gen_seed=7,
                                reward=self.rc, lam=self.state["lagrange"])
                eps = col.collect(greedy=True, fixed=[(f, 3_000_000 + i) for f in c.train_families for i in range(64)])
                clean = open_loop(self.model, eps, self.state["lagrange"])
                col_t = Collector(self.model, 64, {f: 1.0 for f in c.train_families}, "test", seed0=0, gen_seed=7,
                                  reward=self.rc, lam=self.state["lagrange"])
                ood = open_loop(self.model, col_t.collect(greedy=True, fixed=[(f, 5_000 + i) for f in c.train_families
                                                                             for i in range(64)]), self.state["lagrange"])
                bad_model = corrupted(self.model, 0.5)
                bad = open_loop(bad_model, eps, self.state["lagrange"])
                cf_bad = counterfactual_ranking(bad_model, "product", max(10, c.cf_states // 3), 4,
                                                self.state["lagrange"].get("product", 0.0), rc=self.rc) if "product" in fams else None
                probe = Dreamer(self.model, {"mode": "adaptive", "horizon": 8})
                probe.set_reliability(clean)
                probe_bad = Dreamer(bad_model, {"mode": "adaptive", "horizon": 8})
                probe_bad.set_reliability(bad)
                self.state["counterfactual"] = cf
                self.state["wm_validation_final"] = {"clean": clean, "heldout_distribution": ood, "corrupted": bad, "corrupted_counterfactual_product": cf_bad,
                                                     "gate_clean": {"reliability": probe.reliability, "max_h": probe.max_h},
                                                     "gate_corrupted": {"reliability": probe_bad.reliability, "max_h": probe_bad.max_h},
                                                     "n_val_evals_with_wm": len(val)}
                self._sync()
                self.state["compute_s"]["eval"] += time.time() - t0
                self.log("counterfactual ranking: " + "; ".join(f"{f} spearman {v['spearman']:.2f} top1 {v['top1_agreement']:.2f} "
                                                               f"(chance {v['chance_top1']:.2f})" for f, v in cf.items())
                         + f" | gate clean {probe.reliability:.2f} corrupted {probe_bad.reliability:.2f}")
            self.state["phase"] = "done"
            self.state["finished_at"] = time.time()
            self.save_state()
            self.status("done")
            self.log("CAM-v2 run complete")
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    d = yaml.safe_load(Path(a.config).read_text()) if a.config else {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        d[k] = yaml.safe_load(v)
    cfg = TrainConfig2(**d)
    sp = Path(cfg.run_dir) / "state.json"
    if sp.exists() and json.loads(sp.read_text()).get("phase") == "done":
        print("run already complete", flush=True)
        return 0
    return Trainer2(cfg).run()


if __name__ == "__main__":
    raise SystemExit(main())
