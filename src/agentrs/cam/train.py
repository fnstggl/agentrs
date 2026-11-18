"""Continual CAM training (``python -m agentrs.cam.train --config configs/cam/a6000.yaml``).

Loop: persistent workers collect complete episodes with the current CAM version (one batched core) →
PPO + goal-value + world-model updates of all CAM parameters → every ``eval_every`` iterations a
greedy validation (fresh hidden parameters, training distribution) decides promotion (best kept,
regression rolls back) and the world model is evaluated on held-out validation transitions → local
checkpoints (``latest`` every iteration, ``snap_*`` every 30 min, ``best``) with optimizer, RNG and run
state. At the end: sealed test on held-out parameter distributions (unseen counterparty policies /
markets), on the never-trained family (zero-shot), and direct vs planning-assisted acting.
Phases and counters persist in ``<run>/state.json``; a restarted process resumes from ``latest``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from pydantic import BaseModel

from agentrs.cam.learner import eval_world_model, update
from agentrs.cam.model import CAM, CAMConfig
from agentrs.cam.runtime import Workers, summarize


class TrainConfig(BaseModel):
    run_dir: str = "runs/cam"
    hours: float = 6.0
    final_reserve_h: float = 0.25
    train_families: dict = {"negotiation": 0.5, "product": 0.5}
    heldout_families: list = ["bizdev"]
    model: dict = {}
    workers: int = 256
    steps_per_iter: int = 4096
    lr: float = 1e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 2
    mb_episodes: int = 128
    ent_coef: float = 0.01
    v_coef: float = 0.5
    wm_coef: float = 1.0           # 0 = ablation without predictive world-model learning
    wm_sample: int = 4096
    grad_clip: float = 1.0
    target_kl: float = 0.05
    eval_every: int = 10
    eval_episodes: int = 200       # per family
    test_episodes: int = 400
    max_iters: int = 0
    snapshot_every_min: float = 30.0
    seed: int = 0
    gpu_mem_fraction: float = 0.15
    init_from: str = ""            # transfer experiments: start from another run's checkpoint
    planning_k: int = 4


def git_commit() -> dict:
    root = Path(__file__).resolve().parents[3]
    g = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(g("status", "--porcelain"))}


def jdump(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, p)


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.dir = Path(cfg.run_dir)
        self.ck = self.dir / "ckpt"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stop = False
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(cfg.gpu_mem_fraction)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CAM(CAMConfig(**cfg.model)).to(self.dev)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=0.0)
        sp = self.dir / "state.json"
        if sp.exists() and (self.ck / "latest" / "DONE").exists():
            self.state = json.loads(sp.read_text())
            self.load(self.ck / "latest")
            self.state["restarts"] = self.state.get("restarts", 0) + 1
            self.log(f"resumed from ckpt/latest (iteration {self.state['iter']}, version {self.state['version']})")
        else:
            t0 = time.time()
            self.state = {"started_at": t0, "deadline": t0 + cfg.hours * 3600, "phase": "train", "iter": 0, "version": 0,
                          "config": cfg.model_dump(), "git": git_commit(), "counters": {"episodes": 0, "successes": 0,
                          "transitions": 0, "updates": 0, "by_family": {}}, "evals": [], "iters": [], "best": None,
                          "promoted": None, "restarts": 0, "snapshots": [], "last_snapshot": t0,
                          "models": {"cam": self.model.cfg.model_dump(exclude={"vocab"}), "qwen": None,
                                     "params": self.model.n_params()}}
            if cfg.init_from:
                sd = torch.load(Path(cfg.init_from) / "weights.pt", map_location=self.dev)
                self.model.load_state_dict(sd)
                self.state["init_from"] = cfg.init_from
                self.log(f"initialised CAM weights from {cfg.init_from}")
            self.save_state()
            self.log(f"new CAM run: {self.model.n_params()}")
        self.workers = Workers(self.model, cfg.workers, cfg.train_families, "train", seed0=self.state["iter"] * 100_000,
                               gamma=cfg.gamma, gen_seed=cfg.seed + self.state["iter"])
        self.rng = random.Random(cfg.seed + 17 * self.state["iter"])

    # ------------------------------------------------------------------ infra
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.dir / "train.log", "a") as f:
            f.write(line + "\n")

    def save_state(self) -> None:
        jdump(self.dir / "state.json", self.state)

    def save(self, name: str, note: str = "") -> None:
        tmp = self.ck / (name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        torch.save(self.model.state_dict(), tmp / "weights.pt")
        torch.save({"optimizer": self.opt.state_dict()}, tmp / "optimizer.pt")
        torch.save({"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "python": random.getstate(), "numpy": np.random.get_state(), "worker_gen": self.workers.gen.get_state()
                    if hasattr(self, "workers") else None, "seed_ctr": getattr(getattr(self, "workers", None), "seed_ctr", None)},
                   tmp / "rng.pt")
        (tmp / "config.json").write_text(json.dumps(self.model.cfg.model_dump(), indent=1))
        meta = {"name": name, "note": note, "time": time.time(), "version": self.state["version"], "cycle": self.state["iter"],
                "phase": self.state["phase"], "models": {"policy": "CAM (no Qwen)", "policy_revision": None,
                                                          "small": None, "cam_params": self.model.n_params()},
                "git": git_commit(), "rl_lr": self.cfg.lr, "architecture": "agentrs.cam.model.CAM"}
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
        self.opt.load_state_dict(torch.load(path / "optimizer.pt", map_location=self.dev)["optimizer"])
        r = torch.load(path / "rng.pt", weights_only=False)
        torch.set_rng_state(r["torch"])
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])

    def time_left(self) -> float:
        return self.state["deadline"] - time.time()

    def status(self, stage: str, **kw) -> None:
        s = self.state
        jdump(self.dir / "trainer_status.json", {"heartbeat": time.time(), "pid": os.getpid(), "phase": s["phase"],
              "stage": stage, "iter": s["iter"], "version": s["version"], "counters": s["counters"], "best": s["best"],
              "last_iter": s["iters"][-1] if s["iters"] else None, "last_eval": s["evals"][-1] if s["evals"] else None,
              "gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None, **kw})

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, tag: str, split: str, families: list[str], n: int, planning: int = 0, wm: bool = False) -> dict:
        t0 = time.time()
        w = Workers(self.model, min(self.cfg.workers, n * len(families)), {f: 1.0 for f in families}, split,
                    seed0=1_000_000 if split == "val" else 0, gen_seed=12345)
        fixed = [(f, (1_000_000 if split == "val" else 0) + i) for f in families for i in range(n)]
        eps = w.collect(greedy=True, version=f"v{self.state['version']}", planning=planning, fixed=fixed)
        rec = {"tag": tag, "split": split, "planning": planning, "iter": self.state["iter"], "version": self.state["version"],
               "time": time.time(), "wall_s": round(time.time() - t0, 1), "by_family": summarize(eps),
               "env_interactions_so_far": self.state["counters"]["transitions"]}
        if wm:
            rec["world_model"] = eval_world_model(self.model, eps, random.Random(0))
        self.state["evals"].append(rec)
        self.save_state()
        self.log(f"eval {tag}/{split}{' +plan' if planning else ''}: " + ", ".join(
            f"{k} {v['success']}/{v['n']}={v['rate']:.3f} prog {v['final_progress']:.2f}" for k, v in rec["by_family"].items()))
        return rec

    def val_score(self, rec) -> float:
        return float(np.mean([v["rate"] + 0.25 * v["final_progress"] for k, v in rec["by_family"].items()
                              if k in self.cfg.train_families]))

    # ------------------------------------------------------------------ loop
    def train_iter(self) -> None:
        c = self.cfg
        it = self.state["iter"] + 1
        ver = f"v{self.state['version']}"
        t0 = time.time()
        self.status("collect")
        self.workers.stats = {"env_s": 0.0, "net_s": 0.0, "steps": 0, "rounds": 0, "batch": []}
        eps = self.workers.collect(min_steps=c.steps_per_iter, version=ver)
        t1 = time.time()
        snap = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        st = update(self.model, self.opt, eps, ver, {"gamma": c.gamma, "lam": c.lam, "clip": c.clip, "epochs": c.epochs,
                                                     "mb_episodes": c.mb_episodes, "ent_coef": c.ent_coef, "v_coef": c.v_coef,
                                                     "wm_coef": c.wm_coef, "wm_sample": c.wm_sample, "grad_clip": c.grad_clip,
                                                     "target_kl": c.target_kl}, self.rng)
        change = {}
        for n, p in self.model.named_parameters():
            k = n.split(".")[0]
            change[k] = change.get(k, 0.0) + float((p.detach() - snap[n]).pow(2).sum())
        change = {k: math.sqrt(v) for k, v in change.items()}
        del snap
        self.state["version"] += 1
        self.state["iter"] = it
        cnt = self.state["counters"]
        cnt["episodes"] += len(eps)
        cnt["successes"] += sum(e.success for e in eps)
        cnt["transitions"] += sum(len(e.steps) for e in eps)
        cnt["updates"] += st["updates"]
        for f, v in summarize(eps).items():
            d = cnt["by_family"].setdefault(f, {"episodes": 0, "successes": 0})
            d["episodes"] += v["n"]
            d["successes"] += v["success"]
        ws = self.workers.stats
        rec = {"iter": it, "time": time.time(), "version_collected": ver, "train": summarize(eps), "update": st,
               "param_change_l2": change, "collect_s": round(t1 - t0, 1), "update_s": round(time.time() - t1, 1),
               "env_s": round(ws["env_s"], 1), "net_s": round(ws["net_s"], 1),
               "mean_batch": float(np.mean(ws["batch"])) if ws["batch"] else 0, "steps_per_s": ws["steps"] / max(1e-9, t1 - t0)}
        self.state["iters"].append(rec)
        self.state["iters"] = self.state["iters"][-400:]
        with open(self.dir / "iters.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        tr = rec["train"]
        self.log(f"iter {it}: " + ", ".join(f"{k} {v['rate']:.3f} (prog {v['final_progress']:.2f}, valid {v['valid_rate']:.2f}, "
                                            f"info {v['info_action_rate']:.2f})" for k, v in tr.items())
                 + f" | pg {st['pg']:.3f} ent {st['entropy']:.2f} kl {st['kl']:.4f} mism {st['mismatch']:.1e} "
                 f"wm(f {st['feat_nll']:.2f} ev {st['event_bce']:.3f} z {st['latent_cos']:.3f}) upd {st['updates']} "
                 f"| {rec['steps_per_s']:.0f} steps/s collect {rec['collect_s']}s update {rec['update_s']}s")
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

    def gate(self) -> None:
        rec = self.evaluate(f"iter{self.state['iter']}", "val", list(self.cfg.train_families), self.cfg.eval_episodes, wm=True)
        score = self.val_score(rec)
        best = self.state["best"]
        if best is None or score >= best["score"]:
            self.state["best"] = {"version": self.state["version"], "cycle": self.state["iter"], "score": score, "ckpt": "best"}
            self.save("best", f"best validation at iteration {self.state['iter']}")
            self.log(f"new best validation score {score:.4f}")
        elif score < best["score"] - 0.15:
            self.log(f"validation regressed ({score:.3f} vs best {best['score']:.3f}): rolling back to best weights")
            self.model.load_state_dict(torch.load(self.ck / "best" / "weights.pt", map_location=self.dev))
            self.state["version"] += 1
            self.state.setdefault("rollbacks", []).append({"iter": self.state["iter"], "score": score})

    def run(self) -> int:
        c = self.cfg
        if self.state["phase"] == "train":
            if not any(e["tag"] == "untrained" for e in self.state["evals"]):
                self.evaluate("untrained", "val", list(c.train_families), c.eval_episodes, wm=True)
                self.evaluate("untrained", "test", list(c.train_families) + list(c.heldout_families), c.test_episodes)
                self.save("initial", "untrained CAM")
            while not self.stop and self.time_left() > c.final_reserve_h * 3600 and not (self.dir / "STOP").exists() \
                    and not (c.max_iters and self.state["iter"] >= c.max_iters):
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
            done = {(e["tag"], e["split"], e["planning"]) for e in self.state["evals"]}
            fams = list(c.train_families) + list(c.heldout_families)
            if ("final", "test", 0) not in done:
                self.evaluate("final", "test", fams, c.test_episodes, wm=True)
            if (self.ck / "best" / "DONE").exists() and ("best", "test", 0) not in done:
                self.model.load_state_dict(torch.load(self.ck / "best" / "weights.pt", map_location=self.dev))
                self.evaluate("best", "test", fams, c.test_episodes)
                if c.planning_k and ("best", "test", c.planning_k) not in done:
                    self.evaluate("best", "test", list(c.train_families), c.test_episodes, planning=c.planning_k)
            self.state["phase"] = "done"
            self.state["finished_at"] = time.time()
            self.save_state()
            self.status("done")
            self.log("CAM run complete")
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
    cfg = TrainConfig(**d)
    sp = Path(cfg.run_dir) / "state.json"
    if sp.exists() and json.loads(sp.read_text()).get("phase") == "done":
        print("run already complete", flush=True)
        return 0
    return Trainer(cfg).run()


if __name__ == "__main__":
    raise SystemExit(main())
