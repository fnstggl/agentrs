"""The training loop: experience -> trajectory store -> learning update -> changed policy -> checkpoint.

    Trainer(cfg, run_dir).run()

One iteration: the curriculum samples tasks; rollout workers run the current controller on them; the episodes'
new text embeddings, memory records and tools are merged; the algorithm updates the weights; the curriculum is
updated from the evaluator's summaries; metrics are appended to ``metrics.jsonl``; every `eval_every` iterations
the controller is evaluated greedily on tasks of the evaluation split (never used for training); every
`ckpt_every` iterations a checkpoint is written. `resume` continues from ``ckpt/latest.pt`` with the same
counters, optimizer state, curriculum, memory and random state.

Run directory:
    config.json          the full configuration                    meta.json     code hashes, git commit, hardware
    metrics.jsonl        one line per iteration (and per phase)     episodes.jsonl one line per episode summary
    traj/                readable trajectories of sampled episodes  ckpt/         checkpoints
    demos.pkl            demonstration episodes with provenance     eval.jsonl    evaluation results
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import pickle
import subprocess
import time

import numpy as np
import torch

from ..algo import base as algo_base
from ..algo.ppo import bc_update, flat_params
from ..cam.memory import EpisodicMemory
from ..cam.model import CAMController, EmbStore, ModelConfig
from ..grounding.toolsmith import ToolLibrary
from ..world.tasks import SPLITS, TaskSpec, task, task_list
from .curriculum import Curriculum
from .rollout import LocalPool, RolloutPool

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
DEFAULT_CFG = {
    "name": "run", "seed": 0, "device": "auto", "workers": 16, "iters": 50, "episodes_per_iter": 96,
    "eval_every": 5, "eval_episodes": 48, "eval_split": "curriculum", "ckpt_every": 5, "traj_per_iter": 2,
    "llm": {"kind": "none"}, "model": {}, "agent": {}, "core": {"max_decisions": 48},
    "algo": {"name": "ppo"},
    "curriculum": {"arms": None, "adaptive": True, "world_families": ["tiny", "bazaar"]},
    "demos": {"episodes": 0, "bc_epochs": 0, "bc_lr": 3e-4, "max_per_arm": 0},
    "memory": {"enabled": True}, "temperature": 1.0, "store_episodes_iters": 0,
}


def merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def tree_hash(paths: list[str]) -> str:
    h = hashlib.sha256()
    for root in paths:
        for dp, _, fns in sorted(os.walk(root)):
            for fn in sorted(fns):
                if fn.endswith(".py"):
                    p = os.path.join(dp, fn)
                    h.update(os.path.relpath(p, PKG).encode())
                    h.update(open(p, "rb").read())
    return h.hexdigest()[:16]


def code_hashes() -> dict:
    """Hashes of the candidate-editable learner code and of the protected evaluation code."""
    return {"learner": tree_hash([os.path.join(PKG, d) for d in ("cam", "algo")] + [os.path.join(PKG, "train")]),
            "protected": tree_hash([os.path.join(PKG, d) for d in ("evalkit", "world")]),
            "grounding": tree_hash([os.path.join(PKG, d) for d in ("grounding", "actions", "llm")])}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PKG, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - w) / d), min(1.0, (c + w) / d)


def summarize(eps: list[dict]) -> dict:
    """Aggregate evaluator summaries of a set of episodes."""
    s = [e["summary"] for e in eps if e is not None]
    n = len(s)
    if n == 0:
        return {"episodes": 0}
    k = sum(x["success"] for x in s)
    lo, hi = wilson(k, n)
    calls = sum(x["agent"]["calls"] for x in s)
    fails: dict[str, int] = {}
    for x in s:
        for c, v in x["agent"]["failures"].items():
            fails[c] = fails.get(c, 0) + v
    by: dict[str, list] = {}
    for x in s:
        by.setdefault("/".join(str(a) for a in x["arm"]), []).append(float(x["success"]))
    return {"episodes": n, "success": k / n, "success_ci": [round(lo, 4), round(hi, 4)],
            "progress": float(np.mean([x["best_progress"] for x in s])), "return": float(np.mean([x["return"] for x in s])),
            "decisions": int(sum(x["decisions"] for x in s)), "calls": int(calls),
            "valid_rate": (sum(x["agent"]["ok"] for x in s) / calls) if calls else 0.0, "failures": fails,
            "llm_propose": int(sum(x["llm"]["propose"] for x in s)), "llm_gen": int(sum(x["llm"]["gen"] for x in s)),
            "by_arm": {a: round(float(np.mean(v)), 3) for a, v in sorted(by.items())},
            "by_arm_n": {a: len(v) for a, v in sorted(by.items())}}


class Trainer:
    def __init__(self, cfg: dict, run_dir: str, resume: bool = False, pool=None):
        self.cfg = merge(DEFAULT_CFG, cfg)
        self.dir = run_dir
        for d in ("", "ckpt", "traj"):
            os.makedirs(os.path.join(run_dir, d), exist_ok=True)
        c = self.cfg
        dev = c["device"]
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if dev == "auto" else dev
        torch.manual_seed(c["seed"])
        self.rng = np.random.default_rng(c["seed"])
        self.model = CAMController(ModelConfig(**c["model"])).to(self.device)
        self.model.eval()
        self.store = EmbStore()
        self.algo = algo_base.make(c["algo"].get("name", "ppo"), self.model,
                                   dict({k: v for k, v in c["algo"].items() if k != "name"}, seed=c["seed"]), self.device)
        cc = dict(c["curriculum"])
        self.curriculum = Curriculum(cc.get("arms"), cc.get("world_families"), "train", c["seed"],
                                     adaptive=cc.get("adaptive", True), gate=cc.get("gate", True))
        self.memory = EpisodicMemory()
        self.library = ToolLibrary()
        self.mem_synced = 0
        self.iter = 0
        self.env_decisions = 0
        self.episodes = 0
        self.demos: list[dict] = []
        self.wall = {"rollout_s": 0.0, "update_s": 0.0, "eval_s": 0.0}
        self.llm_usage: dict = {}
        wcfg = {"llm": c["llm"], "model": c["model"], "agent": c["agent"], "core": c["core"],
                "env_socket": c.get("env_socket")}
        start = "fork" if (os.environ.get("SUPERLEARNER_ISOLATED") or self.device == "cpu") else "spawn"
        self.pool = pool or (RolloutPool(c["workers"], wcfg, start) if c["workers"] > 0 else LocalPool(wcfg))
        self._errors_logged = 0
        self.theta0 = flat_params(self.model).clone()
        if resume and os.path.exists(os.path.join(run_dir, "ckpt", "latest.pt")):
            self.load(os.path.join(run_dir, "ckpt", "latest.pt"))
        else:
            json.dump(self.cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
            json.dump({"code": code_hashes(), "git": git_commit(), "torch": torch.__version__, "device": self.device,
                       "params": self.model.n_params(), "started": time.time()},
                      open(os.path.join(run_dir, "meta.json"), "w"), indent=1)

    # ------------------------------------------------------------------ plumbing
    def log(self, name: str, rec: dict) -> None:
        with open(os.path.join(self.dir, name), "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def absorb(self, eps: list) -> list[dict]:
        """Merge what the workers learned outside the weights (embeddings, memory, tools) and log summaries."""
        good = [e for e in eps if e is not None]
        for e in good:
            for k, v in e.get("emb_new", []):
                self.store.put(k, v)
            for it in e.get("mem_new", []):
                if self.cfg["memory"]["enabled"]:
                    self.memory.items.append(it)
            lib = e.get("library")
            if lib:
                for name, t in lib["tools"].items():
                    old = self.library.tools.get(name)
                    if old is None or t["version"] > old["version"] or t["stats"]["calls"] > old["stats"]["calls"]:
                        self.library.tools[name] = t
            u = e.get("llm_usage") or {}
            self.llm_usage[e.get("wid", 0)] = u
        self.memory.items = self.memory.items[-self.memory.capacity:]
        self.memory._mat = None
        if len(self.pool.errors) > self._errors_logged:
            with open(os.path.join(self.dir, "errors.log"), "a") as f:
                for e in self.pool.errors[self._errors_logged:]:
                    f.write(f"--- iter {self.iter}\n{e}\n")
            self._errors_logged = len(self.pool.errors)
        return good

    def sync_workers(self) -> None:
        self.pool.set_weights(self.model.state_dict())
        new = self.memory.items[self.mem_synced:]
        self.pool.sync(new, self.library.state_dict(), reset=False)
        self.mem_synced = len(self.memory.items)

    def llm_totals(self) -> dict:
        t: dict[str, float] = {}
        for u in self.llm_usage.values():
            for k, v in u.items():
                if isinstance(v, (int, float)):
                    t[k] = t.get(k, 0) + v
        return t

    def save_traj(self, tag: str, eps: list[dict]) -> None:
        recs = []
        for e in eps:
            recs.append({"summary": e["summary"], "steps": [d.get("human") for d in e["decisions"] if d.get("human")],
                         "policy": {"iter": self.iter, "code": code_hashes()["learner"]}})
        if recs:
            with gzip.open(os.path.join(self.dir, "traj", f"{tag}.json.gz"), "wt") as f:
                json.dump(recs, f, default=str)

    # ------------------------------------------------------------------ demonstrations and warm start
    def collect_demos(self, n: int, arms: list | None = None) -> dict:
        """Run the scripted teachers on training tasks; keep only episodes the evaluator verified as successful."""
        self.sync_workers()
        cur = Curriculum(arms or self.cfg["curriculum"].get("arms"), self.cfg["curriculum"].get("world_families"),
                         "train", self.cfg["seed"] + 77, adaptive=False)
        cur.counter = 5_000_000        # demonstration worlds: a seed range of the training split not used by rollouts
        jobs = [{"spec": s.to_dict(), "mode": "teacher:auto", "seed": i, "episode_id": f"demo{i}", "learn": False,
                 "with_model": True, "tag": "demo"} for i, s in enumerate(cur.sample(n))]
        t0 = time.time()
        eps = self.absorb(self.pool.run(jobs))
        ok = [e for e in eps if e["summary"]["success"]]
        n_verified = len(ok)
        cap = int(self.cfg["demos"].get("max_per_arm") or 0)
        if cap:
            cnt: dict = {}
            kept = []
            for e in ok:
                a = tuple(e["summary"]["arm"])
                if cnt.get(a, 0) < cap:
                    cnt[a] = cnt.get(a, 0) + 1
                    kept.append(e)
            ok = kept
        self.demos = ok
        prov = {"teacher": "train/demos.py:script_auto", "requested": n, "ran": len(eps), "verified_successful": n_verified,
                "kept": len(ok), "max_per_arm": cap,
                "summary_all": summarize(eps), "by_arm_kept": summarize(ok).get("by_arm_n", {}),
                "misses": sum(len(e["summary"]["teacher"]["misses"]) for e in ok), "wall_s": round(time.time() - t0, 1),
                "tasks": [e["summary"]["task"] for e in ok]}
        pickle.dump({"episodes": ok, "provenance": prov, "emb": self.store.items_since(0)},
                    open(os.path.join(self.dir, "demos.pkl"), "wb"))
        self.log("metrics.jsonl", {"phase": "demos", **{k: v for k, v in prov.items() if k != "tasks"}})
        return prov

    def pretrain_bc(self, epochs: int) -> list[dict]:
        if not self.demos or epochs <= 0:
            return []
        opt = torch.optim.AdamW(self.model.parameters(), lr=float(self.cfg["demos"].get("bc_lr", 3e-4)))
        out = []
        for ep in range(epochs):
            m = bc_update(self.model, opt, self.demos, self.store, self.device,
                          int(self.cfg["demos"].get("bc_minibatch_eps") or self.cfg["algo"].get("minibatch_eps", 32)),
                          self.algo.rule, rng=self.rng)
            m.update(phase="bc", epoch=ep, demo_episodes=len(self.demos),
                     param_dist_from_init=float((flat_params(self.model) - self.theta0).norm()))
            self.log("metrics.jsonl", m)
            out.append(m)
        return out

    # ------------------------------------------------------------------ evaluation
    def eval_tasks(self, n: int, split: str | None = None, arms: list | None = None, offset: int = 0) -> list[TaskSpec]:
        """A fixed evaluation set: the same tasks every time it is asked for (so curves are comparable)."""
        split = split or self.cfg["eval_split"]
        fams = [f for f in (self.cfg.get("eval_world_families") or self.cfg["curriculum"].get("world_families")
                            or SPLITS[split][2]) if f in SPLITS[split][2]] or list(SPLITS[split][2])
        arms = arms or [a.key for a in self.curriculum.arms]
        return task_list(arms, fams, n, split, offset + 1000)

    def evaluate(self, tasks: list[TaskSpec], mode: str = "greedy", tag: str = "eval", agent: dict | None = None,
                 save: bool = True) -> dict:
        self.sync_workers()
        t0 = time.time()
        jobs = [{"spec": s.to_dict(), "mode": mode, "seed": 10_000 + i, "episode_id": f"{tag}{i}", "learn": False,
                 "keep_decisions": False, "human": i < 4, "agent": agent or {}, "tag": tag} for i, s in enumerate(tasks)]
        eps = self.absorb(self.pool.run(jobs))
        res = summarize(eps)
        res.update(mode=mode, tag=tag, iter=self.iter, env_decisions=self.env_decisions, wall_s=round(time.time() - t0, 1),
                   errors=len(self.pool.errors))
        self.wall["eval_s"] += time.time() - t0
        if self.device != "cpu":
            torch.cuda.empty_cache()
        if save:
            self.log("eval.jsonl", res)
            self.save_traj(f"{tag}_iter{self.iter:04d}", [e for e in eps if any("human" in d for d in e["decisions"])][:4])
        return res

    def evaluate_hub(self, tag: str = "eval") -> dict:
        """Evaluation when worlds live on the trusted env hub: tasks are started by index, never by spec."""
        n = int(self.cfg.get("hub_eval_n", 0))
        if n <= 0:
            return {}
        self.sync_workers()
        t0 = time.time()
        jobs = [{"eval": i, "mode": "greedy", "seed": 10_000 + i, "episode_id": f"{tag}{i}", "learn": False,
                 "keep_decisions": False, "tag": tag} for i in range(n)]
        res = summarize(self.absorb(self.pool.run(jobs)))
        res.update(mode="greedy", tag=tag, iter=self.iter, env_decisions=self.env_decisions, wall_s=round(time.time() - t0, 1))
        self.wall["eval_s"] += time.time() - t0
        self.log("eval.jsonl", res)
        return res

    # ------------------------------------------------------------------ training
    def train_iter(self) -> dict:
        c = self.cfg
        self.sync_workers()
        specs = self.curriculum.sample(c["episodes_per_iter"])
        t0 = time.time()
        jobs = [{"spec": s.to_dict(), "mode": "sample", "seed": int(self.rng.integers(0, 2**31 - 1)),
                 "episode_id": f"it{self.iter}e{i}", "learn": True, "temperature": c.get("temperature", 1.0),
                 "human": i < c["traj_per_iter"], "tag": "train"} for i, s in enumerate(specs)]
        eps = self.absorb(self.pool.run(jobs))
        t_roll = time.time() - t0
        for e in eps:
            self.curriculum.update(e["summary"], float(np.mean([d["vstd"] for d in e["decisions"]])) if e["decisions"] else 0.0)
            self.log("episodes.jsonl", dict(e["summary"], iter=self.iter))
        self.save_traj(f"train_iter{self.iter:04d}", [e for e in eps if "human" in e["decisions"][0]][: c["traj_per_iter"]])
        keep = int(c.get("store_episodes_iters", 0))
        if keep and eps:       # training-format trajectory store: the last `keep` iterations, with every decision record
            os.makedirs(os.path.join(self.dir, "replay"), exist_ok=True)
            with gzip.open(os.path.join(self.dir, "replay", f"iter{self.iter:04d}.pkl.gz"), "wb") as f:
                pickle.dump({"episodes": [{"summary": e["summary"], "decisions": e["decisions"]} for e in eps],
                             "emb": self.store.items_since(0), "iter": self.iter}, f)
            old_p = os.path.join(self.dir, "replay", f"iter{self.iter - keep:04d}.pkl.gz")
            if os.path.exists(old_p):
                os.unlink(old_p)
        t0 = time.time()
        before = flat_params(self.model).clone()
        m = self.safe_update(eps) if eps else {}
        t_upd = time.time() - t0
        self.iter += 1
        self.env_decisions += sum(len(e["decisions"]) for e in eps)
        self.episodes += len(eps)
        self.wall["rollout_s"] += t_roll
        self.wall["update_s"] += t_upd
        rec = {"phase": "rl", "iter": self.iter, "env_decisions": self.env_decisions, "episodes": self.episodes,
               "rollout_s": round(t_roll, 2), "update_s": round(t_upd, 2), **{f"train_{k}": v for k, v in summarize(eps).items()},
               **m, "param_dist_from_init": float((flat_params(self.model) - self.theta0).norm()),
               "weights_changed": bool((flat_params(self.model) != before).any()),
               "curriculum": self.curriculum.table(), "memory": len(self.memory), "tools": len(self.library.tools),
               "llm": self.llm_totals(), "worker_errors": len(self.pool.errors)}
        self.log("metrics.jsonl", rec)
        return rec

    def init_from(self, path: str) -> None:
        """Start from another run's warm-start checkpoint (weights, frozen-embedding store, memory, demonstrations)
        with fresh counters and a fresh optimiser, so that several runs can branch from the SAME warm start."""
        self.load(path, optimizer=False)
        self.iter = self.env_decisions = self.episodes = 0
        self.wall = {"rollout_s": 0.0, "update_s": 0.0, "eval_s": 0.0}
        from .curriculum import Curriculum
        cc = dict(self.cfg["curriculum"])
        self.curriculum = Curriculum(cc.get("arms"), cc.get("world_families"), "train", self.cfg["seed"],
                                     adaptive=cc.get("adaptive", True), gate=cc.get("gate", True))
        self.rng = np.random.default_rng(self.cfg["seed"])
        demos = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(path))), "demos.pkl")
        if os.path.exists(demos):
            d = pickle.load(open(demos, "rb"))
            self.demos = d["episodes"]
            for k, v in d.get("emb", []):
                self.store.put(k, v)
            pickle.dump(d, open(os.path.join(self.dir, "demos.pkl"), "wb"))
        self.log("metrics.jsonl", {"phase": "init_from", "path": path, "demo_episodes": len(self.demos)})
        self.save("bc.pt")

    def safe_update(self, eps: list[dict]) -> dict:
        """`algo.update`, surviving a GPU out-of-memory error caused by OTHER processes on a shared GPU: the weights
        and optimiser are restored to their state before the update, cached GPU memory is released, and the same
        update is tried again after a pause. The number of retries is logged."""
        if self.device == "cpu":
            return self.algo.update(eps, self.store, self.demos or None)
        snap_m = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        snap_a = pickle.dumps(self.algo.state_dict())
        buf = list(getattr(self.algo, "buffer", []))
        for attempt in range(8):
            try:
                m = self.algo.update(eps, self.store, self.demos or None)
                m["oom_retries"] = attempt
                torch.cuda.empty_cache()
                return m
            except torch.OutOfMemoryError:
                self.model.load_state_dict(snap_m)
                self.algo.load_state_dict(pickle.loads(snap_a))
                if hasattr(self.algo, "buffer"):
                    self.algo.buffer = list(buf)
                torch.cuda.empty_cache()
                time.sleep(float(self.rng.uniform(5, 30)))
        raise RuntimeError("GPU out of memory in 8 attempts")

    def run(self, iters: int | None = None, verbose: bool = True) -> None:
        c = self.cfg
        if self.iter == 0 and c.get("init_from") and not os.path.exists(os.path.join(self.dir, "ckpt", "bc.pt")):
            self.init_from(c["init_from"])
            if verbose:
                print(f"initialised from {c['init_from']} ({len(self.demos)} demonstrations)", flush=True)
        elif self.iter == 0 and c["demos"].get("episodes") and not self.demos:
            prov = self.collect_demos(int(c["demos"]["episodes"]))
            if verbose:
                print(f"demos: {prov['verified_successful']}/{prov['ran']} verified, {prov['kept']} kept; by arm {prov['by_arm_kept']}", flush=True)
            for m in self.pretrain_bc(int(c["demos"].get("bc_epochs", 0))):
                if verbose and (m["epoch"] % 5 == 0):
                    print(f"bc epoch {m['epoch']}: nll {m['bc_nll']:.3f}", flush=True)
            self.save("bc.pt")
        if self.iter == 0 and c["eval_every"]:
            r = self.evaluate(self.eval_tasks(c["eval_episodes"]))
            if verbose:
                print(f"eval @0: success {r['success']:.3f} valid {r['valid_rate']:.2f} {r['by_arm']}", flush=True)
        end = c["iters"] if iters is None else self.iter + iters
        while self.iter < end:
            rec = self.train_iter()
            if verbose:
                print(f"iter {rec['iter']}: train success {rec.get('train_success', 0):.3f} valid {rec.get('train_valid_rate', 0):.2f} "
                      f"ret {rec.get('train_return', 0):.3f} kl {rec.get('kl', 0):.4f} ent {rec.get('entropy', 0):.2f} "
                      f"dθ {rec.get('param_delta', 0):.3f} roll {rec['rollout_s']}s upd {rec['update_s']}s", flush=True)
            if c["eval_every"] and self.iter % c["eval_every"] == 0:
                r = self.evaluate(self.eval_tasks(c["eval_episodes"]))
                if verbose:
                    print(f"eval @{self.iter}: success {r['success']:.3f} valid {r['valid_rate']:.2f} {r['by_arm']}", flush=True)
            if c["ckpt_every"] and self.iter % c["ckpt_every"] == 0:
                self.save()
        self.save()

    # ------------------------------------------------------------------ checkpoints
    def save(self, name: str | None = None) -> str:
        inv = sorted(self.store.key2row.items(), key=lambda kv: kv[1])
        ck = {"model": self.model.state_dict(), "model_cfg": self.model.cfg.to_dict(), "algo": self.algo.state_dict(),
              "algo_name": self.algo.name, "store_keys": np.array([k for k, _ in inv], np.int64),
              "store_mat": self.store.mat[: self.store.n].copy(), "memory": self.memory.state_dict(),
              "library": self.library.state_dict(), "curriculum": self.curriculum.state_dict(), "iter": self.iter,
              "env_decisions": self.env_decisions, "episodes": self.episodes, "wall": self.wall, "cfg": self.cfg,
              "rng": self.rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "theta0": self.theta0.cpu(),
              "code": code_hashes(), "demos_n": len(self.demos)}
        path = os.path.join(self.dir, "ckpt", name or f"iter_{self.iter:04d}.pt")
        torch.save(ck, path)
        if name is None:
            torch.save(ck, os.path.join(self.dir, "ckpt", "latest.pt"))
        return path

    def load(self, path: str, weights: bool = True, memory: bool = True, optimizer: bool = True) -> dict:
        """Restore a checkpoint. `weights` / `memory` can be restored separately (for the ablations that keep one
        and reset the other)."""
        ck = torch.load(path, map_location=self.device, weights_only=False)
        for k, row in zip(ck["store_keys"], ck["store_mat"]):
            if int(k) != 0:
                self.store.put(int(k), row)
        if weights:
            self.model.load_state_dict(ck["model"])
            if optimizer and ck.get("algo_name") == self.algo.name:
                self.algo.load_state_dict(ck["algo"])
            self.theta0 = ck["theta0"].to(self.device)
        if memory:
            self.memory.load_state_dict(ck["memory"])
            self.library.load_state_dict(ck["library"])
            self.mem_synced = 0
            self.pool.sync(self.memory.items, self.library.state_dict(), reset=True)
            self.mem_synced = len(self.memory.items)
        self.curriculum.load_state_dict(ck["curriculum"])
        self.iter, self.env_decisions, self.episodes = ck["iter"], ck["env_decisions"], ck["episodes"]
        self.wall = ck.get("wall", self.wall)
        self.rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        demos = os.path.join(self.dir, "demos.pkl")
        if os.path.exists(demos):
            d = pickle.load(open(demos, "rb"))
            self.demos = d["episodes"]
            for k, v in d.get("emb", []):
                self.store.put(k, v)
        return ck

    def close(self) -> None:
        self.pool.close()
