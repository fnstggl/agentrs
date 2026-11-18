"""Superlearner trainer (``python -m agentrs.sl.train --config ... --set k=v ...``).

One run trains the shared core on a distribution of world families (or adapts a pretrained core to a target
family when ``init_from`` is set), with:
  * recurrent PPO on trial segments + world-model auxiliary loss (``learner``),
  * adaptation-curve evaluations at fixed interaction milestones (per-episode success within K-episode trials,
    with memory ("full") and without adaptation ("none")), on validation tasks of the training families and,
    zero-shot, of held-out families,
  * lifetime checkpoints at fixed interaction milestones (the "experience level" axis of the transfer study),
  * crash safety: atomic checkpoints of weights + optimizer + RNG + counters every iteration window; the sealed
    test results are written before any optional diagnostic; diagnostics are individually persisted and never
    fatal; a finished run is never re-entered.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import subprocess
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from pydantic import BaseModel, Field

from agentrs.sl import worlds as W
from agentrs.sl.learner import PPO
from agentrs.sl.model import SLConfig, SLCore
from agentrs.sl.runner import Runner
from agentrs.sl.world import RewardContract, RewardSpec, Task


class SLTrainConfig(BaseModel):
    run_dir: str
    seed: int = 0
    hours: float = 6.0
    final_reserve_h: float = 0.5
    families: dict = Field(default_factory=lambda: {"cam/negotiation": 1.0})
    heldout_families: list = Field(default_factory=list)       # zero-shot monitors (never trained on)
    model: dict = Field(default_factory=dict)
    learner: dict = Field(default_factory=dict)
    reward: dict = Field(default_factory=dict)
    n_worlds: int = 128
    seg_len: int = 32
    trial_episodes: int = 3
    mode: str = "full"                    # adaptation mode used for training (ablations: gru / episode / none)
    max_env_steps: int = 2_000_000
    max_episode_steps: int = 2000
    eval_at: list = Field(default_factory=lambda: [0, 50_000, 200_000, 500_000, 1_000_000, 2_000_000])
    eval_tasks: int = 64                  # validation tasks per family (one K-episode trial each)
    eval_modes: list = Field(default_factory=lambda: ["full", "none"])
    test_tasks: int = 128
    lifetime_ckpts: list = Field(default_factory=list)       # interaction counts at which to keep a checkpoint
    init_from: Optional[str] = None       # checkpoint dir of a pretrained core (transfer runs)
    init_reset: list = Field(default_factory=list)            # parameter-name prefixes re-initialised after loading
    freeze: list = Field(default_factory=list)                # parameter-name prefixes kept frozen
    wm_coef: float = 0.5
    gpu_mem_fraction: float = 0.15
    device: str = "auto"
    ckpt_every_s: float = 600.0
    diagnostics: list = Field(default_factory=lambda: ["memory_ablation", "world_model"])
    strict_audit: bool = True


def reward_audit(families: list, rc: RewardContract, max_steps: int, n: int = 12, seed: int = 0) -> dict:
    """Adversarial reward-contract audit per family with scripted, model-free policies:
    * ``random``: uniformly random legal actions;
    * ``stall``: always the first legal option (a degenerate repeat policy, which in practice includes invalid /
      time-wasting actions).
    Invariants: every episode of both policies ends within the cap, and the stalling policy never earns a positive
    return without verified success."""
    import random as _r

    from agentrs.sl.runner import Runner  # noqa: F401  (contract helper below mirrors the runner)
    out = {}
    for fam in families:
        if not W.resolve(fam):
            continue
        res = {}
        for pol in ("random", "stall"):
            rng = _r.Random(seed)
            rets, succ, steps, ended = [], [], [], []
            for i in range(n):
                w = W.make_world(fam)
                w.initialize_world(Task(family=fam, seed=900_000 + i, split="train"))
                ags = list(getattr(w, "controlled", None) or w.agents())
                before = {a: w.evaluate_goal(a) for a in ags}
                ret, k, done = 0.0, 0, False
                ov = getattr(w, "reward_overrides", None)
                c = rc if not ov else RewardContract(RewardSpec(**{**rc.s.__dict__, **ov}))
                fixed_len = getattr(w, "agent_controls_length", True) is False
                cap = min(max_steps, 60) if fixed_len else max_steps
                while not done and k < cap:
                    for a in ags:
                        sp = w.available_actions(a)
                        if sp.kind == "discrete":
                            act = sp.labels[0] if pol == "stall" else rng.choice(sp.labels)
                        else:
                            act = []
                            while True:
                                o = sp.allowed(act)
                                if o == ["END"] or not o:
                                    break
                                act.append(o[0] if pol == "stall" else rng.choice(o))
                        w.execute(a, act)
                    w.advance_time()
                    k += 1
                    cons = w.observe_consequences(ags[0])
                    after = w.evaluate_goal(ags[0])
                    ret += c(before[ags[0]], after, cons)
                    before[ags[0]] = after
                    done = cons.done
                rets.append(ret)
                succ.append(bool(before[ags[0]].success))
                steps.append(k)
                ended.append(done or fixed_len)  # episode length outside the agent's control: nothing to exploit
                w.close()
            res[pol] = {"return": float(np.mean(rets)), "success": float(np.mean(succ)), "steps": float(np.mean(steps)),
                        "ended": float(np.mean(ended))}
        r, st = res["random"], res["stall"]
        # invariants: every scripted episode terminates within the cap (no reward for never finishing), and inaction
        # never earns a positive return (preserving resources may beat destructive random play, which is legitimate)
        passed = r["ended"] == 1.0 and st["ended"] == 1.0 and (st["return"] <= 0.0 or st["success"] > 0)
        out[fam] = {**res, "passed": bool(passed)}
    return out


def jdump(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, p)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        return ""


class SLTrainer:
    def __init__(self, cfg: SLTrainConfig):
        self.cfg = c = cfg
        self.rd = Path(c.run_dir)
        self.rd.mkdir(parents=True, exist_ok=True)
        self.ck = self.rd / "ckpt"
        self.ck.mkdir(exist_ok=True)
        dev = c.device if c.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dev = torch.device(dev)
        if self.dev.type == "cuda" and c.gpu_mem_fraction < 1:
            torch.cuda.set_per_process_memory_fraction(c.gpu_mem_fraction)
        for f in list(c.families) + list(c.heldout_families):
            if not W.resolve(f):
                raise ValueError(f"unknown world family {f!r}; available: {sorted(W.FAMILIES)}")
        self.model = SLCore(SLConfig(**c.model)).to(self.dev)
        self.learner = PPO(self.model, c.learner)
        self.rc = RewardContract(RewardSpec(**c.reward))
        sp = self.rd / "state.json"
        if sp.exists():
            self.state = json.loads(sp.read_text())
            self._load(self.ck / "latest")
            self.state["restarts"] = self.state.get("restarts", 0) + 1
            self.log(f"resumed from ckpt/latest (iteration {self.state['iter']}, {self.state['interactions']:,} interactions)")
        else:
            torch.manual_seed(c.seed)
            random.seed(c.seed)
            np.random.seed(c.seed)
            self.state = {"phase": "train", "iter": 0, "interactions": 0, "started_at": time.time(),
                          "config": c.model_dump(), "git": git_commit(), "evals": [], "iters_logged": 0,
                          "compute_s": {"collect": 0.0, "learn": 0.0, "eval": 0.0}, "dt_ref": {}, "lifetime_saved": [],
                          "best": None, "restarts": 0}
            self.state["reward_audit"] = reward_audit(list(c.families) + list(c.heldout_families), self.rc, c.max_episode_steps)
            bad = {f: a for f, a in self.state["reward_audit"].items() if not a["passed"]}
            self.log("reward-contract audit: " + ("PASSED" if not bad else f"FAILED {bad}"))
            if bad and c.strict_audit:
                self.save_state()
                raise SystemExit(f"reward-contract audit failed: {sorted(bad)}")
            if c.init_from:
                self._init_from(Path(c.init_from))
            self._save("initial")
            self._save("latest")
            self.save_state()
        for n, prm in self.model.named_parameters():
            if any(n.startswith(pfx) for pfx in c.freeze):
                prm.requires_grad_(False)
        self.deadline = self.state["started_at"] + c.hours * 3600
        self.runner = None
        self.stop = False
        signal.signal(signal.SIGTERM, self._sigterm)

    # ------------------------------------------------------------------ persistence
    def _sigterm(self, *_):
        self.stop = True

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.rd / "train.log", "a") as f:
            f.write(line + "\n")

    def save_state(self) -> None:
        jdump(self.rd / "state.json", self.state)

    def _save(self, name: str, light: bool = False) -> None:
        """Atomic checkpoint: weights, optimizer, model config, RNG states, run state. ``light`` (milestone and
        lifetime checkpoints used for evaluation / transfer initialisation): weights, config and state only."""
        tmp = self.ck / f".{name}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        torch.save(self.model.state_dict(), tmp / "weights.pt")
        if not light:
            torch.save(self.learner.state_dict(), tmp / "optimizer.pt")
        rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
               "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        torch.save(rng, tmp / "rng.pt")
        jdump(tmp / "config.json", self.model.cfg_dict())
        jdump(tmp / "meta.json", {"version": self.state["iter"], "time": time.time(), "phase": self.state["phase"],
                                  "kind": name, "interactions": self.state["interactions"], "git": self.state.get("git"),
                                  "models": {"policy": "agentrs.sl.SLCore (superlearner core, trained from scratch)"},
                                  "light": light})
        jdump(tmp / "state.json", self.state)
        (tmp / "DONE").write_text(str(time.time()))
        dst = self.ck / name
        if dst.exists():
            old = self.ck / f".{name}.old"
            if old.exists():
                shutil.rmtree(old)
            os.replace(dst, old)
            os.replace(tmp, dst)
            shutil.rmtree(old)
        else:
            os.replace(tmp, dst)

    def _load(self, d: Path) -> None:
        self.model.load_state_dict(torch.load(d / "weights.pt", map_location=self.dev))
        self.learner.load_state_dict(torch.load(d / "optimizer.pt", map_location=self.dev, weights_only=False))
        rng = torch.load(d / "rng.pt", weights_only=False)
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])

    def _init_from(self, d: Path) -> None:
        sd = torch.load(d / "weights.pt", map_location=self.dev)
        own = self.model.state_dict()
        loaded = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape
                  and not any(k.startswith(p) for p in self.cfg.init_reset)}
        self.model.load_state_dict(loaded, strict=False)
        self.state["init_from"] = {"dir": str(d), "loaded": len(loaded), "total": len(own),
                                   "source_state": json.loads((d / "state.json").read_text()).get("interactions") if (d / "state.json").exists() else None}
        self.log(f"initialised {len(loaded)}/{len(own)} tensors from {d}")

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, tag: str, split: str, families: list, n_tasks: int, mode: str) -> dict:
        t0 = time.time()
        self.model.eval()
        rec = {"tag": tag, "split": split, "mode": mode, "interactions": self.state["interactions"], "iter": self.state["iter"],
               "families": {}}
        for fam in families:
            tasks = W.eval_tasks(fam, split, n_tasks)
            n_w = min(len(tasks), self.cfg.n_worlds)
            r = Runner(self.model, {fam: 1.0}, W.make_world, W.sample_task, n_w, trial_episodes=self.cfg.trial_episodes,
                       split=split, seed=12345, mode=mode, contract=self.rc, device=self.dev,
                       max_episode_steps=self.cfg.max_episode_steps, fixed_tasks=tasks, dt_ref=dict(self.state["dt_ref"]))
            try:
                eps = r.run_trials()
            finally:
                r.close()
            curve = []
            for k in range(self.cfg.trial_episodes):
                ek = [e for e in eps if e["episode"] == k]
                if ek:
                    curve.append({"episode": k + 1, "n": len(ek), "success": float(np.mean([e["success"] for e in ek])),
                                  "score": float(np.mean([e["score"] for e in ek])), "progress": float(np.mean([e["progress"] for e in ek])),
                                  "violation": float(np.mean([e["violated"] is not None for e in ek])),
                                  "return": float(np.mean([e["return"] for e in ek])), "steps": float(np.mean([e["steps"] for e in ek]))})
            rec["families"][fam] = {"curve": curve, "auc_success": float(np.mean([p["success"] for p in curve])) if curve else 0.0,
                                    "auc_score": float(np.mean([p["score"] for p in curve])) if curve else 0.0,
                                    "first": curve[0] if curve else None, "last": curve[-1] if curve else None,
                                    "interactions_used": int(r.steps_total)}
        rec["wall_s"] = round(time.time() - t0, 1)
        self.state["compute_s"]["eval"] += time.time() - t0
        self.state["evals"].append(rec)
        self.save_state()
        self.model.train()
        self.log(f"eval {tag}/{split}/{mode}: " + "; ".join(
            f"{f} " + " -> ".join(f"{p['success']:.3f}" for p in v["curve"]) + f" (score {v['auc_score']:.3f})"
            for f, v in rec["families"].items()))
        return rec

    def _val_score(self, rec: dict) -> float:
        return float(np.mean([v["auc_success"] + 0.1 * v["auc_score"] for f, v in rec["families"].items() if f in self.cfg.families]))

    # ------------------------------------------------------------------ training
    def budget_left(self) -> bool:
        return (self.state["interactions"] < self.cfg.max_env_steps and not self.stop
                and time.time() < self.deadline - self.cfg.final_reserve_h * 3600)

    def train_iter(self) -> None:
        c = self.cfg
        if self.runner is None:
            self.runner = Runner(self.model, c.families, W.make_world, W.sample_task, c.n_worlds, trial_episodes=c.trial_episodes,
                                 split="train", seed=c.seed * 1000 + self.state["iter"], mode=c.mode, contract=self.rc,
                                 device=self.dev, max_episode_steps=c.max_episode_steps, dt_ref=self.state["dt_ref"])
        t0 = time.time()
        self.model.eval()
        seg = self.runner.collect(c.seg_len)
        self.model.train()
        t1 = time.time()
        stats = self.learner.update(seg, self.dev, wm_coef=c.wm_coef)
        t2 = time.time()
        n_new = c.seg_len * self.runner.n
        self.state["interactions"] += n_new
        self.state["iter"] += 1
        self.state["compute_s"]["collect"] += t1 - t0
        self.state["compute_s"]["learn"] += t2 - t1
        done = self.runner.trials_done
        self.runner.trials_done = []
        fam_stats = {}
        for e in done:
            fs = fam_stats.setdefault(e["family"], {"episodes": 0, "success": 0, "score": 0.0})
            fs["episodes"] += 1
            fs["success"] += int(e["success"])
            fs["score"] += e["score"]
        for fs in fam_stats.values():
            fs["score"] /= max(1, fs["episodes"])
        rec = {"iter": self.state["iter"], "time": time.time(), "interactions": self.state["interactions"], "collect_s": round(t1 - t0, 1),
               "learn_s": round(t2 - t1, 1), "steps_per_s": round(n_new / max(1e-9, t1 - t0), 1), "learn": stats, "train": fam_stats,
               "env_s": round(self.runner.stats["env_s"], 1), "infer_s": round(self.runner.stats["infer_s"], 1)}
        if self.dev.type == "cuda":
            rec["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        with open(self.rd / "iters.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if self.state["iter"] % 5 == 1:
            self.log(f"iter {self.state['iter']}: {self.state['interactions']:,} interactions, "
                     + ", ".join(f"{k} {v['success']}/{v['episodes']}" for k, v in sorted(fam_stats.items()))
                     + f" | pg {stats['pg']:+.3f} vf {stats['vf']:.3f} ent {stats['ent']:.2f} wm {stats['wm']:.3f} kl {stats['kl']:.4f} "
                     f"mismatch {stats['mismatch']:.1e} | {rec['steps_per_s']} steps/s")

    def _milestones(self) -> None:
        c, s = self.cfg, self.state
        done = {(e["tag"], e["mode"]) for e in s["evals"]}
        for m in c.eval_at:
            tag = f"m{m}"
            if s["interactions"] >= m and all((tag, md) in done for md in c.eval_modes):
                continue
            if s["interactions"] >= m:
                fams = list(c.families) + list(c.heldout_families)
                for md in c.eval_modes:
                    if (tag, md) not in done:
                        rec = self.evaluate(tag, "val", fams, c.eval_tasks, md)
                        if md == "full":
                            sc = self._val_score(rec)
                            if s["best"] is None or sc > s["best"]["score"]:
                                s["best"] = {"score": sc, "interactions": s["interactions"], "tag": tag}
                                self._save("best")
                self._save(tag, light=True)
                self.save_state()
        for m in c.lifetime_ckpts:
            if s["interactions"] >= m and m not in s["lifetime_saved"]:
                self._save(f"life_{m}", light=True)
                s["lifetime_saved"].append(m)
                self.save_state()

    def run(self) -> int:
        c, s = self.cfg, self.state
        if s["phase"] == "done":
            self.log("run already complete")
            return 0
        if s["phase"] == "train":
            last_ck = time.time()
            self._milestones()
            while self.budget_left():
                self.train_iter()
                self._milestones()
                if time.time() - last_ck > c.ckpt_every_s:
                    self._save("latest")
                    self.save_state()
                    last_ck = time.time()
            self._save("latest")
            if self.stop:
                self.save_state()
                self.log("SIGTERM: checkpointed, exiting (resumes on restart)")
                return 3
            s["phase"] = "final"
            s["train_finished_at"] = time.time()
            self._save("final")
            self.save_state()
        if s["phase"] == "final":
            fams = list(c.families) + list(c.heldout_families)
            done = {(e["tag"], e["split"], e["mode"]) for e in s["evals"]}
            if ("final", "test", "full") not in done:
                self.evaluate("final", "test", fams, c.test_tasks, "full")
            self.save_state()  # sealed test results on disk before any optional diagnostic
            for name in c.diagnostics:
                self.optional_diagnostic(name, lambda n=name: DIAGNOSTICS[n](self, fams))
            s["phase"] = "done"
            s["finished_at"] = time.time()
            self.save_state()
            self.log("superlearner run complete")
        return 0

    def optional_diagnostic(self, key: str, fn) -> None:
        """Persisted, never fatal, retried at most once after an interrupted attempt (see agentrs.runtime_safety)."""
        s = self.state
        if key in s.get("diagnostics", {}) or key in s.get("diagnostic_errors", {}):
            return
        att = s.setdefault("diagnostic_attempts", {})
        if att.get(key, 0) >= 2:
            s.setdefault("diagnostic_errors", {})[key] = {"error": "abandoned: interrupted twice", "time": time.time()}
            self.save_state()
            return
        att[key] = att.get(key, 0) + 1
        self.save_state()
        try:
            res = fn()
            s.setdefault("diagnostics", {})[key] = res
        except Exception as e:  # noqa: BLE001
            s.setdefault("diagnostic_errors", {})[key] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
            self.log(f"diagnostic {key} FAILED (recorded; run continues): {type(e).__name__}: {e}")
        self.save_state()


def _diag_memory_ablation(t: SLTrainer, fams) -> dict:
    """Adaptation curves on validation tasks with the trained weights frozen, under every adaptation mode."""
    out = {}
    for md in ("full", "gru", "episode", "none"):
        rec = t.evaluate(f"final-{md}", "val", fams, max(2, t.cfg.eval_tasks // 2), md)
        out[md] = {f: v["auc_success"] for f, v in rec["families"].items()}
    return out


@torch.no_grad()
def _diag_world_model(t: SLTrainer, fams) -> dict:
    """One-step prediction quality of the world model on fresh validation segments (real transitions)."""
    from agentrs.sl.featurize import collate
    from agentrs.sl.model import symlog

    m = t.model
    out = {}
    for fam in fams:
        r = Runner(m, {fam: 1.0}, W.make_world, W.sample_task, 4, trial_episodes=t.cfg.trial_episodes, split="val", seed=99,
                   mode="full", contract=t.rc, device=t.dev, max_episode_steps=t.cfg.max_episode_steps)
        try:
            seg = r.collect(32)
        finally:
            r.close()
        lp, ent, val, wml = t.learner._replay(seg, torch.arange(r.n), "full", t.dev)
        rew = torch.stack([s["r"] for s in seg["steps"]])
        out[fam] = {"wm_loss": float(wml.mean()), "reward_abs": float(rew.abs().mean()), "value_mean": float(val.mean())}
    return out


DIAGNOSTICS = {"memory_ablation": _diag_memory_ablation, "world_model": _diag_world_model}


def load_config(path: Path) -> dict:
    """YAML config with an optional ``base:`` file (relative to the config) merged underneath it."""
    d = yaml.safe_load(path.read_text()) or {}
    if "base" in d:
        b = load_config(path.parent / d.pop("base"))
        for k, v in d.items():
            b[k] = {**b[k], **v} if isinstance(v, dict) and isinstance(b.get(k), dict) else v
        d = b
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    d = load_config(Path(a.config)) if a.config else {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        v = yaml.safe_load(v)
        d[k] = {**d[k], **v} if isinstance(v, dict) and isinstance(d.get(k), dict) and k not in ("families",) else v
    cfg = SLTrainConfig(**d)
    sp = Path(cfg.run_dir) / "state.json"
    if sp.exists() and json.loads(sp.read_text()).get("phase") == "done":
        print("run already complete", flush=True)
        return 0
    return SLTrainer(cfg).run()


if __name__ == "__main__":
    raise SystemExit(main())
