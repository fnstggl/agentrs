"""CAM 2.0 training / evaluation orchestration.

Collection runs many environment workers concurrently (``EnvPool``); every ready stream is acted on in one batched
CAM + frozen-LLM call. Training alternates collection of ``episodes_per_update`` complete episodes with a learner
update; evaluation uses the same runner with ``train=False``: no buffer or memory writes, no optimiser, and a
parameter checksum asserted unchanged afterwards.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from agentrs.cam2.agent import Agent, action_text
from agentrs.cam2.backbone import load_llm
from agentrs.cam2.envs.base import TaskRef
from agentrs.cam2.envs.pool import EnvPool
from agentrs.cam2.featurize import ToolRegistry, make_step_input
from agentrs.cam2.learner import Episode, LearnConfig, Learner, Transition
from agentrs.cam2.model import CAM, CAMConfig, EpisodicMemory, param_checksum
from agentrs.cam2.planner import PlanConfig
from agentrs.cam2.tasks import TrainSampler, protocol_hash, train_eval_tasks, val_tasks


@dataclass
class RunConfig:
    run_dir: str = "runs/cam2/dev"
    env: str = "miniwob"
    llm: str = "Qwen/Qwen3-1.7B"
    workers: int = 16
    seed: int = 0
    env_steps: int = 20_000
    max_minutes: float = 60.0
    max_steps: int = 10
    train_families: Optional[list] = None
    eval_every: int = 10                  # updates
    eval_per_family: int = 4
    eval_families: Optional[list] = None   # None = VAL families + TRAIN families (val seeds)
    ckpt_every: int = 5
    cam: dict = field(default_factory=dict)
    learn: dict = field(default_factory=dict)
    plan: dict = field(default_factory=dict)
    phase: str = "A"
    init_from: Optional[str] = None       # checkpoint to initialise weights from (weights only)
    save_replay: bool = True
    final_eval: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def env_versions() -> dict:
    import importlib.metadata as md

    out = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda}
    for pkg in ["transformers", "browsergym-core", "browsergym-miniwob", "playwright", "gymnasium", "numpy"]:
        try:
            out[pkg] = md.version(pkg)
        except Exception:  # noqa: BLE001
            out[pkg] = None
    try:
        out["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                           cwd=Path(__file__).parent).stdout.strip()
    except Exception:  # noqa: BLE001
        out["git_commit"] = None
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
    return out


def bootstrap_ci(x: list, n: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple:
    if not x:
        return (float("nan"), float("nan"), float("nan"))
    a = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bs = a[rng.integers(0, len(a), size=(n, len(a)))].mean(1)
    return float(a.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


class Runner:
    """Drives all workers of a pool with one agent."""

    def __init__(self, agent: Agent, pool: EnvPool, mem: Optional[EpisodicMemory], max_steps: int, gamma: float,
                 carry_state: bool = False, gen: Optional[torch.Generator] = None):
        self.agent, self.pool, self.mem, self.max_steps, self.gamma = agent, pool, mem, max_steps, gamma
        self.carry_state, self.gen = carry_state, gen
        self.cam = agent.cam
        self.W = pool.n
        self.state = self.cam.init_state(self.W, agent.device)
        self.env_steps = 0
        self.errors = 0
        self.invalid = 0
        self.dropped: list = []
        self.batch_wait = 0.5

    def run(self, next_task: Callable[[], Optional[TaskRef]], n_episodes: int, train: bool, greedy: bool = False,
            on_episode: Optional[Callable] = None, plan_cost: float = 0.0) -> list:
        cam, dev, W = self.cam, self.agent.device, self.W
        S = [dict(active=False, wait=False) for _ in range(W)]
        launched = 0
        done_eps: list = []
        consecutive_errors = 0

        def launch(i, failed: bool = False):
            """Start the next episode on worker ``i``. After an environment failure (browser crash, timeout) the
            SAME task is restarted from scratch (up to 3 tries, so paired evaluations keep identical task sets);
            only then is it dropped and recorded. Persistent failure aborts loudly."""
            nonlocal launched, consecutive_errors
            t, tries = None, 0
            if failed:
                consecutive_errors += 1
                if consecutive_errors > max(50, 4 * W):
                    raise RuntimeError(f"{consecutive_errors} consecutive environment failures; aborting collection")
                tries = S[i].get("tries", 0) + 1
                if tries <= 3:
                    t = S[i]["task"]
                else:
                    self.dropped.append(S[i]["task"].key())
                    tries = 0
            if t is None:
                if launched >= n_episodes:
                    S[i] = dict(active=False, wait=False)
                    return
                t = next_task()
                if t is None:
                    S[i] = dict(active=False, wait=False)
                    return
                launched += 1
            S[i] = dict(active=True, wait=False, task=t, tr=[], k=0, pact="", prev_r=0.0, invalid=False, ret=0.0,
                        tries=tries)
            if not (self.carry_state and train):
                for key in self.state:
                    self.state[key][i] = 0.0
            self.pool.send(i, "reset", t)

        for i in range(W):
            launch(i)
        boot_queue: list = []
        first_ready_at = None
        while any(s["active"] for s in S):
            ready = [i for i in range(W) if S[i]["active"] and S[i]["wait"]]
            # batching window: act when at least half of the busy streams are ready, or 0.5 s after the first one
            # became ready (one batched LLM call per decision round instead of one per straggler)
            n_act = sum(1 for s in S if s["active"])
            if ready and first_ready_at is None:
                first_ready_at = time.time()
            if ready and len(ready) * 2 < n_act and time.time() - first_ready_at < self.batch_wait:
                ready_now = []
            else:
                ready_now = ready
            ready = ready_now
            if ready:
                first_ready_at = None
                steps = [make_step_input(S[i]["obs"], S[i]["tools"], self.agent.reg, S[i]["pact"], S[i]["prev_r"],
                                         S[i]["k"], self.max_steps, S[i]["invalid"]) for i in ready]
                idx = torch.tensor(ready, device=dev)
                st = {k: v[idx] for k, v in self.state.items()}
                mem = mm = None
                if cam.cfg.mem_k > 0 and self.mem is not None:
                    mem, mm = cam.retrieve(self.mem, st["h"])
                out = self.agent.act(steps, st, mem, mm, greedy=greedy, gen=self.gen)
                for j, i in enumerate(ready):
                    for k in self.state:
                        self.state[k][i] = out.state[k][j]
                    s = S[i]
                    call = out.calls[j]
                    if train:
                        g = out.gen[j]
                        tr = Transition(step=steps[j], h0={k: v[j].detach().cpu() for k, v in st.items()},
                                        mem=None if mem is None else mem[j].detach().to(torch.float16).cpu(),
                                        mem_m=None if mm is None else mm[j].detach().cpu(),
                                        cand=int(out.cand[j]), logp_cand=float(out.logp_cand[j]),
                                        delib=int(out.delib[j]), logp_delib=float(out.logp_delib[j]),
                                        bias=None if out.bias is None else out.bias[j, : len(steps[j].cands)].detach().cpu(),
                                        logits=out.logits[j, : len(steps[j].cands)].detach().to(torch.float16).cpu(),
                                        gen=None if g is None else {k: g.get(k) for k in ("ctx", "stub", "tokens", "text", "kind", "prefix")},
                                        logp_text=float(out.logp_text[j]), value=float(out.value[j]),
                                        f=out.f[j].detach().to(torch.float16).cpu(), delib_used=out.delib_used)
                    else:
                        tr = Transition(step=steps[j], h0={}, mem=None, mem_m=None, cand=int(out.cand[j]),
                                        logp_cand=0.0, delib=int(out.delib[j]), logp_delib=0.0, bias=None,
                                        logits=torch.zeros(0), gen=None, logp_text=0.0, value=float(out.value[j]),
                                        f=out.f[j].detach().to(torch.float16).cpu())
                    s["tr"].append(tr)
                    s["last_call"] = call
                    s["last_step"] = steps[j]
                    s["wait"] = False
                    self.pool.send(i, "step", call)
            for i, cmd, st_, v in self.pool.poll(timeout=0.001 if ready else (0.05 if first_ready_at else 2.0)):
                s = S[i]
                if st_ != "ok":
                    self.errors += 1
                    print(f"[runner] worker {i} {cmd} failed: {str(v)[:300]}", flush=True)
                    launch(i, failed=True)
                    continue
                consecutive_errors = 0
                if cmd == "reset":
                    obs, tools = v
                    s.update(obs=obs, tools=tools, wait=True)
                    continue
                sr = v
                self.env_steps += 1
                tr = s["tr"][-1]
                r = float(sr.reward) - (plan_cost if tr.delib == 1 else 0.0)
                tr.reward, tr.terminated, tr.truncated = r, bool(sr.terminated), bool(sr.truncated)
                tr.invalid = bool(sr.info.get("invalid"))
                self.invalid += int(tr.invalid)
                s["ret"] += float(sr.reward)
                s["k"] += 1
                s["pact"] = action_text(s["last_call"], s["last_step"])
                s["prev_r"], s["invalid"] = float(sr.reward), tr.invalid
                s["obs"] = sr.obs
                if sr.terminated or sr.truncated:
                    ep = Episode(tr=s["tr"], task=s["task"], success=bool(sr.info.get("success", sr.reward > 0)),
                                 score=float(sr.info.get("score", float(sr.reward > 0))), ret=s["ret"],
                                 family=s["task"].family)
                    ep.info = {k: v for k, v in sr.info.items() if k in ("eval",)}
                    if sr.truncated and not sr.terminated and train:
                        boot_queue.append((ep, i, s["obs"], s["tools"], s["pact"], s["prev_r"], s["k"], s["invalid"]))
                    else:
                        self._finish(ep, train, on_episode, done_eps)
                    launch(i)
                else:
                    s["wait"] = True
            if boot_queue:
                self._bootstrap(boot_queue, train, on_episode, done_eps)
                boot_queue = []
        return done_eps

    @torch.no_grad()
    def _bootstrap(self, q, train, on_episode, done_eps):
        """Value of the final observation of truncated episodes (bootstrapped, not treated as terminal)."""
        steps = [make_step_input(o, tl, self.agent.reg, pa, pr, k, self.max_steps, inv) for (_, _, o, tl, pa, pr, k, inv) in q]
        from agentrs.cam2.featurize import collate

        st = {}
        for key in self.state:
            st[key] = torch.stack([self.state[key][i] for (_, i, *_r) in q])
        mem = mm = None
        if self.cam.cfg.mem_k > 0 and self.mem is not None:
            mem, mm = self.cam.retrieve(self.mem, st["h"])
        b = collate(self.agent.llm, self.agent.reg, steps, mem, mm, self.agent.device, goal_cond=self.cam.cfg.el_goal_cond)
        f, *_ = self.cam.integrate(b, st)
        v = self.cam.v(f)
        for j, (ep, *_r) in enumerate(q):
            ep.tr[-1].boot_value = float(v[j])
            self._finish(ep, train, on_episode, done_eps)

    def _finish(self, ep: Episode, train: bool, on_episode, done_eps):
        if train and self.mem is not None and self.cam.cfg.mem_k > 0:
            T = len(ep.tr)
            g, rtg = 0.0, []
            for x in reversed(ep.tr):
                g = x.reward + self.gamma * g
                rtg.append(g)
            rtg = rtg[::-1]
            f = torch.stack([x.f for x in ep.tr]).float().to(self.agent.device)
            rows = self.cam.memory_rows(f, torch.tensor(rtg, device=f.device),
                                        torch.full((T,), float(ep.success), device=f.device),
                                        torch.arange(T, device=f.device).float() / max(1, T))
            self.mem.write(rows)
        done_eps.append(ep)
        if on_episode:
            on_episode(ep)


# ====================================================================================================== Trainer
class Trainer:
    def __init__(self, rc: RunConfig, llm=None, pool: Optional[EnvPool] = None, device: Optional[str] = None):
        self.rc = rc
        self.dir = Path(rc.run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        set_seeds(rc.seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.time()
        self.llm = llm or load_llm(rc.llm, device=self.device)
        self.llm_load_s = time.time() - t0
        self.llm_fp_start = self.llm.fingerprint()
        ccfg = CAMConfig(**{**rc.cam, "llm_pooled_dim": self.llm.pooled_dim, "llm_hidden": self.llm.hidden})
        init = self.llm.token_embedding_rows(" action control signal for the answer:") if ccfg.prefix_len else None
        self.cam = CAM(ccfg, prefix_init=init).to(self.device)
        self.lcfg = LearnConfig(**rc.learn)
        self.pcfg = PlanConfig(**rc.plan)
        self.reg = ToolRegistry()
        self.mem = EpisodicMemory(self.cam.mem_raw_dim, ccfg.mem_capacity, self.device) if ccfg.mem_k > 0 else None
        self.learner = Learner(self.cam, self.llm, self.reg, self.lcfg, self.device)
        self.gen = torch.Generator(device=self.device).manual_seed(rc.seed + 17)
        self.agent = Agent(self.llm, self.cam, self.reg, self.device, temperature=self.lcfg.temperature,
                           plan=self.pcfg if self.pcfg.enabled else None)
        self.pool = pool
        self.own_pool = pool is None
        self.replay: list = []
        self.replay_seen = 0
        self.counters = {"env_steps": 0, "episodes": 0, "updates": 0, "wall_s": 0.0, "llm_gen_tokens": 0,
                         "llm_encode_tokens": 0}
        self.log_f = None
        self.sampler = TrainSampler(rc.env, rc.train_families, seed=rc.seed)
        if rc.init_from:
            self.load(rc.init_from, weights_only=True)
        if (self.dir / "ckpt" / "latest.pt").exists():
            self.load(str(self.dir / "ckpt" / "latest.pt"))
            print(f"[trainer] resumed from {self.dir / 'ckpt' / 'latest.pt'} at {self.counters}", flush=True)

    # ---------------------------------------------------------------- infra
    def env_kwargs(self) -> dict:
        return {"max_steps": self.rc.max_steps}

    def ensure_pool(self):
        if self.pool is None:
            self.pool = EnvPool(self.rc.env, self.rc.workers, self.env_kwargs())
        return self.pool

    def log(self, rec: dict) -> None:
        if self.log_f is None:
            self.log_f = open(self.dir / "log.jsonl", "a")
        self.log_f.write(json.dumps(rec, default=float) + "\n")
        self.log_f.flush()

    def describe(self) -> dict:
        return {"run": self.rc.to_dict(), "cam_config": self.cam.cfg.to_dict(), "cam_params": self.cam.n_params(),
                "learn": self.lcfg.to_dict(), "plan": asdict(self.pcfg), "llm": self.llm.describe(),
                "llm_fingerprint_start": self.llm_fp_start, "protocol_hash": protocol_hash(),
                "versions": env_versions(), "env": self.pool.desc[0] if self.pool and self.pool.desc else None}

    # ---------------------------------------------------------------- checkpoints
    def save(self, name: str = "latest") -> str:
        d = self.dir / "ckpt"
        d.mkdir(parents=True, exist_ok=True)
        ck = {"cam": self.cam.state_dict(), "cam_config": self.cam.cfg.to_dict(), "learner": self.learner.state_dict(),
              "learn_config": self.lcfg.to_dict(), "plan": asdict(self.pcfg), "run": self.rc.to_dict(),
              "counters": self.counters, "registry": self.reg.state_dict(),
              "memory": self.mem.state_dict() if self.mem is not None else None,
              "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                      "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                      "gen": self.gen.get_state(), "sampler": self.sampler.rng.getstate()},
              "llm": {"name": self.llm.name, "fingerprint": self.llm_fp_start, "revision": self.llm.revision},
              "replay_seen": self.replay_seen, "checksum": param_checksum(self.cam), "time": time.time()}
        tmp = d / f"{name}.pt.tmp"
        torch.save(ck, tmp)
        os.replace(tmp, d / f"{name}.pt")
        if self.rc.save_replay and self.replay:
            torch.save(self.replay, d / "replay.pt.tmp")
            os.replace(d / "replay.pt.tmp", d / "replay.pt")
        return str(d / f"{name}.pt")

    def load(self, path: str, weights_only: bool = False) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if ck["llm"]["name"] != self.llm.name:
            raise ValueError(f"checkpoint was trained with {ck['llm']['name']}, not {self.llm.name}")
        self.reg.load_state_dict(ck["registry"])
        self.cam.load_state_dict(ck["cam"])
        if ck.get("memory") is not None and self.mem is not None:
            self.mem.load_state_dict(ck["memory"])
        if weights_only:
            rp = Path(path).parent / "replay.pt"
            if rp.exists():  # older tasks' experience travels with the weights (retention terms)
                self.replay = torch.load(rp, weights_only=False)
            return
        self.learner.load_state_dict(ck["learner"])
        self.counters = ck["counters"]
        self.replay_seen = ck.get("replay_seen", 0)
        r = ck["rng"]
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        # RNG states are CPU ByteTensors; ``map_location`` above may have moved them to the GPU
        torch.set_rng_state(r["torch"].cpu())
        if r["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in r["cuda"]])
        self.gen.set_state(r["gen"].cpu())
        self.sampler.rng.setstate(r["sampler"])
        rp = Path(path).parent / "replay.pt"
        if rp.exists():
            self.replay = torch.load(rp, weights_only=False)
        p = ck.get("plan")
        if p:
            self.pcfg.reliability, self.pcfg.u_max = p.get("reliability", 0.0), p.get("u_max", 1e9)
        assert abs(param_checksum(self.cam) - ck["checksum"]) < 1e-6 * max(1.0, abs(ck["checksum"])), "checksum mismatch"

    # ---------------------------------------------------------------- evaluation (no updates)
    @torch.no_grad()
    def evaluate(self, tasks: list, greedy: bool = True, label: str = "val") -> dict:
        pool = self.ensure_pool()
        before = param_checksum(self.cam)
        mem_writes = self.mem.writes if self.mem is not None else 0
        self.cam.eval()
        it = iter(tasks)
        runner = Runner(self.agent, pool, self.mem, self.rc.max_steps, self.lcfg.gamma, gen=self.gen)
        t0 = time.time()
        eps = runner.run(lambda: next(it, None), len(tasks), train=False, greedy=greedy)
        self.cam.train()
        assert param_checksum(self.cam) == before, "evaluation changed CAM parameters"
        assert self.mem is None or self.mem.writes == mem_writes, "evaluation wrote episodic memory"
        fam: dict = {}
        for e in eps:
            fam.setdefault(e.family, []).append(float(e.success))
        succ = [float(e.success) for e in eps]
        m, lo, hi = bootstrap_ci(succ)
        res = {"label": label, "n": len(eps), "success": m, "ci95": [lo, hi],
               "score": float(np.mean([e.score for e in eps])) if eps else float("nan"),
               "per_family": {k: float(np.mean(v)) for k, v in sorted(fam.items())},
               "mean_len": float(np.mean([len(e.tr) for e in eps])) if eps else 0.0,
               "invalid_rate": runner.invalid / max(1, runner.env_steps), "env_steps": runner.env_steps,
               "errors": runner.errors, "dropped": list(runner.dropped), "eval_s": time.time() - t0,
               "episodes": [{"task": e.task.key(), "success": e.success, "score": e.score, "len": len(e.tr)} for e in eps]}
        return res

    # ---------------------------------------------------------------- training
    def train(self) -> dict:
        rc, lc = self.rc, self.lcfg
        pool = self.ensure_pool()
        (self.dir / "config.json").write_text(json.dumps(self.describe(), indent=1, default=str))
        runner = Runner(self.agent, pool, self.mem, rc.max_steps, lc.gamma, carry_state=lc.carry_state, gen=self.gen)
        runner.env_steps = self.counters["env_steps"]
        t_start = time.time() - self.counters["wall_s"]
        recent: list = []
        if self.counters["updates"] == 0:
            self.save("initial")
        while self.counters["env_steps"] < rc.env_steps and (time.time() - t_start) / 60 < rc.max_minutes:
            tc = time.time()
            g0, e0 = self.llm.stats["gen_tokens"], self.llm.stats["encode_tokens"]
            eps = runner.run(self.sampler, lc.episodes_per_update, train=True, plan_cost=0.002 if self.cam.cfg.deliberation else 0.0)
            collect_s = time.time() - tc
            for e in eps:
                e.phase, e.version = rc.phase, self.counters["updates"]
            self.counters["env_steps"] = runner.env_steps
            if not eps:
                print("[trainer] collection returned no complete episodes; collecting again", flush=True)
                continue
            self.counters["episodes"] += len(eps)
            recent += [float(e.success) for e in eps]
            recent = recent[-400:]
            holdout = eps[: min(16, len(eps))] if lc.wm_coef > 0 else None
            stats = self.learner.update(eps, self.replay, holdout=holdout)
            if "wm_reliability" in stats:   # calibration travels with every checkpoint (used if planning is enabled later)
                self.pcfg.reliability = stats["wm_reliability"]
                self.pcfg.u_max = stats.get("wm_u_max", 1e9)
            # replay: reservoir over all past episodes (keeps old task families represented). Replay never
            # recomputes text log-probs, so the stored control vectors are dropped (32 KB per text step)
            for e in eps:
                for x in e.tr:
                    if x.gen is not None:
                        x.gen.pop("prefix", None)
                        x.gen.pop("delta", None)
                self.replay_seen += 1
                if len(self.replay) < lc.replay_capacity:
                    self.replay.append(e)
                else:
                    j = random.randrange(self.replay_seen)
                    if j < lc.replay_capacity:
                        self.replay[j] = e
            self.counters["updates"] += 1
            self.counters["wall_s"] = time.time() - t_start
            self.counters["llm_gen_tokens"] += self.llm.stats["gen_tokens"] - g0
            self.counters["llm_encode_tokens"] += self.llm.stats["encode_tokens"] - e0
            fam: dict = {}
            for e in eps:
                fam.setdefault(e.family, []).append(float(e.success))
            rec = {"type": "update", **self.counters, "train_success": float(np.mean([e.success for e in eps])),
                   "train_success_recent": float(np.mean(recent)), "train_score": float(np.mean([e.score for e in eps])),
                   "ep_len": float(np.mean([len(e.tr) for e in eps])), "collect_s": collect_s,
                   "invalid_steps": runner.invalid, "env_errors": runner.errors,
                   "gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
                   "per_family": {k: float(np.mean(v)) for k, v in fam.items()}, **stats}
            rec["samples"] = [{"task": e.task.key(), "success": e.success,
                                "actions": [self._describe_action(t) for t in e.tr]} for e in eps[:3]]
            self.log(rec)
            print(f"[upd {self.counters['updates']}] steps={self.counters['env_steps']} succ={rec['train_success']:.3f} "
                  f"recent={rec['train_success_recent']:.3f} len={rec['ep_len']:.1f} loss={stats.get('loss', 0):.3f} "
                  f"mis={stats.get('replay_mismatch', 0):.2e} kl={stats.get('approx_kl', 0):.3f} "
                  f"dθ={stats['update_rel_norm']:.2e} col={collect_s:.0f}s upd={stats['update_s']:.0f}s", flush=True)
            if rc.eval_every and self.counters["updates"] % rc.eval_every == 0:
                ev = self.evaluate(val_tasks(rc.env, rc.eval_per_family, rc.eval_families), label="val")
                ev.pop("episodes")
                self.log({"type": "eval", **ev, "eval_env_steps": ev["env_steps"], **self.counters})
                print(f"[eval] val success={ev['success']:.3f} ci={ev['ci95']}", flush=True)
            if rc.ckpt_every and self.counters["updates"] % rc.ckpt_every == 0:
                self.save("latest")
        self.save("latest")
        self.save("final")
        fp_end = self.llm.fingerprint()
        summary = {"counters": self.counters, "llm_fingerprint_unchanged": fp_end == self.llm_fp_start,
                   "cam_params": self.cam.n_params()}
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        return summary

    def _describe_action(self, t: Transition) -> str:
        t_, e, en = t.step.cands[t.cand]
        spec = self.reg.sets[t.step.tools_key]["specs"][t_]
        el = t.step.el_texts[e][:60] if e >= 0 else ""
        txt = f" <- {t.gen['text'][:80]!r}" if t.gen else ""
        return f"{spec.name}({el}{', ' + en if en else ''}){txt} r={t.reward:g}"

    def close(self) -> None:
        if self.own_pool and self.pool is not None:
            self.pool.close()
            self.pool = None
        if self.log_f:
            self.log_f.close()
            self.log_f = None


def main(argv=None):
    if os.environ.get("CAM2_STACKDUMP"):
        import faulthandler

        faulthandler.dump_traceback_later(float(os.environ["CAM2_STACKDUMP"]), repeat=True)
    ap = argparse.ArgumentParser(description="CAM 2.0 training")
    ap.add_argument("--config", help="JSON/YAML RunConfig")
    ap.add_argument("--set", nargs="*", default=[], help="overrides key=value (dotted for cam./learn./plan.)")
    a = ap.parse_args(argv)
    cfg = {}
    if a.config:
        import yaml

        cfg = yaml.safe_load(open(a.config))
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except Exception:  # noqa: BLE001
            pass
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    rc = RunConfig(**cfg)
    tr = Trainer(rc)
    try:
        s = tr.train()
        print(json.dumps(s, indent=1, default=float))
    finally:
        tr.close()


if __name__ == "__main__":
    main()
