"""Real-experience records, persistent replay and the promising-state archive.

Provenance rules: the replay holds only transitions that happened in an environment (``origin =
"real"``). Hindsight-relabelled goals are *views* produced at sampling time (``Batch.relabeled``) and
never overwrite the recorded goal or outcome; imagined transitions live only inside the dream learner
and are never stored here. Rewards are stored as contract components so that the training reward can
be recomputed exactly for a different constraint multiplier or a hindsight goal.
"""
from __future__ import annotations

import gzip
import math
import pickle
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class Step:
    obs: list                 # observation token ids (pre-action)
    prev_act: list            # previous action tokens
    elapsed_prev: float       # simulated hours the previous action took
    status: np.ndarray        # verified status vector at decision time
    feat_obs: np.ndarray      # observable features at decision time
    prev_opt: int
    new_opt: bool
    term_sample: int
    opt: int
    act_tokens: list          # complete action incl. END
    act_ids: list
    logp_b: float             # behaviour log-prob of the full factored decision (term, option, tokens)
    r_shape: float = 0.0      # reward contract components (training reward = sum - lam * viol)
    r_time: float = 0.0
    r_fail: float = 0.0
    r_succ: float = 0.0
    viol: float = 0.0
    disc: float = 1.0         # gamma ** dt
    dt: float = 1.0           # clock units consumed by the action
    elapsed_h: float = 0.0
    events: Optional[np.ndarray] = None
    feat_after: Optional[np.ndarray] = None
    status_after: Optional[np.ndarray] = None
    prog_before: float = 0.0
    prog_after: float = 0.0
    done: bool = False
    valid: bool = True
    learn: bool = True        # False for replayed archive prefixes (belief burn-in only)
    first: bool = False       # first decision of the agent in this episode (no option-termination term)
    sem: Optional[np.ndarray] = None  # frozen semantic vector of the natural-language observation (semantic mode)
    feat: Optional[np.ndarray] = None  # PPO: the actor's stop-gradient input at decision time (exact ratios); dropped after the update

    def reward(self, lam: float = 0.0) -> float:
        return self.r_shape + self.r_time + self.r_fail + self.r_succ - lam * self.viol


@dataclass
class Episode:
    family: str
    split: str
    seed: int
    goal: list
    version: str
    env_version: str = ""
    start: str = "scratch"    # "scratch" | "archive"
    n_prefix: int = 0
    difficulty: float = 1.0
    steps: list = field(default_factory=list)
    success: bool = False
    outcome: str = ""         # success | violation | failure
    cause: Optional[str] = None
    milestones: dict = field(default_factory=dict)
    final: dict = field(default_factory=dict)  # time_left, resources at termination
    hindsight: list = field(default_factory=list)
    collected_iter: int = 0
    origin: str = "real"
    sampled: int = 0
    truncated_by_guard: bool = False
    final_obs: list = field(default_factory=list)  # observation ids after the last action (terminal state)
    final_elapsed: float = 0.0
    goal_sem: Optional[np.ndarray] = None    # semantic mode: frozen vector of the natural-language goal
    goal_feat: Optional[np.ndarray] = None   # PPO: goal encoding used by the actor at collection time
    final_sem: Optional[np.ndarray] = None   # semantic mode: frozen vector of the terminal observation text

    @property
    def n_learn(self) -> int:
        return sum(s.learn for s in self.steps)


@dataclass
class View:
    """An episode as used for one update: possibly with a hindsight goal (sparse success at ``upto``)."""
    ep: Episode
    goal: list
    upto: int
    relabeled: bool = False
    shaping: float = 0.0   # potential-shaping coefficient of the reward contract (for hindsight views)
    goal_sem: Optional[np.ndarray] = None  # semantic mode: frozen vector of this view's (possibly hindsight) goal

    def rewards(self, lam: float) -> list[float]:
        st = self.ep.steps[:self.upto]
        if not self.relabeled:
            return [s.reward(lam) for s in st]
        # hindsight goal "reach the level actually achieved": the same time costs, success at the step that
        # achieved it, no failure charge, and potential shaping on progress *relative to the achieved level*
        # (goals here are thresholds on a progress-proportional quantity, so phi' = min(1, progress / achieved))
        ach = max(1e-6, st[-1].prog_after)
        phi = lambda x: min(1.0, x / ach)  # noqa: E731
        out = []
        for i, s in enumerate(st):
            last = i == len(st) - 1
            sh = self.shaping * (s.disc * (0.0 if last else phi(s.prog_after)) - phi(s.prog_before))
            out.append(s.r_time + sh + (1.0 if last else 0.0))
        return out

    def dones(self) -> list[bool]:
        st = self.ep.steps[:self.upto]
        return [s.done or (self.relabeled and i == len(st) - 1) for i, s in enumerate(st)]

    @property
    def steps(self) -> list[Step]:
        return self.ep.steps[:self.upto]


class Replay:
    """Persistent replay of complete real episodes (episodes are short, so beliefs are reconstructed
    exactly by unrolling each sampled episode from its first step — no stored-state staleness).
    A protected reservoir keeps rare successes and the most advanced partial attempts from eviction."""

    def __init__(self, capacity_steps: int = 300_000, protected_steps: int = 40_000, directory: Optional[Path] = None):
        self.cap, self.pcap = capacity_steps, protected_steps
        self.main: deque = deque()
        self.prot: deque = deque()
        self.n_main = self.n_prot = 0
        self.dir = Path(directory) if directory else None
        self.added = 0
        self.sampled_steps = 0

    def __len__(self):
        return len(self.main) + len(self.prot)

    def steps(self) -> int:
        return self.n_main + self.n_prot

    def _protect(self, e: Episode) -> bool:
        return e.success or (e.outcome != "violation" and e.steps and e.steps[-1].prog_after >= 0.5)

    def add(self, eps: list[Episode], it: int = 0, persist: bool = True) -> None:
        for e in eps:
            e.collected_iter = it
            if self._protect(e):
                self.prot.append(e)
                self.n_prot += len(e.steps)
                while self.n_prot > self.pcap and len(self.prot) > 1:
                    self.n_prot -= len(self.prot.popleft().steps)
            else:
                self.main.append(e)
                self.n_main += len(e.steps)
                while self.n_main > self.cap and len(self.main) > 1:
                    self.n_main -= len(self.main.popleft().steps)
            self.added += 1
        if persist and self.dir and eps:
            self.dir.mkdir(parents=True, exist_ok=True)
            with gzip.open(self.dir / f"shard_{it:06d}.pkl.gz", "wb") as f:
                pickle.dump(eps, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self) -> int:
        if not self.dir or not self.dir.exists():
            return 0
        n = 0
        for p in sorted(self.dir.glob("shard_*.pkl.gz")):
            try:
                with gzip.open(p, "rb") as f:
                    eps = pickle.load(f)
            except (OSError, EOFError, pickle.UnpicklingError):
                continue
            self.add(eps, eps[0].collected_iter if eps else 0, persist=False)
            n += len(eps)
        return n

    def sample(self, n: int, rng: random.Random, her_p: float = 0.0, prot_frac: float = 0.25,
               shaping: float = 0.0) -> list[View]:
        out = []
        for _ in range(n):
            pool = self.prot if (self.prot and (not self.main or rng.random() < prot_frac)) else self.main
            e = pool[rng.randrange(len(pool))]
            e.sampled += 1
            hs = [h for h in e.hindsight]
            if hs and her_p > 0 and rng.random() < her_p:
                h = hs[rng.randrange(len(hs))]
                goal, upto = h[0], h[1]
                upto = len(e.steps) if upto is None else int(upto)
                if upto > e.n_prefix:
                    out.append(View(e, list(goal), upto, relabeled=True, shaping=shaping,
                                    goal_sem=h[2] if len(h) > 2 else None))
                    self.sampled_steps += upto
                    continue
            out.append(View(e, list(e.goal), len(e.steps), goal_sem=e.goal_sem))
            self.sampled_steps += len(e.steps)
        return out

    def stats(self) -> dict:
        allep = list(self.main) + list(self.prot)
        reuse = [e.sampled for e in allep]
        return {"episodes": len(allep), "steps": self.steps(), "protected_episodes": len(self.prot),
                "successes": sum(e.success for e in allep), "with_hindsight_goals": sum(bool(e.hindsight) for e in allep),
                "mean_times_sampled": float(np.mean(reuse)) if reuse else 0.0,
                "replay_ratio": self.sampled_steps / max(1, sum(len(e.steps) for e in allep))}


class Archive:
    """Go-Explore-style archive of promising, still-viable states in resettable simulators. Cells are
    coarse buckets of the *verified, observable* task status (progress, time left, resources,
    milestones); each keeps the most promising state reached, its recorded history (replayed as belief
    burn-in when exploring from it) and how often it was chosen / led to success. Restored states get
    fresh exogenous randomness. Training-only: evaluation always starts from the initial state."""

    def __init__(self, cells_per_family: int = 400, seed: int = 0, investigate: float = 0.0):
        self.cells: dict[str, dict] = {}
        self.cap = cells_per_family
        self.rng = random.Random(seed)
        self.investigate = investigate
        self.min_used = {}            # reverse curriculum: minimum fraction of the horizon already used
        self.stats = {"inserted": 0, "chosen": 0, "succ_from": 0, "done_from": 0}

    @staticmethod
    def _score(st) -> float:
        res = list(st.resources.values())
        ms = st.milestones
        return st.progress + 0.25 * (sum(map(bool, ms.values())) / max(1, len(ms))) + 0.1 * (float(np.mean(res)) if res else 0.0)

    def key(self, fam: str, st) -> tuple:
        return (fam, int(st.progress * 10), int(st.time_left * 5),
                tuple(int(min(2.0, v) * 4) for v in st.resources.values()), sum(map(bool, st.milestones.values())))

    def offer(self, fam: str, env, ep: Episode, slot: dict, u: float = 0.0) -> None:
        st = env.task_status()
        if env.done or st.violated or st.time_left <= 0.05:
            return
        k = self.key(fam, st)
        fc = self.cells.setdefault(fam, {})
        sc = self._score(st)
        cur = fc.get(k)
        if cur is not None and cur["score"] >= sc:
            cur["seen"] += 1
            return
        if cur is None and len(fc) >= self.cap:  # replace the least promising, most explored cell
            worst = min(fc, key=lambda c: fc[c]["score"] - 0.01 * fc[c]["chosen"])
            if fc[worst]["score"] >= sc:
                return
            del fc[worst]
        fc[k] = {"env": env.snapshot(), "steps": list(ep.steps), "seed": ep.seed, "difficulty": ep.difficulty,
                 "prev_act": list(slot["prev_act"]), "opt": int(slot["opt"]), "elapsed": float(slot["elapsed"]),
                 "used": 1.0 - st.time_left, "score": sc, "u": float(u), "seen": 1,
                 "chosen": cur["chosen"] if cur else 0, "succ": cur["succ"] if cur else 0}
        self.stats["inserted"] += 1

    def choose(self, fam: str) -> Optional[dict]:
        fc = self.cells.get(fam)
        if not fc:
            return None
        lo = self.min_used.get(fam, 0.0)
        cand = [c for c in fc.values() if c["used"] >= lo] or list(fc.values())
        w = [(0.1 + c["score"]) * (1 + self.investigate * c["u"]) / math.sqrt(1 + c["chosen"]) for c in cand]
        c = self.rng.choices(cand, w)[0]
        c["chosen"] += 1
        self.stats["chosen"] += 1
        return c

    def result(self, cell: dict, success: bool) -> None:
        cell["succ"] += int(success)
        self.stats["succ_from"] += int(success)
        self.stats["done_from"] += 1

    def size(self) -> dict:
        return {f: len(c) for f, c in self.cells.items()}
