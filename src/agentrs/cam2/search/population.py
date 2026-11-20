"""Candidate records, persistent population state, budgets and asynchronous successive halving (ASHA).

* ``Candidate``: immutable identity (``cid`` = hash of spec, parent, seed and creation index; never reused), the
  typed spec, parent lineage, full mutation history, seeds, origin (``scratch`` / ``inherit``), per-rung results
  (training budget, checkpoint path, outer-validation result, compute consumed) and status.
* ``Population``: all candidates ever created (failed and successful), the controller's hypothesis history and the
  budget ledger, persisted atomically to ``population.json`` after every change; ``Population.load`` resumes.
* ``ASHA``: rung r trains a candidate to ``b0 * eta**r`` environment steps (continuing its own checkpoint, so the
  total is the budget, not an additional one). A candidate is promoted from rung r when it ranks in the top
  ``1/eta`` of the candidates *completed* at rung r (asynchronous: no waiting for a full generation).
* ``Budget``: hard limits on GPU hours, environment steps, candidate count, parameter count, wall clock and
  concurrent GPU jobs; ``exhausted()`` is checked before every job and jobs are themselves capped so the search
  stops cleanly with every result checkpointed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Budget:
    gpu_hours: float = 4.0
    env_steps: int = 200_000
    max_candidates: int = 24
    max_params: int = 120_000_000
    wall_hours: float = 4.0
    max_concurrent_gpu_jobs: int = 1
    # consumed (ledger)
    used_gpu_hours: float = 0.0
    used_env_steps: int = 0
    started_at: float = field(default_factory=time.time)
    wall_offset_s: float = 0.0     # wall time consumed before the last resume

    def wall_hours_used(self) -> float:
        return (self.wall_offset_s + time.time() - self.started_at) / 3600

    def exhausted(self, n_candidates: int) -> Optional[str]:
        if self.used_gpu_hours >= self.gpu_hours:
            return f"GPU-hour budget reached ({self.used_gpu_hours:.2f}/{self.gpu_hours})"
        if self.used_env_steps >= self.env_steps:
            return f"environment-step budget reached ({self.used_env_steps}/{self.env_steps})"
        if self.wall_hours_used() >= self.wall_hours:
            return f"wall-clock budget reached ({self.wall_hours_used():.2f}/{self.wall_hours} h)"
        return None

    def can_create(self, n_candidates: int) -> bool:
        return n_candidates < self.max_candidates

    def remaining_minutes(self) -> float:
        return 60 * min(self.gpu_hours - self.used_gpu_hours, self.wall_hours - self.wall_hours_used())


@dataclass
class Candidate:
    cid: str
    spec: dict
    parent: Optional[str]
    op: str                                  # operator that created it ("seed", "arch.width", "llm.propose", ...)
    mutation: str                            # human-readable description of the change
    history: list                            # full lineage of (op, description) from the seed
    seed: int
    origin: str = "scratch"                  # scratch | inherit
    created: float = field(default_factory=time.time)
    status: str = "new"                      # new | running | ready | failed | stopped
    rung: int = -1                           # highest completed rung
    results: dict = field(default_factory=dict)   # rung (str) -> {budget, ckpt, val, compute, train}
    failure: Optional[str] = None
    validation: dict = field(default_factory=dict)  # admission-check report
    inherit: dict = field(default_factory=dict)     # weight-inheritance report (origin == inherit)
    promoted_from: list = field(default_factory=list)  # rungs from which it was promoted

    def score(self, rung: Optional[int] = None) -> Optional[float]:
        r = self.rung if rung is None else rung
        res = self.results.get(str(r))
        return None if res is None else res["val"]["success"]

    def family(self) -> str:
        """Coarse architecture family (for diversity accounting)."""
        a = self.spec["arch"]
        rule = json.loads(self.spec["learn"]["rule"])
        rname = rule.get("name") or "r" + hashlib.sha1(json.dumps(rule["policy"], sort_keys=True).encode()).hexdigest()[:4]
        return f"{a['core']}/mem{int(a['mem_k'] > 0)}/plan{int(self.spec['plan']['enabled'])}/{rname}"


def make_cid(spec: dict, parent: Optional[str], seed: int, index: int) -> str:
    h = hashlib.sha1(json.dumps({"spec": spec, "parent": parent, "seed": seed, "index": index},
                                sort_keys=True, default=str).encode()).hexdigest()[:10]
    return f"c{index:03d}-{h}"


class Population:
    def __init__(self, root: str, budget: Budget, meta: Optional[dict] = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.budget = budget
        self.cands: dict[str, Candidate] = {}
        self.order: list[str] = []
        self.hypotheses: list = []           # controller history: every proposal and its measured outcome
        self.events: list = []
        self.meta = meta or {}
        self.champion: Optional[str] = None

    # ------------------------------------------------------------------ persistence
    def path(self) -> Path:
        return self.root / "population.json"

    def save(self) -> None:
        d = {"budget": asdict(self.budget) | {"wall_offset_s": self.budget.wall_offset_s + time.time() - self.budget.started_at,
                                              "started_at": time.time()},
             "candidates": [asdict(self.cands[c]) for c in self.order], "hypotheses": self.hypotheses,
             "events": self.events[-2000:], "meta": self.meta, "champion": self.champion}
        tmp = self.path().with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=1, default=str))
        os.replace(tmp, self.path())

    @classmethod
    def load(cls, root: str) -> "Population":
        d = json.loads((Path(root) / "population.json").read_text())
        b = Budget(**d["budget"])
        b.started_at = time.time()   # wall_offset_s holds the time consumed before this resume
        p = cls(root, b, d.get("meta"))
        for c in d["candidates"]:
            cand = Candidate(**c)
            if cand.status == "running":     # interrupted mid-job: the rung is re-run from its checkpoint
                cand.status = "ready" if cand.rung >= 0 else "new"
                if cand.rung >= 0 and cand.promoted_from and cand.promoted_from[-1] == cand.rung:
                    cand.promoted_from.pop()     # the interrupted promotion becomes eligible again
            p.cands[cand.cid] = cand
            p.order.append(cand.cid)
        p.hypotheses, p.events, p.champion = d.get("hypotheses", []), d.get("events", []), d.get("champion")
        return p

    def log(self, kind: str, **kw) -> None:
        self.events.append({"t": time.time(), "kind": kind, **kw})

    # ------------------------------------------------------------------ candidates
    def add(self, spec: dict, parent: Optional[str], op: str, mutation: str, seed: int, origin: str = "scratch") -> Candidate:
        idx = len(self.order)
        cid = make_cid(spec, parent, seed, idx)
        assert cid not in self.cands
        hist = (list(self.cands[parent].history) if parent else []) + [[op, mutation]]
        c = Candidate(cid=cid, spec=spec, parent=parent, op=op, mutation=mutation, history=hist, seed=seed, origin=origin)
        self.cands[cid] = c
        self.order.append(cid)
        self.log("create", cid=cid, parent=parent, op=op, mutation=mutation)
        return c

    def run_dir(self, cid: str) -> Path:
        return self.root / "candidates" / cid

    def completed(self, rung: int) -> list:
        return [c for c in self.cands.values() if str(rung) in c.results]

    def alive(self) -> list:
        return [c for c in self.cands.values() if c.status in ("ready",)]


class ASHA:
    """Asynchronous successive halving over environment-step budgets."""

    def __init__(self, b0: int, eta: int = 3, max_rung: int = 2, min_completed: int = 2):
        self.b0, self.eta, self.max_rung, self.min_completed = b0, eta, max_rung, min_completed

    def budget(self, rung: int) -> int:
        return int(self.b0 * self.eta ** rung)

    def promotable(self, pop: Population) -> list:
        """Candidates eligible for promotion, best first: in the top 1/eta of those completed at their rung."""
        out = []
        for r in range(self.max_rung):
            done = [c for c in pop.completed(r) if c.status == "ready" or str(r + 1) in c.results]
            if len(done) < self.min_completed:
                continue
            k = max(1, len(done) // self.eta)
            top = sorted(done, key=lambda c: (-c.score(r), c.created))[:k]
            for c in top:
                if c.status == "ready" and c.rung == r and r not in c.promoted_from:
                    out.append((r, c))
        out.sort(key=lambda rc: (-rc[0], -rc[1].score(rc[0])))
        return out


def paired_delta(a_eps: list, b_eps: list, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired comparison of two evaluations on the same task list (keyed by task): mean difference in success and a
    bootstrap 95% CI over tasks. Only tasks present in both (environment failures are dropped pairwise)."""
    import numpy as np

    A = {e["task"]: float(e["success"]) for e in a_eps}
    B = {e["task"]: float(e["success"]) for e in b_eps}
    keys = sorted(set(A) & set(B))
    if not keys:
        return {"n": 0, "delta": float("nan"), "ci95": [float("nan"), float("nan")]}
    d = np.array([A[k] - B[k] for k in keys])
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return {"n": len(keys), "delta": float(d.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}
