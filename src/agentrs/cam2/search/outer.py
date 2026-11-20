"""The outer learner: population-based architecture / learning-rule search with real candidate training.

Every candidate is trained by the ordinary CAM 2.0 inner learner (``trainer.Trainer``) on INNER-TRAINING tasks,
then evaluated (greedy, no parameter updates, asserted) on the fixed OUTER-VALIDATION task list. Selection,
promotion and the controller's rewards use only those validation measurements. The locked test split is not
importable from here (``tests/cam2/test_protocol.py``), the protocol hash is pinned when a search starts and
re-checked on resume, and candidates are data (``spec.py``) - the outer learner never executes generated code.

One frozen LLM and one environment-worker pool are shared by all candidates (candidates run one at a time:
``max_concurrent_gpu_jobs = 1`` on this machine; the container's process limit also caps browser workers).

Usage:
  PYTHONPATH=src python -m agentrs.cam2.search.outer --config configs/cam2/search.yaml [--set key=value ...]
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch

from agentrs.cam2.search.controller import make_controller
from agentrs.cam2.search.population import ASHA, Budget, Candidate, Population, paired_delta
from agentrs.cam2.search.spec import (OPERATORS, Limits, SpecError, default_spec, inherit_weights, mutate, spec_hash,
                                      validate)
from agentrs.cam2.tasks import protocol_hash, val_tasks


@dataclass
class SearchConfig:
    root: str = "runs/cam2/search/dev"
    env: str = "miniwob"
    llm: str = "Qwen/Qwen3-1.7B"
    workers: int = 16
    controller: str = "adaptive"            # adaptive | uniform | pbt
    operators: Optional[list] = None        # None = all operators (+ llm.propose when llm_proposals)
    llm_proposals: bool = False
    seed: int = 0
    b0: int = 1500                          # rung-0 environment-step budget
    eta: int = 3
    max_rung: int = 1
    origin: str = "scratch"                 # scratch | inherit (weight inheritance from the parent where shapes match)
    base: dict = field(default_factory=dict)        # seed candidate overrides {"arch": .., "learn": .., "plan": ..}
    fixed_learn: dict = field(default_factory=lambda: {"episodes_per_update": 64, "mb_episodes": 32, "warmup_steps": 10})
    max_steps: int = 10                     # environment step limit per episode (protocol, fixed)
    val_per_family: int = 4
    val_families: Optional[list] = None     # None = held-out VAL families + val seeds of TRAIN families
    budget: dict = field(default_factory=dict)
    job_minutes_cap: float = 30.0
    check_device: str = "cuda"
    reproduce_seeds: list = field(default_factory=list)   # independent seeds for the final scratch reproduction

    def to_dict(self) -> dict:
        return asdict(self)


class OuterLearner:
    def __init__(self, cfg: SearchConfig, llm=None, pool=None):
        from agentrs.cam2.backbone import load_llm
        from agentrs.cam2.envs.pool import EnvPool

        self.cfg = cfg
        self.root = Path(cfg.root)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm = llm or load_llm(cfg.llm, device=self.dev)
        self.dev = self.llm.device          # CAM candidates live where the frozen LLM lives
        self.pool = pool or EnvPool(cfg.env, cfg.workers, {"max_steps": cfg.max_steps})
        self.asha = ASHA(cfg.b0, cfg.eta, cfg.max_rung)
        ops = list(cfg.operators or OPERATORS)
        if cfg.llm_proposals and "llm.propose" not in ops:
            ops.append("llm.propose")
        self.ctrl = make_controller(cfg.controller, seed=cfg.seed, operators=ops,
                                    **({"max_rung": cfg.max_rung} if cfg.controller == "adaptive" else {}))
        self.proposer = None
        if "llm.propose" in ops:
            from agentrs.cam2.search.proposer import LLMProposer

            self.proposer = LLMProposer(self.llm)
        self.rng = random.Random(cfg.seed)
        self.tasks = val_tasks(cfg.env, cfg.val_per_family, cfg.val_families)
        if (self.root / "population.json").exists():
            self.pop = Population.load(str(self.root))
            if self.pop.meta.get("protocol_hash") != protocol_hash():
                raise RuntimeError("the evaluation protocol changed since this search started; refusing to resume")
            if self.pop.meta.get("val_tasks") != [t.key() for t in self.tasks]:
                raise RuntimeError("the outer-validation task list changed since this search started")
            self.rng.setstate(tuple(self.pop.meta["rng"][0:1]) + (tuple(self.pop.meta["rng"][1]),) + (self.pop.meta["rng"][2],))
            print(f"[outer] resumed {len(self.pop.order)} candidates, budget used "
                  f"{self.pop.budget.used_gpu_hours:.2f} GPU-h / {self.pop.budget.used_env_steps} steps", flush=True)
        else:
            self.pop = Population(str(self.root), Budget(**cfg.budget),
                                  meta={"config": cfg.to_dict(), "protocol_hash": protocol_hash(),
                                        "val_tasks": [t.key() for t in self.tasks], "llm": self.llm.describe()})
        self.limits = Limits(max_params=self.pop.budget.max_params)

    # ------------------------------------------------------------------ helpers
    def save(self) -> None:
        self.pop.meta["rng"] = self.rng.getstate()
        self.pop.save()

    def admit(self, c: Candidate) -> bool:
        try:
            rep = validate(c.spec, self.llm.pooled_dim, self.llm.hidden, self.limits, device=self.cfg.check_device)
            c.validation = rep
            return True
        except SpecError as e:
            c.status, c.failure = "failed", f"admission: {e}"
            self.pop.log("rejected", cid=c.cid, reason=str(e))
            return False
        except Exception as e:  # noqa: BLE001  - e.g. CUDA OOM in the check is a resource failure, not a crash
            c.status, c.failure = "failed", f"admission error: {type(e).__name__}: {e}"
            self.pop.log("rejected", cid=c.cid, reason=c.failure)
            return False
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    def run_config(self, c: Candidate, rung: int):
        from agentrs.cam2.trainer import RunConfig

        s = c.spec
        return RunConfig(run_dir=str(self.pop.run_dir(c.cid)), env=self.cfg.env, llm=self.llm.name,
                         workers=self.cfg.workers, seed=c.seed, env_steps=self.asha.budget(rung),
                         max_minutes=max(1.0, min(self.cfg.job_minutes_cap, self.pop.budget.remaining_minutes())),
                         max_steps=self.cfg.max_steps, eval_every=0, ckpt_every=5, cam=dict(s["arch"]),
                         learn={**self.cfg.fixed_learn, **s["learn"]}, plan=dict(s["plan"]), phase=f"search:{c.cid}")

    def train_rung(self, c: Candidate, rung: int) -> None:
        """Train ``c`` up to the rung's budget (continuing its own checkpoint) and evaluate it on outer validation."""
        from agentrs.cam2.trainer import Trainer

        c.status = "running"
        self.save()
        t0 = time.time()
        rc = self.run_config(c, rung)
        tr = None
        try:
            tr = Trainer(rc, llm=self.llm, pool=self.pool, device=self.dev)
            # the trainer's wall-clock cap counts its whole history; give this rung its own allowance
            tr.rc.max_minutes = tr.counters["wall_s"] / 60 + rc.max_minutes
            if c.origin == "inherit" and rung == 0 and tr.counters["updates"] == 0 and c.parent:
                pck = self.pop.run_dir(c.parent) / "ckpt" / "latest.pt"
                if pck.exists():
                    sd = torch.load(pck, map_location=self.dev, weights_only=False)["cam"]
                    c.inherit = inherit_weights(tr.cam, sd)   # optimizer state: fresh (see inherit_weights)
            steps0 = tr.counters["env_steps"]
            summ = tr.train()
            train_s = time.time() - t0
            ev = tr.evaluate(self.tasks, greedy=True, label=f"val_r{rung}")
            if not summ["llm_fingerprint_unchanged"]:
                raise RuntimeError("frozen LLM weights changed during candidate training")
            wall = time.time() - t0
            res = {"budget": self.asha.budget(rung), "trained_env_steps": tr.counters["env_steps"],
                   "ckpt": str(Path(rc.run_dir) / "ckpt" / "latest.pt"),
                   "val": {k: ev[k] for k in ("success", "ci95", "score", "per_family", "mean_len", "invalid_rate",
                                              "errors", "dropped", "n")},
                   "val_episodes": ev["episodes"],
                   "compute": {"wall_s": wall, "train_s": train_s, "eval_s": ev["eval_s"], "gpu_hours": wall / 3600,
                               "env_steps": tr.counters["env_steps"] - steps0 + ev["env_steps"],
                               "llm_gen_tokens": tr.counters["llm_gen_tokens"], "llm_encode_tokens": tr.counters["llm_encode_tokens"],
                               "params": tr.cam.n_params()["total"],
                               "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0},
                   "train": self._train_curve(Path(rc.run_dir))}
            c.results[str(rung)] = res
            c.rung = rung
            c.status = "ready"
            self.pop.budget.used_gpu_hours += wall / 3600
            self.pop.budget.used_env_steps += res["compute"]["env_steps"]
            if rung > 0 and str(rung - 1) in c.results:
                self.pop.log("rung_gain", cid=c.cid, rung=rung, gain=res["val"]["success"] - c.score(rung - 1),
                             cost_h=wall / 3600)
            print(f"[outer] {c.cid} rung {rung}: val={res['val']['success']:.3f} ci={res['val']['ci95']} "
                  f"steps={res['trained_env_steps']} wall={wall/60:.1f}min", flush=True)
        except Exception as e:  # noqa: BLE001  - a crashing candidate is recorded as failed; the search continues
            c.status, c.failure = "failed", f"training: {type(e).__name__}: {e}"
            self.pop.budget.used_gpu_hours += (time.time() - t0) / 3600
            self.pop.log("train_failed", cid=c.cid, rung=rung, error=traceback.format_exc()[-2000:])
            print(f"[outer] {c.cid} rung {rung} FAILED: {e}", flush=True)
        finally:
            if tr is not None:
                tr.close()
                del tr
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.save()

    @staticmethod
    def _train_curve(run_dir: Path) -> list:
        out = []
        p = run_dir / "log.jsonl"
        if p.exists():
            for ln in p.read_text().splitlines():
                r = json.loads(ln)
                if r.get("type") == "update":
                    out.append({k: r.get(k) for k in ("env_steps", "train_success", "train_success_recent", "loss",
                                                      "update_rel_norm", "approx_kl", "wm_reliability")})
        return out

    # ------------------------------------------------------------------ proposal
    def new_child(self) -> Optional[Candidate]:
        parent, op = self.ctrl.propose(self.pop)
        ctx = {"population_rules": list({c.spec["learn"]["rule"] for c in self.pop.cands.values()})}
        seed = self.rng.randrange(1, 2**31)
        raw = None
        try:
            if op == "llm.propose":
                spec, desc, raw = self.proposer.propose(parent.spec, self.pop, self.rng)
            else:
                spec, desc = mutate(parent.spec, op, self.rng, ctx)
        except SpecError as e:
            c = self.pop.add(parent.spec, parent.cid, op, f"(no valid mutation: {e})", seed, self.cfg.origin)
            c.status, c.failure = "failed", f"mutation: {e}"
            self._record(c, parent, raw=raw)
            return None
        c = self.pop.add(spec, parent.cid, op, desc, seed, self.cfg.origin)
        if raw is not None:
            self.pop.log("llm_proposal", cid=c.cid, raw=raw[:600])
        return c

    def _record(self, c: Candidate, parent: Candidate, raw: Optional[str] = None) -> None:
        h = {"child": c.cid, "parent": parent.cid, "op": c.op, "mutation": c.mutation,
             "context": self.ctrl.context(self.pop, parent).tolist() if parent.rung >= 0 else None,
             "outcome": "evaluated" if c.rung >= 0 else "failed", "failure": c.failure}
        if c.rung >= 0:
            h.update(child_score=c.score(0), parent_score=parent.score(0), delta=c.score(0) - parent.score(0),
                     cost_h=c.results["0"]["compute"]["gpu_hours"],
                     paired=paired_delta(c.results["0"]["val_episodes"], parent.results["0"]["val_episodes"]))
        if h["context"] is None:
            h.pop("context")
        self.pop.hypotheses.append(h)

    # ------------------------------------------------------------------ main loop
    def seed_population(self) -> None:
        if self.pop.order:
            return
        spec = default_spec(self.cfg.base)
        c = self.pop.add(spec, None, "seed", "hand-designed CAM 2.0 configuration", self.cfg.seed + 1)
        if not self.admit(c):
            raise RuntimeError(f"the seed candidate fails admission: {c.failure}")
        self.save()

    def est_minutes(self, rung: int) -> float:
        """Expected wall minutes of a job that trains to ``rung`` (from completed jobs; rung r+1 continues rung r)."""
        import numpy as np

        w0 = [c.results["0"]["compute"]["wall_s"] / 60 for c in self.pop.cands.values() if "0" in c.results]
        if not w0:
            return 0.0
        if rung == 0:
            return float(np.median(w0))
        wr = [c.results[str(rung)]["compute"]["wall_s"] / 60 for c in self.pop.cands.values() if str(rung) in c.results]
        return float(np.median(wr)) if wr else float(np.median(w0)) * (self.cfg.eta ** rung - self.cfg.eta ** (rung - 1))

    def run(self) -> dict:
        self.seed_population()
        while True:
            why = self.pop.budget.exhausted(len(self.pop.order))
            if why:
                self.pop.log("stop", reason=why)
                print(f"[outer] stopping: {why}", flush=True)
                break
            nxt = self.ctrl.decide_job(self.pop, self.asha) if not any(
                c.status == "new" for c in self.pop.cands.values()) else ("new", None)
            need = self.est_minutes(nxt[1][0] + 1 if nxt[0] == "promote" else 0)
            if need and self.pop.budget.remaining_minutes() < 1.1 * need:
                # never start a job the budget cannot finish: a truncated rung would be compared unfairly
                why = (f"remaining budget {self.pop.budget.remaining_minutes():.1f} min < expected job "
                       f"{need:.1f} min")
                self.pop.log("stop", reason=why)
                print(f"[outer] stopping: {why}", flush=True)
                break
            pending = [c for c in self.pop.cands.values() if c.status == "new"]
            if pending:                         # admitted but not yet trained (seed, or resumed mid-job)
                c = pending[0]
                if not c.validation and not self.admit(c):
                    self.save()
                    continue
                self.train_rung(c, 0)
                if c.parent:
                    self._record(c, self.pop.cands[c.parent])
                continue
            kind, item = self.ctrl.decide_job(self.pop, self.asha)
            seed = self.pop.cands[self.pop.order[0]]
            if kind == "promote" and item[0] + 1 > seed.rung >= 0 and seed.status == "ready" \
                    and seed.cid != item[1].cid and (seed.rung) not in seed.promoted_from:
                # reference promotion: the hand-designed seed is trained to every rung a candidate reaches, so the
                # champion comparison is always made at a matched budget
                kind, item = "promote", (seed.rung, seed)
                self.pop.log("reference_promotion", cid=seed.cid, rung=seed.rung + 1)
            if kind == "promote":
                r, c = item
                c.promoted_from.append(r)
                self.pop.log("promote", cid=c.cid, rung=r + 1)
                self.train_rung(c, r + 1)
                continue
            if not self.pop.budget.can_create(len(self.pop.order)):
                self.pop.log("stop", reason="candidate budget reached and nothing promotable")
                print("[outer] stopping: candidate budget reached and nothing promotable", flush=True)
                break
            c = self.new_child()
            if c is None:
                self.save()
                continue
            if not self.admit(c):
                self._record(c, self.pop.cands[c.parent])
                self.save()
                continue
            self.train_rung(c, 0)
            self._record(c, self.pop.cands[c.parent])
            self.save()
        res = self.select_champion()
        self.save()
        return res

    # ------------------------------------------------------------------ selection / promotion criteria
    def select_champion(self) -> dict:
        """Champion = best outer-validation success at the highest rung reached by >= 1 candidate, subject to the
        promotion criteria against the seed (hand-designed) candidate at the same rung:
          * improvement: paired delta over the same validation tasks > 0;
          * no catastrophic forgetting: success on TRAIN-family validation instances not more than 0.10 below seed;
          * compute: cost per rung <= 1.5x the seed's unless the paired improvement's CI excludes 0;
          * integrity: invalid-action rate and environment-error count not anomalous (<= seed + 0.05 / no drops).
        A candidate that fails a criterion is reported but not promoted; the seed stays champion."""
        from agentrs.cam2.envs.miniwob import TRAIN_FAMILIES

        seed = self.pop.cands[self.pop.order[0]]
        ev = [c for c in self.pop.cands.values() if c.rung >= 0 and c.status in ("ready", "stopped")]
        if not ev:
            return {"champion": None, "note": "no candidate completed training"}
        top_rung = max(c.rung for c in ev)
        at = [c for c in ev if c.rung >= top_rung]
        best = max(at, key=lambda c: c.score(top_rung))
        report = {"top_rung": top_rung, "best": best.cid, "best_score": best.score(top_rung), "seed": seed.cid,
                  "seed_score": seed.score(top_rung) if str(top_rung) in seed.results else None, "criteria": {}}
        champion = seed.cid
        if best.cid != seed.cid and str(top_rung) in seed.results:
            rb, rs = best.results[str(top_rung)], seed.results[str(top_rung)]
            pd = paired_delta(rb["val_episodes"], rs["val_episodes"])
            trf = lambda r: [v for k, v in r["val"]["per_family"].items() if k in TRAIN_FAMILIES]  # noqa: E731
            tb, ts = trf(rb), trf(rs)
            forget = (sum(ts) / max(1, len(ts))) - (sum(tb) / max(1, len(tb)))
            cost_ratio = rb["compute"]["gpu_hours"] / max(1e-9, rs["compute"]["gpu_hours"])
            crit = {"paired": pd, "improves": pd["delta"] > 0, "significant": pd["ci95"][0] > 0,
                    "train_family_drop": forget, "no_forgetting": forget <= 0.10, "cost_ratio": cost_ratio,
                    "compute_ok": cost_ratio <= 1.5 or pd["ci95"][0] > 0,
                    "integrity_ok": (rb["val"]["invalid_rate"] <= rs["val"]["invalid_rate"] + 0.05
                                     and len(rb["val"]["dropped"]) <= len(rs["val"]["dropped"]) + 2)}
            report["criteria"] = crit
            if crit["improves"] and crit["no_forgetting"] and crit["compute_ok"] and crit["integrity_ok"]:
                champion = best.cid
        elif best.cid != seed.cid:
            report["criteria"] = {"note": "seed was not evaluated at the top rung; comparison at rung 0 only"}
        self.pop.champion = champion
        report["champion"] = champion
        report["champion_is_seed"] = champion == seed.cid
        self.pop.log("champion", **{k: v for k, v in report.items() if k != "criteria"})
        self.pop.meta["selection"] = report
        return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="CAM 2.0 outer learner (architecture / learning-rule search)")
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    cfg: dict = {}
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
    ol = OuterLearner(SearchConfig(**cfg))
    try:
        res = ol.run()
        print(json.dumps(res, indent=1, default=str))
    finally:
        ol.pool.close()


if __name__ == "__main__":
    main()
