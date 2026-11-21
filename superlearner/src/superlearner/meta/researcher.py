"""The Meta-Researcher: an automated loop that tries to find a better learning algorithm for the inner learner.

One round (`MetaResearcher.round`):

  A diagnose   read the champion's learning curves (trusted) and training diagnostics; name the bottleneck
  B hypothesize generate a pool of proposals (LLM-written functions, composed rule programs, learned-rule
               parameters); the trainable surrogate (or, for the control arm, a uniform random choice) picks which
               to spend an experiment on
  C implement  create the candidate's own source tree with the proposed code
  D verify     protected files unchanged, source parses, then a self-test INSIDE the confinement on real episodes
               (likelihood recomputation, finite losses, weights change); invalid candidates are rejected here
  E train      a fresh controller is trained by the candidate's algorithm, confined, against the trusted env hub,
               under a fixed experience budget enforced by the hub
  F compare    learning curves from the hub's own results file, against the champion's runs on the same seeds,
               budgets, tasks, demonstrations and hardware; screening -> development -> confirmation
  G analyze    differences with bootstrap confidence intervals on sample efficiency (AULC), final capability,
               training-time success, wall-clock; Pareto status
  H decide     promote only if the predeclared criterion holds at the confirmation stage; otherwise archive
  I learn      the outcome is added to the ledger; the surrogate is refitted; the learned-rule search is updated
  J repeat

State is saved after every experiment, so a search can be resumed; a candidate never overwrites the champion's
tree or measurements, and nothing is deleted.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import pickle
import threading
import time

import numpy as np

from ..train.trainer import code_hashes, git_commit, merge
from ..world.hub import start_hub
from ..world.tasks import SPLITS, task_list
from . import proposals as P
from .compare import compare, load_trusted, pareto, run_measures, verdict
from .ledger import Ledger
from .surrogate import Surrogate
from .workspace import PKG, create_candidate, require_isolation, run_isolated, verify

DEFAULTS = {
    "name": "meta", "seed": 0, "rounds": 4, "pool": 6, "per_round": 1, "selection": "surrogate",   # or "random"
    "sources": {"rule_space": 4, "llm": 1, "learned_rule": 1},
    "max_parallel": 3, "primary": "aulc_eval", "screen_margin": 0.03, "promote_margin": 0.0, "max_final_drop": 0.05,
    "inner": {
        "workers": 6, "llm": {"kind": "none"}, "eval_every": 2,
        "model": {"d": 128, "layers": 2, "heads": 4, "ff": 256, "bag_dim": 48, "hid": 128},
        "agent": {"use_llm": False}, "core": {"max_decisions": 24},
        "algo": {"name": "ppo", "lr": 0.0003, "epochs": 3, "minibatch_eps": 16, "target_kl": 0.05},
        "curriculum": {"adaptive": True, "world_families": ["tiny"],
                       "arms": [{"family": "found"}, {"family": "fund", "prereq": ["found/0/scratch"]},
                                {"family": "list", "start": "deployed"}, {"family": "service", "start": "listed"}]},
        "demos": {"episodes": 0, "bc_epochs": 0},
    },
    "eval_arms": ["found/0/scratch", "fund/0/scratch", "list/0/deployed", "service/0/listed"],
    "eval_world_families": ["tiny", "entrenched_incumbents"],
    "stages": {
        "screen": {"seeds": [1], "iters": 8, "episodes_per_iter": 32, "budget": 5000, "eval_n": 16, "timeout_s": 1500},
        "develop": {"seeds": [1, 2, 3], "iters": 12, "episodes_per_iter": 32, "budget": 8000, "eval_n": 24, "timeout_s": 2400},
        "confirm": {"seeds": [1, 2, 3, 4, 5], "iters": 12, "episodes_per_iter": 32, "budget": 8000, "eval_n": 24, "timeout_s": 2400},
    },
}


class MetaResearcher:
    def __init__(self, cfg: dict, run_dir: str, llm=None, resume: bool = False):
        self.cfg = merge(DEFAULTS, cfg)
        self.dir = os.path.abspath(run_dir)
        for d in ("candidates", "trusted", "sock"):
            os.makedirs(os.path.join(self.dir, d), exist_ok=True)
        os.chmod(self.dir, 0o755)
        os.chmod(os.path.join(self.dir, "candidates"), 0o755)
        os.chmod(os.path.join(self.dir, "sock"), 0o755)
        self.llm = llm
        c = self.cfg
        self.rng = np.random.default_rng(c["seed"])
        self.ledger = Ledger(os.path.join(self.dir, "ledger.jsonl"))
        self.surrogate = Surrogate(llm.embed if llm is not None else None, seed=c["seed"])
        self.rule_space = P.RuleSpace(c["seed"])
        self.learned = P.LearnedRule(c["seed"])
        self.llm_proposer = P.LLMProposer(llm, c["seed"]) if (llm is not None and c["sources"].get("llm")) else None
        self.champion = {"id": "champion-0", "changes": {}, "genome": P.BASE_GENOME, "hypothesis": "baseline: recurrent PPO with GAE",
                         "measures": {}, "promoted_at": 0}
        self.round_n = 0
        self.isolation = require_isolation()          # fail closed: no confinement, no candidate execution
        self._sem = threading.Semaphore(int(c["max_parallel"]))     # confined training runs in flight
        self._lock = threading.RLock()                               # ledger / state / surrogate
        self._base_lock = threading.Lock()
        self.state_path = os.path.join(self.dir, "state.json")
        if resume and os.path.exists(self.state_path):
            self.load_state()
        else:
            json.dump(self.cfg, open(os.path.join(self.dir, "config.json"), "w"), indent=1)
            json.dump({"code": code_hashes(), "git": git_commit(), "isolation": self.isolation, "started": time.time()},
                      open(os.path.join(self.dir, "meta.json"), "w"), indent=1, default=str)

    # ------------------------------------------------------------------ state
    def save_state(self) -> None:
        s = {"champion": self.champion, "round": self.round_n, "surrogate": self.surrogate.state_dict(),
             "learned": self.learned.state_dict(), "rng": self.rng.bit_generator.state,
             "rule_rng": self.rule_space.rng.bit_generator.state}
        tmp = self.state_path + ".tmp"
        json.dump(s, open(tmp, "w"), default=str)
        os.replace(tmp, self.state_path)

    def load_state(self) -> None:
        s = json.load(open(self.state_path))
        self.champion, self.round_n = s["champion"], s["round"]
        self.champion["measures"] = {st: {int(k): v for k, v in d.items()} for st, d in self.champion["measures"].items()}
        self.surrogate.load_state_dict(s["surrogate"])
        self.learned.load_state_dict(s["learned"])
        self.rng.bit_generator.state = s["rng"]
        self.rule_space.rng.bit_generator.state = s["rule_rng"]

    # ------------------------------------------------------------------ shared demonstrations
    def prepare_demos(self) -> str | None:
        """Collect the demonstration file every candidate warm-starts from (trusted code, done once).
        Demonstrations come from training-split tasks; candidates receive the file read-only."""
        d = self.cfg["inner"].get("demos") or {}
        if not int(d.get("episodes", 0)) or not int(d.get("bc_epochs", 0)):
            return None
        path = os.path.join(self.dir, "demos", "demos.pkl")
        if not os.path.exists(path):
            from ..train.trainer import Trainer
            cfg = merge(self.cfg["inner"], {"name": "demos", "device": "cpu", "workers": int(self.cfg["inner"].get("workers", 4)) * int(self.cfg["max_parallel"])})
            tr = Trainer(cfg, os.path.join(self.dir, "demos"))
            try:
                prov = tr.collect_demos(int(d["episodes"]))
            finally:
                tr.close()
            self.ledger.append({"kind": "demonstrations", "provenance": {k: v for k, v in prov.items() if k != "tasks"}})
        os.chmod(os.path.dirname(path), 0o755)
        os.chmod(path, 0o644)
        self.cfg["demos_path"] = path
        return path

    # ------------------------------------------------------------------ tasks and configs
    def eval_tasks(self, stage: str) -> list:
        st = self.cfg["stages"][stage]
        arms, fams = self.cfg["eval_arms"], [f for f in self.cfg["eval_world_families"] if f in SPLITS["select"][2]]
        return task_list(arms, fams, int(st["eval_n"]), "select", 4000)

    def inner_config(self, stage: str, seed: int) -> dict:
        st = self.cfg["stages"][stage]
        return merge(self.cfg["inner"], {"name": f"{stage}_s{seed}", "seed": int(seed), "iters": int(st["iters"]),
                                         "episodes_per_iter": int(st["episodes_per_iter"]), "ckpt_every": 0,
                                         "traj_per_iter": 0, "device": "cpu"})

    # ------------------------------------------------------------------ candidates
    def candidate_dir(self, cid: str) -> str:
        return os.path.join(self.dir, "candidates", cid)

    def make_candidate(self, cid: str, changes: dict) -> dict:
        rec = create_candidate(self.candidate_dir(cid), changes)
        for dp, dns, fns in os.walk(rec["path"]):          # readable by the unprivileged account, not writable
            os.chmod(dp, 0o755)
            for fn in fns:
                os.chmod(os.path.join(dp, fn), 0o644)
        return rec

    def run_one(self, cid: str, stage: str, seed: int, selftest: bool = False) -> dict:
        """One confined run of a candidate against its own trusted env hub. Returns trusted measures + diagnostics."""
        cdir = self.candidate_dir(cid)
        tag = "selftest" if selftest else f"{stage}_s{seed}"
        work = os.path.join(cdir, "work", tag)
        os.makedirs(work, exist_ok=True)
        tdir = os.path.join(self.dir, "trusted", cid)
        os.makedirs(tdir, exist_ok=True)
        results = os.path.join(tdir, f"{tag}.jsonl")
        if os.path.exists(results):
            os.replace(results, results + f".old{int(time.time())}")     # never silently mix two runs
        sock = os.path.join(self.dir, "sock", hashlib.sha1(f"{cid}{tag}".encode()).hexdigest()[:12] + ".sock")
        st = self.cfg["stages"][stage]
        cfg = self.inner_config(stage, seed)
        budget = int(st["budget"])
        hub = start_hub(sock, None if selftest else self.eval_tasks(stage), cfg=cfg.get("core"), eval_cfg=cfg.get("core"),
                        results_path=results, budget=None if selftest else {"train_decisions": budget})
        job = {"stage": "selftest" if selftest else "train", "config": cfg, "env_socket": sock, "workdir": work,
               "eval_n": 0 if selftest else int(st["eval_n"]), "llm": cfg.get("llm"),
               "demos_path": self.cfg.get("demos_path"), "update_threads": 4}
        job_path = os.path.join(work, "job.json")
        json.dump(job, open(job_path, "w"), indent=1)
        t0 = time.time()
        self._sem.acquire()
        try:
            extra = (os.path.dirname(self.cfg["demos_path"]),) if self.cfg.get("demos_path") else ()
            res = run_isolated("superlearner.meta.candidate_run", [job_path], os.path.join(cdir, "src"), work,
                               timeout_s=300 if selftest else float(st["timeout_s"]), gpu=False, extra_read=extra,
                               log_path=os.path.join(work, "log.txt"))
        finally:
            self._sem.release()
            hub.terminate()
            hub.join(timeout=10)
            try:
                os.unlink(sock)
            except OSError:
                pass
        out = {"returncode": res["returncode"], "timed_out": res["timed_out"], "wall_s": round(time.time() - t0, 1)}
        rp = os.path.join(work, ("selftest" if selftest else "train") + "_result.json")
        untrusted = json.load(open(rp)) if os.path.exists(rp) else {"ok": False, "error": res["stdout"][-800:]}
        out["untrusted"] = {k: untrusted.get(k) for k in ("ok", "error", "checks", "max_logp_error", "iters", "stopped",
                                                           "update_s", "rollout_s", "worker_errors", "params")}
        if selftest:
            out["ok"] = bool(res["returncode"] == 0 and untrusted.get("ok"))
            return out
        m = run_measures(load_trusted(results), budget)
        out.update(m)
        out["ok"] = bool(res["returncode"] == 0 and m["train_decisions"] >= 0.5 * min(budget, int(st["iters"]) * int(st["episodes_per_iter"]) * 4)
                         and m["eval_rounds"] >= 2)
        out["diag"] = self._diagnostics(os.path.join(work, "run", "metrics.jsonl"))
        return out

    @staticmethod
    def _diagnostics(path: str) -> dict:
        """Training diagnostics written by the (untrusted) learner: used for diagnosis text only, never for scores."""
        rows = []
        if os.path.exists(path):
            for line in open(path):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("phase") == "rl":
                    rows.append(r)
        if not rows:
            return {}

        def mean(k, part):
            xs = [r[k] for r in part if isinstance(r.get(k), (int, float))]
            return round(float(np.mean(xs)), 4) if xs else None
        h = max(1, len(rows) // 2)
        return {k: [mean(k, rows[:h]), mean(k, rows[h:])] for k in
                ("kl", "entropy", "explained_variance", "grad_norm", "clip_frac", "v_loss", "pol_loss", "param_delta",
                 "train_valid_rate", "train_success")} | {"early_stops": int(sum(bool(r.get("early_stop")) for r in rows)),
                                                           "iters": len(rows)}

    def run_stage(self, cid: str, stage: str, seeds: list[int] | None = None) -> dict:
        seeds = list(seeds if seeds is not None else self.cfg["stages"][stage]["seeds"])
        out: dict = {}
        with cf.ThreadPoolExecutor(max_workers=int(self.cfg["max_parallel"])) as ex:
            futs = {ex.submit(self.run_one, cid, stage, s): s for s in seeds}
            for f in cf.as_completed(futs):
                s = futs[f]
                try:
                    out[s] = f.result()
                except Exception as e:  # noqa: BLE001
                    out[s] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return out

    def baseline(self, stage: str, seeds: list[int] | None = None) -> dict:
        """Champion measurements for a stage (run once per seed, cached, re-used for every comparison)."""
        seeds = list(seeds if seeds is not None else self.cfg["stages"][stage]["seeds"])
        with self._base_lock:
            cid = self.champion["id"]
            if not os.path.isdir(os.path.join(self.candidate_dir(cid), "src")):
                self.make_candidate(cid, self.champion["changes"])
            have = self.champion["measures"].setdefault(stage, {})
            need = [s for s in seeds if s not in have]
            if need:
                for s, m in self.run_stage(cid, stage, need).items():
                    have[s] = m
                    self.ledger.append({"kind": "baseline_run", "candidate": cid, "stage": stage, "seed": s,
                                        "measures": {k: v for k, v in m.items() if k not in ("eval_curve",)}})
                self.save_state()
            return {s: have[s] for s in seeds}

    # ------------------------------------------------------------------ A: diagnose
    def diagnose(self) -> dict:
        stage = "develop" if self.champion["measures"].get("develop") else "screen"
        runs = self.baseline(stage)
        ok = [m for m in runs.values() if m.get("ok")]
        if not ok:
            return {"text": "the champion's runs failed; no diagnosis", "tags": []}

        def avg(k):
            return float(np.mean([m.get(k, 0.0) for m in ok]))
        d = {}
        for k in ("kl", "entropy", "explained_variance", "clip_frac", "grad_norm", "train_valid_rate", "train_success"):
            pairs = [m["diag"].get(k) for m in ok if m.get("diag", {}).get(k)]
            if pairs:
                d[k] = [float(np.mean([p[0] for p in pairs if p[0] is not None] or [0])),
                        float(np.mean([p[1] for p in pairs if p[1] is not None] or [0]))]
        tags, notes = [], []
        ev = d.get("explained_variance", [0, 0])
        if ev[1] < 0.3:
            tags.append("credit")
            notes.append(f"the critic explains little of the returns (explained variance {ev[0]:.2f} -> {ev[1]:.2f}): "
                         f"advantages are noisy, so credit assignment is a bottleneck")
        es = float(np.mean([m.get("diag", {}).get("early_stops", 0) / max(1, m.get("diag", {}).get("iters", 1)) for m in ok]))
        if es > 0.5 or d.get("clip_frac", [0, 0])[1] > 0.3:
            tags.append("trust_region")
            notes.append(f"{es:.0%} of updates stopped early on the KL limit: policy updates leave the trust region quickly")
        ent = d.get("entropy", [0, 0])
        if ent[1] < 0.6 * ent[0] and avg("final_eval") < 0.7:
            tags.append("exploration")
            notes.append(f"entropy fell from {ent[0]:.2f} to {ent[1]:.2f} while held-out success is {avg('final_eval'):.2f}: premature exploitation")
        if ent[1] > 1.2 * ent[0] and avg("final_eval") <= avg("first_eval"):
            tags.append("stability")
            notes.append(f"entropy rose ({ent[0]:.2f} -> {ent[1]:.2f}) and held-out success did not improve: the objective is destabilising the policy")
        vr = d.get("train_valid_rate", [0, 0])
        if vr[1] < 0.6:
            tags.append("validity")
            notes.append(f"only {vr[1]:.0%} of world commands are accepted late in training")
        if avg("final_eval") - avg("first_eval") < 0.05:
            tags.append("plateau")
            notes.append("held-out success barely moves over the budget (plateau)")
        text = (f"Champion {self.champion['id']} ({self.champion['hypothesis'][:120]}), stage {stage}, {len(ok)} seed(s): "
                f"held-out success {avg('first_eval'):.2f} -> {avg('final_eval'):.2f}, area under the learning curve "
                f"{avg('aulc_eval'):.3f}, training success late {avg('train_success_late'):.2f}. "
                + " ".join(f"{k} {v[0]:.3f}->{v[1]:.3f};" for k, v in d.items()) + " " + " ".join(notes))
        rec = {"text": text, "tags": tags, "stats": d, "aulc_eval": avg("aulc_eval"), "final_eval": avg("final_eval")}
        self.ledger.append({"kind": "diagnosis", "round": self.round_n, **rec})
        return rec

    # ------------------------------------------------------------------ B: hypothesize
    def propose_pool(self, diag: dict) -> list[dict]:
        src = self.cfg["sources"]
        pool: list[dict] = []
        parent = self.champion.get("genome") or P.BASE_GENOME
        seen = {r["candidate"] for r in self.ledger.experiments()}
        tries = 0
        while len([p for p in pool if p["source"] == "rule_space"]) < int(src.get("rule_space", 0)) and tries < 50:
            tries += 1
            p = self.rule_space.propose(parent if "rule" not in str(parent.get("phi", "")) else P.BASE_GENOME)
            if p["id"] not in seen and p["id"] not in {q["id"] for q in pool}:
                pool.append(p)
        for _ in range(int(src.get("learned_rule", 0))):
            pool.append(self.learned.propose())
        if self.llm_proposer is not None:
            for _ in range(int(src.get("llm", 0))):
                p = self.llm_proposer.propose(self.champion["changes"], diag["text"], self.ledger.summary_text())
                if p is None:
                    f = self.llm_proposer.failures[-1]
                    self.ledger.append({"kind": "proposal_rejected", "source": "llm", "round": self.round_n, **f})
                elif p["id"] not in seen:
                    pool.append(p)
        for p in pool:
            p["round"] = self.round_n
        return pool

    def select(self, pool: list[dict]) -> list[int]:
        k = min(int(self.cfg["per_round"]), len(pool))
        if self.cfg["selection"] == "random" or not pool:
            return [int(i) for i in self.rng.permutation(len(pool))[:k]]
        return self.surrogate.select(pool, k, rng=self.rng)

    # ------------------------------------------------------------------ C-H: one experiment
    def experiment(self, prop: dict) -> dict:
        cid = f"c{prop.get('round', self.round_n):03d}-{prop['id']}"
        t0 = time.time()
        rec = {"kind": "experiment", "candidate": cid, "proposal": prop["id"], "round": self.round_n,
               "parent": self.champion["id"], "source": prop["source"], "hypothesis": prop["hypothesis"],
               "expected": prop["expected"], "target": prop["target"], "slots": prop["slots"],
               "lines_changed": prop["lines_changed"], "genome": prop.get("genome"), "stage_reached": "implement",
               "decision": "rejected_invalid", "reject_reason": "", "comparison": {}, "stages": {}}
        try:
            cand = self.make_candidate(cid, prop["changes"])
        except Exception as e:  # noqa: BLE001
            rec["reject_reason"] = f"could not create candidate: {type(e).__name__}: {e}"
            return self._finish(rec, prop, None, False, t0)
        open(os.path.join(cand["path"], "proposal.diff"), "w").write(prop.get("diff", ""))
        json.dump({k: v for k, v in prop.items() if k != "changes"}, open(os.path.join(cand["path"], "proposal.json"), "w"), indent=1, default=str)
        rec.update(editable_hash=cand["editable_hash"], protected_hash=cand["protected_hash"], changed_files=cand["changed"])
        okv, why = verify(cand["path"])
        if not okv:
            rec["reject_reason"] = why
            return self._finish(rec, prop, None, False, t0)
        rec["stage_reached"] = "verify"
        st = self.run_one(cid, "screen", 0, selftest=True)
        rec["stages"]["selftest"] = st
        if not st["ok"]:
            bad = [k for k, v in (st["untrusted"].get("checks") or {}).items() if not v]
            rec["reject_reason"] = f"self-test failed: {bad or st['untrusted'].get('error') or 'crashed'}"[:400]
            return self._finish(rec, prop, None, False, t0)
        prim = self.cfg["primary"]
        effect = None
        for stage in ("screen", "develop", "confirm"):
            base = self.baseline(stage)
            runs = self.run_stage(cid, stage)
            good = {s: m for s, m in runs.items() if m.get("ok")}
            rec["stages"][stage] = {s: {k: v for k, v in m.items() if k not in ("eval_curve", "diag")} for s, m in runs.items()}
            rec["stage_reached"] = stage
            if len(good) < len(runs):
                rec["decision"] = "rejected_invalid"
                bad = next(m for s, m in runs.items() if not m.get("ok"))
                rec["reject_reason"] = (f"training run failed at {stage}: " +
                                        str(bad.get("untrusted", {}).get("error") or ("timeout" if bad.get("timed_out") else "too little experience or no evaluation")))[:400]
                return self._finish(rec, prop, effect, effect is not None, t0)
            cmp = {m_: compare({s: good[s][m_] for s in good}, {s: base[s][m_] for s in good if base[s].get("ok")})
                   for m_ in ("aulc_eval", "final_eval", "aulc_train")}
            cmp["wall_s"] = compare({s: good[s]["wall_s"] for s in good}, {s: base[s]["wall_s"] for s in good if base[s].get("ok")})
            rec["comparison"] = cmp
            effect = cmp[prim]["diff"]
            if stage == "screen":
                if effect < -float(self.cfg["screen_margin"]):
                    rec["decision"] = "rejected_screen"
                    rec["reject_reason"] = f"screening: {prim} {effect:+.3f} below the champion by more than the margin"
                    return self._finish(rec, prop, effect, True, t0)
            elif stage == "develop":
                if cmp[prim]["p_improve"] < 0.8 or effect <= 0:
                    v = verdict(cmp[prim], cmp["final_eval"], self.cfg["promote_margin"], self.cfg["max_final_drop"])
                    rec["decision"] = "harmful" if v == "harmful" else "inconclusive"
                    return self._finish(rec, prop, effect, True, t0)
            else:
                v = verdict(cmp[prim], cmp["final_eval"], self.cfg["promote_margin"], self.cfg["max_final_drop"])
                rec["decision"] = {"improved": "promoted", "harmful": "harmful"}.get(v, "inconclusive")
                if rec["decision"] == "promoted":
                    with self._lock:
                        if self.champion["id"] == rec["parent"]:       # the champion it beat is still the champion
                            self._promote(cid, prop, runs)
                        else:
                            rec["decision"] = "improved_over_previous_champion"
        return self._finish(rec, prop, effect, True, t0)

    def _promote(self, cid: str, prop: dict, confirm_runs: dict) -> None:
        old = self.champion["id"]
        self.champion = {"id": cid, "changes": dict(prop["changes"]), "genome": prop.get("genome") if prop["source"] == "rule_space" else self.champion.get("genome"),
                         "hypothesis": prop["hypothesis"], "measures": {"confirm": confirm_runs}, "promoted_at": self.round_n,
                         "previous": old}
        self.ledger.append({"kind": "promotion", "round": self.round_n, "new_champion": cid, "previous": old})

    def _finish(self, rec: dict, prop: dict, effect: float | None, valid: bool, t0: float) -> dict:
        with self._lock:
            return self._finish_locked(rec, prop, effect, valid, t0)

    def _finish_locked(self, rec: dict, prop: dict, effect: float | None, valid: bool, t0: float) -> dict:
        rec["wall_s"] = round(time.time() - t0, 1)
        rec["effect"] = effect
        self.ledger.append(rec)
        # I: learn from the experiment
        self.surrogate.add(prop, effect, valid)
        fit = self.surrogate.fit()
        if prop["source"] == "learned_rule":
            self.learned.tell(prop["id"], effect)
            step = self.learned.step()
            if step:
                self.ledger.append({"kind": "learned_rule_update", "round": self.round_n, **step})
        self.ledger.append({"kind": "meta_update", "round": self.round_n, "surrogate": fit, "loo": self.surrogate.loo_report()})
        self.save_state()
        return rec

    # ------------------------------------------------------------------ J: the loop
    def round(self) -> list[dict]:
        diag = self.diagnose()
        pool = self.propose_pool(diag)
        mu, sd, pv = self.surrogate.predict(pool) if pool else ([], [], [])
        chosen = self.select(pool)
        self.ledger.append({"kind": "selection", "round": self.round_n, "selection": self.cfg["selection"],
                            "pool": [{"id": p["id"], "source": p["source"], "slots": p["slots"], "hypothesis": p["hypothesis"][:200],
                                      "predicted": float(mu[i]), "spread": float(sd[i]), "p_valid": float(pv[i])} for i, p in enumerate(pool)],
                            "chosen": [pool[i]["id"] for i in chosen], "diagnosis_tags": diag.get("tags")})
        # the chosen candidates are evaluated side by side; confined runs share `max_parallel` slots
        with cf.ThreadPoolExecutor(max_workers=max(1, len(chosen))) as ex:
            out = list(ex.map(lambda i: self.experiment(pool[i]), chosen))
        self.round_n += 1
        self.save_state()
        return out

    def run(self, rounds: int | None = None, verbose: bool = True) -> None:
        n = int(rounds if rounds is not None else self.cfg["rounds"])
        self.prepare_demos()
        for _ in range(n):
            for r in self.round():
                if verbose:
                    d = r.get("comparison", {}).get(self.cfg["primary"], {})
                    print(f"round {r['round']} {r['candidate']} [{r['source']}] -> {r['decision']} at {r['stage_reached']}: "
                          f"{self.cfg['primary']} diff {d.get('diff', float('nan')):+.4f} ci {d.get('ci')} "
                          f"| {r['hypothesis'][:110]} {('| ' + r['reject_reason'][:120]) if r['reject_reason'] else ''}", flush=True)

    def report(self) -> dict:
        exps = self.ledger.experiments()
        dec: dict[str, int] = {}
        for r in exps:
            dec[r["decision"]] = dec.get(r["decision"], 0) + 1
        pts = {r["candidate"]: {"aulc_eval": r["comparison"]["aulc_eval"]["mean_cand"], "final_eval": r["comparison"]["final_eval"]["mean_cand"],
                                "wall_s": r["comparison"]["wall_s"]["mean_cand"]}
               for r in exps if r.get("comparison", {}).get("aulc_eval", {}).get("n_cand")}
        return {"champion": {k: v for k, v in self.champion.items() if k not in ("changes", "measures")},
                "rounds": self.round_n, "experiments": len(exps), "decisions": dec, "table": self.ledger.table(),
                "pareto": pareto(pts) if pts else {}, "surrogate": {"updates": self.surrogate.updates, **self.surrogate.loo_report()},
                "learned_rule": self.learned.history,
                "llm_proposals_rejected_before_running": len(self.ledger.entries("proposal_rejected"))}


def main(a) -> int:
    from ..cli import RUNS, load_config
    cfg = load_config(a.config)
    rd = a.run_dir or os.path.join(RUNS, "meta", cfg.get("name", "meta"))
    llm = None
    if (cfg.get("sources") or DEFAULTS["sources"]).get("llm") and cfg.get("proposer_llm", {"kind": "remote"}).get("kind") != "none":
        from ..cli import DEFAULT_LLM
        from ..llm.server import connect
        spec = dict(DEFAULT_LLM, **cfg.get("proposer_llm", {}))
        if os.path.exists(spec["socket"]):
            llm = connect(spec)
        else:
            print(f"note: no LLM server at {spec['socket']}; running without LLM-written proposals")
    mr = MetaResearcher(cfg, rd, llm, resume=a.resume)
    mr.run(a.rounds)
    rep = mr.report()
    json.dump(rep, open(os.path.join(rd, "report.json"), "w"), indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ("champion", "rounds", "experiments", "decisions", "surrogate")}, indent=1, default=str))
    print(f"ledger: {os.path.join(rd, 'ledger.jsonl')}")
    return 0
