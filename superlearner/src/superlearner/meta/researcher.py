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
import threading
import time

import numpy as np

from ..train.trainer import code_hashes, git_commit, merge
from ..world.hub import start_hub
from ..world.tasks import SPLITS, task_list
from . import proposals as P
from .compare import compare, lifetime_measures, load_trusted, pareto, run_measures, verdict
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
        # mode "lifetime": candidates are CONTINUAL update rules (algo/continual.py); a run is a short lifetime over
        # phases of new task arms, scored by acquisition and retention (compare.lifetime_measures)
        self.life = c.get("mode") == "lifetime"
        self.cont_space = P.ContinualSpace(c["seed"])
        self.best_seen: dict | None = None
        if self.life:
            self.llm_proposer = P.LLMProposer(
                llm, c["seed"], path=P.CONT_PATH, functions=("replay_arm_weights", "anchor_penalty", "fisher_update"),
                note=("These functions are parts of a continual-learning update rule for a neural controller that learns "
                      "task arms in successive phases. `replay_arm_weights(arms, current)` returns the probability of "
                      "replaying each arm (tuples; `current` is the set of arms of the present phase). "
                      "`anchor_penalty(params, anchors, weights)` is the consolidation penalty between the current "
                      "parameters and their values at the end of the last phase (lists of tensors; weights may be None). "
                      "`fisher_update(old, new, decay)` combines per-parameter importance estimates across phases. "
                      "numpy is imported as np.")) if (llm is not None and c["sources"].get("llm")) else None
        else:
            self.llm_proposer = P.LLMProposer(llm, c["seed"]) if (llm is not None and c["sources"].get("llm")) else None
        self.champion = {"id": "champion-0", "changes": {}, "genome": P.BASE_CONTINUAL_GENOME if self.life else P.BASE_GENOME,
                         "hypothesis": ("baseline continual rule: rehearsal reservoirs, online world model, no anchors"
                                        if self.life else "baseline: recurrent PPO with GAE"),
                         "measures": {}, "promoted_at": 0}
        self.round_n = 0
        self.isolation = require_isolation()          # fail closed: no confinement, no candidate execution
        self._sem = threading.Semaphore(int(c["max_parallel"]))     # confined training runs in flight
        self._lock = threading.RLock()                               # ledger / state / surrogate
        self._base_lock = threading.Lock()
        self.state_path = os.path.join(self.dir, "state.json")
        # the base tree every candidate (and the champion's own runs) is built from: frozen when the search starts,
        # so that editing the repository during a search cannot make compared runs use different code
        self.base_pkg = os.path.join(self.dir, "base_src", "superlearner")
        if not os.path.isdir(self.base_pkg):
            import shutil
            shutil.copytree(PKG, self.base_pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if resume and os.path.exists(self.state_path):
            self.load_state()
        else:
            json.dump(self.cfg, open(os.path.join(self.dir, "config.json"), "w"), indent=1)
            from .workspace import tree_hashes
            bh = tree_hashes(self.base_pkg)
            json.dump({"code": code_hashes(), "git": git_commit(), "isolation": self.isolation, "started": time.time(),
                       "base_tree": {"editable": bh["editable"], "protected": bh["protected"]}},
                      open(os.path.join(self.dir, "meta.json"), "w"), indent=1, default=str)

    # ------------------------------------------------------------------ state
    def save_state(self) -> None:
        s = {"champion": self.champion, "round": self.round_n, "surrogate": self.surrogate.state_dict(),
             "learned": self.learned.state_dict(), "rng": self.rng.bit_generator.state,
             "rule_rng": self.rule_space.rng.bit_generator.state, "cont_rng": self.cont_space.rng.bit_generator.state,
             "best_seen": self.best_seen}
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
        if s.get("cont_rng"):
            self.cont_space.rng.bit_generator.state = s["cont_rng"]
        self.best_seen = s.get("best_seen")

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
    def life_lists(self, stage: str) -> tuple[list, list]:
        """(held-out tasks for the hub, index ranges per phase for the learner): one list per phase of the lifetime,
        the phase's arms in held-out seeds of the evaluation world families."""
        st = self.cfg["stages"][stage]
        fams = [f for f in self.cfg["eval_world_families"] if f in SPLITS["select"][2]]
        tasks, lists = [], []
        for j, ph in enumerate(self.cfg["phases"]):
            part = task_list(ph["arms"], fams, int(st["eval_per_arm"]) * len(ph["arms"]), "select", 4000 + 300 * j)
            lists.append({"name": ph.get("name", f"P{j + 1}"), "phase": j, "kind": "seen",
                          "indices": list(range(len(tasks), len(tasks) + len(part)))})
            tasks += part
        return tasks, lists

    def eval_tasks(self, stage: str) -> list:
        if self.life:
            return self.life_lists(stage)[0]
        st = self.cfg["stages"][stage]
        arms, fams = self.cfg["eval_arms"], [f for f in self.cfg["eval_world_families"] if f in SPLITS["select"][2]]
        return task_list(arms, fams, int(st["eval_n"]), "select", 4000)

    def inner_config(self, stage: str, seed: int) -> dict:
        st = self.cfg["stages"][stage]
        if self.life:
            phases = [dict(ph, **{k: st[k] for k in ("demos_per_arm", "observe_steps", "iters", "episodes_per_iter") if k in st})
                      for ph in self.cfg["phases"]]
            return merge(self.cfg["inner"], {
                "name": f"{stage}_s{seed}", "seed": int(seed), "ckpt_every": 0, "traj_per_iter": 0, "device": "cpu",
                "lifetime": {"phases": phases, "hub_lists": self.life_lists(stage)[1], "wm_holdout": 0, "probe_steps": 0,
                             "eval": {"sampled": False, "unseen_worlds": []}}})
        return merge(self.cfg["inner"], {"name": f"{stage}_s{seed}", "seed": int(seed), "iters": int(st["iters"]),
                                         "episodes_per_iter": int(st["episodes_per_iter"]), "ckpt_every": 0,
                                         "traj_per_iter": 0, "device": "cpu"})

    # ------------------------------------------------------------------ candidates
    def candidate_dir(self, cid: str) -> str:
        return os.path.join(self.dir, "candidates", cid)

    def make_candidate(self, cid: str, changes: dict) -> dict:
        rec = create_candidate(self.candidate_dir(cid), changes, self.base_pkg)
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
        tdir = os.path.join(self.dir, "trusted", cid)
        os.makedirs(tdir, exist_ok=True)
        results = os.path.join(tdir, f"{tag}.jsonl")
        done = os.path.join(tdir, f"{tag}.measures.json")
        if os.path.exists(done):
            # this exact run (same candidate tree, stage settings and seed) finished before the search was
            # interrupted: its trusted measures are reused instead of spending the experience again
            out = json.load(open(done))
            if out.get("ok") and out.get("tree") == self._tree_id(cid) and out.get("stage_cfg") == self.cfg["stages"][stage]:
                out["reused"] = True
                return out
        if os.path.isdir(work) and os.listdir(work):
            os.replace(work, work + f".old{int(time.time())}")           # an interrupted attempt: set aside, start clean
        os.makedirs(work, exist_ok=True)
        if os.path.exists(results):
            os.replace(results, results + f".old{int(time.time())}")     # never silently mix two runs
        sock = os.path.join(self.dir, "sock", hashlib.sha1(f"{cid}{tag}".encode()).hexdigest()[:12] + ".sock")
        st = self.cfg["stages"][stage]
        cfg = self.inner_config(stage, seed)
        budget = int(st["budget"])
        hub = start_hub(sock, None if selftest else self.eval_tasks(stage), cfg=cfg.get("core"), eval_cfg=cfg.get("core"),
                        results_path=results, budget=None if selftest else {"train_decisions": budget})
        run_stage = "lifetime" if self.life else "train"
        job = {"stage": "selftest" if selftest else run_stage, "config": cfg, "env_socket": sock, "workdir": work,
               "eval_n": 0 if selftest else int(st.get("eval_n", 0)), "llm": cfg.get("llm"),
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
        rp = os.path.join(work, ("selftest" if selftest else run_stage) + "_result.json")
        untrusted = json.load(open(rp)) if os.path.exists(rp) else {"ok": False, "error": res["stdout"][-800:]}
        out["untrusted"] = {k: untrusted.get(k) for k in ("ok", "error", "checks", "max_logp_error", "iters", "stopped",
                                                           "update_s", "rollout_s", "worker_errors", "params")}
        if selftest:
            out["ok"] = bool(res["returncode"] == 0 and untrusted.get("ok"))
            if out["ok"]:
                json.dump(dict(out, tree=self._tree_id(cid), stage_cfg=self.cfg["stages"][stage]), open(done, "w"), default=str)
            return out
        if self.life:
            m = lifetime_measures(load_trusted(results), self.life_lists(stage)[1], budget)
            out.update(m)
            out["ok"] = bool(res["returncode"] == 0 and m["life_valid"])
        else:
            m = run_measures(load_trusted(results), budget)
            out.update(m)
            out["ok"] = bool(res["returncode"] == 0 and m["train_decisions"] >= 0.5 * min(budget, int(st["iters"]) * int(st["episodes_per_iter"]) * 4)
                             and m["eval_rounds"] >= 2)
        out["diag"] = self._diagnostics(os.path.join(work, "run", "metrics.jsonl"))
        if out["ok"]:
            json.dump(dict(out, tree=self._tree_id(cid), stage_cfg=self.cfg["stages"][stage]), open(done, "w"), default=str)
        return out

    # ------------------------------------------------------------------ inert changes
    def probe(self, cid: str) -> dict:
        """Confined fingerprint of a candidate's update on fixed data (see `candidate_run.probe`). The candidate's
        editable files are run with THIS repository's protected code, so trees of older searches can be probed too."""
        from .workspace import is_editable, tree_hashes
        out_path = os.path.join(self.dir, "probe", cid, "fingerprint.json")
        if os.path.exists(out_path):
            return json.load(open(out_path))
        pkg = os.path.join(self.candidate_dir(cid), "src", "superlearner")
        changes = {rel: open(os.path.join(pkg, rel)).read() for rel in tree_hashes(pkg)["files"] if is_editable(rel)}
        dest = os.path.join(self.dir, "probe", cid)
        os.makedirs(dest, exist_ok=True)
        create_candidate(dest, changes, PKG)
        work = os.path.join(dest, "work")
        os.makedirs(work, exist_ok=True)
        job = {"stage": "probe", "config": self.inner_config("screen", 0), "workdir": work, "demos_path": self.cfg.get("demos_path")}
        jp = os.path.join(work, "job.json")
        json.dump(job, open(jp, "w"), indent=1)
        extra = (os.path.dirname(self.cfg["demos_path"]),) if self.cfg.get("demos_path") else ()
        with self._sem:
            res = run_isolated("superlearner.meta.candidate_run", [jp], os.path.join(dest, "src"), work, timeout_s=600, gpu=False,
                               extra_read=extra, log_path=os.path.join(work, "log.txt"))
        rp = os.path.join(work, "probe_result.json")
        r = json.load(open(rp)) if os.path.exists(rp) else {"ok": False, "error": str(res.get("stdout", ""))[-500:]}
        out = {"ok": bool(res["returncode"] == 0 and r.get("ok")), "fingerprint": r.get("fingerprint"), "error": r.get("error")}
        json.dump(out, open(out_path, "w"), indent=1)
        return out

    def is_inert(self, cid: str) -> bool | None:
        """True if the candidate's update on the probe data is identical to the champion's; None if a probe failed."""
        a, b = self.probe(cid), self.probe(self.champion["id"])
        if not (a.get("ok") and b.get("ok")):
            return None
        fa, fb = a["fingerprint"], b["fingerprint"]
        return set(fa) == set(fb) and all(abs(fa[k] - fb[k]) <= 1e-9 * max(1.0, abs(fb[k])) for k in fb)

    def _tree_id(self, cid: str) -> str:
        """Hash of a candidate's source tree (editable and protected parts), to tie stored measures to the code."""
        try:
            c = json.load(open(os.path.join(self.candidate_dir(cid), "candidate.json")))
            return f"{c.get('editable_hash')}:{c.get('protected_hash')}"
        except (OSError, ValueError):
            return ""

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
            for other, runs in self.champion["measures"].items():       # identical settings: reuse the same seed's run
                if other != stage and self.same_settings(other, stage):
                    for s_, m_ in runs.items():
                        have.setdefault(int(s_), m_)
            need = [s for s in seeds if s not in have]
            if need:
                for s, m in self.run_stage(cid, stage, need).items():
                    have[s] = m
                    self.ledger.append({"kind": "baseline_run", "candidate": cid, "stage": stage, "seed": s,
                                        "measures": {k: v for k, v in m.items() if k not in ("eval_curve",)}})
                self.save_state()
            return {s: have[s] for s in seeds}

    def same_settings(self, a: str, b: str) -> bool:
        """Two stages whose runs are interchangeable (everything equal except which seeds they use)."""
        sa, sb = dict(self.cfg["stages"][a]), dict(self.cfg["stages"][b])
        sa.pop("seeds", None)
        sb.pop("seeds", None)
        return sa == sb

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
        if self.life:
            base_src = self.champion["changes"].get(P.CONT_PATH) or open(os.path.join(self.base_pkg, P.CONT_PATH)).read()
            had = {r.get("proposal") for r in self.ledger.experiments()}
            if self.round_n == 0 and self.cfg.get("named_first", True):
                # first pool: the classic continual-learning methods, one idea each, against the same baseline
                for name, g in self.cont_space.named():
                    p = self.cont_space.proposal(g, parent, name, base_src)
                    if p["id"] not in had:
                        pool.append(p)
            n_want = len(pool) + int(src.get("continual_space", 0)) if not pool else len(pool)
            while len(pool) < n_want and tries < 80:
                tries += 1
                # mutate the champion, or (half of the time) the best rule screened so far; cross the two over
                start = self.best_seen if (self.best_seen is not None and self.rng.random() < 0.5) else parent
                p = self.cont_space.proposal(self.cont_space.mutate(start), parent, None, base_src)
                if p["id"] not in had and p["id"] not in {q["id"] for q in pool} and p["genome"] != parent:
                    pool.append(p)
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
    def experiment(self, prop: dict, cid: str | None = None, known: dict | None = None) -> dict:
        """Take one proposal through verify, self-test, screening, development and confirmation.
        `cid` + `known` continue a candidate that was already screened (its tree exists, its screen runs are given)."""
        existing = cid is not None
        cid = cid or f"c{prop.get('round', self.round_n):03d}-{prop['id']}"
        have: dict = {"screen": {int(k): v for k, v in (known or {}).items()}}
        t0 = time.time()
        rec = {"kind": "experiment", "candidate": cid, "proposal": prop["id"], "round": self.round_n,
               "parent": self.champion["id"], "source": prop["source"], "hypothesis": prop["hypothesis"],
               "expected": prop["expected"], "target": prop["target"], "slots": prop["slots"],
               "lines_changed": prop["lines_changed"], "genome": prop.get("genome"), "stage_reached": "implement",
               "decision": "rejected_invalid", "reject_reason": "", "comparison": {}, "stages": {}}
        try:
            cand = json.load(open(os.path.join(self.candidate_dir(cid), "candidate.json"))) if existing \
                else self.make_candidate(cid, prop["changes"])
        except Exception as e:  # noqa: BLE001
            rec["reject_reason"] = f"could not create candidate: {type(e).__name__}: {e}"
            return self._finish(rec, prop, None, False, t0)
        open(os.path.join(cand["path"], "proposal.diff"), "w").write(prop.get("diff", ""))
        json.dump({k: v for k, v in prop.items() if k != "changes"}, open(os.path.join(cand["path"], "proposal.json"), "w"), indent=1, default=str)
        rec.update(editable_hash=cand["editable_hash"], protected_hash=cand["protected_hash"], changed_files=cand["changed"])
        okv, why = verify(cand["path"], self.base_pkg)
        if not okv:
            rec["reject_reason"] = why
            return self._finish(rec, prop, None, False, t0)
        rec["stage_reached"] = "verify"
        if not existing:
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
            seeds = list(self.cfg["stages"][stage]["seeds"])
            runs = {}
            for other, done in have.items():                 # a seed already run under identical settings is reused
                if other == stage or self.same_settings(other, stage):
                    runs.update({s_: m_ for s_, m_ in done.items() if s_ in seeds})
            todo = [s_ for s_ in seeds if s_ not in runs]
            if todo:
                runs.update(self.run_stage(cid, stage, todo))
            have[stage] = dict(runs)
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

    # ------------------------------------------------------------------ full-feedback rounds (Experiment F)
    def screen_only(self, prop: dict) -> dict:
        """Implement, verify, self-test and screen one proposal (one seed). No later stages, no promotion."""
        cid = f"f{prop.get('round', self.round_n):03d}-{prop['id']}"
        t0 = time.time()
        rec = {"candidate": cid, "proposal": prop["id"], "source": prop["source"], "slots": prop["slots"],
               "hypothesis": prop["hypothesis"][:300], "valid": False, "effect": None, "reject_reason": ""}
        try:
            cand = self.make_candidate(cid, prop["changes"])
            open(os.path.join(cand["path"], "proposal.diff"), "w").write(prop.get("diff", ""))
            okv, why = verify(cand["path"], self.base_pkg)
            if not okv:
                rec["reject_reason"] = why
                return rec
            st = self.run_one(cid, "screen", 0, selftest=True)
            if not st["ok"]:
                bad = [k for k, v in (st["untrusted"].get("checks") or {}).items() if not v]
                rec["reject_reason"] = f"self-test failed: {bad or st['untrusted'].get('error') or 'crashed'}"[:300]
                return rec
            if self.cfg.get("reject_inert", True) and self.cfg.get("demos_path"):
                self.baseline("screen")                                  # makes sure the champion's tree exists
                if self.is_inert(cid):
                    rec["reject_reason"] = "no functional change: its update on the probe data is identical to the champion's"
                    rec["inert"] = True
                    return rec
            runs = self.run_stage(cid, "screen")
            good = {s_: m for s_, m in runs.items() if m.get("ok")}
            if len(good) < len(runs):
                bad = next(m for m in runs.values() if not m.get("ok"))
                rec["reject_reason"] = ("training run failed: " + str(bad.get("untrusted", {}).get("error") or "timeout or too little experience"))[:300]
                return rec
            base = self.baseline("screen")               # cached; in the first round it runs alongside the candidates
            if not all(base[s_].get("ok") for s_ in good):
                raise RuntimeError("the champion's own run failed; no comparison is possible")
            prim = self.cfg["primary"]
            keys = ("aulc_eval", "final_eval", "first_eval", "aulc_train", "train_decisions", "wall_s",
                    "life_forgetting", "life_acquisition", "life_matrix")
            rec.update(valid=True, effect=float(np.mean([good[s_][prim] - base[s_][prim] for s_ in good])),
                       measures={s_: {k: m[k] for k in keys if k in m} for s_, m in good.items()})
            if self.life and prop.get("genome") and (self.best_seen is None or rec["effect"] > self.best_seen.get("_effect", -1e9)):
                self.best_seen = dict(prop["genome"], _effect=rec["effect"])
        except Exception as e:  # noqa: BLE001
            rec["reject_reason"] = f"{type(e).__name__}: {e}"[:300]
        finally:
            rec["wall_s"] = round(time.time() - t0, 1)
        return rec

    def full_feedback_round(self, k: int = 1) -> dict:
        """Screen EVERY proposal of the pool, and record what each selection strategy would have picked.

        The surrogate chooses before any outcome of this round is known and afterwards learns only from its own
        picks (as in normal operation). Because all outcomes are measured, its picks can be compared with uniform
        random selection (the pool mean) and with the best proposal of the pool on exactly the same candidates."""
        diag = self.diagnose()
        pool = self.propose_pool(diag)
        mu, sd, pv = self.surrogate.predict(pool)
        fitted = bool(self.surrogate.models)
        picks = self.surrogate.select(pool, k, rng=self.rng)
        with cf.ThreadPoolExecutor(max_workers=max(1, len(pool)) + 1) as ex:
            fb = ex.submit(self.baseline, "screen")       # the champion's run shares the round's parallel slots
            res = list(ex.map(self.screen_only, pool))
            fb.result()
        penalty = -float(self.cfg["screen_margin"])                    # an invalid candidate wastes the experiment
        vals = [r["effect"] if r["valid"] else penalty for r in res]
        rec = {"kind": "full_feedback", "round": self.round_n, "surrogate_fitted": fitted, "k": k,
               "pool": [dict(r, predicted=float(mu[i]), spread=float(sd[i]), p_valid=float(pv[i]), value=vals[i]) for i, r in enumerate(res)],
               "surrogate_picks": [pool[i]["id"] for i in picks], "value_surrogate": float(np.mean([vals[i] for i in picks])),
               "value_random_expected": float(np.mean(vals)), "value_best": float(np.max(vals)),
               "valid_fraction": float(np.mean([r["valid"] for r in res])), "diagnosis_tags": diag.get("tags")}
        with self._lock:
            self.ledger.append(rec)
            for i in picks:                                                # the meta-policy learns from what it chose
                self.surrogate.add(pool[i], res[i]["effect"], res[i]["valid"])
            fit = self.surrogate.fit()
            for i, p in enumerate(pool):                                   # the learned-rule search uses every evaluation
                if p["source"] == "learned_rule":
                    self.learned.tell(p["id"], res[i]["effect"] if res[i]["valid"] else None)
            step = self.learned.step()
            if step:
                self.ledger.append({"kind": "learned_rule_update", "round": self.round_n, **step})
            self.ledger.append({"kind": "meta_update", "round": self.round_n, "surrogate": fit, "loo": self.surrogate.loo_report()})
            self._pools = getattr(self, "_pools", {})
            for i, p in enumerate(pool):
                json.dump(p, open(os.path.join(self.candidate_dir(res[i]["candidate"]), "proposal_full.json"), "w"), default=str) \
                    if os.path.isdir(self.candidate_dir(res[i]["candidate"])) else None
                if res[i]["valid"]:
                    self._pools[p["id"]] = (p, res[i])
            self.round_n += 1
            self.save_state()
        return rec

    def run_full_feedback(self, rounds: int, k: int = 1, develop_top: int = 2, verbose: bool = True) -> dict:
        self.prepare_demos()
        if not getattr(self, "_pools", None):                 # after a restart: rebuild the screened pool from disk
            self._pools = {}
            for fr in self.ledger.entries("full_feedback"):
                for e in fr["pool"]:
                    pf = os.path.join(self.candidate_dir(e["candidate"]), "proposal_full.json")
                    if e.get("valid") and os.path.exists(pf):
                        self._pools[e["proposal"]] = (json.load(open(pf)), e)
        for _ in range(rounds):
            r = self.full_feedback_round(k)
            if verbose:
                print(f"round {r['round']}: pool {len(r['pool'])} valid {r['valid_fraction']:.2f} | surrogate pick {r['value_surrogate']:+.4f} "
                      f"(fitted={r['surrogate_fitted']}) random {r['value_random_expected']:+.4f} best {r['value_best']:+.4f}", flush=True)
        # the best screened candidates go through development and confirmation (the promotion pipeline)
        top = sorted(getattr(self, "_pools", {}).values(), key=lambda pe: -pe[1]["effect"])[:develop_top]
        out = []
        for prop, scr in top:
            eff = scr["effect"]
            if eff <= 0:
                continue
            # the screened candidate continues: same source tree, its screening run counts as that seed's run
            known = {int(s_): dict(m_, ok=True) for s_, m_ in (scr.get("measures") or {}).items()}
            rec = self.experiment(dict(prop, round=self.round_n), cid=scr["candidate"], known=known)
            out.append(rec)
            if verbose:
                d = rec.get("comparison", {}).get(self.cfg["primary"], {})
                print(f"follow-up {rec['candidate']} [{rec['source']}] screen effect {eff:+.4f} -> {rec['decision']} at {rec['stage_reached']}: "
                      f"diff {d.get('diff', float('nan')):+.4f} ci {d.get('ci')} n={d.get('n_cand')}", flush=True)
            self.round_n += 1
            self.save_state()
        return self.selection_report()

    def selection_report(self) -> dict:
        """Experiment F: learned selection against uniform random selection on identical pools."""
        rounds = self.ledger.entries("full_feedback")
        if not rounds:
            return {}
        rows = [{"round": r["round"], "fitted": r["surrogate_fitted"], "surrogate": r["value_surrogate"],
                 "random": r["value_random_expected"], "best": r["value_best"], "n": len(r["pool"])} for r in rounds]
        fitted = [r for r in rows if r["fitted"]]
        rng = np.random.default_rng(0)

        def boot(xs):
            xs = np.asarray(xs, float)
            if len(xs) < 2:
                return [float("nan"), float("nan")]
            b = xs[rng.integers(0, len(xs), size=(4000, len(xs)))].mean(1)
            return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        d_all = [r["surrogate"] - r["random"] for r in rows]
        d_fit = [r["surrogate"] - r["random"] for r in fitted]
        return {"rounds": rows, "candidates_screened": int(sum(r["n"] for r in rows)),
                "advantage_over_random_all_rounds": {"mean": float(np.mean(d_all)), "ci95": boot(d_all), "n": len(d_all)},
                "advantage_over_random_once_fitted": {"mean": float(np.mean(d_fit)) if d_fit else None, "ci95": boot(d_fit), "n": len(d_fit)},
                "regret_vs_best": {"surrogate": float(np.mean([r["best"] - r["surrogate"] for r in rows])),
                                   "random": float(np.mean([r["best"] - r["random"] for r in rows]))},
                "established": bool(len(d_fit) >= 5 and boot(d_fit)[0] > 0), "surrogate_loo": self.surrogate.loo_report()}

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


def probe_search(run_dir: str) -> dict:
    """For a finished or running search: which screened candidates are functionally inert (their update on the probe
    data equals the champion's)? Reads the search's files; runs only confined probes; writes nothing to its ledger."""
    if not os.path.exists(os.path.join(run_dir, "state.json")):
        raise FileNotFoundError(f"{run_dir} has no state.json (the search has not finished its champion baseline)")
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    mr = MetaResearcher(dict(cfg, sources=dict(cfg.get("sources") or {}, llm=0)), run_dir, None, resume=True)
    mr.cfg["demos_path"] = os.path.join(mr.dir, "demos", "demos.pkl")
    out: dict = {"champion": mr.probe(mr.champion["id"]), "candidates": {}}
    for fr in mr.ledger.entries("full_feedback"):
        for e in fr["pool"]:
            cid = e["candidate"]
            if os.path.isdir(os.path.join(mr.candidate_dir(cid), "src")):
                pr = mr.probe(cid)
                out["candidates"][cid] = {"inert": mr.is_inert(cid), "fingerprint": pr.get("fingerprint"), "probe_ok": pr.get("ok"),
                                          "error": pr.get("error")}
    return out


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
    if getattr(a, "full_feedback", False):
        total = int(cfg.get("rounds", 4))
        done = len(mr.ledger.entries("full_feedback"))
        # --rounds N: N more rounds. Without it: the configured total, minus the rounds a resumed search already has
        todo = int(a.rounds) if a.rounds else max(0, total - done)
        if done:
            print(f"resuming: {done} screening rounds in the ledger, {todo} to run", flush=True)
        sel = mr.run_full_feedback(todo, int(cfg.get("per_round", 1)), int(cfg.get("develop_top", 2)))
        json.dump(sel, open(os.path.join(rd, "selection_report.json"), "w"), indent=1, default=str)
        print(json.dumps({k: v for k, v in sel.items() if k != "rounds"}, indent=1, default=str))
    else:
        mr.run(a.rounds)
    rep = mr.report()
    json.dump(rep, open(os.path.join(rd, "report.json"), "w"), indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ("champion", "rounds", "experiments", "decisions", "surrogate")}, indent=1, default=str))
    print(f"ledger: {os.path.join(rd, 'ledger.jsonl')}")
    return 0
