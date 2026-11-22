"""A learner's lifetime: a sequence of phases, each with new tasks and possibly a new kind of world.

    Lifetime(cfg, run_dir).run()

One phase:

    begin      the update rule is told a phase starts (rehearsal buffers kept or dropped, optional shrink-and-perturb)
    observe    the learner is shown `demos_per_arm` verified demonstrations of the phase's arms and takes supervised
               weight updates on them (with rehearsal of earlier arms if the rule rehearses)
    practice   `iters` iterations of acting in fresh worlds of the phase and updating the weights from the outcome:
               policy and critic (PPO), replay of own successes, world model on every new transition
    close      the rule consolidates (anchors / importance); then EVERY phase's held-out task list is evaluated with
               the current weights - one row of the retention matrix - together with the world model's held-out
               prediction quality on every phase seen so far, plasticity diagnostics and the size of the weight change

Everything a restart needs is in ONE checkpoint file written atomically after every step (weights, optimiser state,
world model, replay reservoirs, consolidation anchors, episodic memory, counters, the rows measured so far), so a
lifetime that is killed continues at the step after the last one that finished (`tests/test_continual.py`).

Conditions are settings, not code paths:

    lifetime.memory        episodic memory and agent-made tools on (training and evaluation)
    lifetime.freeze_after  index of the phase after which the WEIGHTS never change again; later phases only write
                           to episodic memory (the "frozen core plus written memory" baseline)
    algo.rehearsal / algo.anchor / algo.shrink_perturb / algo.group_lr / algo.wm     the continual update rule
    backbone.mode          none | lora | top:k | full   (how much of the language backbone is plastic)

Held-out lists use the `select` split (seeds and worlds never used for training); `lifetime.eval.unseen_worlds`
adds, per phase, the same arms in world families no phase trains in.
"""
from __future__ import annotations

import copy
import gzip
import json
import os
import pickle
import time

import numpy as np
import torch

from ..train.curriculum import Curriculum
from ..train.trainer import Trainer, merge, summarize, wilson
from ..world.tasks import TaskSpec, task_list
from . import plasticity as PL

LIFE_DEFAULTS = {"memory": False, "tools": None, "freeze_after": None, "wm_holdout": 8, "probe_episodes": 12,
                 "probe_steps": 40, "hub_lists": None,
                 "eval": {"per_arm": 8, "unseen_worlds": [], "unseen_per_arm": 4, "sampled": True, "unseen_only_at_end": True,
                          "extra_lists": []},
                 "phases": []}
PHASE_DEFAULTS = {"worlds": ["tiny", "bazaar"], "demos_per_arm": 4, "observe_steps": 120, "iters": 4,
                  "episodes_per_iter": 64, "rehearse_tasks": 0.0}


def arm_dict(key: str) -> dict:
    f, lvl, start = key.split("/")
    return {"family": f, "level": int(lvl), "start": start}


def derive(rows: list[dict], phases: list[dict]) -> dict:
    """Continual-learning measures from the rows of the retention matrix (success after phase k on list j)."""
    names = [p["name"] for p in phases]
    K = len(rows)
    if K == 0:
        return {}
    R = [[r["lists"].get(n, {}).get("success") for n in names] for r in rows]
    out: dict = {"matrix": R, "phases": names[:len(names)], "rows_done": K}
    acq, fwd, forg, ret = [], [], [], []
    for k in range(K):
        prev = R[k - 1][k] if k > 0 else 0.0
        if R[k][k] is not None:
            acq.append(R[k][k] - (prev or 0.0))
            fwd.append(prev or 0.0)
    last = R[-1]
    for j in range(K):
        best = max(R[k][j] for k in range(j, K) if R[k][j] is not None)
        forg.append(best - last[j])
        ret.append(last[j])
    out.update(acquisition=acq, zero_shot_before_training=fwd, forgetting=forg, final=ret,
               mean_final_seen=float(np.mean(ret)), mean_acquisition=float(np.mean(acq)) if acq else 0.0,
               mean_forgetting=float(np.mean(forg[:-1])) if K > 1 else 0.0,
               backward_transfer=float(np.mean([last[j] - R[j][j] for j in range(K - 1)])) if K > 1 else 0.0)
    return out


class Lifetime:
    def __init__(self, cfg: dict, run_dir: str, resume: bool = True, verbose: bool = True):
        self.cfg = cfg = copy.deepcopy(cfg)
        self.L = L = merge(LIFE_DEFAULTS, cfg.get("lifetime") or {})
        self.phases = [merge(PHASE_DEFAULTS, p) for p in L["phases"]]
        for i, p in enumerate(self.phases):
            p.setdefault("name", f"P{i + 1}")
        self.dir, self.verbose = run_dir, verbose
        os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)
        self.ck = os.path.join(run_dir, "ckpt", "latest.pt")
        have = resume and os.path.exists(self.ck)
        mem = bool(L["memory"])
        tools = mem if L["tools"] is None else bool(L["tools"])
        tcfg = merge(cfg, {"eval_every": 0, "ckpt_every": 0, "memory": {"enabled": mem}, "demos": {"episodes": 0},
                           "agent": dict(cfg.get("agent") or {}, use_memory=mem, use_tools=tools),
                           "curriculum": {"arms": [arm_dict(a) for a in self.phases[0]["arms"]], "gate": False,
                                          "world_families": self.phases[0]["worlds"]}})
        tcfg.pop("init_from", None)
        self.tr = tr = Trainer(tcfg, run_dir, resume=False)
        if have:
            tr.load(self.ck)
            tr._trim_logs()
            self.x = tr.extra
            self.say(f"resumed: phase {self.x['phase']} stage {self.x['stage']} iteration {self.x['it']}")
        else:
            self.x = tr.extra = {"phase": 0, "stage": "begin", "it": 0, "rows": [], "probe": [], "holdout": {},
                                 "counter": 0, "phase_log": [], "ref_init": PL.state_of(tr.model), "ref_phase": None,
                                 "digest_init": PL.weights_digest(tr.model), "frozen": False, "restarts": 0}
            json.dump({"cfg": cfg, "phases": self.phases}, open(os.path.join(run_dir, "lifetime.json"), "w"), indent=1)
        if have:
            self.x["restarts"] = self.x.get("restarts", 0) + 1
            if self.x["stage"] == "done" and self.x["phase"] < len(self.phases):
                self.x["stage"] = "begin"          # a finished lifetime given further phases simply goes on
        self.lists = self._eval_lists()

    # ------------------------------------------------------------------ small helpers
    def say(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def save(self) -> None:
        self.tr.save("latest.pt", inline=True)

    def _eval_lists(self) -> list[dict]:
        if self.L.get("hub_lists"):
            # the held-out tasks live on a trusted env hub: this process knows them only as indices (meta-search)
            return [dict(x, kind=x.get("kind", "seen"), tasks=None) for x in self.L["hub_lists"]]
        path = os.path.join(self.dir, "eval_tasks.json")
        if os.path.exists(path):
            raw = json.load(open(path))
            return [dict(x, tasks=[TaskSpec.from_dict(t) for t in x["tasks"]]) for x in raw]
        E = self.L["eval"]
        out = []
        for j, p in enumerate(self.phases):
            out.append({"name": p["name"], "phase": j, "kind": "seen",
                        "tasks": task_list(p["arms"], p["worlds"], int(E["per_arm"]) * len(p["arms"]), "select", 4000 + 300 * j)})
            if E.get("unseen_worlds"):
                out.append({"name": p["name"] + "@unseen", "phase": j, "kind": "unseen",
                            "tasks": task_list(p["arms"], list(E["unseen_worlds"]), int(E["unseen_per_arm"]) * len(p["arms"]),
                                               "select", 7000 + 300 * j)})
        for j, x in enumerate(E.get("extra_lists") or []):      # further held-out lists that belong to no phase
            out.append({"name": x["name"], "phase": 10_000, "kind": "extra",
                        "tasks": task_list(x["arms"], list(x["worlds"]), int(x.get("n", 12)), "select", 9000 + 300 * j)})
        json.dump([dict(x, tasks=[t.to_dict() for t in x["tasks"]]) for x in out], open(path, "w"))
        return out

    def _curriculum(self, k: int, demos: bool = False) -> Curriculum:
        p = self.phases[k]
        arms = [arm_dict(a) for a in p["arms"]]
        old = sorted({a for q in self.phases[:k] for a in q["arms"]} - set(p["arms"]))
        rate = float(p["rehearse_tasks"])
        if rate > 0 and old and not demos:
            arms += [arm_dict(a) for a in old]
        cur = Curriculum(arms, p["worlds"], "train", int(self.cfg.get("seed", 0)) + 101 * k + (77 if demos else 0),
                         adaptive=not demos and not (rate > 0 and old), gate=False)
        if rate > 0 and old and not demos:                   # old arms share `rate` of the practice episodes
            w = np.array([(1 - rate) / len(p["arms"])] * len(p["arms"]) + [rate / len(old)] * len(old))
            cur.weights = lambda w=w: w / w.sum()
        return cur

    def _run(self, jobs: list[dict]) -> list:
        self.tr.sync_workers()
        eps = self.tr.pool.run(jobs)
        self.tr.absorb(eps)
        return eps

    def evaluate(self, lst: dict, tag: str, agent: dict | None = None, mode: str = "greedy") -> dict:
        common = {"mode": mode, "learn": False, "keep_decisions": False, "human": False, "agent": agent or {}, "tag": tag}
        if lst.get("tasks") is None:
            jobs = [dict(common, eval=int(i), seed=10_000 + int(i), episode_id=f"{tag}{i}") for i in lst["indices"]]
        else:
            jobs = [dict(common, spec=s.to_dict(), seed=10_000 + i, episode_id=f"{tag}{i}") for i, s in enumerate(lst["tasks"])]
        t0 = time.time()
        eps = self._run(jobs)
        good = [e for e in eps if e is not None]
        r = summarize(good)
        return {"n": len(good), "success": r.get("success", 0.0), "success_ci": r.get("success_ci"), "by_arm": r.get("by_arm"),
                "valid_rate": r.get("valid_rate"), "decisions": r.get("decisions"), "failures": r.get("failures"),
                "per_task": [None if e is None else bool(e["summary"]["success"]) for e in eps],
                "wall_s": round(time.time() - t0, 1)}

    # ------------------------------------------------------------------ stages
    def begin(self, k: int) -> None:
        p, tr = self.phases[k], self.tr
        info = {}
        if hasattr(tr.algo, "begin_phase"):
            info = tr.algo.begin_phase(k, [tuple([arm_dict(a)["family"], arm_dict(a)["level"], arm_dict(a)["start"]]) for a in p["arms"]])
        self.x["ref_phase"] = PL.state_of(tr.model)
        fa = self.L["freeze_after"]
        self.x["frozen"] = fa is not None and k > int(fa)
        self.x["phase_log"].append({"phase": k, "name": p["name"], "begin": info, "frozen": self.x["frozen"],
                                    "env_decisions_start": tr.env_decisions, "train_success": [], "observe": {}})
        self.say(f"=== phase {k} {p['name']}: arms {p['arms']} worlds {p['worlds']}" + ("  [weights frozen]" if self.x["frozen"] else ""))

    def observe(self, k: int) -> None:
        p, tr = self.phases[k], self.tr
        want = int(p["demos_per_arm"])
        log = self.x["phase_log"][-1]
        if want <= 0:
            return
        cur = self._curriculum(k, demos=True)
        cur.counter = 5_000_000 + 20_000 * k
        got: dict = {}
        kept: list = []
        ran = 0
        mem = bool(self.L["memory"])
        for attempt in range(5):
            need = [a for a in p["arms"] if got.get(a, 0) < want]
            if not need:
                break
            specs = [s for s in cur.sample(4 * want * len(p["arms"])) if "/".join(map(str, s.arm)) in need][: 3 * want * len(need)]
            jobs = [{"spec": s.to_dict(), "mode": "teacher:auto", "seed": 1000 * k + ran + i, "episode_id": f"demo{k}_{ran + i}",
                     "learn": mem, "with_model": True, "tag": "demo"} for i, s in enumerate(specs)]
            ran += len(jobs)
            for e in self._run(jobs):
                if e is None or not e["summary"]["success"]:
                    continue
                a = "/".join(map(str, e["summary"]["arm"]))
                if got.get(a, 0) < want:
                    got[a] = got.get(a, 0) + 1
                    kept.append({"summary": e["summary"], "decisions": e["decisions"]})
        log["observe"] = {"demonstrations": dict(got), "teacher_episodes_run": ran}
        if not self.x["probe"]:
            self.x["probe"] = kept[: int(self.L["probe_episodes"])]
        if self.x["frozen"]:
            self.say(f"observe: {got} demonstrations written to episodic memory only (weights frozen)")
            return
        if hasattr(tr.algo, "observe"):
            hist = tr.algo.observe(kept, tr.store, int(p["observe_steps"]))
            tr.refresh_embeddings()
            for h in hist:
                tr.log("metrics.jsonl", dict(h, life_phase=k))
            if hist:
                log["observe"].update(steps=hist[-1]["step"], nll_first=hist[0]["bc_nll"], nll_last=hist[-1]["bc_nll"],
                                      weights_moved=hist[-1]["param_delta"])
                self.say(f"observe: {got}; imitation loss {hist[0]['bc_nll']:.3f} -> {hist[-1]['bc_nll']:.3f} in {hist[-1]['step']} steps")

    def holdout(self, k: int) -> None:
        """Sampled episodes on training-split tasks of this phase that are never trained on: the world model's
        held-out transitions for this phase (collected before any practice in it)."""
        n = int(self.L["wm_holdout"])
        if n <= 0 or str(k) in self.x["holdout"]:
            return
        cur = self._curriculum(k)
        cur.counter = 6_000_000 + 20_000 * k
        # half sampled from the policy at a high temperature, half uniformly random: transitions with both
        # outcomes (accepted and rejected actions), unlike the trained policy's own nearly always accepted ones
        jobs = [dict({"spec": s.to_dict(), "seed": 500 + i, "episode_id": f"hold{k}_{i}", "learn": False, "tag": "holdout"},
                     **({"mode": "random"} if i % 2 else {"mode": "sample", "temperature": 2.5}))
                for i, s in enumerate(cur.sample(n))]
        self.x["holdout"][str(k)] = [{"summary": e["summary"], "decisions": e["decisions"]} for e in self._run(jobs) if e is not None]

    def practice(self, k: int, it: int) -> dict:
        p, tr = self.phases[k], self.tr
        cur = self._curriculum(k)
        cur.counter = self.x["counter"]
        tr.curriculum = cur
        tr.cfg["episodes_per_iter"] = int(p["episodes_per_iter"])
        if self.x["frozen"]:
            rec = self._practice_frozen(cur, int(p["episodes_per_iter"]))
        else:
            rec = tr.train_iter()
        self.x["counter"] = cur.counter
        log = self.x["phase_log"][-1]
        log["train_success"].append(round(rec.get("train_success", 0.0), 4))
        if "wm_preq_ok_acc" in rec:
            log.setdefault("wm_prequential", []).append([round(rec["wm_preq_ok_acc"], 4), round(rec["wm_preq_majority"], 4)])
        self.say(f"  practice {it + 1}/{p['iters']}: train success {rec.get('train_success', 0):.3f} kl {rec.get('kl', 0):.4f} "
                 f"wm ok-acc {rec.get('wm_ok_acc', 0):.2f} dθ {rec.get('param_delta', 0):.3f} "
                 f"roll {rec.get('rollout_s', 0)}s upd {rec.get('update_s', 0)}s")
        return rec

    def _practice_frozen(self, cur: Curriculum, n: int) -> dict:
        """Acting and writing to episodic memory, with no weight update (the frozen-core baseline)."""
        tr = self.tr
        t0 = time.time()
        jobs = [{"spec": s.to_dict(), "mode": "sample", "seed": int(tr.rng.integers(0, 2**31 - 1)),
                 "episode_id": f"fz{tr.iter}e{i}", "learn": True, "tag": "train"} for i, s in enumerate(cur.sample(n))]
        eps = [e for e in self._run(jobs) if e is not None]
        tr.iter += 1
        tr.env_decisions += sum(len(e["decisions"]) for e in eps)
        tr.episodes += len(eps)
        rec = {"phase": "frozen", "iter": tr.iter, "env_decisions": tr.env_decisions, "rollout_s": round(time.time() - t0, 2),
               "update_s": 0.0, **{f"train_{a}": v for a, v in summarize(eps).items()}, "weights_changed": False,
               "memory": len(tr.memory), "tools": len(tr.library.tools)}
        tr.log("metrics.jsonl", rec)
        return rec

    def wm_report(self) -> dict:
        tr = self.tr
        wm = getattr(tr.algo, "wm", None)
        if wm is None:
            return {}
        from ..cam.worldmodel import heldout_report
        out = {}
        for j, eps in sorted(self.x["holdout"].items(), key=lambda kv: int(kv[0])):
            if eps:
                out[self.phases[int(j)]["name"]] = heldout_report(tr.model, wm, eps, tr.store, tr.device, tr.algo.rule.gamma)
        return out

    def close(self, k: int) -> dict:
        p, tr = self.phases[k], self.tr
        t0 = time.time()
        cons = tr.algo.end_phase(tr.store) if (hasattr(tr.algo, "end_phase") and not self.x["frozen"]) else {}
        row: dict = {"after_phase": k, "name": p["name"], "env_decisions": tr.env_decisions, "episodes": tr.episodes,
                     "iter": tr.iter, "lists": {}, "consolidation": cons, "frozen": self.x["frozen"],
                     "memory_items": len(tr.memory), "tools": len(tr.library.tools),
                     "buffer": len(getattr(tr.algo, "buffer", []) or [])}
        last = k == len(self.phases) - 1
        for lst in self.lists:
            if lst["kind"] == "unseen" and not last and self.L["eval"].get("unseen_only_at_end", True):
                continue                                   # unseen-world lists: once, with the final weights
            row["lists"][lst["name"]] = self.evaluate(lst, f"L{k}_{lst['name']}_")
        if last and self.L["memory"]:              # how much of the final behaviour needs the written memory?
            row["lists_memory_off"] = {lst["name"]: self.evaluate(lst, f"L{k}m0_{lst['name']}_",
                                                                  {"use_memory": False, "use_tools": False})
                                       for lst in self.lists if lst["kind"] == "seen"}
        if self.L["eval"].get("sampled", True):     # the stochastic policy (what RL optimises) on the phases seen so far
            row["lists_sampled"] = {lst["name"]: self.evaluate(lst, f"L{k}s_{lst['name']}_", mode="sample")
                                    for lst in self.lists if lst["kind"] == "seen" and lst["phase"] <= k}
        row["world_model"] = self.wm_report()
        probe = self.x["probe"]
        row["plasticity"] = dict(PL.diagnostics(tr.model, probe, tr.store, tr.device),
                                 probe_fit=PL.probe_fit(tr.model, probe, tr.store, tr.device, int(self.L["probe_steps"])))             if int(self.L["probe_steps"]) > 0 else {}
        row["weights"] = {"digest": PL.weights_digest(tr.model), "since_init": PL.param_change(tr.model, self.x["ref_init"]),
                          "this_phase": PL.param_change(tr.model, self.x["ref_phase"]) if self.x["ref_phase"] else {}}
        if tr.backbone is not None:
            row["backbone"] = dict(tr.backbone.n_params(), version=tr.backbone.version, change_from_pretrained=tr.backbone.change())
        row["optimizer_steps"] = int(getattr(tr.algo, "n_updates", 0))
        row["lr_scale"] = float(getattr(tr.algo, "lr_scale", 1.0))
        row["close_wall_s"] = round(time.time() - t0, 1)
        row["phase_log"] = self.x["phase_log"][-1]
        self.x["rows"].append(row)
        self.write_results()
        seen = [row["lists"][q["name"]]["success"] for q in self.phases[: k + 1]]
        self.say(f"  after {p['name']}: " + "  ".join(f"{q['name']} {row['lists'][q['name']]['success']:.2f}" for q in self.phases)
                 + f"   | mean over seen {np.mean(seen):.3f}")
        return row

    def write_results(self) -> None:
        tr = self.tr
        if "explore" in tr.cfg:
            self.L["explore"] = tr.cfg["explore"]
        out = {"name": self.cfg.get("name"), "seed": self.cfg.get("seed", 0), "phases": self.phases,
               "settings": {"lifetime": {k: v for k, v in self.L.items() if k != "phases"}, "algo": tr.cfg["algo"],
                            "backbone": tr.cfg.get("backbone"), "agent": tr.cfg["agent"]},
               "params": {"controller": tr.model.n_params(), "groups": tr.algo.group_sizes() if hasattr(tr.algo, "group_sizes") else {},
                          "backbone": tr.backbone.n_params() if tr.backbone is not None else None},
               "rows": self.x["rows"], "derived": derive(self.x["rows"], self.phases), "restarts": self.x.get("restarts", 0),
               "digest_init": self.x.get("digest_init"), "wall": tr.wall, "env_decisions": tr.env_decisions,
               "episodes": tr.episodes, "finished": self.x["stage"] == "done"}
        tmp = os.path.join(self.dir, "results.json.tmp")
        json.dump(out, open(tmp, "w"), indent=1, default=str)
        os.replace(tmp, os.path.join(self.dir, "results.json"))

    # ------------------------------------------------------------------ the loop
    def step(self) -> bool:
        """Do the next unit of work and checkpoint. Returns False when the lifetime is complete."""
        x = self.x
        k = x["phase"]
        if x["stage"] == "done" or k >= len(self.phases):
            return False
        if x["stage"] == "begin":
            self.begin(k)
            self.observe(k)
            self.holdout(k)
            x["stage"], x["it"] = "practice", 0
        elif x["stage"] == "practice":
            if x["it"] < int(self.phases[k]["iters"]):
                self.practice(k, x["it"])
                x["it"] += 1
            if x["it"] >= int(self.phases[k]["iters"]):
                x["stage"] = "close"
        elif x["stage"] == "close":
            self.close(k)
            if self.L.get("keep_phase_checkpoints"):
                self.tr.save(f"phase_{k}.pt", inline=True)
            x["phase"], x["stage"], x["it"] = k + 1, ("begin" if k + 1 < len(self.phases) else "done"), 0
        self.save()
        if x["stage"] == "done":
            self.write_results()
            open(os.path.join(self.dir, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
        return x["stage"] != "done"

    def run(self, max_steps: int | None = None) -> dict:
        n = 0
        try:
            while self.step():
                n += 1
                if max_steps is not None and n >= max_steps:
                    break
        finally:
            self.tr.close()
        return json.load(open(os.path.join(self.dir, "results.json"))) if os.path.exists(os.path.join(self.dir, "results.json")) else {}
