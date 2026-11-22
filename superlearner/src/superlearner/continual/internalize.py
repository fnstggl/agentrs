"""The Parametric Internalization Test: is a newly acquired capability IN THE WEIGHTS?

    superlearner pit base   --config configs/pit.yaml --seed 0            the learner before (familiar task families)
    superlearner pit route  --config configs/pit.yaml --seed 0 --family launch --route observe_practice
    superlearner pit report --config configs/pit.yaml                     tables over seeds, families and routes

Procedure (one seed, one unfamiliar family, one route of learning):

  1. `base`: a learner is trained on the familiar families. Its weights are exported: W_before.
  2. W_before is evaluated on held-out tasks of the UNFAMILIAR family (step 7 below, same code path).
  3. `route`: starting from the base learner's full state, the learner meets the unfamiliar family and learns by the
     route's means (practice only / practice with more exploration / observation only / observation and practice).
  4. The change of every parameter group is recorded (L2, relative, digests of the weights before and after).
  5. The weights are exported to a file that holds tensors and nothing else: W_after.
  6. Nothing else of the learning run is used again: no episode, no replay buffer, no episodic memory, no tool
     library, no transcript, no cached embedding, no optimiser.
  7. A NEW PROCESS builds a new learner from the configuration, loads W_after and acts greedily on held-out tasks:
     the unfamiliar family in the kinds of world it practised in and in kinds of world it never saw, related
     families that were never trained, and the familiar families (retention). Episodic memory and tool-making
     are off; the controller's recurrent state starts from zero in every episode; worlds, seeds and all
     identifiers differ from training (the `select` split).
  8. The same tasks under W_before (step 2) give the paired comparison: same network, same tasks, only the weights
     differ.

Controls: `handle` runs W_after with the actor registered under a different name (identifiers derived from the
name then differ from everything seen in training); the `frozen_memory` route keeps W_before and writes the same
experience to episodic memory only (what non-parametric learning alone gives); `memory_observe_practice` trains
with memory on and is evaluated with its memory (the combined system) and, in the fresh process, without it.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from ..train.trainer import merge
from ..world.tasks import TaskSpec, task_list
from . import plasticity as PL
from .lifetime import Lifetime


# ------------------------------------------------------------------------------------------------ configuration
def life_cfg(cfg: dict, seed: int, phases: list[dict], over: dict | None = None, extra_lists: list | None = None) -> dict:
    c = copy.deepcopy(cfg)
    c.pop("pit", None)
    c = merge(c, over or {})
    c["seed"] = seed
    L = c.setdefault("lifetime", {})
    L["phases"] = phases
    L.setdefault("eval", {})
    L["eval"] = dict(L["eval"], unseen_worlds=[], extra_lists=extra_lists or [], sampled=False)
    return c


def export_weights(tr, path: str) -> dict:
    """A file with the learner's weights and nothing else."""
    w = {"model": {k: v.detach().cpu() for k, v in tr.model.state_dict().items()},
         "wm": ({k: v.detach().cpu() for k, v in tr.algo.wm.state_dict().items()} if getattr(tr.algo, "wm", None) is not None else None),
         "backbone": tr.backbone.trainable_state() if (tr.backbone is not None and tr.backbone.trainable) else None,
         "digest": PL.weights_digest(tr.model)}
    torch.save(w, path + ".tmp")
    os.replace(path + ".tmp", path)
    return {"path": path, "digest": w["digest"], "bytes": os.path.getsize(path)}


def eval_lists(P: dict, fam: dict) -> list[dict]:
    base = P["base_phase"]
    worlds = fam.get("worlds") or base["worlds"]
    n = int(P["eval"].get("n", 24))
    out = [{"name": "unfamiliar", "arms": fam["arms"], "worlds": worlds, "n": n, "offset": 12_000},
           {"name": "unfamiliar_unseen_worlds", "arms": fam["arms"], "worlds": P["eval"]["unseen_worlds"],
            "n": int(P["eval"].get("n_unseen", 16)), "offset": 13_000},
           {"name": "familiar", "arms": base["arms"], "worlds": base["worlds"], "n": int(P["eval"].get("n_familiar", 20)),
            "offset": 15_000}]
    if fam.get("related"):
        out.insert(2, {"name": "related_untrained", "arms": fam["related"], "worlds": worlds,
                       "n": int(P["eval"].get("n_related", 16)), "offset": 14_000})
    for x in out:
        x["tasks"] = [t.to_dict() for t in task_list(x["arms"], list(x["worlds"]), x["n"], "select", x["offset"])]
    return out


# ------------------------------------------------------------------------------------------------ fresh evaluation
def fresh_eval_main(argv=None) -> int:
    """Entry point of the new process (step 7). Reads: a configuration, a weights file, task lists. Nothing else."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--handle", default="agent")
    ap.add_argument("--memory-from", dest="memory_from")
    ap.add_argument("--sampled", action="store_true")
    a = ap.parse_args(argv)
    from ..train.trainer import Trainer, summarize
    cfg = json.load(open(a.config))
    with_mem = bool(a.memory_from)
    cfg = merge(cfg, {"eval_every": 0, "ckpt_every": 0, "memory": {"enabled": with_mem}, "demos": {"episodes": 0},
                      "agent": dict(cfg.get("agent") or {}, use_memory=with_mem, use_tools=with_mem)})
    cfg.pop("init_from", None)
    scratch = a.out + ".run"
    shutil.rmtree(scratch, ignore_errors=True)
    tr = Trainer(cfg, scratch)
    out: dict = {"handle": a.handle, "memory": with_mem, "lists": {}, "pid": os.getpid(), "started": time.time()}
    try:
        w = torch.load(a.weights, map_location="cpu", weights_only=False)
        assert set(w) <= {"model", "wm", "backbone", "digest"}, "a weights file holds tensors only"
        tr.model.load_state_dict(w["model"])
        if w.get("backbone") is not None and tr.backbone is not None:
            tr.backbone.load_trainable_state(w["backbone"])
        out["weights_digest"] = PL.weights_digest(tr.model)
        assert out["weights_digest"] == w["digest"]
        if with_mem:                                         # the combined system: the same weights plus the memory written
            ck = torch.load(a.memory_from, map_location="cpu", weights_only=False)
            tr.memory.load_state_dict(ck["memory"])
            tr.library.load_state_dict(ck["library"])
            tr.pool.sync(tr.memory.items, tr.library.state_dict(), reset=True)
            tr.mem_synced = len(tr.memory.items)
        out["state_at_start"] = {"memory_items": len(tr.memory), "tools": len(tr.library.tools), "embedding_rows": tr.store.n - 1,
                                 "replay_buffer": len(getattr(tr.algo, "buffer", []) or []), "texts": len(tr.texts)}
        for li, lst in enumerate(json.load(open(a.tasks))):
            for mode in (["greedy", "sample"] if (a.sampled and li == 0) else ["greedy"]):   # sampled: the unfamiliar list
                jobs = [{"spec": t, "mode": mode, "seed": 10_000 + i, "episode_id": f"pit{i}", "learn": False,
                         "keep_decisions": False, "handle": a.handle, "tag": "pit"} for i, t in enumerate(lst["tasks"])]
                tr.sync_workers()
                eps = tr.pool.run(jobs)
                for attempt in range(3):                  # an episode that did not run is an error, never a failure
                    miss = [i for i, e in enumerate(eps) if e is None]
                    if not miss:
                        break
                    time.sleep(20 * (attempt + 1))
                    for i, e in zip(miss, tr.pool.run([jobs[i] for i in miss])):
                        eps[i] = e
                if any(e is None for e in eps):
                    raise RuntimeError(f"{sum(e is None for e in eps)} of {len(eps)} evaluation episodes did not run "
                                       f"({lst['name']}, {mode}): {tr.pool.errors[-1][-600:] if tr.pool.errors else ''}")
                tr.absorb(eps)
                good = [e for e in eps if e is not None]
                r = summarize(good)
                out["lists"][lst["name"] + ("" if mode == "greedy" else "@sampled")] = {
                    "n": len(good), "success": r.get("success", 0.0), "success_ci": r.get("success_ci"), "by_arm": r.get("by_arm"),
                    "valid_rate": r.get("valid_rate"), "failures": r.get("failures"),
                    "per_task": [None if e is None else bool(e["summary"]["success"]) for e in eps]}
    finally:
        tr.close()
        shutil.rmtree(scratch, ignore_errors=True)
    json.dump(out, open(a.out + ".tmp", "w"), indent=1)
    os.replace(a.out + ".tmp", a.out)
    return 0


def fresh_eval(cfg: dict, weights: str, tasks: list, out: str, handle: str = "agent", memory_from: str | None = None,
               sampled: bool = False, workers: int = 8) -> dict:
    if not os.path.exists(out):
        c = dict(copy.deepcopy(cfg), workers=workers)
        c.pop("lifetime", None)
        c.pop("pit", None)
        json.dump(c, open(out + ".cfg.json", "w"))
        json.dump(tasks, open(out + ".tasks.json", "w"))
        cmd = [sys.executable, "-m", "superlearner.continual.internalize", "--config", out + ".cfg.json", "--weights", weights,
               "--tasks", out + ".tasks.json", "--out", out, "--handle", handle] + (["--sampled"] if sampled else []) \
            + (["--memory-from", memory_from] if memory_from else [])
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"fresh evaluation failed:\n{r.stderr[-3000:]}")
    return json.load(open(out))


# ------------------------------------------------------------------------------------------------ stages
def seed_dir(cfg: dict, out_dir: str, seed: int) -> str:
    return os.path.join(out_dir, f"s{seed}")


def base_name(route: dict) -> str:
    return "base_mem" if route.get("memory") else "base"


def run_base(cfg: dict, out_dir: str, seed: int, memory: bool = False) -> str:
    P = cfg["pit"]
    rd = os.path.join(seed_dir(cfg, out_dir, seed), "base_mem" if memory else "base")
    done = os.path.join(rd, "weights.pt")
    if os.path.exists(done):
        return rd
    c = life_cfg(cfg, seed, [dict(P["base_phase"], name="familiar")], {"lifetime": {"memory": memory}})
    c["name"] = os.path.basename(rd)
    life = Lifetime(c, rd)
    while life.step():
        pass
    export_weights(life.tr, done)
    life.tr.close()
    return rd


def run_route(cfg: dict, out_dir: str, seed: int, family: str, route: str, workers: int = 8) -> dict:
    P = cfg["pit"]
    fam = next(f for f in P["families"] if f["name"] == family)
    R = P["routes"][route]
    sd = seed_dir(cfg, out_dir, seed)
    base = run_base(cfg, out_dir, seed, bool(R.get("memory")))
    rd = os.path.join(sd, f"{family}__{route}")
    res_path = os.path.join(rd, "pit.json")
    if os.path.exists(res_path):
        return json.load(open(res_path))
    os.makedirs(os.path.join(rd, "ckpt"), exist_ok=True)
    if not os.path.exists(os.path.join(rd, "ckpt", "latest.pt")):
        shutil.copy(os.path.join(base, "ckpt", "latest.pt"), os.path.join(rd, "ckpt", "latest.pt"))
    up = {"name": family, "arms": fam["arms"], "worlds": fam.get("worlds") or P["base_phase"]["worlds"],
          "demos_per_arm": int(R.get("demos_per_arm", 0)), "observe_steps": int(R.get("observe_steps", 120)),
          "iters": int(R.get("iters", 0)), "episodes_per_iter": int(R.get("episodes_per_iter", 64))}
    over = copy.deepcopy(R.get("over") or {})
    over.setdefault("lifetime", {})
    over["lifetime"]["memory"] = bool(R.get("memory"))
    if R.get("freeze"):
        over["lifetime"]["freeze_after"] = 0
    c = life_cfg(cfg, seed, [dict(P["base_phase"], name="familiar"), up], over)
    c["name"] = os.path.basename(rd)
    life = Lifetime(c, rd)
    while life.step():
        pass
    tr = life.tr
    row = life.x["rows"][-1]
    wa = export_weights(tr, os.path.join(rd, "weights.pt"))
    tcfg = copy.deepcopy(tr.cfg)
    tr.close()
    # the fresh evaluations are other processes on the same GPU: this one gives its GPU memory back first
    del life, tr
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    lists = eval_lists(P, fam)
    wb = os.path.join(base, "weights.pt")
    before = fresh_eval(tcfg, wb, lists, os.path.join(sd, f"before__{family}__{base_name(R)}.json"), workers=workers)
    after = fresh_eval(tcfg, wa["path"], lists, os.path.join(rd, "after.json"), sampled=True, workers=workers)
    out = {"seed": seed, "family": family, "route": route, "route_settings": R, "arms": fam["arms"],
           "weights_before": {"digest": before["weights_digest"], "file": wb},
           "weights_after": wa, "weights_changed": before["weights_digest"] != wa["digest"],
           "parameter_change": row["weights"]["this_phase"], "optimizer_steps_total": row["optimizer_steps"],
           "learning": {"observe": row["phase_log"].get("observe"), "train_success_by_iter": row["phase_log"].get("train_success"),
                        "env_decisions": row["env_decisions"] - row["phase_log"].get("env_decisions_start", 0)},
           "in_run_eval_with_its_own_state": {k: {"success": v["success"], "n": v["n"]} for k, v in row["lists"].items()},
           "fresh_process": {"before": before["lists"], "after": after["lists"], "state_at_start": after["state_at_start"],
                             "pid": after["pid"]}}
    if P["eval"].get("handle_control") and not R.get("freeze"):
        h = fresh_eval(tcfg, wa["path"], lists[:1], os.path.join(rd, "after_handle.json"), handle=str(P["eval"]["handle_control"]), workers=workers)
        out["fresh_process"]["after_other_actor_name"] = h["lists"]
    if R.get("memory"):                       # the combined system: the same weights with the memory it wrote
        m = fresh_eval(tcfg, wa["path"], lists[:1], os.path.join(rd, "after_memory.json"),
                       memory_from=os.path.join(rd, "ckpt", "latest.pt"), workers=workers)
        out["fresh_process"]["after_with_its_memory"] = m["lists"]
        out["fresh_process"]["memory_state"] = m["state_at_start"]
    json.dump(out, open(res_path + ".tmp", "w"), indent=1, default=str)
    os.replace(res_path + ".tmp", res_path)
    return out


# ------------------------------------------------------------------------------------------------ report
def paired(before: list[list], after: list[list], n_boot: int = 4000, seed: int = 0) -> dict:
    """Difference of success rates on the same tasks; two-level bootstrap (seeds, then tasks within a seed)."""
    pairs = [[(float(b), float(a)) for b, a in zip(bs, as_) if b is not None and a is not None] for bs, as_ in zip(before, after)]
    pairs = [p for p in pairs if p]
    if not pairs:
        return {}
    flat = [x for p in pairs for x in p]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        tot, n = 0.0, 0
        for si in rng.integers(0, len(pairs), size=len(pairs)):
            p = pairs[int(si)]
            for ti in rng.integers(0, len(p), size=len(p)):
                tot += p[int(ti)][1] - p[int(ti)][0]
                n += 1
        diffs.append(tot / n)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"n_tasks": len(flat), "seeds": len(pairs), "before": float(np.mean([b for b, _ in flat])),
            "after": float(np.mean([a for _, a in flat])), "diff": float(np.mean([a - b for b, a in flat])),
            "ci95": [float(lo), float(hi)], "gained": int(sum(a > b for b, a in flat)), "lost": int(sum(a < b for b, a in flat)),
            "by_seed": [[float(np.mean([b for b, _ in p])), float(np.mean([a for _, a in p]))] for p in pairs]}


def report(cfg: dict, out_dir: str) -> dict:
    P = cfg["pit"]
    out: dict = {"families": {}, "config": P}
    seeds = sorted(int(d[1:]) for d in os.listdir(out_dir) if d.startswith("s") and d[1:].isdigit())
    for fam in P["families"]:
        F: dict = {}
        for route in P["routes"]:
            rs = []
            for s in seeds:
                p = os.path.join(out_dir, f"s{s}", f"{fam['name']}__{route}", "pit.json")
                if os.path.exists(p):
                    rs.append(json.load(open(p)))
            if not rs:
                continue
            row: dict = {"seeds": [r["seed"] for r in rs], "lists": {}}
            for name in rs[0]["fresh_process"]["before"]:
                row["lists"][name] = paired([r["fresh_process"]["before"][name]["per_task"] for r in rs],
                                            [r["fresh_process"]["after"][name]["per_task"] for r in rs])
            for extra in ("after_other_actor_name", "after_with_its_memory"):
                if all(extra in r["fresh_process"] for r in rs):
                    row[extra] = paired([r["fresh_process"]["before"]["unfamiliar"]["per_task"] for r in rs],
                                        [r["fresh_process"][extra]["unfamiliar"]["per_task"] for r in rs])
            sm = [r["fresh_process"]["after"].get("unfamiliar@sampled", {}).get("success") for r in rs]
            row["after_sampled_unfamiliar"] = float(np.mean([x for x in sm if x is not None])) if any(x is not None for x in sm) else None
            row["weights_changed"] = [r["weights_changed"] for r in rs]
            row["relative_weight_change"] = {g: float(np.mean([r["parameter_change"].get(g, {}).get("relative", 0.0) for r in rs]))
                                             for g in ("input", "encoder", "recurrent", "policy", "critic", "all")}
            row["env_decisions"] = float(np.mean([r["learning"]["env_decisions"] for r in rs]))
            row["train_success_by_iter"] = [r["learning"]["train_success_by_iter"] for r in rs]
            row["state_at_start_of_fresh_process"] = rs[0]["fresh_process"]["state_at_start"]
            F[route] = row
        out["families"][fam["name"]] = F
    return out


def main(a) -> int:
    from ..cli import RUNS, apply_sets, load_config, resolve_llm
    cfg = apply_sets(resolve_llm(load_config(a.config)), a.set)
    out_dir = a.out_dir or os.path.join(RUNS, cfg.get("name", "pit"))
    os.makedirs(out_dir, exist_ok=True)
    if a.workers is not None:
        cfg["workers"] = a.workers
    if a.stage == "base":
        print(run_base(cfg, out_dir, a.seed, bool(a.memory)))
    elif a.stage == "route":
        r = run_route(cfg, out_dir, a.seed, a.family, a.route, workers=int(cfg.get("workers", 8)))
        b, f = r["fresh_process"]["before"]["unfamiliar"], r["fresh_process"]["after"]["unfamiliar"]
        print(f"{a.family} / {a.route} / seed {a.seed}: unfamiliar family {b['success']:.3f} -> {f['success']:.3f} "
              f"(fresh process, weights only); weights changed: {r['weights_changed']}")
    else:
        rep = report(cfg, out_dir)
        from ..experiments import _save
        _save(a.save_as or "pit", rep)
        for fam, F in rep["families"].items():
            for route, row in F.items():
                u = row["lists"].get("unfamiliar", {})
                print(f"{fam:10s} {route:26s} {u.get('before', 0):.3f} -> {u.get('after', 0):.3f}  diff {u.get('diff', 0):+.3f} "
                      f"CI {u.get('ci95')}  seeds {row['seeds']}")
    return 0


if __name__ == "__main__":
    sys.exit(fresh_eval_main())
