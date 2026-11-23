"""Aggregate the continual-learning runs into the JSON files the documentation's tables are generated from.

    superlearner continual-report [--life runs/life] [--pit runs/pit] [--meta runs/meta/continual1]

Writes, under docs/experiments/:
    continual_lifetime.json      conditions x seeds: retention matrices, acquisition, forgetting, adaptation to the
                                 shifted worlds, unseen worlds, plasticity, world model, weight changes, costs, and
                                 task-paired comparisons between conditions
    life/<run>.json              each run's own results (per-task outcomes kept, so every number can be recomputed)
    pit.json                     the Parametric Internalization Test (written by `superlearner pit report`)
    meta_continual/              the search over update rules: ledger, reports, diffs of every candidate
Nothing here trains or evaluates; it only reads what the runs wrote.
"""
from __future__ import annotations

import glob
import json
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "docs", "experiments")

ORDER = ["frozen_memory", "naive", "memory", "rehearsal", "consolidate", "backbone_lora", "backbone_top4", "backbone_full"]
LABEL = {"frozen_memory": "1 frozen core + written memory", "naive": "2 trainable core, no protection",
         "memory": "3 trainable core + external memory", "rehearsal": "4a trainable core + rehearsal",
         "consolidate": "4b trainable core + rehearsal + consolidation", "backbone_lora": "5 (LoRA) 4b + backbone adapters",
         "backbone_top4": "5 (top 4 blocks) 4b + backbone top blocks", "backbone_full": "5 (full) 4b + whole backbone"}
PAIRS = [("naive", "frozen_memory"), ("memory", "naive"), ("rehearsal", "naive"), ("consolidate", "naive"),
         ("consolidate", "rehearsal"), ("consolidate", "frozen_memory"), ("backbone_lora", "consolidate"),
         ("backbone_top4", "consolidate"), ("backbone_full", "consolidate")]


def boot_paired(a: list[list], b: list[list], n_boot: int = 4000, seed: int = 0) -> dict:
    """Mean of (a - b) over tasks present in both, two-level bootstrap: seeds, then tasks within a seed."""
    pairs = [[(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None] for xs, ys in zip(a, b)]
    pairs = [p for p in pairs if p]
    if not pairs:
        return {}
    flat = [t for p in pairs for t in p]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        tot, n = 0.0, 0
        for si in rng.integers(0, len(pairs), size=len(pairs)):
            p = np.asarray(pairs[int(si)])
            idx = rng.integers(0, len(p), size=len(p))
            tot += float((p[idx, 0] - p[idx, 1]).sum())
            n += len(p)
        diffs.append(tot / n)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"n_tasks": len(flat), "seeds": len(pairs), "a": float(np.mean([x for x, _ in flat])), "b": float(np.mean([y for _, y in flat])),
            "diff": float(np.mean([x - y for x, y in flat])), "ci95": [float(lo), float(hi)],
            "a_only": int(sum(x > y for x, y in flat)), "b_only": int(sum(x < y for x, y in flat))}


def load_life(root: str) -> dict:
    runs: dict = {}
    for p in sorted(glob.glob(os.path.join(root, "*", "results.json"))):
        r = json.load(open(p))
        if not r.get("finished"):
            continue
        holes = [(row["name"], key, k) for row in r["rows"] for key in ("lists", "lists_sampled", "lists_memory_off")
                 for k, v in (row.get(key) or {}).items() if any(x is None for x in v["per_task"])]
        if holes:                                    # an evaluation in which episodes did not run is not a result
            print(f"EXCLUDED {p}: evaluation episodes missing in {holes[:3]}")
            continue
        name = os.path.basename(os.path.dirname(p))
        swap = os.path.join(os.path.dirname(p), "backbone_swap.json")
        if os.path.exists(swap):
            r["_backbone_swap"] = json.load(open(swap))
        for key, fn in (("_verify", "verify.json"), ("_verify_memory", "verify_memory.json")):
            vp = os.path.join(os.path.dirname(p), fn)
            if os.path.exists(vp):
                v = json.load(open(vp))
                if not any(x is None for L in v["lists"].values() for x in L["per_task"]):
                    r[key] = v
        cond, _, sd = name.rpartition("_s")
        runs.setdefault(cond, {})[int(sd)] = r
    return runs


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def cond_summary(rs: dict) -> dict:
    seeds = sorted(rs)
    first = rs[seeds[0]]
    names = [p["name"] for p in first["phases"]]
    K = len(names)
    out: dict = {"seeds": seeds, "phases": names}
    mats = [[[r["rows"][k]["lists"][n]["success"] for n in names] for k in range(K)] for r in (rs[s] for s in seeds)]
    out["matrix_mean"] = np.mean(mats, 0).round(4).tolist()
    out["matrix_by_seed"] = mats
    smats = []
    for s in seeds:
        rows = rs[s]["rows"]
        smats.append([[rows[k].get("lists_sampled", {}).get(n, {}).get("success") for n in names] for k in range(K)])
    out["matrix_sampled_by_seed"] = smats
    fin = [m[-1] for m in mats]
    out["final_by_phase"] = np.mean(fin, 0).round(4).tolist()
    out["final_mean"] = {"mean": float(np.mean([np.mean(f) for f in fin])), "by_seed": [float(np.mean(f)) for f in fin]}
    sfin = [[x for x in sm[-1] if x is not None] for sm in smats]
    out["final_mean_sampled"] = {"mean": _mean([np.mean(f) for f in sfin if f]), "by_seed": [float(np.mean(f)) if f else None for f in sfin]}
    # the five taught phases only (the last phase re-tests earlier arms in other worlds without new demonstrations)
    out["final_mean_taught"] = {"mean": float(np.mean([np.mean(f[: K - 1]) for f in fin])), "by_seed": [float(np.mean(f[: K - 1])) for f in fin]}
    forg, acq, zs, learned = [], [], [], []
    for m in mats:
        forg.append([max(m[k][j] for k in range(j, K)) - m[K - 1][j] for j in range(K - 1)])
        acq.append([m[k][k] - (m[k - 1][k] if k else 0.0) for k in range(K)])
        zs.append([(m[k - 1][k] if k else 0.0) for k in range(K)])
        learned.append([m[k][k] for k in range(K)])
    out["forgetting"] = {"mean": float(np.mean(forg)), "by_phase": np.mean(forg, 0).round(4).tolist(), "by_seed": [float(np.mean(f)) for f in forg]}
    out["acquisition"] = {"mean": float(np.mean(acq)), "by_phase": np.mean(acq, 0).round(4).tolist(), "by_seed": [float(np.mean(f)) for f in acq]}
    out["right_after_learning"] = {"mean": float(np.mean(learned)), "by_phase": np.mean(learned, 0).round(4).tolist()}
    out["zero_shot_before_training"] = np.mean(zs, 0).round(4).tolist()
    out["shift"] = {"before": float(np.mean([m[K - 2][K - 1] for m in mats])), "after": float(np.mean([m[K - 1][K - 1] for m in mats])),
                    "by_seed": [[m[K - 2][K - 1], m[K - 1][K - 1]] for m in mats]}
    un = [[v["success"] for n, v in rs[s]["rows"][-1]["lists"].items() if n.endswith("@unseen")] for s in seeds]
    out["unseen_worlds_final"] = {"mean": _mean([np.mean(u) for u in un if u]), "by_seed": [float(np.mean(u)) if u else None for u in un]}
    moff = [[v["success"] for v in rs[s]["rows"][-1].get("lists_memory_off", {}).values()] for s in seeds]
    if any(moff):
        out["final_mean_memory_off"] = {"mean": _mean([np.mean(u) for u in moff if u]), "by_seed": [float(np.mean(u)) if u else None for u in moff]}
    out["memory_items_final"] = _mean([rs[s]["rows"][-1].get("memory_items") for s in seeds])
    out["tools_final"] = _mean([rs[s]["rows"][-1].get("tools") for s in seeds])
    # learning efficiency over the lifetime: training success in the first practice iteration of each phase, and mean
    out["train_first_iter"] = [_mean([(rs[s]["rows"][k]["phase_log"]["train_success"] or [None])[0] for s in seeds]) for k in range(K)]
    out["train_mean"] = [_mean([float(np.mean(rs[s]["rows"][k]["phase_log"]["train_success"])) if rs[s]["rows"][k]["phase_log"]["train_success"] else None
                                for s in seeds]) for k in range(K)]
    out["env_decisions"] = _mean([rs[s]["env_decisions"] for s in seeds])
    out["episodes"] = _mean([rs[s]["episodes"] for s in seeds])
    out["wall_s"] = {k: _mean([rs[s]["wall"].get(k) for s in seeds]) for k in ("rollout_s", "update_s")}
    out["optimizer_steps"] = _mean([rs[s]["rows"][-1].get("optimizer_steps") for s in seeds])
    # plasticity and representation, per phase
    pl = lambda key: [_mean([_get(rs[s]["rows"][k]["plasticity"], key) for s in seeds]) for k in range(K)]  # noqa: E731
    out["plasticity"] = {"probe_loss_after_40_steps": pl("probe_fit.loss_last"), "probe_chance": _mean([_get(rs[s]["rows"][0]["plasticity"], "probe_fit.chance") for s in seeds]),
                         "dormant_fraction": pl("dormant_fraction"), "state_effective_rank": pl("state_effective_rank")}
    wm = []
    for k in range(K):
        acc, maj, nll, base, cos, cosb = [], [], [], [], [], []
        for s in seeds:
            for v in rs[s]["rows"][k]["world_model"].values():
                acc.append(v["ok_accuracy"]); maj.append(v["ok_accuracy_base_rate"]); nll.append(v["ok_nll"]); base.append(v["ok_nll_base_rate"])
                cos.append(v["next_latent_cosine"]); cosb.append(v["next_latent_cosine_mean_predictor"])
        wm.append({"ok_accuracy": _mean(acc), "majority": _mean(maj), "ok_nll": _mean(nll), "base_nll": _mean(base),
                   "next_cosine": _mean(cos), "next_cosine_mean_predictor": _mean(cosb)})
    out["world_model_heldout"] = wm
    # the held-out transitions of the FIRST phase, re-predicted after every later phase (does the model keep them?)
    out["world_model_first_phase_over_time"] = [_mean([_get(rs[s]["rows"][k]["world_model"], f"{names[0]}.ok_accuracy") for s in seeds]) for k in range(K)]
    pq = [[x for k in range(K) for x in (rs[s]["rows"][k]["phase_log"].get("wm_prequential") or [])] for s in seeds]
    pq = [p for p in pq if p]
    if pq:
        out["world_model_prequential"] = {"accuracy": float(np.mean([x[0] for p in pq for x in p])), "majority": float(np.mean([x[1] for p in pq for x in p])),
                                          "first_iteration": [float(np.mean([p[0][0] for p in pq])), float(np.mean([p[0][1] for p in pq]))],
                                          "last_iteration": [float(np.mean([p[-1][0] for p in pq])), float(np.mean([p[-1][1] for p in pq]))]}
    last = [rs[s]["rows"][-1] for s in seeds]
    out["weights_since_init_relative"] = {g: _mean([_get(r["weights"]["since_init"], f"{g}.relative") for r in last])
                                          for g in ("input", "encoder", "recurrent", "policy", "critic", "all")}
    out["weights_changed_after_first_phase"] = [rs[s]["rows"][0]["weights"]["digest"] != rs[s]["rows"][-1]["weights"]["digest"] for s in seeds]
    bb = [r.get("backbone") for r in last]
    if bb[0]:
        out["backbone"] = {"mode": bb[0]["mode"], "trainable": bb[0]["trainable"], "total": bb[0]["total"],
                           "relative_change_from_pretrained": _mean([(b.get("change_from_pretrained") or {}).get("relative") for b in bb]),
                           "l2_change_from_pretrained": _mean([(b.get("change_from_pretrained") or {}).get("l2") for b in bb])}
    ver = {}
    for key, tag in (("_verify", "weights only"), ("_verify_memory", "with its memory")):
        have = [s for s in seeds if rs[s].get(key)]
        if have:
            L = rs[have[0]][key]["lists"]
            ver[tag] = {"seeds": have, "lists": {n: {"mean": float(np.mean([rs[s][key]["lists"][n]["success"] for s in have])),
                                                     "by_seed": [rs[s][key]["lists"][n]["success"] for s in have],
                                                     "n_tasks": L[n]["n"]} for n in L},
                        "by_arm": {a: float(np.mean([rs[s][key]["lists"][n]["by_arm"][a] for s in have for n in L]))
                                   for a in L[next(iter(L))]["by_arm"]}}
    if ver:
        out["verify"] = ver
    out["params"] = first["params"]
    out["restarts"] = [rs[s].get("restarts", 0) for s in seeds]
    sw = [rs[s].get("_backbone_swap") for s in seeds if rs[s].get("_backbone_swap")]
    if sw:
        pair = boot_paired([[x for v in w["trained backbone"].values() for x in v["per_task"]] for w in sw],
                           [[x for v in w["pretrained backbone"].values() for x in v["per_task"]] for w in sw])
        out["backbone_swap"] = {"seeds": len(sw), "trained": float(np.mean([w["mean"]["trained backbone"] for w in sw])),
                                "pretrained": float(np.mean([w["mean"]["pretrained backbone"] for w in sw])), "paired": pair,
                                "by_phase": {n: [float(np.mean([w["trained backbone"][n]["success"] for w in sw])),
                                                 float(np.mean([w["pretrained backbone"][n]["success"] for w in sw]))] for n in names}}
    return out


def _get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def final_tasks(r: dict, which: str = "seen") -> list:
    """Per-task outcomes of the last row, all lists of a kind concatenated in a fixed order."""
    out = []
    for n, v in sorted(r["rows"][-1]["lists"].items()):
        if (which == "seen") != n.endswith("@unseen"):
            out += v["per_task"]
    return out


def life_summary(root: str) -> dict:
    runs = load_life(root)
    out: dict = {"conditions": {}, "comparisons": {}, "labels": LABEL, "order": [c for c in ORDER if c in runs]}
    for c in out["order"]:
        out["conditions"][c] = cond_summary(runs[c])
    for a, b in PAIRS:
        if a in runs and b in runs:
            seeds = sorted(set(runs[a]) & set(runs[b]))
            if seeds:
                out["comparisons"][f"{a} - {b}"] = {
                    "final, all six lists": boot_paired([final_tasks(runs[a][s]) for s in seeds], [final_tasks(runs[b][s]) for s in seeds]),
                    "final, unseen worlds": boot_paired([final_tasks(runs[a][s], "unseen") for s in seeds], [final_tasks(runs[b][s], "unseen") for s in seeds])}
                vs = [s for s in seeds if runs[a][s].get("_verify") and runs[b][s].get("_verify")]
                if vs:
                    flat = lambda r: [x for n in sorted(r["_verify"]["lists"]) for x in r["_verify"]["lists"][n]["per_task"]]   # noqa: E731
                    out["comparisons"][f"{a} - {b}"]["untouched tasks, fresh process"] = boot_paired([flat(runs[a][s]) for s in vs], [flat(runs[b][s]) for s in vs])
    return out


def main(a) -> int:
    from ..cli import RUNS
    os.makedirs(os.path.join(OUT, "life"), exist_ok=True)
    life_root = a.life or os.path.join(RUNS, "life")
    summ = life_summary(life_root)
    json.dump(summ, open(os.path.join(OUT, "continual_lifetime.json"), "w"), indent=1, default=str)
    n = 0
    for p in sorted(glob.glob(os.path.join(life_root, "*", "results.json"))):
        r = json.load(open(p))
        if r.get("finished"):
            json.dump(r, open(os.path.join(OUT, "life", os.path.basename(os.path.dirname(p)) + ".json"), "w"), default=str)
            n += 1
    print(f"lifetimes: {n} finished runs, conditions {summ['order']}")
    for c in summ["order"]:
        s = summ["conditions"][c]
        print(f"  {LABEL[c]:48s} seeds {s['seeds']}  final mean {s['final_mean']['mean']:.3f}  forgetting {s['forgetting']['mean']:.3f}  "
              f"acquisition {s['acquisition']['mean']:.3f}")
    meta = a.meta or os.path.join(RUNS, "meta", "continual1")
    if os.path.exists(os.path.join(meta, "ledger.jsonl")):
        dst = os.path.join(OUT, "meta_continual")
        os.makedirs(os.path.join(dst, "diffs"), exist_ok=True)
        for f in ("ledger.jsonl", "config.json", "meta.json", "report.json", "selection_report.json"):
            if os.path.exists(os.path.join(meta, f)):
                shutil.copy(os.path.join(meta, f), os.path.join(dst, f))
        for d in sorted(glob.glob(os.path.join(meta, "candidates", "*", "proposal.diff"))):
            shutil.copy(d, os.path.join(dst, "diffs", os.path.basename(os.path.dirname(d)) + ".diff"))
        print(f"meta-search: copied ledger and {len(os.listdir(os.path.join(dst, 'diffs')))} candidate diffs")
    return 0
