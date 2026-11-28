"""cam_live.eval.v3_report — prespecified summaries of the Protocol v3 substantive runs (docs/S/PROTOCOLS.md v3.4/3.5).

    CAM_LIVE_EVAL_HOME=... python -m cam_live.eval.v3_report --out-dir cam_live/docs/evidence/S/v3

Per experiment and condition: objectives achieved / attempted (never-reached and interrupted counted), decisions to
verified achievement (censored at the budget), verified value, executable fraction, backbone-written fraction of the
chosen actions, repeats of failed actions, distinct strategies, prediction NLL, meta-updates (applied / skipped with
reasons, gradient norms, reward-version changes), compute, wall time, configuration and snapshot per run; ranges
across seeds; S_q for the transfer pairs; paired seed differences with exact sign-flip p (Protocol v3.5: with 2-3
seeds the smallest attainable p is 0.25-0.5, so these are reported as descriptive only).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from .stats import diff_ci, mean_ci, speedup


def _load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def objective_rows(res: Dict[str, Any], budget: int) -> List[Dict[str, Any]]:
    out = []
    for o in res.get("objectives", []):
        n = o.get("decisions")
        ach = bool(o.get("achieved"))
        meta = o.get("meta_updates") or []
        out.append({
            "index": o.get("index"), "family": o.get("family"), "problem": o.get("problem"),
            "achieved": ach, "interrupted": bool(o.get("interrupted")), "verified_value": o.get("verified_value"),
            "decisions": n, "n_to_achieve": float(n) if ach and n is not None else float(budget),
            "censored": not ach, "executable_fraction": o.get("executable_fraction"),
            "backbone_chosen_fraction": o.get("backbone_chosen_fraction"),
            "repeats_of_failed_actions": o.get("repeats_of_failed_actions"),
            "distinct_strategies": o.get("distinct_strategies"), "mean_prediction_nll": o.get("mean_prediction_nll"),
            "primitive_ops_total": o.get("primitive_ops_total"), "skill_runs": o.get("skill_runs"),
            "wall_s": o.get("wall_s"),
            "inference_s": ((o.get("compute") or {}).get("seconds") or {}).get("inference"),
            "training_s": ((o.get("compute") or {}).get("seconds") or {}).get("training"),
            "meta_applied": sum(1 for m in meta if m.get("applied")),
            "meta_skipped": dict(Counter(str(m.get("skipped_reason", "")).split(":")[0] for m in meta
                                         if not m.get("applied"))),
            "meta_grad_norms": [m.get("meta_grad_norm") for m in meta if m.get("applied")],
            "reward_changed": bool((o.get("group_changed") or {}).get("reward")),
            "groups_changed": o.get("group_changed"), "mix": o.get("mix"),
            "unavailable_tools": [u.get("tool") for u in (o.get("unavailable_permitted_tools") or [])],
            "lock_hold_s": o.get("lock_hold_s"), "gpu_peak_gb": o.get("gpu_peak_gb"),
            "eval_code": o.get("eval_code")})
    return out


def summarise_condition(runs: Dict[int, Dict[str, Any]], budget: int) -> Dict[str, Any]:
    per_seed = {s: objective_rows(r, budget) for s, r in sorted(runs.items())}
    allrows = [x for rows in per_seed.values() for x in rows]
    att = [x for x in allrows if not x["interrupted"]]

    def seedwise(key: str, f=lambda rows, k: sum(r[k] for r in rows if r[k] is not None) /
                 max(1, sum(1 for r in rows if r[k] is not None))) -> Dict[str, Any]:
        vals = [f(rows, key) for rows in per_seed.values() if rows]
        return {"per_seed": vals, "mean": sum(vals) / len(vals) if vals else None,
                "min": min(vals) if vals else None, "max": max(vals) if vals else None}
    return {
        "seeds": sorted(runs), "objectives_attempted": len(att), "objectives_achieved": sum(x["achieved"] for x in att),
        "objectives_never_reached": sum(1 for x in att if not x["achieved"]),
        "objectives_interrupted": sum(1 for x in allrows if x["interrupted"]),
        "achieved_by_problem": {p: f"{sum(x['achieved'] for x in att if x['problem'] == p)}/"
                                   f"{sum(1 for x in att if x['problem'] == p)}"
                                for p in sorted({x["problem"] for x in att})},
        "decisions_to_achieve_censored": seedwise("n_to_achieve"),
        "verified_value": seedwise("verified_value"),
        "executable_fraction": seedwise("executable_fraction"),
        "backbone_chosen_fraction": seedwise("backbone_chosen_fraction"),
        "repeats_of_failed_actions": seedwise("repeats_of_failed_actions"),
        "distinct_strategies": seedwise("distinct_strategies"),
        "mean_prediction_nll": seedwise("mean_prediction_nll"),
        "meta_updates_applied_total": sum(x["meta_applied"] for x in allrows),
        "meta_skip_reasons": dict(sum((Counter(x["meta_skipped"]) for x in allrows), Counter())),
        "meta_grad_norms": [g for x in allrows for g in x["meta_grad_norms"]][:40],
        "objectives_with_reward_change": sum(x["reward_changed"] for x in allrows),
        "bootstrap_mix_values": sorted({x["mix"] for x in allrows if x["mix"] is not None}),
        "unavailable_tools_seen": sorted({t for x in allrows for t in x["unavailable_tools"]}),
        "max_lock_hold_s": max((x["lock_hold_s"] or 0.0 for x in allrows), default=None),
        "gpu_peak_gb_max": max((x["gpu_peak_gb"] or 0.0 for x in allrows), default=None),
        "wall_s_per_objective": seedwise("wall_s"),
        "learning_curve_by_index": {i: f"{sum(x['achieved'] for x in att if x['index'] == i)}/"
                                       f"{sum(1 for x in att if x['index'] == i)}"
                                    for i in sorted({x["index"] for x in att})},
        "per_seed": per_seed,
        "configuration": {s: r.get("configuration") for s, r in runs.items()},
        "code": {s: r.get("code") for s, r in runs.items()}}


def collect(home: str, exp: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    by: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(home, "results", "v3", exp, "*.json"))):
        r = _load(p)
        if not r:
            continue
        name = os.path.basename(p)[:-5]
        cond, _, s = name.rpartition("_s")
        by[cond][int(s)] = r
    return dict(by)


def transfer(home: str, budget: int) -> Dict[str, Any]:
    by = collect(home, "x2")
    ini, exp_ = by.get("initial", {}), by.get("experienced", {})
    seeds = sorted(set(ini) & set(exp_))
    pairs_i, pairs_e, per = [], [], []
    for s in seeds:
        ri, re_ = objective_rows(ini[s], budget), objective_rows(exp_[s], budget)
        for a, b in zip(ri, re_):
            if a["interrupted"] or b["interrupted"]:
                continue
            pairs_i.append({"reached": a["achieved"], "n": a["n_to_achieve"]})
            pairs_e.append({"reached": b["achieved"], "n": b["n_to_achieve"]})
            per.append({"seed": s, "problem": a["problem"], "initial": a["n_to_achieve"], "initial_achieved":
                        a["achieved"], "experienced": b["n_to_achieve"], "experienced_achieved": b["achieved"],
                        "initial_value": a["verified_value"], "experienced_value": b["verified_value"]})
    out = {"seeds": seeds, "pairs": per,
           "S_q": speedup(pairs_i, pairs_e) if pairs_i else None,
           "value_difference_experienced_minus_initial": diff_ci([p["experienced_value"] or 0.0 for p in per],
                                                                 [p["initial_value"] or 0.0 for p in per],
                                                                 paired=True) if per else None,
           "fresh_process_checks": {f"{v}_s{s}": (r.get("fresh_process") or {}) for v, d in
                                    (("initial", ini), ("experienced", exp_)) for s, r in d.items()},
           "initial": summarise_condition(ini, budget) if ini else None,
           "experienced": summarise_condition(exp_, budget) if exp_ else None,
           "note": "pairs are (seed, held-out objective); with this many pairs no test can reach p < 0.05 "
                   "(Protocol v3.5); S_q is reported with its bootstrap range only"}
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budget", type=int, default=20)
    a = ap.parse_args(argv)
    from .experiments import EVAL_HOME
    os.makedirs(a.out_dir, exist_ok=True)
    report: Dict[str, Any] = {}
    for exp in ("x0", "x1", "x3", "x4"):
        by = collect(EVAL_HOME, exp)
        if by:
            report[exp] = {c: summarise_condition(r, a.budget) for c, r in by.items()}
    if "x1" in report and "x3" in report:
        e = collect(EVAL_HOME, "x1").get("E", {})
        comp = {}
        for c, runs in collect(EVAL_HOME, "x3").items():
            seeds = sorted(set(e) & set(runs))
            fe = [sum(x["n_to_achieve"] for x in objective_rows(e[s], a.budget)) / max(1, len(e[s]["objectives"]))
                  for s in seeds]
            fc = [sum(x["n_to_achieve"] for x in objective_rows(runs[s], a.budget)) /
                  max(1, len(runs[s]["objectives"])) for s in seeds]
            comp[c] = {"paired_seeds": seeds,
                       "mean_decisions_to_achieve_E_minus_c": diff_ci(fe, fc, paired=True) if seeds else None}
        report["x3_vs_E"] = comp
    if collect(EVAL_HOME, "x2"):
        report["x2"] = transfer(EVAL_HOME, a.budget)
    for k, v in report.items():
        with open(os.path.join(a.out_dir, f"summary_{k}.json"), "w") as f:
            json.dump({"executed": True, "experiment": k, **({"conditions": v} if k in ("x0", "x1", "x3", "x4")
                                                             else v)}, f, indent=1, default=str)
    for k, v in report.items():
        if k in ("x0", "x1", "x3", "x4"):
            for c, s in v.items():
                print(f"{k} {c}: achieved {s['objectives_achieved']}/{s['objectives_attempted']} (interrupted "
                      f"{s['objectives_interrupted']}) by problem {s['achieved_by_problem']} decisions-to-achieve "
                      f"{s['decisions_to_achieve_censored']['per_seed']} exec {s['executable_fraction']['mean']} "
                      f"backbone {s['backbone_chosen_fraction']['mean']} meta applied {s['meta_updates_applied_total']}")
        elif k == "x2" and v.get("S_q"):
            print(f"x2: S_q {v['S_q'].get('S_q')} [{v['S_q'].get('lo')}, {v['S_q'].get('hi')}] pairs {len(v['pairs'])}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
