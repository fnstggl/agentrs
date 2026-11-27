"""Aggregate ``choice_diagnostic.py`` reports by arm: means over lifetimes with bootstrap 95 % intervals (resampling
lifetimes), the relation between how good the agent's own predictions were in a lifetime and how good its choices
were, and how both move over lifetimes.

    .venv/bin/python cam_live/docs/evidence/B/summarise_choice.py --out SUMMARY.json ARM=glob [ARM=glob ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import statistics
import sys


def _mean(v):
    v = [x for x in v if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return statistics.fmean(v) if v else None


def _boot(v, n=2000, seed=0):
    v = [x for x in v if x is not None and math.isfinite(x)]
    if len(v) < 2:
        return None
    rng = random.Random(seed)
    ms = sorted(statistics.fmean(rng.choices(v, k=len(v))) for _ in range(n))
    return [ms[int(0.025 * n)], ms[int(0.975 * n) - 1]]


def _pearson(x, y):
    pts = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pts) < 3:
        return None
    mx, my = statistics.fmean(p[0] for p in pts), statistics.fmean(p[1] for p in pts)
    sxy = sum((a - mx) * (b - my) for a, b in pts)
    sx = math.sqrt(sum((a - mx) ** 2 for a, _ in pts))
    sy = math.sqrt(sum((b - my) ** 2 for _, b in pts))
    return sxy / (sx * sy) if sx > 0 and sy > 0 else None


def _slope(x, y):
    pts = [(a, b) for a, b in zip(x, y) if b is not None]
    if len(pts) < 3:
        return None
    mx, my = statistics.fmean(p[0] for p in pts), statistics.fmean(p[1] for p in pts)
    den = sum((a - mx) ** 2 for a, _ in pts)
    return sum((a - mx) * (b - my) for a, b in pts) / den if den > 0 else None


KEYS = ("expected_quality", "quality", "uniform_quality", "greedy_pred_quality", "policy_quality",
        "spearman_pred_truth", "spearman_policy_truth", "policy_share_tv")


def arm_summary(paths):
    lts, rows, reps = [], [], []
    for p in sorted(paths):
        with open(p) as f:
            r = json.load(f)
        reps.append({"file": p.split("/")[-1], "seed": r.get("seed"), "lifetimes": len(r.get("lifetimes", [])),
                     "selection_rule": r.get("selection_rule"), "summary": r.get("summary")})
        for lt in r.get("lifetimes", []):
            lts.append({**lt["mean"], "index": lt["index"], "seed": r.get("seed"), "progress": lt["progress"],
                        "zero_value_share": lt["zero_value_share"], "mix_end": lt.get("mix_end"),
                        "expected_adopters": lt.get("expected_adopters"), "decisions": lt.get("decisions")})
            rows += lt["decision_rows"]
    out = {"runs": reps, "n_lifetimes": len(lts), "n_decisions": len(rows)}
    for k in KEYS + ("progress", "zero_value_share"):
        v = [lt.get(k) for lt in lts]
        out[k] = {"mean": _mean(v), "ci95": _boot(v)}
    out["expected_adopters_per_decision"] = (sum(lt["expected_adopters"] or 0.0 for lt in lts)
                                             / max(1, sum(lt["decisions"] or 0 for lt in lts)))
    # does the choice track the agent's own predictions? (across lifetimes)
    out["corr_lifetime_greedy_pred_quality_vs_expected_quality"] = _pearson(
        [lt.get("greedy_pred_quality") for lt in lts], [lt.get("expected_quality") for lt in lts])
    out["corr_lifetime_spearman_pred_vs_expected_quality"] = _pearson(
        [lt.get("spearman_pred_truth") for lt in lts], [lt.get("expected_quality") for lt in lts])
    out["ratio_expected_quality_to_greedy_pred_quality"] = (
        (out["expected_quality"]["mean"] / out["greedy_pred_quality"]["mean"])
        if out["expected_quality"]["mean"] and out["greedy_pred_quality"]["mean"] else None)
    # per decision: choice quality against how well the predictions ranked that decision's candidates
    dq = [(d.get("spearman_pred_truth"), d.get("expected_quality")) for d in rows
          if d.get("spearman_pred_truth") is not None and d.get("expected_quality") is not None]
    out["corr_decision_spearman_pred_vs_expected_quality"] = _pearson([a for a, _ in dq], [b for _, b in dq])
    bins = {}
    for a, b in dq:
        key = "rho<0.5" if a < 0.5 else ("0.5<=rho<0.8" if a < 0.8 else "rho>=0.8")
        bins.setdefault(key, []).append(b)
    out["expected_quality_by_prediction_rank_quality"] = {k: {"mean": _mean(v), "n": len(v)} for k, v in bins.items()}
    # within a lifetime: early / middle / late decisions (the fast layer adapts the predictions to the world)
    by_pos = {}
    for r in [json.load(open(p_)) for p_ in sorted(paths)]:
        for lt in r.get("lifetimes", []):
            n = len(lt["decision_rows"])
            for j, d in enumerate(lt["decision_rows"]):
                key = "early" if j < n / 3 else ("middle" if j < 2 * n / 3 else "late")
                by_pos.setdefault(key, []).append(d)
    out["within_lifetime"] = {k: {"expected_quality": _mean([d.get("expected_quality") for d in v]),
                                  "greedy_pred_quality": _mean([d.get("greedy_pred_quality") for d in v]),
                                  "spearman_pred_truth": _mean([d.get("spearman_pred_truth") for d in v]),
                                  "n": len(v)} for k, v in by_pos.items()}
    # over lifetimes (learning)
    for k in ("expected_quality", "greedy_pred_quality", "policy_quality", "spearman_pred_truth",
              "spearman_policy_truth"):
        out[f"slope_over_lifetimes_{k}"] = _slope([lt["index"] for lt in lts], [lt.get(k) for lt in lts])
    first = [lt for lt in lts if lt["index"] < 4]
    second = [lt for lt in lts if lt["index"] >= 4]
    for k in ("expected_quality", "policy_quality", "greedy_pred_quality"):
        out[f"{k}_lifetimes_0_3"] = _mean([lt.get(k) for lt in first])
        out[f"{k}_lifetimes_4_7"] = _mean([lt.get(k) for lt in second])
    # reward
    def pos_share(key):
        pos = [(d[key], d["zero_value"]) for d in rows if d.get(key) is not None and d[key] > 0]
        tot = sum(v for v, _ in pos)
        return None if tot <= 0 else sum(v for v, z in pos if z) / tot
    out["share_positive_stored_reward_to_zero_value"] = pos_share("stored_intrinsic")
    out["share_positive_inner_reward_to_zero_value"] = pos_share("inner_total")
    out["mean_inner_reward_zero_value_vs_valued"] = {
        "zero_value": _mean([d.get("inner_total") for d in rows if d["zero_value"]]),
        "valued": _mean([d.get("inner_total") for d in rows if not d["zero_value"]])}
    out["mix_end_by_run"] = [r["summary"].get("mix_trajectory") if r.get("summary") else None for r in reps]
    out["mix_decisions"] = [x for r in reps if r.get("summary") for x in r["summary"].get("mix_decisions", [])]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("arms", nargs="+", help="NAME=glob")
    a = ap.parse_args(argv)
    res = {}
    for spec in a.arms:
        name, pat = spec.split("=", 1)
        paths = glob.glob(pat)
        if paths:
            res[name] = arm_summary(paths)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for name, s in res.items():
        print(f"{name}: lifetimes {s['n_lifetimes']} E[q] {s['expected_quality']['mean']:.3f} {s['expected_quality']['ci95']} "
              f"uniform {s['uniform_quality']['mean']:.3f} greedy-pred {s['greedy_pred_quality']['mean']:.3f} "
              f"policy {s['policy_quality']['mean']:.3f} zero-share {s['zero_value_share']['mean']:.3f} "
              f"adopters/dec {s['expected_adopters_per_decision']:.2f} "
              f"corr(gp,Eq) {s['corr_lifetime_greedy_pred_quality_vs_expected_quality']} "
              f"slope E[q] {s['slope_over_lifetimes_expected_quality']} slope pol {s['slope_over_lifetimes_policy_quality']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
