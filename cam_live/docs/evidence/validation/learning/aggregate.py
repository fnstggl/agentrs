"""Summarise the repeated-task runs of ``run_arm.py`` (one JSON per arm x seed) with confidence intervals.

    .venv/bin/python cam_live/docs/evidence/validation/learning/aggregate.py OUT.json RUN1.json RUN2.json ...

Per arm: success rate per episode index (Wilson 95 % interval over seeds), primitive operations and decisions per
objective (failures count with their full budget), and the change from the first half of the episodes to the
second half per seed (mean over seeds with a percentile bootstrap 95 % interval over seeds). Between arms: the
difference of second-half success rates and of second-half primitive operations (bootstrap over seeds).
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from typing import Any, Dict, List, Sequence


def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def boot_mean(xs: Sequence[float], n: int = 4000, seed: int = 0) -> Dict[str, float]:
    xs = [float(x) for x in xs if x == x]
    if not xs:
        return {"n": 0}
    r = random.Random(seed)
    means = sorted(statistics.fmean(r.choice(xs) for _ in xs) for _ in range(n))
    return {"n": len(xs), "mean": statistics.fmean(xs), "lo": means[int(0.025 * n)], "hi": means[int(0.975 * n) - 1]}


def summarise(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        by_arm.setdefault(r["arm"], []).append(r)
    out: Dict[str, Any] = {"arms": {}, "comparisons": {}}
    halves: Dict[str, Dict[str, List[float]]] = {}
    for arm, rs in sorted(by_arm.items()):
        n_ep = max(len(r["episodes"]) for r in rs)
        budget = rs[0]["max_decisions"]
        per_ep = []
        for i in range(n_ep):
            eps = [r["episodes"][i] for r in rs if len(r["episodes"]) > i]
            k = sum(1 for e in eps if e["achieved"])
            per_ep.append({"episode": i, "n_seeds": len(eps), "successes": k, "success_rate": k / len(eps),
                           "wilson95": wilson(k, len(eps)),
                           "mean_primitive_ops": statistics.fmean(e["primitive_ops"] for e in eps),
                           "mean_decisions": statistics.fmean(e["decisions"] for e in eps),
                           "skill_runs": sum(e.get("skill_runs", 0) for e in eps)})
        first_s, second_s, first_o, second_o, d_s, d_o = [], [], [], [], [], []
        for r in rs:
            eps = r["episodes"]
            h = len(eps) // 2
            if h == 0:
                continue
            a, b = eps[:h], eps[h:]
            sa, sb = statistics.fmean(e["achieved"] for e in a), statistics.fmean(e["achieved"] for e in b)
            oa, ob = statistics.fmean(e["primitive_ops"] for e in a), statistics.fmean(e["primitive_ops"] for e in b)
            first_s.append(sa), second_s.append(sb), first_o.append(oa), second_o.append(ob)
            d_s.append(sb - sa), d_o.append(ob - oa)
        k_all = sum(1 for r in rs for e in r["episodes"] if e["achieved"])
        n_all = sum(len(r["episodes"]) for r in rs)
        succ = [e for r in rs for e in r["episodes"] if e["achieved"]]
        out["arms"][arm] = {
            "seeds": sorted(r["seed"] for r in rs), "episodes_per_seed": n_ep, "decision_budget": budget,
            "overall_success": {"k": k_all, "n": n_all, "rate": k_all / max(1, n_all), "wilson95": wilson(k_all, n_all)},
            "per_episode": per_ep,
            "first_half_success": boot_mean(first_s), "second_half_success": boot_mean(second_s),
            "change_in_success_second_minus_first": boot_mean(d_s),
            "first_half_primitive_ops": boot_mean(first_o), "second_half_primitive_ops": boot_mean(second_o),
            "change_in_primitive_ops_second_minus_first": boot_mean(d_o),
            "primitive_ops_per_success": boot_mean([e["primitive_ops"] for e in succ]),
            "decisions_per_success": boot_mean([e["decisions"] for e in succ]),
            "skill_runs_total": sum(e.get("skill_runs", 0) for r in rs for e in r["episodes"]),
            "skill_runs_ok": sum(e.get("skill_runs_ok", 0) for r in rs for e in r["episodes"]),
            "induce_attempts": sum(e.get("induce_attempts", 0) for r in rs for e in r["episodes"]),
            "errors": sum(len(r.get("errors", [])) for r in rs),
        }
        halves[arm] = {"second_s": second_s, "second_o": second_o}

    def diff(a: str, b: str, key: str) -> Dict[str, float]:
        xa, xb = halves.get(a, {}).get(key, []), halves.get(b, {}).get(key, [])
        if not xa or not xb:
            return {"n": 0}
        r = random.Random(1)
        ds = sorted(statistics.fmean(r.choice(xa) for _ in xa) - statistics.fmean(r.choice(xb) for _ in xb)
                    for _ in range(4000))
        return {"mean": statistics.fmean(xa) - statistics.fmean(xb), "lo": ds[100], "hi": ds[3899],
                "n_a": len(xa), "n_b": len(xb)}
    for a, b in (("learn_skills", "frozen_skills"), ("learn_primitive", "frozen_primitive"),
                 ("learn_skills", "learn_primitive"), ("frozen_skills", "frozen_primitive")):
        out["comparisons"][f"{a} - {b}"] = {"second_half_success": diff(a, b, "second_s"),
                                            "second_half_primitive_ops": diff(a, b, "second_o")}
    return out


def main(argv: Sequence[str]) -> int:
    out, paths = argv[0], argv[1:]
    runs = []
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        r["file"] = p
        runs.append(r)
    res = summarise(runs)
    res["files"] = [r["file"] for r in runs]
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    for arm, a in res["arms"].items():
        print(f"{arm:18s} success {a['overall_success']['k']}/{a['overall_success']['n']} "
              f"first {a['first_half_success'].get('mean', float('nan')):.2f} -> second "
              f"{a['second_half_success'].get('mean', float('nan')):.2f}  "
              f"change {a['change_in_success_second_minus_first']}  ops change "
              f"{a['change_in_primitive_ops_second_minus_first']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
