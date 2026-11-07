"""Paired comparison of Experiment F arms over all seeds (same held-out tasks in every arm)."""

import glob
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2] / "results" / "temporal"
rs = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(R / "expF_s[0-9].json")))]
arms = list(rs[0]["eval"])
Y = {a: np.array([float(row["success"]) for r in rs for row in r["eval"][a]["rows"]]) for a in arms}
rng = np.random.default_rng(0)
out = {"n_tasks": int(len(Y[arms[0]])), "seeds": [r["seed"] for r in rs], "success": {}, "paired_vs_rl_policy": {}}
for a in arms:
    y = Y[a]
    bs = [y[rng.integers(0, len(y), len(y))].mean() for _ in range(4000)]
    out["success"][a] = {"mean": float(y.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
for a in arms:
    d = Y[a] - Y["rl_policy"]
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    out["paired_vs_rl_policy"][a] = {"diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)),
                                                                        float(np.percentile(bs, 97.5))]}
d = Y["planner_temporal_all"] - Y["planner_temporal_self"]
bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
out["temporal_all_minus_temporal_self"] = {"diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)),
                                                                              float(np.percentile(bs, 97.5))]}
out["compute"] = [{"seed": r["seed"], "wall_s": r.get("wall_s"), "self_posts": r["phases"]["rl"]["self_posts"],
                   "bc_wall_s": r["phases"]["bc"]["wall_s"], "rl_wall_s": r["phases"]["rl"]["wall_s"],
                   "wm_wall_s": r["phases"]["wm"]["wall_s"], "eval_wall_s": r["phases"].get("eval_wall_s"),
                   "rl_param_change": r["phases"]["rl"]["param_change"], **r["data"]} for r in rs]
(R / "expF_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "compute"}, indent=1))
print(out["compute"])
