"""Summarise diagnose_f1.py runs: one row per run (arm), means over lifetimes / decisions, bootstrap CIs.

    .venv/bin/python cam_live/docs/evidence/validation/diagnosis/summarise.py OUT.json RUN1.json RUN2.json ...
"""
from __future__ import annotations

import json
import random
import statistics
import sys


def ci(xs, n=4000, seed=0):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    r = random.Random(seed)
    m = sorted(statistics.fmean(r.choice(xs) for _ in xs) for _ in range(n))
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 4), "lo": round(m[100], 4), "hi": round(m[3899], 4)}


def main(argv):
    out, paths = argv[0], argv[1:]
    rows = {}
    for p in paths:
        d = json.load(open(p))
        lts = d["lifetimes"]
        dec = [x for lt in lts for x in lt["decision_rows"]]
        rows[d["label"] or p] = {
            "lifetimes": len(lts), "decisions": len(dec), "optim": d.get("optim"),
            "agent": {k: d["agent_settings"].get(k) for k in ("selection_rule", "value_weight", "proposer_per_tool",
                                                             "use_backbone", "grounded_share", "backbone_share")},
            "verified_progress": ci([lt["progress"] for lt in lts]),
            "money_efficiency": ci([lt["quality"] for lt in lts]),
            "choice_quality": ci([x["quality"] for x in dec]),
            "uniform_quality_in_offer": ci([x["uniform_quality"] for x in dec]),
            "p_of_best": ci([x["p_of_best"] for x in dec]), "p_uniform": ci([x["p_uniform"] for x in dec]),
            "acts_offered": ci([x["n_act_offered"] for x in dec]),
            "wait_fraction": round(sum(1 for x in dec if x["op"] == "wait") / max(1, len(dec)), 3),
            "zero_value_fraction": round(sum(1 for x in dec if x["truth"] == 0) / max(1, len(dec)), 3),
            "reward_hash_changed_lifetimes": sum(1 for lt in lts if lt["reward_hash_changed"]),
            "meta_applied": sum(1 for lt in lts for m in lt["meta"] if m["applied"]),
            "policy_update_heads_delta": ci([(u.get("weight_delta") or {}).get("heads") for lt in lts
                                             for u in lt["policy_updates"]]),
            "reward_vs_truth": d.get("reward_vs_truth"),
            "per_lifetime_progress": [round(lt["progress"], 2) for lt in lts]}
    json.dump(rows, open(out, "w"), indent=1)
    for k, r in rows.items():
        print(f"{k:10s} progress {r['verified_progress'].get('mean')} [{r['verified_progress'].get('lo')}, "
              f"{r['verified_progress'].get('hi')}] choice {r['choice_quality'].get('mean')} vs uniform-in-offer "
              f"{r['uniform_quality_in_offer'].get('mean')} p_best {r['p_of_best'].get('mean')} (uniform "
              f"{r['p_uniform'].get('mean')}) acts {r['acts_offered'].get('mean')} waits {r['wait_fraction']} "
              f"zero-value {r['zero_value_fraction']} meta {r['meta_applied']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
