"""Table of the debug ablation (ps_eval.py results): per lifetime, verified progress and prospective NLL by outcome
kind (count / binary / goal-relevant), for every run in DIR; plus paired on-off differences and retention.

    python ps_abl_table.py DIR
"""
import glob
import json
import os
import sys

d = sys.argv[1]
runs = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*_s*.json")))}
K = ("count", "binary", "goal_relevant")
print("| run | lifetime | family | progress | NLL count (n) | NLL binary (n) | NLL goal-relevant (n) | causal TV | rel. change of b | rho |")
print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
for name, r in runs.items():
    for lt in r["lifetimes"]:
        n = lt.get("nll_final") or lt.get("nll") or {}
        cells = []
        for k in K:
            v = n.get(k) or {}
            cells.append(f"{v['mean']:.3f} ({v['n']})" if v.get("mean") is not None else "-")
        c = lt.get("causal") or {}
        ps = lt.get("persistent_state") or {}
        print(f"| {name} | {lt['index']} | {lt['family']} | {lt['progress']:.3f} | " + " | ".join(cells) +
              f" | {c['mean_tv']:.4f} | {c['mean_rel_db']:.3f} | {ps['rho_mean']:.4f} |" if c.get("mean_tv") is not None
              else f"| {name} | {lt['index']} | {lt['family']} | {lt['progress']:.3f} | " + " | ".join(cells) + " | - | - | - |")
print()
print("| run | world-model NLL of the first phase's records after phase 0 / 1 / 2 / 3 | change 0 -> 3 |")
print("|---|---|---:|")
for name, r in runs.items():
    ret = r.get("retention_first_phase_records") or {}
    vals = [ret.get(f"after_phase_{i}", {}).get("mean_nll") for i in range(4)]
    if all(v is not None for v in vals):
        print(f"| {name} | " + " / ".join(f"{v:.4f}" for v in vals) + f" | {vals[3] - vals[0]:+.4f} ({100 * (vals[3] / vals[0] - 1):+.1f} %) |")
# paired on - off
print()
for seed in sorted({k.split("_s")[-1] for k in runs}):
    on, off = runs.get(f"on_s{seed}"), runs.get(f"off_s{seed}")
    if not on or not off:
        continue
    for label, fams in (("LabWorld (F1/T1)", ("F1", "T1")), ("held-out T2", ("T2",))):
        dp, dn, dc = [], [], []
        for a, b in zip(on["lifetimes"], off["lifetimes"]):
            if a["family"] not in fams:
                continue
            dp.append(a["progress"] - b["progress"])
            for key, acc in (("goal_relevant", dn), ("count", dc)):
                na = (a.get("nll_final") or {}).get(key) or {}
                nb = (b.get("nll_final") or {}).get(key) or {}
                if na.get("mean") is not None and nb.get("mean") is not None:
                    acc.append(na["mean"] - nb["mean"])
        m = lambda x: f"{sum(x) / len(x):+.3f} ({len(x)})" if x else "-"
        print(f"seed {seed}, {label}: progress on - off {m(dp)}; goal-relevant NLL on - off {m(dn)}; "
              f"count NLL on - off {m(dc)}")
