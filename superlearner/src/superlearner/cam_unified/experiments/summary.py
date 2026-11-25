"""Compact summaries of the experiment reports (markdown tables + JSON) for the committed results documents.

    python -m superlearner.cam_unified.experiments.summary exp1 --set dev [--reference cam]
    python -m superlearner.cam_unified.experiments.summary exp2 --set dev
    python -m superlearner.cam_unified.experiments.summary exp3 --set dev
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import stack as S

KS = (0, 1, 2, 4, 8, 16, 32, 64)


def ci(x, nd: int = 2) -> str:
    if not x or x.get("mean") is None:
        return "-"
    return f"{x['mean']:.{nd}f} [{x['lo']:.{nd}f}, {x['hi']:.{nd}f}]"


def exp1_tables(rep: dict, reference: str) -> tuple:
    md, js = [], {}
    for metric in ("logscore", "regret"):
        blocks = rep.get(f"predict/{metric}", {})
        for key, blk in sorted(blocks.items()):
            md.append(f"\n**Prediction `{metric}`, condition {key}** (mean over tasks and seeds, 95% two-level CI on "
                      "the area)\n")
            md.append("| learner | " + " | ".join(f"k={k}" for k in KS) + " | area |")
            md.append("|---" * (len(KS) + 2) + "|")
            for tag, b in sorted(blk.items()):
                if "-minus-" in tag:
                    continue
                cur = b["curve"]
                md.append(f"| {tag} | " + " | ".join(f"{cur[str(k)]['mean']:.2f}" if cur[str(k)]["mean"] is not None
                                                     else "-" for k in KS) + f" | {ci(b['area'])} |")
                js[f"predict/{metric}/{key}/{tag}/area"] = b["area"]
            for tag, b in sorted(blk.items()):
                if "-minus-" in tag:
                    md.append(f"| {tag} (paired) | " + " | ".join(ci(b["k"][str(k)]) for k in KS)
                              + f" | {ci(b['area'])} |")
                    js[f"predict/{metric}/{key}/{tag}/area"] = b["area"]
    eff = rep.get("condition_effects/logscore", {})
    if eff:
        md.append("\n**Condition effects within a learner (paired, log score; C - A = predecessor notes, D - A = delay "
                  "+ hidden gauges)**\n")
        md.append("| learner/split/contrast | k=0 | k=1 | k=2 | k=4 | area |")
        md.append("|---|---|---|---|---|---|")
        for key, b in sorted(eff.items()):
            md.append(f"| {key} | " + " | ".join(ci(b[f"k{k}"]) for k in (0, 1, 2, 4)) + f" | {ci(b['area'])} |")
            js[f"condition_effect/{key}"] = b
    dec = rep.get("decide", {})
    for key, blk in sorted(dec.items()):
        md.append(f"\n**Decisions (agent-selected), condition {key}**: window score after k observations "
                  "(ticks [k, 2k)), episode score, success\n")
        md.append("| learner | n | " + " | ".join(f"w{k}" for k in (0, 1, 2, 4, 8, 16, 32, 64)) +
                  " | episode score | success |")
        md.append("|---" * 12 + "|")
        for tag, b in sorted(blk.items()):
            if "-minus-" in tag:
                md.append(f"| {tag} (paired score) | | " + " | " * 8 + f"{ci(b['score'])} | |")
                js[f"decide/{key}/{tag}/score"] = b["score"]
                continue
            w = b["windows"]
            md.append(f"| {tag} | {b['n_episodes']} | " + " | ".join(
                f"{w[str(k)]['mean']:.2f}" if str(k) in w and w[str(k)]["mean"] is not None else "-"
                for k in (0, 1, 2, 4, 8, 16, 32, 64)) + f" | {ci(b['score'])} | {ci(b['success'])} |")
            js[f"decide/{key}/{tag}/score"] = b["score"]
            js[f"decide/{key}/{tag}/success"] = b["success"]
    return "\n".join(md), js


def exp2_tables(rep: dict) -> tuple:
    md = ["| learner | episode score | success | cum. regret (64 ticks) | obs. needed (reached) | effect log score "
          "area | effect regret area | CPU s / decision |", "|---|---|---|---|---|---|---|---|"]
    js = {}
    for tag, b in sorted(rep["learners"].items()):
        md.append(f"| {tag} | {ci(b['score'])} | {ci(b['succ'])} | {ci(b['cumregret64'], 1)} | "
                  f"{ci(b['needed'], 1)} ({ci(b['reached'])}) | {ci(b['effect_logscore_area'])} | "
                  f"{ci(b['effect_regret_area'])} | {b['cpu_per_decision']['mean']:.3f} |")
        js[tag] = {k: b[k] for k in ("score", "succ", "cumregret64", "cumregret", "needed", "reached",
                                     "effect_logscore_area", "effect_regret_area", "cpu_per_decision",
                                     "wall_per_decision")}
    for key in [k for k in rep if k.endswith("-minus")]:
        md.append(f"\n**Paired differences ({key[:-6]} - other)**\n")
        md.append("| other | cum. regret 64 | episode score | effect log score area | effect regret area |")
        md.append("|---|---|---|---|---|")
        for tag, b in sorted(rep[key].items()):
            md.append(f"| {tag} | {ci(b['cumregret64'], 1)} | {ci(b['score'])} | {ci(b['effect_logscore_area'])} | "
                      f"{ci(b['effect_regret_area'])} |")
            js[f"{key}/{tag}"] = b
    return "\n".join(md), js


def exp3_tables(rep: dict) -> tuple:
    md, js = [], {}
    cats = sorted({k.split("/")[1] for k in rep["cells"]})
    tags = sorted({k.split("/")[0] for k in rep["cells"]})
    md.append("**Log-score area (k = 0..64) per checkpoint and category set**\n")
    md.append("| checkpoint | " + " | ".join(cats) + " |")
    md.append("|---" * (len(cats) + 1) + "|")
    for t in tags:
        md.append(f"| {t} | " + " | ".join(ci(rep["cells"].get(f"{t}/{c}", {}).get("logscore", {}).get("area"))
                                           for c in cats) + " |")
        for c in cats:
            js[f"cells/{t}/{c}/logscore_area"] = rep["cells"].get(f"{t}/{c}", {}).get("logscore", {}).get("area")
    md.append("\n**Contrasts (paired, log-score area; regret area in parentheses)**\n")
    for c, blk in rep["contrasts"].items():
        for name, v in blk.items():
            md.append(f"- {c} {name}: {ci(v.get('logscore'))}" + (f" ({ci(v.get('regret'))})" if v.get("regret")
                                                                    else ""))
            js[f"contrast/{c}/{name}"] = v
    if rep.get("plasticity"):
        md.append("\n**Plasticity (E's instruments)**\n")
        md.append("| checkpoint | dormant fraction | activation eff. rank | Z eff. rank | Z srank | weight change |")
        md.append("|---|---|---|---|---|---|")
        for p in rep["plasticity"]:
            md.append(f"| {p['tag']} s{p['seed']} | {p['dormant_frac']:.3f} | {p.get('activation_rank_mean') or 0:.1f} "
                      f"| {p['z_effective_rank']:.1f} | {p['z_srank']} | {p.get('weight_change_rel', float('nan')):.3f} |")
        js["plasticity"] = rep["plasticity"]
    return "\n".join(md), js


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["exp1", "exp2", "exp3"])
    ap.add_argument("--set", default="dev")
    ap.add_argument("--reference", default="cam")
    ap.add_argument("--out", default=None, help="write the JSON summary here")
    a = ap.parse_args(argv)
    d = os.path.join(S.RUNS_ROOT, a.exp, a.set)
    if a.exp == "exp1":
        md, js = exp1_tables(json.load(open(os.path.join(d, f"report_{a.reference}.json"))), a.reference)
    elif a.exp == "exp2":
        md, js = exp2_tables(json.load(open(os.path.join(d, f"report_{a.reference}.json"))))
    else:
        md, js = exp3_tables(json.load(open(os.path.join(d, "report.json"))))
    print(md)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(js, fh, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
