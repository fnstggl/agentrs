"""Compile CAM experiment results into a markdown report (``runs/cam/report.md``).

    PYTHONPATH=src python scripts/cam_report.py --runs /workspace/agentrs/runs

Matched comparison (full CAM vs no-world-model ablation): validation success per family vs the
number of real environment interactions; held-out world-model prediction vs baselines; sealed test
on held-out distributions and the never-trained family; planning-assisted vs direct acting; the
bizdev learning-speed transfer test (scratch vs pretrained). Only numbers present in the run
directories are reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(d: Path):
    try:
        return json.loads((d / "state.json").read_text())
    except (OSError, ValueError):
        return None


def curve(st, fam):
    return [(e["iter"], e["env_interactions_so_far"], e["by_family"][fam]["rate"], e["by_family"][fam]["final_progress"])
            for e in st["evals"] if e["split"] == "val" and fam in e["by_family"] and not e.get("planning")]


def fmt_curve(c):
    return " → ".join(f"it{i}: {r:.3f}" for i, _, r, _ in c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/workspace/agentrs/runs")
    a = ap.parse_args()
    R = Path(a.runs)
    full, nowm, pfix = load(R / "cam"), load(R / "cam_nowm"), load(R / "cam_pfix")
    md = ["# CAM experiment report", ""]
    if full:
        md += [f"Full CAM: phase **{full['phase']}**, iteration {full['iter']}, "
               f"{full['counters']['episodes']} episodes / {full['counters']['transitions']} real environment "
               f"actions / {full['counters']['successes']} verified successes / {full['counters']['updates']} gradient "
               f"updates of all {full['models']['params']['trainable']:,} CAM parameters.", ""]
        md += ["## Validation success vs. iterations (matched seeds and budgets)", "",
               "| family | full CAM | no world model (ablation C) | product fix (failure-horizon cost + training curriculum) |",
               "|---|---|---|---|"]
        for fam in full["config"]["train_families"]:
            md.append(f"| {fam} | {fmt_curve(curve(full, fam))} | {fmt_curve(curve(nowm, fam)) if nowm else 'n/a'} "
                      f"| {fmt_curve(curve(pfix, fam)) if pfix else 'not started'} |")
        md.append("")
        md += ["## Held-out world-model prediction (validation episodes, never trained on)", ""]
        for name, st in (("full", full), ("no-WM ablation", nowm), ("product fix", pfix)):
            if not st:
                continue
            ev = [e for e in st["evals"] if e.get("world_model")]
            if ev:
                f, l = ev[0]["world_model"], ev[-1]["world_model"]
                keys = ["event_brier", "event_brier_baserate", "feat_mse", "feat_mse_const", "success_brier",
                        "success_brier_baserate", "success_ece", "belief_std"]
                md.append(f"- {name} ({ev[0]['tag']} → {ev[-1]['tag']}): " + ", ".join(
                    f"{k} {f.get(k, float('nan')):.4f}→{l.get(k, float('nan')):.4f}" for k in keys))
        md.append("")
        md += ["## Sealed tests (held-out counterparty policies / preferences / markets; bizdev never trained)", ""]
        for name, st in (("full", full), ("no-WM ablation", nowm), ("product fix", pfix)):
            for e in (st or {}).get("evals", []):
                if e["split"] == "test":
                    md.append(f"- {name} `{e['tag']}`{' +planning' if e.get('planning') else ''}: " + ", ".join(
                        f"{k} {v['success']}/{v['n']} = {v['rate']:.3f} (progress {v['final_progress']:.2f})"
                        for k, v in e["by_family"].items()))
        md.append("")
    if pfix:
        md += ["## Product fix: training-goal curriculum (validation/test always use the full target)", "",
               f"Run phase **{pfix['phase']}**, iteration {pfix['iter']}; current product training difficulty "
               f"{pfix.get('curriculum', {}).get('product', float('nan')):.3f} (fraction of the full target).", ""]
        try:
            rows = [json.loads(x) for x in (R / "cam_pfix" / "iters.jsonl").read_text().splitlines() if x.strip()]
        except OSError:
            rows = []
        pts = [r for r in rows if r["iter"] % 10 == 0 or r is rows[-1]]
        if pts:
            md.append("- product training success at the difficulty used: " + " → ".join(
                f"it{r['iter']}: {r['train'].get('product', {}).get('rate', 0):.2f}@{r.get('curriculum_used', {}).get('product', 1):.2f}"
                f" ({r['train'].get('product', {}).get('avg_steps', 0):.1f} steps)" for r in pts))
        md.append("")
    tr = {arm: load(R / "cam_transfer" / arm) for arm in ("scratch", "pretrained")}
    md += ["## Learning-speed transfer to the never-trained bizdev family", ""]
    if any(tr.values()):
        for arm, st in tr.items():
            if st:
                md.append(f"- {arm}: {fmt_curve(curve(st, 'bizdev'))}")
    else:
        md.append("- not started yet (begins when the main run reaches iteration 150)")
    out = R / "cam" / "report.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
