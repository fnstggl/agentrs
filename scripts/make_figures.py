#!/usr/bin/env python
"""Learning curves (verified val success vs real environment steps) for several runs.

    python scripts/make_figures.py --runs runs/v2/v2_D_s0 runs/v2/v2_D_s1 --out docs/figures/learning_curves.png
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)


def label(cfg: dict, mode: str) -> str:
    v = cfg["variant"]
    name = {"A": "A policy-only", "B": "B imitation", "C": "C model-free RL", "D": "D"}[v]
    if v == "D":
        name = "D+imagination" if cfg.get("use_imagination") else "D (= C weights)"
        if not cfg.get("use_demos", True):
            name += " no-demos"
        if not cfg.get("historical_replay", True):
            name += " no-replay"
    return f"{name} [{mode}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    curves = defaultdict(list)
    for r in a.runs:
        p = Path(r) / "results.json"
        if not p.exists():
            continue
        res = json.loads(p.read_text())
        for mode in ("policy", "planner"):
            pts = [(e["real_env_steps"], e["results"][f"{mode}/{a.split}"]["success_rate"]) for e in res["evals"]
                   if f"{mode}/{a.split}" in e["results"]]
            if len(pts) > 1:
                curves[label(res["config"], mode)].append(pts)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for lab, runs in sorted(curves.items()):
        ln = None
        for pts in runs:
            ln, = ax.plot([x for x, _ in pts], [y for _, y in pts], marker="o", ms=3, lw=1.2, alpha=0.8,
                          color=ln.get_color() if ln is not None else None, label=lab if ln is None else None)
    ax.set_xlabel("real environment steps used for RL")
    ax.set_ylabel(f"verified success ({a.split})")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out, {k: len(v) for k, v in curves.items()})


if __name__ == "__main__":
    main()
