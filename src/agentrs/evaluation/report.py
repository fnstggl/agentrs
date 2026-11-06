"""Aggregate experiment ``results.json`` files into markdown tables (actual measurements only)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from agentrs.evaluation.metrics import bootstrap_ci


def _load(run: str | Path) -> Optional[dict]:
    p = Path(run) / "results.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    r["_dir"] = str(run)
    return r


def _label(r: dict) -> str:
    c = r["config"]
    lab = {"A": "A policy-only", "B": "B imitation", "C": "C model-free RL", "D": "D model-based"}[c["variant"]]
    extras = []
    if c["variant"] == "D":
        if c.get("use_imagination"):
            extras.append("+imagination")
        if not c.get("use_demos", True):
            extras.append("-demos")
        if not c.get("historical_replay", True):
            extras.append("-replay")
        if c.get("use_planning_collect"):
            extras.append("+plan-collect")
    return lab + (" " + " ".join(extras) if extras else "")


def build_report(runs: Sequence[str], out: Optional[str] = None) -> str:
    rs = [r for r in (_load(x) for x in runs) if r is not None and "final" in r]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rs:
        groups[_label(r)].append(r)
    sets = ["val", "test", "test_family"]
    lines = ["| configuration | mode | seeds | " + " | ".join(f"{s} success" for s in sets)
             + " | invalid-action rate (test) | real env steps | train PFLOP | inference PFLOP | wall (min) |",
             "|" + "---|" * (9 + len(sets) - 3)]
    for lab in sorted(groups):
        rr = groups[lab]
        modes = sorted({k.split("/")[0] for r in rr for k in r["final"]})
        for mode in modes:
            cells = []
            for s in sets:
                vals = [r["final"][f"{mode}/{s}"]["success_rate"] for r in rr if f"{mode}/{s}" in r["final"]]
                if not vals:
                    cells.append("-")
                    continue
                if len(vals) > 1:
                    lo, hi = bootstrap_ci(vals)
                    cells.append(f"{np.mean(vals):.3f} [{lo:.2f},{hi:.2f}]")
                else:
                    ci = rr[0]["final"][f"{mode}/{s}"].get("success_ci95", [None, None])
                    cells.append(f"{vals[0]:.3f}" + (f" [{ci[0]:.2f},{ci[1]:.2f}]" if ci and ci[0] is not None else ""))
            inv = [r["final"].get(f"{mode}/test", {}).get("invalid_action_rate") for r in rr]
            inv = [x for x in inv if x is not None]
            steps = np.mean([r.get("real_env_steps", 0) for r in rr])
            tf = np.mean([(r["flops"]["policy_train"] + r["flops"]["wm_train"] + r["flops"]["critic_train"]) / 1e15
                          for r in rr])
            inf = np.mean([r["flops"]["inference"] / 1e15 for r in rr])
            wall = np.mean([r.get("total_wall_s", 0) / 60 for r in rr])
            lines.append(f"| {lab} | {mode} | {len(rr)} | " + " | ".join(cells)
                         + f" | {np.mean(inv) if inv else float('nan'):.3f} | {steps:.0f} | {tf:.3f} | {inf:.3f} | {wall:.1f} |")
    # per-family breakdown on held-out families
    fam_lines = ["", "Held-out family success (test_family):", "",
                 "| configuration | mode | " + " | ".join(_heldout_names(rs)) + " |",
                 "|---|---|" + "---|" * len(_heldout_names(rs))]
    for lab in sorted(groups):
        rr = groups[lab]
        for mode in sorted({k.split("/")[0] for r in rr for k in r["final"]}):
            vals = []
            for f in _heldout_names(rs):
                v = [r["final"].get(f"{mode}/test_family", {}).get("by_family", {}).get(f) for r in rr]
                v = [x for x in v if x is not None]
                vals.append(f"{np.mean(v):.2f}" if v else "-")
            fam_lines.append(f"| {lab} | {mode} | " + " | ".join(vals) + " |")
    md = "\n".join(lines + fam_lines)
    if out:
        Path(out).write_text(md + "\n")
    return md


def _heldout_names(rs) -> list[str]:
    names = set()
    for r in rs:
        for k, v in r["final"].items():
            if k.endswith("/test_family"):
                names |= set(v.get("by_family", {}))
    return sorted(names)


def learning_curve(r: dict, mode: str = "policy", split: str = "val") -> list[tuple[int, float]]:
    pts = []
    for e in r.get("evals", []):
        k = f"{mode}/{split}"
        if k in e["results"]:
            pts.append((e["real_env_steps"], e["results"][k]["success_rate"]))
    return pts
