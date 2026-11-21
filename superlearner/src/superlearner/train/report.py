"""`superlearner report`: what a run measured. Learning curves, area under the learning curve, costs, and the
evidence that the network's weights changed."""
from __future__ import annotations

import json
import os

import numpy as np


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def aulc(xs: list[float], ys: list[float], x_max: float | None = None) -> float:
    """Normalised area under a learning curve y(x) (trapezoid rule, divided by the x-range): the mean height of
    the curve over the experience budget. Higher = capability was acquired earlier."""
    if len(xs) < 2:
        return float(ys[0]) if ys else 0.0
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if x_max is not None and x_max > xs[-1]:
        xs, ys = np.append(xs, x_max), np.append(ys, ys[-1])
    span = xs[-1] - xs[0]
    area = float(np.sum((ys[1:] + ys[:-1]) * 0.5 * np.diff(xs)))
    return area / span if span > 0 else float(ys.mean())


def report(run_dir: str) -> dict:
    cfg = json.load(open(os.path.join(run_dir, "config.json"))) if os.path.exists(os.path.join(run_dir, "config.json")) else {}
    meta = json.load(open(os.path.join(run_dir, "meta.json"))) if os.path.exists(os.path.join(run_dir, "meta.json")) else {}
    metrics = read_jsonl(os.path.join(run_dir, "metrics.jsonl"))
    evals = [e for e in read_jsonl(os.path.join(run_dir, "eval.jsonl")) if e.get("tag") == "eval"]
    rl = [m for m in metrics if m.get("phase") == "rl"]
    bc = [m for m in metrics if m.get("phase") == "bc"]
    demos = [m for m in metrics if m.get("phase") == "demos"]
    curve = [(e["env_decisions"], e["success"], e.get("success_ci"), e.get("valid_rate")) for e in evals]
    rep = {"run_dir": run_dir, "name": cfg.get("name"), "algo": (cfg.get("algo") or {}).get("name"),
           "code": meta.get("code"), "git": meta.get("git"), "params": meta.get("params"),
           "demos": demos[-1] if demos else None, "bc_epochs": len(bc),
           "bc_nll": [round(bc[0]["bc_nll"], 3), round(bc[-1]["bc_nll"], 3)] if bc else None,
           "rl_iters": len(rl), "env_decisions": rl[-1]["env_decisions"] if rl else 0,
           "episodes": rl[-1]["episodes"] if rl else 0,
           "eval_curve": [{"env_decisions": x, "success": round(y, 4), "ci": ci, "valid_rate": v} for x, y, ci, v in curve],
           "aulc_eval_success": round(aulc([c[0] for c in curve], [c[1] for c in curve]), 4) if curve else None,
           "eval_first": curve[0][1] if curve else None, "eval_last": curve[-1][1] if curve else None,
           "eval_by_arm_first": evals[0].get("by_arm") if evals else None,
           "eval_by_arm_last": evals[-1].get("by_arm") if evals else None,
           "train_success": [round(m.get("train_success", 0.0), 3) for m in rl],
           "weights": {"param_dist_from_init": rl[-1].get("param_dist_from_init") if rl else (bc[-1].get("param_dist_from_init") if bc else 0.0),
                       "mean_param_delta_per_update": float(np.mean([m.get("param_delta", 0.0) for m in rl])) if rl else 0.0,
                       "updates_that_changed_weights": int(sum(bool(m.get("weights_changed")) for m in rl)),
                       "mean_grad_norm": float(np.mean([m.get("grad_norm", 0.0) for m in rl])) if rl else 0.0},
           "cost": {"rollout_s": round(sum(m.get("rollout_s", 0) for m in rl), 1),
                    "update_s": round(sum(m.get("update_s", 0) for m in rl), 1),
                    "llm": rl[-1].get("llm") if rl else None},
           "curriculum": rl[-1].get("curriculum") if rl else None}
    lines = [f"run {rep['name']}  ({run_dir})", f"algorithm {rep['algo']}  learner code {(rep['code'] or {}).get('learner')}  "
             f"trainable parameters {(rep['params'] or {}).get('trainable')}"]
    if rep["demos"]:
        lines.append(f"demonstrations: {rep['demos'].get('verified_successful')} verified of {rep['demos'].get('ran')} run; "
                     f"BC epochs {rep['bc_epochs']} nll {rep['bc_nll']}")
    lines.append(f"RL: {rep['rl_iters']} iterations, {rep['episodes']} episodes, {rep['env_decisions']} decisions; "
                 f"rollout {rep['cost']['rollout_s']}s, updates {rep['cost']['update_s']}s")
    lines.append("held-out evaluation (greedy) success by experience:")
    for c in rep["eval_curve"]:
        lines.append(f"  {c['env_decisions']:>9} decisions  success {c['success']:.3f}  95% CI {c['ci']}  valid actions {c['valid_rate']:.2f}")
    lines.append(f"AULC (mean eval success over the experience budget): {rep['aulc_eval_success']}")
    lines.append(f"weights: distance from initialisation {rep['weights']['param_dist_from_init']}, "
                 f"{rep['weights']['updates_that_changed_weights']} updates changed the weights, "
                 f"mean gradient norm {rep['weights']['mean_grad_norm']:.3f}")
    if rep["eval_by_arm_last"]:
        lines.append("eval by arm (first -> last): " + ", ".join(
            f"{a} {(rep['eval_by_arm_first'] or {}).get(a, 0):.2f}->{v:.2f}" for a, v in rep["eval_by_arm_last"].items()))
    rep["text"] = "\n".join(lines)
    return rep
