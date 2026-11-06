"""Plots of learning curves / losses and text rendering of trajectories with
world-model predictions next to the real outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from agentrs.monitoring.tracker import read_metrics


def plot_run(run_dir: str | Path, out_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ms = read_metrics(run_dir)
    written = []
    panels = [("bc/loss", "behaviour cloning loss"), ("wm/loss", "world-model loss"),
              ("wm/diff_nll_per_token", "world-model diff NLL / token"), ("rl/collect_success", "RL collection success"),
              ("rl/rl_entropy", "policy entropy"), ("critic/loss", "critic loss")]
    present = [(k, t) for k, t in panels if any(k in m for m in ms)]
    if present:
        fig, axes = plt.subplots(1, len(present), figsize=(4 * len(present), 3))
        axes = axes if len(present) > 1 else [axes]
        for ax, (k, t) in zip(axes, present):
            pts = [(m["step"], m[k]) for m in ms if k in m]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1)
            ax.set_title(t, fontsize=9)
            ax.set_xlabel("step (env steps for rl/*)", fontsize=8)
        fig.tight_layout()
        p = out_dir / "training_curves.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)
    res = run_dir / "results.json"
    if res.exists():
        r = json.loads(res.read_text())
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for mode in ("policy", "planner"):
            pts = [(e["real_env_steps"], e["results"][f"{mode}/val"]["success_rate"]) for e in r.get("evals", [])
                   if f"{mode}/val" in e["results"]]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=mode)
        ax.set_xlabel("real environment steps used for training")
        ax.set_ylabel("verified success (val)")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        p = out_dir / "learning_curve.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        written.append(p)
    return written


def render_trajectory(store_dir: str, episode_id: Optional[str] = None, wm_path: Optional[str] = None) -> str:
    from agentrs.data.storage import ExperienceStore

    eps = ExperienceStore(store_dir).load_episodes()
    if not eps:
        return "no episodes"
    ep = next((e for e in eps if e.episode_id == episode_id), eps[0])
    wm = None
    if wm_path:
        from agentrs.training.common import load_model

        wm = load_model(wm_path)
    from agentrs.training.world_model import outcome_text

    lines = [f"episode {ep.episode_id} task={ep.task_id} success={ep.success} ({ep.termination_reason})",
             f"GOAL: {ep.goal}"]
    hist: list[tuple[str, str]] = []
    for t in ep.transitions:
        lines.append(f"\n--- step {t.step}: {t.action}  valid={t.action_valid}  reward={t.reward}")
        lines.append("observation:\n  " + t.observation.to_text(20).replace("\n", "\n  "))
        lines.append("actual change:\n  " + t.diff_text.replace("\n", "\n  "))
        if t.wm_prediction is not None and t.wm_prediction.diff_text:
            lines.append("predicted at decision time:\n  " + t.wm_prediction.diff_text.replace("\n", "\n  "))
        if wm is not None:
            pred = wm.predict(t.goal, t.observation, t.action, hist)
            lines.append("world model now predicts:\n  " + pred.diffs[0][0].replace("\n", "\n  ")
                         + f"\n  P(reward)={pred.reward_prob:.2f} P(terminal)={pred.terminal_prob:.2f} "
                           f"epistemic={pred.epistemic:.3f}")
        hist.append((t.action, outcome_text(t)))
    return "\n".join(lines)
