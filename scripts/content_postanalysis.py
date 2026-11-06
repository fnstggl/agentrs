#!/usr/bin/env python
"""Offline analysis of content runs from their stored experience and results:
reward-hacking monitor (windowed, significance-tested), per-iteration hype/engagement/success,
and a summary table row per run.

    python scripts/content_postanalysis.py runs/content/content_C_s0 runs/content/content_E_s0
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentrs.content.evaluation import RewardHackingMonitor  # noqa: E402
from agentrs.data.storage import ExperienceStore  # noqa: E402


def episodes_by_iteration(run: Path):
    st = ExperienceStore(run / "experience")
    out = defaultdict(dict)
    for p in st.shards("real"):
        m = re.search(r"content_rl(\d+)_", p.name)
        if not m:
            continue
        for t in st.iter_transitions(shards=[p]):
            out[int(m.group(1))].setdefault(t.episode_id, []).append(t)
    return out


class _Ep:
    def __init__(self, ts):
        self.transitions = sorted(ts, key=lambda t: t.step)
        self.success = self.transitions[-1].episode_success
        self.family = self.transitions[0].family


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    report = {}
    for r in a.runs:
        run = Path(r)
        res = json.loads((run / "results.json").read_text())
        mon = RewardHackingMonitor(window=10, min_n=40)
        for it, eps in sorted(episodes_by_iteration(run).items()):
            mon.record(it, [_Ep(ts) for ts in eps.values() if next(iter(ts)).family == "content_forum"])
        fin = res["final"]
        report[run.name] = {
            "variant": res["config"]["variant"], "seed": res["config"]["seed"],
            "final": {k: {m: v.get(m) for m in ("success_rate", "first_attempt_success", "quality_partial_mean",
                                                 "format_valid_rate", "factual_accuracy", "engagement_mean_NOT_REWARD",
                                                 "declined_rate", "by_family")}
                      for k, v in fin.items() if isinstance(v, dict) and "success_rate" in v},
            "lm_perplexity": fin.get("lm_perplexity"), "action_regression": fin.get("action_regression"),
            "start_action_regression": res.get("start", {}).get("action_regression"),
            "curve": [(e["tag"], e["real_episodes"], e["policy/val"]["success_rate"]) for e in res["evals"]
                      if "policy/val" in e],
            "hacking": mon.summary(), "regressions": res.get("regressions"),
            "wm_heldout": res.get("wm_heldout"), "wm_heldout_family": res.get("wm_heldout_family"),
            "real_episodes": res.get("real_episodes"), "flops": res.get("flops"),
            "wall_min": round(res.get("total_wall_s", 0) / 60, 1)}
        print(run.name, json.dumps({k: report[run.name][k] for k in ("hacking", "curve")}, default=str))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
