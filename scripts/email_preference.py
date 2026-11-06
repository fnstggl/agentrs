#!/usr/bin/env python
"""What does each email policy propose? Fraction of morning / afternoon / invalid proposals on
held-out tasks (greedy and sampled), from final content-policy checkpoints.

    python scripts/email_preference.py runs/content_email/email_B_s0 runs/content_email/email_C_s0
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agentrs.agent.runtime import run_episode  # noqa: E402
from agentrs.content.policy import ContentGeneratingPolicy, ContentPolicyActor  # noqa: E402
from agentrs.environments.content.env import ContentEnv, make_content_task  # noqa: E402
from agentrs.training.common import load_model  # noqa: E402


def classify(content: str, slots: list[str]) -> str:
    low = content.lower()
    hit = [s for s in slots if s.lower() in low]
    if len(hit) != 1:
        return "invalid"
    return "morning" if hit[0].endswith("am") else "afternoon"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out")
    a = ap.parse_args()
    env, out = ContentEnv(), {}
    for r in a.runs:
        cp = ContentGeneratingPolicy(load_model(os.path.join(r, "checkpoints", "content_policy_final")))
        cp.cfg.max_content_tokens = 40
        res = {}
        for mode in ("greedy", "sample"):
            counts = {"morning": 0, "afternoon": 0, "invalid": 0}
            succ = 0
            for i in range(a.n):
                t = make_content_task("content_email", a.split, 500 + i)
                ep = run_episode(env, t, ContentPolicyActor(cp, mode, seed=i))
                tr = ep.transitions[0]
                c = tr.semantic_action.content if tr.semantic_action else ""
                counts[classify(c, t.private["slots"])] += 1
                succ += int(bool(ep.success))
            res[mode] = {**{k: v / a.n for k, v in counts.items()}, "success": succ / a.n}
        out[os.path.basename(r.rstrip("/"))] = res
        print(os.path.basename(r.rstrip("/")), json.dumps(res))
    if a.out:
        open(a.out, "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
