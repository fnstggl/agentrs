"""cam_live.eval.proposal_probe — what the full-size agent's open-ended proposer writes for an objective.

    python -m cam_live.eval.proposal_probe --family SW --problem rle --rounds 4 --out P.json   (GPU lock)

Builds the real agent (Qwen3-4B + 50m, substantive settings), opens the objective in the open-ended computer,
and calls ``agent.propose`` ``--rounds`` times exactly as the runtime would (same tools, same digest). Records the
backbone's raw text, the parsed actions, the rejection reasons and the merged candidate set. Diagnostic for
change request S-6; it executes nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="SW")
    ap.add_argument("--problem", default="rle")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    from ..nn.gpu import gpu_lock
    from ..validation import OffsetClock
    from .conditions import _CLASSES
    from .experiments import EVAL_HOME
    from .oe_runner import substantive_config
    from .openended import OpenEndedComputer
    root = os.path.join(EVAL_HOME, "oe", "proposal_probe")
    shutil.rmtree(root, ignore_errors=True)
    out = {"family": a.family, "problem": a.problem, "rounds": []}
    with gpu_lock(min_free_gb=12.0):
        clock = OffsetClock()
        cfg = substantive_config("E", 0)
        if a.tokens:
            cfg.sections["agent"]["proposer_max_new_tokens"] = a.tokens
        agent = _CLASSES["E"](cfg, os.path.join(root, "agent"), clock=clock)
        oc = OpenEndedComputer(os.path.join(root, "computer"), clock)
        env, policy, verifier, task = oc.open(a.family, a.problem, 1, "obj_probe", max_decisions=10)
        agent.attach_verifier(verifier)
        agent.set_objective(task)
        agent.ingest(env.observe())
        tools = env.discover_tools()
        out["n_tools"] = len(tools)
        out["prompt_chars"] = len(agent.proposer.prompt(tools, agent._digest(), agent._goal_text()))
        orig = agent.proposer.propose
        captured = {}

        def spy(*args, **kw):
            r = orig(*args, **kw)
            captured["r"] = r
            return r
        agent.proposer.propose = spy
        for i in range(a.rounds):
            t0 = time.time()
            cands = agent.propose(tools, 10)
            r = captured.get("r")
            out["rounds"].append({"seconds": time.time() - t0, "stats": r.stats if r else None,
                                  "raw": r.raw if r else None,
                                  "rejections": [{"reason": x.reason, "detail": x.detail[:200],
                                                  "text": x.fragment[:300]} for x in (r.rejections if r else [])][:20],
                                  "candidates": [{"op": c.operation, "prov": c.provenance,
                                                  "args": json.dumps(c.arguments)[:300]} for c in cands]})
            agent.n_decisions += 1
        oc.close()
        agent.close()
        torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps({"n_tools": out["n_tools"], "prompt_chars": out["prompt_chars"],
                      "stats": [r["stats"] for r in out["rounds"]]}, default=str)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
