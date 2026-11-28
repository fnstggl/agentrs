"""G2-integration probe p07 — is a resource-limit refusal of the authorization service a *resource* obstacle? (OB-01)

The harness classifies a refusal whose every reason starts with "resource: " as FailureClass.RESOURCE_LIMIT
(harness/failures.py refusal_class; harness/auth/limits.py writes exactly such reasons, e.g. "resource:
max_external_actions exhausted ..."). This probe drives the persistent runtime (runtime.loop.LiveRuntime) on LabWorld
with G-rt's scripted stand-in agent and a diagnostic policy that refuses one action with such a limit reason, and
reads what the runtime recorded: the action_result's failure class and the obstacle record's category.
The same thing was observed on the real configuration in my staging run B (browser.click refused with "resource:
max_external_actions exhausted (used 2 incl. reserved, requested 1, limit 2)", recorded as missing_permission /
category permission).

Usage (from /workspace/camlive):
    PYTHONPATH=<copy>/cam_live/src .venv/bin/python cam_live/docs/verification/repro/g2integration/p07_limit_refusal_class.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.abspath(os.path.join(HERE, "..", "rt"))
sys.path.insert(0, RT)
os.environ.setdefault("G_RT_SCRATCH", "/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/"
                                      "G2INTEGRATION/p07")

from cam_live.contracts import AuthorizationDecision, AuthVerdict, FailureClass  # noqa: E402
from cam_live.diagnostics.world import AllowAllPolicy  # noqa: E402
from cam_live.harness.failures import RESOURCE_PREFIX, refusal_class  # noqa: E402
from standin import make_lab_run, read_log  # noqa: E402

broken = 0
REASON = RESOURCE_PREFIX + "max_external_actions exhausted (used 2 incl. reserved, requested 1, limit 2)"


def check(name, ok, detail=""):
    global broken
    broken += 0 if ok else 1
    print(f"[{'HOLDS ' if ok else 'BROKEN'}] {name}" + (f" :: {detail}" if detail else ""))


class LimitPolicy(AllowAllPolicy):
    """Allows everything except the 2nd lab.act, which it refuses for a resource limit (as the real service does)."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def authorize(self, action, task, context=None):
        if action.operation == "lab.act":
            self.n += 1
            if self.n == 2:
                return AuthorizationDecision(AuthVerdict.DENY, action.content_hash(), [REASON], policy_ids=["L-budget"])
        return super().authorize(action, task, context)


def main() -> int:
    check("0 control: the harness classifies this refusal as resource_limit",
          refusal_class([REASON], ["L-budget"]) == FailureClass.RESOURCE_LIMIT, refusal_class([REASON], ["L-budget"]))
    run = make_lab_run("p07", script=("lab.act", "lab.act", "lab.act"), policy=LimitPolicy())
    run.rt.start()
    for _ in range(8):
        run.rt.step()
    run.rt.close()
    results = [r["data"] for r in read_log(run.run_dir, "events.jsonl") if r.get("type") == "action_result"]
    refused = [d for d in results if REASON in str(d.get("failure_detail"))]
    check("1 the refused action was recorded", bool(refused), f"{len(results)} results")
    fc = refused[0].get("failure_class") if refused else None
    check("2 the runtime records the limit refusal as resource_limit (spec obstacle item 3: resource-related)",
          fc == FailureClass.RESOURCE_LIMIT, f"recorded failure_class = {fc!r}")
    obs = [r["data"].get("obstacle", r["data"]) for r in read_log(run.run_dir, "obstacles.jsonl")]
    mine = [o for o in obs if REASON in str(o.get("description"))]
    cat = mine[0].get("category") if mine else None
    check("3 the obstacle record of the limit refusal is categorised as resource", cat == "resource",
          f"category = {cat!r}, failure_class = {mine[0].get('failure_class') if mine else None!r}")
    print(f"== p07_limit_refusal_class: {broken} broken of 4 checks ({run.run_dir})")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
