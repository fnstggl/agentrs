"""G2-integration probe p04 — obstacle records and obstacle handling (OB-01 .. OB-04, AE-08).

Uses the runtime's ObstacleTracker with the real RunLog, the integrated agent's explain_obstacle on the tiny CPU
configuration, the harness's failure taxonomy / retry policy, and a static scan for failure-conditioned recovery
branches in the decision path (runtime loop, agent, predictor, credit, reward).

Usage (from /workspace/camlive):
    PYTHONPATH=<copy>/cam_live/src .venv/bin/python cam_live/docs/verification/repro/g2integration/p04_obstacles.py
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile

import cam_live
from cam_live import contracts as C
from cam_live.contracts import FailureClass
from cam_live.harness import failures as HF
from cam_live.runtime.logs import RunLog
from cam_live.runtime.obstacles import CATEGORY_BY_FAILURE, ObstacleTracker

SCRATCH = os.environ.get("G2I_SCRATCH", "/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/"
                                        "G2INTEGRATION/p04")
FIVE = {"technical", "informational", "strategic", "resource", "permission"}
broken = 0


def check(name, ok, detail=""):
    global broken
    broken += 0 if ok else 1
    print(f"[{'HOLDS ' if ok else 'BROKEN'}] {name}" + (f" :: {detail}" if detail else ""))


def main() -> int:
    os.makedirs(SCRATCH, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="p04_", dir=SCRATCH)
    classes = [c for c in FailureClass.ALL if c != FailureClass.NONE]

    # ---- OB-01 classification into the five categories of the specification ----------------------------------
    rt_bad = {c: CATEGORY_BY_FAILURE.get(c, None) for c in classes if CATEGORY_BY_FAILURE.get(c) not in FIVE}
    hf_map = getattr(HF, "CATEGORY", None) or next((v for k, v in vars(HF).items()
                                                    if isinstance(v, dict) and FailureClass.STRATEGIC in v
                                                    and v.get(FailureClass.STRATEGIC) == "strategic"), {})
    check("1 every failure class of an obstacle record gets one of the five categories (runtime/obstacles.py)",
          not rt_bad, f"unclassified: {rt_bad}")
    diff = {c: (CATEGORY_BY_FAILURE.get(c), hf_map.get(c)) for c in classes
            if hf_map and CATEGORY_BY_FAILURE.get(c) != hf_map.get(c)}
    check("2 the runtime's obstacle categories agree with the harness's failure taxonomy (harness/failures.py)",
          not diff, f"(runtime, harness): {diff}")

    # ---- OB-01 explanations with probabilities from the integrated agent ------------------------------------
    from cam_live.agent import CamLiveAgent
    from cam_live.config import CamLiveConfig
    task = C.TaskSpec(objective_id="ob", objective="obstacle probe", mode="staging", interfaces=["internal"],
                      limits=C.Limits(duration_s=3600, money=0, max_decisions=10, max_external_actions=1))
    clock = C.SimClock(1000.0)
    cfg = CamLiveConfig.debug(seed=0)
    cfg.sections["agent"] = {"use_backbone": False, "durable": False, "planner": False}   # speed; not under test
    agent = CamLiveAgent(cfg, os.path.join(tmp, "agent"), clock=clock, task=task)
    from cam_live.harness.internal import InternalExecutor
    tools = InternalExecutor(os.path.join(tmp, "exec"), clock=clock).tools()
    cands = agent.propose(tools, 4)
    ev = agent.evaluate(cands)
    dec = agent.select(ev)
    res = C.ActionResult(action_id=dec.chosen.action_id, status=C.ResultStatus.DENIED,
                         failure_class=FailureClass.MISSING_PERMISSION,
                         failure_detail="the authorization service refused: missing permission", started=clock.now(),
                         ended=clock.now() + 1)
    agent.record_result(dec, res)
    log = RunLog(os.path.join(tmp, "run"), clock)
    tr = ObstacleTracker(log, clock)
    rec = tr.open(dec.chosen, res, agent)
    ps = [e["p"] for e in rec.explanations]
    top = rec.explanations[0] if rec.explanations else {}
    check("3 the obstacle record carries explanations with probabilities", bool(ps) and all(0 < p <= 1 for p in ps),
          f"{[(e['hypothesis'], round(e['p'], 3)) for e in rec.explanations]}")
    print(f"[NOTE  ] the class the authorization service reported (missing_permission, a refusal by policy) gets "
          f"p = {top.get('p', 0):.3f} as top explanation; explanations are failure-class labels only, "
          f"from counts per operation, not from the failure's content")
    check("4 the obstacle is categorised as permission-related", rec.category == "permission", rec.category)

    # ---- OB-02 / OB-03 linkage -----------------------------------------------------------------------------
    dec2 = agent.select(agent.evaluate(agent.propose(tools, 4)))
    tr.on_decision(dec2)
    check("5 the next decision's candidates and chosen alternative are recorded",
          rec.chosen_alternative == dec2.chosen.action_id and len(rec.alternatives) >= 1,
          f"{len(rec.alternatives)} alternatives")
    ok2 = C.ActionResult(action_id=dec2.chosen.action_id, status=C.ResultStatus.OK, started=clock.now(),
                         ended=clock.now() + 1)
    tr.on_result(dec2.chosen, ok2)
    check("6 the alternative's result is recorded", "finished with status ok" in rec.outcome, rec.outcome)
    unrelated = C.LearningUpdateRecord(update_id="upd_unrelated_probe", ts=clock.now(), trigger="an update on "
                                       "records that do not include the obstacle", timescale="frequent")
    tr.on_learning([unrelated])
    check("7 only learning updates that used the obstacle's experience are linked to it (not every later update)",
          "upd_unrelated_probe" not in rec.learning_update_ids, f"linked: {rec.learning_update_ids}")
    agent.close()

    # ---- OB-04 hard constraints and no failure-recovery tree ------------------------------------------------
    for hard in FailureClass.HARD:
        try:
            HF.RetryPolicy({hard: HF.RetryRule(max_retries=1)})
            ok = False
        except ValueError:
            ok = True
        check(f"8 a retry rule for the hard class {hard} cannot be configured", ok)
        check(f"9 RetryPolicy.decide never retries {hard}", not HF.RetryPolicy().decide(hard, 0).retry)
    src = os.path.dirname(cam_live.__file__)
    hits = []
    for rel in ("runtime/loop.py", "agent.py", "predict/candidates.py", "predict/decision.py", "predict/policy.py",
                "credit/learner.py", "reward/metalearner.py"):
        path = os.path.join(src, rel)
        if not os.path.exists(path):
            continue
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.IfExp)):
                seg = ast.get_source_segment(open(path).read(), node.test) or ""
                if "failure_class" in seg and ("==" in seg or " in " in seg):
                    hits.append(f"{rel}:{node.lineno}: {seg[:90]}")
    print("[NOTE  ] failure-class-conditioned branches in the decision path:\n    " + "\n    ".join(hits or ["none"]))
    check("10 no branch in the decision path picks a recovery action from the failure class",
          all(h.startswith("agent.py:") and "g is not None" in h for h in hits),
          "the one branch abandons a self-generated goal (agent.py), it does not choose an action" if hits else "")
    print(f"== p04_obstacles: {broken} broken of {10 + 2 * (len(FailureClass.HARD) - 1)} checks ({tmp})")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
