"""G-rt probe 15 — a malformed ActionResult from the executor (cost.money is not a number) after the action ran.

    .venv/bin/python cam_live/docs/verification/repro/rt/p15_malformed_result.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from standin import Report, make_slow_run, read_log  # noqa: E402

from cam_live.contracts import RunFiles, SimClock  # noqa: E402

R = Report("p15_malformed_result")
L = make_slow_run("p15", SimClock(), limits={"max_external_actions": 2, "max_decisions": 50})
L.env.result_edit = lambda res: setattr(res.cost, "money", None)       # e.g. an executor that could not price the call
s = L.rt.run(max_iterations=200)
st = json.load(open(os.path.join(L.run_dir, RunFiles.STATE)))
ev = read_log(L.run_dir, RunFiles.EVENTS)
n_res = sum(1 for e in ev if e["type"] == "action_result")
R.note("what happened", {"status": s, "reason": st["status_reason"][:120], "world_executions": len(L.env.executed),
                         "action_result_records": n_res, "results_given_to_learner": len(L.agent.results),
                         "counters": {k: st["counters"][k] for k in ("decisions", "external_actions", "actions_executed")}})
R.check("1 every execution that happened in the environment is counted against max_external_actions",
        st["counters"]["external_actions"] == len(L.env.executed), {"counted": st["counters"]["external_actions"],
                                                                    "executed": len(L.env.executed)})
R.check("2 the limit of 2 external actions is exceeded by at most one", len(L.env.executed) <= 3, len(L.env.executed))
R.check("3 every execution has an action_result record and was reported to the learner",
        n_res == len(L.env.executed) == len(L.agent.results), {"records": n_res, "delivered": len(L.agent.results)})
sys.exit(R.finish())
