"""G2-runtime probe: kill -9 while an ended objective is settling (INT3B settle_after_finish_s), then restart.

    .venv/bin/python cam_live/docs/verification/repro/g2runtime/probe_settle_crash.py

Reuses G-rt's crash machinery (repro/rt/p02_crash.py: world, verifier, policy and clock in this process over the
project RPC; the runtime in a child process killed with SIGKILL at a named point). The p02 scenario ends its
objective by the decision budget (9) with delayed outcomes pending, so the runtime settles. Kill points are chosen
inside the settle phase. Invariants: G-rt's p02 judge (nothing executed twice, every result / outcome logged and
delivered once, counters = ground truth, seq intact, checkpoint complete) plus: a 'settling' event exists, no
decision record after it, exactly one objective_finished, and every outcome that matured inside the window is logged.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RT = os.path.join(os.path.dirname(HERE), "rt")
sys.path.insert(0, HERE)
sys.path.insert(0, RT)
from g2common import Report  # noqa: E402
import p02_crash as P  # noqa: E402
from standin import read_log  # noqa: E402

from cam_live.contracts import RunFiles  # noqa: E402

R = Report("probe_settle_crash")


def settle_checks(info):
    out = {}
    if info.get("summary") is None:
        return {"restart_completes": (False, info.get("stderr", "")[-300:])}
    ev = read_log(info["run_dir"], RunFiles.EVENTS)
    dec = read_log(info["run_dir"], RunFiles.DECISIONS)
    settling = [e for e in ev if e.get("type") == "settling"]
    fin = [e for e in ev if e.get("type") == "objective_finished"]
    out["a_settling_phase_happened"] = (bool(settling), len(settling))
    if settling:
        s0 = settling[0]["seq"]
        late = [d for d in dec if d["seq"] > s0 and not (d["data"].get("situation") or {}).get("written_after_restart")]
        out["no_new_decision_after_settling_began"] = (not late, len(late))
        outs = [e for e in ev if e.get("type") == "outcome" and e["seq"] > s0]
        out["outcomes_measured_while_settling"] = (len(outs) > 0, len(outs))
    out["exactly_one_objective_finished"] = (len(fin) == 1, [{"seq": e["seq"], "status": e["data"].get("status"),
                                                              "settle": e["data"].get("settle"),
                                                              "counters": e["data"].get("counters")} for e in fin])
    if fin:
        out["finished_once_with_settle_record"] = (fin[0]["data"].get("settle") is not None, fin[0]["data"].get("settle"))
    return out


base = P.scenario([], dedup=True)
for n, (ok, d) in settle_checks(base).items():
    R.check(f"{'no kill':30s} {n}", ok, d if not ok else "")

POINTS = ["after_settling_log:1", "after_checkpoint_log:4", "after_outcome_log:2", "after_outcome_log:3",
          "after_reconciled_log:3", "in_reconcile:3", "in_learn:6", "after_objective_finished_log:1",
          "before_wal_reset:5", "after_settling_log:1+after_outcome_log:1"]
for k in POINTS:
    kills = k.split("+")
    info = P.scenario(kills, dedup=True)
    j = P.judge(info)
    j.update(settle_checks(info))
    bad = {n: v[1] for n, v in j.items() if not v[0]}
    R.check(f"{k:30s} all invariants hold", not bad, bad)
sys.exit(R.done())
