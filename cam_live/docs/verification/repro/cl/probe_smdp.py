"""G-cl probe: temporally extended actions (CU-20).

One intervention T with a procedure P and three primitive operations O1..O3 below it (parent_decision_id chain).
Checks on the production path (store, linker, simple credit, trajectory_return, learner trigger unit) and on the
CreditLedger:
  * duration-aware discount composes across levels: exp(-b(t2-t0)) = exp(-b(t1-t0)) exp(-b(t2-t1)) (smdp_target);
  * one event, reported once by the primitive that produced it and once as the answer to T's declared query (same
    dedup key), enters the return once and is credited once over the hierarchy;
  * the learner counts the hierarchy as ONE independent unit, not one per row;
  * CreditLedger refuses counted weight > 1 for one outcome;
  * the production credit path (CreditAssigner -> simple) does not go through CreditLedger (D's own limitation).

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/cl/probe_smdp.py
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys

sys.path.insert(0, "cam_live/src")
sys.path.insert(0, "cam_live/tests/credit")
from conftest import build_record                                                   # noqa: E402
from cam_live.contracts import Level, Observation, OutcomeObservation, SimClock    # noqa: E402
from cam_live.credit.links import OutcomeLink, OutcomeLinker                       # noqa: E402
from cam_live.credit.providers import top_index, trajectory_return                 # noqa: E402
from cam_live.credit.replay import SqliteExperienceStore                           # noqa: E402
from cam_live.credit.simple import credit_from_links                               # noqa: E402
from cam_live.credit.smdp import CreditLedger, DoubleCreditError, discount, smdp_return, smdp_target   # noqa: E402

WORK = os.environ.get("CL_WORK", "/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/G_cl/work_smdp")
FAILS, OUT = [], {}


def check(name, ok, detail=""):
    print(("[PASS] " if ok else "[FAIL] ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def main():
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)
    b = 3e-4
    # composition of the duration-aware discount
    lhs = smdp_target(1.0, 100.0, smdp_target(2.0, 250.0, 5.0, b), b)
    rhs = smdp_return([1.0, 2.0, 5.0], [0.0, 100.0, 350.0], b)
    check("smdp_target composes to the time-discounted return", abs(lhs - rhs) < 1e-12, f"{lhs} vs {rhs}")
    check("discount(t2-t0) = discount(t1-t0) * discount(t2-t1)", abs(discount(350, b) - discount(100, b) * discount(250, b)) < 1e-15)
    st = SqliteExperienceStore(os.path.join(WORK, "s.sqlite"))
    lk = OutcomeLinker(st, clock=SimClock(1e4))
    T = build_record(0, ts=1000.0, horizon=5000.0, objective="o")
    P = build_record(1, ts=1001.0, query=False, objective="o", level=Level.PROCEDURE)
    P.decision.chosen.parent_decision_id = T.decision.decision_id
    Os = []
    for i in range(3):
        o = build_record(2 + i, ts=1002.0 + i, query=False, objective="o", level=Level.PRIMITIVE)
        o.decision.chosen.parent_decision_id = P.decision.decision_id
        Os.append(o)
    for r in [T, P] + Os:
        st.append(r)
    # the primitive O2 produced a 'member joined' event; the same event (same dedup key) answers T's declared query
    ev = Observation(obs_id="ev1", ts=1500.0, kind="event", text="member joined", parent_action_id=Os[1].decision.chosen.action_id,
                     data={"metric": "members", "value": 1.0, "dedup_key": "join-42"})
    r1 = lk.reconcile(events=[ev])
    r2 = lk.reconcile(outcomes=[OutcomeObservation(query_id="q0", status="observed", value=1.0, observed_at=1500.0,
                                                   dedup_key="join-42")])
    links = [OutcomeLink.from_dict(l) for l in st.links()]
    by = {}
    for l in links:
        by[l.exp_id] = by.get(l.exp_id, 0.0) + l.weight
    OUT["links_after_both_reports"] = by
    cred = credit_from_links(links)
    total = sum(v.value for v in cred.values())
    check("same event at two levels: credited once over the hierarchy (simple)", abs(total - 1.0) < 1e-12,
          f"simple credit {({k: v.value for k, v in cred.items()})}")
    st.put_objective_outcome("o", status="achieved", value=1.0, evaluated_at=6000.0, verified=True)
    tr = trajectory_return(st, st.trajectory("o"), beta=b)
    check("trajectory return counts the event once, discounted from the intervention start",
          len(tr.events) == 1 and abs(tr.G - math.exp(-b * 500.0)) < 1e-12, f"G={tr.G}, events={len(tr.events)}")
    ti = top_index(st.trajectory("o"))
    check("all five records map to the one top-level decision", set(ti.values()) == {0}, str(ti))
    # learner independence unit (same rule as ContinualLearner._unit): parent chain to the top
    def unit(rec):
        cur = rec
        while cur.decision.chosen.parent_decision_id:
            cur = st.by_decision(cur.decision.chosen.parent_decision_id)
        return cur.decision.decision_id
    units = {unit(r) for r in st.trajectory("o")}
    check("the hierarchy is one independent unit", len(units) == 1, str(units))
    # ledger: counted weight never above 1
    led = CreditLedger(st)
    ents = led.assign(links)
    check("ledger: counted weight of the event is 1, on the top decision", abs(led.counted_weight("join-42") - 1.0) < 1e-12
          and all((e.exp_id == T.exp_id) == e.counted for e in ents), str([(e.exp_id, e.weight, e.counted) for e in ents]))
    try:
        led.assign_direct("join-42", Os[0].exp_id, 0.5, counted=True)
        check("ledger refuses a second counted assignment of the same outcome", False)
    except DoubleCreditError:
        check("ledger refuses a second counted assignment of the same outcome", True)
    # production path: does the simple credit written by the assigner put credit on non-top records?
    OUT["simple_credit_on_levels"] = {st.get(k).decision.level: v.value for k, v in cred.items()}
    print(json.dumps(OUT, indent=1))
    st.close()
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)


if __name__ == "__main__":
    main()
