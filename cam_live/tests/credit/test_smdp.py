"""Temporally extended actions: traces inside the intervention record, duration-aware discounting, and one outcome
credited once across the three timescales (CU-20)."""
import math

import pytest

from cam_live.contracts import (ActionResult, ActionSpec, Cost, Level, Modality, Observation, PrimitiveOp, ResultStatus,
                                SimClock)
from cam_live.credit.links import OutcomeLinker
from cam_live.credit.replay import SqliteExperienceStore
from cam_live.credit.smdp import (CreditLedger, DoubleCreditError, attach_trace, build_trace, discount, get_trace,
                                  smdp_return, smdp_target, smdp_transition, validate_intervention_record)

N_PROC, OPS_PER = 3, 17            # 3 procedures, 51 primitive operations in total (17 each), like a long browser task


def _hierarchy(st, make_record):
    """One intervention made of three procedures, each of which executed 17 primitive operations. Every level has
    its own experience record (the policy chose at every level); parents are linked by ``parent_decision_id``."""
    t0 = 1000.0
    top = make_record(0, ts=t0, query=False, prefix="I", level=Level.INTERVENTION, probs=[0.6, 0.3, 0.1])
    ops, step_results, steps, procs, prims = [], {}, [], [], []
    t = t0
    for p in range(N_PROC):
        p_start = t
        prec = make_record(p, ts=t, query=False, prefix="P", level=Level.PROCEDURE)
        prec.decision.chosen.parent_decision_id = top.decision.decision_id
        for k in range(OPS_PER):
            dur = 0.5 + 0.1 * k
            op = PrimitiveOp(op="browser.click", modality=Modality.STRUCTURED_BROWSER, started=t, ended=t + dur)
            ops.append(op)
            r = make_record(p * OPS_PER + k, ts=t, query=False, prefix="K", level=Level.PRIMITIVE, duration=dur)
            r.decision.chosen.parent_decision_id = prec.decision.decision_id
            r.result.ops = [op]
            prims.append(r)
            t += dur
        step = ActionSpec(action_id=prec.decision.chosen.action_id, operation="skill.run", level=Level.PROCEDURE,
                          modality=Modality.SKILL, parent_decision_id=top.decision.decision_id)
        steps.append(step)
        step_results[step.action_id] = ActionResult(step.action_id, ResultStatus.OK, started=p_start, ended=t, verified=True)
        prec.result = step_results[step.action_id]
        prec.duration_s = t - p_start
        procs.append(prec)
    top.decision.chosen.steps = steps
    top.decision.candidates[0].action.steps = steps
    top.result = ActionResult(top.decision.chosen.action_id, ResultStatus.OK, started=t0, ended=t, ops=ops,
                              cost=Cost(seconds=t - t0, primitive_ops=len(ops)), verified=True)
    trace = attach_trace(top, step_results)
    with st.batch():
        for r in [top] + procs + prims:
            st.append(r)
    return top, procs, prims, trace


@pytest.fixture
def st(tmp_path):
    s = SqliteExperienceStore(str(tmp_path / "e.sqlite"), durable=False)
    yield s
    s.close()


def test_intervention_record_contains_procedure_and_primitive_traces(st, make_record):
    top, procs, prims, trace = _hierarchy(st, make_record)
    got = st.get(top.exp_id)                                          # after a round trip through the store
    assert validate_intervention_record(got) == []
    tr = get_trace(got)
    assert tr.level == Level.INTERVENTION and tr.n_primitive == N_PROC * OPS_PER == len(got.result.ops) == 51
    assert [c.level for c in tr.children] == [Level.PROCEDURE] * N_PROC
    assert [c.n_primitive for c in tr.children] == [OPS_PER] * N_PROC and tr.op_indices == []
    assert sorted(i for c in tr.children for i in c.op_indices) == list(range(51))      # every operation owned once
    assert tr.children[1].started == pytest.approx(tr.children[0].ended) and tr.termination == "ok"
    tx = smdp_transition(got)
    assert (tx.n_primitive, tx.n_procedures, tx.level) == (51, 3, Level.INTERVENTION)
    assert tx.elapsed_s == pytest.approx(got.result.ended - got.result.started) == pytest.approx(got.duration_s)
    assert tx.probabilities == [0.6, 0.3, 0.1] and tx.propensity == 0.6 and tx.cost.primitive_ops == 51
    assert tx.termination == "ok" and tx.status == ResultStatus.OK
    # without per-step results nothing is guessed: the operations stay with the root
    flat = build_trace(got.decision.chosen, got.result)
    assert flat.op_indices == list(range(51)) and [c.n_primitive for c in flat.children] == [0, 0, 0]
    # a record that lacks the trace or the level is reported
    bare = make_record(5, query=False, prefix="Z", level=Level.PROCEDURE)
    assert validate_intervention_record(bare)[:2] == ["decision level is 'procedure', not intervention", "no trace attached"]
    failed = make_record(6, query=False, prefix="F", status=ResultStatus.TIMEOUT)
    assert attach_trace(failed).termination == ResultStatus.TIMEOUT


def test_duration_aware_discount_is_consistent_across_timescales(st, make_record):
    beta = 3e-3
    top, procs, prims, _ = _hierarchy(st, make_record)
    assert discount(0.0, beta) == 1.0 and discount(100.0, beta) == pytest.approx(math.exp(-0.3))
    assert discount(40.0, beta) * discount(60.0, beta) == pytest.approx(discount(100.0, beta))       # semigroup
    V_after = 7.5                       # value of the state reached when the intervention has finished
    # one backup at intervention level ...
    high = smdp_target(0.0, smdp_transition(top).elapsed_s, V_after, beta)
    # ... equals backing up through the three procedures, and through all 51 primitive operations
    mid = V_after
    for p in reversed(procs):
        mid = smdp_target(0.0, p.duration_s, mid, beta)
    low = V_after
    for k in reversed(prims):
        low = smdp_target(0.0, k.duration_s, low, beta)
    assert high == pytest.approx(mid, rel=1e-12) == pytest.approx(low, rel=1e-12)
    assert high < V_after
    # a per-step discount (gamma per operation) would instead depend on how finely the same time was sliced
    gamma = 0.99
    assert gamma ** 1 * V_after != pytest.approx(gamma ** 51 * V_after)
    # returns of delayed rewards are referenced by time, not by step count
    t0 = top.result.started
    assert smdp_return([1.0, 2.0], [t0, t0 + 600.0], beta) == pytest.approx(1.0 + 2.0 * math.exp(-beta * 600.0))
    assert smdp_return([], [], beta) == 0.0


def test_one_outcome_is_credited_once_across_the_three_timescales(st, make_record):
    top, procs, prims, _ = _hierarchy(st, make_record)
    linker = OutcomeLinker(st, clock=SimClock(1e6))
    ledger = CreditLedger(st)
    # the delayed consequence names a primitive operation deep inside the second procedure
    deep = prims[OPS_PER + 4]
    ev = Observation(obs_id="ev1", ts=5000.0, kind="event",
                     data={"by_action": {deep.decision.chosen.action_id: 12}, "metric": "signups", "value": 12.0})
    res = linker.reconcile(events=[ev]).results[0]
    assert [l.exp_id for l in res.links] == [deep.exp_id]
    entries = ledger.assign(res.links)
    key = res.outcome_key
    counted = [e for e in entries if e.counted]
    inherited = [e for e in entries if not e.counted]
    assert [(e.exp_id, e.level, e.weight, e.credit) for e in counted] == [(top.exp_id, Level.INTERVENTION, 1.0, 12.0)]
    assert {e.exp_id for e in inherited} == {deep.exp_id, procs[1].exp_id}
    assert ledger.counted_weight(key) == 1.0
    assert ledger.by_level(key) == {Level.INTERVENTION: {"counted": 1.0, "inherited": 0.0},
                                    Level.PROCEDURE: {"counted": 0.0, "inherited": 1.0},
                                    Level.PRIMITIVE: {"counted": 0.0, "inherited": 1.0}}
    # 55 records took part; the outcome is one consequential experience and its value is counted once
    assert st.count() == 1 + N_PROC + N_PROC * OPS_PER == 55 and ledger.independent_experiences() == 1
    ledger.apply(entries)
    total = sum(r.credit.get("smdp", 0.0) for r in st.iter_records())
    assert total == 12.0 and st.get(top.exp_id).credit == {"smdp": 12.0}
    assert st.get(deep.exp_id).credit == {"smdp_inherited": 12.0}          # available for procedural learning
    # entering the same links again changes nothing
    ledger.assign(res.links)
    assert ledger.counted_weight(key) == 1.0
    # any attempt to count the same outcome again at another timescale is refused
    for other in (procs[1], deep, prims[0]):
        with pytest.raises(DoubleCreditError):
            ledger.assign_direct(key, other.exp_id, 1.0)
    with pytest.raises(DoubleCreditError):
        ledger.assign_direct(key, procs[0].exp_id, 0.01)
    assert ledger.counted_weight(key) == 1.0 and ledger.independent_experiences() == 1


def test_evidence_at_several_levels_of_one_hierarchy_still_counts_once(st, make_record):
    top, procs, prims, _ = _hierarchy(st, make_record)
    other = make_record(7, ts=1500.0, query=False, prefix="J")           # an unrelated second intervention
    st.append(other)
    linker = OutcomeLinker(st, clock=SimClock(1e6))
    ledger = CreditLedger(st)
    # the event names the intervention itself, one of its procedures, one of its operations, and the other intervention
    amounts = {top.decision.chosen.action_id: 2, procs[0].decision.chosen.action_id: 1,
               prims[3].decision.chosen.action_id: 1, other.decision.chosen.action_id: 4}
    res = linker.reconcile(events=[Observation(obs_id="ev2", ts=9000.0, kind="event", data={"by_action": amounts})]).results[0]
    assert len(res.links) == 4 and sum(l.weight for l in res.links) == pytest.approx(1.0)
    entries = ledger.assign(res.links)
    counted = {e.exp_id: e.weight for e in entries if e.counted}
    assert counted == {top.exp_id: pytest.approx(0.5), other.exp_id: pytest.approx(0.5)}
    assert ledger.counted_weight(res.outcome_key) == pytest.approx(1.0)
    assert ledger.independent_experiences() == 2                     # two interventions share one outcome
    ledger.apply(entries)
    assert sum(r.credit.get("smdp", 0.0) for r in st.iter_records()) == pytest.approx(8.0)      # the event's total
