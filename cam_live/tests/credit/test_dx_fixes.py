"""DX regression tests for the store and the linker (G2_learn.md G2L-2, G2L-3)."""
import pytest

from conftest import build_record

from cam_live.contracts import Observation, OutcomeObservation, OutcomeStatus, Provenance, Quantity, SimClock
from cam_live.credit.links import OutcomeLinker
from cam_live.credit.replay import OutcomeTimingError, SqliteExperienceStore


@pytest.fixture
def store(tmp_path):
    st = SqliteExperienceStore(str(tmp_path / "s.sqlite"))
    yield st
    st.close()


def _obs(q, t, key, value=5.0, status=OutcomeStatus.OBSERVED):
    return OutcomeObservation(query_id=q.query_id, status=status, value=value, observed_at=t, dedup_key=key)


# ---------------------------------------------------------------------------------------------------------- G2L-2
def test_update_refuses_an_appended_outcome_dated_before_its_query_and_writes_nothing(store):
    r = build_record(1)                                   # decision at 1010, query issued at 1015
    store.append(r)
    q = r.pending_queries[0]
    rec = store.get(r.exp_id)
    rec.outcomes.append(_obs(q, 1012.0, "between decision and query"))
    with pytest.raises(OutcomeTimingError, match="before the query was issued"):
        store.update(rec)
    assert store.get(r.exp_id).outcomes == []
    assert store.stats.get("refused_outcomes") == 1
    # the same outcome dated after the query is accepted through update()
    rec = store.get(r.exp_id)
    rec.outcomes.append(_obs(q, 1016.0, "after the query"))
    store.update(rec)
    assert [o.observed_at for o in store.get(r.exp_id).outcomes] == [1016.0]
    # and a later update of the record (credit, labels ...) does not re-judge what is already stored
    rec = store.get(r.exp_id)
    rec.credit["simple"] = 1.0
    store.update(rec)


def test_update_and_append_refuse_an_outcome_dated_before_the_decision(store):
    r = build_record(2)
    store.append(r)
    rec = store.get(r.exp_id)
    stray = OutcomeObservation(query_id="undeclared", status=OutcomeStatus.OBSERVED, value=1.0,
                               observed_at=r.decision.ts - 500.0, dedup_key="stray")
    rec.outcomes.append(stray)
    with pytest.raises(OutcomeTimingError, match="before the decision"):
        store.update(rec)
    r3 = build_record(3)
    r3.outcomes.append(_obs(r3.pending_queries[0], r3.decision.ts - 1.0, "early"))
    with pytest.raises(OutcomeTimingError):
        store.append(r3)
    assert store.get(r3.exp_id) is None
    # an observation time of 0 means "unknown" and is not judged (as in add_outcome); censoring is judged too
    r4 = build_record(4)
    r4.outcomes.append(_obs(r4.pending_queries[0], 0.0, "unknown time"))
    store.append(r4)
    r5 = build_record(5)
    r5.outcomes.append(_obs(r5.pending_queries[0], r5.decision.ts - 1.0, "censored early", status=OutcomeStatus.CENSORED))
    with pytest.raises(OutcomeTimingError):
        store.append(r5)


def test_add_outcome_refuses_an_outcome_before_the_decision_when_the_query_is_unknown(store):
    r = build_record(6)
    store.append(r)
    o = OutcomeObservation(query_id="not-registered", status=OutcomeStatus.OBSERVED, value=2.0,
                           observed_at=r.decision.ts - 10.0, dedup_key="x")
    res = store.add_outcome(o, exp_id=r.exp_id)
    assert not res.accepted and res.reason == "observed before the decision was made"
    o2 = OutcomeObservation(query_id="not-registered", status=OutcomeStatus.OBSERVED, value=2.0,
                            observed_at=r.decision.ts + 10.0, dedup_key="y")
    assert store.add_outcome(o2, exp_id=r.exp_id).accepted


# ---------------------------------------------------------------------------------------------------------- G2L-3
def _event(i, data, value=2.0):
    return Observation(obs_id=f"ev{i}", ts=2000.0 + i, kind="event", text="", data={"metric": "m", **data},
                       quantities=[Quantity("m", value)] if value is not None else [],
                       provenance=Provenance(source="env", trust="environment"))


def test_amounts_that_overflow_when_summed_are_shares_not_a_dropped_outcome(store):
    recs = [build_record(i) for i in (1, 2, 3)]
    for r in recs:
        store.append(r)
    lk = OutcomeLinker(store, clock=SimClock(5000.0))
    a, b, c = (r.decision.chosen.action_id for r in recs)
    res = lk.link_event(_event(1, {"value": 2.0, "by_action": {a: 1e308, b: 1e308}}))
    assert sorted((l.exp_id, l.weight) for l in res.links) == [("e1", 0.5), ("e2", 0.5)]
    assert sum(l.weight * l.value for l in res.links) == pytest.approx(2.0)
    res = lk.link_event(_event(2, {"value": 3.0, "by_action": {a: 1e308, b: 1e308, c: 5e307}}))
    assert [round(l.weight, 12) for l in sorted(res.links, key=lambda l: l.exp_id)] == [0.4, 0.4, 0.2]
    # tiny (subnormal) amounts keep their proportions too
    res = lk.link_event(_event(3, {"value": 1.0, "by_action": {a: 3e-320, b: 1e-320}}))
    assert sorted(round(l.weight, 3) for l in res.links) == [0.25, 0.75]


def test_overflowing_amounts_without_a_stated_value_are_refused_explicitly(store):
    recs = [build_record(i) for i in (1, 2)]
    for r in recs:
        store.append(r)
    lk = OutcomeLinker(store, clock=SimClock(5000.0))
    a, b = (r.decision.chosen.action_id for r in recs)
    res = lk.link_event(_event(4, {"by_action": {a: 1e308, b: 1e308}}, value=None))
    assert not res.links
    assert res.reason == "the outcome value is not finite"           # an explicit refusal, not "no evidence"
    assert any(x.get("reason") == "non_finite_outcome_value" for x in res.refused)
    # with finite amounts and no stated value the summed amount is the outcome (unchanged behaviour)
    res = lk.link_event(_event(5, {"by_action": {a: 1.5, b: 0.5}}, value=None))
    assert sorted((l.exp_id, round(l.weight, 6), l.value) for l in res.links) == [("e1", 0.75, 2.0), ("e2", 0.25, 2.0)]
