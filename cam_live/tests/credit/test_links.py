"""OutcomeLinker: evidence-only links, fractional links under ambiguity, no future leakage, retrospective queue."""
import math

import pytest

from cam_live.contracts import (ActionSpec, AuthorizationDecision, AuthVerdict, ExperienceItem, ItemKind, Observation,
                                OutcomeObservation, OutcomeStatus, Provenance, Quantity, SimClock)
from cam_live.credit.links import (EV_ARM, EV_ARTIFACT, EV_KEY, EV_PARENT, EV_QUERY, EV_SHARED, EV_WINDOW, OutcomeLinker,
                                   split_uncertainty)
from cam_live.credit.replay import SqliteExperienceStore
from cam_live.credit.simple import SimpleAttribution, credit_from_links, time_decayed_attribution
from cam_live.diagnostics.world import LabConfig, LabWorld


@pytest.fixture
def st(tmp_path):
    s = SqliteExperienceStore(str(tmp_path / "experience.sqlite"), durable=False)
    yield s
    s.close()


def _event(obs_id, ts, data=None, quantities=(), refs=None, parent=""):
    return Observation(obs_id=obs_id, ts=ts, kind="event", quantities=list(quantities), data=data or {},
                       refs=refs or {}, parent_action_id=parent)


def test_declared_query_links_exactly_and_queues_retraining(st, make_record):
    clock = SimClock(5000.0)
    for i in range(4):
        st.append(make_record(i))
    linker = OutcomeLinker(st, clock=clock, retro_predecessors=1)
    q = st.get("e1").pending_queries[0]
    obs = OutcomeObservation(q.query_id, OutcomeStatus.OBSERVED, 7.0, observed_at=1200.0, dedup_key="m@1200")
    rep = linker.reconcile(outcomes=[obs])
    (res,) = rep.results
    (link,) = res.links
    assert (link.exp_id, link.weight, link.uncertainty, link.evidence, link.amount) == ("e1", 1.0, 0.0, EV_QUERY, 7.0)
    assert res.unattributed == 0.0 and rep.outcomes_accepted == 1
    # the linked decision and its predecessor are queued, later decisions are not
    assert rep.retrain_marked == ["e1", "e0"]
    qu = {d["exp_id"]: d["reasons"] for d in st.retrain_queue()}
    assert qu == {"e1": ["outcome_matured"], "e0": ["return_changed"]}
    # the same event again: nothing new
    again = linker.reconcile(outcomes=[obs])
    assert again.duplicates == 1 and again.results == [] and len(st.links()) == 1
    assert SimpleAttribution(st).apply() and st.get("e1").credit == {"simple": 7.0}
    assert "simple" not in st.get("e2").credit        # no evidence, no credit (not zero)
    # an observation that carries no time of its own is dated by the clock at reconciliation
    q3 = st.get("e3").pending_queries[0]
    (r3,) = linker.reconcile(outcomes=[OutcomeObservation(q3.query_id, OutcomeStatus.CENSORED)]).results
    assert r3.links[0].exp_id == "e3" and r3.links[0].outcome_ts == 5000.0 and r3.links[0].amount is None
    assert "e3" not in SimpleAttribution(st).credit()  # an occurrence without a value is not a number


def test_attribution_key_beats_recency(st, make_record):
    """The outcome names an early action. Later actions exist; none of them gets anything."""
    for i in range(6):
        st.append(make_record(i))
    linker = OutcomeLinker(st, clock=SimClock(9000.0))
    ev = _event("ev1", 2000.0, {"by_action": {"a1_0": 3, "a2_0": 1}}, [Quantity("adopters.new", 4)])
    res = linker.link_event(ev)
    assert res.evidence == EV_KEY and {l.exp_id: l.weight for l in res.links} == {"e1": 0.75, "e2": 0.25}
    assert all(l.uncertainty == 0.0 and l.value == 4.0 for l in res.links)
    assert sum(l.amount for l in res.links) == pytest.approx(4.0)
    assert "e5" not in {l.exp_id for l in res.links}             # the most recent decision gets nothing
    single = linker.link_event(_event("ev2", 2000.0, {"decision_id": "d3", "value": 2.0, "metric": "x"}))
    assert [(l.exp_id, l.weight, l.value) for l in single.links] == [("e3", 1.0, 2.0)]
    produced = linker.link_event(_event("ev3", 2000.0, parent="a4_0"))
    assert produced.evidence == EV_PARENT and produced.links[0].exp_id == "e4" and produced.links[0].amount is None


def test_no_evidence_means_no_link_never_the_latest_action(st, make_record):
    for i in range(5):
        st.append(make_record(i, query=False))
    linker = OutcomeLinker(st, clock=SimClock(9000.0))
    res = linker.link_event(_event("ev1", 2000.0, {}, [Quantity("something.happened", 1.0)]))
    assert res.links == [] and res.unattributed == 1.0 and "no evidence" in res.reason
    rep = linker.reconcile(events=[_event("ev9", 2000.0, {"by_action": {"ghost": 2}})])
    assert rep.links == [] and rep.results[0].refused == [{"action_id": "ghost", "reason": "unknown_action"}]
    assert st.retrain_count() == 0 and st.links() == []


def test_ambiguous_evidence_gives_fractional_links_with_uncertainty(st, make_record):
    # three decisions touched the same artifact; two decisions declared a window on the same metric
    for i in range(3):
        st.append(make_record(i, artifacts={"page": "art-7"}, query=False))
    for i in (3, 4):
        r = make_record(i, horizon=1000.0)
        r.pending_queries[0].metric = "signups"
        st.append(r)
    st.append(make_record(5, query=False))                      # most recent, no evidence
    linker = OutcomeLinker(st, clock=SimClock(9000.0))
    res = linker.link_event(_event("ev1", 1500.0, {"artifact_id": "art-7", "value": 6.0, "metric": "visits"}))
    assert res.evidence == EV_ARTIFACT and [l.exp_id for l in res.links] == ["e0", "e1", "e2"]
    for l in res.links:
        assert l.weight == pytest.approx(1 / 3) and l.uncertainty == pytest.approx(math.sqrt(1 / 3 * 2 / 3))
        assert l.amount == pytest.approx(2.0)
    assert sum(l.weight for l in res.links) == pytest.approx(1.0)
    win = linker.link_event(_event("ev2", 1100.0, {}, [Quantity("signups", 10.0)]))
    assert win.evidence == EV_WINDOW and {l.exp_id: l.weight for l in win.links} == {"e3": 0.5, "e4": 0.5}
    assert all(l.uncertainty == pytest.approx(0.5) for l in win.links)       # equal support, not 'the later one'
    # outside every declared window: nothing
    late = linker.link_event(_event("ev3", 90000.0, {}, [Quantity("signups", 10.0)]))
    assert late.links == []
    # credit carries the uncertainty
    cr = credit_from_links(res.links + win.links)
    assert cr["e0"].value == pytest.approx(2.0) and cr["e0"].uncertainty == pytest.approx(6.0 * math.sqrt(2 / 9))
    assert cr["e3"].value == pytest.approx(5.0) and cr["e3"].uncertainty == pytest.approx(5.0)


def test_one_event_answering_two_decisions_is_shared_not_counted_twice(st, make_record):
    for i in range(2):
        st.append(make_record(i))
    linker = OutcomeLinker(st, clock=SimClock(9000.0))
    obs = [OutcomeObservation(f"q{i}", OutcomeStatus.OBSERVED, 8.0, observed_at=1300.0, dedup_key="total@1300")
           for i in range(2)]
    linker.reconcile(outcomes=obs)
    links = st.links(outcome_key="total@1300")
    assert sorted((l["exp_id"], l["weight"], l["evidence"]) for l in links) == [("e0", 0.5, EV_SHARED), ("e1", 0.5, EV_SHARED)]
    cr = SimpleAttribution(st).credit()
    assert cr["e0"].value + cr["e1"].value == pytest.approx(8.0)       # the event is credited once in total
    assert len(st.outcome_events()) == 1


def test_experiment_arm_links(st, make_record):
    for i in range(6):
        st.append(make_record(i, experiment_id="x1", arm="A" if i % 2 == 0 else "B"))
    linker = OutcomeLinker(st, clock=SimClock(9000.0))
    res = linker.link_event(_event("ev1", 5000.0, {"experiment_id": "x1", "arm": "B", "value": 3.0, "metric": "m"}))
    assert res.evidence == EV_ARM and sorted(l.exp_id for l in res.links) == ["e1", "e3", "e5"]
    assert sum(l.weight for l in res.links) == pytest.approx(1.0)


def test_no_future_leakage(st, make_record):
    """A decision made after the outcome cannot be its cause, whatever the event claims; linking does not alter
    the decision-time context; the information available at decision time is recorded."""
    items = [ExperienceItem(kind=ItemKind.OBSERVATION, text=f"o{i}", ts=1000.0 + i, ref=f"it{i}") for i in range(4)]
    st.put_items(items)
    early, late = make_record(0, ts=1010.0), make_record(1, ts=3000.0)
    early.context_item_refs = ["it0", "it1", "it2", "it3"]
    st.append(early), st.append(late)
    linker = OutcomeLinker(st, clock=SimClock(9000.0))
    res = linker.link_event(_event("ev1", 2000.0, {"by_action": {"a0_0": 1, "a1_0": 1}}))
    assert [l.exp_id for l in res.links] == ["e0"] and res.links[0].weight == 0.5
    assert res.unattributed == 0.5                          # the refused half is not handed to someone else
    assert res.refused[0]["reason"] == "decision_after_outcome" and res.refused[0]["exp_id"] == "e1"
    info = res.links[0].available_at_decision
    assert info["decision_ts"] == 1010.0 and info["n_context_items"] == 4 and info["latest_context_ts"] == 1003.0
    assert info["leak_refs"] == []
    # an outcome 'observed' before the decision that owns the query is rejected outright
    rep = linker.reconcile(outcomes=[OutcomeObservation("q1", OutcomeStatus.OBSERVED, 1.0, observed_at=2500.0)])
    assert rep.outcomes_accepted == 0 and rep.rejected[0]["reason"] == "observed before the query was issued"
    # linking wrote targets, never context
    before = st.get("e0").context_item_refs
    linker.reconcile(outcomes=[OutcomeObservation("q0", OutcomeStatus.OBSERVED, 2.0, observed_at=1500.0)],
                     events=[_event("ev5", 1600.0, {"action_id": "a0_0", "value": 1.0, "metric": "m"})])
    after = st.get("e0")
    assert after.context_item_refs == before == ["it0", "it1", "it2", "it3"] and len(after.outcomes) == 1
    assert linker.audit_leakage() == []
    # a context that does contain later information is reported
    st.put_item(ExperienceItem(kind=ItemKind.OUTCOME, text="later", ts=1500.0, ref="it_future"))
    bad = make_record(2, ts=1020.0)
    bad.context_item_refs = ["it0", "it_future"]
    st.append(bad)
    assert linker.audit_leakage() == [{"exp_id": "e2", "decision_ts": 1020.0, "leak_refs": ["it_future"]}]


def test_time_decayed_window_is_bounded_and_causal():
    w = time_decayed_attribution(1000.0, [("a", 990.0), ("b", 900.0), ("c", 1001.0), ("d", 100.0)], window_s=200.0,
                                 half_life_s=50.0, background=0.5)
    assert set(w) == {"a", "b"}                                 # c is after the outcome, d outside the window
    ka, kb = 2 ** (-10 / 50), 2 ** (-100 / 50)
    assert w["a"][0] == pytest.approx(ka / (ka + kb + 0.5)) and w["b"][0] == pytest.approx(kb / (ka + kb + 0.5))
    assert sum(x for x, _ in w.values()) < 1.0 and w["a"][1] == pytest.approx(split_uncertainty(w["a"][0]))
    assert time_decayed_attribution(1000.0, [("c", 1001.0)], window_s=200.0, half_life_s=50.0) == {}


def test_labworld_events_are_linked_through_their_attribution_keys(st):
    """End to end on the diagnostic world: delayed adopters arrive as events carrying ``by_action``; they are
    attributed to the interventions that caused them (ground truth known), not to whatever ran last."""
    from conftest import build_record
    world = LabWorld(LabConfig(seed=3))
    allow = AuthorizationDecision(verdict=AuthVerdict.ALLOW)
    linker = OutcomeLinker(st, clock=world.clock)
    recs, cursor = [], ""
    for i in range(6):
        rec = build_record(i, ts=world.clock.now(), query=False, provenance="diagnostic")
        act = ActionSpec(action_id=rec.decision.chosen.action_id, operation="lab.act",
                         arguments={"channel": i % world.cfg.n_channels, "effort": 3})
        assert world.validate_action(act) == []
        res = world.execute(act, allow)
        rec.result, rec.pending_queries = res, list(res.expected_outcomes)
        st.append(rec)
        recs.append(rec)
    # a later action with no effect on adopters at all
    idle = build_record(99, ts=world.clock.now(), query=False, provenance="diagnostic")
    st.append(idle)
    total = 0.0
    for _ in range(40):
        world.clock.advance(world.cfg.outcome_horizon_s / 20)
        events, cursor = world.poll_events(cursor)
        rep = linker.reconcile(events=events)
        total += sum(l.amount for l in rep.links)
    truth = {aid: sum(1 for t, a, _ in world._arrivals if a == aid and t <= world.clock.now())
             for aid in world._actions}
    assert total == sum(truth.values()) > 0
    cr = SimpleAttribution(st).credit()
    for rec in recs:
        aid = rec.decision.chosen.action_id
        assert (cr[rec.exp_id].value if rec.exp_id in cr else 0.0) == truth[aid]
        assert rec.exp_id not in cr or cr[rec.exp_id].provisional      # no declared measurement yet
    assert idle.exp_id not in cr
    # the declared queries then mature. They measure the same adopters, so the declared measurement replaces the
    # provisional event credit instead of being added to it.
    due = st.pending(world.clock.now())
    obs = world.measure([q for _, q in due])
    rep = linker.reconcile(outcomes=obs)
    assert rep.outcomes_accepted == len(recs) == len(obs)
    final = SimpleAttribution(st).apply()
    for rec, o in zip(recs, sorted(obs, key=lambda o: [r.pending_queries[0].query_id for r in recs].index(o.query_id))):
        cv = final[rec.exp_id]
        assert cv.value == o.value and not cv.provisional and cv.uncertainty == 0.0
        assert st.get(rec.exp_id).credit["simple"] == o.value
    assert sum(cv.value for cv in final.values()) <= total              # counted once, not events + queries
    assert SimpleAttribution(st, combine="sum").credit()[recs[0].exp_id].value >= final[recs[0].exp_id].value


# ---- fix M3 (G-cl): attribution amounts are validated; credit never exceeds the outcome ------------------------------
@pytest.mark.parametrize("amounts", [{"a1_0": 3, "a2_0": -1}, {"a1_0": 2.0, "a2_0": float("nan")},
                                     {"a1_0": 1.0, "a2_0": "many"}, {"a1_0": float("inf"), "a2_0": 1.0}])
def test_malformed_attribution_amounts_are_ambiguous_and_never_create_credit(st, make_record, amounts):
    for i in range(3):
        st.append(make_record(i))
    linker = OutcomeLinker(st, clock=SimClock(5000.0))
    rep = linker.reconcile(events=[_event("o1", 1500.0, {"metric": "members", "value": 2.0, "by_action": amounts})])
    w = {l.exp_id: (l.weight, l.uncertainty) for l in rep.links}
    assert set(w) == {"e1", "e2"}
    assert all(abs(x[0] - 0.5) < 1e-12 and x[1] > 0 for x in w.values())       # ambiguous: equal, uncertain
    assert sum(x[0] for x in w.values()) <= 1.0 + 1e-12
    cred = credit_from_links(rep.links)
    assert abs(sum(c.value for c in cred.values())) <= 2.0 + 1e-12
    assert rep.results[0].links[0].detail["malformed_amounts"]


def test_valid_amounts_are_shares_and_weights_never_exceed_one(st, make_record):
    for i in range(3):
        st.append(make_record(i))
    linker = OutcomeLinker(st, clock=SimClock(5000.0))
    rep = linker.reconcile(events=[_event("o1", 1500.0, {"metric": "m", "value": 4.0,
                                                          "by_action": {"a1_0": 3, "a2_0": 1, "a0_0": 0}})])
    w = {l.exp_id: l.weight for l in rep.links}
    assert w == {"e1": 0.75, "e2": 0.25}
    with pytest.raises(ValueError):                                             # the invariant is enforced in _make
        linker._make("k", [(st.get("e1"), -0.5, 0.0)], EV_KEY, 1500.0, 1.0, "m", False, True, {}, [])
    res = linker._make("k", [(st.get("e1"), 0.9, 0.0), (st.get("e2"), 0.6, 0.0)], EV_KEY, 1500.0, 1.0, "m", False,
                       True, {}, [])
    assert abs(sum(l.weight for l in res.links) - 1.0) < 1e-12
    bad = linker.reconcile(events=[_event("o2", 1500.0, {"metric": "m", "value": float("nan"), "action_id": "a1_0"})])
    assert not bad.links
