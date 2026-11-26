"""Support / query windows from one lifetime (spec 7.5; MR-06, MR-10)."""
import random

import pytest

from cam_live.contracts import (ActionResult, ActionSpec, Decision, ExperienceRecord, OutcomeObservation,
                                OutcomeStatus, ResultStatus)
from cam_live.reward.windows import WindowBuilder, held_out_groups, independence_key, verified_outcomes


def rec(i, ts, *, exp="", parent="", outcome=True, verified=True, status=OutcomeStatus.OBSERVED, dedup=None,
        provenance="live", duration=1.0, obs_delay=5.0, result=True, probs=(0.5, 0.5)):
    a = ActionSpec(action_id=f"a{i}", operation="internal.wait", parent_decision_id=parent)
    d = Decision(decision_id=f"d{i}", ts=ts, objective_id="o", chosen=a, probabilities=list(probs),
                 experiment_id=exp)
    outs = [OutcomeObservation(query_id=f"q{i}", status=status, value=1.0, observed_at=ts + obs_delay,
                               verified=verified, dedup_key=dedup if dedup is not None else f"k{i}")] if outcome else []
    return ExperienceRecord(exp_id=f"e{i}", objective_id="o", decision=d,
                            result=ActionResult(action_id=a.action_id, status=ResultStatus.OK) if result else None,
                            outcomes=outs, duration_s=duration, provenance=provenance, seq=i)


def test_independence_key_prefers_experiment_then_parent_decision():
    r = rec(0, 10.0, exp="x")
    assert independence_key(r) == "exp:x" and independence_key(rec(1, 1.0, parent="d0")) == "dec:d0"
    assert independence_key(rec(2, 1.0)) == "dec:d2"
    via_action = rec(3, 1.0); via_action.decision.chosen.experiment_id = "y"
    assert independence_key(via_action) == "exp:y"


def test_independent_outcomes_are_counted_per_group_not_per_row():
    # support: 4 single decisions; query: 3 experiments x 5 correlated rows + 2 single decisions
    recs = [rec(i, float(i)) for i in range(4)]
    i = 4
    for e in range(3):
        for _ in range(5):
            recs.append(rec(i, 100.0 + i, exp=f"E{e}")); i += 1
    recs += [rec(i, 200.0), rec(i + 1, 201.0)]
    pair = WindowBuilder(min_independent_outcomes=5).cut(recs, boundary_ts=50.0)
    assert pair.query.n_rows == 17 and pair.n_independent_query == 5         # 17 rows, 5 independent outcomes
    assert pair.eligible and pair.reason == ""
    strict = WindowBuilder(min_independent_outcomes=8).cut(recs, boundary_ts=50.0)
    assert not strict.eligible and strict.reason == "insufficient_independent_outcomes:5<8"
    assert len(set(pair.query.row_groups)) == 5 and pair.query.row_groups.count("exp:E0") == 5


def test_rows_sharing_one_verified_event_or_parent_are_one_group():
    recs = [rec(0, 0.0)] + [rec(i, 10.0 + i, dedup="same-event") for i in range(1, 5)] \
        + [rec(i, 20.0 + i, parent="d-root") for i in range(5, 8)] + [rec(8, 30.0)]
    pair = WindowBuilder(min_independent_outcomes=1).cut(recs, boundary_ts=5.0)
    assert pair.query.n_rows == 8 and pair.n_independent_query == 3


def test_temporal_separation_gap_and_straddling_groups():
    recs = [rec(0, 0.0, duration=30.0), rec(1, 10.0, exp="LONG"), rec(2, 20.0)]
    recs += [rec(3, 25.0), rec(4, 40.0, exp="LONG"), rec(5, 50.0), rec(6, 60.0)]
    pair = WindowBuilder(min_independent_outcomes=1, min_gap_s=5.0).cut(recs, boundary_ts=22.0)
    assert pair.support.exp_ids == ("e0", "e1", "e2") and pair.support.t_end == 30.0
    # e3 starts before the support transitions ended (+gap); e4 belongs to an experiment that began in support
    assert pair.query.exp_ids == ("e5", "e6") and dict(pair.dropped) == {"inside_gap": 1, "straddles_boundary": 1}
    assert pair.query.t_start >= pair.support.t_end + 5.0 and pair.gap_s == 20.0
    assert "exp:LONG" not in pair.query.groups


def test_only_admissible_verified_outcomes_count():
    recs = [rec(0, 0.0)]
    recs += [rec(1, 10.0), rec(2, 11.0, verified=False), rec(3, 12.0, status=OutcomeStatus.PENDING),
             rec(4, 13.0, status=OutcomeStatus.UNOBSERVABLE), rec(5, 14.0, outcome=False),
             rec(6, 15.0, obs_delay=-3.0),                       # "observed" before its own decision
             rec(7, 16.0, status=OutcomeStatus.CENSORED), rec(8, 17.0, provenance="staging"),
             rec(9, 18.0, result=False)]
    b = WindowBuilder(min_independent_outcomes=1)
    pair = b.cut(recs, boundary_ts=5.0)
    assert pair.query.groups == ("dec:d1", "dec:d7") and pair.n_independent_query == 2
    assert dict(pair.dropped) == {"no_result": 1, "provenance_staging": 1}
    assert "e8" not in pair.query.exp_ids and "e9" not in pair.query.exp_ids
    assert WindowBuilder(provenance="staging").cut(recs, 5.0).query.exp_ids == ("e8",)     # never mixed
    # a signature filter (e.g. ledger membership) is applied on top
    signed = {"k1"}
    flt = WindowBuilder(min_independent_outcomes=1, outcome_filter=lambda r, o: o.dedup_key in signed)
    assert flt.cut(recs, 5.0).query.groups == ("dec:d1",)
    assert [o.dedup_key for o in verified_outcomes(recs[1])] == ["k1"] and verified_outcomes(recs[2]) == []
    dup = rec(20, 1.0); dup.outcomes = dup.outcomes * 3
    assert len(verified_outcomes(dup)) == 1
    assert not WindowBuilder(require_propensities=True).row_eligible(rec(21, 1.0, probs=()))[0]


def test_window_ids_are_deterministic_and_content_addressed():
    recs = [rec(i, float(i * 10)) for i in range(12)]
    b = WindowBuilder(min_independent_outcomes=2)
    p1 = b.cut(recs, boundary_ts=55.0)
    shuffled = recs[:]
    random.Random(3).shuffle(shuffled)
    p2 = WindowBuilder(min_independent_outcomes=2).cut(shuffled, boundary_ts=55.0)
    assert (p1.pair_id, p1.support.window_id, p1.query.window_id) == (p2.pair_id, p2.support.window_id,
                                                                    p2.query.window_id)
    assert p1.support.window_id.startswith("ws_") and p1.query.window_id.startswith("wq_")
    p3 = b.cut(recs[:-1], boundary_ts=55.0)
    assert p3.query.window_id != p1.query.window_id and p3.support.window_id == p1.support.window_id
    assert p3.pair_id != p1.pair_id


def test_blocks_and_pairs_chain_through_one_lifetime_and_report_thin_evidence():
    recs = []
    i = 0
    for e in range(10):                                        # 10 experiments of 3 correlated rows each
        for _ in range(3):
            recs.append(rec(i, float(i * 10), exp=f"E{e}")); i += 1
    recs += [rec(i, float(i * 10))]                            # a trailing single decision
    b = WindowBuilder(min_independent_outcomes=4)
    blocks = b.blocks(recs)
    assert [len(x) for x in blocks] == [10, 12, 9]             # a block closes at its 4th independent outcome
    pairs = b.pairs(recs)
    assert len(pairs) == 2
    first, second = pairs
    assert first.eligible and first.n_independent_query == 4
    assert dict(first.dropped) == {"straddles_boundary": 2}    # the rest of the experiment cut by the boundary
    assert second.support.exp_ids == tuple(r.exp_id for r in blocks[1])
    assert not second.eligible and second.reason == "insufficient_independent_outcomes:3<4"
    assert max(first.support.t_end, 0) <= first.query.t_start
    all_q = set(first.query.exp_ids)
    assert not all_q & set(first.support.exp_ids)
    assert WindowBuilder(min_independent_outcomes=4).pairs(recs[:5]) == []     # one block only: nothing to pair
    empty = b.cut(recs, boundary_ts=-1.0)
    assert not empty.eligible and empty.reason == "empty_support_window"


def test_held_out_groups_are_deterministic():
    groups = [f"exp:E{i}" for i in range(200)]
    h1, h2 = held_out_groups(groups, 0.25, salt="p1"), held_out_groups(list(reversed(groups)), 0.25, salt="p1")
    assert h1 == h2 and 25 <= len(h1) <= 75
    assert held_out_groups(groups, 0.25, salt="p2") != h1
    assert held_out_groups(groups, 0.0) == set()
