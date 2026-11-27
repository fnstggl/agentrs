"""Gate I (MASTER_SPEC Part 17): large-scale social reach. Owner: workstream C2 (I.n2 measured here for N).

Two runs of the ordinary engine on the ONE population (`crucible.population`, internet profile, columnar path;
the crowd stand-in is not used): members who write and publish through content.publish, platform recommenders, the
consumption engine as population activities, sharing to each member's contacts, exact audience accounting with
decision provenance:

* the **positive control** (I.1-I.7): 10,000,000 identities of `pop`; the experiment is designed so that a large
  cascade is expected (one informative post by an experimenter account at the start on the population's most popular
  news topic - the control item - and a trending-weighted mixture). It runs hour by hour until the control item has
  1,000,000 unique viewers and the world has 10,000,000 qualifying distinct consumption events, or 30 simulated
  hours, whichever is first;
* the **organic run** (I.n1): 2,500,000 identities, an ordinary configuration (no favoured topic, no seeded post,
  ordinary mixture and sharing), 24 simulated hours; the most viewed item's reach is measured.

Every number is recomputed from the recorded `cols.batch` decision blobs (`audience.recount`), never read from a
counter alone. These runs show capacity and that reach arises from the ordinary mechanisms; they make no claim that
the cascades are empirically realistic (I.n2).

Run through the slot scheduler (about 20 minutes, up to ~15 GB of memory):
PYTHONPATH=src python -m crucible.scale.slots --cpu 4 --label "ws-C2: gate I" -- python -m pytest -q tests/v5/test_gate_I.py
"""
import glob
import json
import os
import re
import time

import numpy as np
import pytest

import crucible
from crucible.acceptance.v5 import evidence
from crucible.core.kernel import World
from crucible.social import audience as AUD
from crucible.social import cascade
from social_helpers import assert_met, failed_ticks  # noqa: E402

crucible.load_all_modules()
CMD = ('PYTHONPATH=src python -m crucible.scale.slots --cpu 4 --label "ws-C2: gate I" -- '
       'python -m pytest -q tests/v5/test_gate_I.py')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N_PC = int(os.environ.get("CRUCIBLE_GATE_I_N", 10_000_000))
N_ORGANIC = int(os.environ.get("CRUCIBLE_GATE_I_ORGANIC_N", 2_500_000))
H = 3600
pytestmark = [pytest.mark.slow, pytest.mark.timeout(4 * 3600)]


@pytest.fixture(scope="module")
def positive_control():
    t0 = time.time()
    w = World.create(cascade.pop_internet_spec(N_PC, 7, positive_control=True))
    item = cascade.positive_control_post(w)
    t_create = time.time() - t0
    progress = []
    while w.now < 30 * H:
        progress += cascade.run(w, w.now + H, step=H)
        last = progress[-1]
        if int(AUD.item_col(w, "viewers")[item]) >= 1_000_000 and last["distinct_views"] >= 10_000_000:
            break
    t_run = time.time() - t0 - t_create
    m = cascade.measure(w, item)
    m["materialised_sample"] = cascade.materialise_sample(w, 1000)
    m["distinct_state_rows"] = cascade.distinct_state_rows(w)
    m.update(create_wall_s=round(t_create, 1), run_wall_s=round(t_run, 1), simulated_hours=w.now / H,
             events=w.seq)
    return {"world": w, "m": m, "progress": progress, "item": item}


@pytest.fixture(scope="module")
def organic():
    t0 = time.time()
    w = World.create(cascade.pop_internet_spec(N_ORGANIC, 8, positive_control=False))
    progress = cascade.run(w, 24 * H, step=2 * H)
    m = cascade.measure(w)
    m.update(run_wall_s=round(time.time() - t0, 1))
    return {"world": w, "m": m, "progress": progress}


def _notes(pc):
    p = pc["progress"]
    traj = [(r["t"] // H, r["top_item"], r["top_unique_viewers"], r["distinct_views"]) for r in p]
    return (f"Positive-control run: {pc['m']['distinct_identities']} identities, {pc['m']['simulated_hours']:.0f} "
            f"simulated hours, {pc['m']['events']} events, create {pc['m']['create_wall_s']} s, run "
            f"{pc['m']['run_wall_s']} s, control item {pc['item']} reached {pc['m']['item_unique_viewers']} unique "
            f"viewers. Hourly (hour, top item, its unique viewers, distinct qualifying views): {traj}")


def test_I1_ten_million_identities(positive_control):
    m = positive_control["m"]
    evidence.record("I.1", {"distinct_identities": m["distinct_identities"],
                            "distinct_identities_with_qualifying_consumption":
                                m["distinct_identities_with_qualifying_consumption"],
                            "distinct_state_rows": m["distinct_state_rows"],
                            "identities_with_decisions": m["identities_with_decisions"],
                            "materialised_sample_principals": m["materialised_sample"], "estimated": False},
                    command=CMD, notes="Identities are rows of the columnar population table (each with its own "
                    "interests, attention ledger, satisfaction, share propensity, inbox and activity state); an "
                    "individual of crucible.population becomes a kernel principal p<i> on first need (a random sample of 1,000 was "
                    "materialised). distinct_state_rows counts distinct (interests, attention budget) rows by "
                    "64-bit hash. Exact, individual-level: no estimated-population mode is used. " + _notes(positive_control))
    assert_met('I.1', {"distinct_identities": m["distinct_identities"],
               "distinct_identities_with_qualifying_consumption":
               m["distinct_identities_with_qualifying_consumption"],
               "distinct_state_rows": m["distinct_state_rows"],
               "identities_with_decisions": m["identities_with_decisions"],
               "materialised_sample_principals": m["materialised_sample"], "estimated": False})
    assert m["distinct_identities"] == N_PC and m["materialised_sample"] == 1000
    ft = failed_ticks(positive_control["world"])
    assert ft["failed_ticks"] == 0, ft["failed_tick_errors"]


def test_I2_verified_exposure_accounting(positive_control):
    m = positive_control["m"]
    evidence.record("I.2", {"exposure_recount_mismatches": m["exposure_recount_mismatches"],
                            "exposures_checked": m["exposures_checked"], "impression_rows": m["impression_rows"]},
                    command=CMD, notes="Every impression row of every recorded impress.<platform> batch blob was "
                    "re-read; impressions and distinct (item, identity) exposures per item were recomputed "
                    "independently and compared with the state counters and, member by member, with each item's "
                    "exact exposed set. " + _notes(positive_control))
    assert_met('I.2', {"exposure_recount_mismatches": m["exposure_recount_mismatches"],
               "exposures_checked": m["exposures_checked"], "impression_rows": m["impression_rows"]})
    assert m["exposures_checked"] > 0


def test_I3_verified_distinct_viewer_accounting(positive_control):
    m = positive_control["m"]
    evidence.record("I.3", {"distinct_viewer_recount_mismatches": m["distinct_viewer_recount_mismatches"],
                            "distinct_viewers_checked": m["distinct_viewers_checked"],
                            "consumption_rows": m["consumption_rows"],
                            "total_recount_mismatches": m["total_recount_mismatches"]},
                    command=CMD, notes="Views and unique viewers per item recomputed from every consumption row "
                    "(the view rule applied to watch_s and the item's duration) and compared with the counters and "
                    "the exact viewer sets member by member; all ten audience quantities were recounted. " +
                    _notes(positive_control))
    assert_met('I.3', {"distinct_viewer_recount_mismatches": m["distinct_viewer_recount_mismatches"],
               "distinct_viewers_checked": m["distinct_viewers_checked"],
               "consumption_rows": m["consumption_rows"],
               "total_recount_mismatches": m["total_recount_mismatches"]})
    assert m["distinct_viewers_checked"] > 0


def test_I4_finite_attention(positive_control):
    m = positive_control["m"]
    evidence.record("I.4", {"attention_budget_violations": m["attention_budget_violations"],
                            "attention_ledger_violations": m["attention_ledger_violations"],
                            "daily_consumption_over_budget": m["daily_consumption_over_budget"]},
                    command=CMD, notes="The attention ledger (left >= 0, spent <= daily budget) checked for all "
                    "identities, and per identity and day the seconds consumed in the recorded consumption rows "
                    "compared with the daily budget. " + _notes(positive_control))
    assert_met('I.4', {"attention_budget_violations": m["attention_budget_violations"],
               "attention_ledger_violations": m["attention_ledger_violations"],
               "daily_consumption_over_budget": m["daily_consumption_over_budget"]})
    assert m["consumption_rows"] > 0


def test_I5_organic_propagation(positive_control):
    m = positive_control["m"]
    evidence.record("I.5", {"forced_view_assignments": m["forced_view_assignments"] + m["decision_rows_without_provenance"],
                            "views_without_prior_exposure": m["forced_view_assignments"],
                            "decision_rows_without_provenance": m["decision_rows_without_provenance"],
                            "share_generations_observed": m["share_generations_observed"],
                            "share_generation_views": json.dumps(m["share_generation_views"])},
                    command=CMD, notes="A forced view is a consumption row without an earlier recorded exposure "
                    "of the same identity to the same item (the consumption API also refuses such rows), or any "
                    "audience decision row without decision provenance (not listed by the population's own session "
                    "of the deciding individual: population.check_provenance). Share "
                    "generations: qualifying views reached through an inbox share of generation g (1 = shared by "
                    "someone who found it in a feed). " + _notes(positive_control))
    assert_met('I.5', {"forced_view_assignments": m["forced_view_assignments"] + m["decision_rows_without_provenance"],
               "views_without_prior_exposure": m["forced_view_assignments"],
               "decision_rows_without_provenance": m["decision_rows_without_provenance"],
               "share_generations_observed": m["share_generations_observed"],
               "share_generation_views": json.dumps(m["share_generation_views"])})
    assert m["share_generations_observed"] >= 1


def test_I6_positive_control_cascade(positive_control):
    m = positive_control["m"]
    w = positive_control["world"]
    it = positive_control["item"]
    rec = AUD.info(w, it)
    evidence.record("I.6", {"positive_control_unique_viewers": m["item_unique_viewers"],
                            "positive_control_unique_exposed": m["item_unique_exposed"],
                            "positive_control_item": it, "top_item": m["top_item"],
                            "forced_view_assignments": m["forced_view_assignments"] +
                            m["decision_rows_without_provenance"]},
                    command=CMD, notes=f"Control item {it} ('{rec['title']}', {rec['words']} words, by "
                    f"{rec['creator']}, an experimenter account: scripted publication) is an informative text post "
                    "(real news abstracts) published through the ordinary tools at the start; it reached people "
                    "only through feeds and shares, every view a population decision. " + _notes(positive_control))
    assert_met('I.6', {"positive_control_unique_viewers": m["item_unique_viewers"],
               "positive_control_unique_exposed": m["item_unique_exposed"],
               "positive_control_item": it, "top_item": m["top_item"],
               "forced_view_assignments": m["forced_view_assignments"] +
               m["decision_rows_without_provenance"]})
    assert m["item"] == it


def test_I7_ten_million_qualifying_distinct_consumption_events(positive_control):
    m = positive_control["m"]
    evidence.record("I.7", {"qualifying_distinct_consumption_events": m["qualifying_distinct_consumption_events"],
                            "distinct_identities_with_qualifying_consumption":
                                m["distinct_identities_with_qualifying_consumption"],
                            "qualifying_views": m["qualifying_views"], "consumption_rows": m["consumption_rows"],
                            "distinct_viewer_recount_mismatches": m["distinct_viewer_recount_mismatches"],
                            "run_wall_s": m["run_wall_s"]},
                    command=CMD, notes="Qualifying distinct consumption events = distinct (identity, item) pairs with "
                    "at least one consumption row meeting the view rule, recounted exactly from the recorded batches; "
                    "distinct_identities_with_qualifying_consumption counts the identities among them (spec 4.6's "
                    "condition for an exact claim of that many unique viewers). Capacity, not realism. " +
                    _notes(positive_control))
    assert_met('I.7', {"qualifying_distinct_consumption_events": m["qualifying_distinct_consumption_events"],
               "distinct_identities_with_qualifying_consumption":
               m["distinct_identities_with_qualifying_consumption"],
               "qualifying_views": m["qualifying_views"], "consumption_rows": m["consumption_rows"],
               "distinct_viewer_recount_mismatches": m["distinct_viewer_recount_mismatches"],
               "run_wall_s": m["run_wall_s"]})
    assert m["qualifying_distinct_consumption_events"] == m["distinct_viewers_checked"]


def test_In1_organic_large_adoption(organic):
    m = organic["m"]
    w = organic["world"]
    traj = [(r["t"] // H, r["top_item"], r["top_unique_viewers"]) for r in organic["progress"]]
    evidence.record("I.n1", {"organic_cascade_unique_viewers": m["top_item_unique_viewers"],
                             "forced_view_assignments": m["forced_view_assignments"] +
                             m["decision_rows_without_provenance"],
                             "organic_item": m["top_item"], "distinct_identities": m["distinct_identities"],
                             "share_generations_observed": m["share_generations_observed"],
                             "total_recount_mismatches": m["total_recount_mismatches"],
                             "items": m["items"]},
                    command=CMD, notes=f"Ordinary configuration ({m['distinct_identities']} identities of the one "
                    "population, no seeded post, only the population's own creators, ordinary mixture: interest "
                    "retrieval, social, freshness, exploration): "
                    "the most viewed item of 24 simulated hours. Every view recounted from recorded batches. "
                    f"Trajectory (hour, top item, its unique viewers): {traj}")
    assert_met('I.n1', {"organic_cascade_unique_viewers": m["top_item_unique_viewers"],
               "forced_view_assignments": m["forced_view_assignments"] +
               m["decision_rows_without_provenance"],
               "organic_item": m["top_item"], "distinct_identities": m["distinct_identities"],
               "share_generations_observed": m["share_generations_observed"],
               "total_recount_mismatches": m["total_recount_mismatches"],
               "items": m["items"]})
    assert m["total_recount_mismatches"] == 0
    ft = failed_ticks(w)
    assert ft["failed_ticks"] == 0, ft["failed_tick_errors"]


def test_In2_no_unsupported_realism_claims():
    """Every document a realism claim could hide in (README, status and architecture documents, docs/**, evidence
    notes, every module docstring; `crucible.research.v5.integrity.corpus`) is scanned clause by clause: a clause that
    asserts realism of large-scale viral dynamics is a claim unless a negation governs the assertion in that clause
    ("... with no hand tuning" disclaims nothing). The scanner must first classify its planted claims and
    disclaimers correctly. No validated evidence of realistic ten-million-view dynamics exists, so every claim found
    is unsupported."""
    from crucible.research.v5 import integrity as G
    r = G.scan_claims("realism_cascades", G.corpus(ROOT))
    evidence.record("I.n2", {"unsupported_realism_claims": r["unsupported_claims"], "statements_checked":
                             r["statements_checked"], "files_scanned": r["files_scanned"],
                             "planted_controls_correct": r["controls"]["ok"],
                             "validated_cascade_realism_evidence": False},
                    notes=f"Claims found: {r['claims'][:10]}; planted controls {r['controls']}")
    assert_met('I.n2', {"unsupported_realism_claims": r["unsupported_claims"], "statements_checked":
               r["statements_checked"], "files_scanned": r["files_scanned"],
               "planted_controls_correct": r["controls"]["ok"],
               "validated_cascade_realism_evidence": False})
    assert r["controls"]["ok"], r["controls"]
    assert r["files_scanned"] > 100 and r["statements_checked"] >= 1
    assert r["unsupported_claims"] == 0, r["claims"]
