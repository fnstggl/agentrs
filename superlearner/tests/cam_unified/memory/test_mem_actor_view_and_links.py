"""W9 memory hygiene (plan 2026-11-26 §1 Memory; audit 'archive' HashedEventEncoder + delayed-outcome findings):

* the default memory event encoder reads the actor view only: canaries planted in the record's reward, cost, done,
  extra labels, training provenance and goal id leave event tokens, segment keys and memory values bitwise
  identical (the audit measured L2 4.79 from the reward alone); the legacy encoder is a declared baseline refused
  under observation_only;
* the archive indexes delayed-outcome links (attributed_to, Observation.caused_by) both ways, survives a restart and
  a lost index, backfills an index written before the links table existed, and filters on them without a scan.
"""
from __future__ import annotations

import copy
import os
import sqlite3

import pytest
import torch

from memhelpers import light_events
from superlearner.cam_unified.interfaces import Observation, ObsKind
from superlearner.cam_unified.memory.archive import EventArchive, row_links
from superlearner.cam_unified.memory.featurize import ACTOR_VIEW_FIELDS, HashedEventEncoder, actor_view
from superlearner.cam_unified.memory.retrieval import Retriever
from superlearner.cam_unified.testing import tiny_config

CANARY = 7_391_113.25


def _plant(rec):
    r = copy.deepcopy(rec)
    r.reward = CANARY
    r.cost = -CANARY
    r.done = not r.done
    r.goal_id = "goal-canary-7391113"
    r.extra = dict(r.extra, success=True, verified_outcome=CANARY, utility=0.93, family="rapid:canary_family",
                   label={"status": "success", "value": CANARY})
    r.training_provenance = {"used_in_updates": ["u-canary"]}
    return r


def test_event_tokens_are_bitwise_identical_under_evaluator_side_canaries():
    torch.manual_seed(0)
    enc = HashedEventEncoder(48)
    evs = light_events(12)
    planted = [_plant(e) for e in evs]
    a, b = enc(evs), enc(planted)
    assert torch.equal(a, b)
    for e, p in zip(evs, planted):
        assert enc.tokens(e) == enc.tokens(p) and enc.numeric(e) == enc.numeric(p)
        assert not any("canary" in t or "7391113" in t for t in enc.tokens(p))
    # the encoder still sees what the actor saw: another observation value / result changes the token
    other = copy.deepcopy(evs[0])
    other.observations[0].value = other.observations[0].value + 50.0
    assert not torch.equal(enc([other]), a[:1])
    other = copy.deepcopy(evs[0])
    other.result.ok = not other.result.ok
    assert not torch.equal(enc([other]), a[:1])
    assert set(actor_view(evs[0])) == set(ACTOR_VIEW_FIELDS) and "reward" not in actor_view(evs[0])


def test_legacy_encoder_reads_the_reward_and_is_refused_under_observation_only():
    torch.manual_seed(0)
    leg = HashedEventEncoder(48, view="legacy")
    evs = light_events(4)
    assert not torch.equal(leg(evs), leg([_plant(e) for e in evs]))       # the leak the audit measured
    with pytest.raises(ValueError, match="observation_only"):
        HashedEventEncoder(48, view="legacy", reward_mode="observation_only")
    with pytest.raises(ValueError):
        HashedEventEncoder(48, view="labels")


def test_retriever_memory_keys_and_values_ignore_the_canaries():
    """The Retriever's default event encoder is the actor-view encoder, so segment keys / memory values (actor
    inputs when memory is on) cannot carry evaluator values either."""
    torch.manual_seed(0)
    cfg = tiny_config(mem_segment=8, mem_segment_tokens=2)
    ret = Retriever(cfg)
    assert isinstance(ret.event_encoder, HashedEventEncoder) and ret.event_encoder.view == "actor"
    evs = light_events(8)
    with torch.no_grad():
        x1 = ret.encode_records(evs, target=True)
        x2 = ret.encode_records([_plant(e) for e in evs], target=True)
    assert torch.equal(x1, x2)


def _linked_events(n=30):
    evs = light_events(n, episode_len=10)
    for i, e in enumerate(evs):
        if i % 5 == 3:                              # a delayed outcome of the decision 3 steps earlier
            e.attributed_to = evs[i - 3].event_id
        if i % 7 == 6:                              # a result observation caused by this event's own action
            e.result.observations = [Observation(ObsKind.EVENT, "delivery", value=1.0, caused_by=evs[i - 1].event_id)]
        if i == 20:                                 # an observation caused by an older decision (third party)
            e.observations.append(Observation(ObsKind.NUMERIC, "payout", value=3.0, caused_by=evs[2].event_id))
    return evs


def _expected(evs):
    want = {}
    for e in evs:
        if e.attributed_to:
            want.setdefault(e.attributed_to, []).append((e.event_id, "attributed_to"))
        for o in list(e.observations) + list(e.result.observations if e.result else []):
            if o.caused_by:
                want.setdefault(o.caused_by, []).append((e.event_id, "caused_by"))
    return want


def test_links_are_indexed_both_ways_and_filterable(tmp_path):
    evs = _linked_events()
    want = _expected(evs)
    arc = EventArchive(str(tmp_path / "a"), chunk_rows=8)
    arc.append(evs[:17])                            # two sealed chunks + WAL tail
    arc.append(evs[17:])
    for src, lst in want.items():
        got = [(e, k) for e, _, k in arc.consequences(src)]
        assert sorted(got) == sorted(lst), src
    assert arc.causes(evs[20].event_id) == [(evs[2].event_id, "caused_by")] or \
        (evs[2].event_id, "caused_by") in arc.causes(evs[20].event_id)
    src = evs[0].event_id
    assert arc.count_where(attributed_to=src) == 1 and arc.count_where(caused_by=src) == 0
    assert [r.event_id for b in arc.scan(attributed_to=[src]) for r in b] == [evs[3].event_id]
    assert arc.count_where(linked_to=evs[2].event_id) == len(want[evs[2].event_id])
    assert arc.count_where(attributed_to=[]) == 0
    arc.close()
    # restart: the WAL tail is re-indexed, the links survive; a lost index is rebuilt with them
    arc = EventArchive(str(tmp_path / "a"), chunk_rows=8)
    for src, lst in want.items():
        assert sorted((e, k) for e, _, k in arc.consequences(src)) == sorted(lst)
    arc.close()
    os.remove(str(tmp_path / "a" / "index.sqlite"))
    arc = EventArchive(str(tmp_path / "a"), chunk_rows=8)
    for src, lst in want.items():
        assert sorted((e, k) for e, _, k in arc.consequences(src)) == sorted(lst)
    arc.close()


def test_index_from_before_the_links_table_is_backfilled(tmp_path):
    evs = _linked_events()
    root = str(tmp_path / "a")
    arc = EventArchive(root, chunk_rows=8)
    arc.append(evs)
    arc.close()
    db = sqlite3.connect(os.path.join(root, "index.sqlite"))
    db.execute("DROP TABLE links")
    db.commit()
    db.close()
    arc = EventArchive(root, chunk_rows=8)
    want = _expected(evs)
    for src, lst in want.items():
        assert sorted((e, k) for e, _, k in arc.consequences(src)) == sorted(lst)
    arc.close()


def test_row_links_reads_attribution_and_caused_by_from_row_dicts():
    d = {"attributed_to": "e1", "observations": '[{"caused_by": "e2"}, {"caused_by": null}]',
         "result": '{"observations": [{"caused_by": "e3"}, {"caused_by": "e2"}]}'}
    assert row_links(d) == [("e1", "attributed_to"), ("e2", "caused_by"), ("e3", "caused_by")]
    assert row_links({"attributed_to": None, "observations": "[]", "result": "null"}) == []
