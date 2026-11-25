"""Consolidator: raw events -> segments -> longer representations with source links, artifacts and weight links;
restart without loss or duplication; version migration recomputed from raw events."""
from __future__ import annotations

import pytest
import torch

from memhelpers import light_events
from superlearner.cam_unified.memory.archive import EventArchive
from superlearner.cam_unified.memory.artifacts import ArtifactStore
from superlearner.cam_unified.memory.consolidate import Consolidator
from superlearner.cam_unified.memory.index import MemoryIndex
from superlearner.cam_unified.memory.retrieval import Retriever
from superlearner.cam_unified.testing import tiny_config


def _setup(root, seg=8, fanout=4, artifacts=False):
    cfg = tiny_config(mem_segment=seg, mem_segment_tokens=2)
    art = ArtifactStore(str(root / "art")) if artifacts else None
    arc = EventArchive(str(root / "a"), chunk_rows=256, artifacts=art, inline_limit=64)
    idx = MemoryIndex(str(root / "i"), cfg.mem_key_dim, 2, cfg.d_model, device="cpu")
    torch.manual_seed(0)
    ret = Retriever(cfg, idx, fanout=fanout)
    return cfg, arc, idx, ret, Consolidator(arc, idx, ret)


def _all_segments(idx, level):
    q = "SELECT seg_id FROM segments WHERE level=? ORDER BY gseq_start"
    return idx.segments([r[0] for r in idx._db.execute(q, (level,))])


def test_hierarchy_with_source_links(tmp_path):
    cfg, arc, idx, ret, con = _setup(tmp_path, artifacts=True)
    evs = light_events(203, env="envA", episode_len=50) + light_events(77, env="envB", episode_len=40, seed=1)
    evs[5].observations[0].content = "exact original document " * 20       # offloaded to the artifact store
    arc.append(evs)
    out = con.step()
    assert out["events"] == 280
    l1 = _all_segments(idx, 1)
    by_ep = {}
    for s in l1:
        assert 1 <= s["n_events"] <= 8 and len(s["event_ids"]) == s["n_events"]
        by_ep.setdefault(s["episode_id"], []).extend(s["event_ids"])
    want = {}
    for e in evs:
        want.setdefault(e.episode_id, []).append(e.event_id)
    open_ids = {k.split("|")[2]: [x[0] for x in v] for k, v in con.state["open1"].items()}
    assert sorted(open_ids) == ["envA-ep4", "envB-ep1"]    # unfinished episodes keep a partial segment open
    assert [len(v) for _, v in sorted(open_ids.items())] == [3, 5]
    for ep, ids in want.items():                            # every event, in order, exactly once
        assert by_ep.get(ep, []) + open_ids.get(ep, []) == ids, ep
    seg5 = next(s for s in l1 if evs[5].event_id in s["event_ids"])
    assert len(seg5["artifact_refs"]) == 1 and arc.artifacts.get_text(seg5["artifact_refs"][0]).startswith("exact")
    assert set(seg5["entities"]) >= {evs[5].observations[0].entity}
    l2 = _all_segments(idx, 2)
    assert l2 and all(len(s["child_ids"]) == 4 for s in l2)
    for s in l2:
        kids = idx.segments(s["child_ids"])
        assert {k["env_id"] for k in kids} == {s["env_id"]}
        assert idx.source_events(s["seg_id"]) == [e for k in kids for e in k["event_ids"]]
        assert s["t_start"] == min(k["t_start"] for k in kids) and s["n_events"] == sum(k["n_events"] for k in kids)
    v = ret.key_version
    assert idx.count(v) == len(l1) + len(l2)
    con.step(close_all=True)
    assert not con.state["open1"] and not con.state["open2"]
    assert sum(s["n_events"] for s in _all_segments(idx, 1)) == 280
    rec = con.source_records(l1[0]["seg_id"])
    assert [r.event_id for r in rec] == l1[0]["event_ids"]
    arc.close()


def test_resume_after_restart_without_loss_or_duplicates(tmp_path):
    evs = light_events(500, episode_len=64)
    # single pass reference
    _, arc1, idx1, _, con1 = _setup(tmp_path / "ref")
    arc1.append(evs)
    con1.step()
    ref = [(s["seg_id"], tuple(s["event_ids"])) for s in _all_segments(idx1, 1)]
    ref2 = [(s["seg_id"], tuple(s["child_ids"])) for s in _all_segments(idx1, 2)]
    arc1.close()
    # interrupted: half the events, close everything, reopen, the rest
    root = tmp_path / "run"
    _, arc, idx, ret, con = _setup(root)
    arc.append(evs[:230])
    con.step(max_events=170)
    con.step()
    idx.close()
    arc.close()
    _, arc, idx, ret, con = _setup(root)                     # fresh objects on the same directories
    assert con.state["cursor"] == 229
    before_crash = con.state_dict()
    arc.append(evs[230:])
    con.step()
    got = [(s["seg_id"], tuple(s["event_ids"])) for s in _all_segments(idx, 1)]
    assert got == ref and [(s["seg_id"], tuple(s["child_ids"])) for s in _all_segments(idx, 2)] == ref2
    # crash after the segments were indexed but before the cursor was stored: the older (consistent) state replays
    con.load_state_dict(before_crash)
    con.step()
    assert idx.count() == len(ref) + len(ref2)             # deterministic ids: nothing duplicated
    assert con.state["cursor"] == 499
    assert len({s for s, _ in got}) == len(got)
    arc.close()


def test_version_migration_recomputed_from_raw_events(tmp_path):
    cfg, arc, idx, ret, con = _setup(tmp_path)
    arc.append(light_events(400, episode_len=40))
    con.step()
    con.ensure_current()
    v0 = ret.key_version
    assert idx.current_version == v0
    q = torch.nn.functional.normalize(torch.randn(2, cfg.mem_key_dim), dim=-1)
    with torch.no_grad():                                     # online encoders change (training)
        for p in list(ret.segment_encoder.parameters()) + list(ret.event_encoder.parameters()):
            p.add_(0.05 * torch.randn_like(p))
    v1 = ret.sync_target()
    _, ids_old, _ = ret.retrieve(q, 3)                        # still served by the complete previous version
    assert idx.current_version == v0 and len(ids_old[0]) == 3
    r = con.reencode_step(max_items=10)
    assert r["reencoded"] == 10 and not r["promoted"] and 0 < r["coverage"] < 1
    r = con.reencode_step(max_items=None)
    assert r["promoted"] and r["coverage"] == 1.0 and r["dropped"] == [v0]
    assert idx.versions() == {v1: idx.count()} and idx.current_version == v1
    # the migrated keys are exactly what the new target encoders compute from the raw events
    l1 = _all_segments(idx, 1)[:6]
    groups = [arc.get(s["event_ids"]) for s in l1]
    k_ref, m_ref = con._encode1(groups, l1, target=True)
    k_idx = idx.keys(v1, idx.kids_of(v1, [s["seg_id"] for s in l1]))
    torch.testing.assert_close(k_idx, k_ref.half().float(), atol=2e-3, rtol=0)
    l2 = _all_segments(idx, 2)[:2]
    k2_ref, _ = con._encode2(l2, v1, target=True)
    torch.testing.assert_close(idx.keys(v1, idx.kids_of(v1, [s["seg_id"] for s in l2])), k2_ref.half().float(),
                               atol=2e-3, rtol=0)
    _, ids_new, _ = ret.retrieve(q, 3)
    assert len(ids_new[0]) == 3
    arc.close()


def test_weight_links_record_parametric_consolidation(tmp_path):
    _, arc, idx, ret, con = _setup(tmp_path)
    arc.append(light_events(64, episode_len=64))
    con.step()
    seg = _all_segments(idx, 1)[0]
    con.link_weights([seg["seg_id"]], "update-42", model_version=9)
    assert idx.weight_links(seg["seg_id"]) == [{"update_id": "update-42", "model_version": 9}]
    rec = arc.get(seg["event_ids"][:1])[0]
    assert rec.training_provenance["used_in_updates"] == ["update-42"]
    arc.close()


def test_invalid_hierarchy_shapes_are_rejected(tmp_path):
    cfg, arc, idx, ret, _ = _setup(tmp_path)
    with pytest.raises(ValueError):
        Consolidator(arc, idx, ret, fanout=64)               # longer than the long encoder can attend
    with pytest.raises(ValueError):
        Consolidator(arc, idx, ret, segment_len=10_000)
    arc.close()


def _perturb_and_sync(ret, scale=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in list(ret.segment_encoder.parameters()) + list(ret.long_encoder.parameters()) + \
                list(ret.event_encoder.parameters()):
            p.add_(scale * torch.randn(p.shape, generator=g))
    return ret.sync_target()


def test_step_during_a_migration_builds_every_group_and_interleaves_with_reencode(tmp_path):
    """step() between sync_target() and promotion: level-2 groups that straddle the sync are built (children
    re-encoded from raw events on demand), nothing is lost or duplicated, retrieval keeps being served by the old
    version, and the promoted version equals a from-scratch encoding of everything with the new encoders."""
    cfg, arc, idx, ret, con = _setup(tmp_path, seg=8, fanout=4)
    evs = light_events(152, episode_len=10_000)
    arc.append(evs[:48])                                       # 6 level-1 -> one full group + 2 open children
    con.step()
    con.ensure_current()
    v0 = ret.key_version
    assert [len(g) for g in con.state["open2"].values()] == [2]
    v1 = _perturb_and_sync(ret)
    assert v1 != v0 and ret.migration_pending
    q = torch.nn.functional.normalize(torch.randn(3, cfg.mem_key_dim), dim=-1)
    arc.append(evs[48:88])
    r = con.step()                                             # completes the straddling group in v1
    assert r["segments2"] == 1 and r["children_reencoded"] == 2 and r["migration_pending"]
    assert idx.get_state("consolidator") == con.state           # persisted == in memory
    _, ids, _ = ret.retrieve(q, 4)                              # still served (by v0)
    assert idx.current_version == v0 and len(ids[0]) == 4
    assert ret.sync_target() == v1 and ret.sync_info["reason"] == "unchanged"
    with torch.no_grad():                                       # a second sync before promotion is held back ...
        next(ret.segment_encoder.parameters()).add_(0.5)
    assert ret.sync_target() == v1 and ret.sync_info["reason"] == "migration-pending"
    assert len(ret.retrieve(q, 4)[1][0]) == 4                   # ... so the served version stays readable
    with torch.no_grad():
        next(ret.segment_encoder.parameters()).sub_(0.5)
    part = con.reencode_step(max_items=3)
    assert part["reencoded"] == 3 and not part["promoted"]
    arc.append(evs[88:])
    r = con.step()
    assert r["segments1"] == 8 and r["segments2"] == 2
    done = con.reencode_step(max_items=None)
    assert done["promoted"] and done["dropped"] == [v0] and idx.current_version == v1
    con.step(close_all=True)
    l1, l2 = _all_segments(idx, 1), _all_segments(idx, 2)
    ids1 = [e for s in l1 for e in s["event_ids"]]
    assert ids1 == [f"env0-0-{i:09d}" for i in range(152)]     # every event exactly once, in order
    assert [len(s["child_ids"]) for s in l2] == [4] * 4 + [3]   # 19 level-1 segments -> groups of 4 (+ the rest)
    assert [c for s in l2 for c in s["child_ids"]] == [s["seg_id"] for s in l1]
    assert idx.coverage(v1) == 1.0 and idx.versions() == {v1: len(l1) + len(l2)}
    # reference: the same events encoded from scratch with the v1 encoders
    _, arc_r, idx_r, _, _ = _setup(tmp_path / "ref", seg=8, fanout=4)
    arc_r.append(evs)
    con_r = Consolidator(arc_r, idx_r, ret)
    con_r.step(close_all=True)
    for lv in (1, 2):
        segs = [s["seg_id"] for s in _all_segments(idx_r, lv)]
        assert segs == [s["seg_id"] for s in _all_segments(idx, lv)]
        torch.testing.assert_close(idx.keys(v1, idx.kids_of(v1, segs)), idx_r.keys(v1, idx_r.kids_of(v1, segs)),
                                   atol=2e-3, rtol=0)
    arc.close()
    arc_r.close()


def test_a_failed_step_changes_nothing_and_the_retry_matches_a_clean_run(tmp_path, monkeypatch):
    evs = light_events(300, episode_len=70)
    _, arc_r, idx_r, _, con_r = _setup(tmp_path / "ref")
    arc_r.append(evs)
    con_r.step()
    ref = [(s["seg_id"], tuple(s["event_ids"]), tuple(s["child_ids"])) for lv in (1, 2) for s in _all_segments(idx_r, lv)]
    cfg, arc, idx, ret, con = _setup(tmp_path / "run")
    arc.append(evs[:100])
    con.step()
    before_state, before_count = con.state_dict(), idx.count()
    before_keys = idx.count(ret.key_version)
    arc.append(evs[100:])
    calls = {"n": 0}
    real = Consolidator._encode2

    def boom(self, metas, version, target=True):           # fails after level-1 segments were already added
        calls["n"] += 1
        raise RuntimeError("simulated failure while encoding level 2")
    monkeypatch.setattr(Consolidator, "_encode2", boom)
    with pytest.raises(RuntimeError, match="simulated"):
        con.step()
    assert calls["n"] == 1
    assert con.state == before_state and idx.get_state("consolidator") == before_state   # cursor did not move
    assert idx.count() == before_count and idx.count(ret.key_version) == before_keys      # no partial segments
    monkeypatch.setattr(Consolidator, "_encode2", real)
    con.step()
    got = [(s["seg_id"], tuple(s["event_ids"]), tuple(s["child_ids"])) for lv in (1, 2) for s in _all_segments(idx, lv)]
    assert got == ref
    k = idx.kids_of(ret.key_version, [g[0] for g in got])
    assert (k >= 0).all() and sorted(k.tolist()) == list(range(len(got)))   # dense kids: the rollback left no gap
    idx.close()
    re = MemoryIndex(str(tmp_path / "run" / "i"), cfg.mem_key_dim, 2, cfg.d_model, device="cpu")
    assert re.count(ret.key_version) == len(got)
    re.close()
    arc.close()
    arc_r.close()
