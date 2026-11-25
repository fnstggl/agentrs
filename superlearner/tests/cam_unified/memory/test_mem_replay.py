"""ReplaySampler: stratum proportions and membership, whole-lifetime uniformity (not FIFO), contiguous episode slices,
later tag updates, restart persistence, bounded caches."""
from __future__ import annotations

import collections
import os
import random

import numpy as np
import pytest

from memhelpers import light_events
from superlearner.cam_unified.memory.archive import EventArchive
from superlearner.cam_unified.memory.replay import DEFAULT_MIX, STRATA, ActiveCache, ReplaySampler


def _tagged_archive(root, n=20_000, seed=0):
    """Four families of very different sizes; known failure / skill / innovation / surprise tags."""
    rng = random.Random(seed)
    a = EventArchive(root, chunk_rows=4096)
    fams = ["big"] * 70 + ["mid"] * 20 + ["small"] * 8 + ["tiny"] * 2
    evs = light_events(n, episode_len=50, seed=seed)
    truth = {}
    for e in evs:
        e.extra["family"] = fams[(int(e.event_id[-9:]) // 50) % 100]
        r = rng.random()
        if r < 0.05:
            e.extra["success"] = True
        if 0.05 <= r < 0.07:
            e.extra["innovation"] = True
        e.extra["surprise"] = rng.gauss(0, 1) + (8.0 if rng.random() < 0.01 else 0.0)
        truth[e.event_id] = e
    a.append(evs)
    return a, truth


def _is(stratum, e):
    if stratum == "failures":
        return (not e.result.ok) or e.extra.get("success") is False
    if stratum == "skills":
        return bool(e.extra.get("success")) or bool(e.extra.get("skill"))
    if stratum == "innovations":
        return bool(e.extra.get("innovation"))
    if stratum == "rare":
        return e.extra["surprise"] > 3.0
    return True


def test_strata_proportions_and_membership(tmp_path):
    a, truth = _tagged_archive(str(tmp_path / "a"))
    rs = ReplaySampler(a, recent_window=2000, seed=1)
    assert rs.sync() == 20_000
    sizes = rs.stratum_sizes()
    assert sizes["uniform"] == 20_000 and sizes["recent"] == 2000
    assert sizes["failures"] == sum(1 for e in truth.values() if not e.result.ok)
    assert sizes["skills"] == sum(1 for e in truth.values() if e.extra.get("success"))
    assert sizes["innovations"] == sum(1 for e in truth.values() if e.extra.get("innovation"))
    assert 150 <= sizes["rare"] <= 400        # the ~1% outliers + the far Gaussian tail (prequential z > 2)
    anchors = rs.sample_anchors(20_000)
    by = collections.Counter(s for s, _ in anchors)
    for s in STRATA:
        assert abs(by[s] / 20_000 - DEFAULT_MIX[s]) < 0.015, (s, by[s])
    meta = {m["gseq"]: m for b in a.scan_meta(columns=["gseq", "event_id", "family"]) for m in b}
    for s, g in anchors:
        e = truth[meta[g]["event_id"]]
        if s in ("failures", "skills", "innovations"):
            assert _is(s, e), (s, e.event_id)
        if s == "rare":
            assert e.extra["surprise"] > 1.0                 # prequential z: early statistics are noisy
        if s == "recent":
            assert g >= 18_000
    rare = [truth[meta[g]["event_id"]].extra["surprise"] for s, g in anchors if s == "rare"]
    assert np.mean(np.array(rare) > 4.0) > 0.6                 # dominated by the genuine outliers (+8 shift)
    dom = collections.Counter(meta[g]["family"] for s, g in anchors if s == "domain")
    tot = sum(dom.values())
    assert all(abs(dom[f] / tot - 0.25) < 0.05 for f in ("big", "mid", "small", "tiny")), dom   # balanced domains
    under = collections.Counter(meta[g]["family"] for s, g in anchors if s == "underrepresented")
    assert under["tiny"] > under["small"] > under["mid"] > under["big"], under       # inverse to their size
    uni = collections.Counter(meta[g]["family"] for s, g in anchors if s == "uniform")
    assert under["tiny"] / sum(under.values()) > 10 * uni["tiny"] / sum(uni.values())
    a.close()


def test_uniform_stratum_covers_the_whole_lifetime_not_fifo(tmp_path):
    a = EventArchive(str(tmp_path / "a"), chunk_rows=8192)
    for i in range(0, 60_000, 10_000):
        a.append(light_events(10_000, start=i))
    rs = ReplaySampler(a, mix={"uniform": 0.5, "recent": 0.5}, recent_window=1000, seed=3)
    rs.sync()
    anchors = rs.sample_anchors(20_000)
    uni = np.array([g for s, g in anchors if s == "uniform"])
    rec = np.array([g for s, g in anchors if s == "recent"])
    assert abs((uni < 600).mean() - 0.01) < 0.004            # the oldest 1% is drawn ~1% of the time
    assert abs(np.median(uni) - 30_000) < 1_500 and rec.min() >= 59_000
    a.close()


def test_slices_are_contiguous_episode_windows(tmp_path):
    a = EventArchive(str(tmp_path / "a"), chunk_rows=1000)
    a.append(light_events(1000, episode_len=37) + light_events(200, env="envB", episode_len=5, seed=4))
    rs = ReplaySampler(a, seed=5)
    rs.sync()
    for sl in rs.sample(300, length=16):
        recs = sl.records
        assert len({r.episode_id for r in recs}) == 1 and recs[0].episode_id == sl.episode_id
        steps = [r.step for r in recs]
        assert steps == list(range(steps[0], steps[0] + len(steps)))
        ep_len = 37 if sl.episode_id.startswith("env0") else 5
        assert len(sl) == min(16, ep_len)
        assert sl.anchor_gseq in [m["gseq"] for m in a.meta(sl.event_ids)]
    a.close()


def test_later_tags_enter_strata_and_empty_strata_pass_their_mass(tmp_path):
    a = EventArchive(str(tmp_path / "a"), chunk_rows=1000)
    evs = light_events(500)
    for e in evs:
        e.result.ok = True                                    # no failures yet
    a.append(evs)
    rs = ReplaySampler(a, seed=0)
    rs.sync()
    assert rs.stratum_sizes()["skills"] == 0 and rs.stratum_sizes()["failures"] == 0
    anchors = rs.sample_anchors(1000)
    assert {s for s, _ in anchors} <= {"recent", "uniform", "domain", "underrepresented"}
    assert abs(sum(rs.last_mix.values()) - 1) < 1e-9
    a.tag([e.event_id for e in evs[100:150]], success=True)       # episode outcome known later
    a.tag([e.event_id for e in evs[300:310]], success=False)
    rs.sync()
    assert rs.stratum_sizes()["skills"] == 50 and rs.stratum_sizes()["failures"] == 10
    got = {g for s, g in rs.sample_anchors(3000) if s == "skills"}
    assert got <= set(range(100, 150)) and len(got) > 30
    a.close()


def test_sampler_state_persists_and_restores(tmp_path):
    a, _ = _tagged_archive(str(tmp_path / "a"), n=5000)
    rs = ReplaySampler(a, seed=7)
    rs.sync()
    st = rs.state_dict()
    first = rs.sample_anchors(50)
    sizes = rs.stratum_sizes()
    rs.close()
    re = ReplaySampler(a, seed=123)                           # reopened: membership tables and cursor persisted
    assert re.stratum_sizes() == sizes and re.cursor == 4999 and re.sync() == 0
    re.load_state_dict(st)
    assert re.sample_anchors(50) == first                     # replay cursors / RNG restored exactly
    a.append(light_events(100, start=5000))
    assert re.sync() == 100 and re.stratum_sizes()["uniform"] == 5100
    a.close()


def test_caches_are_bounded(tmp_path):
    a = EventArchive(str(tmp_path / "a"), chunk_rows=2000)
    a.append(light_events(6000))
    rs = ReplaySampler(a, cache_events=500, seed=1)
    rs.sync()
    for _ in range(10):
        rs.sample(20, 16)
        assert len(rs._cache) <= 500
    ac = ActiveCache(rs, capacity=40, length=8)
    ac.refresh()
    assert len(ac.slices) == 40
    ac.refresh(15)
    assert len(ac.slices) == 40
    assert all(len(s) <= 8 for s in ac.batch(64))
    a.close()


def test_episode_ids_reused_across_environments_and_actors_never_mix(tmp_path):
    """Harness-local episode counters ("ep0", "ep1", ...) repeat across environments and actors: every slice stays
    inside one (env, actor, episode) stream, is contiguous in step order and contains its anchor."""
    a = EventArchive(str(tmp_path / "a"), chunk_rows=2000)
    evs, streams = [], {}
    for k, (env, actor) in enumerate([("envA", "a0"), ("envB", "a0"), ("envA", "a1"), ("envC", "a2")]):
        part = light_events(300, env=env, actor=actor, episode_len=100, seed=k)
        for e in part:
            e.event_id = f"{env}-{actor}-{e.event_id}"
            e.episode_id = f"ep{e.step // 1000 + int(e.event_id[-9:]) // 100}"     # "ep0".."ep2" in every stream
            streams.setdefault((env, actor, e.episode_id), []).append(e.step)
        evs += part
    random.Random(0).shuffle(evs)                              # streams interleave in the archive
    a.append(evs)
    rs = ReplaySampler(a, seed=2)
    rs.sync()
    for sl in rs.sample(400, length=24):
        key = {(r.env_id, r.actor_id, r.episode_id) for r in sl.records}
        assert key == {(sl.env_id, sl.actor_id, sl.episode_id)}
        steps = [r.step for r in sl.records]
        assert steps == list(range(steps[0], steps[0] + 24)) and len(sl) == 24
        assert sl.anchor_gseq in [m["gseq"] for m in a.meta(sl.event_ids)]
    assert a.episode_steps("ep1", env_id="envB", actor_id="a0") == sorted(
        a.episode_steps("ep1", env_id="envB", actor_id="a0"))
    assert [s for s, _, _ in a.episode_steps("ep1", env_id="envB", actor_id="a0")] == list(range(100))
    a.close()


def test_draw_cost_does_not_grow_with_episode_length(tmp_path):
    """A lifelong episode without resets: a draw fetches a window around its anchor (indexed), never the episode."""
    import time
    import tracemalloc
    a = EventArchive(str(tmp_path / "a"), chunk_rows=65536, fsync=False)
    for i in range(0, 40_000, 10_000):
        a.append(light_events(10_000, start=i, episode_len=10 ** 9))
    rs = ReplaySampler(a, seed=0)
    rs.sync()
    plan = a._db.execute("EXPLAIN QUERY PLAN SELECT step, event_id, gseq FROM events INDEXED BY ev_stream WHERE "
                         "episode_id=? AND env_id=? AND actor_id=? AND step <= ? AND (step < ? OR gseq < ?) ORDER BY "
                         "step DESC, gseq DESC LIMIT ?", ("x", "y", "z", 5, 5, 5, 3)).fetchall()
    assert any("ev_stream" in str(r) and "step<?" in str(r).replace(" ", "") for r in plan), plan
    rs.sample(4, 32, with_records=False)
    tracemalloc.start()
    t0 = time.time()
    sl = rs.sample(64, 32, with_records=False)
    dt = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(sl) == 64 and all(len(s) == 32 and s.steps == list(range(s.steps[0], s.steps[0] + 32)) for s in sl)
    assert peak < 1_000_000, peak                               # a 40k-row episode list alone is several MB
    assert dt < 2.0, dt
    a.close()


def test_corrected_tags_move_events_between_strata_and_restore_exactly(tmp_path):
    a = EventArchive(str(tmp_path / "a"), chunk_rows=1000)
    evs = light_events(600, episode_len=60)
    for e in evs:
        e.result.ok = True
        e.extra["family"] = "fam_a" if int(e.event_id[-9:]) < 300 else "fam_b"
    a.append(evs)
    rs = ReplaySampler(a, seed=4)
    rs.sync()
    a.tag([e.event_id for e in evs[100:150]], success=True)
    rs.sync()
    assert rs.members("skills") == list(range(100, 150)) and rs.members("failures") == []
    st = rs.state_dict()
    want_draws = rs.sample_anchors(200)
    a.tag([e.event_id for e in evs[120:140]], success=False)    # the outcome was corrected
    a.tag([e.event_id for e in evs[0:10]], family="fam_b")       # a family re-labelled
    rs.sync()
    assert rs.members("skills") == list(range(100, 120)) + list(range(140, 150))
    assert rs.members("failures") == list(range(120, 140))
    assert rs.members("dom:fam_a") == list(range(10, 300)) and rs.members("dom:fam_b")[:10] == list(range(10))
    sizes = rs.stratum_sizes()
    assert sizes["skills"] == 30 and sizes["failures"] == 20 and rs.domains() == {"fam_a": 290, "fam_b": 310}
    got = rs.sample_anchors(3000)
    assert {g for s, g in got if s == "skills"} <= set(range(100, 120)) | set(range(140, 150))
    assert {g for s, g in got if s == "failures"} <= set(range(120, 140))
    assert rs.stats["flips"] == 30
    a.tag([e.event_id for e in evs[120:125]], success=True)     # corrected back: rows are revived, not appended
    rs.sync()
    assert rs.members("skills") == list(range(100, 125)) + list(range(140, 150)) and rs._n["skills"] == 50
    rs.load_state_dict(st)                                      # the checkpoint's exact membership and draws
    assert rs.members("skills") == list(range(100, 150)) and rs.members("failures") == []
    assert rs.domains() == {"fam_a": 300, "fam_b": 300}
    assert rs.sample_anchors(200) == want_draws
    rs.sync()                                                   # later corrections re-enter from the change log
    assert rs.members("failures") == list(range(125, 140)) and rs.domains() == {"fam_a": 290, "fam_b": 310}
    rs.close()
    re = ReplaySampler(a, seed=0)                               # persisted, not re-derived
    assert re.members("failures") == list(range(125, 140)) and re.sync() == 0
    re.close()
    a.close()


def test_archive_index_rebuild_rederives_membership(tmp_path):
    root = str(tmp_path / "a")
    a = EventArchive(root, chunk_rows=200)
    evs = light_events(500)
    a.append(evs)
    a.tag([e.event_id for e in evs[:40]], success=True)
    rs = ReplaySampler(a, seed=1)
    rs.sync()
    before = {s: rs.members(s) for s in ("skills", "failures")}
    rs.close()
    a.close()
    for suf in ("", "-wal", "-shm"):
        if os.path.exists(os.path.join(root, "index.sqlite" + suf)):
            os.unlink(os.path.join(root, "index.sqlite" + suf))
    b = EventArchive(root, chunk_rows=200)
    rs = ReplaySampler(b, seed=1)
    b.tag([evs[45].event_id], success=True)                    # change number 1 again in the rebuilt index
    rs.sync()
    assert rs.stats["rebuilds"] == 1
    assert rs.members("skills") == before["skills"] + [45] and rs.members("failures") == before["failures"]
    rs.close()
    b.close()
