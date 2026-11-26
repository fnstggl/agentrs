"""Blob retention and compaction (crucible.substrate.retention) and the reference-counted process pool of store-less
worlds (crucible.substrate.artifacts): bounded growth while replay, resume and recounts stay exact."""
import gc
import os

import numpy as np
import pytest

import crucible
from crucible.core.errors import InvariantViolation
from crucible.core.kernel import World
from crucible.population import analysis
from crucible.population.worlds import population_world
from crucible.storage.sqlite import SQLiteStore
from crucible.substrate import artifacts as A
from crucible.substrate import retention

crucible.load_all_modules()
H = 3600


def _fresh_reads():
    """Forget every byte this process cached, so reads go to the store (as in a new process)."""
    A._CHUNKS.clear()


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    import torch
    torch.set_num_threads(1)
    db = str(tmp_path_factory.mktemp("ret") / "w.sqlite")
    st = SQLiteStore(db)
    sp = population_world("development", 3, n=300, sites=8, strategic=False,
                          adaptive={"learn_every_s": 3 * H, "checkpoint_every": 1, "min_examples": 16})
    w = World.create(sp, world_id="ret", store=st)
    w.advance_time(9 * H)
    old_snap = w.snapshot()
    w.advance_time(27 * H)
    w.flush()
    return {"world": w, "store": st, "db": db, "old_snap": old_snap,
            "visits": analysis.visit_log(w), "digest": w.digest()}


@pytest.mark.timeout(900)
def test_pack_compresses_blobs_and_every_read_stays_exact(stored):
    w, st = stored["world"], stored["store"]
    before = retention.stats(w)
    out = retention.pack(w, min_bytes=0)
    after = retention.stats(w)
    assert out["packed"] > 100 and out["chunks"] >= 1
    assert out["stored_bytes"] * 2 < out["raw_bytes"]                    # at least 2x on cohort blobs
    assert after["live_blob_bytes"] < before["live_blob_bytes"] - out["raw_bytes"] // 2
    assert after["blobs_packed"] == out["packed"] and after["blobs"] == before["blobs"]
    roots = {r[len(A.REF_PREFIX):] for r in retention.roots(w)}
    assert roots and all(len(st.get_blob(A.blob_key(h))) != A.PTR_LEN for h in roots)   # roots stay unpacked
    _fresh_reads()
    # replay from genesis in this process from the store alone (a new store object, no caches)
    st2 = SQLiteStore(stored["db"])
    rep = World.replay(w.spec, st2.read_events("ret"), world_id="ret-replay", store=st2)
    assert rep.digest() == stored["digest"]
    # recounts that read blobs referenced from events (the population's visit log) are unchanged
    from crucible.core.kernel import Event
    v = analysis.visit_log(rep, [Event.from_dict(d) for d in st2.read_events("ret")])
    assert len(v["ind"]) == len(stored["visits"]["ind"]) > 0
    assert set(v) == set(stored["visits"]) and all(np.array_equal(v[k], stored["visits"][k]) for k in v)
    # resume (latest snapshot + replay) works on the packed store
    res = World.load(st2, "ret")
    assert res.digest() == stored["digest"]
    assert retention.pack(w, min_bytes=0)["packed"] == 0                   # idempotent


def test_a_damaged_chunk_makes_its_blobs_unavailable_never_wrong(tmp_path):
    from crucible.core.kernel import WorldSpec
    st = SQLiteStore(str(tmp_path / "d.sqlite"))
    w = World.create(WorldSpec("dmg", 1, [{"name": m, "config": {}} for m in ("identity", "bank", "admin",
                                                                              "artifacts")]), store=st)
    refs = []

    def put(ctx):
        for i in range(40):
            refs.append(A.put_array_blob(ctx, {"x": np.arange(i, i + 500, dtype=np.int32)}))
    w._run_system("test", put)
    assert retention.pack(w, min_bytes=0)["packed"] == 40
    (key,) = list(st.blob_sizes("pack/"))
    comp = st.get_blob(key)
    st.put_blob(key, comp[:-10] + bytes(10))          # damage the chunk
    _fresh_reads()
    w.runtime.pop("artifacts", None)
    for ref in refs[:3]:
        with pytest.raises(Exception) as ei:
            A.get_array_blob(w, ref)
        assert "not available" in str(ei.value)
    st.put_blob(key, comp)
    _fresh_reads()
    assert np.array_equal(A.get_array_blob(w, refs[5])["x"], np.arange(5, 505, dtype=np.int32))


@pytest.mark.timeout(900)
def test_collect_drops_only_superseded_checkpoints_no_retained_snapshot_needs(stored):
    w, st = stored["world"], stored["store"]
    sup = retention.superseded(w)
    assert sup, "the adaptive population must have replaced a checkpoint for this test to mean anything"
    snap_roots = retention.snapshot_roots(st, stored["old_snap"])
    assert snap_roots and snap_roots & sup              # the old snapshot needs a now-superseded checkpoint
    first = retention.collect(w)
    assert first["kept_roots"] >= 1
    for r in snap_roots:                                  # ...so it is kept: resuming from that snapshot still works
        assert st.get_blob(A.blob_key(r[len(A.REF_PREFIX):])) is not None
    # once only newer snapshots are retained, the superseded checkpoints go
    w.snapshot()
    pr = retention.prune_snapshots(w, keep_last=1, keep_every_s=None)
    assert pr["deleted"] >= 1 and len(st.list_snapshots("ret")) == 1
    out = retention.collect(w)
    dropped = sup - retention.roots(w) - set().union(*[retention.snapshot_roots(st, s["id"])
                                                       for s in st.list_snapshots("ret")])
    assert out["dropped"] + first["dropped"] >= len([r for r in dropped if True]) > 0
    for r in dropped:
        assert st.get_blob(A.blob_key(r[len(A.REF_PREFIX):])) is None
    _fresh_reads()
    st2 = SQLiteStore(stored["db"])
    assert World.load(st2, "ret").digest() == w.digest()                 # resume is exact after collection
    # never anything a cols.batch applies or the current state roots
    for r in retention.roots(w):
        assert st.get_blob(A.blob_key(r[len(A.REF_PREFIX):])) is not None


def test_collect_keeps_everything_when_the_store_holds_another_world(tmp_path):
    import torch
    torch.set_num_threads(1)
    st = SQLiteStore(str(tmp_path / "f.sqlite"))
    sp = population_world("development", 5, n=200, sites=4, strategic=False,
                          adaptive={"learn_every_s": 2 * H, "checkpoint_every": 1, "min_examples": 8})
    w = World.create(sp, world_id="parent", store=st)
    w.advance_time(4 * H)
    f = w.fork("parent-fork", store=st)
    w.advance_time(14 * H)
    out = retention.collect(w)
    assert out["dropped"] == 0 and "other world" in out.get("reason", "")
    assert f.world_id in [x["id"] for x in st.list_worlds()]


# ---------------------------------------------------------------------------------------------- process pool
def _storeless(seed):
    from crucible.core.kernel import WorldSpec
    w = World.create(WorldSpec("pool", seed, [{"name": m, "config": {}} for m in ("identity", "bank", "admin",
                                                                                  "artifacts")]))
    refs = []

    def put(ctx):
        refs.append(A.put_array_blob(ctx, {"x": np.full(1000, seed, np.int64)}))
    w._run_system("test", put)
    return w, refs[0]


def test_storeless_worlds_release_their_pool_bytes_when_closed_or_collected():
    base = A.pool_stats()["blobs"]
    w1, r1 = _storeless(101)
    w2, r2 = _storeless(102)
    h1, h2 = r1[len(A.REF_PREFIX):], r2[len(A.REF_PREFIX):]
    assert h1 in A._POOL and h2 in A._POOL
    fork = w1.fork("pool-fork")                                  # a fork holds its own lease
    w1.close()
    assert h1 in A._POOL and np.array_equal(A.get_array_blob(fork, r1)["x"], np.full(1000, 101, np.int64))
    fork.close()
    assert h1 not in A._POOL                                     # no live world holds it: evicted
    del w2
    gc.collect()                                                  # never closed: released by garbage collection
    assert h2 not in A._POOL
    assert A.pool_stats()["blobs"] <= base


def test_a_world_cannot_reach_another_worlds_pool_bytes():
    w1, r1 = _storeless(201)
    w2, _ = _storeless(202)
    assert r1[len(A.REF_PREFIX):] in A._POOL
    with pytest.raises(Exception):
        A.get_array_blob(w2, r1)                                 # w2's state does not record that REF
    w1.close()
    w2.close()
