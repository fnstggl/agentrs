"""Blob retention and compaction (crucible.substrate.retention) and the reference-counted process pool of store-less
worlds (crucible.substrate.artifacts): bounded growth while replay, resume and recounts stay exact."""
import gc
import os

import numpy as np
import pytest

import crucible
from crucible.core.errors import InvariantViolation
from crucible.core.kernel import Module, World, register_module
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
        ctx.emit("note", {"what": "test blobs", "refs": list(refs)}, vis="internal")   # the record names them (D-78)
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


@register_module
class _SysBlobTable(Module):
    """A column table for test_system_array_blobs_are_not_world_state."""
    name = "sysblob_test"
    depends = ("cols",)

    def install(self, world):
        from crucible.substrate import columns as C
        C.declare(world, "tbl", 1000, {"x": ("int64", (), 0)})


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


# ---------------------------------------------------------------------------------------------- cost per compaction
@pytest.mark.timeout(900)
def test_periodic_compaction_reads_only_what_was_written_since_the_last_one(tmp_path, monkeypatch):
    """At day 365 of a stored population world one weekly compaction took 19.3 s: collect re-read every cols.batch
    event of the log, every retained snapshot was unpickled for its roots and time, and pack and collect each listed
    the size of every blob in the store (222k keys). Now (D-78) each compaction reads only the events written since
    the previous one, snapshot metadata once per process, and the sizes of exactly the blobs it considers - with the
    same result; and the position survives in the store, so a new process does not re-read the log."""
    import torch
    torch.set_num_threads(1)
    st = SQLiteStore(str(tmp_path / "inc.sqlite"))
    sp = population_world("development", 4, n=200, sites=4, strategic=False,
                          adaptive={"learn_every_s": 2 * H, "checkpoint_every": 1, "min_examples": 8})
    w = World.create(sp, world_id="inc", store=st)
    windows, unpickled, listed = [], [], []
    orig_read, orig_get, orig_sizes = st.read_events, st.get_snapshot, st.blob_sizes

    def counting_read(wid, from_seq=0, to_seq=None, types=None):
        out = orig_read(wid, from_seq, to_seq, types=types)
        if to_seq is not None and to_seq - from_seq > 1:          # the scan's windows (1-event reads verify)
            windows.append((from_seq, to_seq, len(out)))
        return out

    def counting_get(sid):
        unpickled.append(sid)
        return orig_get(sid)

    def counting_sizes(prefix=""):
        listed.append(prefix)
        return orig_sizes(prefix)
    monkeypatch.setattr(st, "read_events", counting_read)
    monkeypatch.setattr(st, "get_snapshot", counting_get)
    monkeypatch.setattr(st, "blob_sizes", counting_sizes)
    outs = []
    for day in range(1, 4):
        w.advance_time(day * 8 * H)
        w.snapshot()
        outs.append(retention.compact(w, prune=True, min_bytes=0))
    # one window per compaction, each starting where the previous one stopped: every event read exactly once
    assert len(windows) == 3 and windows[0][0] == 0
    assert all(windows[k + 1][0] == windows[k][1] for k in range(2))
    assert sum(n for _, _, n in windows) == w.seq
    # no listing of every blob: only full keys (indexed lookups of the blobs considered)
    assert listed and all(len(p) == len("art/") + 64 for p in listed), sorted(set(p[:8] for p in listed))
    assert sum(o["pack"]["packed"] for o in outs) > 50
    every = orig_read("inc", 0, None, types=["cols.batch"])
    assert w.runtime["retention.scan"]["batch"] == {e["payload"]["ref"] for e in every}
    # every snapshot was unpickled at most once for its metadata, however many compactions looked at it
    assert len(unpickled) == len(set(unpickled)) <= 3
    # a new process continues from the position kept in the store: no window starts at 0 again
    retention._SNAP_META.clear()
    w.runtime.pop("retention.scan")
    windows.clear()
    w.advance_time(w.now + 4 * H)
    w.flush()
    sc = retention._scan(w, st)
    assert windows and windows[0][0] > 0 and sc["batch"] == {e["payload"]["ref"] for e in
                                                             orig_read("inc", 0, None, types=["cols.batch"])}
    # ...and agrees with a cold computation from the log (no persisted position): the same candidates and batch REFs
    w.runtime["retention.scan"] = None
    st.delete_blob(retention._scan_key(w))
    cold = retention._scan(w, st)
    assert cold["batch"] == sc["batch"] and set(cold["pending"]) >= set(sc["pending"])
    w.flush()
    assert World.load(SQLiteStore(str(tmp_path / "inc.sqlite")), "inc").digest() == w.digest()


def test_system_array_blobs_are_not_world_state(tmp_path):
    """D-78: a cohort decision's blob is named by its hash-chained cols.batch event and kept in the store; the state
    gets no metadata row and the log no artifact.stored event for it. So a snapshot's size does not grow with the
    number of recorded batches (at day 365 of a population world 222,702 such rows made each snapshot hold the
    whole blob history), while replay, resume and recounts stay exact."""
    from crucible.core.kernel import WorldSpec
    from crucible.substrate import columns as C
    mods = [{"name": m, "config": {}} for m in ("identity", "bank", "admin", "artifacts", "cols", "sysblob_test")]
    st = SQLiteStore(str(tmp_path / "sb.sqlite"))
    w = World.create(WorldSpec("sysblobs", 3, mods), world_id="sb", store=st)
    sizes = []
    refs = []

    def batches(k):
        def run(ctx):
            rng = ctx.rng("b", k)
            for j in range(50):
                idx = np.sort(rng.choice(1000, 200, replace=False))
                refs.append(C.batch(ctx, "tbl", "test.add", idx, {"x": rng.integers(0, 9, 200)}, mode="add"))
        return run
    for k in range(4):
        w._run_system(f"batch{k}", batches(k))
        sizes.append(len(w.snapshot_blob()))
    assert w.state.count("artifacts") == 0
    assert not [e for e in w.log if e.type == "artifact.stored"]
    assert len(set(refs)) == 200 and all(st.get_blob(A.blob_key(r[7:])) is not None for r in refs)
    # 50 to 200 batches: the snapshot holds the columns, not the blob history
    assert max(sizes) - min(sizes) < 2000, sizes
    w.flush()
    truth = C.col(w, "tbl", "x").copy()
    _fresh_reads()
    st2 = SQLiteStore(str(tmp_path / "sb.sqlite"))
    rep = World.replay(w.spec, st2.read_events("sb"), world_id="sb-replay", store=st2)
    assert rep.digest() == w.digest() and np.array_equal(C.col(rep, "tbl", "x"), truth)
    w.snapshot()
    assert World.load(st2, "sb").digest() == w.digest()
    # a store-less world of another lineage cannot reach these blobs; a replay of this lineage without a store can
    other = World.create(WorldSpec("sysblobs", 4, mods), world_id="other")
    with pytest.raises(Exception):
        A.get_array_blob(other, refs[0])
    rep2 = World.replay(w.spec, [e.to_dict() for e in w.log], world_id="sb-replay2")
    assert np.array_equal(C.col(rep2, "tbl", "x"), truth)
    # participant tools never see a system blob
    w.system_call("admin.join", {"handle": "eve", "endowment": 0})
    r = w.execute("p:eve", {"tool": "artifact.info", "args": {"ref": refs[0]}})
    assert not r.ok and r.error["code"] == "not_found"
    other.close()


def test_a_year_of_cohort_batches_leaves_the_snapshot_size_unchanged(tmp_path):
    """Wave 3b (D-78, A.json store growth): 365 simulated days of a kernel world writing four cohort batches a day
    (one system array blob each), a snapshot and `retention.compact(prune=True)` every week. Before D-78 every system
    blob kept an `artifacts` row, so each snapshot carried the whole blob history (a 365-day population world: 1.4 MB
    per snapshot in the first quarter, 3.3 MB in the last). Now the snapshot holds the world's state only - its size
    stays that of the first week while the blob count grows - and replay from genesis and resume stay exact."""
    from crucible.substrate import bench
    r = bench.store_growth("kernel", 365, db=str(tmp_path / "g.db"), n=20_000, rows=2_000, batches_per_day=4)
    rows = r["rows"]
    assert r["done"] and len(rows) == 53 and rows[-1]["day"] == 365
    assert rows[-1]["blobs"] >= 4 * 365                      # every batch wrote a system blob into the store
    assert all(x["artifact_rows"] == 0 for x in rows)         # none of them is a row of the world's state
    first = rows[0]["snapshot_bytes_latest"]
    assert max(x["snapshot_bytes_latest"] for x in rows) <= 1.05 * first
    assert min(x["snapshot_bytes_latest"] for x in rows) >= 0.95 * first
    # retained snapshot bytes grow with the number of retained snapshots only (pruning keeps the last three and one
    # per 30 days), never with the history each one carries
    assert all(x["snapshot_bytes_retained"] <= 1.05 * first * x["snapshots"] for x in rows)
    assert rows[-1]["blobs_packed"] > 0                       # compaction packed the batch blobs
    assert r["end"]["replay_from_genesis_exact"] and r["end"]["resume_exact"]


# ---------------------------------------------------------------------------------------------- blob lineage (D-78)
@register_module
class _LineageAnnotator(Module):
    """Writes a cohort batch at genesis and then annotates a nested value of its own config, as the industrial genesis
    of a civilisation does (`crucible.worlds.open_industry` records each chain entry's `_out`/`_mach`)."""
    name = "lineage_annotator"
    depends = ("cols",)

    def install(self, world):
        from crucible.substrate import columns as C
        C.declare(world, "lin_tbl", 100, {"x": ("int64", (), 0)})

    def genesis(self, ctx):
        from crucible.substrate import columns as C
        C.batch(ctx, "lin_tbl", "lin.init", np.arange(10), {"x": np.arange(10)}, mode="set")
        self.config["chain"][0]["_out"] = "annotated at genesis"


def _annotating_spec(seed):
    from crucible.core.kernel import WorldSpec
    return WorldSpec("lineage", seed, [{"name": m, "config": ({"chain": [{"a": 1}]} if m == "lineage_annotator" else {})}
                                       for m in ("identity", "bank", "admin", "artifacts", "cols", "lineage_annotator")])


def test_a_module_annotating_its_config_at_genesis_keeps_the_worlds_blob_lineage():
    """Verifier finding (wave-3b repair, D-78): `lineage()` cached `spec.hash()` at the first system-blob write while a
    module's nested config was the spec's own object, so a module annotating it at genesis moved the spec hash under
    the world: every store-less replay, fork and history query computed another lineage and could not read the world's
    own cohort blobs (all 11 civilisation profiles: replay raised InvariantViolation, a fork NotFound). A module now
    owns a detached copy of its config, and the lineage is the hash of the spec genesis ran from (`World.genesis_spec`,
    recorded by the genesis note, carried by snapshots and replays)."""
    from crucible.substrate import columns as C
    from crucible.substrate import history
    sp = _annotating_spec(7)
    h0 = sp.hash()
    w = World.create(sp, world_id="lin")
    assert w.modules["lineage_annotator"].config["chain"][0]["_out"] == "annotated at genesis"
    assert sp.hash() == h0 and w.spec.hash() == h0 and "_out" not in sp.modules[-1]["config"]["chain"][0]
    genesis = w.log[0]
    assert genesis.payload["what"] == "genesis" and genesis.payload["spec"] == h0 == A.lineage(w) == w.genesis_spec
    truth = C.col(w, "lin_tbl", "x").copy()
    rep = World.replay(w.spec, [e.to_dict() for e in w.log], world_id="lin-replay")
    assert rep.digest() == w.digest() and np.array_equal(C.col(rep, "lin_tbl", "x"), truth)
    f = w.fork("lin-fork")
    f.advance_time(f.now + 60)
    assert np.array_equal(C.col(f, "lin_tbl", "x"), truth) and A.lineage(f) == h0
    past = history.state_at(w, seq=int(w.seq))
    assert past.digest() == w.digest()
    for x in (rep, f, past):
        x.close()
    w.close()


def test_the_blob_lineage_is_the_spec_genesis_ran_from_not_the_spec_objects_later_state():
    """Even if code outside the world changes the spec object after genesis, the world's lineage stays the hash its
    genesis note records; a snapshot carries it, so a fork, a resume of that snapshot and a replay of the log (which
    takes it from the genesis note) read the world's system blobs."""
    from crucible.substrate import columns as C
    sp = _annotating_spec(8)
    w = World.create(sp, world_id="lin2")
    h0 = w.genesis_spec
    truth = C.col(w, "lin_tbl", "x").copy()
    w.spec.params["changed_after_genesis"] = True             # the object hashes differently now
    assert w.spec.hash() != h0 and A.lineage(w) == h0
    blob = w.snapshot_blob()
    for x in (World.from_blob(blob, world_id="lin2-resume"), w.fork("lin2-fork"),
              World.replay(w.spec, [e.to_dict() for e in w.log], world_id="lin2-replay")):
        assert x.genesis_spec == h0 == A.lineage(x)
        assert np.array_equal(C.col(x, "lin_tbl", "x"), truth)
        x.close()
    w.close()


@pytest.mark.timeout(600)
def test_a_storeless_civilisation_replays_forks_and_answers_history_queries():
    """The verifier's probes p05/p07/p09 (wave-3b repair): right after genesis a store-less civilisation (whose
    industrial genesis annotates its chain config) must replay from its own log to its state, fork and keep running,
    and answer `history.state_at`; on the D-78 head before the repair all three raised."""
    from crucible.civ.generator import generate
    from crucible.substrate import history, integrity
    import torch
    torch.set_num_threads(1)
    w = generate("balanced", 0, "development")
    assert w.spec.hash() == w.genesis_spec == w.log[0].payload["spec"]
    assert any(e.type == "cols.batch" for e in w.log)
    assert integrity.replay_check(w)["replay_exact"] is True
    f = w.fork("civ-fork")
    f.advance_time(f.now + 600)
    assert f.seq > w.seq
    past = history.state_at(w, seq=int(w.seq) // 2)
    assert past.seq == int(w.seq) // 2
    for x in (f, past):
        x.close()
    w.close()
