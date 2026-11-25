"""EventArchive: lifetime ingestion without cutoff, durability, recovery, explicit storage limits, streaming scans."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from memhelpers import light_events
from superlearner.cam_unified.interfaces import EventRecord, Provenance
from superlearner.cam_unified.memory.archive import EventArchive
from superlearner.cam_unified.memory.artifacts import ArtifactStore
from superlearner.cam_unified.memory.errors import ArchiveLocked, StorageLimitExceeded
from superlearner.cam_unified.testing import sample_events

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "src"))


def _canon(r: EventRecord) -> dict:
    d = r.to_dict()
    d.pop("training_provenance")
    return json.loads(json.dumps(d))


def _run(script: str, *args) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([SRC, HERE]), OMP_NUM_THREADS="2")
    return subprocess.run([sys.executable, "-c", textwrap.dedent(script), *args], env=env, capture_output=True,
                          text=True, timeout=600)


def test_more_than_256_events_roundtrip_exactly(tmp_path):
    a = EventArchive(str(tmp_path / "a"), chunk_rows=128)
    evs = []
    for k in range(3):
        evs += sample_events(100, env_id=f"env{k}", seed=k)
    a.append(evs)
    assert a.count() == 300 and a.info()["chunks"] == 2 and a.info()["wal_rows"] == 44
    got = a.get([e.event_id for e in evs])
    assert [_canon(g) for g in got] == [_canon(e) for e in evs]
    scanned = [r.event_id for b in a.scan(batch_size=37) for r in b]
    assert scanned == [e.event_id for e in evs]
    a.close()


def test_more_than_10k_events_without_any_cutoff(tmp_path):
    root = str(tmp_path / "a")
    a = EventArchive(root, chunk_rows=1000)
    evs = light_events(12_000)
    for i in range(0, len(evs), 700):
        a.append(evs[i:i + 700])
    a.flush()
    assert a.count() == 12_000 and a.info()["chunks"] == 12
    assert a.get([evs[-1].event_id])[0].event_id == evs[-1].event_id
    n, last = 0, -1
    for b in a.scan(batch_size=1024):
        assert len(b) <= 1024
        n += len(b)
        last = b[-1].step
    assert n == 12_000 and last == evs[-1].step
    a.close()
    b = EventArchive(root, chunk_rows=1000)            # ingestion continues across restarts
    b.append(light_events(3_500, start=12_000))
    assert b.count() == 15_500 and b.count_where(gseq_min=12_000) == 3_500
    assert b.get(["env0-0-000015499"])[0].step == 15_499 % 100
    b.close()


def test_rejects_non_real_provenance_all_or_nothing(tmp_path):
    a = EventArchive(str(tmp_path / "a"), chunk_rows=64)
    evs = light_events(10)
    evs[5].provenance = Provenance.IMAGINED
    with pytest.raises(ValueError, match="REAL"):
        a.append(evs)
    assert a.count() == 0
    evs[5].provenance = Provenance.COUNTERFACTUAL
    with pytest.raises(ValueError):
        a.append(evs)
    assert a.count() == 0
    a.close()
    b = EventArchive(str(tmp_path / "b"), chunk_rows=64, allow_counterfactual=True)
    b.append(evs)
    assert b.count() == 10 and b.meta([evs[5].event_id])[0]["provenance"] == "counterfactual"
    evs2 = light_events(1, seed=9)
    evs2[0].provenance = Provenance.IMAGINED
    with pytest.raises(ValueError):
        b.append(evs2)
    b.close()


def test_redelivered_events_are_not_duplicated(tmp_path):
    a = EventArchive(str(tmp_path / "a"), chunk_rows=50)
    evs = light_events(120)
    a.append(evs[:80])
    a.append(evs[40:])
    a.append(evs[:10] + evs[:10])
    assert a.count() == 120 and a.stats["duplicates"] == 40 + 20
    assert sorted(r.event_id for b in a.scan() for r in b) == sorted(e.event_id for e in evs)
    a.close()


def test_scan_filters_match_reference(tmp_path):
    a = EventArchive(str(tmp_path / "a"), chunk_rows=300)
    evs = []
    for k, fam in enumerate(["alpha", "beta", "gamma"]):
        evs += light_events(700, env=f"env{k}", episode_len=50, seed=k, family=fam)
    for i, e in enumerate(evs):
        if i % 11 == 0:
            e.extra["success"] = True
        e.entity_links = [f"link{i % 5}"]
    a.append(evs)
    cases = [
        ({"env_id": "env1"}, lambda e: e.env_id == "env1"),
        ({"family": ["alpha", "gamma"], "failed": True}, lambda e: e.extra.get("family") in ("alpha", "gamma")
         and not e.result.ok),
        ({"t_min": 5.0, "t_max": 20.0}, lambda e: 5.0 <= e.t_sim <= 20.0),
        ({"entity": "link3"}, lambda e: "link3" in e.entity_links),
        ({"entity": "ent4", "env_id": "env2"}, lambda e: e.env_id == "env2" and e.observations[0].entity == "ent4"),
        ({"success": True, "model_version": 0}, lambda e: e.extra.get("success") and e.model_version == 0),
        ({"episode_id": "env0-ep3", "step_min": 10, "step_max": 19}, lambda e: e.episode_id == "env0-ep3"
         and 10 <= e.step <= 19),
        ({"reward_min": 0.5, "done": False}, lambda e: e.reward >= 0.5 and not e.done),
    ]
    for f, ref in cases:
        got = [r.event_id for b in a.scan(batch_size=128, **f) for r in b]
        assert got == [e.event_id for e in evs if ref(e)], f
        assert a.count_where(**f) == len(got)
    with pytest.raises(ValueError, match="unknown archive filter"):
        list(a.scan(colour="red"))
    a.close()


_CRASH = """
import os, sys
from memhelpers import light_events
from superlearner.cam_unified.memory.archive import EventArchive
root, point = sys.argv[1], sys.argv[2]
a = EventArchive(root, chunk_rows=500, autoflush=True)
a.append(light_events(1200))
def boom(*args, **kw):
    os._exit(17)
if point == "after_tmp":
    EventArchive._rename_chunk = boom
elif point == "after_rename":
    EventArchive._commit_chunk_index = boom
elif point == "after_index":
    EventArchive._write_manifest = boom
a.append(light_events(400, start=1200))
os._exit(0)
"""


@pytest.mark.parametrize("point", ["after_tmp", "after_rename", "after_index"])
def test_restart_recovery_after_crash_mid_chunk_write(tmp_path, point):
    root = str(tmp_path / "a")
    r = _run(_CRASH, root, point)
    assert r.returncode == 17, r.stderr[-2000:]
    a = EventArchive(root, chunk_rows=500)
    rec = a.recovery
    assert a.count() == 1600, (a.count(), rec)
    if point == "after_tmp":                          # the half-made chunk is discarded; its rows are in the WAL
        assert rec["removed_tmp"] == 1 and rec["adopted_chunks"] == 0
        assert a.info()["chunks"] == 2 and a.count_where(chunk=-1) == 600
    else:                                             # the renamed (complete) chunk is adopted
        assert rec["adopted_chunks"] == 1
        assert a.info()["chunks"] == 3 and a.count_where(chunk=-1) == 100
    evs = light_events(1600)
    got = a.get([e.event_id for e in evs])
    assert [g.event_id for g in got] == [e.event_id for e in evs]
    assert a.verify()["ok"]
    a.append(light_events(10, start=1600))            # ingestion continues after recovery
    assert a.count() == 1610 and a.info()["chunks"] == 3 and a.count_where(chunk=-1) == 110
    a.close()


def test_torn_wal_line_is_ignored(tmp_path):
    root = str(tmp_path / "a")
    a = EventArchive(root, chunk_rows=1000)
    a.append(light_events(30))
    a.close()
    wal = os.path.join(root, "wal", sorted(os.listdir(os.path.join(root, "wal")))[0])
    with open(wal, "a") as fh:
        fh.write('{"gseq": 30, "event_id": "half-writt')       # crash in the middle of a line
    b = EventArchive(root, chunk_rows=1000)
    assert b.count() == 30 and b.recovery["wal_torn_lines"] == 1
    b.append(light_events(5, start=30))
    b.close()
    c = EventArchive(root, chunk_rows=1000)
    assert c.count() == 35 and c.recovery["wal_torn_lines"] == 0
    c.close()


def test_lost_index_is_rebuilt_with_tags_and_training_provenance(tmp_path):
    root = str(tmp_path / "a")
    a = EventArchive(root, chunk_rows=100)
    evs = light_events(250)
    a.append(evs)
    a.tag([evs[3].event_id, evs[240].event_id], success=True, surprise=4.5)
    a.mark_used([evs[3].event_id], "update-17")
    a.close()
    for suf in ("", "-wal", "-shm"):
        p = os.path.join(root, "index.sqlite" + suf)
        if os.path.exists(p):
            os.unlink(p)
    b = EventArchive(root, chunk_rows=100)
    assert b.recovery["index_rebuilt"] and b.count() == 250
    m = b.meta([evs[3].event_id, evs[240].event_id, evs[4].event_id])
    assert [x["success"] for x in m] == [1, 1, None] and m[0]["surprise"] == 4.5
    assert b.get([evs[3].event_id])[0].training_provenance == {"used_in_updates": ["update-17"]}
    assert b.count_where(entity="ent3") == sum(1 for e in evs if e.observations[0].entity == "ent3")
    b.close()


def test_storage_limit_raises_and_drops_nothing(tmp_path):
    root = str(tmp_path / "a")
    a = EventArchive(root, chunk_rows=200, max_bytes=2_400_000, cold_policy=())
    accepted = []
    with pytest.raises(StorageLimitExceeded) as ei:
        for i in range(0, 100_000, 100):
            batch = light_events(100, start=i)
            a.append(batch)
            accepted += batch
    assert ei.value.limit == 2_400_000
    assert a.count() == len(accepted) > 0
    assert a.used_bytes() <= 2_400_000
    with pytest.raises(KeyError):
        a.get([f"env0-0-{len(accepted):09d}"])            # the rejected call's records are not half-written
    assert [r.event_id for r in a.get([e.event_id for e in accepted])] == [e.event_id for e in accepted]
    a.close()
    b = EventArchive(root, chunk_rows=200, max_bytes=10_000_000, cold_policy=())   # operator raises the budget
    b.append(light_events(100, start=len(accepted)))
    assert b.count() == len(accepted) + 100
    b.close()


def _fill(a, limit_events: int, batch: int = 250, start: int = 0) -> tuple:
    """Append until the budget raises; check the budget is never exceeded and a refused call leaves no trace."""
    n = start
    try:
        while n < limit_events:
            a.append(_text_events(batch, n))
            n += batch
            assert a.max_bytes is None or a.used_bytes() <= a.max_bytes
    except StorageLimitExceeded:
        assert a.count() == n
        with pytest.raises(KeyError):
            a.get([f"env0-0-{n:09d}"])
        return n, True
    return n, False


def _text_events(n: int, start: int) -> list:
    """Events carrying ~2 KB of document-like text each (so chunk storage dominates, as in real lifetimes)."""
    import random
    evs = light_events(n, start=start)
    for e in evs:
        rng = random.Random(int(e.event_id[-9:]))
        e.observations[0].content = " ".join(["tok%d" % rng.randrange(20000) for _ in range(300)])
    return evs


def test_cold_policies_extend_capacity_and_keep_data_readable(tmp_path):
    probe = EventArchive(str(tmp_path / "probe"), chunk_rows=500)
    for i in range(0, 3000, 500):
        probe.append(_text_events(500, i))
    limit = int(probe.used_bytes() * 0.8)
    probe.close()
    none = EventArchive(str(tmp_path / "none"), chunk_rows=500, max_bytes=limit, cold_policy=())
    n_none, raised = _fill(none, 60_000)
    assert raised and none.stats["compressed_chunks"] == 0
    none.close()
    comp = EventArchive(str(tmp_path / "comp"), chunk_rows=500, max_bytes=limit, cold_policy=("compress",))
    n_comp, raised = _fill(comp, 60_000)
    assert raised and comp.stats["compressed_chunks"] >= 1 and comp.info()["tiers"]["z"] >= 1
    assert n_comp > n_none                             # the policy bought capacity instead of dropping anything
    assert comp.verify()["ok"] and len(comp.get([f"env0-0-{i:09d}" for i in range(0, n_comp, 97)])) == len(
        range(0, n_comp, 97))
    comp.close()
    cold = str(tmp_path / "coldvol")
    mv = EventArchive(str(tmp_path / "mv"), chunk_rows=500, max_bytes=limit, cold_policy=("compress", "move"),
                      cold_dir=cold, cold_max_bytes=2 * limit)
    n_mv, raised = _fill(mv, 200_000)
    assert raised                                      # the cold volume's own budget is explicit too
    assert n_mv > n_comp and mv.stats["moved_chunks"] >= 1 and 0 < mv.cold_bytes() <= 2 * limit
    print(f"capacity at {limit} bytes: no policy {n_none}, compress {n_comp}, compress+move {n_mv} events")
    assert any(c["loc"] == "cold" for c in mv.chunks()) and mv.verify()["ok"]
    ids = [f"env0-0-{i:09d}" for i in range(0, n_mv, 211)]
    assert [r.event_id for r in mv.get(ids)] == ids
    mv.close()
    re = EventArchive(str(tmp_path / "mv"), chunk_rows=500)          # cold location comes from the manifest
    assert re.count() == n_mv and re.verify()["ok"] and len(re.get(ids)) == len(ids)
    re.close()


_SCAN_MEM = """
import sys, tracemalloc, json, resource
import pyarrow as pa
from superlearner.cam_unified.memory.archive import EventArchive
a = EventArchive(sys.argv[1], readonly=True)
for b in a.scan(batch_size=8, gseq_max=20):       # warm-up: lazy imports of the first decode are not working set
    pass
tracemalloc.start()
n = 0
for b in a.scan(batch_size=256, gseq_max=int(sys.argv[2])):
    n += len(b)
    del b
cur, peak = tracemalloc.get_traced_memory()
print(json.dumps({"n": n, "py_peak": peak, "arrow_peak": pa.default_memory_pool().max_memory()}))
"""


def test_scan_is_memory_bounded(tmp_path):
    root = str(tmp_path / "a")
    a = EventArchive(root, chunk_rows=4000, row_group_rows=512)
    text = "lorem ipsum dolor sit amet " * 120            # ~3.2 KB of text per event
    total = 0
    for i in range(0, 14_000, 2000):
        evs = light_events(2000, start=i, content=text)
        a.append(evs)
        total += sum(len(e.to_json()) for e in evs)
    a.close()
    peaks = {}
    for upto in (3_499, 13_999):                      # a quarter of the lifetime vs all of it
        r = _run(_SCAN_MEM, root, str(upto))
        assert r.returncode == 0, r.stderr[-2000:]
        peaks[upto] = m = json.loads(r.stdout.strip().splitlines()[-1])
        assert m["n"] == upto + 1
    q, f = peaks[3_499], peaks[13_999]
    assert total > 45e6
    assert f["py_peak"] < 1.3 * q["py_peak"] + 1e6, peaks        # memory does not grow with the scanned length
    assert f["arrow_peak"] < 1.3 * q["arrow_peak"] + 1e6, peaks
    assert f["py_peak"] + f["arrow_peak"] < 0.4 * total, peaks   # and is a fraction of the archived data


def test_long_content_offloaded_to_artifacts_and_resolved(tmp_path):
    art = ArtifactStore(str(tmp_path / "art"))
    a = EventArchive(str(tmp_path / "a"), chunk_rows=64, artifacts=art, inline_limit=100)
    evs = sample_events(5)
    code = "def f(x):\n    return x * 2\n" * 50
    evs[2].observations[2].content = code
    a.append(evs)
    a.seal()
    o = a.get([evs[2].event_id])[0].observations[2]
    assert len(o.content) == 100 and o.artifact_ref.startswith("sha256:") and o.meta["truncated"]
    assert a.resolve(o) == code
    a.close()


def test_backup_is_verified_and_openable(tmp_path):
    a = EventArchive(str(tmp_path / "a"), chunk_rows=100)
    evs = light_events(450)
    a.append(evs)
    a.tag([evs[0].event_id], skill="lever_tuning")
    rep = a.backup(str(tmp_path / "bk"))
    assert rep["chunks_verified"] == [0, 1, 2, 3] and rep["count"] == 450
    a.close()
    b = EventArchive(str(tmp_path / "bk"), chunk_rows=100)
    assert b.count() == 450 and b.verify()["ok"] and b.meta([evs[0].event_id])[0]["skill"] == "lever_tuning"
    assert [r.event_id for r in b.get([evs[449].event_id])] == [evs[449].event_id]
    b.close()


def test_single_writer_and_concurrent_readonly_reader(tmp_path):
    root = str(tmp_path / "a")
    a = EventArchive(root, chunk_rows=100)
    a.append(light_events(250))
    a.flush()
    with pytest.raises(ArchiveLocked):
        EventArchive(root)
    r = EventArchive(root, readonly=True)
    assert r.count() == 250 and len(r.get(["env0-0-000000249"])) == 1
    with pytest.raises(PermissionError):
        r.append(light_events(1, start=999))
    r.close()
    a.close()


def test_reader_follows_a_live_writer_and_sizing_is_persistent(tmp_path):
    root = str(tmp_path / "a")
    w = EventArchive(root, chunk_rows=100)
    w.append(light_events(250))
    w.flush()
    r = EventArchive(root, readonly=True)
    assert r.count() == 250 and r.chunk_rows == 100           # chunk size comes from the manifest
    w.append(light_events(300, start=250))                    # seals 3 more chunks after the reader opened
    w.flush()
    ids = [r_.event_id for b in r.scan(batch_size=128) for r_ in b]
    assert ids == [f"env0-0-{i:09d}" for i in range(550)]     # new chunks and WAL rows are picked up
    assert r.count() == 550 and r.get(["env0-0-000000549"])[0].step == 49
    r.close()
    w.close()
    again = EventArchive(root)
    assert again.chunk_rows == 100
    again.close()


def _du(path: str) -> int:
    tot = 0
    for root, _d, files in os.walk(path):
        for f in files:
            tot += os.lstat(os.path.join(root, f)).st_size
    return tot


def test_default_policy_budget_bounds_everything_under_the_root(tmp_path):
    """With the DEFAULT cold policy the explicit budget bounds the archive's whole directory: chunks, WAL, tag logs,
    SQLite files, manifest and a replay sampler database living in the same root."""
    from superlearner.cam_unified.config import make_config
    from superlearner.cam_unified.memory.replay import ReplaySampler
    limit = 3_000_000
    root = str(tmp_path / "a")
    a = EventArchive(root, make_config("dev", archive_max_bytes=limit, archive_chunk_rows=300))
    assert a.cold_policy == ("compress",) and not a.cold_external
    rs = ReplaySampler(a)                                     # its database is root/replay.sqlite
    n, raised = 0, False
    try:
        while n < 100_000:
            a.append(_text_events(150, n))
            n += 150
            a.flush()
            assert _du(root) <= limit                         # measured on disk, not just the archive's own estimate
            assert a.used_bytes() == _du(root)                # the accounting covers every file under root
            a.tag([f"env0-0-{n - 1:09d}"], success=True)
            rs.sync()
    except StorageLimitExceeded:
        raised = True
    assert raised and a.count() == n and a.stats["compressed_chunks"] >= 1
    assert _du(root) <= limit + 200_000                       # + the sampler's growth since the last append
    assert os.path.getsize(os.path.join(root, "replay.sqlite")) > 0
    ids = [f"env0-0-{i:09d}" for i in range(0, n, 157)]
    assert [r.event_id for r in a.get(ids)] == ids            # nothing was dropped to make room
    rs.close()
    a.close()


def test_move_policy_requires_an_external_cold_volume_with_its_own_budget(tmp_path):
    with pytest.raises(ValueError, match="outside the archive root"):
        EventArchive(str(tmp_path / "a"), cold_policy=("compress", "move"))
    EventArchive(str(tmp_path / "a")).close()               # the failed open released the writer lock
    with pytest.raises(ValueError, match="outside the archive root"):
        EventArchive(str(tmp_path / "b"), cold_policy=("move",), cold_dir=str(tmp_path / "b" / "cold2"),
                     cold_max_bytes=10**6)
    with pytest.raises(ValueError, match="cold_max_bytes"):
        EventArchive(str(tmp_path / "c"), cold_policy=("move",), cold_dir=str(tmp_path / "cold"))
    ok = EventArchive(str(tmp_path / "d"), cold_policy=("move",), cold_dir=str(tmp_path / "cold"), cold_max_bytes=10**6)
    assert ok.cold_external
    ok.close()


_TAIL_RSS = """
import json, os, sys
from memhelpers import light_events
from superlearner.cam_unified.memory.archive import EventArchive
root, n, cb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
a = EventArchive(root, chunk_rows=65536, chunk_bytes=cb, fsync=False)
w = light_events(200, start=10**8)
for e in w:
    e.observations[0].content = os.urandom(5000).hex()
a.append(w)
a.seal()                                          # warm-up: imports and the first Parquet write are not working set


def hwm_kb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1])


def rss_kb():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1])


with open("/proc/self/clear_refs", "w") as fh:  # reset the peak-RSS mark to the current RSS
    fh.write("5")
base = rss_kb()
mx = 0
for i in range(0, n, 200):
    evs = light_events(200, start=i)
    for e in evs:
        e.observations[0].content = os.urandom(5000).hex()     # ~11 KB per record
    a.append(evs)
    mx = max(mx, a.info()["wal_bytes"])
peak = hwm_kb()
print(json.dumps({"growth_mb": (peak - base) / 1024, "max_wal_bytes": mx, "chunks": a.info()["chunks"],
                  "count": a.count(), "base_mb": base / 1024}))
a.close()
"""


def test_unsealed_tail_is_bounded_by_bytes_not_only_rows(tmp_path):
    """Large records: the tail is sealed by ``chunk_bytes`` long before ``chunk_rows``; peak memory does not grow
    with the number of records ingested (12k x 11 KB records would hold ~135 MB in a row-bounded tail)."""
    cb = 4_000_000
    out = {}
    for n in (3000, 12000):
        r = _run(_TAIL_RSS, str(tmp_path / f"a{n}"), str(n), str(cb))
        assert r.returncode == 0, r.stderr[-2000:]
        out[n] = m = json.loads(r.stdout.strip().splitlines()[-1])
        assert m["count"] == n + 200 and m["max_wal_bytes"] < cb and m["chunks"] >= (n + 200) * 11_000 // (2 * cb)
    print("tail memory growth (MB) by records ingested:", {n: round(m["growth_mb"], 1) for n, m in out.items()})
    assert 0 < out[3000]["growth_mb"] and out[12000]["growth_mb"] < out[3000]["growth_mb"] + 25
    assert out[12000]["growth_mb"] < 90
    re = EventArchive(str(tmp_path / "a12000"))
    assert re.count() == 12200 and re.info()["wal_bytes"] < cb and re.chunk_rows == 65536
    ids = [f"env0-0-{i:09d}" for i in range(0, 12000, 1999)]
    assert [x.event_id for x in re.get(ids)] == ids
    re.close()


def test_tag_logs_are_not_relogged_on_restart(tmp_path):
    root = str(tmp_path / "a")
    a = EventArchive(root, chunk_rows=100)
    evs = light_events(250)
    a.append(evs)
    a.tag([evs[10].event_id], success=True)                 # a sealed row
    a.tag([evs[230].event_id], success=False, surprise=3.5)  # a row still in the WAL tail
    a.tag([evs[10].event_id], success=False)                 # a correction
    n_changes = len(a.tag_changes(0))
    assert n_changes == 3 and a.tag_changes(0, with_keys=True)[1][2] == ["success", "surprise"]
    epoch = a.index_epoch
    a.close()
    for _ in range(3):
        b = EventArchive(root, chunk_rows=100)
        assert len(b.tag_changes(0)) == n_changes            # replaying the tag log changes nothing committed
        m = b.meta([evs[10].event_id, evs[230].event_id])
        assert [x["success"] for x in m] == [0, 0] and m[1]["surprise"] == 3.5
        assert b.index_epoch == epoch
        b.close()
    for suf in ("", "-wal", "-shm"):                         # a lost index is rebuilt with a new epoch
        if os.path.exists(os.path.join(root, "index.sqlite" + suf)):
            os.unlink(os.path.join(root, "index.sqlite" + suf))
    c = EventArchive(root, chunk_rows=100)
    assert c.index_epoch != epoch and [x["success"] for x in c.meta([evs[10].event_id, evs[230].event_id])] == [0, 0]
    c.close()
