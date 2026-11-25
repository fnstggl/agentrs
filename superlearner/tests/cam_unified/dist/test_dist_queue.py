"""ExperienceQueue: segment integrity, dedup, crash recovery on both sides, backpressure, delayed outcomes, ordering,
and a real kill -9 of writer processes (manifest F01)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from superlearner.cam_unified.dist.queue import (CorruptSegment, ExperienceQueue, SpoolFull, WireTrace,
                                                  decode_segment, encode_record, encode_segment)
from superlearner.cam_unified.dist.sinks import JsonlArchive
from superlearner.cam_unified.interfaces import DecisionTrace, EventRecord, Observation, ObsKind, new_id
from superlearner.cam_unified.testing import sample_events

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))


def _events(n, actor="a0", version=0, seed=0):
    out = sample_events(n, seed=seed)
    for e in out:
        e.actor_id = actor
        e.model_version = version
    return out


def _trace(e: EventRecord) -> DecisionTrace:
    return DecisionTrace(decision_id=new_id("d_"), event_id=e.event_id, env_id=e.env_id, policy_version=e.model_version,
                         t_sim=e.t_sim, objective="obj", candidates=[{"action": {"tool": "x"}, "score": 0.5}],
                         selected={"tool": "x", "args": {}}, outcome={"ok": True})


def test_segment_roundtrip_checksum_and_corruption():
    evs = _events(5)
    data = encode_segment("a0", 3, 123, [("event", e) for e in evs] + [("trace", _trace(evs[0]))])
    header, items = decode_segment(data)
    assert header["actor_id"] == "a0" and header["seq"] == 3 and header["n"] == 6
    assert header["model_versions"] == [0, 0] and header["kinds"] == {"event": 5, "trace": 1}
    back = [EventRecord.from_dict(d) for k, d, _ in items if k == "event"]
    assert [b.to_dict() for b in back] == [e.to_dict() for e in evs]
    # one flipped byte in the body -> checksum mismatch
    i = data.index(b"set_price")
    bad = data[:i] + b"X" + data[i + 1:]
    with pytest.raises(CorruptSegment, match="checksum"):
        decode_segment(bad)
    with pytest.raises(CorruptSegment):
        decode_segment(data[:-20])           # truncated
    with pytest.raises(CorruptSegment):
        decode_segment(data.replace(b"cam-unified-seg/1", b"cam-unified-seg/9"))


def test_encode_record_equivalent_to_contract_to_dict():
    evs = _events(30, seed=3)
    delayed = EventRecord(event_id=new_id("e_"), env_id="env0", actor_id="a0", episode_id="ep", step=3,
                          model_version=2, t_sim=1.5, observations=[Observation(ObsKind.EVENT, "payout", value=1.0,
                                                                                 unit="points", caused_by="a_x",
                                                                                 meta={"key": "payoff"})],
                          attributed_to=evs[0].event_id, extra={"kind": "delayed_outcome"})
    for e in evs + [delayed]:
        d = json.loads(json.dumps(encode_record(e)))
        assert EventRecord.from_dict(d).to_dict() == e.to_dict()
        assert len(json.dumps(d)) <= len(e.to_json())          # compact: default fields dropped
    tr = _trace(evs[1])
    assert json.loads(json.dumps(encode_record(tr))) == tr.to_dict()
    # lazy trace decoding keeps the exact record and its routing fields
    data = encode_segment("a0", 0, 1, [("trace", tr)])
    _, items = decode_segment(data, lazy_traces=True)
    wt = items[0][1]
    assert isinstance(wt, WireTrace)
    assert (wt.decision_id, wt.event_id, wt.env_id, wt.policy_version) == (tr.decision_id, tr.event_id, tr.env_id,
                                                                            tr.policy_version)
    assert wt.load().to_dict() == tr.to_dict()


def test_dedup_across_and_within_segments(tmp_path):
    q = ExperienceQueue(str(tmp_path / "spool"))
    sink = JsonlArchive(str(tmp_path / "arch"))
    w = q.writer("a0", batch_records=1000)
    evs = _events(20)
    for e in evs:
        w.put(e)
    w.put(evs[3])                             # duplicate within one segment
    w.flush()
    for e in evs[:10]:                        # re-sent later (e.g. a retry after an uncertain commit)
        w.put(e)
    w.flush()
    c = q.consumer()
    r = c.ingest(sink)
    assert len(r.events) == 20 and r.n_dup == 11 and len(r.segments) == 2
    assert sink.count() == 20 and c.seen_count() == 20
    assert {e.event_id for e in r.events} == {e.event_id for e in evs}
    # a third copy after everything was consumed is still recognised
    for e in evs[5:8]:
        w.put(e)
    w.flush()
    r2 = c.ingest(sink)
    assert r2.events == [] and r2.n_dup == 3 and sink.count() == 20
    assert sorted(os.listdir(q.done)) and not q.pending()


def test_writer_recovers_committed_tmp_and_quarantines_partial(tmp_path):
    q = ExperienceQueue(str(tmp_path / "spool"))
    w = q.writer("a1", batch_records=5)
    for e in _events(10, actor="a1"):
        w.put(e)                               # two committed segments (seq 0, 1)
    assert w.next_seq == 2
    # crash between fsync and rename: a complete tmp for seq 2
    evs = _events(4, actor="a1", seed=9)
    good = encode_segment("a1", 2, time.time_ns(), [("event", e) for e in evs])
    with open(os.path.join(q.incoming, ".a1-0000000002.tmp"), "wb") as fh:
        fh.write(good)
    # crash mid-write: a truncated tmp for seq 3
    with open(os.path.join(q.incoming, ".a1-0000000003.tmp"), "wb") as fh:
        fh.write(good[: len(good) // 2])
    w2 = q.writer("a1", batch_records=5)       # the restarted actor
    assert w2.stats["recovered_tmp"] == 1 and w2.stats["quarantined_tmp"] == 1
    assert w2.next_seq == 3                    # continues after the recovered segment, never reuses a number
    assert not [f for f in os.listdir(q.incoming) if f.endswith(".tmp")]
    assert any(f.endswith(".reason.json") for f in os.listdir(q.quarantine))
    sink = JsonlArchive(str(tmp_path / "arch"))
    c = q.consumer()
    r = c.ingest(sink)
    assert len(r.events) == 14 and sink.count() == 14
    assert {e.event_id for e in evs} <= {e.event_id for e in r.events}
    assert c.gaps() == {}


class _CrashOnCommit(Exception):
    pass


def test_consumer_crash_between_sink_and_commit_is_reconciled(tmp_path, monkeypatch):
    q = ExperienceQueue(str(tmp_path / "spool"))
    w = q.writer("a0", batch_records=8)
    evs = _events(24)
    for e in evs:
        w.put(e)
    sink = JsonlArchive(str(tmp_path / "arch"))
    c = q.consumer()
    orig = c._note_seq

    def boom(*a, **k):                         # fails inside the seen-set transaction, after the sink append
        raise _CrashOnCommit()
    monkeypatch.setattr(c, "_note_seq", boom)
    with pytest.raises(_CrashOnCommit):
        c.ingest(sink, max_segments=2)
    assert sink.count() == 16                  # the sink got the records
    assert c.seen_count() == 0                 # but the seen-set rolled back; segments are 'claimed'
    c.close()
    c2 = q.consumer()                          # the restarted learner
    r = c2.ingest(sink)
    assert r.n_recovered_in_sink == 16
    assert sink.count() == 24                  # nothing appended twice, nothing lost
    assert len({e.event_id for b in sink.scan() for e in b}) == 24
    assert c2.seen_count() == 24 and not q.pending()
    del orig


def test_backpressure_block_waits_for_consumer(tmp_path):
    q = ExperienceQueue(str(tmp_path / "spool"), max_spool_bytes=20_000, policy="block")
    w = q.writer("a0", batch_records=10)
    evs = _events(100)
    done = threading.Event()

    def produce():
        for e in evs:
            w.put(e)
        w.flush()
        done.set()
    th = threading.Thread(target=produce)
    th.start()
    time.sleep(0.5)
    assert not done.is_set(), "writer should be blocked by the full spool"
    assert q.spool_bytes() <= 20_000
    sink = JsonlArchive(str(tmp_path / "arch"))
    c = q.consumer()
    got = []
    t0 = time.time()
    while not (done.is_set() and not q.pending()) and time.time() - t0 < 30:
        got += c.ingest(sink, max_segments=1).events
        time.sleep(0.02)
    th.join(5)
    assert done.is_set()
    assert len(got) == 100 and sink.count() == 100
    assert w.stats["blocked_s"] > 0.3 and w.stats["shed_items"] == 0


def test_backpressure_block_timeout_and_raise_policies(tmp_path):
    q = ExperienceQueue(str(tmp_path / "s1"), max_spool_bytes=5_000, policy="block", block_timeout_s=0.3)
    w = q.writer("a0", batch_records=5)
    with pytest.raises(SpoolFull):
        for e in _events(50):
            w.put(e)
    assert len(w.buf) > 0                      # the experience is still held, not dropped
    q2 = ExperienceQueue(str(tmp_path / "s2"), max_spool_bytes=5_000, policy="raise")
    w2 = q2.writer("a0", batch_records=5)
    with pytest.raises(SpoolFull):
        for e in _events(50):
            w2.put(e)


def test_backpressure_shed_is_recorded_not_silent(tmp_path):
    q = ExperienceQueue(str(tmp_path / "spool"), max_spool_bytes=12_000, policy="shed")
    w = q.writer("a0", batch_records=5)
    evs = _events(40)
    for e in evs:
        w.put(e)
    w.flush()
    assert w.stats["shed_segments"] > 0
    ledger = q.shed_ledger("a0")
    shed_ids = {i for row in ledger for i in row["ids"]}
    assert len(shed_ids) == w.stats["shed_items"] and all(row["reason"] for row in ledger)
    sink = JsonlArchive(str(tmp_path / "arch"))
    c = q.consumer()
    got = {e.event_id for e in c.ingest(sink, max_segments=100).events}
    assert got.isdisjoint(shed_ids) and got | shed_ids == {e.event_id for e in evs}
    gaps = c.gaps().get("a0", [])
    assert sorted(gaps) == sorted(row["seq"] for row in ledger if row["seq"] < max(
        int(n.split("-")[2][:10]) for n in os.listdir(q.done)))


def test_delayed_outcomes_traces_and_version_tags(tmp_path):
    q = ExperienceQueue(str(tmp_path / "spool"))
    w = q.writer("a0", batch_records=100)
    evs = _events(6, version=7)
    delayed = EventRecord(event_id=new_id("e_"), env_id="env0", actor_id="a0", episode_id=evs[0].episode_id, step=6,
                          model_version=8, t_sim=6.0, observations=[Observation(ObsKind.EVENT, "late", value=3.0)],
                          attributed_to=evs[2].event_id)
    for e in evs:
        w.put(e, _trace(e))
    w.put(delayed)
    name = w.flush()
    with open(os.path.join(q.incoming, name), "rb") as fh:
        header, _ = decode_segment(fh.read())
    assert header["model_versions"] == [7, 8]
    sink = JsonlArchive(str(tmp_path / "arch"))
    r = q.consumer().ingest(sink)
    assert [d.event_id for d in r.delayed] == [delayed.event_id]
    assert r.delayed[0].attributed_to == evs[2].event_id
    assert len(r.traces) == 6 and {t.event_id for t in r.traces} == {e.event_id for e in evs}
    assert all(e.model_version == 7 for e in r.events if e.action is not None)
    assert [x.event_id for b in sink.scan(attributed_to=evs[2].event_id) for x in b] == [delayed.event_id]


def test_corrupt_committed_segment_is_quarantined_and_reported(tmp_path):
    q = ExperienceQueue(str(tmp_path / "spool"))
    w = q.writer("a0", batch_records=5)
    for e in _events(15):
        w.put(e)
    names = q.pending()
    p = os.path.join(q.incoming, names[1])
    data = bytearray(open(p, "rb").read())
    data[len(data) // 2] ^= 0xFF
    open(p, "wb").write(bytes(data))
    sink = JsonlArchive(str(tmp_path / "arch"))
    c = q.consumer()
    r = c.ingest(sink)
    assert [n for n, _ in r.quarantined] == [names[1]]
    assert sink.count() == 10
    assert c.gaps() == {"a0": [1]}               # the loss is visible, not silent
    assert os.path.exists(os.path.join(q.quarantine, names[1] + ".reason.json"))


def test_consumes_in_commit_order_across_writers(tmp_path):
    q = ExperienceQueue(str(tmp_path / "spool"))
    wa, wb = q.writer("a", batch_records=1), q.writer("b", batch_records=1)
    order = []
    for i in range(10):
        e = _events(1, actor="a" if i % 2 == 0 else "b", seed=i)[0]
        (wa if i % 2 == 0 else wb).put(e)
        order.append(e.event_id)
    r = q.consumer().ingest(None, max_segments=100)
    assert [e.event_id for e in r.events] == order


_WRITER = r"""
import sys, time
sys.path.insert(0, {src!r})
from superlearner.cam_unified.dist.queue import ExperienceQueue
from superlearner.cam_unified.interfaces import EventRecord, Observation, ObsKind
q = ExperienceQueue({root!r})
w = q.writer({actor!r}, batch_records=7)
print("ready", flush=True)
for i in range({n}):
    w.put(EventRecord(event_id="e_{actor}_%06d" % i, env_id="env", actor_id={actor!r}, episode_id="ep", step=i,
                      model_version=i // 50, t_sim=float(i),
                      observations=[Observation(ObsKind.NUMERIC, value=float(i), unit="u")]))
    time.sleep({sleep})
w.flush()
print("done", flush=True)
"""


def _spawn_writer(root, actor, n, sleep):
    code = _WRITER.format(src=SRC, root=root, actor=actor, n=n, sleep=sleep)
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "ready"
    return p


def test_killed_writer_processes_restart_without_loss_or_duplicates(tmp_path):
    root = str(tmp_path / "spool")
    ExperienceQueue(root)
    pa = _spawn_writer(root, "wa", 600, 0.002)
    pb = _spawn_writer(root, "wb", 300, 0.001)
    time.sleep(0.6)
    pa.send_signal(signal.SIGKILL)               # crash mid-stream, possibly mid-segment
    pa.wait()
    assert pb.wait(60) == 0
    q = ExperienceQueue(root)
    committed_before = {n for n in q.pending() if "-wa-" in n}
    assert committed_before, "the killed writer had committed some segments"
    # the restarted actor re-sends its whole stream (as an actor replaying a local log would)
    pa2 = _spawn_writer(root, "wa", 600, 0.0)
    assert pa2.wait(60) == 0
    sink = JsonlArchive(str(tmp_path / "arch"))
    c = q.consumer()
    while c.ingest(sink, max_segments=50).segments:
        pass
    ids = [e.event_id for b in sink.scan() for e in b]
    assert len(ids) == len(set(ids)) == 900       # every event exactly once
    assert set(ids) == {f"e_wa_{i:06d}" for i in range(600)} | {f"e_wb_{i:06d}" for i in range(300)}
    st = c.stats()
    assert st["gaps"] == {}                      # sequence numbers continued across the crash
    assert st["totals"]["dup"] > 0               # the re-sent prefix was recognised as duplicates
    assert not [f for f in os.listdir(q.incoming) if f.endswith(".tmp")]
    versions = {e.model_version for b in sink.scan(actor_id="wa") for e in b}
    assert versions == set(range(12))            # model-version tags survive transport


def test_fast_event_decoder_equals_contract_from_dict():
    from superlearner.cam_unified.dist.queue import decode_event
    from superlearner.cam_unified.interfaces import EvidenceKind, ExecutionResult, Provenance
    evs = _events(20, seed=5)
    evs[0].result = ExecutionResult(False, [Observation(ObsKind.ERROR, "boom", evidence=EvidenceKind.INFERRED)],
                                    error="x")
    evs[1].action = None
    evs[2].provenance = Provenance.COUNTERFACTUAL
    for e in evs:
        d = json.loads(json.dumps(encode_record(e)))
        assert decode_event(json.loads(json.dumps(d))) == EventRecord.from_dict(d) == e


_WAL_WRITER = r"""
import os, sys, time
sys.path.insert(0, {src!r})
from superlearner.cam_unified.dist.queue import ExperienceQueue
from superlearner.cam_unified.interfaces import EventRecord
q = ExperienceQueue({root!r})
w = q.writer("wa", batch_records=10**6)          # never commits on its own
for i in range({n}):
    w.put(EventRecord(event_id="e_wal_%06d" % i, env_id="w", actor_id="wa", episode_id="ep", step=i,
                      model_version=0, t_sim=float(i), observations=[]))
print("ready", flush=True)
time.sleep(60)
"""


def test_writer_wal_recovers_buffered_items_after_sigkill(tmp_path):
    """An actor killed with SIGKILL before committing its buffer loses nothing: the next writer for the actor id
    commits the write-ahead log."""
    import signal
    import subprocess
    import sys
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
    root = str(tmp_path / "spool")
    p = subprocess.Popen([sys.executable, "-c", _WAL_WRITER.format(src=src, root=root, n=300)],
                         stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "ready"
    p.send_signal(signal.SIGKILL)
    p.wait(30)
    q = ExperienceQueue(root)
    assert q.pending() == []                           # nothing was committed by the dead process
    w = q.writer("wa", batch_records=10)
    assert w.stats["wal_recovered_items"] == 300 and w.stats["wal_torn_lines"] == 0
    assert len(q.pending()) == 1
    c = q.consumer(lazy_traces=False)
    r = c.ingest(None)
    assert [e.event_id for e in r.events] == ["e_wal_%06d" % i for i in range(300)]
    # a torn final line (killed inside write) is dropped and recorded, never silent
    with open(w.wal_path, "ab") as fh:
        fh.write(b'{"k":"event","d":{"event_id":"e_to')
    w2 = q.writer("wa", batch_records=10)
    assert w2.stats["wal_torn_lines"] == 1 and w2.stats["wal_recovered_items"] == 0
    assert any("torn" in e.get("reason", "") for e in q.shed_ledger("wa"))
    # after a commit the log is empty: a restart re-commits nothing
    w2.put(EventRecord(event_id="e_new", env_id="w", actor_id="wa", episode_id="ep", step=0, model_version=0,
                       t_sim=0.0, observations=[]))
    assert os.path.getsize(w2.wal_path) > 0
    w2.flush()
    assert os.path.getsize(w2.wal_path) == 0
    w3 = q.writer("wa", batch_records=10)
    assert w3.stats["wal_recovered_items"] == 0


def test_blocked_segment_is_ordered_by_its_actual_commit(tmp_path):
    """A writer that waited for spool space sorts after segments committed while it waited (its commit time is
    taken after admission, not before the wait)."""
    root = str(tmp_path / "spool")
    q = ExperienceQueue(root, max_spool_bytes=10_000, policy="block")
    big = ExperienceQueue(root, max_spool_bytes=1 << 30)          # same spool, writers with room to commit
    fill = big.writer("fill", batch_records=1000)
    for e in _events(10, actor="fill"):                          # ~14.6 kB: over the 10 kB limit
        fill.put(e)
    fill.flush()
    wb = q.writer("blk", batch_records=1000)
    for e in _events(3, actor="blk", seed=1):                    # ~4.5 kB
        wb.put(e)
    th = threading.Thread(target=wb.flush)
    th.start()
    time.sleep(0.3)                                               # "blk" is waiting for space
    assert th.is_alive() and q.pending()[0].split("-")[1] == "fill"
    wc = big.writer("late", batch_records=1000)
    for e in _events(3, actor="late", seed=2):
        wc.put(e)
    wc.flush()                                                    # commits while "blk" waits
    c = q.consumer()
    assert c.ingest(None, max_segments=1).segments[0].split("-")[1] == "fill"   # frees room for "blk" only
    th.join(10)
    assert not th.is_alive() and wb.stats["blocked_s"] > 0.2
    assert [n.split("-")[1] for n in q.pending()] == ["late", "blk"]
