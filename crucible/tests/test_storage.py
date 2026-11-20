"""Store conformance suite: the same tests run against MemoryStore, SQLiteStore (temp file) and PostgresStore
(throwaway cluster from pg_fixture.py; marked `postgres`, skipped when the binaries are missing).

Covers the raw store contract (worlds, contiguous append-only events, snapshots, leases, blobs), the kernel's
use of it (fork lineage, crash recovery with World.load, a second writer being rejected), concurrent writers
(threads, and processes for the multi-process backends) and a small throughput benchmark (printed only).
"""
from __future__ import annotations

import copy
import multiprocessing as mp
import os
import signal
import threading
import time

import pytest

from crucible.core.canonical import canonical_bytes
from crucible.core.kernel import World, WorldSpec
from crucible.services import admin as _admin, bank as _bank, identity as _identity  # noqa: F401  (registers modules)
from crucible.storage.base import StoreConflict
from crucible.storage.memory import MemoryStore
from crucible.storage.sqlite import SQLiteStore
from pg_fixture import pg_dsn, pg_fresh_dsn  # noqa: F401  (pytest fixtures)

SPEC = WorldSpec("storage-test", 5, [{"name": "identity"}, {"name": "bank"}, {"name": "admin"}],
                 params={"note": "spec round-trip", "neg0": -0.0, "x": 0.1, "big": 2**70})
BACKENDS = ["memory", "sqlite", pytest.param("postgres", marks=pytest.mark.postgres)]
MULTIPROC = ["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)]


# ====================================================================== backend plumbing
def open_store(spec: tuple):
    kind, arg = spec
    if kind == "sqlite":
        return SQLiteStore(arg)
    if kind == "postgres":
        from crucible.storage.postgres import PostgresStore
        return PostgresStore(arg)
    raise ValueError(kind)


class Backend:
    """Opens handles on one store. Memory has a single shared instance; SQL backends open independent handles
    (connections) to the same database, like separate processes would."""

    def __init__(self, kind: str, arg: str | None) -> None:
        self.kind, self.arg = kind, arg
        self._mem = MemoryStore() if kind == "memory" else None
        self.handles: list = []

    @property
    def spec(self) -> tuple:
        return (self.kind, self.arg)

    def open(self):
        s = self._mem if self._mem is not None else open_store(self.spec)
        self.handles.append(s)
        return s

    def close(self) -> None:
        for s in self.handles:
            s.close()


def _make_backend(kind, request, tmp_path) -> Backend:
    if kind == "memory":
        return Backend("memory", None)
    if kind == "sqlite":
        return Backend("sqlite", str(tmp_path / "crucible.db"))
    return Backend("postgres", request.getfixturevalue("pg_fresh_dsn"))


@pytest.fixture(params=BACKENDS)
def backend(request, tmp_path):
    b = _make_backend(request.param, request, tmp_path)
    yield b
    b.close()


@pytest.fixture(params=MULTIPROC)
def mp_backend(request, tmp_path):
    b = _make_backend(request.param, request, tmp_path)
    yield b
    b.close()


def ev(seq: int, **payload) -> dict:
    return {"seq": seq, "t": seq * 10, "type": "test.ev", "actor": "p:a", "cmd": seq if seq % 2 else None,
            "payload": payload or {"i": seq, "nested": {"l": [1, 2.5, "x"]}}, "vis": ["p:a", "p:b"] if seq % 3 else "public",
            "hash": f"{seq:032x}"}


def same(a, b) -> bool:
    """Exact equality as the kernel sees it (canonical bytes distinguish -0.0 from 0.0, 1 from 1.0)."""
    return canonical_bytes(a) == canonical_bytes(b)


# ====================================================================== worlds
def test_create_list_info(backend):
    s = backend.open()
    assert s.list_worlds() == [] and s.world_info("w1") is None
    s.create_world("w1", SPEC.to_dict(), None)
    s.create_world("w0", {"family": "f", "seed": 1, "modules": [], "params": {}}, None)
    with pytest.raises(StoreConflict):
        s.create_world("w1", SPEC.to_dict(), None)
    assert [w["id"] for w in s.list_worlds()] == ["w0", "w1"]
    info = s.world_info("w1")
    assert same(info["spec"], SPEC.to_dict()) and WorldSpec.from_dict(info["spec"]).hash() == SPEC.hash()
    assert info["parent"] is None and info["status"] == "active" and info["writer"] is None
    assert info["lease_until"] == 0 and abs(info["created"] - time.time()) < 60
    info["spec"]["seed"] = 999            # a returned record is a copy
    assert s.world_info("w1")["spec"]["seed"] == SPEC.seed
    s.set_status("w1", "archived")
    assert s.world_info("w1")["status"] == "archived" and s.world_info("w0")["status"] == "active"
    assert s.last_seq("w1") == 0 and s.read_events("w1") == []
    # another handle sees the same data
    assert [w["id"] for w in backend.open().list_worlds()] == ["w0", "w1"]


def test_contiguous_appends_and_read_ranges(backend):
    s = backend.open()
    s.create_world("w", {}, None)
    s.create_world("other", {}, None)
    evs = [ev(i) for i in range(1, 11)]
    s.append_events("w", evs[:3])
    s.append_events("w", evs[3:4])
    s.append_events("w", [])                    # no-op
    s.append_events("w", evs[4:])
    s.append_events("other", [ev(1, only="other")])
    assert s.last_seq("w") == 10 and s.last_seq("other") == 1
    got = s.read_events("w")
    assert same(got, evs)
    assert [e["seq"] for e in s.read_events("w", from_seq=3)] == list(range(4, 11))
    assert [e["seq"] for e in s.read_events("w", from_seq=3, to_seq=6)] == [4, 5, 6]
    assert [e["seq"] for e in s.read_events("w", to_seq=2)] == [1, 2]
    assert s.read_events("w", from_seq=10) == [] and s.read_events("w", from_seq=0, to_seq=0) == []
    assert [e["seq"] for e in s.read_events("w", from_seq=8, to_seq=99)] == [9, 10]
    assert same(s.read_events("other"), [ev(1, only="other")])
    got[0]["payload"]["i"] = "mutated"          # reads are copies
    assert s.read_events("w", to_seq=1)[0]["payload"]["i"] == 1


def test_noncontiguous_and_duplicate_appends_raise(backend):
    s = backend.open()
    s.create_world("w", {}, None)
    with pytest.raises(StoreConflict):
        s.append_events("w", [ev(2)])           # first event must be 1
    with pytest.raises(StoreConflict):
        s.append_events("w", [ev(0)])
    s.append_events("w", [ev(i) for i in range(1, 6)])
    for bad in ([ev(7)], [ev(5)], [ev(3)], [ev(1)],      # gap, duplicates
                [ev(6), ev(8)],                          # gap inside a batch (regression: SQL backends took it)
                [ev(6), ev(6)], [ev(6), ev(7), ev(7)],   # duplicate inside a batch
                [ev(6), ev(5)]):
        with pytest.raises(StoreConflict):
            s.append_events("w", bad)
        assert s.last_seq("w") == 5             # atomic: nothing of a rejected batch is kept
    assert [e["seq"] for e in s.read_events("w")] == [1, 2, 3, 4, 5]
    s.append_events("w", [ev(6), ev(7)])
    assert s.last_seq("w") == 7


def test_fork_base_continues_parent_sequence(backend):
    s = backend.open()
    s.create_world("root", {}, None)
    s.append_events("root", [ev(i) for i in range(1, 6)])
    s.create_world("child", {}, {"world": "root", "seq": 5, "hash": ev(5)["hash"]})
    assert s.last_seq("child") == 5 and s.read_events("child") == []
    with pytest.raises(StoreConflict):
        s.append_events("child", [ev(1)])
    s.append_events("child", [ev(6), ev(7)])
    assert s.last_seq("child") == 7 and s.last_seq("root") == 5
    assert [w["id"] for w in s.lineage("child")] == ["child", "root"]
    assert s.world_info("child")["parent"] == {"world": "root", "seq": 5, "hash": ev(5)["hash"]}


def test_payload_roundtrip_is_exact(backend):
    """Regression (Postgres): JSONB turned -0.0 into 0 and rejected NUL characters and unpaired surrogates, so a
    reloaded event no longer matched its chain hash (or could not be stored at all)."""
    s = backend.open()
    s.create_world("w", {}, None)
    tricky = {"neg0": -0.0, "one_f": 1.0, "one_i": 1, "tenth": 0.1, "tiny": 1e-7, "huge": 1.5e300, "big": 2**70,
              "neg": -(2**63), "nul": "a\x00b", "sur": "x\ud800y", "uni": "héllo ☃ \U0001f600", "quote": '"\\\n\t',
              "empty": {}, "list": [], "nested": [[{"a": None, "b": True, "c": False}]]}
    s.append_events("w", [ev(1, **tricky)])
    back = s.read_events("w")[0]["payload"]
    assert same(back, tricky)
    assert backend.open().read_events("w")[0]["hash"] == ev(1)["hash"]

    # and through the kernel: World.load verifies the chain over values like these
    w = World.create(SPEC, "w-tricky", store=s)
    w.system_call("admin.join", {"handle": "tricky", "info": tricky})
    w.flush()
    r = World.load(backend.open(), "w-tricky")
    assert (r.digest(), r.last_hash) == (w.digest(), w.last_hash)


# ====================================================================== snapshots, leases, blobs
def test_snapshots(backend):
    s = backend.open()
    s.create_world("w", {}, None)
    s.create_world("v", {}, None)
    assert s.list_snapshots("w") == [] and s.latest_snapshot("w") is None
    blob = bytes(range(256)) * 1000
    s.put_snapshot("s-b", "w", 20, "h20", b"second")
    s.put_snapshot("s-a", "w", 10, "h10", blob)
    s.put_snapshot("s-c", "v", 99, "h99", b"other world")
    s.put_snapshot("s-a", "w", 10, "h10", b"overwrite attempt")   # snapshots are immutable
    meta, data = s.get_snapshot("s-a")
    assert data == blob and isinstance(data, bytes)
    assert {k: meta[k] for k in ("id", "world", "seq", "hash", "bytes")} == \
        {"id": "s-a", "world": "w", "seq": 10, "hash": "h10", "bytes": len(blob)}
    assert [m["id"] for m in s.list_snapshots("w")] == ["s-a", "s-b"]
    assert s.latest_snapshot("w")["id"] == "s-b" and s.latest_snapshot("v")["id"] == "s-c"
    with pytest.raises(KeyError):
        s.get_snapshot("s-missing")
    assert backend.open().get_snapshot("s-b")[1] == b"second"


def test_leases(backend):
    s, s2 = backend.open(), backend.open()
    s.create_world("w", {}, None)
    assert s.acquire("w", "a", ttl_s=30)
    assert not s2.acquire("w", "b", ttl_s=30)
    assert s.acquire("w", "a", ttl_s=30)            # renewal by the holder
    info = s2.world_info("w")
    assert info["writer"] == "a" and info["lease_until"] > time.time() + 20
    s2.release("w", "b")                            # not the holder: no effect
    assert not s2.acquire("w", "b")
    s.release("w", "a")
    assert s2.world_info("w")["writer"] is None
    assert s2.acquire("w", "b", ttl_s=0.2)
    assert not s.acquire("w", "a")
    time.sleep(0.35)                                # expiry
    assert s.acquire("w", "a", ttl_s=30)
    assert s.world_info("w")["writer"] == "a"


def test_blobs(backend):
    s = backend.open()
    assert s.get_blob("k") is None
    data = os.urandom(1 << 20)
    s.put_blob("k", data)
    s.put_blob("empty", b"")
    assert s.get_blob("k") == data and s.get_blob("empty") == b""
    s.put_blob("k", b"v2")                          # blobs are overwritable (content-addressed by callers)
    assert backend.open().get_blob("k") == b"v2"


# ====================================================================== kernel on top of the store
def _script(w: World, start: int, stop: int) -> None:
    """Deterministic command mix (joins, transfers incl. failing ones, orgs, time)."""
    for i in range(start, stop):
        k = i % 7
        if k == 0:
            w.system_call("admin.join", {"handle": f"u{i}", "endowment": 1000 + i})
        elif k in (1, 2, 3):
            users = sorted(a for a in w.state.ids("principals") if a.startswith("p:"))
            if len(users) >= 2:
                a, b = users[i % len(users)], users[(i + 1) % len(users)]
                src = w.modules["bank"].primary_account(w.state, a)
                w.execute(a, {"tool": "bank.transfer", "args": {"from_account": src, "to": b, "amount": 1 + (i * 37) % 900}})
        elif k == 4:
            users = sorted(a for a in w.state.ids("principals") if a.startswith("p:"))
            if users:
                w.execute(users[0], {"tool": "id.create_org", "args": {"handle": f"org{i}", "name": f"Org {i}"}})
        elif k == 5:
            w.advance_time(w.now + 600)
        else:
            w.execute("p:u0", {"tool": "bank.transfer", "args": {"from_account": "acct_000001", "to": "nobody", "amount": 5}})


def test_world_fork_lineage(backend):
    s = backend.open()
    w = World.create(SPEC, "root", store=s)
    _script(w, 0, 15)
    f1 = w.fork("f1", store=s)
    _script(f1, 15, 25)
    f2 = f1.fork("f2", store=s)
    _script(f2, 25, 30)
    _script(w, 30, 35)
    for x in (w, f1, f2):
        x.flush()
    assert [i["id"] for i in s.lineage("f2")] == ["f2", "f1", "root"]
    assert [i["id"] for i in s.lineage("root")] == ["root"]
    assert s.world_info("f1")["parent"]["world"] == "root"
    assert s.world_info("f2")["parent"]["world"] == "f1"
    s2 = backend.open()
    for x in (w, f1, f2):
        y = World.load(s2, x.world_id)
        assert (y.digest(), y.last_hash, y.seq) == (x.digest(), x.last_hash, x.seq)
    assert len({w.digest(), f1.digest(), f2.digest()}) == 3


def _crash_writer(store_spec: tuple, conn) -> None:
    """Child process: write through a World, snapshot, write, flush, write more, then die by SIGKILL."""
    store = open_store(store_spec)
    w = World.create(SPEC, "crash", store=store)
    _script(w, 0, 20)
    sid = w.snapshot()
    _script(w, 20, 40)
    w.flush()
    conn.send({"sid": sid, "snap_seq": store.latest_snapshot("crash")["seq"], "digest": w.digest(),
               "hash": w.last_hash, "seq": w.seq})
    _script(w, 40, 48)                               # never flushed: lost in the crash
    conn.send({"unflushed_seq": w.seq})
    os.kill(os.getpid(), signal.SIGKILL)


def _oracle(stop_flush: int = 40) -> World:
    w = World.create(SPEC, "crash")
    _script(w, 0, stop_flush)
    return w


def test_world_load_after_crash(backend):
    if backend.kind == "memory":
        # in-process "crash": the World object is dropped without close()
        s = backend.open()
        w = World.create(SPEC, "crash", store=s)
        _script(w, 0, 20)
        sid = w.snapshot()
        _script(w, 20, 40)
        w.flush()
        info = {"sid": sid, "digest": w.digest(), "hash": w.last_hash, "seq": w.seq}
        _script(w, 40, 48)
        assert w.seq > info["seq"]
        del w
    else:
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe(duplex=False)
        p = ctx.Process(target=_crash_writer, args=(backend.spec, child))
        p.start()
        info, lost = parent.recv(), parent.recv()
        p.join(60)
        assert p.exitcode == -signal.SIGKILL
        assert lost["unflushed_seq"] > info["seq"]
    s = backend.open()
    assert s.last_seq("crash") == info["seq"]
    assert s.latest_snapshot("crash")["id"] == info["sid"] and s.latest_snapshot("crash")["seq"] < info["seq"]
    w2 = World.load(s, "crash")                      # latest snapshot + replay of the events after it
    assert (w2.digest(), w2.last_hash, w2.seq) == (info["digest"], info["hash"], info["seq"])
    full = World.replay(SPEC, s.read_events("crash"))  # and the same from genesis
    assert (full.digest(), full.last_hash) == (info["digest"], info["hash"])
    oracle = _oracle()                               # and the same as an in-memory run of the durable prefix
    assert (oracle.digest(), oracle.last_hash) == (info["digest"], info["hash"])
    assert w2.check_invariants() == [] and w2.modules["bank"].check_ledger_consistency(w2) == []
    # the recovered world carries on writing where the durable log ends
    _script(w2, 40, 48)
    w2.flush()
    _script(oracle, 40, 48)
    assert (w2.digest(), w2.last_hash) == (oracle.digest(), oracle.last_hash)
    assert World.load(backend.open(), "crash").last_hash == oracle.last_hash


def test_second_world_writer_is_rejected(backend):
    s1, s2 = backend.open(), backend.open()
    w1 = World.create(SPEC, "shared", store=s1)
    _script(w1, 0, 8)
    w1.flush()
    w2 = World.load(s2, "shared")
    assert w2.last_hash == w1.last_hash
    assert s1.acquire("shared", "kernel-1") and not s2.acquire("shared", "kernel-2")
    # even with the lease bypassed, contiguity stops the second writer from forking history in place
    _script(w1, 8, 12)
    _script(w2, 8, 12)
    w1.flush()
    with pytest.raises(StoreConflict):
        w2.flush()
    w3 = World.load(backend.open(), "shared")
    assert (w3.digest(), w3.last_hash) == (w1.digest(), w1.last_hash)


# ====================================================================== concurrency
ROUNDS = 25


def _race(store, world_id: str, writer: str, base: int, rounds: int, barrier, out: list) -> None:
    """Every round, all writers try to append the same next seq at the same moment."""
    try:
        for r in range(rounds):
            barrier.wait()
            seq = base + r + 1
            try:
                store.append_events(world_id, [ev(seq, writer=writer, round=r)])
                out.append((r, writer, "ok"))
            except StoreConflict:
                out.append((r, writer, "conflict"))
    finally:
        store.close()


def _check_race(store, world_id: str, base: int, rounds: int, results: list, writers: list[str]) -> None:
    by_round: dict[int, list] = {}
    for r, wr, outcome in results:
        by_round.setdefault(r, []).append((wr, outcome))
    assert sorted(by_round) == list(range(rounds))
    winners = {}
    for r, outs in by_round.items():
        assert sorted(o for _, o in outs) == ["conflict"] * (len(writers) - 1) + ["ok"], (r, outs)
        winners[r] = next(wr for wr, o in outs if o == "ok")
    evs = store.read_events(world_id, from_seq=base)
    assert [e["seq"] for e in evs] == list(range(base + 1, base + rounds + 1))
    assert all(e["payload"]["writer"] == winners[e["payload"]["round"]] for e in evs)


def test_concurrent_writer_threads(backend):
    s = backend.open()
    s.create_world("race", {}, None)
    s.append_events("race", [ev(1), ev(2)])
    writers = ["t0", "t1", "t2"]
    barrier = threading.Barrier(len(writers))
    results: list = []
    stores = [backend.open() for _ in writers]
    ts = [threading.Thread(target=_race, args=(st, "race", wr, 2, ROUNDS, barrier, results))
          for st, wr in zip(stores, writers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(120)
    assert not any(t.is_alive() for t in ts)
    _check_race(backend.open(), "race", 2, ROUNDS, results, writers)


def _race_proc(store_spec: tuple, writer: str, base: int, rounds: int, barrier, q) -> None:
    out: list = []
    _race(open_store(store_spec), "race", writer, base, rounds, barrier, out)
    q.put(out)


def test_concurrent_writer_processes(mp_backend):
    s = mp_backend.open()
    s.create_world("race", {}, None)
    s.append_events("race", [ev(1)])
    ctx = mp.get_context("spawn")
    writers = ["p0", "p1"]
    barrier, q = ctx.Barrier(len(writers), timeout=120), ctx.Queue()
    ps = [ctx.Process(target=_race_proc, args=(mp_backend.spec, wr, 1, ROUNDS, barrier, q)) for wr in writers]
    for p in ps:
        p.start()
    results = [x for _ in ps for x in q.get(timeout=180)]
    for p in ps:
        p.join(60)
        assert p.exitcode == 0
    _check_race(s, "race", 1, ROUNDS, results, writers)


# ====================================================================== throughput (printed, not asserted)
def test_throughput_benchmark(backend, capsys):
    s = backend.open()
    s.create_world("bench", {}, None)
    n, batch = 20_000, 500
    evs = [{"seq": i, "t": i, "type": "bank.txn", "actor": "p:alice", "cmd": i // 3,
            "payload": {"id": f"tx_{i:06d}", "postings": [["acct_000001", -125], ["acct_000002", 125]],
                        "kind": "transfer", "memo": "benchmark payment " * 3, "ref": None, "by": "p:alice"},
            "vis": ["p:alice", "p:bob"], "hash": f"{i:032x}"} for i in range(1, n + 1)]
    t0 = time.perf_counter()
    for i in range(0, n, batch):
        s.append_events("bench", evs[i:i + batch])
    t_app = time.perf_counter() - t0
    single = 300
    t0 = time.perf_counter()
    for i in range(n + 1, n + single + 1):
        s.append_events("bench", [dict(evs[0], seq=i)])
    t_single = time.perf_counter() - t0
    t0 = time.perf_counter()
    got = s.read_events("bench")
    t_read = time.perf_counter() - t0
    assert len(got) == n + single
    line = (f"[storage-bench] {backend.kind:8s} append(batch={batch}) {n / t_app:10,.0f} ev/s | "
            f"append(batch=1) {single / t_single:8,.0f} ev/s | read {len(got) / t_read:10,.0f} ev/s")
    with capsys.disabled():
        print("\n" + line)
