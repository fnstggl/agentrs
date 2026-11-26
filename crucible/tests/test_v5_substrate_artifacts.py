"""Crucible 5.0 substrate: content-addressed artifacts (contract 1.1)."""
import base64
import hashlib
import os

import numpy as np
import pytest

import crucible
from crucible.core.errors import CapacityExceeded, InvalidAction, NotFound
from crucible.core.kernel import World, WorldSpec
from crucible.storage.memory import MemoryStore
from crucible.storage.sqlite import SQLiteStore
from crucible.substrate import artifacts as A
from crucible.worldgen.admission import admit

crucible.load_all_modules()


def spec(art_cfg=None, seed=1, extra=()):
    mods = [{"name": "identity", "config": {}}, {"name": "bank", "config": {}}, {"name": "admin", "config": {}},
            {"name": "artifacts", "config": art_cfg or {}}, *extra]
    return WorldSpec("art-test", seed, mods)


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def world(art_cfg=None, store=None, strict=True, endow=100_000):
    w = World.create(spec(art_cfg), store=store, strict=strict)
    for h in ("alice", "bob", "carol"):
        w.system_call("admin.join", {"handle": h, "endowment": endow})
    return w


def put(w, actor, data: bytes, mime="text/plain", **kw):
    r = w.execute(actor, {"tool": "artifact.put", "args": {"data_b64": b64(data), "mime": mime, **kw}})
    return r


def test_put_is_content_addressed_idempotent_and_verified():
    w = world()
    r = put(w, "p:alice", b"hello world")
    assert r.ok, r.error
    ref = r.result["ref"]
    assert ref == "sha256:" + hashlib.sha256(b"hello world").hexdigest()
    assert r.result["new"] is True
    n = w.seq
    r2 = put(w, "p:alice", b"hello world")
    assert r2.ok and r2.result["ref"] == ref and r2.result["new"] is False
    # the second put records the command but no new artifact event
    assert [e.type for e in w.log[n:]].count("artifact.stored") == 0
    assert A.get(w, ref) == b"hello world"
    i = A.info(w, ref)
    assert i["owner"] == "p:alice" and i["size"] == 11 and i["mime"] == "text/plain" and i["kind"] == "file"
    # bytes are not in the event log (only the REF and metadata)
    stored = [e for e in w.log if e.type == "artifact.stored"]
    assert len(stored) == 1 and "data" not in stored[0].payload


def test_tampered_bytes_are_never_returned():
    st = MemoryStore()
    w = world(store=st)
    ref = put(w, "p:alice", b"original bytes").result["ref"]
    h = A.parse_ref(ref)
    st.put_blob(A.blob_key(h), b"substituted bytes")
    w.runtime.pop("artifacts", None)         # drop the verified cache: the next read goes to the store
    A._POOL.pop(h, None)
    with pytest.raises(NotFound):
        A.get(w, ref)


def test_permissions_owner_grant_publish_and_no_existence_oracle():
    w = world()
    ref = put(w, "p:alice", b"private notes").result["ref"]
    bob_get = w.execute("p:bob", {"tool": "artifact.get", "args": {"ref": ref}})
    missing = "sha256:" + "0" * 64
    bob_missing = w.execute("p:bob", {"tool": "artifact.get", "args": {"ref": missing}})
    # unreadable and nonexistent look the same
    assert not bob_get.ok and not bob_missing.ok
    assert bob_get.error["code"] == bob_missing.error["code"] == "not_found"
    assert not w.execute("p:bob", {"tool": "artifact.info", "args": {"ref": ref}}).ok
    assert not A.readable(w, "p:bob", ref) and A.readable(w, "p:alice", ref)
    # bob cannot publish or grant alice's artifact
    assert not w.execute("p:bob", {"tool": "artifact.publish", "args": {"ref": ref}}).ok
    assert w.execute("p:alice", {"tool": "artifact.grant", "args": {"ref": ref, "grantee": "p:bob"}}).ok
    g = w.execute("p:bob", {"tool": "artifact.get", "args": {"ref": ref}})
    assert g.ok and base64.b64decode(g.result["data_b64"]) == b"private notes"
    assert not A.readable(w, "p:carol", ref)
    assert w.execute("p:alice", {"tool": "artifact.revoke", "args": {"ref": ref, "grantee": "p:bob"}}).ok
    assert not A.readable(w, "p:bob", ref)
    assert w.execute("p:alice", {"tool": "artifact.publish", "args": {"ref": ref}}).ok
    assert A.readable(w, "p:carol", ref) and A.info(w, ref)["published"]
    pub = [e for e in w.log if e.type == "artifact.published"]
    assert pub and pub[-1].vis == "public"
    # the private stored event was visible to the owner only
    st_ev = [e for e in w.log if e.type == "artifact.stored"][0]
    assert st_ev.vis == ("p:alice",)


def test_holder_of_identical_bytes_sees_its_own_record_not_the_owner():
    w = world()
    ref = put(w, "p:alice", b"same bytes", meta={"title": "alice's"}).result["ref"]
    n = w.seq
    r = put(w, "p:bob", b"same bytes", mime="application/octet-stream", meta={"title": "bob's"})
    assert r.ok and r.result["ref"] == ref and r.result["new"] is True     # new for bob: nothing leaks
    seen_dup = [(e.type, sorted(e.payload)) for e in w.log[n - w.log_base:] if w.visible(e, {"p:bob"})]
    n = w.seq
    fresh = put(w, "p:bob", b"bytes nobody had")
    seen_new = [(e.type, sorted(e.payload)) for e in w.log[n - w.log_base:] if w.visible(e, {"p:bob"})]
    assert seen_dup == seen_new and fresh.result["new"] is True     # identical observations: no existence oracle
    assert put(w, "p:bob", b"same bytes").result["new"] is False
    v = w.execute("p:bob", {"tool": "artifact.info", "args": {"ref": ref}}).result
    assert v["owner"] == "p:bob" and v["meta"] == {"title": "bob's"} and v["mime"] == "application/octet-stream"
    assert A.info(w, ref)["owner"] == "p:alice"
    # both pay quota for what they stored
    assert A.usage(w, "p:alice")["bytes"] == 10 and A.usage(w, "p:bob")["bytes"] == 10 + len(b"bytes nobody had")
    assert w.check_invariants() == []


def test_size_limit_quota_and_bad_input():
    w = world({"max_bytes": 1000, "quota_bytes": 2500})
    assert put(w, "p:alice", b"x" * 1000).ok
    big = put(w, "p:alice", b"y" * 1001)
    assert not big.ok and big.error["code"] == "capacity_exceeded"
    assert put(w, "p:alice", b"z" * 1000).ok
    over = put(w, "p:alice", b"w" * 600)
    assert not over.ok and over.error["code"] == "capacity_exceeded"
    assert A.usage(w, "p:alice") == {"owner": "p:alice", "bytes": 2000, "count": 2}
    bad = w.execute("p:alice", {"tool": "artifact.put", "args": {"data_b64": "not base64!", "mime": "text/plain"}})
    assert not bad.ok and bad.error["code"] == "invalid_action"
    bad_mime = w.execute("p:alice", {"tool": "artifact.put", "args": {"data_b64": b64(b"a"), "mime": "nonsense"}})
    assert not bad_mime.ok
    batch = put(w, "p:alice", b"q", kind="batch")
    assert not batch.ok           # 'batch' is reserved for system array blobs
    assert w.check_invariants() == []


def test_parents_provenance_must_be_readable():
    w = world()
    a = put(w, "p:alice", b"draft").result["ref"]
    b = put(w, "p:alice", b"final", parents=[a]).result["ref"]
    assert A.info(w, b)["parents"] == [a]
    # bob cannot claim lineage from alice's private artifact
    r = put(w, "p:bob", b"bob's remix", parents=[a])
    assert not r.ok and r.error["code"] == "not_found"
    w.execute("p:alice", {"tool": "artifact.publish", "args": {"ref": a}})
    assert put(w, "p:bob", b"bob's remix", parents=[a]).ok


def test_storage_fee_is_charged_through_bank_in_the_same_transaction():
    w = world({"fee_cents_per_mib": 100}, endow=150)
    data = b"m" * (1 << 20)                       # exactly 1 MiB -> 100 cents
    r = put(w, "p:alice", data)
    assert r.ok, r.error
    bank = w.modules["bank"]
    acct = bank.primary_account(w.state, "p:alice")
    assert bank.balance(w.state, acct) == 50
    td0, n_art = w.state.table_digests(), w.state.count("artifacts")
    r2 = put(w, "p:alice", b"n" * (1 << 20))      # cannot pay: nothing of the put survives
    assert not r2.ok and r2.error["code"] == "insufficient_funds"
    assert w.state.count("artifacts") == n_art and bank.balance(w.state, acct) == 50
    assert A.usage(w, "p:alice")["bytes"] == 1 << 20
    assert sum(a["balance"] for a in w.state.tables["accounts"].values()) == 0
    assert w.state.table_digests() == td0         # only the rejection itself is recorded (cmd.rejected)
    assert w.log[-1].type == "cmd.rejected"
    assert bank.check_ledger_consistency(w) == []


def test_array_blobs_are_npz_without_pickle_and_deterministic():
    arrays = {"idx": np.arange(10, dtype=np.int32), "v": np.linspace(0, 1, 10, dtype=np.float32)}
    b1, b2 = A.npz_bytes(arrays), A.npz_bytes({"v": arrays["v"].copy(), "idx": arrays["idx"].copy()})
    assert b1 == b2
    with np.load(__import__("io").BytesIO(b1), allow_pickle=False) as z:     # a standard npz
        assert set(z.files) == {"idx", "v"}
    back = A.npz_arrays(b1)
    assert np.array_equal(back["idx"], arrays["idx"]) and back["v"].dtype == np.float32
    assert not back["v"].flags.writeable
    with pytest.raises(InvalidAction):
        A.npz_bytes({"o": np.array([object()], dtype=object)})


def _log_events(w):
    return [e.to_dict() for e in w.log]


def test_replay_regenerate_snapshot_fork_with_store_and_without():
    for store in (None, MemoryStore()):
        w = world(store=store)
        refs = [put(w, "p:alice", f"doc {i}".encode() * (i + 1)).result["ref"] for i in range(5)]
        w.execute("p:alice", {"tool": "artifact.publish", "args": {"ref": refs[0]}})
        w.execute("p:alice", {"tool": "artifact.grant", "args": {"ref": refs[1], "grantee": "p:bob"}})
        put(w, "p:bob", b"doc 0")      # holder
        evs = _log_events(w)
        r = World.replay(w.spec, evs)
        assert r.digest() == w.digest()
        g = World.regenerate(w.spec, evs)
        assert g.last_hash == w.last_hash
        f = w.fork("art-fork")
        assert f.digest() == w.digest()
        assert A.get(f, refs[3]) == A.get(w, refs[3])
        x = put(f, "p:carol", b"only in the fork").result["ref"]
        assert A.exists(f, x) and not A.exists(w, x)
        with pytest.raises(NotFound):
            A.get(w, x)
        r2 = World.from_blob(w.snapshot_blob())
        assert r2.digest() == w.digest() and A.get(r2, refs[4]) == A.get(w, refs[4])


def test_sqlite_store_load_in_fresh_world_reads_blobs_from_store(tmp_path):
    path = str(tmp_path / "w.db")
    st = SQLiteStore(path)
    w = world(store=st)
    ref = put(w, "p:alice", os.urandom(3000)).result["ref"]
    data = A.get(w, ref)
    w.flush()
    wid, d = w.world_id, w.digest()
    w.close()
    st.close()
    for h in list(A._POOL):
        A._POOL.pop(h)
    st2 = SQLiteStore(path)
    w2 = World.load(st2, wid)
    assert w2.digest() == d
    assert A.get(w2, ref) == data
    # export to a different store
    dest = MemoryStore()
    rep = A.export(w2, dest)
    assert rep["missing"] == [] and rep["copied"] == 1 and dest.get_blob(A.blob_key(A.parse_ref(ref))) == data
    st2.close()


def test_artifacts_module_passes_admission():
    rep = admit(["artifacts"], steps=300)
    assert rep.passed, (rep.checks, rep.errors[:5])
