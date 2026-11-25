"""MemoryIndex: exact top-k vs brute force, metadata prefilters, version isolation, stale detection + re-encoding from
raw events, crash truncation, and delayed-cue retrieval under heavy interference (synthetic keys)."""
from __future__ import annotations

import os
import time

import numpy as np
import pytest
import torch

from memhelpers import light_events
from superlearner.cam_unified.memory.archive import EventArchive
from superlearner.cam_unified.memory.errors import IncompatibleMemoryVersion
from superlearner.cam_unified.memory.index import MemoryIndex
from superlearner.cam_unified.memory.retrieval import Retriever
from superlearner.cam_unified.testing import tiny_config

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _fill(idx, version, n, kd, T, D, seed=0, envs=3):
    rng = np.random.default_rng(seed)
    keys = _unit(rng.standard_normal((n, kd))).astype(np.float32)
    vals = rng.standard_normal((n, T, D)).astype(np.float32)
    metas = [{"seg_id": f"s{i:06d}", "level": 1 + (i % 4 == 0), "env_id": f"env{i % envs}", "episode_id": f"ep{i // 50}",
              "t_start": float(i), "t_end": float(i + 1), "gseq_start": i * 10, "gseq_end": i * 10 + 9,
              "event_ids": [f"e{i}_{j}" for j in range(3)], "entities": [f"ent{i % 7}", f"ent{(i * 3) % 11}"]}
             for i in range(n)]
    idx.add_segments(metas)
    for i in range(0, n, 997):                       # several add() calls crossing shard boundaries
        idx.add(version, [m["seg_id"] for m in metas[i:i + 997]], keys[i:i + 997], vals[i:i + 997])
    return keys, vals, metas


def _brute(keys16, q, k, allowed=None):
    s = q.astype(np.float32) @ keys16.astype(np.float32).T
    if allowed is not None:
        s = np.where(allowed[None, :], s, -np.inf)
    order = np.argsort(-s, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(s, order, 1)


@pytest.mark.parametrize("device", DEVICES)
def test_topk_equals_brute_force(tmp_path, device):
    kd, T, D, n = 32, 2, 16, 5000
    idx = MemoryIndex(str(tmp_path / "i"), kd, T, D, shard_rows=1500, device=device, search_chunk=700)
    keys, vals, metas = _fill(idx, "v1", n, kd, T, D)
    q = _unit(np.random.default_rng(1).standard_normal((9, kd))).astype(np.float32)
    res = idx.search("v1", torch.from_numpy(q), 10)
    order, sc = _brute(keys.astype(np.float16), q, 10)
    assert (res.kids == order).all()
    np.testing.assert_allclose(res.scores.numpy(), sc, rtol=1e-5, atol=1e-5)
    assert res.seg_ids[3] == [f"s{i:06d}" for i in order[3]]
    v = idx.values("v1", res.kids[0])
    np.testing.assert_allclose(v.numpy(), vals[order[0]].astype(np.float16).astype(np.float32))
    idx.close()


@pytest.mark.parametrize("device", DEVICES)
def test_filtered_topk_equals_brute_force(tmp_path, device):
    kd, T, D, n = 32, 2, 8, 4000
    idx = MemoryIndex(str(tmp_path / "i"), kd, T, D, shard_rows=1024, device=device, search_chunk=512)
    keys, _vals, metas = _fill(idx, "v1", n, kd, T, D)
    q = _unit(np.random.default_rng(2).standard_normal((5, kd))).astype(np.float32)
    filters = [{"env_id": "env1"},                                        # broad -> mask path
               {"t_min": 100.0, "t_max": 180.0},                          # narrow -> gather path
               {"entity": "ent3", "level": 1},
               None,
               {"exclude_episode": ["ep0", "ep1"], "gseq_max": 30_000, "env_id": ["env0", "env2"]}]

    def ok(m, f):
        if f is None:
            return True
        return ((f.get("env_id") is None or m["env_id"] in ([f["env_id"]] if isinstance(f["env_id"], str)
                                                             else f["env_id"]))
                and (f.get("t_min") is None or m["t_end"] >= f["t_min"])
                and (f.get("t_max") is None or m["t_start"] <= f["t_max"])
                and (f.get("entity") is None or f["entity"] in m["entities"])
                and (f.get("level") is None or m["level"] == f["level"])
                and (f.get("exclude_episode") is None or m["episode_id"] not in f["exclude_episode"])
                and (f.get("gseq_max") is None or m["gseq_end"] <= f["gseq_max"]))
    res = idx.search("v1", torch.from_numpy(q), 7, filters)
    for b, f in enumerate(filters):
        allowed = np.array([ok(m, f) for m in metas])
        order, sc = _brute(keys.astype(np.float16), q[b:b + 1], 7, allowed)
        assert (res.kids[b] == order[0]).all(), (b, f)
        assert all(ok(metas[k], f) for k in res.kids[b])
    empty = idx.search("v1", torch.from_numpy(q[:1]), 3, {"env_id": "nowhere"})
    assert (empty.kids == -1).all() and torch.isinf(empty.scores).all() and empty.seg_ids == [[None] * 3]
    with pytest.raises(ValueError, match="unknown memory filter"):
        idx.search("v1", torch.from_numpy(q[:1]), 3, {"colour": "red"})
    idx.close()


def test_versions_are_isolated(tmp_path):
    kd, T, D = 16, 1, 4
    idx = MemoryIndex(str(tmp_path / "i"), kd, T, D, shard_rows=100, device="cpu")
    k1, _, metas = _fill(idx, "old", 300, kd, T, D, seed=0)
    target = metas[42]["seg_id"]
    rng = np.random.default_rng(5)
    k2 = _unit(rng.standard_normal((300, kd))).astype(np.float32)
    q = k2[42:43] + 0.01 * rng.standard_normal((1, kd)).astype(np.float32)     # matches seg 42's NEW key
    idx.add("new", [target], k2[42:43], np.zeros((1, T, D)))
    assert idx.search("new", torch.from_numpy(q), 1).seg_ids == [[target]]
    old = idx.search("old", torch.from_numpy(q), 5)
    order, _ = _brute(k1.astype(np.float16), q, 5)
    assert (old.kids == order).all()                 # computed against old keys only, never the new one
    assert idx.coverage("new") == pytest.approx(1 / 300) and len(idx.missing("new")) == 299
    with pytest.raises(ValueError, match="re-encode"):
        idx.set_current("new")
    with pytest.raises(KeyError):
        idx.search("never", torch.from_numpy(q), 1)
    idx.set_current("old")
    assert idx.stale_versions() == ["new"]
    idx.close()


def test_retriever_refuses_keys_of_an_unknown_encoder_version(tmp_path):
    cfg = tiny_config(mem_segment=4, mem_segment_tokens=2)
    idx = MemoryIndex(str(tmp_path / "i"), cfg.mem_key_dim, 2, cfg.d_model, device="cpu")
    _fill(idx, "seg-7-deadbeefdeadbeef", 20, cfg.mem_key_dim, 2, cfg.d_model)
    idx.set_current("seg-7-deadbeefdeadbeef")
    ret = Retriever(cfg, idx)
    with pytest.raises(IncompatibleMemoryVersion):
        ret.retrieve(torch.randn(2, cfg.mem_key_dim), 3)
    idx.close()


def test_stale_version_reencoded_from_raw_events(tmp_path):
    """The callback recomputes keys from the archived raw events; the new version then serves searches alone."""
    arc = EventArchive(str(tmp_path / "a"), chunk_rows=200)
    evs = light_events(400)
    arc.append(evs)
    kd, T, D = 8, 1, 3
    idx = MemoryIndex(str(tmp_path / "i"), kd, T, D, shard_rows=7, device="cpu")
    segs = [{"seg_id": f"s{i}", "level": 1, "event_ids": [e.event_id for e in evs[i * 10:(i + 1) * 10]],
             "t_start": evs[i * 10].t_sim, "t_end": evs[i * 10 + 9].t_sim, "env_id": "env0"} for i in range(40)]
    idx.add_segments(segs)

    def encoder(scale):
        def fn(metas):
            ks, vs = [], []
            for m in metas:
                recs = arc.get(m["event_ids"])
                v = np.array([r.observations[0].value for r in recs], dtype=np.float32)
                ks.append(np.concatenate([v[:4], v[4:8]]) * scale / 100.0)
                vs.append(np.array([[v.mean(), v.min(), v.max()]], dtype=np.float32))
            return np.stack(ks), np.stack(vs)
        return fn
    idx.reencode("A", encoder(1.0))
    idx.set_current("A")
    assert sorted(idx.missing("B")) == sorted(f"s{i}" for i in range(40))
    n = idx.reencode("B", encoder(-1.0), batch=6, max_items=13)
    assert n == 13 and 0 < idx.coverage("B") < 1                       # background migration, step by step
    while idx.missing("B"):
        idx.reencode("B", encoder(-1.0), batch=6, max_items=13)
    want, _ = encoder(-1.0)(idx.segments([f"s{i}" for i in range(40)]))
    np.testing.assert_allclose(idx.keys("B", idx.kids_of("B", [f"s{i}" for i in range(40)])).numpy(),
                               want.astype(np.float16).astype(np.float32))
    idx.set_current("B")
    assert idx.stale_versions() == ["A"]
    idx.drop_version("A")
    assert idx.versions() == {"B": 40} and idx.stale_versions() == []
    with pytest.raises(KeyError):
        idx.search("A", torch.zeros(1, kd), 1)
    idx.close()
    arc.close()


def test_persistence_and_truncation_of_uncommitted_rows(tmp_path):
    kd, T, D = 16, 2, 4
    root = str(tmp_path / "i")
    idx = MemoryIndex(root, kd, T, D, shard_rows=50, device="cpu")
    keys, _, _ = _fill(idx, "v", 120, kd, T, D)
    q = torch.from_numpy(_unit(np.random.default_rng(3).standard_normal((4, kd))).astype(np.float32))
    before = idx.search("v", q, 6)
    idx.close()
    vdir = os.path.join(root, "versions", os.listdir(os.path.join(root, "versions"))[0])
    with open(os.path.join(vdir, "keys_00002.f16"), "ab") as fh:        # rows written, commit never happened
        fh.write(np.ones((5, kd), np.float16).tobytes())
    with open(os.path.join(vdir, "vals_00007.f16"), "wb") as fh:        # a whole uncommitted shard
        fh.write(np.ones((3, T * D), np.float16).tobytes())
    re = MemoryIndex(root, kd, T, D, shard_rows=50, device="cpu")
    assert os.path.getsize(os.path.join(vdir, "keys_00002.f16")) == 20 * kd * 2
    assert os.path.getsize(os.path.join(vdir, "vals_00007.f16")) == 0
    after = re.search("v", q, 6)
    assert (after.kids == before.kids).all() and re.count("v") == 120
    with pytest.raises(ValueError):
        MemoryIndex(root, kd + 1, T, D)                                 # dims are part of the index identity
    re.close()


@pytest.mark.parametrize("device", DEVICES[-1:])
def test_exact_search_finds_cues_among_200k_interfering_keys(tmp_path, device):
    """Index-level check (not a learned-retention test; see test_mem_retrieval for learned delayed cues through the
    archive): exact search over sharded fp16 keys keeps finding cue keys as up to 200k interfering keys (random +
    'near' distractors at cosine ~0.6) are added, i.e. nothing is evicted or approximated away; reports latency."""
    kd, T, D = 64, 1, 4
    idx = MemoryIndex(str(tmp_path / "i"), kd, T, D, shard_rows=65536, device=device)
    rng = np.random.default_rng(0)
    n_cues = 64
    cues = _unit(rng.standard_normal((n_cues, kd))).astype(np.float32)
    total = 0

    def add(keys):
        nonlocal total
        ids = [f"k{total + i}" for i in range(len(keys))]
        idx.add_segments([{"seg_id": s, "level": 1, "t_start": float(total + i), "t_end": float(total + i)}
                          for i, s in enumerate(ids)])
        idx.add("v", ids, keys, np.zeros((len(keys), T, D), np.float32))
        total += len(keys)
    add(cues)
    report = {}
    for delay in (10, 1_000, 20_000, 200_000):
        while total - n_cues < delay:
            m = min(50_000, delay - (total - n_cues))
            noise = _unit(rng.standard_normal((m, kd))).astype(np.float32)
            near = cues[rng.integers(0, n_cues, m // 10)]
            noise[: m // 10] = _unit(0.75 * near + 0.66 * _unit(rng.standard_normal((m // 10, kd))))
            add(noise)
        q = _unit(cues + 0.45 * _unit(rng.standard_normal((n_cues, kd)))).astype(np.float32)
        t0 = time.time()
        res = idx.search("v", torch.from_numpy(q), 5)
        lat = time.time() - t0
        hit1 = float(np.mean([res.seg_ids[i][0] == f"k{i}" for i in range(n_cues)]))
        hit5 = float(np.mean([f"k{i}" in res.seg_ids[i] for i in range(n_cues)]))
        report[delay] = (hit1, hit5, lat)
        assert hit5 >= 0.95, (delay, report)
    print("delayed cue retrieval (hit@1, hit@5, search seconds for 64 queries) by #interfering keys:", report)
    idx.close()


@pytest.mark.parametrize("device", DEVICES)
def test_nonfinite_keys_are_refused_and_nonfinite_queries_match_nothing(tmp_path, device):
    kd, T, D = 8, 1, 4
    idx = MemoryIndex(str(tmp_path / "i"), kd, T, D, device=device)
    keys = _unit(np.random.default_rng(0).standard_normal((100, kd))).astype(np.float32)
    idx.add_segments([{"seg_id": f"s{i}", "level": 1} for i in range(100)])
    bad = keys.copy()
    bad[37] = np.nan
    with pytest.raises(ValueError, match="non-finite keys"):
        idx.add("v", [f"s{i}" for i in range(100)], bad, np.zeros((100, T, D)))
    big = np.zeros((100, T, D), np.float32)
    big[3] = 1e6                                             # beyond fp16: would be stored as inf
    with pytest.raises(ValueError, match="non-finite values"):
        idx.add("v", [f"s{i}" for i in range(100)], keys, big)
    assert idx.count("v") == 0
    idx.add("v", [f"s{i}" for i in range(100)], keys, np.zeros((100, T, D)))
    q = torch.from_numpy(np.stack([keys[5], np.full(kd, np.nan, np.float32), keys[9]]))
    r = idx.search("v", q, 3)
    assert r.kids[0, 0] == 5 and r.kids[2, 0] == 9
    assert (r.kids[1] == -1).all() and torch.isinf(r.scores[1]).all() and r.seg_ids[1] == [None] * 3
    assert not torch.isnan(r.scores).any() and idx.stats["nonfinite_queries"] == 1
    idx.close()


def test_transaction_rollback_restores_the_index_exactly(tmp_path):
    kd, T, D = 8, 2, 4
    root = str(tmp_path / "i")
    idx = MemoryIndex(root, kd, T, D, shard_rows=16, device="cpu")
    keys, vals, metas = _fill(idx, "v", 40, kd, T, D)
    q = torch.from_numpy(_unit(np.random.default_rng(5).standard_normal((3, kd))).astype(np.float32))
    before = idx.search("v", q, 5)
    n_seg = idx.count()
    new = [{"seg_id": f"n{i}", "level": 1, "t_start": 100.0 + i, "t_end": 101.0 + i} for i in range(30)]
    rng = np.random.default_rng(9)
    with pytest.raises(RuntimeError, match="boom"):
        with idx.transaction():
            idx.add_segments(new)
            idx.add("v", [m["seg_id"] for m in new], _unit(rng.standard_normal((30, kd))), rng.standard_normal((30, T, D)))
            idx.add_segments([{"seg_id": "w0", "level": 1}])
            idx.add("brand-new-version", ["w0"], _unit(rng.standard_normal((1, kd))), np.zeros((1, T, D)))
            idx.set_state("cursor", 123)
            assert idx.count("v") == 70                       # visible inside the transaction
            raise RuntimeError("boom")
    assert idx.count() == n_seg and idx.count("v") == 40 and idx.get_state("cursor") is None
    assert "brand-new-version" not in idx.versions() and idx.stats["rollbacks"] == 1
    after = idx.search("v", q, 5)
    assert (after.kids == before.kids).all()
    vdirs = os.listdir(os.path.join(root, "versions"))
    assert len(vdirs) == 1                                    # the uncommitted version's files are gone
    shard = os.path.join(root, "versions", vdirs[0], "keys_00002.f16")
    assert os.path.getsize(shard) == 8 * kd * 2               # rows 32..39 only; the 30 rolled-back rows were cut
    idx.add_segments(new)                                     # the index keeps working after the rollback
    k = idx.add("v", [m["seg_id"] for m in new], _unit(rng.standard_normal((30, kd))), rng.standard_normal((30, T, D)))
    assert k.tolist() == list(range(40, 70))
    idx.close()


def test_pinned_versions_are_not_dropped(tmp_path):
    kd, T, D = 8, 1, 4
    idx = MemoryIndex(str(tmp_path / "i"), kd, T, D, device="cpu")
    _fill(idx, "v0", 20, kd, T, D)
    _fill(idx, "v1", 20, kd, T, D, seed=1)
    idx.set_current("v1")
    idx.pin("ckpt:/x/life", ["v0"])
    with pytest.raises(ValueError, match="pinned"):
        idx.drop_version("v0")
    assert idx.pinned_versions() == {"v0"} and idx.pointer()["pins"] == {"ckpt:/x/life": ["v0"]}
    idx.unpin("ckpt:/x/life")
    idx.drop_version("v0")
    assert list(idx.versions()) == ["v1"]
    idx.close()
