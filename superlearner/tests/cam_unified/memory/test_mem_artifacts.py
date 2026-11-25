"""ArtifactStore: content addressing, deduplication, codecs, integrity, explicit limits, durability."""
from __future__ import annotations

import hashlib
import os

import pytest

from superlearner.cam_unified.memory.artifacts import ArtifactStore
from superlearner.cam_unified.memory.errors import StorageLimitExceeded


@pytest.mark.parametrize("codec", ["gzip", "lzma", "none"])
def test_roundtrip_and_content_address(tmp_path, codec):
    st = ArtifactStore(str(tmp_path / codec), codec=codec)
    doc = ("def solve(x):\n    return sorted(x)\n" * 200).encode()
    ref = st.put(doc)
    assert ref == "sha256:" + hashlib.sha256(doc).hexdigest()
    assert st.get(ref) == doc and st.exists(ref) and st.verify(ref)
    assert st.get_text(st.put("héllo wörld")) == "héllo wörld"
    assert st.get_json(st.put({"b": [1, 2], "a": 3})) == {"a": 3, "b": [1, 2]}
    if codec != "none":
        assert st.size_bytes() < len(doc)            # compressed on disk


def test_deduplication_and_codec_independent_refs(tmp_path):
    a = ArtifactStore(str(tmp_path / "a"), codec="gzip")
    blob = os.urandom(10_000) + b"x" * 50_000
    r1, r2 = a.put(blob), a.put(blob)
    assert r1 == r2 and a.stats["dedup_hits"] == 1 and len(a) == 1
    b = ArtifactStore(str(tmp_path / "b"), codec="lzma")
    assert b.put(blob) == r1                          # same reference whatever the codec
    assert a.put({"k": 1, "z": 2}) == a.put({"z": 2, "k": 1})   # canonical JSON
    assert ArtifactStore(str(tmp_path / "a")).get(r1) == blob   # survives reopening


def test_corruption_is_detected_not_returned(tmp_path):
    st = ArtifactStore(str(tmp_path / "s"), codec="none")
    ref = st.put(b"exact original bytes" * 20)
    h = ref[7:]
    p = os.path.join(str(tmp_path / "s"), "objects", h[:2], h[2:4], f"{h}.raw")
    with open(p, "r+b") as fh:
        fh.write(b"X")
    assert not st.verify(ref)
    with pytest.raises(IOError):
        st.get(ref)
    with pytest.raises(KeyError):
        st.get("sha256:" + "0" * 64)
    with pytest.raises(ValueError):
        st.get("md5:abc")


def test_explicit_storage_limit(tmp_path):
    st = ArtifactStore(str(tmp_path / "s"), codec="none", max_bytes=5_000)
    st.put(os.urandom(3_000))
    with pytest.raises(StorageLimitExceeded):
        st.put(os.urandom(3_000))
    assert len(st) == 1
    no_tmp = [f for r, _, fs in os.walk(str(tmp_path / "s")) for f in fs if f.startswith(".tmp")]
    assert no_tmp == []
