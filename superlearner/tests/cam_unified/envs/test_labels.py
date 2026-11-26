"""Trainer-only label channel: released after resolution, once, without private keys (OB-03, SE-03)."""
import pytest

from superlearner.cam_unified.envs.labels import LabelRecord, LabelStore


def test_label_store_appends_once_rejects_pending_and_private_keys_and_survives_reopen(tmp_path):
    s = LabelStore(str(tmp_path))
    r = LabelRecord("rapid:x#0", 1, "success", 0.7, 1234.0, 2.5, "crucible.rapid")
    assert s.append(r) and not s.append(r)
    with pytest.raises(ValueError):
        s.append(LabelRecord("rapid:x#0", 2, "pending", None, None, None, "e"))
    with pytest.raises(ValueError):
        s.append(LabelRecord("rapid:x#0", 3, "failure", 0.0, 1.0, 1.0, "e", meta={"oracle": 5.0}))
    s2 = LabelStore(str(tmp_path))
    assert len(s2) == 1 and not s2.append(r)
    assert list(s2.by_episode()) == [("rapid:x#0", 1)]
