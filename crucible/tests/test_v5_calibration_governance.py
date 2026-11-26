"""Pseudonym keys are kept apart from the data they protect (SP-5.1; verifier note of the wave-3 repair pass)."""
import os

import pytest

from crucible.calibration import audit_sp5, gharchive as G


@pytest.fixture()
def roots(tmp_path, monkeypatch):
    data, keys = tmp_path / "data", tmp_path / "keys"
    (data / G.DATASET).mkdir(parents=True)
    monkeypatch.setenv("CRUCIBLE_DATA", str(data))
    monkeypatch.setenv(G.KEY_DIR_ENV, str(keys))
    return data, keys


def test_new_key_is_created_outside_the_data_root(roots):
    data, keys = roots
    k = G.pseudonym_key()
    assert len(k) == 32 and G.pseudonym_key() == k
    p = G.key_path()
    assert p.startswith(str(keys)) and oct(os.stat(p).st_mode)[-3:] == "600"
    assert not os.path.exists(G.legacy_key_path())
    assert audit_sp5.co_located_keys(str(data))["files"] == []


def test_legacy_key_in_the_dataset_directory_is_moved_unchanged(roots):
    data, keys = roots
    legacy = G.legacy_key_path()
    with open(legacy, "wb") as fh:
        fh.write(b"\x07" * 32)
    assert audit_sp5.co_located_keys(str(data))["files"] == [f"{G.DATASET}/{G.SALT_FILE}"]
    assert G.pseudonym_key(create=False) == b"\x07" * 32
    assert not os.path.exists(legacy) and os.path.exists(G.key_path())
    assert audit_sp5.co_located_keys(str(data))["files"] == []


def test_two_different_keys_are_refused(roots):
    G.pseudonym_key()
    with open(G.legacy_key_path(), "wb") as fh:
        fh.write(b"\x01" * 32)
    with pytest.raises(ValueError):
        G.pseudonym_key()


def test_key_directory_inside_the_data_root_is_refused(roots, monkeypatch):
    data, _ = roots
    monkeypatch.setenv(G.KEY_DIR_ENV, str(data / "keys"))
    with pytest.raises(ValueError):
        G.pseudonym_key()
