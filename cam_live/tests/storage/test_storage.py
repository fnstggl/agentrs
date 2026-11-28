"""Disk guard and retention policy."""
import os

import pytest

from cam_live import storage


def _ckpt(parent, name, optim=True, size=1000):
    d = os.path.join(parent, name)
    os.makedirs(os.path.join(d, "weights"))
    open(os.path.join(d, "weights", "experiential.safetensors"), "wb").write(b"w" * size)
    if optim:
        os.makedirs(os.path.join(d, "optim"))
        open(os.path.join(d, "optim", "experiential.pt"), "wb").write(b"o" * 2 * size)
    return d


def test_guard_refuses_before_writing_and_warns(tmp_path):
    free = storage.free_gb(str(tmp_path))
    g = storage.DiskGuard(str(tmp_path), min_free_gb=max(0.0, free - 1.0), warn_free_gb=free + 10)
    assert g.check(0)["level"] == "warn"
    with pytest.raises(storage.DiskSpaceLow) as e:
        g.check(int(2 * storage.GB), "checkpoint")
    assert "not writing" in str(e.value)
    assert storage.DiskGuard(str(tmp_path), min_free_gb=free + 1).status()["level"] == "low"
    assert storage.DiskGuard(str(tmp_path / "does" / "not" / "exist"), min_free_gb=0, warn_free_gb=0).check()["level"] == "ok"


def test_retention_keeps_records_newest_weights_and_protected(tmp_path):
    run = tmp_path / "run"
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    paths = []
    for i in range(5):
        p = _ckpt(str(ck), f"ckpt_{i:08d}")
        os.utime(p, (1000 + i, 1000 + i))
        paths.append(p)
    pre = _ckpt(str(run), "ckpt_pre")
    exp = _ckpt(str(run), "export_post", optim=False)
    for rec in ("decisions.jsonl", "experience.sqlite", "journal.md"):
        (run / rec).write_text("record")
    res = storage.apply_retention(str(run), keep=2, keep_optim=1)
    assert res["removed"] == 4 and res["bytes"] > 0          # 3 old checkpoints + optim of the second newest
    assert sorted(os.listdir(ck)) == ["ckpt_00000003", "ckpt_00000004"]
    assert os.path.isdir(os.path.join(paths[4], "optim")) and not os.path.isdir(os.path.join(paths[3], "optim"))
    assert os.path.isdir(os.path.join(pre, "optim")) and os.path.isdir(exp)
    assert all((run / r).read_text() == "record" for r in ("decisions.jsonl", "experience.sqlite", "journal.md"))
    res2 = storage.apply_retention(str(run), keep=1, keep_optim=0, drop_protected_optim=True)
    assert sorted(os.listdir(ck)) == ["ckpt_00000004"] and not os.path.isdir(os.path.join(pre, "optim"))
    assert os.path.isfile(os.path.join(pre, "weights", "experiential.safetensors"))
    with pytest.raises(ValueError):
        storage.apply_retention(str(run), keep=0)


def test_dry_run_and_drop_groups(tmp_path):
    run = tmp_path / "eval" / "E_F1_s0"
    for lt in ("lt00", "lt01"):
        _ckpt(str(run / lt / "checkpoints"), "ckpt_00000001")
    plan = storage.apply_retention(str(tmp_path), keep=1, drop_groups=("*/lt[0-9][0-9]/checkpoints",), dry_run=True)
    assert plan["removed"] == 2 and all(os.path.isdir(p) for p in plan["paths"])
    storage.apply_retention(str(tmp_path), keep=1, drop_groups=("*/lt[0-9][0-9]/checkpoints",))
    assert not any(os.path.isdir(p) for p in plan["paths"])


def test_retention_never_follows_a_symlinked_checkpoint_out_of_the_tree(tmp_path):
    import os
    from cam_live.storage import apply_retention
    outside = tmp_path / "outside" / "ckpt_00000009"
    (outside / "optim").mkdir(parents=True)
    (outside / "optim" / "adam.pt").write_bytes(b"x" * 10)
    (outside / "weights").mkdir()
    root = tmp_path / "runs" / "r1" / "checkpoints"
    for i in (1, 2, 3):
        d = root / f"ckpt_0000000{i}"
        (d / "weights").mkdir(parents=True)
        (d / "optim").mkdir()
    os.symlink(str(outside), str(root / "ckpt_00000099"))
    res = apply_retention(str(tmp_path / "runs"), keep=1, keep_optim=0)
    assert (outside / "optim" / "adam.pt").exists()
    assert all(not p.startswith(str(tmp_path / "outside")) for p in res["paths"])
    assert os.path.islink(str(root / "ckpt_00000099"))
