"""Backup: consistent snapshots, verified upload, and a real restore (local rclone target)."""
import json
import shutil

import pytest
import torch

from agentrs.backup import backup_once, make_target, select_files
from agentrs.training.common import save_checkpoint

pytestmark = pytest.mark.skipif(shutil.which("rclone") is None, reason="rclone not installed")


def _model():
    return torch.nn.Linear(4, 3)


def test_upload_verify_restore_roundtrip(tmp_path):
    runs = tmp_path / "work" / "runs"
    m = _model()
    save_checkpoint(runs / "exp" / "checkpoints" / "policy_final", m)
    (runs / "exp" / "checkpoints" / "rl_state").mkdir(parents=True)
    (runs / "exp" / "checkpoints" / "rl_state" / "state.pt").write_bytes(b"incomplete")  # no DONE marker
    (runs / "exp" / "checkpoints" / "half.tmp").mkdir()
    (runs / "exp" / "checkpoints" / "half.tmp" / "weights.safetensors").write_bytes(b"partial")
    (runs / "exp" / "metrics.jsonl").write_text('{"step": 1}\n')
    target = make_target(f"rclone::local:{tmp_path / 'remote'}")
    state = tmp_path / "state"
    r = backup_once(target, runs, state_dir=state)
    assert r["verified"] and r["changed"] == 3
    remote = target.remote_files()
    assert "runs/exp/checkpoints/policy_final/weights.safetensors" in remote
    assert not any("half.tmp" in p or "rl_state" in p for p in remote)  # incomplete state never uploaded
    # second pass: nothing changed
    assert backup_once(target, runs, state_dir=state)["changed"] == 0
    # restore onto a "fresh machine" and load the weights
    fresh = tmp_path / "fresh"
    target.download(fresh)
    sd = __import__("safetensors.torch", fromlist=["load_file"]).load_file(
        str(fresh / "runs/exp/checkpoints/policy_final/weights.safetensors"))
    assert torch.equal(sd["weight"], m.weight.detach())
    manifest = json.loads((fresh / "BACKUP_MANIFEST.json").read_text())
    assert len(manifest["files"]) == 3


def test_selection_skips_lock_and_hidden(tmp_path):
    runs = tmp_path / "runs"
    (runs / "cache").mkdir(parents=True)
    (runs / "cache" / "bc_x.lock").write_text("")
    (runs / "cache" / ".hidden").write_text("")
    (runs / "cache" / "summary.json").write_text("{}")
    files = [f.name for fs in select_files(runs).values() for f in fs]
    assert files == ["summary.json"]
