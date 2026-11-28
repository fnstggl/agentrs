"""HX-6 (G2-harness R3-M1): the command channel's lock file is never opened through a planted symlink."""
import os

import pytest

from cam_live.runtime.control import ControlChannel


def test_a_symlink_planted_as_the_lock_file_is_refused_and_its_target_untouched(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    ch = ControlChannel(str(run))
    os.symlink(str(victim), ch.path + ".lock")
    with pytest.raises(OSError):
        ch.send("pause", wake=False)
    assert victim.read_text() == "keep me"
    assert os.path.islink(ch.path + ".lock")
    assert not os.path.exists(ch.path)


def test_a_dangling_symlink_does_not_create_its_target(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    target = tmp_path / "created_by_root"
    ch = ControlChannel(str(run))
    os.symlink(str(target), ch.path + ".lock")
    with pytest.raises(OSError):
        ch.send("pause", wake=False)
    assert not os.path.exists(target)


def test_commands_are_still_appended_through_the_safe_lock(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    ch = ControlChannel(str(run))
    a = ch.send("pause", wake=False)
    b = ch.send("resume", wake=False)
    assert a != b
    assert os.path.isfile(ch.path + ".lock") and not os.path.islink(ch.path + ".lock")
