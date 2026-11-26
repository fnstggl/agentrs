"""Replay burn-in slices (plan 2026-11-26 §2 BPTT: 16 burn-in decisions without gradient, then 8 graded ones; plan
§1 Memory "Replay: burn-in slices")."""
from __future__ import annotations

from memhelpers import light_events
from superlearner.cam_unified.memory.archive import EventArchive
from superlearner.cam_unified.memory.replay import ReplaySampler


def _archive(root):
    a = EventArchive(root, chunk_rows=500)
    a.append(light_events(1200, episode_len=50))
    return a


def test_burn_in_slices_precede_a_graded_window_that_holds_the_anchor(tmp_path):
    a = _archive(str(tmp_path / "a"))
    rs = ReplaySampler(a, seed=3)
    rs.sync()
    sl = rs.sample(64, 8, burn_in=16)
    assert len(sl) == 64
    for s in sl:
        b = s.meta["burn_in"]
        assert 0 <= b <= 16 and len(s) - b <= 8 and len(s) <= 24
        graded = s.event_ids[b:]
        assert graded and len(graded) == min(8, len(s) - b)
        steps = s.steps
        assert steps == list(range(steps[0], steps[0] + len(steps)))          # contiguous in its own stream
        assert all(r.episode_id == s.episode_id for r in s.records)
        if b < 16:
            assert steps[0] == 0                                              # short only at the episode start
    assert any(s.meta["burn_in"] == 16 for s in sl)
    a.close()


def test_burn_in_zero_draws_exactly_what_it_drew_before(tmp_path):
    a = _archive(str(tmp_path / "a"))
    r1, r2 = ReplaySampler(a, seed=7, path=str(tmp_path / "r1.sqlite")), ReplaySampler(a, seed=7,
                                                                                     path=str(tmp_path / "r2.sqlite"))
    r1.sync()
    r2.sync()
    x = [s.event_ids for s in r1.sample(32, 8)]
    y = [s.event_ids for s in r2.sample(32, 8, burn_in=0)]
    assert x == y and all("burn_in" not in s.meta for s in r2.sample(4, 8))
    a.close()
