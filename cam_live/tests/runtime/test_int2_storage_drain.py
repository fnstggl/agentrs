"""Storage guard and drain-after-exhaustion of the runtime (INT2; storage rule of 2026-11-27, S-1).

* below the configured free-space minimum the runtime writes no checkpoint: it pauses the run with a resource
  obstacle (logged as ``obstacle`` / ``storage_low``, journalled) and resumes normally once space is available;
* runtime checkpoints: the newest ``keep_checkpoints`` (default 2) are kept, optimizer state only in the newest;
* the integrated agent's learner promotions check the guard first; a refusal reaches the runtime, which pauses;
* ``drain_after_exhaustion``: after the decision budget the run keeps observing — no new decision — until the
  verifier's verdict is terminal or the deadline passes, so the experience gets its outcomes.
"""
from __future__ import annotations

import json
import os

import pytest

from cam_live.contracts import RunFiles
from cam_live.diagnostics.world import LabConfig
from cam_live.runtime.loop import RuntimeConfig
from cam_live.storage import DiskGuard, DiskSpaceLow, free_gb
from rt_helpers import make_lab


def _events(run_dir, t):
    p = os.path.join(run_dir, RunFiles.EVENTS)
    return [json.loads(l) for l in open(p)] and [e for e in (json.loads(l) for l in open(p)) if e["type"] == t]


def test_runtime_pauses_instead_of_checkpointing_when_disk_is_low(tmp_path):
    lab = make_lab(str(tmp_path / "run"), rt_cfg=RuntimeConfig(fsync=False, min_free_gb=1e9, warn_free_gb=2e9))
    rt = lab.rt
    for _ in range(4):
        rt.step()
        if rt.status != "running":
            break
    assert rt.status == "paused" and "storage" in rt.status_reason
    assert not os.path.isdir(os.path.join(rt.run_dir, RunFiles.CHECKPOINTS)) or \
        not os.listdir(os.path.join(rt.run_dir, RunFiles.CHECKPOINTS))          # nothing was written
    obs = _events(rt.run_dir, "obstacle")
    assert obs and obs[-1]["data"]["kind"] == "storage_low" and obs[-1]["data"]["class"] == "resource"
    assert _events(rt.run_dir, "paused")
    assert os.path.exists(os.path.join(rt.run_dir, "journal.md"))
    # space is available again (operator): resume, and the next commit is written
    rt.guard = DiskGuard(rt.run_dir, 0.0, 0.0)
    rt.cfg.min_free_gb = 0.0
    rt._apply_command("resume", {}, "operator")
    for _ in range(6):
        rt.step()
    assert rt.status in ("running", "completed", "exhausted", "failed")
    assert os.listdir(os.path.join(rt.run_dir, RunFiles.CHECKPOINTS))
    rt.close()


def test_drain_after_exhaustion_waits_for_the_verdict(tmp_path):
    cfg = LabConfig(seed=3, target=200, outcome_horizon_s=3 * 3600.0)
    plain = make_lab(str(tmp_path / "plain"), lab_cfg=cfg, task_edit=lambda t: setattr(t.limits, "max_decisions", 4),
                     rt_cfg=RuntimeConfig(fsync=False))
    for _ in range(30):
        plain.rt.step()
        if plain.rt.status != "running":
            break
    assert plain.rt.status == "exhausted" and len(plain.rt.pending) > 0       # outcomes left unmeasured
    lab = make_lab(str(tmp_path / "drain"), lab_cfg=cfg, task_edit=lambda t: setattr(t.limits, "max_decisions", 4),
                   rt_cfg=RuntimeConfig(fsync=False, drain_after_exhaustion=True))
    for _ in range(400):
        lab.rt.step()
        if lab.rt.status != "running":
            break
    rt = lab.rt
    assert rt.counters["decisions"] == 4                                      # nothing decided while draining
    assert _events(rt.run_dir, "draining")
    assert rt.status in ("failed", "exhausted", "completed")
    vo = rt._last_verified.get("outcome", {})
    assert vo.get("status") in ("failed", "achieved")                         # the verifier's terminal verdict
    assert len(rt.pending) == 0                                               # every due outcome was measured
    assert lab.clock.now() >= rt._deadline() or vo.get("status") == "achieved"
    rt.close()
