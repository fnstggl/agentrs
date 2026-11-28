"""G2-integration probe p03 — storage guard, retention and disk monitoring (OP-01, OP-02, OP-03).

The integrated agent on the tiny CPU configuration with ``storage.min_free_gb`` set far above the free space of
this disk, so every guarded write must be refused before anything is written; ``tools/prune_runs.py`` on synthetic
run trees in the scratch directory; the doctor's and dashboard's disk reporting.

Usage (from /workspace/camlive):
    PYTHONPATH=<copy>/cam_live/src .venv/bin/python cam_live/docs/verification/repro/g2integration/p03_storage.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

from cam_live import contracts as C
from cam_live.config import CamLiveConfig, PROJECT_ROOT
from cam_live.storage import DiskGuard, DiskSpaceLow, apply_retention, free_gb

SCRATCH = os.environ.get("G2I_SCRATCH", "/tmp/claude-0/-workspace/8e531f56-63d9-4df0-a584-c2fb645312ce/scratchpad/"
                                        "G2INTEGRATION/p03")
broken = 0


def check(name, ok, detail=""):
    global broken
    broken += 0 if ok else 1
    print(f"[{'HOLDS ' if ok else 'BROKEN'}] {name}" + (f" :: {detail}" if detail else ""))


def ckpt(path, optim=True):
    os.makedirs(os.path.join(path, "weights"), exist_ok=True)
    with open(os.path.join(path, "weights", "w.bin"), "wb") as f:
        f.write(b"w" * 1000)
    if optim:
        os.makedirs(os.path.join(path, "optim"), exist_ok=True)
        with open(os.path.join(path, "optim", "o.bin"), "wb") as f:
            f.write(b"o" * 2000)
    with open(os.path.join(path, "meta.json"), "w") as f:
        f.write("{}")


def main() -> int:
    os.makedirs(SCRATCH, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="p03_", dir=SCRATCH)
    free = free_gb(tmp)
    huge = free * 10 + 1000.0
    print(f"free space at the scratch directory: {free:.1f} GB; minimum used by the probe: {huge:.0f} GB")

    # ---- OP-01 the guard itself --------------------------------------------------------------------------------
    g = DiskGuard(tmp, huge, huge * 2)
    try:
        g.check(0, "probe")
        ok = False
    except DiskSpaceLow:
        ok = True
    check("1 DiskGuard.check raises DiskSpaceLow below the minimum", ok)

    # ---- OP-01 the integrated agent --------------------------------------------------------------------------
    from cam_live.agent import CamLiveAgent
    cfg = CamLiveConfig.debug(seed=0)
    cfg.sections["storage"] = {"min_free_gb": huge, "warn_free_gb": huge * 2}
    task = C.TaskSpec(objective_id="st", objective="storage probe", mode="staging", interfaces=["internal"],
                      limits=C.Limits(duration_s=3600, money=0, max_decisions=10, max_external_actions=1))
    agent = CamLiveAgent(cfg, os.path.join(tmp, "agent"), clock=C.SimClock(1000.0), task=task)
    dest = os.path.join(tmp, "ckpt_save")
    try:
        agent.save(dest)
        ok = False
    except DiskSpaceLow:
        ok = not os.path.exists(dest)
    check("2 agent.save is refused below the minimum and writes nothing", ok)
    if agent.ckpts is not None:
        try:
            agent.promote("probe")
            ok = False
        except DiskSpaceLow:
            ok = True
        check("3 agent.promote (learner checkpoint) is refused below the minimum", ok)
    exp = os.path.join(tmp, "export_weights")
    try:
        agent.export_weights(exp)
        wrote = os.path.exists(exp)
        err = ""
    except DiskSpaceLow as e:
        wrote, err = os.path.exists(exp), str(e)
    check("4 agent.export_weights (weights-only export, what `export-checkpoint` of a running run calls) is refused "
          "below the minimum", not wrote, f"export written: {wrote} {err}")

    # `cam-live export-checkpoint` of a stopped run: the committed checkpoint is exported from disk
    cfg_ok = CamLiveConfig.debug(seed=0)
    agent2 = CamLiveAgent(cfg_ok, os.path.join(tmp, "agent2"), clock=C.SimClock(1000.0), task=task)
    run = os.path.join(tmp, "run")
    os.makedirs(os.path.join(run, "checkpoints"), exist_ok=True)
    cpath = os.path.join(run, "checkpoints", "ckpt_00000001")
    cid = agent2.save(cpath)
    agent2.close()
    with open(os.path.join(run, C.RunFiles.STATE), "w") as f:
        json.dump({"committed": {"ckpt": {"id": cid, "path": cpath}}, "task": task.to_dict()}, f)
    out = os.path.join(tmp, "exported")
    env = dict(os.environ, CAM_LIVE_MIN_FREE_GB=str(huge), CAM_LIVE_WARN_FREE_GB=str(huge * 2),
               CAM_LIVE_HOME=os.path.join(tmp, "home"), CAM_LIVE_GPU_LOCK=os.path.join(tmp, "gpu.lock"))
    p = subprocess.run([sys.executable, "-m", "cam_live.runtime.cli", "export-checkpoint", "--run-dir", run,
                        "--out", out], capture_output=True, text=True, env=env, timeout=600)
    check("5 `cam-live export-checkpoint` (stopped run) writes nothing below the minimum (CAM_LIVE_MIN_FREE_GB)",
          not os.path.exists(out), f"exit {p.returncode}; export written: {os.path.exists(out)} "
          f"({sum(len(fs) for _d, _s, fs in os.walk(out)) if os.path.exists(out) else 0} files)")
    agent.close()

    # ---- OP-02 retention (tools/prune_runs.py through cam_live.storage) ---------------------------------------
    root = os.path.join(tmp, "tree")
    grp = os.path.join(root, "runA", "checkpoints")
    for i in (1, 2, 3, 4):
        ckpt(os.path.join(grp, f"ckpt_{i:08d}"))
    ckpt(os.path.join(root, "runA", "ckpt_pre"))
    ckpt(os.path.join(root, "runA", "export_post"))
    with open(os.path.join(root, "runA", "events.jsonl"), "w") as f:
        f.write('{"a": 1}\n')
    res = apply_retention(root, keep=2, keep_optim=1)
    left = sorted(os.listdir(grp))
    check("6 newest 2 kept, older removed", left == ["ckpt_00000003", "ckpt_00000004"], left)
    check("7 optimizer state only in the newest", os.path.isdir(os.path.join(grp, "ckpt_00000004", "optim"))
          and not os.path.isdir(os.path.join(grp, "ckpt_00000003", "optim")))
    check("8 protected checkpoints and records are kept", all(os.path.exists(os.path.join(root, "runA", x))
                                                             for x in ("ckpt_pre/weights", "export_post/weights",
                                                                       "events.jsonl")))
    # a symbolic link inside the tree that points at a checkpoint-shaped directory outside it
    outside = os.path.join(tmp, "outside_tree", "ckpt_00000001")
    ckpt(outside)
    grp2 = os.path.join(root, "runB", "checkpoints")
    ckpt(os.path.join(grp2, "ckpt_00000009"))
    os.symlink(outside, os.path.join(grp2, "ckpt_00000005"))
    res = apply_retention(root, keep=2, keep_optim=1)
    check("9 retention never deletes anything outside the tree it is applied to (symlinked checkpoint)",
          os.path.isdir(os.path.join(outside, "optim")) and os.path.isdir(os.path.join(outside, "weights")),
          f"outside optim exists: {os.path.isdir(os.path.join(outside, 'optim'))}; actions: "
          f"{[p.replace(tmp, '') for p in res['paths']]}")
    # the learner's own promotion directory: how many checkpoints keep their optimizer state
    # (credit.checkpoint.CheckpointManager keep_last=2 keep_promoted=1, used by CamLiveAgent)
    from cam_live.credit.checkpoint import CheckpointManager
    import inspect
    src = inspect.getsource(CheckpointManager.prune)
    check("10 learner promotions (CheckpointManager.prune) drop optimizer state from all but the newest",
          "optim" in src, "prune() removes whole checkpoints only; kept ones (up to keep_last + keep_promoted) all "
                          "keep their optimizer state" if "optim" not in src else "")

    # ---- OP-03 monitoring ------------------------------------------------------------------------------------
    from cam_live.runtime import doctor
    cfgd = CamLiveConfig.debug(seed=0)
    cfgd.home = os.path.join(tmp, "home")
    cfgd.sections["storage"] = {"min_free_gb": huge, "warn_free_gb": huge * 2}
    disk = next(c for c in doctor.storage_checks(cfgd) if c.get("name") == "disk" or c.get("check") == "disk")
    check("11 doctor's disk check reports the configured minimum (storage.min_free_gb) as not met",
          not disk.get("ok", disk.get("passed", True)), json.dumps(disk)[:300])
    from cam_live.runtime.dashboard import RunReader  # noqa: F401  (import check only)
    print(f"== p03_storage: {broken} broken of 11 checks ({tmp})")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
