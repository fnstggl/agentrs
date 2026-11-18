"""Download a superlearner checkpoint from GitHub Releases and verify the restore.

Checks: the bundle verifies (part / tar / per-file sha256, done by gh_backup.restore); every weight tensor equals the
local checkpoint it was made from; the optimizer state loads into the learner; the RNG state loads; and the restored
network reproduces the local network's action log-probabilities and values on the same live observations (and differs
from a freshly initialised network).

    PYTHONPATH=src:/workspace/agentrs/src python scripts/verify_sl_restore.py --run /workspace/agentrs/runs/sl/P_NPV_s1 \
        --local-ckpt initial [--tag ckpt-...]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--local-ckpt", default="initial")
    ap.add_argument("--tag")
    ap.add_argument("--dest", default="/workspace/restore_staging/sl_verify")
    a = ap.parse_args()
    from agentrs.gh_backup import restore
    from agentrs.sl import worlds as W
    from agentrs.sl.learner import PPO
    from agentrs.sl.model import SLConfig, SLCore
    from agentrs.sl.runner import Runner

    t0 = time.time()
    res = restore(Path(a.dest), tag=a.tag, run_name=Path(a.run).name)
    b = Path(res["dir"])
    info = json.loads((b / "BUNDLE_INFO.json").read_text())
    rck = b / "payload" / "ckpt" / info["checkpoint"]
    lck = Path(a.run) / "ckpt" / a.local_ckpt
    rep = {"tag": res["tag"], "bundle_checkpoint": info["checkpoint"], "compared_with_local": str(lck), "files": res["files"]}
    rw = torch.load(rck / "weights.pt", map_location="cpu")
    lw = torch.load(lck / "weights.pt", map_location="cpu")
    rep["tensors"] = len(rw)
    rep["all_tensors_equal_local"] = set(rw) == set(lw) and all(torch.equal(rw[k], lw[k]) for k in rw)
    cfg = SLConfig(**json.loads((rck / "config.json").read_text()))
    restored, local = SLCore(cfg), SLCore(cfg)
    restored.load_state_dict(rw)
    local.load_state_dict(lw)
    torch.manual_seed(12345)
    fresh = SLCore(cfg)
    ppo = PPO(restored, {})
    ppo.load_state_dict(torch.load(rck / "optimizer.pt", map_location="cpu", weights_only=False))
    rep["optimizer_loaded"] = True
    torch.load(rck / "rng.pt", weights_only=False)
    rep["rng_loaded"] = True
    st = json.loads((rck / "state.json").read_text())
    fams = list(st["config"]["families"])

    def run(m):
        torch.manual_seed(7)
        r = Runner(m, {f: 1.0 for f in fams}, W.make_world, W.sample_task, 6, trial_episodes=2, seed=3, greedy=True)
        seg = r.collect(12)
        r.close()
        return torch.stack([s["lp"] for s in seg["steps"]]), torch.stack([s["v"] for s in seg["steps"]])

    (lr_, vr), (ll, vl), (lf, vf) = run(restored), run(local), run(fresh)
    rep["max_abs_logp_diff_restored_vs_local"] = float((lr_ - ll).abs().max())
    rep["max_abs_value_diff_restored_vs_local"] = float((vr - vl).abs().max())
    rep["mean_abs_value_diff_vs_fresh"] = float((vr - vf).abs().mean())
    rep["passed"] = bool(rep["all_tensors_equal_local"] and rep["max_abs_logp_diff_restored_vs_local"] == 0.0
                         and rep["max_abs_value_diff_restored_vs_local"] == 0.0 and rep["mean_abs_value_diff_vs_fresh"] > 0)
    rep["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(rep, indent=1))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
