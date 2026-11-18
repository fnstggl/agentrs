"""Download the latest CAM checkpoint from GitHub Releases, verify it (part/tar/per-file sha256),
compare it byte-for-byte with the local checkpoint, load it into a fresh CAM, and check that the
restored network reproduces the original's decision log-probs exactly on real recorded decisions
(and differs from an untrained CAM), and that the optimizer state loads.

    PYTHONPATH=src python scripts/verify_cam_restore.py --run /workspace/agentrs/runs/cam --dest /tmp/cam_restore
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from agentrs.cam.learner import policy_terms, unroll
from agentrs.cam.model import CAM, CAMConfig
from agentrs.cam.runtime import Workers
from agentrs.gh_backup import restore, sha256_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/workspace/agentrs/runs/cam")
    ap.add_argument("--dest", default="/tmp/cam_restore")
    ap.add_argument("--tag")
    ap.add_argument("--out")
    a = ap.parse_args()
    t0 = time.time()
    res = restore(Path(a.dest), tag=a.tag, run_name=Path(a.run).name)
    b = Path(res["dir"])
    info = json.loads((b / "BUNDLE_INFO.json").read_text())
    rck = b / "payload" / "ckpt" / info["checkpoint"]
    lck = Path(a.run) / "ckpt" / info["checkpoint"]
    rep = {"tag": res["tag"], "checkpoint": info["checkpoint"], "files": res["files"]}
    same = {f.name: sha256_file(f) == sha256_file(lck / f.name) for f in rck.iterdir()
            if f.is_file() and (lck / f.name).exists()}
    rep["byte_identical_to_local"] = bool(same) and all(same.values())
    cfg = CAMConfig(**json.loads((rck / "config.json").read_text()))
    torch.manual_seed(0)
    restored = CAM(cfg)
    restored.load_state_dict(torch.load(rck / "weights.pt", map_location="cpu"))
    original = CAM(cfg)
    original.load_state_dict(torch.load(lck / "weights.pt", map_location="cpu")) if (lck / "weights.pt").exists() else None
    fresh = CAM(cfg)
    opt = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    opt.load_state_dict(torch.load(rck / "optimizer.pt", map_location="cpu")["optimizer"])
    eps = Workers(restored, 16, {"negotiation": 0.5, "product": 0.5}, seed0=777_000, gen_seed=7).collect(min_episodes=16, version="r")
    with torch.no_grad():
        lr_ = policy_terms(restored, eps, *unroll(restored, eps))[0]
        lo = policy_terms(original, eps, *unroll(original, eps))[0]
        lf = policy_terms(fresh, eps, *unroll(fresh, eps))[0]
        lo2 = policy_terms(original, eps, *unroll(original, eps))[0]  # same weights again: numerical floor
    rep["decisions"] = int(lr_.numel())
    rep["max_abs_logp_diff_restored_vs_original"] = float((lr_ - lo).abs().max())
    rep["max_abs_logp_diff_same_model_rerun"] = float((lo2 - lo).abs().max())
    rep["mean_abs_logp_diff_vs_untrained"] = float((lr_ - lf).abs().mean())
    rep["optimizer_state_loaded"] = len(opt.state) > 0
    rep["state"] = {k: json.loads((rck / "state.json").read_text()).get(k) for k in ("iter", "version", "phase")}
    rep["passed"] = (rep["byte_identical_to_local"] and rep["max_abs_logp_diff_restored_vs_original"] <= 1e-4
                     and rep["mean_abs_logp_diff_vs_untrained"] > 0)
    rep["seconds"] = round(time.time() - t0)
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
