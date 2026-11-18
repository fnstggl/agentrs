"""Download a CAM-v2 checkpoint from GitHub Releases, verify it (part / tar / per-file sha256), compare it
byte-for-byte with the local checkpoint, load it into a fresh CAM2, and check that the restored network
reproduces the original's decision log-probabilities and world-model predictions on recorded real
episodes (and differs from an untrained CAM2), and that all three optimizers load.

    PYTHONPATH=src:/workspace/agentrs/src python scripts/verify_cam2_restore.py --run /workspace/agentrs/runs/cam2/cam2_awr_s1
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import random
import time
from pathlib import Path

import torch

from agentrs.cam.v2.data import View
from agentrs.cam.v2.learner import OffPolicy, _grammar, decision_logprob, observe
from agentrs.cam.v2.model import CAM2, CAM2Config
from agentrs.gh_backup import restore, sha256_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dest", default="/tmp/cam2_restore")
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
    same = {f.name: sha256_file(f) == sha256_file(lck / f.name) for f in rck.iterdir() if f.is_file() and (lck / f.name).exists()}
    rep["byte_identical_to_local"] = bool(same) and all(same.values())
    cfg = CAM2Config(**json.loads((rck / "config.json").read_text()))
    torch.manual_seed(0)
    restored, original, fresh = CAM2(cfg), CAM2(cfg), CAM2(cfg)
    restored.load_state_dict(torch.load(rck / "weights.pt", map_location="cpu"))
    original.load_state_dict(torch.load(lck / "weights.pt", map_location="cpu"))
    OffPolicy(restored, {}).load_state_dict(torch.load(rck / "optimizer.pt", map_location="cpu"))
    rep["optimizers_loaded"] = True
    shard = sorted((Path(a.run) / "replay").glob("shard_*.pkl.gz"))[-1]
    with gzip.open(shard, "rb") as f:
        eps = pickle.load(f)
    random.Random(0).shuffle(eps)
    views = [View(e, list(e.goal), len(e.steps)) for e in eps[:16]]

    def score(m):
        torch.manual_seed(1)
        with torch.no_grad():
            ob = observe(m, views)
            dec = [(i, t) for i, v in enumerate(views) for t in range(len(v.steps)) if v.steps[t].learn]
            k = torch.tensor([ob["pos"][x] for x in dec])
            gi = torch.tensor([i for i, _ in dec])
            lp, _ = decision_logprob(m, ob["feat"][k], ob["goal"][gi], [views[i].steps[t] for i, t in dec],
                                     [_grammar(views[i].ep.family) for i, _ in dec])
            rew = m.heads(ob["feat"][k], ob["goal"][gi])["rew"]
        return lp, rew

    lr_, rr = score(restored)
    lo, ro = score(original)
    lf, _ = score(fresh)
    rep["decisions"] = int(lr_.numel())
    rep["max_abs_logp_diff_restored_vs_original"] = float((lr_ - lo).abs().max())
    rep["max_abs_reward_pred_diff_restored_vs_original"] = float((rr - ro).abs().max())
    rep["mean_abs_logp_diff_vs_untrained"] = float((lr_ - lf).abs().mean())
    rep["state"] = {k: json.loads((rck / "state.json").read_text()).get(k) for k in ("iter", "version", "phase")}
    rep["passed"] = (rep["byte_identical_to_local"] and rep["max_abs_logp_diff_restored_vs_original"] <= 1e-5
                     and rep["max_abs_reward_pred_diff_restored_vs_original"] <= 1e-5 and rep["mean_abs_logp_diff_vs_untrained"] > 0)
    rep["seconds"] = round(time.time() - t0)
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
