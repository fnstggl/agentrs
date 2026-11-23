"""An independent check of the finished lifetimes on tasks nothing has looked at.

    superlearner continual-verify [--life runs/life] [--workers 8] [--parallel 3]

Why it exists: the protection used in the compared conditions was chosen from a short calibration, and that
calibration, the lifetimes' own held-out lists and the meta-search all draw their tasks from the same seed range of
the `select` split. This evaluation uses a seed range none of them touched (offsets 30,000+), fixed before any
outcome was seen, the same tasks for every run.

Each run's final weights are exported to a tensors-only file and evaluated greedily in a NEW process, with no
episodic memory, tools, replay buffer or cached embedding (`internalize.fresh_eval`): this is "generalization without
episode retrieval" for the whole lifetime. Runs that wrote episodic memory are also evaluated with that memory.
"""
from __future__ import annotations

import concurrent.futures as cf
import glob
import json
import os

import torch

from ..cam.model import CAMController, ModelConfig
from ..world.tasks import task_list
from . import plasticity as PL
from .internalize import fresh_eval

ARMS = ["found/0/scratch", "list/0/deployed", "fund/0/scratch", "deploy/0/coded", "service/0/listed", "repair/0/scratch",
        "service/0/deployed", "service/0/coded"]
SEEN = ["tiny", "bazaar", "cheap_compute_closed_markets", "expensive_compute_open_distribution"]
UNSEEN = ["entrenched_incumbents", "fragmented_networks"]


def verify_lists() -> list[dict]:
    out = [{"name": "kinds_of_world_trained_in", "arms": ARMS, "worlds": SEEN, "n": 72, "offset": 30_000},
           {"name": "kinds_of_world_never_trained_in", "arms": ARMS, "worlds": UNSEEN, "n": 40, "offset": 31_000}]
    for x in out:
        x["tasks"] = [t.to_dict() for t in task_list(x["arms"], x["worlds"], x["n"], "select", x["offset"])]
    return out


def export_final(run_dir: str, path: str) -> str:
    """The final weights of a lifetime as a tensors-only file (same format as the internalization test)."""
    ckp = os.path.join(run_dir, "ckpt", "latest.pt")
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    m = CAMController(ModelConfig(**ck["model_cfg"]))
    m.load_state_dict(ck["model"])
    bbp = ckp + ".backbone.pt"
    wm = (ck.get("algo") or {}).get("wm")
    w = {"model": {k: v.cpu() for k, v in m.state_dict().items()},
         "wm": {k: v.cpu() for k, v in wm.items()} if wm is not None else None,
         "backbone": torch.load(bbp, map_location="cpu", weights_only=False) if os.path.exists(bbp) else None,
         "digest": PL.weights_digest(m)}
    torch.save(w, path + ".tmp")
    os.replace(path + ".tmp", path)
    return path


def verify_run(run_dir: str, lists: list[dict], workers: int = 8) -> dict:
    cfg = json.load(open(os.path.join(run_dir, "lifetime.json")))["cfg"]
    out = os.path.join(run_dir, "verify.json")
    mem = bool((cfg.get("lifetime") or {}).get("memory"))
    outm = os.path.join(run_dir, "verify_memory.json")
    if os.path.exists(out) and (not mem or os.path.exists(outm)):
        return {"run": run_dir, "reused": True}
    wp = os.path.join(run_dir, "weights_final.pt")
    export_final(run_dir, wp)
    try:
        fresh_eval(cfg, wp, lists, out, workers=workers)
        if mem:
            fresh_eval(cfg, wp, lists, outm, memory_from=os.path.join(run_dir, "ckpt", "latest.pt"), workers=workers)
    finally:
        if os.path.exists(wp):
            os.unlink(wp)                        # the checkpoint holds the same weights
    return {"run": run_dir}


def main(a) -> int:
    from ..cli import RUNS
    root = a.life or os.path.join(RUNS, "life")
    runs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "DONE")))
    lists = verify_lists()
    print(f"{len(runs)} finished lifetimes, {sum(len(x['tasks']) for x in lists)} tasks", flush=True)
    with cf.ThreadPoolExecutor(max_workers=int(a.parallel)) as ex:
        futs = {ex.submit(verify_run, r, lists, int(a.workers)): r for r in runs}
        failed = 0
        for f in cf.as_completed(futs):
            try:
                f.result()
                print("verified", os.path.basename(futs[f]), flush=True)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print("FAILED", os.path.basename(futs[f]), str(e)[-600:], flush=True)
    return 1 if failed else 0
