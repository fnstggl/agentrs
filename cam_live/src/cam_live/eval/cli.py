"""cam_live.eval.cli — command line of the scientific evaluation (spec Part 16).

    python -m cam_live.eval.cli run --condition E --family F1 --seed 0 --lifetimes 8 --decisions 16 --out R.json
    python -m cam_live.eval.cli batch --plan baselines_cpu [--jobs 10]   # the prespecified CPU batch (PROTOCOLS.md)
    python -m cam_live.eval.cli anchors --family F1 --seeds 0-7 --out A.json
    python -m cam_live.eval.cli analyse [--which all]                  # results JSON -> docs/evidence/S/*.json
    python -m cam_live.eval.cli meta | integrity | continual | shadow | skills | scaling ...

``cam-live evaluate-transfer --checkpoint <ckpt>`` runs :func:`cam_live.eval.transfer.main`.
Every command writes JSON with ``"executed": true`` only after the experiment ran.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence


def _seeds(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def cmd_run(a: argparse.Namespace) -> int:
    import contextlib
    from .experiments import run_condition
    lock = contextlib.nullcontext()
    if a.real or a.gpu_lock:                      # heavy GPU work only under the shared lock, <= ~15 min per hold
        from ..nn.gpu import gpu_lock
        lock = gpu_lock(min_free_gb=a.min_free_gb, timeout_s=a.lock_timeout_s)
    with lock:
        res = run_condition(a.condition, a.family, a.seed, n_lifetimes=a.lifetimes, max_decisions=a.decisions,
                            real=a.real, scale=a.scale, out=a.out, save_dir=a.save_dir, k=a.k, threads=a.threads,
                            budget_s=a.budget_s, device=a.device)
    lts = res["lifetimes"]
    print(json.dumps({"condition": a.condition, "family": a.family, "seed": a.seed,
                      "quality": [round(x["quality"], 3) for x in lts],
                      "progress": [round(x["progress"], 3) for x in lts],
                      "wall_s": round(res["wall_seconds"], 1)}))
    return 0


def cmd_anchors(a: argparse.Namespace) -> int:
    from .runner import run_anchor
    out = {"executed": True, "family": a.family, "decisions": a.decisions, "anchors": {}}
    for kind in ("uniform", "oracle"):
        out["anchors"][kind] = [run_anchor(kind, a.family, s, max_decisions=a.decisions, k=a.k)
                                for s in _seeds(a.world_seeds)]
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m cam_live.eval.cli", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one condition on one family for a sequence of lifetimes")
    r.add_argument("--condition", required=True)
    r.add_argument("--family", default="F1")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--lifetimes", type=int, default=8)
    r.add_argument("--decisions", type=int, default=16)
    r.add_argument("--k", type=int, default=16)
    r.add_argument("--real", action="store_true")
    r.add_argument("--scale", default="")
    r.add_argument("--out", default="")
    r.add_argument("--save-dir", dest="save_dir", default="")
    r.add_argument("--threads", type=int, default=3)
    r.add_argument("--budget-s", dest="budget_s", type=float, default=0.0,
                   help="stop starting new lifetimes after this many seconds (GPU chunks: <= 780)")
    r.add_argument("--device", default="", help="override the device (e.g. cuda for the tiny-backbone scaling runs)")
    r.add_argument("--gpu-lock", dest="gpu_lock", action="store_true", help="hold the shared GPU lock")
    r.add_argument("--min-free-gb", dest="min_free_gb", type=float, default=14.0)
    r.add_argument("--lock-timeout-s", dest="lock_timeout_s", type=float, default=3600.0)
    r.set_defaults(fn=cmd_run)
    an = sub.add_parser("anchors", help="uniform / oracle anchors of the competence scale")
    an.add_argument("--family", default="F1")
    an.add_argument("--world-seeds", dest="world_seeds", required=True)
    an.add_argument("--decisions", type=int, default=16)
    an.add_argument("--k", type=int, default=16)
    an.add_argument("--out", required=True)
    an.set_defaults(fn=cmd_anchors)
    from . import batch as _batch
    _batch.add_commands(sub)
    a = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
