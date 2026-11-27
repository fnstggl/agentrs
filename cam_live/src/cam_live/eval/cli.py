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
    _add_analyse(sub)
    from . import extra as _extra
    _extra.add_commands(sub)
    a = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    return int(a.fn(a) or 0)



# ------------------------------------------------------------------------------------------------ analyse
EVIDENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "docs", "evidence", "S")


def _write(name: str, obj: dict) -> str:
    path = os.path.abspath(os.path.join(EVIDENCE, name))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    return path


def cmd_analyse(a: argparse.Namespace) -> int:
    from . import analysis
    from .experiments import EVAL_HOME
    from .integrity import reward_vs_progress
    from .runner import run_anchor
    res_dir = os.path.join(EVAL_HOME, "results", a.plan)
    by = analysis.load(res_dir)
    if not by:
        print(f"no executed results in {res_dir}", file=sys.stderr)
        return 1
    seeds = sorted({r["seed"] for rs in by.values() for r in rs})
    n_lt = max(len(r["lifetimes"]) for rs in by.values() for r in rs)
    fam = next(iter(by.values()))[0]["family"]
    from .experiments import WORLD_BASE
    worlds = [WORLD_BASE[fam] + 100 * s + i for s in seeds for i in range(n_lt)]
    anchors = {"anchors": {k: [run_anchor(k, fam, w, max_decisions=16) for w in worlds] for k in ("uniform", "oracle")}}
    common = {"executed": True, "plan": a.plan, "results_dir": res_dir, "seeds": seeds, "family": fam,
              "conditions": {c: [r["seed"] for r in rs] for c, rs in by.items()},
              "scale": {c: {"backbone": rs[0]["backbone"], "core": rs[0]["scale"], "real": rs[0]["real_backbone"]}
                        for c, rs in by.items()}}
    tag = a.tag or a.plan
    out = []
    out.append(_write(f"{tag}_baselines.json", {**common, **analysis.baselines(by, anchors)}))
    learners = [c for c in ("E", "D", "E_mix0", "E_no_credit", "C") if c in by]
    out.append(_write(f"{tag}_calibration.json", {**common, **{c: analysis.calibration(by[c]) for c in learners}}))
    out.append(_write(f"{tag}_action_quality.json", {**common, **{c: analysis.action_quality(by[c])
                                                                for c in learners}}))
    if "E" in by:
        out.append(_write(f"{tag}_delayed_credit.json", {**common, **analysis.delayed_credit(
            by["E"], by.get("E_no_credit", []))}))
        out.append(_write(f"{tag}_shadow.json", {**common, **{c: analysis.shadow(by[c]) for c in learners}}))
        out.append(_write(f"{tag}_reward_vs_progress.json", {**common, **{c: reward_vs_progress(by[c])
                                                                          for c in ("E", "E_mix0", "D") if c in by}}))
    print("\n".join(out))
    return 0


def _add_analyse(sub) -> None:
    an = sub.add_parser("analyse", help="prespecified analyses of a results plan -> docs/evidence/S/*.json")
    an.add_argument("--plan", default="main_cpu")
    an.add_argument("--tag", default="")
    an.set_defaults(fn=cmd_analyse)


if __name__ == "__main__":
    sys.exit(main())
