"""Command line for the content encoders (run from crucible/ with PYTHONPATH=src; anything that loads models goes
through the slot scheduler: `python -m crucible.scale.slots --cpu 2 --gpu-mb 3000 -- python -m ...`).

    python -m crucible.semantic.encoders fetch                     # download the pinned model files (once, ~1.2 GB)
    python -m crucible.semantic.encoders info [--modality video]   # ids, versions, dims, parameter counts
    python -m crucible.semantic.encoders encode FILE --modality image [--out vec.npy]
    python -m crucible.semantic.encoders validate [--modality audio] [--json out.json]   # separation suites
    python -m crucible.semantic.encoders audit                     # creator-assigned quality field audit
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m crucible.semantic.encoders")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    i = sub.add_parser("info")
    i.add_argument("--modality")
    e = sub.add_parser("encode")
    e.add_argument("file")
    e.add_argument("--modality", required=True)
    e.add_argument("--out")
    v = sub.add_parser("validate")
    v.add_argument("--modality", action="append")
    v.add_argument("--json")
    sub.add_parser("audit")
    a = ap.parse_args(argv)
    from . import MODALITIES, encode_bytes, encoder_info, fetch, stats
    if a.cmd == "fetch":
        print(json.dumps(fetch(), indent=1))
    elif a.cmd == "info":
        for m in ([a.modality] if a.modality else MODALITIES):
            info = encoder_info(m)
            print(json.dumps({k: info[k] for k in ("id", "version", "dim", "params", "preprocessing_hash", "device")}))
    elif a.cmd == "encode":
        vec = encode_bytes(open(a.file, "rb").read(), a.modality)
        if a.out:
            np.save(a.out, vec)
        print(json.dumps({"dim": int(vec.size), "norm": float(np.linalg.norm(vec)), "head": vec[:8].round(5).tolist()}))
    elif a.cmd == "validate":
        from .validation import run
        res = run(a.modality or MODALITIES)
        for m, r in res.items():
            print(f"{m:6s} ratio {r['separation_ratio']:8.2f}  auc {r['auc_semantic_farther']:.4f}  "
                  f"nuisance {r['mean_nuisance_distance']:.4f}  semantic {r['mean_semantic_distance']:.4f}")
        if a.json:
            json.dump({"results": res, "stats": stats()}, open(a.json, "w"), indent=1)
    elif a.cmd == "audit":
        from .audit import quality_fields
        rep = quality_fields()
        print(json.dumps({k: rep[k] for k in ("files_scanned", "reads", "creator_assigned_quality_fields_read",
                                              "creator_assignable_count", "out_of_scope_files", "exempt")}, indent=1))
        return 1 if rep["creator_assigned_quality_fields_read"] else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
