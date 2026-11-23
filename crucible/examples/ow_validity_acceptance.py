"""Acceptance rate of generated mechanisms under the validity checks, per family (all three engines).

For every mechanism family, draws `--n` mechanisms (seeds `--first` ...) and runs `mechanisms.validity.check_mechanism`
on each first draw: how many pass, why the others were rejected, and the solvability verdicts. A family whose
first draws often fail is a generator problem (fix the generator, do not loosen the check); a draw that fails is
redrawn deterministically by `sample_valid` when tasks are built.

Usage: PYTHONPATH=src python examples/ow_validity_acceptance.py [--n 12] [--first 0] [--workers 3]
           [--out docs/experiments/ow_validity_acceptance.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def one(args):
    fam, n, first = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from crucible.mechanisms.sim import ensure_engines
    from crucible.mechanisms.validity import acceptance
    ensure_engines()
    t0 = time.time()
    r = acceptance([fam], n, first)[fam]
    r["wall_s"] = round(time.time() - t0, 1)
    return fam, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "experiments",
                                                  "ow_validity_acceptance.json"))
    a = ap.parse_args()
    from crucible.mechanisms.base import MECH_FAMILIES
    from crucible.mechanisms.sim import ensure_engines
    ensure_engines()
    fams = sorted(MECH_FAMILIES)
    out = {"n_per_family": a.n, "first_seed": a.first, "families": {}}
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for fam, r in pool.imap_unordered(one, [(f, a.n, a.first) for f in fams]):
            f = MECH_FAMILIES[fam]
            out["families"][fam] = {"engine": f.engine, "split": f.split, "transfer": f.transfer, "seats": f.seats, **r}
            print(f"{fam:20s} {f.engine:8s} accepted {r['accepted']}/{r['n']} reasons {r['reasons']} "
                  f"solvable {r['solvable']} cpu/mech {r['cpu_s_per_mechanism']}s", flush=True)
            tot = sum(x["accepted"] for x in out["families"].values())
            out["total"] = {"accepted": tot, "n": sum(x["n"] for x in out["families"].values())}
            with open(a.out, "w") as fh:
                json.dump(out, fh, indent=1, sort_keys=True)
    print("total", out["total"], "->", a.out)


if __name__ == "__main__":
    main()
