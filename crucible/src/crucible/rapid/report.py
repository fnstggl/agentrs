"""Registry-wide measurements for the Tier 1 gates (evaluator side).

    python -m crucible.rapid report [--seeds 0 1 2] [--json out.json]

`admission_report(seeds)` admits every registered family and returns the reports, the summary counts, the
fingerprints, which families count as structurally distinct, and the per-family reasons they are distinct (the
machine-readable distinctness registry of Part 7.5). Results are cached on disk under `$CRUCIBLE_CACHE/rapid/`
keyed by a digest of this package's source code and the seeds, so a change to any rapid module recomputes them.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

PKG = os.path.dirname(os.path.abspath(__file__))


def source_digest() -> str:
    h = hashlib.blake2b(digest_size=16)
    for root, _, files in sorted(os.walk(PKG)):
        for f in sorted(files):
            if f.endswith(".py"):
                p = os.path.join(root, f)
                h.update(os.path.relpath(p, PKG).encode())
                h.update(open(p, "rb").read())
    import numpy
    h.update(numpy.__version__.encode())
    return h.hexdigest()


def cache_path(seeds) -> str:
    base = os.environ.get("CRUCIBLE_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "crucible")
    return os.path.join(base, "rapid", f"admission-{source_digest()}-{'-'.join(map(str, seeds))}.json")


def _admit_one(args):
    name, seeds = args
    from .admission import admit
    return name, json.loads(json.dumps(admit(name, seeds), default=_jsonable))


def admission_report(seeds=(0, 1, 2), use_cache: bool = True, jobs: int = 1) -> dict:
    from .families import FAMILIES
    path = cache_path(seeds)
    if use_cache and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    t0 = time.process_time()
    names = sorted(FAMILIES)
    if jobs > 1:
        import concurrent.futures as cf
        import multiprocessing as mp
        with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("fork")) as ex:
            reps = dict(ex.map(_admit_one, [(n, tuple(seeds)) for n in names]))
    else:
        reps = dict(_admit_one((n, tuple(seeds))) for n in names)
    return assemble(seeds, reps, cpu_s=time.process_time() - t0)


def assemble(seeds, reps: dict, cpu_s: float = 0.0, write: bool = True) -> dict:
    """The registry-wide report from per-family admission reports `reps` (name -> `admit(name, seeds)`, JSON
    form) - computed here or by separate processes (shards) on the same source tree - and its cache file."""
    from .admission import static_isolation, summary
    from .families import FAMILIES, REGISTRY
    from .fingerprints import counted, nearest
    names = sorted(FAMILIES)
    if sorted(reps) != names:
        raise ValueError(f"reports for {sorted(set(names) ^ set(reps))} missing or unknown")
    fps = {n: REGISTRY[n]["fingerprint"] for n in names}
    kept, dup = counted(fps)
    distinct = {n: dict(nearest(n, fps), distinct_because=FAMILIES[n].distinct_because,
                        verification=reps[n]["checks"]["distinct"].get("verification"),
                        recreates_4_0_sealed=reps[n]["checks"]["distinct"].get("recreates_4_0_sealed"),
                        exposes_4_0_novel_motifs=reps[n]["checks"]["distinct"].get("exposes_4_0_novel_motifs"))
                for n in fps}
    out = {"seeds": list(seeds), "source_digest": source_digest(), "reports": reps, "summary": summary(reps),
           "fingerprints": fps, "counted": kept, "cosmetic_duplicates": dup, "distinctness": distinct,
           "static_isolation": static_isolation(), "cpu_s": cpu_s}
    out = json.loads(json.dumps(out, default=_jsonable))
    if not write:
        return out
    path = cache_path(seeds)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, path)
    return out


def robust_report(seeds=(0, 1, 2), holdout=(10, 11, 12), use_cache: bool = True, jobs: int = 1) -> dict:
    """Admission on the admission seeds and on held-out seeds: a family counts only if it is admitted on both and
    its fingerprint is counted (not a cosmetic duplicate)."""
    a = admission_report(seeds, use_cache, jobs)
    h = admission_report(holdout, use_cache, jobs)
    both = sorted(n for n in a["reports"] if a["reports"][n]["admitted"] and h["reports"].get(n, {}).get("admitted")
                  and n in a["counted"])
    return {"admission": a, "holdout": h, "robust": both,
            "robust_train": [n for n in both if a["reports"][n]["split"] == "train"],
            "robust_eval": [n for n in both if a["reports"][n]["split"] in ("val", "test")]}


def _jsonable(x):
    import numpy as np
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, float) and x != x:
        return None
    return str(x)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m crucible.rapid")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="admit every family and print the gate counts")
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    r.add_argument("--json", default=None)
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--robust", action="store_true", help="also admit on held-out seeds 10-12")
    f = sub.add_parser("families", help="the family registry with fingerprints and distinctness reasons")
    f.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "report":
        if a.robust:
            rr = robust_report(tuple(a.seeds), use_cache=not a.no_cache, jobs=a.jobs)
            rep = rr["admission"]
            print(json.dumps({"robust_train": len(rr["robust_train"]), "robust_eval": len(rr["robust_eval"]),
                              "robust": rr["robust"]}, indent=1))
        else:
            rep = admission_report(tuple(a.seeds), use_cache=not a.no_cache, jobs=a.jobs)
        print(json.dumps(rep["summary"], indent=1))
        if a.json:
            json.dump(rep, open(a.json, "w"), indent=1)
        return 0
    from .families import REGISTRY
    from .fingerprints import nearest
    fps = {n: REGISTRY[n]["fingerprint"] for n in REGISTRY}
    out = {n: dict(REGISTRY[n], why_distinct=nearest(n, fps)) for n in REGISTRY}
    txt = json.dumps(out, indent=1, default=str)
    if a.json:
        open(a.json, "w").write(txt)
    else:
        print(txt)
    return 0
