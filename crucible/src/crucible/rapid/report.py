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


def admission_report(seeds=(0, 1, 2), use_cache: bool = True) -> dict:
    from .admission import admit, static_isolation, summary
    from .families import FAMILIES, REGISTRY
    from .fingerprints import counted, nearest
    path = cache_path(seeds)
    if use_cache and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    t0 = time.process_time()
    reps = {n: admit(n, seeds) for n in sorted(FAMILIES)}
    fps = {n: REGISTRY[n]["fingerprint"] for n in sorted(FAMILIES)}
    kept, dup = counted(fps)
    distinct = {n: dict(nearest(n, fps), distinct_because=FAMILIES[n].distinct_because) for n in fps}
    out = {"seeds": list(seeds), "source_digest": source_digest(), "reports": reps, "summary": summary(reps),
           "fingerprints": fps, "counted": kept, "cosmetic_duplicates": dup, "distinctness": distinct,
           "static_isolation": static_isolation(), "cpu_s": time.process_time() - t0}
    out = json.loads(json.dumps(out, default=_jsonable))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, path)
    return out


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
    f = sub.add_parser("families", help="the family registry with fingerprints and distinctness reasons")
    f.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "report":
        rep = admission_report(tuple(a.seeds), use_cache=not a.no_cache)
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
