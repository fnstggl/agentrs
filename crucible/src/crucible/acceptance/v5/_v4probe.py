"""Standalone probe of the Crucible 4.0 benchmark definitions, run as a script against any source tree:

    PYTHONPATH=<tree>/src python -P _v4probe.py definitions --seeds 0 1
    PYTHONPATH=<tree>/src python -P _v4probe.py execute --seeds 0 1 --days 1 [--families a,b]

`definitions` prints, as JSON, every registered world family (split and the SHA-256 of the generated WorldSpec per
seed, kernel version excluded), every mechanism family (engine, split, transfer kind and the hash of the generated
hidden mechanism per seed) and every useful-work category family (kind, relation, split and the hash of the
instantiated category's data per seed). Hashes are computed here with hashlib, independently of the tree's own
digest code, so the same script compares Crucible 4.0 (c07098b) with the current tree. Nothing is printed except
hashes and names: held-out families are hashed, never shown.

`execute` builds each world family's world on each seed and advances it by the given number of simulated days,
reporting per (family, seed) whether it ran, its event count, its final state digest and the error if any. With
`--deterministic` the worlds use 4.0's deterministic program-cost model (spec.params cost_model), the mode in which
4.0 guarantees identical trajectories across fresh executions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time


def _plain(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if callable(x):
        return f"<callable {getattr(x, '__qualname__', type(x).__name__)}>"
    if isinstance(x, (set, frozenset)):
        return sorted(_plain(v) for v in x)
    if hasattr(x, "__dict__"):
        return {"__type__": type(x).__name__, **_plain({k: v for k, v in vars(x).items() if not k.startswith("_")})}
    return f"<{type(x).__name__}>"            # never a repr: it may carry a memory address


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def definitions(seeds: list[int]) -> dict:
    import crucible
    crucible.load_all_modules()
    from crucible.mechanisms import base as MB
    from crucible.mechanisms import load_engines
    from crucible.usefulness import category as UC
    from crucible.worldgen import generator as G
    load_engines()
    out = {"world": {}, "mech": {}, "useful": {}}
    for name in sorted(G.FAMILIES):
        f = G.FAMILIES[name]
        d = {}
        for s in seeds:
            try:
                spec = G.generate({"family": name}, s).to_dict()
                spec.pop("kernel", None)
                d[str(s)] = _h(spec)
            except Exception as exc:                      # noqa: BLE001 - reported as a definition difference
                d[str(s)] = f"error:{type(exc).__name__}"
        out["world"][name] = {"split": f.split, "digests": d}
    for name in sorted(MB.MECH_FAMILIES):
        f = MB.MECH_FAMILIES[name]
        d = {}
        for s in seeds:
            try:
                d[str(s)] = _h(MB.make_mechanism(name, s))
            except Exception as exc:                      # noqa: BLE001
                d[str(s)] = f"error:{type(exc).__name__}"
        out["mech"][name] = {"engine": f.engine, "split": f.split, "transfer": f.transfer, "digests": d}
    UC.load_families()
    for name in sorted(UC.FAMILIES):
        f = UC.FAMILIES[name]
        d = {}
        for s in seeds:
            try:
                cat = f.instantiate(s)
                data = {k: v for k, v in vars(cat).items() if not callable(v) and not k.startswith("_")}
                d[str(s)] = _h(data)
            except Exception as exc:                      # noqa: BLE001
                d[str(s)] = f"error:{type(exc).__name__}"
        out["useful"][name] = {"kind": f.kind, "relation": f.relation, "split": f.split, "digests": d}
    return out


def execute(seeds: list[int], days: float, only: list[str] | None, deterministic: bool = False) -> dict:
    import crucible
    crucible.load_all_modules()
    from crucible.core.kernel import World
    from crucible.worldgen import generator as G
    out = {}
    for name in sorted(only or G.FAMILIES):
        out[name] = {}
        for s in seeds:
            t0 = time.time()
            try:
                spec = G.generate({"family": name}, s)
                if deterministic:          # 4.0's reproducible mode: program cost from the deterministic model
                    spec.params["cost_model"] = "deterministic"
                w = World.create(spec)
                n0 = len(w.log)
                w.advance_time(w.now + int(days * 86400))
                out[name][str(s)] = {"ok": True, "events": len(w.log), "events_during_run": len(w.log) - n0,
                                     "state_digest": w.digest(), "wall_s": round(time.time() - t0, 2)}
            except Exception as exc:                      # noqa: BLE001 - this is the measurement
                out[name][str(s)] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                                     "wall_s": round(time.time() - t0, 2)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("definitions", "execute"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--days", type=float, default=1.0)
    ap.add_argument("--families", default="")
    ap.add_argument("--deterministic", action="store_true")
    a = ap.parse_args()
    if a.what == "definitions":
        res = definitions(a.seeds)
    else:
        res = execute(a.seeds, a.days, [f for f in a.families.split(",") if f] or None, a.deterministic)
    sys.stdout.write("\n@@JSON@@" + json.dumps(res, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
