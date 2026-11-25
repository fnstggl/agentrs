"""Command-line runner of the capability diagnostics (exact, reproducible commands for the committed results).

    python -m superlearner.cam_unified.capabilities.run <capability> --seed S --out result.json [--set key=value ...]

capability: beliefs | reasoning | imagination | exploration | memory | causal. `--set` overrides fields of the
capability's settings dataclass (values parsed as JSON when possible, e.g. --set train_rounds=300
--set "hops=[1,2,3]"). Working files (archives, indices) go under --root."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time


def _settings(cap: str):
    if cap == "beliefs":
        from .beliefs import BeliefSettings, run_beliefs
        return BeliefSettings, run_beliefs
    if cap == "reasoning":
        from .reasoning import ReasoningSettings, run_reasoning
        return ReasoningSettings, run_reasoning
    if cap == "imagination":
        from .imagination import ImaginationSettings, run_imagination
        return ImaginationSettings, run_imagination
    if cap == "exploration":
        from .exploration import ExplorationSettings, run_exploration
        return ExplorationSettings, run_exploration
    if cap == "memory":
        from .memory import MemorySettings, run_memory
        return MemorySettings, run_memory
    if cap == "causal":
        from .causal import CausalSettings, run_causal
        return CausalSettings, run_causal
    raise SystemExit(f"unknown capability {cap!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capability")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--log-every", type=int, default=20)
    a = ap.parse_args(argv)
    S, run = _settings(a.capability)
    kw = {"seed": a.seed}
    fields = {f.name for f in dataclasses.fields(S)}
    for item in a.set:
        k, v = item.split("=", 1)
        if k not in fields:
            raise SystemExit(f"{S.__name__} has no field {k!r}")
        try:
            kw[k] = json.loads(v)
        except json.JSONDecodeError:
            kw[k] = v
        if isinstance(kw[k], list):
            kw[k] = tuple(kw[k])
    st = S(**kw)
    t0 = time.time()
    root = a.root or os.path.join(os.path.dirname(os.path.abspath(a.out)), "work")
    kws = {"log_every": a.log_every}
    import inspect
    if "root" in inspect.signature(run).parameters:
        kws["root"] = root
    res = run(st, **kws)
    res["command"] = " ".join(["python -m superlearner.cam_unified.capabilities.run"] + list(argv or sys.argv[1:]))
    res["wall_s_total"] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(f"wrote {a.out} ({res['wall_s_total']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
