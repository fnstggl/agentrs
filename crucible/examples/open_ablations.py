"""Ablations of open-world mechanisms on a fixed world configuration with independent seeds.

Each mechanism is switched off through a world parameter; the same agents (capacity inventor, maintenance
inventor) act in every run. Output: per-flag means/sds of structural outcomes (see evaluation/open_endedness.py).

Usage: PYTHONPATH=crucible/src python crucible/examples/open_ablations.py [days] [seeds...]
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import crucible  # noqa: E402

crucible.load_all_modules()
from crucible.agents.inventors import CapacityInventor, MaintenanceInventor  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.evaluation.open_endedness import ablation  # noqa: E402
from crucible.worldgen import generate  # noqa: E402


def make_world(params, seed):
    return World.create(generate({"family": "open_mixed", "overrides": {"n_consumers": 2500, "firms_per_need": 3,
                                                                        "workers": 24, "params": params}}, seed))


def make_agents(w):
    out = []
    for h, pol in [("ada", CapacityInventor(open_source=True)), ("mo", MaintenanceInventor())]:
        w.system_call("admin.join", {"handle": h, "endowment": 3_000_000})
        out.append((f"p:{h}", pol))
    return out


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seeds = [int(x) for x in sys.argv[2:]] or [1, 2]
    t0 = time.time()
    flags = [None, "ablate_adoption", "ablate_sharing", "ablate_requests", "ablate_entry", "ablate_diffusion"]
    res = ablation(make_world, make_agents, days, seeds, flags)
    res["config"] = {"family": "open_mixed", "days": days, "seeds": seeds, "wall_s": round(time.time() - t0, 1)}
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "experiments", "open_ablations.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    for flag, s in res["summary"].items():
        print(flag, {k: v["mean"] for k, v in s.items() if k != "institutions.deals"})


if __name__ == "__main__":
    main()
