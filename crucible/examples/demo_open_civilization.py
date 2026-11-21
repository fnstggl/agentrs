"""Demonstration 5 - a long-running open civilisation, with its structure measured over time.

World `open_mixed`: service firms with daily demand cycles, a multi-tier industrial chain, households, workers,
compute providers, lenders, simulated entrepreneurs, negotiation and institutions. Agents: three competent-agent
controls (capacity automation, predictive maintenance, process research) plus an idle agent. Every 5 days the
independent census (`crucible.evaluation.open_endedness.census`) records the world's structure; the full event
timeline is persisted (sqlite) so it can be replayed and browsed in the dashboard.

Usage: PYTHONPATH=crucible/src python crucible/examples/demo_open_civilization.py [seed] [days]
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import crucible  # noqa: E402

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.inventors import CapacityInventor, MaintenanceInventor, ProcessResearcher  # noqa: E402
from crucible.agents.scripted import IdlePolicy  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.evaluation.open_endedness import census  # noqa: E402
from crucible.storage.postgres import open_store  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 90
    t0 = time.time()
    root = os.path.join(os.path.dirname(__file__), "..", "..", "crucible-data")
    os.makedirs(root, exist_ok=True)
    store = open_store(f"sqlite:///{os.path.join(root, f'open_civ_{seed}.db')}")
    w = World.create(generate({"family": "open_mixed", "overrides": {"n_consumers": 3000, "firms_per_need": 3,
                                                                     "workers": 30}}, seed),
                     world_id=f"open-civ-{seed}", store=store)
    agents = []
    for h, pol in [("ada", CapacityInventor(open_source=True)), ("mo", MaintenanceInventor()),
                   ("rhea", ProcessResearcher()), ("ivan", IdlePolicy())]:
        w.system_call("admin.join", {"handle": h, "endowment": 6_000_000})
        agents.append((f"p:{h}", pol))
    run = Runner(w, agents, dt=3600, record=False, seed=seed)
    timeline = [census(w)]
    for d in range(days):
        run.run(24)
        if (d + 1) % 5 == 0:
            timeline.append(census(w))
            w.flush()
            c = timeline[-1]
            print(f"day {c['day']:3d}  events {c['events']:7d}  caps {c['capabilities']['published']} "
                  f"(adopted {c['capabilities']['with_independent_adopters']})  markets "
                  f"{c['technology']['emergent_capability_markets']}  notes sold {c['industry']['process_notes_sold']}  "
                  f"retail {c['industry']['retail_units_sold']}  orgs+{c['organisations']['created_after_genesis']}  "
                  f"deals {c['institutions']['deals']}  wall {time.time() - t0:.0f}s", flush=True)
    w.flush()
    assert w.check_invariants() == [] and w.modules["bank"].check_ledger_consistency(w) == []
    t1 = time.time()
    w2 = World.load(store, w.world_id)
    replay_ok = w2.digest() == w.digest()
    out = {"family": "open_mixed", "seed": seed, "days": days, "agents": [a for a, _ in agents],
           "store": f"crucible-data/open_civ_{seed}.db", "world_id": w.world_id, "replay_from_store_identical": replay_ok,
           "replay_s": round(time.time() - t1, 1), "wall_s": round(t1 - t0, 1), "census": timeline,
           "final": timeline[-1]}
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "experiments", "open_civilization.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("replay from store identical:", replay_ok, " events", w.seq, " wall", out["wall_s"], "s")


if __name__ == "__main__":
    main()
