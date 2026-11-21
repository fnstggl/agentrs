"""Demonstration 4 - an emergent supply chain under real constraints, a supplier failure, and an agent-made
optimisation, compared across four forks of one world.

World: `open_fragile_suppliers` (thin freight network; one component maker supplies both assemblers). At the fork
point the same world continues in four branches: base; supplier failure (every machine of the shared component
maker breaks - an operator intervention); predictive maintenance invented by an agent (adoption decided by the
firms' own trials); failure + maintenance. Measured from state: who trades with whom (the chain as it exists),
downstream stockouts and supply shortfalls, retail units sold, substitutions, operating results.

Usage: PYTHONPATH=crucible/src python crucible/examples/demo_open_supplychain.py [seed] [days]
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
from crucible.agents.inventors import MaintenanceInventor  # noqa: E402
from crucible.agents.scripted import IdlePolicy  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.economy.operations import operating_result  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400


def measure(w, d0, d1):
    st = w.state
    ops = w.modules["ops"]
    firms = sorted(m["org"] for _, m in st.scan("minds") if m["kind"] == "firm" and m["role"] not in ("provider", "utility"))
    loss = {f: {} for f in firms}
    op = {}
    for f in firms:
        op[f] = 0
        for d in range(d0, d1):
            r = ops.day(st, f, d)
            op[f] += operating_result(r)
            for k, v in (r or {"loss_n": {}})["loss_n"].items():
                loss[f][k] = loss[f].get(k, 0) + v
    edges = {}
    for _, t in st.scan("trade_orders"):
        if t["t"] >= d0 * DAY:
            k = f"{t['seller']}->{t['buyer']}"
            edges[k] = edges.get(k, 0) + t["qty"]
    return {"operating": op, "losses": loss, "trade_edges": edges,
            "retail_sold": sum(r.get("sold_total", 0) for _, r in st.scan("retail")),
            "substitutions": [dict(e, firm=k) for k, m in st.scan("minds") for e in m["memory"] if e["kind"] == "substitute"],
            "maintenance_adopters": sorted(i["installer"] for _, i in st.scan("cap_installs") if i["status"] == "active")}


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    t0 = time.time()
    w = World.create(generate({"family": "open_fragile_suppliers"}, seed), world_id="sc-base")
    w.system_call("admin.join", {"handle": "mo", "endowment": 1_000_000})
    Runner(w, [("p:mo", IdlePolicy())], dt=3600, record=False).run(24 * 8)
    fork_day = w.now // DAY
    shared = "o:forge1"
    branches = {"base": w, "failure": w.fork("sc-failure"), "maintenance": w.fork("sc-maint"),
                "failure+maintenance": w.fork("sc-fail-maint")}
    ind = w.modules["industry"]
    for name in ("failure", "failure+maintenance"):
        b = branches[name]
        for mid in b.state.find_ids("machines", "owner", shared):
            b._run_system("shock", lambda ctx, mid=mid: b.modules["industry"].force_breakdown(ctx, mid, parts_days=7))
    runners = {n: Runner(b, [("p:mo", MaintenanceInventor() if "maintenance" in n else IdlePolicy())], dt=3600,
                         record=False, seed=1) for n, b in branches.items()}
    for _ in range(days):
        for n in branches:
            runners[n].run(24)
    res = {n: measure(b, fork_day, b.now // DAY) for n, b in branches.items()}
    for b in branches.values():
        assert b.check_invariants() == []
    down = ["o:assem1", "o:assem2"]
    summary = {n: {"downstream_stockouts": sum(r["losses"][f].get("stockout", 0) for f in down),
                   "downstream_supply_shortfall": sum(r["losses"][f].get("supply_shortfall", 0) for f in down),
                   "retail_sold": r["retail_sold"], "substitutions": len(r["substitutions"]),
                   "chain_operating_total": sum(r["operating"].values()),
                   "maintenance_adopters": r["maintenance_adopters"]} for n, r in res.items()}
    out = {"family": "open_fragile_suppliers", "seed": seed, "fork_day": fork_day, "days": days,
           "shared_supplier": shared, "summary": summary, "detail": res, "wall_s": round(time.time() - t0, 1)}
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "experiments", "open_supplychain.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps(summary, indent=1))
    print("trade edges (base):", res["base"]["trade_edges"])
    del ind


if __name__ == "__main__":
    main()
