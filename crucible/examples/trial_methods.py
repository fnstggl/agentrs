"""Validate in-world trial designs against matched counterfactual worlds (the operator's ground truth).

For predictive maintenance in `open_industry` (an intervention whose effects outlast a day), run each world twice:
firms use (a) day-level switchback trials, (b) sequential randomised blocks with washout. At the moment every trial
starts, the world is snapshotted; afterwards `crucible.evaluation.causal.installation_effect` forks that snapshot,
keeps or removes the installation for 16 days and measures the installer's true effect. Reported: each trial's
estimate, its standard error, the counterfactual effect, carry-over diagnostics, and per-design bias/RMSE.

Usage: PYTHONPATH=crucible/src python crucible/examples/trial_methods.py [days] [seeds...]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import crucible  # noqa: E402

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.inventors import MaintenanceInventor  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.evaluation.causal import installation_effect  # noqa: E402
from crucible.worldgen import generate  # noqa: E402


def run(seed, days, switchback):
    w = World.create(generate({"family": "open_industry", "overrides": {"params": {"trial_design_switchback": switchback}}},
                              seed))
    w.system_call("admin.join", {"handle": "mo", "endowment": 1_000_000})
    r = Runner(w, [("p:mo", MaintenanceInventor())], dt=3600, record=False)
    snaps = {}
    for _ in range(days * 24):
        r.run(1)
        for iid, i in w.state.scan("cap_installs"):
            if iid not in snaps and w.now // 86400 >= i["start_day"]:
                snaps[iid] = w.snapshot_blob()
    rows = []
    for iid, blob in sorted(snaps.items()):
        i = w.state.tables["cap_installs"][iid]
        ev = w.state.get("cap_evidence", i.get("evidence") or "")
        if ev is None:
            continue
        truth = installation_effect(World.from_blob(blob, world_id=f"cf-{seed}-{iid}"), iid, days=16)
        d = ev["data"]
        rows.append({"seed": seed, "install": iid, "firm": i["installer"], "design": i.get("design") or "switchback",
                     "estimate": d["effect_per_day"], "se": d["se"], "days_total": d.get("days_total"),
                     "carryover_suspected": d.get("carryover_suspected"), "truth": truth["effect_per_day"],
                     "truth_se": truth["se_over_days"], "decision": i["status"]})
    return rows


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    seeds = [int(x) for x in sys.argv[2:]] or [1, 2, 3]
    t0 = time.time()
    rows = []
    for seed in seeds:
        for sb in (True, False):
            rows += run(seed, days, sb)
            print(f"seed {seed} switchback={sb}: {len(rows)} trials so far ({time.time() - t0:.0f}s)", flush=True)
    summary = {}
    for design in sorted({r["design"] for r in rows}):
        rs = [r for r in rows if r["design"] == design]
        err = np.array([r["estimate"] - r["truth"] for r in rs], float)
        sign = np.mean([np.sign(r["estimate"]) == np.sign(r["truth"]) for r in rs])
        summary[design] = {"trials": len(rs), "bias": round(float(err.mean()), 1), "rmse": round(float(np.sqrt((err ** 2).mean())), 1),
                           "sign_agreement": round(float(sign), 3),
                           "mean_days": round(float(np.mean([r["days_total"] or 0 for r in rs])), 1),
                           "carryover_flags": sum(bool(r["carryover_suspected"]) for r in rs)}
    out = {"family": "open_industry", "capability": "predictive maintenance (MaintenanceInventor)", "days": days,
           "seeds": seeds, "summary": summary, "trials": rows, "wall_s": round(time.time() - t0, 1)}
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "experiments", "trial_methods.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
