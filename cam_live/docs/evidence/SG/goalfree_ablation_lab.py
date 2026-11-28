"""SG diagnostic (debug model on CPU, LabWorld, the agent API in the runtime's order): does a goal-free period before an
assigned objective change how well the objective goes? Same world seed and agent seed per pair; arm "goalfree" spends
``--free`` decisions without an objective (goal-free mode chooses its own learning goals) and then gets the objective,
arm "none" gets the objective at once. The goal-free period runs on its own instance of the same world (same seed:
same mechanism; its own clock-independent state, budget and adopters), the objective on a fresh instance in both arms. Measured on the objective: verified adopters after its decisions and 12 h of
drain, its decisions' failures, and (goal-free arm) the transfer evidence the objective's outcomes produced.

    PYTHONPATH=cam_live/src .venv/bin/python cam_live/docs/evidence/SG/goalfree_ablation_lab.py --seeds 0,1,2 \
        --free 8 --decisions 16 --out OUT.json
"""
import argparse
import json
import os
import statistics
import sys
import tempfile

import torch


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--free", type=int, default=8)
    ap.add_argument("--decisions", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "..", "..", "tests", "integration"))
    from int_helpers import DriverState, agent_config, drive, lab_config
    from cam_live.agent import CamLiveAgent
    from cam_live.contracts import Limits, SimClock, TaskSpec, Trust
    from cam_live.diagnostics.world import LabVerifier, LabWorld, lab_task

    rows = []
    for s in [int(x) for x in a.seeds.split(",")]:
        for arm in ("none", "goalfree"):
            clock = SimClock()
            agent = CamLiveAgent(agent_config(seed=s), tempfile.mkdtemp(prefix="sg_gfab_"), clock=clock)
            try:
                free_ops = []
                if arm == "goalfree":
                    # the goal-free period runs on its own instance of the same world (same mechanism, its own time,
                    # budget and adopters), so that only what was learned can carry over - not resources spent or
                    # adopters gained (a first version shared one world: the period's waits and spending were a
                    # confound)
                    gworld = LabWorld(lab_config(300 + s, target=30), clock)
                    gver = LabVerifier(gworld)
                    agent.attach_verifier(gver)
                    agent.set_objective(TaskSpec(objective_id=f"free_{s}", objective="", success={},
                                                 family="lab/integration",
                                                 limits=Limits(duration_s=2 * 86400.0, money=40.0, max_decisions=a.free),
                                                 mode="diagnostic", origin=Trust.SELF))
                    free_ops = [d["operation"] for d in drive(agent, gworld, clock, gver, DriverState(), a.free)]
                world = LabWorld(lab_config(300 + s, target=30), clock)
                verifier = LabVerifier(world)
                agent.attach_verifier(verifier)
                st = DriverState()
                before = float(verifier.verify(lab_task(world, "probe")).value)
                task = lab_task(world, f"assigned_{s}")
                agent.set_objective(task)
                out = drive(agent, world, clock, verifier, st, a.decisions)
                clock.advance(12 * 3600.0)
                drive(agent, world, clock, verifier, st, 1)           # collect what matured
                v = verifier.verify(task)
                rows.append({"seed": s, "arm": arm, "free_decisions": len(free_ops), "free_ops": free_ops,
                             "adopters_before": before, "adopters_after": float(v.value),
                             "gain": float(v.value) - before, "objective_ops": [d["operation"] for d in out],
                             "goal_free_goals": len(agent.goal_free.selector.history),
                             "transfer_points": len(agent.goal_free.transfer.points),
                             "transfer_evidence": int(agent.counters["transfer_evidence"]),
                             "transfer_by_key": agent.goal_free.transfer.by_key()})
                print(json.dumps({k: rows[-1][k] for k in ("seed", "arm", "gain", "goal_free_goals", "transfer_points")}),
                      flush=True)
            finally:
                agent.close()
    summ = {}
    for arm in ("none", "goalfree"):
        g = [r["gain"] for r in rows if r["arm"] == arm]
        summ[arm] = {"n": len(g), "mean_gain": statistics.mean(g) if g else None, "gains": g}
    pairs = [(r2["gain"] - r1["gain"]) for r1 in rows for r2 in rows
             if r1["seed"] == r2["seed"] and r1["arm"] == "none" and r2["arm"] == "goalfree"]
    summ["paired_difference"] = {"values": pairs, "mean": statistics.mean(pairs) if pairs else None,
                                 "positive": sum(d > 0 for d in pairs), "n": len(pairs)}
    json.dump({"what": __doc__.strip().splitlines()[0], "args": vars(a), "rows": rows, "summary": summ},
              open(a.out, "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
