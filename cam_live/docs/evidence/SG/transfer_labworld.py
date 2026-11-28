"""SG diagnostic (debug model, LabWorld): does goal-free learning improve the prediction of a later assigned objective's
outcomes, and does the learned transfer estimate track it? Per seed: a goal-free period of 24 self-chosen probes, then
12 outcomes of a later objective either in the same world (mechanism persists) or in a re-drawn world (mechanism
changed). gain = NLL_reference - NLL_goal_free on the same later outcome (nats).

    PYTHONPATH=cam_live/src .venv/bin/python cam_live/docs/evidence/SG/transfer_labworld.py --seeds 0,1,2,3,4 --out OUT.json
"""
import argparse
import json
import statistics
import sys

import torch


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    from cam_live.config import CamLiveConfig
    from cam_live.contracts import ActionSpec, Level, Modality, OutcomeKind, OutcomeQuery, new_id
    from cam_live.diagnostics.world import AllowAllPolicy, LabConfig, LabWorld, lab_task
    from cam_live.nn.model import build_model
    from cam_live.reward.selfgoals import GoalFreeBudget, GoalFreeMode

    def probe(world, ch):
        tools = {t.name: t for t in world.discover_tools()}
        act = ActionSpec(new_id("act"), "lab.probe", Modality.API, Level.INTERVENTION, arguments={"channel": ch},
                         est_cost=tools["lab.probe"].est_cost)
        return act, OutcomeQuery(new_id("q"), f"lab.probe.{ch}", OutcomeKind.REAL, 600.0, description="reading")

    rows = []
    for s in [int(x) for x in a.seeds.split(",")]:
        for later in ("same", "redrawn"):
            torch.manual_seed(s)
            model = build_model(CamLiveConfig.debug(seed=s))
            world = LabWorld(LabConfig(n_channels=3, seed=100 + s))
            gf = GoalFreeMode(model, model.new_fast_state(), GoalFreeBudget(seconds=1e7, decisions=24, money=1e3))
            b = torch.zeros(1, model.d_model)
            pol, task = AllowAllPolicy(), lab_task(world)
            for _ in range(24):
                goals = gf.propose(b, [probe(world, ch) for ch in range(3)])
                if not goals:
                    break
                g = goals[0]
                res = world.execute(g.action, pol.authorize(g.action, task, {}))
                gf.resolve(g, res.observations[0].quantities[0].value, res.cost)
            before = [c.transfer for c, _, _ in gf.candidates(b, [probe(world, ch) for ch in range(3)])]
            w2 = world if later == "same" else LabWorld(LabConfig(n_channels=3, seed=900 + s,
                                                                   w_mean=(-1.0, 0.5, -0.3)))
            t2 = lab_task(w2)
            gains = []
            for i in range(12):
                act, q = probe(w2, i % 3)
                y = w2.execute(act, pol.authorize(act, t2, {})).observations[0].quantities[0].value
                phi = gf._features(b, [(act, q)])[0]
                gains.append(gf.observe_assigned(gf.key_of(act, q), phi, q.kind, y, objective_id="later"))
            after = [c.transfer for c, _, _ in gf.candidates(b, [probe(w2, ch) for ch in range(3)])]
            rows.append({"seed": s, "later": later, "goal_free_goals": len(gf.selector.history),
                         "mean_gain_nats": statistics.mean(gains), "positive": sum(g > 0 for g in gains),
                         "n": len(gains), "transfer_before": before, "transfer_after": after})
            print(json.dumps(rows[-1]), flush=True)
    summ = {}
    for later in ("same", "redrawn"):
        r = [x for x in rows if x["later"] == later]
        summ[later] = {"mean_gain_nats": statistics.mean(x["mean_gain_nats"] for x in r),
                       "seeds_with_positive_mean_gain": sum(x["mean_gain_nats"] > 0 for x in r), "seeds": len(r),
                       "mean_transfer_after": statistics.mean(statistics.mean(x["transfer_after"]) for x in r),
                       "transfer_before_values": sorted({v for x in r for v in x["transfer_before"]})}
    json.dump({"what": __doc__.strip().splitlines()[0], "rows": rows, "summary": summ}, open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
