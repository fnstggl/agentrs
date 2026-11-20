"""Two learning agents in one world: competition (pricing) and cooperation (joint advertising public good).

    PYTHONPATH=crucible/src python crucible/examples/multiagent_learning.py --days 40 --seeds 1,2,3

Competition: two PricingLearners serve the same need in one world (plus incumbents); control: two fixed-price
Builders. Cooperation: two AdLearners serve different needs and decide daily whether to fund a joint campaign that
advertises both; condition B adds a cost-sharing contract (each pays the other 50 % of a 3,000-cent daily budget
when it contributes... implemented as a fixed daily transfer clause between the firms). Everything measured from
authoritative state; the learners only see their own analytics and public market data.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import multiprocessing as mp  # noqa: E402

DAY = 86400


def run(job):
    import numpy as np
    import crucible
    crucible.load_all_modules()
    from crucible.agents.base import Runner
    from crucible.agents.learners import AdLearner, PricingLearner
    from crucible.agents.scripted import Builder
    from crucible.core.kernel import World
    from crucible.goals.metrics import m_independent_revenue
    from crucible.worldgen import generate
    kind, seed, days = job
    spec = generate({"family": "tiny", "overrides": {"n_consumers": 2500, "firms_per_need": 1,
                                                     "needs": ["route.shortest", "pack.knapsack"],
                                                     "params": {"cost_model": "deterministic"}}}, seed)
    w = World.create(spec)
    if kind == "compete_learn":
        pols = {"ana": PricingLearner("route.shortest", seed=seed), "ben": PricingLearner("route.shortest", seed=seed + 100)}
    elif kind == "compete_fixed":
        pols = {"ana": Builder(need="route.shortest", advertise=False), "ben": Builder(need="route.shortest", advertise=False)}
    elif kind in ("coop_learn", "coop_contract"):
        pols = {"ana": AdLearner("route.shortest", "ben", seed=seed), "ben": AdLearner("pack.knapsack", "ana", seed=seed + 100)}
    for h in pols:
        w.system_call("admin.join", {"handle": h, "endowment": 2_000_000})
    run = Runner(w, [(f"p:{h}", p) for h, p in pols.items()], dt=3600, seed=seed)
    if kind == "coop_contract":
        run.run(24 * 2)   # both launched; then they sign a cost-sharing agreement (fixed prefix, documented)
        orgs = {h: [m["org"] for m in w.state.find("members", "person", f"p:{h}")][0] for h in pols}
        r = w.execute("p:ana", {"tool": "contracts.propose", "args": {
            "parties": [orgs["ana"], orgs["ben"]], "as_org": orgs["ana"], "title": "co-marketing cost share",
            "clauses": [{"type": "payment", "from": orgs["ana"], "to": orgs["ben"], "amount": 1_500, "every_s": DAY,
                         "count": None, "start_in_s": 3600},
                        {"type": "payment", "from": orgs["ben"], "to": orgs["ana"], "amount": 1_500, "every_s": DAY,
                         "count": None, "start_in_s": 3600},
                        {"type": "notice", "seconds": DAY}]}})
        if r.ok:
            w.execute("p:ben", {"tool": "contracts.sign", "args": {"contract": r.result["contract"], "as_org": orgs["ben"]}})
    t = time.time()
    curve = []
    for d in range(days):
        run.run(24)
        row = {"day": d + 1}
        for h in pols:
            orgs = [m["org"] for m in w.state.find("members", "person", f"p:{h}") if m["role"] == "owner"]
            row[h] = sum(m_independent_revenue(w, o, DAY) for o in orgs)
        row["shares"] = w.modules["consumers"].market_shares(w)
        curve.append(row)
    out = {"kind": kind, "seed": seed, "days": days, "wall_s": round(time.time() - t, 1), "curve": curve,
           "invariants": w.check_invariants()}
    for h, p in pols.items():
        out[f"{h}_prices"] = getattr(p, "prices", None)
        out[f"{h}_contrib"] = getattr(p, "contrib", None)
        out[f"{h}_weights"] = p.weights() if hasattr(p, "weights") else (p.pol.W.tolist() if hasattr(p, "pol") else None)
        out[f"{h}_log"] = getattr(p, "log", [])[:5]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=40)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default="crucible/docs/experiments/multiagent_learning.json")
    a = ap.parse_args()
    jobs = [(k, int(s), a.days) for k in ("compete_learn", "compete_fixed", "coop_learn", "coop_contract")
            for s in a.seeds.split(",")]
    with mp.get_context("spawn").Pool(min(12, len(jobs))) as pool:
        res = pool.map(run, jobs)
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    for r in res:
        tail = r["curve"][-10:]
        print(r["kind"], r["seed"], {h: round(sum(x[h] for x in tail) / len(tail) / 100, 2) for h in ("ana", "ben")},
              "contrib(last10)", (r["ana_contrib"] or [])[-10:], (r["ben_contrib"] or [])[-10:],
              "prices(last)", (r["ana_prices"] or [None])[-1], (r["ben_prices"] or [None])[-1], r["invariants"])


if __name__ == "__main__":
    main()
