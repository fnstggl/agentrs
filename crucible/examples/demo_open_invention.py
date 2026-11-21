"""Demonstration 1 - a new product category arising during a world's lifetime, with a counterfactual fork.

For each invention class (cloud capacity automation in `open_digital`; predictive maintenance in `open_industry`):

1. generate a persistent world; independent firms (Tier-2 minds) run their businesses with default routines;
2. run until a firm *publicly* reports a problem from its own books (an `ops.request`) - this is the fork point;
3. fork: in the factual branch a scripted inventor (a competent-agent control) reads the requests, writes a new
   kind of service, deploys it, publishes it as a capability and answers the requests; in the counterfactual
   branch the same agent does nothing;
4. run both branches in lockstep and record, per day: adoption, verified trial outcomes, firms' operating results,
   the losses the invention targets, and secondary effects (new requests, entrants, infrastructure prices);
5. regenerate the factual branch from its event log and check it is identical.

Usage:  PYTHONPATH=crucible/src python crucible/examples/demo_open_invention.py [seed] [days]
Writes crucible/docs/experiments/open_invention.json
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
from crucible.agents.inventors import CapacityInventor, MaintenanceInventor  # noqa: E402
from crucible.agents.scripted import IdlePolicy  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.economy.operations import operating_result  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400


def firm_metrics(w, firms, day):
    st = w.state
    ops = w.modules["ops"]
    out = {}
    for f in firms:
        r = ops.day(st, f, day)
        out[f] = {"operating": operating_result(r), "losses": dict(r["losses"]) if r else {},
                  "loss_events": dict(r["loss_n"]) if r else {},
                  "compute": -(r["flows"].get("compute", 0)) if r else 0}
    return out


def run_case(name, family, inventor, seed, days, overrides=None):
    t0 = time.time()
    spec = generate({"family": family, "overrides": overrides or {}}, seed)
    w = World.create(spec, world_id=f"{name}-factual")
    w.system_call("admin.join", {"handle": "ada", "endowment": 1_000_000})
    firms = sorted(m["org"] for _, m in w.state.scan("minds") if m["kind"] == "firm" and m["role"] not in
                   ("provider", "utility"))
    # 1. run with an idle agent until the first public problem report: the fork point
    idle = Runner(w, [("p:ada", IdlePolicy())], dt=3600, record=False)
    while not w.state.count("ops_requests"):
        idle.run(24)
    fork_day = w.now // DAY
    cf = w.fork(f"{name}-counterfactual")
    fac = Runner(w, [("p:ada", inventor)], dt=3600, record=False, seed=1)
    cfr = Runner(cf, [("p:ada", IdlePolicy())], dt=3600, record=False, seed=1)
    timeline = []
    for d in range(days):
        fac.run(24)
        cfr.run(24)
        day = w.now // DAY - 1
        st = w.state
        timeline.append({
            "day": day,
            "factual": firm_metrics(w, firms, day), "counterfactual": firm_metrics(cf, firms, day),
            "installs": {i["installer"]: i["status"] for _, i in st.scan("cap_installs")},
            "capabilities": st.count("capabilities"), "requests": st.count("ops_requests"),
            "requests_cf": cf.state.count("ops_requests"),
            "provider_prices": {k: p["price"] for k, p in sorted(st.scan("providers"))},
            "provider_prices_cf": {k: p["price"] for k, p in sorted(cf.state.scan("providers"))}})
    st = w.state
    cid = getattr(inventor, "capability", None)
    evidence = [{"by": e["by"], "independent": e["independent"], **{k: e["data"].get(k) for k in (
        "effect_per_day", "se", "rel_effect", "days_on", "days_off")}} for _, e in st.scan("cap_evidence")
        if e["kind"] == "outcome"]
    adopters = sorted(i["installer"] for _, i in st.scan("cap_installs") if i["status"] == "active")
    rejected = sorted(i["installer"] for _, i in st.scan("cap_installs") if i["status"] == "removed")
    total = {}
    for branch in ("factual", "counterfactual"):
        total[branch] = {f: {"operating": sum(t[branch][f]["operating"] for t in timeline),
                             "losses": sum(sum(t[branch][f]["losses"].values()) for t in timeline),
                             "loss_events": sum(sum(t[branch][f]["loss_events"].values()) for t in timeline),
                             "compute": sum(t[branch][f]["compute"] for t in timeline)} for f in firms}
    effect = {f: total["factual"][f]["operating"] - total["counterfactual"][f]["operating"] for f in firms}
    # secondary effects: problems reported only in the factual branch after the invention appeared
    pub_t = st.tables["capabilities"][cid]["t"] if cid else None
    new_kinds = sorted({tag for _, r in st.scan("ops_requests") if pub_t and r["t"] > pub_t for tag in r["tags"]}
                       - {tag for _, r in cf.state.scan("ops_requests") for tag in r["tags"]})
    entrants = [c["host"] for _, c in st.scan("capabilities") if c["improves"] == cid]
    cats = w.modules["caps"].categories(w, 1)
    markets = [dict(m, id=k) for k, m in st.scan("cap_markets")]
    # 5. exact regeneration of the factual branch from its log
    t1 = time.time()
    regen_ok = None
    if w.log_base == 0 or True:
        evs = [e.to_dict() for e in w.log] if w.log_base == 0 else None
        if evs is not None:
            w2 = World.regenerate(w.spec, evs)
            regen_ok = (w2.last_hash == w.last_hash and w2.digest() == w.digest())
    res = {"case": name, "family": family, "seed": seed, "fork_day": fork_day, "days": days,
           "capability": cid, "adopters": adopters, "rejected_after_trial": rejected, "verified_outcomes": evidence,
           "independent_adopters": w.modules["caps"].summary(w, cid)["independent_adopters"] if cid else 0,
           "inventor_revenue": sum(p["cents"] for _, p in st.scan("cap_payers")),
           "categories": cats, "markets": markets,
           "counterfactual_effect_operating_cents": effect,
           "totals": total, "secondary": {"new_problem_tags_after_invention": new_kinds, "entrants": entrants,
                                          "provider_price_change": {k: timeline[-1]["provider_prices"][k] -
                                                                    timeline[-1]["provider_prices_cf"][k]
                                                                    for k in timeline[-1]["provider_prices"]}},
           "regeneration_identical": regen_ok, "events": w.seq,
           "wall_s": round(time.time() - t0, 1), "regen_wall_s": round(time.time() - t1, 1),
           "timeline": timeline}
    assert w.check_invariants() == [] and cf.check_invariants() == []
    return res


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    out = {"cases": [run_case("capacity", "open_digital", CapacityInventor(open_source=True), seed, days,
                              {"firms_per_need": 3, "workers": 20}),
                     run_case("maintenance", "open_industry", MaintenanceInventor(), seed, days)]}
    for c in out["cases"]:
        print(f"== {c['case']} ({c['family']} seed {seed}): fork day {c['fork_day']}, capability {c['capability']}")
        print(f"   adopters {c['adopters']}  rejected {c['rejected_after_trial']}  "
              f"independent adopters {c['independent_adopters']}  inventor revenue {c['inventor_revenue']}")
        for e in c["verified_outcomes"]:
            print(f"   trial {e['by']}: {e['effect_per_day']:+.0f} +/- {e['se']:.0f} cents/day (rel {e['rel_effect']})")
        print("   counterfactual effect on operating result (cents, whole run):",
              {k: v for k, v in c["counterfactual_effect_operating_cents"].items()})
        print("   secondary:", c["secondary"], " markets:", [m["label"] for m in c["markets"]])
        print("   regeneration identical:", c["regeneration_identical"], " events", c["events"], " wall", c["wall_s"], "s")
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "experiments", "open_invention.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("wrote", os.path.normpath(path))


if __name__ == "__main__":
    main()
