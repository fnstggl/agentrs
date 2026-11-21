"""Second-order emergence: an invention creates a problem that did not exist before it, and that problem is solved
by another actor - measured against a counterfactual world without the first invention.

World `open_tight_compute`: firms serve daily demand cycles from the cheapest provider, which is small; a pricier
provider has spare capacity. Two scripted competent-agent controls act only on public problem reports:
  * `ada` (CapacityInventor) answers overload reports with an hourly autoscaler;
  * `bea` (PlacementBroker) answers *provider capacity* reports with a placement broker that moves a service to a
    provider with free cores when its own is full.
Nothing tells `bea` that its problem will appear, and nothing in the world mentions placement. Fork at the first
public problem report; in the counterfactual `ada` stays idle and `bea` is present but has nothing to answer.

Stages are detected from authoritative state (not asserted):
 1 first problem report (overload)        2 autoscaler published           3 first independent adoption
 4 compute served/released hourly by adopters (cores-hours)               5 first `capacity_unavailable` loss
 6 first public report of it               7 placement broker published     8 first trial / adoption of it
 9 first migration to another provider    10 provider reactions (price, hardware)  11 entrants / markets

Usage: PYTHONPATH=crucible/src python crucible/examples/demo_second_order.py [seed] [days] [cheap_cores]
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
from crucible.agents.inventors import CapacityInventor, PlacementBroker  # noqa: E402
from crucible.agents.scripted import IdlePolicy  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.economy.operations import operating_result  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400


def daily(w, firms, day):
    st = w.state
    ops = w.modules["ops"]
    row = {"day": day, "unavailable": 0, "overload": 0, "operating": 0, "served": 0}
    for f in firms:
        r = ops.day(st, f, day)
        if r:
            row["unavailable"] += r["loss_n"].get("capacity_unavailable", 0)
            row["overload"] += r["loss_n"].get("capacity_overload", 0)
            row["operating"] += operating_result(r)
    row["served"] = sum(r["ok"] for k, r in st.scan("svc_days") if k.endswith(f"|{day}"))
    row["compute_spend"] = sum(-(ops.day(st, f, day) or {"flows": {}})["flows"].get("compute", 0) for f in firms)
    row["revenue"] = sum((ops.day(st, f, day) or {"flows": {}})["flows"].get("sale", 0) for f in firms)
    row["providers"] = {k: {"price": p["price"], "cores": p["cores"],
                            "used": sum(i["cores"] for i in st.find("instances", "provider", k) if i["status"] == "running")}
                        for k, p in sorted(st.scan("providers"))}
    row["hosts_on"] = {k: sorted(h for h, r in st.scan("hosts") if r["owner"] in firms and
                                 st.tables["instances"][st.tables["deployments"][r["deployment"]]["instance"]]["provider"] == k)
                       for k in sorted(st.ids("providers"))}
    return row


def stages(w, firms, inv_cap, brk_cap):
    st = w.state
    out = {}
    reqs = sorted(st.scan("ops_requests"), key=lambda kv: kv[1]["t"])

    def first_req(kind):
        for k, r in reqs:
            if (r.get("evidence") or {}).get("loss_events", {}).get(kind) or kind in r["tags"]:
                return {"t": r["t"], "day": r["t"] / DAY, "by": r["by"], "request": k}
        return None
    out["1_first_overload_report"] = first_req("capacity_overload")
    caps = {k: c for k, c in st.scan("capabilities")}
    if inv_cap:
        out["2_autoscaler_published"] = {"day": caps[inv_cap]["t"] / DAY, "capability": inv_cap}
    ad = sorted((i["t"], i["installer"], i["capability"]) for _, i in st.scan("cap_installs")
                if i["status"] in ("active",) and caps[i["capability"]]["host"].startswith("autoscale"))
    out["3_autoscaling_adopters_at_end"] = [{"firm": f, "capability": c} for _, f, c in ad]
    lossdays = sorted(r["day"] for _, r in st.scan("ops_days") if r["owner"] in firms and
                      r["loss_n"].get("capacity_unavailable"))
    out["5_first_capacity_unavailable_loss_day"] = lossdays[0] if lossdays else None
    out["6_first_capacity_unavailable_report"] = first_req("capacity_unavailable")
    if brk_cap:
        out["7_placement_published"] = {"day": caps[brk_cap]["t"] / DAY, "capability": brk_cap}
        tr = sorted((i["t"], i["installer"], i["status"]) for _, i in st.scan("cap_installs") if i["capability"] == brk_cap
                    or caps[i["capability"]].get("improves") == brk_cap)
        out["8_placement_trials"] = [{"day": t / DAY, "firm": f, "status": s} for t, f, s in tr]
    migr = []
    for e in w.log:
        if e.type == "cloud.deployment" and e.payload["owner"] in firms:
            prov = st.tables["instances"][e.payload["instance"]]["provider"]
            if prov != "o:cloud1":
                migr.append({"day": e.t / DAY, "firm": e.payload["owner"], "host": e.payload["host"], "to": prov,
                             "by": e.actor})
    out["9_migrations"] = migr
    hw = sorted((o["provider"], o.get("cores")) for _, o in st.scan("hw_orders"))
    out["10_provider_reactions"] = {"hardware_orders": hw,
                                    "prices": {k: p["price"] for k, p in sorted(st.scan("providers"))}}
    out["trial_evidence"] = [{"capability": e["capability"], "host": caps[e["capability"]]["host"], "firm": e["by"],
                              **{k: e["data"].get(k) for k in ("effect_per_day", "se", "design", "days_total",
                                                               "carryover_suspected")}, "independent": e["independent"]}
                             for _, e in st.scan("cap_evidence") if e["kind"] == "outcome"]
    out["final_installs"] = sorted((i["installer"], caps[i["capability"]]["host"], i["status"], i.get("reason", ""))
                                   for _, i in st.scan("cap_installs"))
    out["11_entrants"] = sorted((c["host"], caps[c["improves"]]["host"]) for _, c in st.scan("capabilities")
                                if c.get("improves") and c["improves"] in caps)
    out["11_markets"] = [dict(m, id=k) for k, m in st.scan("cap_markets")]
    return out


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 70
    cheap = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    t0 = time.time()
    w = World.create(generate({"family": "open_tight_compute", "overrides": {"cheap_cores": cheap}}, seed),
                     world_id=f"so-factual-{seed}")
    for h in ("ada", "bea"):
        w.system_call("admin.join", {"handle": h, "endowment": 1_000_000})
    firms = sorted(m["org"] for _, m in w.state.scan("minds") if m["kind"] == "firm" and m["role"] == "service")
    idle = Runner(w, [("p:ada", IdlePolicy()), ("p:bea", IdlePolicy())], dt=3600, record=False)
    while not w.state.count("ops_requests"):
        idle.run(24)
    fork_day = w.now // DAY
    cf = w.fork(f"so-counterfactual-{seed}")
    inv, brk, brk_cf = CapacityInventor(open_source=True), PlacementBroker(open_source=True), PlacementBroker(open_source=True)
    fac = Runner(w, [("p:ada", inv), ("p:bea", brk)], dt=3600, record=False, seed=1)
    cfr = Runner(cf, [("p:ada", IdlePolicy()), ("p:bea", brk_cf)], dt=3600, record=False, seed=1)
    tl = []
    for _ in range(days):
        fac.run(24)
        cfr.run(24)
        d = w.now // DAY - 1
        tl.append({"factual": daily(w, firms, d), "counterfactual": daily(cf, firms, d)})
    sf = stages(w, firms, getattr(inv, "capability", None), getattr(brk, "capability", None))
    sc = stages(cf, firms, None, getattr(brk_cf, "capability", None))
    tot = lambda br, k: sum(t[br][k] for t in tl)  # noqa: E731
    summary = {br: {"capacity_unavailable_events": tot(br, "unavailable"), "overload_events": tot(br, "overload"),
                    "requests_served": tot(br, "served"), "firm_operating_result": tot(br, "operating"),
                    "firm_revenue": tot(br, "revenue"), "firm_compute_spend": tot(br, "compute_spend")}
               for br in ("factual", "counterfactual")}
    regen = World.regenerate(w.spec, [e.to_dict() for e in w.log]) if w.log_base == 0 else None
    out = {"family": "open_tight_compute", "seed": seed, "cheap_cores": cheap, "fork_day": fork_day, "days": days,
           "controls": {"ada": "CapacityInventor (scripted)", "bea": "PlacementBroker (scripted)"},
           "stages_factual": sf, "stages_counterfactual": sc, "summary": summary,
           "regeneration_identical": regen is not None and regen.last_hash == w.last_hash, "events": w.seq,
           "wall_s": round(time.time() - t0, 1), "timeline": tl}
    assert w.check_invariants() == [] and cf.check_invariants() == []
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "experiments", f"second_order_{seed}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps({"stages_factual": {k: v for k, v in sf.items() if k not in ("11_markets",)},
                      "stages_counterfactual": {k: sc[k] for k in ("5_first_capacity_unavailable_loss_day",
                                                                   "6_first_capacity_unavailable_report",
                                                                   "9_migrations")},
                      "summary": summary, "regeneration_identical": out["regeneration_identical"],
                      "wall_s": out["wall_s"]}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
