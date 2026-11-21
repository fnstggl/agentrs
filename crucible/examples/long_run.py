"""Part 24: long-running worlds over several seeds and families, with weekly liveness checks and invariants.

The worlds contain, all at once:
* persistent people and organisations, money and employment;
* a synthetic web (generated apps and adversarial sites), software infrastructure with scarce compute;
* contracts, negotiation, production (in `open_mixed`), forums and mail;
* firm minds, NPC innovators and entrepreneurs.

Two scripted controls (an inventor and a placement broker) act as ordinary agents. Everything else is the world's
own dynamics.

Every simulated week the script records structural measures: technology, adoption and abandonment, markets, firm
entry and exit, output, constraints, diffusion, relationships, reliability, dependencies, emergent problems,
interaction and strategy. It also records liveness:
* the daily processes fired 7 times;
* demand was served;
* events kept arriving;
* the invariants hold.

A week with no *structural* change is flagged. Event counts do not count as change. If two consecutive weeks are
flagged, the script records that evolution stopped and the probable cause (the processes that went silent).

Usage: PYTHONPATH=crucible/src python crucible/examples/long_run.py [days] [family:seed ...]
"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import crucible  # noqa: E402

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.inventors import CapacityInventor, PlacementBroker  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.evaluation.open_internet import measure  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400
WEEK = 7 * DAY


def make(family, seed):
    ov = {"module_config": {"civ": {"innovators": 2}, "webgen": {"sites": 3},
                            "adversary": {"injection_pages": 2, "fraud_capability": True, "spam_posts": 2}}}
    if family == "open_tight_compute":
        ov["cheap_cores"] = 40
    if family == "open_mixed":
        ov.update({"n_consumers": 2500, "firms_per_need": 3, "workers": 24})
    w = World.create(generate({"family": family, "overrides": ov}, seed), world_id=f"long-{family}-{seed}")
    ag = []
    for h, pol in (("ada", CapacityInventor(open_source=True)), ("bea", PlacementBroker(open_source=True))):
        w.system_call("admin.join", {"handle": h, "endowment": 2_000_000})
        ag.append((f"p:{h}", pol))
    return w, ag


def snapshot(w):
    st = w.state
    s = {"capabilities": st.count("capabilities") if "capabilities" in st.tables else 0,
         "installs_by_status": dict(Counter(i["status"] for _, i in st.scan("cap_installs"))) if "cap_installs" in st.tables else {},
         "markets": st.count("cap_markets") if "cap_markets" in st.tables else 0,
         "orgs_active": sum(1 for _, o in st.scan("orgs") if o["status"] == "active"),
         "orgs_inactive": sum(1 for _, o in st.scan("orgs") if o["status"] != "active"),
         "designs": st.count("designs") if "designs" in st.tables else 0,
         "loss_kinds": sorted({k for _, r in st.scan("ops_days") for k, n in r["loss_n"].items() if n}) if "ops_days" in st.tables else [],
         "request_tags": sorted({t for _, r in st.scan("ops_requests") for t in r["tags"]}) if "ops_requests" in st.tables else [],
         "deps": st.count("cap_deps") if "cap_deps" in st.tables else 0,
         "deals": st.count("deals") if "deals" in st.tables else 0,
         "contracts": st.count("contracts") if "contracts" in st.tables else 0,
         "machines": st.count("machines") if "machines" in st.tables else 0,
         "provider_prices": {k: p["price"] for k, p in sorted(st.scan("providers"))} if "providers" in st.tables else {},
         "provider_cores": {k: p["cores"] for k, p in sorted(st.scan("providers"))} if "providers" in st.tables else {},
         "web_pages": st.count("web_pages") if "web_pages" in st.tables else 0,
         "beliefs_from_others": sum(1 for _, m in st.scan("minds") for b in m["beliefs"].values() if b.get("src") == "contact"),
         "contacts": sum(len(m["contacts"]) for _, m in st.scan("minds"))}
    return s


STRUCTURAL = ("capabilities", "installs_by_status", "markets", "orgs_active", "orgs_inactive", "designs",
              "loss_kinds", "request_tags", "deps", "deals", "contracts", "machines", "provider_cores", "web_pages")


def week_activity(w, t0, t1):
    types = Counter(e.type for e in w.log if t0 <= e.t < t1)
    served = sum(r["ok"] for k, r in w.state.scan("svc_days") if t0 // DAY <= int(k.rsplit("|", 1)[1]) < t1 // DAY)
    procs = Counter(e.payload.get("proc") for e in w.log if t0 <= e.t < t1 and e.type == "proc")
    failed = Counter(e.payload.get("proc") for e in w.log if t0 <= e.t < t1 and e.type == "proc.failed")
    return {"events": sum(types.values()), "event_types": len(types), "requests_served": served,
            "process_failures": dict(failed),
            "ops_daily_runs": procs.get("ops.daily", 0), "society_people_runs": procs.get("society.people", 0),
            "society_firms_runs": procs.get("society.firms", 0), "top_types": types.most_common(6)}


def run(family, seed, days):
    t_wall = time.time()
    w, ag = make(family, seed)
    r = Runner(w, ag, dt=3600, record=False, seed=seed)
    weeks = []
    prev = snapshot(w)
    flat = 0
    stopped = None
    for k in range(days // 7):
        t0 = w.now
        r.run(24 * 7)
        snap = snapshot(w)
        act = week_activity(w, t0, w.now)
        changed = [key for key in STRUCTURAL if snap[key] != prev[key]]
        inv = w.check_invariants()
        live = {"daily_processes_ok": act["ops_daily_runs"] >= 6 and act["society_people_runs"] >= 6,   # 6 in week 1
                "no_process_failures": not act["process_failures"],
                "demand_served": act["requests_served"] > 0, "events": act["events"] > 0, "invariants_ok": not inv}
        flat = 0 if changed else flat + 1
        if flat >= 2 and stopped is None:
            stopped = {"week": k + 1, "silent_processes": [p for p in ("ops_daily_runs", "society_people_runs",
                                                                       "society_firms_runs") if act[p] == 0],
                       "event_types": act["event_types"]}
        weeks.append({"week": k + 1, "day": w.now // DAY, "changed": changed, "liveness": live, "activity": act,
                      "snapshot": snap, "invariant_errors": inv[:3]})
        prev = snap
        print(family, seed, "week", k + 1, "changed", changed, "live", all(live.values()),
              "wall", round(time.time() - t_wall), flush=True)
    m = measure(w, 0)
    return {"family": family, "seed": seed, "days": days, "weeks": weeks, "evolution_stopped": stopped,
            "measures": m, "events": w.seq, "wall_s": round(time.time() - t_wall, 1),
            "replay_identical": World.replay(w.spec, [e.to_dict() for e in w.log]).digest() == w.digest()}


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    specs = sys.argv[2:] or ["open_tight_compute:1", "open_tight_compute:2", "open_mixed:1", "open_mixed:2"]
    out = []
    for s in specs:
        fam, seed = s.split(":")
        out.append(run(fam, int(seed), days))
        path = os.path.join(os.path.dirname(__file__), "..", "docs", "experiments", "long_run.json")
        json.dump({"days": days, "runs": out}, open(path, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
