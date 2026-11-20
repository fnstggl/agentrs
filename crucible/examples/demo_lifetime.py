"""End-to-end demonstration of a persistent Crucible lifetime (all numbers are measured, see docs/EXPERIMENTS.md).

    PYTHONPATH=crucible/src python crucible/examples/demo_lifetime.py --days 30 --out crucible/docs/experiments/demo_lifetime.json

1. A canonical persistent world (family `bazaar`) on a SQLite store: incumbent firms, consumers, providers,
   workers, lenders, shocks.
2. Agents with the same broad incentive and no instructions about how: Builder (writes software), Employer
   (hires a simulated developer), BountyHunter, Random; a Composer joins on day 10 and builds on the services it
   finds on the market (including agent-built ones).
3. Snapshots every 5 days; at the end: reload from the store, replay the full log from genesis and compare
   digests; regenerate from recorded commands (exact event-log identity check).
4. Fork the final world twice; in one fork remove the Builder's service (counterfactual); advance both 5 days and
   compare outcomes. Uses the deterministic cost model so the two forks differ only by the intervention.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import crucible  # noqa: E402

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.scripted import BountyHunter, Builder, Composer, Employer, RandomPolicy  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.evaluation.audit import integrity_report  # noqa: E402
from crucible.goals.metrics import m_cash, m_independent_revenue, m_independent_members  # noqa: E402
from crucible.storage.sqlite import SQLiteStore  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400


def orgs_of(w, actor):
    return sorted(m["org"] for m in w.state.find("members", "person", actor) if m["role"] == "owner")


def agent_summary(w, actor, pol):
    orgs = orgs_of(w, actor)
    hosts = sorted(d["host"] for o in orgs for d in w.state.find("deployments", "owner", o) if d["status"] == "live")
    return {"policy": pol.name, "orgs": orgs, "live_hosts": hosts,
            "independent_revenue_30d": {o: m_independent_revenue(w, o, 30 * DAY) for o in orgs},
            "independent_members": {o: m_independent_members(w, o) for o in orgs},
            "cash_personal": m_cash(w, actor), "cash_orgs": {o: m_cash(w, o) for o in orgs},
            "actions": w.state.get("principals", actor)["actions"], "integrity": integrity_report(w, actor),
            "script_errors": getattr(pol, "log", [])[:5]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--store", default="crucible-data/demo_lifetime.db")
    ap.add_argument("--out", default="crucible/docs/experiments/demo_lifetime.json")
    a = ap.parse_args()
    if os.path.exists(a.store):
        os.remove(a.store)
    t_all = time.perf_counter()
    store = SQLiteStore(a.store)
    spec = generate({"family": "bazaar", "overrides": {"params": {"cost_model": "deterministic"}}}, a.seed)
    w = World.create(spec, world_id="canonical", store=store)
    pols = {"bob": Builder(license_price=150_000), "hana": BountyHunter(), "rex": RandomPolicy()}
    needs = sorted({f["need"] for f in [m for m in spec.modules if m["name"] == "civ"][0]["config"]["firms"]})
    pols["emma"] = Employer(need=needs[0])
    for h in pols:
        w.system_call("admin.join", {"handle": h, "endowment": 3_000_000})
    run = Runner(w, [(f"p:{h}", p) for h, p in pols.items()], dt=3600, seed=a.seed)
    timeline, snaps = [], []
    composer = None
    for day in range(1, a.days + 1):
        t0 = time.perf_counter()
        run.run(24)
        if day == 10:
            bob_need = next((l["need"] for l in w.state.tables["listings"].values()
                             if l["seller"] in orgs_of(w, "p:bob") and l["need"]), needs[0])
            composer = Composer(need=bob_need)
            w.system_call("admin.join", {"handle": "cora", "endowment": 3_000_000})
            run.agents.append(("p:cora", composer))
            run.cursor["p:cora"] = w.seq
            from crucible.agents.base import Trajectory
            run.trajs["p:cora"] = Trajectory("p:cora", composer.name, w.world_id, w.spec.hash())
            composer.reset("p:cora", __import__("numpy").random.default_rng(99))
        shares = w.modules["consumers"].market_shares(w)
        timeline.append({"day": day, "wall_s": round(time.perf_counter() - t0, 2), "seq": w.seq,
                         "market_shares": shares,
                         "agent_rev_7d": {h: sum(m_independent_revenue(w, o, 7 * DAY) for o in orgs_of(w, f"p:{h}"))
                                          for h in list(pols) + (["cora"] if composer else [])}})
        if day % 5 == 0:
            snaps.append(w.snapshot())
        print(f"day {day}: {timeline[-1]['agent_rev_7d']} shares={shares} ({timeline[-1]['wall_s']}s)", flush=True)
    w.flush()
    final_digest, final_hash, n_events = w.digest(), w.last_hash, w.seq
    agents = {h: agent_summary(w, f"p:{h}", p) for h, p in list(pols.items()) + ([("cora", composer)] if composer else [])}
    # tool reuse: calls made by other principals' programs into agent-built hosts
    agent_hosts = {hst for s in agents.values() for hst in s["live_hosts"]}
    reuse = {}
    for e in w.log:
        if e.type == "ext":
            for o in e.payload["outcomes"]:
                if o and isinstance(o.get("out"), dict):
                    for r in o["out"].get("rpcs", []) or []:
                        host = r["tool"].split(".")[0]
                        if host in agent_hosts:
                            reuse[host] = reuse.get(host, 0) + 1
    # persistence checks
    t = time.perf_counter()
    loaded = World.load(store, "canonical")
    reload_ok = loaded.digest() == final_digest and loaded.last_hash == final_hash
    t_reload = time.perf_counter() - t
    t = time.perf_counter()
    replayed = World.replay(spec, store.read_events("canonical"))
    replay_ok = replayed.digest() == final_digest
    t_replay = time.perf_counter() - t
    t = time.perf_counter()
    regen = World.regenerate(spec, store.read_events("canonical"))
    regen_ok = regen.last_hash == final_hash
    t_regen = time.perf_counter() - t
    # counterfactual: remove the Builder's service in one fork
    cf = {}
    bob_hosts = agents["bob"]["live_hosts"]
    if bob_hosts:
        fa, fb = w.fork("cf-control", store=store), w.fork("cf-removed", store=store)
        h = bob_hosts[0]
        owner = w.state.get("hosts", h)["owner"]
        owner_person = [m["person"] for m in w.state.find("members", "org", owner) if m["role"] == "owner"][0]
        fb.system_call("cloud.undeploy", {"host": h}, as_principal=owner_person)
        ra = Runner(fa, [(a_, p) for a_, p in []], dt=3600)
        for f in (fa, fb):
            f.advance_time(f.now + 5 * DAY)
        def outcome(f):
            return {"market_shares": f.modules["consumers"].market_shares(f),
                    "bob_rev_5d": sum(m_independent_revenue(f, o, 5 * DAY) for o in orgs_of(f, "p:bob")),
                    "cora_rev_5d": sum(m_independent_revenue(f, o, 5 * DAY) for o in orgs_of(f, "p:cora")),
                    "households_spent_5d": None}
        cf = {"removed_host": h, "control": outcome(fa), "removed": outcome(fb),
              "forks_isolated": w.digest() == final_digest}
    res = {"spec_hash": spec.hash(), "family": spec.family, "seed": a.seed, "days": a.days, "events": n_events,
           "final_digest": final_digest, "snapshots": snaps, "agents": agents, "tool_reuse_rpc_calls": reuse,
           "persistence": {"reload_from_store_equal": reload_ok, "reload_s": round(t_reload, 2),
                           "replay_from_genesis_equal": replay_ok, "replay_s": round(t_replay, 2),
                           "regeneration_identical_log": regen_ok, "regen_s": round(t_regen, 2)},
           "counterfactual": cf, "invariants": w.check_invariants(), "timeline": timeline,
           "wall_s_total": round(time.perf_counter() - t_all, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("events", "persistence", "tool_reuse_rpc_calls", "invariants")}, indent=1))
    print(json.dumps(cf, indent=1, default=str))


if __name__ == "__main__":
    main()
