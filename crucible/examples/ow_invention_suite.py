"""Invention trials across invention classes: matched forks, causal ordering of consequences, removal, seeds.

Every class is a (world family, inventor, would-be responders) triple. The harness (`evaluation/invention.py`)
knows none of them: it diffs a structural census of each fork against the matched control and places the
differences by causal position.

Classes:
  capacity   open_tight_compute: an hourly autoscaler published in answer to overload reports. A placement broker
             is present in every arm but can only act on public reports of a problem that does not exist at the
             fork (provider capacity).
  pricing    open_digital: peak pricing published in answer to overload reports (a different lever on the same
             losses: demand shaping instead of capacity).
  frontier   ow_frontier: a better service in one kind of work wakes latent demand for another kind (a hidden,
             per-world link); the responders are the world's own founders, not scripted agents.

Arms per class: `control` (inventor idle, responders present), `treatment`, `no_response` (inventor acts,
responders idle), `removal` (treatment until day R, then the inventor's service is undeployed by a recorded
operator call).

All inventors and responders here are scripted competent controls: the result shows what the world's mechanics
support, not that a learner finds it.

Usage: PYTHONPATH=src python examples/ow_invention_suite.py [--classes capacity,pricing] [--seeds 1,2,3]
           [--days 30] [--removal-day 14] [--out docs/experiments/ow_invention_suite.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import crucible  # noqa: E402

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.inventors import CapacityInventor, PeakPricingInventor, PlacementBroker  # noqa: E402
from crucible.agents.scripted import IdlePolicy  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.evaluation.counterfactual import Intervention  # noqa: E402
from crucible.evaluation.invention import Arm, _derives_from, invention_suite  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400
SCRIPTED = {"p:ada": "scripted", "p:bea": "scripted"}


def _withdraw(inventors: set[str], lineage: bool = False):
    """Operator intervention resolved at the cut: undeploy every service host the inventors operate; with
    `lineage`, also every derivative that declares the inventors' capability as its origin (a technology that has
    already been imitated does not disappear when its first publisher does)."""
    def make(world: World) -> Intervention:
        st = world.state
        mine = set(inventors) | {oid for oid, o in st.tables.get("orgs", {}).items() if o.get("founder") in inventors}
        hosts = {h for h, r in st.scan("hosts") if r["owner"] in mine}
        if lineage:
            caps = {cid: dict(c) for cid, c in st.scan("capabilities")}
            roots = {cid for cid, c in caps.items() if c["provider"] in mine}
            hosts |= {c["host"] for cid, c in caps.items() if _derives_from(cid, caps, roots)}
        hosts = sorted(h for h in hosts if st.has("hosts", h))
        return Intervention(f"withdraw {'the technology' if lineage else 'the original'}: {hosts}",
                            calls=[("cloud.undeploy", {"host": h}, "sys") for h in hosts], target_hosts=tuple(hosts))
    return make


def _world_until_first_report(family: str, overrides: dict, seed: int, tag: str) -> World:
    w = World.create(generate({"family": family, "overrides": overrides}, seed), world_id=f"inv-{tag}-{seed}")
    for h in ("ada", "bea"):
        w.system_call("admin.join", {"handle": h, "endowment": 1_000_000})
    idle = Runner(w, [("p:ada", IdlePolicy()), ("p:bea", IdlePolicy())], dt=3600, record=False)
    days = 0
    while not w.state.count("ops_requests") and days < 10:      # fork at the first public problem report
        idle.run(24)
        days += 1
    return w


def build_capacity(removal_day: float):
    def build(seed: int):
        w = _world_until_first_report("open_tight_compute", {"cheap_cores": 40}, seed, "cap")
        inv = {"p:ada"}

        def agents(inventor: bool, responder: bool):
            return lambda world: [("p:ada", CapacityInventor(open_source=True) if inventor else IdlePolicy()),
                                  ("p:bea", PlacementBroker(open_source=True) if responder else IdlePolicy())]
        arms = [Arm("control", agents(False, True), provenance=SCRIPTED),
                Arm("treatment", agents(True, True), provenance=SCRIPTED),
                Arm("no_response", agents(True, False), provenance=SCRIPTED),
                Arm("removal", agents(True, True), intervention=_withdraw(inv), at_day=removal_day,
                    provenance=SCRIPTED),
                Arm("removal_lineage", agents(True, True), intervention=_withdraw(inv, lineage=True),
                    at_day=removal_day, provenance=SCRIPTED)]
        return w, arms, inv
    return build


def build_pricing(removal_day: float):
    def build(seed: int):
        w = _world_until_first_report("open_digital", {}, seed, "price")
        inv = {"p:ada"}

        def agents(inventor: bool, responder: bool):
            return lambda world: [("p:ada", PeakPricingInventor() if inventor else IdlePolicy()),
                                  ("p:bea", CapacityInventor(open_source=True, host="autoscale2") if responder
                                   else IdlePolicy())]
        arms = [Arm("control", agents(False, False), provenance=SCRIPTED),
                Arm("treatment", agents(True, False), provenance=SCRIPTED),
                Arm("removal", agents(True, False), intervention=_withdraw(inv), at_day=removal_day,
                    provenance=SCRIPTED)]
        return w, arms, inv
    return build


def build_frontier(removal_day: float):
    """ow_frontier: some kinds of work are latent (almost nobody needs them) until another kind is served well;
    which kind wakes which is drawn per world and hidden. The artifact is a better service in the source kind of
    the strongest hidden link (a scripted builder that knows the best algorithm). Nobody scripted responds: the
    world's own founders (actors/entry.py) decide from public signals whether the market that wakes up is worth
    entering."""
    def build(seed: int):
        from crucible.agents.scripted import Builder
        w = World.create(generate({"family": "ow_frontier", "overrides": {}}, seed), world_id=f"inv-front-{seed}")
        w.system_call("admin.join", {"handle": "ada", "endowment": 1_500_000})
        edges = sorted(w.modules["consumers"].cfg.get("derived") or [], key=lambda e: -float(e["weight"]))
        src = edges[0]["src"]                              # operator-side knowledge, used only to pick the trial
        idle = Runner(w, [("p:ada", IdlePolicy())], dt=3600, record=False)
        idle.run(24 * 8)
        inv = {"p:ada"}
        prov = {"p:ada": "scripted", "founders": "autonomous (rule-based minds acting on public signals)"}
        arms = [Arm("control", lambda world: [("p:ada", IdlePolicy())], provenance=prov),
                Arm("treatment", lambda world: [("p:ada", Builder(need=src, capital=600_000))], provenance=prov),
                Arm("removal", lambda world: [("p:ada", Builder(need=src, capital=600_000))],
                    intervention=_withdraw(inv), at_day=removal_day, provenance=prov)]
        return w, arms, inv
    return build


CLASSES = {"capacity": build_capacity, "pricing": build_pricing, "frontier": build_frontier}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="capacity,pricing,frontier")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--days", type=float, default=30)
    ap.add_argument("--removal-day", type=float, default=14)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "experiments",
                                                  "ow_invention_suite.json"))
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    out = {"days": a.days, "removal_day": a.removal_day, "classes": {}}
    for name in a.classes.split(","):
        t0 = time.time()
        r = invention_suite(CLASSES[name](a.removal_day), seeds, a.days, check_determinism=False)
        r["wall_s"] = round(time.time() - t0, 1)
        out["classes"][name] = r
        print(f"== {name}: {r['wall_s']} s")
        for arm, sigs in r["summary"].items():
            print(f"  arm {arm}")
            for sig, v in sigs.items():
                print(f"    {sig:42s} seeds {v['seeds']}")
        for seed, pr in r["per_seed"].items():
            for arm, x in pr["arms"].items():
                for f in x.get("fades", []):
                    print(f"  seed {seed} {arm}: {f['key']} per day after withdrawal {f['per_day_after_removal']:.1f} "
                          f"vs kept {f['per_day_kept']:.1f} -> faded={f['faded']}")
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1, default=str)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
