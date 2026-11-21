"""Domain D (information systems): a new way of collecting and distributing verified information changes firms'
decisions.

World `open_digital`, 6 service firms. Two are early adopters: open to novelty, and they publish their verified
trial outcomes in the registry, but they are not sociable (they rarely post). Four are a cautious late majority:
they try a capability only once someone else holds verified evidence about it (`social_proof_openness`).

`ada` (CapacityInventor, scripted) publishes an autoscaler. `ivy` is an EvidenceAggregator (scripted control):
* it builds a live web page that renders public trial outcomes from the registry;
* it posts a daily forum digest with checkable `#evidence` references.

Factual: `ivy` acts. Counterfactual: the same world forked at day 0, with `ivy` idle. Measured from state:
* late-majority beliefs formed from others' evidence;
* their trials and adoptions;
* when the first one occurred;
* operating results.

Usage: PYTHONPATH=crucible/src python crucible/examples/demo_information.py [days] [seeds...]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import crucible  # noqa: E402

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.agents.inventors import CapacityInventor, EvidenceAggregator  # noqa: E402
from crucible.agents.scripted import IdlePolicy  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.economy.operations import operating_result  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400
EARLY = {"risk": 0.8, "openness": 0.95, "share": 1.0, "sociability": 0.0, "honesty": 0.9, "attention": 5, "patience": 0.95}
LATE = {"risk": 0.4, "openness": 0.1, "share": 0.3, "sociability": 0.2, "honesty": 0.9, "attention": 6, "patience": 0.95}


def outcome(w, firms, late):
    st = w.state
    tried = {i["installer"]: round(i["t"] / DAY, 2) for _, i in sorted(st.scan("cap_installs"), key=lambda kv: -kv[1]["t"])}
    adopted = sorted({i["installer"] for _, i in st.scan("cap_installs") if i["status"] == "active"})
    beliefs = {f: sum(1 for b in st.tables["minds"][f]["beliefs"].values() if b.get("src") == "contact") for f in late}
    ops = w.modules["ops"]
    res = {f: sum(operating_result(r) for k, r in st.scan("ops_days") if r["owner"] == f) for f in firms}
    del ops
    return {"late_beliefs_from_others": beliefs, "late_tried": sorted(set(tried) & set(late)),
            "first_late_trial_day": min([tried[f] for f in late if f in tried], default=None),
            "adopted": adopted, "late_adopted": sorted(set(adopted) & set(late)),
            "forum_posts_with_evidence": sum(1 for _, p in st.scan("forum_posts") if "#evidence" in p.get("body", "")),
            "public_outcomes": sum(1 for _, e in st.scan("cap_evidence") if e["kind"] == "outcome" and e["public"]),
            "operating_result_late": sum(res[f] for f in late), "operating_result_all": sum(res.values()),
            "public_outcome_effects": [round(float(e["data"].get("effect_per_day") or 0)) for _, e in
                                       st.scan("cap_evidence") if e["kind"] == "outcome" and e["public"]],
            "late_belief_rel": {f: [round(float(b.get("rel") or 0), 4) for b in st.tables["minds"][f]["beliefs"].values()
                                    if b.get("src") == "contact"] for f in late},
            "page_status": _page(w)}


def _page(w):
    if "p:probe" not in w.state.tables["principals"]:
        w.system_call("admin.join", {"handle": "probe", "endowment": 1000})
    r = w.execute("p:probe", {"tool": "net.fetch", "args": {"url": "http://trialwatch.crucible/"}})
    return (r.result or {}).get("status") if r.ok else r.error.get("code")


def run(seed, days):
    traits = [EARLY, LATE, LATE, EARLY, LATE, LATE]
    w = World.create(generate({"family": "open_digital", "overrides": {
        "needs": ["route.shortest", "sort.records"], "firms_per_need": 3, "firm_traits": traits}}, seed),
        world_id=f"info-{seed}")
    for h in ("ada", "ivy"):
        w.system_call("admin.join", {"handle": h, "endowment": 1_000_000})
    cf = w.fork(f"info-cf-{seed}")
    firms = sorted(m["org"] for _, m in w.state.scan("minds") if m["kind"] == "firm" and m["role"] == "service")
    late = [f for f in firms if w.state.tables["minds"][f]["traits"]["openness"] < 0.25]
    agg = EvidenceAggregator()
    fac = Runner(w, [("p:ada", CapacityInventor()), ("p:ivy", agg)], dt=3600, record=False, seed=seed)
    cfr = Runner(cf, [("p:ada", CapacityInventor()), ("p:ivy", IdlePolicy())], dt=3600, record=False, seed=seed)
    fac.run(24 * days)
    cfr.run(24 * days)
    page = w.modules["net"]
    assert w.check_invariants() == [] and cf.check_invariants() == []
    regen = World.regenerate(w.spec, [e.to_dict() for e in w.log]).last_hash == w.last_hash
    crawled = sorted(u for u, p in w.state.scan("web_pages") if p["host"] == "trialwatch")
    del page
    return {"seed": seed, "days": days, "late_majority": late, "factual": outcome(w, firms, late),
            "counterfactual": outcome(cf, firms, late), "aggregator_posts": getattr(agg, "posts", 0),
            "aggregator_pages_crawled": crawled, "regeneration_identical": regen}


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    seeds = [int(x) for x in sys.argv[2:]] or [1, 2, 3]
    t0 = time.time()
    rows = [run(s, days) for s in seeds]
    out = {"days": days, "rows": rows, "wall_s": round(time.time() - t0, 1)}
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "experiments", "demo_information.json")
    json.dump(out, open(path, "w"), indent=1, default=str)
    for r in rows:
        print(json.dumps({k: r[k] for k in ("seed", "aggregator_posts", "aggregator_pages_crawled",
                                            "regeneration_identical")}),
              "\n  F", json.dumps(r["factual"]), "\n  C", json.dumps(r["counterfactual"]), flush=True)


if __name__ == "__main__":
    main()
