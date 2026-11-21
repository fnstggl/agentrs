"""Demonstration 2 - endogenous technological change, measured against a counterfactual fork.

World: `open_scarce_energy` (small regional grids that do not expand). Fork at the first public problem report
about defects, breakdowns or energy; factual branch: a research agent (`ProcessResearcher`, a competent-agent
control) runs designed experiments on its own machines and sells process knowledge; counterfactual: it idles.
Nothing scripts what the firms do with the knowledge: their minds compare its verified evidence with their own
runs and decide to buy and apply it - or not.

Each stage of the chain is *detected* from authoritative state, not asserted:
  bottleneck -> improvement published -> adoption by independent firms -> energy/yield per unit of adopters
  -> grid curtailment and output -> prices and sales downstream -> competitors' responses -> new problem reports

Usage: PYTHONPATH=crucible/src python crucible/examples/demo_open_techchange.py [seed] [days]
Writes crucible/docs/experiments/open_techchange.json
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
from crucible.agents.inventors import ProcessResearcher  # noqa: E402
from crucible.agents.scripted import IdlePolicy  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.economy.operations import operating_result  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400
TAGS = {"defects", "machine_downtime", "energy_curtailment", "cost-energy"}


def runs_stats(w, org, t0, t1):
    st = w.state
    made = units = 0
    kwh = 0.0
    for _, r in st.scan("runs"):
        if r["owner"] == org and r["status"] == "done" and t0 <= r["t0"] < t1:
            made += r["made"]
            units += r["units"]
            kwh += r["kwh"]
    return {"units": units, "made": made, "yield": round(made / units, 4) if units else None,
            "kwh_per_good_unit": round(kwh / made, 4) if made else None}


def loss_n(w, org, kind, d0, d1):
    ops = w.modules["ops"]
    return sum((ops.day(w.state, org, d) or {"loss_n": {}})["loss_n"].get(kind, 0) for d in range(d0, d1))


def snapshot(w, firms, day):
    st = w.state
    ops = w.modules["ops"]
    return {"day": day,
            "operating": {f: operating_result(ops.day(st, f, day)) for f in firms},
            "retail": {k: {"price": r["price"], "sold": r.get("sold_y", 0)} for k, r in sorted(st.scan("retail"))},
            "offers": {k: o["price"] for k, o in sorted(st.scan("offers"))},
            "grid": {r: round((st.get("grid_days", f"{r}|{day}") or {}).get("used", 0), 1) for r in sorted(st.ids("regions"))}}


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 45
    t0 = time.time()
    w = World.create(generate({"family": "open_scarce_energy"}, seed), world_id="tech-factual")
    w.system_call("admin.join", {"handle": "rhea", "endowment": 6_000_000})
    firms = sorted(m["org"] for _, m in w.state.scan("minds") if m["kind"] == "firm" and m["role"] not in
                   ("provider", "utility"))
    idle = Runner(w, [("p:rhea", IdlePolicy())], dt=3600, record=False)
    while not any(TAGS & set(r["tags"]) for _, r in w.state.scan("ops_requests")):
        idle.run(24)
    fork_t = w.now
    bottleneck = sorted(((r["t"], r["by"], r["tags"]) for _, r in w.state.scan("ops_requests")
                         if TAGS & set(r["tags"])))[0]
    cf = w.fork("tech-counterfactual")
    agent = ProcessResearcher()
    fac = Runner(w, [("p:rhea", agent)], dt=3600, record=False, seed=1)
    cfr = Runner(cf, [("p:rhea", IdlePolicy())], dt=3600, record=False, seed=1)
    timeline = []
    for _ in range(days):
        fac.run(24)
        cfr.run(24)
        day = w.now // DAY - 1
        timeline.append({"factual": snapshot(w, firms, day), "counterfactual": snapshot(cf, firms, day)})
    st = w.state
    notes = {k: dict(n) for k, n in st.scan("process_notes")}
    buys = sorted(((r["t"], r["holder"], r["note"]) for _, r in st.scan("note_rights")))
    pub_t = min((n["t"] for n in notes.values()), default=None)
    adopters = sorted({h for _, h, _ in buys})
    end = w.now
    chain = {"bottleneck": {"t": bottleneck[0], "by": bottleneck[1], "tags": bottleneck[2]},
             "improvement_published": {"t": pub_t, "notes": {k: {"model": n["model"], "evidence": n["evidence"],
                                                                  "buyers": n["buyers"]} for k, n in notes.items()},
                                       "researcher_findings": {m: {k: v for k, v in f.items() if k != "grid"}
                                                               for m, f in agent.findings.items()}},
             "adoption": [{"t": t, "firm": h, "note": n} for t, h, n in buys]}
    eff = {}
    for f in adopters:
        t_adopt = min(t for t, h, _ in buys if h == f)
        eff[f] = {"before": runs_stats(w, f, fork_t, t_adopt), "after": runs_stats(w, f, t_adopt, end),
                  "counterfactual_after": runs_stats(cf, f, t_adopt, end)}
    chain["efficiency_of_adopters"] = eff
    d_pub = (pub_t or end) // DAY
    d_end = end // DAY
    chain["market"] = {
        "curtailment_after_publication": {"factual": sum(loss_n(w, f, "energy_curtailment", d_pub, d_end) for f in firms),
                                          "counterfactual": sum(loss_n(cf, f, "energy_curtailment", d_pub, d_end) for f in firms)},
        "units_made_after_publication": {"factual": sum(runs_stats(w, f, pub_t or end, end)["made"] for f in firms),
                                         "counterfactual": sum(runs_stats(cf, f, pub_t or end, end)["made"] for f in firms)},
        "retail_units_sold": {"factual": sum(r.get("sold_total", 0) for _, r in st.scan("retail")),
                              "counterfactual": sum(r.get("sold_total", 0) for _, r in cf.state.scan("retail"))},
        "operating_after_publication": {
            br: {f: sum(t[br]["operating"][f] for t in timeline if t[br]["day"] >= d_pub) for f in firms}
            for br in ("factual", "counterfactual")}}
    non = [f for f in firms if f not in adopters]
    resp = []
    for f in non:
        mm = st.tables["minds"][f]
        resp += [{"firm": f, **e} for e in mm["memory"] if e["t"] >= (pub_t or end) and e["kind"] in
                 ("substitute", "learned_process", "invest", "trial_start")]
    chain["responses_of_others"] = resp
    later_tags = sorted({tag for _, r in st.scan("ops_requests") if pub_t and r["t"] > pub_t for tag in r["tags"]}
                        - {tag for _, r in cf.state.scan("ops_requests") if r["t"] > fork_t for tag in r["tags"]})
    chain["new_problem_reports_only_in_factual"] = later_tags
    regen = World.regenerate(w.spec, [e.to_dict() for e in w.log]) if w.log_base == 0 else None
    out = {"family": "open_scarce_energy", "seed": seed, "days": days, "fork_t": fork_t, "chain": chain,
           "regeneration_identical": regen is not None and regen.last_hash == w.last_hash,
           "events": w.seq, "wall_s": round(time.time() - t0, 1), "timeline": timeline}
    assert w.check_invariants() == [] and cf.check_invariants() == []
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "experiments", "open_techchange.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps({k: v for k, v in chain.items() if k != "improvement_published"}, indent=1, default=str)[:4000])
    print("published:", {k: v["evidence"] for k, v in chain["improvement_published"]["notes"].items()})
    print("regeneration identical:", out["regeneration_identical"], "events", w.seq, "wall", out["wall_s"], "s")


if __name__ == "__main__":
    main()
