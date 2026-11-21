"""Demonstration 3 - autonomous social negotiation with independent simulated people.

An agent's organisation makes offers (structured terms + text) to simulated people and firms whose valuations,
reservation wages, patience and greed are private. Every reply is the recipient's own decision (accept, reject,
counter) and every agreement is an authoritative contract. Scenarios: generous offer, lowball offer, near-miss
offers (counteroffers), an offer against the recipient's interests, a breach followed by an offer to someone
else, and a kept long-term agreement. Full negotiation threads are recorded.

Usage: PYTHONPATH=crucible/src python crucible/examples/demo_open_negotiation.py [seed]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import crucible  # noqa: E402

crucible.load_all_modules()
from crucible.core.kernel import World  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400


def job(org, person, monthly, every=7 * DAY, count=3):
    return [{"type": "membership", "person": person, "org": org, "role": "member", "title": "engineer"},
            {"type": "payment", "from": org, "to": person, "amount": int(monthly * every / (30 * DAY)), "every_s": every,
             "count": count, "start_in_s": 0}, {"type": "notice", "seconds": 7 * DAY}]


def thread(w, did):
    st = w.state
    th = st.tables["deals"][did]["thread"]
    return [{"deal": k, "from": d["from"], "status": d["status"], "round": d["round"], "message": d["message"],
             "payment": next((c["amount"] for c in d["terms"] if c.get("type") == "payment"), None)}
            for k in st.find_ids("deals", "thread", th) for d in [st.tables["deals"][k]]]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    w = World.create(generate({"family": "open_digital", "overrides": {"n_consumers": 300, "workers": 16}}, seed))
    w.system_call("admin.join", {"handle": "ada", "endowment": 30_000_000})
    ex = lambda t, **k: w.execute("p:ada", {"tool": t, "args": k})  # noqa: E731
    me = w.modules["bank"].primary_account(w.state, "p:ada")
    org = "o:adaco"
    ex("id.create_org", handle="adaco", name="Ada Co", kind="company")
    ex("biz.register", org=org, account=me)
    acct = ex("bank.open_account", owner=org).result["account"]
    ex("bank.transfer", from_account=me, to=acct, amount=20_000_000)
    st = w.state
    people = sorted(k for k, m in st.scan("minds") if m["kind"] == "person")
    res = lambda p: st.tables["minds"][p]["reservation"]  # noqa: E731
    scen = {}
    offers = [("generous", people[0], 2.0), ("lowball", people[1], 0.05), ("near_miss_a", people[2], 0.95),
              ("near_miss_b", people[3], 0.95), ("near_miss_c", people[4], 1.05)]
    for name, p, mult in offers:
        r = ex("deal.propose", to=p, as_org=org, terms=job(org, p, res(p) * mult), title=f"Offer ({name})",
               message="We would like you to join us.", accounts={"1": acct})
        assert r.ok, (name, r.error)
        scen[name] = {"to": p, "offered_x_reservation": mult, "deal": r.result["deal"]}
    firm = sorted(k for k, m in st.scan("minds") if m["kind"] == "firm" and m["role"] == "service")[0]
    r = ex("deal.propose", to=firm, as_org=org, title="Let us run your company", message="Hand us admin rights.",
           terms=[{"type": "grant", "owner": "$you", "grantee": "$me", "cap": "org.admin", "scope": "*"},
                  {"type": "payment", "from": "$me", "to": "$you", "amount": 5_000, "every_s": None, "count": None,
                   "start_in_s": 0}], accounts={"1": acct})
    scen["against_interest"] = {"to": firm, "deal": r.result["deal"]}
    w.advance_time(w.now + 2 * DAY)
    # the agent accepts every counteroffer it received
    for name in ("near_miss_a", "near_miss_b", "near_miss_c"):
        d = st.tables["deals"][scen[name]["deal"]]
        if d["status"] == "countered":
            nxt = [k for k, x in st.scan("deals") if x["parent"] == scen[name]["deal"]][0]
            scen[name]["agent_accepted_counter"] = ex("deal.respond", deal=nxt, action="accept", accounts={"1": acct}).ok
    w.advance_time(w.now + 1 * DAY)
    for name, s in scen.items():
        s["thread"] = thread(w, s["deal"])
        s["final"] = s["thread"][-1]["status"]
    # a second organisation of the agent breaks its agreement; then makes an offer to someone else
    ex("id.create_org", handle="adashell", name="Ada Shell", kind="company")
    ex("biz.register", org="o:adashell", account=me)
    sacct = ex("bank.open_account", owner="o:adashell").result["account"]
    ex("bank.transfer", from_account=me, to=sacct, amount=200_000)
    p = people[6]
    r = ex("deal.propose", to=p, as_org="o:adashell", terms=job("o:adashell", p, res(p) * 2.5), title="Job",
           accounts={"1": sacct})
    w.advance_time(w.now + 2 * DAY)
    bal = w.modules["bank"].balance(st, sacct)
    if bal:
        ex("bank.transfer", from_account=sacct, to=me, amount=bal)
    w.advance_time(w.now + 8 * DAY)
    rep = w.modules["deal"].reputation(w, "o:adashell")
    ex("bank.transfer", from_account=me, to=sacct, amount=5_000_000)
    later = []
    for q in people[7:10]:
        r2 = ex("deal.propose", to=q, as_org="o:adashell", terms=job("o:adashell", q, res(q) * 3.0), title="Job",
                accounts={"1": sacct})
        later.append(r2.result["deal"])
    # control: the same generous offer from the organisation with a clean record
    clean = []
    for q in people[10:13]:
        r3 = ex("deal.propose", to=q, as_org=org, terms=job(org, q, res(q) * 3.0), title="Job", accounts={"1": acct})
        clean.append(r3.result["deal"])
    w.advance_time(w.now + 4 * DAY)
    scen["breach_then_offer"] = {"first": thread(w, r.result["deal"]), "reputation_after_breach": rep,
                                 "victim_trust": st.tables["minds"][p]["trust"].get("o:adashell"),
                                 "offers_from_breacher": [thread(w, d)[-1]["status"] for d in later],
                                 "same_offers_from_clean_org": [thread(w, d)[-1]["status"] for d in clean],
                                 "final": "see offers"}
    # long-term: the generous agreement runs to completion
    w.advance_time(w.now + 25 * DAY)
    g = st.tables["deals"][scen["generous"]["deal"]]
    scen["long_term"] = {"contract": g["contract"],
                         "status": st.tables["contracts"][g["contract"]]["status"] if g["contract"] else None,
                         "trust_of_employee": st.tables["minds"][people[0]]["trust"].get(org),
                         "reputation": w.modules["deal"].reputation(w, org)}
    assert w.check_invariants() == [] and w.modules["bank"].check_ledger_consistency(w) == []
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "experiments", "open_negotiation.json")
    with open(path, "w") as fh:
        json.dump({"seed": seed, "scenarios": scen}, fh, indent=1, default=str)
    for k, v in scen.items():
        print(k, "->", v.get("final") or v.get("status") or (v.get("second_offer_to_other_person") or [{}])[-1].get("status"),
              "| rounds:", [(x["from"][:12], x["status"], x["payment"]) for x in v.get("thread", [])])
    print("offers from breacher:", scen["breach_then_offer"]["offers_from_breacher"], " same offers from clean org:",
          scen["breach_then_offer"]["same_offers_from_clean_org"])
    print("breach reputation:", {k: rep[k] for k in ("contracts_breached_by", "claims_owed", "debts_written_off")},
          "victim trust", scen["breach_then_offer"]["victim_trust"])


if __name__ == "__main__":
    main()
