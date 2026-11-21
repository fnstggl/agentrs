"""Domain C (social organisation): a governance change, decided by members' votes, changes a group's performance.

World `open_industry`. Initial condition (set once, by the system, and documented as such): one producer, the
forge with the most operators, is a worker cooperative. Twelve simulated people are members. The constitution
splits profits equally among all members, and any amendment needs a members' vote (threshold 0.5, quorum 0.4).

The world's incentive model (`gov.effort`, applied to the operators' effective hours in production) makes equal
sharing among many members dilute effort. The effort formula is a modelling assumption, not a finding.

Fork at day 2:
Two constitutions are compared: amendments need 50% of *all* members, or 30% (abstentions count against).

* **factual:** the cooperative's founder (a scripted organiser move, the only scripted act) proposes amending
  the distribution to a labour basis. Members vote from their trust in the proposer. If the amendment passes,
  the constitution changes; otherwise it does not.
* **counterfactual:** no proposal.

Measured from state over 30 days: the proposal outcome and the votes, effective operators, units made,
operating result, and the distributions paid to members.

Usage: PYTHONPATH=crucible/src python crucible/examples/demo_social_org.py [days] [seeds...]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import crucible  # noqa: E402

crucible.load_all_modules()
from crucible.agents.base import Runner  # noqa: E402
from crucible.core.kernel import World  # noqa: E402
from crucible.economy.operations import operating_result  # noqa: E402
from crucible.worldgen import generate  # noqa: E402

DAY = 86400
def rules(threshold):
    """Amendments need `threshold` of *all* members voting yes (abstentions count against), quorum 0.4."""
    return [{"match": {"tool": "gov.amend"}, "require": {"type": "vote", "electorate": "members",
                                                         "threshold": threshold, "quorum": 0.4, "window_s": 3 * DAY}}]


def measure(w, org, d0):
    st = w.state
    ind = w.modules["industry"]
    runs = [r for _, r in st.scan("runs") if r["owner"] == org and r["t0"] >= d0 * DAY] if "runs" in st.tables else []
    days = [r for k, r in st.scan("ops_days") if r["owner"] == org and r["day"] >= d0]
    paid = [e.payload for e in w.log if e.type == "gov.distributed" and e.payload["org"] == org]
    return {"effective_operators": round(ind.operators(st, org), 3),
            "constitution": st.tables["constitutions"].get(org, {}).get("distribution"),
            "runs": len(runs), "units_made": sum((r.get("made") or 0) for r in runs),
            "operating_result": sum(operating_result(r) for r in days),
            "distributed_to_members": sum(p.get("paid", 0) for p in paid)}


def run(seed, days, threshold=0.5):
    w = World.create(generate({"family": "open_industry"}, seed), world_id=f"coop-{seed}")
    st = w.state
    own = st.meta.get("own_operators") or {}
    org = sorted((-n, o) for o, n in own.items() if "forge" in o)[0][1]
    founder = next(m["person"] for m in st.find("members", "org", org) if m["role"] == "owner")
    people = sorted(k for k, m in st.scan("minds") if m["kind"] == "person")[:12]
    idm = w.modules["identity"]
    w._run_system("coop-setup", lambda ctx: [idm.add_member(ctx, org, p, "member") for p in people])
    RULES = rules(threshold)
    r = w.execute(founder, {"tool": "gov.adopt", "args": {"org": org, "rules": RULES,
                                                          "distribution": {"basis": "equal", "share": 0.5}}})
    assert r.ok, r.error
    Runner(w, [], dt=3600, record=False).run(48)
    cf = w.fork(f"coop-cf-{seed}")
    d0 = w.now // DAY
    before = measure(w, org, d0)
    p = w.execute(founder, {"tool": "gov.propose", "args": {
        "org": org, "tool": "gov.amend", "args": {"org": org, "rules": RULES,
                                                  "distribution": {"basis": "labor", "share": 0.5}},
        "reason": "Share profits by recorded working hours, so that those who work more keep more."}})
    assert p.ok, p.error
    pid = p.result.get("proposal")
    Runner(w, [], dt=3600, record=False).run(24 * days)
    Runner(cf, [], dt=3600, record=False).run(24 * days)
    prop = st.tables["proposals"].get(pid, {})
    votes = [e.payload for e in w.log if e.type == "gov.vote" and e.payload.get("proposal") == pid]
    assert w.check_invariants() == [] and cf.check_invariants() == []
    return {"seed": seed, "threshold": threshold, "cooperative": org, "members": len(st.find_ids("members", "org", org)), "fork_day": d0,
            "proposal": {"id": pid, "status": prop.get("status"),
                         "votes_for": sum(1 for v in votes if v.get("approve")),
                         "votes_against": sum(1 for v in votes if not v.get("approve"))},
            "before": before, "factual": measure(w, org, d0), "counterfactual": measure(cf, org, d0),
            # the initial condition is a system intervention (not a recorded command), so the check is event replay
            "replay_identical": World.replay(w.spec, [e.to_dict() for e in w.log]).digest() == w.digest()}


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seeds = [int(x) for x in sys.argv[2:]] or [1, 2, 3]
    t0 = time.time()
    rows = [run(s, days, thr) for thr in (0.5, 0.3) for s in seeds]
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "experiments", "demo_social_org.json")
    json.dump({"days": days, "rows": rows, "wall_s": round(time.time() - t0, 1)}, open(path, "w"), indent=1, default=str)
    for r in rows:
        print(json.dumps(r, default=str), flush=True)


if __name__ == "__main__":
    main()
