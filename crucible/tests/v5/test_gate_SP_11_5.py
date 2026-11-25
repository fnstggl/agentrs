"""SP-11.5: long-horizon experience - actions with different durations, delayed events, asynchronous workflows,
long-running programs, concurrent projects, persistent commitments, consequences long after the action, and causal
provenance the learner can investigate (no privileged causal answers).

A builder operating in a 4.0 economy (tiny + workflows + experiments) over 16 simulated days through the
`ExperienceEnv` interface: it deploys a service a client uses for two weeks, signs a contract with three weekly
payments, schedules a workflow eight days ahead and runs two experiments at once. Every feature is measured from what
the builder itself observes."""
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.experience.registry import make
from crucible.experience.scripted import Recording, call, deploy_service, discover, need_input

pytestmark = pytest.mark.timeout(900)
CMD = "PYTHONPATH=src python -m pytest tests/v5/test_gate_SP_11_5.py"
DAY = 86400
BAL = {"tool": "bank.accounts", "where": {"owner": "$self"}, "field": "balance", "aggregate": "sum"}


def test_SP_11_5_long_horizon_experience():
    env = make(0, "tiny", 3, "observation_only", agents=["builder", "client", "payer"], endowment=3_000_000,
               extra_modules=["wf"], overrides={"n_consumers": 200})
    b, c, p = env.agents()
    rec = Recording(env)                           # the builder's own record of what it did (with action_seq)

    def bx(tool, **args):
        return rec.execute(b, {"tool": tool, "args": args})
    deploy_service(rec, b, handle="longco", host="longsvc", price=3)
    discover(env, c, "longsvc")
    # a persistent commitment: three weekly payments from the payer to the builder
    clause = {"type": "payment", "from": p, "to": b, "amount": 20_000, "every_s": 7 * DAY, "count": 3,
              "start_in_s": 3600}
    cid = call(env, p, "contracts.propose", parties=[p, b], title="weekly retainer", clauses=[clause])["contract"]
    sign = bx("contracts.sign", contract=cid)
    # an asynchronous workflow, eight days ahead
    flow = bx("wf.define", name="weekly books", steps=[{"id": "books", "tool": "bank.accounts", "args": {}}])
    later = bx("wf.run_later", flow=flow["result"]["flow"], in_s=8 * DAY)
    # two concurrent projects: experiments with different deadlines
    e1 = bx("exp.register", hypothesis="the retainer's first payment raises my balance within 3 days",
            measure=BAL, prediction={"op": "increase", "min_change": 10_000}, deadline_in_s=3 * DAY)
    e2 = bx("exp.register", hypothesis="after 10 days the service has been called at least twice",
            measure={"tool": "cloud.status", "args": {"host": "longsvc", "days": 14}, "path": ["days"],
                     "field": "requests", "aggregate": "sum"},
            prediction={"op": ">=", "value": 2}, deadline_in_s=10 * DAY)
    bx("forum.post", board="general", title="longsvc is live", body="Sorting as a service, 3 cents a call.")
    results = [r for a, _, r in rec.log if a == b]
    assert all(r["ok"] for r in results), [r for r in results if not r["ok"]]
    max_pending = 0
    caused = []
    service_days = set()
    for day in range(16):
        if day in (1, 7, 14):
            r = env.execute(c, {"tool": "longsvc.solve", "args": need_input("sort.records", day)})
            assert r["ok"] and r["result"]["status"] == 200, r
            service_days.add(day)
        env.advance(DAY)
        o = env.observe(b)
        max_pending = max(max_pending, sum(1 for m in o["missing"] if m.startswith("exp:")))
        caused += [e for e in o["events"] if e["cause"]]
    status = bx("cloud.status", host="longsvc", days=16)["result"]
    days_with_requests = sum(1 for d in status["days"] if d.get("requests", 0) > 0)
    contract = bx("contracts.view", contract=cid)
    results = [r for a, _, r in rec.log if a == b]
    paid_events = [e for e in caused if e["type"] == "contracts.paid"]
    wf_done = [e for e in caused if e["type"] == "wf.job_done"]
    exp_done = [e for e in caused if e["type"] == "exp.resolved"]
    durations = sorted({int(r["cost_s"]) for r in results})
    mine = {r["action_seq"]: r["tool"] for r in results if "action_seq" in r}
    provenance_ok = bool(caused) and all(e["cause"]["action_seq"] in mine and
                                         mine[e["cause"]["action_seq"]] == e["cause"]["tool"]
                                         for e in caused if e["cause"]["action_seq"] in mine)
    linked = sum(1 for e in caused if e["cause"]["action_seq"] in mine)
    features = {
        "action_durations": len(durations) >= 3,
        "delayed_events": any(e["cause"]["via"] == "scheduled" and e["cause"]["delay_s"] > 0 for e in caused),
        "asynchronous_workflows": bool(wf_done) and wf_done[0]["payload"]["ok"] and
        wf_done[0]["cause"]["action_seq"] == later["action_seq"],
        "long_running_programs": days_with_requests >= 2 and status["status"] == "live" and len(service_days) == 3,
        "concurrent_projects": max_pending >= 2 and len(exp_done) == 2,
        "persistent_commitments": len({e["t"] // DAY for e in paid_events}) >= 3 or
        (contract["ok"] and len(paid_events) >= 3),
        "consequences_long_after": any(e["cause"]["delay_s"] >= 7 * DAY for e in caused),
        "causal_provenance": provenance_ok and linked == len(caused),
    }
    metrics = {"long_horizon_features": sum(1 for v in features.values() if v), "caused_events": len(caused),
               "distinct_durations": len(durations), "max_concurrent_experiments": max_pending,
               "contract_payments_observed": len(paid_events), "service_days_with_requests": days_with_requests,
               "max_delay_days": round(max((e["cause"]["delay_s"] for e in caused), default=0) / DAY, 3)}
    record("SP-11.5", metrics, command=CMD, notes=f"features: {features}; durations (cost_s): {durations}; "
           f"caused event types: {sorted({e['type'] for e in caused})}")
    assert all(features.values()), features
    assert metrics["long_horizon_features"] >= 7
