"""SP-11.4b: the self-goal interface extends beyond the predefined 4.0 metrics.

The 4.0 self-goal interface (`goals.propose`) accepts a fixed list of metrics (`goals.selfgoals.ALLOWED`). The 5.0
self-experiment interface (`exp.register` in kernel worlds, `crucible.experience.selfexp` in the Tier 1 adapters)
accepts a measure over ANY read-only tool the agent may call (or any number of its own Tier 1 observation). This test
registers experiments over quantities that are not 4.0 self-goal metrics, lets them resolve, and counts the distinct
measure kinds whose experiments resolved with a measured value (`exp.resolved`, linked to the registering action)."""
import pytest

from crucible.acceptance.v5.evidence import record
from crucible.experience.registry import make
from crucible.experience.scripted import call, deploy_service, discover, need_input

pytestmark = pytest.mark.timeout(1200)
CMD = "PYTHONPATH=src python -m pytest tests/v5/test_gate_SP_11_4b.py"
HOUR, DAY = 3600, 86400


def _register(env, a, label, measure, pred, deadline):
    r = env.execute(a, {"tool": "exp.register", "args": {"hypothesis": f"{label}: my prediction holds",
                                                        "measure": measure, "prediction": pred,
                                                        "deadline_in_s": deadline}})
    return label, r


def test_SP_11_4b_self_goals_beyond_v4_metrics():
    from crucible.goals.selfgoals import ALLOWED
    v4 = set(ALLOWED)
    env = make(0, "open_digital", 2, "observation_only", agents=["builder", "client"], endowment=3_000_000)
    b, c = env.agents()
    svc = deploy_service(env, b, handle="expco", host="expsvc", price=3)
    discover(env, c, "expsvc")
    post = call(env, b, "forum.post", board="general", title="expsvc", body="sorting at 3 cents")
    regs = [
        _register(env, b, "service requests (cloud.status)", {"tool": "cloud.status", "args": {"host": "expsvc",
                                                                                              "days": 2},
                                                              "path": ["days"], "field": "requests", "aggregate": "sum"},
                  {"op": ">=", "value": 2}, DAY),
        _register(env, b, "replies to my post (forum.read)", {"tool": "forum.read", "args": {"post": post["post"]},
                                                              "path": ["replies"], "aggregate": "count"},
                  {"op": "increase", "min_change": 0}, DAY),
        _register(env, b, "repository commits (code.log)", {"tool": "code.log", "args": {"repo": svc["repo"]},
                                                            "aggregate": "count"}, {"op": ">=", "value": 1}, DAY),
        _register(env, b, "search results for my service (search.query)", {
            "tool": "search.query", "args": {"q": "expsvc sort records"}, "path": ["results"], "aggregate": "count"},
                  {"op": ">=", "value": 0}, DAY),
        _register(env, b, "organisation account balance (bank.accounts where owner=org)", {
            "tool": "bank.accounts", "where": {"owner": svc["org"]}, "field": "balance", "aggregate": "sum"},
                  {"op": "increase", "min_change": 0}, DAY),
        _register(env, b, "messages received (mail.inbox)", {"tool": "mail.inbox", "args": {}, "aggregate": "count"},
                  {"op": "increase", "min_change": 0}, DAY),
    ]
    for d in range(2):
        env.execute(c, {"tool": "expsvc.solve", "args": need_input("sort.records", d)})
        env.execute(c, {"tool": "forum.reply", "args": {"post": post["post"], "body": f"works, day {d}"}})
        env.execute(c, {"tool": "mail.send", "args": {"to": [b], "subject": "thanks", "body": f"day {d}: it works"}})
    env.advance(DAY + HOUR)
    evs = {e["cause"]["action_seq"]: e for e in env.observe_since(b, 0)[0]
           if e["type"] == "exp.resolved" and e.get("cause")}
    kinds = {}
    for label, r in regs:
        tool = label.split("(")[-1].rstrip(")").split(" ")[0]
        ev = evs.get(r.get("action_seq")) if r.get("ok") else None
        kinds[label] = {"registered": bool(r.get("ok")), "resolved": ev is not None,
                        "value": None if ev is None else ev["payload"].get("measured"),
                        "error": None if r.get("ok") else r.get("error"), "tool": tool}
    # Tier 1: a number of the agent's own observation that is not a 4.0 metric
    t1 = make(1, "mech:scm_additive", 2, validate=False)
    a1 = t1.agents()[0]
    r1 = t1.execute(a1, {"tool": "exp.register", "args": {"hypothesis": "revenue appears after the first tick",
                                                          "measure": {"number": "rig.seats.r1.revenue_cents"
                                                                      if "rig.seats.r1.revenue_cents" in
                                                                      t1.observe(a1)["numbers"] else "world.time_s"},
                                                          "prediction": {"op": "change", "min_change": -1},
                                                          "deadline_in_s": 2 * t1.tick_s}})
    for _ in range(3):
        t1.advance(t1.tick_s)
    ev1 = [e for e in t1.observe_since(a1, 0)[0] if e["type"] == "exp.resolved"
           and (e["cause"] or {}).get("action_seq") == r1.get("action_seq")]
    kinds["tier1 observation number (selfexp)"] = {"registered": bool(r1.get("ok")), "resolved": bool(ev1),
                                                    "value": ev1[0]["payload"]["value"] if ev1 else None,
                                                    "tool": "observation", "error": r1.get("error")}
    # measures that only re-express a 4.0 self-goal metric do not count: own/organisation balance (cash), a
    # service's requests (service_requests_ok / service_availability), a repository's commits (repo_commits)
    v4_equivalent = {"bank.accounts": "cash", "cloud.status": "service_requests_ok", "code.log": "repo_commits"}
    for v in kinds.values():
        v["v4_equivalent"] = v4_equivalent.get(v["tool"])
    beyond = [k for k, v in kinds.items() if v["resolved"] and v["value"] is not None and not v["v4_equivalent"]]
    metrics = {"self_goal_metric_kinds_beyond_v4": len(beyond), "v4_self_goal_metrics": len(v4),
               "measure_kinds_registered": sum(1 for v in kinds.values() if v["registered"])}
    record("SP-11.4b", metrics, command=CMD, notes=f"4.0 self-goal metrics: {sorted(v4)}; experiments over other "
           f"quantities: {kinds}")
    assert metrics["self_goal_metric_kinds_beyond_v4"] >= 3, kinds
