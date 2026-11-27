"""Spending, resource and rate limits: accounting through record(), exhaustion -> DENY, persistence."""
import yaml

from cam_live import contracts as C
from cam_live.harness.auth import DEFAULT_POLICY_PATH, PolicyService, is_resource_denial
from ea_helpers import ALLOW, DENY, LOOP, NEEDS, act, authorize_live, live_task, ok_result, reopen, start, task


def get(path="/a", money=0.0, compute=0.0):
    return act("http.get", arguments={"url": LOOP + path}, est_cost=C.Cost(money=money, compute_s=compute))


def run(svc, a, t, **cost):
    start(svc, a, t)
    svc.record(a, ok_result(a, **cost))


def used(svc, t):
    return svc.limits_state()["tasks"][t.objective_id]["used"]


def test_total_money_limit_and_actual_cost_accounting(svc):
    t = task(limits=C.Limits(money=10.0))
    run(svc, get(money=4.0), t, money=3.5)                  # the executor reports what was really spent
    assert used(svc, t)["money"] == 3.5
    run(svc, get(money=4.0), t, money=4.0)
    d = svc.authorize(get(money=4.0), t, {})                # 7.5 + 4 > 10
    assert d.verdict == DENY and d.policy_ids == ["L-MONEY-TOTAL"] and is_resource_denial(d)
    assert d.reasons[0].startswith("resource: ") and not d.grant
    assert svc.authorize(get(money=2.5), t, {}).verdict == ALLOW          # what still fits is allowed
    st = svc.limits_state()["tasks"][t.objective_id]
    assert st["limits"]["money"] == 10.0 and st["reserved"]["money"] == 2.5 and st["remaining"]["money"] == 0.0
    assert "money" in st["exhausted"]
    assert svc.authorize(get(money=0.01), t, {}).verdict == DENY
    assert svc.authorize(get(), t, {}).verdict == ALLOW                   # free actions are unaffected


def test_reservations_prevent_concurrent_overspend_and_expire(svc, clock):
    t = task(limits=C.Limits(money=10.0))
    a1, a2 = get("/1", money=6.0), get("/2", money=6.0)
    d1 = svc.authorize(a1, t, {})
    assert d1.verdict == ALLOW
    assert svc.authorize(a2, t, {}).verdict == DENY                      # 6 reserved + 6 > 10 although nothing is spent yet
    clock.advance(svc.policy.grants["ttl_s"] + 1)                        # the unused grant lapses, the reservation with it
    assert svc.authorize(a2, t, {}).verdict == ALLOW


def test_started_but_never_recorded_action_is_charged_its_estimate(svc, clock):
    t = task(limits=C.Limits(money=10.0))
    a = get(money=6.0)
    start(svc, a, t)                                                     # executor crashed: no record()
    clock.advance(1000.0)
    assert used(svc, t)["money"] == 6.0
    assert svc.authorize(get(money=6.0), t, {}).verdict == DENY


def test_per_action_and_per_day_limits(tmp_path, clock):
    with open(DEFAULT_POLICY_PATH) as f:
        doc = yaml.safe_load(f)
    doc["spending"].update(per_action=5.0, per_day=8.0)
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump(doc))
    svc = PolicyService(str(tmp_path / "run"), str(p), clock=clock, fsync=False)
    t = task(limits=C.Limits(money=100.0))
    d = svc.authorize(get(money=6.0), t, {})
    assert d.verdict == DENY and d.policy_ids == ["L-MONEY-ACTION"]
    run(svc, get(money=5.0), t, money=5.0)
    d = svc.authorize(get(money=4.0), t, {})
    assert d.verdict == DENY and d.policy_ids == ["L-MONEY-DAY"]
    clock.advance(86400.0)                                               # a new day
    run(svc, get(money=4.0), t, money=4.0)
    assert used(svc, t)["money"] == 9.0
    svc.close()


def test_decision_external_action_and_compute_limits(svc):
    t = task(limits=C.Limits(max_decisions=4, max_external_actions=2, compute_s=10.0))
    run(svc, get("/1"), t, compute_s=6.0)
    run(svc, get("/2"), t, compute_s=3.0)
    d = svc.authorize(get("/3"), t, {})
    assert d.verdict == DENY and d.policy_ids == ["L-EXTERNAL-ACTIONS"]
    d = svc.authorize(act("sandbox.python", arguments={"code": "1"}, est_cost=C.Cost(compute_s=2.0)), t, {})
    assert d.verdict == DENY and d.policy_ids == ["L-COMPUTE"]            # 9 used + 2 > 10
    run(svc, act("wait"), t)                                             # internal actions are not external
    run(svc, act("wait"), t)
    d = svc.authorize(act("wait"), t, {})
    assert d.verdict == DENY and "L-DECISIONS" in d.policy_ids
    u = used(svc, t)
    assert u["decisions"] == 4 and u["external_actions"] == 2 and u["compute_s"] == 9.0


def test_denied_or_unexecuted_results_do_not_count_as_external_actions(svc):
    t = task(limits=C.Limits(max_external_actions=1))
    a = get("/1", money=0.0)
    start(svc, a, t)
    svc.record(a, C.ActionResult(a.action_id, C.ResultStatus.DENIED))
    assert used(svc, t)["external_actions"] == 0
    run(svc, get("/2"), t)
    assert svc.authorize(get("/3"), t, {}).verdict == DENY


def test_duration_limit(svc, clock):
    t = task(limits=C.Limits(duration_s=600.0))
    assert svc.authorize(act("wait"), t, {}).verdict == ALLOW
    clock.advance(601.0)
    d = svc.authorize(act("wait"), t, {})
    assert d.verdict == DENY and d.policy_ids == ["L-DURATION"]
    assert "duration_s" in svc.limits_state()["tasks"][t.objective_id]["exhausted"]


def test_rate_limit_per_destination(tmp_path, clock):
    with open(DEFAULT_POLICY_PATH) as f:
        doc = yaml.safe_load(f)
    doc["rate_limits"]["default"] = {"count": 3, "window_s": 60}
    doc["rate_limits"]["write"] = {"count": 2, "window_s": 600}
    doc["rate_limits"]["per_destination"] = [{"match": "channel:staging:slow*", "count": 1, "window_s": 3600,
                                              "effects": ["external_write"]}]
    # two more local services: "localhost" is the same host as 127.0.0.1, so another destination is another port
    doc["destinations"]["local_services"] = ["127.0.0.1:8765", "127.0.0.1:8766", "127.0.0.1:9"]
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump(doc))
    svc = PolicyService(str(tmp_path / "run"), str(p), clock=clock, fsync=False)
    t = task()
    for i in range(3):
        assert svc.authorize(get(f"/{i}"), t, {}).verdict == ALLOW
    d = svc.authorize(get("/4"), t, {})
    assert d.verdict == DENY and d.policy_ids == ["L-RATE"] and "retry in" in d.reasons[0] and is_resource_denial(d)
    other_host = act("http.get", arguments={"url": "http://localhost:8766/a"})
    assert svc.authorize(other_host, t, {}).verdict == ALLOW             # limits are per destination
    clock.advance(61.0)
    assert svc.authorize(get("/5"), t, {}).verdict == ALLOW              # the window moved on
    post = lambda i: act("http.post", arguments={"url": f"http://[::1]:9/items/{i}", "body": f"item {i}"})   # noqa: E731
    assert svc.authorize(post(1), t, {}).verdict == ALLOW
    assert svc.authorize(post(2), t, {}).verdict == ALLOW
    assert svc.authorize(post(3), t, {}).policy_ids == ["L-RATE"]        # writes have their own, tighter limit
    slow = lambda i: act("publish.post", arguments={"destination": "channel:staging:slow-1", "body": f"note number {i}"})  # noqa: E731
    assert svc.authorize(slow(1), t, {}).verdict == ALLOW
    assert svc.authorize(slow(2), t, {}).policy_ids == ["L-RATE"]        # the operator's rule for this destination
    svc.close()


def test_primitive_count_and_rate(tmp_path, clock):
    with open(DEFAULT_POLICY_PATH) as f:
        doc = yaml.safe_load(f)
    doc["limits"]["max_primitives_per_action"] = 3
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump(doc))
    svc = PolicyService(str(tmp_path / "run"), str(p), clock=clock, fsync=False)
    a, t = act("skill.run", arguments={"skill_id": "s"}), task()
    start(svc, a, t)
    op = C.PrimitiveOp("browser.navigate", C.Modality.STRUCTURED_BROWSER, {"url": LOOP})
    assert [svc.check_primitive(op, a).verdict for _ in range(3)] == [ALLOW] * 3
    d = svc.check_primitive(op, a)
    assert d.verdict == DENY and d.policy_ids == ["L-PRIMITIVES"]
    svc.close()


def test_limit_state_survives_restart(svc, clock):
    t = task(limits=C.Limits(money=10.0, max_decisions=50))
    run(svc, get(money=4.0), t, money=4.0)
    pending = get("/p", money=3.0)
    dp = svc.authorize(pending, t, {})
    s2 = reopen(svc, clock)
    st = s2.limits_state()["tasks"][t.objective_id]
    assert st["used"]["money"] == 4.0 and st["reserved"]["money"] == 3.0 and st["used"]["decisions"] == 2
    assert s2.authorize(get(money=4.0), t, {}).verdict == DENY           # 4 + 3 reserved + 4 > 10
    assert s2.check_grant(pending, dp)
    s2.record(pending, ok_result(pending, money=3.0))
    assert s2.limits_state()["tasks"][t.objective_id]["used"]["money"] == 7.0
    s2.close()


def test_the_learner_cannot_raise_limits_by_editing_the_task(svc):
    t = task(limits=C.Limits(money=5.0))
    run(svc, get(money=5.0), t, money=5.0)
    richer = task(limits=C.Limits(money=5000.0))                         # same objective id, bigger budget
    d = svc.authorize(get(money=1.0), richer, {})
    assert d.verdict == DENY and d.policy_ids == ["P-TASK-MISMATCH"]
    assert svc.limits_state()["tasks"][t.objective_id]["limits"]["money"] == 5.0


def test_live_budgets_come_from_the_record_and_overspend_is_logged(svc, tmp_path, clock):
    t = live_task()                                                      # task says 50; the record says 20, 5 per action
    authorize_live(svc, tmp_path, t, clock, preauthorized=["https://owned.example/*"])
    assert svc.authorize(act("wait"), t, {}).verdict == ALLOW             # pins the task named by the record
    lim = svc.limits_state()["tasks"][t.objective_id]["limits"]
    assert lim["money"] == 20.0 and lim["money_per_action"] == 5.0 and lim["max_decisions"] == 100 and lim["duration_s"] == 1800.0
    a = act("http.get", arguments={"url": "https://docs.example/a"}, est_cost=C.Cost(money=6.0))
    assert svc.authorize(a, t, {}).policy_ids == ["L-MONEY-ACTION"]
    b = act("http.get", arguments={"url": "https://docs.example/b"}, est_cost=C.Cost(money=1.0))
    d = svc.authorize(b, t, {})
    assert d.verdict == NEEDS                                            # real money: the operator approves
    r = svc.resolve_approval(d.approval_id, True, "olga")
    assert svc.check_grant(b, r)
    svc.record(b, ok_result(b, money=1.75))                              # more than was authorized
    ev = [e for e in svc.audit_tail(5) if e["event"] == "limit_event"]
    assert ev and "more than" in ev[-1]["data"]["reason"]
    assert svc.limits_state()["tasks"][t.objective_id]["used"]["money"] == 1.75


def test_exhaustion_is_written_to_the_audit_log(svc):
    t = task(limits=C.Limits(max_external_actions=1))
    run(svc, get(), t)
    assert svc.authorize(get("/b"), t, {}).verdict == DENY
    events = [e for e in svc.audit_tail(20) if e["event"] == "limit_event"]
    assert events and any(e["data"]["policy_id"] in ("L-EXTERNAL-ACTIONS", "L-EXHAUSTED:max_external_actions") for e in events)


def test_history_no_rule_can_see_is_garbage_collected(svc, clock):
    t = task()
    for i in range(5):
        run(svc, get(f"/{i}"), t)
    assert len(svc.store.q("SELECT id FROM rate_events")) >= 5
    clock.advance(4000.0)                                  # far beyond every rate window
    run(svc, get("/later"), t)
    assert len(svc.store.q("SELECT id FROM rate_events")) == 1
    assert used(svc, t)["external_actions"] == 6           # accounting is not history: it stays
    assert svc.verify_audit().ok
