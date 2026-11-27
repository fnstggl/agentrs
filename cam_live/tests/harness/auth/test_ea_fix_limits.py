"""Regression tests for G-sec M1 / G-env C2: every limit is charged per declared step and per external-effect
primitive, never once per ActionSpec (docs/EA/FIXES.md)."""
import pytest
import yaml

from cam_live import contracts as C
from cam_live.harness.auth import DEFAULT_POLICY_PATH, PolicyService, is_resource_denial
from ea_helpers import ALLOW, DENY, LOOP, NEEDS, act, authorize_live, live_task, ok_result, prim, start, task

API = C.Modality.API
WRITE_CAP = 10                      # default_policy.yaml rate_limits.write.count


def post(i, site=LOOP):
    return act("http.post", API, arguments={"url": f"{site}/items/{i}", "json": {"text": f"item {i}"}})


def net(i, method="POST", site=LOOP):
    return prim("net.request", API, method=method, url=f"{site}/items/{i}")


def used(svc, t):
    return svc.limits_state()["tasks"][t.objective_id]["used"]


def policy_with(tmp_path, mutate):
    with open(DEFAULT_POLICY_PATH) as f:
        doc = yaml.safe_load(f)
    mutate(doc)
    p = tmp_path / "policy.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return str(p)


def test_a_program_declaring_more_writes_than_the_window_allows_is_refused_up_front(svc):
    t = task(limits=C.Limits(max_external_actions=1000, max_decisions=1000, duration_s=1e6))
    prog = act("program", C.Modality.INTERNAL, level=C.Level.PROCEDURE, steps=[post(i) for i in range(WRITE_CAP + 1)])
    d = svc.authorize(prog, t, {})
    assert d.verdict == DENY and "L-RATE" in d.policy_ids and is_resource_denial(d)
    assert used(svc, t)["external_actions"] == 0


def test_declared_steps_prepay_their_realisations_and_extra_primitives_are_charged(svc):
    t = task(limits=C.Limits(max_external_actions=1000, max_decisions=1000, duration_s=1e6))
    prog = act("program", C.Modality.INTERNAL, level=C.Level.PROCEDURE, steps=[post(i) for i in range(4)])
    start(svc, prog, t)
    for i in range(4):                       # the declared steps: realised against their prepaid share
        assert svc.check_primitive(net(i), prog).verdict == ALLOW
    n = 4
    while svc.check_primitive(net(100 + n), prog).verdict == ALLOW:     # undeclared writes: charged anew
        n += 1
    assert n == WRITE_CAP                    # never more writes than the window allows, batched or not
    d = svc.check_primitive(net(999), prog)
    assert d.verdict == DENY and d.policy_ids == ["L-RATE"]
    svc.record(prog, ok_result(prog))
    u = used(svc, t)
    assert u["external_actions"] == WRITE_CAP and u["decisions"] == 1       # one decision, ten external writes
    assert svc.authorize(post(1000), t, {}).verdict == DENY                 # the same window, one by one


def test_skill_steps_are_charged_once_each_and_never_beyond_the_window(svc):
    """The skill executor declares each step (note "skill:...") and the step's own executor then reports the
    realised request: one write per step, not two -- and a skill gets exactly the writes of its steps one by one."""
    t = task(limits=C.Limits(max_external_actions=1000, max_decisions=1000, duration_s=1e6))
    run = act("skill.run", C.Modality.SKILL, arguments={"skill_id": "s1", "arguments": {}})
    start(svc, run, t)
    steps = 0
    for i in range(WRITE_CAP + 5):
        decl = C.PrimitiveOp("http.request", API, {"method": "POST", "url": f"{LOOP}/api/items", "json": {"n": i}},
                             note=f"skill:s1:v1:step{i}")
        if svc.check_primitive(decl, run).verdict != ALLOW:
            break
        real = prim("http.request", API, method="POST", url=f"{LOOP}/api/items", hop=0, body_sha256="x")
        assert svc.check_primitive(real, run).verdict == ALLOW              # uses the declared step's share
        steps += 1
    assert steps == WRITE_CAP
    unannounced = prim("http.request", API, method="POST", url=f"{LOOP}/api/items", hop=0)
    assert svc.check_primitive(unannounced, run).verdict == DENY            # no share left, window full
    svc.record(run, ok_result(run))
    assert used(svc, t)["external_actions"] == WRITE_CAP


def test_max_external_actions_bounds_primitives_inside_one_action(svc):
    t = task(limits=C.Limits(max_external_actions=3, max_decisions=100, duration_s=1e6))
    run = act("skill.run", C.Modality.SKILL, arguments={"skill_id": "s2"})
    start(svc, run, t)
    verdicts = [svc.check_primitive(prim("browser.navigate", C.Modality.STRUCTURED_BROWSER, url=f"{LOOP}/p{i}"),
                                    run) for i in range(5)]
    assert [v.verdict for v in verdicts] == [ALLOW] * 3 + [DENY] * 2
    assert "L-EXTERNAL-ACTIONS" in verdicts[3].policy_ids
    svc.record(run, ok_result(run))
    assert used(svc, t)["external_actions"] == 3
    assert svc.authorize(act("browser.navigate", arguments={"url": LOOP + "/x"}), t, {}).verdict == DENY


def test_a_partly_executed_program_is_charged_for_what_it_did(svc):
    t = task(limits=C.Limits(max_external_actions=100, max_decisions=100, duration_s=1e6))
    prog = act("program", C.Modality.INTERNAL, level=C.Level.PROCEDURE, steps=[post(i) for i in range(5)])
    start(svc, prog, t)
    for i in range(2):
        assert svc.check_primitive(net(i), prog).verdict == ALLOW
    svc.record(prog, C.ActionResult(action_id=prog.action_id, status=C.ResultStatus.DENIED))
    assert used(svc, t)["external_actions"] == 2


def test_global_write_rate_and_task_configured_rates_within_the_mode_caps(svc):
    res = {"owned": ["channel:staging:*"], "preauthorized": ["channel:staging:*"],
           "rate_limits": {"write_global": {"count": 3, "window_s": 600}, "write": {"count": 100000, "window_s": 1}}}
    t = task(resources=res, limits=C.Limits(max_external_actions=100, max_decisions=100, duration_s=1e6))
    msg = lambda c: act("publish.post", API, arguments={"destination": f"channel:staging:{c}", "body": f"note {c}"})  # noqa: E731
    assert [svc.authorize(msg(c), t, {}).verdict for c in "abc"] == [ALLOW] * 3
    d = svc.authorize(msg("d"), t, {})                       # a fourth destination: the task-wide write limit
    assert d.verdict == DENY and "L-RATE" in d.policy_ids and "all destinations" in d.reasons[0]
    from cam_live.harness.auth.evaluate import TaskCtx  # noqa: F401  (type only)
    tc, _ = svc._task_ctx(t, 0.0)
    cap = svc.policy.modes["staging"].rate_caps["write"]
    assert tc.rates["write"].count == cap.count and tc.rates["write"].window_s == cap.window_s   # clamped
    bad = task(oid="bad-rates", resources={"rate_limits": {"write": {"count": "many"}}})
    assert "P-TASK-INVALID" in svc.authorize(act("wait"), bad, {}).policy_ids


def test_live_record_sets_the_write_rate_within_the_live_cap(svc, tmp_path, clock):
    t = live_task(oid="obj-live-rate")
    authorize_live(svc, tmp_path, t, clock, preauthorized=["https://owned.example/*"],
                   rate_limits={"write": {"count": 2, "window_s": 600}})
    w = lambda i: act("http.post", API, arguments={"url": f"https://owned.example/n/{i}", "json": {"n": i}})  # noqa: E731
    assert [svc.authorize(w(i), t, {}).verdict for i in range(3)] == [ALLOW, ALLOW, DENY]
    t2 = live_task(oid="obj-live-rate2")
    authorize_live(svc, tmp_path, t2, clock, record_id="rec-2", rate_limits={"write": {"count": 10 ** 6, "window_s": 1}})
    tc, _ = svc._task_ctx(t2, clock.now())
    cap = svc.policy.modes["live"].rate_caps["write"]
    assert (tc.rates["write"].count, tc.rates["write"].window_s) == (cap.count, cap.window_s)


def test_hard_bound_on_external_primitives_per_action(tmp_path, clock):
    p = policy_with(tmp_path, lambda doc: doc["limits"].__setitem__("max_external_primitives_per_action", 2))
    svc = PolicyService(str(tmp_path / "run"), p, clock=clock, fsync=False)
    t = task(limits=C.Limits(max_external_actions=100, max_decisions=100, duration_s=1e6))
    prog = act("program", C.Modality.INTERNAL, level=C.Level.PROCEDURE, steps=[post(i) for i in range(3)])
    assert "L-PRIMITIVES" in svc.authorize(prog, t, {}).policy_ids
    run = act("skill.run", C.Modality.SKILL, arguments={"skill_id": "s3"})
    start(svc, run, t)
    v = [svc.check_primitive(prim("browser.navigate", C.Modality.STRUCTURED_BROWSER, url=f"{LOOP}/{i}"), run).verdict
         for i in range(3)]
    assert v == [ALLOW, ALLOW, DENY]
    svc.close()


def test_money_is_charged_per_primitive(svc):
    t = task(limits=C.Limits(money=10.0, max_external_actions=100, max_decisions=100, duration_s=1e6))
    run = act("skill.run", C.Modality.SKILL, arguments={"skill_id": "s4"})
    start(svc, run, t)
    pay = prim("payment.send", API, payee=f"{LOOP}/pay", amount=6.0)
    assert svc.check_primitive(pay, run).verdict == ALLOW
    d = svc.check_primitive(prim("payment.send", API, payee=f"{LOOP}/pay", amount=6.0), run)
    assert d.verdict == DENY and "L-MONEY-TOTAL" in d.policy_ids          # 12 > 10, although the action declared 0
    svc.record(run, ok_result(run))
    assert used(svc, t)["money"] == 6.0


def test_interactions_on_staging_pages_are_not_external_writes(svc):
    t = task(limits=C.Limits(max_external_actions=1000, max_decisions=1000, duration_s=1e6))
    for i in range(WRITE_CAP * 2):
        a = act("browser.fill", C.Modality.STRUCTURED_BROWSER, arguments={"ref": "e3", "text": f"v{i}",
                                                                          "page_url": LOOP + "/items/new"})
        d = svc.authorize(a, t, {})
        assert d.verdict == ALLOW, (i, d.reasons)
    effects = [e for e in svc.audit_tail(5) if e["event"] == "authorize"][-1]["data"]["effects"]
    assert effects == ["input:url:127.0.0.1:8765/items/new"]
