"""ComputerEnvironment core (no Chromium): contract, authorization, at-most-once, serialisation, events, outcomes,
verification, logging, restart recovery. The side effects are real HTTP requests to the staging sites."""
import json
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from cam_live.contracts import (EXEC_INTERFACE_VERSION, ActionResult, AuthorizationDecision, AuthVerdict, EnvironmentAdapter,
                                FailureClass, Level, Limits, Modality, ModalityExecutor, Observation, OutcomeKind,
                                OutcomeObservation, OutcomeQuery, OutcomeStatus, Quantity, ResultStatus, RunFiles,
                                SimClock, ToolDescriptor, Trust, VerificationCondition, WallClock, new_id)
from cam_live.harness.env import PROGRAM, ComputerEnvironment, validate_args
from cam_live.harness.internal import InternalExecutor

from eb_support import SRC, Driver, GrantPolicy, HttpExecutor, make_action


def build(run_dir, policy=None, clock=None, **kw):
    policy = policy or GrantPolicy()
    env = ComputerEnvironment(str(run_dir), policy, clock=clock, **kw)
    env.register(InternalExecutor(str(run_dir), clock=env.clock))
    http = HttpExecutor(env.clock)
    env.register(http)
    return env, policy, http


@pytest.fixture
def rig(tmp_path, task):
    env, policy, http = build(tmp_path)
    d = Driver(env, policy, task)
    d.http = http
    yield d
    env.close()


def log_records(run_dir):
    with open(os.path.join(str(run_dir), RunFiles.EVENTS)) as f:
        return [json.loads(x) for x in f]


# ---- contract ------------------------------------------------------------------------------------------------------
def test_environment_contract_conformance(rig, sites):
    env = rig.env
    assert isinstance(env, EnvironmentAdapter) and env.interface_version == EXEC_INTERFACE_VERSION
    assert isinstance(env.executor(Modality.INTERNAL), ModalityExecutor)
    tools = env.discover_tools()
    names = {t.name for t in tools}
    assert all(isinstance(t, ToolDescriptor) for t in tools) and {"wait", "internal.inspect", "internal.note",
                                                                   "http.get", "http.post", PROGRAM} <= names
    assert all(t.description and t.arg_schema.get("type") == "object" for t in tools)
    obs = env.observe()
    assert obs and all(isinstance(o, Observation) and o.provenance.trust == Trust.UNTRUSTED for o in obs)
    res = rig.ok("http.get", url=sites.urls["structured"] + "/report")
    assert isinstance(res, ActionResult) and res.ops and res.cost.primitive_ops == 1 and res.ended >= res.started
    events, cur = env.poll_events("")
    assert isinstance(events, list) and isinstance(cur, str)
    assert isinstance(env.get_resource_state(), dict) and isinstance(env.get_capabilities(), dict)
    assert all(isinstance(x, OutcomeQuery) for x in env.get_pending_outcomes())
    assert all(isinstance(x, Quantity) for x in env.get_observable_metrics())
    assert env.measure([]) == []
    caps = env.get_capabilities()
    assert caps["provenance"] == "staging" and caps["live"] is False and caps["features"]["reward"] is False
    # observations and consequences only: no reward anywhere in what the environment returns
    blob = json.dumps([res.to_dict(), [o.to_dict() for o in obs], env.get_resource_state(), caps], default=str).lower()
    assert "reward\":" not in blob.replace('"reward": false', "")


def test_validate_action_reports_structural_errors(rig, sites):
    env, url = rig.env, sites.urls["structured"] + "/"
    assert env.validate_action(make_action("http.get", url=url)) == []
    assert "unknown operation" in env.validate_action(make_action("nope.op"))[0]
    assert any("required" in e for e in env.validate_action(make_action("http.get")))
    assert any("not a known argument" in e for e in env.validate_action(make_action("wait", seconds=1, extra=2)))
    assert any("<=" in e or ">=" in e for e in env.validate_action(make_action("wait", seconds=-5)))
    wrong = make_action("http.get", url=url)
    wrong.modality = Modality.CODE
    assert any("belongs to modality" in e for e in env.validate_action(wrong))
    assert any("at least one step" in e for e in env.validate_action(make_action(PROGRAM)))
    bad_prog = make_action(PROGRAM, steps=[make_action("http.get", url=url), make_action("wait", seconds="x")])
    assert any(e.startswith("step 1 (wait)") for e in env.validate_action(bad_prog))
    assert any("does not take steps" in e for e in env.validate_action(make_action("http.get", url=url, steps=[
        make_action("wait", seconds=1)])))
    assert validate_args({"type": "object", "properties": {"n": {"type": "integer", "enum": [1, 2]}}}, {"n": 3})
    res = rig.do("wait", seconds="soon")
    assert res.status == ResultStatus.FAILED and res.failure_class == FailureClass.INVALID_ARGUMENTS


# ---- authorization -----------------------------------------------------------------------------------------------
def test_nothing_runs_without_a_verified_grant(rig, sites):
    env, url = rig.env, sites.urls["structured"] + "/items"
    a = make_action("http.get", url=url)
    for bad in (None, AuthorizationDecision(AuthVerdict.DENY, a.content_hash(), ["no"]),
                AuthorizationDecision(AuthVerdict.ALLOW, a.content_hash(), grant="forged"),
                AuthorizationDecision(AuthVerdict.ALLOW, "0" * 64, grant="x")):
        res = env.execute(a, bad)
        assert res.status == ResultStatus.DENIED and res.failure_class in FailureClass.HARD and not res.ops
    # a grant issued for one action does not cover a different one
    other = make_action("http.get", url=sites.urls["structured"] + "/report")
    res = env.execute(other, rig.policy.authorize(a, rig.task))
    assert res.status == ResultStatus.DENIED
    expired = rig.policy.authorize(a, rig.task)
    expired.expires_at = env.clock.now() - 1
    assert env.execute(a, expired).status == ResultStatus.DENIED
    assert rig.http.calls == [] and rig.policy.primitives == []
    assert env.execute(a, rig.policy.authorize(a, rig.task)).status == ResultStatus.OK and rig.http.calls == [a.action_id]


def test_needs_approval_then_same_action_runs_after_approval(tmp_path, task, sites):
    env, policy, http = build(tmp_path, GrantPolicy(approve_ops={"http.post"}))
    try:
        a = make_action("http.post", url=sites.urls["dynamic"] + "/api/refresh")
        g0 = sites.dynamic.generation
        first = env.execute(a, policy.authorize(a, task))
        assert first.status == ResultStatus.NEEDS_APPROVAL and sites.dynamic.generation == g0
        (pending,) = policy.approvals()
        done = env.execute(a, policy.resolve_approval(pending["approval_id"], True, "operator"))
        assert done.status == ResultStatus.OK and sites.dynamic.generation == g0 + 1
    finally:
        env.close()


def test_every_primitive_is_checked_and_a_denial_is_hard(tmp_path, task, sites):
    s = sites.urls["structured"]
    policy = GrantPolicy()
    env, policy, http = build(tmp_path, policy)
    d = Driver(env, policy, task)
    try:
        steps = [make_action("http.get", url=s + "/items"), make_action("wait", seconds=0),
                 make_action("http.get", url=s + "/report")]
        res = d.submit(make_action(PROGRAM, steps=steps, level=Level.PROCEDURE))
        assert res.status == ResultStatus.OK and [o.op for o in res.ops] == ["http.get", "wait", "http.get"]
        assert [p[0] for p in policy.primitives] == ["http.get", "wait", "http.get"]
        assert {p[2] for p in policy.primitives} == {res.action_id}          # checks are bound to the granted action
        # the policy starts refusing one destination after the grant: the primitive check stops the program there
        policy.primitives.clear()
        prog = make_action(PROGRAM, steps=[make_action("http.get", url=s + "/items"),
                                           make_action("http.post", url=s + "/items/1/archive"),
                                           make_action("http.get", url=s + "/report")])
        grant = policy.authorize(prog, task)
        policy.deny_urls.append("/items/1/archive")
        res = env.execute(prog, grant)
        assert res.status == ResultStatus.DENIED and res.failure_class in FailureClass.HARD
        assert "step 1 (http.post)" in res.failure_detail and res.retries == 0
        assert [p[0] for p in policy.primitives] == ["http.get", "http.post"]         # nothing after the denial
        assert sites.structured.items[0]["status"] == "active" and len(http.calls) == 3   # the POST never left
    finally:
        env.close()


def test_emergency_stop_refuses_everything(rig, sites):
    rig.policy.emergency_stop("test", "operator")
    a = make_action("wait", seconds=0)
    res = rig.env.execute(a, AuthorizationDecision(AuthVerdict.ALLOW, a.content_hash(), grant=rig.policy._grant(a)))
    assert res.status == ResultStatus.DENIED and "emergency stop" in res.failure_detail


# ---- coordination --------------------------------------------------------------------------------------------------
def test_duplicate_submission_executes_once(rig, sites):
    g0 = sites.dynamic.generation
    a = make_action("http.post", url=sites.urls["dynamic"] + "/api/refresh")
    grant = rig.policy.authorize(a, rig.task)
    first = rig.env.execute(a, grant)
    again = rig.env.execute(a, grant)
    assert first.status == ResultStatus.OK and again.to_dict() == first.to_dict()
    assert sites.dynamic.generation == g0 + 1 and rig.http.calls.count(a.action_id) == 1
    assert rig.env.get_resource_state()["duplicates_suppressed"] == 1
    # the same id with different content is refused rather than run
    b = make_action("http.post", url=sites.urls["dynamic"] + "/api/refresh?x=1", aid=a.action_id)
    res = rig.env.execute(b, rig.policy.authorize(b, rig.task))
    assert res.status == ResultStatus.FAILED and res.failure_class == FailureClass.INVALID_ARGUMENTS
    assert sites.dynamic.generation == g0 + 1


def test_concurrent_duplicates_execute_once_and_share_the_result(rig, sites):
    n0 = sites.structured.slow_started
    a = make_action("http.post", url=sites.urls["structured"] + "/slow?ms=300")
    grant = rig.policy.authorize(a, rig.task)
    out = []
    threads = [threading.Thread(target=lambda: out.append(rig.env.execute(a, grant))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(out) == 4 and all(r.status == ResultStatus.OK and r.to_dict() == out[0].to_dict() for r in out)
    assert sites.structured.slow_started == n0 + 1


def test_writes_are_serialised_reads_run_concurrently(rig, sites):
    site, url = sites.structured, sites.urls["structured"] + "/slow?ms=350"

    def burst(op):
        site.slow_max_active = 0
        acts = [make_action(op, url=url) for _ in range(3)]
        grants = [rig.policy.authorize(a, rig.task) for a in acts]
        ts = [threading.Thread(target=rig.env.execute, args=(a, g)) for a, g in zip(acts, grants)]
        t0 = time.time()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return site.slow_max_active, time.time() - t0

    w_max, w_secs = burst("http.post")
    r_max, r_secs = burst("http.get")
    assert w_max == 1 and w_secs >= 3 * 0.35           # state-changing operations never overlap
    assert r_max >= 2 and r_secs < 3 * 0.35            # read-only work does


_CRASH = r"""
import sys, time
sys.path.insert(0, {src!r}); sys.path.insert(0, {here!r})
from eb_support import GrantPolicy, HttpExecutor, make_action
from cam_live.contracts import TaskSpec
from cam_live.harness.env import ComputerEnvironment
env = ComputerEnvironment({run_dir!r}, GrantPolicy()); env.register(HttpExecutor())
a = make_action({op!r}, url={url!r}, aid={aid!r})
print("starting", flush=True)
env.execute(a, env.policy.authorize(a, TaskSpec("o", "x")))
"""


@pytest.mark.parametrize("op,rerun", [("http.post", False), ("http.get", True)])
def test_crash_retry_never_repeats_a_state_changing_action(tmp_path, task, sites, op, rerun):
    aid, url = new_id("act"), sites.urls["structured"] + "/slow?ms=4000"
    n0 = sites.structured.slow_started
    code = _CRASH.format(src=SRC, here=os.path.dirname(__file__), run_dir=str(tmp_path), op=op, url=url, aid=aid)
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "starting", proc.stderr.read()
    t0 = time.time()
    while sites.structured.slow_started == n0 and time.time() - t0 < 20:
        time.sleep(0.02)
    assert sites.structured.slow_started == n0 + 1
    proc.send_signal(signal.SIGKILL)                   # the process dies while the request is in flight
    proc.wait()
    env, policy, http = build(tmp_path)
    try:
        assert any(r["type"] == "env.started" and r["recovered_interrupted"] == [aid] for r in log_records(tmp_path))
        a = make_action(op, url=url, aid=aid)             # the retry: same id, same content
        res = env.execute(a, policy.authorize(a, task))
        if rerun:
            assert res.status == ResultStatus.OK and sites.structured.slow_started == n0 + 2
        else:
            assert res.status == ResultStatus.FAILED and res.failure_class == FailureClass.UNKNOWN
            assert "not executed again" in res.failure_detail and http.calls == []
            assert sites.structured.slow_started == n0 + 1
            assert env.execute(a, policy.authorize(a, task)).to_dict() == res.to_dict()      # and stays that way
    finally:
        env.close()


def test_running_action_can_be_cancelled(rig):
    a = make_action("wait", seconds=20, block=True)
    grant = rig.policy.authorize(a, rig.task)
    out = []
    t = threading.Thread(target=lambda: out.append(rig.env.execute(a, grant)))
    t0 = time.time()
    t.start()
    while rig.env.get_resource_state()["in_flight"] == 0 and time.time() - t0 < 5:
        time.sleep(0.01)
    assert rig.env.cancel_action(a.action_id) is True and rig.env.cancel_action("act_unknown") is False
    t.join(timeout=5)
    assert out and out[0].status == ResultStatus.CANCELLED and time.time() - t0 < 5


def test_one_environment_per_run_directory(rig):
    with pytest.raises(RuntimeError, match="already owns"):
        ComputerEnvironment(rig.env.run_dir, GrantPolicy())


# ---- staleness -----------------------------------------------------------------------------------------------------
def test_outdated_observation_reference_fails_before_any_executor_runs(rig, sites):
    url = sites.urls["structured"] + "/items"
    res = rig.do("http.get", url=url, obs_id="obs_does_not_exist")
    assert res.status == ResultStatus.FAILED and res.failure_class == FailureClass.STALE_OBSERVATION
    assert rig.http.calls == [] and rig.policy.primitives == []
    live = rig.ok("http.get", url=url).observations[0]
    assert rig.env.is_stale(live.obs_id) is None
    assert rig.do("http.get", url=url, obs_id=live.obs_id).status == ResultStatus.OK
    # surface generations: a newer observation of the same surface outdates the older one
    p = rig.env.perception
    o1, o2 = p.event("a"), p.event("b")
    o1.data.update(surface="screen:1", generation="g1")
    o2.data.update(surface="screen:1", generation="g2")
    rig.env.emit_event(o1)
    assert rig.env.is_stale(o1.obs_id) is None
    rig.env.emit_event(o2)
    assert "earlier state" in rig.env.is_stale(o1.obs_id) and rig.env.is_stale(o2.obs_id) is None
    res = rig.do("http.get", url=url, obs_id=o1.obs_id)
    assert res.failure_class == FailureClass.STALE_OBSERVATION


# ---- verification --------------------------------------------------------------------------------------------------
def test_verify_is_evaluated_against_the_resulting_state(rig, sites):
    s = sites.urls["structured"]
    ok = rig.do("http.get", url=s + "/report", verify=VerificationCondition("report shown", "text_present",
                                                                            {"text": "Total quantity"}))
    assert ok.status == ResultStatus.OK and ok.verified is True
    bad = rig.do("http.get", url=s + "/report", verify=VerificationCondition("", "text_present", {"text": "No such text"}))
    assert bad.status == ResultStatus.FAILED and bad.verified is False
    assert bad.failure_class == FailureClass.WRONG_SITE_ASSUMPTION and bad.ops[0].status == ResultStatus.OK
    strategic = rig.do("http.get", url=s + "/report", level=Level.INTERVENTION,
                       verify=VerificationCondition("", "text_present", {"text": "No such text"}))
    assert strategic.failure_class == FailureClass.STRATEGIC
    absent = rig.do("http.get", url=s + "/report", verify=VerificationCondition("", "text_present",
                                                                                {"text": "No such text", "absent": True}))
    assert absent.verified is True
    assert rig.do("http.get", url=s + "/report", verify=VerificationCondition("", "url_matches",
                                                                              {"pattern": r"/report$"})).verified is True
    m = rig.do("http.get", url=s + "/report", verify=VerificationCondition("", "metric", {
        "name": "doc.report.summary.active_items", "op": ">=", "value": 1}))
    assert m.verified is True
    m = rig.do("http.get", url=s + "/report", verify=VerificationCondition("", "metric", {
        "name": "doc.report.summary.active_items", "op": ">", "value": 10 ** 6}))
    assert m.verified is False and m.status == ResultStatus.FAILED
    unknown = rig.do("http.get", url=s + "/report", verify=VerificationCondition("", "metric", {"name": "no.metric"}))
    assert unknown.status == ResultStatus.OK and unknown.verified is None       # not checkable is not "passed"
    probe = make_action("http.get", url=s + "/api/state").to_dict()
    prog = rig.do("http.get", url=s + "/report", verify=VerificationCondition("", "program", {"steps": [probe]}))
    assert prog.verified is True and len(prog.ops) == 2
    writer = make_action("http.post", url=s + "/items/2/archive").to_dict()
    refused = rig.do("http.get", url=s + "/report", verify=VerificationCondition("", "program", {"steps": [writer]}))
    assert refused.verified is None and sites.structured.items[1]["status"] == "active"   # checks may not write
    assert rig.do("wait", seconds=0).verified is None


# ---- events, waiting, outcomes, resources ---------------------------------------------------------------------------
def test_event_stream_cursor_and_scheduled_wait(tmp_path, task):
    env, policy, _ = build(tmp_path)
    d = Driver(env, policy, task)
    try:
        assert env.poll_events("") == ([], "0")
        c1 = env.emit_event(env.perception.event("job.finished", "job 1 finished", quantities=[Quantity("job.rows", 7)]))
        env.emit_event(env.perception.event("job.finished", "job 2 finished"))
        events, cur = env.poll_events("")
        assert [e.text for e in events] == ["job 1 finished", "job 2 finished"] and cur == "2" and c1 == "1"
        assert env.poll_events(c1)[0][0].text == "job 2 finished" and env.poll_events(cur) == ([], "2")
        t0 = time.time()
        res = d.ok("wait", seconds=0.4, reason="let it settle")
        assert time.time() - t0 < 0.3                                   # scheduling does not block or burn anything
        wake = env.get_resource_state()["next_wake_at"]
        assert wake == pytest.approx(res.started + 0.4, abs=0.05)
        assert env.poll_events(cur) == ([], "2")
        time.sleep(0.5)
        events, cur2 = env.poll_events(cur)
        assert len(events) == 1 and events[0].data["event"] == "wait.finished" and events[0].data["action_id"] == res.action_id
        assert env.poll_events(cur2) == ([], cur2) and env.get_resource_state()["next_wake_at"] is None
        with pytest.raises(ValueError):
            env.poll_events("not-a-cursor")
    finally:
        env.close()


def test_wait_advances_a_simulated_clock_and_blocks_only_on_request(tmp_path, task):
    clock = SimClock(5000.0)
    env, policy, _ = build(tmp_path / "sim", clock=clock)
    try:
        res = Driver(env, policy, task).ok("wait", seconds=86400)
        assert clock.now() == 5000.0 + 86400 and res.cost.seconds == 86400 and env.get_capabilities()["simulated_time"]
    finally:
        env.close()
    env, policy, _ = build(tmp_path / "wall")
    try:
        t0 = time.time()
        res = Driver(env, policy, task).ok("wait", seconds=0.3, block=True)
        assert 0.28 <= time.time() - t0 < 2 and res.ops[0].note == "slept"
    finally:
        env.close()


def test_pending_outcomes_and_measure_through_a_source(tmp_path, task, sites):
    lab = sites.labsite.world
    env, policy, _ = build(tmp_path, clock=sites.clock)
    env.add_source(sites.outcome_source())
    d = Driver(env, policy, task)
    try:
        before = {q.query_id for q in env.get_pending_outcomes()}
        d.ok("http.post", url=sites.urls["labsite"] + "/act", form={"channel": "0", "effort": "3"})
        new = [q for q in env.get_pending_outcomes() if q.query_id not in before]
        assert len(new) == 1 and new[0].kind == OutcomeKind.COUNT and new[0].metric.startswith("lab.adopters.from.")
        q = new[0]
        assert env.measure([q])[0].status == OutcomeStatus.PENDING        # not due: no value is invented
        assert q.query_id in {x.query_id for x in env.get_pending_outcomes()}
        d.ok("wait", seconds=q.due_at - sites.clock.now() + 1)
        (obs,) = env.measure([q])
        aid = q.metric.rsplit(".", 1)[1]
        truth = sum(1 for t, a, _ in lab._arrivals if a == aid and t <= q.due_at)
        assert obs.status == OutcomeStatus.OBSERVED and obs.value == float(truth) and obs.verified is False
        assert q.query_id not in {x.query_id for x in env.get_pending_outcomes()}
        ghost = OutcomeQuery(new_id("q"), "no.such.metric", issued_at=0.0)
        assert env.measure([ghost])[0].status == OutcomeStatus.UNOBSERVABLE and env.measure([ghost])[0].value is None
        snap = OutcomeQuery(new_id("q"), "lab.adopters.total", kind=OutcomeKind.COUNT, issued_at=sites.clock.now())
        assert env.measure([snap])[0].value == float(lab.total()) and lab.total() >= truth
        assert {m.name for m in env.get_observable_metrics()} >= {"lab.adopters.total", "env.actions"}
    finally:
        env.close()


def test_resource_accounting_and_limits(tmp_path, task, sites):
    env, policy, _ = build(tmp_path, limits=Limits(max_external_actions=2))
    d = Driver(env, policy, task)
    try:
        url = sites.urls["dynamic"] + "/api/refresh"
        d.ok("http.get", url=sites.urls["dynamic"] + "/api/figures")
        d.ok("http.post", url=url)
        d.ok("http.post", url=url)
        g = sites.dynamic.generation
        res = d.do("http.post", url=url)
        assert res.status == ResultStatus.FAILED and res.failure_class == FailureClass.RESOURCE_LIMIT
        assert sites.dynamic.generation == g
        r = env.get_resource_state()
        assert r["actions"] == 3 and r["ok"] == 3 and r["external_actions"] == 2 and r["primitive_ops"] == 3
        assert r["remaining"]["external_actions"] == 0 and r["by_modality"]["api"]["actions"] == 3
        assert r["seconds"] > 0 and r["policy_limits"] == {"primitive_checks": 3}
        assert d.do("http.get", url=sites.urls["dynamic"] + "/api/figures").status == ResultStatus.OK   # reads still fine
        assert policy.records == [(a, ResultStatus.OK) for a, _ in policy.records] and len(policy.records) == 4
    finally:
        env.close()


# ---- logging, secrets, persistence ----------------------------------------------------------------------------------
class _Broker:
    """Executor-side handle with the same two methods the authorization workstream's handle has."""
    VALUE = "tok-9f8e7d6c5b4a"

    def get(self, name):
        return self.VALUE

    def scrub(self, text):
        return text.replace(self.VALUE, "[secret:demo]")


def test_execution_log_is_complete_and_secrets_are_scrubbed(tmp_path, task, sites):
    env, policy, _ = build(tmp_path, secrets=_Broker())
    d = Driver(env, policy, task)
    try:
        s = sites.urls["structured"]
        prog = make_action(PROGRAM, steps=[make_action("http.get", url=s + "/echo?value=" + _Broker.VALUE),
                                           make_action("internal.note", text="checked the echo page"),
                                           make_action("http.get", url=s + "/report")])
        res = d.submit(prog)
        denied = d.do("http.get", url=s + "/items", obs_id="obs_gone")
        assert res.status == ResultStatus.OK and denied.status == ResultStatus.FAILED
        assert _Broker.VALUE not in json.dumps(res.to_dict()) and "[secret:demo]" in res.observations[0].text
    finally:
        env.close()
    raw = open(os.path.join(str(tmp_path), RunFiles.EVENTS)).read()
    assert _Broker.VALUE not in raw and "[secret:demo]" in raw
    recs = [r for r in log_records(tmp_path) if r["src"] == "computer_environment"]
    assert [r["seq"] for r in recs] == sorted(r["seq"] for r in recs) and all("ts" in r and "wall" in r for r in recs)
    mine = [r for r in recs if r.get("action_id") == prog.action_id or (r.get("action") or {}).get("action_id") == prog.action_id]
    kinds = [r["type"] for r in mine]
    assert kinds[0] == "env.action_submitted" and kinds[1] == "env.action_started" and kinds[-1] == "env.action_result"
    assert mine[0]["action"]["steps"][1]["arguments"]["text"] == "checked the echo page"
    assert mine[0]["authorization"]["verdict"] == "allow" and "grant" not in mine[0]["authorization"]
    assert kinds.count("env.primitive_check") == 3 and kinds.count("env.primitive_op") == len(res.ops) == 3
    logged_obs = {r["observation"]["obs_id"] for r in mine if r["type"] == "env.observation"}
    assert logged_obs == {o.obs_id for o in res.observations} and len(logged_obs) == 3
    final = mine[-1]
    assert final["status"] == "ok" and final["observation_ids"] == [o.obs_id for o in res.observations]
    assert final["primitive_ops"] == 3 and final["cost"]["primitive_ops"] == 3
    refused = [r for r in recs if r["type"] == "env.action_refused"]
    assert refused and refused[-1]["failure_class"] == FailureClass.STALE_OBSERVATION
    assert recs[0]["type"] == "env.started" and recs[-1]["type"] == "env.closed"


def test_state_survives_restart(tmp_path, task, sites):
    env, policy, _ = build(tmp_path)
    d = Driver(env, policy, task)
    d.ok("internal.note", text="remember this", tags=["t"])
    with open(os.path.join(env.workspace, "artifact.txt"), "w") as f:
        f.write("kept")
    env.emit_event(env.perception.event("e1", "first"))
    _, cursor = env.poll_events("")
    env.emit_event(env.perception.event("e2", "second"))
    q = OutcomeQuery(new_id("q"), "env.actions", kind=OutcomeKind.COUNT, horizon_s=3600, issued_at=env.clock.now())
    env.add_pending(q)
    d.ok("wait", seconds=3600, reason="later")
    done = d.ok("http.get", url=sites.urls["structured"] + "/report")
    before = env.get_resource_state()
    env.close()

    env2, policy2, http2 = build(tmp_path)                       # a new process would do exactly this
    d2 = Driver(env2, policy2, task)
    try:
        assert [x.query_id for x in env2.get_pending_outcomes()] == [q.query_id]
        events, cur = env2.poll_events(cursor)                   # the old cursor still means the same position
        assert [e.text for e in events] == ["second"] and cur == "2"
        after = env2.get_resource_state()
        assert after["actions"] == before["actions"] == 3 and after["next_wake_at"] == before["next_wake_at"]
        assert open(os.path.join(env2.workspace, "artifact.txt")).read() == "kept"
        notes = d2.ok("internal.inspect", what="notes").observations[0].data["records"]
        assert [n["text"] for n in notes] == ["remember this"]
        same = make_action("http.get", url=sites.urls["structured"] + "/report", aid=done.action_id)
        again = env2.execute(same, policy2.authorize(same, task))
        assert again.to_dict() == done.to_dict() and http2.calls == []    # at-most-once holds across the restart
        assert env2.is_stale(done.observations[0].obs_id)                 # observations do not outlive the process
    finally:
        env2.close()


def test_internal_inspect_reads_own_records(rig, sites):
    done = rig.ok("http.get", url=sites.urls["structured"] + "/report")
    rig.do("http.get", url=sites.urls["structured"] + "/items", obs_id="obs_none")
    recs = rig.ok("internal.inspect", what="results", limit=5).observations[0].data["records"]
    assert recs[-1]["action_id"] == done.action_id and recs[-1]["status"] == "ok"
    one = rig.ok("internal.inspect", what="action", action_id=done.action_id).observations[0].data["records"]
    assert [r["type"] for r in one][:2] == ["env.action_submitted", "env.action_started"]
    assert rig.ok("internal.inspect", what="resources").observations[0].data["records"]["actions"] >= 2
    tools = rig.ok("internal.inspect", what="tools").observations[0].data["records"]
    assert {"name": "wait", "modality": "internal", "available": True} in tools
    assert rig.do("internal.inspect", what="action").failure_class == FailureClass.INVALID_ARGUMENTS


def test_executor_failures_do_not_kill_the_environment(tmp_path, task):
    class Broken:
        modality = Modality.CODE

        def tools(self):
            return [ToolDescriptor("broken.run", Modality.CODE, "Raise.", {"type": "object", "properties": {}})]

        def validate(self, action):
            return []

        def execute(self, action, ctx):
            raise RuntimeError("boom")

        def close(self):
            pass

    env = ComputerEnvironment(str(tmp_path), GrantPolicy())
    env.register(Broken())
    try:
        with pytest.raises(ValueError):
            env.register(Broken())
        a = make_action("broken.run")
        a.modality = Modality.CODE
        res = env.execute(a, env.policy.authorize(a, task))
        assert res.status == ResultStatus.FAILED and "boom" in res.failure_detail and res.observations
        assert env.get_resource_state()["failed"] == 1
    finally:
        env.close()
