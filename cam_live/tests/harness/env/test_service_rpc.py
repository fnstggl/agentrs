"""ExecRPC v1: the environment served over the versioned RPC, RemoteEnvironment and RemotePolicy clients, the
separate service process and its restart. The policy lives in the test process and is reached by the service over
a real authorization socket."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import pytest

from cam_live.contracts import (EXEC_INTERFACE_VERSION, ActionResult, AuthorizationDecision, AuthVerdict, EnvironmentAdapter,
                                FailureClass, Modality, Observation, OutcomeKind, OutcomeObservation, OutcomeQuery,
                                OutcomeStatus, PolicyEngine, PrimitiveOp, Provenance, Quantity, ResultStatus, TaskSpec,
                                ToolDescriptor, Trust, new_id)
from cam_live.harness.env import ComputerEnvironment
from cam_live.harness.internal import InternalExecutor
from cam_live.harness.rpc import RpcServer
from cam_live.harness.service import (EXEC_METHODS, EXEC_RPC_VERSIONS, POLICY_METHODS, RemoteEnvironment, RemotePolicy,
                                      VersionMismatch, negotiate, serve)

from eb_support import HERE, SRC, Driver, GrantPolicy, HttpExecutor, make_action

TASK = TaskSpec("obj_rpc", "exercise the rpc boundary", mode="staging")


@pytest.fixture
def short_dir():
    d = tempfile.mkdtemp(prefix="ebs")            # unix socket paths are limited to ~100 characters
    yield d
    shutil.rmtree(d, ignore_errors=True)


def spawn_service(run_dir, *extra):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([SRC, HERE]))
    proc = subprocess.Popen([sys.executable, "-m", "cam_live.harness.service", "--run-dir", run_dir,
                             "--socket", os.path.join(run_dir, "sock", "exec.sock"), *extra],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    line = proc.stdout.readline()
    if not line:
        raise AssertionError("service did not start: " + proc.stderr.read()[-2000:])
    return proc, json.loads(line)


def test_version_negotiation_rules():
    assert negotiate(["1.0"]) == "1.0" and negotiate(["0.9", "1.0", "7.3"]) == "1.0"
    assert negotiate(["1.0", "1.2"], ["1.0", "1.2", "1.10"]) == "1.2"
    with pytest.raises(VersionMismatch):
        negotiate(["2.0"])
    assert {"observe", "discover_tools", "validate_action", "execute", "poll_events", "get_resource_state",
            "get_pending_outcomes", "get_observable_metrics", "get_capabilities", "measure", "ping", "close",
            "hello"} <= set(EXEC_METHODS)
    assert set(POLICY_METHODS) == {m for m in dir(PolicyEngine) if not m.startswith("_")}


def test_rpc_round_trip_of_the_whole_adapter(short_dir, sites):
    policy = GrantPolicy()
    env = ComputerEnvironment(short_dir, policy, clock=sites.clock)
    env.register(InternalExecutor(short_dir, clock=env.clock))
    env.register(HttpExecutor(env.clock))
    env.add_source(sites.outcome_source())
    sock = os.path.join(short_dir, "sock", "exec.sock")
    server = serve(env, sock, group=None)
    try:
        remote = RemoteEnvironment(sock)
        assert isinstance(remote, EnvironmentAdapter)
        assert remote.rpc_version == "1.0" and remote.interface_version == EXEC_INTERFACE_VERSION
        assert remote.ping()["ok"] is True and set(remote.server_info["methods"]) == set(EXEC_METHODS)
        with pytest.raises(VersionMismatch):
            RemoteEnvironment(sock, versions=["9.9"])
        d = Driver(remote, policy, TASK)
        tools = remote.discover_tools()
        assert all(isinstance(t, ToolDescriptor) for t in tools)
        assert [t.to_dict() for t in tools] == [t.to_dict() for t in env.discover_tools()]
        obs = remote.observe()
        assert all(isinstance(o, Observation) and isinstance(o.provenance, Provenance) for o in obs)
        assert obs[-1].provenance.trust == Trust.UNTRUSTED and all(isinstance(q, Quantity) for q in obs[-1].quantities)
        a = make_action("http.get", url=sites.urls["structured"] + "/report")
        assert remote.validate_action(a) == [] and "unknown operation" in remote.validate_action(make_action("x.y"))[0]
        res = d.submit(a)
        assert isinstance(res, ActionResult) and res.status == ResultStatus.OK
        assert isinstance(res.ops[0], PrimitiveOp) and isinstance(res.observations[0], Observation)
        local = env._stored_result(a.action_id)
        assert res.to_dict() == local.to_dict()                       # nothing is lost or altered on the wire
        q = {x.name: x for x in res.observations[0].quantities}
        assert q["doc.report.summary.total_value"].unit == "credits" and isinstance(q["doc.report.summary.total_value"].value, float)
        assert remote.execute(a, policy.authorize(a, TASK)).to_dict() == res.to_dict()      # at-most-once over RPC too
        assert remote.execute(a, AuthorizationDecision(AuthVerdict.DENY)).to_dict() == res.to_dict()
        fresh = make_action("http.get", url=sites.urls["structured"] + "/report")
        denied = remote.execute(fresh, AuthorizationDecision(AuthVerdict.ALLOW, fresh.content_hash(), grant="forged"))
        assert denied.status == ResultStatus.DENIED and denied.failure_class in FailureClass.HARD
        env.emit_event(env.perception.event("tick", "an event", quantities=[Quantity("n", 1.5, "u")]))
        events, cur = remote.poll_events("")
        assert isinstance(events, list) and isinstance(cur, str) and cur == "1"
        assert events[0].text == "an event" and events[0].quantities[0].to_dict() == Quantity("n", 1.5, "u").to_dict()
        assert remote.poll_events(cur) == ([], "1")
        d.ok("http.post", url=sites.urls["labsite"] + "/act", form={"channel": "2", "effort": "1"})
        pend = remote.get_pending_outcomes()
        assert pend and all(isinstance(x, OutcomeQuery) for x in pend)
        out = remote.measure(pend[:1])
        assert isinstance(out[0], OutcomeObservation) and out[0].status in (OutcomeStatus.PENDING, OutcomeStatus.OBSERVED)
        assert {m.name for m in remote.get_observable_metrics()} >= {"env.actions", "lab.adopters.total"}
        rs, caps = remote.get_resource_state(), remote.get_capabilities()
        assert rs["actions"] == 2 and rs["duplicates_suppressed"] == 2
        assert caps["exec_rpc"]["versions"] == list(EXEC_RPC_VERSIONS) and caps["modalities"] == ["api", "internal"]
        assert remote.cancel_action("act_unknown") is False
        # calls from several threads do not queue behind one long call
        slow = make_action("http.get", url=sites.urls["structured"] + "/slow?ms=600")
        grant = policy.authorize(slow, TASK)
        t = threading.Thread(target=remote.execute, args=(slow, grant))
        t.start()
        t0 = time.time()
        assert remote.ping()["ok"] and time.time() - t0 < 0.4
        t.join()
        remote.close()
    finally:
        server.stop()
        env.close()


def test_remote_policy_round_trip_and_fail_closed(short_dir):
    policy = GrantPolicy(approve_ops={"http.post"})
    sock = os.path.join(short_dir, "auth.sock")
    server = RpcServer(sock, policy, POLICY_METHODS, name="auth").start()
    rp = RemotePolicy(sock, timeout=5.0)
    assert isinstance(rp, PolicyEngine)
    a = make_action("http.get", url="http://127.0.0.1:1/x")
    dec = rp.authorize(a, TASK, {})
    assert isinstance(dec, AuthorizationDecision) and dec.verdict == AuthVerdict.ALLOW and rp.check_grant(a, dec) is True
    assert rp.check_grant(make_action("http.get", url="http://127.0.0.1:1/y"), dec) is False
    assert rp.check_primitive(PrimitiveOp("http.get", "api", {"url": "http://127.0.0.1:1/x"}), a).verdict == AuthVerdict.ALLOW
    assert policy.primitives[-1] == ("http.get", {"url": "http://127.0.0.1:1/x"}, a.action_id)
    rp.record(a, ActionResult(a.action_id, ResultStatus.OK))
    assert policy.records == [(a.action_id, ResultStatus.OK)]
    post = make_action("http.post", url="http://127.0.0.1:1/x")
    need = rp.authorize(post, TASK, {})
    assert need.verdict == AuthVerdict.NEEDS_APPROVAL and rp.approvals()[0]["approval_id"] == need.approval_id
    ok = rp.resolve_approval(need.approval_id, True, "operator", "fine")
    assert ok.verdict == AuthVerdict.ALLOW and rp.check_grant(post, ok)
    assert rp.limits_state() == {"primitive_checks": 1} and rp.stopped() is False
    rp.revoke("/blocked", "operator")
    assert rp.check_primitive(PrimitiveOp("http.get", "api", {"url": "http://h/blocked"}), a).verdict == AuthVerdict.DENY
    rp.emergency_stop("test", "operator")
    assert rp.stopped() is True and rp.authorize(a, TASK, {}).verdict == AuthVerdict.DENY
    server.stop()                                         # the authorization service is gone: everything is refused
    assert rp.stopped() is True and rp.check_grant(a, dec) is False
    assert rp.check_primitive(PrimitiveOp("http.get", "api", {}), a).verdict == AuthVerdict.DENY
    assert rp.authorize(a, TASK, {}).verdict == AuthVerdict.DENY
    with pytest.raises(ConnectionError):
        RemotePolicy(sock, connect_timeout=0.3)


def test_service_process_with_authorization_socket_and_restart(short_dir, sites):
    policy = GrantPolicy()
    auth_sock = os.path.join(short_dir, "sock", "auth.sock")
    auth = RpcServer(auth_sock, policy, POLICY_METHODS, name="auth").start()
    args = ("--auth-socket", auth_sock, "--no-browser", "--executor", "eb_support:HttpExecutor")
    proc, ready = spawn_service(short_dir, *args)
    try:
        assert ready["ready"] is True and ready["pid"] == proc.pid and ready["versions"] == ["1.0"]
        remote = RemoteEnvironment(ready["socket"])
        assert remote.ping()["pid"] == proc.pid != os.getpid()
        d = Driver(remote, policy, TASK)
        s = sites.urls["structured"]
        res = d.ok("http.get", url=s + "/report")
        assert res.observations[0].data["title"].startswith("Report") if "title" in res.observations[0].data else True
        # the service asked *our* policy, over the socket, for the grant and for the primitive
        assert policy.primitives == [("http.get", {"url": s + "/report"}, res.action_id)]
        assert policy.records == [(res.action_id, ResultStatus.OK)]
        act = make_action("http.get", url=s + "/items")
        grant = policy.authorize(act, TASK)
        policy.deny_urls.append("/items")
        hits = sites.structured.hit_count("/items")
        denied = remote.execute(act, grant)
        assert denied.status == ResultStatus.DENIED and denied.failure_class in FailureClass.HARD
        assert sites.structured.hit_count("/items") == hits
        policy.deny_urls.clear()
        d.ok("wait", seconds=7200, reason="later")
        remote.execute(make_action("internal.note", text="x"), AuthorizationDecision(AuthVerdict.DENY))
        q = OutcomeQuery(new_id("q"), "env.actions", kind=OutcomeKind.COUNT, horizon_s=10 ** 6, issued_at=time.time())
        assert remote.measure([q])[0].status == OutcomeStatus.PENDING
        wake = remote.get_resource_state()["next_wake_at"]
        assert wake is not None
        remote.shutdown()
        assert proc.wait(timeout=20) == 0 and not os.path.exists(ready["socket"])
        # a new service process on the same run directory: accounting, scheduled wake and at-most-once survive
        proc, ready = spawn_service(short_dir, *args)
        remote = RemoteEnvironment(ready["socket"])
        rs = remote.get_resource_state()
        assert rs["actions"] == 3 and rs["denied"] == 2 and rs["next_wake_at"] == wake and remote.ping()["pid"] == proc.pid
        again = remote.execute(make_action("http.get", url=s + "/report", aid=res.action_id),
                               policy.authorize(make_action("http.get", url=s + "/report", aid=res.action_id), TASK))
        assert again.to_dict() == res.to_dict() and len(policy.records) == 3
        # the authorization service disappears: the executor process fails closed
        auth.stop()
        orphan = make_action("http.get", url=s + "/report")
        refused = remote.execute(orphan, AuthorizationDecision(AuthVerdict.ALLOW, orphan.content_hash(),
                                                               grant=policy._grant(orphan)))
        assert refused.status == ResultStatus.DENIED and refused.failure_class == FailureClass.MISSING_PERMISSION
        remote.shutdown()
        assert proc.wait(timeout=20) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        auth.stop()


def test_service_refuses_to_start_without_authorization(short_dir):
    env = dict(os.environ, PYTHONPATH=SRC)
    out = subprocess.run([sys.executable, "-m", "cam_live.harness.service", "--run-dir", short_dir, "--socket",
                          os.path.join(short_dir, "exec.sock"), "--no-browser"], capture_output=True, text=True, env=env)
    assert out.returncode == 2 and "auth-socket is required" in out.stderr
    out = subprocess.run([sys.executable, "-m", "cam_live.harness.service", "--run-dir", short_dir, "--socket",
                          os.path.join(short_dir, "exec.sock"), "--no-browser", "--diagnostic-allow-all",
                          "--provenance", "live"], capture_output=True, text=True, env=env)
    assert out.returncode == 2


@pytest.mark.browser
def test_browser_through_the_service_process(short_dir, sites):
    policy = GrantPolicy()
    auth_sock = os.path.join(short_dir, "sock", "auth.sock")
    auth = RpcServer(auth_sock, policy, POLICY_METHODS, name="auth").start()
    proc, ready = spawn_service(short_dir, "--auth-socket", auth_sock)
    try:
        remote = RemoteEnvironment(ready["socket"])
        d = Driver(remote, policy, TASK)
        # without --executor the service composes every executor of the harness (cam_live.harness.factories)
        assert remote.get_capabilities()["modalities"] == sorted(["internal", "structured_browser", "code", "visual",
                                                                  "api", "skill"])
        assert ready["compose"] == "all"
        s = sites.urls["structured"]
        o = d.ok("browser.navigate", url=s + "/items").observations[0]
        assert o.kind == "page" and o.provenance.trust == Trust.UNTRUSTED
        assert {x.name: x.value for x in o.quantities}["page.items.items.beta_gasket.unit_price"] == 0.125
        nxt = d.ok("browser.click", ref=Driver.ref(o, "link", "Report"), obs_id=o.obs_id).observations[0]
        assert nxt.data["url"] == s + "/report" and "Total quantity" in nxt.text
        stale = d.do("browser.click", ref=Driver.ref(o, "link", "Report"), obs_id=o.obs_id)
        assert stale.failure_class == FailureClass.STALE_OBSERVATION
        # reads of the page's own origin (e.g. its icon) are judged per origin and action: browser.request
        assert [p[0] for p in policy.primitives if p[0] != "browser.request"] == \
            ["browser.navigate", "browser.click", "browser.navigate"]
        from urllib.parse import urlsplit
        extra = [p[1] for p in policy.primitives if p[0] == "browser.request"]
        assert all(urlsplit(x["url"]).netloc == urlsplit(s).netloc and x["method"] == "GET" for x in extra), extra
        observed = remote.observe()
        assert [x.kind for x in observed] == ["page", "metric"] and observed[0].data["url"] == s + "/report"
        log = open(os.path.join(short_dir, "events.jsonl")).read()
        assert '"env.primitive_check"' in log and '"env.action_result"' in log
        remote.shutdown()
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        auth.stop()
