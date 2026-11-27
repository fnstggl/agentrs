"""The service's peer-uid authorization matrix and a real-socket round trip (single-uid mode; the same matrix with
real uids is in test_ea_privileged.py)."""
import os

import pytest

from cam_live import contracts as C
from cam_live.harness import rpc
from cam_live.harness.auth import PolicyService
from cam_live.harness.auth.client import AuthClient
from cam_live.harness.auth.service import EXECUTOR, LEARNER, METHODS, OPERATOR, VERIFIER, AuthService, default_roles, serve
from ea_helpers import ALLOW, DENY, LOOP, NEEDS, act, ok_result, task

UID = {LEARNER: 2001, EXECUTOR: 2002, VERIFIER: 2003, OPERATOR: 0, "sandbox": 2004, "stranger": 2999}
ROLES = {LEARNER: {2001}, EXECUTOR: {2002}, VERIFIER: {2003}, OPERATOR: {0}}
SECRET = "tok_Zx81kQ0pLm4RvT7yWb2N"
EXPECTED = {     # who, besides the operator, may call what
    LEARNER: {"authorize", "check_grant", "approvals", "stopped", "paused", "limits_state", "audit_head",
              "emergency_stop", "pause"},
    EXECUTOR: {"check_grant", "check_primitive", "record", "secret_get", "secret_has", "scrub", "scrub_obj", "approvals", "stopped",
               "paused", "limits_state", "audit_head", "emergency_stop", "pause"},
    VERIFIER: {"scrub", "scrub_obj", "approvals", "stopped", "paused", "limits_state", "audit_head", "emergency_stop",
               "pause"},
    "sandbox": set(), "stranger": set(),
}
OPERATOR_ONLY = {"resolve_approval", "reset_stop", "resume", "revoke", "reload_policy", "register_task",
                 "register_live_authorization", "put_secret", "secret_names", "audit_tail", "verify_audit",
                 "register_local_service", "unregister_local_service", "local_services"}


def test_declared_matrix_is_the_specified_one():
    for role, allowed in EXPECTED.items():
        if role in ROLES:
            assert {m for m, roles in METHODS.items() if role in roles} == allowed, role
    assert {m for m, roles in METHODS.items() if not roles} == OPERATOR_ONLY
    assert set(METHODS) == set().union(*EXPECTED.values()) | OPERATOR_ONLY
    # the whole PolicyEngine contract is served
    for name in ("authorize", "check_grant", "check_primitive", "record", "approvals", "resolve_approval",
                 "emergency_stop", "stopped", "revoke", "limits_state"):
        assert name in METHODS and callable(getattr(AuthService, name))


def _args(svc, method, ctx):
    a, d, t = ctx["action"], ctx["decision"], ctx["task"]
    return {
        "authorize": (act("wait"), t, {}), "check_grant": (a, d), "approvals": (), "stopped": (), "paused": (),
        "limits_state": (), "audit_head": (), "emergency_stop": ("drill", "x"), "pause": ("drill", "x"),
        "check_primitive": (C.PrimitiveOp("net.request", C.Modality.API, {"method": "GET", "url": LOOP}), a),
        "record": (a, ok_result(a)), "secret_get": ("site_token", a, d, LOOP + "/x"), "secret_has": ("site_token",), "scrub": ("text",),
        "scrub_obj": ({"k": "v"},), "resolve_approval": ("apr_none", True, "x", ""), "reset_stop": ("x",), "resume": ("x",),
        "revoke": ("grants", "x"), "reload_policy": ("x",), "register_task": (task(oid="other"), "x"),
        "register_live_authorization": ({"record_id": "r"}, "x"), "put_secret": ("n2", "another-secret-value", ["127.0.0.1"]),
        "secret_names": (), "audit_tail": (5,), "verify_audit": (),
        "register_local_service": ("127.0.0.1:18650", "x"), "unregister_local_service": ("127.0.0.1:18650", "x"),
        "local_services": (),
    }[method]


@pytest.mark.parametrize("who", ["learner", "executor", "verifier", "sandbox", "stranger", "operator"])
def test_every_method_for_every_caller(tmp_path, clock, who):
    svc = PolicyService(str(tmp_path / "run"), clock=clock, fsync=False)
    svc.put_secret("site_token", SECRET, ["127.0.0.1:8765"])
    current = {"uid": UID[who]}
    target = AuthService(svc, ROLES, uid_getter=lambda: current["uid"])
    t = task()
    for method in METHODS:
        # fresh granted, started action for each method (set up directly on the policy object)
        svc.stop.reset(); svc.stop.resume()
        a = act("http.get", arguments={"url": LOOP + "/x"})
        d = svc.authorize(a, t, {})
        if method not in ("check_grant",):
            assert svc.check_grant(a, d)
        allowed = who == "operator" or method in EXPECTED[who]
        try:
            getattr(target, method)(*_args(svc, method, {"action": a, "decision": d, "task": t}))
            raised = None
        except PermissionError as e:
            raised = e
        except Exception as e:  # noqa: BLE001 — permitted callers may still get a domain error (bad record, ...)
            raised = None if allowed else e
        if allowed:
            assert raised is None or "may not call" not in str(raised), (who, method)
        else:
            assert isinstance(raised, PermissionError) and "may not call" in str(raised), (who, method)
    denied = {e["data"]["method"] for e in svc.audit_tail(500) if e["event"] == "rpc_denied"}
    if who != "operator":
        assert denied == set(METHODS) - EXPECTED[who]
        assert all(e["data"]["by"] == f"uid={UID[who]}" for e in svc.audit_tail(500) if e["event"] == "rpc_denied")
    svc.close()


def test_learner_cannot_consume_a_grant_and_identity_is_attested_by_the_service(tmp_path, clock):
    svc = PolicyService(str(tmp_path / "run"), clock=clock, fsync=False)
    cur = {"uid": 2001}
    target = AuthService(svc, ROLES, uid_getter=lambda: cur["uid"])
    a, t = act("http.get", arguments={"url": LOOP}), task()
    d = target.authorize(a, t, {})
    assert target.check_grant(a, d) is True and target.check_grant(a, d, True) is True      # the learner only looks
    cur["uid"] = 2002
    assert target.check_grant(a, d) is True and target.check_grant(a, d) is False           # the executor uses it up
    cur["uid"] = 2001
    target.emergency_stop("because", "operator (claimed)")
    stop = svc.limits_state()["stop"]
    assert "uid=2001" in stop["by"] and "learner" in stop["by"]          # the audit shows who really asked
    with pytest.raises(PermissionError):
        target.reset_stop("operator (claimed)")
    cur["uid"] = 0
    assert target.reset_stop("olga") is True
    svc.close()


def test_responses_to_the_learner_are_scrubbed(tmp_path, clock):
    svc = PolicyService(str(tmp_path / "run"), clock=clock, fsync=False)
    svc.put_secret("site_token", SECRET, ["127.0.0.1:8765"])
    # force a secret into state the learner can list (cannot happen through the policy path, which denies it)
    svc.approvals_q.request(bind_hash="h", action_hash="h", task_id="t", kind="action", preview={"render": f"key={SECRET}"},
                            action={}, reasons=[f"contains {SECRET}"], policy_ids=["P"], now=clock.now())
    learner = AuthService(svc, ROLES, uid_getter=lambda: 2001)
    assert SECRET not in repr(learner.approvals()) and "[REDACTED:site_token]" in repr(learner.approvals())
    svc.close()


def test_real_socket_round_trip(tmp_path):
    """One uid, real unix socket: records survive the wire, grants are single use, errors come back typed."""
    run = str(tmp_path / "run")
    sock = os.path.join(run, "sock", "auth.sock")
    server = serve(run, sock, None, roles=default_roles(single_uid=True), fsync=False)
    try:
        c = AuthClient(sock, timeout=20.0, server_uids={os.getuid()})
        assert isinstance(c, C.PolicyEngine)
        assert c.ping()["name"] == "auth" and set(c.ping()["methods"]) == set(METHODS)
        assert oct(os.stat(sock).st_mode & 0o777) == "0o660"
        t = task(resources={"owned": ["channel:discord:900000000000000011"]})
        a = act("http.get", arguments={"url": LOOP + "/x"}, est_cost=C.Cost(seconds=1.5))
        d = c.authorize(a, t, {"page_url": LOOP})
        assert isinstance(d, C.AuthorizationDecision) and d.verdict == ALLOW and d.action_hash == a.content_hash()
        assert c.check_grant(a, d) is True and c.check_grant(a, d) is False
        p = c.check_primitive(C.PrimitiveOp("net.request", C.Modality.API, {"method": "GET", "url": LOOP + "/x"}), a)
        assert p.verdict == ALLOW
        c.put_secret("site_token", SECRET, ["127.0.0.1:8765"], ["site"], "olga")
        assert c.secrets.get("site_token", a, d, LOOP + "/x") == SECRET
        with pytest.raises(PermissionError):
            c.secrets.get("site_token", a, d, "http://localhost:9/")
        assert c.secrets.scrub(f"echo {SECRET}") == "echo [REDACTED:site_token]"
        c.record(a, C.ActionResult(a.action_id, C.ResultStatus.OK, cost=C.Cost(seconds=1.0),
                                   observations=[C.Observation("o1", 1.0, "page", text="hello", provenance=C.Provenance(uri=LOOP))]))
        need = c.authorize(act("discord.send_message", arguments={"channel_id": "900000000000000011", "content": "notes " * 10}), t, {})
        assert need.verdict == NEEDS and c.approvals()[0]["approval_id"] == need.approval_id
        ok = c.resolve_approval(need.approval_id, True, "olga", "fine")
        assert ok.verdict == ALLOW and ok.grant
        assert c.limits_state()["tasks"][t.objective_id]["used"]["external_actions"] == 1
        assert c.stopped() is False
        c.emergency_stop("drill", "olga")
        assert c.stopped() is True and c.authorize(act("wait"), t, {}).verdict == DENY
        assert c.reset_stop("olga") is True
        with pytest.raises(rpc.RemoteError) as e:
            c.revoke("nonsense", "olga")
        assert e.value.kind == "ValueError"
        assert c.verify_audit()["ok"] is True
        wrong = AuthClient(sock, server_uids={os.getuid() + 12345})      # the client checks who is listening
        with pytest.raises(PermissionError):
            wrong.stopped()
        c.close()
    finally:
        server.target.policy.close()
        server.stop()
