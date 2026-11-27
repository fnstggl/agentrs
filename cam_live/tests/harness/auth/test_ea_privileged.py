"""Separated mode with real unix users: the kernel, not the policy, keeps the learner and the sandbox away from the
vault, the audit log and the verifier's files; the service authorizes callers by their real uid; the launcher
starts every service under its own uid without privileges.

    pytest -m privileged cam_live/tests/harness/auth/test_ea_privileged.py        (as root)
"""
import base64
import errno
import grp
import hashlib
import json
import os
import pwd
import signal
import stat
import sys
import time

import pytest

from cam_live import contracts as C
from cam_live.harness import SERVICE_GROUP, SERVICE_USERS, rpc
from cam_live.harness.auth.client import connect
from cam_live.harness.auth.launcher import ServiceSpec, launch, prepare_run_dir
from cam_live.harness.auth.service import METHODS
from ea_helpers import ALLOW, DENY, LOOP, act, jout, run_as, task
from test_ea_launcher import DUMMY, alive, wait_for
from test_ea_service import EXPECTED, OPERATOR_ONLY

pytestmark = pytest.mark.privileged
CANARY = "CANARY-51c0e7aa94d2b6f3-Xw"
FAST = dict(separated=True, poll_s=0.05, restart_backoff_s=0.05, stop_grace_s=2.0)
U = SERVICE_USERS

TRY_OPEN = r"""
import errno, json, os, sys
out = {}
for spec in sys.argv[1:]:
    how, path = spec.split("=", 1)
    try:
        if how == "read":
            with open(path, "rb") as f: f.read(1)
        elif how == "list":
            os.listdir(path)
        elif how == "write":
            with open(path, "ab") as f: f.write(b"x")
        elif how == "create":
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600); os.close(fd)
        elif how == "unlink":
            os.unlink(path)
        elif how == "rename":
            os.rename(path, path + ".moved")
        elif how == "replace":
            fd = os.open(path + ".new", os.O_WRONLY | os.O_CREAT, 0o600); os.close(fd)
            try:
                os.rename(path + ".new", path)
            finally:
                if os.path.exists(path + ".new"): os.unlink(path + ".new")
        out[spec] = "ok"
    except OSError as e:
        out[spec] = errno.errorcode.get(e.errno, str(e.errno))
print(json.dumps(out))
"""

CALL_ALL = r"""
import json, sys
from cam_live import contracts as C
from cam_live.harness import rpc
from cam_live.harness.auth.client import connect
run = sys.argv[1]
a, d, t = rpc.from_wire(json.loads(sys.argv[2]))
calls = {
    "authorize": (C.ActionSpec(C.new_id("a"), "wait", "internal"), t, {}), "check_grant": (a, d), "approvals": (),
    "stopped": (), "paused": (), "limits_state": (), "audit_head": (),
    "check_primitive": (C.PrimitiveOp("net.request", "api", {"method": "GET", "url": "http://127.0.0.1:8765/x"}), a),
    "secret_get": ("site_token", a, d, "http://127.0.0.1:8765/x"), "record": (a, C.ActionResult(a.action_id, "ok")),
    "emergency_stop": ("drill", "x"), "pause": ("drill", "x"),
    "scrub": ("text",), "scrub_obj": ({"k": "v"},), "resolve_approval": ("apr_none", True, "x", ""), "reset_stop": ("x",),
    "resume": ("x",), "revoke": ("secret:does-not-exist", "x"), "reload_policy": ("x",),
    "register_task": (C.TaskSpec("other-task", "o"), "x"), "register_live_authorization": ({"record_id": "r"}, "x"),
    "put_secret": ("n2", "another-secret-value", ["127.0.0.1"]), "secret_names": (), "audit_tail": (5,), "verify_audit": (),
    "register_local_service": ("127.0.0.1:18650", "x"), "unregister_local_service": ("127.0.0.1:18650", "x"),
    "local_services": (),
}
out, raw = {}, []
try:
    c = connect(run)
    c.ping()
except Exception as e:
    print(json.dumps({"connect": type(e).__name__}))
    sys.exit(0)
for m, args in calls.items():
    try:
        raw.append(json.dumps(rpc.to_wire(c._rpc.call(m, *args)), default=str))
        out[m] = "ok"
    except rpc.RemoteError as e:
        raw.append(e.message)
        out[m] = "denied" if (e.kind == "PermissionError" and "may not call" in e.message) else "ok:" + e.kind
    except Exception as e:
        out[m] = "fail:" + type(e).__name__
print(json.dumps({"calls": out, "raw": "\n".join(raw)}))
"""


def owner(path):
    st = os.lstat(path)
    return pwd.getpwuid(st.st_uid).pw_name, grp.getgrgid(st.st_gid).gr_name, oct(stat.S_IMODE(st.st_mode))


def forms(v):
    raw = v.encode()
    return [v, raw.hex(), base64.b64encode(raw).decode(), v[::-1]]


@pytest.fixture
def separated(shared_dir, service_users):
    run = os.path.join(shared_dir, "run")
    h = launch(run, ["auth"], dict(FAST))
    yield run, h
    h.stop()


def test_service_users_exist_and_are_distinct(service_users):
    assert set(service_users) == set(U) and len(set(service_users.values())) == len(U)
    assert all(uid != 0 for uid in service_users.values())
    members = set(grp.getgrnam(SERVICE_GROUP).gr_mem)
    assert set(U.values()) <= members


def test_run_dir_ownership_and_modes(shared_dir, service_users):
    run = os.path.join(shared_dir, "run")
    lay = prepare_run_dir(run, separated=True)
    assert owner(run)[0] == "root" and owner(run)[2] == "0o1771"
    assert owner(os.path.join(run, "sock")) == ("root", SERVICE_GROUP, "0o1770")
    for rel, user, mode in (("auth", U["auth"], "0o700"), ("auth/keys", U["auth"], "0o700"), ("vault", U["auth"], "0o700"),
                            ("audit.jsonl", U["auth"], "0o600"), ("auth/keys/grant.key", U["auth"], "0o600"),
                            ("auth/policy.yaml", U["auth"], "0o600"), ("verifier", U["verifier"], "0o700"),
                            ("verifier.jsonl", U["verifier"], "0o600"), ("browser_profile", U["browser"], "0o700"),
                            ("exec", U["browser"], "0o700"), ("exec/grant.key", U["browser"], "0o600"),
                            ("learner", U["learner"], "0o700"), ("sandbox", U["sandbox"], "0o2770")):
        o = owner(os.path.join(run, rel))
        assert (o[0], o[2]) == (user, mode), (rel, o)
    assert owner(os.path.join(run, "sandbox"))[1] == grp.getgrgid(pwd.getpwnam(U["browser"]).pw_gid).gr_name
    assert lay["uids"]["auth"] == service_users["auth"] and lay["executor_uid"] == service_users["browser"]


def test_learner_and_sandbox_uids_cannot_read_vault_audit_or_verifier_files(separated, service_users):
    run, h = separated
    root = connect(run)
    root.put_secret("site_token", CANARY, ["127.0.0.1:8765"], [], "operator")
    root.authorize(act("wait"), task(), {})
    fd = os.open(os.path.join(run, "verifier.jsonl"), os.O_WRONLY | os.O_APPEND)      # (no O_CREAT: see launcher._mk)
    os.write(fd, b'{"outcome": "signed"}\n')
    os.close(fd)
    with open(os.path.join(run, "verifier", "verifier.key"), "w") as f:
        f.write("verifier-signing-key")
    os.chown(os.path.join(run, "verifier", "verifier.key"), service_users["verifier"], -1)
    os.chmod(os.path.join(run, "verifier", "verifier.key"), 0o600)
    with open(os.path.join(run, "browser_profile", "Cookies"), "w") as f:
        f.write("session=abc")
    os.chown(os.path.join(run, "browser_profile", "Cookies"), service_users["browser"], -1)
    os.chmod(os.path.join(run, "browser_profile", "Cookies"), 0o600)
    P = lambda *p: os.path.join(run, *p)                                   # noqa: E731
    secret_file = P("vault", "site_token.secret.json")
    protected = [f"read={secret_file}", f"list={P('vault')}", f"read={P('audit.jsonl')}", f"write={P('audit.jsonl')}",
                 f"read={P('auth', 'state.sqlite')}", f"read={P('auth', 'keys', 'grant.key')}", f"read={P('auth', 'audit.head')}",
                 f"read={P('auth', 'policy.yaml')}", f"write={P('auth', 'policy.yaml')}", f"create={P('auth', 'STOP.x')}",
                 f"create={P('vault', 'mine.secret.json')}", f"read={P('verifier.jsonl')}", f"write={P('verifier.jsonl')}",
                 f"read={P('verifier', 'verifier.key')}", f"list={P('verifier')}", f"read={P('browser_profile', 'Cookies')}",
                 f"read={P('exec', 'grant.key')}"]
    for user in (U["learner"], U["sandbox"]):
        res = jout(run_as(user, TRY_OPEN, *protected))
        assert res == {k: "EACCES" for k in protected}, (user, res)
        # neither can they remove or replace the logs or the service socket (root-owned sticky directories)
        res = jout(run_as(user, TRY_OPEN, f"unlink={P('audit.jsonl')}", f"rename={P('audit.jsonl')}", f"replace={P('audit.jsonl')}",
                          f"unlink={P('verifier.jsonl')}", f"rename={P('vault')}", f"unlink={P('sock', 'auth.sock')}",
                          f"replace={P('sock', 'auth.sock')}", f"rename={P('auth')}"))
        assert set(res.values()) <= {"EPERM", "EACCES"}, (user, res)
    assert os.path.exists(P("audit.jsonl")) and os.path.exists(secret_file) and os.path.exists(P("sock", "auth.sock"))
    # positive controls: the right uid can, and the learner can still write its own logs in the run directory
    assert jout(run_as(U["auth"], TRY_OPEN, f"read={secret_file}", f"read={P('audit.jsonl')}")) == {
        f"read={secret_file}": "ok", f"read={P('audit.jsonl')}": "ok"}
    assert jout(run_as(U["verifier"], TRY_OPEN, f"read={P('verifier.jsonl')}", f"read={secret_file}")) == {
        f"read={P('verifier.jsonl')}": "ok", f"read={secret_file}": "EACCES"}
    assert jout(run_as(U["browser"], TRY_OPEN, f"read={P('exec', 'grant.key')}", f"read={secret_file}", f"read={P('audit.jsonl')}")) == {
        f"read={P('exec', 'grant.key')}": "ok", f"read={secret_file}": "EACCES", f"read={P('audit.jsonl')}": "EACCES"}
    assert jout(run_as(U["learner"], TRY_OPEN, f"create={P('events.jsonl')}", f"create={P('learner', 'state.json')}")) == {
        f"create={P('events.jsonl')}": "ok", f"create={P('learner', 'state.json')}": "ok"}
    # the service process itself is out of reach: no environment, no memory
    pid = h.status()["services"]["auth"]["pid"]
    res = jout(run_as(U["learner"], TRY_OPEN, f"read=/proc/{pid}/environ", f"read=/proc/{pid}/mem"))
    assert set(res.values()) <= {"EACCES", "EPERM"}
    assert root.verify_audit()["ok"]


def test_rpc_peer_uid_authorization_matrix(separated, service_users):
    run, h = separated
    assert h.status()["services"]["auth"]["uid"] == service_users["auth"]
    root = connect(run)
    root.put_secret("site_token", CANARY, ["127.0.0.1:8765"], [], "operator")
    t = task()
    who = {"learner": U["learner"], "executor": U["browser"], "verifier": U["verifier"], "sandbox": U["sandbox"]}
    for role, user in who.items():
        root.reset_stop("test"); root.resume("test")
        a = act("http.get", arguments={"url": LOOP + "/x"})
        d = root.authorize(a, t, {})
        assert d.verdict == ALLOW
        if role != "executor":
            assert root.check_grant(a, d)                       # give the non-executor callers a running action to refer to
        res = jout(run_as(user, CALL_ALL, run, json.dumps(rpc.to_wire([a, d, t]))))
        got = {m for m, r in res["calls"].items() if r.startswith("ok")}
        denied = {m for m, r in res["calls"].items() if r == "denied"}
        assert got == EXPECTED[role], (role, res["calls"])
        assert denied == set(METHODS) - EXPECTED[role], (role, res["calls"])
        assert got.isdisjoint(OPERATOR_ONLY)
        if role == "executor":
            assert res["calls"]["secret_get"] == "ok" and CANARY in res["raw"]       # the executor is given the secret ...
        else:
            assert all(f not in res["raw"] for f in forms(CANARY)), role             # ... nobody else ever sees it
    root.reset_stop("test"); root.resume("test")
    # a uid outside the service group cannot even reach the socket
    out = jout(run_as("nobody", CALL_ALL, run, json.dumps(rpc.to_wire([a, d, t])), extra_groups=()))
    assert out == {"connect": "PermissionError"}
    # root is the operator
    a = act("http.get", arguments={"url": LOOP + "/y"})
    d = root.authorize(a, t, {})
    assert root.check_grant(a, d)
    res = jout(run_as("root", CALL_ALL, run, json.dumps(rpc.to_wire([a, d, t])), extra_groups=()))
    assert all(r.startswith("ok") for r in res["calls"].values()), res["calls"]
    root.reset_stop("test"); root.resume("test")
    tail = root.audit_tail(400)
    uids = {e["data"]["by"] for e in tail if e["event"] == "rpc_denied"}
    assert {f"uid={service_users[r]}" for r in ("learner", "browser", "verifier", "sandbox")} <= uids
    stops = [e["data"]["operator"] for e in tail if e["event"] == "emergency_stop"]
    assert any(f"uid={service_users['learner']}" in s for s in stops)       # recorded with the real caller, not "x"


LEARNER_FLOW = r"""
import json, sys
from cam_live import contracts as C
from cam_live.harness import rpc
from cam_live.harness.auth.client import connect
run, canary_forms = sys.argv[1], json.loads(sys.argv[2])
c = connect(run)
t = C.TaskSpec("canary-task", "an objective", mode="staging", resources={"owned": ["channel:discord:900000000000000011"]})
raw = []
def call(m, *a):
    try:
        r = c._rpc.call(m, *a)
        raw.append(json.dumps(rpc.to_wire(r), default=str)); return r
    except rpc.RemoteError as e:
        raw.append(e.kind + ": " + e.message); return e
a = C.ActionSpec("act-login", "http.post", "api", arguments={"url": "http://127.0.0.1:8765/api/login",
                 "headers": {"Authorization": "secret:site_token"}})
d = call("authorize", a, t, {})
for f in canary_forms:                                   # guesses and exfiltration attempts
    call("authorize", C.ActionSpec(C.new_id("a"), "http.post", "api", arguments={"url": "http://127.0.0.1:8765/p", "body": f}), t, {})
call("authorize", C.ActionSpec(C.new_id("a"), "discord.send_message", "api",
     arguments={"channel_id": "900000000000000011", "content": "status update " * 5}), t, {})
for m, args in (("approvals", ()), ("limits_state", ()), ("audit_head", ()), ("stopped", ()), ("paused", ()),
                ("check_grant", (a, d)), ("secret_get", ("site_token", a, d, "http://127.0.0.1:8765/api/login")),
                ("secret_names", ()), ("audit_tail", (100,)), ("scrub", ("x",)), ("scrub_obj", ({"a": 1},))):
    call(m, *args)
print(json.dumps({"raw": "\n".join(raw), "wire": rpc.to_wire([a, d, t])}))
"""

EXECUTOR_FLOW = r"""
import hashlib, json, sys
from cam_live import contracts as C
from cam_live.harness import rpc
from cam_live.harness.auth.client import connect
run = sys.argv[1]
a, d, t = rpc.from_wire(json.loads(sys.argv[2]))
c = connect(run)
assert c.check_grant(a, d) is True
ctx = C.ExecContext(clock=C.WallClock(), check=c.check_primitive, obs_sink=None, run_dir=run, secrets=c.secrets)
op = C.PrimitiveOp("net.request", "api", {"method": "POST", "url": "http://127.0.0.1:8765/api/login",
                                           "headers": {"Authorization": "secret:site_token"}})
assert ctx.check(op, a).verdict == "allow"
token = ctx.secrets.get("site_token", a, d, "http://127.0.0.1:8765/api/login")
echoed = "welcome; debug echo: " + token + " / " + token.encode().hex() + " / " + token[::-1]
obs = ctx.secrets.scrub_record(C.Observation("obs-1", 1.0, "tool_result", text=echoed, data={"cookie": token},
                                             provenance=C.Provenance(uri="http://127.0.0.1:8765/api/login?t=" + token)))
res = C.ActionResult(a.action_id, "ok", observations=[obs], ops=[op])
c.record(a, res)
try:
    ctx.secrets.get("site_token", a, d, "http://localhost:8765/api/login"); wrong = "released"
except PermissionError:
    wrong = "refused"
print(json.dumps({"token_sha256": hashlib.sha256(token.encode()).hexdigest(), "result": rpc.to_wire(res), "wrong_destination": wrong}))
"""


def test_canary_secret_reaches_only_the_executor(separated, service_users):
    """Real uids, real sockets: the learner authorizes, the executor obtains the secret for that granted action and
    scrubs what comes back; the canary is found in nothing the learner or the sandbox uid can obtain."""
    run, h = separated
    root = connect(run)
    root.put_secret("site_token", CANARY, ["127.0.0.1:8765"], [], "operator")
    all_forms = forms(CANARY)
    learner = jout(run_as(U["learner"], LEARNER_FLOW, run, json.dumps(all_forms)))
    ex = jout(run_as(U["browser"], EXECUTOR_FLOW, run, json.dumps(learner["wire"])))
    assert ex["token_sha256"] == hashlib.sha256(CANARY.encode()).hexdigest()         # the executor really had it
    assert ex["wrong_destination"] == "refused"
    learner2 = jout(run_as(U["learner"], LEARNER_FLOW, run, json.dumps(all_forms)))   # and again after execution
    sandbox = run_as(U["sandbox"], CALL_ALL, run, json.dumps(learner["wire"]))
    sb = jout(sandbox)
    assert all(r == "denied" for r in sb["calls"].values())
    sb_files = jout(run_as(U["sandbox"], TRY_OPEN, f"read={run}/vault/site_token.secret.json", f"list={run}/vault",
                           f"read={run}/audit.jsonl", f"read={run}/logs/auth.log"))
    assert set(sb_files.values()) == {"EACCES"}
    visible = [learner["raw"], learner2["raw"], json.dumps(ex["result"]), sb["raw"], sandbox.stdout, sandbox.stderr]
    for blob in visible:
        for f in all_forms:
            assert f not in blob
    assert "[REDACTED:site_token]" in json.dumps(ex["result"])                       # the observation was scrubbed, not dropped
    assert "X-CREDENTIAL-LEAK" in learner["raw"] and "may not call secret_get" in learner["raw"]
    # server side: audit and service log never contain the value; the access is recorded by name with the caller's uid
    for path in (os.path.join(run, "audit.jsonl"), os.path.join(run, "logs", "auth.log")):
        with open(path) as f:
            text = f.read()
        assert all(x not in text for x in all_forms), path
    acc = [e for e in root.audit_tail(500) if e["event"] in ("secret_access", "secret_denied")]
    assert [e["data"]["by"] for e in acc if e["event"] == "secret_access"] == [f"uid={service_users['browser']}"]
    assert any(e["event"] == "secret_denied" for e in acc) and all(e["data"]["name"] == "site_token" for e in acc)
    holders = []
    for dirpath, _d, files in os.walk(run):
        for fn in files:
            try:
                with open(os.path.join(dirpath, fn), "rb") as f:
                    if CANARY.encode() in f.read():
                        holders.append(os.path.relpath(os.path.join(dirpath, fn), run))
            except OSError:
                pass
    assert holders == ["vault/site_token.secret.json"]
    assert root.verify_audit()["ok"]


def test_launcher_starts_supervises_and_stops_services_under_their_own_uids(shared_dir, service_users, monkeypatch):
    monkeypatch.setenv("EA_TEST_API_TOKEN", "would-be-inherited")
    run = os.path.join(shared_dir, "run")
    specs = ["auth",
             ServiceSpec(name="worker", role="sandbox", socket="worker.sock", argv=[sys.executable, DUMMY], args=["--spawn-child"]),
             ServiceSpec(name="thinker", role="learner", socket="thinker.sock", argv=[sys.executable, DUMMY]),
             ServiceSpec(name="exec-like", role="browser", socket="exec.sock", argv=[sys.executable, DUMMY])]
    h = launch(run, specs, dict(FAST))
    try:
        st = h.status()
        assert st["separated"] and all(s["state"] == "running" for s in st["services"].values()), st
        want = {"auth": "auth", "worker": "sandbox", "thinker": "learner", "exec-like": "browser"}
        shared_gid = grp.getgrnam(SERVICE_GROUP).gr_gid
        for name, role in want.items():
            s = st["services"][name]
            assert s["uid"] == service_users[role]
            assert os.stat(f"/proc/{s['pid']}").st_uid == service_users[role]         # the kernel agrees
            with open(f"/proc/{s['pid']}/status") as f:
                status = f.read()
            assert "NoNewPrivs:\t1" in status and "CapEff:\t0000000000000000" in status and "CapPrm:\t0000000000000000" in status
            assert f"Uid:\t{service_users[role]}\t{service_users[role]}\t{service_users[role]}" in status
        for name in ("worker", "thinker", "exec-like"):
            info = rpc.RpcClient(st["services"][name]["socket"]).call("whoami")
            assert info["uid"] == service_users[want[name]] and shared_gid in info["groups"] and 0 not in info["groups"]
            assert "EA_TEST_API_TOKEN" not in info["env_keys"] and info["no_new_privs"] == "1"
            assert not any(k in info["env_keys"] for k in ("OPEN_BUTTON_TOKEN", "CONTAINER_API_KEY", "WEB_PASSWORD", "JUPYTER_TOKEN"))
        connect(run).put_secret("site_token", CANARY, ["127.0.0.1:8765"], [], "operator")
        secret_file = os.path.join(run, "vault", "site_token.secret.json")
        for name in ("worker", "thinker", "exec-like"):                              # a running service cannot open the vault
            c = rpc.RpcClient(st["services"][name]["socket"])
            assert c.call("can_read", secret_file) == {"ok": False, "errno": errno.EACCES}
            assert c.call("can_read", os.path.join(run, "audit.jsonl")) == {"ok": False, "errno": errno.EACCES}
        assert rpc.RpcClient(st["services"]["exec-like"]["socket"]).call("can_read", os.path.join(run, "exec", "grant.key")) == {"ok": True}
        # crash -> restarted under the same uid
        old = st["services"]["worker"]["pid"]
        assert old > 1
        os.kill(old, signal.SIGKILL)
        new = wait_for(lambda: (lambda s: s if s["state"] == "running" and s["pid"] not in (0, old) else None)(h.status()["services"]["worker"]))
        assert new and new["restarts"] == 1 and os.stat(f"/proc/{new['pid']}").st_uid == service_users["sandbox"]
        # emergency stop from the operator: everything but the authorization service goes away, children included
        tree = wait_for(lambda: (lambda p: p if len(p) >= 2 else None)(h._tree(h._svcs["worker"])))
        victims = tree + [h.status()["services"][n]["pid"] for n in ("thinker", "exec-like")]
        connect(run).emergency_stop("drill", "operator")
        halted = wait_for(lambda: (lambda s: s if all(v["state"] == "halted" for n, v in s["services"].items() if n != "auth") else None)(h.status()))
        assert halted and halted["services"]["auth"]["state"] == "running"
        assert wait_for(lambda: not any(alive(p) for p in victims))
        assert connect(run).stopped()
    finally:
        final = h.stop()
    assert all(s["pid"] == 0 for s in final["services"].values())
    left = [p for p in os.listdir("/proc") if p.isdigit() and _tag_of(p) == h.tag]
    assert left == []
    # relaunch on the same run directory (e.g. after a reboot): state, audit chain and the stop all persisted
    h2 = launch(run, ["auth", specs[1]], dict(FAST))
    try:
        st2 = h2.status()
        assert st2["halted"] and st2["services"]["auth"]["state"] == "running" and st2["services"]["worker"]["state"] == "halted"
        c = connect(run)
        assert c.stopped() and c.verify_audit()["ok"] and [n["name"] for n in c.secret_names()] == ["site_token"]
        conf = [e for e in c.audit_tail(10) if e["event"] == "service_confinement"]
        assert conf[-1]["data"]["landlock"] == "enforced"
        assert c.reset_stop("operator") is True
        assert h2.resume()["services"]["worker"]["state"] == "running"
    finally:
        h2.stop()


def _tag_of(pid):
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            for e in f.read().split(b"\0"):
                if e.startswith(b"CAM_LIVE_RUN_TAG="):
                    return e.split(b"=", 1)[1].decode()
    except OSError:
        pass
    return ""


def test_service_uids_cannot_regain_privileges(service_users):
    code = ("import os, json\nout = {}\n"
            "for name, fn in (('setuid0', lambda: os.setuid(0)), ('setgid0', lambda: os.setgid(0)), ('chown', lambda: os.chown('/tmp', 0, 0))):\n"
            "    try:\n        fn(); out[name] = 'ok'\n    except OSError as e:\n        out[name] = e.errno\n"
            "print(json.dumps(out))\n")
    for user in U.values():
        assert jout(run_as(user, code)) == {"setuid0": errno.EPERM, "setgid0": errno.EPERM, "chown": errno.EPERM}, user
