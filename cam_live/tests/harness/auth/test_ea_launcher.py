"""Launcher in single-uid mode: starts real services, supervises and restarts them, halts on emergency stop, stops
everything it started. The same with separate uids is in test_ea_privileged.py."""
import json
import os
import signal
import sys
import time

import pytest

from cam_live import contracts as C
from cam_live.harness import rpc
from cam_live.harness.auth import launcher
from cam_live.harness.auth.client import connect
from cam_live.harness.auth.launcher import LauncherConfig, ServiceSpec, launch, prepare_run_dir
from ea_helpers import ALLOW, DENY, LOOP, act, task

DUMMY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "svc_dummy.py")
FAST = dict(separated=False, poll_s=0.05, restart_backoff_s=0.05, stop_grace_s=2.0)


def dummy(name, role, socket="", *args, **kw):
    return ServiceSpec(name=name, role=role, socket=socket, argv=[sys.executable, DUMMY], args=list(args), **kw)


def wait_for(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        v = cond()
        if v:
            return v
        time.sleep(0.03)
    return cond()


def alive(pid):
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            return f.read().rsplit(b")", 1)[1].split()[0] != b"Z"
    except OSError:
        return False


def session_pids(sid):
    out = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                with open(f"/proc/{name}/stat", "rb") as f:
                    if int(f.read().rsplit(b")", 1)[1].split()[3]) == sid:
                        out.append(int(name))
            except (OSError, IndexError, ValueError):
                pass
    return out


def test_prepare_run_dir_layout_single_uid(tmp_path):
    run = str(tmp_path / "run")
    lay = prepare_run_dir(run, separated=False)
    for rel, mode in (("auth", "0o700"), ("vault", "0o700"), ("auth/keys", "0o700"), ("browser_profile", "0o700"),
                      ("sandbox", "0o700"), ("learner", "0o700"), ("verifier", "0o700"), ("exec", "0o700"), ("sock", "0o700")):
        assert oct(os.stat(os.path.join(run, rel)).st_mode & 0o7777) == mode, rel
    for rel in ("audit.jsonl", "verifier.jsonl", "auth/keys/grant.key", "exec/grant.key", "auth/policy.yaml"):
        assert oct(os.stat(os.path.join(run, rel)).st_mode & 0o777) == "0o600", rel
    with open(os.path.join(run, "auth/keys/grant.key"), "rb") as a, open(os.path.join(run, "exec/grant.key"), "rb") as b:
        key = a.read()
        assert key == b.read() and len(key) == 32
    assert lay["policy"] == os.path.join(run, "auth", "policy.yaml")
    prepare_run_dir(run, separated=False)                     # idempotent: the key is kept
    with open(os.path.join(run, "auth/keys/grant.key"), "rb") as a:
        assert a.read() == key
    os.symlink("/etc/passwd", os.path.join(run, "evil"))
    os.rename(os.path.join(run, "audit.jsonl"), os.path.join(run, "audit.old"))
    os.symlink(os.path.join(run, "evil"), os.path.join(run, "audit.jsonl"))
    with pytest.raises((RuntimeError, OSError)):              # never follows a planted symlink
        prepare_run_dir(run, separated=False)


def test_launch_auth_and_services_then_stop(tmp_path, monkeypatch):
    monkeypatch.setenv("EA_TEST_API_TOKEN", "would-be-inherited")
    monkeypatch.setenv("EA_TEST_DB_PASSWORD", "would-be-inherited")
    run = str(tmp_path / "run")
    h = launch(run, ["auth", dummy("worker", "sandbox", "worker.sock", "--spawn-child"), dummy("quiet", "learner")],
               dict(FAST))
    try:
        st = h.status()
        assert {n: s["state"] for n, s in st["services"].items()} == {"auth": "running", "worker": "running", "quiet": "running"}
        assert st["separated"] is False and not st["halted"]
        assert all(s["pid"] and alive(s["pid"]) and s["ready"] for s in st["services"].values())
        # the authorization service really works through its socket
        c = connect(run)
        a = act("http.get", arguments={"url": LOOP})
        d = c.authorize(a, task(), {})
        assert d.verdict == ALLOW and c.check_grant(a, d)
        info = rpc.RpcClient(st["services"]["worker"]["socket"]).call("whoami")
        assert info["pid"] == st["services"]["worker"]["pid"] and info["home"] == os.path.join(run, "sandbox")
        # the environment is built from an allowlist: nothing token-like is inherited
        allowed = set(launcher.ENV_PASSTHROUGH) | {"PYTHONPATH", "PYTHONUNBUFFERED", "HOME", "CAM_LIVE_RUN_TAG",
                                                   "CAM_LIVE_SERVICE", "LC_CTYPE"}
        assert set(info["env_keys"]) <= allowed and "CAM_LIVE_RUN_TAG" in info["env_keys"]
        assert "EA_TEST_API_TOKEN" not in info["env_keys"] and "EA_TEST_DB_PASSWORD" not in info["env_keys"]
        worker_pid = st["services"]["worker"]["pid"]
        assert wait_for(lambda: len(session_pids(worker_pid)) == 2)       # the service and the child it spawned
        pids = [s["pid"] for s in st["services"].values()] + session_pids(worker_pid)
    finally:
        final = h.stop()
    assert all(s["state"] == "stopped" and s["pid"] == 0 for s in final["services"].values())
    assert wait_for(lambda: not any(alive(p) for p in pids))              # children of the services are gone too
    assert not os.path.exists(os.path.join(run, "sock", "auth.sock")) and final["stopped"]
    assert h.stop()["stopped"]                                            # idempotent


def test_crashed_service_is_restarted_and_a_crash_loop_gives_up(tmp_path):
    run = str(tmp_path / "run")
    marker = str(tmp_path / "starts.jsonl")
    h = launch(run, ["auth", dummy("worker", "sandbox", "worker.sock"),
                     dummy("flaky", "learner", "", "--exit-after", "0.05", "--marker", marker)],
               dict(FAST, max_restarts=3, restart_window_s=60.0))
    try:
        first = h.status()["services"]["worker"]["pid"]
        assert first > 1
        os.kill(first, signal.SIGKILL)                                    # crash
        st = wait_for(lambda: (lambda s: s if s["state"] == "running" and s["pid"] not in (0, first) else None)(
            h.status()["services"]["worker"]))
        assert st and st["restarts"] == 1 and st["last_exit"] == -9
        assert rpc.RpcClient(st["socket"]).call("whoami")["pid"] == st["pid"]       # the restarted one serves
        rpc.RpcClient(st["socket"]).call("crash")                                   # crash from inside, exit code 7
        st2 = wait_for(lambda: (lambda s: s if s["restarts"] == 2 and s["state"] == "running" else None)(
            h.status()["services"]["worker"]))
        assert st2 and st2["last_exit"] == 7
        flaky = wait_for(lambda: (lambda s: s if s["state"] == "failed" else None)(h.status()["services"]["flaky"]), 20.0)
        assert flaky and flaky["restarts"] == 3 and "crashed 4 times" in flaky["reason"]
        with open(marker) as f:
            assert len(f.readlines()) == 4                                # started once + three restarts, then left alone
        assert h.status()["services"]["auth"]["restarts"] == 0
        kinds = [e["kind"] for e in h.status()["events"]]
        assert "exit" in kinds and "gave_up" in kinds
    finally:
        h.stop()


def test_authorization_service_is_restarted_with_its_state(tmp_path):
    run = str(tmp_path / "run")
    h = launch(run, ["auth"], dict(FAST))
    try:
        c = connect(run)
        t = task(limits=C.Limits(max_decisions=3))
        a = act("wait")
        d = c.authorize(a, t, {})
        pid = h.status()["services"]["auth"]["pid"]
        assert pid > 1
        os.kill(pid, signal.SIGKILL)
        assert wait_for(lambda: (lambda s: s["state"] == "running" and s["restarts"] == 1 and s["pid"] not in (0, pid))(
            h.status()["services"]["auth"]))
        c2 = connect(run)
        assert c2.check_grant(a, d) is True                               # the grant issued before the crash is known
        assert c2.limits_state()["tasks"][t.objective_id]["used"]["decisions"] == 1
        assert c2.verify_audit()["ok"]
    finally:
        h.stop()


def test_emergency_stop_halts_everything_but_the_authorization_service(tmp_path):
    run = str(tmp_path / "run")
    h = launch(run, ["auth", dummy("worker", "sandbox", "worker.sock", "--spawn-child", "--detached-child"),
                     dummy("thinker", "learner")], dict(FAST))
    try:
        st = h.status()["services"]
        worker_pid = st["worker"]["pid"]
        tagged = wait_for(lambda: (lambda p: p if len(p) >= 3 else None)(h._tree(h._svcs["worker"])))
        assert tagged and len(tagged) >= 3                                # service, child, detached grandchild
        c = connect(run)
        c.emergency_stop("operator pressed stop", "olga")
        halted = wait_for(lambda: (lambda s: s if s["halted"] and all(
            v["state"] == "halted" for n, v in s["services"].items() if n != "auth") else None)(h.status()))
        assert halted, h.status()
        assert halted["services"]["auth"]["state"] == "running"           # stays up: answers "stopped", keeps the audit
        assert wait_for(lambda: not any(alive(p) for p in tagged + [st["thinker"]["pid"]]))
        assert c.stopped() and c.authorize(act("wait"), task(), {}).verdict == DENY
        time.sleep(0.4)                                                   # nothing is restarted while stopped
        assert all(v["pid"] == 0 for n, v in h.status()["services"].items() if n != "auth")
        with pytest.raises(RuntimeError):
            h.resume()                                                    # not while the stop is engaged
        assert c.reset_stop("olga") is True
        resumed = h.resume()
        assert all(v["state"] == "running" for v in resumed["services"].values()) and not resumed["halted"]
        assert resumed["services"]["worker"]["pid"] not in (0, worker_pid)
    finally:
        h.stop()


def test_launch_with_stop_already_engaged_starts_only_the_authorization_service(tmp_path):
    run = str(tmp_path / "run")
    h = launch(run, ["auth"], dict(FAST))
    connect(run).emergency_stop("before the restart", "olga")
    h.stop()
    h2 = launch(run, ["auth", dummy("worker", "sandbox", "worker.sock")], dict(FAST))      # e.g. after a reboot
    try:
        st = h2.status()
        assert st["halted"] and st["services"]["auth"]["state"] == "running" and st["services"]["worker"]["state"] == "halted"
        assert st["services"]["worker"]["pid"] == 0
        assert connect(run).stopped() is True
    finally:
        h2.stop()


def test_unavailable_modules_are_reported_not_faked(tmp_path):
    run = str(tmp_path / "run")
    h = launch(run, ["auth", ServiceSpec(name="ghost", module="cam_live.harness.auth.no_such_module", role="learner")], dict(FAST))
    try:
        g = h.status()["services"]["ghost"]
        assert g["state"] == "unavailable" and "not importable" in g["reason"] and g["pid"] == 0
        assert h.status()["services"]["auth"]["state"] == "running"
    finally:
        h.stop()
    with pytest.raises(KeyError):
        launch(str(tmp_path / "r2"), ["no-such-service"], dict(FAST))
    with pytest.raises(ValueError):
        LauncherConfig.coerce({"separated": False, "typo_setting": 1})


def test_nothing_runs_without_the_authorization_service(tmp_path):
    run = str(tmp_path / "run")
    broken_auth = ServiceSpec(name="auth", role="auth", socket="auth.sock", argv=[sys.executable, "-c", "import sys; sys.exit(4)"],
                              ready_timeout_s=3.0, stop_on_estop=False)
    marker = str(tmp_path / "started")
    with pytest.raises(RuntimeError) as e:
        launch(run, [broken_auth, dummy("worker", "sandbox", "", "--marker", marker)], dict(FAST))
    assert "authorization service did not start" in str(e.value)
    assert not os.path.exists(marker)                                     # the other service was never started


def test_launcher_main_runs_until_signalled(tmp_path):
    import subprocess
    from ea_helpers import SRC
    run = str(tmp_path / "run")
    p = subprocess.Popen([sys.executable, "-m", "cam_live.harness.auth.launcher", "--run-dir", run, "--services", "auth",
                          "--no-separation"], env=dict(os.environ, PYTHONPATH=SRC), stdout=subprocess.PIPE, text=True)
    try:
        assert wait_for(lambda: os.path.exists(os.path.join(run, "sock", "auth.sock")), 20.0)
        assert connect(run).stopped() is False
    finally:
        p.send_signal(signal.SIGTERM)
        out, _ = p.communicate(timeout=20)
    assert p.returncode == 0 and json.loads(out)["services"]["auth"]["state"] == "running"
    assert not os.path.exists(os.path.join(run, "sock", "auth.sock"))


def test_authorization_service_confines_its_own_writes_with_landlock(tmp_path):
    """Kernel-enforced: after start-up the service process can write only below its own directories."""
    import subprocess
    from cam_live.harness.auth import landlock as ll
    from ea_helpers import SRC
    if ll.abi_version() < 1:
        pytest.skip("kernel without Landlock")
    inside, outside = tmp_path / "inside", tmp_path / "outside"
    inside.mkdir(); outside.mkdir()
    (inside / "log").write_text("")
    (outside / "existing").write_text("keep")
    code = (
        "import errno, json, os, sys\n"
        "from cam_live.harness.auth import landlock as ll\n"
        "inside, outside = sys.argv[1], sys.argv[2]\n"
        "ok = ll.restrict_writes([inside])\n"
        "out = {'applied': ok}\n"
        "def t(name, fn):\n"
        "    try:\n        fn(); out[name] = 'ok'\n    except OSError as e:\n        out[name] = errno.errorcode[e.errno]\n"
        "t('write_inside', lambda: open(os.path.join(inside, 'new'), 'w').write('x'))\n"
        "t('append_inside', lambda: open(os.path.join(inside, 'log'), 'a').write('x'))\n"
        "t('unlink_inside', lambda: os.unlink(os.path.join(inside, 'new')))\n"
        "t('read_outside', lambda: open(os.path.join(outside, 'existing')).read())\n"
        "t('create_outside', lambda: open(os.path.join(outside, 'new'), 'w'))\n"
        "t('overwrite_outside', lambda: open(os.path.join(outside, 'existing'), 'w'))\n"
        "t('unlink_outside', lambda: os.unlink(os.path.join(outside, 'existing')))\n"
        "t('mkdir_outside', lambda: os.mkdir(os.path.join(outside, 'd')))\n"
        "t('child_create_outside', lambda: (_ for _ in ()).throw(OSError(errno.EACCES, 'x')) if os.system("
        "'touch ' + os.path.join(outside, 'by-child') + ' 2>/dev/null') else None)\n"
        "print(json.dumps(out))\n")
    cp = subprocess.run([sys.executable, "-c", code, str(inside), str(outside)], env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"},
                        capture_output=True, text=True, timeout=30)
    assert cp.returncode == 0, cp.stderr
    out = json.loads(cp.stdout)
    assert out == {"applied": True, "write_inside": "ok", "append_inside": "ok", "unlink_inside": "ok", "read_outside": "ok",
                   "create_outside": "EACCES", "overwrite_outside": "EACCES", "unlink_outside": "EACCES",
                   "mkdir_outside": "EACCES", "child_create_outside": "EACCES"}
    assert (outside / "existing").read_text() == "keep" and not (outside / "by-child").exists()
    # the launched service applies it to itself and keeps working (state, vault, audit, stop flag)
    run = str(tmp_path / "run")
    h = launch(run, ["auth"], dict(FAST))
    try:
        c = connect(run)
        conf = [e for e in c.audit_tail(20) if e["event"] == "service_confinement"]
        assert conf and conf[-1]["data"]["landlock"] == "enforced"
        c.put_secret("site_token", "tok_Zx81kQ0pLm4RvT7yWb2N", ["127.0.0.1"], [], "olga")
        a = act("http.get", arguments={"url": LOOP})
        d = c.authorize(a, task(), {})
        assert d.verdict == ALLOW and c.check_grant(a, d)
        c.emergency_stop("drill", "olga")
        assert c.stopped() and c.reset_stop("olga") and c.verify_audit()["ok"]
        with open(os.path.join(run, "logs", "auth.log")) as f:
            assert "landlock enforced" in f.read()
    finally:
        h.stop()
