"""Regression tests of the E-fix workstream for the composed environment (M6): ``harness.factories`` builds every
executor from the task, and the environment service as ``cam-live run`` starts it registers all of them, scrubs
through the authorization service's broker and — separated, under the browser uid — starts Chromium (privileged
test, run with ``-m privileged``)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import pytest

from cam_live.contracts import AuthVerdict, Modality, ResultStatus, TaskSpec
from cam_live.harness.factories import ALL_MODALITIES, all_executors, default_sandbox_workspace
from cam_live.harness.rpc import RpcServer
from cam_live.harness.service import POLICY_METHODS, RemoteEnvironment, build_environment, find_task_file

from e_fix_support import HERE, SRC, FourArgBroker
from eb_support import Driver, GrantPolicy, make_action

SECRET = "k3y-e-fix-FACTORY-77ab31"


def task(tmp_path, **res):
    site = tmp_path / "site"
    site.mkdir(exist_ok=True)
    resources = {"owned_sites": [{"name": "site", "root": str(site)}],
                 "discord": {"guild_id": "123456789012345678", "token_secret": "discord_bot_token"}}
    resources.update(res)
    return TaskSpec("obj_factories", "exercise the composition", resources=resources, mode="staging")


def test_all_executors_registers_every_modality_configured_from_the_task(tmp_path):
    run = str(tmp_path / "run")
    t = task(tmp_path)
    env = build_environment(run, GrantPolicy(), compose="all", task=t, browser=True)
    try:
        caps = env.get_capabilities()
        assert sorted(caps["modalities"]) == sorted(ALL_MODALITIES)
        tools = {x.name: x for x in env.discover_tools()}
        for name in ("wait", "browser.navigate", "sandbox.python", "visual.click", "http.request", "publish.submit",
                     "discord.send_message", "skill.run", "skill.induce"):
            assert name in tools, name
        # configured from task.resources: an owned site for publishing, the guild and its token for Discord
        assert tools["publish.draft"].available and tools["publish.submit"].available
        assert tools["discord.get_identity"].available, tools["discord.get_identity"].unavailable_reason
        assert tools["visual.click"].available                       # its screens are the browser's sessions
        assert caps["executors"][Modality.API] == {} or isinstance(caps["executors"][Modality.API], dict)
    finally:
        env.close()
    bare = all_executors(str(tmp_path / "bare"), browser_enabled=False)
    assert [e.modality for e in bare] == [m for m in ALL_MODALITIES if m != "structured_browser"]
    api = [e for e in bare if e.modality == Modality.API][0]
    why = {t.name: t.unavailable_reason for t in api.tools() if not t.available}
    assert "no owner-authorized Discord guild" in why["discord.send_message"]          # no task: said so, not hidden
    assert "no owned publishing destination" in why["publish.submit"]
    assert default_sandbox_workspace(str(tmp_path / "x" / "exec")) == str(tmp_path / "x" / "exec" / "sandbox")
    (tmp_path / "x" / "sandbox").mkdir(parents=True)
    assert default_sandbox_workspace(str(tmp_path / "x" / "exec")) == str(tmp_path / "x" / "sandbox")


class AuthWithBroker(GrantPolicy):
    """GrantPolicy plus the executor-side broker methods of the authorization service (secret_get, scrub,
    scrub_obj), served over AuthRPC like the real service."""

    def __init__(self):
        super().__init__()
        self.broker = FourArgBroker({"api_key": SECRET})

    def secret_get(self, name, action, decision, destination):
        return self.broker.get(name, action, decision, destination)

    def scrub(self, text):
        return self.broker.scrub(text)

    def scrub_obj(self, obj):
        return json.loads(self.broker.scrub(json.dumps(obj)))


def test_the_service_composes_everything_and_scrubs_through_the_authorization_service(tmp_path):
    R = tempfile.mkdtemp(prefix="efx", dir="/tmp")            # unix socket paths are limited to ~107 bytes
    try:
        shutil.copyfile(os.path.join(SRC, "..", "tasks", "lab_staging.yaml"), os.path.join(R, "task.yaml"))
        run = os.path.join(R, "exec")
        os.makedirs(run)
        policy = AuthWithBroker()
        auth_sock = os.path.join(R, "auth.sock")
        auth = RpcServer(auth_sock, policy, POLICY_METHODS + ("secret_get", "scrub", "scrub_obj"), name="auth").start()
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([SRC, HERE]))
        # exactly what runtime/cli.py gives the exec service: no --executor, no --secrets, no --task
        proc = subprocess.Popen([sys.executable, "-m", "cam_live.harness.service", "--run-dir", run, "--socket",
                                 os.path.join(R, "exec.sock"), "--auth-socket", auth_sock, "--provenance", "staging",
                                 "--no-browser"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        try:
            ready = json.loads(proc.stdout.readline() or "{}")
            assert ready.get("ready") is True, proc.stderr.read()[-2000:]
            assert ready["compose"] == "all" and ready["secret_scrubbing"] is True
            assert ready["task_file"] == os.path.join(R, "task.yaml") == find_task_file(run)
            assert ready["modalities"] == sorted(m for m in ALL_MODALITIES if m != "structured_browser")
            remote = RemoteEnvironment(ready["socket"])
            caps = remote.get_capabilities()
            assert caps["features"]["secret_scrubbing"] is True
            d = Driver(remote, policy, TaskSpec("o", "x"))
            d.ok("internal.note", text=f"the key is {SECRET} and it must not come back")
            notes = d.ok("internal.inspect", what="notes").observations[0]
            assert SECRET not in json.dumps(notes.to_dict()) and "[secret]" in json.dumps(notes.to_dict())
            assert SECRET not in open(os.path.join(run, "events.jsonl")).read()
            remote.shutdown()
            assert proc.wait(timeout=30) == 0
        finally:
            if proc.poll() is None:
                proc.kill()
            auth.stop()
    finally:
        shutil.rmtree(R, ignore_errors=True)


@pytest.mark.privileged
@pytest.mark.browser
def test_separated_service_as_cam_live_run_starts_it_registers_everything_and_runs_chromium_as_the_browser_uid():
    if os.geteuid() != 0:
        pytest.skip("needs root")
    from cam_live.harness import SOCK_AUTH, SOCK_EXEC, uid_of
    from cam_live.harness.auth.client import connect
    from cam_live.harness.auth.launcher import LauncherConfig, launch
    from cam_live.harness.browser import SHARED_BROWSERS_DIR, find_chromium
    from cam_live.harness.staging import StagingSites
    try:
        uid_b = uid_of("browser")
    except KeyError:
        pytest.skip("service users do not exist")
    if not find_chromium(uid_b):
        pytest.fail(f"no shared Chromium under {SHARED_BROWSERS_DIR}: run python -m cam_live.harness.browser --install-shared")
    R = tempfile.mkdtemp(prefix="efs", dir="/tmp")
    os.chmod(R, 0o755)
    sites = StagingSites().start()
    sock = os.path.join(R, "sock")
    lc = LauncherConfig(extra_args={"exec": ["--auth-socket", os.path.join(sock, SOCK_AUTH), "--provenance", "staging",
                                             "--run-dir", os.path.join(R, "exec")]})        # as runtime/cli.py
    handle = launch(R, ["auth", "exec"], lc)
    try:
        auth = connect(R)
        t = TaskSpec("obj_sep", "exercise the separated environment", mode="staging")
        auth.register_task(t, "operator")
        s = sites.urls["structured"]
        auth.register_local_service(s.split("//", 1)[1].split("/", 1)[0], "operator", "staging site of this test")
        env = RemoteEnvironment(os.path.join(sock, SOCK_EXEC), connect_timeout=60.0)
        caps = env.get_capabilities()
        assert sorted(caps["modalities"]) == sorted(ALL_MODALITIES) and caps["features"]["secret_scrubbing"] is True
        pid = env.ping()["pid"]
        assert os.stat(f"/proc/{pid}").st_uid == uid_b
        a = make_action("browser.navigate", url=s + "/items")
        dec = auth.authorize(a, t, {})
        assert dec.verdict == AuthVerdict.ALLOW, dec.reasons
        res = env.execute(a, dec)
        assert res.status == ResultStatus.OK, res.failure_detail
        assert res.observations[0].data["url"] == s + "/items"
        bcaps = env.get_capabilities()["executors"]["structured_browser"]
        assert bcaps["version"] and bcaps["chromium_executable"].startswith(SHARED_BROWSERS_DIR)
        assert bcaps["uid_separation"]["enabled"] is True and bcaps["uid_separation"]["user"] == "cambrowser"
        chrome = []
        for st in os.listdir("/proc"):
            if not st.isdigit():
                continue
            try:
                cmd = open(f"/proc/{st}/cmdline", "rb").read().decode("utf-8", "replace")
            except OSError:
                continue
            if "--proxy-server=" in cmd and os.path.join(R, "exec") in cmd:
                chrome.append(os.stat(f"/proc/{st}").st_uid)
        assert chrome and set(chrome) == {uid_b}                 # Chromium runs as the browser uid, behind its guard
    finally:
        try:
            handle.stop()
        finally:
            sites.stop()
            time.sleep(0.3)
            shutil.rmtree(R, ignore_errors=True)


@pytest.mark.browser
def test_the_composed_visual_executor_acts_on_the_browsers_screen(tmp_path):
    """visual.screenshot / visual.click through the composed environment on the browser executor's BrowserScreen;
    the click's effect is seen by the site and every gesture is a checked primitive."""
    from e_fix_support import CountingServer, html
    srv = CountingServer()
    srv.routes["/hit"] = lambda *a: (200, {"Content-Type": "application/json"}, "{}")
    srv.page("/v", html("visual", "<button id=b style='position:absolute;left:100px;top:100px;width:200px;"
                                  "height:80px' onclick=\"fetch('/hit',{method:'POST'})\">PRESS</button>"))
    pol = GrantPolicy()
    env = build_environment(str(tmp_path / "run"), pol, compose="all", task=task(tmp_path))
    try:
        d = Driver(env, pol, TaskSpec("o", "x"))
        d.ok("browser.navigate", url=srv.url + "/v")
        shot = make_action("visual.screenshot")
        shot.modality = Modality.VISUAL
        r = env.execute(shot, pol.authorize(shot, None))
        assert r.status == ResultStatus.OK and r.observations[0].kind == "image", r.failure_detail
        oid = r.observations[0].obs_id
        click = make_action("visual.click", observation_id=oid, x=200, y=140)
        click.modality = Modality.VISUAL
        rc = env.execute(click, pol.authorize(click, None))
        time.sleep(0.5)
        assert rc.status == ResultStatus.OK, (rc.failure_class, rc.failure_detail)
        assert srv.count("/hit", "POST") == 1
        assert any(p[0] == "visual.click" for p in pol.primitives)
        assert any(p[0] == "browser.request" and "/hit" in p[1].get("url", "") for p in pol.primitives)
    finally:
        env.close()
        srv.close()
