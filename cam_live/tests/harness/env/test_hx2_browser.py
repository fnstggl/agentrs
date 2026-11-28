"""G2-harness D4 and D6 on real Chromium (run with ``-m browser``), real PolicyService with the default policy, real
ComputerEnvironment; staging sites the task owns and pre-authorizes, so only the fixed constraints can refuse.

D4  a site's own CAPTCHA (a field labelled as the CAPTCHA answer, an "I am not a robot" checkbox) is neither
    answered nor submitted: the server receives no submission carrying it (before: ``captcha_input=XK4P9`` arrived)
D6  a page script's read of a path the task denies is refused even on the page's own origin (before: the script
    read ``/private/data`` and the page showed it)"""
import time

import pytest

from cam_live import contracts as C
from cam_live.harness.auth import PolicyService
from cam_live.harness.browser import BrowserExecutor
from cam_live.harness.env import ComputerEnvironment
from cam_live.harness.internal import InternalExecutor

from e_fix_support import CountingServer, html
from eb_support import Driver

pytestmark = pytest.mark.browser
PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                    "1f15c4890000000d49444154789c6300010000050001d0bf1d0e0000000049454e44ae426082")


@pytest.fixture
def rig(tmp_path):
    srv = CountingServer()
    srv.routes["/captcha.png"] = lambda *a: (200, {"Content-Type": "image/png"}, PNG)
    srv.routes["/signup"] = lambda m, p, q, h, b: (200, {}, html("thanks", "<h1>thanks</h1>")) if m == "POST" else (
        200, {}, html("Sign up", """<h1>Create an account</h1><form method=post action=/signup>
          <label for=em>Email</label><input id=em name=email>
          <img src=/captcha.png alt="CAPTCHA image: type these characters to prove you are human">
          <label for=c>CAPTCHA answer</label><input id=c name=captcha_input>
          <button type=submit>Sign up</button></form>"""))
    srv.routes["/join"] = lambda m, p, q, h, b: (200, {}, html("joined", "<h1>ok</h1>")) if m == "POST" else (
        200, {}, html("Join", """<h1>Join</h1><form method=post action=/join>
          <label><input type=checkbox name=human_check value=1> I am not a robot (human verification)</label>
          <button type=submit>Continue</button></form>"""))
    srv.routes["/notes"] = lambda m, p, q, h, b: (200, {}, html("saved", "<h1>saved</h1>")) if m == "POST" else (
        200, {}, html("Notes", """<form method=post action=/notes><label for=t>Note</label><input id=t name=text>
          <button type=submit>Save</button></form>"""))
    srv.routes["/private/data"] = lambda *a: (200, {"Content-Type": "application/json"}, '{"secret_rows": 3}')
    srv.routes["/public/data"] = lambda *a: (200, {"Content-Type": "application/json"}, '{"rows": 1}')
    srv.page("/public", html("public", """<h1>public page</h1><div id=o>pending</div><div id=p>pending</div><script>
      fetch('/private/data').then(r => r.text()).then(t => document.getElementById('o').innerText = 'read: ' + t)
        .catch(e => document.getElementById('o').innerText = 'refused');
      fetch('/public/data').then(r => r.text()).then(t => document.getElementById('p').innerText = 'pub: ' + t)
        .catch(e => document.getElementById('p').innerText = 'pub refused');</script>"""))
    run_dir = str(tmp_path / "run")
    svc = PolicyService(run_dir, fsync=False)
    task = C.TaskSpec(objective_id="obj-hx2-browser", objective="exercise the fixed constraints", mode="staging",
                      resources={"owned": [srv.host], "preauthorized": [srv.host], "deny": [f"{srv.host}/private/*"]},
                      limits=C.Limits(duration_s=3600, max_decisions=200, max_external_actions=200))
    svc.register_task(task, "operator")
    env = ComputerEnvironment(run_dir, svc)
    env.register(InternalExecutor(run_dir, clock=env.clock))
    env.register(BrowserExecutor(run_dir, clock=env.clock))
    yield srv, Driver(env, svc, task)
    env.close()
    svc.close()
    srv.close()


def posts(srv, path):
    return [body for (m, p, hd, body) in srv.log if m == "POST" and p.startswith(path)]


def test_a_sites_own_captcha_is_neither_answered_nor_submitted(rig):
    srv, b = rig
    o = b.ok("browser.navigate", url=srv.url + "/signup").observations[0]
    email = b.do("browser.fill", ref=Driver.ref(o, "textbox", "Email"), obs_id=o.obs_id, text="a@owned.example")
    answer = b.do("browser.fill", ref=Driver.ref(o, "textbox", "CAPTCHA answer"), obs_id=o.obs_id, text="XK4P9")
    assert email.status == C.ResultStatus.OK
    assert answer.status == C.ResultStatus.DENIED and "human-verification" in answer.failure_detail, answer.failure_detail
    o2 = b.inspect()
    b.do("browser.click", ref=Driver.ref(o2, "button", "Sign up"), obs_id=o2.obs_id)
    time.sleep(1.0)
    assert posts(srv, "/signup") == []                 # the form with the CAPTCHA field never left the browser
    o = b.ok("browser.navigate", url=srv.url + "/join").observations[0]
    tick = b.do("browser.click", ref=Driver.ref(o, "checkbox", "I am not a robot (human verification)"), obs_id=o.obs_id)
    assert tick.status == C.ResultStatus.DENIED, (tick.status, tick.failure_detail)
    time.sleep(0.5)
    assert not [x for x in posts(srv, "/join") if b"human_check" in x]
    # an ordinary form of the same site still works
    o = b.ok("browser.navigate", url=srv.url + "/notes").observations[0]
    assert b.do("browser.fill", ref=Driver.ref(o, "textbox", "Note"), obs_id=o.obs_id, text="hello").status == "ok"
    o2 = b.inspect()
    b.do("browser.click", ref=Driver.ref(o2, "button", "Save"), obs_id=o2.obs_id)
    deadline = time.time() + 10
    while time.time() < deadline and not posts(srv, "/notes"):
        time.sleep(0.2)
    assert posts(srv, "/notes") == [b"text=hello"]


def test_a_page_script_cannot_read_a_denied_path_of_its_own_origin(rig):
    srv, b = rig
    direct = b.do("browser.navigate", url=srv.url + "/private/data")
    assert direct.status != C.ResultStatus.OK
    srv.log.clear()
    res = b.do("browser.navigate", url=srv.url + "/public")
    assert res.status == C.ResultStatus.OK, res.failure_detail
    deadline = time.time() + 10
    text = ""
    while time.time() < deadline:
        text = b.inspect().text
        if "pub:" in text or "pub refused" in text:
            break
        time.sleep(0.3)
    time.sleep(0.5)
    text = b.inspect().text
    got = [p for (m, p, hd, body) in srv.log if m == "GET" and p.startswith("/private")]
    assert got == [] and "secret_rows" not in text and "refused" in text, (got, text[-200:])
    assert "pub:" in text and '"rows": 1' in text                  # an allowed same-origin read still works
