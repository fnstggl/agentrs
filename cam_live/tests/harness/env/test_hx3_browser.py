"""G2-harness round-2 findings on real Chromium (run with ``-m browser``), real PolicyService (default policy), staging
site owned and pre-authorized by the task, ``<site>/private/*`` on the task's deny list. Both tests fail on 0a30cd8.

R2-M1  ordinary forms (a newsletter form with Drupal Antibot's hidden key, a quiz answer) are submitted
R2-m2  a denied path is not requested through <img>, <link rel=stylesheet>, <script src> or <link rel=prefetch>"""
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
    ok = lambda *a: (200, {}, html("thanks", "<h1>thanks</h1>"))  # noqa: E731
    srv.routes["/join"] = lambda m, p, q, h, b: ok() if m == "POST" else (200, {}, html("Join", """<h1>Newsletter</h1>
      <form method=post action=/join><label for=e>Email</label><input id=e name=email>
      <input type=hidden name=antibot_key value="k1"><button type=submit>Subscribe</button></form>"""))
    srv.routes["/quiz"] = lambda m, p, q, h, b: ok() if m == "POST" else (200, {}, html("Weekly challenge", """
      <h1>Weekly coding challenge</h1><form method=post action=/quiz><label for=a>Your answer</label>
      <input id=a name=challenge_answer><button type=submit>Submit answer</button></form>"""))
    srv.routes["/private/pixel.png"] = lambda *a: (200, {"Content-Type": "image/png"}, PNG)
    srv.routes["/private/style.css"] = lambda *a: (200, {"Content-Type": "text/css"}, ".x{color:red}")
    srv.routes["/private/data.js"] = lambda *a: (200, {"Content-Type": "text/javascript"}, "window.__leak=1;")
    srv.routes["/private/prefetch"] = lambda *a: (200, {}, "prefetched")
    srv.routes["/ok.png"] = lambda *a: (200, {"Content-Type": "image/png"}, PNG)
    srv.page("/public", html("public", "<h1>public</h1><img src=/ok.png><img src=/private/pixel.png>",
                             head="<link rel=stylesheet href=/private/style.css><script src=/private/data.js></script>"
                                  "<link rel=prefetch href=/private/prefetch>"))
    run_dir = str(tmp_path / "run")
    svc = PolicyService(run_dir, fsync=False)
    task = C.TaskSpec(objective_id="obj-hx3-browser", objective="round-2 findings", mode="staging",
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


def _wait_post(srv, path, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        got = [b for (m, p, h, b) in srv.log if m == "POST" and p == path]
        if got:
            return got
        time.sleep(0.2)
    return []


def test_ordinary_forms_with_antibot_keys_or_quiz_answers_are_submitted(rig):
    srv, b = rig
    o = b.ok("browser.navigate", url=srv.url + "/join").observations[0]
    assert b.do("browser.fill", ref=Driver.ref(o, "textbox", "Email"), obs_id=o.obs_id, text="a@x.org").status == "ok"
    o2 = b.inspect()
    b.do("browser.click", ref=Driver.ref(o2, "button", "Subscribe"), obs_id=o2.obs_id)
    assert _wait_post(srv, "/join") and b"antibot_key=k1" in _wait_post(srv, "/join")[0]
    o = b.ok("browser.navigate", url=srv.url + "/quiz").observations[0]
    assert b.do("browser.fill", ref=Driver.ref(o, "textbox", "Your answer"), obs_id=o.obs_id, text="42").status == "ok"
    o2 = b.inspect()
    b.do("browser.click", ref=Driver.ref(o2, "button", "Submit answer"), obs_id=o2.obs_id)
    assert _wait_post(srv, "/quiz") == [b"challenge_answer=42"]


def test_a_denied_path_is_not_requested_through_any_subresource(rig):
    srv, b = rig
    res = b.do("browser.navigate", url=srv.url + "/public")
    assert res.status == C.ResultStatus.OK, res.failure_detail
    time.sleep(1.5)
    private = [p for (m, p, h, bd) in srv.log if p.startswith("/private")]
    assert private == [], private
    assert srv.count("/ok.png") >= 1                     # an allowed image of the same origin still loads
