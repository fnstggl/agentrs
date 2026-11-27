"""OP-09 at execution time on real Chromium (run with ``-m browser``): a page of a staging site the task owns and
pre-authorizes embeds a human-verification challenge. The real authorization service (default policy) lets the
page and the challenge frame load, but the challenge's answer request is refused by the browser's network guard
before it leaves Chromium, while the page's ordinary write request goes through. Before the hard constraints the
answer request was sent (the site is owned and pre-authorized)."""
import time

import pytest

from cam_live import contracts as C
from cam_live.harness.auth import PolicyService
from cam_live.harness.browser import BrowserExecutor
from cam_live.harness.env import ComputerEnvironment
from cam_live.harness.internal import InternalExecutor

from e_fix_support import CountingServer, html

pytestmark = pytest.mark.browser

CHALLENGE = "/cdn-cgi/challenge-platform/h/b/turnstile/if/ov2/av0/rcv0/0/w1/light/normal"
ANSWER = "/cdn-cgi/challenge-platform/h/b/flow/ov1/0:17:1"


@pytest.fixture
def rig(tmp_path):
    srv = CountingServer()
    srv.routes[ANSWER] = lambda *a: (200, {"Content-Type": "application/json"}, '{"success": true}')
    srv.routes["/api/notes"] = lambda *a: (200, {"Content-Type": "application/json"}, "{}")
    srv.page(CHALLENGE, html("challenge", f"""<button id=cb>Verify you are human</button><script>
      fetch('{ANSWER}', {{method: 'POST', body: 'answer=solved'}}).then(r => document.title = 'answered')
        .catch(e => document.title = 'refused');</script>"""))
    srv.page("/signup", html("signup", f"""<h1>Sign up</h1><iframe id=ch src="{CHALLENGE}"></iframe><div id=o></div>
      <script>fetch('/api/notes', {{method: 'POST', body: 'n=1'}}).then(() =>
        document.getElementById('o').innerText = 'note-saved');</script>"""))
    run_dir = str(tmp_path / "run")
    svc = PolicyService(run_dir, fsync=False)
    task = C.TaskSpec(objective_id="obj-hx-browser", objective="exercise the network guard", mode="staging",
                      resources={"owned": [srv.host], "preauthorized": [srv.host]},
                      limits=C.Limits(duration_s=3600, max_decisions=100, max_external_actions=100))
    svc.register_task(task, "operator")
    env = ComputerEnvironment(run_dir, svc)
    env.register(InternalExecutor(run_dir, clock=env.clock))
    env.register(BrowserExecutor(run_dir, clock=env.clock))
    yield srv, svc, env, task
    env.close()
    svc.close()
    srv.close()


def test_a_challenge_frame_loads_but_its_answer_never_leaves_the_browser(rig):
    srv, svc, env, task = rig
    a = C.ActionSpec(action_id=C.new_id("act"), operation="browser.navigate", modality=C.Modality.STRUCTURED_BROWSER,
                     arguments={"url": srv.url + "/signup"})
    dec = svc.authorize(a, task, {})
    assert dec.verdict == C.AuthVerdict.ALLOW, dec.reasons
    res = env.execute(a, dec)
    assert res.status == C.ResultStatus.OK, res.failure_detail
    deadline = time.time() + 15
    while time.time() < deadline and not (srv.count("/api/notes", "POST") and srv.count(CHALLENGE, "GET")):
        time.sleep(0.2)
    time.sleep(1.0)
    assert srv.count("/signup", "GET") == 1
    assert srv.count(CHALLENGE, "GET") == 1          # loading the challenge frame is a read: permitted
    assert srv.count("/api/notes", "POST") == 1      # an ordinary write of the owned page: permitted
    assert srv.count(ANSWER, "POST") == 0            # answering the challenge: refused before it was sent
    checks = [r for r in env.records("env.primitive_check", 500, a.action_id)]
    refused = [r for r in checks if ANSWER in r["op"]["arguments"].get("url", "")]
    assert refused and all(r["verdict"] == C.AuthVerdict.DENY for r in refused), checks
    assert any("human-verification" in " ".join(r.get("reasons") or []) for r in refused), refused
