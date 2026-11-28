"""G2-harness probe g06 (real Chromium, real PolicyService + default policy): are a page's own same-origin GET
requests judged per URL? This is the mechanism behind D6 (a Discord page's same-origin reads of Discord's API,
which tests/harness/auth/test_hx_platform_constraints.py:136 asserts are refused as URL-level primitives).

Discord cannot be contacted, so the same mechanism is shown with a URL-level rule the policy *does* apply to a
staging site: the task's deny list names one path of the site (``<site>/private/*``). A navigation to that path is
refused; the question is whether a script of an allowed page of the same site can still read it.

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/g2harness/g06_browser_same_origin_reads.py
"""
import time

from _g2 import C, Report, cleanup, fresh_dir  # noqa: I001

from cam_live.harness.auth import PolicyService
from cam_live.harness.browser import BrowserExecutor
from cam_live.harness.env import ComputerEnvironment
from cam_live.harness.internal import InternalExecutor
from e_fix_support import CountingServer, html
from eb_support import Driver

R = Report("g06_browser_same_origin_reads")
srv = CountingServer()
srv.routes["/private/data"] = lambda *a: (200, {"Content-Type": "application/json"}, '{"secret_rows": 3}')
srv.page("/public", html("public", """<h1>public page</h1><div id=o>pending</div><script>
  fetch('/private/data').then(r => r.text()).then(t => document.getElementById('o').innerText = 'read: ' + t)
    .catch(e => document.getElementById('o').innerText = 'refused');</script>"""))

d = fresh_dir("g06")
run_dir = d + "/run"
svc = PolicyService(run_dir, fsync=False)
task = C.TaskSpec(objective_id="obj-g2h-origin", objective="probe per-URL read rules", mode="staging",
                  resources={"owned": [srv.host], "preauthorized": [srv.host], "deny": [f"{srv.host}/private/*"]},
                  limits=C.Limits(duration_s=3600, max_decisions=100, max_external_actions=100))
svc.register_task(task, "operator")
env = ComputerEnvironment(run_dir, svc)
env.register(InternalExecutor(run_dir, clock=env.clock))
env.register(BrowserExecutor(run_dir, clock=env.clock))
b = Driver(env, svc, task)
try:
    direct = b.do("browser.navigate", url=srv.url + "/private/data")
    R.check(direct.status != "ok", "control: navigating to the denied path is refused",
            f"{direct.status} {str(direct.failure_detail)[:120]!r}")
    srv.log.clear()
    res = b.do("browser.navigate", url=srv.url + "/public")
    time.sleep(2.0)
    got = [p for (m, p, hd, body) in srv.log if m == "GET" and p.startswith("/private")]
    o = b.inspect()
    R.check(not got, "a script of an allowed page cannot read a path the task denies on the same origin",
            f"navigate {res.status}; server received {got}; page says {o.text[-80:]!r}")
finally:
    env.close()
    svc.close()
    srv.close()
    cleanup(d)
R.done()
