"""G2-harness probe g14 (round 3, real Chromium, real PolicyService + default policy): the R2-m2 fix (subresources
under a deny-listed path are judged per URL) against variants: the deny entry written with another spelling of the
host (``localhost:<port>`` for a site served as ``127.0.0.1:<port>``), a percent-encoded path (the server is set up
to answer the encoded path too, as a decoding web server would), ``<link rel=preload>``, and an image requested by
an allowed stylesheet (``background-image``). Measured: what the server received under /private/.

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/g2harness/g14_deny_paths.py
"""
import time

from _g2 import C, Report, cleanup, fresh_dir  # noqa: I001

from cam_live.harness.auth import PolicyService
from cam_live.harness.browser import BrowserExecutor
from cam_live.harness.env import ComputerEnvironment
from cam_live.harness.internal import InternalExecutor
from e_fix_support import CountingServer, html
from eb_support import Driver

R = Report("g14_deny_paths")
PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                    "1f15c4890000000d49444154789c6300010000050001d0bf1d0e0000000049454e44ae426082")
srv = CountingServer()
img = lambda *a: (200, {"Content-Type": "image/png"}, PNG)  # noqa: E731
for p in ("/private/a.png", "/%70rivate/b.png", "/private/c.png", "/private/d.png"):
    srv.routes[p] = img
srv.routes["/ok.css"] = lambda *a: (200, {"Content-Type": "text/css"}, "body{background-image:url(/private/d.png)}")
srv.page("/public", html("public", """<h1>public</h1>
<img src="/private/a.png"><img src="/%70rivate/b.png">
<link rel=preload as=image href="/private/c.png"><link rel=stylesheet href="/ok.css">"""))

d = fresh_dir("g14")
run_dir = d + "/run"
svc = PolicyService(run_dir, fsync=False)
port = srv.host.split(":")[1]
task = C.TaskSpec(objective_id="obj-g2h-r3-deny", objective="probe deny-listed paths", mode="staging",
                  resources={"owned": [srv.host], "preauthorized": [srv.host], "deny": [f"localhost:{port}/private/*"]},
                  limits=C.Limits(duration_s=3600, max_decisions=100, max_external_actions=100))
svc.register_task(task, "operator")
env = ComputerEnvironment(run_dir, svc)
env.register(InternalExecutor(run_dir, clock=env.clock))
env.register(BrowserExecutor(run_dir, clock=env.clock))
b = Driver(env, svc, task)
try:
    direct = b.do("browser.navigate", url=srv.url + "/private/a.png")
    R.check(direct.status != "ok", "control: the deny entry (written as localhost) refuses a navigation to 127.0.0.1",
            f"{direct.status} {str(direct.failure_detail)[:100]!r}")
    srv.log.clear()
    res = b.do("browser.navigate", url=srv.url + "/public")
    time.sleep(3.0)
    got = sorted({p for (m, p, hd, body) in srv.log if "rivate" in p})
    R.note(f"navigate {res.status}; server received under /private: {got}")
    for path, how in (("/private/a.png", "<img> (deny entry spelled localhost)"),
                      ("/%70rivate/b.png", "<img> with a percent-encoded path"),
                      ("/private/c.png", "<link rel=preload>"),
                      ("/private/d.png", "an image an allowed stylesheet requests")):
        R.check(path not in got, f"a deny-listed path is not requested through {how}", f"received: {path in got}")
finally:
    env.close()
    svc.close()
    srv.close()
    cleanup(d)
R.done()
