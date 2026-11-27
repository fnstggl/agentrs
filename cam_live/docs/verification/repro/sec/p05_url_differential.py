"""Destination parsing: does the policy judge the same host a real client connects to?

The policy normalises URLs with urllib.parse.urlsplit (destinations.py normalise_url). WHATWG clients (Chromium)
treat "\\" as "/" in http(s) URLs, so in  http://HOST_B\\@HOST_A/  urlsplit sees user-info "HOST_B\\" and host
HOST_A, while Chromium connects to HOST_B. normalise_url rejects whitespace but not a backslash.

The probe uses two loopback servers: A (127.0.0.1) is allowed; B (127.0.0.2:80) is outside the
staging envelope and on the task's deny list. It shows
  1. the policy decision for the crafted URL (expected for a sound policy: DENY, or the destination it reports = B),
  2. where Chromium really goes with that string,
  3. whether the integrated path (ComputerEnvironment + BrowserExecutor of workstream EB + the real PolicyService)
     lets the request reach B.
Everything stays on 127.0.0.1.
"""
import http.server
import threading

from _common import ALLOW, C, Report, cleanup, fresh_dir
from ea_helpers import act, prim, task
from cam_live.harness.auth import PolicyService
from cam_live.harness.auth import destinations as D

R = Report("p05_url_differential")
hits = {"A": [], "B": []}


def server(name, host="127.0.0.1", port=0):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits[name].append(self.path)
            body = f"<html><body><h1>server {name}</h1></body></html>".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    s = http.server.ThreadingHTTPServer((host, port), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


# B listens on 127.0.0.2:80 (needs root): a host OUTSIDE the staging envelope [127.0.0.1, localhost, ::1], reached
# without a port so that the crafted user-info part contains no ":" (a "user:pass@" form is caught by the policy's
# K-BASIC-AUTH-URL credential pattern; a bare "host\\@" is not).
sa, sb = server("A"), server("B", "127.0.0.2", 80)
PA = sa.server_address[1]
B_HOST = "127.0.0.2"
crafted = f"http://{B_HOST}\\@127.0.0.1:{PA}/x"
print(f"  allowed A = 127.0.0.1:{PA}; forbidden B = {B_HOST}:80; crafted URL = {crafted!r}")

d = fresh_dir("p05")
svc = PolicyService(d, fsync=False)
t = task(oid="obj-url", resources={"deny": [B_HOST]},
         interfaces=[C.Modality.STRUCTURED_BROWSER, C.Modality.API, C.Modality.INTERNAL],
         limits=C.Limits(duration_s=3600, max_external_actions=100, max_decisions=100))
MB = C.Modality.STRUCTURED_BROWSER

plain = svc.authorize(act("browser.navigate", MB, arguments={"url": f"http://{B_HOST}/x"}), t)
R.check(plain.verdict != ALLOW, "the denied destination B is refused when named plainly", ",".join(plain.policy_ids))
seen = D.normalise_url(crafted)
a = act("browser.navigate", MB, arguments={"url": crafted})
dec = svc.authorize(a, t)
print(f"  policy reads the crafted URL as {seen.key} -> {dec.verdict} {dec.policy_ids}")

# 2. where does Chromium go?
from playwright.sync_api import sync_playwright
with sync_playwright() as pw:
    br = pw.chromium.launch()
    page = br.new_page()
    page.goto(crafted)
    chromium_url = page.url
    br.close()
went_b = any(True for _ in hits["B"])
print(f"  Chromium resolved it to {chromium_url}; hits A={hits['A']} B={hits['B']}")
R.check(not (dec.verdict == ALLOW and went_b),
        "the policy and Chromium agree on the host of a URL (no parser differential)",
        f"policy allowed {seen.key}; Chromium fetched {chromium_url}")

# httpx (the HTTP/API executor's client)
import httpx
try:
    hx = httpx.URL(crafted)
    hx_host = f"{hx.host}:{hx.port}"
except Exception as e:  # noqa: BLE001
    hx_host = f"rejected ({type(e).__name__})"
print(f"  httpx reads it as {hx_host}")

# 3. integrated path: EB's browser executor behind the real policy
hits["A"].clear(); hits["B"].clear()
from cam_live.harness.browser import BrowserExecutor
from cam_live.harness.env import ComputerEnvironment
from cam_live.harness.internal import InternalExecutor
run = fresh_dir("p05run")
env = ComputerEnvironment(run, svc)
env.register(InternalExecutor(run, clock=env.clock))
env.register(BrowserExecutor(run, clock=env.clock))
try:
    a2 = act("browser.navigate", MB, arguments={"url": crafted})
    g = svc.authorize(a2, t)
    res = env.execute(a2, g) if g.verdict == ALLOW else None
    status = res.status if res is not None else f"not executed ({g.verdict})"
    final = ""
    if res is not None and res.observations:
        final = str(res.observations[0].data.get("url", ""))
    print(f"  integrated: status={status} detail={(res.failure_detail if res else '')[:160]!r} final={final!r} "
          f"hits A={hits['A']} B={hits['B']}")
    R.check(not hits["B"], "integrated path: no request reaches the denied server B",
            f"B received {hits['B']}")
finally:
    env.close()
    svc.close()
    sa.shutdown(); sb.shutdown()
    cleanup(d); cleanup(run)
R.done()
