"""Real-world abuse paths of the authorisation-gated bridge (Part 16, SP-16b): no real-world spam, fabricated
engagement, covert influence, surveillance or impersonation.

`check()` runs in a FRESH interpreter (a process that never hosted a world; the bridge refuses to exist in one that
did) and plays an operator with a valid authorisation for a loopback staging service. It returns, per attempt,
whether the bridge refused it:

* authorisation documents naming third-party platforms (social networks, messaging and mail gateways, review and
  ranking sites, advertising and search, code hosting and wikis), including subdomains and mixed-case names;
* requests that set identity headers (a browser User-Agent, From, X-Forwarded-For, Forwarded, Via, Referer, ...);
* and, as the positive control, that an ordinary authorised GET reaches the staging service carrying the bridge's
  automation disclosure as its User-Agent.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

PLATFORM_URLS = ("https://api.twitter.com/2/", "https://x.com/", "https://www.reddit.com/api/", "https://graph.facebook.com/",
                 "https://api.telegram.org/", "https://discord.com/api/", "https://API.LinkedIn.com/v2/",
                 "https://www.yelp.com/", "https://www.trustpilot.com/", "https://api.github.com/repos/",
                 "https://en.wikipedia.org/w/", "https://api.sendgrid.com/v3/", "https://www.googleapis.com/youtube/",
                 "https://news.ycombinator.com/")
SPOOFED_HEADERS = ({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"}, {"From": "alice@example.org"},
                   {"X-Forwarded-For": "203.0.113.7"}, {"Forwarded": "for=198.51.100.9"}, {"Via": "1.1 human-proxy"},
                   {"Referer": "https://news.example.org/"}, {"user-agent": "curl/8.0"})

_CODE = r"""
import http.server, json, os, sys, tempfile, threading, time
from crucible.realworld import RealWorldBridge, BridgeRefused, AuthorizationError, load_authorization
from crucible.realworld.authorization import ACK
seen = []
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        seen.append(dict(self.headers))
        b = b"ok"
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
d = tempfile.mkdtemp()
def doc(urls):
    return {"acknowledgement": ACK, "authorised_by": "Red Team <operator@localhost>", "experiment": "rt-abuse",
            "purpose": "abuse-path probe against a loopback staging service", "base_urls": urls,
            "methods": ["GET", "POST"], "limits": {"max_requests": 50, "max_rate_per_s": 1000.0, "max_bytes": 100000,
            "max_wall_s": 600}, "expires": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3600)),
            "approval_required": True, "allow_private_addresses": ["127.0.0.1:%d" % port]}
def write(name, body):
    p = os.path.join(d, name)
    with open(p, "w") as fh:
        json.dump(body, fh)
    os.chmod(p, 0o600)
    return p
out = {"platforms": {}, "spoofed": {}}
for i, u in enumerate(json.loads(sys.argv[1])):
    try:
        load_authorization(write("p%d.json" % i, doc(["http://127.0.0.1:%d/api/" % port, u])))
        out["platforms"][u] = "accepted"
    except AuthorizationError as e:
        out["platforms"][u] = "refused"
b = RealWorldBridge.from_file(write("ok.json", doc(["http://127.0.0.1:%d/api/" % port])), log_path=os.path.join(d, "log"))
r = b.get("http://127.0.0.1:%d/api/x" % port)
out["control_status"] = r.status
out["control_user_agent"] = (seen[-1] if seen else {}).get("User-Agent")
n_before = len(seen)
for hdr in json.loads(sys.argv[2]):
    try:
        b.get("http://127.0.0.1:%d/api/y" % port, headers=hdr)
        out["spoofed"][json.dumps(hdr)] = "sent"
    except BridgeRefused:
        out["spoofed"][json.dumps(hdr)] = "refused"
out["requests_reaching_staging_with_spoofed_headers"] = len(seen) - n_before
b.close()
srv.shutdown()
print(json.dumps(out))
"""


def check(src_root: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("CRUCIBLE_HOSTS_WORLDS",)}
    env["PYTHONPATH"] = src_root
    r = subprocess.run([sys.executable, "-c", _CODE, json.dumps(list(PLATFORM_URLS)), json.dumps(list(SPOOFED_HEADERS))],
                       capture_output=True, text=True, timeout=180, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"the abuse-path check failed to run: {r.stderr[-1500:]}")
    out = json.loads(r.stdout.strip().splitlines()[-1])
    ua = str(out.get("control_user_agent") or "")
    out["abuse_paths"] = sorted([f"platform:{u}" for u, v in out["platforms"].items() if v != "refused"] +
                                [f"spoof:{h}" for h, v in out["spoofed"].items() if v != "refused"] +
                                ([] if "automated research request" in ua else ["no_automation_disclosure"]))
    out["control_ok"] = out.get("control_status") == 200
    return out
