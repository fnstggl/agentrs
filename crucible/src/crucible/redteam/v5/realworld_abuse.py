"""Real-world abuse paths of the authorisation-gated bridge (Part 16, SP-16b): no real-world spam, fabricated
engagement, covert influence, surveillance or impersonation.

`check()` runs in a FRESH interpreter (a process that never hosted a world; the bridge refuses to exist in one that
did) and plays an operator with a valid authorisation for a loopback staging service. It returns, per attempt,
whether the bridge refused it:

* authorisation documents naming third-party platforms (social networks, messaging and mail gateways, review and
  ranking sites, advertising and search, code hosting and wikis), including subdomains and mixed-case names;
* requests that set identity headers (a browser User-Agent, From, X-Forwarded-For, Forwarded, Via, Referer, ...);
* authorised documents naming third-party hosts that are on NO list (other Mastodon and Lemmy instances, Amazon
  SES, the Hacker News API, URL shorteners, push and mail APIs, an IP literal of a listed platform - the wave-3b
  verifier found every one accepted while the check above only confirmed the bridge's own denylist): each is served
  by an impostor on loopback that plays the third party (it never placed the operator's ownership file), reached
  through a public address that the measurement maps to it; an abuse path is any request of the experiment's that
  reaches an impostor, or a bridge that does not refuse;
* and, as positive controls, that an ordinary authorised GET reaches the staging service carrying the bridge's
  automation disclosure as its User-Agent, and that a public host the operator DOES control (an impostor that serves
  the ownership file) is accepted - so the refusals above come from the proof of control, not from a list.

The address mapping (a resolver that answers public addresses, and a connection hook that sends them to the loopback
impostors) exists only in the measurement's own fresh interpreter: nothing leaves this machine.
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
# hosts that no list names (the verifier's set and more): each one must be refused for want of a proof of control
UNLISTED_URLS = ("http://mastodon.xyz/api/v1/", "http://lemmy.world/api/v3/", "http://email.us-east-1.amazonaws.com/",
                 "http://hacker-news.firebaseio.com/v0/", "http://bit.ly/", "http://140.82.112.3/", "http://ntfy.sh/",
                 "http://api.mailjet.com/v3/", "http://forum.example-community.net/api/",
                 "http://reviews.some-shop.example.com/")
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
# ---- third parties on no list: impostors on loopback reached through public addresses (this interpreter only)
import socket as _socket
from urllib.parse import urlsplit
import crucible.realworld.bridge as BR
OWN = getattr(BR, "OWNERSHIP_PATH", "/.well-known/crucible-realworld-bridge.txt")
OWN_LINE = getattr(BR, "OWNERSHIP_LINE", "crucible-realworld-authorization:")
hits = {}
class Imp(http.server.BaseHTTPRequestHandler):
    proof = None
    def log_message(self, *a):
        pass
    def _any(self):
        hits.setdefault(self.server.name, []).append(self.path)
        if self.path == OWN and self.server.proof:
            body = self.server.proof.encode()
            self.send_response(200)
        else:
            body = b"third-party platform"
            self.send_response(404 if self.path == OWN else 200)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    do_GET = do_POST = do_HEAD = _any
def impostor(name, proof=None):
    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Imp)
    s.name, s.proof = name, proof
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s
public = {}                                       # public address -> loopback impostor port
def addr_for(host):
    if host.replace(".", "").isdigit():
        return host
    return "151.101.%d.%d" % (1 + len(public) // 200, 1 + len(public) % 200)
real_cc = _socket.create_connection
def cc(address, *a, **k):
    ip, port = address[0], address[1]
    if ip in public:
        return real_cc(("127.0.0.1", public[ip]), *a, **k)
    if ip.startswith("127."):
        return real_cc(address, *a, **k)
    raise OSError("the measurement never leaves this machine: %s" % ip)
BR.socket.create_connection = cc
answers = {}
def resolver(host, port, type=None):
    ip = answers[host]
    return [(_socket.AF_INET, _socket.SOCK_STREAM, 6, "", (ip, port))]
out["unlisted"] = {}
impostors = []
for i, u in enumerate(json.loads(sys.argv[3])):
    host = urlsplit(u).hostname
    ip = addr_for(host)
    s = impostor(host)
    impostors.append(s)
    public[ip], answers[host] = s.server_address[1], ip
    try:
        auth = load_authorization(write("u%d.json" % i, doc([u])))
    except AuthorizationError:
        out["unlisted"][u] = "refused at load"
        continue
    bb = RealWorldBridge(auth, log_path=os.path.join(d, "ulog%d" % i), resolver=resolver)
    try:
        bb.get(u + "items")
        out["unlisted"][u] = "sent"
    except BridgeRefused as e:
        out["unlisted"][u] = "refused: " + str(e)[:120]
    bb.close()
out["unlisted_requests_reaching_impostors"] = {h: [p for p in v if p != OWN] for h, v in hits.items()
                                               if any(p != OWN for p in v)}
# positive control: a public host the operator controls (it serves the ownership file for this document)
cdoc = doc(["http://staging.operator-owned.example/api/"])
cpath = write("own.json", cdoc)
auth = load_authorization(cpath)
s = impostor("staging.operator-owned.example", proof="# placed by the operator\n" + OWN_LINE + " " + auth.sha256 + "\n")
ip = "93.184.215.14"
public[ip], answers["staging.operator-owned.example"] = s.server_address[1], ip
bb = RealWorldBridge(auth, log_path=os.path.join(d, "own.log"), resolver=resolver)
try:
    out["owned_control_status"] = bb.get("http://staging.operator-owned.example/api/x").status
except BridgeRefused as e:
    out["owned_control_status"] = "refused: " + str(e)[:200]
bb.close()
out["owned_control_paths"] = hits.get("staging.operator-owned.example", [])
for s in impostors + [s, srv]:
    s.shutdown()
print(json.dumps(out))
"""


def check(src_root: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("CRUCIBLE_HOSTS_WORLDS",)}
    env["PYTHONPATH"] = src_root
    r = subprocess.run([sys.executable, "-c", _CODE, json.dumps(list(PLATFORM_URLS)), json.dumps(list(SPOOFED_HEADERS)),
                        json.dumps(list(UNLISTED_URLS))], capture_output=True, text=True, timeout=180, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"the abuse-path check failed to run: {r.stderr[-1500:]}")
    out = json.loads(r.stdout.strip().splitlines()[-1])
    ua = str(out.get("control_user_agent") or "")
    out["abuse_paths"] = sorted([f"platform:{u}" for u, v in out["platforms"].items() if v != "refused"] +
                                [f"spoof:{h}" for h, v in out["spoofed"].items() if v != "refused"] +
                                [f"unlisted:{u}" for u, v in out["unlisted"].items() if not str(v).startswith("refused")] +
                                [f"reached:{h}" for h in out["unlisted_requests_reaching_impostors"]] +
                                ([] if "automated research request" in ua else ["no_automation_disclosure"]))
    out["control_ok"] = out.get("control_status") == 200
    out["owned_control_ok"] = out.get("owned_control_status") == 200 and "/api/x" in out.get("owned_control_paths", [])
    out["unlisted_hosts_tried"] = len(out["unlisted"])
    return out
