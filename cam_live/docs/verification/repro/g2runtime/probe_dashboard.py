"""G2-runtime probe: dashboard credentials, CSRF, same-origin and security headers (LX-10).

    .venv/bin/python cam_live/docs/verification/repro/g2runtime/probe_dashboard.py

FastAPI TestClient against runtime.dashboard.create_app on a run directory with one pending approval (no network;
the in-process client has no socket peer, so the uid check is not exercised here: tests/runtime/
test_rt_fix_dashboard.py covers it with a camlearn process).
"""
import json
import os
import secrets
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g2common import Report, scratch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from cam_live.contracts import RunFiles  # noqa: E402
from cam_live.runtime.dashboard import CSP, create_app  # noqa: E402
from cam_live.runtime.logs import atomic_write_json  # noqa: E402

R = Report("probe_dashboard")
run = os.path.join(scratch("dash"), "run")
os.makedirs(run)
atomic_write_json(os.path.join(run, RunFiles.STATE), {"version": 2, "status": "running", "approvals": {
    "apr_1": {"approval_id": "apr_1", "status": "pending", "operation": "x.y", "preview": "p"}}})
TOK, OP = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
c = TestClient(create_app(run, TOK, operator_token=OP), raise_server_exceptions=False)
page = c.get(f"/?token={TOK}")
csrf = page.cookies.get("cam_live_csrf") or c.cookies.get("cam_live_csrf")
H = {"Authorization": f"Bearer {TOK}"}
OK = {**H, "X-CAM-Operator": OP, "X-CSRF-Token": csrf, "Origin": "http://testserver"}


def control_written():
    p = os.path.join(run, RunFiles.CONTROL)
    return json.load(open(p))["commands"] if os.path.exists(p) else []


n0 = len(control_written())
cases = {
    "viewing token only": ({**H}, None),
    "operator credential in the query string": ({**H, "X-CSRF-Token": csrf}, {"operator": OP}),
    "operator credential as a cookie": ({**H, "X-CSRF-Token": csrf, "Cookie": f"X-CAM-Operator={OP}"}, None),
    "no CSRF header": ({**H, "X-CAM-Operator": OP}, None),
    "CSRF header not matching the cookie": ({**H, "X-CAM-Operator": OP, "X-CSRF-Token": "x" * 43}, None),
    "Sec-Fetch-Site cross-site": ({**OK, "Sec-Fetch-Site": "cross-site"}, None),
    "Origin null": ({**OK, "Origin": "null"}, None),
    "Origin of another site": ({**OK, "Origin": "http://evil.example"}, None),
    "DNS-rebinding page (Host = Origin = evil) without the secrets": (
        {"Host": "evil.example:8787", "Origin": "http://evil.example:8787", "Sec-Fetch-Site": "same-origin"}, None),
    "wrong operator credential": ({**OK, "X-CAM-Operator": "y" * 32}, None),
}
for label, (hdr, params) in cases.items():
    r = c.post("/api/control/resume", headers=hdr, params=params or {})
    r2 = c.post("/api/approvals/apr_1/approve", headers=hdr, params=params or {}, json={})
    R.check(f"refused — {label}", r.status_code in (401, 403) and r2.status_code in (401, 403),
            {"resume": r.status_code, "approve": r2.status_code})
R.check("no command was written by any refused request", len(control_written()) == n0, len(control_written()) - n0)
r = c.post("/api/control/pause", headers=OK, json={"reason": "probe"})
R.check("control: viewing token + operator credential + CSRF + same origin is accepted", r.status_code == 200, r.status_code)

# security headers on every kind of response
with open(os.path.join(run, RunFiles.STATE), "w") as f:
    f.write("[1, 2, 3]")                                   # a state file that is valid JSON but not an object
fresh = TestClient(create_app(run, TOK, operator_token=OP), raise_server_exceptions=False)   # no cookies
resp = {"page": page, "401": fresh.get("/api/status"), "404": c.get("/api/logs/nope", headers=H),
        "403": c.post("/api/control/resume", headers=H), "422": c.get("/api/learning?n=abc", headers=H),
        "app.js": c.get("/app.js", headers=H), "status on a malformed state file": c.get("/api/status", headers=H)}
for k, r in resp.items():
    R.check(f"CSP + nosniff + DENY on the {k} response (HTTP {r.status_code})",
            r.headers.get("content-security-policy") == CSP and r.headers.get("x-content-type-options") == "nosniff"
            and r.headers.get("x-frame-options") == "DENY", dict(r.headers))
sys.exit(R.done())
