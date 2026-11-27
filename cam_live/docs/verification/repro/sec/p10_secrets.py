"""Secret vault / broker: release only to a granted, running action for a permitted and bound destination;
scrubbing; nothing secret in errors, decisions, approvals or the audit log.
"""
import base64
import codecs
import json
import os
import urllib.parse

from _common import ALLOW, DENY, C, Report, cleanup, fresh_dir
from ea_helpers import act, authorize_live, live_task, prim, start
from cam_live.harness.auth import PolicyService
from cam_live.harness.auth.vault import SecretError

R = Report("p10_secrets")
d = fresh_dir("p10")
clock = C.SimClock()
svc = PolicyService(d, clock=clock, fsync=False)
SECRET = "sk-CANARY-7f3a9b2c4d5e6f708192a3b4c5d6e7f8"
t = live_task(oid="obj-sec")
authorize_live(svc, d, t, clock, owned=["https://api.owned.example/*", "https://owned.example/*"],
               read=["api.owned.example", "owned.example", "docs.example"],
               preauthorized=["https://api.owned.example/*", "https://owned.example/*"])
svc.put_secret("site_token", SECRET, ["https://api.owned.example/v1/*"])
API = C.Modality.API
DEST = "https://api.owned.example/v1/items"


def refused(what, fn):
    try:
        val = fn()
    except SecretError as e:
        R.check(SECRET not in str(e), f"no release: {what}", str(e)[:70])
        return
    R.defect(f"no release: {what}", f"got a value of length {len(val)}")


a = act("http.post", API, arguments={"url": DEST, "headers": {"authorization": "secret:site_token"}, "json": {"a": 1}})
dec = svc.authorize(a, t)
assert dec.verdict == ALLOW, dec.reasons
refused("granted but not yet running (grant not consumed)", lambda: svc.secret_get("site_token", a, dec, DEST))
never = act("http.post", API, arguments={"url": DEST, "json": {"a": 2}})
refused("no decision at all", lambda: svc.secret_get("site_token", never, C.AuthorizationDecision(verdict=ALLOW, action_hash=never.content_hash()), DEST))
assert svc.check_grant(a, dec)
R.check(svc.secret_get("site_token", a, dec, DEST) == SECRET, "release to the running granted action for the bound destination")
refused("destination the secret is not bound to (owned, permitted)", lambda: svc.secret_get("site_token", a, dec, "https://owned.example/echo"))
refused("destination outside the task", lambda: svc.secret_get("site_token", a, dec, "https://evil.example/v1/x"))
refused("bound path prefix as a sibling host", lambda: svc.secret_get("site_token", a, dec, "https://api.owned.example.evil.example/v1/x"))
refused("path traversal out of the bound path", lambda: svc.secret_get("site_token", a, dec, "https://api.owned.example/v1/../../echo"))
refused("unknown secret name", lambda: svc.secret_get("nope", a, dec, DEST))
refused("another action's decision", lambda: svc.secret_get("site_token", never, dec, DEST))
other = act("wait", C.Modality.INTERNAL)
od = start(svc, other, t)
got = None
try:
    got = svc.secret_get("site_token", other, od, DEST)
except SecretError:
    pass
R.note(f"an unrelated running action (internal 'wait', no destination of its own) obtains the secret for the bound "
       f"destination: {got == SECRET} - release is bound to (running grant, destination string supplied by the "
       f"executor), not to the action's own declared destinations")
svc.record(a, C.ActionResult(action_id=a.action_id, status=C.ResultStatus.OK))
refused("after the action was recorded (closed)", lambda: svc.secret_get("site_token", a, dec, DEST))
b = act("http.post", API, arguments={"url": DEST, "json": {"a": 3}}, timeout_s=5.0)
bd = start(svc, b, t)
clock.advance(5.0 + svc.policy.grants["exec_slack_s"] + 1)
refused("after the execution window", lambda: svc.secret_get("site_token", b, bd, DEST))
c = act("http.post", API, arguments={"url": DEST, "json": {"a": 4}})
cd = start(svc, c, t)
svc.revoke("secret:site_token", "operator")
refused("revoked secret", lambda: svc.secret_get("site_token", c, cd, DEST))
svc.put_secret("site_token", SECRET, ["https://api.owned.example/v1/*"])
svc.emergency_stop("probe", "G-sec")
refused("emergency stop", lambda: svc.secret_get("site_token", c, cd, DEST))
svc.reset_stop("operator")

# ---- the value never appears in anything the learner can see or in the audit log
leak = act("http.post", API, arguments={"url": "https://owned.example/echo", "json": {"note": SECRET}})
lv = svc.authorize(leak, t)
R.check(lv.verdict == DENY and "X-CREDENTIAL-LEAK" in lv.policy_ids and SECRET not in json.dumps(lv.to_dict()),
        "an action carrying the literal value is denied and the decision does not echo it")
for name, form in (("base64", base64.b64encode(SECRET.encode()).decode()), ("hex", SECRET.encode().hex()),
                   ("url-quoted", urllib.parse.quote("x=" + SECRET + "&", safe="")), ("reversed", SECRET[::-1]),
                   ("base64 inside a longer blob", base64.b64encode(b"Authorization: Bearer " + SECRET.encode() + b"\r\n").decode())):
    v = svc.authorize(act("http.post", API, arguments={"url": "https://owned.example/echo", "json": {"note": form}}), t)
    R.check(v.verdict == DENY and "X-CREDENTIAL-LEAK" in v.policy_ids, f"value in {name} form is recognised in an action")
audit = open(os.path.join(d, C.RunFiles.AUDIT)).read()
state = open(os.path.join(d, "auth", "state.sqlite"), "rb").read()
R.check(SECRET not in audit and base64.b64encode(SECRET.encode()).decode() not in audit, "audit log holds no secret value")
R.check(SECRET.encode() not in state, "authorization state database holds no secret value")
R.check(SECRET not in json.dumps(svc.approvals()) and SECRET not in json.dumps(svc.limits_state(), default=str),
        "approvals() / limits_state() hold no secret value")

# ---- scrubbing: what it knows and what it does not
page = f"token={SECRET} b64={base64.b64encode(SECRET.encode()).decode()} hex={SECRET.encode().hex()} rev={SECRET[::-1]}"
R.check(SECRET not in svc.scrub(page) and svc.scrub(page).count("[REDACTED:site_token]") == 4,
        "scrub removes the value, base64, hex and reversed forms")
unknown = {
    "all but the last character": SECRET[:-1],
    "split in two halves": SECRET[:20] + " " + SECRET[20:],
    "upper-cased": SECRET.upper(),
    "rot13": codecs.encode(SECRET, "rot13"),
    "base32": base64.b32encode(SECRET.encode()).decode(),
    "base64 of base64": base64.b64encode(base64.b64encode(SECRET.encode())).decode(),
    "hex with colons": ":".join(f"{x:02x}" for x in SECRET.encode()),
    "one character per line": "\n".join(SECRET),
    "html entities": "".join(f"&#{ord(ch)};" for ch in SECRET),
    "utf-16 bytes as text": SECRET.encode("utf-16-le").decode("latin-1"),
}
passed = [k for k, v in unknown.items() if svc.scrub(f"echo: {v}") == f"echo: {v}"]
R.note(f"scrub leaves {len(passed)} of {len(unknown)} simple transformations untouched: {', '.join(passed)} "
       f"(EA status: 'not arbitrary transformations'; note that a value missing ONE character also passes)")
svc.close()
cleanup(d)
R.done()
