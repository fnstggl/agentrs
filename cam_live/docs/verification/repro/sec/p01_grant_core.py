"""Core grant properties of PolicyService (real service object, SimClock, no mocks).

Attacks: execute without a grant, replay, reuse after consumption, edit after authorization, forged / truncated /
re-signed token, expiry, unknown tool, malformed policy, policy tampering, internal error.
"""
import base64
import copy
import hashlib
import hmac
import json
import os

from _common import ALLOW, DENY, C, Report, cleanup, fresh_dir
from ea_helpers import act, prim, task
from cam_live.harness.auth import PolicyService

R = Report("p01_grant_core")
d = fresh_dir("p01")
clock = C.SimClock()
svc = PolicyService(d, clock=clock, fsync=False)
t = task()
MB = C.Modality.STRUCTURED_BROWSER

# 1. a primitive with no grant at all
a = act("browser.navigate", arguments={"url": "http://127.0.0.1:8765/"})
dec = svc.check_primitive(prim("browser.navigate", MB, url="http://127.0.0.1:8765/"), a)
R.check(dec.verdict == DENY and "P-NO-ACTIVE-GRANT" in dec.policy_ids, "primitive without any grant is denied")

# 2. authorized but grant not consumed -> primitive still denied
dec = svc.authorize(a, t)
assert dec.verdict == ALLOW, dec.reasons
p = svc.check_primitive(prim("browser.navigate", MB, url="http://127.0.0.1:8765/"), a)
R.check(p.verdict == DENY, "primitive before check_grant(consume) is denied")

# 3. edit after authorization (any field, incl. rationale / action_id)
for field, val in (("rationale", "x"), ("target", "http://127.0.0.1:1/"), ("action_id", "other")):
    b = copy.deepcopy(a)
    setattr(b, field, val)
    R.check(svc.check_grant(b, dec, consume=False) is False, f"grant refuses an action edited in {field!r}")
b = copy.deepcopy(a)
b.arguments["url"] = "http://127.0.0.1:9999/other"
d2 = copy.deepcopy(dec)
d2.action_hash = b.content_hash()
R.check(svc.check_grant(b, d2) is False, "decision re-pointed at an edited action is refused")

# 4. single use + replay
R.check(svc.check_grant(a, dec) is True, "valid grant accepted once")
R.check(svc.check_grant(a, dec) is False, "second consumption refused (single use)")

# 5. forged tokens
body, sig = dec.grant.split(".")
payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
a2 = act("browser.navigate", arguments={"url": "http://127.0.0.1:8765/two"})
forged = dict(payload, h=a2.content_hash(), n="00" * 16)
fb = base64.urlsafe_b64encode(json.dumps(forged, sort_keys=True, separators=(",", ":")).encode()).decode().rstrip("=")
for name, tok in (("payload swapped, old signature", fb + "." + sig),
                  ("signed with a guessed key", fb + "." + base64.urlsafe_b64encode(
                      hmac.new(b"\0" * 32, base64.urlsafe_b64decode(fb + "=" * (-len(fb) % 4)), hashlib.sha256).digest()).decode().rstrip("=")),
                  ("empty", ""), ("garbage", "a.b")):
    fd = C.AuthorizationDecision(verdict=ALLOW, action_hash=a2.content_hash(), grant=tok)
    R.check(svc.check_grant(a2, fd) is False, f"forged grant refused ({name})")

# 6. a grant the executor-side key could mint (the executor holds a key copy): nonce never issued by the service
from cam_live.harness.auth.grants import GrantSigner
with open(os.path.join(d, "auth", "keys", "grant.key"), "rb") as f:
    key = f.read()
minted = GrantSigner(key).issue(a2.content_hash(), t.objective_id, t.content_hash(), clock.now(), clock.now() + 60, 0)
fd = C.AuthorizationDecision(verdict=ALLOW, action_hash=a2.content_hash(), grant=minted["token"])
R.check(svc.check_grant(a2, fd) is False, "well-signed grant with a nonce the service never issued is refused")

# 7. expiry
a3 = act("browser.navigate", arguments={"url": "http://127.0.0.1:8765/three"})
d3 = svc.authorize(a3, t)
clock.advance(svc.policy.grants["ttl_s"] + 1)
R.check(svc.check_grant(a3, d3) is False, "expired grant refused")

# 8. execution window closes: primitives after timeout_s + slack are denied
a4 = act("browser.navigate", arguments={"url": "http://127.0.0.1:8765/four"}, timeout_s=5.0)
d4 = svc.authorize(a4, t)
assert svc.check_grant(a4, d4)
ok1 = svc.check_primitive(prim("browser.navigate", MB, url="http://127.0.0.1:8765/four"), a4).verdict
clock.advance(5.0 + svc.policy.grants["exec_slack_s"] + 1)
ok2 = svc.check_primitive(prim("browser.navigate", MB, url="http://127.0.0.1:8765/four"), a4).verdict
R.check(ok1 == ALLOW and ok2 == DENY, "primitives are allowed only inside the execution window", f"{ok1} then {ok2}")

# 9. unknown tool / unknown modality / malformed
R.check(svc.authorize(act("teleport.now", C.Modality.API), t).verdict == DENY, "unknown tool denied")
R.check(svc.authorize(act("browser.navigate", "telepathy", arguments={"url": "http://127.0.0.1:1/"}), t).verdict == DENY,
        "unknown modality denied")
R.check(svc.authorize("not an action", t).verdict == DENY, "non-ActionSpec denied")           # type: ignore[arg-type]
R.check(svc.authorize(a, "not a task").verdict == DENY, "non-TaskSpec denied")                 # type: ignore[arg-type]

# 10. invalid / tampered policy file -> deny all
svc.close()
bad = os.path.join(d, "bad.yaml")
src = open(os.path.join(os.path.dirname(__import__("cam_live.harness.auth").harness.auth.__file__), "default_policy.yaml")).read()
for name, text in (("unknown key", src + "\nextra_key: 1\n"), ("not yaml", "::: {"), ("empty", ""),
                   ("effect typo", src.replace("effect: external_read", "effect: external_reed", 1)),
                   ("bad regex", src.replace('regex: "-----BEGIN', 'regex: "(((-----BEGIN', 1))):
    with open(bad, "w") as f:
        f.write(text)
    s2 = PolicyService(fresh := fresh_dir("p01b"), policy_path=bad, clock=clock, fsync=False)
    v = s2.authorize(act("wait", C.Modality.INTERNAL), t)
    R.check(v.verdict == DENY and "P-POLICY-INVALID" in v.policy_ids, f"invalid policy ({name}) denies everything")
    s2.close()
    cleanup(fresh)

# 11. valid policy, then the file is replaced by an invalid one and reloaded -> deny all (no stale policy)
good = os.path.join(d, "good.yaml")
open(good, "w").write(src)
s3 = PolicyService(fresh_dir("p01c"), policy_path=good, clock=clock, fsync=False)
assert s3.authorize(act("wait", C.Modality.INTERNAL), t).verdict == ALLOW
open(good, "w").write(src + "\nrogue: true\n")
s3.reload_policy("operator")
v = s3.authorize(act("wait", C.Modality.INTERNAL), t)
R.check(v.verdict == DENY, "reload of a tampered (invalid) policy file leaves the service denying")
run3 = s3.run_dir
s3.close()
cleanup(run3)
cleanup(d)
R.done()
