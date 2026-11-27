"""Live mode needs a signed operator record: missing, forged, altered, expired, wrong task version, revoked.
Also: what "signed operator record" is worth when the service object lives in the learner's own process.
"""
import base64
import copy
import os

from _common import ALLOW, DENY, C, Report, cleanup, fresh_dir
from ea_helpers import act, authorize_live, install_operator_key, live_record, live_task
from cam_live.harness.auth import PolicyService, liveauth

R = Report("p08_live_record")
d = fresh_dir("p08")
clock = C.SimClock()
svc = PolicyService(d, clock=clock, fsync=False)
t = live_task(oid="obj-live-1")
wait = lambda: act("wait", C.Modality.INTERNAL)          # noqa: E731

v = svc.authorize(wait(), t)
R.check(v.verdict == DENY and "P-LIVE-AUTH" in v.policy_ids, "live task without a record is denied (even an internal wait)")
priv = install_operator_key(svc, d)
good = liveauth.sign_live_authorization(live_record(t, clock.now()), priv)


def rejected(rec, what):
    try:
        svc.register_live_authorization(rec, "operator")
    except liveauth.LiveAuthorizationError:
        R.hold(f"{what} is rejected at registration")
        return
    v = svc.authorize(wait(), t)
    R.check(v.verdict == DENY, f"{what} is rejected", f"registered; authorize -> {v.verdict}")


unsigned = {k: v for k, v in good.items() if k != "signature"}
rejected(unsigned, "an unsigned record")
alt = copy.deepcopy(good); alt["budgets"]["money"] = 1e9
rejected(alt, "a record with an altered budget")
alt = copy.deepcopy(good); alt["owned"].append("*")
rejected(alt, "a record with an added destination")
alt = copy.deepcopy(good); alt["not_after"] += 10 * 86400
rejected(alt, "a record with an extended time window")
other_priv, other_pub = os.path.join(d, "attacker.key"), os.path.join(d, "attacker.pub")
liveauth.generate_operator_key(other_priv, other_pub)
rejected(liveauth.sign_live_authorization(live_record(t, clock.now()), other_priv), "a record signed with another key")
alt = copy.deepcopy(good); alt["signature"] = base64.b64encode(b"\0" * 64).decode()
rejected(alt, "a record with a zero signature")
alt = copy.deepcopy(good); alt["extra"] = 1
rejected(alt, "a record with an unknown field")

svc.register_live_authorization(good, "operator")
R.check(svc.authorize(wait(), t).verdict == ALLOW, "the genuine record enables the task")
t_edit = C.TaskSpec.from_dict(dict(t.to_dict(), objective="a different objective"))
v = svc.authorize(wait(), t_edit)
R.check(v.verdict == DENY and "P-LIVE-AUTH" in v.policy_ids, "an edited live task is not covered by the record")
t_other = live_task(oid="obj-live-2")
R.check(svc.authorize(wait(), t_other).verdict == DENY, "the record does not cover another objective id")
# expiry, on a second service so that the clock of the main one stays inside the window
dx = fresh_dir("p08x")
cx = C.SimClock()
sx = PolicyService(dx, clock=cx, fsync=False)
authorize_live(sx, dx, t, cx)
assert sx.authorize(wait(), t).verdict == ALLOW
cx.advance(3601.0)
v = sx.authorize(wait(), t)
R.check(v.verdict == DENY and ("L-WINDOW" in v.policy_ids or "L-DURATION" in v.policy_ids),
        "an expired record denies", ",".join(v.policy_ids))
sx.close()
cleanup(dx)
svc.revoke("live:rec-1", "operator")
R.check(svc.authorize(wait(), t).verdict == DENY, "a revoked record denies")

# a not-yet-valid record
t3 = live_task(oid="obj-live-3")
early = liveauth.sign_live_authorization(live_record(t3, clock.now(), record_id="rec-3", not_before=clock.now() + 100,
                                                     not_after=clock.now() + 200), priv)
svc.register_live_authorization(early, "operator")
v = svc.authorize(wait(), t3)
R.check(v.verdict == DENY and "L-WINDOW" in v.policy_ids, "a record whose window has not started denies")

# window is checked by authorize; what about an action already granted when the window ends?
t4 = live_task(oid="obj-live-4")
rec4 = liveauth.sign_live_authorization(live_record(t4, clock.now(), record_id="rec-4", not_after=clock.now() + 50), priv)
svc.register_live_authorization(rec4, "operator")
a = act("browser.navigate", C.Modality.STRUCTURED_BROWSER, arguments={"url": "https://owned.example/x"}, timeout_s=800.0)
g = svc.authorize(a, t4)
assert g.verdict == ALLOW and svc.check_grant(a, g)
clock.advance(600.0)                                     # 550 s after the operator's window closed
from ea_helpers import prim
p = svc.check_primitive(prim("net.request", C.Modality.STRUCTURED_BROWSER, method="GET", url="https://owned.example/x"), a)
R.note(f"primitive {int(600 - 50)} s after the record's not_after, inside the action's execution window -> {p.verdict} "
       f"{p.policy_ids} (check_primitive / secret_get do not look at the window; bounded by exec_window_s = "
       f"{svc.policy.grants['exec_window_s']:g} s)")
svc.close()
cleanup(d)

# --------------------------------------------------- in-process service: who is "the operator"?
d = fresh_dir("p08b")
clock = C.SimClock()
svc = PolicyService(d, clock=clock, fsync=False)           # what tests, diagnostics and staging use
t = live_task(oid="obj-self")
mine_priv, mine_pub = os.path.join(d, "mine.key"), os.path.join(d, "mine.pub")
liveauth.generate_operator_key(mine_priv, mine_pub)
import shutil
shutil.copy(mine_pub, os.path.join(svc.keys_dir, "operator.pub"))          # same uid: nothing stops this
svc.register_live_authorization(liveauth.sign_live_authorization(
    live_record(t, clock.now(), owned=["*"], read=["*"], preauthorized=["*"]), mine_priv), "learner")
v = svc.authorize(act("http.post", C.Modality.API, arguments={"url": "https://anything.example/x", "json": {"a": 1}}), t)
R.note(f"single-uid / in-process mode: a caller holding the PolicyService object installed its own operator key, "
       f"signed a record for '*' and got {v.verdict} for a live write. Stated in SECURITY.md section 2 ('single-uid "
       f"development mode ... nothing in section 1 applies'); live mode is not refused in that mode by this layer.")
svc.close()
cleanup(d)
R.done()
