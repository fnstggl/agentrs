"""Emergency stop and pause: do they void grants, block every call, and survive a restart?

DEFECT probed: the stop can be engaged two ways (docs/EA/README.md "Emergency stop": `policy.emergency_stop(...)`
or `touch R/auth/STOP`; stop.py: "can be set by the operator with nothing but a shell when everything else is
down"). Only the API path bumps the revocation epoch (policy.py emergency_stop: `if first: self._void_grants`).
A stop engaged through the file flag voids nothing: after reset-stop, grants issued before the stop are accepted
again and an in-flight action keeps its execution window.
"""
import os

from _common import ALLOW, DENY, C, Report, cleanup, fresh_dir
from ea_helpers import act, prim, reopen, task
from cam_live.harness.auth import PolicyService

R = Report("p02_stop_pause")
MB = C.Modality.STRUCTURED_BROWSER
URL = "http://127.0.0.1:8765/"


def nav(i):
    return act("browser.navigate", arguments={"url": f"{URL}{i}"}, timeout_s=600.0)


# ---------------------------------------------------------------- A. stop through the API
d = fresh_dir("p02a")
clock = C.SimClock()
svc = PolicyService(d, clock=clock, fsync=False)
t = task()
a_unused, a_running = nav(1), nav(2)
d_unused, d_running = svc.authorize(a_unused, t), svc.authorize(a_running, t)
assert svc.check_grant(a_running, d_running)
svc.emergency_stop("probe", "G-sec")
R.check(svc.authorize(nav(3), t).policy_ids == ["P-STOP"], "API stop: authorize denied with P-STOP")
R.check(svc.check_grant(a_unused, d_unused) is False, "API stop: unused grant refused while stopped")
R.check(svc.check_primitive(prim("browser.navigate", MB, url=URL + "2"), a_running).verdict == DENY,
        "API stop: in-flight primitive refused while stopped")
svc = reopen(svc, clock)
R.check(svc.stopped() and svc.authorize(nav(4), t).verdict == DENY, "API stop survives a service restart")
svc.reset_stop("operator")
R.check(svc.check_grant(a_unused, d_unused) is False, "API stop: grant issued before the stop stays void after reset")
R.check(svc.check_primitive(prim("browser.navigate", MB, url=URL + "2"), a_running).verdict == DENY,
        "API stop: in-flight action stays dead after reset")
svc.close()
cleanup(d)

# ---------------------------------------------------------------- B. stop through the documented file flag
d = fresh_dir("p02b")
clock = C.SimClock()
svc = PolicyService(d, clock=clock, fsync=False)
a_unused, a_running = nav(1), nav(2)
d_unused, d_running = svc.authorize(a_unused, t), svc.authorize(a_running, t)
assert svc.check_grant(a_running, d_running)
open(os.path.join(d, "auth", "STOP"), "w").close()                 # `touch R/auth/STOP`
R.check(svc.stopped() and svc.authorize(nav(3), t).verdict == DENY, "file-flag stop: seen by the next decision")
R.check(svc.check_grant(a_unused, d_unused) is False, "file-flag stop: unused grant refused while stopped")
epoch_before = svc.limits_state()["epoch"]
svc.emergency_stop("operator also clicks stop in the dashboard", "operator")   # no effect on grants: not 'first'
svc.reset_stop("operator")
clock.advance(1.0)
ok_unused = svc.check_grant(a_unused, d_unused)
ok_running = svc.check_primitive(prim("browser.navigate", MB, url=URL + "2"), a_running).verdict
R.check(ok_unused is False, "file-flag stop: grant issued BEFORE the stop is void after reset",
        f"check_grant -> {ok_unused}; epoch {epoch_before} -> {svc.limits_state()['epoch']}")
R.check(ok_running == DENY, "file-flag stop: action in flight BEFORE the stop is dead after reset",
        f"check_primitive -> {ok_running}")
svc.close()
cleanup(d)

# ---------------------------------------------------------------- C. pause
d = fresh_dir("p02c")
clock = C.SimClock()
svc = PolicyService(d, clock=clock, fsync=False)
a_unused, a_running = nav(1), nav(2)
d_unused, d_running = svc.authorize(a_unused, t), svc.authorize(a_running, t)
assert svc.check_grant(a_running, d_running)
svc.pause("probe", "G-sec")
R.check(svc.authorize(nav(3), t).policy_ids == ["P-PAUSE"], "pause: authorize denied with P-PAUSE")
R.check(svc.check_grant(a_unused, d_unused) is False and
        svc.check_primitive(prim("browser.navigate", MB, url=URL + "2"), a_running).verdict == DENY,
        "pause: grants and in-flight primitives refused while paused")
svc = reopen(svc, clock)
R.check(svc.paused() and svc.authorize(nav(4), t).verdict == DENY, "pause survives a service restart")
svc.resume("operator")
after_unused = svc.check_grant(a_unused, d_unused)
after_running = svc.check_primitive(prim("browser.navigate", MB, url=URL + "2"), a_running).verdict
R.note(f"pause does NOT void grants (by design, stop.py docstring): after resume the pre-pause grant is "
       f"accepted={after_unused}, the in-flight action continues={after_running == ALLOW}")
svc.close()
cleanup(d)
R.done()
