"""Pause and emergency stop: checked first, across processes, across restarts, operator-only reset."""
import json
import os
import subprocess
import sys

import pytest

from cam_live import contracts as C
from cam_live.harness.auth import PolicyService
from cam_live.harness.auth.stop import StopSwitch, is_stopped
from ea_helpers import ALLOW, DENY, LOOP, SRC, act, ok_result, prim, reopen, start, task

EVERY_MODALITY = [
    lambda: act("browser.navigate", arguments={"url": LOOP}),
    lambda: act("visual.screenshot"),
    lambda: act("sandbox.python", arguments={"code": "1"}),
    lambda: act("http.get", arguments={"url": LOOP}),
    lambda: act("skill.run", arguments={"skill_id": "s"}),
    lambda: act("wait"),
]


def test_stop_blocks_every_modality_immediately(svc):
    t = task()
    running = [(a, start(svc, a, t)) for a in (mk() for mk in EVERY_MODALITY)]      # six executions in progress
    waiting = act("http.get", arguments={"url": LOOP + "/later"})
    d_wait = svc.authorize(waiting, t, {})
    svc.put_secret("site_token", "tok_Zx81kQ0pLm4RvT7yWb2N", ["127.0.0.1:8765"])
    assert not svc.stopped()
    svc.emergency_stop("operator pressed the button", "olga")
    assert svc.stopped() and svc.limits_state()["stopped"] is True
    for mk in EVERY_MODALITY:
        d = svc.authorize(mk(), t, {})
        assert d.verdict == DENY and d.policy_ids == ["P-STOP"] and not d.grant
    assert svc.check_grant(waiting, d_wait) is False                                 # a grant not yet used
    for a, d in running:                                                             # actions already executing
        p = svc.check_primitive(prim("net.request", a.modality, method="GET", url=LOOP), a)
        assert p.verdict == DENY and p.policy_ids == ["P-STOP"]
    a0, d0 = running[3]
    with pytest.raises(PermissionError):
        svc.secret_get("site_token", a0, d0, LOOP)
    assert svc.resolve_approval("apr_x", True, "olga").verdict == DENY


def test_stop_survives_restart_and_reset_is_explicit(svc, clock):
    t = task()
    a = act("http.get", arguments={"url": LOOP})
    d = start(svc, a, t)
    svc.emergency_stop("power cut drill", "olga")
    s2 = reopen(svc, clock)                                                          # service restarted
    assert s2.stopped()
    assert s2.authorize(act("wait"), t, {}).policy_ids == ["P-STOP"]
    assert s2.limits_state()["stop"]["reason"] == "power cut drill"
    assert s2.reset_stop("olga") is True
    assert not s2.stopped()
    assert s2.authorize(act("wait"), t, {}).verdict == ALLOW
    # what was running before the stop does not resume by itself
    assert s2.check_primitive(prim("net.request", C.Modality.API, method="GET", url=LOOP), a).policy_ids == ["P-NO-ACTIVE-GRANT"]
    assert s2.check_grant(a, d) is False
    ev = [e["event"] for e in s2.audit_tail(50)]
    assert "emergency_stop_reset" in ev
    s2.close()


def test_flag_set_by_another_process_is_seen_by_the_next_decision(svc, run_dir):
    t = task()
    assert svc.authorize(act("wait"), t, {}).verdict == ALLOW
    code = ("import sys, time; sys.path.insert(0, sys.argv[1])\n"
            "from cam_live.harness.auth.stop import StopSwitch\n"
            "StopSwitch(sys.argv[2]).engage('from another process', 'operator-shell', time.time())\n")
    subprocess.run([sys.executable, "-c", code, SRC, os.path.join(run_dir, "auth")], check=True)
    assert is_stopped(run_dir)
    d = svc.authorize(act("wait"), t, {})                                            # no restart, no notification
    assert d.verdict == DENY and "from another process" in d.reasons[0]
    assert svc.stopped()


def test_a_bare_or_corrupt_flag_file_still_stops(svc, run_dir):
    with open(os.path.join(run_dir, "auth", "STOP"), "w") as f:
        f.write("")                                                                  # `touch R/auth/STOP`
    assert svc.stopped() and svc.authorize(act("wait"), task(), {}).verdict == DENY
    with open(os.path.join(run_dir, "auth", "STOP"), "w") as f:
        f.write("\x00\xff not json")
    assert svc.stopped()


def test_first_cause_is_kept(svc):
    svc.emergency_stop("first", "a")
    svc.emergency_stop("second", "b")
    assert svc.limits_state()["stop"]["reason"] == "first"
    assert svc.limits_state()["epoch"] == 1                                          # grants voided once


def test_pause_blocks_new_work_and_resume_restores_it(svc, clock):
    t = task()
    a = act("http.get", arguments={"url": LOOP})
    d = svc.authorize(a, t, {})
    svc.pause("operator is reviewing", "olga")
    assert svc.paused() and not svc.stopped()
    dd = svc.authorize(act("wait"), t, {})
    assert dd.verdict == DENY and dd.policy_ids == ["P-PAUSE"]
    assert svc.check_grant(a, d) is False
    s2 = reopen(svc, clock)
    assert s2.paused()                                                               # survives restart too
    assert s2.resume("olga") is True
    assert s2.check_grant(a, d) is True                                              # a pause, unlike a stop, voids nothing
    assert s2.check_primitive(prim("net.request", C.Modality.API, method="GET", url=LOOP), a).verdict == ALLOW
    s2.close()


def test_stop_switch_files(tmp_path):
    sw = StopSwitch(str(tmp_path / "auth"))
    assert sw.engaged() is None and sw.blocked() is None and sw.reset() is False
    sw.engage("why", "who", 12.5)
    assert json.load(open(sw.stop_path)) == {"reason": "why", "by": "who", "ts": 12.5}
    assert oct(os.stat(sw.stop_path).st_mode & 0o777) == "0o600"
    assert "why" in sw.blocked()
    assert sw.reset() is True and sw.engaged() is None
