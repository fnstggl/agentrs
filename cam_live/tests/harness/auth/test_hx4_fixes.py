"""Fixes of the G2-harness round-3 findings at 85101a2 (docs/verification/G2_harness.md "Round 3"). Each test fails
on 85101a2. (R3-M1's race is tested as root in tests/runtime/test_hx4_safe_writes.py.)

R3-m1  a restart no longer rewrites the audit head over a tail removed while the service was down (IH-16)
R3-m2  configuring one's own site's CAPTCHA, an "Are you a bot?" FAQ link and an e-mail code field are not
       refused as human verification; real answers still are (OP-09)
R3-M1  safefs: no symlink in a directory another uid can write is followed (unit level, any uid)"""
import os

import pytest

from cam_live import contracts as C
from cam_live.harness import safefs
from cam_live.harness.auth import PolicyService
from cam_live.harness.auth import platform_constraints as PC
from ea_helpers import ALLOW, DENY, act, prim, start

MB, API = C.Modality.STRUCTURED_BROWSER, C.Modality.API


def _svc_with_entries(run, n=6, clock=None):
    svc = PolicyService(run, fsync=False, clock=clock)
    t = C.TaskSpec(objective_id="obj-hx4", objective="o", mode="staging",
                   limits=C.Limits(duration_s=3600, max_decisions=500, max_external_actions=10))
    for _ in range(n):
        svc.authorize(act("wait", C.Modality.INTERNAL, arguments={"seconds": 1}), t, {})
    return svc, t


def _lines(p):
    return [x for x in open(p, "rb").read().split(b"\n") if x.strip()]


# ------------------------------------------------------------------------------------------------------- R3-m1
def test_a_tail_removed_while_the_service_was_down_stays_detected_after_restarts(tmp_path):
    run = str(tmp_path / "run")
    svc, t = _svc_with_entries(run)
    log = svc.audit.path
    svc.close()
    ls = _lines(log)
    open(log, "wb").write(b"\n".join(ls[:-3]) + b"\n")          # the tail removed while the service is down
    for _ in range(3):                                           # every later start, and appends in between
        s = PolicyService(run, fsync=False)
        s.authorize(act("wait", C.Modality.INTERNAL, arguments={"seconds": 1}), t, {})
        v = s.verify_audit()
        assert not v.ok and any("discontinuity" in e for e in v.errors), v.errors
        s.close()


def test_a_head_rolled_back_far_or_a_replaced_entry_is_reported_and_a_clean_restart_is_not(tmp_path):
    run = str(tmp_path / "far")
    svc, t = _svc_with_entries(run, n=3)
    head = svc.audit.head_path
    old = open(head, "rb").read()
    for _ in range(70):
        svc.authorize(act("wait", C.Modality.INTERNAL, arguments={"seconds": 1}), t, {})
    svc.close()
    open(head, "wb").write(old)                                  # 70+ entries behind: not what a stop leaves
    s = PolicyService(run, fsync=False)
    assert not s.verify_audit().ok
    s.close()
    # a missing head with a non-empty log, after a restart
    run2 = str(tmp_path / "missing")
    svc, t = _svc_with_entries(run2, n=3)
    h2 = svc.audit.head_path
    svc.close()
    os.unlink(h2)
    s = PolicyService(run2, fsync=False)
    assert not s.verify_audit().ok
    s.close()
    # an ordinary stop and start: no alarm
    run3 = str(tmp_path / "clean")
    svc, t = _svc_with_entries(run3, n=3)
    svc.close()
    for _ in range(2):
        s = PolicyService(run3, fsync=False)
        assert s.verify_audit().ok, s.verify_audit().errors
        s.close()


def test_an_out_of_band_anchor_detects_a_log_cut_back_with_its_head(tmp_path):
    """The launcher (root) keeps the head it saw at the last stop; a log and head both rewritten while the service
    was down (what only the auth uid or root could do) no longer contain the anchored entry."""
    from cam_live.harness.auth import AuditLog
    path, head, key = str(tmp_path / "a.jsonl"), str(tmp_path / "a.head"), b"k" * 32
    log = AuditLog(path, key=key, head_path=head, fsync=False)
    for i in range(5):
        log.append("e", float(i), n=i)
    anchor = (log.head()["seq"], log.head()["hash"])
    log.close()
    ls = _lines(path)
    open(path, "wb").write(b"\n".join(ls[:2]) + b"\n")
    os.unlink(head)
    log = AuditLog(path, key=key, head_path=head, fsync=False, anchor=anchor)
    assert log.discontinuity() is not None
    log.close()


# ------------------------------------------------------------------------------------------------------- R3-m2
def test_configuring_a_sites_captcha_and_ordinary_controls_pass(svc):
    t = C.TaskSpec(objective_id="obj-hx4-captcha", objective="o", mode="staging",
                   resources={"owned": ["127.0.0.1:8765"], "preauthorized": ["127.0.0.1:8765"]})
    LOOP = "http://127.0.0.1:8765"
    for body in ({"enable_captcha": True, "captcha_provider": "hcaptcha"},
                 {"site_name": "Reading circle", "require_captcha_on_signup": True},
                 {"captcha_site_key": "10000000-ffff", "captcha_enabled": True}):
        d = svc.authorize(act("http.post", API, arguments={"url": LOOP + "/admin/settings", "json": body}), t, {})
        assert d.verdict == ALLOW, (body, d.policy_ids, d.reasons)
    for body in ({"g-recaptcha-response": "03AF"}, {"captcha_answer": "XK4P9"}, {"captcha": "XK4P9"},
                 {"captcha_input": "x"}):
        d = svc.authorize(act("http.post", API, arguments={"url": LOOP + "/signup", "json": body}), t, {})
        assert d.verdict == DENY and PC.PID_HUMAN in d.policy_ids, body
    nav = act("browser.navigate", MB, arguments={"url": LOOP + "/admin"})
    start(svc, nav, t)
    for op, role, name, want in (("browser.click", "checkbox", "Enable CAPTCHA on sign-up", ALLOW),
                                 ("browser.click", "button", "Get reCAPTCHA keys", ALLOW),
                                 ("browser.click", "link", "Are you a bot?", ALLOW),
                                 ("browser.fill", "textbox", "Type the code shown in the e-mail we sent you", ALLOW),
                                 ("browser.fill", "textbox", "CAPTCHA site key", ALLOW),
                                 ("browser.fill", "textbox", "CAPTCHA answer", DENY),
                                 ("browser.click", "checkbox", "I'm not a robot", DENY),
                                 ("browser.click", "button", "Verify you are human", DENY),
                                 ("browser.click", "checkbox", "Are you a robot?", DENY),
                                 ("browser.fill", "textbox", "Type the characters you see in the image", DENY)):
        r = svc.check_primitive(prim(op, MB, page_url=LOOP + "/admin", element={"role": role, "name": name}), nav)
        assert r.verdict == want, (name, r.verdict, r.policy_ids)


# ------------------------------------------------------------------------------------------------------- R3-M1
def test_safefs_never_follows_a_symlink_in_a_writable_directory(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    os.chmod(d, 0o1777)                                          # writable by others
    victim = tmp_path / "victim"
    victim.write_text("precious")
    os.symlink(str(victim), str(d / "target"))
    with pytest.raises(OSError):
        safefs.write_atomic(str(d / "target"), b"x")
    assert victim.read_text() == "precious"
    os.symlink(str(tmp_path), str(d / "sub"))
    with pytest.raises(OSError):
        safefs.write_atomic(str(d / "sub" / "victim"), b"x")
    with pytest.raises(OSError):
        safefs.read_nofollow(str(d / "target"))
    assert victim.read_text() == "precious"
    safefs.write_atomic(str(d / "ok.json"), b"{}", 0o640)
    assert (d / "ok.json").read_bytes() == b"{}" and oct(os.stat(d / "ok.json").st_mode & 0o777) == "0o640"
    assert not [x for x in os.listdir(d) if x.endswith(".tmp")]
