"""G2-harness probe g13 (round 3, policy level): after the precise human-verification recognition (R2-M1 fix), are
ordinary operations still refused, and are real human-verification answers still refused? Real PolicyService,
default policy, a live task whose record owns and pre-authorizes https://owned.example/* and reads the public web.

Ordinary (must be allowed): configuring the operator's OWN site's CAPTCHA (a settings form, an "Enable CAPTCHA on
sign-up" checkbox, a "Get reCAPTCHA keys" button), a FAQ link "Are you a bot?" on the community's own page, entering
an e-mail confirmation code labelled "Type the code shown in the e-mail", plus the round-2 cases (honeypots, quiz
answers, turnstile counts, "Run security check").
Human verification (must stay refused): a CAPTCHA answer field, "I'm not a robot", g-recaptcha-response, a
self-hosted challenge answer key.

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/g2harness/g13_round3_false_refusals.py
"""
from _g2 import ALLOW, C, Report, cleanup, dec_str, fresh_dir  # noqa: I001

from cam_live.harness.auth import PolicyService
from ea_helpers import authorize_live, live_task

R = Report("g13_round3_false_refusals")
API, MB = C.Modality.API, C.Modality.STRUCTURED_BROWSER
d = fresh_dir("g13")
clk = C.SimClock()
svc = PolicyService(d + "/run", clock=clk, fsync=False)
t = live_task(oid="obj-g13", resources={"authorized_tools": ["browser.*", "http.*", "internal.*", "wait"]})
authorize_live(svc, d, t, clk, owned=["https://owned.example/*"], read=["https://*", "http://*"],
               preauthorized=["https://owned.example/*"])


def act(op, modality, **args):
    return C.ActionSpec(action_id=C.new_id("g2h"), operation=op, modality=modality, arguments=args)


def human(dec):
    return "X-HUMAN-VERIFICATION" in dec.policy_ids


try:
    for body, ordinary in (({"enable_captcha": True, "captcha_provider": "hcaptcha"}, True),
                           ({"site_name": "Reading circle", "require_captcha_on_signup": True}, True),
                           ({"botcheck": "x", "email": "a@owned.example"}, True),
                           ({"challenge_answer": "42"}, True), ({"turnstile_count": 1200}, True),
                           ({"email": "a@owned.example", "g-recaptcha-response": "03AFcWeA"}, False),
                           ({"captcha_answer": "XK4P9"}, False)):
        dec = svc.authorize(act("http.post", API, url="https://owned.example/admin/settings", json=body), t, {})
        if ordinary:
            R.check(not human(dec), f"ordinary form body {sorted(body)} is not refused as human verification",
                    dec_str(dec))
        else:
            R.check(human(dec), f"control: human-verification answer {sorted(body)} stays refused", dec_str(dec))
    nav = act("browser.navigate", MB, url="https://owned.example/admin")
    dn = svc.authorize(nav, t, {})
    assert dn.verdict == ALLOW and svc.check_grant(nav, dn), dn.reasons
    for role, name, ordinary in (("checkbox", "Enable CAPTCHA on sign-up", True),
                                 ("button", "Get reCAPTCHA keys", True),
                                 ("link", "Are you a bot?", True),
                                 ("textbox", "Type the code shown in the e-mail we sent you", True),
                                 ("button", "Run security check", True), ("button", "Turnstile counts", True),
                                 ("textbox", "CAPTCHA answer", False), ("checkbox", "I'm not a robot", False),
                                 ("button", "Verify you are human", False)):
        op = "browser.click" if role in ("button", "link", "checkbox") else "browser.fill"
        args = {"page_url": "https://owned.example/admin", "element": {"role": role, "name": name}}
        if op == "browser.fill":
            args["text"] = "123456"
        p = svc.check_primitive(C.PrimitiveOp(op=op, modality=MB, arguments=args), nav)
        if ordinary:
            R.check(not human(p), f"{op} on the ordinary {role} {name!r} is not refused as human verification",
                    dec_str(p))
        else:
            R.check(human(p), f"control: {op} on {role} {name!r} stays refused", dec_str(p))
finally:
    svc.close()
    cleanup(d)
R.done()
