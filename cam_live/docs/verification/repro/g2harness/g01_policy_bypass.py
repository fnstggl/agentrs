"""G2-harness probe g01: try to get past the hard platform constraints (OP-09), the data-use rule (IH-15) and
revocation (IH-17) at the authorization layer, with the real PolicyService and the default policy.

Live records below mirror the planning task (tasks/discord_100_planning.yaml): the public web is readable
("https://*", "http://*"), the api modality is an interface, the bot credential is bound to discord.com
(harness/credentials.py default binding) and no message-content permission is granted.

Run from /workspace/camlive:  .venv/bin/python cam_live/docs/verification/repro/g2harness/g01_policy_bypass.py
"""
from _g2 import ALLOW, DENY, NEEDS, C, Report, cleanup, dec_str, fresh_dir  # noqa: I001

from cam_live.harness.auth import PolicyService
from cam_live.harness.auth import platform_constraints as PC
from ea_helpers import act, authorize_live, live_task, prim, reopen, start, task

R = Report("g01_policy_bypass")
MB, MAPI = C.Modality.STRUCTURED_BROWSER, C.Modality.API
GUILD, CHAN = "1575680989060010034", "1575680990334947328"
TOKEN = ".".join(["MTU3NTY4MTc4MDQ2MjEyOTIzMg", "G2hCan", "canary-bot-token-value-0123456789abcdef"])  # fake canary, built at run time (push protection)


def live_svc(tag, **over):
    d = fresh_dir(tag)
    clock = C.SimClock()
    svc = PolicyService(d + "/run", clock=clock, fsync=False)
    t = live_task(oid=f"obj-{tag}", data_permissions=over.pop("data_permissions", {}))
    rec = dict(owned=["https://owned.example/*", "channel:discord:*", f"account:discord:{GUILD}"],
               read=["https://*", "http://*"], preauthorized=["https://owned.example/*"])
    rec.update(over)
    authorize_live(svc, d, t, clock, **rec)
    return d, clock, svc, t


def refused_by(d, pid):
    return d.verdict == DENY and pid in d.policy_ids and not d.grant


# ---------------------------------------------------------------------------------------------------------------
# A. OP-09: the browser opening Discord pages that run with a user's session
# ---------------------------------------------------------------------------------------------------------------
d, clock, svc, t = live_svc("g01a")
try:
    for url in ("https://discord.com/developers/applications",
                f"https://discord.com/developers/applications/1575681780462129232/bot",
                "https://discord.com/developers/teams"):
        dec = svc.authorize(act("browser.navigate", MB, arguments={"url": url}), t, {})
        R.check(refused_by(dec, PC.PID_ACCOUNT),
                f"browser.navigate to Discord's developer portal (a user-session application) is refused: {url}",
                dec_str(dec))
    dec = svc.authorize(act("browser.navigate", MB, arguments={"url": "https://discord.com/developers/docs/intro"}), t, {})
    R.check(dec.verdict == ALLOW, "control: Discord's developer documentation stays readable", dec_str(dec))
    # What the real browser reports for a page's same-origin GET (browser.py _read_allowed: origin + "/", once
    # per origin and action) versus what tests/harness/env/test_hx_platform_constraints.py checks (the API URL).
    nav = act("browser.navigate", MB, arguments={"url": "https://discord.com/developers/docs/intro"})
    start(svc, nav, t)
    as_tested = svc.check_primitive(prim("browser.request", MB, url="https://discord.com/api/v9/users/@me",
                                         method="GET", via="page", resource_type="xhr"), nav)
    as_sent = svc.check_primitive(prim("browser.request", MB, url="https://discord.com/", method="GET",
                                       resource_type="xhr", session="default"), nav)
    R.note(f"same-origin XHR of a Discord page to /api/v9/users/@me: as HX's test reports it -> {dec_str(as_tested)}; "
           f"as browser.py actually reports it (origin + '/', resource_type xhr) -> {dec_str(as_sent)}")
    R.check(as_sent.verdict != ALLOW,
            "a Discord page's own same-origin read of Discord's API (user session) is refused as the browser reports it",
            f"browser.py:1512-1521 judges reads per origin; reported url 'https://discord.com/' -> {dec_str(as_sent)}")
    # spelling variants: informational (Discord's own routing of these spellings cannot be checked offline)
    for url in ("https://discord.com/LOGIN", "https://discord.com/%6Cogin", "https://discord.com/Channels/@me"):
        dec = svc.authorize(act("browser.navigate", MB, arguments={"url": url}), t, {})
        R.note(f"spelling variant {url} -> {dec_str(dec)}")
finally:
    svc.close()
    cleanup(d)

# ---------------------------------------------------------------------------------------------------------------
# B. IH-15: message text through the api modality with the bot credential, without any content permission
# ---------------------------------------------------------------------------------------------------------------
d, clock, svc, t = live_svc("g01b")
try:
    svc.put_secret("discord_bot_token", TOKEN, ["discord.com", "gateway.discord.gg", "*.discord.gg"], ["discord"],
                   by="operator")
    ctl = svc.authorize(act("discord.read_messages", MAPI, arguments={"guild_id": GUILD, "channel_id": CHAN}), t, {})
    R.check(ctl.verdict != ALLOW, "control: discord.read_messages without read_message_content is refused", dec_str(ctl))
    for op in ("http.get", "api.get"):
        url = f"https://discord.com/api/v10/channels/{CHAN}/messages?limit=100"
        a = act(op, MAPI, arguments={"url": url, "auth": {"scheme": "bot", "secret": "secret:discord_bot_token"}})
        dec = svc.authorize(a, t, {})
        released = ""
        if dec.verdict == ALLOW and svc.check_grant(a, dec):
            try:
                v = svc.secret_get("discord_bot_token", a, dec, url)
                released = "bot token RELEASED for it" if v == TOKEN else "other value"
            except PermissionError as e:
                released = f"secret refused: {e}"
        R.check(dec.verdict != ALLOW,
                f"{op} of a channel's message history with the bot credential needs the message-content permission "
                f"(data-use rule of IH-15 / PL-DISCORD-MESSAGE-CONTENT)", f"{dec_str(dec)}; {released}")
    ctl = svc.authorize(act("discord.list_members", MAPI, arguments={"guild_id": GUILD}), t, {})
    R.check(ctl.verdict != ALLOW, "control: discord.list_members without member_list / personal-data permission is "
                                  "refused", dec_str(ctl))
    url = f"https://discord.com/api/v10/guilds/{GUILD}/members?limit=1000"
    a = act("http.get", MAPI, arguments={"url": url, "auth": {"scheme": "bot", "secret": "secret:discord_bot_token"}})
    dec = svc.authorize(a, t, {})
    R.check(dec.verdict != ALLOW, "http.get of the server's member list (user names, ids) with the bot credential "
                                  "needs the member_list / personal-data permission", dec_str(dec))
    # the same request with a train / store flag, which the data-use table does read
    a = act("http.get", MAPI, arguments={"url": f"https://discord.com/api/v10/channels/{CHAN}/messages",
                                         "auth": {"scheme": "bot", "secret": "secret:discord_bot_token"},
                                         "store": True, "train": True})
    dec = svc.authorize(a, t, {})
    R.note(f"the same http.get with store/train flags set -> {dec_str(dec)}")
finally:
    svc.close()
    cleanup(d)

# ---------------------------------------------------------------------------------------------------------------
# C. OP-09: answers to a CAPTCHA under field names the fixed list does not contain
# ---------------------------------------------------------------------------------------------------------------
d, clock, svc, t = live_svc("g01c")
try:
    ctl = svc.authorize(act("http.post", MAPI, arguments={"url": "https://owned.example/signup",
                                                          "json": {"email": "a@owned.example",
                                                                   "g-recaptcha-response": "03AFcWeA"}}), t, {})
    R.check(refused_by(ctl, PC.PID_HUMAN), "control: a g-recaptcha-response answer is refused", dec_str(ctl))
    for args in ({"json": {"email": "a@owned.example", "captcha_input": "XK4P9"}},
                 {"json": {"email": "a@owned.example", "recaptcha_response_field": "XK4P9"}},
                 {"json": {"email": "a@owned.example", "captchaResponseToken": "P1_eyJ0"}},
                 {"body": "email=a%40owned.example&captcha_input=XK4P9"}):
        dec = svc.authorize(act("http.post", MAPI, arguments=dict(url="https://owned.example/signup", **args)), t, {})
        R.check(refused_by(dec, PC.PID_HUMAN), f"a CAPTCHA answer under the key {list(args.values())[0]!r} is refused",
                dec_str(dec))
finally:
    svc.close()
    cleanup(d)

# ---------------------------------------------------------------------------------------------------------------
# D. IH-17: revocation by destination (any spelling), task, secret, live record; in flight; after a restart
# ---------------------------------------------------------------------------------------------------------------
d = fresh_dir("g01d")
clock = C.SimClock()
svc = PolicyService(d + "/run", clock=clock, fsync=False)
try:
    svc.register_local_service("127.0.0.1:8766", "operator", note="second owned local service of this probe")
    st = task(oid="obj-g01d", resources={"owned": ["127.0.0.1:8765", "127.0.0.1:8766"],
                                         "preauthorized": ["127.0.0.1:8765", "127.0.0.1:8766"]},
              limits=C.Limits(duration_s=3600, max_decisions=200, max_external_actions=200))
    svc.register_task(st, "operator")
    svc.put_secret("site_pw", "Zq7-canary-site-password-91", ["127.0.0.1:8765"], ["site"], by="operator")
    a0 = act("http.post", MAPI, arguments={"url": "http://127.0.0.1:8765/notes", "json": {"n": 1}})
    start(svc, a0, st)
    svc.revoke("destination:127.0.0.1:8765", "operator")
    inflight = svc.check_primitive(prim("http.request", MAPI, url="http://127.0.0.1:8765/notes", method="POST"), a0)
    R.check(inflight.verdict == DENY, "an action in flight stops at its next primitive after a destination revocation",
            dec_str(inflight))
    for url in ("http://127.0.0.1:8765/x", "http://localhost:8765/x", "http://[::1]:8765/x", "http://127.1:8765/x",
                "http://LOCALHOST.:8765/x", "http://0x7f000001:8765/x"):
        dec = svc.authorize(act("http.post", MAPI, arguments={"url": url, "json": {}}), st, {})
        R.check(dec.verdict == DENY, f"revoked destination refused under the spelling {url}", dec_str(dec))
    other = svc.authorize(act("http.post", MAPI, arguments={"url": "http://127.0.0.1:8766/x", "json": {}}), st, {})
    R.check(other.verdict == ALLOW, "control: the other owned local service is still allowed", dec_str(other))
    svc = reopen(svc, clock)
    dec = svc.authorize(act("http.post", MAPI, arguments={"url": "http://localhost:8765/y", "json": {}}), st, {})
    R.check(dec.verdict == DENY, "the destination revocation survives a restart of the service", dec_str(dec))
    a1 = act("http.post", MAPI, arguments={"url": "http://127.0.0.1:8766/z", "json": {}})
    dec1 = start(svc, a1, st)
    svc.revoke("secret:site_pw", "operator")
    try:
        svc.secret_get("site_pw", a1, dec1, "http://127.0.0.1:8766/z")
        R.defect("a revoked secret is not released")
    except PermissionError as e:
        R.hold("a revoked secret is not released", str(e)[:100])
    svc.revoke(f"task:{st.objective_id}", "operator")
    dec = svc.authorize(act("http.post", MAPI, arguments={"url": "http://127.0.0.1:8766/z", "json": {}}), st, {})
    R.check(dec.verdict == DENY, "a revoked task gets no further grant", dec_str(dec))
    try:
        svc.revoke("everything", "operator")
        R.defect("an unknown revocation scope is an error, not a silent no-op")
    except ValueError:
        R.hold("an unknown revocation scope is an error, not a silent no-op")
    kinds = [e.get("kind") or e.get("event") for e in svc.audit_tail(200)]
    R.check(sum(1 for k in kinds if k == "revocation") >= 3, "every revocation is audited",
            f"{sum(1 for k in kinds if k == 'revocation')} revocation entries")
    v = svc.verify_audit()
    R.check(bool(getattr(v, "ok", False)), "the audit chain verifies after the revocations", str(v)[:160])
finally:
    svc.close()
    cleanup(d)

d, clock, svc, t = live_svc("g01e")
try:
    svc.revoke("live:rec-1", "operator")
    dec = svc.authorize(act("browser.navigate", MB, arguments={"url": "https://owned.example/a"}), t, {})
    R.check(dec.verdict == DENY, "a revoked live record authorizes nothing", dec_str(dec))
finally:
    svc.close()
    cleanup(d)

R.done()
