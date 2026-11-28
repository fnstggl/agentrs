"""Fixes of the G2-harness round-2 findings at 0a30cd8 (docs/verification/G2_harness.md "Round 2"), against the real
PolicyService with the default policy. Each test fails on 0a30cd8.

R2-M1  the structural human-verification recognition is precise: ordinary fields / controls pass, real CAPTCHA
       fields, provider tokens and humanity-proof controls are refused (OP-09)
R2-m1  the Discord adapter's standing belongs to the adapter's own action, not to a program containing one; the
       attachment CDN is Discord data as a whole (IH-15, CU-25)
R2-m2  an origin's read decision names the paths the deny lists name there (the browser judges those per URL)
R2-m3  an audit head left one entry behind by a stop is reconciled at open; tampering is still detected (IH-16)
(R2-C1 is tested in tests/harness/auth/test_hx_sandbox_host_privileged.py, R2-m2 / R2-M1 in Chromium in
tests/harness/env/test_hx3_browser.py.)"""
import json
import os
import shutil

import pytest

from cam_live import contracts as C
from cam_live.harness.auth import AuditLog, PolicyService, verify_audit_log
from cam_live.harness.auth import platform_constraints as PC
from cam_live.harness.auth.vault import SecretError
from ea_helpers import ALLOW, DENY, act, authorize_live, live_task, prim, start

MB, API = C.Modality.STRUCTURED_BROWSER, C.Modality.API
CANARY = ".".join(["hx3canary", "notatoken", "y" * 32])
OWNED = ["https://owned.example/*", "channel:discord:*", "account:discord:900000000000000001"]


@pytest.fixture
def lt(svc, tmp_path, clock):
    t = live_task(resources={"authorized_tools": ["http.*", "api.*", "browser.*", "discord.*", "wait", "program"]})
    authorize_live(svc, tmp_path, t, clock, owned=OWNED, read=["https://*", "http://*"],
                   preauthorized=["https://owned.example/*"], accounts=["account:discord:900000000000000001"])
    svc.put_secret("discord_bot_token", CANARY, ["discord.com", "gateway.discord.gg"], ["discord"], by="operator")
    return t


# ------------------------------------------------------------------------------------------------------- R2-M1
def test_ordinary_fields_and_controls_are_not_taken_for_human_verification(svc, lt):
    for body in ({"name": "Ann", "email": "ann@owned.example", "botcheck": "x"},          # a honeypot
                 {"name": "Ann", "antibot_key": "a1b2"},                                  # Drupal Antibot
                 {"answer": "42", "challenge_answer": "42"},                              # a quiz
                 {"title": "Turnstile counts at the north gate", "turnstile_count": 1200},
                 {"security_check": "passed", "challenge_token": "t1", "bot_detection": "on"}):
        d = svc.authorize(act("http.post", API, arguments={"url": "https://owned.example/api/forms", "json": body}), lt, {})
        assert d.verdict == ALLOW, (body, d.policy_ids, d.reasons)
    nav = act("browser.navigate", MB, arguments={"url": "https://owned.example/settings"})
    start(svc, nav, lt)
    for role, name in (("button", "Run security check"), ("button", "Security checkup"),
                       ("button", "Bot protection settings"), ("link", "Turnstile counts"),
                       ("button", "CAPTCHA settings"), ("textbox", "Your answer"), ("button", "Submit answer")):
        r = svc.check_primitive(prim("browser.click", MB, page_url="https://owned.example/settings",
                                     element={"role": role, "name": name}), nav)
        assert r.verdict == ALLOW, (name, r.policy_ids, r.reasons)
    for fields in (["name", "email", "message", "botcheck"], ["email", "antibot_key"], ["challenge_answer"]):
        r = svc.check_primitive(prim("browser.request", MB, url="https://owned.example/contact", method="POST",
                                     via="page", resource_type="document", form_fields=fields), nav)
        assert r.verdict == ALLOW, (fields, r.policy_ids, r.reasons)


def test_real_human_verification_is_still_refused(svc, lt):
    H = PC.PID_HUMAN
    for body in ({"email": "a@x.org", "captcha_input": "XK4P9"}, {"recaptcha_response_field": "x"},
                 {"captchaResponseToken": "P1"}, {"g-recaptcha-response": "03AF"}, {"h-captcha-response": "P1_e"},
                 {"cf-turnstile-response": "0.x"}, {"human_check": "1"}):
        d = svc.authorize(act("http.post", API, arguments={"url": "https://owned.example/signup", "json": body}), lt, {})
        assert d.verdict == DENY and H in d.policy_ids, (body, d.policy_ids)
    nav = act("browser.navigate", MB, arguments={"url": "https://owned.example/signup"})
    start(svc, nav, lt)
    for op, role, name in (("browser.fill", "textbox", "CAPTCHA answer"), ("browser.fill", "textbox", "Captcha"),
                           ("browser.click", "checkbox", "I am not a robot (human verification)"),
                           ("browser.click", "button", "Verify you are human"),
                           ("browser.click", "button", "Get a new CAPTCHA challenge"),
                           ("browser.click", "button", "Refresh captcha"),
                           ("browser.click", "button", "Select all images with traffic lights")):
        r = svc.check_primitive(prim(op, MB, page_url="https://owned.example/signup",
                                     element={"role": role, "name": name}), nav)
        assert r.verdict == DENY and H in r.policy_ids, (name, r.verdict, r.policy_ids)
    r = svc.check_primitive(prim("browser.request", MB, url="https://owned.example/signup", method="POST", via="page",
                                 form_fields=["email", "captcha_input"]), nav)
    assert r.verdict == DENY and H in r.policy_ids
    assert not PC.is_token_key("botcheck") and not PC.is_token_key("antibot_key")
    assert not PC.is_token_key("challenge_answer") and not PC.is_token_key("turnstile_count")
    assert PC.is_token_key("cf-turnstile-response") and PC.is_token_key("captcha_input")


# ------------------------------------------------------------------------------------------------------- R2-m1
def test_a_discord_step_does_not_lend_the_adapters_standing_to_its_program(svc, lt):
    stats = act("discord.get_guild_stats", API, arguments={"guild_id": "900000000000000001"})
    fetch = act("http.get", API, arguments={"url": "https://docs.example.org/r"})
    prog = act("program", C.Modality.INTERNAL, steps=[stats, fetch])
    d = svc.authorize(prog, lt, {})
    assert d.verdict == ALLOW, d.reasons
    assert svc.check_grant(prog, d)
    for u in ("https://discord.com/api/v10/channels/1/messages", "https://cdn.discordapp.com/attachments/1/2/f.txt"):
        hop = svc.check_primitive(prim("http.request", API, url=u, method="GET", via="redirect", hop=1), prog)
        assert hop.verdict == DENY and PC.PID_ADAPTER in hop.policy_ids, (u, hop.policy_ids)
    with pytest.raises(SecretError):
        svc.secret_get("discord_bot_token", prog, d, "https://discord.com/api/v10/guilds/900000000000000001")
    # the adapter's own action still gets it
    d2 = svc.authorize(stats, lt, {})
    assert d2.verdict == ALLOW and svc.check_grant(stats, d2)
    assert svc.secret_get("discord_bot_token", stats, d2, "https://discord.com/api/v10/guilds/1") == CANARY


def test_images_of_the_attachment_cdn_are_not_loaded_by_pages(svc, lt):
    pg = act("browser.navigate", MB, arguments={"url": "https://docs.example.org/gallery"})
    start(svc, pg, lt)
    for host in ("https://cdn.discordapp.com/", "https://media.discordapp.net/"):
        r = svc.check_primitive(prim("browser.request", MB, url=host, method="GET", resource_type="image",
                                     session="default"), pg)
        assert r.verdict == DENY and PC.PID_ADAPTER in r.policy_ids, (host, r.policy_ids)


# ------------------------------------------------------------------------------------------------------- R2-m2
def test_an_origins_read_decision_names_its_denied_paths(svc):
    t = C.TaskSpec(objective_id="obj-hx3-paths", objective="o", mode="staging",
                   resources={"owned": ["127.0.0.1:8765"], "preauthorized": ["127.0.0.1:8765"],
                              "deny": ["127.0.0.1:8765/private/*", "127.0.0.1:8765/admin"]})
    LOOP = "http://127.0.0.1:8765"
    nav = act("browser.navigate", MB, arguments={"url": LOOP + "/public"})
    start(svc, nav, t)
    r = svc.check_primitive(prim("browser.request", MB, url=LOOP + "/", method="GET", resource_type="image",
                                 session="default"), nav)
    assert r.verdict == ALLOW
    hint = [x for x in r.reasons if x.startswith(PolicyService.DENY_PATHS_PREFIX)]
    assert hint and set(json.loads(hint[0][len(PolicyService.DENY_PATHS_PREFIX):])) == {"/private/*", "/admin"}
    r2 = svc.check_primitive(prim("browser.request", MB, url=LOOP + "/private/pixel.png", method="GET",
                                  resource_type="image", session="default"), nav)
    assert r2.verdict == DENY
    from cam_live.harness.browser import _under_paths
    assert _under_paths(LOOP + "/private/pixel.png", ["/private/*"]) and _under_paths(LOOP + "/%70rivate/x", ["/private/*"])
    assert _under_paths(LOOP + "/a/../private/x", ["/private/*"]) and not _under_paths(LOOP + "/public/x.png", ["/private/*"])


# ------------------------------------------------------------------------------------------------------- R2-m3
def _lagging(tmp_path, n=4):
    path, head = str(tmp_path / "audit.jsonl"), str(tmp_path / "audit.head")
    key = b"k" * 32
    log = AuditLog(path, key=key, head_path=head, fsync=False)
    for i in range(n - 1):
        log.append("decision", float(i), n=i)
    saved = open(head, "rb").read()
    log.append("decision", float(n), n=n)              # line written and synced, then the process stops ...
    log.close()
    with open(head, "wb") as f:                         # ... before the head file was replaced
        f.write(saved)
    return path, head, key


def test_a_head_left_behind_by_a_stop_is_reconciled_at_open(tmp_path):
    path, head, key = _lagging(tmp_path)
    assert not verify_audit_log(path, key, head_path=head).ok          # the false alarm of G2-harness R2-m3
    AuditLog(path, key=key, head_path=head, fsync=False).close()        # the service starts again
    v = verify_audit_log(path, key, head_path=head)
    assert v.ok and v.entries == 4, v.errors


def test_reconciliation_never_hides_tampering(tmp_path):
    key = b"k" * 32
    # an entry appended after the head without the key (bad MAC): not reconciled
    path, head, _ = _lagging(tmp_path / "a")
    lines = open(path, "rb").read().splitlines()
    e = json.loads(lines[-1])
    e["mac"] = "0" * 64
    lines[-1] = json.dumps(e, sort_keys=True, separators=(",", ":")).encode()
    open(path, "wb").write(b"\n".join(lines) + b"\n")
    AuditLog(path, key=key, head_path=head, fsync=False).close()
    assert not verify_audit_log(path, key, head_path=head).ok
    # a truncated log (head ahead): not reconciled
    path, head, _ = _lagging(tmp_path / "b")
    AuditLog(path, key=key, head_path=head, fsync=False).close()        # reconciled: head = 4
    lines = open(path, "rb").read().splitlines()
    open(path, "wb").write(b"\n".join(lines[:-2]) + b"\n")
    AuditLog(path, key=key, head_path=head, fsync=False).close()
    assert not verify_audit_log(path, key, head_path=head).ok
    # an edited entry after the head: not reconciled
    path, head, _ = _lagging(tmp_path / "c")
    lines = open(path, "rb").read().splitlines()
    e = json.loads(lines[-1])
    e["data"] = {"n": 999}
    lines[-1] = json.dumps(e, sort_keys=True, separators=(",", ":")).encode()
    open(path, "wb").write(b"\n".join(lines) + b"\n")
    AuditLog(path, key=key, head_path=head, fsync=False).close()
    assert not verify_audit_log(path, key, head_path=head).ok


def test_verifying_while_the_service_appends_never_raises_a_false_alarm(tmp_path):
    """The relaunched service appends its start entries while a client verifies (G2-harness g10): the log and the head
    are read under the log's lock, so the moment between an entry and its head is never seen."""
    import threading
    path, head, key = str(tmp_path / "audit.jsonl"), str(tmp_path / "audit.head"), b"k" * 32
    log = AuditLog(path, key=key, head_path=head, fsync=False)
    log.append("service_start", 0.0)
    stop = threading.Event()

    def writer():
        for i in range(400):                   # bounded: each verification reads the whole log
            if stop.is_set():
                break
            log.append("decision", float(i), n=i, pad="x" * 200)

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        bad, n = [], 0
        while (t.is_alive() or n < 20) and n < 300:
            v = verify_audit_log(path, key, head_path=head)
            n += 1
            if not v.ok:
                bad.append(v.errors)
    finally:
        stop.set()
        t.join(5)
        log.close()
    assert bad == [], bad[:3]
