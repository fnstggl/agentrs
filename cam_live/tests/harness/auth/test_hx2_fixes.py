"""Fixes of the G2-harness findings at 4d6044a (docs/verification/G2_harness.md), against the real PolicyService with
the default policy, the operator's real planning task (tasks/discord_100_planning.yaml) and the live record
tools/authorize_live.py builds for it (--read 'https://*' --read 'http://*'), signed with a throw-away key.

D1  Discord data and the bot credential only through the Discord adapter (IH-15, IH-14, CU-25)
D3  Discord's developer portal is not opened by the browser (OP-09)
D4  a site's own CAPTCHA is recognised structurally, not by an exact key list (OP-09)
D5  resources.authorized_tools is enforced (IH-08)
D6  page scripts' reads of Discord are refused (OP-09; the browser part is in tests/harness/env/test_hx2_browser.py)
D7  the launched authorization service sees the instance's public address (CU-25)
Nothing is sent anywhere: every decision and every secret release is checked in-process."""
import importlib.util
import os
import shutil

import pytest

from cam_live import contracts as C
from cam_live.harness.auth import PolicyService, liveauth
from cam_live.harness.auth import platform_constraints as PC
from cam_live.harness.auth.vault import SecretError
from ea_helpers import ALLOW, DENY, NEEDS, act, authorize_live, live_task, prim, start

CAM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MB = C.Modality.STRUCTURED_BROWSER
#: a fake credential built at run time (no token-shaped literal in the repository)
CANARY = ".".join(["hx2canary", "notatoken", "z" * 32])
CHAN = "1575680990334947328"


def _authorize_live_tool():
    spec = importlib.util.spec_from_file_location("hx2_authorize_live", os.path.join(CAM, "tools", "authorize_live.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def planning(tmp_path):
    """(service, planning task, guild id) exactly as the operator would start the planning launch."""
    from cam_live.runtime.tasks import load_task
    AL = _authorize_live_tool()
    task = load_task(os.path.join(CAM, "tasks", "discord_100_planning.yaml"), for_run=True)
    svc = PolicyService(str(tmp_path / "run"), fsync=False)
    priv, pub = str(tmp_path / "op.key"), str(tmp_path / "op.pub")
    liveauth.generate_operator_key(priv, pub)
    shutil.copy(pub, os.path.join(svc.keys_dir, "operator.pub"))
    rec = AL.build_record(task, hours=6, owned=[], read=["https://*", "http://*"], preauthorized=[], accounts=[],
                          issued_by="hx2 test")
    svc.register_task(task, "operator")
    svc.register_live_authorization(liveauth.sign_live_authorization(rec, priv), "operator")
    svc.put_secret("discord_bot_token", CANARY, ["discord.com", "gateway.discord.gg", "*.discord.gg"], ["discord"],
                   by="operator")          # the default binding of harness/credentials.py
    yield svc, task, str(task.resources["discord"]["guild_id"])
    svc.close()


def released(svc, action, dec, url):
    if dec.verdict != ALLOW or not svc.check_grant(action, dec):
        return False
    try:
        return svc.secret_get("discord_bot_token", action, dec, url) == CANARY
    except SecretError:
        return False


# ------------------------------------------------------------------------------------------------------------- D1
def test_discord_data_is_reached_only_through_the_adapter_under_the_planning_task(planning):
    svc, task, guild = planning
    bot = {"scheme": "bot", "secret": "secret:discord_bot_token"}
    msgs = f"https://discord.com/api/v10/channels/{CHAN}/messages"
    members = f"https://discord.com/api/v10/guilds/{guild}/members?limit=1000"
    for op, args, modality in [
        ("http.get", {"url": msgs, "auth": bot}, C.Modality.API),
        ("api.get", {"url": msgs, "auth": bot}, C.Modality.API),
        ("http.get", {"url": members, "auth": bot}, C.Modality.API),
        ("http.get", {"url": members}, C.Modality.API),
        ("http.request", {"method": "GET", "url": "https://discordapp.com/api/v9/channels/1/messages"}, C.Modality.API),
        ("http.get", {"url": "https://cdn.discordapp.com/attachments/1/2/file.png"}, C.Modality.API),
        ("browser.navigate", {"url": "https://cdn.discordapp.com/attachments/1/2/notes.pdf"}, MB),
        ("browser.navigate", {"url": "https://media.discordapp.net/attachments/1/2/a.png"}, MB),
        ("sandbox.python", {"code": f"import urllib.request\nurllib.request.urlopen('{msgs}')"}, C.Modality.CODE),
        ("http.get", {"url": "https://docs.example.org/page", "auth": bot}, C.Modality.API),
    ]:
        a = act(op, modality, arguments=args)
        d = svc.authorize(a, task, {})
        assert d.verdict == DENY and not d.grant, (op, args, d.verdict, d.policy_ids, d.reasons)
        assert not released(svc, a, d, args.get("url", msgs))
    # the adapter's own operations are judged by their data-use rules (refused here: no permission in the task)
    d = svc.authorize(act("discord.read_messages", C.Modality.API, arguments={"guild_id": guild, "channel_id": CHAN}),
                      task, {})
    assert d.verdict == DENY and ({"P-DATA-USE", "X-DATA-COLLECTION"} & set(d.policy_ids)), d.policy_ids
    d = svc.authorize(act("discord.list_members", C.Modality.API, arguments={"guild_id": guild}), task, {})
    assert d.verdict == DENY and "X-DATA-COLLECTION" in d.policy_ids, d.policy_ids


def test_the_bot_credential_is_released_only_to_discord_operations(planning):
    svc, task, guild = planning
    # an aggregate read of the adapter: the credential is released for Discord's API
    stats = act("discord.get_guild_stats", C.Modality.API, arguments={"guild_id": guild})
    d = svc.authorize(stats, task, {})
    assert d.verdict == ALLOW, d.reasons
    assert released(svc, stats, d, "https://discord.com/api/v10/guilds/" + guild)
    # a browser action that is allowed (public web readable) cannot obtain it, not even for discord.com
    nav = act("browser.navigate", MB, arguments={"url": "https://discord.com/developers/docs/intro"})
    d = svc.authorize(nav, task, {})
    assert d.verdict == ALLOW, d.reasons
    assert svc.check_grant(nav, d)
    with pytest.raises(SecretError) as e:
        svc.secret_get("discord_bot_token", nav, d, "https://discord.com/api/v10/users/@me")
    assert "released only to the discord adapter" in str(e.value)
    # nor a program unless one of its steps is a Discord operation
    assert PC.credential_platform("secret:discord_bot_token") == "discord"
    assert PC.credential_platform("site_token", ["discord"]) == "discord" and PC.credential_platform("site_token") == ""


def test_the_adapters_own_transport_primitives_stay_covered(svc, tmp_path, clock):
    t = live_task()
    authorize_live(svc, tmp_path, t, clock, owned=["channel:discord:*", "account:discord:guild-000001",
                                                  "https://discord.com/api/*"],
                   read=["discord.com"], preauthorized=["channel:discord:*"])
    msg = act("discord.send_message", C.Modality.API, arguments={"channel_id": "123456789",
                                                                 "content": "release notes are up"})
    start(svc, msg, t)
    url = "https://discord.com/api/v10/channels/123456789/messages"
    assert svc.check_primitive(prim("net.request", C.Modality.API, method="POST", url=url), msg).verdict != DENY
    # the same request reported under a generic HTTP action is refused
    raw = act("http.post", arguments={"url": "http://127.0.0.1:8765/x", "json": {}})
    start(svc, raw, task_for(svc, t))
    r = svc.check_primitive(prim("http.request", C.Modality.API, method="POST", url=url, hop=1), raw)
    assert r.verdict == DENY and PC.PID_ADAPTER in r.policy_ids, r.policy_ids


def task_for(svc, live):
    t = C.TaskSpec(objective_id="obj-hx2-staging", objective="o", mode="staging",
                   resources={"owned": ["127.0.0.1:8765"], "preauthorized": ["127.0.0.1:8765"]})
    return t


# ------------------------------------------------------------------------------------------------------------- D5
def test_the_tasks_authorized_tools_are_enforced(planning):
    svc, task, guild = planning
    assert task.resources["authorized_tools"] == ["browser.*", "sandbox.*", "discord.*", "internal.*", "wait"]
    d = svc.authorize(act("http.get", arguments={"url": "https://docs.example.org/page"}), task, {})
    assert d.verdict == DENY and "P-TOOL-NOT-AUTHORIZED" in d.policy_ids, d.policy_ids
    for op, args, m in [("browser.navigate", {"url": "https://docs.example.org/page"}, MB),
                        ("wait", {"seconds": 1}, C.Modality.INTERNAL),
                        ("internal.note", {"text": "a note"}, C.Modality.INTERNAL),
                        ("discord.get_guild_stats", {"guild_id": guild}, C.Modality.API)]:
        d = svc.authorize(act(op, m, arguments=args), task, {})
        assert "P-TOOL-NOT-AUTHORIZED" not in d.policy_ids, (op, d.policy_ids, d.reasons)
    prog = act("program", C.Modality.INTERNAL, steps=[act("wait", C.Modality.INTERNAL, arguments={"seconds": 1}),
                                                       act("http.get", arguments={"url": "https://docs.example.org/"})])
    d = svc.authorize(prog, task, {})
    assert d.verdict == DENY and "P-TOOL-NOT-AUTHORIZED" in d.policy_ids
    d = svc.authorize(act("skill.run", C.Modality.SKILL, arguments={"skill_id": "x"}), task, {})
    assert d.verdict == DENY and "P-TOOL-NOT-AUTHORIZED" in d.policy_ids


def test_authorized_tools_accept_modality_names_and_an_absent_list_changes_nothing(svc):
    t = C.TaskSpec(objective_id="obj-hx2-tools", objective="o", mode="staging", interfaces=["api", "internal", "code"],
                   resources={"authorized_tools": ["api", "wait"]})
    assert svc.authorize(act("http.get", arguments={"url": "http://127.0.0.1:8765/x"}), t, {}).verdict == ALLOW
    d = svc.authorize(act("sandbox.python", arguments={"code": "print(1)"}), t, {})
    assert d.verdict == DENY and "P-TOOL-NOT-AUTHORIZED" in d.policy_ids
    free = C.TaskSpec(objective_id="obj-hx2-free", objective="o", mode="staging", interfaces=["code"])
    assert svc.authorize(act("sandbox.python", arguments={"code": "print(1)"}), free, {}).verdict == ALLOW


# ------------------------------------------------------------------------------------------------------- D3 / D6
def test_discords_developer_portal_and_page_script_reads_are_refused_in_the_browser(planning):
    svc, task, guild = planning
    U = PC.PID_ACCOUNT
    for url in ("https://discord.com/developers/applications", "https://discord.com/developers/applications/1/bot",
                "https://discord.com/developers/teams", "https://discord.com/developers/servers",
                "https://canary.discord.com/developers/applications/1/oauth2"):
        d = svc.authorize(act("browser.navigate", MB, arguments={"url": url}), task, {})
        assert d.verdict == DENY and U in d.policy_ids, (url, d.policy_ids)
    for url in ("https://discord.com/developers/docs/intro", "https://discord.com/developers/docs/topics/gateway",
                "https://discord.com/guidelines"):
        d = svc.authorize(act("browser.navigate", MB, arguments={"url": url}), task, {})
        assert d.verdict == ALLOW, (url, d.policy_ids, d.reasons)
    nav = act("browser.navigate", MB, arguments={"url": "https://discord.com/developers/docs/intro"})
    start(svc, nav, task)
    for rtype in ("xhr", "fetch"):
        for url in ("https://discord.com/", "https://discord.com/api/v9/users/@me", "https://discord.com/channels/@me"):
            r = svc.check_primitive(prim("browser.request", MB, url=url, method="GET", resource_type=rtype,
                                         session="default"), nav)
            assert r.verdict == DENY and (U in r.policy_ids or PC.PID_ADAPTER in r.policy_ids), (url, r.policy_ids)
    # static subresources of the documentation (styles, images) stay readable
    r = svc.check_primitive(prim("browser.request", MB, url="https://discord.com/", method="GET",
                                 resource_type="stylesheet", session="default"), nav)
    assert r.verdict == ALLOW, r.reasons


# ------------------------------------------------------------------------------------------------------------- D4
def test_a_sites_own_captcha_is_recognised_structurally(svc):
    H = PC.PID_HUMAN
    t = C.TaskSpec(objective_id="obj-hx2-captcha", objective="o", mode="staging",
                   resources={"owned": ["127.0.0.1:8765"], "preauthorized": ["127.0.0.1:8765"]})
    LOOP = "http://127.0.0.1:8765"
    for body in ({"email": "a@owned.example", "captcha_input": "XK4P9"},
                 {"email": "a@owned.example", "recaptcha_response_field": "XK4P9"},
                 {"email": "a@owned.example", "captchaResponseToken": "P1_eyJ0"},
                 {"human_check": "1"}, {"nested": {"Captcha-Answer": "x"}}):
        d = svc.authorize(act("http.post", arguments={"url": LOOP + "/signup", "json": body}), t, {})
        assert d.verdict == DENY and H in d.policy_ids, (body, d.policy_ids)
    d = svc.authorize(act("http.post", arguments={"url": LOOP + "/signup", "body": "email=a%40owned.example&captcha_input=XK4P9"}), t, {})
    assert d.verdict == DENY and H in d.policy_ids
    d = svc.authorize(act("http.get", arguments={"url": LOOP + "/check?recaptcha_response_field=abc"}), t, {})
    assert d.verdict == DENY and H in d.policy_ids
    assert svc.authorize(act("http.post", arguments={"url": LOOP + "/signup", "json": {"email": "a@x.org"}}), t, {}).verdict == ALLOW
    # what the browser reports: the element an input goes to, the fields a page submits
    nav = act("browser.navigate", MB, arguments={"url": LOOP + "/signup"})
    start(svc, nav, t)
    cases = [
        (prim("browser.fill", MB, page_url=LOOP + "/signup", element={"role": "textbox", "name": "CAPTCHA answer"}), DENY),
        (prim("browser.click", MB, page_url=LOOP + "/join",
              element={"role": "checkbox", "name": "I am not a robot (human verification)"}), DENY),
        (prim("browser.click", MB, page_url=LOOP + "/x", element={"role": "button", "name": "Verify you are human"}), DENY),
        (prim("browser.request", MB, url=LOOP + "/signup", method="POST", form_fields=["email", "captcha_input"]), DENY),
        (prim("browser.navigate", MB, url=LOOP + "/join", method="POST", via="page", form_fields=["human_check"]), DENY),
        (prim("browser.fill", MB, page_url=LOOP + "/signup", element={"role": "textbox", "name": "Email"}), ALLOW),
        (prim("browser.click", MB, page_url=LOOP + "/signup", element={"role": "button", "name": "Sign up"}), ALLOW),
        (prim("browser.request", MB, url=LOOP + "/signup", method="POST", form_fields=["email", "name"]), ALLOW),
    ]
    for p, want in cases:
        r = svc.check_primitive(p, nav)
        assert r.verdict == want, (p.op, p.arguments, r.verdict, r.policy_ids, r.reasons)
        if want == DENY:
            assert H in r.policy_ids
    assert PC.is_token_key("recaptcha_response_field") and PC.is_token_key("captchaResponseToken")
    assert not PC.is_token_key("email") and not PC.is_token_key("comment")


# ------------------------------------------------------------------------------------------------------------- D7
def test_the_launcher_passes_the_public_address_to_the_services():
    from cam_live.harness.auth.launcher import ENV_PASSTHROUGH
    assert "PUBLIC_IPADDR" in ENV_PASSTHROUGH
