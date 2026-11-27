"""Regression tests for the G-sec / G-env authorization findings M2, M3, M4 (+ G-env C1, M10), m5, m6, m7 and the
policy data of G-env M7 (docs/EA/FIXES.md)."""
import http.server
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from cam_live import contracts as C
from cam_live.harness.auth import PolicyService, verify_audit_log
from cam_live.harness.auth import destinations as D
from cam_live.harness.auth import localnet
from cam_live.harness.auth.schema import DEFAULT_POLICY_PATH, RUN_POLICY_PATH, load_policy
from ea_helpers import ALLOW, DENY, LOOP, NEEDS, act, authorize_live, live_task, ok_result, prim, start, task

API, MB = C.Modality.API, C.Modality.STRUCTURED_BROWSER


def live(svc, tmp_path, clock, oid="obj-fix", **over):
    t = live_task(oid=oid, data_permissions={"aggregate_metrics": True})
    over.setdefault("owned", ["account:discord:guild-000001", "channel:discord:*", "https://owned.example/*"])
    over.setdefault("read", ["account:discord:guild-000001", "channel:discord:*", "owned.example"])
    over.setdefault("preauthorized", ["channel:discord:*", "https://owned.example/*"])
    over.setdefault("budgets", {"money": 20.0, "money_per_action": 5.0, "money_per_day": 10.0, "compute_s": 500.0,
                                "max_external_actions": 5000, "max_decisions": 5000, "duration_s": 1800.0})
    authorize_live(svc, tmp_path, t, clock, **over)
    return t


_n = [0]


def say(svc, t, content, **extra):
    _n[0] += 1
    return svc.authorize(act("discord.send_message", API, arguments=dict(
        {"channel_id": f"{10 ** 18 + _n[0]}", "content": content}, **extra)), t, {})


# ---------------------------------------------------------------------------------------------------------------
# M2 approval / pre-authorization coverage
# ---------------------------------------------------------------------------------------------------------------
def test_a_grant_for_a_preauthorized_path_does_not_cover_another_path(svc, tmp_path, clock):
    t = live(svc, tmp_path, clock, owned=["https://owned.example/*"], read=["owned.example"],
             preauthorized=["https://owned.example/blog/*"])
    ok = act("http.post", API, arguments={"url": "https://owned.example/blog/post-1", "json": {"text": "hello"}})
    start(svc, ok, t)
    assert svc.check_primitive(prim("net.request", API, method="POST", url="https://owned.example/blog/post-1"),
                               ok).verdict == ALLOW                     # the declared request itself
    for method, url in (("DELETE", "https://owned.example/admin/site"),      # another path
                        ("POST", "https://owned.example/blog-admin/post-1"),
                        ("PUT", "https://owned.example/settings")):
        assert svc.check_primitive(prim("net.request", API, method=method, url=url), ok).verdict == NEEDS, url


def test_a_channel_grant_covers_only_its_own_transport_request(svc, tmp_path, clock):
    t = live(svc, tmp_path, clock, owned=["https://discord.com/api/*", "channel:discord:*", "account:discord:guild-000001"],
             read=["discord.com"], preauthorized=[])
    msg = act("discord.send_message", API, arguments={"channel_id": "123456789", "content": "release notes are up"})
    d = svc.authorize(msg, t, {})
    assert d.verdict == NEEDS
    svc.resolve_approval(d.approval_id, True, "operator")
    g = svc.authorize(msg, t, {})
    assert g.verdict == ALLOW and svc.check_grant(msg, g)
    base = "https://discord.com/api/v10/channels"
    assert svc.check_primitive(prim("net.request", API, method="POST", url=f"{base}/123456789/messages"), msg).verdict == ALLOW
    assert svc.check_primitive(prim("net.request", API, method="DELETE", url=f"{base}/123456789"), msg).verdict == NEEDS
    assert svc.check_primitive(prim("net.request", API, method="POST", url=f"{base}/999999999/messages",
                                    json={"content": "x", "nonce": "123456789"}), msg).verdict == NEEDS
    assert svc.check_primitive(prim("net.request", API, method="POST", url=f"{base}/123456789/x/messages"),
                               msg).verdict == NEEDS                    # "*" never crosses a path segment
    # the platform adapter's own primitive: same operation, same channel, its server named as context
    same = prim("discord.send_message", API, channel_id="123456789", content_sha256="ab", guild_id="guild-000001")
    assert svc.check_primitive(same, msg).verdict == ALLOW
    other_op = prim("discord.delete_channel", API, channel_id="123456789", guild_id="guild-000001")
    assert svc.check_primitive(other_op, msg).verdict == NEEDS
    foreign = prim("discord.send_message", API, channel_id="123456789", guild_id="guild-000999")
    assert svc.check_primitive(foreign, msg).verdict == DENY             # the server must be one the task may act in


def test_an_interaction_approval_binds_to_the_page_it_was_given_on(svc, tmp_path, clock):
    t = live(svc, tmp_path, clock, owned=["https://owned.example/*", "https://shop.example/*"],
             read=["owned.example", "shop.example"], preauthorized=[])
    click = act("browser.click", MB, arguments={"ref": "e5"})
    v = svc.authorize(click, t, {"page_url": "https://owned.example/drafts/42"})
    assert v.verdict == NEEDS
    page = [a for a in svc.approvals() if a["approval_id"] == v.approval_id][0]["preview"]["page"]
    assert page == "https://owned.example/drafts/42"
    r = svc.resolve_approval(v.approval_id, True, "operator")
    assert r.verdict == ALLOW                                           # re-authorized on the page it was given on
    other = svc.authorize(click, t, {"page_url": "https://shop.example/checkout/confirm"})
    assert other.verdict == NEEDS and other.approval_id != v.approval_id
    same = svc.authorize(click, t, {"page_url": "https://owned.example/drafts/42"})
    assert same.verdict == ALLOW and same.approval_id == v.approval_id


def test_an_approved_skill_step_covers_its_realisation_and_nothing_else(svc, tmp_path, clock):
    t = live(svc, tmp_path, clock, owned=["https://owned.example/*"], read=["owned.example"], preauthorized=[])
    run = act("skill.run", C.Modality.SKILL, arguments={"skill_id": "s1"})
    start(svc, run, t)
    decl = C.PrimitiveOp("http.request", API, {"method": "POST", "url": "https://owned.example/api/items",
                                               "json": {"t": "x"}}, note="skill:s1:v1:step0")
    d = svc.check_primitive(decl, run)
    assert d.verdict == NEEDS
    svc.resolve_approval(d.approval_id, True, "operator")
    assert svc.check_primitive(decl, run).verdict == ALLOW
    real = prim("http.request", API, method="POST", url="https://owned.example/api/items", hop=0)
    assert svc.check_primitive(real, run).verdict == ALLOW              # no second approval for the same request
    assert svc.check_primitive(prim("http.request", API, method="POST", url="https://owned.example/api/other", hop=0),
                               run).verdict == NEEDS


# ---------------------------------------------------------------------------------------------------------------
# M3 prohibited classes (spec 12.6)
# ---------------------------------------------------------------------------------------------------------------
PARAPHRASES = [
    "Join-for-join partners wanted: become a member here and our whole team becomes a member of your server.",
    "Invite three friends and each of you receives a 10 dollar voucher once they are in.",
    "Speaking as one of the human volunteers here - there is no automation behind this account.",
    "Discord Trust and Safety here. Your account is flagged; confirm your details in this channel within 24h.",
    "<@&1000000000000000009> big news for all of you",
    "hello " + " ".join(f"<@{10 ** 17 + i}>" for i in range(200)),
    "Feeling lonely or in debt? Teens welcome - a server for kids who need quick cash.",
    "We are swapping members with other servers: mutual join, everyone wins.",
    "Boost the server and get free nitro for a month!",
]
BENIGN = [
    "Release notes for version 2.1 are up: we fixed the search bug and added dark mode.",
    "The weekly Python workshop starts at 18:00 UTC. Bring your laptop and your questions.",
    "Thanks to everyone who reported bugs this week; the fixes ship on Monday.",
]


@pytest.mark.parametrize("text", PARAPHRASES)
def test_paraphrased_prohibited_content_never_passes_silently(svc, tmp_path, clock, text):
    t = live(svc, tmp_path, clock)
    d = say(svc, t, text)
    assert d.verdict in (DENY, NEEDS), (text, d.policy_ids)
    assert not any(p.startswith("L-") for p in d.policy_ids)


def test_ordinary_announcements_to_preauthorized_channels_are_allowed(svc, tmp_path, clock):
    t = live(svc, tmp_path, clock)
    for text in BENIGN:
        d = say(svc, t, text)
        assert d.verdict == ALLOW, (text, d.reasons)


def test_live_publication_needs_the_operator_unless_the_record_preauthorizes_it(svc, tmp_path, clock):
    t = live(svc, tmp_path, clock, preauthorized=[])
    d = say(svc, t, BENIGN[0])
    assert d.verdict == NEEDS and "P-PUBLISH-APPROVAL" in d.policy_ids
    w = svc.authorize(act("http.post", API, arguments={"url": "https://owned.example/api/post", "json": {"t": "x"}}), t, {})
    assert w.verdict == NEEDS


def test_structure_is_judged_under_any_field_name_and_inside_string_bodies(svc, tmp_path, clock):
    t = live(svc, tmp_path, clock)
    body = '{"username": "Elon Musk", "avatar_url": "https://x/y.png", "content": "welcome"}'
    j = svc.authorize(act("http.post", API, arguments={"url": "https://owned.example/api/webhook", "body": body}), t, {})
    assert j.verdict == DENY and "X-IDENTITY" in j.policy_ids
    form = "content=welcome&username=Support+Team"
    f = svc.authorize(act("http.post", API, arguments={"url": "https://owned.example/api/webhook", "body": form}), t, {})
    assert f.verdict == DENY and "X-IDENTITY" in f.policy_ids
    for key, values in (("rcpt", [f"u{i}@x.example" for i in range(50)]), ("whatever", [f"+1555000{i:04d}" for i in range(9)]),
                        ("notify_people", [str(10 ** 17 + i) for i in range(20)])):
        d = svc.authorize(act("http.post", API, arguments={"url": "https://owned.example/api/mail",
                                                           "json": {key: values, "text": "hi"}}), t, {})
        assert d.verdict == DENY and "X-MASS-OUTREACH" in d.policy_ids, key
    ok = svc.authorize(act("http.post", API, arguments={"url": "https://owned.example/api/items",
                                                        "json": {"tags": ["a", "b", "c", "d", "e"]}}), t, {})
    assert ok.verdict == ALLOW


def test_disguised_secrets_are_refused(svc, tmp_path, clock):
    t = live(svc, tmp_path, clock)
    svc.put_secret("bot_token", "tok-ABCDEFGH12345678-xyz", ["channel:discord:*"])
    for text, extra in (("debug: t o k - A B C D E F G H 1 2 3 4 5 6 7 8 - x y z", {}),
                        ("debug: TOK_abcdefgh_12345678_XYZ", {}),
                        ("debug part 1: tok-ABCDEFGH1", {"embed_note": "part 2: 2345678-xyz"})):
        d = say(svc, t, text, **extra)
        assert d.verdict == DENY and "X-CREDENTIAL-LEAK" in d.policy_ids, text
    assert say(svc, t, "the build of 12345678 is green").verdict == ALLOW


def test_profiling_in_generated_code_goes_to_the_operator(svc, tmp_path, clock):
    t = live(svc, tmp_path, clock)
    code = ("import json\nrows=json.load(open('events.json'))\nscores={r['user']:{'lonely':r['late_night_msgs']>20} "
            "for r in rows}\njson.dump(scores,open('per_user_traits.json','w'))")
    d = svc.authorize(act("sandbox.python", C.Modality.CODE, arguments={"code": code}), t, {})
    assert d.verdict == NEEDS and "X-PROFILING-CODE" in d.policy_ids
    plain = svc.authorize(act("sandbox.python", C.Modality.CODE, arguments={"code": "portraits = load(); print(len(portraits))"}), t, {})
    assert plain.verdict == ALLOW


# ---------------------------------------------------------------------------------------------------------------
# M4 / G-env C1, M10: this machine's control plane
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("url", ["http://localhost:8765/x", "http://localhost.:8765/x", "http://127.1:8765/x",
                                 "http://2130706433:8765/x", "http://0177.0.0.1:8765/x", "http://0x7f000001:8765/x",
                                 "http://[::1]:8765/x", "http://[::ffff:127.0.0.1]:8765/x", "http://0.0.0.0:8765/x",
                                 "http://127.9.9.9:8765/x", "http://app.localhost:8765/x", "http://%31%32%37.0.0.1:8765/x"])
def test_every_spelling_of_loopback_is_one_host(url):
    assert D.normalise_url(url).key == "url:127.0.0.1:8765/x"


@pytest.mark.parametrize("url", ["http://127.0.0.1:11111/capabilities/provision", "http://localhost:11111/x",
                                 "http://[::1]:8080/tree", "http://0.0.0.0:22/", "http://127.0.0.1/",
                                 "http://169.254.169.254/latest/meta-data/", "http://metadata.google.internal/x",
                                 "http://10.0.0.1/", "http://192.168.1.1/", "http://172.17.0.1/", "http://[fe80::1]/",
                                 "http://100.100.100.200/", "http://router/", "http://printer.local/"])
def test_this_machine_and_its_network_are_unreachable_in_every_mode(svc, tmp_path, clock, url):
    staging = svc.authorize(act("http.post", API, arguments={"url": url, "json": {"pip": ["x"]}}), task(), {})
    assert staging.verdict == DENY
    t = live(svc, tmp_path, clock, oid="obj-everything", read=["*"], owned=["*"], preauthorized=["*"])
    for a in (act("http.get", API, arguments={"url": url}), act("browser.navigate", MB, arguments={"url": url})):
        d = svc.authorize(a, t, {})
        assert d.verdict == DENY and "P-LOCAL-ADDRESS" in d.policy_ids, (url, d.reasons)


def test_own_interface_addresses_are_local():
    for addr in localnet.self_addresses():
        assert localnet.local_kind(addr), addr


def test_registered_local_services_and_their_audit(svc):
    t = task()
    nav = lambda port: act("browser.navigate", MB, arguments={"url": f"http://localhost:{port}/"})   # noqa: E731
    assert svc.authorize(nav(18600), t, {}).verdict == DENY
    svc.register_local_service("[::1]:18600", "operator", note="staging site started by the harness")
    assert svc.authorize(nav(18600), t, {}).verdict == ALLOW
    assert svc.authorize(nav(18601), t, {}).verdict == DENY             # exactly that port
    assert {"address": "127.0.0.1:18600", "source": "registered"}.items() <= svc.local_services()[-1].items()
    assert svc.unregister_local_service("127.0.0.1:18600", "operator")
    assert svc.authorize(nav(18600), t, {}).verdict == DENY
    with pytest.raises(ValueError):
        svc.register_local_service("example.org:80", "operator")          # not an address of this machine
    with pytest.raises(ValueError):
        svc.register_local_service("127.0.0.1", "operator")               # a port is required
    events = [e["event"] for e in svc.audit_tail(50)]
    assert "local_service_registered" in events and "local_service_unregistered" in events


def test_in_process_staging_services_are_reachable_but_other_processes_are_not(svc, tmp_path):
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    child = subprocess.Popen([sys.executable, "-c", "import socket,sys,time\ns=socket.socket();s.bind(('127.0.0.1',0));"
                              "s.listen();print(s.getsockname()[1],flush=True);time.sleep(60)"],
                             stdout=subprocess.PIPE, text=True)
    try:
        foreign = int(child.stdout.readline())
        mine = httpd.server_address[1]
        t = task()
        nav = lambda port: act("browser.navigate", MB, arguments={"url": f"http://localhost:{port}/"})   # noqa: E731
        assert svc.authorize(nav(mine), t, {}).verdict == ALLOW         # served by this very process
        d = svc.authorize(nav(foreign), t, {})
        assert d.verdict == DENY and "P-LOCAL-ADDRESS" in d.policy_ids  # another process listens there
    finally:
        child.kill()
        child.wait()
        httpd.shutdown()
        httpd.server_close()


def test_a_denied_loopback_port_is_denied_under_every_alias(svc):
    t = task(resources={"deny": ["127.0.0.1:8765"]})
    for host in ("127.0.0.1", "localhost", "localhost.", "[::1]", "127.1", "2130706433", "0.0.0.0", "[::ffff:7f00:1]"):
        d = svc.authorize(act("browser.navigate", MB, arguments={"url": f"http://{host}:8765/secret"}), t, {})
        assert d.verdict == DENY and "P-DEST-DENIED" in d.policy_ids, host


def test_a_public_name_that_resolved_to_this_machine_is_refused(svc, tmp_path, clock):
    t = live(svc, tmp_path, clock, oid="obj-peer", read=["*"])
    a = act("http.get", API, arguments={"url": "https://rebind.example/x"})
    start(svc, a, t)
    assert svc.check_primitive(prim("http.request", API, method="GET", url="https://rebind.example/x",
                                    peer_ip="203.0.113.7"), a).verdict == ALLOW
    d = svc.check_primitive(prim("http.request", API, method="GET", url="https://rebind.example/x", peer_ip="127.0.0.1"), a)
    assert d.verdict == DENY and "P-LOCAL-ADDRESS" in d.policy_ids


# ---------------------------------------------------------------------------------------------------------------
# m5 file-flag stop, m6 URL reading, m7 audit head
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("how", ["touch", "api"])
def test_a_stop_set_by_any_means_voids_grants_for_good(svc, run_dir, clock, how):
    t = task()
    nav = lambda i: act("browser.navigate", MB, arguments={"url": f"{LOOP}/{i}"})   # noqa: E731
    a_unused, a_running = nav(1), nav(2)
    d_unused, d_running = svc.authorize(a_unused, t, {}), svc.authorize(a_running, t, {})
    assert svc.check_grant(a_running, d_running)
    if how == "touch":
        open(os.path.join(run_dir, "auth", "STOP"), "w").close()
    else:
        svc.emergency_stop("operator", "operator")
    assert svc.authorize(nav(3), t, {}).verdict == DENY
    svc.reset_stop("operator")
    clock.advance(1.0)
    assert svc.check_grant(a_unused, d_unused) is False
    assert svc.check_primitive(prim("browser.navigate", MB, url=f"{LOOP}/2"), a_running).verdict == DENY
    assert svc.authorize(nav(4), t, {}).verdict == ALLOW               # new work after the reset is fine


def test_a_stop_flag_set_while_the_service_was_down_voids_grants_on_restart(run_dir, clock):
    s = PolicyService(run_dir, clock=clock, fsync=False)
    a = act("browser.navigate", MB, arguments={"url": LOOP + "/z"})
    d = s.authorize(a, task(), {})
    s.close()
    open(os.path.join(run_dir, "auth", "STOP"), "w").close()           # offline `opctl stop` / touch
    s = PolicyService(run_dir, clock=clock, fsync=False)
    s.reset_stop("operator")
    assert s.check_grant(a, d) is False
    s.close()


@pytest.mark.parametrize("raw,host", [("http://evil.example\\@good.example/", "evil.example"),
                                      ("http:\\\\evil.example\\path", "evil.example"),
                                      ("http:/\\evil.example/", "evil.example"),
                                      ("https:evil.example/x", "evil.example"),
                                      ("http://good.example@evil.example/", "evil.example"),
                                      ("http://a@b@evil.example/", "evil.example"),
                                      ("http://evil.example?\\@good.example/", "evil.example"),
                                      ("http://%65vil.example/", "evil.example")])
def test_urls_are_read_as_a_whatwg_parser_reads_them(raw, host):
    assert D.normalise_url(raw).host == host


@pytest.mark.parametrize("raw", ["http://evil.example/\tx", "http://ev\x00il.example/", "http://evil.example/\x7f"])
def test_control_characters_are_refused_not_stripped(raw):
    with pytest.raises(D.DestinationError):
        D.normalise_url(raw)


def test_the_policy_judges_the_host_chromium_would_contact(svc):
    t = task(resources={"deny": ["127.0.0.1:8765"]})
    crafted = "http://127.0.0.1:8765\\@127.0.0.1:18700/x"
    assert D.normalise_url(crafted).port == 8765
    assert svc.authorize(act("browser.navigate", MB, arguments={"url": crafted}), t, {}).verdict == DENY


def test_an_emptied_log_without_its_head_fails_verification(svc):
    for i in range(3):
        svc.authorize(act("wait"), task(), {})
    path, head, key = svc.audit.path, svc.audit.head_path, svc.audit.key
    assert verify_audit_log(path, key=key).ok                           # finds <run>/auth/audit.head by itself
    svc.close()
    open(path, "wb").close()
    os.unlink(head)
    assert not verify_audit_log(path, key=key).ok
    assert not verify_audit_log(path).ok
    assert verify_audit_log(path, require_head=False).ok               # explicitly: chain consistency only


# ---------------------------------------------------------------------------------------------------------------
# policy data for legitimate work (G-env M7) and the run policy
# ---------------------------------------------------------------------------------------------------------------
def test_programs_are_authorized_step_by_step(svc):
    t = task(resources={"deny": ["127.0.0.1:8765/forbidden/*"]})
    steps = [act("internal.note", C.Modality.INTERNAL, arguments={"text": "x"}),
             act("http.request", API, arguments={"method": "POST", "url": LOOP + "/w", "json": {"i": 1}})]
    assert svc.authorize(act("program", C.Modality.INTERNAL, level=C.Level.PROCEDURE, steps=steps), t, {}).verdict == ALLOW
    bad = steps + [act("http.request", API, arguments={"method": "POST", "url": LOOP + "/forbidden/x"})]
    d = svc.authorize(act("program", C.Modality.INTERNAL, level=C.Level.PROCEDURE, steps=bad), t, {})
    assert d.verdict == DENY and "P-DEST-DENIED" in d.policy_ids


def test_uploads_from_the_workspace_are_allowed(svc, run_dir):
    t = task()
    ws = os.path.join(run_dir, "workspace")
    os.makedirs(ws)
    open(os.path.join(ws, "ok.txt"), "w").write("fine")
    os.symlink("/etc/passwd", os.path.join(ws, "link.txt"))
    up = lambda p: act("browser.upload", MB, arguments={"path": p, "page_url": LOOP + "/up", "selector": "input"})  # noqa: E731
    assert svc.authorize(up(os.path.join(ws, "ok.txt")), t, {}).verdict == ALLOW
    for p in (os.path.join(ws, "link.txt"), "/etc/passwd", os.path.join(run_dir, "env", "state.json"),
              os.path.join(run_dir, "exec", "env", "state.json"), os.path.join(run_dir, "vault", "x.secret.json")):
        assert svc.authorize(up(p), t, {}).verdict == DENY, p
    exec_ws = os.path.join(run_dir, "exec", "workspace", "a.txt")      # the in-process runtime's workspace
    assert svc.authorize(up(exec_ws), t, {}).verdict == ALLOW


def test_discord_member_events_need_the_member_list_permission(svc, tmp_path, clock):
    """Master decision (2026-11-27, G-sec p07): per-member join/leave events and the gateway event stream process the
    member list, so the task must grant data_permissions.member_list; the output stays pseudonymous and aggregate,
    so collect_personal_data is not needed. Plain counts (get_guild_stats) stay aggregate reads."""
    for op, args in (("discord.listen_events", {"guild_id": "guild-000001", "seconds": 10}),
                     ("discord.get_member_events", {"guild_id": "guild-000001"})):
        t = live(svc, tmp_path, clock, oid=f"obj-nomembers-{op[8:]}")          # aggregate_metrics only
        d = svc.authorize(act(op, API, arguments=args), t, {})
        assert d.verdict == DENY and {"P-DATA-USE", "X-DATA-COLLECTION"} & set(d.policy_ids), (op, d.verdict, d.policy_ids)
        t2 = live_task(oid=f"obj-members-{op[8:]}", data_permissions={"aggregate_metrics": True, "member_list": True})
        authorize_live(svc, tmp_path, t2, clock, record_id=f"rec-members-{op[8:]}",
                       owned=["account:discord:guild-000001", "channel:discord:*"],
                       read=["account:discord:guild-000001", "channel:discord:*"])
        d = svc.authorize(act(op, API, arguments=args), t2, {})
        assert d.verdict == ALLOW, (op, d.reasons)
        effects = [e for e in svc.audit_tail(5) if e["event"] == "authorize"][-1]["data"]["effects"]
        assert effects == ["read:account:discord:guild-000001"]
    t3 = live(svc, tmp_path, clock, oid="obj-counts")
    assert svc.authorize(act("discord.get_guild_stats", API, arguments={"guild_id": "guild-000001"}), t3, {}).verdict == ALLOW
    t4 = live_task(oid="obj-no-aggregates", data_permissions={"aggregate_metrics": False})
    authorize_live(svc, tmp_path, t4, clock, record_id="rec-noagg", read=["account:discord:guild-000001"])
    d = svc.authorize(act("discord.get_guild_stats", API, arguments={"guild_id": "guild-000001"}), t4, {})
    assert d.verdict == DENY and "P-DATA-USE" in d.policy_ids


def test_cross_origin_subresource_checks_are_reads_by_method(svc):
    run = act("browser.navigate", MB, arguments={"url": LOOP + "/page"})
    start(svc, run, task())
    assert svc.check_primitive(prim("browser.request", MB, url=LOOP + "/", method="GET", resource_type="image"),
                               run).verdict == ALLOW
    d = svc.check_primitive(prim("browser.request", MB, url="http://127.0.0.1:11111/", method="POST"), run)
    assert d.verdict == DENY and "P-LOCAL-ADDRESS" in d.policy_ids


def test_the_run_policy_is_the_fixed_default_and_is_what_a_run_loads(svc):
    with open(RUN_POLICY_PATH, "rb") as a, open(DEFAULT_POLICY_PATH, "rb") as b:
        assert a.read() == b.read()
    assert svc.policy_path == RUN_POLICY_PATH
    pol = load_policy(RUN_POLICY_PATH)
    assert pol.modes["staging"].own_process_services and not pol.modes["live"].own_process_services
    assert pol.local_services == [("127.0.0.1", 8765)] and pol.rate_write_global is not None


# ---------------------------------------------------------------------------------------------------------------
# requests from E (docs/change_requests/EA.md "Requests from E")
# ---------------------------------------------------------------------------------------------------------------
def test_a_browser_tunnel_is_a_read_level_check_not_an_external_write(svc):
    """E->EA-1: the browser network layer reports tunnels it was not told about as browser.request CONNECT."""
    run = act("browser.navigate", MB, arguments={"url": LOOP + "/page"})
    start(svc, run, task(limits=C.Limits(max_external_actions=100, max_decisions=100, duration_s=1e6)))
    for _ in range(15):                          # more than the write window: never charged as writes
        d = svc.check_primitive(prim("browser.request", MB, url=LOOP + "/", method="CONNECT", resource_type="tunnel",
                                     via="network"), run)
        assert d.verdict == ALLOW, d.reasons
    assert svc.check_primitive(prim("browser.request", MB, url="http://127.0.0.1:11111/", method="CONNECT"),
                               run).verdict == DENY


def test_approval_previews_say_when_content_was_cut(svc, tmp_path, clock):
    """E->EA-3: a long publication is shown in full (up to 64 KiB) and the preview says whether it was cut."""
    t = live(svc, tmp_path, clock, preauthorized=[])
    text = "Weekly notes. " + "x" * 9000
    d = say(svc, t, text)
    entry = [a for a in svc.approvals() if a["approval_id"] == d.approval_id][0]["preview"]["content"][0]
    assert entry["text"] == text and entry["content_chars"] == len(text) and entry["content_truncated"] is False


def test_secret_has_tells_presence_only(svc):
    """E->EA-4: whether a secret exists, never its value or bindings."""
    svc.put_secret("bot_token", "tok-ABCDEFGH12345678-xyz", ["channel:discord:*"])
    assert svc.secret_has("bot_token") is True and svc.secret_has("nope") is False
    svc.revoke("secret:bot_token", "operator")
    assert svc.secret_has("bot_token") is False
