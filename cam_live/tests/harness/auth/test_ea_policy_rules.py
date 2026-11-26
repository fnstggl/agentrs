"""Every rule family of the authorization policy: allow / deny / needs-approval, and fail-closed behaviour."""
import os

import pytest
import yaml

from cam_live import contracts as C
from cam_live.harness.auth import DEFAULT_POLICY_PATH, PolicyError, PolicyService, liveauth, load_policy
from ea_helpers import ALLOW, DENY, LOOP, NEEDS, act, authorize_live, install_operator_key, live_record, live_task, task

GUILD, CHAN, CHAN2, CHAN3 = "900000000000000001", "900000000000000011", "900000000000000022", "900000000000000033"
TEXT_A = "The weekly notes are published: the build log, the open questions and the next milestone are listed there."
TEXT_B = "A completely different announcement about maintenance windows, scheduled for the coming weekend at noon."


def policy_file(tmp_path, mutate, name="policy.yaml"):
    with open(DEFAULT_POLICY_PATH) as f:
        doc = yaml.safe_load(f)
    mutate(doc)
    p = tmp_path / name
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return str(p)


def discord_task(pre=False, **kw):
    res = {"owned": [f"channel:discord:{CHAN}", f"channel:discord:{CHAN2}", f"channel:discord:{CHAN3}"],
           "accounts": [f"account:discord:{GUILD}"]}
    if pre:
        res["preauthorized"] = ["channel:discord:*"]
    return task(resources=res, **kw)


def send(channel=CHAN, content=TEXT_A, **extra):
    return act("discord.send_message", arguments=dict({"channel_id": channel, "content": content}, **extra))


def pids(dec):
    return set(dec.policy_ids)


# ---------------------------------------------------------------------------------------------------------------
# task mode
# ---------------------------------------------------------------------------------------------------------------
def test_diagnostic_mode_allows_simulation_and_nothing_external(svc):
    t = task(mode="diagnostic", oid="diag")
    assert svc.authorize(act("lab.act", arguments={"channel": 1, "effort": 2}, est_cost=C.Cost(money=3.0)), t, {}).verdict == ALLOW
    assert svc.authorize(act("wait", arguments={"seconds": 60}), t, {}).verdict == ALLOW
    assert svc.authorize(act("sandbox.python", arguments={"code": "print(1)"}), t, {}).verdict == ALLOW
    d = svc.authorize(act("browser.navigate", arguments={"url": LOOP + "/"}), t, {})
    assert d.verdict == DENY and pids(d) & {"P-MODE-ENVELOPE", "P-MODE-EFFECT"}
    assert svc.authorize(act("http.post", arguments={"url": LOOP + "/x", "body": "a"}), t, {}).verdict == DENY


def test_staging_mode_reaches_only_the_local_staging_services(svc):
    t = task()
    assert svc.authorize(act("browser.navigate", arguments={"url": LOOP + "/page"}), t, {}).verdict == ALLOW
    assert svc.authorize(act("http.post", arguments={"url": LOOP + "/api/items", "json": {"a": 1}}), t, {}).verdict == ALLOW
    d = svc.authorize(act("browser.navigate", arguments={"url": "https://public.example/"}), t, {})
    assert d.verdict == DENY and "P-MODE-ENVELOPE" in pids(d)
    d = svc.authorize(act("lab.act", arguments={}), t, {})
    assert d.verdict == DENY and "P-TOOL-MODE" in pids(d)
    # a staging task cannot widen the envelope by claiming to own a public site
    t2 = task(oid="greedy", resources={"owned": ["https://public.example/*"], "preauthorized": ["https://public.example/*"]})
    assert svc.authorize(act("http.post", arguments={"url": "https://public.example/post", "body": "x"}), t2, {}).verdict == DENY


def test_live_mode_requires_a_signed_operator_record(svc, tmp_path, clock):
    t = live_task()
    nav = lambda: act("browser.navigate", arguments={"url": "https://a.docs.example/guide"})   # noqa: E731
    d = svc.authorize(nav(), t, {})
    assert d.verdict == DENY and pids(d) == {"P-LIVE-AUTH"}
    priv = install_operator_key(svc, tmp_path)
    rec = live_record(t, clock.now())
    # unsigned, tampered and wrongly signed records are refused at registration
    with pytest.raises(liveauth.LiveAuthorizationError):
        svc.register_live_authorization(dict(rec), "operator")
    signed = liveauth.sign_live_authorization(rec, priv)
    tampered = dict(signed, owned=signed["owned"] + ["https://extra.example/*"])
    with pytest.raises(liveauth.LiveAuthorizationError):
        svc.register_live_authorization(tampered, "operator")
    other = os.path.join(str(tmp_path), "other.key")
    liveauth.generate_operator_key(other, other + ".pub")
    with pytest.raises(liveauth.LiveAuthorizationError):
        svc.register_live_authorization(liveauth.sign_live_authorization(rec, other), "operator")
    assert svc.authorize(nav(), t, {}).verdict == DENY
    svc.register_live_authorization(signed, "operator")
    assert svc.authorize(nav(), t, {}).verdict == ALLOW
    # the record names one exact version of the task
    edited = live_task(objective="a test objective, edited")
    d = svc.authorize(nav(), edited, {})
    assert d.verdict == DENY and pids(d) == {"P-LIVE-AUTH"}
    # outside the record's destinations and time window
    d = svc.authorize(act("browser.navigate", arguments={"url": "https://elsewhere.example/"}), t, {})
    assert d.verdict == DENY and "P-MODE-ENVELOPE" in pids(d)
    clock.advance(3700.0)
    d = svc.authorize(nav(), t, {})
    assert d.verdict == DENY and pids(d) & {"L-WINDOW", "L-DURATION"}


def test_live_record_time_window_not_started_and_revocation(svc, tmp_path, clock):
    t = live_task()
    authorize_live(svc, tmp_path, t, clock, not_before=clock.now() + 100.0, not_after=clock.now() + 500.0)
    nav = lambda: act("browser.navigate", arguments={"url": "https://docs.example/"})   # noqa: E731
    d = svc.authorize(nav(), t, {})
    assert d.verdict == DENY and "L-WINDOW" in pids(d)
    clock.advance(150.0)
    assert svc.authorize(nav(), t, {}).verdict == ALLOW
    svc.revoke("live:rec-1", "operator")
    d = svc.authorize(nav(), t, {})
    assert d.verdict == DENY and pids(d) == {"P-LIVE-AUTH"}


def test_live_mode_never_assumes_unlimited_resources(svc, tmp_path, clock):
    t = live_task(limits=C.Limits())                     # nothing configured in the task
    zero = {k: 0 for k in liveauth.BUDGET_KEYS}
    authorize_live(svc, tmp_path, t, clock, budgets=zero)
    d = svc.authorize(act("wait", arguments={"seconds": 60}), t, {})
    assert d.verdict == DENY and pids(d) == {"L-UNSET"}
    assert all(r.startswith("resource: ") for r in d.reasons)


def test_unknown_mode_and_malformed_task_are_denied(svc):
    assert "P-TASK-INVALID" in pids(svc.authorize(act("wait"), task(mode="production"), {}))
    bad = task(oid="bad")
    bad.limits = {"money": 1}                             # not a Limits record
    assert "P-TASK-INVALID" in pids(svc.authorize(act("wait"), bad, {}))
    assert svc.authorize(act("wait"), {"objective_id": "x"}, {}).verdict == DENY
    bad2 = task(oid="bad2", resources={"owned": ["channel:discord"]})        # malformed pattern
    assert "P-TASK-INVALID" in pids(svc.authorize(act("wait"), bad2, {}))


# ---------------------------------------------------------------------------------------------------------------
# tools and modalities per task
# ---------------------------------------------------------------------------------------------------------------
def test_unknown_tools_are_denied(svc):
    for op in ("teleport.now", "shell", "os.system", "auth.resolve_approval", "policy.reload"):
        d = svc.authorize(act(op, C.Modality.API), task(), {})
        assert d.verdict == DENY and pids(d) == {"P-TOOL-UNKNOWN"}, op


def test_modality_must_match_the_tool(svc):
    d = svc.authorize(act("browser.navigate", C.Modality.INTERNAL, arguments={"url": LOOP}), task(), {})
    assert d.verdict == DENY and "P-MODALITY" in pids(d)
    d = svc.authorize(act("sandbox.python", C.Modality.STRUCTURED_BROWSER, arguments={"code": "1"}), task(), {})
    assert d.verdict == DENY and "P-MODALITY" in pids(d)


def test_task_interfaces_limit_modalities_and_tools(svc):
    t = task(oid="code-only", interfaces=["code"])
    assert svc.authorize(act("sandbox.python", arguments={"code": "1"}), t, {}).verdict == ALLOW
    assert svc.authorize(act("wait"), t, {}).verdict == ALLOW                     # internal: no effect
    d = svc.authorize(act("browser.navigate", arguments={"url": LOOP}), t, {})
    assert d.verdict == DENY and pids(d) == {"P-INTERFACE"}
    t2 = task(oid="by-tool", interfaces=["browser.navigate", "http.get"])
    assert svc.authorize(act("browser.navigate", arguments={"url": LOOP}), t2, {}).verdict == ALLOW
    assert svc.authorize(act("http.get", arguments={"url": LOOP}), t2, {}).verdict == ALLOW
    assert "P-INTERFACE" in pids(svc.authorize(act("http.post", arguments={"url": LOOP, "body": "x"}), t2, {}))


def test_live_task_without_interfaces_authorizes_no_tool(svc, tmp_path, clock):
    t = live_task(interfaces=[])
    authorize_live(svc, tmp_path, t, clock)
    d = svc.authorize(act("browser.navigate", arguments={"url": "https://docs.example/"}), t, {})
    assert d.verdict == DENY and "P-INTERFACE" in pids(d)


# ---------------------------------------------------------------------------------------------------------------
# destinations
# ---------------------------------------------------------------------------------------------------------------
def test_url_hosts_and_path_patterns(svc, tmp_path, clock):
    t = live_task()
    authorize_live(svc, tmp_path, t, clock, read=["*.docs.example", "https://paths.example/public/*"])
    ok = ["https://a.docs.example/x", "https://paths.example/public/a", "https://PATHS.example:443/public/b/../c"]
    no = ["https://docs.example.evil.example/", "https://paths.example/private/a", "https://paths.example/public/../private/a",
          "http://paths.example/public/a", "https://a.docs.example@evil.example/"]
    for u in ok:
        assert svc.authorize(act("browser.navigate", arguments={"url": u}), t, {}).verdict == ALLOW, u
    for u in no:
        d = svc.authorize(act("browser.navigate", arguments={"url": u}), t, {})
        assert d.verdict == DENY and pids(d) & {"P-MODE-ENVELOPE", "P-DEST-NOT-ALLOWED", "X-CIRCUMVENTION"}, u


def test_deny_lists_override_everything(svc, tmp_path, clock):
    t = task(resources={"deny": ["127.0.0.1:9999"]})
    assert svc.authorize(act("browser.navigate", arguments={"url": "http://127.0.0.1:8765/"}), t, {}).verdict == ALLOW
    d = svc.authorize(act("browser.navigate", arguments={"url": "http://127.0.0.1:9999/"}), t, {})
    assert d.verdict == DENY and "P-DEST-DENIED" in pids(d)
    p = policy_file(tmp_path, lambda doc: doc["destinations"].__setitem__("deny", ["localhost"]))
    s2 = PolicyService(str(tmp_path / "r2"), p, clock=clock, fsync=False)
    d = s2.authorize(act("http.get", arguments={"url": "http://localhost:8000/"}), task(), {})
    assert d.verdict == DENY and "P-DEST-DENIED" in pids(d)
    s2.close()


def test_api_endpoints_by_method(svc, tmp_path, clock):
    t = live_task()
    authorize_live(svc, tmp_path, t, clock)
    assert svc.authorize(act("http.get", arguments={"url": "https://docs.example/api/v1/items"}), t, {}).verdict == ALLOW
    assert svc.authorize(act("api.request", arguments={"method": "HEAD", "url": "https://docs.example/"}), t, {}).verdict == ALLOW
    for op, args in (("http.post", {"url": "https://docs.example/api/v1/items", "json": {"a": 1}}),
                     ("api.request", {"method": "DELETE", "url": "https://docs.example/api/v1/items/3"}),
                     ("api.request", {"url": "https://docs.example/api/v1/items"})):       # no method: assume a write
        d = svc.authorize(act(op, arguments=args), t, {})
        assert d.verdict == DENY and "P-WRITE-NON-OWNED" in pids(d), op
    d = svc.authorize(act("http.post", arguments={"url": "https://owned.example/api/pages", "json": {"t": "x"}}), t, {})
    assert d.verdict == NEEDS and pids(d) == {"P-WRITE-APPROVAL"}


def test_accounts_and_channels(svc):
    t = discord_task()
    d = svc.authorize(send(CHAN), t, {})
    assert d.verdict == NEEDS and pids(d) == {"P-PUBLISH-APPROVAL"} and d.approval_id
    d = svc.authorize(send("900000000000009999"), t, {})
    assert d.verdict == DENY and "P-WRITE-NON-OWNED" in pids(d)
    assert svc.authorize(send(CHAN), discord_task(pre=True, oid="pre"), {}).verdict == ALLOW
    assert svc.authorize(act("discord.get_guild_stats", arguments={"guild_id": GUILD}), t, {}).verdict == ALLOW
    d = svc.authorize(act("discord.get_guild_stats", arguments={"guild_id": "900000000000000777"}), t, {})
    assert d.verdict == DENY and "P-DEST-NOT-ALLOWED" in pids(d)
    assert "P-DEST-MISSING" in pids(svc.authorize(act("discord.send_message", arguments={"content": TEXT_A}), t, {}))


def test_file_roots(svc, run_dir):
    t = task()
    assert svc.authorize(act("sandbox.write_file", arguments={"path": "out/report.txt", "content": "x"}), t, {}).verdict == ALLOW
    assert svc.authorize(act("sandbox.read_file", arguments={"path": os.path.join(run_dir, "artifacts", "a.bin")}), t, {}).verdict == ALLOW
    for path in ("/etc/passwd", "../../etc/passwd", os.path.join(run_dir, "audit.jsonl"), os.path.join(run_dir, "verifier.jsonl"),
                 os.path.join(run_dir, "browser_profile", "Cookies"), "/root/.ssh/id_rsa"):
        d = svc.authorize(act("sandbox.read_file", arguments={"path": path}), t, {})
        assert d.verdict == DENY and pids(d) & {"P-FILE-ROOT", "P-DEST-INVALID"}, path


# ---------------------------------------------------------------------------------------------------------------
# side-effect classes, interaction, publishing approvals
# ---------------------------------------------------------------------------------------------------------------
def test_side_effect_classes(svc, tmp_path, clock):
    t = live_task()
    authorize_live(svc, tmp_path, t, clock, preauthorized=["https://owned.example/auto/*"])
    cases = [
        (act("wait", arguments={"seconds": 5}), ALLOW, None),                                         # none
        (act("sandbox.python", arguments={"code": "print(2+2)"}), ALLOW, None),                       # local
        (act("browser.navigate", arguments={"url": "https://docs.example/a"}), ALLOW, None),          # external_read
        (act("http.post", arguments={"url": "https://docs.example/a", "body": "x"}), DENY, "P-WRITE-NON-OWNED"),
        (act("http.post", arguments={"url": "https://owned.example/manual/a", "body": "x"}), NEEDS, "P-WRITE-APPROVAL"),
        (act("http.post", arguments={"url": "https://owned.example/auto/a", "body": "x"}), ALLOW, None),
    ]
    for a, verdict, pid in cases:
        d = svc.authorize(a, t, {})
        assert d.verdict == verdict, (a.operation, a.arguments, d.reasons)
        assert pid is None or pid in pids(d)
        assert d.reasons and d.policy_ids


def test_browser_interaction_depends_on_ownership(svc, tmp_path, clock):
    t = live_task()
    authorize_live(svc, tmp_path, t, clock, read=["docs.example", "search.example", "owned.example"],
                   interactive=["search.example"])
    click = lambda url: act("browser.click", arguments={"ref": "e3", "page_url": url})          # noqa: E731
    fill = lambda url: act("browser.fill", arguments={"ref": "e4", "text": "query", "page_url": url})   # noqa: E731
    assert svc.authorize(click("https://docs.example/a"), t, {}).verdict == ALLOW          # following links: reading
    d = svc.authorize(fill("https://docs.example/a"), t, {})
    assert d.verdict == NEEDS and pids(d) == {"P-INTERACT-APPROVAL"}                       # sending input: operator
    assert svc.authorize(fill("https://search.example/"), t, {}).verdict == ALLOW          # operator-listed
    d = svc.authorize(click("https://owned.example/admin"), t, {})
    assert d.verdict == NEEDS and pids(d) == {"P-WRITE-APPROVAL"}                          # owned: a write
    d = svc.authorize(fill("https://unknown.example/"), t, {})
    assert d.verdict == DENY
    d = svc.authorize(act("visual.type", arguments={"text": "hello", "page_url": "https://docs.example/a"}), t, {})
    assert d.verdict == NEEDS
    assert svc.authorize(act("visual.screenshot"), t, {}).verdict == ALLOW


def test_interaction_needs_a_known_location(svc):
    t = task()
    d = svc.authorize(act("browser.click", arguments={"ref": "e1"}), t, {})
    assert d.verdict == DENY and pids(d) == {"P-DEST-MISSING"}
    assert svc.authorize(act("browser.click", arguments={"ref": "e1"}), t, {"page_url": LOOP + "/a"}).verdict == ALLOW


def test_publishing_approvals(svc, tmp_path, clock):
    t = live_task()
    authorize_live(svc, tmp_path, t, clock, preauthorized=["https://owned.example/notes/*"])
    pub = lambda dest: act("publish.post", arguments={"destination": dest, "title": "Notes", "body": TEXT_A})  # noqa: E731
    d = svc.authorize(pub("https://owned.example/blog/1"), t, {})
    assert d.verdict == NEEDS and pids(d) == {"P-PUBLISH-APPROVAL"}
    assert svc.authorize(pub("https://owned.example/notes/1"), t, {}).verdict == ALLOW
    assert svc.authorize(pub("https://docs.example/blog/1"), t, {}).verdict == DENY
    assert svc.authorize(act("publish.draft", arguments={"title": "Notes", "body": TEXT_A}), t, {}).verdict == ALLOW
    # the operator can require an approval for every publication, pre-authorized or not
    p = policy_file(tmp_path, lambda doc: doc["publishing"].__setitem__("require_approval", "always"))
    s2 = PolicyService(str(tmp_path / "r2"), p, clock=clock, fsync=False)
    authorize_live(s2, tmp_path, t, clock, preauthorized=["https://owned.example/notes/*"])
    assert s2.authorize(pub("https://owned.example/notes/1"), t, {}).verdict == NEEDS
    s2.close()


def test_spending_real_money_needs_approval_simulated_does_not(svc, tmp_path, clock):
    t = live_task()
    authorize_live(svc, tmp_path, t, clock)
    buy = act("http.get", arguments={"url": "https://docs.example/a"}, est_cost=C.Cost(money=2.0))
    d = svc.authorize(buy, t, {})
    assert d.verdict == NEEDS and pids(d) == {"P-SPEND-APPROVAL"}
    pay = act("payment.create", arguments={"payee": "https://owned.example/billing", "amount": 3.0, "purpose": "hosting"})
    d = svc.authorize(pay, t, {})
    assert d.verdict == NEEDS and "P-SPEND-APPROVAL" in pids(d)
    st = task(oid="sim", limits=C.Limits(money=10.0))
    assert svc.authorize(act("http.get", arguments={"url": LOOP}, est_cost=C.Cost(money=2.0)), st, {}).verdict == ALLOW


def test_generated_code_network_access(svc, tmp_path, clock):
    code = act("sandbox.python", arguments={"code": "import urllib.request", "network": True, "hosts": ["docs.example"]})
    d = svc.authorize(code, task(), {})
    assert d.verdict == DENY and "P-CODE-NETWORK" in pids(d)
    t = live_task()
    authorize_live(svc, tmp_path, t, clock)
    d = svc.authorize(code, t, {})
    assert d.verdict == NEEDS and pids(d) == {"P-CODE-NETWORK"}
    nohosts = act("sandbox.python", arguments={"code": "x", "network": True})
    assert svc.authorize(nohosts, t, {}).verdict == DENY
    bad = act("sandbox.python", arguments={"code": "x", "allow_network": True, "hosts": ["elsewhere.example"]})
    assert svc.authorize(bad, t, {}).verdict == DENY


# ---------------------------------------------------------------------------------------------------------------
# data use and platform rules
# ---------------------------------------------------------------------------------------------------------------
def test_data_use_permissions(svc):
    read = lambda **kw: act("discord.read_messages", arguments=dict({"guild_id": GUILD}, **kw))   # noqa: E731
    d = svc.authorize(read(), discord_task(), {})
    assert d.verdict == DENY and pids(d) == {"P-DATA-USE"}
    t_ok = discord_task(oid="may-read", data_permissions={"read_message_content": True})
    assert svc.authorize(read(), t_ok, {}).verdict == ALLOW
    d = svc.authorize(read(store=True), t_ok, {})
    assert d.verdict == DENY and "P-DATA-USE" in pids(d)                    # storing needs its own permission
    t_all = discord_task(oid="may-store", data_permissions={"read_message_content": True, "store_message_content": True,
                                                             "train_on_message_content": True})
    d = svc.authorize(read(add_to_dataset=True), t_all, {})
    assert d.verdict == DENY and "PL-DISCORD-MESSAGE-CONTENT" in pids(d)    # the platform rule still forbids it
    # aggregates are allowed by default; member lists and personal fields are not
    assert svc.authorize(act("discord.get_guild_stats", arguments={"guild_id": GUILD}), discord_task(), {}).verdict == ALLOW
    d = svc.authorize(act("discord.list_members", arguments={"guild_id": GUILD}), discord_task(), {})
    assert d.verdict == DENY and "X-DATA-COLLECTION" in pids(d)
    d = svc.authorize(act("discord.get_guild_stats", arguments={"guild_id": GUILD, "fields": ["member_count", "email"]}),
                      discord_task(), {})
    assert d.verdict == DENY and "X-DATA-COLLECTION" in pids(d)
    d = svc.authorize(act("discord.get_guild_stats", arguments={"guild_id": GUILD, "fields": ["dms"]}),
                      discord_task(oid="p", data_permissions={"collect_personal_data": True}), {})
    assert d.verdict == DENY and "P-DATA-USE" in pids(d)                    # private messages: never
    assert svc.authorize(act("discord.list_members", arguments={"guild_id": GUILD}),
                         discord_task(oid="m", data_permissions={"member_list": True, "collect_personal_data": True}), {}).verdict == ALLOW


def test_platform_rules_are_data_the_operator_edits(svc, tmp_path, clock):
    t = live_task()
    authorize_live(svc, tmp_path, t, clock, read=["discord.com", "forum.example"], owned=["https://forum.example/mine/*"],
                   preauthorized=["https://forum.example/mine/*"])
    assert svc.authorize(act("browser.navigate", arguments={"url": "https://discord.com/developers/docs"}), t, {}).verdict == ALLOW
    d = svc.authorize(act("browser.click", arguments={"ref": "e1", "page_url": "https://discord.com/channels/1/2"}), t, {})
    assert d.verdict == DENY and "PL-DISCORD-NO-USER-AUTOMATION" in pids(d)
    post = lambda: act("publish.post", arguments={"destination": "https://forum.example/mine/a", "body": TEXT_A})   # noqa: E731
    assert svc.authorize(post(), t, {}).verdict == ALLOW

    def add_rule(doc):
        doc["platforms"]["forum"] = {"match": ["forum.example"], "rules": [
            {"id": "PL-FORUM-NO-AUTOMATED-POSTS", "when": {"kinds": ["publish"]}, "verdict": "deny",
             "reason": "the forum's terms forbid automated posting"}]}
    p = policy_file(tmp_path, add_rule)
    out = svc.reload_policy("operator", p)
    assert out["valid"]
    d = svc.authorize(post(), t, {})
    assert d.verdict == DENY and "PL-FORUM-NO-AUTOMATED-POSTS" in pids(d)
    assert any("forum's terms" in r for r in d.reasons)


# ---------------------------------------------------------------------------------------------------------------
# prohibited classes (operator specification 12.6)
# ---------------------------------------------------------------------------------------------------------------
def test_mass_outreach_recipient_count_and_destinations(svc):
    t = discord_task(pre=True)
    assert svc.authorize(send(mentions=["u1", "u2"]), t, {}).verdict == ALLOW
    d = svc.authorize(send(content=TEXT_B, mentions=["u1", "u2", "u3", "u4"]), t, {})
    assert d.verdict == DENY and "X-MASS-OUTREACH" in pids(d)
    many = act("skill.run", arguments={"skill_id": "s1"},
               steps=[act("http.post", arguments={"url": f"http://{h}/api", "body": f"note {i} " + "x" * i})
                      for i, h in enumerate(["127.0.0.1:1", "localhost:2", "[::1]:3", "127.0.0.2:4"])])
    d = svc.authorize(many, task(oid="many", resources={}), {})
    assert d.verdict == DENY           # 127.0.0.2 is outside the envelope; and with it gone the count rule holds:
    four = act("skill.run", arguments={"skill_id": "s1"},
               steps=[send(c, f"separate message number {i} for this channel only") for i, c in
                      enumerate([CHAN, CHAN2, CHAN3, "900000000000000044"])])
    t4 = task(oid="four", resources={"owned": ["channel:discord:*"], "preauthorized": ["channel:discord:*"]})
    d = svc.authorize(four, t4, {})
    assert d.verdict == DENY and "X-MASS-OUTREACH" in pids(d)


def test_near_duplicate_content_across_destinations(svc):
    t = discord_task(pre=True)
    assert svc.authorize(send(CHAN, TEXT_A), t, {}).verdict == ALLOW
    assert svc.authorize(send(CHAN2, TEXT_A.replace("weekly", "WEEKLY") + " !!"), t, {}).verdict == ALLOW
    d = svc.authorize(send(CHAN3, "  " + TEXT_A.upper() + "  :)"), t, {})
    assert d.verdict == DENY and "X-MASS-OUTREACH-DUPLICATE" in pids(d)
    assert svc.authorize(send(CHAN3, TEXT_B), t, {}).verdict == ALLOW          # different content is fine
    assert svc.authorize(send(CHAN, TEXT_A), t, {}).verdict == ALLOW           # the same place again is not outreach


def test_direct_messages_need_opt_in(svc):
    dm = lambda who: act("staging.send_dm", arguments={"recipient": f"account:staging:{who}", "content": "hello there"})  # noqa: E731
    t = task(resources={"opted_in_recipients": ["user-000123"]})
    assert svc.authorize(dm("user-000123"), t, {}).verdict == ALLOW
    d = svc.authorize(dm("user-000999"), t, {})
    assert d.verdict == DENY and "X-DM-OPT-IN" in pids(d)
    d = svc.authorize(act("discord.send_dm", arguments={"user_id": "900000000000000555", "content": "hi"}), discord_task(oid="d"), {})
    assert d.verdict == DENY and {"X-DM-OPT-IN", "PL-DISCORD-NO-DM"} <= pids(d)


def test_deceptive_identity(svc, tmp_path, clock):
    t = discord_task(pre=True)
    for extra in ({"username_override": "Server Admin"}, {"impersonate": "a member"}, {"embed": {"as_user": "u1"}},
                  {"avatar_url": "http://127.0.0.1/a.png"}, {"identities": ["a", "b"]}):
        d = svc.authorize(send(**extra), t, {})
        assert d.verdict == DENY and "X-IDENTITY" in pids(d), extra
    d = svc.authorize(send(content="Honestly, I am a real person and not a bot, trust me on this one."), t, {})
    assert d.verdict == DENY and "C-HUMAN-CLAIM" in pids(d)
    d = svc.authorize(send(content="Message from the official support team about your account settings."), t, {})
    assert d.verdict == NEEDS and "C-AUTHORITY-CLAIM" in pids(d)

    def need_disclosure(doc):
        doc["identity"]["disclosure_text"] = "automated account"
        doc["identity"]["require_disclosure_for_kinds"] = ["publish"]
    s2 = PolicyService(str(tmp_path / "r2"), policy_file(tmp_path, need_disclosure), clock=clock, fsync=False)
    d = s2.authorize(send(), t, {})
    assert d.verdict == DENY and pids(d) == {"X-DISCLOSURE"}
    assert s2.authorize(send(content=TEXT_A + " (Automated Account)"), t, {}).verdict == ALLOW
    s2.close()

    def required_not_configured(doc):
        doc["platforms"]["discord"]["require_disclosure"] = True
    s3 = PolicyService(str(tmp_path / "r3"), policy_file(tmp_path, required_not_configured, "p3.yaml"), clock=clock, fsync=False)
    d = s3.authorize(send(), t, {})
    assert d.verdict == DENY and pids(d) == {"X-DISCLOSURE"}
    s3.close()


def test_artificial_membership_and_join_for_join(svc):
    t = discord_task(pre=True)
    for op in ("discord.add_member", "discord.add_members", "discord.join_server", "discord.accept_invite",
               "discord.create_account"):
        d = svc.authorize(act(op, arguments={"guild_id": GUILD, "user_id": "900000000000000555"}), t, {})
        assert d.verdict == DENY and "X-MEMBERSHIP" in pids(d), op
    d = svc.authorize(send(fake_accounts=20), t, {})
    assert d.verdict == DENY and "X-MEMBERSHIP" in pids(d)
    d = svc.authorize(send(content="Anyone up for join for join? j4j welcome in this channel right now."), t, {})
    assert d.verdict == DENY and "C-JOIN-FOR-JOIN" in pids(d)
    d = svc.authorize(send(content="If you join my server then I'll join yours afterwards, deal? Just say so here."), t, {})
    assert d.verdict == NEEDS and "C-RECIPROCAL-JOIN" in pids(d)


def test_paying_or_rewarding_for_engagement(svc, tmp_path, clock):
    t = discord_task(pre=True)
    d = svc.authorize(send(reward_for_join={"amount": 5}), t, {})
    assert d.verdict == DENY and "X-INCENTIVE" in pids(d)
    d = svc.authorize(send(content="Win a $50 gift card: every member who invites three friends gets a reward."), t, {})
    assert d.verdict == NEEDS and "C-PAID-ENGAGEMENT" in pids(d)
    lt = live_task()
    authorize_live(svc, tmp_path, lt, clock)
    pay = lambda purpose: act("payment.create", arguments={"payee": "https://owned.example/billing", "amount": 2.0,   # noqa: E731
                                                           "purpose": purpose})
    d = svc.authorize(pay("pay people to join; 2 per member"), lt, {})
    assert d.verdict == DENY and "X-INCENTIVE" in pids(d)
    d = svc.authorize(pay("hosting invoice"), lt, {})
    assert d.verdict == NEEDS and "X-INCENTIVE" not in pids(d)


def test_profiling_and_vulnerable_audiences(svc):
    t = discord_task(pre=True)
    d = svc.authorize(send(user_traits={"u1": "anxious"}), t, {})
    assert d.verdict == DENY and "X-PROFILING" in pids(d)
    d = svc.authorize(act("discord.get_guild_stats", arguments={"guild_id": GUILD, "profile_users": True}), t, {})
    assert d.verdict == DENY and "X-PROFILING" in pids(d)
    for aud in ({"audience": "teens under 18"}, {"target_audience": ["people in debt"]}, {"min_age": 13},
                {"segment": {"label": "lonely members"}}):
        d = svc.authorize(send(**aud), t, {})
        assert d.verdict == DENY and "X-VULNERABLE" in pids(d), aud
    d = svc.authorize(send(audience="people who opened the documentation"), t, {})
    assert d.verdict == NEEDS and "X-AUDIENCE-TARGETING" in pids(d)
    d = svc.authorize(send(content="@everyone the weekly notes are up, have a look at the changes when you can."), t, {})
    assert d.verdict == NEEDS and "X-MASS-OUTREACH-BROADCAST" in pids(d)


def test_credential_leakage(svc, run_dir):
    import base64
    svc.put_secret("site_token", "tok_Zx81kQ0pLm4RvT7yWb2N", ["https://api.owned.example/*"])
    t = task()
    v = "tok_Zx81kQ0pLm4RvT7yWb2N"
    leaks = [
        act("http.post", arguments={"url": LOOP + "/paste", "body": "the key is " + v}),
        act("http.post", arguments={"url": LOOP + "/paste", "body": base64.b64encode(("x" + v + "yz").encode()).decode()}),
        act("http.get", arguments={"url": LOOP + "/?k=" + v.encode().hex()}),
        act("browser.fill", arguments={"ref": "e1", "text": v[::-1], "page_url": LOOP}),
        act("sandbox.python", arguments={"code": f"open('{run_dir}/vault/site_token.secret.json').read()"}),
        act("sandbox.shell", arguments={"cmd": f"cat {run_dir}/auth/keys/grant.key"}),
        act("http.post", arguments={"url": LOOP + "/x", "headers": {"Authorization": "Bearer abcdefghijklmnopqrstuvwxyz012345"}}),
        act("http.post", arguments={"url": LOOP + "/login", "json": {"user": "a", "password": "hunter2hunter2"}}),
        act("http.get", arguments={"url": "http://user:pass@127.0.0.1:8765/"}),
        act("sandbox.python", arguments={"code": "k = '-----BEGIN RSA PRIVATE KEY-----'"}),
    ]
    for a in leaks:
        d = svc.authorize(a, t, {})
        assert d.verdict == DENY and "X-CREDENTIAL-LEAK" in pids(d), a.arguments
        assert v not in " ".join(d.reasons)
    by_name = act("http.post", arguments={"url": LOOP + "/x", "headers": {"Authorization": "secret:site_token"}})
    assert svc.authorize(by_name, t, {}).verdict == ALLOW          # a reference by vault name is not a leak


# ---------------------------------------------------------------------------------------------------------------
# programs (steps), reasons, fail-closed
# ---------------------------------------------------------------------------------------------------------------
def test_steps_are_evaluated_recursively(svc):
    t = discord_task()
    inner_bad = act("skill.run", arguments={"skill_id": "s"}, steps=[
        act("browser.navigate", arguments={"url": LOOP}),
        act("skill.run", arguments={"skill_id": "t"}, steps=[act("http.post", arguments={"url": "https://public.example/x", "body": "b"})])])
    d = svc.authorize(inner_bad, t, {})
    assert d.verdict == DENY and "P-MODE-ENVELOPE" in pids(d)
    inner_needs = act("skill.run", arguments={"skill_id": "s"}, steps=[act("wait"), send(CHAN)])
    assert svc.authorize(inner_needs, t, {}).verdict == NEEDS
    fine = act("skill.run", arguments={"skill_id": "s"}, steps=[act("wait"), act("browser.navigate", arguments={"url": LOOP})])
    assert svc.authorize(fine, t, {}).verdict == ALLOW
    unknown_inside = act("skill.run", arguments={"skill_id": "s"}, steps=[act("warp.drive", C.Modality.API)])
    assert "P-TOOL-UNKNOWN" in pids(svc.authorize(unknown_inside, t, {}))


def test_malformed_actions_are_denied(svc):
    t = task()
    deep = act("skill.run")
    cur = deep
    for _ in range(10):
        nxt = act("skill.run")
        cur.steps = [nxt]
        cur = nxt
    wide = act("skill.run", steps=[act("wait") for _ in range(200)])
    bad = [act("browser.navigate", "telepathy", arguments={"url": LOOP}), act("wait", arguments={"x": object()}),
           act("wait", arguments={1: "non-string key"}), act("rm -rf /", C.Modality.CODE), act("wait", level="cosmic"),
           act("wait", est_cost=C.Cost(money=-5.0)), act("wait", est_cost=C.Cost(money=float("nan"))),
           act("wait", timeout_s=-1), act("wait", target="x" * 5000), act("wait", arguments={"blob": "y" * 300000}),
           deep, wide, act("wait", steps=["not an action"]), act("wait", arguments=["not", "a", "mapping"])]
    for a in bad:
        d = svc.authorize(a, t, {})
        assert d.verdict == DENY and "P-MALFORMED" in pids(d), (a.operation, d.reasons)
        assert not d.grant
    for junk in (None, "browser.navigate http://x", {"operation": "wait"}, 42):
        d = svc.authorize(junk, t, {})
        assert d.verdict == DENY and not d.grant


def test_every_decision_carries_reasons_and_policy_ids(svc):
    t = discord_task()
    actions = [act("wait"), act("nope.tool", C.Modality.API), send(CHAN), send("900000000000009999"),
               act("browser.navigate", arguments={"url": "https://public.example"}), act("sandbox.python", arguments={"code": "1"})]
    seen = set()
    for a in actions:
        d = svc.authorize(a, t, {})
        seen.add(d.verdict)
        assert d.reasons and d.policy_ids and all(isinstance(r, str) and r for r in d.reasons)
        assert d.action_hash == a.content_hash()
        assert bool(d.grant) == (d.verdict == ALLOW)
    assert seen == {ALLOW, DENY, NEEDS}


@pytest.mark.parametrize("breakage", ["syntax", "unknown_key", "bad_regex", "wrong_type", "bad_effect", "bad_pattern",
                                      "missing_section", "live_without_record", "missing_file"])
def test_malformed_policy_fails_closed(tmp_path, clock, breakage):
    def mutate(doc):
        if breakage == "unknown_key":
            doc["allow_everything"] = True
        elif breakage == "bad_regex":
            doc["prohibited"]["content_patterns"][0]["regex"] = "(unclosed"
        elif breakage == "wrong_type":
            doc["grants"]["ttl_s"] = "soon"
        elif breakage == "bad_effect":
            doc["tools"][0]["effect"] = "harmless"
        elif breakage == "bad_pattern":
            doc["modes"]["staging"]["owned"] = ["channel:discord"]
        elif breakage == "missing_section":
            del doc["prohibited"]
        elif breakage == "live_without_record":
            doc["modes"]["live"]["requires_authorization_record"] = False
    p = policy_file(tmp_path, mutate)
    if breakage == "syntax":
        open(p, "w").write("version: 1\n  tools: [unbalanced\n")
    if breakage == "missing_file":
        p = str(tmp_path / "does-not-exist.yaml")
    with pytest.raises(PolicyError):
        load_policy(p)
    s = PolicyService(str(tmp_path / "run"), p, clock=clock, fsync=False)
    for a in (act("wait"), act("browser.navigate", arguments={"url": LOOP}), act("sandbox.python", arguments={"code": "1"})):
        d = s.authorize(a, task(), {})
        assert d.verdict == DENY and d.policy_ids == ["P-POLICY-INVALID"] and not d.grant
    assert s.limits_state()["policy"]["valid"] is False
    s.emergency_stop("still works", "operator")
    assert s.stopped()
    s.reset_stop("operator")
    # a valid policy brings it back; a broken reload locks it again
    assert s.reload_policy("operator", DEFAULT_POLICY_PATH)["valid"] is True
    assert s.authorize(act("wait"), task(), {}).verdict == ALLOW
    assert s.reload_policy("operator", p)["valid"] is False
    assert s.authorize(act("wait"), task(), {}).verdict == DENY
    s.close()


def test_internal_errors_fail_closed(svc, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("unexpected")
    monkeypatch.setattr(svc._eval, "node", boom)
    a = act("wait")
    d = svc.authorize(a, task(), {})
    assert d.verdict == DENY and d.policy_ids == ["P-INTERNAL-ERROR"] and not d.grant
    assert svc.audit_tail(1)[0]["data"]["verdict"] == DENY


def test_policy_implements_the_contract(svc):
    assert isinstance(svc, C.PolicyEngine)
