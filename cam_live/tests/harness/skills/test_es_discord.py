"""Discord adapter against the local STAGING service (spec 12.5; requirement IH-05).

Nothing here talks to Discord. Every result is a staging result and is labelled as such by the adapter."""
import json
import socket

import httpx
import pytest

from cam_live.contracts import FailureClass, Modality, ModalityExecutor, ResultStatus, TaskSpec
from cam_live.harness import discord as discord_mod
from cam_live.harness.discord import (CONTENT_PERMISSION, LIVE_BASE_URL, DiscordAPIError, DiscordClient,
                                      DiscordExecutor, DiscordUnavailable, environment_of, member_id_hash)
from cam_live.harness.staging_discord import STAGING_HEADER, STAGING_LABEL, StagingDiscord, StagingDiscordServer
from es_support import ScopedPolicy, act, make_ctx

SCOPES = ("discord.read", "discord.write")


@pytest.fixture
def staging():
    server = StagingDiscordServer(StagingDiscord(rate_limit=(1000, 1.0)))
    yield server
    server.close()


def client_for(staging, **kw):
    return DiscordClient(staging.base_url, lambda: staging.state.token, **kw)


def executor_for(staging, **kw):
    kw.setdefault("secrets", {"discord_bot_token": staging.state.token})
    kw.setdefault("guild_id", staging.state.guild_id)
    return DiscordExecutor(staging.base_url, **kw)


def dact(name, args=None):
    return act(f"discord.{name}", args or {}, scope="discord.read" if name in (
        "guild_stats", "list_channels", "channel_activity", "list_invites", "list_events") else "discord.write")


# ---- the client ---------------------------------------------------------------------------------------------------
def test_members_bots_and_declared_test_accounts(staging):
    st = staging.state
    humans = [st.add_human(f"person{i}") for i in range(7)]
    st.add_human("pending person", pending=True)
    bots = [st.add_bot("helper bot"), st.add_bot("music bot")]
    tests = [st.add_test_account("operator test 1"), st.add_test_account("operator test 2")]
    c = client_for(staging)
    g = c.guild(st.guild_id)
    assert g["approximate_member_count"] == 13 and g["id"] == st.guild_id         # 8 people + 2 bots + 2 tests + the app
    members = c.list_members(st.guild_id, test_account_ids=st.test_account_ids, page_size=5)   # paged: 5 + 5 + 3
    assert len(members) == 13 and sum(1 for r in st.request_log if r["path"].endswith("/members")) == 3
    counts = c.member_counts(members)
    assert counts == {"total": 13, "bots": 3, "test_accounts": 2, "humans": 8, "pending": 1, "humans_not_pending": 7,
                      "distinct": 8}
    # data minimisation: nothing but what verification needs leaves the client
    assert all(set(m) == {"id_hash", "bot", "joined_at", "pending", "test_account"} for m in members)
    blob = json.dumps(members)
    assert "person" not in blob and "helper bot" not in blob and not any(uid in blob for uid in humans + bots + tests)
    assert member_id_hash(st.guild_id, humans[0]) in {m["id_hash"] for m in members}
    assert member_id_hash(st.guild_id, humans[0], "salt") != member_id_hash(st.guild_id, humans[0])
    # a member leaves; the same pseudonyms identify who stayed
    st.remove_member(humans[0])
    later = c.list_members(st.guild_id, test_account_ids=st.test_account_ids)
    assert c.member_counts(later)["humans"] == 7
    assert {m["id_hash"] for m in later} < {m["id_hash"] for m in members}
    st.add_human("late joiner")
    newest = c.list_members(st.guild_id, test_account_ids=st.test_account_ids)
    assert c.member_retention(members, newest) == {"cohort": 8, "retained": 7, "left": 1, "joined_since": 1}
    # without the members intent the list is refused by the platform (403 Missing Access)
    st.members_intent = False
    with pytest.raises(DiscordAPIError) as e:
        c.list_members(st.guild_id)
    assert (e.value.status, e.value.code) == (403, 50001)
    c.close()


def test_channels_messages_invites_events(staging):
    st = staging.state
    c = client_for(staging)
    assert [x["name"] for x in c.list_channels(st.guild_id)] == ["general"]
    cat = c.create_channel(st.guild_id, "resources", 4)
    ch = c.create_channel(st.guild_id, "guides", 0, topic="how-to documents", parent_id=cat["id"])
    assert ch["type"] == 0 and ch["parent_id"] == cat["id"] and st.channels[ch["id"]]["topic"] == "how-to documents"
    assert c.channel(ch["id"])["guild_id"] == st.guild_id
    m = c.send_message(ch["id"], "first words", nonce="n-1")
    assert set(m) == {"id", "channel_id", "timestamp", "edited_timestamp", "content_sha256", "content_length"}
    assert st.messages[ch["id"]][0]["content"] == "first words" and st.messages[ch["id"]][0]["author"]["bot"] is True
    assert c.send_message(ch["id"], "first words", nonce="n-1")["id"] == m["id"]      # idempotent with a nonce
    assert len(st.messages[ch["id"]]) == 1
    e = c.edit_message(ch["id"], m["id"], "revised words")
    assert e["id"] == m["id"] and e["edited_timestamp"] and st.messages[ch["id"]][0]["content"] == "revised words"
    inv = c.create_invite(ch["id"], max_age=3600, max_uses=25, unique=True)
    assert inv["url"] == f"https://discord.gg/{inv['code']}" and inv["uses"] == 0 and inv["max_uses"] == 25
    st.use_invite(inv["code"], "newcomer 1")
    st.use_invite(inv["code"], "newcomer 2")
    listed = c.list_invites(st.guild_id)
    assert [(i["code"], i["uses"], i["inviter_is_bot"]) for i in listed] == [(inv["code"], 2, True)]
    assert "inviter" not in listed[0] and "newcomer" not in json.dumps(listed)
    ev = c.create_scheduled_event(st.guild_id, "open session", "2027-01-10T18:00:00+00:00",
                                  scheduled_end_time="2027-01-10T19:00:00+00:00", location="the guides channel",
                                  description="questions and answers")
    st.mark_interested(ev["id"], 4)
    events = c.list_scheduled_events(st.guild_id)
    assert [(x["name"], x["user_count"], x["location"]) for x in events] == [("open session", 4, "the guides channel")]
    # error bodies carry Discord's codes
    for call, status, code in ((lambda: c.send_message("123456789", "x"), 404, 10003),
                               (lambda: c.guild("123456789"), 404, 10004),
                               (lambda: c.edit_message(ch["id"], "123456789", "x"), 404, 10008),
                               (lambda: c.send_message(ch["id"], "x" * 2001), 400, 50035),
                               (lambda: c.create_invite(ch["id"], max_uses=1000), 400, 50035)):
        with pytest.raises(DiscordAPIError) as err:
            call()
        assert (err.value.status, err.value.code) == (status, code)
    c.close()


def test_auth_header_user_agent_and_staging_label(staging):
    seen = []
    c = client_for(staging, transport=_Spy(seen))
    c.guild(staging.state.guild_id)
    req = seen[0]
    assert req.headers["authorization"] == f"Bot {staging.state.token}" and req.headers["user-agent"].startswith("DiscordBot (")
    assert req.url.path.endswith(f"/api/v10/guilds/{staging.state.guild_id}") and req.url.params["with_counts"] == "true"
    raw = httpx.get(staging.base_url + "/users/@me", headers={"Authorization": f"Bot {staging.state.token}"})
    assert raw.headers[STAGING_HEADER] == STAGING_LABEL and raw.json()["bot"] is True
    assert c.environment == "staging" and environment_of(LIVE_BASE_URL) == "live"
    # a wrong token is a 401 from the service
    bad = DiscordClient(staging.base_url, lambda: "wrong-token")
    with pytest.raises(DiscordAPIError) as e:
        bad.guild(staging.state.guild_id)
    assert e.value.status == 401
    # a user token is never sent in any form: the only scheme the client knows is "Bot"
    assert all(r.headers["authorization"].startswith("Bot ") for r in seen)
    c.close()
    bad.close()


class _Spy(httpx.HTTPTransport):
    def __init__(self, seen):
        super().__init__()
        self.seen = seen

    def handle_request(self, request):
        self.seen.append(request)
        return super().handle_request(request)


def test_rate_limits_are_read_from_headers_and_waited_for(staging):
    st = staging.state
    st.set_rate_limit(2, 0.3)
    waits = []

    def sleep(s):
        waits.append(s)
        import time
        time.sleep(s)

    c = client_for(staging, sleep=sleep)
    for _ in range(5):                                   # 5 calls through a 2-per-0.3s bucket
        c.list_channels(st.guild_id)
    codes = [r for r in st.request_log if r["path"].endswith("/channels")]
    assert len(codes) == 5                               # the client paused on "remaining: 0" instead of drawing a 429
    assert len(waits) == 2 and all(0 < w <= 0.3 for w in waits) and c.rate_limit_waits_s == pytest.approx(sum(waits))
    assert c.last_rate_limit.limit == 2.0 and c.last_rate_limit.bucket
    # a 429 that does arrive (another client used the bucket) is retried only after the stated retry_after
    other = client_for(staging)
    st.set_rate_limit(1, 0.25)
    other.list_channels(st.guild_id)
    waits.clear()
    fresh = client_for(staging, sleep=sleep)
    fresh.list_channels(st.guild_id)                     # first answer is 429 with retry_after; second succeeds
    assert len(waits) == 1 and 0 < waits[0] <= 0.25 and fresh.requests_sent == 2
    # a global limit is honoured across routes; a wait beyond the bound is reported, not slept through
    st.set_rate_limit(1000, 1.0)
    st.block_globally(120.0)
    with pytest.raises(DiscordAPIError) as e:
        fresh.guild(st.guild_id)
    assert e.value.status == 429 and e.value.is_global and 100 < e.value.retry_after_s <= 120
    sent = fresh.requests_sent
    with pytest.raises(DiscordAPIError) as e:
        fresh.list_invites(st.guild_id)                  # a different route: held back locally, nothing is sent
    assert e.value.sent is False and fresh.requests_sent == sent
    for x in (c, other, fresh):
        x.close()


# ---- the executor -------------------------------------------------------------------------------------------------
def test_executor_tools_describe_capabilities_only(staging):
    ex = executor_for(staging)
    assert isinstance(ex, ModalityExecutor) and ex.modality == Modality.API
    tools = {t.name: t for t in ex.tools()}
    assert set(tools) == {"discord.guild_stats", "discord.list_channels", "discord.create_channel", "discord.send_message",
                          "discord.edit_message", "discord.channel_activity", "discord.list_invites",
                          "discord.create_invite", "discord.list_events", "discord.create_event"}
    assert all(t.available and not t.unavailable_reason for t in tools.values())
    assert tools["discord.send_message"].side_effects == "external_write" and tools["discord.guild_stats"].side_effects == "external_read"
    # descriptions say what a tool does; there is no advice on when or why, and no direct-message or user-account tool
    text = " ".join(t.description.lower() for t in tools.values())
    for word in ("should", "grow", "attract", "engage", "promot", "best", "strategy", "schedule a post", "tip", "recommend"):
        assert word not in text, word
    assert not any(k in " ".join(tools) for k in ("dm", "direct", "user", "friend", "relationship"))
    assert not any(hasattr(DiscordClient, m) for m in ("create_dm", "send_dm", "login", "list_dms", "user_messages"))
    assert "users/@me/channels" not in open(discord_mod.__file__).read().split('"""', 2)[2]
    ex.close()


def test_stats_messages_invites_through_the_executor(staging):
    st = staging.state
    for i in range(5):
        st.add_human(f"p{i}")
    st.add_bot("b")
    st.add_test_account("t")
    ex, pol = executor_for(staging, test_account_ids=st.test_account_ids), ScopedPolicy(scopes=SCOPES)
    ctx = make_ctx(pol)
    res = ex.execute(dact("guild_stats"), ctx)
    assert res.status == ResultStatus.OK
    obs = res.observations[0]
    q = {x.name: x.value for x in obs.quantities}
    assert q["discord.members.humans"] == 5.0 and q["discord.members.bots"] == 2.0 and q["discord.members.test_accounts"] == 1.0
    assert q["discord.members.approximate"] == 8.0 and q["discord.members.total"] == 8.0 and "discord.ratelimit.remaining" in q
    # every observation says it is a staging result
    assert obs.provenance.source == "discord:staging" and obs.data["staging"] is True and obs.text.startswith("[staging]")
    assert obs.provenance.trust == "untrusted"
    assert pol.checks[0].op == "discord.guild_stats" and pol.checks[0].arguments["environment"] == "staging"
    made = ex.execute(dact("create_channel", {"name": "notes", "type": "text"}), ctx)
    cid = made.artifacts["channel_id"]
    sent = ex.execute(dact("send_message", {"channel_id": cid, "content": "written by the agent"}), ctx)
    assert sent.status == ResultStatus.OK and st.messages[cid][0]["content"] == "written by the agent"
    # the API's echo of the message is not turned into an observation (the text is the agent's own action argument)
    assert "written by the agent" not in json.dumps([o.to_dict() for o in sent.observations])
    assert pol.checks[-1].arguments["content"] == "written by the agent"           # the policy judged the literal text
    edited = ex.execute(dact("edit_message", {"channel_id": cid, "message_id": sent.artifacts["message_id"],
                                              "content": "second version"}), ctx)
    assert edited.status == ResultStatus.OK and st.messages[cid][0]["content"] == "second version"
    inv = ex.execute(dact("create_invite", {"channel_id": cid, "max_age_s": 600, "unique": True}), ctx)
    st.use_invite(inv.artifacts["invite_code"])
    invites = ex.execute(dact("list_invites"), ctx)
    q = {x.name: x.value for x in invites.observations[0].quantities}
    assert q["discord.invites.count"] == 1.0 and q["discord.invites.uses"] == 1.0
    ev = ex.execute(dact("create_event", {"name": "session", "start_time": "2027-01-10T18:00:00+00:00",
                                          "end_time": "2027-01-10T19:00:00+00:00", "location": "online"}), ctx)
    assert ev.status == ResultStatus.OK and ex.execute(dact("list_events"), ctx).observations[0].data["events"][0]["name"] == "session"
    assert ex.execute(dact("create_event", {"name": "s", "start_time": "2027-01-10T18:00:00+00:00"}), ctx).failure_class \
        == FailureClass.INVALID_ARGUMENTS
    ex.close()


def test_message_content_is_dropped_by_default(staging):
    st = staging.state
    alice, bob = st.add_human("alice"), st.add_human("bob")
    st.post_member_message(st.general_id, alice, "my private opinion about pineapples")
    st.post_member_message(st.general_id, alice, "a second remark")
    st.post_member_message(st.general_id, bob, "bob's sentence")
    pol = ScopedPolicy(scopes=SCOPES)
    ex = executor_for(staging)
    res = ex.execute(dact("channel_activity", {"channel_id": st.general_id}), make_ctx(pol))
    assert res.status == ResultStatus.OK
    q = {x.name: x.value for x in res.observations[0].quantities}
    assert q["discord.channel.messages"] == 3.0 and q["discord.channel.human_messages"] == 3.0
    assert q["discord.channel.distinct_human_authors"] == 2.0 and q["discord.channel.bot_messages"] == 0.0
    blob = json.dumps(res.to_dict())
    for leaked in ("pineapples", "second remark", "bob's sentence", "alice", "bob", alice, bob):
        assert leaked not in blob, leaked
    assert res.observations[0].data["activity"]["content_included"] is False and "not retained" in res.observations[0].text
    # the client drops it too unless explicitly told otherwise
    c = client_for(staging)
    assert "contents" not in c.channel_activity(st.general_id, guild_id=st.guild_id)
    c.close()
    # only an explicit data permission in the task lets text through; authors stay pseudonymous even then
    task = TaskSpec(objective_id="o", objective="x", resources={"discord": {"guild_id": st.guild_id}},
                    data_permissions={CONTENT_PERMISSION: True})
    allowed = DiscordExecutor.from_task(task, base_url=staging.base_url, secrets={"discord_bot_token": st.token})
    res = allowed.execute(dact("channel_activity", {"channel_id": st.general_id}), make_ctx(pol))
    assert "pineapples" in res.observations[0].text and res.observations[0].data["activity"]["content_included"] is True
    assert "alice" not in json.dumps(res.to_dict()) and alice not in json.dumps(res.to_dict())
    denied = DiscordExecutor.from_task(TaskSpec(objective_id="o", objective="x",
                                                resources={"discord": {"guild_id": st.guild_id}},
                                                data_permissions={CONTENT_PERMISSION: False}),
                                       base_url=staging.base_url, secrets={"discord_bot_token": st.token})
    assert "pineapples" not in json.dumps(denied.execute(dact("channel_activity", {"channel_id": st.general_id}),
                                                         make_ctx(pol)).to_dict())
    for x in (ex, allowed, denied):
        x.close()


def test_direct_messages_and_other_servers_are_never_touched(staging):
    st = staging.state
    dm = st.add_dm_channel()
    ex, pol = executor_for(staging), ScopedPolicy(scopes=SCOPES)
    for name, args in (("channel_activity", {"channel_id": dm}), ("send_message", {"channel_id": dm, "content": "hello"}),
                       ("create_invite", {"channel_id": dm}), ("channel_activity", {"channel_id": "999999999999"})):
        res = ex.execute(dact(name, args), make_ctx(pol))
        assert res.status == ResultStatus.DENIED and res.failure_class == FailureClass.MISSING_PERMISSION
        assert "not a channel of the authorized server" in res.failure_detail and res.retries == 0
    paths = [r["path"] for r in st.request_log]
    assert not any(dm in p for p in paths) and all(p.endswith("/channels") for p in paths)   # only the guild's channel list was read
    assert "private words" not in json.dumps([m for m in st.messages[dm] if m["author"].get("bot")])
    assert len(st.messages[dm]) == 1                                                        # nothing was posted there
    ex.close()


def test_tools_are_unavailable_without_credentials_and_nothing_is_sent(staging, monkeypatch):
    st = staging.state
    connections = []
    real_connect = socket.socket.connect
    monkeypatch.setattr(socket.socket, "connect", lambda self, addr: (connections.append(addr), real_connect(self, addr))[1])
    pol = ScopedPolicy(scopes=SCOPES)
    cases = {
        "no token": (DiscordExecutor(staging.base_url, guild_id=st.guild_id, secrets={}),
                     "blocked: no Discord bot token provided (secret 'discord_bot_token')"),
        "no broker": (DiscordExecutor(LIVE_BASE_URL, guild_id=st.guild_id), "blocked: no Discord bot token provided (secret 'discord_bot_token')"),
        "no guild": (DiscordExecutor(staging.base_url, secrets={"discord_bot_token": st.token}),
                     "blocked: no owner-authorized Discord guild is configured"),
    }
    for label, (ex, reason) in cases.items():
        tools = ex.tools()
        assert len(tools) == 10 and all(not t.available and t.unavailable_reason == reason for t in tools), label
        for name, args in (("guild_stats", {}), ("send_message", {"channel_id": st.general_id, "content": "x"})):
            res = ex.execute(dact(name, args), make_ctx(pol))
            assert res.status == ResultStatus.DENIED and res.failure_class == FailureClass.MISSING_PERMISSION, label
            assert res.failure_detail == reason and res.retries == 0 and res.ops == []
        ex.close()
    assert st.request_log == [] and connections == [] and pol.checks == []
    # the client itself refuses before any network activity
    c = DiscordClient(LIVE_BASE_URL, lambda: None)
    with pytest.raises(DiscordUnavailable, match="no Discord bot token"):
        c.guild("1234567")
    assert c.requests_sent == 0 and connections == []
    c.close()


def test_denials_platform_rules_and_missing_permissions_are_hard_failures(staging):
    st = staging.state
    ex = executor_for(staging)
    # the authorization layer refuses: nothing reaches the service
    pol = ScopedPolicy(scopes=("discord.read",))
    res = ex.execute(dact("send_message", {"channel_id": st.general_id, "content": "x"}), make_ctx(pol))
    assert res.status == ResultStatus.DENIED and st.request_log == []
    waiting = ScopedPolicy(scopes=SCOPES, approval_rule=lambda op, a: op.op == "discord.send_message")
    res = ex.execute(dact("send_message", {"channel_id": st.general_id, "content": "x"}), make_ctx(waiting))
    assert res.status == ResultStatus.NEEDS_APPROVAL and st.messages[st.general_id] == [] and len(waiting.approvals()) == 1
    # the bot lacks a Discord permission: 403 / 50013 -> missing permission, sent once, never retried
    pol = ScopedPolicy(scopes=SCOPES)
    st.set_permission("SEND_MESSAGES", False)
    n = len(st.request_log)
    res = ex.execute(dact("send_message", {"channel_id": st.general_id, "content": "x"}), make_ctx(pol))
    assert res.status == ResultStatus.FAILED and res.failure_class == FailureClass.MISSING_PERMISSION and res.retries == 0
    assert sum(1 for r in st.request_log[n:] if r["path"].endswith("/messages")) == 1
    st.set_permission("MANAGE_GUILD", False)
    assert ex.execute(dact("list_invites"), make_ctx(pol)).failure_class == FailureClass.MISSING_PERMISSION
    # a platform maximum (error code 30013) is a platform restriction
    for i in range(499):
        st.add_channel(f"c{i}")
    res = ex.execute(dact("create_channel", {"name": "one too many"}), make_ctx(pol))
    assert res.failure_class == FailureClass.PLATFORM_RESTRICTION and res.retries == 0 and "30013" in res.failure_detail
    # without the members intent, statistics degrade to the approximate counts and say so
    st.members_intent = False
    res = ex.execute(dact("guild_stats"), make_ctx(pol))
    assert res.status == ResultStatus.OK and res.observations[0].data["members_listed"] is False
    assert "discord.members.humans" not in {x.name for x in res.observations[0].quantities}
    ex.close()


def test_rate_limited_and_unreachable_service_through_the_executor(staging):
    st = staging.state
    waits = []
    ex, pol = executor_for(staging, sleep=waits.append), ScopedPolicy(scopes=SCOPES)
    st.block_globally(500.0)
    res = ex.execute(dact("list_channels"), make_ctx(pol))
    assert res.status == ResultStatus.FAILED and res.failure_class == FailureClass.RESOURCE_LIMIT and res.retries == 0
    assert waits == [] and 400 < {x.name: x.value for o in res.observations for x in o.quantities}["exec.failure.retry_after_s"] <= 500
    ex.close()
    down = DiscordExecutor("http://127.0.0.1:9/api/v10", guild_id=st.guild_id, secrets={"discord_bot_token": "t"},
                           sleep=lambda s: None)
    res = down.execute(dact("guild_stats"), make_ctx(pol))
    assert res.failure_class == FailureClass.TRANSIENT_NETWORK and res.retries == 3
    # a write that is not idempotent is not repeated after an uncertain failure
    res = down.execute(dact("create_channel", {"name": "x"}), make_ctx(pol))
    assert res.failure_class == FailureClass.TRANSIENT_NETWORK and res.retries == 0
    down.close()
