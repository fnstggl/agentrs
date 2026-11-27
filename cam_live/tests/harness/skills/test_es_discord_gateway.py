"""The bounded Gateway window (``cam_live.harness.discord_gateway``) and ``discord.listen_events`` against the local
STAGING Gateway (spec 12.5; requirement IH-05).

Nothing here talks to Discord. Every result is a staging result and is labelled as such by the adapter."""
import json
import random
import threading

import pytest

from cam_live.contracts import FailureClass, ResultStatus, TaskSpec
from cam_live.harness.discord import CONTENT_PERMISSION, DiscordExecutor
from cam_live.harness.discord_gateway import (CLOSE_CODES, DEFAULT_INTENTS, FORBIDDEN_INTENTS, GatewayError,
                                              GatewayWindow, describe_close, gateway_connect_url)
from cam_live.harness.discord_perms import INTENT_BITS, intents_value, member_id_hash
from cam_live.harness.staging_discord import StagingDiscord, StagingDiscordServer, StagingGateway
from es_support import GrantBoundBroker, ScopedPolicy, act, make_ctx

TOKEN = "MTIzNDU2Nzg5MDEyMzQ1Njc4.staging.not-a-real-token-0123456789abcdef"


@pytest.fixture
def staging():
    server = StagingDiscordServer(StagingDiscord(token=TOKEN, rate_limit=(1000, 1.0)))
    yield server
    server.close()


def window_for(staging, seconds=0.5, **kw):
    kw.setdefault("guild_id", staging.state.guild_id)
    return GatewayWindow(staging.gateway.url, kw.pop("token", lambda: staging.state.token), seconds=seconds, **kw)


def executor_for(staging, **kw):
    kw.setdefault("secrets", {"discord_bot_token": staging.state.token})
    kw.setdefault("guild_id", staging.state.guild_id)
    return DiscordExecutor(staging.base_url, **kw)


def listen(ex, staging, pol, seconds=1, secrets=None, **args):
    action = act("discord.listen_events", dict(args, seconds=seconds, guild_id=staging.state.guild_id), scope="discord")
    return ex.execute(action, make_ctx(pol, secrets=secrets)), action


# ---- the protocol ----------------------------------------------------------------------------------------------------
def test_protocol_constants_come_from_the_documentation():
    assert gateway_connect_url("wss://gateway.discord.gg/") == "wss://gateway.discord.gg/?v=10&encoding=json"
    assert gateway_connect_url("ws://127.0.0.1:1234") == "ws://127.0.0.1:1234/?v=10&encoding=json"
    assert intents_value(DEFAULT_INTENTS) == (1 << 0) | (1 << 1) | (1 << 6) | (1 << 9) | (1 << 16) == 66115
    assert INTENT_BITS["GUILD_PRESENCES"] == 8 and INTENT_BITS["MESSAGE_CONTENT"] == 15 and INTENT_BITS["DIRECT_MESSAGES"] == 12
    assert [CLOSE_CODES[c][2] for c in (4004, 4010, 4011, 4012, 4013, 4014)] == [False] * 6      # codes not to reconnect on
    assert "Authentication failed" in describe_close(4004) and "token" in describe_close(4004)
    assert "Invalid intent(s)" in describe_close(4013) and "Disallowed intent(s)" in describe_close(4014)
    assert "not enabled or not approved" in describe_close(4014) and "without a close code" in describe_close(None)
    # the intents this adapter never asks for, and the bound on a window
    for bad in FORBIDDEN_INTENTS:
        with pytest.raises(ValueError, match="never requested"):
            GatewayWindow("ws://127.0.0.1:1", lambda: "t", guild_id="1", seconds=1, intents=DEFAULT_INTENTS + (bad,))
    assert "GUILD_PRESENCES" in FORBIDDEN_INTENTS and "DIRECT_MESSAGES" in FORBIDDEN_INTENTS
    with pytest.raises(ValueError, match="only together with keep_content"):
        GatewayWindow("ws://127.0.0.1:1", lambda: "t", guild_id="1", seconds=1, intents=DEFAULT_INTENTS + ("MESSAGE_CONTENT",))
    for seconds in (0, 61, 3600):
        with pytest.raises(ValueError, match="at most 60 seconds"):
            GatewayWindow("ws://127.0.0.1:1", lambda: "t", guild_id="1", seconds=seconds)
    assert GatewayWindow("ws://127.0.0.1:1", lambda: "t", guild_id="1", seconds=1, keep_content=True).intents_value == 66115 | (1 << 15)


def test_window_receives_member_add_and_remove_and_reduces_them(staging):
    st, gw = staging.state, staging.gateway
    tester = st.add_test_account("declared test account")
    made = {}

    def happenings():
        made["joined"] = st.add_human("a newcomer", pending=True)
        made["bot"] = st.add_bot("a helper bot")
        st.update_member(made["joined"], pending=False)
        st.remove_member(made["joined"])
        st.remove_member(tester)

    gw.on_ready(happenings)
    out = window_for(staging, 0.5, id_salt="pepper", test_account_ids=st.test_account_ids).run()
    assert out["session_ok"] and out["ended_by"] == "window" and 0.45 <= out["window_s"] < 3.0
    assert out["counts"] == {"READY": 1, "GUILD_CREATE": 1, "GUILD_MEMBER_ADD": 2, "GUILD_MEMBER_UPDATE": 1, "GUILD_MEMBER_REMOVE": 2}
    ready, created, add, add_bot, update, remove, remove_test = out["events"]
    assert ready == {"event": "READY", "guilds": 1, "configured_guild_listed": True}
    assert created == {"event": "GUILD_CREATE", "member_count": 2, "channels": 1, "unavailable": False}
    h = member_id_hash(st.guild_id, made["joined"], "pepper")
    assert add == {"event": "GUILD_MEMBER_ADD", "id_hash": h, "bot": False, "test_account": False, "pending": True,
                   "joined_at": add["joined_at"]} and add["joined_at"].startswith("20")
    assert add_bot["bot"] is True and add_bot["id_hash"] == member_id_hash(st.guild_id, made["bot"], "pepper")
    assert update == dict(add, event="GUILD_MEMBER_UPDATE", pending=False)
    assert remove == {"event": "GUILD_MEMBER_REMOVE", "id_hash": h, "bot": False, "test_account": False}
    assert remove_test == {"event": "GUILD_MEMBER_REMOVE", "id_hash": member_id_hash(st.guild_id, tester, "pepper"),
                           "bot": False, "test_account": True}
    # nothing raw left the module: no id, no name, no session id, no resume address
    blob = json.dumps(out)
    for leaked in (made["joined"], made["bot"], tester, "newcomer", "helper bot", "declared test", "staging-app",
                   st.bot_user["id"], "staging-session", gw.url, TOKEN):
        assert leaked not in blob, leaked
    # what the client sent: one Identify with exactly the documented default intents, then a normal close (1000)
    assert [f["op"] for f in gw.frames_received if f["op"] != 1] == [2]
    ident = gw.identify_log[0]
    assert ident["intents"] == 66115 and ident["token_ok"] and ident["outcome"] == "ready"
    assert ident["intent_names"] == ["GUILDS", "GUILD_INVITES", "GUILD_MEMBERS", "GUILD_MESSAGES", "GUILD_SCHEDULED_EVENTS"]
    assert set(ident["properties"]) == {"os", "browser", "device"} and TOKEN not in json.dumps(gw.identify_log)
    assert _wait(lambda: gw.closes_received == [1000])


def _wait(cond, timeout=2.0):
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def test_invites_messages_channels_and_scheduled_events_are_reduced(staging):
    st, gw = staging.state, staging.gateway
    human = st.add_human("the inviter")

    def happenings():
        cid = st.add_channel("a new room")
        st.channels[cid]["name"] = "renamed room"
        st._emit("CHANNEL_UPDATE", dict(st.channels[cid]))
        st.post_member_message(st.general_id, human, "eleven chars")
        gw.emit("INVITE_CREATE", {"channel_id": st.general_id, "code": "abc123", "guild_id": st.guild_id, "max_age": 600,
                                  "max_uses": 5, "uses": 0, "temporary": False, "inviter": dict(st.members[human]["user"]),
                                  "created_at": "2026-11-27T00:00:00+00:00", "expires_at": None})
        gw.emit("INVITE_DELETE", {"channel_id": st.general_id, "code": "abc123", "guild_id": st.guild_id})
        st.guild["name"] = "a new name"
        st._emit("GUILD_UPDATE", st._guild_object(False))
        gw.emit("GUILD_SCHEDULED_EVENT_CREATE", {"id": "777000111", "guild_id": st.guild_id, "name": "open hour",
                                                 "creator": dict(st.members[human]["user"]), "creator_id": human})
        gw.emit("GUILD_SCHEDULED_EVENT_USER_ADD", {"guild_scheduled_event_id": "777000111", "user_id": human, "guild_id": st.guild_id})
        gw.emit("GUILD_ROLE_CREATE", {"guild_id": st.guild_id, "role": {"id": "5", "name": "r"}})
        gw.emit("TYPING_START", {"guild_id": st.guild_id, "user_id": human})                # needs an intent we do not ask for
        gw.emit("PRESENCE_UPDATE", {"guild_id": st.guild_id, "user": {"id": human}, "status": "online"})
        gw.emit("MESSAGE_CREATE", {"id": "1", "channel_id": "424242", "author": {"id": human}, "content": "a direct message"})

    gw.on_ready(happenings)
    out = window_for(staging, 0.5).run()
    by = {}
    for e in out["events"]:
        by.setdefault(e["event"], []).append(e)
    assert by["CHANNEL_CREATE"][0]["name"] == "a new room" and by["CHANNEL_UPDATE"][0]["name"] == "renamed room"
    assert set(by["CHANNEL_CREATE"][0]) == {"event", "id", "name", "type"} and by["CHANNEL_CREATE"][0]["type"] == 0
    assert by["MESSAGE_CREATE"] == [{"event": "MESSAGE_CREATE", "channel_id": st.general_id, "bot": False, "length": 0,
                                     "author_id_hash": member_id_hash(st.guild_id, human)}]     # no content intent: no text, length 0
    assert by["INVITE_CREATE"] == [{"event": "INVITE_CREATE", "code": "abc123", "channel_id": st.general_id, "max_age": 600, "max_uses": 5}]
    assert by["INVITE_DELETE"] == [{"event": "INVITE_DELETE", "code": "abc123", "channel_id": st.general_id}]
    assert by["GUILD_UPDATE"] == [{"event": "GUILD_UPDATE", "name": "a new name"}]
    assert by["GUILD_SCHEDULED_EVENT_CREATE"] == [{"event": "GUILD_SCHEDULED_EVENT_CREATE", "id": "777000111", "name": "open hour"}]
    assert by["GUILD_SCHEDULED_EVENT_USER_ADD"] == [{"event": "GUILD_SCHEDULED_EVENT_USER_ADD", "id": "777000111"}]
    # other events of the guild are counted by name and dropped; an event without a guild (a direct message) is dropped
    assert out["dropped"] == {"other_guild": 0, "no_guild": 1, "unreduced": {"GUILD_ROLE_CREATE": 1}}
    assert "PRESENCE_UPDATE" not in json.dumps(out) and "TYPING_START" not in json.dumps(out)   # never delivered: no such intent
    blob = json.dumps(out)
    for leaked in (human, "the inviter", "eleven chars", "a direct message", st.guild["owner_id"], "creator"):
        assert leaked not in blob, leaked
    # with the content permission the same message arrives with its text (authors stay pseudonymous)
    gw.on_ready(lambda: st.post_member_message(st.general_id, human, "eleven chars"))
    out = window_for(staging, 0.4, keep_content=True).run()
    msg = [e for e in out["events"] if e["event"] == "MESSAGE_CREATE"][0]
    assert msg["content"] == "eleven chars" and msg["length"] == 12 and out["content_included"] is True
    assert gw.identify_log[-1]["intents"] == 66115 | (1 << 15) and human not in json.dumps(out)


def test_events_of_other_guilds_are_counted_and_dropped(staging):
    st, gw = staging.state, staging.gateway
    other = st.add_other_guild("another server")
    stranger = {"id": "888000111222", "username": "a stranger"}

    def happenings():
        gw.emit_other_guild("GUILD_MEMBER_ADD", {"user": stranger, "joined_at": "2026-01-01T00:00:00+00:00"})
        gw.emit_other_guild("MESSAGE_CREATE", {"id": "9", "channel_id": "5550001", "author": stranger, "content": "elsewhere"})
        gw.emit_other_guild("GUILD_UPDATE", {"name": "another server renamed"})
        gw.emit_other_guild("INVITE_CREATE", {"code": "othercode", "channel_id": "5550001"})
        st.add_human("one of ours")

    gw.on_ready(happenings)
    out = window_for(staging, 0.5).run()
    assert out["guilds"] == 2 and out["configured_guild_listed"] is True
    assert out["dropped"]["other_guild"] == 5                       # its Guild Create and the four events above
    assert out["counts"] == {"READY": 1, "GUILD_CREATE": 1, "GUILD_MEMBER_ADD": 1}
    blob = json.dumps(out)
    for leaked in (other, "another server", "a stranger", "888000111222", "elsewhere", "othercode", "5550001"):
        assert leaked not in blob, leaked
    # a bot that is not in the configured guild at all gets a session and no event of it
    st.remove_bot_from_guild()
    out = window_for(staging, 0.3).run()
    assert out["session_ok"] and out["configured_guild_listed"] is False and out["counts"] == {"READY": 1}
    assert out["dropped"]["other_guild"] == 1


def test_heartbeats_jitter_ack_request_and_a_connection_that_stops_answering():
    with StagingDiscordServer(StagingDiscord(token=TOKEN, rate_limit=(1000, 1.0)), heartbeat_interval_ms=100) as staging:
        st, gw = staging.state, staging.gateway
        gw.on_ready(lambda: (st.add_human("x"), gw.send_op(1)))          # an event, then the gateway asks for a heartbeat now
        out = window_for(staging, 0.65, rng=random.Random(7)).run()
        hb = out["heartbeats"]
        assert hb["interval_s"] == 0.1 and hb["requested_by_gateway"] == 1 and out["ended_by"] == "window"
        assert 3 <= hb["sent"] <= 9 and hb["acknowledged"] >= hb["sent"] - 1      # about one per 0.1 s, plus the requested one
        beats = [f["d"] for f in gw.frames_received if f["op"] == 1]
        assert len(beats) == hb["sent"] and beats[-1] == 3              # each carries the last sequence number received
        assert all(b is None or isinstance(b, int) for b in beats) and beats == sorted(beats, key=lambda v: v or 0)
        # the first heartbeat waits interval * jitter: with jitter ~1 none is sent in a window shorter than the interval
        gw.frames_received.clear()

        class Late(random.Random):
            def random(self):
                return 0.999

        out = window_for(staging, 0.08, rng=Late()).run()
        assert out["heartbeats"]["sent"] == 0 and not [f for f in gw.frames_received if f["op"] == 1]
        # no ACK between two heartbeats: the window ends and the connection is closed with a code other than 1000
        assert _wait(lambda: gw.closes_received == [1000, 1000])
        gw.closes_received.clear()
        gw.acknowledge_heartbeats = False
        out = window_for(staging, 5.0, rng=random.Random(7)).run()
        assert out["ended_by"] == "heartbeat_not_acknowledged" and out["window_s"] < 4.0 and out["heartbeats"]["acknowledged"] == 0
        assert _wait(lambda: gw.closes_received == [4000])


def test_reconnect_invalid_session_and_refused_identifies(staging):
    st, gw = staging.state, staging.gateway
    # opcode 7 / opcode 9 during a session: the window ends early and says why; there is no second Identify
    gw.on_ready(lambda: (st.add_human("x"), gw.send_op(7)))
    out = window_for(staging, 5.0).run()
    assert out["ended_by"] == "reconnect_requested" and out["session_ok"] and out["window_s"] < 4.0 and out["counts"]["GUILD_MEMBER_ADD"] == 1
    gw.on_ready(lambda: gw.send_op(9, False))
    out = window_for(staging, 5.0).run()
    assert out["ended_by"] == "invalid_session" and out["resumable"] is False and out["window_s"] < 4.0
    assert len(gw.identify_log) == 2 and not [f for f in gw.frames_received if f["op"] == 6]        # never resumed
    # opcode 9 in answer to Identify: no session was started
    gw.identify_response = "invalid_session"
    with pytest.raises(GatewayError) as e:
        window_for(staging, 5.0).run()
    assert e.value.kind == "limit" and "Invalid Session (opcode 9" in str(e.value) and TOKEN not in str(e.value)
    gw.identify_response = "reconnect"
    with pytest.raises(GatewayError) as e:
        window_for(staging, 5.0).run()
    assert e.value.kind == "network" and "opcode 7" in str(e.value)
    # 4014: a privileged intent the application has not enabled
    st.application["flags"] = 1 << 19                                  # message content only: no members intent
    with pytest.raises(GatewayError) as e:
        window_for(staging, 5.0).run()
    assert (e.value.kind, e.value.close_code) == ("permission", 4014)
    assert str(e.value) == ("gateway close code 4014 (Disallowed intent(s)): A disallowed intent for a Gateway Intent "
                            "was sent: an intent that is not enabled or not approved for the application.")
    assert window_for(staging, 0.2, intents=("GUILDS", "GUILD_MESSAGES")).run()["session_ok"]       # without it: fine
    # 4013: intent bits that do not exist
    w = window_for(staging, 5.0)
    w.intents_value |= 1 << 30
    with pytest.raises(GatewayError) as e:
        w.run()
    assert (e.value.kind, e.value.close_code) == ("permission", 4013) and "Invalid intent(s)" in str(e.value)
    # 4004: a wrong token; the token is in no message
    wrong = "wrong-token-" + "z" * 40
    with pytest.raises(GatewayError) as e:
        window_for(staging, 5.0, token=lambda: wrong).run()
    assert (e.value.kind, e.value.close_code) == ("permission", 4004) and "Authentication failed" in str(e.value)
    assert wrong not in str(e.value) and wrong not in repr(e.value) and wrong not in json.dumps(gw.identify_log)
    # a close in the middle of a session with an ordinary code ends the window and is reported
    st.application["flags"] = (1 << 15) | (1 << 19)
    gw.on_ready(lambda: gw.close_sessions(4009, "Session timed out."))
    out = window_for(staging, 5.0).run()
    assert out["ended_by"] == "closed:4009" and out["close_code"] == 4009 and out["session_ok"]
    assert [x["outcome"] for x in gw.identify_log] == ["ready", "ready", "op:9", "op:7", "close:4014", "ready", "close:4013",
                                                      "close:4004", "ready"]


def test_the_token_is_in_no_exception_and_no_result(staging):
    st, gw = staging.state, staging.gateway
    # no token: nothing connects
    with pytest.raises(GatewayError, match="no Discord bot token was released for the gateway destination") as e:
        window_for(staging, 1.0, token=lambda: None).run()
    assert e.value.kind == "permission" and e.value.close_code is None and gw.identify_log == []
    # an address nobody listens on
    with pytest.raises(GatewayError) as e:
        GatewayWindow("ws://127.0.0.1:9", lambda: TOKEN, guild_id=st.guild_id, seconds=1, open_timeout_s=2.0).run()
    assert e.value.kind == "network" and TOKEN not in str(e.value)

    # a transport that fails and repeats what it was given in its error text: the text is cleaned
    def echoing_connect(url, **kw):
        raise OSError(f"proxy said no; payload was {{'token': '{TOKEN}'}}")

    with pytest.raises(GatewayError) as e:
        window_for(staging, 1.0, connect=echoing_connect).run()
    assert TOKEN not in str(e.value) and "[token]" in str(e.value) and e.value.__cause__ is None and e.value.__context__ is None

    class Flaky:
        def __init__(self, ws):
            self.ws = ws

        def send(self, data):
            raise OSError("write failed while sending " + data)                 # would carry the Identify payload

        def recv(self, timeout=None):
            return self.ws.recv(timeout=timeout)

        def close(self, code=1000):
            self.ws.close(code)

    from websockets.sync.client import connect as real_connect
    with pytest.raises(GatewayError) as e:
        window_for(staging, 1.0, connect=lambda url, **kw: Flaky(real_connect(url, **kw).__enter__())).run()
    assert TOKEN not in str(e.value) and "[token]" in str(e.value) and e.value.__context__ is None
    # the window object does not keep the token, and nothing is logged about the frames even with DEBUG logging on
    import io
    import logging
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    for name in ("websockets", "websockets.client", "cam_live.harness.discord_gateway.transport"):
        logging.getLogger(name).setLevel(logging.DEBUG)
    try:
        w = window_for(staging, 0.2)
        w.run()
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
    assert TOKEN not in json.dumps({k: repr(v) for k, v in vars(w).items()})
    assert TOKEN not in stream.getvalue() and "token" not in stream.getvalue() and '"op": 2' not in stream.getvalue()
    # through the executor: the whole serialized result, including failures
    ex, pol = executor_for(staging), ScopedPolicy(scopes=("discord",))
    gw.on_ready(lambda: st.add_human("x"))
    ok, _ = listen(ex, staging, pol, 1, max_events=3)
    assert ok.status == ResultStatus.OK and TOKEN not in json.dumps(ok.to_dict())
    assert TOKEN not in json.dumps([c.to_dict() for c in pol.checks])
    st.token = "rotated-" + "y" * 50                                    # the stored token is no longer valid: 401, then 4004
    bad, _ = listen(ex, staging, pol, 1)
    assert bad.status == ResultStatus.FAILED and bad.failure_class == FailureClass.MISSING_PERMISSION
    assert TOKEN not in json.dumps(bad.to_dict()) and st.token not in json.dumps(bad.to_dict())
    ex.close()


# ---- discord.listen_events through the executor ----------------------------------------------------------------------
def test_listen_events_through_the_executor(staging):
    st, gw = staging.state, staging.gateway
    human = st.add_human("writer")
    ex, pol = executor_for(staging, test_account_ids=st.test_account_ids), ScopedPolicy(scopes=("discord",))

    def happenings():
        made = st.add_human("joins during the window")
        st.post_member_message(st.general_id, human, "words that must stay out")
        st.remove_member(made)

    gw.on_ready(happenings)
    res, action = listen(ex, staging, pol, 1)
    assert res.status == ResultStatus.OK and res.verified is True
    obs = res.observations[0]
    w, q = obs.data["window"], {x.name: x.value for x in obs.quantities}
    assert obs.text.startswith("[staging] event window of 1.") and obs.data["staging"] is True and obs.provenance.source == "discord:staging"
    assert w["ended_by"] == "window" and 0.95 <= w["window_s"] < 4.0 and w["session_ok"]
    assert w["counts"] == {"READY": 1, "GUILD_CREATE": 1, "GUILD_MEMBER_ADD": 1, "MESSAGE_CREATE": 1, "GUILD_MEMBER_REMOVE": 1}
    assert q["discord.gateway.events"] == 5.0 and q["discord.gateway.guild_member_add"] == 1.0 and q["discord.gateway.dropped_other_guild"] == 0.0
    assert q["discord.gateway.session_starts_remaining"] == 999.0 and q["discord.gateway.window_s"] == w["window_s"]
    blob = json.dumps(res.to_dict())
    assert "must stay out" not in blob and "writer" not in blob and human not in blob and "not retained" in obs.text
    # one checked operation, deterministic, naming the intents; no message-content read without the permission
    assert [c.op for c in pol.checks] == ["discord.listen_events"]
    assert pol.checks[0].arguments == {"seconds": 1, "guild_id": st.guild_id, "reads": "gateway events", "aggregate_only": True,
                                       "intents": list(DEFAULT_INTENTS), "environment": "staging"}
    assert [r["path"].split("/api/v10")[1] for r in st.request_log] == ["/applications/@me", "/gateway/bot"]
    assert gw.identify_log[0]["intents"] == 66115 and "GUILD_PRESENCES" not in gw.identify_log[0]["intent_names"]
    # max_events bounds the window
    gw.on_ready(lambda: [st.add_human() for _ in range(10)])
    res, _ = listen(ex, staging, pol, 30, max_events=4)
    w = res.observations[0].data["window"]
    assert w["ended_by"] == "max_events" and w["truncated"] and len(w["events"]) == 4 and w["window_s"] < 15.0
    # a refused check: nothing is asked of the service, nothing connects
    n, m = len(st.request_log), len(gw.identify_log)
    deny = ScopedPolicy(scopes=("discord",), deny=lambda op, a: "listening is not permitted")
    res, _ = listen(ex, staging, deny, 1)
    assert res.status == ResultStatus.DENIED and len(st.request_log) == n and len(gw.identify_log) == m
    for bad in (0, 61, 1.5, "10"):
        r, _ = listen(ex, staging, pol, bad)
        assert r.failure_class == FailureClass.INVALID_ARGUMENTS, bad
    assert len(gw.identify_log) == m
    ex.close()


def test_listen_events_with_the_content_permission_is_its_own_checked_read(staging):
    st, gw = staging.state, staging.gateway
    human = st.add_human("writer")
    task = TaskSpec(objective_id="o", objective="x", resources={"discord": {"guild_id": st.guild_id}},
                    data_permissions={CONTENT_PERMISSION: True})
    keep = DiscordExecutor.from_task(task, base_url=staging.base_url, secrets={"discord_bot_token": st.token})
    pol = ScopedPolicy(scopes=("discord",))
    gw.on_ready(lambda: st.post_member_message(st.general_id, human, "text under permission"))
    res, _ = listen(keep, staging, pol, 1, max_events=3)
    assert res.status == ResultStatus.OK and [c.op for c in pol.checks] == ["discord.listen_events", "discord.read_messages"]
    assert pol.checks[0].arguments["aggregate_only"] is False and pol.checks[0].arguments["intents"][-1] == "MESSAGE_CONTENT"
    assert pol.checks[1].arguments["content_retained"] is True
    msg = [e for e in res.observations[0].data["window"]["events"] if e["event"] == "MESSAGE_CREATE"][0]
    assert msg["content"] == "text under permission" and human not in json.dumps(res.to_dict())
    assert gw.identify_log[-1]["intents"] == 66115 | (1 << 15) and "message text retained" in res.observations[0].text
    # a policy that refuses message reading stops the window before it connects
    n = len(gw.identify_log)
    strict = ScopedPolicy(scopes=("discord",), deny=lambda op, a: "platform: message content may not be read" if op.op == "discord.read_messages" else "")
    res, _ = listen(keep, staging, strict, 1)
    assert res.status == ResultStatus.DENIED and res.failure_class == FailureClass.PLATFORM_RESTRICTION and len(gw.identify_log) == n
    # the application lacks the message-content intent: refused before any Identify
    st.application["flags"] = 1 << 15
    keep._app_state = None
    res, _ = listen(keep, staging, pol, 1)
    assert res.failure_class == FailureClass.MISSING_PERMISSION and len(gw.identify_log) == n
    assert res.failure_detail == ("the application lacks the MESSAGE_CONTENT privileged gateway intent; the application "
                                  "owner must enable it on the Bot page of the developer portal")
    keep.close()


def test_listen_events_hard_failures_limits_and_destinations(staging):
    st, gw = staging.state, staging.gateway
    pol = ScopedPolicy(scopes=("discord",))
    ex = executor_for(staging)
    # the application lacks the members intent: refused without a doomed Identify, and reported as a missing capability
    st.application["flags"] = 0
    res, _ = listen(ex, staging, pol, 1)
    assert res.status == ResultStatus.FAILED and res.failure_class == FailureClass.MISSING_PERMISSION and res.retries == 0
    assert "lacks the GUILD_MEMBERS privileged gateway intent" in res.failure_detail and gw.identify_log == []
    assert "discord.listen_events" in {c["capability"] for c in ex.missing_capabilities()}
    # the intent is withdrawn after the adapter last looked: the Gateway closes with 4014 -> hard failure with its meaning
    st.application["flags"] = (1 << 15) | (1 << 19)
    ex._app_state = None
    assert listen(ex, staging, pol, 1, max_events=2)[0].status == ResultStatus.OK
    st.application["flags"] = 0
    res, _ = listen(ex, staging, pol, 1)
    assert res.status == ResultStatus.FAILED and res.failure_class == FailureClass.MISSING_PERMISSION and res.retries == 0
    assert "gateway close code 4014 (Disallowed intent(s))" in res.failure_detail and gw.identify_log[-1]["outcome"] == "close:4014"
    assert "discord.listen_events" in {c["capability"] for c in ex.missing_capabilities()}
    st.application["flags"] = (1 << 15) | (1 << 19)
    ex._app_state = None
    # Invalid Session in answer to Identify: a failure, not an empty success; not repeated
    gw.identify_response = "invalid_session"
    n = len(gw.identify_log)
    res, _ = listen(ex, staging, pol, 1)
    assert res.status == ResultStatus.FAILED and res.failure_class == FailureClass.RESOURCE_LIMIT and res.retries == 0
    assert "Invalid Session" in res.failure_detail and len(gw.identify_log) == n + 1
    # few session starts left today: no Identify is spent
    st.session_starts_remaining = 50
    n = len(gw.identify_log)
    res, _ = listen(ex, staging, pol, 1)
    assert res.failure_class == FailureClass.RESOURCE_LIMIT and "gateway session starts remain" in res.failure_detail
    assert len(gw.identify_log) == n
    st.session_starts_remaining = 1000
    # a gateway address that is not where the token may go: refused, nothing connects
    real_url, gw.url = gw.url, "ws://203.0.113.7:443"
    res, _ = listen(ex, staging, pol, 1)
    assert res.status == ResultStatus.DENIED and "not a Discord gateway address" in res.failure_detail and len(gw.identify_log) == n
    gw.url = real_url
    live = DiscordExecutor(guild_id=st.guild_id)
    assert live.gateway_url_allowed("wss://gateway.discord.gg") and live.gateway_url_allowed("wss://gateway-us-east1-b.discord.gg/")
    for url in ("ws://gateway.discord.gg", "wss://gateway.discord.gg.example.org", "wss://example.org", real_url, "wss://discord.gg.evil.test"):
        assert not live.gateway_url_allowed(url), url
    assert ex.gateway_url_allowed(real_url) and not ex.gateway_url_allowed("wss://gateway.discord.gg")
    # the secret must be bound to the gateway destination as well as to the REST one
    rest_only = GrantBoundBroker({"discord_bot_token": st.token}, {"discord_bot_token": staging.base_url})
    bound = DiscordExecutor(staging.base_url, guild_id=st.guild_id, credential_probe=lambda name: True)
    action = act("discord.listen_events", {"seconds": 1, "guild_id": st.guild_id}, scope="discord")
    ctx = make_ctx(pol, secrets=rest_only)
    ctx.authorization = rest_only.grant(action)
    res = bound.execute(action, ctx)
    assert res.status == ResultStatus.DENIED and res.failure_class == FailureClass.MISSING_PERMISSION and len(gw.identify_log) == n
    assert res.failure_detail == ("blocked: the bot token (secret 'discord_bot_token') was not released for the gateway "
                                  "destination 127.0.0.1; the operator must bind the secret to it")
    assert rest_only.calls[-1]["destination"] == real_url and st.token not in json.dumps(res.to_dict())
    caps = [c for c in bound.missing_capabilities() if c["capability"] == "discord.listen_events"]
    assert caps and "gateway" in caps[0]["how_to_grant"]
    both = GrantBoundBroker({"discord_bot_token": st.token}, {"discord_bot_token": (staging.base_url, real_url)})
    action = act("discord.listen_events", {"seconds": 1, "max_events": 2, "guild_id": st.guild_id}, scope="discord")
    ctx = make_ctx(pol, secrets=both)
    ctx.authorization = both.grant(action)
    res = bound.execute(action, ctx)
    assert res.status == ResultStatus.OK and [c["destination"] for c in both.calls] == [staging.base_url, real_url]
    assert bound.missing_capabilities() == []
    for x in (ex, live, bound):
        x.close()


def test_a_cancelled_action_ends_the_window(staging):
    st, gw = staging.state, staging.gateway
    cancel = threading.Event()
    gw.on_ready(cancel.set)
    out = window_for(staging, 30.0, cancel=cancel).run()
    assert out["ended_by"] == "cancelled" and out["window_s"] < 10.0 and out["session_ok"]
    assert isinstance(staging.gateway, StagingGateway) and staging.gateway.environment == "staging"
