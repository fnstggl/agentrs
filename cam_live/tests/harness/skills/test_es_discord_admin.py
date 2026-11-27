"""Discord administration tools, the operator's permission scope and the permission pre-check, against the local
STAGING service (spec 12.5; requirement IH-05).

Nothing here talks to Discord. Every result is a staging result and is labelled as such by the adapter."""
import json
import os
import stat

import pytest

from cam_live.contracts import FailureClass, ResultStatus, SimClock, TaskSpec
from cam_live.harness import discord as discord_mod
from cam_live.harness.discord import AUDIT_REASON, TOOLS, DiscordClient, DiscordExecutor
from cam_live.harness.discord_perms import (ALL_PERMISSIONS, DEFAULT_SCOPE, NEVER_GRANTED, PERMISSION_BITS,
                                            base_permissions, bits_of, channel_permissions, granted_intents,
                                            member_id_hash, names_of, normalise_scope, timed_out_permissions)
from cam_live.harness.staging_discord import StagingDiscord, StagingDiscordServer
from es_support import ScopedPolicy, act, make_ctx

NEW_WRITES = ("modify_guild", "modify_channel", "delete_channel", "set_channel_permissions", "create_role",
              "modify_role", "delete_role", "delete_message", "pin_message", "unpin_message", "delete_invite")
NEW_READS = ("get_identity", "get_permissions", "get_guild_settings", "list_roles", "get_member_events",
             "listen_events", "get_audit_summary")
MISSING = "the bot lacks {} in this server; the server owner must grant it"


@pytest.fixture
def staging():
    server = StagingDiscordServer(StagingDiscord(rate_limit=(1000, 1.0)))
    yield server
    server.close()


def executor_for(staging, **kw):
    kw.setdefault("secrets", {"discord_bot_token": staging.state.token})
    kw.setdefault("guild_id", staging.state.guild_id)
    return DiscordExecutor(staging.base_url, **kw)


def run(ex, staging, pol, name, args=None, clock=None, **kw):
    action = act(f"discord.{name}", dict(args or {}, guild_id=staging.state.guild_id), scope="discord", **kw)
    return ex.execute(action, make_ctx(pol, clock=clock))


def sent(st, since=0, methods=("POST", "PUT", "PATCH", "DELETE")):
    """Requests in the staging request log that would change something."""
    return [(r["method"], r["path"]) for r in st.request_log[since:] if r["method"] in methods]


def count(st, suffix, since=0):
    return sum(1 for r in st.request_log[since:] if r["path"].endswith(suffix))


# ---- permission table and algorithm ----------------------------------------------------------------------------------
def test_permission_bits_match_the_documented_values():
    assert bits_of(DEFAULT_SCOPE) == 2269727275740401 and len(DEFAULT_SCOPE) == 18      # the operator's permission integer
    assert bits_of(["SEND_MESSAGES", "ADD_REACTIONS"]) == 0x800 | 0x40 == 2112            # the example of the documentation
    assert names_of("2112") == ["ADD_REACTIONS", "SEND_MESSAGES"] and names_of(0) == []
    assert (PERMISSION_BITS["ADMINISTRATOR"], PERMISSION_BITS["MANAGE_ROLES"], PERMISSION_BITS["PIN_MESSAGES"],
            PERMISSION_BITS["CREATE_EVENTS"], PERMISSION_BITS["MODERATE_MEMBERS"]) == (3, 28, 51, 44, 40)
    assert len(PERMISSION_BITS) == 52 and len(set(PERMISSION_BITS.values())) == 52 and 47 not in PERMISSION_BITS.values()
    assert names_of(ALL_PERMISSIONS) == sorted(PERMISSION_BITS)
    with pytest.raises(ValueError, match="unknown Discord permission"):
        bits_of(["SEND_MESSAGE"])
    with pytest.raises(ValueError, match="unknown Discord permission"):
        normalise_scope(["VIEW_CHANNEL", "FLY"])
    with pytest.raises(ValueError):
        DiscordExecutor("http://127.0.0.1:9/api/v10", guild_id="12345", scope=["FLY"])
    assert normalise_scope(None) == DEFAULT_SCOPE and normalise_scope(["VIEW_CHANNEL", "VIEW_CHANNEL"]) == ("VIEW_CHANNEL",)
    # application flags 565248 = bits 13, 15, 19: the three "limited" privileged intents
    assert granted_intents(565248) == {"GUILD_PRESENCES": True, "GUILD_MEMBERS": True, "MESSAGE_CONTENT": True}
    assert granted_intents(1 << 15) == {"GUILD_PRESENCES": False, "GUILD_MEMBERS": True, "MESSAGE_CONTENT": False}
    assert granted_intents(1 << 18)["MESSAGE_CONTENT"] is True and granted_intents(0)["GUILD_MEMBERS"] is False


def test_permission_algorithm_roles_everyone_administrator_overwrites():
    g, me = "100", "555"
    b = lambda *names: bits_of(names)                                                # noqa: E731
    roles = [{"id": g, "permissions": str(b("VIEW_CHANNEL", "READ_MESSAGE_HISTORY"))},
             {"id": "201", "permissions": str(b("SEND_MESSAGES", "EMBED_LINKS"))},
             {"id": "202", "permissions": str(b("MANAGE_CHANNELS"))},
             {"id": "203", "permissions": str(b("ADMINISTRATOR"))}]
    # @everyone | the member's roles; a role the member does not have adds nothing
    base = base_permissions(g, roles, ["201"])
    assert names_of(base) == ["EMBED_LINKS", "READ_MESSAGE_HISTORY", "SEND_MESSAGES", "VIEW_CHANNEL"]
    assert names_of(base_permissions(g, roles, [])) == ["READ_MESSAGE_HISTORY", "VIEW_CHANNEL"]
    assert names_of(base_permissions(g, roles, ["201", "202", "999"])) == sorted(
        ["EMBED_LINKS", "READ_MESSAGE_HISTORY", "SEND_MESSAGES", "VIEW_CHANNEL", "MANAGE_CHANNELS"])
    # ADMINISTRATOR (or owning the guild) is every permission
    assert base_permissions(g, roles, ["203"]) == ALL_PERMISSIONS == base_permissions(g, roles, [], is_owner=True)
    ow = lambda i, t, allow=(), deny=(): {"id": i, "type": t, "allow": str(b(*allow)), "deny": str(b(*deny))}   # noqa: E731
    # no overwrites: unchanged
    assert channel_permissions(base, [], g, ["201"], me) == base
    # the @everyone overwrite: deny first, then allow
    p = channel_permissions(base, [ow(g, 0, allow=["ADD_REACTIONS"], deny=["SEND_MESSAGES", "ADD_REACTIONS"])], g, ["201"], me)
    assert names_of(p) == ["ADD_REACTIONS", "EMBED_LINKS", "READ_MESSAGE_HISTORY", "VIEW_CHANNEL"]
    # role overwrites: all denies of the member's roles, then all allows (an allow of one role beats a deny of another)
    p = channel_permissions(base_permissions(g, roles, ["201", "202"]),
                            [ow(g, 0, deny=["VIEW_CHANNEL"]), ow("201", 0, deny=["SEND_MESSAGES", "EMBED_LINKS"]),
                             ow("202", 0, allow=["SEND_MESSAGES", "VIEW_CHANNEL"]), ow("777", 0, deny=["MANAGE_CHANNELS"])],
                            g, ["201", "202"], me)
    assert names_of(p) == ["MANAGE_CHANNELS", "READ_MESSAGE_HISTORY", "SEND_MESSAGES", "VIEW_CHANNEL"]
    # the member's own overwrite is applied last
    p = channel_permissions(base, [ow("201", 0, allow=["ATTACH_FILES"]), ow(me, 1, allow=["PIN_MESSAGES"], deny=["ATTACH_FILES"])],
                            g, ["201"], me)
    assert "PIN_MESSAGES" in names_of(p) and "ATTACH_FILES" not in names_of(p)
    # an overwrite of another member, or a member overwrite that carries a role's id, is not applied
    assert channel_permissions(base, [ow("556", 1, deny=["VIEW_CHANNEL"])], g, ["201"], me) == base
    # ADMINISTRATOR bypasses overwrites
    assert channel_permissions(ALL_PERMISSIONS, [ow(g, 0, deny=["VIEW_CHANNEL", "SEND_MESSAGES"])], g, ["203"], me) == ALL_PERMISSIONS
    # a timed-out member keeps only VIEW_CHANNEL and READ_MESSAGE_HISTORY; administrators are exempt
    assert names_of(timed_out_permissions(base)) == ["READ_MESSAGE_HISTORY", "VIEW_CHANNEL"]
    assert timed_out_permissions(ALL_PERMISSIONS) == ALL_PERMISSIONS


def test_get_permissions_agrees_with_the_service_for_roles_and_overwrites(staging):
    st = staging.state
    ex, pol = executor_for(staging), ScopedPolicy(scopes=("discord",))
    res = run(ex, staging, pol, "get_permissions")
    d = res.observations[0].data
    assert res.status == ResultStatus.OK and d["permissions"] == sorted(DEFAULT_SCOPE) == names_of(st.bot_permissions())
    assert d["missing_from_scope"] == [] and d["broader_than_scope"] == [] and d["administrator"] is False
    assert d["staging"] is True and res.observations[0].text.startswith("[staging]")
    # an extra role and a narrower bot role
    st.set_bot_permissions(["VIEW_CHANNEL", "SEND_MESSAGES", "MANAGE_CHANNELS"])
    extra = st.add_role("helpers", ["CONNECT", "SPEAK"])
    st.assign_role(st.bot_user["id"], extra)
    d = run(ex, staging, pol, "get_permissions").observations[0].data
    assert d["permissions"] == names_of(st.bot_permissions()) == sorted(
        ["VIEW_CHANNEL", "READ_MESSAGE_HISTORY", "SEND_MESSAGES", "MANAGE_CHANNELS", "CONNECT", "SPEAK"])
    assert d["broader_than_scope"] == ["CONNECT", "SPEAK"] and "MANAGE_ROLES" in d["missing_from_scope"]
    q = {x.name: x.value for x in run(ex, staging, pol, "get_permissions").observations[0].quantities}
    assert q["discord.permissions.broader_than_scope"] == 2.0 and q["discord.permissions.administrator"] == 0.0
    # channel overwrites: @everyone denied, a role of the bot allowed again, the bot itself denied
    ch = st.add_channel("rules")
    st.set_overwrite(ch, st.guild_id, deny=["SEND_MESSAGES", "VIEW_CHANNEL"])
    st.set_overwrite(ch, extra, allow=["VIEW_CHANNEL", "ADD_REACTIONS"])
    st.set_overwrite(ch, st.bot_user["id"], deny=["MANAGE_CHANNELS"], member=True)
    d = run(ex, staging, pol, "get_permissions", {"channel_id": ch}).observations[0].data
    assert d["permissions"] == names_of(st.bot_permissions(ch)) == sorted(
        ["VIEW_CHANNEL", "READ_MESSAGE_HISTORY", "ADD_REACTIONS", "CONNECT", "SPEAK"]) and d["channel_id"] == ch
    # ADMINISTRATOR: everything, overwrites notwithstanding, and all of it is reported as beyond the scope
    st.make_bot_administrator()
    d = run(ex, staging, pol, "get_permissions", {"channel_id": ch}).observations[0].data
    assert d["administrator"] is True and d["permissions"] == sorted(PERMISSION_BITS) == names_of(st.bot_permissions(ch))
    assert d["broader_than_scope"] == sorted(set(PERMISSION_BITS) - set(DEFAULT_SCOPE)) and d["missing_from_scope"] == []
    # only the bot's own member object was ever read
    members = [r["path"] for r in st.request_log if "/members/" in r["path"]]
    assert members and all(p.endswith(f"/members/{st.bot_user['id']}") for p in members)
    ex.close()


# ---- the operator's scope --------------------------------------------------------------------------------------------
def test_scope_refusal_even_when_the_bot_is_administrator(staging):
    st = staging.state
    st.make_bot_administrator()
    scope = [n for n in DEFAULT_SCOPE if n not in ("MANAGE_ROLES", "MANAGE_GUILD", "MANAGE_MESSAGES")]
    task = TaskSpec(objective_id="o", objective="x", resources={"discord": {"guild_id": st.guild_id, "scope": scope}})
    ex = DiscordExecutor.from_task(task, base_url=staging.base_url, secrets={"discord_bot_token": st.token})
    assert ex.scope == tuple(scope)
    tools = {t.name: t for t in ex.tools()}
    blocked = {"discord.create_role": "MANAGE_ROLES", "discord.modify_role": "MANAGE_ROLES",
               "discord.delete_role": "MANAGE_ROLES", "discord.set_channel_permissions": "MANAGE_ROLES",
               "discord.modify_guild": "MANAGE_GUILD", "discord.list_invites": "MANAGE_GUILD"}
    for name, t in tools.items():
        if name in blocked:
            assert not t.available and t.unavailable_reason == f"outside the operator's permission scope: needs {blocked[name]}"
        else:
            assert t.available and t.unavailable_reason == "", name
    assert tools["discord.create_role"].auth_scope == "MANAGE_ROLES" and tools["discord.get_identity"].auth_scope == "discord.read"
    assert tools["discord.send_message"].auth_scope == "VIEW_CHANNEL+SEND_MESSAGES"
    assert tools["discord.set_channel_permissions"].auth_scope == "VIEW_CHANNEL+MANAGE_ROLES"
    pol = ScopedPolicy(scopes=("discord",))
    for name, args in (("create_role", {"name": "r"}), ("modify_guild", {"name": "new name"}), ("list_invites", {}),
                       ("delete_role", {"role_id": st.bot_role_id}),
                       ("set_channel_permissions", {"channel_id": st.general_id, "role_id": "everyone", "deny": ["SEND_MESSAGES"]})):
        res = run(ex, staging, pol, name, args)
        assert res.status == ResultStatus.DENIED and res.failure_class == FailureClass.MISSING_PERMISSION, name
        assert res.failure_detail.startswith("outside the operator's permission scope: needs ") and res.ops == [] and res.retries == 0
    assert st.request_log == [] and pol.checks == []                 # refused before any request and before any check
    # a permission that only the arguments bring in is judged the same way: an event in a voice channel needs CONNECT
    voice = st.add_channel("stage", 2)
    res = run(ex, staging, pol, "create_event", {"name": "e", "start_time": "2027-01-10T18:00:00+00:00", "channel_id": voice})
    assert res.status == ResultStatus.DENIED and res.failure_detail == "outside the operator's permission scope: needs CONNECT"
    # deleting another author's message needs MANAGE_MESSAGES, which this scope leaves out: refused, message kept
    other = st.post_member_message(st.general_id, st.add_human("h"), "words")
    res = run(ex, staging, pol, "delete_message", {"channel_id": st.general_id, "message_id": other})
    assert res.status == ResultStatus.DENIED and "needs MANAGE_MESSAGES" in res.failure_detail
    assert [m["id"] for m in st.messages[st.general_id]] == [other] and sent(st) == []
    # what the role allows beyond the scope is reported, not used
    d = run(ex, staging, pol, "get_permissions").observations[0].data
    assert d["administrator"] is True and {"MANAGE_ROLES", "MANAGE_GUILD", "BAN_MEMBERS"} <= set(d["broader_than_scope"])
    caps = {c["capability"]: c for c in ex.missing_capabilities()}
    assert set(caps) == set(blocked) and "resources.discord.scope" in caps["discord.create_role"]["how_to_grant"]
    assert caps["discord.modify_guild"]["reason"] == "outside the operator's permission scope: needs MANAGE_GUILD"
    ex.close()


# ---- permission pre-check --------------------------------------------------------------------------------------------
def test_precheck_refuses_without_the_doomed_request(staging):
    st = staging.state
    clock = SimClock()
    ex, pol = executor_for(staging), ScopedPolicy(scopes=("discord",))
    st.revoke_bot("MANAGE_CHANNELS")
    res = run(ex, staging, pol, "create_channel", {"name": "notes"}, clock)
    assert res.status == ResultStatus.FAILED and res.failure_class == FailureClass.MISSING_PERMISSION and res.retries == 0
    assert res.failure_detail == MISSING.format("MANAGE_CHANNELS")
    assert sent(st) == [] and len(st.channels) == 1                              # the doomed POST was never made
    assert [r["path"].rsplit("/", 1)[-1] for r in st.request_log] == ["@me", "roles", st.bot_user["id"]]
    assert pol.checks[0].op == "discord.create_channel" and res.ops[0].status == ResultStatus.FAILED
    # the bot's permissions are read once and kept: further refusals cost no request at all
    n = len(st.request_log)
    for _ in range(3):
        assert run(ex, staging, pol, "create_channel", {"name": "notes"}, clock).failure_detail == MISSING.format("MANAGE_CHANNELS")
    assert len(st.request_log) == n
    # ... for 300 seconds; then they are read again and a permission granted in the meantime is seen
    st.grant_bot("MANAGE_CHANNELS")
    clock.advance(299.0)
    assert run(ex, staging, pol, "create_channel", {"name": "notes"}, clock).status == ResultStatus.FAILED
    clock.advance(2.0)
    made = run(ex, staging, pol, "create_channel", {"name": "notes"}, clock)
    assert made.status == ResultStatus.OK and count(st, "/roles") == 2 and count(st, "/users/@me") == 1
    # channel overwrites count: the bot's role is denied SEND_MESSAGES in one channel only
    quiet, open_ = made.artifacts["channel_id"], st.general_id
    st.set_overwrite(quiet, st.bot_role_id, deny=["SEND_MESSAGES"])
    clock.advance(301.0)
    n = len(st.request_log)
    res = run(ex, staging, pol, "send_message", {"channel_id": quiet, "content": "x"}, clock)
    assert res.failure_class == FailureClass.MISSING_PERMISSION and res.failure_detail == MISSING.format("SEND_MESSAGES")
    assert sent(st, n) == [] and st.messages[quiet] == []
    assert run(ex, staging, pol, "send_message", {"channel_id": open_, "content": "x"}, clock).status == ResultStatus.OK
    # several missing permissions are named together, in the order the route needs them
    st.set_bot_permissions([])
    st.set_everyone_permissions([])
    clock.advance(301.0)
    res = run(ex, staging, pol, "get_channel_activity", {"channel_id": open_}, clock)
    assert res.failure_detail == MISSING.format("VIEW_CHANNEL, READ_MESSAGE_HISTORY")
    # a tool that needs no permission is not held up by any of this
    assert run(ex, staging, pol, "list_roles", {}, clock).status == ResultStatus.OK
    ex.close()


def test_a_403_that_arrives_anyway_drops_the_cached_permissions(staging):
    st = staging.state
    ex, pol = executor_for(staging), ScopedPolicy(scopes=("discord",))
    assert run(ex, staging, pol, "create_channel", {"name": "a"}).status == ResultStatus.OK and count(st, "/roles") == 1
    assert run(ex, staging, pol, "create_channel", {"name": "b"}).status == ResultStatus.OK and count(st, "/roles") == 1
    # the service refuses although the roles (as the adapter last read them) allow it: sent once, 403, hard failure
    st.set_permission("MANAGE_CHANNELS", False)
    n = len(st.request_log)
    res = run(ex, staging, pol, "create_channel", {"name": "c"})
    assert res.failure_class == FailureClass.MISSING_PERMISSION and res.retries == 0 and "50013" in res.failure_detail
    assert sent(st, n) == [("POST", f"/api/v10/guilds/{st.guild_id}/channels")]
    # the cache was dropped: the next action reads the roles again and sees the revocation without another POST
    st.set_permission("MANAGE_CHANNELS", True)
    st.revoke_bot("MANAGE_CHANNELS")
    n = len(st.request_log)
    res = run(ex, staging, pol, "create_channel", {"name": "d"})
    assert res.failure_detail == MISSING.format("MANAGE_CHANNELS") and count(st, "/roles") == 2 and sent(st, n) == []
    ex.close()


# ---- guild settings --------------------------------------------------------------------------------------------------
def test_rename_guild_and_restore(staging):
    st = staging.state
    st.add_human("resident")
    ex, pol = executor_for(staging), ScopedPolicy(scopes=("discord",))
    before = run(ex, staging, pol, "get_guild_settings").observations[0]
    s = before.data["settings"]
    assert s["name"] == "staging guild" and s["verification_level"] == 1 and s["preferred_locale"] == "en-US"
    assert set(s) == {"id", "name", "description", "features", "verification_level", "default_message_notifications",
                      "explicit_content_filter", "system_channel_id", "rules_channel_id", "public_updates_channel_id",
                      "preferred_locale", "approximate_member_count", "approximate_presence_count"}
    blob = json.dumps(before.to_dict())
    assert st.guild["owner_id"] not in blob and "owner" not in blob and "staging-app" not in blob and "resident" not in blob
    assert {q.name: q.value for q in before.quantities}["discord.members.approximate"] == 2.0
    renamed = run(ex, staging, pol, "modify_guild", {"name": s["name"] + " (integration test)"})
    assert renamed.status == ResultStatus.OK and st.guild["name"] == "staging guild (integration test)"
    assert renamed.observations[0].data["changed"] == ["name"] and st.guild["owner_id"] not in json.dumps(renamed.to_dict())
    patch = st.request_log[-1]
    assert (patch["method"], patch["path"], patch["audit_reason"]) == ("PATCH", f"/api/v10/guilds/{st.guild_id}", AUDIT_REASON)
    assert pol.checks[-1].op == "discord.modify_guild" and pol.checks[-1].arguments["name"] == "staging guild (integration test)"
    restored = run(ex, staging, pol, "modify_guild", {"name": s["name"]})
    assert restored.status == ResultStatus.OK and st.guild["name"] == "staging guild"
    assert run(ex, staging, pol, "get_guild_settings").observations[0].data["settings"] == s
    # other settings, a channel of the server as system channel, and what is refused
    res = run(ex, staging, pol, "modify_guild", {"description": "about tides", "verification_level": 2,
                                                 "default_message_notifications": 1, "explicit_content_filter": 2,
                                                 "system_channel_id": st.general_id, "preferred_locale": "en-GB"})
    assert res.status == ResultStatus.OK and st.guild["system_channel_id"] == st.general_id and st.guild["verification_level"] == 2
    assert st.guild["description"] == "about tides" and st.guild["preferred_locale"] == "en-GB"
    n = len(st.request_log)
    for bad in ({}, {"name": "x"}, {"name": "  x  "}, {"verification_level": 9}, {"icon": "data"}, {"features": ["COMMUNITY"]},
                {"owner_id": "123456"}):
        assert run(ex, staging, pol, "modify_guild", bad).failure_class == FailureClass.INVALID_ARGUMENTS, bad
    dm = st.add_dm_channel()
    res = run(ex, staging, pol, "modify_guild", {"system_channel_id": dm})
    assert res.status == ResultStatus.DENIED and "not a channel of the authorized server" in res.failure_detail
    assert sent(st, n) == [] and st.guild["system_channel_id"] == st.general_id
    ex.close()


# ---- channels --------------------------------------------------------------------------------------------------------
def test_channel_create_modify_permissions_delete(staging):
    st = staging.state
    ex, pol = executor_for(staging), ScopedPolicy(scopes=("discord",))
    made = run(ex, staging, pol, "create_channel", {"name": "cam-integration-test"})
    cid = made.artifacts["channel_id"]
    assert st.request_log[-1]["audit_reason"] == AUDIT_REASON
    cat = run(ex, staging, pol, "create_channel", {"name": "archive", "type": "category"}).artifacts["channel_id"]
    res = run(ex, staging, pol, "modify_channel", {"channel_id": cid, "name": "renamed", "topic": "what it is for",
                                                   "position": 5, "parent_id": cat, "nsfw": False, "rate_limit_per_user": 30})
    assert res.status == ResultStatus.OK
    ch = st.channels[cid]
    assert (ch["name"], ch["topic"], ch["position"], ch["parent_id"], ch["rate_limit_per_user"]) == ("renamed", "what it is for", 5, cat, 30)
    assert res.observations[0].data["changed"] == ["name", "topic", "position", "parent_id", "nsfw", "rate_limit_per_user"]
    assert st.request_log[-1]["method"] == "PATCH" and st.request_log[-1]["audit_reason"] == AUDIT_REASON
    assert run(ex, staging, pol, "modify_channel", {"channel_id": cid, "topic": None, "parent_id": None}).status == ResultStatus.OK
    assert st.channels[cid]["topic"] is None and st.channels[cid]["parent_id"] is None
    # permission overwrites: for a role (here everyone), by name, as overwrite type 0
    res = run(ex, staging, pol, "set_channel_permissions", {"channel_id": cid, "role_id": "everyone", "deny": ["SEND_MESSAGES"]})
    assert res.status == ResultStatus.OK
    assert st.channels[cid]["permission_overwrites"] == [{"id": st.guild_id, "type": 0, "allow": "0",
                                                          "deny": str(bits_of(["SEND_MESSAGES"]))}]
    put = st.request_log[-1]
    assert (put["method"], put["path"], put["audit_reason"]) == ("PUT", f"/api/v10/channels/{cid}/permissions/{st.guild_id}", AUDIT_REASON)
    helpers = st.add_role("helpers")
    res = run(ex, staging, pol, "set_channel_permissions", {"channel_id": cid, "role_id": helpers,
                                                            "allow": ["SEND_MESSAGES", "ATTACH_FILES"], "deny": ["EMBED_LINKS"]})
    assert res.status == ResultStatus.OK and res.observations[0].data["overwrite"]["everyone"] is False
    stored = {o["id"]: o for o in st.channels[cid]["permission_overwrites"]}[helpers]
    assert stored == {"id": helpers, "type": 0, "allow": str(bits_of(["SEND_MESSAGES", "ATTACH_FILES"])), "deny": str(bits_of(["EMBED_LINKS"]))}
    # what is never written: a member's overwrite, a dangerous allow, a name outside the scope, allow and deny at once
    n = len(st.request_log)
    human = st.add_human("someone")
    res = run(ex, staging, pol, "set_channel_permissions", {"channel_id": cid, "role_id": human, "deny": ["SEND_MESSAGES"]})
    assert res.status == ResultStatus.FAILED and res.failure_detail == "the role is not a role of the authorized server; nothing was changed"
    for never in NEVER_GRANTED:
        res = run(ex, staging, pol, "set_channel_permissions", {"channel_id": cid, "role_id": "everyone", "allow": [never]})
        assert res.failure_class == FailureClass.INVALID_ARGUMENTS and never in res.failure_detail, never
    for bad in ({"allow": ["CONNECT"]}, {"deny": ["SPEAK"]}, {"allow": ["SEND_MESSAGES"], "deny": ["SEND_MESSAGES"]},
                {"allow": ["NOT_A_PERMISSION"]}, {"allow": "SEND_MESSAGES"}, {"user_id": human}, {"member_id": human}):
        res = run(ex, staging, pol, "set_channel_permissions", dict({"channel_id": cid, "role_id": "everyone"}, **bad))
        assert res.failure_class == FailureClass.INVALID_ARGUMENTS, bad
    assert sent(st, n) == [] and all(o["type"] == 0 for o in st.channels[cid]["permission_overwrites"])
    # a permission the bot does not hold itself cannot be handed out: refused before the request
    st.revoke_bot("ATTACH_FILES")
    ex._perm_state = None
    res = run(ex, staging, pol, "set_channel_permissions", {"channel_id": cid, "role_id": helpers, "deny": ["ATTACH_FILES"]})
    assert res.failure_detail == MISSING.format("ATTACH_FILES") and sent(st, n) == []
    # deleting: only a channel of the server; the adapter forgets it afterwards
    dm = st.add_dm_channel()
    for target in (dm, "999999999999"):
        res = run(ex, staging, pol, "delete_channel", {"channel_id": target})
        assert res.status == ResultStatus.DENIED and "not a channel of the authorized server" in res.failure_detail
    assert sent(st, n) == [] and dm in st.channels
    res = run(ex, staging, pol, "delete_channel", {"channel_id": cid})
    assert res.status == ResultStatus.OK and cid not in st.channels and res.observations[0].data["deleted"] is True
    assert (st.request_log[-1]["method"], st.request_log[-1]["audit_reason"]) == ("DELETE", AUDIT_REASON)
    assert run(ex, staging, pol, "modify_channel", {"channel_id": cid, "name": "again"}).status == ResultStatus.DENIED
    ex.close()


# ---- roles -----------------------------------------------------------------------------------------------------------
def test_roles_create_modify_delete_and_the_refusal_list(staging):
    st = staging.state
    ex, pol = executor_for(staging), ScopedPolicy(scopes=("discord",))
    listed = run(ex, staging, pol, "list_roles").observations[0].data["roles"]
    assert [(r["name"], r["everyone"], r["managed"]) for r in listed] == [("@everyone", True, False), ("staging-app", False, True)]
    assert listed[1]["permissions"] == sorted(DEFAULT_SCOPE) and "permission_bits" not in listed[0]
    made = run(ex, staging, pol, "create_role", {"name": "cam-integration-test-role"})
    rid = made.artifacts["role_id"]
    assert made.status == ResultStatus.OK and st.roles[rid]["permissions"] == "0" and st.roles[rid]["name"] == "cam-integration-test-role"
    assert "not given to anyone" in made.observations[0].text and st.request_log[-1]["audit_reason"] == AUDIT_REASON
    assert all(rid not in m["roles"] for m in st.members.values())
    res = run(ex, staging, pol, "modify_role", {"role_id": rid, "name": "writers", "permissions": ["SEND_MESSAGES", "ATTACH_FILES"],
                                                "color": 0x3366FF, "hoist": True, "mentionable": True})
    role = st.roles[rid]
    assert res.status == ResultStatus.OK and role["name"] == "writers" and names_of(role["permissions"]) == ["ATTACH_FILES", "SEND_MESSAGES"]
    assert role["colors"]["primary_color"] == 0x3366FF and role["hoist"] and role["mentionable"]
    assert res.observations[0].data["role"]["permissions"] == ["ATTACH_FILES", "SEND_MESSAGES"]
    second = run(ex, staging, pol, "create_role", {"name": "readers", "permissions": ["VIEW_CHANNEL", "READ_MESSAGE_HISTORY"]}).artifacts["role_id"]
    assert names_of(st.roles[second]["permissions"]) == ["READ_MESSAGE_HISTORY", "VIEW_CHANNEL"]
    # the refusal list: never ADMINISTRATOR or anything that acts on members, on the server or on everyone
    n = len(st.request_log)
    for never in NEVER_GRANTED:
        for name, args in (("create_role", {"name": "x", "permissions": [never]}),
                           ("modify_role", {"role_id": rid, "permissions": ["SEND_MESSAGES", never]})):
            res = run(ex, staging, pol, name, args)
            assert res.failure_class == FailureClass.INVALID_ARGUMENTS and never in res.failure_detail, (name, never)
    assert set(NEVER_GRANTED) == {"ADMINISTRATOR", "KICK_MEMBERS", "BAN_MEMBERS", "MODERATE_MEMBERS", "MANAGE_GUILD",
                                  "MANAGE_ROLES", "MANAGE_WEBHOOKS", "MENTION_EVERYONE"}
    assert run(ex, staging, pol, "create_role", {"name": "x", "permissions": ["CONNECT"]}).failure_class == FailureClass.INVALID_ARGUMENTS
    assert run(ex, staging, pol, "modify_role", {"role_id": rid}).failure_class == FailureClass.INVALID_ARGUMENTS
    # never the @everyone role, never a managed role, never a role at or above the bot's own, never a foreign id
    for name in ("modify_role", "delete_role"):
        extra = {"name": "y"} if name == "modify_role" else {}
        assert run(ex, staging, pol, name, dict({"role_id": st.guild_id}, **extra)).failure_class == FailureClass.INVALID_ARGUMENTS
        res = run(ex, staging, pol, name, dict({"role_id": st.bot_role_id}, **extra))
        assert res.status == ResultStatus.DENIED and "managed by an integration are never modified or deleted" in res.failure_detail
        res = run(ex, staging, pol, name, dict({"role_id": "424242424242"}, **extra))
        assert res.status == ResultStatus.FAILED and "not a role of the authorized server" in res.failure_detail
    above = st.add_role("moderators", ["VIEW_CHANNEL"])              # created by the owner above the bot's role
    res = run(ex, staging, pol, "delete_role", {"role_id": above})
    assert res.failure_class == FailureClass.MISSING_PERMISSION and "below its own highest role" in res.failure_detail
    assert sent(st, n) == [] and {rid, second, above, st.bot_role_id, st.guild_id} == set(st.roles)
    # a permission the bot lacks cannot be put into a role
    st.revoke_bot("ATTACH_FILES")
    ex._perm_state = None
    res = run(ex, staging, pol, "create_role", {"name": "z", "permissions": ["ATTACH_FILES"]})
    assert res.failure_detail == MISSING.format("ATTACH_FILES") and sent(st, n) == []
    for target in (rid, second):
        assert run(ex, staging, pol, "delete_role", {"role_id": target}).status == ResultStatus.OK
    assert rid not in st.roles and second not in st.roles and st.request_log[-1]["audit_reason"] == AUDIT_REASON
    # no request ever addressed a member other than the bot itself, and none gave a role to anyone
    assert all(p.endswith(f"/members/{st.bot_user['id']}") for p in (r["path"] for r in st.request_log) if "/members" in p)
    assert all(r["method"] == "GET" for r in st.request_log if "/members" in r["path"])
    ex.close()


# ---- messages --------------------------------------------------------------------------------------------------------
def test_delete_pin_and_unpin_messages(staging):
    st = staging.state
    ex, pol = executor_for(staging), ScopedPolicy(scopes=("discord",))
    cid = st.general_id
    own = run(ex, staging, pol, "send_message", {"channel_id": cid, "content": "the bot's words"}).artifacts["message_id"]
    human = st.add_human("writer")
    theirs = st.post_member_message(cid, human, "a member's private-ish words")
    pinned = run(ex, staging, pol, "pin_message", {"channel_id": cid, "message_id": own})
    assert pinned.status == ResultStatus.OK and st.messages[cid][0]["pinned"] is True
    put = st.request_log[-1]
    assert (put["method"], put["path"], put["audit_reason"]) == ("PUT", f"/api/v10/channels/{cid}/messages/pins/{own}", AUDIT_REASON)
    unpinned = run(ex, staging, pol, "unpin_message", {"channel_id": cid, "message_id": own})
    assert unpinned.status == ResultStatus.OK and st.messages[cid][0]["pinned"] is False
    assert (st.request_log[-1]["method"], st.request_log[-1]["path"]) == ("DELETE", f"/api/v10/channels/{cid}/messages/pins/{own}")
    # deleting the bot's own message: one check, recorded as its own
    n_checks = len(pol.checks)
    res = run(ex, staging, pol, "delete_message", {"channel_id": cid, "message_id": own})
    d = res.observations[0].data
    assert res.status == ResultStatus.OK and d["own"] is True and d["deleted"] is True
    assert [c.op for c in pol.checks[n_checks:]] == ["discord.delete_message"]
    assert st.request_log[-1]["method"] == "DELETE" and st.request_log[-1]["audit_reason"] is None   # not a route with a reason
    # deleting another author's message is a different act: it is put to the policy again, saying so
    n_checks = len(pol.checks)
    res = run(ex, staging, pol, "delete_message", {"channel_id": cid, "message_id": theirs})
    d = res.observations[0].data
    assert res.status == ResultStatus.OK and d["own"] is False and d["author_is_bot"] is False and st.messages[cid] == []
    assert [c.op for c in pol.checks[n_checks:]] == ["discord.delete_message", "discord.delete_message"]
    assert pol.checks[-1].arguments["own"] is False and "own" not in pol.checks[-2].arguments
    blob = json.dumps(res.to_dict())
    assert "private-ish" not in blob and "writer" not in blob and human not in blob      # neither text nor author leave
    # a policy that refuses that second check keeps the message
    kept = st.post_member_message(cid, human, "stays")
    strict = ScopedPolicy(scopes=("discord",), deny=lambda op, a: "moderation is not permitted" if op.arguments.get("own") is False else "")
    n = len(st.request_log)
    res = run(ex, staging, strict, "delete_message", {"channel_id": cid, "message_id": kept})
    assert res.status == ResultStatus.DENIED and [m["id"] for m in st.messages[cid]] == [kept] and sent(st, n) == []
    # without MANAGE_MESSAGES the bot can still delete what it posted itself, and nothing else
    st.revoke_bot("MANAGE_MESSAGES")
    ex._perm_state = None
    res = run(ex, staging, pol, "delete_message", {"channel_id": cid, "message_id": kept})
    assert res.failure_detail == MISSING.format("MANAGE_MESSAGES") and sent(st, n) == []
    again = run(ex, staging, pol, "send_message", {"channel_id": cid, "content": "x"}).artifacts["message_id"]
    assert run(ex, staging, pol, "delete_message", {"channel_id": cid, "message_id": again}).status == ResultStatus.OK
    # only messages of the server: a direct-message channel is never touched
    dm = st.add_dm_channel()
    for name in ("delete_message", "pin_message", "unpin_message"):
        res = run(ex, staging, pol, name, {"channel_id": dm, "message_id": st.messages[dm][0]["id"]})
        assert res.status == ResultStatus.DENIED and "not a channel of the authorized server" in res.failure_detail
    assert not any(dm in r["path"] for r in st.request_log) and len(st.messages[dm]) == 1
    assert run(ex, staging, pol, "pin_message", {"channel_id": cid, "message_id": "123456789"}).failure_class == FailureClass.WRONG_SITE_ASSUMPTION
    ex.close()


# ---- invites ---------------------------------------------------------------------------------------------------------
def test_delete_invite_only_of_the_authorized_guild(staging):
    st = staging.state
    ex, pol = executor_for(staging), ScopedPolicy(scopes=("discord",))
    code = run(ex, staging, pol, "create_invite", {"channel_id": st.general_id, "max_age_s": 600, "max_uses": 1}).artifacts["invite_code"]
    foreign = st.add_foreign_invite()
    n = len(st.request_log)
    res = run(ex, staging, pol, "delete_invite", {"code": foreign})
    assert res.status == ResultStatus.DENIED and res.failure_class == FailureClass.MISSING_PERMISSION
    assert res.failure_detail == "the invite does not belong to the authorized server; nothing was deleted"
    assert foreign in st.invites and sent(st, n) == []
    assert [(r["method"], r["path"]) for r in st.request_log[n:]][0] == ("GET", f"/api/v10/invites/{foreign}")
    assert "someone" not in json.dumps(res.to_dict())                              # the inviter of the foreign invite
    res = run(ex, staging, pol, "delete_invite", {"code": "nosuchcode"})
    assert res.failure_class == FailureClass.WRONG_SITE_ASSUMPTION and "10006" in res.failure_detail
    assert run(ex, staging, pol, "delete_invite", {"code": "../guilds"}).failure_class == FailureClass.INVALID_ARGUMENTS
    res = run(ex, staging, pol, "delete_invite", {"code": code})
    assert res.status == ResultStatus.OK and code not in st.invites and res.artifacts["invite_code"] == code
    assert (st.request_log[-1]["method"], st.request_log[-1]["path"], st.request_log[-1]["audit_reason"]) == (
        "DELETE", f"/api/v10/invites/{code}", AUDIT_REASON)
    # MANAGE_CHANNELS in the invite's channel, or MANAGE_GUILD in the server (the documented alternative)
    code = run(ex, staging, pol, "create_invite", {"channel_id": st.general_id, "unique": True}).artifacts["invite_code"]
    st.revoke_bot("MANAGE_CHANNELS")
    ex._perm_state = None
    assert run(ex, staging, pol, "delete_invite", {"code": code}).status == ResultStatus.OK
    code = run(ex, staging, pol, "create_invite", {"channel_id": st.general_id, "unique": True}).artifacts["invite_code"]
    st.revoke_bot("MANAGE_GUILD")
    ex._perm_state = None
    n = len(st.request_log)
    res = run(ex, staging, pol, "delete_invite", {"code": code})
    assert res.failure_detail == MISSING.format("MANAGE_CHANNELS") and code in st.invites and sent(st, n) == []
    ex.close()


# ---- membership changes ----------------------------------------------------------------------------------------------
def test_member_event_diff_baseline_bots_and_declared_test_accounts(staging, tmp_path):
    st = staging.state
    state_dir = str(tmp_path / "state")
    early = [st.add_human(f"early{i}") for i in range(3)]
    tester = st.add_test_account("operator test")
    clock = SimClock()
    pol = ScopedPolicy(scopes=("discord",))
    ex = executor_for(staging, test_account_ids=st.test_account_ids, state_dir=state_dir, id_salt="pepper")
    invite = run(ex, staging, pol, "create_invite", {"channel_id": st.general_id, "unique": True}, clock).artifacts["invite_code"]
    first = run(ex, staging, pol, "get_member_events", {}, clock)
    d, q = first.observations[0].data, {x.name: x.value for x in first.observations[0].quantities}
    assert first.status == ResultStatus.OK and d["baseline"] is True and d["persisted"] is True
    assert "baseline established" in first.observations[0].text and "joined" not in d and "discord.members.joined" not in q
    assert d["member_counts"]["humans"] == 3 and q["discord.members.humans"] == 3.0 and q["discord.members.test_accounts"] == 1.0
    assert pol.checks[-1].op == "discord.get_member_events" and pol.checks[-1].arguments["aggregate_only"] is True
    # between the readings: two people join through the invite, one joins otherwise, a bot and a test account are
    # added, one early member and the first test account leave
    clock.advance(600.0)
    joiners = [st.use_invite(invite, "newcomer a"), st.use_invite(invite, "newcomer b"), st.add_human("walk-in")]
    newbot, tester2 = st.add_bot("music bot"), st.add_test_account("operator test 2")
    st.remove_member(early[0])
    st.remove_member(tester)
    ex2 = executor_for(staging, test_account_ids=st.test_account_ids, state_dir=state_dir, id_salt="pepper")   # a new process
    second = run(ex2, staging, pol, "get_member_events", {}, clock)
    d = second.observations[0].data
    assert d["baseline"] is False and d["since_s"] == 600.0
    assert d["joined"] == {"people": 3, "bots": 1, "test_accounts": 1} and d["left"] == {"people": 1, "bots": 0, "test_accounts": 1}
    assert d["joined_at"] == sorted(st.members[j]["joined_at"] for j in joiners) and len(d["joined_at"]) == 3
    assert d["member_counts"] == {"total": 8, "bots": 2, "test_accounts": 1, "humans": 5, "pending": 0,
                                  "humans_not_pending": 5, "distinct": 5}
    assert d["invites"] == [{"code": invite, "uses": 2, "delta": 2, "new": False}] and d["invites_gone"] == 0
    qs = {x.name: x for x in second.observations[0].quantities}
    assert (qs["discord.members.joined"].value, qs["discord.members.left"].value) == (3.0, 1.0)
    assert (qs["discord.members.joined_bots"].value, qs["discord.members.left_test_accounts"].value) == (1.0, 1.0)
    assert qs["discord.members.joined"].kind == "count" and qs["discord.members.joined"].exposure == 600.0
    assert qs["discord.invites.uses_delta"].value == 2.0 and qs["discord.members.humans"].value == 5.0
    # aggregates only: no id, no name, no pseudonym in what is returned
    blob = json.dumps(second.to_dict())
    everyone = early + joiners + [tester, tester2, newbot]
    assert not any(uid in blob for uid in everyone) and "newcomer" not in blob and "walk-in" not in blob and "music bot" not in blob
    assert member_id_hash(st.guild_id, joiners[0], "pepper") not in blob
    # the snapshot: one private file, pseudonyms only
    files = os.listdir(state_dir)
    assert len(files) == 1 and files[0].startswith("member_snapshot_") and st.guild_id not in files[0]
    path = os.path.join(state_dir, files[0])
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600 and stat.S_IMODE(os.stat(state_dir).st_mode) == 0o700
    raw = open(path).read()
    snap = json.loads(raw)
    assert not any(uid in raw for uid in everyone + [st.guild_id, st.bot_user["id"]]) and "newcomer" not in raw and "staging-app" not in raw
    assert all(set(m) == {"id_hash", "bot", "joined_at", "pending", "test_account"} for m in snap["members"]) and len(snap["members"]) == 8
    assert member_id_hash(st.guild_id, joiners[0], "pepper") in {m["id_hash"] for m in snap["members"]}
    assert snap["invites"] == {invite: 2} and snap["version"] == 1
    # nothing changed: zeros, not a new baseline
    clock.advance(60.0)
    third = run(ex2, staging, pol, "get_member_events", {}, clock).observations[0].data
    assert third["baseline"] is False and third["joined"] == third["left"] == {"people": 0, "bots": 0, "test_accounts": 0}
    assert third["invites"][0]["delta"] == 0
    # an unreadable snapshot is not guessed at: the reading is a baseline again and says why
    open(path, "w").write("{not json")
    again = run(ex2, staging, pol, "get_member_events", {}, clock).observations[0].data
    assert again["baseline"] is True and again["previous_snapshot_unreadable"] is True
    # without MANAGE_GUILD the membership part still works and the invite part says what is missing
    st.revoke_bot("MANAGE_GUILD")
    ex2._perm_state = None
    st.add_human("late")
    last = run(ex2, staging, pol, "get_member_events", {}, clock).observations[0].data
    assert last["joined"]["people"] == 1 and "invites" not in last and "MANAGE_GUILD" in last["invites_unavailable"]
    # without the members intent the list is refused by the platform: a hard failure, and the snapshot is kept
    st.members_intent = False
    before = open(path).read()
    res = run(ex2, staging, pol, "get_member_events", {}, clock)
    assert res.failure_class == FailureClass.MISSING_PERMISSION and open(path).read() == before
    assert {"discord.get_member_events", "discord.listen_events"} <= {c["capability"] for c in ex2.missing_capabilities()}
    for x in (ex, ex2):
        x.close()


def test_member_events_without_a_state_directory_use_the_run_directory_or_memory(staging, tmp_path):
    st = staging.state
    pol = ScopedPolicy(scopes=("discord",))
    ex = executor_for(staging)
    a = act("discord.get_member_events", {"guild_id": st.guild_id}, scope="discord")
    first = ex.execute(a, make_ctx(pol)).observations[0].data
    assert first["baseline"] is True and first["persisted"] is False
    st.add_human("x")
    assert ex.execute(act("discord.get_member_events", {"guild_id": st.guild_id}, scope="discord"),
                      make_ctx(pol)).observations[0].data["joined"]["people"] == 1
    run_dir = str(tmp_path / "run")
    other = executor_for(staging)
    res = other.execute(act("discord.get_member_events", {"guild_id": st.guild_id}, scope="discord"), make_ctx(pol, run_dir=run_dir))
    assert res.observations[0].data["persisted"] is True and len(os.listdir(os.path.join(run_dir, "discord_state"))) == 1
    for x in (ex, other):
        x.close()


# ---- identity, audit, capabilities -----------------------------------------------------------------------------------
def test_identity_other_guilds_and_missing_capabilities(staging):
    st = staging.state
    ex, pol = executor_for(staging), ScopedPolicy(scopes=("discord",))
    assert ex.missing_capabilities() == []                                     # nothing known to be missing; nothing was asked
    assert st.request_log == []
    res = run(ex, staging, pol, "get_identity")
    d, q = res.observations[0].data, {x.name: x.value for x in res.observations[0].quantities}
    assert res.status == ResultStatus.OK and d["bot"] == {"id": st.bot_user["id"], "name": "staging-app", "bot": True}
    assert d["application"] == {"name": "staging app", "bot_public": False,
                                "intents": {"GUILD_PRESENCES": True, "GUILD_MEMBERS": True, "MESSAGE_CONTENT": True}}
    assert d["in_guild"] is True and d["other_guilds"] == 0 and d["warnings"] == []
    assert q["discord.identity.in_guild"] == 1.0 and q["discord.identity.other_guilds"] == 0.0
    assert sorted(r["path"].split("/api/v10")[1] for r in st.request_log) == ["/applications/@me", "/users/@me", "/users/@me/guilds"]
    # the bot is in two other servers: the count is reported as a warning quantity, nothing else about them
    a, b = st.add_other_guild("secret project"), st.add_other_guild("friends")
    st.application["bot_public"] = True
    res = run(ex, staging, pol, "get_identity")
    d = res.observations[0].data
    assert d["other_guilds"] == 2 and len(d["warnings"]) == 2 and "2 other server(s)" in d["warnings"][0]
    assert {x.name: x.value for x in res.observations[0].quantities}["discord.identity.other_guilds"] == 2.0
    blob = json.dumps(res.to_dict())
    assert a not in blob and b not in blob and "secret project" not in blob and "friends" not in blob
    # missing intents
    st.application["flags"] = 0
    d = run(ex, staging, pol, "get_identity").observations[0].data
    assert d["application"]["intents"] == {"GUILD_PRESENCES": False, "GUILD_MEMBERS": False, "MESSAGE_CONTENT": False}
    caps = ex.missing_capabilities()
    assert {c["capability"] for c in caps} == {"discord.listen_events", "discord.get_member_events", "discord.get_guild_stats"}
    assert all("GUILD_MEMBERS" in c["reason"] and "Developer Portal" in c["how_to_grant"] for c in caps)
    assert all(set(c) == {"capability", "reason", "how_to_grant"} for c in caps)
    st.application["flags"] = (1 << 15) | (1 << 19)
    run(ex, staging, pol, "get_identity")
    # a Discord permission the bot lacks, learned from an executed action
    st.revoke_bot("MANAGE_ROLES", "PIN_MESSAGES")
    run(ex, staging, pol, "get_permissions")
    caps = {c["capability"]: c for c in ex.missing_capabilities()}
    assert set(caps) == {"discord.create_role", "discord.modify_role", "discord.delete_role",
                         "discord.set_channel_permissions", "discord.pin_message", "discord.unpin_message"}
    assert caps["discord.pin_message"]["reason"] == MISSING.format("PIN_MESSAGES") and "Roles" in caps["discord.pin_message"]["how_to_grant"]
    # the bot was not added to the server (what the operator's bot looks like before the owner adds it)
    st.remove_bot_from_guild()
    res = run(ex, staging, pol, "get_identity")
    assert res.status == ResultStatus.OK and res.observations[0].data["in_guild"] is False
    assert ex.missing_capabilities() == [{"capability": "discord", "reason": "the bot is not a member of the authorized server",
                                          "how_to_grant": discord_mod._HOW_TO["guild"]}]
    n = len(st.request_log)
    fresh = executor_for(staging)
    res = run(fresh, staging, pol, "create_channel", {"name": "x"})
    assert res.failure_class == FailureClass.MISSING_PERMISSION and sent(st, n) == []
    assert res.failure_detail == "the bot is not a member of the authorized server; the server owner must add it"
    assert fresh.missing_capabilities()[0]["capability"] == "discord"
    res = run(fresh, staging, pol, "get_guild_settings")                       # a route that needs no permission: 404 Unknown Guild
    assert res.failure_class == FailureClass.MISSING_PERMISSION and res.retries == 0 and "10004" in res.failure_detail
    assert res.failure_detail.startswith("the bot is not a member of the authorized server; the server owner must add it")
    # no token, no guild: one entry, and nothing is contacted to say so
    n = len(st.request_log)
    no_token = DiscordExecutor(staging.base_url, guild_id=st.guild_id, secrets={})
    no_guild = DiscordExecutor(staging.base_url, secrets={"discord_bot_token": st.token})
    assert no_token.missing_capabilities()[0]["reason"] == "blocked: no Discord bot token provided (secret 'discord_bot_token')"
    assert "credentials" in no_token.missing_capabilities()[0]["how_to_grant"]
    assert no_guild.missing_capabilities()[0]["reason"] == "blocked: no owner-authorized Discord guild is configured"
    assert len(st.request_log) == n
    for x in (ex, fresh, no_token, no_guild):
        x.close()


def test_audit_summary_has_counts_and_times_only(staging):
    st = staging.state
    ex, pol = executor_for(staging), ScopedPolicy(scopes=("discord",))
    cid = run(ex, staging, pol, "create_channel", {"name": "one"}).artifacts["channel_id"]
    run(ex, staging, pol, "create_channel", {"name": "two"})
    run(ex, staging, pol, "modify_channel", {"channel_id": cid, "topic": "t"})
    run(ex, staging, pol, "create_role", {"name": "r"})
    st.audit_log.append({"id": st._snowflake(), "user_id": st.guild["owner_id"], "target_id": st.add_human("victim"),
                         "action_type": 20, "reason": "a private reason", "changes": []})
    st.audit_log.append({"id": st._snowflake(), "user_id": None, "target_id": None, "action_type": 9999})
    res = run(ex, staging, pol, "get_audit_summary", {"limit": 50})
    a = res.observations[0].data["audit"]
    assert res.status == ResultStatus.OK and a["entries"] == 6
    assert a["by_action"] == {"CHANNEL_CREATE": 2, "CHANNEL_UPDATE": 1, "MEMBER_KICK": 1, "ROLE_CREATE": 1, "UNKNOWN_9999": 1}
    assert set(a) == {"entries", "by_action", "newest", "oldest"} and a["newest"] >= a["oldest"] and a["newest"].startswith("20")
    q = {x.name: x.value for x in res.observations[0].quantities}
    assert q["discord.audit.entries"] == 6.0 and q["discord.audit.channel_create"] == 2.0 and q["discord.audit.member_kick"] == 1.0
    blob = json.dumps(res.to_dict())
    for leaked in (st.guild["owner_id"], st.bot_user["id"], "victim", "a private reason", AUDIT_REASON, "staging-app", cid):
        assert leaked not in blob.replace(st.guild_id, ""), leaked            # no actor, no target, no reason
    assert pol.checks[-1].arguments["aggregate_only"] is True
    assert run(ex, staging, pol, "get_audit_summary", {"limit": 2}).observations[0].data["audit"]["entries"] == 2
    assert discord_mod.snowflake_iso("175928847299117063") == "2016-04-30T11:18:25.796000+00:00"   # example of the API reference
    st.revoke_bot("VIEW_AUDIT_LOG")
    ex._perm_state = None
    n = len(st.request_log)
    res = run(ex, staging, pol, "get_audit_summary")
    assert res.failure_detail == MISSING.format("VIEW_AUDIT_LOG") and count(st, "/audit-logs", n) == 0
    ex.close()


# ---- every write passes the authorization check; nothing targets a member ----------------------------------------------
def _write_sequence(staging, pol):
    """One pass over every new write tool, on a fresh staging state. Returns the results by tool name."""
    st = staging.state
    ex = executor_for(staging)
    out = {}
    cid = run(ex, staging, pol, "create_channel", {"name": "cam-integration-test"}).artifacts.get("channel_id", st.general_id)
    mid = run(ex, staging, pol, "send_message", {"channel_id": cid, "content": "text"}).artifacts.get("message_id", "123456789")
    code = run(ex, staging, pol, "create_invite", {"channel_id": cid, "max_age_s": 600, "max_uses": 1}).artifacts.get("invite_code", "abcdefgh")
    out["modify_guild"] = run(ex, staging, pol, "modify_guild", {"name": "staging guild (integration test)"})
    out["modify_channel"] = run(ex, staging, pol, "modify_channel", {"channel_id": cid, "topic": "topic"})
    out["set_channel_permissions"] = run(ex, staging, pol, "set_channel_permissions",
                                         {"channel_id": cid, "role_id": "everyone", "deny": ["SEND_MESSAGES"]})
    out["create_role"] = run(ex, staging, pol, "create_role", {"name": "cam-integration-test-role"})
    rid = out["create_role"].artifacts.get("role_id", "123456789")
    out["modify_role"] = run(ex, staging, pol, "modify_role", {"role_id": rid, "mentionable": True})
    out["delete_role"] = run(ex, staging, pol, "delete_role", {"role_id": rid})
    out["pin_message"] = run(ex, staging, pol, "pin_message", {"channel_id": cid, "message_id": mid})
    out["unpin_message"] = run(ex, staging, pol, "unpin_message", {"channel_id": cid, "message_id": mid})
    out["delete_message"] = run(ex, staging, pol, "delete_message", {"channel_id": cid, "message_id": mid})
    out["delete_invite"] = run(ex, staging, pol, "delete_invite", {"code": code})
    out["delete_channel"] = run(ex, staging, pol, "delete_channel", {"channel_id": cid})
    ex.close()
    return out


def test_every_new_write_tool_passes_through_the_check_with_deterministic_arguments():
    runs = []
    for _ in range(2):
        with StagingDiscordServer(StagingDiscord(rate_limit=(1000, 1.0))) as staging:
            pol = ScopedPolicy(scopes=("discord",))
            out = _write_sequence(staging, pol)
            assert set(out) == set(NEW_WRITES) and all(r.status == ResultStatus.OK for r in out.values()), {
                k: (r.status, r.failure_detail) for k, r in out.items() if r.status != ResultStatus.OK}
            # every write the service received carried the fixed audit reason where the route takes one
            st = staging.state
            reasons = {(r["method"], r["path"].split("/")[-2] if r["path"].split("/")[-1].isdigit() else r["path"].split("/")[-1]): r["audit_reason"]
                       for r in st.request_log if r["method"] != "GET"}
            no_reason = {k for k, v in reasons.items() if v is None}
            assert no_reason == {("POST", "messages"), ("DELETE", "messages")}, no_reason
            assert set(reasons.values()) == {None, AUDIT_REASON}
            runs.append([(c.op, json.dumps(c.arguments, sort_keys=True)) for c in pol.checks])
            ops = [c.op for c in pol.checks]
            assert {f"discord.{n}" for n in NEW_WRITES} <= set(ops)
            assert all(c.arguments["guild_id"] == st.guild_id and c.arguments["environment"] == "staging" for c in pol.checks)
    assert runs[0] == runs[1]                              # the same actions produce the same checked operations
    # a policy that refuses a write tool's operation stops it before anything is changed
    for name in NEW_WRITES:
        with StagingDiscordServer(StagingDiscord(rate_limit=(1000, 1.0))) as staging:
            op_name = f"discord.{name}"
            pol = ScopedPolicy(scopes=("discord",), deny=lambda op, a, op_name=op_name: "refused by the test policy" if op.op == op_name else "")
            st = staging.state
            out = _write_sequence(staging, pol)
            res = out[name]
            assert res.status == ResultStatus.DENIED and res.failure_class == FailureClass.MISSING_PERMISSION, name
            assert res.failure_detail == "refused by the test policy" and res.retries == 0
            first = [i for i, c in enumerate(pol.checks) if c.op == op_name][0]
            started = pol.checks[first].started
            method = {"modify_guild": "PATCH", "modify_channel": "PATCH", "modify_role": "PATCH", "create_role": "POST",
                      "set_channel_permissions": "PUT", "pin_message": "PUT"}.get(name, "DELETE")
            marker = {"modify_guild": f"/guilds/{st.guild_id}", "create_role": "/roles", "modify_role": "/roles/",
                      "delete_role": "/roles/", "set_channel_permissions": "/permissions/", "pin_message": "/pins/",
                      "unpin_message": "/pins/", "delete_invite": "/invites/", "delete_message": "/messages/",
                      "modify_channel": "/channels/", "delete_channel": "/channels/"}[name]
            hits = [r for r in st.request_log if r["method"] == method and marker in r["path"]
                    and (name != "modify_guild" or r["path"].endswith(marker))
                    and (name not in ("modify_channel", "delete_channel") or r["path"].count("/") == 4)
                    and (name != "delete_message" or "/pins/" not in r["path"])]
            assert hits == [], (name, hits, started)


def test_no_tool_names_or_takes_a_member_as_target(staging):
    st = staging.state
    ex = executor_for(staging)
    tools = {t.name: t for t in ex.tools()}
    assert set(tools) == set(TOOLS) and len(tools) == 28
    assert {f"discord.{n}" for n in NEW_WRITES + NEW_READS} <= set(tools)
    for name in NEW_WRITES:
        assert tools[f"discord.{name}"].side_effects == "external_write" and tools[f"discord.{name}"].available, name
    for name in NEW_READS:
        assert tools[f"discord.{name}"].side_effects == "external_read", name
    # no tool acts on a member: none is named for it, and no schema has a field that could carry a member
    verbs = ("kick", "ban", "timeout", "time_out", "mute", "nick", "rename_member", "assign", "grant_role", "add_role",
             "remove_member", "prune", "dm", "direct", "user", "friend", "relationship", "whisper")
    assert not [n for n in tools for v in verbs if v in n.split(".", 1)[1]]
    assert [n for n in tools if "member" in n] == ["discord.get_member_events"]          # an aggregate reading
    fields = {p for t in tools.values() for p in t.arg_schema["properties"]}
    # ("rate_limit_per_user" is Discord's name for a channel's slow mode: a number of seconds, not a person)
    assert not [f for f in fields - {"rate_limit_per_user"}
                if any(w in f for w in ("user", "member", "recipient", "target", "nick", "author"))], fields
    assert tools["discord.modify_channel"].arg_schema["properties"]["rate_limit_per_user"]["type"] == "integer"
    assert fields == {"guild_id", "channel_id", "message_id", "role_id", "code", "name", "type", "topic", "parent_id",
                      "content", "limit", "max_age_s", "max_uses", "unique", "start_time", "end_time", "description",
                      "location", "verification_level", "default_message_notifications", "explicit_content_filter",
                      "system_channel_id", "rules_channel_id", "public_updates_channel_id", "preferred_locale",
                      "position", "nsfw", "rate_limit_per_user", "allow", "deny", "permissions", "color", "hoist",
                      "mentionable", "seconds", "max_events"}
    for t in tools.values():
        assert t.arg_schema["additionalProperties"] is False and t.arg_schema["properties"]["guild_id"]["const"] == st.guild_id
    # the client has no method that changes a member, and no route for it appears in the adapter's source
    assert not [m for m in dir(DiscordClient) if any(v in m for v in ("kick", "ban", "timeout", "nick", "add_member",
                                                                      "remove_member", "member_role", "prune", "dm"))]
    source = open(discord_mod.__file__).read().split('"""', 2)[2]
    for route in ("/bans", "/prune", "/members/@me", "users/@me/channels", "/recipients", "/bulk-ban", "/members/search",
                  "\"nick\"", "/thread-members", "/roles/{role_id}/members", "members/{user"):
        assert route not in source, route
    # the one member route in the source is a GET of the bot's own member object (its roles, for the permission check)
    assert source.count("/members/") == 2 and source.count('"GET", f"/guilds/{guild_id}/members/{bot_id}"') == 1
    assert source.count('"type": 0}') == 1 and '"type": 1' not in source          # overwrites are written for roles only
    # descriptions say what a tool does; there is no advice on when or why
    text = " ".join(t.description.lower() for t in tools.values())
    for word in ("should", "grow", "attract", "engage", "promot", "best", "strategy", "tip", "recommend", "useful", "consider"):
        assert word not in text, word
    ex.close()
