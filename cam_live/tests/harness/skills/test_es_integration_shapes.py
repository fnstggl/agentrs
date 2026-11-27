"""The shapes the other harness services use at the seams: a grant-bound secret broker, the environment's own
sub-step dispatch, and operation names the authorization policy can classify. (Local stand-ins only; the real
services belong to other workstreams.)"""
import json

from cam_live.contracts import (ActionResult, FailureClass, Level, Modality, ResultStatus, TaskSpec,
                                VerificationCondition, WallClock)
from cam_live.harness.discord import CONTENT_PERMISSION, DiscordExecutor
from cam_live.harness.http_api import HttpApiExecutor, grant_bound, resolve_secret
from cam_live.harness.skills import SkillExecutor, SkillLibrary, Trace, evaluate_condition
from cam_live.harness.staging_discord import StagingDiscord, StagingDiscordServer
from es_support import GrantBoundBroker, NotesApp, ScopedPolicy, act, make_ctx


def test_grant_bound_broker_releases_only_for_the_granted_action_and_destination(notes):
    http, pol = HttpApiExecutor(), ScopedPolicy()
    broker = GrantBoundBroker({"app_token": NotesApp.TOKEN}, {"app_token": notes.url})
    assert grant_bound(broker) and not grant_bound({"a": "b"}) and resolve_secret(broker, "app_token") is None
    action = act("http.get", {"url": notes.url + "/api/private", "auth": {"secret": "app_token"}})
    ctx = make_ctx(pol, secrets=broker)
    ctx.authorization = broker.grant(action)                       # what the environment places on the context
    res = http.execute(action, ctx)
    assert res.status == ResultStatus.OK and NotesApp.TOKEN not in json.dumps(res.to_dict())
    assert broker.calls[-1] == {"name": "app_token", "action_id": action.action_id, "destination": notes.url + "/api/private"}
    # an action that was not granted gets no secret and nothing is sent
    n = len(notes.requests)
    other = act("http.get", {"url": notes.url + "/api/private", "auth": {"secret": "app_token"}})
    res = http.execute(other, ctx)
    assert res.failure_class == FailureClass.MISSING_PERMISSION and len(notes.requests) == n
    http.close()


def test_skill_steps_use_the_environment_dispatch_and_the_skill_actions_grant(tmp_path, notes):
    http = HttpApiExecutor()
    broker = GrantBoundBroker({"app_token": NotesApp.TOKEN}, {"app_token": notes.url})
    pol = ScopedPolicy()
    lib = SkillLibrary(str(tmp_path), WallClock())
    ex = SkillExecutor(lib, [])                                    # no executors of its own: the environment dispatches
    ex.inducer.set_tools(http.tools())
    dispatched = []

    def ctx_for(action):
        ctx = make_ctx(pol, secrets=broker)
        ctx.authorization = broker.grant(action)
        ctx.dispatch = lambda step: (dispatched.append(step.operation), http.execute(step, ctx))[1]
        return ctx

    def episode(marker):
        pairs = []
        for path in ("/api/private", "/api/items"):
            a = act("http.get", {"url": notes.url + path, "query": {"m": marker}, "auth": {"secret": "app_token"}})
            pairs.append((a, http.execute(a, ctx_for(a))))
            assert pairs[-1][1].status == ResultStatus.OK
        return Trace.from_pairs(pairs)

    induce = act("skill.induce", {"traces": [["x"]]}, modality=Modality.SKILL)
    skill, report = ex.induce_validate_register([episode("one"), episode("two")], ctx_for(induce), action=induce)
    assert skill is not None, report
    assert skill.steps[0].arguments["auth"] == {"secret": "app_token"}            # a name; never a value
    assert NotesApp.TOKEN not in skill.to_json()
    run = act("skill.run", {"skill_id": skill.skill_id, "arguments": {"m": "three"}}, modality=Modality.SKILL)
    dispatched.clear()
    res = ex.execute(run, ctx_for(run))
    assert res.status == ResultStatus.OK and dispatched == ["http.get", "http.get"]
    # the broker was asked in the name of the granted skill.run action, once per step, for each step's own URL
    asked = broker.calls[-2:]
    assert [c["action_id"] for c in asked] == [run.action_id, run.action_id]
    assert [c["destination"].split("?")[0] for c in asked] == [notes.url + "/api/private", notes.url + "/api/items"]
    assert NotesApp.TOKEN not in json.dumps(res.to_dict())
    http.close()


def test_discord_tools_name_the_server_and_use_a_probe_for_availability():
    with StagingDiscordServer(StagingDiscord(rate_limit=(1000, 1.0))) as staging:
        st = staging.state
        broker = GrantBoundBroker({"discord_bot_token": st.token}, {"discord_bot_token": staging.base_url})
        # without a probe a grant-bound broker cannot say whether the token exists; with one it can, without releasing it
        blind = DiscordExecutor(staging.base_url, guild_id=st.guild_id, secrets=broker)
        assert not blind.tools()[0].available
        ex = DiscordExecutor(staging.base_url, guild_id=st.guild_id, secrets=broker,
                             credential_probe=lambda name: name in broker.values)
        tools = {t.name: t for t in ex.tools()}
        assert all(t.available for t in tools.values()) and broker.calls[-1]["action_id"] is None and len(broker.calls) == 1
        # every tool takes the server id, fixed to the authorized one, so each action states its destination
        for t in tools.values():
            assert t.arg_schema["properties"]["guild_id"] == {"type": "string", "const": st.guild_id}
            assert "guild_id" in t.arg_schema["required"]
        pol = ScopedPolicy(scopes=("discord.read", "discord.write"))
        ok = act("discord.send_message", {"guild_id": st.guild_id, "channel_id": st.general_id, "content": "hello"},
                 scope="discord.write")
        ctx = make_ctx(pol, secrets=broker)
        ctx.authorization = broker.grant(ok)
        assert ex.execute(ok, ctx).status == ResultStatus.OK and st.messages[st.general_id][0]["content"] == "hello"
        assert pol.checks[0].arguments["guild_id"] == st.guild_id and pol.checks[0].arguments["channel_id"] == st.general_id
        for bad in ({"guild_id": "9999999", "channel_id": st.general_id, "content": "x"},
                    {"channel_id": st.general_id, "content": "x"}):
            res = ex.execute(act("discord.send_message", bad, scope="discord.write"), ctx)
            assert res.failure_class == FailureClass.INVALID_ARGUMENTS
        assert len(st.messages[st.general_id]) == 1
        # an action whose grant the broker does not know gets no token: denied, nothing sent
        n = len(st.request_log)
        stray = act("discord.list_channels", {"guild_id": st.guild_id}, scope="discord.read")
        res = ex.execute(stray, make_ctx(pol, secrets=broker))
        assert res.status == ResultStatus.DENIED and res.failure_class == FailureClass.MISSING_PERMISSION
        assert len(st.request_log) == n
        for x in (blind, ex):
            x.close()


def test_retaining_message_text_is_put_to_the_policy_as_its_own_operation():
    with StagingDiscordServer(StagingDiscord(rate_limit=(1000, 1.0))) as staging:
        st = staging.state
        st.post_member_message(st.general_id, st.add_human("m"), "words of a member")
        secrets = {"discord_bot_token": st.token}
        args = {"guild_id": st.guild_id, "channel_id": st.general_id}
        task = TaskSpec(objective_id="o", objective="x", resources={"discord": {"guild_id": st.guild_id}},
                        data_permissions={CONTENT_PERMISSION: True})
        keep = DiscordExecutor.from_task(task, base_url=staging.base_url, secrets=secrets)
        drop = DiscordExecutor(staging.base_url, guild_id=st.guild_id, secrets=secrets)
        pol = ScopedPolicy(scopes=("discord.read",))
        drop.execute(act("discord.get_channel_activity", args, scope="discord.read"), make_ctx(pol))
        assert [c.op for c in pol.checks] == ["discord.get_channel_activity", "discord.list_channels"]
        assert pol.checks[0].arguments["aggregate_only"] is True
        pol2 = ScopedPolicy(scopes=("discord.read",))
        res = keep.execute(act("discord.get_channel_activity", args, scope="discord.read"), make_ctx(pol2))
        assert [c.op for c in pol2.checks][:2] == ["discord.get_channel_activity", "discord.read_messages"]
        assert pol2.checks[0].arguments["aggregate_only"] is False and "words of a member" in res.observations[0].text
        # a policy that refuses message reading stops it even though the task's data permission is set
        strict = ScopedPolicy(scopes=("discord.read",),
                              deny=lambda op, a: "platform: message content may not be read" if op.op == "discord.read_messages" else "")
        n = len(st.request_log)
        res = keep.execute(act("discord.get_channel_activity", args, scope="discord.read"), make_ctx(strict))
        assert res.status == ResultStatus.DENIED and res.failure_class == FailureClass.PLATFORM_RESTRICTION
        assert len(st.request_log) == n and "words of a member" not in json.dumps(res.to_dict())
        for x in (keep, drop):
            x.close()


def test_publishing_to_a_channel_under_a_grant_bound_broker(tmp_path):
    from cam_live.harness.publishing import DISCORD_CHANNEL, Destination, PublishingExecutor
    with StagingDiscordServer(StagingDiscord(rate_limit=(1000, 1.0))) as staging:
        st = staging.state
        broker = GrantBoundBroker({"discord_bot_token": st.token}, {"discord_bot_token": staging.base_url})
        discord = DiscordExecutor(staging.base_url, guild_id=st.guild_id, credential_probe=lambda n: True)
        pub = PublishingExecutor(str(tmp_path), [Destination("news", DISCORD_CHANNEL, channel_id=st.general_id)], discord=discord)
        pol = ScopedPolicy(scopes=("publish.draft", "publish.write", "discord.write", "discord.read"))
        plain = make_ctx(pol, secrets=broker)
        d = pub.execute(act("publish.draft", {"destination": "news", "content": "words"}, scope="publish.draft"), plain)
        draft_id = d.artifacts["draft_id"]
        addr = pub.execute(act("publish.preview", {"draft_id": draft_id}, scope="publish.draft"), plain).observations[0].data["where"]["address"]
        submit = act("publish.submit", {"draft_id": draft_id, "destination": addr}, scope="publish.write")
        # the broker does not know this action's grant: no token, nothing sent - and that is a fault, not a rejection
        res = pub.execute(submit, plain)
        assert res.status == ResultStatus.DENIED and st.messages[st.general_id] == [] and pub.drafts()[0]["state"] == "previewed"
        # with the grant on the context the token is released in the name of the publish.submit action
        ctx = make_ctx(pol, secrets=broker)
        ctx.authorization = broker.grant(submit)
        res = pub.execute(submit, ctx)
        assert res.status == ResultStatus.OK and [m["content"] for m in st.messages[st.general_id]] == ["words"]
        assert broker.calls[-1]["action_id"] == submit.action_id and pub.drafts()[0]["state"] == "published"
        assert getattr(ctx, "granted_action", None) is None                  # restored afterwards
        # an update may state the draft's destination; another one is refused
        ok = pub.execute(act("publish.update", {"draft_id": draft_id, "content": "w2", "destination": addr}, scope="publish.draft"), ctx)
        bad = pub.execute(act("publish.update", {"draft_id": draft_id, "content": "w3", "destination": "channel:discord:1"},
                              scope="publish.draft"), ctx)
        assert ok.status == ResultStatus.OK and bad.failure_class == FailureClass.INVALID_ARGUMENTS
        assert pub.drafts()[0]["revision"] == 2
        discord.close()


def test_condition_operators_accept_both_spellings():
    res = ActionResult(action_id="a", status=ResultStatus.OK)
    from cam_live.contracts import Observation, Quantity
    obs = [Observation(obs_id="o", ts=0.0, kind="metric", quantities=[Quantity("site.views", 12.0)], data={"n": 3})]
    for op, value, want in ((">=", 12, True), ("ge", 13, False), ("==", 12, True), ("eq", 12, True), ("<", 12, False),
                            ("!=", 5, True), ("contains", "1", True)):
        cond = VerificationCondition("c", "metric", {"name": "site.views", "op": op, "value": value})
        assert evaluate_condition(cond, obs) is want, op
    assert evaluate_condition(VerificationCondition("c", "metric", {"name": "missing", "op": "eq", "value": 1}), obs) is None
    assert evaluate_condition(VerificationCondition("c", "metric", {"path": "n", "op": "delta_eq", "value": 1}), obs) is None
    assert evaluate_condition(VerificationCondition("c", "url_matches", {"pattern": "x"}), obs) is None
    assert evaluate_condition(VerificationCondition(), obs) is None and res.verified is None
    assert Level.PROCEDURE == "procedure"
