"""IH-15: Discord message content is neither stored nor trained on without explicit permission (spec 12.5, 12.6).

The real stack against the local STAGING Discord service (nothing talks to Discord): authorization service with
the default policy, computer environment, Discord executor. Message text that the adapter puts into an
observation is kept -- it enters the experience stream and is learned from -- so the adapter's own flag
(``discord_message_content``) together with permission to *read* message content is not enough: storing and
training must be permitted too, and the default policy's platform rule refuses both for Discord until the operator
records the express permission by editing that rule. Before this fix, read permission alone let the text through."""
import json
import os

import pytest
import yaml

from cam_live import contracts as C
from cam_live.harness.auth import DEFAULT_POLICY_PATH, PolicyService
from cam_live.harness.discord import CONTENT_PERMISSION, DiscordExecutor
from cam_live.harness.env import ComputerEnvironment
from cam_live.harness.http_api import ApiModality
from cam_live.harness.staging_discord import StagingDiscord, StagingDiscordServer

SECRETS = ("pineapples are a pizza topping", "my second remark", "bob's sentence")


@pytest.fixture
def staging():
    st = StagingDiscord(token="staging-" + os.urandom(16).hex(), rate_limit=(1000, 1.0))
    server = StagingDiscordServer(st)
    alice, bob = st.add_human("alice"), st.add_human("bob")
    st.post_member_message(st.general_id, alice, SECRETS[0])
    st.post_member_message(st.general_id, alice, SECRETS[1])
    st.post_member_message(st.general_id, bob, SECRETS[2])
    yield server
    server.close()


def stack(tmp_path, server, perms, policy_path=None):
    st = server.state
    run_dir = str(tmp_path / f"run{len(os.listdir(tmp_path))}")
    owned = [f"account:discord:{st.guild_id}", "channel:discord:*"]
    task = C.TaskSpec(objective_id="obj-hx-data-use", objective="read the channel", mode="staging",
                      resources={"discord": {"guild_id": st.guild_id, "base_url": server.base_url},
                                 "owned": owned, "preauthorized": owned},
                      interfaces=["api"], data_permissions=dict(perms),
                      limits=C.Limits(duration_s=3600, max_external_actions=100, max_decisions=100))
    svc = PolicyService(run_dir, policy_path, fsync=False)
    assert svc.policy is not None, svc.policy_error
    rest = server.base_url.split("//", 1)[1].split("/", 1)[0]
    svc.put_secret("discord_bot_token", st.token, [rest, server.gateway.url.split("//", 1)[1]], ["discord"], by="test")
    svc.register_local_service(rest, by="test", note="staging Discord service of this test")
    svc.register_task(task, "operator")
    env = ComputerEnvironment(run_dir, svc, secrets=svc.secrets_handle(), provenance="staging")
    names = {s["name"] for s in svc.secret_names()}
    ex = DiscordExecutor.from_task(task, base_url=server.base_url, secrets=svc.secrets_handle(),
                                   credential_probe=lambda n: n in names,
                                   state_dir=os.path.join(run_dir, "discord_state"))
    env.register(ApiModality([ex]))
    return svc, env, task


def activity(svc, env, task, channel_id, guild_id):
    a = C.ActionSpec(action_id=C.new_id("act"), operation="discord.get_channel_activity", modality=C.Modality.API,
                     arguments={"channel_id": channel_id, "guild_id": guild_id, "limit": 50})
    dec = svc.authorize(a, task, {})
    assert dec.verdict == C.AuthVerdict.ALLOW, dec.reasons
    return env.execute(a, dec)


def leaked(res):
    blob = json.dumps(res.to_dict(), default=str)
    return [s for s in SECRETS if s in blob]


def test_read_permission_alone_does_not_let_message_text_into_the_experience_stream(tmp_path, staging):
    st = staging.state
    perms = {"aggregate_metrics": True, CONTENT_PERMISSION: True, "read_message_content": True,
             "store_message_content": False, "train_on_message_content": False}
    svc, env, task = stack(tmp_path, staging, perms)
    res = activity(svc, env, task, st.general_id, st.guild_id)
    try:
        assert not leaked(res), leaked(res)
        assert res.status == C.ResultStatus.DENIED, (res.status, res.failure_detail)
        assert "train_on_message_content" in res.failure_detail or "store_message_content" in res.failure_detail \
            or "stored or trained" in res.failure_detail, res.failure_detail
        # without the adapter flag the counts are read as before (aggregate metrics need no further permission)
        svc2, env2, task2 = stack(tmp_path, staging, {"aggregate_metrics": True})
        res2 = activity(svc2, env2, task2, st.general_id, st.guild_id)
        assert res2.status == C.ResultStatus.OK and not leaked(res2)
        q = {x.name: x.value for x in res2.observations[0].quantities}
        assert q["discord.channel.messages"] == 3.0 and q["discord.channel.distinct_human_authors"] == 2.0
        env2.close()
        svc2.close()
    finally:
        env.close()
        svc.close()


def test_store_and_train_permissions_and_the_platform_rule_decide(tmp_path, staging):
    st = staging.state
    every = {"aggregate_metrics": True, CONTENT_PERMISSION: True, "read_message_content": True,
             "store_message_content": True, "train_on_message_content": True}
    # every task permission, default policy: Discord's rule still refuses storing / training on message content
    svc, env, task = stack(tmp_path, staging, every)
    res = activity(svc, env, task, st.general_id, st.guild_id)
    assert res.status == C.ResultStatus.DENIED and not leaked(res), (res.status, leaked(res))
    assert "PL-DISCORD-MESSAGE-CONTENT" in json.dumps([r for r in env.records("env.primitive_check", 50,
                                                                              res.action_id)], default=str)
    env.close()
    svc.close()
    # the operator records Discord's express permission by editing the platform rule (rules are data): only then,
    # and only with every task permission, is the text kept
    with open(DEFAULT_POLICY_PATH) as f:
        pol = yaml.safe_load(f)
    pol["policy_id"] = "default+express-content-permission"
    pol["platforms"]["discord"]["rules"] = [r for r in pol["platforms"]["discord"]["rules"]
                                            if r["id"] != "PL-DISCORD-MESSAGE-CONTENT"]
    path = str(tmp_path / "policy_express.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(pol, f, sort_keys=False)
    svc, env, task = stack(tmp_path, staging, every, policy_path=path)
    res = activity(svc, env, task, st.general_id, st.guild_id)
    assert res.status == C.ResultStatus.OK, res.failure_detail
    assert set(leaked(res)) == set(SECRETS)
    env.close()
    svc.close()
    # same edited policy, but no permission to train: refused again
    svc, env, task = stack(tmp_path, staging, dict(every, train_on_message_content=False), policy_path=path)
    res = activity(svc, env, task, st.general_id, st.guild_id)
    assert res.status == C.ResultStatus.DENIED and not leaked(res)
    env.close()
    svc.close()
