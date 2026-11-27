"""``tools/discord_live_check.py`` against the STAGING services: the sequence, its exit codes and its cleanup.

The script drives the real computer environment and the real authorization service (in-process). Nothing here
talks to Discord; the one test of the live entry point shows that it stops before any connection is made."""
import importlib.util
import json
import os
import socket

import pytest

from cam_live import contracts as C
from cam_live.harness.discord_perms import DEFAULT_SCOPE

TOOL = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools", "discord_live_check.py")
spec = importlib.util.spec_from_file_location("discord_live_check", TOOL)
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)

WRITE_OPS = ["create_channel", "send_message", "edit_message", "pin_message", "unpin_message", "create_invite",
             "delete_invite", "modify_channel", "set_channel_permissions", "create_role", "modify_role", "delete_role",
             "modify_guild", "modify_guild", "delete_channel"]


def staging_run(tmp_path, *extra, mutate=None, name="run"):
    out = str(tmp_path / f"{name}.json")
    seen = {}

    def grab(st, resources):
        seen["st"] = st
        if mutate is not None:
            mutate(st, resources)

    args = lc.main.__globals__["argparse"].Namespace(
        run_dir=str(tmp_path / name), task="", policy=None, skip_writes=False, await_reply_s=0.0, approve_wait_s=0.0,
        listen_s=1, out=out, staging=True, live_policy=False, print_record_template=False, print_policy_rule=False,
        write_policy="")
    for flag in extra:
        if isinstance(flag, tuple):
            setattr(args, flag[0], flag[1])
        else:
            setattr(args, flag, True)
    code = lc.run_staging(args, mutate=grab)
    return code, json.load(open(out)), seen["st"], open(out).read()


def ops_of(ev, outcome=None):
    return [s["operation"].split(".", 1)[1] for s in ev["steps"] if outcome is None or s["outcome"] == outcome]


def test_full_sequence_in_staging_leaves_the_server_as_it_was(tmp_path, capsys):
    code, ev, st, raw = staging_run(tmp_path, ("await_reply_s", 1.0))
    assert code == 0 and ev["exit_code"] == 0 and ev["mode"] == "staging" and ev["environment"] == "staging" and ev["staging"] is True
    assert ev["summary"]["by_outcome"] == {"ok": len(ev["steps"])} and ev["summary"]["not_ok"] == []
    done = ops_of(ev)
    assert done[:6] == ["get_identity", "get_permissions", "get_guild_settings", "list_channels", "get_guild_stats", "list_invites"]
    assert [o for o in done if o in set(WRITE_OPS)] == WRITE_OPS and done[-1] == "delete_channel"
    assert done[-5:-1] == ["get_member_events", "get_member_events", "listen_events", "get_audit_summary"]
    # every step went validate -> authorize -> execute: an allow verdict, and the adapter's own checked operations
    assert all(s["verdict"] == "allow" and s["checked_ops"][0] == s["operation"] for s in ev["steps"])
    # the server is as it was found: no temporary channel, role or invite, the original name
    after = ev["staging_state_after"]
    assert after == {"channels": ["general"], "roles": ["@everyone", "staging-app"], "guild_name": "staging guild", "invites": 0}
    assert st.guild["name"] == "staging guild" and "cam-integration-test" not in [c["name"] for c in st.channels.values()]
    renames = [s["arguments"]["name"] for s in ev["steps"] if s["operation"] == "discord.modify_guild"]
    assert renames == ["staging guild (integration test)", "staging guild"]
    # what the steps measured
    by_note = {s["note"]: s for s in ev["steps"] if s.get("note")}
    assert by_note["reads back the bot's message"]["bot_message_read_back"] is True
    assert ev["reply"]["human_reply_arrived"] is True
    assert by_note["first reading"]["data"]["baseline"] is True
    assert by_note["second reading"]["data"]["joined"] == {"people": 1, "bots": 1, "test_accounts": 0}
    window = [s for s in ev["steps"] if s["operation"] == "discord.listen_events"][0]["data"]["window"]
    assert window["session_ok"] and window["counts"]["GUILD_MEMBER_ADD"] == 1 and window["counts"]["GUILD_MEMBER_REMOVE"] == 1
    assert ev["staging_requests"]["gateway_identifies"] == ["ready"] and ev["staging_requests"]["writes"] == len(WRITE_OPS)
    assert all("limit" in s["rate_limit"] and "remaining" in s["rate_limit"] for s in ev["steps"] if s["operation"] != "discord.listen_events")
    assert ev["write_budget"]["enough"] is True and ev["write_budget"]["needed"] == 16 and ev["missing_capabilities"] == []
    # the token is nowhere: not in the evidence, not in what was printed
    printed = capsys.readouterr().out
    assert ev["token_free"] is True and ev["secrets_checked"] == 1
    assert st.token not in raw and st.token not in printed and "[REDACTED" not in raw
    assert "No request was sent to Discord" in ev["staging_note"] and printed.count("\n") >= len(ev["steps"])
    # the audit log of the authorization service saw every step and holds no secret either
    audit = open(tmp_path / "run" / "audit.jsonl").read()
    assert audit.count('"authorize"') >= len(ev["steps"]) and st.token not in audit


def test_the_policys_live_mode_with_a_signed_record(tmp_path):
    code, ev, st, raw = staging_run(tmp_path, "live_policy")
    assert code == 0 and ev["policy_mode"] == "live" and ev["environment"] == "staging"       # live rules, staging services
    assert ev["summary"]["by_outcome"] == {"ok": len(ev["steps"])} and ev["live_policy"]["record_id"].startswith("discord-live-check-")
    assert ev["live_policy"]["preauthorized"] == [f"account:discord:{st.guild_id}", "channel:discord:*"]
    assert ev["staging_state_after"]["guild_name"] == "staging guild" and st.token not in raw
    # without the record the same task is refused by the authorization service before anything is sent
    args = lc.main.__globals__["argparse"].Namespace(run_dir=str(tmp_path / "bare"), task="", policy=None)
    task = C.TaskSpec(objective_id="t", objective="x", mode="live", interfaces=["api"],
                      resources={"discord": {"guild_id": st.guild_id}},
                      limits=C.Limits(duration_s=60, max_external_actions=5, max_decisions=5))
    record = lc.record_template(task, st.guild_id)
    assert record["task_hash"] == task.content_hash() and record["task_id"] == "t"
    assert record["read"] == ["discord.com", "gateway.discord.gg"] and record["budgets"]["max_external_actions"] > 0
    assert args.run_dir.endswith("bare")


def test_scope_and_permission_refusals_are_reported_and_cleanup_still_runs(tmp_path):
    def narrow(st, resources):
        resources["discord"]["scope"] = [n for n in DEFAULT_SCOPE if n != "MANAGE_ROLES"]      # the operator's choice
        st.revoke_bot("PIN_MESSAGES")                                                           # the owner's choice

    code, ev, st, raw = staging_run(tmp_path, mutate=narrow)
    assert code == 1 and ev["exit_code"] == 1
    assert ev["summary"]["scope_refusals"] == ["discord.set_channel_permissions", "discord.create_role"]
    assert ops_of(ev, "scope_refusal") == ["set_channel_permissions", "create_role"]
    refused = [s for s in ev["steps"] if s["outcome"] == "scope_refusal"]
    assert all(s["status"] == "not_executed" and s["verdict"] == "" and "needs MANAGE_ROLES" in s["failure_detail"] for s in refused)
    assert ops_of(ev, "permission_refusal") == ["pin_message", "unpin_message"]
    assert ev["summary"]["permission_refusals"][0] == ("discord.pin_message: the bot lacks PIN_MESSAGES in this server; "
                                                       "the server owner must grant it")
    assert ops_of(ev, "skipped") == ["modify_role, delete_role"]
    assert {c["capability"] for c in ev["missing_capabilities"]} >= {"discord.create_role", "discord.pin_message"}
    # everything else ran, and the server was put back
    assert ev["staging_state_after"] == {"channels": ["general"], "roles": ["@everyone", "staging-app"],
                                         "guild_name": "staging guild", "invites": 0}
    assert ops_of(ev)[-1] == "delete_channel" and ev["steps"][-1]["outcome"] == "ok"
    assert not [r for r in st.request_log if "/pins/" in r["path"] or "/roles" in r["path"] and r["method"] != "GET"]
    # with only the scope narrowed (nothing the scope allows fails) the exit code is 0
    code, ev, st, raw = staging_run(tmp_path, mutate=lambda st, r: r["discord"].update(
        scope=[n for n in DEFAULT_SCOPE if n != "MANAGE_ROLES"]), name="scope_only")
    assert code == 0 and ev["summary"]["by_outcome"]["scope_refusal"] == 2 and "permission_refusal" not in ev["summary"]["by_outcome"]


def test_stops_after_identity_when_the_bot_is_not_in_the_server(tmp_path, capsys):
    code, ev, st, raw = staging_run(tmp_path, mutate=lambda st, r: st.remove_bot_from_guild())
    assert code == 3 and ops_of(ev) == ["get_identity"] and ev["steps"][0]["data"]["in_guild"] is False
    assert "is not a member of server" in capsys.readouterr().out
    assert [r["method"] for r in st.request_log] == ["GET", "GET", "GET"] and st.token not in raw
    assert ev["missing_capabilities"][0]["reason"] == "the bot is not a member of the authorized server"


def test_stops_before_any_write_when_the_policy_write_rate_is_too_low(tmp_path, capsys):
    from cam_live.harness.auth.schema import DEFAULT_POLICY_PATH
    code, ev, st, raw = staging_run(tmp_path, ("policy", DEFAULT_POLICY_PATH))
    assert code == 4 and ev["write_budget"] == {"known": True, "needed": 16, "allowed": 10, "window_s": 3600.0, "used": 0,
                                                "enough": False}
    assert ops_of(ev) == ["get_identity", "get_permissions", "get_guild_settings", "list_channels", "get_guild_stats", "list_invites"]
    assert all(r["method"] == "GET" for r in st.request_log) and len(st.channels) == 1
    assert "--print-policy-rule" in capsys.readouterr().out
    rule = lc.policy_rule(st.guild_id)
    assert rule[0] == {"match": f"account:discord:{st.guild_id}", "count": 60, "window_s": 3600, "effects": ["external_write"]}
    # read-only is not held up by the write rate
    code, ev, st, raw = staging_run(tmp_path, "skip_writes", ("policy", DEFAULT_POLICY_PATH), name="reads")
    assert code == 0 and all(r["method"] == "GET" for r in st.request_log)
    assert ops_of(ev)[-4:] == ["get_member_events", "get_member_events", "listen_events", "get_audit_summary"]


def test_cleanup_runs_when_the_sequence_is_interrupted(tmp_path, monkeypatch):
    seen = {}
    real_step = lc.Check.step

    def step_then_fail(self, name, args=None, **kw):
        rec = real_step(self, name, args, **kw)
        if kw.get("note") == "rename":                     # the rename reached the server; then the script is interrupted
            raise KeyboardInterrupt()
        return rec

    monkeypatch.setattr(lc.Check, "step", step_then_fail)
    with pytest.raises(KeyboardInterrupt):
        staging_run(tmp_path, mutate=lambda st, r: seen.update(st=st))
    st = seen["st"]
    assert st.guild["name"] == "staging guild"                                              # restored in the finally
    assert [c["name"] for c in st.channels.values()] == ["general"]                         # the temporary channel is gone
    assert sorted(r["name"] for r in st.roles.values()) == ["@everyone", "staging-app"]
    patches = [r for r in st.request_log if r["method"] == "PATCH" and r["path"].endswith(f"/guilds/{st.guild_id}")]
    assert len(patches) == 2 and st.request_log[-1]["method"] == "DELETE"


def test_the_live_entry_point_stops_before_any_connection_without_an_authorization_record(tmp_path, monkeypatch, capsys):
    from cam_live.harness.auth import PolicyService
    run_dir = str(tmp_path / "run")
    secret = "not-a-real-token-" + "q" * 48
    svc = PolicyService(run_dir, fsync=False)
    svc.put_secret("discord_bot_token", secret, ["discord.com", "gateway.discord.gg"])
    svc.close()
    task = C.TaskSpec(objective_id="discord-check", objective="integration check", mode="live", interfaces=["api"],
                      resources={"discord": {"guild_id": "123456789012345678"}},
                      limits=C.Limits(duration_s=600, max_external_actions=100, max_decisions=100))
    task_file = str(tmp_path / "task.json")
    json.dump(task.to_dict(), open(task_file, "w"))
    attempts = []
    monkeypatch.setattr(socket.socket, "connect", lambda self, addr: attempts.append(addr) or (_ for _ in ()).throw(
        AssertionError(f"a connection to {addr} was attempted")))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: attempts.append(a) or (_ for _ in ()).throw(
        AssertionError(f"a name lookup for {a[:1]} was attempted")))
    out = str(tmp_path / "live.json")
    code = lc.main(["--run-dir", run_dir, "--task", task_file, "--out", out])
    ev = json.load(open(out))
    assert code == 1 and attempts == [] and ops_of(ev) == ["get_identity"]
    step = ev["steps"][0]
    assert step["verdict"] == "deny" and step["status"] == "denied" and step["outcome"] == "authorization_denied"
    assert "P-LIVE-AUTH" in step["policy_ids"] and "signed operator authorization" in step["failure_detail"]
    assert ev["mode"] == "live" and ev["environment"] == "live" and ev["token_free"] is True and secret not in open(out).read()
    assert secret not in capsys.readouterr().out
    # the record the operator would sign for exactly this task file
    assert lc.main(["--run-dir", run_dir, "--task", task_file, "--print-record-template"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["task_hash"] == task.content_hash() and record["accounts"] == ["account:discord:123456789012345678"]
    assert lc.main(["--run-dir", run_dir, "--task", task_file, "--print-policy-rule"]) == 0
    assert json.loads(capsys.readouterr().out)["rate_limits"]["per_destination"][0]["match"] == "account:discord:123456789012345678"
    # a task that is not a live task, or a run directory without the token, is a setup error; nothing is contacted
    staging_task = str(tmp_path / "staging.json")
    json.dump(C.TaskSpec(objective_id="s", objective="x", mode="staging", resources={"discord": {"guild_id": "123456789012345678"}}).to_dict(),
              open(staging_task, "w"))
    assert lc.main(["--run-dir", run_dir, "--task", staging_task]) == 2
    assert lc.main(["--run-dir", str(tmp_path / "empty"), "--task", task_file]) == 2
    assert lc.main(["--task", task_file]) == 2 and attempts == []
    err = capsys.readouterr()
    assert "no Discord bot token provided" in err.out and "mode 'live'" in err.err


def test_the_documented_live_preparation_yields_grants_and_releases_the_token_only_to_discord(tmp_path, monkeypatch, capsys):
    """The commands of docs/ES/INTEGRATION.md ("Live check"), run on a made-up token and server id. Only the
    authorization service is asked: no action is executed, so nothing can be sent anywhere."""
    from cam_live.harness import credentials
    from cam_live.harness.auth import PolicyService, opctl
    from cam_live.harness.auth.vault import SecretError
    attempts = []
    monkeypatch.setattr(socket.socket, "connect", lambda self, addr: attempts.append(addr) or (_ for _ in ()).throw(OSError("blocked")))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: attempts.append(a) or (_ for _ in ()).throw(OSError("blocked")))
    R, guild = str(tmp_path / "run"), "123456789012345678"
    made_up = "made-up-token-" + "k" * 56
    # 1. the token, bound to discord.com and the gateway host by default
    stored = credentials.store(R, "discord_bot_token", made_up, verifier=False)
    assert stored["destinations"] == ["discord.com", "gateway.discord.gg", "*.discord.gg"]
    # 2. the task file
    task_file = str(tmp_path / "discord_check_task.yaml")
    open(task_file, "w").write(
        "version: 1\nobjective_id: discord-live-check\nmode: live\n"
        'objective: "Integration check of the Discord adapter in the authorized server."\n'
        f'resources:\n  discord:\n    guild_id: "{guild}"\ninterfaces: [api]\n'
        "limits:\n  duration_s: 3600\n  money: 0\n  compute_s: 0\n  max_external_actions: 200\n  max_decisions: 200\n")
    task = lc.load_task(task_file)
    assert task.mode == "live" and task.resources["discord"]["guild_id"] == guild
    # 3. operator key, 4. record: template -> sign -> register
    key, pub = str(tmp_path / "operator.key"), os.path.join(R, "auth", "keys", "operator.pub")
    assert opctl.main(["init-operator-key", "--private", key, "--public", pub]) == 0
    capsys.readouterr()
    assert lc.main(["--task", task_file, "--print-record-template"]) == 0
    record_file = str(tmp_path / "record.json")
    open(record_file, "w").write(capsys.readouterr().out)
    assert opctl.main(["sign-live", record_file, "--private", key]) == 0
    signed = json.loads(capsys.readouterr().out)
    svc = PolicyService(R, fsync=False)
    assert svc.register_live_authorization(signed, "operator")["task_id"] == "discord-live-check"
    svc.close()
    # 5. the policy with the write rate of the check
    policy_file = str(tmp_path / "policy.yaml")
    assert lc.main(["--task", task_file, "--write-policy", policy_file]) == 0 and os.path.exists(policy_file)
    capsys.readouterr()
    # what the check will get from the authorization service, step by step
    policy = PolicyService(R, policy_file, fsync=False)
    assert policy.policy is not None and policy.policy.policy_id == "default-1+discord-live-check"

    def authorize(op, **args):
        action = C.ActionSpec(action_id=C.new_id("t"), operation=f"discord.{op}", modality="api", arguments=dict(args, guild_id=guild))
        return action, policy.authorize(action, task, {})

    for op, args in (("get_identity", {}), ("create_channel", {"name": "cam-integration-test"}),
                     ("send_message", {"channel_id": "223456789012345678", "content": "text of the check"}),
                     ("modify_guild", {"name": "a name (integration test)"}), ("listen_events", {"seconds": 10}),
                     ("delete_channel", {"channel_id": "223456789012345678"})):
        action, decision = authorize(op, **args)
        assert decision.verdict == "allow", (op, decision.reasons)
    # more writes than the default policy's ten per hour are allowed by the written policy
    assert all(authorize("modify_channel", channel_id="223456789012345678", topic=f"t{i}")[1].verdict == "allow" for i in range(14))
    # another server is not this task's to write to
    other = C.ActionSpec(action_id=C.new_id("t"), operation="discord.create_channel", modality="api",
                         arguments={"name": "x", "guild_id": "999999999999999999"})
    assert policy.authorize(other, task, {}).verdict == "deny"
    # the token is released to a granted, executing action for the REST and the gateway destination, and nowhere else
    action, decision = authorize("listen_events", seconds=10)
    with pytest.raises(SecretError):
        policy.secret_get("discord_bot_token", action, decision, "https://discord.com/api/v10")      # not executing yet
    assert policy.check_grant(action, decision) is True
    assert policy.secret_get("discord_bot_token", action, decision, "https://discord.com/api/v10") == made_up
    assert policy.secret_get("discord_bot_token", action, decision, "wss://gateway.discord.gg") == made_up
    for elsewhere in ("https://example.org/api", "wss://gateway.example.org", "https://discord.com.example.org/api/v10"):
        with pytest.raises(SecretError):
            policy.secret_get("discord_bot_token", action, decision, elsewhere)
    policy.close()
    assert attempts == []
