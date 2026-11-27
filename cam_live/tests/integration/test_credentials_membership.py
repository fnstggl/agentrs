"""Operator credential entry and the verifier's own Discord membership source (staging service; nothing live)."""
import io
import json
import os
import stat
import subprocess
import sys

import pytest

from cam_live import contracts as C
from cam_live.harness import credentials as cred
from cam_live.harness.auth.vault import SecretBroker
from cam_live.harness.discord_membership import (DiscordMembershipSource, membership_source, snowflake_time)
from cam_live.harness.staging_discord import StagingDiscord, StagingDiscordServer

TOKEN = "staging-" + "x" * 60


def test_store_puts_the_value_only_in_vault_and_verifier_copy(tmp_path):
    R = str(tmp_path / "run")
    out = cred.store(R, "discord_bot_token", TOKEN)
    assert TOKEN not in json.dumps(out)
    for p in (out["vault"], out["verifier_copy"]):
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
        assert stat.S_IMODE(os.stat(os.path.dirname(p)).st_mode) == 0o700
    found = []
    for root, _dirs, files in os.walk(R):
        for f in files:
            if TOKEN in open(os.path.join(root, f), errors="replace").read():
                found.append(os.path.relpath(os.path.join(root, f), R))
    assert sorted(found) == ["vault/discord_bot_token.secret.json", "verifier/secrets/discord_bot_token"]
    rows = cred.listing(R)
    assert rows[0]["name"] == "discord_bot_token" and rows[0]["verifier_copy"] and TOKEN not in json.dumps(rows)
    assert "discord.com" in rows[0]["destinations"]
    assert cred.verifier_secret(R, "discord_bot_token") == TOKEN
    b = SecretBroker(os.path.join(R, "vault"))
    assert b.scrub(f"Authorization: Bot {TOKEN}") == "Authorization: Bot [REDACTED:discord_bot_token]"


def test_bad_values_and_loose_files_are_refused(tmp_path):
    R = str(tmp_path / "run")
    with pytest.raises(ValueError):
        cred.store(R, "discord_bot_token", "short")
    with pytest.raises(ValueError):
        cred.store(R, "discord_bot_token", "a" * 40 + " " + "b" * 40)
    with pytest.raises(ValueError):
        cred.store(R, "other_secret", "y" * 20)                      # unknown name needs an explicit destination
    f = tmp_path / "tok"
    f.write_text(TOKEN)
    os.chmod(f, 0o644)
    with pytest.raises(PermissionError):
        cred.read_value(str(f))
    os.chmod(f, 0o600)
    assert cred.read_value(str(f)) == TOKEN
    assert cred.read_value(stream=io.StringIO(TOKEN + "\n")) == TOKEN
    cred.store(R, "discord_bot_token", TOKEN)
    os.chmod(os.path.join(R, "verifier", "secrets", "discord_bot_token"), 0o644)
    with pytest.raises(PermissionError):
        cred.verifier_secret(R, "discord_bot_token")


def test_cli_never_prints_the_value_and_revoke_removes_the_copy(tmp_path):
    R = str(tmp_path / "run")
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    p = subprocess.run([sys.executable, "-m", "cam_live.harness.credentials", "set", "discord_bot_token", "--run-dir", R],
                       input=TOKEN + "\n", capture_output=True, text=True, env=env)
    assert p.returncode == 0 and TOKEN not in p.stdout + p.stderr
    p = subprocess.run([sys.executable, "-m", "cam_live.harness.credentials", "list", "--run-dir", R],
                       capture_output=True, text=True, env=env)
    assert "discord_bot_token" in p.stdout and TOKEN not in p.stdout
    p = subprocess.run([sys.executable, "-m", "cam_live.harness.credentials", "revoke", "discord_bot_token", "--run-dir", R],
                       capture_output=True, text=True, env=env)
    assert json.loads(p.stdout)["revoked"] == ["discord_bot_token"]
    assert not os.path.exists(os.path.join(R, "verifier", "secrets", "discord_bot_token"))
    with pytest.raises(PermissionError):
        SecretBroker(os.path.join(R, "vault")).release("discord_bot_token", None)


def test_membership_source_reads_members_with_the_verifier_copy(tmp_path):
    st = StagingDiscord(token=TOKEN, rate_limit=(1000, 1.0))
    R = str(tmp_path / "run")
    cred.store(R, "discord_bot_token", TOKEN)
    with StagingDiscordServer(st) as srv:
        humans = [st.add_human() for _ in range(5)]
        st.add_human(pending=True)
        st.add_bot()
        tester = st.add_test_account()
        task = C.TaskSpec(objective_id="t", objective="o", mode="staging",
                          resources={"discord": {"guild_id": st.guild_id, "base_url": srv.base_url}})
        src = membership_source(task, R)
        assert src.live is False and src.name == "discord:staging"
        ms = src.list_members()
        by_id = {m.id: m for m in ms}
        assert set(humans) <= set(by_id) and tester in by_id          # declared test accounts are the verifier's to exclude
        assert sum(1 for m in ms if m.is_bot) >= 1
        assert sum(1 for m in ms if m.pending) == 1
        assert all(m.account_created_at > 0 and m.joined_at > 0 for m in ms)
        assert st.guild["owner_id"] not in by_id
        st.remove_member(humans[0])
        assert humans[0] not in {m.id for m in src.list_members()}
        assert src.resource_usage()["discord.verifier.requests"] >= 4


def test_membership_source_pages_and_excludes_the_owner(tmp_path):
    st = StagingDiscord(token=TOKEN, rate_limit=(1000, 1.0))
    with StagingDiscordServer(st) as srv:
        ids = [st.add_human() for _ in range(7)]
        st.guild["owner_id"] = ids[3]
        src = DiscordMembershipSource(st.guild_id, lambda: TOKEN, base_url=srv.base_url, page_size=3)
        ms = src.list_members()                                       # the application's own bot user is a member too
        assert {m.id for m in ms if not m.is_bot} == set(ids) - {ids[3]}
        assert [m.id for m in ms if m.is_bot] == [st.bot_user["id"]]
        assert src.last == {"listed": 7, "pages": 3, "owner_excluded": True}


def test_snowflake_time_matches_the_documented_epoch():
    assert snowflake_time("175928847299117063") == pytest.approx(1462015105.796, abs=1e-3)   # example in the API reference
    with pytest.raises(ValueError):
        DiscordMembershipSource("", lambda: TOKEN)
