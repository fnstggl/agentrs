"""OP-07: credential entry by the operator, isolated from the learner, training data and logs (run with
``-m privileged`` as root; uses the real service users).

The operator CLI (``python -m cam_live.harness.credentials set``) stores a value into a run directory prepared
by the real launcher with the authorization service running under its own uid. Then: the value is in exactly two
files (vault, verifier copy), both unreadable for the learner and sandbox uids; the CLI never prints it and never
takes it from argv; the running authorization service sees the secret by name, never shows its value in
decisions, the audit log or a listing, and its scrubber redacts it from text."""
import json
import os
import subprocess
import sys

import pytest

from cam_live import contracts as C
from cam_live.harness import SERVICE_USERS
from cam_live.harness.auth.client import connect
from cam_live.harness.auth.launcher import launch
from ea_helpers import SRC, act, jout, run_as, task

pytestmark = pytest.mark.privileged
U = SERVICE_USERS
FAST = dict(separated=True, poll_s=0.05, restart_backoff_s=0.05, stop_grace_s=2.0)
VALUE = "MTU3NTY4MTc4MDQ2MjEyOTIzMg.HXtest.operator-entered-" + "q" * 30
TRY_READ = r"""
import errno, json, os, sys
out = {}
for p in sys.argv[1:]:
    try:
        if os.path.isdir(p):
            os.listdir(p)
        else:
            with open(p, "rb") as f: f.read(1)
        out[p] = "ok"
    except OSError as e:
        out[p] = errno.errorcode.get(e.errno, str(e.errno))
print(json.dumps(out))
"""


def test_operator_entered_credential_is_isolated_from_learner_sandbox_training_data_and_logs(shared_dir, service_users):
    run = os.path.join(shared_dir, "run")
    h = launch(run, ["auth"], dict(FAST))
    try:
        cli = [sys.executable, "-m", "cam_live.harness.credentials"]
        env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": SRC, "HOME": "/root"}
        cp = subprocess.run(cli + ["set", "discord_bot_token", "--run-dir", run], input=VALUE, text=True,
                            capture_output=True, env=env, timeout=60)
        assert cp.returncode == 0, cp.stderr
        assert VALUE not in cp.stdout + cp.stderr
        listed = subprocess.run(cli + ["list", "--run-dir", run], text=True, capture_output=True, env=env, timeout=60)
        assert listed.returncode == 0 and "discord_bot_token" in listed.stdout and VALUE not in listed.stdout
        # argv never carries a value: there is no option for it
        bad = subprocess.run(cli + ["set", "discord_bot_token", VALUE, "--run-dir", run], text=True,
                             capture_output=True, env=env, timeout=60)
        assert bad.returncode != 0
        vault_file = os.path.join(run, "vault", "discord_bot_token.secret.json")
        copy = os.path.join(run, "verifier", "secrets", "discord_bot_token")
        assert os.stat(vault_file).st_uid == service_users["auth"] and os.stat(copy).st_uid == service_users["verifier"]
        # exactly two files hold the value
        holders = []
        for root, _d, files in os.walk(run):
            for f in files:
                p = os.path.join(root, f)
                try:
                    if VALUE.encode() in open(p, "rb").read():
                        holders.append(os.path.relpath(p, run))
                except OSError:
                    pass
        assert sorted(holders) == ["vault/discord_bot_token.secret.json", "verifier/secrets/discord_bot_token"]
        # the learner and the sandbox cannot read either, nor list the directories
        for user in (U["learner"], U["sandbox"]):
            res = jout(run_as(user, TRY_READ, vault_file, copy, os.path.join(run, "vault"),
                              os.path.join(run, "verifier", "secrets")))
            assert set(res.values()) == {"EACCES"}, (user, res)
        # the running authorization service knows it by name only
        c = connect(run)
        t = task(oid="obj-hx-cred")
        c.register_task(t, "operator")
        names = c.secret_names()
        assert any(n["name"] == "discord_bot_token" for n in names) and VALUE not in json.dumps(names, default=str)
        d = c.authorize(act("http.get", arguments={"url": "http://127.0.0.1:8765/x",
                                                   "auth": {"scheme": "bot", "secret": "secret:discord_bot_token"}}), t, {})
        assert VALUE not in json.dumps(d.to_dict())
        # the learner writing the value into an action is refused as a credential leak, and it is redacted from text
        leak = c.authorize(act("http.post", arguments={"url": "http://127.0.0.1:8765/x", "body": f"token {VALUE}"}), t, {})
        assert leak.verdict == C.AuthVerdict.DENY and "X-CREDENTIAL-LEAK" in leak.policy_ids
        assert VALUE not in c.scrub(f"Authorization: Bot {VALUE}")
        assert VALUE not in json.dumps(c.audit_tail(200), default=str)
        # revoking removes the verifier copy and the vault entry stops being releasable
        rv = subprocess.run(cli + ["revoke", "discord_bot_token", "--run-dir", run], text=True, capture_output=True,
                            env=env, timeout=60)
        assert rv.returncode == 0 and not os.path.exists(copy)
        # (listed before, see above): no longer listed as available
        assert not [n for n in c.secret_names() if n["name"] == "discord_bot_token" and not n.get("revoked")]
    finally:
        h.stop()
