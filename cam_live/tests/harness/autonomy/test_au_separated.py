"""AU (privileged): the standing profile in the separated deployment -- the authorization service runs as its own
uid (launcher), the operator registers the signed standing record over AuthRPC as root, and the learner's uid gets
routine operations granted without approval and retained ones queued, but can neither register a record, change
the policy nor read the vault."""
import json
import os
import time

import pytest

from cam_live.harness import SERVICE_USERS
from cam_live.harness.auth import liveauth
from cam_live.harness.auth.client import connect
from cam_live.harness.auth.launcher import launch

from au_support import GUILD, PROFILE, live_task, standing_record
from ea_helpers import jout, run_as

pytestmark = pytest.mark.privileged
FAST = dict(separated=True, poll_s=0.05, restart_backoff_s=0.05, stop_grace_s=2.0)

LEARNER = r"""
import json, sys
from cam_live import contracts as C
from cam_live.harness import rpc
from cam_live.harness.auth.client import connect
run, task_json, guild = sys.argv[1], sys.argv[2], sys.argv[3]
c = connect(run)
t = C.TaskSpec.from_dict(json.loads(task_json))
out = {}
def call(name, m, *a):
    try:
        r = c._rpc.call(m, *a)
        out[name] = rpc.to_wire(r)
    except rpc.RemoteError as e:
        out[name] = "refused: " + e.kind
A = lambda op, **kw: C.ActionSpec(C.new_id("a"), op, "api", arguments=kw)
call("send", "authorize", A("discord.send_message", guild_id=guild, channel_id="222222222222222222",
                            content="the notes of today are pinned"), t, {})
call("deploy", "authorize", A("http.request", method="PUT", url="https://deploy.host.example/sites/cam/a.html",
                              body="<p>a</p>"), t, {})
call("delete", "authorize", A("discord.delete_channel", guild_id=guild, channel_id="222222222222222222"), t, {})
call("register", "register_live_authorization", {"record_id": "forged"}, "learner")
call("reload", "reload_policy", "learner")
call("secret", "put_secret", "x_token", "abcdefgh12345678", ["https://evil.example/*"], [], "learner")
print(json.dumps(out))
"""


def test_routine_grants_and_retained_approvals_across_real_uids(shared_dir, service_users, tmp_path):
    priv, pub = str(tmp_path / "op.key"), str(tmp_path / "op.pub")
    liveauth.generate_operator_key(priv, pub)
    run = os.path.join(shared_dir, "run")
    h = launch(run, ["auth"], dict(FAST, operator_pubkey=pub))
    try:
        root = connect(run)
        t = live_task(oid="obj-au-sep", resources={})
        signed = liveauth.sign_live_authorization(standing_record(t, time.time()), priv)
        assert root.register_live_authorization(signed, "operator")["record_id"] == signed["record_id"]
        out = jout(run_as(SERVICE_USERS["learner"], LEARNER, run, json.dumps(t.to_dict()), GUILD))
        out = {k: (v["d"] if isinstance(v, dict) and "d" in v else v) for k, v in out.items()}
        assert out["send"]["verdict"] == "allow", out["send"]
        assert out["deploy"]["verdict"] == "allow", out["deploy"]
        assert out["delete"]["verdict"] == "needs_approval" and "R-IRREVERSIBLE" in out["delete"]["policy_ids"]
        for k in ("register", "reload", "secret"):
            assert str(out[k]).startswith("refused"), (k, out[k])
        events = [e["event"] for e in root.audit_tail(200)]
        assert "live_authorization_registered" in events and "approval_requested" in events
        reg = [e for e in root.audit_tail(200) if e["event"] == "live_authorization_registered"][-1]
        assert reg["data"]["standing"] == PROFILE
        assert root.verify_audit()["ok"]
    finally:
        h.stop()
