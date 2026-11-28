"""AU: the operator's tools for standing authorization and scoped credentials -- ``tools/authorize_live.py
--standing`` (what it signs, what it refuses, what it shows), ``harness.credentials`` (destination and operation
binding, expiry, generated values that are never printed, revocation picked up by a running service) and the
request kinds CAM may use to ask for an account or a verification step."""
import io
import json
import os
import sys
import time

import pytest

from cam_live import contracts as C
from cam_live.harness import credentials as CR
from cam_live.harness.auth import PolicyService, destinations as D, liveauth
from cam_live.harness.auth.vault import SecretBroker, SecretError
from cam_live.harness.operator_requests import KINDS, OperatorRequests

from au_support import PROFILE

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "tools"))
TASKS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "tasks"))
sys.path.insert(0, TOOLS)
import authorize_live as AL  # noqa: E402
from cam_live.runtime.tasks import load_task  # noqa: E402


@pytest.fixture
def planning():
    return load_task(os.path.join(TASKS, "discord_100_planning.yaml"), for_run=True)


def test_standing_record_of_the_planning_task(planning, tmp_path):
    rec = AL.build_record(planning, hours=8, owned=[], read=["https://*", "http://*"], preauthorized=[], accounts=[],
                          issued_by="operator", standing=PROFILE, hosting=["https://deploy.host.example/*"],
                          code_hosts=["pypi.org", "files.pythonhosted.org"], now=1_800_000_000.0)
    guild = str(planning.resources["discord"]["guild_id"])
    assert rec["standing"] == {"profile": PROFILE, "code_network": ["pypi.org", "files.pythonhosted.org"]}
    assert {f"account:discord:{guild}", "channel:discord:*", "https://deploy.host.example/*"} <= set(rec["owned"])
    assert set(rec["preauthorized"]) == {f"account:discord:{guild}", "channel:discord:*",
                                         "https://deploy.host.example/*"}
    assert {"https://*", "http://*", "pypi.org", "files.pythonhosted.org"} <= set(rec["read"])
    assert rec["task_hash"] == planning.content_hash() and rec["budgets"]["money"] == float(planning.limits.money)
    text = AL.summary(planning, rec)
    for needle in ("STANDING AUTHORIZATION", "R-IRREVERSIBLE", "R-NEW-PERMISSION", "R-PERSON-MESSAGE",
                   "P-SPEND-APPROVAL", "X-HUMAN-VERIFICATION", "pypi.org"):
        assert needle in text, needle
    # the record verifies and registers against the shipped policy
    priv, pub = str(tmp_path / "k"), str(tmp_path / "k.pub")
    liveauth.generate_operator_key(priv, pub)
    signed = liveauth.sign_live_authorization(rec, priv)
    svc = PolicyService(str(tmp_path / "run"), fsync=False, clock=C.SimClock(1_800_000_100.0))
    try:
        os.replace(pub, os.path.join(svc.keys_dir, "operator.pub"))
        assert svc.register_live_authorization(signed, "operator")["task_id"] == planning.objective_id
    finally:
        svc.close()


def test_without_standing_the_record_is_what_it_was(planning):
    rec = AL.build_record(planning, hours=8, owned=[], read=["https://*"], preauthorized=[], accounts=[],
                          issued_by="operator", now=1_800_000_000.0)
    assert "standing" not in rec and rec["preauthorized"] == []
    with pytest.raises(AL.StandingError):
        AL.build_record(planning, hours=8, owned=[], read=[], preauthorized=[], accounts=[], issued_by="o",
                        hosting=["https://deploy.host.example/*"])


@pytest.mark.parametrize("bad", [dict(preauthorized=["https://*"]), dict(hosting=["*"]),
                                 dict(integrations=["https://*.com/*"]), dict(code_hosts=["*"]),
                                 dict(code_hosts=["https://*"])])
def test_a_standing_profile_names_places_not_everything(planning, bad):
    kw = dict(hours=8, owned=[], read=[], preauthorized=[], accounts=[], issued_by="o", standing=PROFILE)
    kw.update(bad)
    with pytest.raises((AL.StandingError, liveauth.LiveAuthorizationError)):
        AL.build_record(planning, **kw)


def test_the_cli_refuses_an_unknown_profile_and_signs_a_known_one(planning, tmp_path, capsys):
    priv, pub = str(tmp_path / "k"), str(tmp_path / "k.pub")
    liveauth.generate_operator_key(priv, pub)
    task = os.path.join(TASKS, "discord_100_planning.yaml")
    out = str(tmp_path / "s.json")
    assert AL.main(["--task", task, "--private", priv, "--out", out, "--hours", "2", "--standing", "nope",
                    "--yes"]) == 2
    assert not os.path.exists(out)
    assert AL.main(["--task", task, "--private", priv, "--out", out, "--hours", "2", "--standing", PROFILE,
                    "--read", "https://*", "--code-host", "pypi.org", "--yes"]) == 0
    signed = json.load(open(out))
    assert oct(os.stat(out).st_mode & 0o777) == "0o600"
    assert liveauth.verify_live_authorization(signed, liveauth.load_public_key(pub))["standing"]["profile"] == PROFILE
    assert "STANDING AUTHORIZATION" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------------- credentials
def _broker(tmp_path):
    return SecretBroker(str(tmp_path / "vault"))


def test_a_credential_is_bound_to_destinations_operations_and_a_lifetime(tmp_path):
    run = str(tmp_path / "run")
    value = "v-" + os.urandom(12).hex()
    out = CR.store(run, "dns_token", value, dest=["https://api.dns.example/*"], operations=["http.request"],
                   expires_hours=2.0, verifier=False)
    assert value not in json.dumps(out) and out["operations"] == ["http.request"]
    b = SecretBroker(os.path.join(run, "vault"))
    dest = D.parse_destination("https://api.dns.example/v1/zones/1/records")
    now = time.time()
    assert b.release("dns_token", dest, now=now, operation="http.request") == value
    for kw, why in ((dict(now=now, operation="browser.fill"), "operation"),
                    (dict(now=now + 3 * 3600, operation="http.request"), "expired"),
                    (dict(now=None, operation="http.request"), "expired")):
        with pytest.raises(SecretError, match=why):
            b.release("dns_token", dest, **kw)
    with pytest.raises(SecretError, match="not bound"):
        b.release("dns_token", D.parse_destination("https://evil.example/x"), now=now, operation="http.request")
    with pytest.raises(ValueError):
        CR.store(run, "x_token", value, dest=["https://a.example/*"], expires_hours=0, verifier=False)
    with pytest.raises(ValueError):
        CR.store(run, "x_token", value, dest=[], verifier=False)        # never unbound


def test_a_generated_credential_is_stored_and_never_printed(tmp_path, capsys):
    run = str(tmp_path / "run")
    assert CR.main(["set", "svc_password", "--run-dir", run, "--dest", "https://service.example/*",
                    "--op", "browser.fill", "--generate", "--no-verifier"]) == 0
    printed = capsys.readouterr().out
    rec = json.load(open(os.path.join(run, "vault", "svc_password.secret.json")))
    assert len(rec["value"]) >= 32 and rec["value"] not in printed and json.loads(printed)["generated"] is True
    assert rec["operations"] == ["browser.fill"] and rec["destinations"] == ["https://service.example/*"]
    listing = json.dumps(CR.listing(run))
    assert rec["value"] not in listing


def test_a_running_service_picks_up_a_new_and_a_revoked_credential(tmp_path):
    run = str(tmp_path / "run")
    svc = PolicyService(run, fsync=False)
    try:
        assert not svc.secret_has("late_token")
        CR.store(run, "late_token", "late-" + os.urandom(10).hex(), dest=["https://api.late.example/*"],
                 verifier=False)
        assert svc.secret_has("late_token")
        assert CR.main(["revoke", "late_token", "--run-dir", run]) == 0
        names = {x["name"]: x for x in svc.secret_names()}
        assert names["late_token"]["revoked"] is True
    finally:
        svc.close()


def test_cam_can_ask_for_an_account_or_a_verification_step_and_asking_grants_nothing(tmp_path):
    assert {"account", "verification", "credential", "permission"} <= set(KINDS)
    reqs = OperatorRequests(str(tmp_path))
    r = reqs.request("account", "An account on a hosting service", "to deploy what I build", "")
    v = reqs.request("verification", "E-mail confirmation", "the service sent a confirmation link", "")
    assert {x["request_id"] for x in reqs.open_requests()} == {r["request_id"], v["request_id"]}
    with pytest.raises(ValueError):
        reqs.request("everything", "x" * 5, "y" * 5, "")
