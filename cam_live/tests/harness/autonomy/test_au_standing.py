"""AU: standing authorization for routine, reversible operations inside CAM's designated infrastructure, with the
operator's approval retained for irreversible operations, new external permissions, messages to people, money and
sensitive data (operator amendment 2026-11-28, "OPERATIONAL AUTONOMY"). Every decision here is made by the real
authorization service (``PolicyService``) under the shipped run policy (``configs/authorization.yaml``) on a live
task with an operator-signed record."""
import copy
import os

import pytest
import yaml

from cam_live import contracts as C
from cam_live.harness.auth import PolicyService, liveauth
from cam_live.harness.auth.schema import DEFAULT_POLICY_PATH, RUN_POLICY_PATH, compile_policy

from au_support import (ALLOW, DENY, GUILD, NEEDS, PROFILE, act, approve_and_retry, decide, live_record, live_task,
                        sign_register, standing_record)

HOST = "https://deploy.host.example"
API = "https://api.integration.example"
CH = "222222222222222222"


@pytest.fixture
def lt():
    return live_task(resources={})


@pytest.fixture
def standing(svc, tmp_path, clock, lt):
    return sign_register(svc, tmp_path, standing_record(lt, clock.now()))


def dsc(op, **kw):
    return dict(guild_id=GUILD, channel_id=CH, **kw)


def test_policy_files_are_identical_and_name_the_profile():
    assert open(DEFAULT_POLICY_PATH, "rb").read() == open(RUN_POLICY_PATH, "rb").read()
    pol = compile_policy(yaml.safe_load(open(RUN_POLICY_PATH)), raw=b"x")
    assert PROFILE in pol.standing.profiles and pol.standing.retained_modes == ["live"]
    assert {r.cls for r in pol.standing.retained} == {"irreversible", "new_external_permission", "message_to_person"}


def test_routine_operations_on_designated_infrastructure_need_no_approval(svc, lt, standing):
    for op, args in [("discord.send_message", dsc("x", content="Notes from today's session are in #notes.")),
                     ("discord.create_channel", dict(guild_id=GUILD, name="notes", type="text")),
                     ("discord.modify_channel", dsc("x", topic="weekly notes")),
                     ("discord.create_invite", dsc("x", max_age_s=86400)),
                     ("discord.pin_message", dsc("x", message_id="333333333333333333")),
                     ("http.request", dict(method="PUT", url=HOST + "/sites/cam/index.html", body="<h1>hello</h1>")),
                     ("http.request", dict(method="POST", url=HOST + "/sites/cam/deploys", json={"files": 3})),
                     ("http.request", dict(method="PATCH", url=API + "/v1/projects/7", json={"title": "t"})),
                     ("browser.navigate", dict(url="https://docs.python.org/3/library/zipfile.html")),
                     ("http.get", dict(url="https://pypi.org/pypi/httpx/json"))]:
        d = decide(svc, lt, op, None, **args)
        assert d.verdict == ALLOW, (op, args, d.verdict, d.reasons)


def test_the_same_writes_without_a_standing_profile_still_ask(svc, tmp_path, clock, lt):
    # a record that does not pre-authorize: every write to an owned destination needs the operator (unchanged)
    sign_register(svc, tmp_path, standing_record(lt, clock.now(), preauthorized=[], standing=None)
                  if False else {k: v for k, v in standing_record(lt, clock.now(), preauthorized=[]).items()
                                 if k != "standing"})
    d = decide(svc, lt, "discord.send_message", None, **dsc("x", content="hello there"))
    assert d.verdict == NEEDS and "P-PUBLISH-APPROVAL" in d.policy_ids, (d.verdict, d.policy_ids)
    d = decide(svc, lt, "http.request", None, method="PUT", url=HOST + "/sites/cam/a.html", body="x")
    assert d.verdict == NEEDS and "P-WRITE-APPROVAL" in d.policy_ids


@pytest.mark.parametrize("op,args,pid", [
    ("discord.delete_channel", dsc("x"), "R-IRREVERSIBLE"),
    ("discord.delete_message", dsc("x", message_id="333333333333333333"), "R-IRREVERSIBLE"),
    ("discord.delete_role", dict(guild_id=GUILD, role_id="444444444444444444"), "R-IRREVERSIBLE"),
    ("http.request", dict(method="DELETE", url=HOST + "/sites/cam"), "R-IRREVERSIBLE-HTTP"),
    ("http.request", dict(method="POST", url=HOST + "/v1/tokens", json={"scope": "deploy"}), "R-NEW-PERMISSION"),
    ("http.request", dict(method="POST", url=API + "/oauth/authorize", json={"client": "x"}), "R-NEW-PERMISSION"),
    ("http.request", dict(method="PUT", url=API + "/v1/projects/7/collaborators/someone", json={}), "R-NEW-PERMISSION"),
    ("http.request", dict(method="POST", url=API + "/v1/messages", json={"to": "person@example.org", "text": "hi"}),
     "R-PERSON-MESSAGE"),
])
def test_retained_classes_keep_the_operators_approval(svc, lt, standing, op, args, pid):
    a = act(op, arguments=args)
    d, d2 = approve_and_retry(svc, lt, a)
    assert pid in d.policy_ids, (op, d.policy_ids, d.reasons)
    assert "approval is retained" in " ".join(d.reasons)
    # the approved grant covers the operation's own primitive at execution time (and nothing else)
    assert svc.check_grant(a, d2)
    if op.startswith("discord."):
        p = C.PrimitiveOp(op=op, modality=C.Modality.API, arguments=dict(args, environment="live"))
    else:
        p = C.PrimitiveOp(op="http.request", modality=C.Modality.API,
                          arguments={"method": args["method"], "url": args["url"], "hop": 0})
    pd = svc.check_primitive(p, a)
    assert pd.verdict == ALLOW, pd.reasons


def test_a_retained_operation_inside_a_routine_grant_asks_at_execution(svc, lt, standing):
    # a routine PUT is granted without approval; a DELETE its executor issues under that grant is not covered
    a = act("http.request", arguments=dict(method="PUT", url=HOST + "/sites/cam/a.html", body="x"))
    d = svc.authorize(a, lt, {})
    assert d.verdict == ALLOW and svc.check_grant(a, d)
    p = C.PrimitiveOp(op="http.request", modality=C.Modality.API,
                      arguments={"method": "DELETE", "url": HOST + "/sites/cam", "hop": 0})
    pd = svc.check_primitive(p, a)
    assert pd.verdict != ALLOW and "R-IRREVERSIBLE-HTTP" in pd.policy_ids, (pd.verdict, pd.policy_ids)


def test_money_and_sensitive_data_are_not_opened_by_a_standing_profile(svc, lt, standing):
    d = decide(svc, lt, "payment.purchase", None, amount=5.0, payee=HOST + "/billing")
    assert d.verdict == NEEDS and "P-SPEND-APPROVAL" in d.policy_ids, (d.verdict, d.policy_ids)
    d = decide(svc, lt, "discord.list_members", None, guild_id=GUILD)
    assert d.verdict == DENY, d.reasons                    # member_list is not granted by the task
    d = decide(svc, lt, "discord.read_messages", None, **dsc("x"))
    assert d.verdict == DENY, d.reasons                    # message content is not granted by the task


def test_registration_and_input_elsewhere_keep_approval_and_the_hard_constraints_hold(svc, lt, standing):
    d = decide(svc, lt, "browser.fill", C.Modality.STRUCTURED_BROWSER, ref="e3", text="cam@mail.example",
               page_url="https://service.example/signup")
    assert d.verdict == NEEDS and "P-INTERACT-APPROVAL" in d.policy_ids, (d.verdict, d.policy_ids)
    d = decide(svc, lt, "http.request", None, method="POST", url="https://service.example/api/register",
               json={"email": "cam@mail.example"})
    assert d.verdict == DENY and "P-WRITE-NON-OWNED" in d.policy_ids
    # never, whatever is pre-authorized: a CAPTCHA answer, the Discord client, Discord outside the adapter
    d = decide(svc, lt, "http.request", None, method="POST", url=HOST + "/signup",
               json={"email": "a@b.example", "h-captcha-response": "P1_eyJ0"})
    assert d.verdict == DENY and any(p.startswith("X-HUMAN") for p in d.policy_ids), d.policy_ids
    d = decide(svc, lt, "browser.click", C.Modality.STRUCTURED_BROWSER, ref="e1",
               page_url="https://discord.com/channels/@me")
    assert d.verdict == DENY, d.reasons
    d = decide(svc, lt, "http.get", None, url="https://discord.com/api/v10/users/@me")
    assert d.verdict == DENY and any(p.startswith("X-") for p in d.policy_ids), d.policy_ids


def test_generated_code_reaches_the_standing_hosts_without_a_per_job_approval(svc, lt, standing):
    run = lambda hosts: decide(svc, lt, "sandbox.python", C.Modality.CODE, code="print(1)", network="scoped",  # noqa
                               hosts=hosts)
    assert run(["pypi.org"]).verdict == ALLOW
    assert run(["pypi.org", "files.pythonhosted.org:443"]).verdict == ALLOW
    d = run(["pypi.org", "example.org"])                   # readable, but not a standing code host
    assert d.verdict == NEEDS and "P-CODE-NETWORK" in d.policy_ids
    d = run([])                                            # network without naming hosts
    assert d.verdict == DENY
    d = run(["discord.com"])                               # generated code never talks to Discord
    assert d.verdict == DENY and any(p.startswith("X-") for p in d.policy_ids), d.policy_ids


def test_without_standing_code_network_asks_and_in_staging_it_is_closed(svc, tmp_path, clock, lt):
    rec = standing_record(lt, clock.now())
    rec.pop("standing")
    sign_register(svc, tmp_path, rec)
    d = decide(svc, lt, "sandbox.python", C.Modality.CODE, code="print(1)", network="scoped", hosts=["pypi.org"])
    assert d.verdict == NEEDS and "P-CODE-NETWORK" in d.policy_ids
    st = C.TaskSpec(objective_id="obj-au-staging", objective="x", mode="staging")
    d = decide(svc, st, "sandbox.python", C.Modality.CODE, code="print(1)", network="scoped", hosts=["pypi.org"])
    assert d.verdict == DENY and "P-CODE-NETWORK" in d.policy_ids


def test_each_connection_goes_only_to_a_host_the_run_declared(svc, lt, standing):
    a = act("sandbox.shell", C.Modality.CODE, arguments=dict(command="python -m pip download x", network="scoped",
                                                            hosts=["pypi.org", "files.pythonhosted.org"]))
    d = svc.authorize(a, lt, {})
    assert d.verdict == ALLOW and svc.check_grant(a, d)
    run = C.PrimitiveOp(op="sandbox.shell", modality=C.Modality.CODE,
                        arguments={"sha256": "0" * 64, "bytes": 10, "network": "scoped", "cwd": ".", "wall_s": 30.0,
                                   "hosts": ["pypi.org", "files.pythonhosted.org"]})
    assert svc.check_primitive(run, a).verdict == ALLOW

    def connect(host, port=443):
        return svc.check_primitive(C.PrimitiveOp(op="net.connect", modality=C.Modality.CODE, arguments={
            "host": host, "port": port, "ip": "", "private": False, "url": f"tcp://{host}:{port}",
            "from": "sandbox.shell", "dest": f"{host}:{port}"}), a)
    assert connect("pypi.org").verdict == ALLOW
    assert connect("files.pythonhosted.org").verdict == ALLOW
    for host in ("example.org", "docs.python.org", "discord.com", "127.0.0.1"):
        r = connect(host)
        assert r.verdict == DENY, (host, r.reasons)
    r = connect("example.org")
    assert "P-CODE-NETWORK" in r.policy_ids and "declared" in " ".join(r.reasons)


def test_an_unknown_profile_is_refused_at_registration_and_ignored_by_an_older_policy(svc, tmp_path, clock, lt):
    with pytest.raises(liveauth.LiveAuthorizationError, match="not a standing profile"):
        sign_register(svc, tmp_path, standing_record(lt, clock.now(), standing={"profile": "anything-goes"}))
    # a run whose policy copy predates the standing section (no profiles): a standing record cannot be registered,
    # and nothing it would pre-authorize for code is opened
    doc = yaml.safe_load(open(RUN_POLICY_PATH))
    doc.pop("standing")
    doc["tools"] = [t for t in doc["tools"] if t["id"] != "T-CODE-CONNECT"]
    old = tmp_path / "old_policy.yaml"
    old.write_text(yaml.safe_dump(doc))
    s2 = PolicyService(str(tmp_path / "run2"), str(old), clock=clock, fsync=False)
    try:
        assert s2.policy is not None, s2.policy_error
        with pytest.raises(liveauth.LiveAuthorizationError):
            sign_register(s2, tmp_path, standing_record(lt, clock.now()))
        rec = standing_record(lt, clock.now())
        rec.pop("standing")
        sign_register(s2, tmp_path, rec)
        d = decide(s2, lt, "http.request", None, method="DELETE", url=HOST + "/sites/cam")
        assert d.verdict == ALLOW                          # the old meaning: pre-authorized, nothing retained
        d = decide(s2, lt, "sandbox.python", C.Modality.CODE, code="1", network="scoped", hosts=["pypi.org"])
        assert d.verdict == NEEDS
    finally:
        s2.close()


def test_stop_revocation_and_the_window_end_standing_authorization(svc, tmp_path, clock, lt, standing):
    put = lambda: decide(svc, lt, "http.request", None, method="PUT", url=HOST + "/sites/cam/a.html", body="x")  # noqa
    assert put().verdict == ALLOW
    svc.emergency_stop("operator test", "operator")
    assert put().verdict == DENY
    assert svc.reset_stop("operator")
    assert put().verdict == ALLOW
    svc.revoke(f"destination:{HOST}/*", "operator")
    d = put()
    assert d.verdict == DENY, d.reasons
    clock.advance(7200.0)                                  # past not_after of the signed record
    d = decide(svc, lt, "discord.send_message", None, **dsc("x", content="still there?"))
    assert d.verdict == DENY, d.reasons


def test_a_new_signed_record_adds_an_approved_integration_while_the_run_continues(svc, tmp_path, clock, lt, standing):
    url = "https://api.newservice.example/v1/items"
    d = decide(svc, lt, "http.request", None, method="POST", url=url, json={"name": "a"})
    assert d.verdict == DENY                               # not CAM's: not owned
    rec = standing_record(lt, clock.now(), record_id="rec-2")
    rec["owned"].append("https://api.newservice.example/*")
    rec["preauthorized"].append("https://api.newservice.example/*")
    clock.advance(1.0)
    sign_register(svc, tmp_path, rec)
    d = decide(svc, lt, "http.request", None, method="POST", url=url, json={"name": "a"})
    assert d.verdict == ALLOW, d.reasons


def test_a_record_without_a_profile_keeps_the_meaning_of_its_explicit_pre_authorizations(svc, tmp_path, clock, lt):
    # e.g. the Discord live check pre-authorizes deleting the test channel it created; no retained rule applies
    rec = standing_record(lt, clock.now())
    rec.pop("standing")
    sign_register(svc, tmp_path, rec)
    assert decide(svc, lt, "discord.delete_channel", None, **dsc("x")).verdict == ALLOW
    assert decide(svc, lt, "http.request", None, method="DELETE", url=HOST + "/sites/cam").verdict == ALLOW


def test_staging_mode_keeps_its_meaning(svc):
    st = C.TaskSpec(objective_id="obj-au-staging2", objective="x", mode="staging",
                    resources={"owned": ["channel:discord:*", f"account:discord:{GUILD}"],
                               "preauthorized": ["channel:discord:*", f"account:discord:{GUILD}"]})
    d = decide(svc, st, "discord.delete_channel", None, **dsc("x"))
    assert d.verdict == ALLOW, d.reasons                   # retained approval applies to live mode only


def test_a_signed_record_cannot_carry_a_wildcard_code_network():
    t = live_task(resources={})
    for bad in (["*"], ["https://*"], ["*.com"], ["py*.org"]):
        with pytest.raises(liveauth.LiveAuthorizationError):
            liveauth.validate_shape(standing_record(t, 1.0, standing={"profile": PROFILE, "code_network": bad}))
    with pytest.raises(liveauth.LiveAuthorizationError):
        liveauth.validate_shape(standing_record(t, 1.0, standing={"profile": PROFILE, "extra": True}))
