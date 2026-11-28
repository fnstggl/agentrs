"""AU round 2 (G3-act G3A-2): under a standing profile, consequence keeps the operator's approval BY DEFAULT and is
judged from what a request does, not from how it is spelt. Routine = declared: a typed adapter operation of the
policy's ``routine_operations`` or a generic request / page interaction matching the signed record's
``standing.routine`` manifest (method + destination path). Everything else asks (``R-NOT-ROUTINE``). A
declared-routine request still asks when its effective method is overridden, it can spend money, delete /
transfer, create a key / token / account / access, address a person, or read / send personal data the task does
not permit. Money asks on every live record. Real ``PolicyService``, shipped run policy, live task, signed record."""
import os

import pytest
import yaml

from cam_live import contracts as C
from cam_live.harness.auth import PolicyService, liveauth
from cam_live.harness.auth import destinations as D
from cam_live.harness.auth.schema import RUN_POLICY_PATH

from au_support import (ALLOW, DENY, GUILD, NEEDS, PROFILE, ROUTINE, act, live_task, sign_register,
                        standing_record)

HOST = "https://deploy.host.example"
API = "https://api.integration.example"
CH = "222222222222222222"
EVERYTHING = [{"methods": ["GET", "HEAD", "PUT", "PATCH", "POST"], "dest": HOST + "/**"},
              {"methods": ["GET", "HEAD", "PUT", "PATCH", "POST"], "dest": API + "/**"}]

#: (label, operation, modality, arguments): consequential attempts of the G3-act probe and further spellings
CONSEQUENTIAL = [
    ("DELETE", "http.request", None, dict(method="DELETE", url=HOST + "/sites/cam")),
    ("override header", "http.request", None, dict(method="POST", url=HOST + "/sites/cam",
                                                   headers={"X-HTTP-Method-Override": "DELETE"})),
    ("override header lower case", "http.request", None, dict(method="PUT", url=HOST + "/sites/cam/a.html",
                                                              headers={"x-http-method": "delete"}, body="x")),
    ("override X-Method-Override", "http.request", None, dict(method="POST", url=HOST + "/sites/cam",
                                                              headers={"X-Method-Override": "DELETE"})),
    ("override _method form body", "http.request", None, dict(method="POST", url=HOST + "/sites/cam",
                                                              body="_method=DELETE")),
    ("override _method json", "http.request", None, dict(method="POST", url=HOST + "/sites/cam",
                                                         json={"_method": "DELETE"})),
    ("override _method url query", "http.request", None, dict(method="POST", url=HOST + "/sites/cam?_method=DELETE")),
    ("override _method query object", "http.request", None, dict(method="POST", url=HOST + "/sites/cam",
                                                                 query={"_method": "DELETE"})),
    ("domain transfer", "http.request", None, dict(method="POST", url=API + "/v1/domains/cam-example.com/transfer",
                                                   json={"authCode": "x"})),
    ("DNS delete through POST", "http.request", None, dict(method="POST", url=API + "/api/json/v3/dns/delete/x.com/1",
                                                           json={})),
    ("domain buy", "http.request", None, dict(method="POST", url=HOST + "/v4/domains/buy", json={"name": "x.com"})),
    ("order", "http.request", None, dict(method="POST", url=API + "/v1/orders", json={"plan": "pro", "quantity": 12})),
    ("charge", "http.request", None, dict(method="POST", url=API + "/v1/charges", json={"amount": 50000})),
    ("checkout", "http.request", None, dict(method="POST", url=API + "/v1/checkout/sessions", json={"price": "p1"})),
    ("plan upgrade", "http.request", None, dict(method="PATCH", url=HOST + "/v1/sites/7", json={"plan": "enterprise"})),
    ("domain purchase", "http.request", None, dict(method="POST", url=API + "/v1/domains/purchase",
                                                   json={"domain": "x.com", "period": 1})),
    ("registrar create with cost", "http.request", None, dict(method="POST", url=API + "/api/json/v3/domain/create/x.com",
                                                              json={"cost": 1100})),
    ("slack-style DM", "http.request", None, dict(method="POST", url=API + "/api/chat.postMessage",
                                                  json={"channel": "U012AB3CD", "text": "hi"})),
    ("telegram-style chat_id", "http.request", None, dict(method="POST", url=API + "/bot/sendMessage",
                                                          json={"chat_id": 123456789, "text": "hi"})),
    ("email field", "http.request", None, dict(method="POST", url=API + "/v1/send",
                                               json={"email": "person@example.org", "message": "hi"})),
    ("recipient in url query", "http.request", None, dict(method="POST", url=API + "/v3/messages?to=p@example.org",
                                                          body="hello")),
    ("recipient in query object", "http.request", None, dict(method="POST", url=API + "/v3/messages",
                                                             query={"to": "p@example.org"}, body="hello")),
    ("an e-mail address under any key", "http.request", None, dict(method="POST", url=API + "/v1/notes",
                                                                   json={"contact": "person@example.org", "n": "x"})),
    ("a phone number under any key", "http.request", None, dict(method="POST", url=API + "/v1/notes",
                                                                json={"reach": "+44 20 7946 0958"})),
    ("ssh key", "http.request", None, dict(method="POST", url=API + "/user/keys", json={"key": "ssh-ed25519 AAAA"})),
    ("deploy key", "http.request", None, dict(method="POST", url=HOST + "/api/v1/deploy_keys", json={})),
    ("token singular", "http.request", None, dict(method="POST", url=API + "/v1/token", json={"scope": "all"})),
    ("share access", "http.request", None, dict(method="POST", url=HOST + "/v1/sites/7/access",
                                                json={"user": "x", "role": "admin"})),
    ("team member", "http.request", None, dict(method="PUT", url=API + "/v1/teams/1/users/someone", json={})),
    ("webhook", "http.request", None, dict(method="POST", url=HOST + "/api/v1/hooks", json={"url": "https://t.example"})),
    ("API-Keys upper case", "http.request", None, dict(method="POST", url=API + "/v1/API-Keys", json={})),
    ("customers read", "http.get", None, dict(url=API + "/v1/customers")),
    ("personal fields read", "http.get", None, dict(url=API + "/v1/users", query={"fields": "email,phone,address"})),
    ("members export", "http.get", None, dict(url=API + "/v1/members/export")),
    ("members as a json file", "http.get", None, dict(url=API + "/v1/members.json")),
    ("payment op", "payment.purchase", None, dict(amount=5.0, payee=HOST + "/billing")),
    ("discord delete", "discord.delete_channel", None, dict(guild_id=GUILD, channel_id=CH)),
    ("discord unknown write", "discord.purge_messages", None, dict(guild_id=GUILD, channel_id=CH)),
]
ROUTINE_OPS = [
    ("deploy a file", "http.request", None, dict(method="PUT", url=HOST + "/sites/cam/index.html", body="<h1>x</h1>")),
    ("a file whose name is a word of the lists", "http.request", None,
     dict(method="PUT", url=HOST + "/sites/cam/members.html", body="<p>our members page</p>")),
    ("start a deployment", "http.request", None, dict(method="POST", url=HOST + "/sites/cam/deploys", json={"files": 3})),
    ("update a project", "http.request", None, dict(method="PATCH", url=API + "/v1/projects/7", json={"title": "t"})),
    ("discord message", "discord.send_message", None, dict(guild_id=GUILD, channel_id=CH, content="Notes are up.")),
    ("discord channel", "discord.create_channel", None, dict(guild_id=GUILD, name="notes", type="text")),
    ("public web read", "http.get", None, dict(url="https://pypi.org/pypi/httpx/json")),
]


def service(tmp_path, n, lt, routine=None, **over):
    d = tmp_path / f"run{n}"
    d.mkdir()
    clock = C.SimClock()
    s = PolicyService(str(d), clock=clock, fsync=False)
    rec = standing_record(lt, clock.now(), **over)
    if routine is not None:
        rec["standing"]["routine"] = routine
    sign_register(s, tmp_path, rec)
    return s


@pytest.mark.parametrize("manifest", ["declared", "everything-declared"])
def test_every_consequential_attempt_waits_and_the_routine_set_runs(tmp_path, manifest):
    lt = live_task(resources={})
    routine = None if manifest == "declared" else EVERYTHING
    allowed = []
    for i, (label, op, mod, args) in enumerate(CONSEQUENTIAL):
        s = service(tmp_path, i, lt, routine)          # a fresh service: no rate limit masks a verdict
        try:
            d = s.authorize(act(op, mod, arguments=args), lt, {})
        finally:
            s.close()
        if d.verdict == ALLOW:
            allowed.append((label, d.policy_ids))
    assert allowed == [], allowed
    for i, (label, op, mod, args) in enumerate(ROUTINE_OPS):
        s = service(tmp_path, 100 + i, lt, routine)
        try:
            d = s.authorize(act(op, mod, arguments=args), lt, {})
        finally:
            s.close()
        assert d.verdict == ALLOW, (label, d.policy_ids, d.reasons)


def test_an_override_on_a_declared_route_asks_as_a_deletion_and_its_approval_covers_execution(svc, tmp_path, clock):
    lt = live_task(resources={})
    sign_register(svc, tmp_path, standing_record(lt, clock.now(), standing={
        "profile": PROFILE, "code_network": [], "routine": [{"methods": ["POST"], "dest": HOST + "/sites/*"}]}))
    a = act("http.request", arguments=dict(method="POST", url=HOST + "/sites/cam",
                                           headers={"X-HTTP-Method-Override": "DELETE"}))
    d = svc.authorize(a, lt, {})
    assert d.verdict == NEEDS and "R-IRREVERSIBLE-HTTP" in d.policy_ids, d.policy_ids
    svc.resolve_approval(d.approval_id, True, "operator")
    d = svc.authorize(a, lt, {})
    assert d.verdict == ALLOW and svc.check_grant(a, d)
    p = C.PrimitiveOp(op="http.request", modality=C.Modality.API,
                      arguments={"method": "POST", "url": HOST + "/sites/cam", "hop": 0})
    assert svc.check_primitive(p, a).verdict == ALLOW
    # a plain POST on the same declared route is routine
    assert svc.authorize(act("http.request", arguments=dict(method="POST", url=HOST + "/sites/cam", json={"x": 1})),
                         lt, {}).verdict == ALLOW


def test_one_segment_and_any_segment_wildcards(svc, tmp_path, clock):
    lt = live_task(resources={})
    sign_register(svc, tmp_path, standing_record(lt, clock.now(), standing={
        "profile": PROFILE, "code_network": [], "routine": [{"methods": ["POST"], "dest": HOST + "/v1/sites/*"},
                                                            {"methods": ["PUT"], "dest": HOST + "/files/**"}]}))
    post = lambda path: svc.authorize(act("http.request", arguments=dict(method="POST", url=HOST + path, json={})),  # noqa
                                      lt, {})
    assert post("/v1/sites/7").verdict == ALLOW
    d = post("/v1/sites/7/access")                         # '*' is one segment: a sub-resource is not declared
    assert d.verdict == NEEDS and "R-NOT-ROUTINE" in d.policy_ids
    put = svc.authorize(act("http.request", arguments=dict(method="PUT", url=HOST + "/files/a/b/c.css", body="x")),
                        lt, {})
    assert put.verdict == ALLOW


def test_money_asks_on_every_live_record_even_without_a_profile(svc, tmp_path, clock):
    lt = live_task(resources={})
    rec = standing_record(lt, clock.now())
    rec.pop("standing")                                    # explicit pre-authorization, no profile
    sign_register(svc, tmp_path, rec)
    for path, body in (("/v1/orders", {"sku": "x"}), ("/v1/notes", {"amount": 5}), ("/v1/subscriptions", {})):
        d = svc.authorize(act("http.request", arguments=dict(method="POST", url=API + path, json=body)), lt, {})
        assert d.verdict == NEEDS and "P-SPEND-APPROVAL" in d.policy_ids, (path, d.policy_ids)
    d = svc.authorize(act("http.request", arguments=dict(method="POST", url=API + "/v1/notes", json={"t": "x"})), lt, {})
    assert d.verdict == ALLOW                               # the record's explicit pre-authorization otherwise


def test_personal_data_through_a_declared_read_needs_the_tasks_permission_or_an_approval(tmp_path):
    for perm, want in ((False, NEEDS), (True, ALLOW)):
        lt = live_task(oid=f"obj-pd-{int(perm)}", resources={},
                       data_permissions={"collect_personal_data": True} if perm else {})
        s = service(tmp_path, int(perm), lt, EVERYTHING)
        try:
            d = s.authorize(act("http.get", arguments=dict(url=API + "/v1/customers")), lt, {})
            assert d.verdict == want, (perm, d.policy_ids)
            if want == NEEDS:
                assert "R-PERSONAL-DATA" in d.policy_ids
            assert s.authorize(act("http.get", arguments=dict(url=API + "/v1/projects/7")), lt, {}).verdict == ALLOW
        finally:
            s.close()


def test_page_interactions_on_owned_pages_are_routine_only_where_declared(tmp_path):
    lt = live_task(resources={})
    click = dict(ref="e12", page_url=HOST + "/sites/cam/settings")
    s = service(tmp_path, 1, lt)
    try:
        d = s.authorize(act("browser.click", C.Modality.STRUCTURED_BROWSER, arguments=click), lt, {})
        assert d.verdict == NEEDS and "R-NOT-ROUTINE" in d.policy_ids
    finally:
        s.close()
    s = service(tmp_path, 2, lt, ROUTINE + [{"methods": ["INTERACT"], "dest": HOST + "/preview/**"}])
    try:
        d = s.authorize(act("browser.click", C.Modality.STRUCTURED_BROWSER,
                            arguments=dict(ref="e3", page_url=HOST + "/preview/index.html")), lt, {})
        assert d.verdict == ALLOW, d.policy_ids
        d = s.authorize(act("browser.click", C.Modality.STRUCTURED_BROWSER, arguments=click), lt, {})
        assert d.verdict == NEEDS
    finally:
        s.close()


def test_a_record_cannot_declare_a_deletion_or_a_wildcard_host_routine():
    t = live_task(resources={})
    for bad in ([{"methods": ["DELETE"], "dest": HOST + "/sites/**"}],
                [{"methods": ["PUT"], "dest": "https://*.host.example/sites/**"}],
                [{"methods": ["PUT"], "dest": HOST + "/"}],
                [{"methods": ["PUT"], "dest": HOST + "/**", "extra": 1}],
                [{"methods": ["ANY"], "dest": HOST + "/x"}]):
        with pytest.raises(liveauth.LiveAuthorizationError):
            liveauth.validate_shape(standing_record(t, 1.0, standing={"profile": PROFILE, "routine": bad}))
    for bad in (["*.co.uk"], ["*.github.io"], ["*.pythonhosted.org"], ["8.8.8.8"], ["localhost"], ["127.0.0.1"]):
        with pytest.raises(liveauth.LiveAuthorizationError):
            liveauth.validate_shape(standing_record(t, 1.0, standing={"profile": PROFILE, "code_network": bad}))
    liveauth.validate_shape(standing_record(t, 1.0, standing={"profile": PROFILE,
                                                              "code_network": ["pypi.org", "127.0.0.1:8080"]}))


def test_an_older_policy_without_routine_operations_keeps_everything_under_approval(tmp_path):
    doc = yaml.safe_load(open(RUN_POLICY_PATH))
    doc["standing"].pop("routine_operations")
    doc["standing"].pop("consequence")
    old = tmp_path / "old.yaml"
    old.write_text(yaml.safe_dump(doc))
    clock = C.SimClock()
    s = PolicyService(str(tmp_path / "run"), str(old), clock=clock, fsync=False)
    try:
        assert s.policy is not None, s.policy_error
        lt = live_task(resources={})
        sign_register(s, tmp_path, standing_record(lt, clock.now()))
        d = s.authorize(act("discord.send_message", arguments=dict(guild_id=GUILD, channel_id=CH, content="hi")), lt, {})
        assert d.verdict == NEEDS and "R-NOT-ROUTINE" in d.policy_ids
        d = s.authorize(act("http.request", arguments=dict(method="PUT", url=HOST + "/sites/cam/a.html", body="x")),
                        lt, {})
        assert d.verdict == ALLOW                      # declared in the record's manifest
    finally:
        s.close()
